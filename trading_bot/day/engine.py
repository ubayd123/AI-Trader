"""The intraday engine: turns a stream of quotes into trades, inside hard risk limits.

The same engine runs live (quotes polled from eToro), in the dry run (quotes
from Yahoo, simulated fills) and in the backtest (historical bars replayed
as quotes), so what is backtested is the code that trades.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .bars import NEW_YORK, BarSeries, Quote, session_bounds
from .config import DayConfig
from .strategy import entry_signal

log = logging.getLogger(__name__)


class BrokerError(Exception):
    pass


@dataclass
class Spec:
    """What the broker allows for one instrument."""

    min_size: float = 0.0  # smallest position size (shares, or pounds per point for a spread bet)
    size_decimals: int = 6  # sizes are rounded down to this many decimals
    min_notional: float = 0.0  # smallest exposure, in account currency
    margin_pct: float = 1.0  # cash tied up as a fraction of the exposure; 1.0 means no leverage
    min_stop_distance: float = 0.0  # closest a stop may sit to the price, in price units
    tradable: bool = True


@dataclass
class OpenFill:
    position_id: object
    units: float
    price: float
    order_id: object = 0
    stop: Optional[float] = None  # the levels the broker actually set, when it reports them
    target: Optional[float] = None


@dataclass
class CloseFill:
    price: float
    pnl: Optional[float] = None  # broker-reported profit, when available


@dataclass
class Closure:
    """A position the broker closed by itself (stop-loss or take-profit)."""

    position_id: object
    price: float
    reason: str
    pnl: Optional[float] = None


@dataclass
class DayPosition:
    symbol: str
    position_id: object  # eToro: a number. IG: a deal id string
    units: float  # shares, or pounds per point
    entry: float
    amount: float  # exposure (units x entry price)
    stop: float
    target: float
    opened_at: str
    reason: str
    order_id: object = 0
    margin: float = 0.0  # cash tied up; 0 means the full amount (no leverage)

    @property
    def cash_used(self) -> float:
        return self.margin or self.amount


def _tick(price: float) -> float:
    """Round a price level to what the broker accepts: cents for shares above $1."""
    return round(price, 2) if price >= 1 else round(price, 4)


@dataclass
class DayState:
    budget: float  # what the bot started with, in the account's currency
    realized: float = 0.0
    positions: list[DayPosition] = field(default_factory=list)
    peak_equity: float = 0.0
    session: str = ""
    day_start_equity: float = 0.0
    day_realized: float = 0.0
    trades_today: int = 0
    halted_today: bool = False
    halted: bool = False
    halt_reason: str = ""
    cooldowns: dict[str, str] = field(default_factory=dict)
    excluded_today: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> Optional["DayState"]:
        if not path.is_file():
            return None
        raw = json.loads(path.read_text())
        raw["positions"] = [DayPosition(**p) for p in raw.get("positions", [])]
        return cls(**raw)


class DayEngine:
    def __init__(self, cfg: DayConfig, broker, state: DayState, persist: bool = True):
        self.cfg = cfg
        self.broker = broker
        self.state = state
        self.persist = persist
        self.series = {s: BarSeries(s) for s in cfg.symbols}
        self.last_quotes: dict[str, Quote] = {}
        self.trades: list[dict] = []  # every fill this run (the backtest reads this)
        self.blocked: dict[str, str] = {}  # instrument -> why the bot cannot trade it (shown by `check`)
        self._clock_skew = math.inf  # seconds this computer's clock runs ahead of the quote timestamps
        if state.peak_equity <= 0:
            state.peak_equity = state.budget + state.realized
        if state.day_start_equity <= 0:
            state.day_start_equity = state.budget + state.realized

    # ---------- accounting ----------
    def _bid(self, pos: DayPosition) -> float:
        quote = self.last_quotes.get(pos.symbol)
        return quote.bid if quote else pos.entry

    def equity(self) -> float:
        st = self.state
        return st.budget + st.realized + sum((self._bid(p) - p.entry) * p.units for p in st.positions)

    def cash(self) -> float:
        """Budget not tied up in open positions (as margin, for leveraged trades)."""
        st = self.state
        return st.budget + st.realized - sum(p.cash_used for p in st.positions)

    def spec(self, symbol: str) -> Spec:
        if hasattr(self.broker, "spec"):
            return self.broker.spec(symbol)
        return Spec(min_notional=self.cfg.min_position_usd)

    def _block(self, symbol: str, why: str) -> None:
        if self.blocked.get(symbol) != why:
            self.blocked[symbol] = why
            if self.persist:
                log.info("Not trading %s: %s", symbol, why)

    def day_pnl(self) -> float:
        return self.equity() - self.state.day_start_equity

    def _save(self) -> None:
        if self.persist:
            self.state.save(self.cfg.state_path)

    def _record(self, now: datetime, side: str, pos: DayPosition, price: float, pnl: Optional[float], reason: str) -> None:
        self.state.trades_today += 1
        record = {"time": now.isoformat(), "env": self.cfg.label, "broker": self.broker.name, "side": side,
                  "symbol": pos.symbol, "position_id": pos.position_id, "units": pos.units, "price": price,
                  "value": round(pos.units * price, 4), "pnl": None if pnl is None else round(pnl, 4),
                  "currency": self.cfg.currency, "reason": reason}
        self.trades.append(record)
        if self.persist:
            self.cfg.trades_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.trades_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            cur = self.cfg.currency
            log.info("%s %-6s %.4f @ %.2f (%s%.2f exposure)%s | %s", side.upper(), pos.symbol, pos.units, price, cur,
                     pos.units * price, "" if pnl is None else f"  P&L {cur}{pnl:+.3f}", reason)

    def _settle(self, now: datetime, pos: DayPosition, price: float, pnl: Optional[float], reason: str) -> None:
        """Book a closed position into the ledger."""
        st = self.state
        if pnl is None:
            pnl = (price - pos.entry) * pos.units
        st.realized += pnl
        st.day_realized += pnl
        st.positions = [p for p in st.positions if p.position_id != pos.position_id]
        if pnl < 0:
            st.cooldowns[pos.symbol] = (now + timedelta(minutes=self.cfg.cooldown_minutes)).isoformat()
        self._record(now, "sell", pos, price, pnl, reason)
        self._save()

    def _close(self, now: datetime, pos: DayPosition, reason: str) -> bool:
        quote = self.last_quotes.get(pos.symbol)
        if quote is None:
            return False
        try:
            fill = self.broker.close(pos, quote, reason)
        except BrokerError as exc:
            log.error("Could not close %s (%s): %s", pos.symbol, reason, exc)
            return False
        self._settle(now, pos, fill.price, fill.pnl, reason)
        return True

    def flatten(self, now: datetime, reason: str) -> None:
        for pos in list(self.state.positions):
            self._close(now, pos, reason)

    # ---------- the main step ----------
    def on_quotes(self, now: datetime, quotes: dict[str, Quote]) -> None:
        st, cfg = self.state, self.cfg
        self.ensure_session(now)
        session = st.session
        self.last_quotes.update(quotes)
        if quotes:
            # The smallest gap ever seen between our clock and a quote's timestamp is the clock difference;
            # anything beyond that is genuine staleness. This keeps a wrong PC clock from blocking all trades.
            self._clock_skew = min(self._clock_skew, min((now - q.time).total_seconds() for q in quotes.values()))

        # 1. exits the broker made by itself (stop-loss / take-profit orders)
        for c in self.broker.sync(st.positions, quotes, now):
            pos = next((p for p in st.positions if p.position_id == c.position_id), None)
            if pos:
                self._settle(now, pos, c.price, c.pnl, c.reason)

        # 2. account-level limits, checked on every quote
        equity = self.equity()
        st.peak_equity = max(st.peak_equity, equity)
        if not st.halted and equity <= st.peak_equity * (1 - cfg.max_total_drawdown_pct):
            st.halted = True
            st.halt_reason = (f"{session}: equity {cfg.currency}{equity:.2f} is {cfg.max_total_drawdown_pct:.0%} below "
                              f"its peak of {cfg.currency}{st.peak_equity:.2f}. Stopped until `day reset`.")
            log.error("KILL SWITCH: %s", st.halt_reason)
            self.flatten(now, "kill switch: total drawdown")
        if not st.halted_today and self.day_pnl() <= -cfg.daily_loss_limit_pct * st.day_start_equity:
            st.halted_today = True
            log.warning("Daily loss limit reached (%.2f%% of %s%.2f). No more trades today.",
                        100 * self.day_pnl() / st.day_start_equity, cfg.currency, st.day_start_equity)
            self.flatten(now, "daily loss limit")

        # 3. bars: everything below runs once a minute, when a bar completes
        completed = [s for s, q in quotes.items() if s in self.series and self.series[s].update(q, now) is not None]
        if not completed:
            return
        _, session_end = session_bounds(now)
        if now >= session_end - timedelta(minutes=cfg.flatten_before_close):
            self.flatten(now, "end of day")
        elif st.halted or st.halted_today:
            self.flatten(now, "trading halted")  # retry anything an earlier flatten could not sell
        else:
            self._manage(now)
            if now < session_end - timedelta(minutes=cfg.stop_entries_before_close):
                self._enter(now, completed)
        self._save()

    def ensure_session(self, now: datetime) -> None:
        """Start a new trading day (fresh counters and bars) the first time it is seen."""
        session = now.astimezone(NEW_YORK).date().isoformat()
        if session != self.state.session:
            self._start_session(session)

    def seed(self, symbol: str, bars: list) -> None:
        """Load today's earlier bars so indicators are ready straight after a (re)start."""
        if symbol in self.series and not self.series[symbol].bars:
            self.series[symbol].seed(bars)

    def _start_session(self, session: str) -> None:
        st = self.state
        st.session = session
        st.day_start_equity = self.equity()
        st.day_realized = 0.0
        st.trades_today = 0
        st.halted_today = False
        st.cooldowns = {}
        st.excluded_today = []
        self.blocked = {}
        for series in self.series.values():
            series.reset_session()

    def _manage(self, now: datetime) -> None:
        for pos in list(self.state.positions):
            held = now - datetime.fromisoformat(pos.opened_at)
            if held >= timedelta(minutes=self.cfg.max_hold_minutes):
                self._close(now, pos, "time limit")
            elif self._bid(pos) <= pos.stop * 0.999:
                self._close(now, pos, "stop-loss (fallback)")  # the broker's own stop should have fired
            elif self.cfg.exit_at_mean and self.cfg.mode != "breakout":
                features = self.series[pos.symbol].features(self.cfg.lookback, self.cfg.trend_bars)
                if features and features.z >= 0:
                    self._close(now, pos, "back to average")

    def _enter(self, now: datetime, symbols: list[str]) -> None:
        st, cfg = self.state, self.cfg
        if st.halted or st.halted_today:
            return
        session_start, _ = session_bounds(now)
        if now < session_start + timedelta(minutes=cfg.skip_open_minutes):
            return
        held = {p.symbol for p in st.positions}
        signals = []
        for symbol in symbols:
            if symbol in held or symbol in st.excluded_today:
                continue
            if st.cooldowns.get(symbol, "") > now.isoformat():
                continue
            features = self.series[symbol].features(cfg.lookback, cfg.trend_bars)
            signal = features and entry_signal(features, cfg)
            if signal:
                signals.append(signal)
        for signal in sorted(signals, key=lambda s: s.score, reverse=True):
            # every open position still needs one trade to close, so reserve room for those
            if st.trades_today + len(st.positions) + 2 > cfg.max_trades_per_day:
                break
            if len(st.positions) >= cfg.max_positions:
                break
            quote = self.last_quotes[signal.symbol]
            if quote.spread_pct > cfg.max_spread_pct or quote.bid <= 0:
                continue
            if (now - quote.time).total_seconds() - self._clock_skew > cfg.max_quote_age_seconds:
                continue  # stale price
            plan = self._size(signal, quote)
            if plan is None:
                continue
            amount, units, stop_pct, target_pct, margin = plan
            # risk gate: if every open stop were hit, plus this one, the day must still be inside the limit
            worst = st.day_realized + sum((p.stop - p.entry) * p.units for p in st.positions)
            new_risk = amount * (stop_pct + quote.spread_pct)
            if worst - new_risk < -cfg.daily_loss_limit_pct * st.day_start_equity:
                self._block(signal.symbol, "one more stop-out could breach today's loss limit")
                continue
            stop = _tick(quote.ask * (1 - stop_pct))
            target = _tick(quote.ask * (1 + target_pct))
            try:
                fill = self.broker.open(signal.symbol, amount, stop, target, quote)
            except BrokerError as exc:
                log.error("Could not buy %s: %s", signal.symbol, exc)
                continue
            exposure = fill.units * fill.price
            pos = DayPosition(signal.symbol, fill.position_id, fill.units, fill.price, exposure,
                              fill.stop or stop, fill.target or target, now.isoformat(), signal.reason, fill.order_id,
                              margin=0.0 if margin >= amount else exposure * margin / amount)
            st.positions.append(pos)
            self._record(now, "buy", pos, fill.price, None, signal.reason)
            self._save()

    def precheck(self, quotes: dict[str, Quote]) -> dict[str, str]:
        """Which markets cannot be traded at all with this budget, and why. Run once when a session starts,
        so a too-small account is told straight away instead of silently never trading."""
        cfg, equity, cur = self.cfg, self.equity(), self.cfg.currency
        for symbol, quote in quotes.items():
            if symbol not in self.series or quote.ask <= 0:
                continue
            spec = self.spec(symbol)
            floor = max(spec.min_notional, spec.min_size * quote.ask)
            if spec.margin_pct >= 1.0:
                if equity * (1 - cfg.cash_reserve_pct) < floor:
                    self._block(symbol, f"the budget is below the minimum trade of {cur}{floor:.2f}")
                continue
            stop_pct = max(cfg.min_stop_pct, 1.05 * spec.min_stop_distance / quote.ask)
            loss = floor * (stop_pct + quote.spread_pct)
            risk_cash = equity * cfg.risk_per_trade_pct
            if loss > risk_cash:
                self._block(symbol, (
                    f"the smallest trade the broker allows ({cur}{floor:,.0f} exposure) would lose about "
                    f"{cur}{loss:.2f} if stopped out; the limit per trade is {cur}{risk_cash:.2f} "
                    f"({cfg.risk_per_trade_pct:.2%} of {cur}{equity:.2f})"))
            elif floor * spec.margin_pct > equity * (1 - cfg.cash_reserve_pct):
                self._block(symbol, f"the smallest trade needs {cur}{floor * spec.margin_pct:,.2f} of margin, more than the budget")
        return dict(self.blocked)

    def slots(self) -> int:
        """How many unleveraged positions of the broker's minimum size the budget can hold."""
        usable = self.equity() * (1 - self.cfg.cash_reserve_pct)
        return min(self.cfg.max_positions, int(usable // self.cfg.min_position_usd))

    def _size(self, signal, quote):
        """Decide the trade size. Returns (exposure, units, stop %, target %, cash tied up) or None.

        Unleveraged (shares): the budget is split into equal slots.
        Leveraged (spread bets): sized so that a stop-out loses risk_per_trade_pct of the account. If the
        broker's minimum size would risk more than that, the trade is refused rather than oversized.
        """
        cfg, spec = self.cfg, self.spec(signal.symbol)
        if not spec.tradable:
            self._block(signal.symbol, "the market is not open for dealing")
            return None
        stop_pct = max(signal.stop_pct, 1.05 * spec.min_stop_distance / quote.ask)
        target_pct = signal.target_pct * stop_pct / signal.stop_pct
        equity = self.equity()
        floor = max(spec.min_notional, spec.min_size * quote.ask)
        free = self.cash() - equity * cfg.cash_reserve_pct
        cur = cfg.currency
        if spec.margin_pct >= 1.0:
            slots = self.slots()
            if slots == 0:
                self._block(signal.symbol, f"the budget is below the minimum trade of {cur}{floor:.2f}")
                return None
            amount = min(max(equity * (1 - cfg.cash_reserve_pct) / slots, floor), self.cash())
            amount = math.floor(amount * 100) / 100
            if amount < floor:
                self._block(signal.symbol, f"not enough free budget for the minimum trade of {cur}{floor:.2f}")
                return None
            self.blocked.pop(signal.symbol, None)
            return amount, amount / quote.ask, stop_pct, target_pct, amount
        risk_cash = equity * cfg.risk_per_trade_pct
        amount = risk_cash / (stop_pct + quote.spread_pct)
        if amount < floor:
            self._block(signal.symbol, (
                f"the smallest trade the broker allows ({cur}{floor:,.0f} exposure) would lose about "
                f"{cur}{floor * (stop_pct + quote.spread_pct):.2f} if stopped out; the limit per trade is "
                f"{cur}{risk_cash:.2f} ({cfg.risk_per_trade_pct:.2%} of {cur}{equity:.2f})"))
            return None
        margin_pct = spec.margin_pct
        amount = min(amount, max(free, 0.0) / margin_pct)
        scale = 10 ** spec.size_decimals
        units = math.floor(amount / quote.ask * scale + 1e-9) / scale
        amount = units * quote.ask
        if units <= 0 or units < spec.min_size or amount < floor * 0.999:
            self._block(signal.symbol, f"not enough free budget for the minimum trade (it needs {cur}{floor * margin_pct:,.2f} of margin)")
            return None
        self.blocked.pop(signal.symbol, None)
        return amount, units, stop_pct, target_pct, amount * margin_pct

    # ---------- reporting ----------
    def status(self) -> dict:
        st = self.state
        equity = self.equity()
        return {
            "env": self.cfg.label, "currency": self.cfg.currency, "budget": round(st.budget, 2),
            "equity": round(equity, 2),
            "return_pct": round((equity / st.budget - 1) * 100, 2) if st.budget else 0.0,
            "today_pnl": round(self.day_pnl(), 3), "trades_today": st.trades_today,
            "halted_today": st.halted_today, "halted": st.halted, "halt_reason": st.halt_reason,
            "blocked": dict(self.blocked),
            "positions": [{"symbol": p.symbol, "units": round(p.units, 5), "entry": p.entry, "stop": p.stop,
                           "target": p.target, "exposure": round(self._bid(p) * p.units, 2)} for p in st.positions],
        }
