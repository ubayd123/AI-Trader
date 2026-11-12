"""Simulated broker, historical replay (backtest) and Yahoo intraday data."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .bars import NEW_YORK, Quote
from .config import IG_MARKETS, DayConfig
from .engine import Closure, CloseFill, DayEngine, DayPosition, DayState, OpenFill, Spec

log = logging.getLogger(__name__)


class SimBroker:
    """Fills at the quoted bid/ask plus slippage, and runs stop-loss / take-profit orders like eToro does."""

    name = "sim"

    def __init__(self, cfg: DayConfig):
        self.cfg = cfg
        self.slip = cfg.slippage_bp / 10_000
        self._next_id = 1
        self._prev_bid: dict[str, float] = {}
        self.costs = 0.0  # spread + slippage paid, in account currency

    def spec(self, symbol: str) -> Spec:
        market = IG_MARKETS.get(symbol) if self.cfg.broker == "ig" else None
        if market:  # IG's published standard minimums and retail margin
            return Spec(min_size=market["min_size"], size_decimals=2, margin_pct=market["margin"])
        return Spec(min_notional=self.cfg.min_position_usd)

    def open(self, symbol: str, amount: float, stop: float, target: float, quote: Quote) -> OpenFill:
        price = quote.ask * (1 + self.slip)
        self.costs += amount * (price / quote.mid - 1)
        self._next_id += 1
        return OpenFill(self._next_id, amount / price, price)

    def _sell(self, pos: DayPosition, raw: float, mid: float) -> float:
        price = raw * (1 - self.slip)
        self.costs += max(pos.units * (mid - price), 0.0) if raw <= mid else pos.units * raw * self.slip
        return price

    def close(self, pos: DayPosition, quote: Quote, reason: str) -> CloseFill:
        return CloseFill(self._sell(pos, quote.bid, quote.mid))

    def sync(self, positions: list[DayPosition], quotes: dict[str, Quote], now: datetime) -> list[Closure]:
        closures = []
        for pos in positions:
            quote = quotes.get(pos.symbol)
            if quote is None:
                continue
            prev = self._prev_bid.get(pos.symbol)
            if quote.bid <= pos.stop:  # stop checked first: conservative when both are touched
                raw = pos.stop if prev is not None and prev > pos.stop else quote.bid  # gapped through: worse fill
                closures.append(Closure(pos.position_id, self._sell(pos, raw, raw * (1 + quote.spread_pct / 2)), "stop-loss"))
            elif quote.bid >= pos.target:
                raw = pos.target if prev is not None and prev < pos.target else quote.bid
                closures.append(Closure(pos.position_id, self._sell(pos, raw, raw * (1 + quote.spread_pct / 2)), "take-profit"))
        for symbol, quote in quotes.items():
            self._prev_bid[symbol] = quote.bid
        return closures


def spread_fraction(cfg: DayConfig, symbol: str, price: float) -> float:
    """Bid/ask spread as a fraction of price, for simulated quotes."""
    market = IG_MARKETS.get(symbol) if cfg.broker == "ig" else None
    if market:
        return market["spread"] / price  # IG quotes its minimum spread in points
    return cfg.spread_pct(symbol)


def load_yahoo_minutes(symbols: list[str], days: int = 29, interval: str = "1m") -> dict[str, pd.DataFrame]:
    """Regular-session intraday bars from Yahoo (UTC index). Yahoo keeps about 30 days of 1m and 60 days of 5m.

    IG market names (US500, EURUSD, ...) are fetched from their Yahoo proxy and scaled to IG points.
    """
    import yfinance as yf

    out = {}
    end = datetime.now(timezone.utc)
    step = 7 if interval == "1m" else 55
    for symbol in symbols:
        frames = []
        cursor = end - timedelta(days=days)
        while cursor < end:
            stop = min(cursor + timedelta(days=step), end)
            try:
                ticker = IG_MARKETS.get(symbol, {}).get("yahoo", symbol)
                frame = yf.Ticker(ticker).history(interval=interval, start=cursor, end=stop, prepost=False, auto_adjust=False)
            except Exception as exc:
                log.warning("No %s data for %s (%s to %s): %s", interval, symbol, cursor.date(), stop.date(), exc)
                frame = None
            if frame is not None and not frame.empty:
                frames.append(frame[["Open", "High", "Low", "Close"]])
            cursor = stop
        if not frames:
            continue
        bars = pd.concat(frames)
        bars = bars[~bars.index.duplicated(keep="last")].sort_index().dropna()
        local = bars.index.tz_convert(NEW_YORK)
        minutes = local.hour * 60 + local.minute
        bars = bars[(minutes >= 570) & (minutes < 960)]
        bars.index = bars.index.tz_convert(timezone.utc)
        bars.columns = ["open", "high", "low", "close"]
        bars = bars * IG_MARKETS.get(symbol, {}).get("scale", 1)
        if float((bars["high"] == bars["low"]).mean()) > 0.9:
            log.warning("Yahoo's %s data has no range inside each minute, so it cannot be backtested; skipping", symbol)
            continue
        out[symbol] = bars
    return out


def replay(cfg: DayConfig, data: dict[str, pd.DataFrame], budget: float = 66.0,
           start: Optional[str] = None, end: Optional[str] = None) -> dict:
    """Backtest: replay historical minute bars through the real engine with a simulated broker.

    Each bar becomes four quotes in the order open, low, high, close. Seeing the low
    before the high is the pessimistic assumption for a long position with a stop.
    """
    symbols = [s for s in cfg.symbols if s in data]
    index = sorted(set().union(*(data[s].index for s in symbols)))
    if start:
        index = [t for t in index if t >= pd.Timestamp(start, tz="UTC")]
    if end:
        index = [t for t in index if t < pd.Timestamp(end, tz="UTC")]
    if not index:
        raise ValueError("No intraday data in the requested window")
    rows = {s: {t: tuple(r) for t, r in zip(data[s].index, data[s].itertuples(index=False, name=None))} for s in symbols}
    step = (index[1] - index[0]).total_seconds() if len(index) > 1 else 60
    step = min(step, 300)  # bar length in seconds (60 for 1m data, 300 for 5m)

    broker = SimBroker(cfg)
    engine = DayEngine(cfg, broker, DayState(budget=budget), persist=False)
    daily: dict[str, float] = {}
    curve = []
    for t in index:
        ts = t.to_pydatetime()
        for k in range(4):  # open, low, high, close
            now = ts + timedelta(seconds=k * step / 4)
            quotes = {}
            for s in symbols:
                row = rows[s].get(t)
                if row is not None:
                    price = (row[0], row[2], row[1], row[3])[k]
                    half = spread_fraction(cfg, s, price) / 2
                    quotes[s] = Quote(s, price * (1 - half), price * (1 + half), now)
            if quotes:
                engine.on_quotes(now, quotes)
        daily[engine.state.session] = engine.equity()
        curve.append(engine.equity())
    engine.flatten(index[-1].to_pydatetime() + timedelta(seconds=step), "end of data")
    return summarise(engine, broker, daily, curve, budget)


def summarise(engine: DayEngine, broker: SimBroker, daily: dict[str, float], curve: list[float], budget: float) -> dict:
    sells = [t for t in engine.trades if t["side"] == "sell"]
    wins = [t["pnl"] for t in sells if t["pnl"] > 0]
    losses = [t["pnl"] for t in sells if t["pnl"] <= 0]
    days = list(daily)
    equity = pd.Series([budget] + [daily[d] for d in days])
    day_returns = equity.pct_change().dropna()
    series = pd.Series(curve) if curve else equity
    reasons: dict[str, int] = {}
    for t in sells:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    final = engine.equity()
    return {
        "days": len(days), "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
        "currency": engine.cfg.currency, "start": budget, "final": round(final, 2), "total_return_pct": round((final / budget - 1) * 100, 2),
        "avg_day_pct": round(float(day_returns.mean()) * 100, 3) if len(day_returns) else 0.0,
        "best_day_pct": round(float(day_returns.max()) * 100, 2) if len(day_returns) else 0.0,
        "worst_day_pct": round(float(day_returns.min()) * 100, 2) if len(day_returns) else 0.0,
        "winning_days_pct": round(100 * float((day_returns > 0).mean())) if len(day_returns) else 0,
        "max_drawdown_pct": round(float((series / series.cummax() - 1).min()) * 100, 2),
        "trades": len(engine.trades), "trades_per_day": round(len(engine.trades) / max(len(days), 1), 1),
        "round_trips": len(sells), "win_rate_pct": round(100 * len(wins) / len(sells)) if sells else None,
        "avg_win": round(sum(wins) / len(wins), 4) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 4) if losses else None,
        "costs": round(broker.costs, 2), "gross_pnl": round(final - budget + broker.costs, 2),
        "exit_reasons": reasons, "kill_switch": engine.state.halted, "not_traded": dict(engine.blocked),
    }
