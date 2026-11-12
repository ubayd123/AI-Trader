"""One trading cycle: data -> risk checks -> exits -> entries -> persist."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from .brokers import BrokerError, PositionMissing
from .config import Config
from .data import is_market_open
from .models import Order, Portfolio, Snapshot
from .strategy import candidates, circuit_breaker, entry_orders, exit_orders, update_high_water

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg: Config, broker, data, advisor=None, platform=None):
        self.cfg = cfg
        self.broker = broker
        self.data = data
        self.advisor = advisor
        self.platform = platform
        self.portfolio = Portfolio.load(cfg.state_path, cfg.starting_cash_gbp)

    # --- execution ---
    def _execute(self, order: Order, snap: Snapshot, now: datetime) -> Optional[dict]:
        if not self.broker.supports(order.market):
            log.info("SKIP %s %s: %s broker does not trade %s", order.side, order.symbol, self.broker.name, order.market)
            return None
        if self.cfg.enforce_market_hours and not is_market_open(order.market, now):
            log.info("WAIT %s %s (%s): market closed, will retry next cycle", order.side, order.symbol, order.reason)
            return None
        try:
            fill = self.broker.execute(order, snap)
        except PositionMissing as exc:
            log.warning("%s. Removing %s from the bot's ledger.", exc, order.symbol)
            self.portfolio.positions.pop(order.symbol, None)
            return None
        except BrokerError as exc:
            log.error("FAILED %s %s: %s", order.side, order.symbol, exc)
            return None

        pnl = None
        if order.side == "buy":
            self.portfolio.apply_buy(order, fill, now.isoformat())
        else:
            pnl = self.portfolio.apply_sell(order, fill)
        # Persist immediately: after a real order the ledger must never be lost.
        self.portfolio.save(self.cfg.state_path)
        record = {"time": now.isoformat(), "broker": self.broker.name, "side": order.side, "symbol": order.symbol,
                  "market": order.market, "qty": fill.qty, "price_gbp": fill.price_gbp, "fee_gbp": fill.fee_gbp,
                  "value_gbp": fill.qty * fill.price_gbp, "pnl_gbp": pnl, "estimated": fill.estimated,
                  "reason": order.reason}
        with self.cfg.trades_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info("%s %s %.6f @ £%.2f (£%.2f)%s | %s", order.side.upper(), order.symbol, fill.qty, fill.price_gbp,
                 record["value_gbp"], f" P&L £{pnl:+.2f}" if pnl is not None else "", order.reason)
        if self.platform and snap.usd_price:
            self.platform.publish_trade(order.market, order.side, order.symbol, snap.usd_price, fill.qty,
                                        order.reason, now)
        return record

    # --- the cycle ---
    def run_cycle(self, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now(timezone.utc)
        today = now.date()
        pf = self.portfolio
        snaps = {s: v for s, v in self.data.snapshots(now).items() if self.broker.supports(v.market)}
        if not snaps:
            raise RuntimeError("No market data available; nothing was traded")

        update_high_water(pf, snaps)
        prices = {s: v.price for s, v in snaps.items()}
        was_halted = pf.halted
        circuit_breaker(pf, pf.equity(prices), self.cfg, today)
        if pf.halted and not was_halted:
            log.error("KILL SWITCH: %s. Selling everything.", pf.halt_reason)

        advice = None
        if self.advisor and not pf.halted:
            picks = candidates(pf, snaps, self.cfg, today)
            if pf.positions or picks:
                context = self.platform.market_context() if self.platform else None
                advice = self.advisor.advise(pf, snaps, picks, context)

        trades = []
        for order in exit_orders(pf, snaps, self.cfg, today, advice, liquidate=pf.halted):
            trades.append(self._execute(order, snaps[order.symbol], now))
        for order in entry_orders(pf, snaps, self.cfg, today, advice):
            trades.append(self._execute(order, snaps[order.symbol], now))

        pf.save(self.cfg.state_path)
        return {"time": now.isoformat(), "trades": [t for t in trades if t], "ai_used": advice is not None,
                **self.status(prices)}

    def status(self, prices: Optional[dict[str, float]] = None) -> dict:
        pf = self.portfolio
        prices = prices or {}
        equity = pf.equity(prices)
        return {
            "broker": self.broker.name, "equity_gbp": round(equity, 2), "cash_gbp": round(pf.cash, 2),
            "return_pct": round((equity / pf.starting_cash - 1) * 100, 2) if pf.starting_cash else 0.0,
            "realized_pnl_gbp": round(pf.realized_pnl, 2), "fees_paid_gbp": round(pf.fees_paid, 2),
            "halted": pf.halted, "halt_reason": pf.halt_reason,
            "positions": {s: {**asdict(p), "price": prices.get(s),
                              "value_gbp": round(p.qty * prices.get(s, p.avg_price), 2)}
                          for s, p in pf.positions.items()},
        }
