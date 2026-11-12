"""Plain data structures shared by the strategy, engine, brokers and backtest."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Snapshot:
    """Indicator snapshot for one instrument. All prices are in GBP."""

    symbol: str
    market: str
    price: float
    sma_fast: float
    sma_slow: float
    ret_20: float
    ret_60: float
    rsi: float
    vol: float  # annualised volatility of daily returns
    usd_price: Optional[float] = None

    @property
    def trend_ok(self) -> bool:
        """Entry condition: price and the fast average are both above the slow average."""
        return self.price > self.sma_slow and self.sma_fast > self.sma_slow

    def trend_broken(self, buffer_pct: float) -> bool:
        """Exit condition. The buffer stops one noisy close from forcing a (fee-paying) exit."""
        return self.price < self.sma_slow * (1 - buffer_pct)

    @property
    def score(self) -> float:
        """Risk-adjusted momentum: blended 20/60-day return per unit of volatility."""
        return (0.5 * self.ret_20 + 0.5 * self.ret_60) / max(self.vol, 0.05)


@dataclass
class Position:
    market: str
    qty: float
    avg_price: float  # GBP, including buy costs
    high_water: float  # highest GBP price seen since entry
    opened_at: str


@dataclass
class Order:
    side: str  # buy | sell
    symbol: str
    market: str
    reason: str
    gbp_amount: float = 0.0  # buys: cash to spend (fees included)
    qty: float = 0.0  # sells: quantity to sell
    stop: bool = False  # sells: triggered by a stop (starts a re-entry cooldown)


@dataclass
class Fill:
    qty: float
    price_gbp: float  # per unit, before fees
    fee_gbp: float
    estimated: bool = False


@dataclass
class Portfolio:
    """The bot's own ledger. With a real broker this is the bot's ring-fenced budget."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    peak_equity: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    halted_until: str = ""  # ISO date the pause ends; empty = manual reset
    cooldowns: dict[str, str] = field(default_factory=dict)  # symbol -> ISO date re-entry allowed
    starting_cash: float = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p.qty * prices.get(s, p.avg_price) for s, p in self.positions.items())

    def apply_buy(self, order: Order, fill: Fill, when: str) -> None:
        cost = fill.qty * fill.price_gbp + fill.fee_gbp
        pos = self.positions.get(order.symbol)
        if pos:
            total = pos.qty + fill.qty
            pos.avg_price = (pos.qty * pos.avg_price + cost) / total
            pos.qty = total
            pos.high_water = max(pos.high_water, fill.price_gbp)
        else:
            self.positions[order.symbol] = Position(order.market, fill.qty, cost / fill.qty, fill.price_gbp, when)
        self.cash -= cost
        self.fees_paid += fill.fee_gbp

    def apply_sell(self, order: Order, fill: Fill) -> float:
        pos = self.positions[order.symbol]
        proceeds = fill.qty * fill.price_gbp - fill.fee_gbp
        pnl = proceeds - fill.qty * pos.avg_price
        pos.qty -= fill.qty
        if pos.qty <= 1e-9:
            del self.positions[order.symbol]
        self.cash += proceeds
        self.realized_pnl += pnl
        self.fees_paid += fill.fee_gbp
        return pnl

    # --- persistence ---
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written ledger

    @classmethod
    def load(cls, path: Path, starting_cash: float) -> "Portfolio":
        if not path.is_file():
            return cls(cash=starting_cash, peak_equity=starting_cash, starting_cash=starting_cash)
        raw = json.loads(path.read_text())
        raw["positions"] = {s: Position(**p) for s, p in raw.get("positions", {}).items()}
        return cls(**raw)
