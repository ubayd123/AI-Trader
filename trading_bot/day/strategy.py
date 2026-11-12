"""Entry rules for the intraday bot. Pure functions: no I/O, no state.

Three modes, all long-only:
  dip       buy a short pullback inside an intraday uptrend
  breakout  buy a new high inside an intraday uptrend
  revert    buy any sharp short-term drop, expecting a bounce
Exits are the same for all: a take-profit, a stop-loss, a time limit, and
everything is sold before the close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .bars import Features
from .config import DayConfig


@dataclass
class Signal:
    symbol: str
    score: float  # larger = stronger; used to rank when slots are limited
    stop_pct: float  # distance below the entry price
    target_pct: float  # distance above the entry price
    reason: str


def entry_signal(f: Features, cfg: DayConfig) -> Optional[Signal]:
    if f.unit <= 0:
        return None
    uptrend = f.close > f.trend and f.trend_rising
    if cfg.mode == "dip":
        if not (uptrend and f.z <= -cfg.z_entry):
            return None
        score, reason = -f.z, f"dip in uptrend (z {f.z:.1f})"
    elif cfg.mode == "breakout":
        if not (uptrend and f.session_return > 0 and f.close > f.channel_high and f.z >= cfg.z_entry):
            return None
        score, reason = f.z, f"breakout above {cfg.lookback}-min high (z {f.z:.1f})"
    else:  # revert
        if f.z > -cfg.z_entry:
            return None
        score, reason = -f.z, f"sharp drop (z {f.z:.1f})"
    stop = min(max(cfg.stop_mult * f.unit, cfg.min_stop_pct), cfg.max_stop_pct)
    target = stop * cfg.target_mult / cfg.stop_mult
    return Signal(f.symbol, score, stop, target, reason)
