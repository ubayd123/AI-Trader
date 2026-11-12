"""Indicators computed from a daily close series (pandas Series, oldest first)."""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from .config import Config
from .models import Snapshot

MIN_BARS = 61


def rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff().dropna().tail(period)
    gains = delta.clip(lower=0).sum()
    losses = -delta.clip(upper=0).sum()
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    return float(100 - 100 / (1 + gains / losses))


def snapshot(symbol: str, market: str, closes: pd.Series, cfg: Config, usd_price: Optional[float] = None) -> Optional[Snapshot]:
    closes = closes.dropna()
    if len(closes) < max(MIN_BARS, cfg.sma_slow + 1):
        return None
    price = float(closes.iloc[-1])
    if not math.isfinite(price) or price <= 0:
        return None
    periods = 365 if market == "crypto" else 252
    return Snapshot(
        symbol=symbol,
        market=market,
        price=price,
        sma_fast=float(closes.tail(cfg.sma_fast).mean()),
        sma_slow=float(closes.tail(cfg.sma_slow).mean()),
        ret_20=price / float(closes.iloc[-21]) - 1,
        ret_60=price / float(closes.iloc[-61]) - 1,
        rsi=rsi(closes),
        vol=float(closes.pct_change().tail(20).std() * math.sqrt(periods)),
        usd_price=usd_price,
    )
