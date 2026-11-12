"""Quotes, one-minute bars built from them, and the indicators the strategy reads."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
SESSION_OPEN, SESSION_CLOSE = time(9, 30), time(16, 0)


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    time: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else math.inf


@dataclass
class Bar:
    time: datetime  # start of the minute (UTC)
    open: float
    high: float
    low: float
    close: float


@dataclass
class Features:
    """Indicator snapshot for one share at the close of a one-minute bar."""

    symbol: str
    close: float
    mean: float  # average close over `lookback` bars
    z: float  # how many standard deviations the close is from that average
    trend: float  # average close over `trend_bars` bars
    trend_rising: bool
    channel_high: float  # highest high of the previous `lookback` bars
    unit: float  # average one-minute range as a fraction of price: the strategy's volatility unit
    session_return: float  # change since the first bar of today's session


def session_bounds(now: datetime) -> tuple[datetime, datetime]:
    """Today's regular US session (UTC). Holidays and half-days are not modelled."""
    local = now.astimezone(NEW_YORK)
    start = local.replace(hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute, second=0, microsecond=0)
    end = local.replace(hour=SESSION_CLOSE.hour, minute=SESSION_CLOSE.minute, second=0, microsecond=0)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def in_session(now: datetime) -> bool:
    start, end = session_bounds(now)
    return now.astimezone(NEW_YORK).weekday() < 5 and start <= now < end


def next_session_open(now: datetime) -> datetime:
    day = now
    for _ in range(8):
        start, end = session_bounds(day)
        if day.astimezone(NEW_YORK).weekday() < 5 and now < end:
            return start
        day = (day.astimezone(NEW_YORK) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return now


class BarSeries:
    """Builds one-minute bars for one share from a stream of quotes (mid prices)."""

    def __init__(self, symbol: str, maxlen: int = 240):
        self.symbol = symbol
        self.bars: deque[Bar] = deque(maxlen=maxlen)
        self.current: Optional[Bar] = None
        self.session_open: Optional[float] = None

    def reset_session(self) -> None:
        self.bars.clear()
        self.current = None
        self.session_open = None

    def seed(self, bars: list[Bar]) -> None:
        """Warm-up history (oldest first), e.g. today's candles fetched at start-up."""
        for bar in bars:
            if self.session_open is None:
                self.session_open = bar.open
            self.bars.append(bar)

    def update(self, quote: Quote, when: Optional[datetime] = None) -> Optional[Bar]:
        """Feed a quote; returns the bar that just completed when the minute rolls over.

        `when` is the time the quote was observed (defaults to the quote's own timestamp).
        """
        minute = (when or quote.time).replace(second=0, microsecond=0)
        price = quote.mid
        done = None
        if self.current is not None and minute > self.current.time:
            done = self.current
            self.bars.append(done)
            self.current = None
        if self.current is None:
            if self.bars and minute <= self.bars[-1].time:
                return done  # late or duplicate quote for a minute already closed
            self.current = Bar(minute, price, price, price, price)
            if self.session_open is None:
                self.session_open = price
        else:
            bar = self.current
            bar.high, bar.low, bar.close = max(bar.high, price), min(bar.low, price), price
        return done

    def features(self, lookback: int, trend_bars: int) -> Optional[Features]:
        need = trend_bars + 10
        if len(self.bars) < need:
            return None
        bars = list(self.bars)[-need:]
        closes = [b.close for b in bars]
        window = closes[-lookback:]
        mean = sum(window) / lookback
        var = sum((c - mean) ** 2 for c in window) / lookback
        std = math.sqrt(var)
        close = closes[-1]
        trend = sum(closes[-trend_bars:]) / trend_bars
        trend_before = sum(closes[-trend_bars - 10:-10]) / trend_bars
        ranges = [(b.high - b.low) / b.close for b in bars[-lookback:] if b.close > 0]
        unit = sum(ranges) / len(ranges) if ranges else 0.0
        return Features(
            symbol=self.symbol, close=close, mean=mean,
            z=(close - mean) / std if std > 0 else 0.0,
            trend=trend, trend_rising=trend > trend_before,
            channel_high=max(b.high for b in bars[-lookback - 1:-1]),
            unit=unit,
            session_return=close / self.session_open - 1 if self.session_open else 0.0,
        )
