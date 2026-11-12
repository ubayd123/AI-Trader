"""Market data via yfinance, converted to GBP."""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .config import Config
from .indicators import snapshot
from .models import Snapshot

log = logging.getLogger(__name__)
NEW_YORK = ZoneInfo("America/New_York")
MAX_STALE_DAYS = 5


def is_market_open(market: str, now: Optional[datetime] = None) -> bool:
    """Crypto trades 24/7. US stocks: Mon-Fri 09:30-16:00 New York (14:30-21:00 UK). Holidays not modelled."""
    if market == "crypto":
        return True
    local = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    return local.weekday() < 5 and time(9, 30) <= local.time() < time(16, 0)


def _daily_closes(ticker: str, period: str) -> pd.Series:
    import yfinance as yf

    frame = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    closes = frame["Close"].dropna()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    return closes[~closes.index.duplicated(keep="last")]


class MarketData:
    def __init__(self, cfg: Config, period: str = "1y"):
        self.cfg = cfg
        self.period = period
        self._fx: Optional[pd.Series] = None

    def gbpusd(self) -> pd.Series:
        if self._fx is None:
            self._fx = _daily_closes("GBPUSD=X", self.period)
            if self._fx.empty:
                raise RuntimeError("Could not fetch the GBP/USD rate")
        return self._fx

    def closes_gbp(self, market: str, symbol: str) -> pd.Series:
        """Daily closes in GBP. The last bar is the latest price when the market is open."""
        if market == "crypto":
            return _daily_closes(f"{symbol}-GBP", self.period)
        usd = _daily_closes(symbol.replace(".", "-"), self.period)
        if usd.empty:
            return usd
        fx = self.gbpusd().reindex(usd.index.union(self.gbpusd().index)).ffill().reindex(usd.index)
        return (usd / fx).dropna()

    def snapshots(self, today: Optional[datetime] = None) -> dict[str, Snapshot]:
        today = today or datetime.now(timezone.utc)
        fx_now = float(self.gbpusd().iloc[-1])
        out: dict[str, Snapshot] = {}
        for market, symbol in self.cfg.watchlist:
            try:
                closes = self.closes_gbp(market, symbol)
            except Exception as exc:  # one bad ticker must not stop the cycle
                log.warning("No data for %s: %s", symbol, exc)
                continue
            if closes.empty or (today.replace(tzinfo=None) - closes.index[-1]).days > MAX_STALE_DAYS:
                log.warning("Skipping %s: price data missing or stale", symbol)
                continue
            snap = snapshot(symbol, market, closes, self.cfg, usd_price=float(closes.iloc[-1]) * fx_now)
            if snap:
                out[symbol] = snap
            else:
                log.warning("Skipping %s: not enough price history", symbol)
        return out
