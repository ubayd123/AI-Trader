"""Approximate quotes from Yahoo for the dry run (no eToro keys needed)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .bars import Bar, Quote, session_bounds
from .config import IG_MARKETS, DayConfig

log = logging.getLogger(__name__)


def spread_fraction(cfg: DayConfig, symbol: str, price: float) -> float:
    from .sim import spread_fraction as _spread

    return _spread(cfg, symbol, price)


class YahooFeed:
    """Latest one-minute prices from Yahoo, with the broker's typical spread applied around them.

    Good enough to watch the bot work. These are not the broker's prices: use the demo account for a real test.
    """

    def __init__(self, cfg: DayConfig):
        self.cfg = cfg
        self._frames: dict = {}

    def _refresh(self) -> None:
        import yfinance as yf

        for symbol in self.cfg.symbols:
            try:
                market = IG_MARKETS.get(symbol, {}) if self.cfg.broker == "ig" else {}
                frame = yf.Ticker(market.get("yahoo", symbol)).history(period="1d", interval="1m", prepost=False, auto_adjust=False)
                if frame is not None and not frame.empty:
                    self._frames[symbol] = frame[["Open", "High", "Low", "Close"]] * market.get("scale", 1)
            except Exception as exc:
                log.debug("Yahoo data for %s unavailable: %s", symbol, exc)

    def quotes(self, now: datetime) -> dict[str, Quote]:
        self._refresh()
        out = {}
        for symbol, frame in self._frames.items():
            stamp = frame.index[-1].to_pydatetime().astimezone(timezone.utc)
            if now - stamp > timedelta(minutes=3):
                continue  # stale: the engine must not trade on it
            price = float(frame["Close"].iloc[-1])
            half = spread_fraction(self.cfg, symbol, price) / 2
            out[symbol] = Quote(symbol, price * (1 - half), price * (1 + half), now)
        return out

    def history(self, symbol: str, minutes: int = 150) -> list[Bar]:
        if symbol not in self._frames:
            self._refresh()
        frame = self._frames.get(symbol)
        if frame is None:
            return []
        rows = frame.tail(minutes + 1).iloc[:-1]  # drop the minute still forming
        return [Bar(t.to_pydatetime().astimezone(timezone.utc), float(r.Open), float(r.High), float(r.Low), float(r.Close))
                for t, r in rows.iterrows()]

    def gbpusd(self) -> float:
        import yfinance as yf

        return float(yf.Ticker("GBPUSD=X").history(period="5d")["Close"].dropna().iloc[-1])


def todays_bars(bars: list[Bar], now: datetime) -> list[Bar]:
    """Only completed bars from today's regular session (so indicators never mix in yesterday)."""
    start, end = session_bounds(now)
    current = now.replace(second=0, microsecond=0)
    return [b for b in bars if start <= b.time < min(end, current)]
