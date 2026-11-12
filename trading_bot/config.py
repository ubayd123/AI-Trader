"""Bot configuration. Every value can be overridden with a BOT_* environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

# Large, liquid US shares across sectors. All can be bought in fractions by UK investors
# (US-listed ETFs such as SPY cannot: they lack the UK retail disclosure documents).
# Crypto is supported ("crypto:BTC") but off by default: ~1% each-way costs are heavy at £50.
DEFAULT_WATCHLIST = ",".join(f"us-stock:{s}" for s in (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ", "WMT", "COST", "V"))
MARKETS = ("us-stock", "crypto")


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (no dependency). Existing environment variables win."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_watchlist(text: str) -> list[tuple[str, str]]:
    """'us-stock:AAPL,crypto:BTC' -> [('us-stock', 'AAPL'), ('crypto', 'BTC')]"""
    out: list[tuple[str, str]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        market, sep, symbol = item.partition(":")
        if not sep:
            market, symbol = "us-stock", market
        market, symbol = market.strip().lower(), symbol.strip().upper()
        if market not in MARKETS:
            raise ValueError(f"Unsupported market '{market}' in watchlist (use one of {MARKETS})")
        if (market, symbol) not in out:
            out.append((market, symbol))
    if not out:
        raise ValueError("Watchlist is empty")
    return out


@dataclass
class Config:
    # --- capital ---
    starting_cash_gbp: float = 50.0
    watchlist: list[tuple[str, str]] = field(default_factory=lambda: parse_watchlist(DEFAULT_WATCHLIST))

    # --- strategy ---
    sma_fast: int = 20
    sma_slow: int = 50
    rsi_max_entry: float = 75.0  # do not chase overbought names
    exit_buffer_pct: float = 0.03  # exit when price is 3% below the slow average
    regime_filter: bool = True  # no new buys while the S&P 500 is below its 200-day average
    regime_sma: int = 200
    vol_target: float = 0.0  # >0: shrink positions in shares more volatile than this (annualised)

    # --- risk limits (enforced in code; the AI cannot override them) ---
    max_positions: int = 3
    cash_reserve_pct: float = 0.05  # always keep 5% in cash
    min_trade_gbp: float = 2.0  # below this, fees/rounding make a trade pointless
    stop_loss_pct: float = 0.10  # sell if 10% below average entry
    trailing_stop_pct: float = 0.20  # sell if 20% below the high since entry
    max_drawdown_pct: float = 0.25  # kill switch: sell everything and pause trading
    halt_days: int = 30  # length of that pause; 0 = stay stopped until `reset-halt`
    cooldown_days: int = 5  # no re-entry for N days after a stop-out
    enforce_market_hours: bool = True

    # --- costs (used by the paper broker and backtest) ---
    stock_fee_pct: float = 0.0015  # Trading 212 FX fee on USD shares
    crypto_fee_pct: float = 0.01  # typical UK retail crypto spread+fee
    slippage_pct: float = 0.0005

    # --- execution ---
    broker: str = "paper"  # paper | trading212
    t212_env: str = "demo"  # demo | live
    t212_quantity_decimals: int = 3
    state_dir: str = "trading_bot/data"

    # --- AI advisor (optional; needs Anthropic credentials) ---
    ai_enabled: bool = True
    ai_model: str = "claude-opus-5-5"
    ai_min_confidence: float = 0.55
    ai_news: bool = True  # show Claude recent headlines for candidates and holdings

    # --- AI-Trader platform sync (optional) ---
    platform_url: str = "https://ai4trade.ai/api"

    def validate(self) -> "Config":
        if self.starting_cash_gbp <= 0:
            raise ValueError("starting_cash_gbp must be positive")
        if self.max_positions < 1:
            raise ValueError("max_positions must be >= 1")
        for name in ("cash_reserve_pct", "stop_loss_pct", "trailing_stop_pct", "max_drawdown_pct"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.broker not in ("paper", "trading212"):
            raise ValueError("broker must be 'paper' or 'trading212'")
        if self.t212_env not in ("demo", "live"):
            raise ValueError("t212_env must be 'demo' or 'live'")
        if self.sma_fast >= self.sma_slow:
            raise ValueError("sma_fast must be smaller than sma_slow")
        return self

    def fee_pct(self, market: str) -> float:
        return self.crypto_fee_pct if market == "crypto" else self.stock_fee_pct

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir) / f"state_{self.broker}.json"

    @property
    def trades_path(self) -> Path:
        return Path(self.state_dir) / f"trades_{self.broker}.jsonl"

    @property
    def equity_path(self) -> Path:
        return Path(self.state_dir) / f"equity_{self.broker}.jsonl"

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for f in fields(cls):
            raw = os.environ.get(f"BOT_{f.name.upper()}")
            if raw is None or raw == "":
                continue
            current = getattr(cfg, f.name)
            if f.name == "watchlist":
                value = parse_watchlist(raw)
            elif isinstance(current, bool):
                value = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw.strip()
            setattr(cfg, f.name, value)
        return cfg.validate()
