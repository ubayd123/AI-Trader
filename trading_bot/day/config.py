"""Settings for the intraday bot. Every value can be overridden with a DAY_* environment variable.

Two brokers are supported:
  ig     IG spread betting (the default). Leveraged: sizes are in pounds per point.
  etoro  eToro real shares, no leverage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

# Full bid/ask spread in basis points (1bp = 0.01%), measured on eToro during the
# regular US session on 10 Nov 2026. Used by the backtest and the dry run; live
# trading always uses the real quote. Symbols not listed fall back to default_spread_bp.
MEASURED_SPREAD_BP = {
    "GOOGL": 0.9, "NVDA": 1.0, "AAPL": 1.2, "AMZN": 1.5, "MSFT": 1.8, "INTC": 2.3, "TSLA": 2.5,
    "META": 2.9, "AVGO": 3.5, "MU": 3.5, "AMD": 4.1, "PLTR": 4.3, "SOFI": 6.2, "COIN": 7.8,
}
# Liquid US shares, tightest eToro spread first.
ALL_SYMBOLS = "GOOGL,NVDA,AAPL,AMZN,MSFT,INTC,TSLA,META,AVGO,MU,AMD,PLTR"

# IG spread-betting markets (daily funded bets). The live bot reads the real minimum size, margin and
# spread from IG for each market; the figures here are IG's published standard values and are only
# used by the backtest and the dry run. "yahoo" is the price source for those, "scale" converts it
# to IG points (1 point = 1 index point, or 1 pip for FX).
IG_MARKETS = {
    "US500": {"epic": "IX.D.SPTRD.DAILY.IP", "yahoo": "^GSPC", "scale": 1, "spread": 0.4, "min_size": 0.5, "margin": 0.05},
    "USTECH": {"epic": "IX.D.NASDAQ.CASH.IP", "yahoo": "^NDX", "scale": 1, "spread": 1.0, "min_size": 0.5, "margin": 0.05},
    "EURUSD": {"epic": "CS.D.EURUSD.TODAY.IP", "yahoo": "EURUSD=X", "scale": 10000, "spread": 0.6, "min_size": 0.5, "margin": 0.0333},
    "GBPUSD": {"epic": "CS.D.GBPUSD.TODAY.IP", "yahoo": "GBPUSD=X", "scale": 10000, "spread": 0.9, "min_size": 0.5, "margin": 0.0333},
    "WALLST": {"epic": "IX.D.DOW.DAILY.IP", "yahoo": "^DJI", "scale": 1, "spread": 2.4, "min_size": 0.5, "margin": 0.05},
    "FTSE": {"epic": "IX.D.FTSE.DAILY.IP", "yahoo": "^FTSE", "scale": 1, "spread": 1.0, "min_size": 0.5, "margin": 0.05},
}
IG_DEFAULT = ["US500", "USTECH", "EURUSD", "GBPUSD"]

# rapid:     what was asked for, about 100 trades a day. To trade that often the bot has to act on
#            weak signals.
# selective: only sharp drops, far fewer trades. Lost far less in testing.
PRESETS = {
    "rapid": {"z_entry": 1.0, "etoro_symbols": ALL_SYMBOLS.split(",")[:5]},
    "selective": {"z_entry": 3.0, "etoro_symbols": ALL_SYMBOLS.split(",")},
}


@dataclass
class DayConfig:
    # --- money ---
    broker: str = "ig"  # ig | etoro
    budget_gbp: float = 50.0  # IG: used as is. eToro: converted to USD once (eToro accounts are in USD)
    budget_usd: float = 0.0  # eToro only: set this instead to fix the budget in dollars
    preset: str = "rapid"  # rapid | selective
    symbols: list[str] = field(default_factory=lambda: list(IG_DEFAULT))
    min_position_usd: float = 10.0  # eToro's minimum trade
    max_positions: int = 6
    cash_reserve_pct: float = 0.03
    risk_per_trade_pct: float = 0.0025  # leveraged brokers: a stop-out costs at most 0.25% of the account

    # --- the limits you asked for ---
    max_trades_per_day: int = 100  # buys + sells
    daily_loss_limit_pct: float = 0.02  # stop for the day once down 2%
    max_total_drawdown_pct: float = 0.10  # stop completely 10% below the peak, until `day reset`

    # --- session (New York time) ---
    skip_open_minutes: int = 10  # spreads are wide just after the open
    stop_entries_before_close: int = 20  # minutes
    flatten_before_close: int = 5  # minutes: everything is sold, nothing is held overnight

    # --- strategy (see day/strategy.py) ---
    mode: str = "revert"  # revert | dip | breakout
    lookback: int = 20  # bars for the mean / z-score / channel
    trend_bars: int = 60  # bars for the trend average
    z_entry: float = 1.0  # how far below its 20-minute average a share must be (in standard deviations)
    stop_mult: float = 8.0  # stop distance, in units of recent 1-minute range
    target_mult: float = 8.0
    min_stop_pct: float = 0.0015
    max_stop_pct: float = 0.0100
    exit_at_mean: bool = True  # dip/revert: sell once the price is back at its recent average
    max_hold_minutes: int = 30
    cooldown_minutes: int = 10  # per share, after a losing trade
    max_spread_pct: float = 0.0008  # never buy when the live spread is wider than 0.08%
    max_quote_age_seconds: int = 20

    # --- costs for the backtest / dry run ---
    default_spread_bp: float = 5.0
    slippage_bp: float = 0.5  # per side, on top of the spread

    # --- execution ---
    etoro_env: str = "demo"  # demo | real (the account type, for either broker)
    dry_run: bool = False  # Yahoo prices + simulated fills; needs no eToro keys
    poll_seconds: float = 5.0
    sync_seconds: float = 10.0
    state_dir: str = "trading_bot/data"

    # --- optional Claude pre-market screen (one call a day) ---
    ai_enabled: bool = True
    ai_model: str = "claude-opus-5-5"

    def validate(self) -> "DayConfig":
        self.symbols = [s.strip().upper() for s in self.symbols if s.strip()]
        if not self.symbols:
            raise ValueError("No symbols configured")
        if self.broker not in ("ig", "etoro"):
            raise ValueError("broker must be 'ig' or 'etoro'")
        if self.preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}")
        if not 0 < self.risk_per_trade_pct <= self.daily_loss_limit_pct:
            raise ValueError("risk_per_trade_pct must be positive and no larger than daily_loss_limit_pct")
        if self.etoro_env not in ("demo", "real"):
            raise ValueError("etoro_env must be 'demo' or 'real'")
        if self.mode not in ("dip", "breakout", "revert"):
            raise ValueError("mode must be dip, breakout or revert")
        if not 0 < self.daily_loss_limit_pct < 1 or not 0 < self.max_total_drawdown_pct < 1:
            raise ValueError("loss limits must be between 0 and 1")
        if self.max_positions < 1 or self.max_trades_per_day < 2:
            raise ValueError("max_positions must be >= 1 and max_trades_per_day >= 2")
        if self.min_stop_pct <= 0 or self.max_stop_pct < self.min_stop_pct:
            raise ValueError("stop limits are inconsistent")
        if self.trend_bars < self.lookback:
            raise ValueError("trend_bars must be at least lookback")
        return self

    def spread_pct(self, symbol: str) -> float:
        """Share spreads for the eToro backtest (IG markets use points: see IG_MARKETS)."""
        return MEASURED_SPREAD_BP.get(symbol, self.default_spread_bp) / 10_000

    @property
    def currency(self) -> str:
        return "£" if self.broker == "ig" else "$"

    @property
    def label(self) -> str:
        return f"{self.broker}_{'dry' if self.dry_run else self.etoro_env}"

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir) / f"day_state_{self.label}.json"

    @property
    def trades_path(self) -> Path:
        return Path(self.state_dir) / f"day_trades_{self.label}.jsonl"

    @classmethod
    def from_env(cls, **overrides) -> "DayConfig":
        """Preset first, then DAY_* environment variables, then explicit overrides."""
        cfg = cls()
        preset = overrides.get("preset") or os.environ.get("DAY_PRESET") or cfg.preset
        broker = (overrides.get("broker") or os.environ.get("DAY_BROKER") or cfg.broker).strip().lower()
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}")
        if broker not in ("ig", "etoro"):
            raise ValueError("broker must be 'ig' or 'etoro'")
        cfg.preset, cfg.broker = preset, broker
        cfg.z_entry = PRESETS[preset]["z_entry"]
        cfg.symbols = list(IG_DEFAULT) if broker == "ig" else list(PRESETS[preset]["etoro_symbols"])
        for f in fields(cls):
            raw = os.environ.get(f"DAY_{f.name.upper()}")
            if raw is None or raw == "":
                continue
            current = getattr(cfg, f.name)
            if f.name == "symbols":
                value = raw.split(",")
            elif isinstance(current, bool):
                value = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw.strip()
            setattr(cfg, f.name, value)
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg.validate()
