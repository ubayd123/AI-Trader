"""Decision rules: long-only trend following with hard risk limits.

The same two functions drive live trading and the backtest, so what you
backtest is what runs. The optional AI advisor can veto entries, rank
candidates and ask for exits; it can never widen these limits.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from .config import Config
from .models import Order, Portfolio, Snapshot

Advice = dict[str, dict]  # symbol -> {"action": buy|hold|sell|avoid, "confidence": float, "reason": str}


def _confident(advice: Optional[Advice], symbol: str, action: str, cfg: Config) -> Optional[str]:
    a = (advice or {}).get(symbol)
    if a and a.get("action") == action and float(a.get("confidence", 0)) >= cfg.ai_min_confidence:
        return str(a.get("reason", "")).strip() or action
    return None


def update_high_water(portfolio: Portfolio, snaps: dict[str, Snapshot]) -> None:
    for symbol, pos in portfolio.positions.items():
        if symbol in snaps:
            pos.high_water = max(pos.high_water, snaps[symbol].price)


def circuit_breaker(portfolio: Portfolio, equity: float, cfg: Config, today: date) -> bool:
    """Drawdown kill switch. Returns True while the bot must be flat (sell everything, no entries).

    It trips when equity falls max_drawdown_pct below its peak, then pauses
    trading for halt_days (0 = until `reset-halt` is run) and restarts with
    drawdown measured from the equity at that point.
    """
    if portfolio.halted:
        expired = cfg.halt_days > 0 and portfolio.halted_until and today.isoformat() >= portfolio.halted_until
        if not (expired and not portfolio.positions):
            return True
        portfolio.halted, portfolio.halt_reason, portfolio.halted_until = False, "", ""
        portfolio.peak_equity = equity
    portfolio.peak_equity = max(portfolio.peak_equity, equity)
    if equity > portfolio.peak_equity * (1 - cfg.max_drawdown_pct):
        return False
    portfolio.halted = True
    portfolio.halted_until = (today + timedelta(days=cfg.halt_days)).isoformat() if cfg.halt_days > 0 else ""
    portfolio.halt_reason = (f"{today}: equity £{equity:.2f} fell {cfg.max_drawdown_pct:.0%} below its peak of "
                             f"£{portfolio.peak_equity:.2f}; paused "
                             + (f"until {portfolio.halted_until}" if portfolio.halted_until else "until reset-halt"))
    return True


def exit_orders(portfolio: Portfolio, snaps: dict[str, Snapshot], cfg: Config, today: date,
                advice: Optional[Advice] = None, liquidate: bool = False) -> list[Order]:
    orders = []
    for symbol, pos in portfolio.positions.items():
        snap = snaps.get(symbol)
        if snap is None:
            continue  # no fresh price: never trade blind
        reason = None
        stopped = False
        if liquidate:
            reason = "kill switch: max drawdown breached"
        elif snap.price <= pos.avg_price * (1 - cfg.stop_loss_pct):
            reason, stopped = f"stop-loss ({cfg.stop_loss_pct:.0%} below entry)", True
        elif snap.price <= pos.high_water * (1 - cfg.trailing_stop_pct):
            reason, stopped = f"trailing stop ({cfg.trailing_stop_pct:.0%} off high)", True
        elif snap.trend_broken(cfg.exit_buffer_pct):
            reason = f"trend broken (price below its {cfg.sma_slow}-day average)"
        else:
            ai = _confident(advice, symbol, "sell", cfg)
            if ai:
                reason = f"AI exit: {ai}"
        if reason:
            if stopped:
                portfolio.cooldowns[symbol] = (today + timedelta(days=cfg.cooldown_days)).isoformat()
            orders.append(Order("sell", symbol, pos.market, reason, qty=pos.qty))
    return orders


def candidates(portfolio: Portfolio, snaps: dict[str, Snapshot], cfg: Config, today: date) -> list[Snapshot]:
    """Instruments that pass the entry rules, best first."""
    out = []
    for symbol, snap in snaps.items():
        if symbol in portfolio.positions:
            continue
        if portfolio.cooldowns.get(symbol, "") > today.isoformat():
            continue
        if snap.trend_ok and snap.score > 0 and snap.rsi < cfg.rsi_max_entry:
            out.append(snap)
    return sorted(out, key=lambda s: s.score, reverse=True)


def entry_orders(portfolio: Portfolio, snaps: dict[str, Snapshot], cfg: Config, today: date,
                 advice: Optional[Advice] = None) -> list[Order]:
    if portfolio.halted:
        return []
    picks = candidates(portfolio, snaps, cfg, today)
    if advice is not None:
        # With an advisor, a candidate needs an explicit confident "buy".
        picks = [s for s in picks if _confident(advice, s.symbol, "buy", cfg)]
        picks.sort(key=lambda s: float(advice[s.symbol]["confidence"]), reverse=True)

    prices = {s: snap.price for s, snap in snaps.items()}
    equity = portfolio.equity(prices)
    reserve = equity * cfg.cash_reserve_pct
    slot = (equity - reserve) / cfg.max_positions
    spendable = portfolio.cash - reserve
    free_slots = cfg.max_positions - len(portfolio.positions)

    orders = []
    for snap in picks:
        if free_slots <= 0:
            break
        amount = min(slot, spendable)
        if amount < cfg.min_trade_gbp:
            break
        why = _confident(advice, snap.symbol, "buy", cfg) if advice is not None else None
        reason = f"AI entry: {why}" if why else f"uptrend, momentum score {snap.score:.2f}, RSI {snap.rsi:.0f}"
        orders.append(Order("buy", snap.symbol, snap.market, reason, gbp_amount=round(amount, 2)))
        spendable -= amount
        free_slots -= 1
    return orders
