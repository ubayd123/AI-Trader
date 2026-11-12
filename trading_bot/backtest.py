"""Backtest the rule-based strategy (no AI) on daily closes with the same code as live trading.

Signals use each day's close and fill at that close plus slippage and fees,
which matches running the bot shortly before the US close (about 20:45 UK).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from .brokers import PaperBroker
from .config import Config
from .indicators import MIN_BARS, snapshot
from .models import Portfolio
from .strategy import circuit_breaker, entry_orders, exit_orders, update_high_water


def run_backtest(cfg: Config, series: dict[str, tuple[str, pd.Series]]) -> dict:
    """series: symbol -> (market, GBP daily closes)."""
    broker = PaperBroker(cfg)
    pf = Portfolio(cash=cfg.starting_cash_gbp, peak_equity=cfg.starting_cash_gbp, starting_cash=cfg.starting_cash_gbp)
    calendar = sorted(set().union(*(s.index for _, s in series.values())))
    warmup = max(MIN_BARS, cfg.sma_slow + 1)
    last_price: dict[str, float] = {}
    curve, sells, n_trades, halts, start_prices, bench_first = [], [], 0, 0, {}, None

    for day in calendar:
        snaps = {}
        for symbol, (market, closes) in series.items():
            if day not in closes.index:
                continue  # that market is closed today
            hist = closes.loc[:day]
            last_price[symbol] = float(hist.iloc[-1])
            if len(hist) >= warmup:
                snap = snapshot(symbol, market, hist.tail(warmup + 5), cfg)
                if snap:
                    snaps[symbol] = snap
        if not snaps:
            continue
        if bench_first is None:
            bench_first, start_prices = day, dict(last_price)

        today = day.date()
        update_high_water(pf, snaps)
        was_halted = pf.halted
        circuit_breaker(pf, pf.equity(last_price), cfg, today)
        halts += pf.halted and not was_halted
        orders = exit_orders(pf, snaps, cfg, today, liquidate=pf.halted)
        for order in orders:
            sells.append(pf.apply_sell(order, broker.execute(order, snaps[order.symbol])))
        buys = entry_orders(pf, snaps, cfg, today)
        for order in buys:
            pf.apply_buy(order, broker.execute(order, snaps[order.symbol]), datetime.combine(today, datetime.min.time()).isoformat())
        n_trades += len(orders) + len(buys)
        curve.append((day, pf.equity(last_price)))

    if not curve:
        raise ValueError("Not enough price history to backtest")
    equity = pd.Series(dict(curve))
    held = [s for s in start_prices if s in last_price]
    benchmark = sum(last_price[s] / start_prices[s] for s in held) / len(held) - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / cfg.starting_cash_gbp - 1
    return {
        "start": equity.index[0].date().isoformat(), "end": equity.index[-1].date().isoformat(),
        "start_gbp": cfg.starting_cash_gbp, "final_gbp": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(total * 100, 1),
        "annualised_return_pct": round(((1 + total) ** (1 / years) - 1) * 100, 1),
        "max_drawdown_pct": round(float((equity / equity.cummax() - 1).min()) * 100, 1),
        "trades": n_trades, "closed_trades": len(sells),
        "win_rate_pct": round(100 * sum(p > 0 for p in sells) / len(sells), 0) if sells else None,
        "fees_paid_gbp": round(pf.fees_paid, 2), "kill_switch_trips": int(halts),
        "buy_and_hold_watchlist_pct": round(benchmark * 100, 1),
    }
