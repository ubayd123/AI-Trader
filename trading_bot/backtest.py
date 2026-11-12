"""Backtest the rule-based strategy (no AI) on daily closes with the same code as live trading.

Signals use each day's close and fill at that close plus slippage and fees,
which matches running the bot shortly before the US close (about 20:45 UK).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from .brokers import PaperBroker
from .config import Config
from .indicators import MIN_BARS, snapshot
from .models import Portfolio
from .strategy import circuit_breaker, entry_orders, exit_orders, regime_ok, start_cooldown, update_high_water


def run_backtest(cfg: Config, series: dict[str, tuple[str, pd.Series]], index_closes: Optional[pd.Series] = None,
                 start: Optional[str] = None, end: Optional[str] = None) -> dict:
    """series: symbol -> (market, GBP daily closes). index_closes: S&P 500 closes for the regime filter.

    start/end (ISO dates) restrict the trading window; earlier history is still used to warm up indicators.
    """
    broker = PaperBroker(cfg)
    pf = Portfolio(cash=cfg.starting_cash_gbp, peak_equity=cfg.starting_cash_gbp, starting_cash=cfg.starting_cash_gbp)
    calendar = sorted(set().union(*(s.index for _, s in series.values())))
    warmup = max(MIN_BARS, cfg.sma_slow + 1)
    last_price: dict[str, float] = {}
    curve, bench, sells, n_trades, halts, invested, start_prices = [], [], [], 0, 0, 0, None

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
        if not snaps or (start and day < pd.Timestamp(start)) or (end and day > pd.Timestamp(end)):
            continue
        if start_prices is None:
            start_prices = dict(last_price)

        today = day.date()
        update_high_water(pf, snaps)
        was_halted = pf.halted
        circuit_breaker(pf, pf.equity(last_price), cfg, today)
        halts += pf.halted and not was_halted
        orders = exit_orders(pf, snaps, cfg, today, liquidate=pf.halted)
        for order in orders:
            sells.append(pf.apply_sell(order, broker.execute(order, snaps[order.symbol])))
            start_cooldown(pf, order, cfg, today)
        market_ok = regime_ok(index_closes.loc[:day] if index_closes is not None else None, cfg)
        buys = entry_orders(pf, snaps, cfg, today, regime_ok=market_ok)
        for order in buys:
            pf.apply_buy(order, broker.execute(order, snaps[order.symbol]), datetime.combine(today, datetime.min.time()).isoformat())
        n_trades += len(orders) + len(buys)
        invested += bool(pf.positions)
        curve.append((day, pf.equity(last_price)))
        bench.append(sum(last_price[s] / p for s, p in start_prices.items()) / len(start_prices))

    if not curve:
        raise ValueError("Not enough price history to backtest")
    equity = pd.Series(dict(curve))
    hold = pd.Series(bench, index=equity.index)
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
        "buy_and_hold_final_gbp": round(cfg.starting_cash_gbp * float(hold.iloc[-1]), 2),
        "buy_and_hold_return_pct": round((float(hold.iloc[-1]) - 1) * 100, 1),
        "buy_and_hold_max_drawdown_pct": round(float((hold / hold.cummax() - 1).min()) * 100, 1),
        "days_invested_pct": round(100 * invested / len(curve)),
    }
