"""Command line: python -m trading_bot <command>"""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import time
from pathlib import Path

from .config import Config, load_dotenv

log = logging.getLogger("trading_bot")


def _engine(cfg: Config, args, with_ai: bool = True):
    from .advisor import make_advisor
    from .brokers import make_broker
    from .data import MarketData
    from .engine import Engine
    from .platform_client import AITraderClient

    token = os.environ.get("AI4TRADE_TOKEN", "")
    platform = AITraderClient(cfg.platform_url, token) if token else None
    advisor = make_advisor(cfg) if with_ai and not getattr(args, "no_ai", False) else None
    return Engine(cfg, make_broker(cfg, getattr(args, "confirm_live", False)), MarketData(cfg), advisor, platform)


def _print_report(report: dict) -> None:
    mode = report["broker"].upper()
    print(f"\n[{mode}] equity £{report['equity_gbp']:.2f} | cash £{report['cash_gbp']:.2f} | "
          f"return {report['return_pct']:+.2f}% | realised P&L £{report['realized_pnl_gbp']:+.2f} | "
          f"fees £{report['fees_paid_gbp']:.2f}")
    for symbol, p in report["positions"].items():
        print(f"  {symbol:<6} {p['qty']:.6f} @ £{p['avg_price']:.2f}  now worth £{p['value_gbp']:.2f}")
    if not report["positions"]:
        print("  (no open positions)")
    if report["halted"]:
        print(f"  HALTED: {report['halt_reason']}")


def cmd_run(cfg, args):
    report = _engine(cfg, args).run_cycle()
    if not report["trades"]:
        print("No trades this cycle.")
    _print_report(report)


def cmd_loop(cfg, args):
    engine = _engine(cfg, args)
    print(f"Running every {args.interval}s. Ctrl+C to stop.")
    while True:
        try:
            _print_report(engine.run_cycle())
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # keep the loop alive through data/network hiccups
            log.error("Cycle failed: %s", exc)
        time.sleep(args.interval)


def cmd_status(cfg, args):
    engine = _engine(cfg, args, with_ai=False)
    try:
        prices = {s: v.price for s, v in engine.data.snapshots().items()}
    except Exception as exc:
        log.warning("Live prices unavailable (%s); valuing positions at cost", exc)
        prices = {}
    _print_report(engine.status(prices))


def cmd_backtest(cfg, args):
    from .backtest import run_backtest
    from .data import MarketData

    data = MarketData(cfg, period=args.period)
    series = {}
    for market, symbol in cfg.watchlist:
        closes = data.closes_gbp(market, symbol)
        if not closes.empty:
            series[symbol] = (market, closes)
    result = run_backtest(cfg, series, data.index_closes() if cfg.regime_filter else None)
    print(json.dumps(result, indent=2))
    print("\nPast performance does not predict future returns. This backtest ignores the AI advisor.")


def cmd_report(cfg, args):
    from .report import build_report

    report = build_report(cfg)
    if not report.get("cycles"):
        print("No history yet. Run `python -m trading_bot run` first.")
        return
    print(json.dumps(report, indent=2))


def cmd_reset_halt(cfg, args):
    engine = _engine(cfg, args, with_ai=False)
    pf = engine.portfolio
    pf.halted, pf.halt_reason, pf.halted_until = False, "", ""
    pf.peak_equity = pf.cash + sum(p.qty * p.avg_price for p in pf.positions.values())
    pf.save(cfg.state_path)
    print("Kill switch cleared. Drawdown is now measured from the current equity.")


def cmd_register(cfg, args):
    from .data import MarketData
    from .platform_client import AITraderClient

    password = secrets.token_urlsafe(18)
    usd = cfg.starting_cash_gbp * float(MarketData(cfg, period="5d").gbpusd().iloc[-1])
    result = AITraderClient(cfg.platform_url).register(args.name, password, round(usd, 2))
    print(f"Registered '{args.name}' on {cfg.platform_url} with ${usd:.2f} of simulated capital.")
    print("Add these to your .env (they are shown once):\n")
    print(f"AI4TRADE_TOKEN={result['token']}\nAI4TRADE_AGENT_PASSWORD={password}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="trading_bot", description="£50 AI trading bot (UK): swing mode, and `day` for rapid eToro trading")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, text in (("run", cmd_run, "run one trading cycle"), ("loop", cmd_loop, "run cycles forever")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--no-ai", action="store_true", help="rules only, skip the Claude advisor")
        p.add_argument("--confirm-live", action="store_true", help="required for Trading 212 live (real money)")
        p.set_defaults(fn=fn)
        if name == "loop":
            p.add_argument("--interval", type=int, default=3600, help="seconds between cycles")
    sub.add_parser("status", help="show the bot's portfolio").set_defaults(fn=cmd_status)
    p = sub.add_parser("backtest", help="backtest the rules on history")
    p.add_argument("--period", default="3y", help="history to test, e.g. 1y, 3y, 5y")
    p.set_defaults(fn=cmd_backtest)
    sub.add_parser("report", help="performance so far: return, drawdown, win rate, fees").set_defaults(fn=cmd_report)
    sub.add_parser("reset-halt", help="clear the drawdown kill switch").set_defaults(fn=cmd_reset_halt)
    p = sub.add_parser("register", help="register the bot on the AI-Trader platform")
    p.add_argument("name")
    p.set_defaults(fn=cmd_register)

    from .day.cli import add_parser as add_day_parser

    add_day_parser(sub)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("yfinance", "urllib3", "httpx", "anthropic", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv(Path("trading_bot/.env"))
    load_dotenv(Path(".env"))
    try:
        if args.command == "day":
            args.fn(args)
        else:
            args.fn(Config.from_env(), args)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        if args.verbose:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
