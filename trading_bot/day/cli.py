"""Commands: python -m trading_bot day <command>"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from .bars import NEW_YORK, Quote, in_session
from .config import IG_MARKETS, PRESETS, DayConfig
from .engine import BrokerError, DayEngine, DayPosition, DayState

log = logging.getLogger("trading_bot.day")


def _config(args) -> DayConfig:
    cfg = DayConfig.from_env(preset=getattr(args, "preset", None), broker=getattr(args, "broker", None),
                             dry_run=bool(getattr(args, "dry_run", False)) or None)
    if getattr(args, "real", False):
        cfg.etoro_env = "real"
    if cfg.broker == "ig":
        cfg.poll_seconds = max(cfg.poll_seconds, 6.0)  # stay well inside IG's 30 requests a minute
        cfg.sync_seconds = max(cfg.sync_seconds, 15.0)
    return cfg


def _env(cfg: DayConfig, name: str) -> str:
    """Credentials: BROKER_REAL_NAME for the real account if set, otherwise BROKER_NAME."""
    prefix = cfg.broker.upper()
    if cfg.etoro_env == "real" and os.environ.get(f"{prefix}_REAL_{name}"):
        return os.environ[f"{prefix}_REAL_{name}"]
    return os.environ.get(f"{prefix}_{name}", "")


def _ig(cfg: DayConfig):
    from .ig import IGBroker, IGClient, IGFeed, resolve_epics

    epics = resolve_epics(cfg.symbols)
    client = IGClient.login(_env(cfg, "USERNAME"), _env(cfg, "PASSWORD"), _env(cfg, "API_KEY"), cfg.etoro_env,
                            _env(cfg, "ACCOUNT_ID"))
    return client, IGFeed(client, epics), IGBroker(cfg, client, epics)


def _etoro(cfg: DayConfig):
    from .etoro import EtoroBroker, EtoroClient, EtoroFeed

    client = EtoroClient(_env(cfg, "API_KEY"), _env(cfg, "USER_KEY"), cfg.etoro_env)
    ids = client.instruments(cfg.symbols)
    missing = [s for s in cfg.symbols if s not in ids]
    if missing:
        log.warning("Not found on eToro, skipping: %s", ", ".join(missing))
        cfg.symbols = [s for s in cfg.symbols if s in ids]
    return client, EtoroFeed(client, ids), EtoroBroker(cfg, client, ids)


def _budget(cfg: DayConfig, args, feed) -> float:
    if getattr(args, "budget", None):
        return float(args.budget)
    if cfg.broker == "ig":
        return cfg.budget_gbp  # IG spread bets are in pounds
    if cfg.budget_usd > 0:
        return cfg.budget_usd
    rate = feed.gbpusd()
    budget = round(cfg.budget_gbp * rate, 2)
    log.info("Budget: £%.2f = $%.2f at %.4f (fixed from now on; eToro accounts are in dollars)", cfg.budget_gbp, budget, rate)
    return budget


def _build(cfg: DayConfig, args, trading: bool = True):
    """Returns (engine, feed). Dry run: Yahoo prices and simulated fills. Otherwise the broker's demo or real account."""
    if cfg.dry_run:
        from .feeds import YahooFeed
        from .sim import SimBroker

        feed, broker = YahooFeed(cfg), SimBroker(cfg)
    else:
        if cfg.etoro_env == "real" and trading and not getattr(args, "confirm_live", False):
            raise BrokerError("Refusing to trade real money without --confirm-live. Use the demo account first.")
        _, feed, broker = _ig(cfg) if cfg.broker == "ig" else _etoro(cfg)
    state = DayState.load(cfg.state_path)
    if state is None:
        state = DayState(budget=_budget(cfg, args, feed))
    elif getattr(args, "budget", None) and abs(float(args.budget) - state.budget) > 0.005:
        log.warning("Ignoring --budget: this ledger already started with %s%.2f. Delete %s to start again.",
                    cfg.currency, state.budget, cfg.state_path)
    return DayEngine(cfg, broker, state), feed


def _print_status(status: dict) -> None:
    cur = status["currency"]
    print(f"\n[{status['env'].upper()}] equity {cur}{status['equity']:.2f} of {cur}{status['budget']:.2f} "
          f"({status['return_pct']:+.2f}%) | today {cur}{status['today_pnl']:+.3f} | {status['trades_today']} trades today")
    for p in status["positions"]:
        print(f"  {p['symbol']:<7} {p['units']:.4f} @ {p['entry']:.2f}  stop {p['stop']:.2f}  target {p['target']:.2f}")
    if not status["positions"]:
        print("  (no open positions)")
    for symbol, why in status.get("blocked", {}).items():
        print(f"  not trading {symbol}: {why}")
    if status["halted"]:
        print(f"  KILL SWITCH: {status['halt_reason']}")
    elif status["halted_today"]:
        print("  Stopped for today: daily loss limit reached.")


def cmd_run(args) -> None:
    from .runner import run

    cfg = _config(args)
    if cfg.dry_run:
        cfg.poll_seconds = max(cfg.poll_seconds, 20.0)  # be gentle with Yahoo
    engine, feed = _build(cfg, args)
    screen = None
    if cfg.ai_enabled and not args.no_ai and cfg.broker == "etoro":  # the news screen is about individual shares
        from ..config import Config
        from ..data import MarketData
        from .screen import premarket_screen

        news = MarketData(Config())
        screen = lambda: premarket_screen(cfg, news.headlines)  # noqa: E731
    name = "IG spread betting" if cfg.broker == "ig" else "eToro"
    mode = "DRY RUN (Yahoo prices, simulated fills)" if cfg.dry_run else f"{name} {cfg.etoro_env.upper()} account"
    print(f"Intraday bot: {mode}, preset '{cfg.preset}', markets {', '.join(cfg.symbols)}.")
    print(f"Budget {cfg.currency}{engine.state.budget:,.2f}. Up to {cfg.max_trades_per_day} trades a day, daily loss limit "
          f"{cfg.daily_loss_limit_pct:.0%}" + (f", {cfg.risk_per_trade_pct:.2%} at risk per trade." if cfg.broker == "ig" else "."))
    print("Ctrl+C stops the bot and sells its open positions.")
    run(cfg, engine, feed, screen)
    _print_status(engine.status())


def cmd_backtest(args) -> None:
    from .sim import load_yahoo_minutes, replay

    cfg = _config(args)
    print(f"Downloading {args.days} days of one-minute prices for {', '.join(cfg.symbols)}...")
    data = load_yahoo_minutes(cfg.symbols, days=args.days)
    if not data:
        raise BrokerError("No usable price history was downloaded.")
    budget = float(args.budget) if args.budget else (cfg.budget_gbp if cfg.broker == "ig" else cfg.budget_usd or cfg.budget_gbp * 1.32)
    result = replay(cfg, data, round(budget, 2))
    print(json.dumps(result, indent=2))
    cur = result["currency"]
    print(f"\nPreset '{cfg.preset}' on {', '.join(data)}: {cur}{result['start']:,.2f} became {cur}{result['final']:,.2f} over "
          f"{result['days']} trading days ({result['total_return_pct']:+.1f}%), {result['trades_per_day']:.0f} trades a day, "
          f"best day {result['best_day_pct']:+.2f}%, worst day {result['worst_day_pct']:+.2f}%.")
    for symbol, why in result["not_traded"].items():
        print(f"Not traded, {symbol}: {why}")
    print("Past results do not predict future ones. Costs use the broker's published minimum spreads plus slippage.")


def _check_ig(cfg: DayConfig, args) -> None:
    client, feed, broker = _ig(cfg)
    funds = client.balance()
    print(f"Logged in to IG {cfg.etoro_env.upper()}: spread betting account {client.account_id}.")
    print(f"Account balance £{float(funds.get('balance') or 0):,.2f}, available £{float(funds.get('available') or 0):,.2f}.")
    budget = float(args.budget) if getattr(args, "budget", None) else cfg.budget_gbp
    risk_cash = budget * cfg.risk_per_trade_pct
    print(f"Bot budget £{budget:,.2f}: at most £{risk_cash:.2f} at risk per trade ({cfg.risk_per_trade_pct:.2%}) and "
          f"£{budget * cfg.daily_loss_limit_pct:.2f} a day ({cfg.daily_loss_limit_pct:.0%}).\n")
    tradable = 0
    for name in cfg.symbols:
        d = broker.details(name, refresh=True)
        instrument, snap = d.get("instrument") or {}, d.get("snapshot") or {}
        spec = broker.spec(name)
        bid, offer = float(snap.get("bid") or 0), float(snap.get("offer") or 0)
        if not offer:
            print(f"  {name:<7} {instrument.get('name', '?')}: no price ({snap.get('marketStatus')})")
            continue
        exposure = spec.min_size * offer
        stop_pct = max(cfg.min_stop_pct, 1.05 * spec.min_stop_distance / offer)
        loss = exposure * (stop_pct + (offer - bid) / offer)
        ok = loss <= risk_cash and exposure * spec.margin_pct <= budget * (1 - cfg.cash_reserve_pct)
        tradable += ok
        print(f"  {name:<7} {instrument.get('name', '?')} [{snap.get('marketStatus')}]  bid {bid:g} / offer {offer:g} "
              f"(spread {offer - bid:.2f} pts, {(offer - bid) / offer:.4%})")
        print(f"          smallest bet £{spec.min_size:g} a point = £{exposure:,.0f} exposure, £{exposure * spec.margin_pct:,.2f} margin; "
              f"a {stop_pct:.2%} stop on it loses £{loss:.2f}  ->  {'OK' if ok else 'TOO BIG for this budget'}")
    if tradable == 0:
        print(f"\nWith a budget of £{budget:,.2f} the bot cannot trade any of these markets inside your limits: IG's "
              "smallest bet is too large. See the README section 'IG and a small account'.")
    else:
        print(f"\n{tradable} of {len(cfg.symbols)} markets can be traded inside your limits.")
    print(f"US session is {'OPEN' if in_session(datetime.now(timezone.utc)) else 'CLOSED'} now (the bot trades 14:40-20:40 UK time).")


def _check_etoro(cfg: DayConfig, args) -> None:
    from .etoro import EtoroClient

    client = EtoroClient(_env(cfg, "API_KEY"), _env(cfg, "USER_KEY"), cfg.etoro_env)
    ids = client.instruments(cfg.symbols)
    print(f"Keys accepted for the eToro {cfg.etoro_env.upper()} account.")
    missing = [s for s in cfg.symbols if s not in ids]
    print(f"Shares found: {len(ids)}/{len(cfg.symbols)}" + (f" (missing: {', '.join(missing)})" if missing else ""))
    cash = float(client.portfolio().get("credit") or 0)
    budget = cfg.budget_usd or cfg.budget_gbp * 1.32
    print(f"Cash available: ${cash:,.2f}" + ("" if cash >= budget else
          f"  <-- less than the bot's budget of about ${budget:.0f}: orders will be rejected until you deposit"))
    for item in client.eligibility(list(ids)):
        real_long = [c for c in item.get("leverageConfigs") or []
                     if str(c.get("settlementType")).lower() == "real" and str(c.get("direction")).lower() == "long"]
        flags = []
        if not item.get("allowOpenPosition"):
            flags.append("opening positions is NOT allowed")
        if not real_long:
            flags.append("cannot be bought as a real share")
        if float(item.get("minPositionExposure") or 0) > cfg.min_position_usd:
            flags.append(f"minimum trade is ${item['minPositionExposure']}")
        if item.get("requiresW8Ben"):
            flags.append("needs a W-8BEN form on your eToro account")
        print(f"  {item.get('symbol'):<6} " + ("; ".join(flags) if flags else "ok"))
    names = {v: k for k, v in ids.items()}
    wide = []
    for instrument_id, rate in client.rates(list(ids.values())).items():
        if rate.get("bid") and rate.get("ask"):
            quote = Quote(names[instrument_id], float(rate["bid"]), float(rate["ask"]), datetime.now(timezone.utc))
            if quote.spread_pct > cfg.max_spread_pct:
                wide.append(f"{quote.symbol} {quote.spread_pct:.3%}")
    open_now = in_session(datetime.now(timezone.utc))
    print(f"US market is {'OPEN' if open_now else 'CLOSED'} now." +
          (f" Spread wider than the {cfg.max_spread_pct:.2%} limit: {', '.join(wide)}" if wide else " All spreads inside the limit.") +
          ("" if open_now else " (Spreads are normally wide outside market hours.)"))


def cmd_check(args) -> None:
    """Read-only: log-in, account, market rules and live spreads. Places no orders."""
    cfg = _config(args)
    (_check_ig if cfg.broker == "ig" else _check_etoro)(cfg, args)


def cmd_test_order(args) -> None:
    """Open the smallest possible position in one market and close it straight away, to prove the order path."""
    cfg = _config(args)
    if cfg.dry_run:
        raise BrokerError("test-order needs the broker's demo (or real) account, not --dry-run")
    engine, feed = _build(cfg, args)
    now = datetime.now(timezone.utc)
    if not in_session(now):
        raise BrokerError("The US session is closed. Run this between 14:30 and 21:00 UK time on a weekday.")
    symbol = cfg.symbols[0]
    quote = feed.quotes(now).get(symbol)
    if quote is None:
        raise BrokerError(f"No live price for {symbol} right now.")
    spec = engine.spec(symbol)
    amount = max(spec.min_notional, spec.min_size * quote.ask)
    stop, target = round(quote.ask * 0.97, 2), round(quote.ask * 1.03, 2)
    cur = cfg.currency
    print(f"Opening the smallest {symbol} position ({cur}{amount:,.2f} exposure) at about {quote.ask:g} on the "
          f"{cfg.broker.upper()} {cfg.etoro_env.upper()} account...")
    fill = engine.broker.open(symbol, amount, stop, target, quote)
    print(f"  opened {fill.units:g} at {fill.price:g} (position {fill.position_id})")
    pos = DayPosition(symbol, fill.position_id, fill.units, fill.price, amount, stop, target, now.isoformat(), "test", fill.order_id)
    closed = engine.broker.close(pos, quote, "test")
    print(f"  closed at {closed.price:g}. Round trip cost {cur}{(fill.price - closed.price) * fill.units:.4f}.")
    print("Order path works: open, confirm, close, confirm.")


def cmd_status(args) -> None:
    cfg = _config(args)
    state = DayState.load(cfg.state_path)
    if state is None:
        print(f"No ledger yet for '{cfg.label}'. Start with: python -m trading_bot day run --dry-run")
        return
    from .sim import SimBroker

    _print_status(DayEngine(cfg, SimBroker(cfg), state, persist=False).status())


def cmd_flatten(args) -> None:
    cfg = _config(args)
    engine, feed = _build(cfg, args)
    if not engine.state.positions:
        print("The bot has no open positions.")
        return
    now = datetime.now(timezone.utc)
    engine.last_quotes.update(feed.quotes(now))
    engine.flatten(now, "manual flatten")
    _print_status(engine.status())


def cmd_reset(args) -> None:
    cfg = _config(args)
    state = DayState.load(cfg.state_path)
    if state is None:
        print("Nothing to reset.")
        return
    state.halted, state.halt_reason, state.halted_today = False, "", False
    state.peak_equity = state.budget + state.realized
    state.save(cfg.state_path)
    print("Kill switch cleared. Drawdown is now measured from the current equity.")


def day_report(trades: list[dict], budget: float) -> dict:
    sells = [t for t in trades if t.get("side") == "sell" and t.get("pnl") is not None]
    days: dict[str, dict] = {}
    for t in trades:
        day = datetime.fromisoformat(t["time"]).astimezone(NEW_YORK).date().isoformat()
        row = days.setdefault(day, {"trades": 0, "pnl": 0.0})
        row["trades"] += 1
        row["pnl"] = round(row["pnl"] + (t.get("pnl") or 0.0), 4)
    wins = [t["pnl"] for t in sells if t["pnl"] > 0]
    losses = [t["pnl"] for t in sells if t["pnl"] <= 0]
    reasons: dict[str, int] = {}
    for t in sells:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    total = round(sum(t["pnl"] for t in sells), 4)
    return {
        "days": len(days), "trades": len(trades), "round_trips": len(sells),
        "realised_pnl": total, "return_pct": round(100 * total / budget, 2) if budget else None,
        "win_rate_pct": round(100 * len(wins) / len(sells)) if sells else None,
        "avg_win": round(sum(wins) / len(wins), 4) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 4) if losses else None,
        "winning_days": sum(1 for d in days.values() if d["pnl"] > 0),
        "exit_reasons": reasons, "by_day": days,
    }


def cmd_report(args) -> None:
    cfg = _config(args)
    if not cfg.trades_path.is_file():
        print(f"No trades recorded yet for '{cfg.label}'.")
        return
    trades = [json.loads(line) for line in cfg.trades_path.read_text().splitlines() if line.strip()]
    state = DayState.load(cfg.state_path)
    print(json.dumps(day_report(trades, state.budget if state else 0.0), indent=2))


def cmd_markets(args) -> None:
    print("IG markets the bot knows by name (use DAY_SYMBOLS=US500,FTSE or any IG epic):")
    for name, market in IG_MARKETS.items():
        print(f"  {name:<7} {market['epic']:<24} min spread {market['spread']} pts, standard minimum bet £{market['min_size']} a point")


def add_parser(sub) -> None:
    day = sub.add_parser("day", help="rapid intraday trading (IG spread betting, or eToro shares)")
    dsub = day.add_subparsers(dest="day_command", required=True)

    def common(p, trading=False):
        p.add_argument("--broker", choices=["ig", "etoro"], help="default: ig")
        p.add_argument("--preset", choices=sorted(PRESETS), help="rapid (trades often) or selective (sharp drops only)")
        p.add_argument("--budget", type=float, help="the bot's budget (pounds on IG, dollars on eToro); default £50")
        p.add_argument("--dry-run", action="store_true", help="Yahoo prices and simulated fills; no broker log-in needed")
        p.add_argument("--real", action="store_true", help="use the real account instead of the demo account")
        if trading:
            p.add_argument("--confirm-live", action="store_true", help="required with --real: trade real money")

    p = dsub.add_parser("run", help="run the intraday bot")
    common(p, trading=True)
    p.add_argument("--no-ai", action="store_true", help="skip the Claude pre-market screen (eToro shares only)")
    p.set_defaults(fn=cmd_run)
    p = dsub.add_parser("backtest", help="replay recent one-minute prices through the bot")
    common(p)
    p.add_argument("--days", type=int, default=29, help="calendar days of history (Yahoo keeps about 30)")
    p.set_defaults(fn=cmd_backtest)
    p = dsub.add_parser("check", help="check log-in, account, market rules and spreads (places no orders)")
    common(p)
    p.set_defaults(fn=cmd_check)
    p = dsub.add_parser("test-order", help="open and immediately close the smallest position, to test the order path")
    common(p, trading=True)
    p.set_defaults(fn=cmd_test_order)
    for name, fn, text in (("status", cmd_status, "show the bot's ledger"), ("report", cmd_report, "results so far"),
                           ("reset", cmd_reset, "clear the kill switch"), ("markets", cmd_markets, "list known IG markets")):
        p = dsub.add_parser(name, help=text)
        common(p)
        p.set_defaults(fn=fn)
    p = dsub.add_parser("flatten", help="sell every position the bot holds, now")
    common(p, trading=True)
    p.set_defaults(fn=cmd_flatten)
