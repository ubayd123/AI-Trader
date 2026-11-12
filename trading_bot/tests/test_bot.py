import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from trading_bot.advisor import ClaudeAdvisor
from trading_bot.backtest import run_backtest
from trading_bot.brokers import BrokerError, PaperBroker, Trading212Broker, make_broker
from trading_bot.config import Config, parse_watchlist
from trading_bot.data import is_market_open
from trading_bot.engine import Engine
from trading_bot.indicators import snapshot
from trading_bot.models import Fill, Order, Portfolio, Position, Snapshot
from trading_bot.report import summarise
from trading_bot.strategy import circuit_breaker, entry_orders, exit_orders, regime_ok

TODAY = date(2026, 11, 12)
OPEN = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)  # Thursday 10:00 New York
CLOSED = datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)  # Saturday


def snap(symbol, price=100.0, up=True, market="us-stock", **kw):
    base = dict(symbol=symbol, market=market, price=price, sma_fast=price * (0.97 if up else 1.03),
                sma_slow=price * (0.92 if up else 1.08), ret_20=0.05 if up else -0.05,
                ret_60=0.12 if up else -0.12, rsi=60.0, vol=0.25, usd_price=price * 1.3)
    base.update(kw)
    return Snapshot(**base)


@pytest.fixture
def cfg(tmp_path):
    return Config(state_dir=str(tmp_path), ai_enabled=False).validate()


def portfolio(cash=50.0, **positions):
    return Portfolio(cash=cash, positions=positions, peak_equity=50.0, starting_cash=50.0)


def held(price=100.0, qty=0.1, high=None):
    return Position("us-stock", qty, price, high or price, "2026-11-01T00:00:00")


# ---------- config ----------
def test_watchlist_parsing_and_validation():
    assert parse_watchlist("aapl, crypto:btc,AAPL") == [("us-stock", "AAPL"), ("crypto", "BTC")]
    with pytest.raises(ValueError):
        parse_watchlist("forex:EURUSD")
    with pytest.raises(ValueError):
        Config(max_drawdown_pct=1.5).validate()


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("BOT_STARTING_CASH_GBP", "75")
    monkeypatch.setenv("BOT_AI_ENABLED", "false")
    monkeypatch.setenv("BOT_WATCHLIST", "crypto:ETH")
    c = Config.from_env()
    assert (c.starting_cash_gbp, c.ai_enabled, c.watchlist) == (75.0, False, [("crypto", "ETH")])


# ---------- indicators ----------
def test_snapshot_detects_uptrend_and_needs_history(cfg):
    idx = pd.date_range("2026-01-01", periods=120, freq="B")
    rising = pd.Series([100 * 1.004 ** i for i in range(120)], index=idx)
    s = snapshot("UP", "us-stock", rising, cfg)
    assert s.trend_ok and s.score > 0 and s.ret_20 > 0
    assert not snapshot("DOWN", "us-stock", rising[::-1].set_axis(idx), cfg).trend_ok
    assert snapshot("NEW", "us-stock", rising.head(30), cfg) is None


# ---------- entries ----------
def test_entries_respect_slots_reserve_and_ranking(cfg):
    snaps = {s: snap(s, ret_60=r) for s, r in [("A", 0.10), ("B", 0.30), ("C", 0.20), ("D", 0.05)]}
    snaps["E"] = snap("E", up=False)
    orders = entry_orders(portfolio(), snaps, cfg, TODAY)
    assert [o.symbol for o in orders] == ["B", "C", "A"]  # best momentum first, max 3, downtrend excluded
    assert sum(o.gbp_amount for o in orders) <= 50 * (1 - cfg.cash_reserve_pct) + 0.01
    assert all(o.gbp_amount == pytest.approx(15.83, abs=0.01) for o in orders)


def test_no_entry_when_overbought_cooling_down_halted_or_too_small(cfg):
    assert entry_orders(portfolio(), {"A": snap("A", rsi=80)}, cfg, TODAY) == []
    pf = portfolio()
    pf.cooldowns["A"] = (TODAY + timedelta(days=2)).isoformat()
    assert entry_orders(pf, {"A": snap("A")}, cfg, TODAY) == []
    pf = portfolio()
    pf.halted = True
    assert entry_orders(pf, {"A": snap("A")}, cfg, TODAY) == []
    assert entry_orders(portfolio(cash=3.0), {"A": snap("A")}, cfg, TODAY) == []  # £3: slots under min trade


def test_ai_can_veto_and_rank_but_not_add(cfg):
    snaps = {"A": snap("A", ret_60=0.3), "B": snap("B", ret_60=0.1), "BAD": snap("BAD", up=False)}
    advice = {"A": {"action": "avoid", "confidence": 0.9, "reason": "spike"},
              "B": {"action": "buy", "confidence": 0.8, "reason": "steady"},
              "BAD": {"action": "buy", "confidence": 1.0, "reason": "trust me"}}
    orders = entry_orders(portfolio(), snaps, cfg, TODAY, advice)
    assert [o.symbol for o in orders] == ["B"] and "AI entry" in orders[0].reason
    advice["B"]["confidence"] = 0.3  # below ai_min_confidence
    assert entry_orders(portfolio(), snaps, cfg, TODAY, advice) == []


# ---------- exits ----------
def test_stop_loss_trailing_stop_trend_and_ai_exits(cfg):
    pf = portfolio(cash=0, SL=held(100), TR=held(100, high=130), TB=held(100), AI=held(100), OK=held(100))
    snaps = {"SL": snap("SL", 89), "TR": snap("TR", 103), "TB": snap("TB", 100, up=False),
             "AI": snap("AI", 105), "OK": snap("OK", 105)}
    advice = {"AI": {"action": "sell", "confidence": 0.9, "reason": "deteriorating"}}
    orders = exit_orders(pf, snaps, cfg, TODAY, advice)
    reasons = {o.symbol: o.reason for o in orders}
    assert set(reasons) == {"SL", "TR", "TB", "AI"}
    assert "stop-loss" in reasons["SL"] and "trailing" in reasons["TR"] and "trend" in reasons["TB"]
    assert {o.symbol for o in orders if o.stop} == {"SL", "TR"}  # only stop-outs trigger a cooldown


def test_regime_filter_blocks_new_buys_only(cfg):
    idx = pd.date_range("2025-01-01", periods=260, freq="B")
    falling = pd.Series([5000 * 0.999 ** i for i in range(260)], index=idx)
    assert regime_ok(falling[::-1].set_axis(idx), cfg) and not regime_ok(falling, cfg)
    assert regime_ok(None, cfg) and regime_ok(falling.head(50), cfg)  # no data: do not block
    assert regime_ok(falling, Config(regime_filter=False))
    assert entry_orders(portfolio(), {"A": snap("A")}, cfg, TODAY, regime_ok=False) == []
    pf = portfolio(cash=0, A=held(100))
    assert exit_orders(pf, {"A": snap("A", 85)}, cfg, TODAY)  # exits still work in a bad market


def test_volatility_sizing_shrinks_wild_shares(cfg):
    cfg.vol_target = 0.25
    orders = entry_orders(portfolio(), {"CALM": snap("CALM", vol=0.20), "WILD": snap("WILD", vol=0.50)}, cfg, TODAY)
    size = {o.symbol: o.gbp_amount for o in orders}
    assert size["CALM"] == pytest.approx(15.83, abs=0.01) and size["WILD"] == pytest.approx(7.92, abs=0.01)


def test_exit_buffer_prevents_whipsaw_and_missing_price_never_sells(cfg):
    pf = portfolio(cash=0, A=held(100), B=held(100))
    just_below = snap("A", 100, sma_slow=101, sma_fast=102)  # 1% under the average: inside the 3% buffer
    assert exit_orders(pf, {"A": just_below}, cfg, TODAY) == []


def test_circuit_breaker_trips_pauses_and_resumes(cfg):
    pf = portfolio(cash=37.0)
    assert circuit_breaker(pf, 37.0, cfg, TODAY) is True  # 26% below the £50 peak
    assert pf.halted and pf.halted_until == (TODAY + timedelta(days=30)).isoformat()
    assert circuit_breaker(pf, 37.0, cfg, TODAY + timedelta(days=5)) is True
    assert circuit_breaker(pf, 37.0, cfg, TODAY + timedelta(days=30)) is False
    assert not pf.halted and pf.peak_equity == 37.0
    manual = Config(halt_days=0)
    pf = portfolio(cash=37.0)
    circuit_breaker(pf, 37.0, manual, TODAY)
    assert circuit_breaker(pf, 37.0, manual, TODAY + timedelta(days=900)) is True


# ---------- ledger & paper broker ----------
def test_paper_round_trip_accounts_for_every_penny(cfg):
    pf, broker = portfolio(), PaperBroker(cfg)
    buy = Order("buy", "A", "us-stock", "t", gbp_amount=15.0)
    pf.apply_buy(buy, broker.execute(buy, snap("A", 100)), "now")
    assert pf.cash == pytest.approx(35.0)  # spends exactly the budget, fees included
    sell = Order("sell", "A", "us-stock", "t", qty=pf.positions["A"].qty)
    pnl = pf.apply_sell(sell, broker.execute(sell, snap("A", 110)))
    assert "A" not in pf.positions
    assert pf.cash == pytest.approx(50.0 + pnl) and 1.3 < pnl < 1.5 and pf.fees_paid > 0


def test_portfolio_persists(tmp_path):
    pf = portfolio(cash=10, A=held())
    pf.save(tmp_path / "s.json")
    again = Portfolio.load(tmp_path / "s.json", 50)
    assert again == pf
    assert Portfolio.load(tmp_path / "missing.json", 50).cash == 50


# ---------- engine ----------
class FakeData:
    def __init__(self, snaps, index=None):
        self.snaps, self.index = snaps, index

    def snapshots(self, now=None):
        return self.snaps

    def index_closes(self):
        return self.index

    def headlines(self, symbol):
        return [{"date": "2026-11-11", "title": f"{symbol} news", "summary": ""}]


def test_engine_cycle_buys_persists_and_publishes(cfg):
    published = []
    platform = SimpleNamespace(publish_trade=lambda *a: published.append(a), market_context=lambda: None)
    engine = Engine(cfg, PaperBroker(cfg), FakeData({"A": snap("A"), "B": snap("B", up=False)}), platform=platform)
    report = engine.run_cycle(OPEN)
    assert [t["symbol"] for t in report["trades"]] == ["A"] and report["cash_gbp"] < 50
    assert published[0][:3] == ("us-stock", "buy", "A") and published[0][3] == pytest.approx(130.0)
    assert len(cfg.trades_path.read_text().splitlines()) == 1
    # A fresh engine reloads the ledger, then exits when the trend breaks.
    engine = Engine(cfg, PaperBroker(cfg), FakeData({"A": snap("A", 99, up=False)}))
    report = engine.run_cycle(OPEN + timedelta(days=1))
    assert report["positions"] == {} and report["trades"][0]["side"] == "sell"


def test_engine_stop_out_starts_cooldown_and_records_equity(cfg):
    engine = Engine(cfg, PaperBroker(cfg), FakeData({"A": snap("A", 88)}))
    engine.portfolio = portfolio(cash=40.0, A=held(100, qty=0.1))
    report = engine.run_cycle(OPEN)
    assert report["trades"][0]["side"] == "sell" and "A" in engine.portfolio.cooldowns
    assert json.loads(cfg.equity_path.read_text().splitlines()[-1])["equity_gbp"] == pytest.approx(report["equity_gbp"], abs=0.01)


def test_engine_respects_bad_market_and_feeds_news_to_ai(cfg):
    idx = pd.date_range("2025-01-01", periods=260, freq="B")
    bear = pd.Series([5000 * 0.999 ** i for i in range(260)], index=idx)
    engine = Engine(cfg, PaperBroker(cfg), FakeData({"A": snap("A")}, index=bear))
    report = engine.run_cycle(OPEN)
    assert report["trades"] == [] and report["market_ok"] is False

    seen = {}

    def advise(pf, snaps, picks, context, news):
        seen.update(news=news, picks=[p.symbol for p in picks])
        return {"A": {"action": "buy", "confidence": 0.9, "reason": "ok"}}

    engine = Engine(cfg, PaperBroker(cfg), FakeData({"A": snap("A")}), advisor=SimpleNamespace(advise=advise))
    report = engine.run_cycle(OPEN)
    assert seen["picks"] == ["A"] and seen["news"]["A"][0]["title"] == "A news"
    assert report["ai_used"] and [t["symbol"] for t in report["trades"]] == ["A"]


def test_engine_waits_when_market_closed(cfg):
    assert is_market_open("us-stock", OPEN) and not is_market_open("us-stock", CLOSED)
    assert is_market_open("crypto", CLOSED)
    engine = Engine(cfg, PaperBroker(cfg), FakeData({"A": snap("A")}))
    assert engine.run_cycle(CLOSED)["trades"] == []


def test_engine_kill_switch_liquidates(cfg):
    engine = Engine(cfg, PaperBroker(cfg), FakeData({"A": snap("A", 70), "B": snap("B")}))
    engine.portfolio = portfolio(cash=0.0, A=held(100, qty=0.5))  # equity £35 vs £50 peak
    report = engine.run_cycle(OPEN)
    assert report["halted"] and report["positions"] == {} and len(report["trades"]) == 1


def test_engine_survives_broker_failure(cfg):
    class Broken(PaperBroker):
        def execute(self, order, s):
            raise BrokerError("boom")

    engine = Engine(cfg, Broken(cfg), FakeData({"A": snap("A")}))
    report = engine.run_cycle(OPEN)
    assert report["trades"] == [] and report["cash_gbp"] == 50.0


# ---------- Trading 212 ----------
class FakeT212:
    """Minimal fake of the Trading 212 REST API."""

    def __init__(self, cash=100.0, fills=True, currency="GBP"):
        self.cash, self.fills, self.currency = cash, fills, currency
        self.qty, self.cost, self.calls, self.history = 0.0, 0.0, [], []

    def request(self, method, url, auth=None, headers=None, timeout=None, params=None, json=None):
        self.calls.append((method, url.split("/api/v0")[1], json))
        body = None
        if url.endswith("/account/summary"):
            body = {"currency": self.currency, "cash": {"availableToTrade": self.cash}}
        elif url.endswith("/equity/positions"):
            body = [{"instrument": {"ticker": params["ticker"]}, "quantity": self.qty,
                     "quantityAvailableForTrading": self.qty, "walletImpact": {"totalCost": self.cost}}] if self.qty else []
        elif url.endswith("/history/orders"):
            body = {"items": self.history}
        elif url.endswith("/orders/market"):
            if self.fills:
                self.qty += json["quantity"]
                self.cost += json["quantity"] * 100.2
            body = {"id": 7}
        return SimpleNamespace(status_code=200, content=b"x", json=lambda: body, headers={}, text="")


def t212(cfg, fake):
    cfg.broker = "trading212"
    return Trading212Broker(cfg, "key", "secret", session=fake, sleep=lambda s: None)


def test_t212_buy_then_sell(cfg):
    fake = FakeT212()
    broker = t212(cfg, fake)
    fill = broker.execute(Order("buy", "AAPL", "us-stock", "t", gbp_amount=15.0), snap("AAPL", 100))
    order = next(c for c in fake.calls if c[1] == "/equity/orders/market")[2]
    assert order == {"ticker": "AAPL_US_EQ", "quantity": 0.149, "extendedHours": False}
    assert fill.qty == 0.149 and fill.price_gbp == pytest.approx(100.2) and fill.qty * fill.price_gbp <= 15.0
    fake.history = [{"order": {"id": 7}, "fill": {"quantity": -0.149, "walletImpact": {"currency": "GBP", "netValue": 16.3}}}]
    fill = broker.execute(Order("sell", "AAPL", "us-stock", "t", qty=0.149), snap("AAPL", 110))
    sent = [c[2]["quantity"] for c in fake.calls if c[1] == "/equity/orders/market"]
    assert sent == [0.149, -0.149] and fake.qty == pytest.approx(0)
    assert not fill.estimated and fill.qty * fill.price_gbp == pytest.approx(16.3)  # exact proceeds from history


def test_t212_sell_estimates_when_history_lags(cfg):
    fake = FakeT212()
    fake.qty, fake.cost = 0.2, 20.0
    fill = t212(cfg, fake).execute(Order("sell", "AAPL", "us-stock", "t", qty=0.2), snap("AAPL", 110))
    assert fill.estimated and fill.price_gbp == 110


def test_t212_guards(cfg, monkeypatch):
    buy = Order("buy", "AAPL", "us-stock", "t", gbp_amount=15.0)
    with pytest.raises(BrokerError, match="available"):
        t212(cfg, FakeT212(cash=5.0)).execute(buy, snap("AAPL"))
    with pytest.raises(BrokerError, match="GBP"):
        t212(cfg, FakeT212(currency="EUR")).execute(buy, snap("AAPL"))
    fake = FakeT212(fills=False)
    with pytest.raises(BrokerError, match="cancelled"):
        t212(cfg, fake).execute(buy, snap("AAPL"))
    assert fake.calls[-1][:2] == ("DELETE", "/equity/orders/7")
    assert not t212(cfg, FakeT212()).supports("crypto")
    cfg.t212_env = "live"
    monkeypatch.setenv("T212_API_KEY", "k")
    with pytest.raises(BrokerError, match="confirm-live"):
        make_broker(cfg)
    assert make_broker(cfg, confirm_live=True).base == "https://live.trading212.com/api/v0"


# ---------- AI advisor ----------
def fake_claude(payload=None, stop_reason="end_turn", error=None):
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        if error:
            raise error
        return SimpleNamespace(stop_reason=stop_reason,
                               content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=json.dumps(payload))])

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create))), seen


def test_advisor_parses_clamps_and_ignores_unknown_symbols(cfg):
    client, seen = fake_claude({"market_view": "ok", "decisions": [
        {"symbol": "A", "action": "buy", "confidence": 1.7, "reason": "steady"},
        {"symbol": "ZZZ", "action": "buy", "confidence": 0.9, "reason": "not offered"}]})
    news = {"A": [{"date": "2026-11-11", "title": "A beats forecasts", "summary": ""}]}
    advice = ClaudeAdvisor(cfg, client).advise(portfolio(), {"A": snap("A")}, [snap("A")], headlines=news)
    assert advice == {"A": {"action": "buy", "confidence": 1.0, "reason": "steady"}}
    assert json.loads(seen["messages"][0]["content"])["candidates"][0]["recent_headlines"] == news["A"]
    assert seen["model"] == "claude-opus-5-5" and "temperature" not in seen
    assert json.loads(seen["messages"][0]["content"])["candidates"][0]["symbol"] == "A"


def test_advisor_failures_fall_back_to_rules(cfg):
    for client, _ in (fake_claude(stop_reason="refusal"), fake_claude(error=RuntimeError("no network"))):
        assert ClaudeAdvisor(cfg, client).advise(portfolio(), {"A": snap("A")}, [snap("A")]) is None
    assert [o.symbol for o in entry_orders(portfolio(), {"A": snap("A")}, cfg, TODAY, None)] == ["A"]


# ---------- backtest ----------
def test_backtest_profits_in_uptrend_and_stays_out_of_downtrend(cfg):
    idx = pd.date_range("2025-01-01", periods=300, freq="B")
    wiggle = [1 + 0.01 * (-1) ** i for i in range(300)]  # day-to-day noise so RSI is realistic
    up = pd.Series([100 * 1.003 ** i * w for i, w in enumerate(wiggle)], index=idx)
    down = pd.Series([100 * 0.997 ** i * w for i, w in enumerate(wiggle)], index=idx)
    result = run_backtest(cfg, {"UP": ("us-stock", up), "DOWN": ("us-stock", down)})
    assert result["final_gbp"] > 50 and result["trades"] == 1 and result["kill_switch_trips"] == 0
    flat = run_backtest(cfg, {"DOWN": ("us-stock", down)})
    assert flat["final_gbp"] == 50.0 and flat["trades"] == 0


# ---------- report ----------
def test_report_summarises_trades_and_drawdown():
    trades = [{"side": "buy", "fee_gbp": 0.02}, {"side": "sell", "pnl_gbp": 2.0, "fee_gbp": 0.03, "reason": "trend broken (x)"},
              {"side": "sell", "pnl_gbp": -1.0, "fee_gbp": 0.02, "reason": "stop-loss (10% below entry)"}]
    equity = [{"time": "2026-11-01T00:00:00", "equity_gbp": 50}, {"time": "2026-11-05T00:00:00", "equity_gbp": 55},
              {"time": "2026-11-09T00:00:00", "equity_gbp": 49.5}]
    r = summarise(trades, equity, 50.0)
    assert (r["closed_trades"], r["win_rate_pct"], r["realised_pnl_gbp"], r["fees_gbp"]) == (2, 50, 1.0, 0.07)
    assert r["return_pct"] == -1.0 and r["max_drawdown_pct"] == -10.0
    assert r["exit_reasons"] == {"trend broken": 1, "stop-loss": 1}
    assert summarise([], [], 50.0)["trades"] == 0
