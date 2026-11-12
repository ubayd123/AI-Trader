"""Tests for the intraday bot. The IG and eToro APIs are emulated from their published references."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from trading_bot.day.bars import BarSeries, Features, Quote, in_session, next_session_open
from trading_bot.day.cli import day_report
from trading_bot.day.config import IG_DEFAULT, PRESETS, DayConfig
from trading_bot.day.engine import BrokerError, Closure, CloseFill, DayEngine, DayPosition, DayState, OpenFill, Spec
from trading_bot.day.etoro import BASE_URL, EtoroBroker, EtoroClient, EtoroError, EtoroFeed, EtoroUncertain, RateLimiter
from trading_bot.day.feeds import todays_bars
from trading_bot.day.runner import run
from trading_bot.day.screen import premarket_screen
from trading_bot.day.sim import SimBroker, replay
from trading_bot.day.strategy import entry_signal

OPEN = datetime(2026, 11, 12, 14, 30, tzinfo=timezone.utc)  # Thursday 09:30 New York
SPREAD = 0.0002


def quote(symbol, price, when, spread=SPREAD):
    return Quote(symbol, price * (1 - spread / 2), price * (1 + spread / 2), when)


class FakeBroker:
    name = "fake"

    def __init__(self):
        self.opened, self.closed, self.pending = [], [], []
        self.fail_open = False

    def open(self, symbol, amount, stop, target, q):
        if self.fail_open:
            raise BrokerError("rejected")
        self.opened.append((symbol, amount, stop, target))
        return OpenFill(len(self.opened), amount / q.ask, q.ask, 500 + len(self.opened))

    def close(self, pos, q, reason):
        self.closed.append((pos.symbol, reason))
        return CloseFill(q.bid)

    def sync(self, positions, quotes, now):
        out, self.pending = self.pending, []
        return out


@pytest.fixture
def cfg(tmp_path):
    return DayConfig(broker="etoro", symbols=["AAA", "BBB"], state_dir=str(tmp_path), ai_enabled=False).validate()


def make_engine(cfg, broker=None, budget=66.0, **state):
    return DayEngine(cfg, broker or FakeBroker(), DayState(budget=budget, **state))


def minute(engine, i, prices, spread=SPREAD):
    """One minute of trading: two quotes per share, giving each bar a small range and `price` as its close."""
    t = OPEN + timedelta(minutes=i)
    engine.on_quotes(t, {s: quote(s, p + 0.04, t, spread) for s, p in prices.items()})
    t += timedelta(seconds=30)
    engine.on_quotes(t, {s: quote(s, p, t, spread) for s, p in prices.items()})


def warm_up(engine, symbols=("AAA",), upto=69):
    """Seventy quiet minutes (the indicators need 70 bars)."""
    for i in range(upto):
        minute(engine, i, {s: 100 + 0.01 * (-1) ** i for s in symbols})


def dip(engine, at=69, price=99.80, symbols=("AAA",)):
    minute(engine, at, {s: price for s in symbols})  # the bar that closes sharply lower
    minute(engine, at + 1, {s: price for s in symbols})  # its close is seen here: the bot buys


def held(symbol="AAA", pid=1, entry=100.0, units=0.1, stop=99.0, target=101.0, at=OPEN + timedelta(minutes=70)):
    return DayPosition(symbol, pid, units, entry, entry * units, stop, target, at.isoformat(), "test", 501)


# ---------------- config ----------------
def test_presets_env_and_validation(monkeypatch):
    ig = DayConfig.from_env()
    assert (ig.broker, ig.symbols, ig.z_entry, ig.currency, ig.label) == ("ig", IG_DEFAULT, 1.0, "£", "ig_demo")
    assert DayConfig().max_trades_per_day == 100 and DayConfig().daily_loss_limit_pct == 0.02
    etoro = DayConfig.from_env(broker="etoro")
    assert etoro.symbols == PRESETS["rapid"]["etoro_symbols"] and etoro.currency == "$"
    selective = DayConfig.from_env(broker="etoro", preset="selective")
    assert selective.z_entry == 3.0 and len(selective.symbols) == 12
    monkeypatch.setenv("DAY_BROKER", "etoro")
    monkeypatch.setenv("DAY_PRESET", "selective")
    monkeypatch.setenv("DAY_SYMBOLS", "nvda, aapl")
    monkeypatch.setenv("DAY_DAILY_LOSS_LIMIT_PCT", "0.01")
    c = DayConfig.from_env()
    assert (c.broker, c.symbols, c.z_entry, c.daily_loss_limit_pct) == ("etoro", ["NVDA", "AAPL"], 3.0, 0.01)
    assert DayConfig.from_env(dry_run=True).state_path.name == "day_state_etoro_dry.json"
    assert c.spread_pct("NVDA") == pytest.approx(0.0001) and c.spread_pct("ZZZ") == pytest.approx(0.0005)
    for bad in (dict(etoro_env="live"), dict(mode="yolo"), dict(daily_loss_limit_pct=2), dict(preset="turbo"),
                dict(broker="robinhood"), dict(risk_per_trade_pct=0.05)):
        with pytest.raises(ValueError):
            DayConfig(**bad).validate()


# ---------------- bars & sessions ----------------
def test_bars_are_built_from_quotes_per_minute():
    series = BarSeries("AAA")
    assert series.update(quote("AAA", 100, OPEN)) is None
    assert series.update(quote("AAA", 101, OPEN + timedelta(seconds=20))) is None
    assert series.update(quote("AAA", 99, OPEN + timedelta(seconds=40))) is None
    bar = series.update(quote("AAA", 100.5, OPEN + timedelta(minutes=1)))
    assert (bar.open, bar.high, bar.low, bar.close) == (100, 101, 99, 99) and series.session_open == 100
    assert series.update(quote("AAA", 50, OPEN)) is None and len(series.bars) == 1  # late quote ignored
    assert series.features(20, 60) is None  # not enough history yet


def test_features_measure_dip_trend_and_range():
    series = BarSeries("AAA")
    for i in range(80):
        t = OPEN + timedelta(minutes=i)
        price = 100 + 0.02 * i
        series.update(quote("AAA", price + 0.05, t))
        series.update(quote("AAA", price if i < 79 else price - 1.0, t + timedelta(seconds=30)))
    series.update(quote("AAA", 100, OPEN + timedelta(minutes=80)))
    f = series.features(20, 60)
    assert f.z < -2 and f.trend_rising and f.unit > 0 and f.session_return > 0 and f.close < f.mean


def test_session_clock():
    assert in_session(OPEN) and in_session(OPEN + timedelta(hours=6, minutes=29))
    assert not in_session(OPEN - timedelta(minutes=1)) and not in_session(OPEN + timedelta(hours=6, minutes=30))
    saturday = datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)
    assert not in_session(saturday)
    assert next_session_open(saturday) == datetime(2026, 11, 16, 14, 30, tzinfo=timezone.utc)
    assert next_session_open(OPEN - timedelta(hours=3)) == OPEN
    bars = [SimpleNamespace(time=OPEN - timedelta(days=1)), SimpleNamespace(time=OPEN), SimpleNamespace(time=OPEN + timedelta(minutes=5))]
    assert todays_bars(bars, OPEN + timedelta(minutes=5, seconds=10)) == [bars[1]]  # not yesterday, not the forming minute


# ---------------- entry rules ----------------
def feat(**kw):
    base = dict(symbol="AAA", close=100.0, mean=100.3, z=-2.0, trend=99.5, trend_rising=True, channel_high=100.5,
                unit=0.0005, session_return=0.01)
    base.update(kw)
    return Features(**base)


def test_entry_rules_for_each_mode():
    revert = DayConfig(mode="revert", z_entry=1.5)
    assert entry_signal(feat(z=-1.6), revert).stop_pct == pytest.approx(0.004)  # 8 x 0.05%
    assert entry_signal(feat(z=-1.4), revert) is None and entry_signal(feat(unit=0), revert) is None
    assert entry_signal(feat(unit=0.01), revert).stop_pct == 0.01  # capped
    assert entry_signal(feat(unit=0.00001), revert).stop_pct == 0.0015  # floored
    dip_cfg = DayConfig(mode="dip", z_entry=1.5)
    assert entry_signal(feat(), dip_cfg) and entry_signal(feat(trend_rising=False), dip_cfg) is None
    assert entry_signal(feat(close=99.0), dip_cfg) is None  # below its trend
    breakout = DayConfig(mode="breakout", z_entry=1.5)
    assert entry_signal(feat(z=2.0, close=100.6), breakout)
    assert entry_signal(feat(z=2.0, close=100.4), breakout) is None  # no new high
    assert entry_signal(feat(z=2.0, close=100.6, session_return=-0.01), breakout) is None


# ---------------- engine ----------------
def test_engine_buys_a_dip_with_sized_position_stop_and_target(cfg):
    engine = make_engine(cfg)
    warm_up(engine)
    assert engine.state.positions == []
    dip(engine)
    (symbol, amount, stop, target), = engine.broker.opened
    pos, = engine.state.positions
    assert symbol == "AAA" and amount == pytest.approx(66 * 0.97 / 6, abs=0.01)  # six $10.67 slots
    assert stop < pos.entry < target and stop == round(stop, 2)
    assert engine.state.trades_today == 1 and engine.cash() == pytest.approx(66 - amount, abs=0.01)
    assert json.loads(cfg.trades_path.read_text().splitlines()[0])["side"] == "buy"
    assert DayState.load(cfg.state_path).positions[0].symbol == "AAA"  # survives a restart


def test_engine_sells_when_price_returns_to_average(cfg):
    engine = make_engine(cfg)
    warm_up(engine)
    dip(engine)
    minute(engine, 71, {"AAA": 100.0})
    minute(engine, 72, {"AAA": 100.0})
    assert engine.broker.closed == [("AAA", "back to average")] and engine.state.positions == []
    assert engine.state.realized > 0 and engine.state.trades_today == 2 and "AAA" not in engine.state.cooldowns


def test_engine_time_limit_and_end_of_day(cfg):
    cfg.exit_at_mean = False
    engine = make_engine(cfg)
    warm_up(engine)
    dip(engine)
    for i in range(71, 102):
        minute(engine, i, {"AAA": 99.80})
    assert engine.broker.closed == [("AAA", "time limit")]

    engine = make_engine(cfg, positions=[held(at=OPEN + timedelta(hours=6, minutes=20))])
    late = OPEN + timedelta(hours=6, minutes=24)
    engine.on_quotes(late, {"AAA": quote("AAA", 100, late)})
    engine.on_quotes(late + timedelta(minutes=1), {"AAA": quote("AAA", 100, late + timedelta(minutes=1))})
    assert engine.broker.closed == [("AAA", "end of day")]  # five minutes before the close: nothing is held overnight


def test_no_entries_near_the_open_or_the_close(cfg):
    engine = make_engine(cfg)
    engine.seed("AAA", [SimpleNamespace(time=OPEN - timedelta(minutes=80 - i), open=100, high=100.04, low=100, close=100 + 0.01 * (-1) ** i)
                        for i in range(80)])
    minute(engine, 3, {"AAA": 99.5})
    minute(engine, 4, {"AAA": 99.5})
    assert engine.broker.opened == []  # first ten minutes are skipped


def test_broker_stop_loss_is_booked_and_starts_a_cooldown(cfg):
    broker = FakeBroker()
    engine = make_engine(cfg, broker)
    warm_up(engine)
    dip(engine)
    pos = engine.state.positions[0]
    broker.pending = [Closure(pos.position_id, pos.stop, "stop-loss")]
    minute(engine, 71, {"AAA": 99.0})
    assert engine.state.positions == [] and engine.state.realized < 0
    assert engine.trades[-1]["reason"] == "stop-loss" and "AAA" in engine.state.cooldowns
    for i in range(72, 78):
        minute(engine, i, {"AAA": 98.5})
    assert len(broker.opened) == 1  # no re-entry while cooling down, however sharp the drop


def test_daily_loss_limit_stops_trading_for_the_day(cfg):
    broker = FakeBroker()
    engine = make_engine(cfg, broker)
    warm_up(engine, symbols=("AAA", "BBB"))
    engine.state.positions.append(held("BBB", pid=9, entry=100.0, units=1.0, stop=98.0))
    minute(engine, 69, {"AAA": 100.0, "BBB": 98.6})  # the BBB position is now down $1.40 = 2.1% of $66
    assert engine.state.halted_today and engine.state.positions == []
    assert ("BBB", "daily loss limit") in broker.closed
    engine.state.positions.append(held("AAA", pid=11))  # e.g. a sale that failed during the halt
    minute(engine, 70, {"AAA": 100.0, "BBB": 98.6})
    assert ("AAA", "trading halted") in broker.closed and engine.state.positions == []
    dip(engine, at=71)
    assert broker.opened == []  # nothing more today
    engine.on_quotes(OPEN + timedelta(days=1), {"AAA": quote("AAA", 100, OPEN + timedelta(days=1))})
    assert not engine.state.halted_today and engine.state.trades_today == 0  # a new day starts fresh


def test_risk_gate_blocks_a_trade_that_could_breach_the_daily_limit(cfg):
    engine = make_engine(cfg)
    warm_up(engine)
    engine.state.day_realized = -1.30  # $1.32 is the 2% limit; one more stop-out would cross it
    engine.state.realized = -1.30
    dip(engine)
    assert engine.broker.opened == [] and not engine.state.halted_today


def test_trade_cap_total_drawdown_and_reset_of_peak(cfg):
    cfg.max_trades_per_day = 2
    engine = make_engine(cfg)
    warm_up(engine, symbols=("AAA", "BBB"))
    dip(engine, symbols=("AAA", "BBB"))
    assert len(engine.broker.opened) == 1  # a second buy would leave no room to sell both

    engine = make_engine(cfg, peak_equity=74.0)  # $66 is more than 10% below a $74 peak
    warm_up(engine, upto=3)
    assert engine.state.halted and "10%" in engine.state.halt_reason
    warm_up(engine)
    dip(engine)
    assert engine.broker.opened == []


def test_wide_spread_stale_quote_small_budget_and_broker_failure_block_entries(cfg):
    engine = make_engine(cfg)
    warm_up(engine)
    minute(engine, 69, {"AAA": 99.8})
    minute(engine, 70, {"AAA": 99.8}, spread=0.002)  # 0.2% spread: too expensive
    assert engine.broker.opened == []

    engine = make_engine(cfg)
    warm_up(engine)
    minute(engine, 69, {"AAA": 99.8})
    t = OPEN + timedelta(minutes=70)
    engine.on_quotes(t, {"AAA": Quote("AAA", 99.79, 99.81, t - timedelta(minutes=2))})  # two-minute-old price
    assert engine.broker.opened == []

    engine = make_engine(cfg)  # a computer clock that is 90 seconds fast must not block fresh quotes
    for i in range(69):
        t = OPEN + timedelta(minutes=i)
        for half, extra in ((0, 0.04), (30, 0.0)):
            late = t + timedelta(seconds=half)
            engine.on_quotes(late + timedelta(seconds=90), {"AAA": quote("AAA", 100 + 0.01 * (-1) ** i + extra, late)})
    for i, price in ((69, 99.8), (70, 99.8)):
        t = OPEN + timedelta(minutes=i)
        engine.on_quotes(t + timedelta(seconds=90), {"AAA": quote("AAA", price + 0.04, t)})
        engine.on_quotes(t + timedelta(seconds=120), {"AAA": quote("AAA", price, t + timedelta(seconds=30))})
    assert len(engine.broker.opened) == 1

    engine = make_engine(cfg, budget=9.0)  # below eToro's $10 minimum
    warm_up(engine)
    dip(engine)
    assert engine.broker.opened == [] and engine.slots() == 0

    broker = FakeBroker()
    broker.fail_open = True
    engine = make_engine(cfg, broker)
    warm_up(engine)
    dip(engine)
    assert engine.state.positions == [] and engine.state.trades_today == 0


def test_excluded_shares_are_not_traded(cfg):
    engine = make_engine(cfg)
    warm_up(engine, upto=1)
    engine.state.excluded_today = ["AAA"]
    for i in range(1, 69):
        minute(engine, i, {"AAA": 100 + 0.01 * (-1) ** i})
    dip(engine)
    assert engine.broker.opened == []


# ---------------- simulated broker & replay ----------------
def test_sim_broker_fills_and_conservative_stops(cfg):
    broker = SimBroker(cfg)
    t = OPEN
    fill = broker.open("AAA", 10.0, 99.0, 101.0, quote("AAA", 100, t))
    assert fill.price > 100.01 and fill.units == pytest.approx(10 / fill.price) and broker.costs > 0
    pos = held(entry=fill.price, units=fill.units, stop=99.0, target=101.0)
    assert broker.sync([pos], {"AAA": quote("AAA", 100, t)}, t) == []
    hit = broker.sync([pos], {"AAA": quote("AAA", 98.5, t)}, t)[0]  # traded through the stop: filled at the stop
    assert hit.reason == "stop-loss" and hit.price == pytest.approx(99.0 * (1 - 0.00005))
    gap = SimBroker(cfg).sync([pos], {"AAA": quote("AAA", 98.5, t)}, t)[0]  # first quote already below: worse fill
    assert gap.price < 98.6
    broker2 = SimBroker(cfg)
    broker2.sync([pos], {"AAA": quote("AAA", 100, t)}, t)
    assert broker2.sync([pos], {"AAA": quote("AAA", 101.5, t)}, t)[0].reason == "take-profit"
    assert broker.close(pos, quote("AAA", 100, t), "x").price < 100


def test_replay_backtest_on_synthetic_prices(cfg):
    idx = pd.date_range(OPEN, periods=390, freq="1min")
    base = [100 + 0.01 * (-1) ** i for i in range(390)]
    for i in (100, 101, 200, 201):
        base[i] = 99.75  # two sharp dips that recover
    frame = pd.DataFrame({"open": base, "high": [p + 0.04 for p in base], "low": [p - 0.01 for p in base], "close": base}, index=idx)
    cfg.symbols = ["AAA"]
    cfg.z_entry = 2.0
    result = replay(cfg, {"AAA": frame}, budget=66.0)
    assert result["days"] == 1 and result["round_trips"] == 2 and result["win_rate_pct"] == 100
    assert result["final"] > 66 and result["costs"] > 0 and result["exit_reasons"] == {"back to average": 2}
    falling = frame.mul(pd.Series([1 - 0.0004 * i for i in range(390)], index=idx), axis=0)
    lost = replay(cfg, {"AAA": falling}, budget=66.0)
    assert lost["final"] < 66 and lost["worst_day_pct"] >= -2.2  # the daily limit holds


# ---------------- eToro API emulation ----------------
def resp(body=None, status=200, headers=None):
    return SimpleNamespace(status_code=status, content=b"x" if body is not None else b"", json=lambda: body,
                           headers=headers or {}, text=json.dumps(body))


class FakeEtoro:
    """In-memory eToro: orders, positions, portfolio, history. Shapes follow the public API reference."""

    def __init__(self):
        self.calls, self.orders, self.by_ref, self.positions, self.closes, self.trades = [], {}, {}, {}, {}, []
        self.queue = []  # canned responses or exceptions served first
        self.fill, self.reject, self.lag = True, None, False
        self.ids = 100

    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        path = url.replace(BASE_URL, "")
        self.calls.append(SimpleNamespace(method=method, path=path, params=params, body=json, headers=headers))
        if self.queue:
            item = self.queue.pop(0)
            if isinstance(item, Exception):
                raise item
            if item is not None:
                return item
        if method == "POST" and path.endswith("/orders"):
            return self._open(json, headers["x-request-id"])
        if "orders:lookup" in path:
            oid = params.get("orderId") or self.by_ref.get(params.get("referenceId"))
            return resp(self.orders[oid]) if oid in self.orders else resp({"title": "not found"}, 404)
        if method == "DELETE":
            self.orders[int(path.rsplit("/", 1)[1])]["status"] = {"id": 7, "name": "Canceled"}
            return resp({})
        if "market-close-orders/positions/" in path:
            pid = int(path.rsplit("/", 1)[1])
            if pid not in self.positions:
                return resp({"title": "position not found"}, 404)
            pos = self.positions.pop(pid)
            self.ids += 1
            self.closes[self.ids] = {"statusID": 3, "positions": [{"positionID": pid, "rate": 100.5, "units": pos["units"]}]}
            return resp({"orderForClose": {"orderID": self.ids, "positionID": pid}, "token": "t"})
        if "/close-orders/" in path:
            return resp(self.closes[int(path.rsplit("/", 1)[1])])
        if path.endswith("/portfolio"):
            listed = [] if self.lag else [{"positionID": pid, **p} for pid, p in self.positions.items()]
            return resp({"clientPortfolio": {"positions": listed, "credit": 100000.0}})
        if path.endswith("/history") and "trade" in path:
            return resp(self.trades)
        if path.endswith("/market-data/rates"):
            return resp({"results": [{"instrumentId": int(i), "bid": 99.99, "ask": 100.01, "date": "2026-11-12T15:00:00.5",
                                      "quoteType": "realtime"} for i in params["instrumentIds"].split(",")]})
        if path.endswith("/market-data/instruments"):
            known = {"NVDA": 1137, "AAPL": 1001}
            return resp({"results": [{"symbol": s, "instrumentId": known[s]} for s in params["symbols"].split(",") if s in known]})
        if "/history/candles/" in path:
            rows = [{"fromDate": f"2026-11-12T14:{31 - i:02d}:00Z", "open": 100, "high": 101, "low": 99, "close": 100.5} for i in range(2)]
            return resp({"interval": "OneMinute", "candles": [{"instrumentId": 1137, "candles": rows}]})
        raise AssertionError(f"unexpected request {method} {path}")

    def _open(self, body, ref):
        self.ids += 1
        oid = self.ids
        order = {"orderId": oid, "status": {"id": 1, "name": "Received"}, "positionExecutions": []}
        if self.reject:
            order["status"] = {"id": 4, "name": "Rejected", "errorMessage": self.reject}
        elif self.fill:
            pid = oid + 5000
            units = body["amount"] / 100.01
            self.positions[pid] = {"instrumentID": body["instrumentId"], "units": units, "openRate": 100.01}
            order["status"] = {"id": 3, "name": "Filled"}
            order["positionExecutions"] = [{"positionId": pid, "state": "open", "remainingUnits": units,
                                            "openingData": {"units": units, "avgPrice": 100.01}}]
        self.orders[oid], self.by_ref[ref] = order, oid
        return resp({"orderId": oid, "token": "t", "referenceId": ref})


def etoro(cfg, fake, env="demo"):
    cfg.etoro_env = env
    client = EtoroClient("app-key", "user-key", env, session=fake, sleep=lambda s: None, clock=lambda: 0.0)
    ticks = iter(range(0, 10**6, 100))
    return client, EtoroBroker(cfg, client, {"NVDA": 1137, "AAPL": 1001}, sleep=lambda s: None, clock=lambda: next(ticks))


def test_client_sends_keys_and_uses_demo_or_real_paths(cfg):
    fake = FakeEtoro()
    client, _ = etoro(cfg, fake)
    assert client.instruments(["NVDA", "AAPL", "NOPE"]) == {"NVDA": 1137, "AAPL": 1001}
    call = fake.calls[-1]
    assert call.headers["x-api-key"] == "app-key" and call.headers["x-user-key"] == "user-key"
    assert len(call.headers["x-request-id"]) == 36 and call.params["symbols"] == "NVDA,AAPL,NOPE"
    assert client.rates([1137])[1137]["ask"] == 100.01
    bars = client.candles(1137, 2)
    assert [b.time.minute for b in bars] == [30, 31] and bars[0].close == 100.5  # oldest first
    client.portfolio()
    assert fake.calls[-1].path == "/api/v1/trading/info/demo/portfolio"
    order_id = client.open_order(1137, 10.0, 99.0, 101.0, "ref-1")
    sent = fake.calls[-1]
    assert sent.path == "/api/v2/trading/execution/demo/orders" and sent.headers["x-request-id"] == "ref-1"
    assert sent.body == {"action": "open", "transaction": "buy", "instrumentId": 1137, "orderType": "mkt",
                         "settlementType": "real", "leverage": 1, "amount": 10.0, "orderCurrency": "usd",
                         "stopLossRate": 99.0, "takeProfitRate": 101.0, "stopLossType": "fixed"}
    assert client.order_info(reference_id="ref-1")["orderId"] == order_id

    real_fake = FakeEtoro()
    real, _ = etoro(cfg, real_fake, env="real")
    real.portfolio()
    oid = real.open_order(1137, 10.0, 99.0, 101.0, "r")
    real.order_info(order_id=oid)
    real.close_info(real.close_position(oid + 5000, 1137))
    assert [c.path for c in real_fake.calls] == [
        "/api/v1/trading/info/portfolio", "/api/v2/trading/execution/orders", "/api/v2/trading/info/orders:lookup",
        f"/api/v1/trading/execution/market-close-orders/positions/{oid + 5000}",
        f"/api/v1/trading/info/real/close-orders/{oid + 1}"]
    assert real_fake.calls[3].body == {"InstrumentID": 1137}


def test_client_errors_rate_limits_and_retries(cfg):
    fake = FakeEtoro()
    client, _ = etoro(cfg, fake)
    fake.queue = [resp({}, 429, {"Retry-After": "2"})]
    assert client.rates([1137])  # waited and retried
    fake.queue = [resp({"errorCode": "Unauthorized"}, 401)]
    with pytest.raises(EtoroError, match="Demo environment"):
        client.portfolio()
    fake.queue = [requests.ConnectionError("down"), None]
    assert "positions" in client.portfolio()  # reads are retried
    fake.queue = [requests.Timeout("slow")]
    with pytest.raises(EtoroUncertain):
        client.open_order(1137, 10, 99, 101, "ref")  # orders are never blindly re-sent
    fake.queue = [resp({"title": "bad"}, 400)]
    with pytest.raises(EtoroError, match="HTTP 400"):
        client.rates([1])
    with pytest.raises(EtoroError, match="not set"):
        EtoroClient("", "")

    waited, now = [], [0.0]
    limiter = RateLimiter(2, clock=lambda: now[0], sleep=lambda s: (waited.append(s), now.__setitem__(0, now[0] + s)))
    for _ in range(3):
        limiter.wait()
    assert waited and waited[0] == pytest.approx(60.05)  # the third call in a minute has to wait


def test_feed_turns_rates_into_quotes(cfg):
    fake = FakeEtoro()
    client, _ = etoro(cfg, fake)
    feed = EtoroFeed(client, {"NVDA": 1137})
    q = feed.quotes(OPEN)["NVDA"]
    assert (q.bid, q.ask) == (99.99, 100.01) and q.time == datetime(2026, 11, 12, 15, 0, 0, 500000, tzinfo=timezone.utc)
    fake.queue = [resp({"results": [{"instrumentId": 1137, "bid": 1, "ask": 2, "quoteType": "delayed"}]})]
    assert feed.quotes(OPEN) == {}  # delayed prices are never traded
    assert feed.gbpusd() == pytest.approx(100.0) and len(feed.history("NVDA")) == 2


def test_broker_opens_confirms_and_closes(cfg):
    fake = FakeEtoro()
    _, broker = etoro(cfg, fake)
    fill = broker.open("NVDA", 10.0, 99.0, 101.0, quote("NVDA", 100, OPEN))
    assert fill.price == 100.01 and fill.units == pytest.approx(10 / 100.01) and fill.order_id == 101
    pos = held("NVDA", pid=fill.position_id, entry=fill.price, units=fill.units)
    closed = broker.close(pos, quote("NVDA", 100.5, OPEN), "time limit")
    assert closed.price == 100.5 and fake.positions == {}


def test_broker_handles_rejections_timeouts_and_unfilled_orders(cfg):
    fake = FakeEtoro()
    _, broker = etoro(cfg, fake)
    fake.reject = "Insufficient funds"
    with pytest.raises(BrokerError, match="Insufficient funds"):
        broker.open("NVDA", 10.0, 99.0, 101.0, quote("NVDA", 100, OPEN))

    fake = FakeEtoro()
    _, broker = etoro(cfg, fake)
    real_request = fake.request

    def flaky(method, url, **kw):  # the order is created, but the reply is lost
        if method == "POST" and url.endswith("/orders"):
            real_request(method, url, **kw)
            raise requests.Timeout("reply lost")
        return real_request(method, url, **kw)

    fake.request = flaky
    fill = broker.open("NVDA", 10.0, 99.0, 101.0, quote("NVDA", 100, OPEN))
    assert fill.position_id and len(fake.positions) == 1  # found by request id: exactly one position, no duplicate

    fake = FakeEtoro()
    _, broker = etoro(cfg, fake)
    fake.fill = False
    with pytest.raises(BrokerError, match="did not fill"):
        broker.open("NVDA", 10.0, 99.0, 101.0, quote("NVDA", 100, OPEN))
    assert any(c.method == "DELETE" for c in fake.calls)  # cancelled rather than left hanging


def test_broker_sync_detects_etoro_stop_and_target_closes(cfg):
    fake = FakeEtoro()
    _, broker = etoro(cfg, fake)
    fill = broker.open("NVDA", 10.0, 99.0, 101.0, quote("NVDA", 100, OPEN))
    pos = held("NVDA", pid=fill.position_id, entry=fill.price, units=fill.units, stop=99.0, target=101.0)
    pos.order_id = fill.order_id
    quotes = {"NVDA": quote("NVDA", 100, OPEN)}
    assert broker.sync([pos], quotes, OPEN) == []  # still open

    fake.lag = True  # portfolio view lags, but the order still reports the position open
    assert broker.sync([pos], quotes, OPEN) == []
    fake.lag = False

    del fake.positions[pos.position_id]  # eToro's stop-loss fired
    fake.orders[fill.order_id]["positionExecutions"][0]["state"] = "closed"
    fake.trades = [{"positionId": pos.position_id, "closeRate": 98.97, "netProfit": -0.104}]
    closure, = broker.sync([pos], quotes, OPEN)
    assert (closure.reason, closure.price, closure.pnl) == ("stop-loss", 98.97, -0.104)

    fake.trades = [{"positionId": pos.position_id, "closeRate": 101.02, "netProfit": 0.1}]
    assert broker.sync([pos], quotes, OPEN)[0].reason == "take-profit"

    fake.trades = []  # history not updated yet: estimated from where the price is
    low = broker.sync([pos], {"NVDA": quote("NVDA", 98.9, OPEN)}, OPEN)[0]
    assert low.reason == "stop-loss (estimated)" and low.price == 99.0 and low.pnl is None


def test_broker_close_of_an_already_closed_position_uses_history(cfg):
    fake = FakeEtoro()
    _, broker = etoro(cfg, fake)
    pos = held("NVDA", pid=777)
    fake.trades = [{"positionId": 777, "closeRate": 99.0, "netProfit": -0.1}]
    closed = broker.close(pos, quote("NVDA", 100, OPEN), "time limit")
    assert closed.price == 99.0 and closed.pnl == -0.1
    fake.trades = []
    with pytest.raises(BrokerError):
        broker.close(pos, quote("NVDA", 100, OPEN), "time limit")


def test_engine_with_etoro_broker_end_to_end(cfg):
    cfg.symbols = ["NVDA"]
    fake = FakeEtoro()
    _, broker = etoro(cfg, fake)
    engine = DayEngine(cfg, broker, DayState(budget=66.0))
    warm_up(engine, symbols=("NVDA",))
    dip(engine, symbols=("NVDA",))
    assert len(fake.positions) == 1 and engine.state.positions[0].order_id
    minute(engine, 71, {"NVDA": 100.0})
    minute(engine, 72, {"NVDA": 100.0})
    assert fake.positions == {} and engine.state.positions == [] and engine.state.trades_today == 2
    assert engine.trades[-1]["reason"] == "back to average" and engine.trades[-1]["price"] == 100.5


# ---------------- runner, AI screen, report ----------------
def test_runner_waits_for_the_open_warms_up_and_flattens_on_stop(cfg):
    cfg.symbols = ["AAA"]
    cfg.poll_seconds = 30
    now = [OPEN - timedelta(minutes=2)]
    slept, screens = [], []

    class Feed:
        def history(self, symbol, minutes=150):
            return [SimpleNamespace(time=OPEN - timedelta(days=1), open=1, high=1, low=1, close=1)]

        def quotes(self, when):
            return {"AAA": quote("AAA", 100, when)}

    def sleep(seconds):
        slept.append(seconds)
        now[0] += timedelta(seconds=seconds)

    engine = make_engine(cfg)
    run(cfg, engine, Feed(), screen=lambda: screens.append(1) or ["ZZZ"], sleep=sleep, clock=lambda: now[0], max_loops=8)
    assert slept[:2] == [60.0, 60.0] and engine.state.session == "2026-11-12"  # waited for 14:30 UK
    assert screens == [1] and engine.state.excluded_today == ["ZZZ"]  # screened once per session
    assert len(engine.series["AAA"].bars) == 2 and engine.series["AAA"].bars[0].close == 100  # yesterday not loaded

    engine.state.positions.append(held())

    def interrupt(seconds):
        raise KeyboardInterrupt

    run(cfg, engine, Feed(), sleep=interrupt, clock=lambda: now[0])
    assert engine.state.positions == [] and engine.broker.closed[-1] == ("AAA", "bot stopped")


def test_ai_screen_names_shares_to_avoid_and_fails_safe(cfg):
    cfg.symbols = ["NVDA", "AAPL"]
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        body = {"summary": "quiet", "avoid": [{"symbol": "nvda", "reason": "earnings tonight"}, {"symbol": "TSLA", "reason": "n/a"}]}
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=json.dumps(body))])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert premarket_screen(cfg, lambda s: [{"title": f"{s} news"}], client) == ["NVDA"]  # unknown symbols ignored
    assert seen["model"] == "claude-opus-5-5" and "NVDA news" in seen["messages"][0]["content"]

    def boom(**kwargs):
        raise RuntimeError("no credentials")

    broken = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=boom)))
    assert premarket_screen(cfg, lambda s: [], broken) == []
    refused = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **k: SimpleNamespace(stop_reason="refusal", content=[]))))
    assert premarket_screen(cfg, lambda s: [], refused) == []


def test_day_report_groups_by_day():
    trades = [{"time": "2026-11-12T15:00:00+00:00", "side": "buy", "pnl": None, "reason": "dip"},
              {"time": "2026-11-12T15:10:00+00:00", "side": "sell", "pnl": 0.05, "reason": "back to average"},
              {"time": "2026-11-13T15:00:00+00:00", "side": "buy", "pnl": None, "reason": "dip"},
              {"time": "2026-11-13T15:20:00+00:00", "side": "sell", "pnl": -0.11, "reason": "stop-loss"}]
    report = day_report(trades, 66.0)
    assert (report["days"], report["round_trips"], report["win_rate_pct"], report["winning_days"]) == (2, 2, 50, 1)
    assert report["realised_pnl"] == -0.06 and report["return_pct"] == -0.09
    assert report["by_day"]["2026-11-13"] == {"trades": 2, "pnl": -0.11}


# =====================================================================
# Leverage: sizing by risk, margin and the broker's minimum size
# =====================================================================
class LeveragedBroker(FakeBroker):
    def __init__(self, **spec):
        super().__init__()
        self._spec = Spec(**{"min_size": 0.5, "size_decimals": 2, "margin_pct": 0.05, **spec})

    def spec(self, symbol):
        return self._spec


def test_leveraged_trade_is_sized_by_risk_and_ties_up_only_margin(cfg):
    engine = make_engine(cfg, LeveragedBroker(), budget=10_000.0)
    warm_up(engine)
    dip(engine)
    (symbol, amount, stop, target), = engine.broker.opened
    pos, = engine.state.positions
    loss_at_stop = (pos.entry - pos.stop) * pos.units
    assert 20 < loss_at_stop <= 25.5  # 0.25% of £10,000, never more
    assert pos.units == round(pos.units, 2) and pos.units >= 0.5
    assert pos.margin == pytest.approx(pos.amount * 0.05, rel=1e-6)
    assert engine.cash() == pytest.approx(10_000 - pos.margin)  # exposure of thousands, margin of a few hundred


def test_minimum_bet_too_big_for_a_small_account_is_refused_not_oversized(cfg):
    engine = make_engine(cfg, LeveragedBroker(), budget=50.0)
    warm_up(engine)
    dip(engine)
    assert engine.broker.opened == [] and engine.state.positions == []
    why = engine.blocked["AAA"]
    assert "smallest trade" in why and "limit per trade" in why
    assert engine.status()["blocked"] == {"AAA": why}


def test_leverage_respects_free_margin_min_stop_distance_and_closed_markets(cfg):
    cfg.risk_per_trade_pct = 0.02  # a big risk budget, so margin becomes the binding limit
    engine = make_engine(cfg, LeveragedBroker(margin_pct=0.2, min_size=0.01), budget=100.0)
    warm_up(engine)
    dip(engine)
    pos, = engine.state.positions
    assert pos.margin <= 100 * (1 - cfg.cash_reserve_pct) + 1e-6 and engine.cash() >= 100 * cfg.cash_reserve_pct - 1e-6

    engine = make_engine(cfg, LeveragedBroker(min_stop_distance=1.0), budget=10_000.0)  # stops must be 1 point away
    warm_up(engine)
    dip(engine)
    _, _, stop, target = engine.broker.opened[0]
    ask = engine.state.positions[0].entry
    assert ask - stop >= 1.0 and target - ask >= 1.0

    engine = make_engine(cfg, LeveragedBroker(tradable=False), budget=10_000.0)
    warm_up(engine)
    dip(engine)
    assert engine.broker.opened == [] and "not open" in engine.blocked["AAA"]


def test_ig_backtest_small_budget_cannot_trade_and_large_budget_can(cfg):
    cfg.broker, cfg.symbols, cfg.z_entry = "ig", ["US500"], 2.0
    idx = pd.date_range(OPEN, periods=390, freq="1min")
    base = [7600 + 1.0 * (-1) ** i for i in range(390)]
    for i in (100, 101, 200, 201):
        base[i] = 7585.0
    frame = pd.DataFrame({"open": base, "high": [p + 3 for p in base], "low": [p - 1 for p in base], "close": base}, index=idx)
    small = replay(cfg, {"US500": frame}, budget=50.0)
    assert small["trades"] == 0 and small["final"] == 50.0 and "smallest trade" in small["not_traded"]["US500"]
    big = replay(cfg, {"US500": frame}, budget=10_000.0)
    assert big["round_trips"] == 2 and big["final"] > 10_000 and big["currency"] == "£" and big["not_traded"] == {}
    assert SimBroker(cfg).spec("US500") == Spec(min_size=0.5, size_decimals=2, margin_pct=0.05)


# =====================================================================
# IG, emulated from IG's API reference and the trading-ig library
# =====================================================================
from trading_bot.day.ig import IGBroker, IGClient, IGError, IGFeed, resolve_epics  # noqa: E402

US500 = "IX.D.SPTRD.DAILY.IP"


class FakeIG:
    """In-memory IG. Method names, arguments and reply shapes follow trading_ig.IGService."""

    def __init__(self, accounts=None, default="SB1"):
        self.accounts = accounts or [{"accountId": "SB1", "accountType": "SPREADBET"}, {"accountId": "CF1", "accountType": "CFD"}]
        self.current, self.logins, self.n = default, 0, 0
        self.calls, self.positions, self.activities, self.fail = [], {}, [], []
        self.reject = None
        self.markets = {US500: {
            "instrument": {"epic": US500, "name": "US 500", "expiry": "DFB", "marginFactor": 5, "marginFactorUnit": "PERCENTAGE",
                           "currencies": [{"code": "USD", "isDefault": False}, {"code": "GBP", "isDefault": True}]},
            "dealingRules": {"minDealSize": {"unit": "POINTS", "value": 0.5},
                             "minNormalStopOrLimitDistance": {"unit": "POINTS", "value": 4.0}},
            "snapshot": {"bid": 7599.8, "offer": 7600.2, "marketStatus": "TRADEABLE", "delayTime": 0, "decimalPlacesFactor": 1}}}

    def price(self, mid, epic=US500):
        self.markets[epic]["snapshot"].update(bid=round(mid - 0.2, 1), offer=round(mid + 0.2, 1))

    def _fail(self):
        if self.fail:
            raise self.fail.pop(0)

    def create_session(self, *a, **k):
        self.logins += 1
        kind = next(acc["accountType"] for acc in self.accounts if acc["accountId"] == self.current)
        return {"currentAccountId": self.current, "accountType": kind, "accounts": self.accounts, "currencyIsoCode": "GBP",
                "accountInfo": {"balance": 10000.0, "available": 10000.0}}

    def switch_account(self, account_id, default):
        self.current = account_id

    def fetch_accounts(self):
        return {"accounts": [{**acc, "balance": {"balance": 10000.0, "available": 9000.0}} for acc in self.accounts]}

    def fetch_market_by_epic(self, epic):
        self._fail()
        return self.markets[epic]

    def fetch_markets_by_epics(self, epics, detailed=True):
        self._fail()
        self.calls.append(("snapshots", epics, detailed))
        return [{"instrument": {"epic": e}, "snapshot": self.markets[e]["snapshot"]} for e in epics.split(",")]

    def fetch_historical_prices_by_epic(self, epic, numpoints=None, pagesize=None):
        def px(v):
            return {"bid": v - 0.2, "ask": v + 0.2}

        return {"prices": [{"snapshotTimeUTC": f"2026-11-12T14:{31 + i}:00", "openPrice": px(7600), "highPrice": px(7603),
                            "lowPrice": px(7599), "closePrice": px(7601 + i)} for i in range(2)]}

    def fetch_open_positions(self):
        return {"positions": [{"position": {"dealId": d, **p}, "market": {"epic": p["epic"]}} for d, p in self.positions.items()]}

    def fetch_account_activity(self, from_date=None, detailed=False):
        return {"activities": self.activities}

    def create_open_position(self, **kw):
        self.calls.append(("open", kw))
        if self.reject:
            return {"dealStatus": "REJECTED", "reason": self.reject}
        self.n += 1
        deal, level = f"DIAAAA{self.n}", self.markets[kw["epic"]]["snapshot"]["offer"]
        self.positions[deal] = {"epic": kw["epic"], "level": level, "size": kw["size"]}
        return {"dealStatus": "ACCEPTED", "reason": "SUCCESS", "dealId": deal, "dealReference": f"REF{self.n}", "level": level,
                "size": kw["size"], "stopLevel": kw["stop_level"], "limitLevel": kw["limit_level"],
                "affectedDeals": [{"dealId": deal, "status": "OPENED"}]}

    def close_open_position(self, **kw):
        self.calls.append(("close", kw))
        if kw["deal_id"] not in self.positions:
            return {"dealStatus": "REJECTED", "reason": "POSITION_NOT_FOUND"}
        pos = self.positions.pop(kw["deal_id"])
        level = self.markets[pos["epic"]]["snapshot"]["bid"]
        return {"dealStatus": "ACCEPTED", "reason": "SUCCESS", "dealId": kw["deal_id"], "level": level, "size": kw["size"],
                "profit": round((level - pos["level"]) * kw["size"], 2), "profitCurrency": "GBP"}


def ig(cfg, fake=None):
    cfg.broker, cfg.symbols = "ig", ["US500"]
    fake = fake or FakeIG()
    tick = iter(range(0, 10**7, 100))
    client = IGClient.login("user", "pass", "key", service=fake, sleep=lambda s: None, clock=lambda: next(tick))
    epics = resolve_epics(cfg.symbols)
    return fake, client, IGFeed(client, epics), IGBroker(cfg, client, epics, clock=lambda: next(tick))


def test_ig_login_picks_the_spread_betting_account_and_explains_failures(cfg):
    fake, client, _, _ = ig(cfg, FakeIG(default="CF1"))
    assert (client.account_id, client.account_type, fake.current) == ("SB1", "SPREADBET", "SB1")  # switched from the CFD default
    assert client.balance() == {"balance": 10000.0, "available": 9000.0}
    with pytest.raises(IGError, match="spread betting accounts"):
        IGClient.login("u", "p", "k", service=FakeIG(accounts=[{"accountId": "CF1", "accountType": "CFD"}], default="CF1"))
    with pytest.raises(IGError, match="not set"):
        IGClient.login("", "", "")

    class BadLogin(FakeIG):
        def create_session(self, *a, **k):
            raise Exception("error.security.invalid-details")

    with pytest.raises(IGError, match="username or password"):
        IGClient.login("u", "p", "k", service=BadLogin())
    assert resolve_epics(["US500", "IX.D.FTSE.DAILY.IP"]) == {"US500": US500, "IX.D.FTSE.DAILY.IP": "IX.D.FTSE.DAILY.IP"}
    with pytest.raises(IGError, match="not a known IG market"):
        resolve_epics(["APPLE"])


def test_ig_feed_quotes_only_live_tradeable_prices(cfg):
    fake, _, feed, _ = ig(cfg)
    q = feed.quotes(OPEN)["US500"]
    assert (q.bid, q.ask, q.time) == (7599.8, 7600.2, OPEN) and fake.calls[-1] == ("snapshots", US500, False)
    fake.markets[US500]["snapshot"]["marketStatus"] = "EDITS_ONLY"
    assert feed.quotes(OPEN) == {}
    fake.markets[US500]["snapshot"].update(marketStatus="TRADEABLE", delayTime=15)
    assert feed.quotes(OPEN) == {}
    bars = feed.history("US500")
    assert [(b.time.minute, b.open, b.high, b.low, b.close) for b in bars] == [(31, 7600, 7603, 7599, 7601), (32, 7600, 7603, 7599, 7602)]


def test_ig_broker_reads_market_rules(cfg):
    fake, _, _, broker = ig(cfg)
    assert broker.spec("US500") == Spec(min_size=0.5, size_decimals=2, margin_pct=0.05, min_stop_distance=4.0, tradable=True)
    fake.markets[US500]["dealingRules"]["minNormalStopOrLimitDistance"] = {"unit": "PERCENTAGE", "value": 0.1}
    fake.markets[US500]["snapshot"]["marketStatus"] = "CLOSED"
    broker.details("US500", refresh=True)
    spec = broker.spec("US500")
    assert spec.min_stop_distance == pytest.approx(7.6002) and not spec.tradable


def test_ig_broker_opens_and_closes_a_spread_bet(cfg):
    fake, _, _, broker = ig(cfg)
    fill = broker.open("US500", 7676.2, 7575.83, 7624.51, Quote("US500", 7599.8, 7600.2, OPEN))
    kind, sent = fake.calls[-1]
    assert kind == "open" and sent == {
        "currency_code": "GBP", "direction": "BUY", "epic": US500, "expiry": "DFB", "force_open": True,
        "guaranteed_stop": False, "level": None, "limit_distance": None, "limit_level": 7624.5, "order_type": "MARKET",
        "quote_id": None, "size": 1.01, "stop_distance": None, "stop_level": 7575.8, "trailing_stop": False,
        "trailing_stop_increment": None}
    assert (fill.position_id, fill.units, fill.price, fill.stop, fill.target) == ("DIAAAA1", 1.01, 7600.2, 7575.8, 7624.5)
    pos = DayPosition("US500", fill.position_id, fill.units, fill.price, fill.units * fill.price, fill.stop, fill.target,
                      OPEN.isoformat(), "test")
    fake.price(7610.0)
    closed = broker.close(pos, Quote("US500", 7609.8, 7610.2, OPEN), "time limit")
    assert closed.price == 7609.8 and closed.pnl == pytest.approx(9.7, abs=0.01)  # (7609.8 - 7600.2) x £1.01 a point
    assert fake.calls[-1][1] == {"deal_id": "DIAAAA1", "direction": "SELL", "epic": None, "expiry": None, "level": None,
                                 "order_type": "MARKET", "quote_id": None, "size": 1.01}
    assert fake.positions == {}


def test_ig_rejections_are_explained(cfg):
    fake, _, _, broker = ig(cfg)
    fake.reject = "INSUFFICIENT_FUNDS"
    with pytest.raises(BrokerError, match="not enough money"):
        broker.open("US500", 7600.0, 7575.8, 7624.5, Quote("US500", 7599.8, 7600.2, OPEN))
    fake.reject = "SOME_NEW_REASON"
    with pytest.raises(BrokerError, match="some new reason"):
        broker.open("US500", 7600.0, 7575.8, 7624.5, Quote("US500", 7599.8, 7600.2, OPEN))


def test_ig_session_expiry_and_rate_limits_are_retried(cfg):
    fake, client, feed, _ = ig(cfg)
    fake.fail = [Exception("error.security.client-token-invalid")]
    assert feed.quotes(OPEN) and fake.logins == 2  # logged in again, then repeated the request
    fake.fail = [Exception("error.public-api.exceeded-account-allowance")]
    assert feed.quotes(OPEN)  # waited, then repeated
    fake.fail = [Exception("boom"), Exception("boom")]
    with pytest.raises(IGError, match="boom"):
        feed.quotes(OPEN)


def test_ig_sync_books_positions_closed_by_igs_own_stop_or_limit(cfg):
    fake, _, _, broker = ig(cfg)
    quote = Quote("US500", 7599.8, 7600.2, OPEN)
    fill = broker.open("US500", 7676.2, 7575.8, 7624.5, quote)
    pos = DayPosition("US500", fill.position_id, fill.units, fill.price, 7676.2, 7575.8, 7624.5, OPEN.isoformat(), "test")
    assert broker.sync([pos], {"US500": quote}, OPEN) == []  # still open
    del fake.positions[pos.position_id]  # IG's stop fired
    assert broker.sync([pos], {"US500": quote}, OPEN) == []  # first miss: look once more before believing it
    fake.activities = [{"details": {"level": 7575.3, "actions": [{"actionType": "POSITION_CLOSED", "affectedDealId": pos.position_id}]}}]
    closure, = broker.sync([pos], {"US500": quote}, OPEN)
    assert (closure.reason, closure.price) == ("stop-loss", 7575.3)  # the real level, slippage included
    fake.activities[0]["details"]["level"] = 7624.5
    assert broker.sync([pos], {"US500": quote}, OPEN)[0].reason == "take-profit"
    fake.activities = []  # no record yet: after a few looks, book it at the level that must have triggered
    assert broker.sync([pos], {"US500": Quote("US500", 7570, 7570.4, OPEN)}, OPEN)[0].reason == "stop-loss (estimated)"
    # closing a position IG has already closed falls back to the activity record
    fake.activities = [{"details": {"level": 7575.3, "actions": [{"actionType": "POSITION_CLOSED", "affectedDealId": pos.position_id}]}}]
    assert broker.close(pos, quote, "time limit").price == 7575.3
    fake.activities = []
    with pytest.raises(BrokerError):
        broker.close(pos, quote, "time limit")


def ig_minute(engine, feed, fake, i, price):
    t = OPEN + timedelta(minutes=i)
    fake.price(price + 3.0)
    engine.on_quotes(t, feed.quotes(t))
    fake.price(price)
    engine.on_quotes(t + timedelta(seconds=30), feed.quotes(t + timedelta(seconds=30)))


def ig_session(cfg, budget):
    fake, _, feed, broker = ig(cfg)
    engine = DayEngine(cfg, broker, DayState(budget=budget))
    for i in range(69):
        ig_minute(engine, feed, fake, i, 7600 + (-1) ** i)
    ig_minute(engine, feed, fake, 69, 7585.0)  # a sharp drop
    ig_minute(engine, feed, fake, 70, 7585.0)
    return fake, feed, engine


def test_engine_trades_ig_end_to_end_with_a_funded_account(cfg):
    fake, feed, engine = ig_session(cfg, budget=10_000.0)
    pos, = engine.state.positions
    assert len(fake.positions) == 1 and pos.position_id == "DIAAAA1" and pos.units >= 0.5
    assert (pos.entry - pos.stop) * pos.units <= 25.5 and pos.entry - pos.stop >= 4.0  # £25 risk, IG's stop distance
    assert pos.margin == pytest.approx(pos.amount * 0.05, rel=1e-6)
    ig_minute(engine, feed, fake, 71, 7600.0)
    ig_minute(engine, feed, fake, 72, 7600.0)
    assert fake.positions == {} and engine.state.positions == []
    sell = engine.trades[-1]
    assert sell["reason"] == "back to average" and sell["currency"] == "£" and sell["pnl"] == pytest.approx(engine.state.realized)
    assert 5 < engine.state.realized < 25  # about 15 points x £1 a point


def test_precheck_and_runner_warn_when_nothing_can_be_traded(cfg, caplog):
    fake, _, feed, broker = ig(cfg)
    engine = DayEngine(cfg, broker, DayState(budget=50.0))
    blocked = engine.precheck(feed.quotes(OPEN))
    assert "smallest trade" in blocked["US500"] and "£3,800" in blocked["US500"]
    assert DayEngine(cfg, broker, DayState(budget=10_000.0)).precheck(feed.quotes(OPEN)) == {}
    fake.markets[US500]["instrument"]["marginFactor"] = 20  # a market needing 20% margin: £760 for the smallest bet
    broker.details("US500", refresh=True)
    loose = DayConfig(broker="ig", symbols=["US500"], risk_per_trade_pct=0.02, state_dir=cfg.state_dir).validate()
    assert "margin" in DayEngine(loose, broker, DayState(budget=300.0)).precheck(feed.quotes(OPEN))["US500"]
    fake.markets[US500]["instrument"]["marginFactor"] = 5
    broker.details("US500", refresh=True)
    now = [OPEN + timedelta(minutes=1)]

    def sleep(seconds):
        now[0] += timedelta(seconds=seconds)

    feed.history = lambda name, minutes=80: []
    with caplog.at_level("WARNING"):
        run(cfg, engine, feed, sleep=sleep, clock=lambda: now[0], max_loops=3)
    assert "NOTHING CAN BE TRADED" in caplog.text and not any(c[0] == "open" for c in fake.calls)


def test_engine_refuses_ig_trades_on_a_fifty_pound_account(cfg):
    fake, feed, engine = ig_session(cfg, budget=50.0)
    assert fake.positions == {} and not any(c[0] == "open" for c in fake.calls)
    assert "smallest trade" in engine.blocked["US500"] and "£3,79" in engine.blocked["US500"]  # 0.5 x 7,585 points


def test_check_command_tells_a_small_account_the_truth(cfg, monkeypatch, capsys):
    import trading_bot.day.cli as cli

    fake, client, feed, broker = ig(cfg)
    monkeypatch.setattr(cli, "_ig", lambda c: (client, feed, broker))
    for name in ("DAY_BROKER", "DAY_PRESET", "DAY_SYMBOLS", "DAY_BUDGET_GBP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DAY_SYMBOLS", "US500")
    args = SimpleNamespace(preset=None, broker="ig", dry_run=False, real=False, budget=None)
    cli.cmd_check(args)
    out = capsys.readouterr().out
    assert "spread betting account SB1" in out and "TOO BIG for this budget" in out and "cannot trade any" in out
    args.budget = 10_000.0
    cli.cmd_check(args)
    out = capsys.readouterr().out
    assert "->  OK" in out and "1 of 1 markets can be traded" in out
