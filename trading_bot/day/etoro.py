"""eToro Public API client, broker and quote feed.

Built from eToro's published API reference (https://api-portal.etoro.com):
  open    POST /api/v2/trading/execution[/demo]/orders
  confirm GET  /api/v2/trading/info[/demo]/orders:lookup
  close   POST /api/v1/trading/execution[/demo]/market-close-orders/positions/{positionId}
  rates   GET  /api/v2/market-data/rates
Limits: 20 trade requests a minute, 60 account reads, 120 market-data reads.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from .bars import Bar, Quote
from .config import DayConfig
from .engine import BrokerError, Closure, CloseFill, DayPosition, OpenFill

log = logging.getLogger(__name__)
BASE_URL = "https://public-api.etoro.com"
FILLED = {3, 5}  # Filled, PartiallyFilled
FAILED = {4, 7, 8, 9, 10}  # Rejected, Canceled, Expired, ...


class EtoroError(BrokerError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class EtoroUncertain(EtoroError):
    """A request that changes something may or may not have gone through (timeout or server error)."""


class RateLimiter:
    """At most `per_minute` calls in any 60 seconds; waits when the budget is used up."""

    def __init__(self, per_minute: int, clock=time.monotonic, sleep=time.sleep):
        self.per_minute, self.clock, self.sleep = per_minute, clock, sleep
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        now = self.clock()
        while self.calls and now - self.calls[0] >= 60:
            self.calls.popleft()
        if len(self.calls) >= self.per_minute:
            self.sleep(60 - (now - self.calls[0]) + 0.05)
            now = self.clock()
            while self.calls and now - self.calls[0] >= 60:
                self.calls.popleft()
        self.calls.append(now)


def _parse_time(text) -> Optional[datetime]:
    try:
        stamp = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


class EtoroClient:
    def __init__(self, api_key: str, user_key: str, env: str = "demo", session=None, sleep=time.sleep,
                 clock=time.monotonic):
        if not api_key or not user_key:
            raise EtoroError("ETORO_API_KEY and ETORO_USER_KEY are not set (see trading_bot/day/README.md)")
        self.api_key, self.user_key, self.demo = api_key, user_key, env == "demo"
        self.session = session or requests.Session()
        self.sleep = sleep
        # a little under eToro's limits, to leave headroom
        self.limiters = {"market": RateLimiter(100, clock, sleep), "info": RateLimiter(50, clock, sleep),
                         "execution": RateLimiter(18, clock, sleep)}

    # ---------- transport ----------
    def _request(self, method: str, path: str, pool: str, params=None, body=None, request_id: Optional[str] = None):
        headers = {"x-request-id": request_id or str(uuid.uuid4()), "x-api-key": self.api_key,
                   "x-user-key": self.user_key, "content-type": "application/json"}
        for attempt in range(3):
            self.limiters[pool].wait()
            try:
                resp = self.session.request(method, BASE_URL + path, headers=headers, params=params, json=body,
                                            timeout=15)
            except requests.RequestException as exc:
                if method == "GET" and attempt < 2:
                    self.sleep(1 + attempt)
                    continue
                raise EtoroUncertain(f"{method} {path}: {exc}") from exc
            if resp.status_code == 429 and attempt < 2:
                self.sleep(min(float(resp.headers.get("Retry-After") or 5), 65))
                continue
            if resp.status_code >= 500:
                if method == "GET" and attempt < 2:
                    self.sleep(1 + attempt)
                    continue
                raise EtoroUncertain(f"{method} {path}: HTTP {resp.status_code}", resp.status_code)
            if resp.status_code in (401, 403):
                raise EtoroError("eToro rejected the API keys. Check the key pair, that the key was created for the "
                                 f"{'Demo' if self.demo else 'Real'} environment, and that it has Write permission.",
                                 resp.status_code)
            if resp.status_code >= 400:
                raise EtoroError(f"{method} {path} failed: HTTP {resp.status_code} {resp.text[:300]}", resp.status_code)
            return resp.json() if resp.content else None
        raise EtoroError(f"{method} {path}: rate limited", 429)

    def _env(self, real: str, demo: str) -> str:
        return demo if self.demo else real

    # ---------- market data ----------
    def instruments(self, symbols: list[str]) -> dict[str, int]:
        data = self._request("GET", "/api/v2/market-data/instruments", "market",
                             params={"symbols": ",".join(symbols), "pageSize": 100})
        found = {r["symbol"].upper(): int(r["instrumentId"]) for r in (data or {}).get("results", [])}
        return {s: found[s] for s in symbols if s in found}

    def rates(self, instrument_ids: list[int]) -> dict[int, dict]:
        data = self._request("GET", "/api/v2/market-data/rates", "market",
                             params={"instrumentIds": ",".join(str(i) for i in instrument_ids)})
        return {int(r["instrumentId"]): r for r in (data or {}).get("results", [])}

    def candles(self, instrument_id: int, count: int = 150) -> list[Bar]:
        """The most recent one-minute candles, oldest first."""
        data = self._request("GET", f"/api/v1/market-data/instruments/{instrument_id}/history/candles/desc/OneMinute/{count}",
                             "market")
        rows = ((data or {}).get("candles") or [{}])[0].get("candles") or []
        bars = []
        for row in rows:
            when = _parse_time(row.get("fromDate"))
            if when:
                bars.append(Bar(when, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])))
        return sorted(bars, key=lambda b: b.time)

    # ---------- account ----------
    def portfolio(self) -> dict:
        data = self._request("GET", self._env("/api/v1/trading/info/portfolio", "/api/v1/trading/info/demo/portfolio"), "info")
        return (data or {}).get("clientPortfolio", data or {})

    def history(self, since: datetime) -> list[dict]:
        path = self._env("/api/v1/trading/info/trade/history", "/api/v1/trading/info/trade/demo/history")
        data = self._request("GET", path, "info", params={"minDate": since.strftime("%Y-%m-%d"), "page": 1, "pageSize": 100})
        return data if isinstance(data, list) else []

    def eligibility(self, symbols: list[str]) -> list[dict]:
        path = self._env("/api/v2/trading/info/eligibility", "/api/v2/trading/info/demo/eligibility")
        data = self._request("POST", path, "info", body={"symbols": symbols, "currency": "USD"})
        return (data or {}).get("eligibilities", [])

    # ---------- orders ----------
    def open_order(self, instrument_id: int, amount: float, stop: float, target: float, request_id: str) -> int:
        body = {"action": "open", "transaction": "buy", "instrumentId": instrument_id, "orderType": "mkt",
                "settlementType": "real", "leverage": 1, "amount": amount, "orderCurrency": "usd",
                "stopLossRate": stop, "takeProfitRate": target, "stopLossType": "fixed"}
        path = self._env("/api/v2/trading/execution/orders", "/api/v2/trading/execution/demo/orders")
        data = self._request("POST", path, "execution", body=body, request_id=request_id)
        return int(data["orderId"])

    def cancel_order(self, order_id: int) -> None:
        path = self._env("/api/v2/trading/execution/orders/", "/api/v2/trading/execution/demo/orders/")
        self._request("DELETE", path + str(order_id), "execution")

    def order_info(self, order_id: Optional[int] = None, reference_id: Optional[str] = None) -> dict:
        path = self._env("/api/v2/trading/info/orders:lookup", "/api/v2/trading/info/demo/orders:lookup")
        params = {"orderId": order_id} if order_id is not None else {"referenceId": reference_id}
        return self._request("GET", path, "info", params=params) or {}

    def close_position(self, position_id: int, instrument_id: int) -> int:
        path = self._env("/api/v1/trading/execution/market-close-orders/positions/",
                         "/api/v1/trading/execution/demo/market-close-orders/positions/")
        data = self._request("POST", path + str(position_id), "execution", body={"InstrumentID": instrument_id})
        return int(data["orderForClose"]["orderID"])

    def close_info(self, order_id: int) -> dict:
        path = self._env("/api/v1/trading/info/real/close-orders/", "/api/v1/trading/info/demo/close-orders/")
        return self._request("GET", path + str(order_id), "info") or {}


class EtoroFeed:
    """Live bid/ask quotes for the watchlist: one request per poll."""

    def __init__(self, client: EtoroClient, ids: dict[str, int]):
        self.client, self.ids = client, ids
        self.symbols = {v: k for k, v in ids.items()}
        self._warned_delayed = False

    def quotes(self, now: datetime) -> dict[str, Quote]:
        out = {}
        for instrument_id, rate in self.client.rates(list(self.ids.values())).items():
            symbol = self.symbols.get(instrument_id)
            bid, ask = rate.get("bid"), rate.get("ask")
            if not symbol or not bid or not ask:
                continue
            if rate.get("quoteType") == "delayed":
                if not self._warned_delayed:
                    log.warning("eToro is sending delayed quotes for %s; it will not be traded", symbol)
                    self._warned_delayed = True
                continue
            out[symbol] = Quote(symbol, float(bid), float(ask), _parse_time(rate.get("date")) or now)
        return out

    def history(self, symbol: str, minutes: int = 150) -> list[Bar]:
        return self.client.candles(self.ids[symbol], minutes)

    def gbpusd(self) -> float:
        rate = self.client.rates([2]).get(2) or {}  # instrument 2 is GBP/USD
        return (float(rate["bid"]) + float(rate["ask"])) / 2


class EtoroBroker:
    """Places real eToro orders (demo or real account). Shares only, no leverage, long only."""

    name = "etoro"
    POLLS = 12
    POLL_SECONDS = 0.5

    def __init__(self, cfg: DayConfig, client: EtoroClient, ids: dict[str, int], sleep=time.sleep,
                 clock=time.monotonic):
        self.cfg, self.client, self.ids, self.sleep, self.clock = cfg, client, ids, sleep, clock
        self._last_sync = -1e9
        self._orphans: list[tuple[int, str]] = []  # (order id, symbol) for orders whose outcome is unknown
        self._missing: dict[int, int] = {}  # position id -> syncs since it vanished
        self._history_ok = True

    # ---------- open ----------
    def _fill_from_order(self, info: dict) -> Optional[OpenFill]:
        for execution in info.get("positionExecutions") or []:
            opening = execution.get("openingData") or {}
            units = opening.get("units") or execution.get("remainingUnits")
            price = opening.get("avgPrice")
            if execution.get("positionId") and units and price:
                return OpenFill(int(execution["positionId"]), float(units), float(price), int(info.get("orderId") or 0))
        return None

    def _await_open(self, order_id: int) -> Optional[OpenFill]:
        """Poll until the order fills. Raises if eToro rejected it; None if still undecided."""
        for _ in range(self.POLLS):
            self.sleep(self.POLL_SECONDS)
            info = self.client.order_info(order_id=order_id)
            status = (info.get("status") or {})
            if status.get("id") in FILLED:
                fill = self._fill_from_order(info)
                if fill:
                    return fill
            elif status.get("id") in FAILED:
                raise BrokerError(f"eToro {status.get('name', 'rejected')} the order: {status.get('errorMessage') or 'no reason given'}")
        return None

    def open(self, symbol: str, amount: float, stop: float, target: float, quote: Quote) -> OpenFill:
        request_id = str(uuid.uuid4())
        try:
            order_id = self.client.open_order(self.ids[symbol], amount, stop, target, request_id)
        except EtoroUncertain as exc:
            # The order may exist. eToro lets us look it up by the request id we sent.
            try:
                order_id = int(self.client.order_info(reference_id=request_id)["orderId"])
            except (EtoroError, KeyError, TypeError, ValueError):
                raise BrokerError(f"Order for {symbol} could not be confirmed ({exc}). Check the eToro app.") from exc
        fill = self._await_open(order_id)
        if fill:
            return fill
        try:  # not filled in time: cancel it so it cannot fill later behind the bot's back
            self.client.cancel_order(order_id)
        except EtoroError:
            pass
        fill = self._await_open_once(order_id)
        if fill:
            return fill
        self._orphans.append((order_id, symbol))
        raise BrokerError(f"Order {order_id} for {symbol} did not fill in time; cancelled or being cleaned up")

    def _await_open_once(self, order_id: int) -> Optional[OpenFill]:
        try:
            info = self.client.order_info(order_id=order_id)
        except EtoroError:
            return None
        return self._fill_from_order(info) if (info.get("status") or {}).get("id") in FILLED else None

    # ---------- close ----------
    def _closed_trade(self, position_id: int, now: Optional[datetime] = None) -> Optional[dict]:
        if not self._history_ok:
            return None
        try:
            since = (now or datetime.now(timezone.utc)) - timedelta(days=2)
            for trade in self.client.history(since):
                if int(trade.get("positionId") or 0) == position_id:
                    return trade
        except EtoroError as exc:
            log.warning("eToro trade history unavailable (%s); closing prices will be estimated", exc)
            self._history_ok = False
        return None

    def close(self, pos: DayPosition, quote: Quote, reason: str) -> CloseFill:
        try:
            order_id = self.client.close_position(pos.position_id, self.ids[pos.symbol])
        except EtoroUncertain as exc:
            raise BrokerError(f"Close request for {pos.symbol} may not have gone through ({exc}); will retry") from exc
        except EtoroError as exc:
            trade = self._closed_trade(pos.position_id)  # perhaps eToro's stop or target already closed it
            if trade:
                return CloseFill(float(trade["closeRate"]), trade.get("netProfit"))
            raise BrokerError(str(exc)) from exc
        for _ in range(self.POLLS):
            self.sleep(self.POLL_SECONDS)
            info = self.client.close_info(order_id)
            status = info.get("statusID")
            if status in FILLED:
                for closed in info.get("positions") or []:
                    if closed.get("rate"):
                        return CloseFill(float(closed["rate"]))
            elif status in FAILED:
                raise BrokerError(f"eToro did not close {pos.symbol}: {info.get('errorMessage') or 'status ' + str(status)}")
        raise BrokerError(f"Close order {order_id} for {pos.symbol} not confirmed yet; will check again")

    # ---------- positions eToro closed by itself ----------
    def sync(self, positions: list[DayPosition], quotes: dict[str, Quote], now: datetime) -> list[Closure]:
        if self.clock() - self._last_sync < self.cfg.sync_seconds or not (positions or self._orphans):
            return []
        self._last_sync = self.clock()
        try:
            portfolio = self.client.portfolio()
        except EtoroError as exc:
            log.warning("Could not read the eToro portfolio: %s", exc)
            return []
        open_ids = {int(p["positionID"]) for p in portfolio.get("positions") or [] if p.get("positionID")}
        self._clean_orphans()
        closures = []
        for pos in positions:
            if pos.position_id in open_ids:
                self._missing.pop(pos.position_id, None)
                continue
            seen = self._missing.get(pos.position_id, 0) + 1
            self._missing[pos.position_id] = seen
            state = self._position_state(pos)
            if state == "open":
                continue  # the portfolio view is lagging: eToro still reports the position as open
            trade = self._closed_trade(pos.position_id, now)
            if trade and trade.get("closeRate"):
                rate = float(trade["closeRate"])
                reason = "take-profit" if rate >= pos.target * 0.9995 else "stop-loss"
                closures.append(Closure(pos.position_id, rate, reason, trade.get("netProfit")))
            elif state == "closed" or seen >= 3:
                # Closed, but the trade history has not caught up: book it at the level that must have triggered.
                bid = quotes[pos.symbol].bid if pos.symbol in quotes else pos.stop
                at_target = bid >= (pos.entry + pos.target) / 2
                closures.append(Closure(pos.position_id, pos.target if at_target else pos.stop,
                                        ("take-profit" if at_target else "stop-loss") + " (estimated)"))
        return closures

    def _position_state(self, pos: DayPosition) -> Optional[str]:
        """'open' or 'closed' according to the order that created the position; None if unknown."""
        if not pos.order_id:
            return None
        try:
            info = self.client.order_info(order_id=pos.order_id)
        except EtoroError:
            return None
        for execution in info.get("positionExecutions") or []:
            if int(execution.get("positionId") or 0) == pos.position_id:
                return str(execution.get("state") or "").lower() or None
        return None

    def _clean_orphans(self) -> None:
        """An order we gave up on may still have filled. If so, close that position straight away."""
        for order_id, symbol in list(self._orphans):
            try:
                info = self.client.order_info(order_id=order_id)
                status = (info.get("status") or {}).get("id")
                if status in FILLED:
                    fill = self._fill_from_order(info)
                    if fill:
                        log.warning("Late fill on order %s (%s): closing that position", order_id, symbol)
                        self.client.close_position(fill.position_id, self.ids[symbol])
                    self._orphans.remove((order_id, symbol))
                elif status in FAILED:
                    self._orphans.remove((order_id, symbol))
            except EtoroError as exc:
                log.warning("Could not resolve order %s yet: %s", order_id, exc)
