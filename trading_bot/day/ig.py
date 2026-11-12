"""IG spread betting through IG's trading API (https://labs.ig.com), via the community `trading-ig` library.

What IG's API can trade: spread betting and CFD accounts only (not share dealing or ISAs).
This bot supports spread betting accounts: sizes are pounds per point, profits are in pounds
and, for UK residents, free of capital gains tax. Every trade is leveraged.

Order flow, as documented by IG:
  open     POST /positions/otc, then GET /confirms/{dealReference} for the outcome
  close    DELETE /positions/otc by deal id, then the confirmation
  prices   GET /markets?epics=... (one request for the whole watchlist)
IG's published limits: 30 non-trading and 100 trading requests a minute per account,
and 10,000 historical price points a week. Demo limits can be lower.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from .bars import Bar, Quote
from .config import IG_MARKETS, DayConfig
from .engine import BrokerError, Closure, CloseFill, DayPosition, OpenFill, Spec
from .etoro import RateLimiter

log = logging.getLogger(__name__)

FRIENDLY = {
    "invalid-details": "IG rejected the username or password. Demo and live accounts have different logins.",
    "api-key-invalid": "IG rejected the API key. Demo and live accounts need their own keys.",
    "api-key-disabled": "This IG API key is disabled. Enable it in My IG > Settings > API keys.",
    "api-key-revoked": "This IG API key has been revoked. Create a new one in My IG > Settings > API keys.",
    "stockbroking-not-supported": "Your default IG account is a share dealing or ISA account. IG's API only trades "
                                  "spread betting and CFD accounts: open a spread betting account and set IG_ACCOUNT_ID.",
    "exceeded-api-key-allowance": "IG's request limit for this API key was reached; slowing down.",
    "exceeded-account-allowance": "IG's request limit for this account was reached; slowing down.",
    "exceeded-account-trading-allowance": "IG's trading-request limit was reached; slowing down.",
    "exceeded-account-historical-data-allowance": "IG's weekly allowance of historical prices is used up.",
    "too-many-failed-attempts": "Too many failed log-ins. IG has locked API access for a while.",
    "'CST'": "IG did not start a session: wrong username or password, or demo details used on live (or the reverse).",
}
REJECTIONS = {
    "INSUFFICIENT_FUNDS": "not enough money in the IG account for the margin",
    "MARKET_CLOSED": "the market is closed",
    "MARKET_CLOSED_WITH_EDITS": "the market is closed",
    "MARKET_OFFLINE": "the market is offline",
    "MINIMUM_ORDER_SIZE_ERROR": "the size is below IG's minimum for this market",
    "ATTACHED_ORDER_LEVEL_ERROR": "the stop or limit is too close to the price",
    "STOP_ORDER_LEVEL_ERROR": "the stop is too close to the price",
    "REJECT_CFD_ORDER_ON_SPREADBET_ACCOUNT": "that is a CFD market; a spread betting account needs the spread-bet version",
    "REJECT_SPREADBET_ORDER_ON_CFD_ACCOUNT": "that is a spread-bet market, but this is a CFD account",
    "MARKET_ROLLED": "the market has rolled to a new contract",
    "UNKNOWN": "IG gave no reason",
}


class IGError(BrokerError):
    pass


def _friendly(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    for code, message in FRIENDLY.items():
        if code in text:
            return message
    return f"IG API error: {text[:200]}"


def resolve_epics(symbols: list[str]) -> dict[str, str]:
    """Market name -> IG epic. A name from IG_MARKETS, or a raw epic such as IX.D.FTSE.DAILY.IP."""
    out = {}
    for symbol in symbols:
        if symbol in IG_MARKETS:
            out[symbol] = IG_MARKETS[symbol]["epic"]
        elif "." in symbol:
            out[symbol] = symbol
        else:
            raise IGError(f"'{symbol}' is not a known IG market. Use one of {', '.join(IG_MARKETS)} or an IG epic.")
    return out


class IGClient:
    """A thin, defensive wrapper over trading_ig.IGService: one place for errors, limits and re-login."""

    def __init__(self, service, session: dict, sleep=time.sleep, clock=time.monotonic):
        self.service = service
        self.session = session or {}
        self.sleep = sleep
        self.limiter = RateLimiter(24, clock, sleep)  # IG allows 30 non-trading requests a minute per account
        self.account_id = self.session.get("currentAccountId", "")
        self.account_type = self.session.get("accountType", "")
        self.currency = self.session.get("currencyIsoCode", "GBP")

    @classmethod
    def login(cls, username: str, password: str, api_key: str, env: str = "demo", account_id: str = "",
              service=None, sleep=time.sleep, clock=time.monotonic) -> "IGClient":
        if not (username and password and api_key):
            raise IGError("IG_USERNAME, IG_PASSWORD and IG_API_KEY are not set (see trading_bot/day/README.md)")
        if service is None:
            try:
                from trading_ig import IGService
            except ImportError as exc:
                raise IGError("The IG library is missing. Run: pip install trading-ig") from exc
            service = IGService(username, password, api_key, acc_type="DEMO" if env == "demo" else "LIVE",
                                return_dataframe=False, return_munch=False)
        try:
            session = service.create_session()
        except Exception as exc:
            raise IGError(_friendly(exc)) from exc
        client = cls(service, session, sleep, clock)
        accounts = session.get("accounts") or []
        wanted = account_id or next((a["accountId"] for a in accounts if a.get("accountType") == "SPREADBET"), "")
        if wanted and wanted != client.account_id:
            client._call(service.switch_account, wanted, False)
            client.account_id = wanted
            client.account_type = next((a.get("accountType", "") for a in accounts if a.get("accountId") == wanted), "")
        if client.account_type != "SPREADBET":
            raise IGError(f"This bot trades IG spread betting accounts; the selected account is "
                          f"'{client.account_type or 'unknown'}'. Open a spread betting account (or set IG_ACCOUNT_ID to it).")
        return client

    def _call(self, fn, *args, limited: bool = True, **kwargs):
        for attempt in range(2):
            if limited:
                self.limiter.wait()
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # the library raises plain Exception with IG's error code as text
                text = f"{exc.__class__.__name__} {exc}"
                if attempt == 0 and ("TokenInvalid" in text or "client-token" in text or "oauth-token" in text):
                    try:  # the session expired: log in again and repeat the call once
                        self.service.create_session()
                        continue
                    except Exception as login_exc:
                        raise IGError(_friendly(login_exc)) from login_exc
                if attempt == 0 and ("ApiExceeded" in text or "exceeded-" in text) and "historical" not in text:
                    self.sleep(10)
                    continue
                raise IGError(_friendly(exc)) from exc

    # ---------- account & markets ----------
    def balance(self) -> dict:
        data = self._call(self.service.fetch_accounts) or {}
        for account in data.get("accounts", []) if isinstance(data, dict) else []:
            if account.get("accountId") == self.account_id:
                return account.get("balance") or {}
        return self.session.get("accountInfo") or {}

    def market(self, epic: str) -> dict:
        return self._call(self.service.fetch_market_by_epic, epic) or {}

    def snapshots(self, epics: list[str]) -> list[dict]:
        return self._call(self.service.fetch_markets_by_epics, ",".join(epics), detailed=False) or []

    def prices(self, epic: str, points: int) -> list[dict]:
        data = self._call(self.service.fetch_historical_prices_by_epic, epic, numpoints=points, pagesize=points) or {}
        return data.get("prices", []) if isinstance(data, dict) else []

    def positions(self) -> list[dict]:
        data = self._call(self.service.fetch_open_positions) or {}
        return data.get("positions", []) if isinstance(data, dict) else []

    def activity(self, since: datetime) -> list[dict]:
        data = self._call(self.service.fetch_account_activity, from_date=since, detailed=True) or {}
        return data.get("activities", []) if isinstance(data, dict) else []

    # ---------- dealing ----------
    def open(self, epic: str, expiry: str, currency: str, size: float, stop: float, limit: float) -> dict:
        return self._call(
            self.service.create_open_position, limited=False,
            currency_code=currency, direction="BUY", epic=epic, expiry=expiry, force_open=True,
            guaranteed_stop=False, level=None, limit_distance=None, limit_level=limit, order_type="MARKET",
            quote_id=None, size=size, stop_distance=None, stop_level=stop, trailing_stop=False,
            trailing_stop_increment=None) or {}

    def close(self, deal_id: str, size: float) -> dict:
        return self._call(
            self.service.close_open_position, limited=False,
            deal_id=deal_id, direction="SELL", epic=None, expiry=None, level=None, order_type="MARKET",
            quote_id=None, size=size) or {}


def _mid(price: dict) -> Optional[float]:
    bid, ask = (price or {}).get("bid"), (price or {}).get("ask")
    return (float(bid) + float(ask)) / 2 if bid is not None and ask is not None else None


class IGFeed:
    """Live bid/offer for the watchlist: one request per poll."""

    def __init__(self, client: IGClient, epics: dict[str, str]):
        self.client, self.epics = client, epics
        self.names = {epic: name for name, epic in epics.items()}

    def quotes(self, now: datetime) -> dict[str, Quote]:
        out = {}
        for item in self.client.snapshots(list(self.epics.values())):
            name = self.names.get((item.get("instrument") or {}).get("epic"))
            snap = item.get("snapshot") or {}
            bid, offer = snap.get("bid"), snap.get("offer")
            if not name or bid is None or offer is None:
                continue
            if snap.get("marketStatus") != "TRADEABLE" or (snap.get("delayTime") or 0) > 0:
                continue  # closed, suspended or delayed: never trade on it
            out[name] = Quote(name, float(bid), float(offer), now)
        return out

    def history(self, name: str, minutes: int = 80) -> list[Bar]:
        """Recent one-minute bars (mid prices). Uses IG's weekly allowance, so only a short warm-up is requested."""
        bars = []
        for row in self.client.prices(self.epics[name], minutes):
            stamp = str(row.get("snapshotTimeUTC") or "")
            values = [_mid(row.get(key)) for key in ("openPrice", "highPrice", "lowPrice", "closePrice")]
            if stamp and all(v is not None for v in values):
                when = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
                bars.append(Bar(when, *values))
        return sorted(bars, key=lambda b: b.time)


class IGBroker:
    """Opens and closes IG spread bets (long only). Every position carries a stop and a limit held by IG."""

    name = "ig"

    def __init__(self, cfg: DayConfig, client: IGClient, epics: dict[str, str], clock=time.monotonic):
        self.cfg, self.client, self.epics, self.clock = cfg, client, epics, clock
        self._details: dict[str, dict] = {}
        self._last_sync = -1e9
        self._missing: dict[str, int] = {}

    # ---------- market rules ----------
    def details(self, name: str, refresh: bool = False) -> dict:
        if refresh or name not in self._details:
            self._details[name] = self.client.market(self.epics[name])
        return self._details[name]

    def spec(self, name: str) -> Spec:
        d = self.details(name)
        instrument, rules, snap = d.get("instrument") or {}, d.get("dealingRules") or {}, d.get("snapshot") or {}
        price = float(snap.get("offer") or snap.get("bid") or 0)
        min_size = float((rules.get("minDealSize") or {}).get("value") or 0)
        margin = float(instrument.get("marginFactor") or 0)
        margin_pct = margin / 100 if instrument.get("marginFactorUnit", "PERCENTAGE") == "PERCENTAGE" else 0.2
        stop_rule = rules.get("minNormalStopOrLimitDistance") or {}
        stop_distance = float(stop_rule.get("value") or 0)
        if stop_rule.get("unit") == "PERCENTAGE":
            stop_distance = price * stop_distance / 100
        return Spec(min_size=min_size, size_decimals=2, margin_pct=margin_pct or 0.2, min_stop_distance=stop_distance,
                    tradable=snap.get("marketStatus") == "TRADEABLE")

    def _level(self, name: str, level: float) -> float:
        """Round a price level to the number of decimals IG quotes this market in."""
        places = (self.details(name).get("snapshot") or {}).get("decimalPlacesFactor")
        return round(level, int(places) if places is not None else 1)

    @staticmethod
    def _check(confirm: dict, what: str) -> None:
        if confirm.get("dealStatus") != "ACCEPTED":
            reason = str(confirm.get("reason") or "UNKNOWN")
            raise BrokerError(f"IG rejected the {what}: {REJECTIONS.get(reason, reason.replace('_', ' ').lower())}")

    # ---------- dealing ----------
    def open(self, name: str, amount: float, stop: float, target: float, quote: Quote) -> OpenFill:
        instrument = self.details(name).get("instrument") or {}
        currencies = instrument.get("currencies") or [{}]
        currency = next((c.get("code") for c in currencies if c.get("isDefault")), currencies[0].get("code")) or "GBP"
        size = round(amount / quote.ask, 2)
        confirm = self.client.open(self.epics[name], instrument.get("expiry") or "DFB", currency, size,
                                   self._level(name, stop), self._level(name, target))
        self._check(confirm, "trade")
        opened = [d.get("dealId") for d in confirm.get("affectedDeals") or [] if d.get("status") == "OPENED"]
        deal_id = (opened[0] if opened else None) or confirm.get("dealId")
        if not deal_id or confirm.get("level") is None:
            raise BrokerError("IG accepted the trade but did not report its deal id or level. Check the IG platform.")
        return OpenFill(deal_id, float(confirm.get("size") or size), float(confirm["level"]),
                        confirm.get("dealReference") or 0, confirm.get("stopLevel"), confirm.get("limitLevel"))

    def close(self, pos: DayPosition, quote: Quote, reason: str) -> CloseFill:
        try:
            confirm = self.client.close(pos.position_id, pos.units)
            self._check(confirm, "closing trade")
        except BrokerError as exc:
            level = self._closed_level(pos)  # perhaps IG's stop or limit closed it first
            if level is not None:
                return CloseFill(level)
            raise BrokerError(str(exc)) from exc
        if confirm.get("level") is None:
            raise BrokerError("IG closed the position but did not report the level; will reconcile shortly")
        profit = confirm.get("profit") if confirm.get("profitCurrency") in (None, "GBP") else None
        return CloseFill(float(confirm["level"]), float(profit) if profit is not None else None)

    # ---------- positions IG closed by itself (stop or limit) ----------
    def _closed_level(self, pos: DayPosition, now: Optional[datetime] = None) -> Optional[float]:
        try:
            since = (now or datetime.now(timezone.utc)) - timedelta(days=2)
            for item in self.client.activity(since.replace(tzinfo=None)):
                details = item.get("details") or {}
                closed = any(a.get("affectedDealId") == pos.position_id and "CLOSE" in str(a.get("actionType", ""))
                             for a in details.get("actions") or [])
                level = details.get("level", item.get("level"))
                if closed and level is not None:
                    return float(level)
        except BrokerError as exc:
            log.warning("IG activity history unavailable (%s); closing levels will be estimated", exc)
        return None

    def sync(self, positions: list[DayPosition], quotes: dict[str, Quote], now: datetime) -> list[Closure]:
        if not positions or self.clock() - self._last_sync < self.cfg.sync_seconds:
            return []
        self._last_sync = self.clock()
        try:
            open_ids = {(p.get("position") or {}).get("dealId") for p in self.client.positions()}
        except BrokerError as exc:
            log.warning("Could not read IG positions: %s", exc)
            return []
        closures = []
        for pos in positions:
            if pos.position_id in open_ids:
                self._missing.pop(pos.position_id, None)
                continue
            seen = self._missing.get(pos.position_id, 0) + 1
            self._missing[pos.position_id] = seen
            if seen < 2:
                continue  # give IG's position list one more look before believing it
            level = self._closed_level(pos, now)
            if level is not None:
                reason = "take-profit" if level >= pos.target - 1e-9 else "stop-loss" if level <= pos.stop + 1e-9 else "closed at IG"
                closures.append(Closure(pos.position_id, level, reason))
            elif seen >= 4:
                bid = quotes[pos.symbol].bid if pos.symbol in quotes else pos.stop
                at_target = bid >= (pos.entry + pos.target) / 2
                closures.append(Closure(pos.position_id, pos.target if at_target else pos.stop,
                                        ("take-profit" if at_target else "stop-loss") + " (estimated)"))
        return closures
