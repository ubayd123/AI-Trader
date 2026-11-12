"""Order execution: a paper broker (default) and Trading 212 (demo or live)."""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Optional

import requests

from .config import Config
from .models import Fill, Order, Snapshot

log = logging.getLogger(__name__)


class BrokerError(Exception):
    pass


class PositionMissing(BrokerError):
    """The broker no longer holds a position the bot's ledger thinks it owns."""


class PaperBroker:
    """Simulated fills at the latest price, with realistic UK retail costs."""

    name = "paper"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def supports(self, market: str) -> bool:
        return True

    def execute(self, order: Order, snap: Snapshot) -> Fill:
        fee_rate = self.cfg.fee_pct(order.market)
        if order.side == "buy":
            price = snap.price * (1 + self.cfg.slippage_pct)
            qty = order.gbp_amount / (price * (1 + fee_rate))
        else:
            price = snap.price * (1 - self.cfg.slippage_pct)
            qty = order.qty
        if qty <= 0:
            raise BrokerError("Order quantity is zero")
        return Fill(qty=qty, price_gbp=price, fee_gbp=qty * price * fee_rate)


class Trading212Broker:
    """Trading 212 Invest / Stocks ISA via the public API (https://docs.trading212.com/api).

    Shares and ETFs only (no crypto). The account must be in GBP. Orders are
    market orders by quantity, as the API does not support value orders.
    """

    name = "trading212"
    FILL_POLLS = 6
    POLL_SECONDS = 1.5

    def __init__(self, cfg: Config, api_key: str, api_secret: str = "", session: Optional[requests.Session] = None,
                 sleep=time.sleep):
        if not api_key:
            raise BrokerError("T212_API_KEY is not set")
        self.cfg = cfg
        self.base = f"https://{cfg.t212_env}.trading212.com/api/v0"
        self.session = session or requests.Session()
        self.auth = (api_key, api_secret) if api_secret else None
        self.headers = {} if api_secret else {"Authorization": api_key}  # legacy single-key scheme
        self.sleep = sleep

    def supports(self, market: str) -> bool:
        return market == "us-stock"

    def ticker(self, symbol: str) -> str:
        return os.environ.get(f"T212_TICKER_{symbol.upper()}") or f"{symbol.upper()}_US_EQ"

    def _request(self, method: str, path: str, **kwargs):
        for attempt in range(3):
            resp = self.session.request(method, self.base + path, auth=self.auth, headers=self.headers,
                                        timeout=20, **kwargs)
            if resp.status_code == 429 and attempt < 2:
                reset = float(resp.headers.get("x-ratelimit-reset", 0) or 0)
                self.sleep(min(max(reset - time.time(), 1.0), 60.0))
                continue
            if resp.status_code >= 400:
                raise BrokerError(f"Trading 212 {method} {path} failed: HTTP {resp.status_code} {resp.text[:200]}")
            return resp.json() if resp.content else None
        raise BrokerError(f"Trading 212 {method} {path} rate limited")

    def account(self) -> dict:
        summary = self._request("GET", "/equity/account/summary")
        if summary.get("currency") != "GBP":
            raise BrokerError(f"Account currency is {summary.get('currency')}; this bot needs a GBP account")
        return summary

    def _position(self, ticker: str) -> tuple[float, float, float]:
        """(quantity, sellable quantity, total cost in GBP) for one ticker; zeros if not held."""
        for pos in self._request("GET", "/equity/positions", params={"ticker": ticker}) or []:
            if pos.get("instrument", {}).get("ticker") == ticker:
                return (float(pos.get("quantity") or 0), float(pos.get("quantityAvailableForTrading") or 0),
                        float((pos.get("walletImpact") or {}).get("totalCost") or 0))
        return 0.0, 0.0, 0.0

    def _floor(self, qty: float) -> float:
        scale = 10 ** self.cfg.t212_quantity_decimals
        return math.floor(qty * scale + 1e-9) / scale

    def _place_and_wait(self, ticker: str, signed_qty: float, before_qty: float) -> tuple[float, float]:
        """Place a market order; return (new quantity, new total cost). Cancels if it does not fill."""
        placed = self._request("POST", "/equity/orders/market",
                               json={"ticker": ticker, "quantity": signed_qty, "extendedHours": False})
        for _ in range(self.FILL_POLLS):
            self.sleep(self.POLL_SECONDS)
            qty, _, cost = self._position(ticker)
            if abs((qty - before_qty) - signed_qty) < 10 ** -(self.cfg.t212_quantity_decimals + 1):
                return qty, cost
        order_id = (placed or {}).get("id")
        try:
            self._request("DELETE", f"/equity/orders/{order_id}")
            raise BrokerError(f"Order {order_id} for {ticker} did not fill in time and was cancelled")
        except BrokerError as exc:
            if "was cancelled" in str(exc):
                raise
            raise BrokerError(f"Order {order_id} for {ticker} did not fill in time and could not be cancelled "
                              f"({exc}). CHECK THE TRADING 212 APP before running the bot again.") from exc

    def execute(self, order: Order, snap: Snapshot) -> Fill:
        ticker = self.ticker(order.symbol)
        before_qty, sellable, before_cost = self._position(ticker)
        if order.side == "buy":
            available = float(self.account().get("cash", {}).get("availableToTrade") or 0)
            if available < order.gbp_amount:
                raise BrokerError(f"Only £{available:.2f} available in Trading 212; need £{order.gbp_amount:.2f}")
            # 0.5% headroom so price moves and the FX fee cannot push us over budget.
            qty = self._floor(order.gbp_amount / (snap.price * (1.005 + self.cfg.stock_fee_pct)))
            if qty <= 0:
                raise BrokerError(f"£{order.gbp_amount:.2f} is too small to buy {order.symbol} at this precision")
            new_qty, new_cost = self._place_and_wait(ticker, qty, before_qty)
            cost = new_cost - before_cost
            if cost <= 0:  # cost basis unavailable: fall back to an estimate
                return Fill(qty, snap.price, qty * snap.price * self.cfg.stock_fee_pct, estimated=True)
            return Fill(qty=qty, price_gbp=cost / qty, fee_gbp=0.0)  # FX fee is inside the cost basis
        if sellable <= 0:
            raise PositionMissing(f"Trading 212 holds no sellable {order.symbol}")
        qty = self._floor(min(order.qty, sellable))
        if qty <= 0:
            raise PositionMissing(f"Trading 212 holding of {order.symbol} is too small to sell")
        self._place_and_wait(ticker, -qty, before_qty)
        # The API does not return the fill value synchronously; proceeds are estimated from the latest price.
        return Fill(qty=qty, price_gbp=snap.price, fee_gbp=qty * snap.price * self.cfg.stock_fee_pct, estimated=True)


def make_broker(cfg: Config, confirm_live: bool = False):
    if cfg.broker == "paper":
        return PaperBroker(cfg)
    if cfg.t212_env == "live" and not confirm_live:
        raise BrokerError("Refusing to trade real money without --confirm-live. "
                          "Run in paper or Trading 212 demo mode first.")
    return Trading212Broker(cfg, os.environ.get("T212_API_KEY", ""), os.environ.get("T212_API_SECRET", ""))
