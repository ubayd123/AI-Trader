"""Optional sync with the AI-Trader platform (this repo's service, or https://ai4trade.ai).

Publishes each executed trade as a realtime signal and reads the market-intel
overview as context for the AI advisor. Failures never block trading.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

log = logging.getLogger(__name__)


class AITraderClient:
    def __init__(self, base_url: str, token: str = "", session: Optional[requests.Session] = None):
        self.base = base_url.rstrip("/")
        self.token = token
        self.session = session or requests.Session()

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def register(self, name: str, password: str, initial_balance_usd: float) -> dict:
        resp = self.session.post(f"{self.base}/claw/agents/selfRegister", timeout=20,
                                 json={"name": name, "password": password, "initial_balance": initial_balance_usd})
        resp.raise_for_status()
        return resp.json()

    def publish_trade(self, market: str, action: str, symbol: str, usd_price: float, quantity: float,
                      note: str, when: datetime) -> bool:
        if not self.token:
            return False
        try:
            resp = self.session.post(f"{self.base}/signals/realtime", headers=self._headers, timeout=20, json={
                "market": market, "action": action, "symbol": symbol, "price": round(usd_price, 6),
                "quantity": round(quantity, 8), "content": note[:500],
                "executed_at": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            if resp.status_code >= 400:
                log.warning("AI-Trader rejected signal for %s: %s %s", symbol, resp.status_code, resp.text[:200])
                return False
            return True
        except requests.RequestException as exc:
            log.warning("Could not publish signal for %s: %s", symbol, exc)
            return False

    def market_context(self, max_age: timedelta = timedelta(days=3)) -> Optional[dict]:
        """Macro/news overview, only if it is fresh enough to be worth showing the advisor."""
        try:
            data = self.session.get(f"{self.base}/market-intel/overview", timeout=15).json()
            updated = datetime.fromisoformat(str(data["last_updated_at"]).replace("Z", "+00:00"))
            if not data.get("available") or datetime.now(timezone.utc) - updated > max_age:
                return None
            return {"as_of": data["last_updated_at"], "macro_verdict": data.get("macro_verdict"),
                    "macro_summary": data.get("macro_summary"), "etf_flows": data.get("etf_summary"),
                    "news_activity": data.get("news_status")}
        except Exception:
            return None
