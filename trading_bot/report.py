"""Performance summary from the bot's own trade log and equity history."""
from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarise(trades: list[dict], equity: list[dict], starting_cash: float) -> dict:
    closed = [t for t in trades if t.get("side") == "sell" and t.get("pnl_gbp") is not None]
    wins = [t["pnl_gbp"] for t in closed if t["pnl_gbp"] > 0]
    losses = [t["pnl_gbp"] for t in closed if t["pnl_gbp"] <= 0]
    out = {
        "trades": len(trades), "closed_trades": len(closed),
        "win_rate_pct": round(100 * len(wins) / len(closed)) if closed else None,
        "average_win_gbp": round(sum(wins) / len(wins), 2) if wins else None,
        "average_loss_gbp": round(sum(losses) / len(losses), 2) if losses else None,
        "realised_pnl_gbp": round(sum(t["pnl_gbp"] for t in closed), 2),
        "fees_gbp": round(sum(t.get("fee_gbp") or 0 for t in trades), 2),
        "exit_reasons": {},
    }
    for t in closed:
        key = t.get("reason", "").split(" (")[0].split(":")[0]
        out["exit_reasons"][key] = out["exit_reasons"].get(key, 0) + 1
    if equity:
        peak, worst = starting_cash, 0.0
        for point in equity:
            peak = max(peak, point["equity_gbp"])
            worst = min(worst, point["equity_gbp"] / peak - 1)
        out.update(first_cycle=equity[0]["time"][:10], last_cycle=equity[-1]["time"][:10], cycles=len(equity),
                   equity_gbp=round(equity[-1]["equity_gbp"], 2),
                   return_pct=round((equity[-1]["equity_gbp"] / starting_cash - 1) * 100, 2),
                   max_drawdown_pct=round(worst * 100, 2))
    return out


def build_report(cfg) -> dict:
    return summarise(_read_jsonl(cfg.trades_path), _read_jsonl(cfg.equity_path), cfg.starting_cash_gbp)
