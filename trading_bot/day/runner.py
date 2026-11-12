"""The live loop: wait for the US session, poll quotes, feed the engine."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from .bars import NEW_YORK, in_session, next_session_open
from .config import DayConfig
from .engine import DayEngine
from .feeds import todays_bars

log = logging.getLogger(__name__)
LONDON = ZoneInfo("Europe/London")


def run(cfg: DayConfig, engine: DayEngine, feed, screen: Optional[Callable[[], list]] = None, sleep=time.sleep,
        clock=lambda: datetime.now(timezone.utc), max_loops: Optional[int] = None) -> None:
    prepared = None  # the session we have warmed up for
    told_closed = False
    last_status = None
    loops = 0
    try:
        while max_loops is None or loops < max_loops:
            loops += 1
            now = clock()
            if not in_session(now):
                opens = next_session_open(now)
                if not told_closed:
                    log.info("US market is closed. Next session opens %s UK time. Waiting...",
                             opens.astimezone(LONDON).strftime("%a %d %b at %H:%M"))
                    told_closed = True
                sleep(max(1.0, min(60.0, (opens - now).total_seconds())))
                continue
            told_closed = False
            session = now.astimezone(NEW_YORK).date().isoformat()
            if prepared != session:
                engine.ensure_session(now)
                for symbol in cfg.symbols:
                    try:
                        engine.seed(symbol, todays_bars(feed.history(symbol), now))
                    except Exception as exc:
                        log.warning("No warm-up history for %s (%s); it will start once enough bars build up", symbol, exc)
                if screen and not engine.state.excluded_today:
                    engine.state.excluded_today = screen()
                prepared = session
                log.info("Session %s: equity %s%.2f, trading %s", session, cfg.currency, engine.equity(),
                         ", ".join(s for s in cfg.symbols if s not in engine.state.excluded_today))
            try:
                quotes = feed.quotes(now)
            except Exception as exc:
                log.warning("No quotes this round: %s", exc)
                sleep(cfg.poll_seconds)
                continue
            engine.on_quotes(now, quotes)
            stamp = now.replace(minute=now.minute - now.minute % 5, second=0, microsecond=0)
            if stamp != last_status:
                last_status = stamp
                st = engine.state
                log.info("equity %s%.2f | today %+.2f%% | %d trades | %d open%s", cfg.currency, engine.equity(),
                         100 * engine.day_pnl() / st.day_start_equity if st.day_start_equity else 0.0,
                         st.trades_today, len(st.positions),
                         " | STOPPED FOR TODAY" if st.halted_today else " | KILL SWITCH" if st.halted else "")
            sleep(cfg.poll_seconds)
    except KeyboardInterrupt:
        log.info("Stopping. Selling the bot's open positions...")
        engine.flatten(clock(), "bot stopped")
        log.info("Stopped. Equity %s%.2f", cfg.currency, engine.equity())
