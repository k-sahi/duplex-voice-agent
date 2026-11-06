"""Fixed-rate tick loop with drift-free deadlines and timing stats."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


def _pct(xs, p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


@dataclass
class TickStats:
    period_ms: float
    ticks: int = 0
    overruns: int = 0  # tick work took longer than one period
    skipped: int = 0   # ticks dropped because the loop fell behind
    lateness_ms: deque = field(default_factory=lambda: deque(maxlen=50_000))
    work_ms: deque = field(default_factory=lambda: deque(maxlen=50_000))

    def summary(self) -> dict:
        return {
            "ticks": self.ticks,
            "overruns": self.overruns,
            "skipped": self.skipped,
            "lateness_p50_ms": round(_pct(self.lateness_ms, 50), 3),
            "lateness_p99_ms": round(_pct(self.lateness_ms, 99), 3),
            "lateness_max_ms": round(max(self.lateness_ms, default=0.0), 3),
            "work_p50_ms": round(_pct(self.work_ms, 50), 4),
            "work_p99_ms": round(_pct(self.work_ms, 99), 4),
        }


class TickLoop:
    """Calls on_tick(n) every period_ms.

    Deadlines are computed from the start time (start + n * period), so small
    delays never accumulate into drift. If the loop falls more than one tick
    behind, it skips ahead instead of bursting to catch up.

    on_tick must be fast and synchronous: no model inference here. Slow work
    runs in separate asyncio tasks and hands results back through queues.
    """

    def __init__(self, on_tick: Callable[[int], None], period_ms: float = 20.0) -> None:
        self.on_tick = on_tick
        self.period = period_ms / 1000
        self.stats = TickStats(period_ms)
        self._running = False

    async def run(self, duration_s: float | None = None) -> None:
        loop = asyncio.get_running_loop()
        start = loop.time()
        n = 0
        self._running = True
        while self._running:
            deadline = start + n * self.period
            delay = deadline - loop.time()
            await asyncio.sleep(delay if delay > 0 else 0)  # always yield to other tasks

            self.stats.lateness_ms.append((loop.time() - deadline) * 1000)
            t0 = time.perf_counter()
            self.on_tick(n)
            work_ms = (time.perf_counter() - t0) * 1000
            self.stats.work_ms.append(work_ms)
            self.stats.ticks += 1
            if work_ms > self.period * 1000:
                self.stats.overruns += 1

            n += 1
            behind = loop.time() - (start + n * self.period)
            if behind > self.period:
                skip = int(behind // self.period)
                self.stats.skipped += skip
                n += skip

            if duration_s is not None and loop.time() - start >= duration_s:
                break
        self._running = False

    def stop(self) -> None:
        self._running = False
