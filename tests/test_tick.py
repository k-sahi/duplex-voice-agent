import asyncio
import time

from engine.tick import TickLoop


def test_ticks_at_the_right_rate():
    count = []
    loop = TickLoop(count.append, period_ms=20)
    asyncio.run(loop.run(duration_s=0.5))
    assert 22 <= loop.stats.ticks <= 27          # ~25 ticks in 500 ms
    assert count == sorted(count)                # tick numbers only go up


def test_slow_tick_is_detected_and_skipped():
    loop = TickLoop(lambda n: time.sleep(0.05), period_ms=20)  # 50 ms work, 20 ms budget
    asyncio.run(loop.run(duration_s=0.3))
    assert loop.stats.overruns > 0
    assert loop.stats.skipped > 0
