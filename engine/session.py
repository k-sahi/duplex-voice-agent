"""One conversation: owns the state and is advanced only by tick()."""
from __future__ import annotations

from collections import deque
from typing import Callable

from engine.audio import FRAME_MS, OutputBuffer
from engine.state_machine import (
    Action, AgentAudioFinished, AgentAudioStarted, Event, FlushPlayback,
    State, transition,
)


class Session:
    """Events come in from anywhere via post(); they are only applied on a tick.

    Each tick:
      1. drain the inbox and run the state machine
      2. execute FlushPlayback immediately (barge-in stops within this tick)
      3. send exactly one audio frame (agent speech or silence)
      4. derive AgentAudioStarted/Finished from what was actually sent
    """

    def __init__(self, send_audio: Callable[[bytes], None],
                 on_action: Callable[[Action], None] | None = None) -> None:
        self.state = State.IDLE
        self.output = OutputBuffer()
        self._inbox: deque[Event] = deque()
        self._send_audio = send_audio
        self._on_action = on_action or (lambda a: None)
        self.log: list[tuple[int, str]] = []  # (t_ms, description)

    def post(self, event: Event) -> None:
        self._inbox.append(event)

    def _apply(self, event: Event, t_ms: int) -> None:
        old = self.state
        self.state, actions = transition(self.state, event)
        if self.state != old:
            self.log.append((t_ms, f"{old.name} -> {self.state.name} on {type(event).__name__}"))
        for action in actions:
            if isinstance(action, FlushPlayback):
                dropped = self.output.flush()
                self.log.append((t_ms, f"flushed {dropped} queued frames"))
            self._on_action(action)

    def tick(self, n: int) -> None:
        t_ms = n * FRAME_MS

        while self._inbox:
            self._apply(self._inbox.popleft(), t_ms)

        frame, is_agent_audio = self.output.next_frame()
        self._send_audio(frame)

        if is_agent_audio and self.state == State.THINKING:
            self._apply(AgentAudioStarted(t_ms), t_ms)
        elif not is_agent_audio and self.state == State.SPEAKING and self.output.done:
            self.output.done = False
            self._apply(AgentAudioFinished(t_ms), t_ms)
