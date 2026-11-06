from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


# ---------- Events (inputs) ----------
@dataclass(frozen=True)
class UserSpeechStarted:
    t_ms: int

@dataclass(frozen=True)
class UserSpeechEnded:
    t_ms: int

@dataclass(frozen=True)
class TranscriptFinal:
    t_ms: int
    text: str

@dataclass(frozen=True)
class AgentAudioStarted:
    t_ms: int

@dataclass(frozen=True)
class AgentAudioFinished:
    t_ms: int

Event = (UserSpeechStarted | UserSpeechEnded | TranscriptFinal
         | AgentAudioStarted | AgentAudioFinished)


# ---------- Actions (outputs) ----------
@dataclass(frozen=True)
class StartASR: ...

@dataclass(frozen=True)
class FinalizeASR: ...

@dataclass(frozen=True)
class StartResponse:
    text: str

@dataclass(frozen=True)
class CancelAgentJobs:
    reason: str

@dataclass(frozen=True)
class FlushPlayback: ...

Action = StartASR | FinalizeASR | StartResponse | CancelAgentJobs | FlushPlayback


def transition(state: State, event: Event) -> tuple[State, list[Action]]:
    match (state, event):
        case (State.IDLE, UserSpeechStarted()):
            return State.LISTENING, [StartASR()]
        case (State.THINKING | State.SPEAKING, UserSpeechStarted()):
            return State.LISTENING, [
                CancelAgentJobs(reason="barge_in"),
                FlushPlayback(),
                StartASR(),
            ]
        case (State.LISTENING, UserSpeechEnded()):
            return State.LISTENING, [FinalizeASR()]
        case (State.LISTENING, TranscriptFinal(text=text)) if text.strip():
            return State.THINKING, [StartResponse(text=text)]
        case (State.LISTENING, TranscriptFinal()):
            return State.IDLE, []
        case (State.THINKING, AgentAudioStarted()):
            return State.SPEAKING, []
        case (State.SPEAKING, AgentAudioFinished()):
            return State.IDLE, []
        case _:
            return state, []