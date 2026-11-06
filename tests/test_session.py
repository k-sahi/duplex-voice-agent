from engine.audio import FRAME_BYTES, SILENCE
from engine.session import Session
from engine.state_machine import (
    CancelAgentJobs, State, StartResponse, TranscriptFinal,
    UserSpeechEnded, UserSpeechStarted,
)

AUDIO = b"\x01" * FRAME_BYTES


def make_session():
    sent, actions = [], []
    return Session(sent.append, actions.append), sent, actions


def test_every_tick_sends_exactly_one_frame():
    s, sent, _ = make_session()
    for n in range(10):
        s.tick(n)
    assert len(sent) == 10 and all(len(f) == FRAME_BYTES for f in sent)


def test_speaking_state_follows_audio_actually_sent():
    s, sent, actions = make_session()
    s.post(UserSpeechStarted(0)); s.tick(0)
    s.post(UserSpeechEnded(0)); s.tick(1)
    s.post(TranscriptFinal(0, "hi")); s.tick(2)
    assert s.state == State.THINKING and StartResponse("hi") in actions

    s.output.push(AUDIO * 3)
    s.output.mark_done()
    s.tick(3)
    assert s.state == State.SPEAKING
    s.tick(4); s.tick(5)          # frames 2 and 3
    assert s.state == State.SPEAKING
    s.tick(6)                     # buffer empty + done -> finished
    assert s.state == State.IDLE


def test_barge_in_stops_audio_on_the_same_tick():
    s, sent, actions = make_session()
    s.state = State.SPEAKING
    s.output.push(AUDIO * 50)
    s.tick(0)
    assert sent[-1] == AUDIO

    s.post(UserSpeechStarted(20))
    s.tick(1)
    assert s.state == State.LISTENING
    assert sent[-1] == SILENCE          # no stale speech after the interrupt
    assert len(s.output) == 0
    assert CancelAgentJobs("barge_in") in actions
