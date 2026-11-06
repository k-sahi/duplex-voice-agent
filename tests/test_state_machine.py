from engine.state_machine import (
    State, transition,
    UserSpeechStarted, UserSpeechEnded, TranscriptFinal,
    AgentAudioStarted, AgentAudioFinished,
    StartASR, FinalizeASR, StartResponse, CancelAgentJobs, FlushPlayback,
)


def test_full_happy_path():
    s = State.IDLE
    s, a = transition(s, UserSpeechStarted(0));          assert s == State.LISTENING and a == [StartASR()]
    s, a = transition(s, UserSpeechEnded(1000));         assert a == [FinalizeASR()]
    s, a = transition(s, TranscriptFinal(1100, "hi"));   assert s == State.THINKING and a == [StartResponse("hi")]
    s, a = transition(s, AgentAudioStarted(1400));       assert s == State.SPEAKING
    s, a = transition(s, AgentAudioFinished(3000));      assert s == State.IDLE


def test_barge_in_while_speaking():
    s, a = transition(State.SPEAKING, UserSpeechStarted(0))
    assert s == State.LISTENING
    assert a == [CancelAgentJobs("barge_in"), FlushPlayback(), StartASR()]


def test_barge_in_while_thinking():
    s, a = transition(State.THINKING, UserSpeechStarted(0))
    assert s == State.LISTENING
    assert CancelAgentJobs("barge_in") in a


def test_empty_transcript_returns_to_idle():
    s, a = transition(State.LISTENING, TranscriptFinal(0, "   "))
    assert s == State.IDLE and a == []


def test_late_event_from_cancelled_job_is_ignored():
    s, a = transition(State.LISTENING, AgentAudioStarted(0))
    assert s == State.LISTENING and a == []