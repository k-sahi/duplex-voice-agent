from engine.speech_gate import SpeechGate


def run(gate, probs):
    return [r for r in (gate.update(p) for p in probs) if r]


def test_needs_100ms_of_confident_speech_to_start():
    g = SpeechGate()
    assert run(g, [0.9] * 4) == []          # 80 ms: not yet
    assert run(g, [0.9]) == ["start"]       # 100 ms


def test_flicker_does_not_start():
    assert run(SpeechGate(), [0.9, 0.9, 0.1, 0.9, 0.9, 0.1] * 5) == []


def test_ends_after_400ms_quiet():
    g = SpeechGate()
    run(g, [0.9] * 5)
    assert run(g, [0.1] * 19) == []
    assert run(g, [0.1]) == ["end"]


def test_hysteresis_middle_values_keep_speaking():
    g = SpeechGate()
    run(g, [0.9] * 5)
    assert run(g, [0.4] * 100) == [] and g.speaking  # between 0.35 and 0.5: no change


def test_a_word_resets_the_end_countdown():
    g = SpeechGate()
    run(g, [0.9] * 5)
    assert run(g, [0.1] * 15 + [0.9] + [0.1] * 15) == []
