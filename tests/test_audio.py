from engine.audio import FRAME_BYTES, SILENCE, OutputBuffer


def test_push_splits_into_frames_and_keeps_remainder():
    buf = OutputBuffer()
    buf.push(b"\x01" * (FRAME_BYTES + 100))
    assert len(buf) == 1
    buf.mark_done()  # remainder padded into a second frame
    assert len(buf) == 2
    _, _ = buf.next_frame()
    last, is_audio = buf.next_frame()
    assert is_audio and len(last) == FRAME_BYTES and last.endswith(b"\x00")


def test_empty_buffer_returns_silence():
    frame, is_audio = OutputBuffer().next_frame()
    assert frame == SILENCE and not is_audio


def test_flush_drops_everything():
    buf = OutputBuffer()
    buf.push(b"\x01" * FRAME_BYTES * 10)
    assert buf.flush() == 10
    assert len(buf) == 0 and buf.next_frame()[0] == SILENCE
