from pathlib import Path
import numpy as np
import soundfile as sf
from preprocess_audio import make_segments, load_and_preprocess


def test_make_segments_creates_five_second_segment():
    audio = np.ones(16000, dtype=np.float32)
    segments = make_segments(audio)
    assert len(segments) == 1
    assert len(segments[0]) == 80000


def test_make_segments_splits_long_audio():
    audio = np.ones(16000 * 6, dtype=np.float32)
    segments = make_segments(audio)
    assert len(segments) == 2
    assert all(len(segment) == 80000 for segment in segments)


def test_make_segments_zero_pads_short_audio():
    audio = np.ones(1000, dtype=np.float32)
    segments = make_segments(audio)
    assert len(segments) == 1
    assert len(segments[0]) == 80000
    assert np.all(segments[0][1000:] == 0)


def test_load_and_preprocess_returns_valid_audio(tmp_path):
    audio_path = tmp_path / "sample.wav"
    original_audio = np.sin(
        2 * np.pi * 440 * np.arange(16000) / 16000
    ).astype(np.float32)
    sf.write(audio_path, original_audio, 16000)

    result = load_and_preprocess(audio_path)
    audio = next(item for item in result if isinstance(item, np.ndarray))

    assert audio.ndim == 1
    assert len(audio) > 0
    assert np.all(np.isfinite(audio))
