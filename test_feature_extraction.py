import numpy as np
import soundfile as sf

from extract_audio_features import extract_audio_features, feature_columns, load_labels


def test_extract_audio_features_returns_finite_fixed_schema(tmp_path):
    sample_rate = 16000
    samples = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.5 * np.sin(2 * np.pi * 440 * samples)
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, audio, sample_rate)

    features = extract_audio_features(audio_path)

    assert len(features) == 134
    assert list(features) == feature_columns()
    assert all(np.isfinite(value) for value in features.values())
    assert "gtcc_01_mean" in features
    assert "spectral_contrast_01_mean" in features
    assert "f0_mean_hz" in features
    assert "syllabic_rate_proxy" in features
    assert 420.0 < features["f0_mean_hz"] < 460.0
    assert features["voiced_fraction"] > 0.9


def test_load_labels_rejects_duplicate_filenames(tmp_path):
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "filename,label\nD_1.wav,human\nD_1.wav,synthetic\n",
        encoding="utf-8",
    )

    try:
        load_labels(labels_path)
    except ValueError as error:
        assert "Duplicate filename" in str(error)
    else:
        raise AssertionError("Expected duplicate labels to be rejected.")
