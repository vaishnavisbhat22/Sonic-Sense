from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import extract_audio_features as extractor


FIXTURE_DIR = Path(__file__).resolve().parent / "testdata"


def _assert_shared_decode_parity(audio_path, monkeypatch):
    decoder_sources = []
    original_read = extractor.sf.read

    def tracked_read(source, *args, **kwargs):
        decoder_sources.append(source)
        return original_read(source, *args, **kwargs)

    monkeypatch.setattr(extractor.sf, "read", tracked_read)
    path_features = extractor.extract_audio_features(audio_path)
    bytes_features = extractor.extract_audio_features_from_bytes(
        audio_path.read_bytes()
    )

    assert len(decoder_sources) == 2
    assert Path(decoder_sources[0]) == audio_path
    assert isinstance(decoder_sources[1], BytesIO)
    assert list(path_features) == extractor.feature_columns()
    assert list(bytes_features) == extractor.feature_columns()
    np.testing.assert_array_equal(
        [path_features[column] for column in extractor.feature_columns()],
        [bytes_features[column] for column in extractor.feature_columns()],
    )


@pytest.mark.parametrize(
    ("fixture_name", "expected_channels"),
    [
        ("mp3_mono.mp3", 1),
        ("mp3_stereo.mp3", 2),
    ],
)
def test_mp3_path_and_bytes_use_the_same_decoder(
    fixture_name,
    expected_channels,
    monkeypatch,
):
    audio_path = FIXTURE_DIR / fixture_name

    assert sf.info(audio_path).format == "MP3"
    assert sf.info(audio_path).channels == expected_channels
    _assert_shared_decode_parity(audio_path, monkeypatch)


@pytest.mark.parametrize("channels", [1, 2])
def test_ogg_opus_path_and_bytes_use_the_same_decoder(
    tmp_path,
    monkeypatch,
    channels,
):
    sample_rate = 48000
    sample_count = sample_rate
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    channel_signals = [
        0.3 * np.sin(2 * np.pi * (220 + 160 * channel) * time)
        for channel in range(channels)
    ]
    audio = (
        channel_signals[0]
        if channels == 1
        else np.column_stack(channel_signals)
    )
    audio_path = tmp_path / "opus-content-named-mp3.mp3"
    sf.write(audio_path, audio, sample_rate, format="OGG", subtype="OPUS")

    assert sf.info(audio_path).format == "OGG"
    assert sf.info(audio_path).subtype == "OPUS"
    assert sf.info(audio_path).channels == channels
    _assert_shared_decode_parity(audio_path, monkeypatch)


@pytest.mark.parametrize("channels", [1, 2])
def test_m4a_path_and_bytes_use_the_same_decoder(
    tmp_path,
    monkeypatch,
    channels,
):
    import av

    sample_rate = 32000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    channel_signals = [
        0.3 * np.sin(2 * np.pi * (220 + 160 * channel) * time)
        for channel in range(channels)
    ]
    audio = (
        channel_signals[0][np.newaxis, :]
        if channels == 1
        else np.vstack(channel_signals)
    )
    audio_path = tmp_path / "sample.m4a"

    with av.open(str(audio_path), mode="w", format="mp4") as container:
        stream = container.add_stream("aac", rate=sample_rate)
        stream.layout = "mono" if channels == 1 else "stereo"
        frame = av.AudioFrame.from_ndarray(
            audio,
            format="fltp",
            layout=stream.layout.name,
        )
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    assert audio_path.stat().st_size > 0
    _assert_shared_decode_parity(audio_path, monkeypatch)
