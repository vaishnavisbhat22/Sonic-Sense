"""Focused API and schema tests using generated audio, not labeled recordings."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app import inference
from backend.app.inference import (
    FeatureSchemaError,
    _build_input_frame,
    load_model_and_metadata,
)
from backend.app.main import app


client = TestClient(app)


def _wav_bytes():
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.3 * np.sin(2 * np.pi * 440 * time)
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV")
    return output.getvalue()


def test_health_checks_model_readiness():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_name": "XGBoost"}


def test_model_classes_and_feature_schema_match_metadata():
    model, metadata = load_model_and_metadata()

    assert list(model.classes_) == [0, 1]
    assert metadata["xgboost_class_mapping"] == {
        "0": "synthetic",
        "1": "human",
    }
    assert len(metadata["feature_columns"]) == 134


def test_uploaded_wav_returns_prediction_contract():
    response = client.post(
        "/predict",
        files={"audio": ("sample.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in {"human", "synthetic"}
    assert result["model_name"] == "XGBoost"
    assert result["processing_time_ms"] >= 0
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["human_probability"] <= 1
    assert 0 <= result["synthetic_probability"] <= 1
    assert result["human_probability"] + result["synthetic_probability"] == pytest.approx(1)


def test_unsupported_upload_type_is_rejected():
    response = client.post(
        "/predict",
        files={"audio": ("sample.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 415


def test_empty_upload_is_rejected():
    response = client.post(
        "/predict",
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400


def test_undecodable_audio_is_rejected():
    response = client.post(
        "/predict",
        files={"audio": ("invalid.wav", b"not a wav file", "audio/wav")},
    )

    assert response.status_code == 422


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(inference, "MAX_UPLOAD_BYTES", 4)
    response = client.post(
        "/predict",
        files={"audio": ("large.wav", b"12345", "audio/wav")},
    )

    assert response.status_code == 413


def test_missing_features_are_rejected():
    _, metadata = load_model_and_metadata()
    incomplete = {name: 0.0 for name in metadata["feature_columns"][:-1]}

    with pytest.raises(FeatureSchemaError, match="feature names or count"):
        _build_input_frame(incomplete, metadata["feature_columns"])


def test_api_rejects_missing_extracted_features(monkeypatch):
    monkeypatch.setattr(
        inference,
        "extract_audio_features_from_bytes",
        lambda audio_bytes: {"mfcc_01_mean": 0.0},
    )
    response = client.post(
        "/predict",
        files={"audio": ("sample.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Audio features could not be matched to the detection model."
    )


def test_incorrect_feature_count_is_rejected():
    _, metadata = load_model_and_metadata()
    wrong_count = {
        "unused_feature_{0}".format(index): 0.0
        for index in range(134)
    }

    with pytest.raises(FeatureSchemaError, match="feature names or count"):
        _build_input_frame(wrong_count, metadata["feature_columns"])
