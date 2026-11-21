from pathlib import Path

import numpy as np
import pytest

import compare_inference as diagnostic
from backend.app import inference
from backend.app.inference import (
    AudioInputError,
    FeatureSchemaError,
    InferenceExecutionError,
)
from extract_audio_features import feature_columns


class FakeModel:
    def __init__(self, classes=(0, 1), probabilities=(0.2, 0.8), prediction=1):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)
        self.prediction = prediction

    def predict(self, features):
        return np.asarray([self.prediction])

    def predict_proba(self, features):
        return np.asarray([self.probabilities])


def _metadata():
    return {
        "feature_columns": feature_columns(),
        "best_model": "XGBoost",
        "xgboost_class_mapping": {
            "0": "synthetic",
            "1": "human",
        },
    }


def _features():
    return {name: 0.0 for name in feature_columns()}


def _response(prediction="human", confidence=0.8, human=0.8, synthetic=0.2):
    return {
        "prediction": prediction,
        "confidence": confidence,
        "human_probability": human,
        "synthetic_probability": synthetic,
        "model_name": "XGBoost",
        "processing_time_ms": 1.0,
    }


def _run_diagnostic(path, model, metadata, printer):
    return diagnostic.run_diagnostic(
        [path],
        None,
        model,
        metadata,
        metadata["feature_columns"],
        printer=printer,
    )


def test_training_extraction_failure_returns_nonzero_without_partial_result(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    messages = []

    def fail_extraction(_):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(diagnostic, "extract_audio_features", fail_extraction)

    exit_code = _run_diagnostic(audio_path, FakeModel(), _metadata(), messages.append)

    assert exit_code != 0
    assert any("Training-style feature extraction failed" in message for message in messages)
    with pytest.raises(diagnostic.DiagnosticFailure, match="decoder failed"):
        diagnostic.diagnose_file(
            audio_path,
            b"audio",
            FakeModel(),
            _metadata(),
            feature_columns(),
        )


def test_prediction_failure_returns_nonzero(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    messages = []

    monkeypatch.setattr(diagnostic, "extract_audio_features", lambda _: _features())
    monkeypatch.setattr(
        diagnostic,
        "backend_predict_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model failed")),
    )

    exit_code = _run_diagnostic(audio_path, FakeModel(), _metadata(), messages.append)

    assert exit_code != 0
    assert any("Training-style prediction failed" in message for message in messages)


def test_inference_rejects_incorrect_model_class_mapping():
    with pytest.raises(
        InferenceExecutionError,
        match="could not analyze this audio",
    ) as error:
        inference.predict_features(
            _features(),
            model=FakeModel(classes=(0, 2)),
            metadata=_metadata(),
        )
    assert "classes" in str(error.value.__cause__)


def test_inference_rejects_incorrect_metadata_class_mapping():
    metadata = _metadata()
    metadata["xgboost_class_mapping"] = {
        "0": "human",
        "1": "synthetic",
    }

    with pytest.raises(
        InferenceExecutionError,
        match="could not analyze this audio",
    ) as error:
        inference.predict_features(
            _features(),
            model=FakeModel(),
            metadata=metadata,
        )
    assert "class mapping" in str(error.value.__cause__)


def test_inference_rejects_probability_dimension_mismatch():
    with pytest.raises(
        InferenceExecutionError,
        match="could not analyze this audio",
    ) as error:
        inference.predict_features(
            _features(),
            model=FakeModel(probabilities=(1.0,)),
            metadata=_metadata(),
        )
    assert "probability shape" in str(error.value.__cause__)


def test_inference_maps_probabilities_using_model_class_order():
    result = inference.predict_features(
        _features(),
        model=FakeModel(classes=(1, 0), probabilities=(0.8, 0.2)),
        metadata=_metadata(),
    )

    assert result["prediction"] == "human"
    assert result["human_probability"] == pytest.approx(0.8)
    assert result["synthetic_probability"] == pytest.approx(0.2)


def test_inference_rejects_invalid_probability_values():
    with pytest.raises(
        InferenceExecutionError,
        match="could not analyze this audio",
    ) as error:
        inference.predict_features(
            _features(),
            model=FakeModel(probabilities=(1.2, -0.2)),
            metadata=_metadata(),
        )
    assert "[0, 1]" in str(error.value.__cause__)


def test_inference_rejects_probabilities_that_do_not_sum_to_one():
    with pytest.raises(
        InferenceExecutionError,
        match="could not analyze this audio",
    ) as error:
        inference.predict_features(
            _features(),
            model=FakeModel(probabilities=(0.2, 0.2)),
            metadata=_metadata(),
        )
    assert "sum to 1" in str(error.value.__cause__)


def test_prediction_disagreement_returns_nonzero(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    messages = []
    features = _features()

    monkeypatch.setattr(diagnostic, "extract_audio_features", lambda _: features)
    monkeypatch.setattr(
        diagnostic,
        "extract_audio_features_from_bytes",
        lambda _: features,
    )
    monkeypatch.setattr(
        diagnostic,
        "backend_predict_features",
        lambda *args, **kwargs: _response(),
    )
    monkeypatch.setattr(
        diagnostic,
        "backend_predict_audio",
        lambda *args, **kwargs: _response(
            "synthetic",
            confidence=0.8,
            human=0.2,
            synthetic=0.8,
        ),
    )

    exit_code = _run_diagnostic(audio_path, FakeModel(), _metadata(), messages.append)

    assert exit_code != 0
    assert any("predictions disagree" in message for message in messages)


def test_missing_ground_truth_labels_are_rejected():
    paths = [Path("first.wav"), Path("second.wav")]

    with pytest.raises(diagnostic.GroundTruthError, match="exactly one"):
        diagnostic.validate_ground_truth_labels(paths, ["human"])
    with pytest.raises(diagnostic.GroundTruthError, match="cannot be blank"):
        diagnostic.validate_ground_truth_labels(paths, ["human", None])
    with pytest.raises(diagnostic.GroundTruthError, match="required"):
        diagnostic.evaluate_ground_truth(["human"], None)


def test_ground_truth_metrics_use_verified_labels():
    metrics = diagnostic.evaluate_ground_truth(
        ["human", "synthetic", "human", "synthetic"],
        ["human", "synthetic", "synthetic", "human"],
    )

    assert metrics["confusion_matrix"] == {
        "human": {"human": 1, "synthetic": 1},
        "synthetic": {"human": 1, "synthetic": 1},
    }
    assert metrics["correct"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1


def test_diagnostic_calls_actual_backend_wrapper(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_bytes = b"valid wav payload"
    audio_path.write_bytes(audio_bytes)
    model = FakeModel()
    metadata = _metadata()
    features = _features()

    monkeypatch.setattr(diagnostic, "extract_audio_features", lambda _: features)
    monkeypatch.setattr(
        diagnostic,
        "extract_audio_features_from_bytes",
        lambda _: features,
    )
    monkeypatch.setattr(
        inference,
        "load_model_and_metadata",
        lambda: (model, metadata),
    )
    monkeypatch.setattr(
        inference,
        "extract_audio_features_from_bytes",
        lambda _: features,
    )

    result = diagnostic.diagnose_file(
        audio_path,
        audio_bytes,
        model,
        metadata,
        metadata["feature_columns"],
    )

    assert result["wrapper_response"]["prediction"] == "human"
    assert result["wrapper_response"]["human_probability"] == pytest.approx(0.8)


def test_cli_files_without_labels_skip_prompt(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    metadata = _metadata()
    received = {}

    monkeypatch.setattr(
        diagnostic,
        "load_artifacts",
        lambda: (FakeModel(), metadata, metadata["feature_columns"], {0: "synthetic", 1: "human"}),
    )
    monkeypatch.setattr(
        diagnostic,
        "prompt_ground_truth_labels",
        lambda paths: pytest.fail("CLI invocation should not prompt for labels"),
    )

    def fake_run(paths, labels, model, metadata, columns):
        received["labels"] = labels
        return 0

    monkeypatch.setattr(diagnostic, "run_diagnostic", fake_run)

    assert diagnostic.main([str(audio_path)]) == 0
    assert received["labels"] is None


def test_backend_wrapper_still_rejects_unsupported_uploads():
    with pytest.raises(AudioInputError, match="Unsupported audio file type"):
        inference.predict_audio("sample.txt", b"not audio")


def test_shared_inference_rejects_invalid_feature_schema():
    with pytest.raises(FeatureSchemaError, match="feature names or count"):
        inference.predict_features(
            {"not_a_model_feature": 1.0},
            model=FakeModel(),
            metadata=_metadata(),
        )
