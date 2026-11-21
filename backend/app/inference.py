"""Model loading and shared audio inference for the FastAPI backend."""

import json
import logging
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from extract_audio_features import (
    extract_audio_features_from_bytes,
    feature_columns as extractor_feature_columns,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"
METADATA_PATH = PROJECT_ROOT / "results" / "model_metadata.json"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
EXPECTED_CLASS_MAPPING = {0: "synthetic", 1: "human"}


def _metadata_class_mapping(metadata):
    mapping = {
        int(label): name
        for label, name in metadata["xgboost_class_mapping"].items()
    }
    if mapping != EXPECTED_CLASS_MAPPING:
        raise ValueError("Metadata class mapping does not match the expected labels.")
    return mapping


class AudioInputError(ValueError):
    """Raised when an uploaded file does not meet the input requirements."""

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


class AudioProcessingError(ValueError):
    """Raised when validly sized audio cannot be decoded or featurized."""


class FeatureSchemaError(ValueError):
    """Raised when extracted features do not match the model schema."""


class ModelUnavailableError(RuntimeError):
    """Raised when the model or its metadata cannot be loaded safely."""


class InferenceExecutionError(RuntimeError):
    """Raised when the loaded model cannot produce a valid prediction."""


@lru_cache(maxsize=1)
def load_model_and_metadata():
    """Load artifacts once and check their label and feature contracts."""

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        model = joblib.load(MODEL_PATH)

        columns = metadata["feature_columns"]
        if (
            not isinstance(columns, list)
            or len(columns) != 134
            or columns != extractor_feature_columns()
        ):
            raise ValueError(
                "Metadata feature columns do not match the 134-column extractor schema."
            )
        if metadata.get("number_of_features") != len(columns):
            raise ValueError("Metadata feature count is inconsistent.")

        mapping = _metadata_class_mapping(metadata)
        classes = [int(label) for label in model.classes_]
        if (
            mapping != EXPECTED_CLASS_MAPPING
            or len(classes) != len(set(classes))
            or set(classes) != set(mapping)
        ):
            raise ValueError("Model classes do not match the documented class mapping.")
        if getattr(model, "n_features_in_", None) != len(columns):
            raise ValueError("Model feature count does not match metadata.")

        model_columns = getattr(model, "feature_names_in_", None)
        if model_columns is not None and list(model_columns) != columns:
            raise ValueError("Model feature ordering does not match metadata.")
        if metadata.get("best_model") != "XGBoost":
            raise ValueError("Metadata does not identify the expected XGBoost model.")
    except Exception as error:
        LOGGER.exception("Unable to load or validate SonicSense model artifacts.")
        raise ModelUnavailableError(
            "The detection model is unavailable or has an invalid configuration."
        ) from error

    return model, metadata


def _validate_upload(filename, audio_bytes):
    if not filename or Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise AudioInputError(
            "Unsupported audio file type. Use WAV, FLAC, MP3, OGG, or M4A.",
            status_code=415,
        )
    if not audio_bytes:
        raise AudioInputError("The uploaded audio file is empty.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise AudioInputError(
            f"Audio file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit.",
            status_code=413,
        )


def _build_input_frame(features, columns):
    if not isinstance(features, dict):
        raise FeatureSchemaError("Feature extraction did not return a feature mapping.")

    missing = [column for column in columns if column not in features]
    extra = [column for column in features if column not in columns]
    if missing or extra or len(features) != len(columns):
        raise FeatureSchemaError(
            "Extracted feature names or count do not match the model schema."
        )

    try:
        values = np.asarray([features[column] for column in columns], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise FeatureSchemaError("Extracted features must all be numeric.") from error
    if values.size != 134 or not np.all(np.isfinite(values)):
        raise FeatureSchemaError(
            "Extracted features must contain exactly 134 finite values."
        )

    return pd.DataFrame([values], columns=columns)


def predict_features(features, model=None, metadata=None, started=None):
    """Validate and classify an extracted feature mapping using backend rules."""

    if model is None or metadata is None:
        model, metadata = load_model_and_metadata()
    if started is None:
        started = time.perf_counter()
    input_frame = _build_input_frame(features, metadata["feature_columns"])

    try:
        class_ids = [int(value) for value in model.classes_]
        class_mapping = _metadata_class_mapping(metadata)
        if (
            len(class_ids) != len(set(class_ids))
            or set(class_ids) != set(class_mapping)
        ):
            raise ValueError("Model classes do not match the documented class mapping.")

        predicted_class = int(model.predict(input_frame)[0])
        probability_matrix = np.asarray(model.predict_proba(input_frame), dtype=float)
        if probability_matrix.shape != (1, len(class_ids)):
            raise ValueError("Model returned an unexpected class or probability shape.")
        probabilities = probability_matrix[0]
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Model returned non-finite probabilities.")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Model returned probabilities outside [0, 1].")
        if not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("Model probabilities do not sum to 1.")
        if predicted_class not in class_mapping:
            raise ValueError("Model returned an unknown predicted class.")

        probability_by_label = {
            class_mapping[class_id]: float(probabilities[index])
            for index, class_id in enumerate(class_ids)
        }
        prediction = class_mapping[predicted_class]
        confidence = probability_by_label[prediction]
    except Exception as error:
        LOGGER.exception("SonicSense model inference failed.")
        raise InferenceExecutionError(
            "The detection model could not analyze this audio."
        ) from error

    return {
        "prediction": prediction,
        "confidence": confidence,
        "human_probability": probability_by_label["human"],
        "synthetic_probability": probability_by_label["synthetic"],
        "model_name": metadata["best_model"],
        "processing_time_ms": (time.perf_counter() - started) * 1000,
    }


def predict_audio(filename, audio_bytes):
    """Run the validated backend upload and inference pipeline."""

    _validate_upload(filename, audio_bytes)
    model, metadata = load_model_and_metadata()
    started = time.perf_counter()

    try:
        features = extract_audio_features_from_bytes(audio_bytes)
    except Exception as error:
        LOGGER.info("Audio decoding or feature extraction failed: %s", error)
        raise AudioProcessingError(
            "The audio could not be decoded or analyzed."
        ) from error

    return predict_features(
        features,
        model=model,
        metadata=metadata,
        started=started,
    )
