"""Compare training-style extraction with the FastAPI inference pipeline."""

import argparse
import sys
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np

from backend.app.inference import (
    EXPECTED_CLASS_MAPPING,
    predict_audio as backend_predict_audio,
    predict_features as backend_predict_features,
    load_model_and_metadata,
)
from extract_audio_features import (
    extract_audio_features,
    extract_audio_features_from_bytes,
    feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_RTOL = 1e-4
FEATURE_ATOL = 1e-5
PROBABILITY_ATOL = 1e-6
CLASS_NAMES = ("human", "synthetic")
RESPONSE_FIELDS = {
    "prediction",
    "confidence",
    "human_probability",
    "synthetic_probability",
    "model_name",
    "processing_time_ms",
}


class DiagnosticFailure(RuntimeError):
    """Raised when a required pipeline comparison fails."""


class GroundTruthError(ValueError):
    """Raised when supplied ground-truth labels are incomplete or invalid."""


def select_audio_files():
    root = Tk()
    root.withdraw()
    try:
        selected_paths = filedialog.askopenfilenames(
            title="Select audio recordings",
            filetypes=[
                ("Audio files", "*.wav *.flac *.mp3 *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    return [Path(path) for path in selected_paths]


def load_artifacts():
    """Use the backend's artifact validation as the single source of truth."""

    model, metadata = load_model_and_metadata()
    columns = metadata["feature_columns"]
    if columns != feature_columns():
        raise ValueError("Feature columns do not match the extractor schema.")

    return model, metadata, columns, dict(EXPECTED_CLASS_MAPPING)


def normalize_label(value):
    return str(value).strip().lower()


def validate_ground_truth_labels(audio_paths, labels):
    if labels is None:
        return None
    if len(labels) != len(audio_paths):
        raise GroundTruthError(
            "Provide exactly one ground-truth label per audio file."
        )

    normalized = [normalize_label(label) if label is not None else "" for label in labels]
    invalid = [
        label for label in normalized
        if label not in CLASS_NAMES
    ]
    if invalid:
        raise GroundTruthError(
            "Ground-truth labels must be HUMAN or SYNTHETIC; "
            "labels cannot be blank or inferred from predictions."
        )
    return normalized


def prompt_ground_truth_labels(audio_paths, input_fn=input):
    labels = [
        input_fn(
            f"Verified ground truth for {path.name} "
            "(HUMAN/SYNTHETIC; blank for all files to skip evaluation): "
        )
        for path in audio_paths
    ]
    if all(not normalize_label(label) for label in labels):
        return None
    return validate_ground_truth_labels(audio_paths, labels)


def validate_feature_output(features, columns, pipeline_name):
    if not isinstance(features, dict):
        raise DiagnosticFailure(
            f"{pipeline_name} feature extraction did not return a mapping."
        )
    if list(features.keys()) != columns:
        raise DiagnosticFailure(
            f"{pipeline_name} feature names/order do not match model metadata."
        )

    try:
        values = np.asarray([features[column] for column in columns], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DiagnosticFailure(
            f"{pipeline_name} features must all be numeric."
        ) from error
    if values.shape != (len(columns),) or not np.all(np.isfinite(values)):
        raise DiagnosticFailure(
            f"{pipeline_name} features must be {len(columns)} finite values."
        )
    return values


def validate_response(response, metadata, pipeline_name):
    if not isinstance(response, dict) or not RESPONSE_FIELDS.issubset(response):
        raise DiagnosticFailure(
            f"{pipeline_name} returned an invalid inference response contract."
        )
    if response["prediction"] not in EXPECTED_CLASS_MAPPING.values():
        raise DiagnosticFailure(f"{pipeline_name} returned an unknown prediction.")
    if response["model_name"] != metadata.get("best_model"):
        raise DiagnosticFailure(f"{pipeline_name} returned an unexpected model name.")

    try:
        probabilities = {
            "human": float(response["human_probability"]),
            "synthetic": float(response["synthetic_probability"]),
        }
        confidence = float(response["confidence"])
        processing_time = float(response["processing_time_ms"])
    except (TypeError, ValueError) as error:
        raise DiagnosticFailure(
            f"{pipeline_name} returned non-numeric response values."
        ) from error

    values = [*probabilities.values(), confidence, processing_time]
    if not np.all(np.isfinite(values)):
        raise DiagnosticFailure(f"{pipeline_name} returned non-finite values.")
    if any(value < 0 or value > 1 for value in probabilities.values()):
        raise DiagnosticFailure(f"{pipeline_name} returned invalid probabilities.")
    if not np.isclose(
        sum(probabilities.values()),
        1.0,
        rtol=0.0,
        atol=PROBABILITY_ATOL,
    ):
        raise DiagnosticFailure(f"{pipeline_name} probabilities do not sum to 1.")
    if not np.isclose(
        confidence,
        probabilities[response["prediction"]],
        rtol=0.0,
        atol=PROBABILITY_ATOL,
    ):
        raise DiagnosticFailure(
            f"{pipeline_name} confidence does not match the predicted class."
        )
    if processing_time < 0:
        raise DiagnosticFailure(f"{pipeline_name} returned a negative processing time.")


def responses_match(left, right):
    if left["prediction"] != right["prediction"] or left["model_name"] != right["model_name"]:
        return False
    for field in ("confidence", "human_probability", "synthetic_probability"):
        if not np.isclose(
            float(left[field]),
            float(right[field]),
            rtol=0.0,
            atol=PROBABILITY_ATOL,
        ):
            return False
    return True


def _run_stage(stage_name, operation):
    try:
        return operation()
    except Exception as error:
        root_cause = error
        while root_cause.__cause__ is not None:
            root_cause = root_cause.__cause__
        detail = f"{type(error).__name__}: {error}"
        if root_cause is not error:
            detail += (
                f"; cause {type(root_cause).__name__}: {root_cause}"
            )
        raise DiagnosticFailure(
            f"{stage_name} failed ({detail})"
        ) from error


def diagnose_file(audio_path, audio_bytes, model, metadata, columns):
    """Run each extraction route and the real backend wrapper for one file."""

    training_features = _run_stage(
        "Training-style feature extraction",
        lambda: extract_audio_features(audio_path),
    )
    training_values = validate_feature_output(
        training_features,
        columns,
        "Training-style",
    )
    training_response = _run_stage(
        "Training-style prediction",
        lambda: backend_predict_features(
            training_features,
            model=model,
            metadata=metadata,
        ),
    )
    validate_response(training_response, metadata, "Training-style prediction")

    bytes_features = _run_stage(
        "Bytes-based feature extraction",
        lambda: extract_audio_features_from_bytes(audio_bytes),
    )
    bytes_values = validate_feature_output(bytes_features, columns, "Bytes-based")
    bytes_response = _run_stage(
        "Bytes-based prediction",
        lambda: backend_predict_features(
            bytes_features,
            model=model,
            metadata=metadata,
        ),
    )
    validate_response(bytes_response, metadata, "Bytes-based prediction")

    wrapper_response = _run_stage(
        "FastAPI inference wrapper",
        lambda: backend_predict_audio(audio_path.name, audio_bytes),
    )
    validate_response(wrapper_response, metadata, "FastAPI inference wrapper")

    differences = np.abs(training_values - bytes_values)
    if not np.allclose(
        training_values,
        bytes_values,
        rtol=FEATURE_RTOL,
        atol=FEATURE_ATOL,
    ):
        raise DiagnosticFailure(
            "Training-style and bytes-based features differ beyond tolerance "
            f"(mean absolute difference={float(np.mean(differences)):.8g}, "
            f"maximum={float(np.max(differences)):.8g})."
        )

    predictions = {
        training_response["prediction"],
        bytes_response["prediction"],
        wrapper_response["prediction"],
    }
    if len(predictions) != 1:
        raise DiagnosticFailure(
            "Training-style, bytes-based, and FastAPI predictions disagree: "
            f"{training_response['prediction']}, {bytes_response['prediction']}, "
            f"{wrapper_response['prediction']}."
        )
    if not responses_match(bytes_response, wrapper_response):
        raise DiagnosticFailure(
            "The bytes-based result does not match the FastAPI wrapper response."
        )

    return {
        "audio_path": audio_path,
        "training_values": training_values,
        "bytes_values": bytes_values,
        "mean_absolute_difference": float(np.mean(differences)),
        "maximum_absolute_difference": float(np.max(differences)),
        "training_response": training_response,
        "bytes_response": bytes_response,
        "wrapper_response": wrapper_response,
    }


def evaluate_ground_truth(predictions, labels):
    if labels is None:
        raise GroundTruthError(
            "Ground-truth labels are required for classification metrics."
        )
    if len(labels) != len(predictions) or not predictions:
        raise GroundTruthError(
            "Provide one ground-truth label for every completed prediction."
        )

    normalized_labels = validate_ground_truth_labels(
        [None] * len(predictions),
        labels,
    )
    normalized_predictions = [normalize_label(value) for value in predictions]
    if any(value not in CLASS_NAMES for value in normalized_predictions):
        raise ValueError("Predictions must be HUMAN or SYNTHETIC labels.")

    matrix = {
        actual: {predicted: 0 for predicted in CLASS_NAMES}
        for actual in CLASS_NAMES
    }
    for actual, predicted in zip(normalized_labels, normalized_predictions):
        matrix[actual][predicted] += 1

    class_metrics = {}
    for label in CLASS_NAMES:
        true_positive = matrix[label][label]
        false_positive = sum(
            matrix[other][label]
            for other in CLASS_NAMES
            if other != label
        )
        false_negative = sum(
            matrix[label][other]
            for other in CLASS_NAMES
            if other != label
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else None
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator
            else None
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else 0.0 if precision is not None and recall is not None else None
        )
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    correct = sum(matrix[label][label] for label in CLASS_NAMES)
    total = len(normalized_labels)
    return {
        "confusion_matrix": matrix,
        "correct": correct,
        "false_positives": matrix["human"]["synthetic"],
        "false_negatives": matrix["synthetic"]["human"],
        "true_positives": matrix["synthetic"]["synthetic"],
        "true_negatives": matrix["human"]["human"],
        "accuracy": correct / total,
        "class_metrics": class_metrics,
    }


def print_ground_truth_metrics(metrics, printer=print):
    matrix = metrics["confusion_matrix"]
    printer("\nGround-truth confusion matrix (rows=actual, columns=predicted):")
    printer("                 HUMAN  SYNTHETIC")
    for actual, title in (("human", "HUMAN"), ("synthetic", "SYNTHETIC")):
        printer(
            f"{title:12} {matrix[actual]['human']:7d} "
            f"{matrix[actual]['synthetic']:10d}"
        )
    printer(f"Correct predictions: {metrics['correct']}")
    printer(
        "False positives (actual HUMAN predicted SYNTHETIC): "
        f"{metrics['false_positives']}"
    )
    printer(
        "False negatives (actual SYNTHETIC predicted HUMAN): "
        f"{metrics['false_negatives']}"
    )
    printer(f"Accuracy: {metrics['accuracy']:.4f}")
    printer("Class-wise metrics:")
    for label in CLASS_NAMES:
        values = metrics["class_metrics"][label]
        formatted = {
            key: "n/a" if value is None else f"{value:.4f}"
            for key, value in values.items()
        }
        printer(
            f"  {label.upper()}: precision={formatted['precision']}, "
            f"recall={formatted['recall']}, f1={formatted['f1']}"
        )


def run_diagnostic(
    audio_paths,
    labels,
    model,
    metadata,
    columns,
    printer=print,
):
    try:
        labels = validate_ground_truth_labels(audio_paths, labels)
    except GroundTruthError as error:
        printer(f"Ground-truth error: {error}")
        return 2
    if not audio_paths:
        printer("No audio files were selected.")
        return 2

    results = []
    failed = False
    for index, audio_path in enumerate(audio_paths):
        try:
            audio_bytes = audio_path.read_bytes()
            result = diagnose_file(
                audio_path,
                audio_bytes,
                model,
                metadata,
                columns,
            )
        except Exception as error:
            printer(
                f"[FAIL] {audio_path}: "
                f"{type(error).__name__}: {error}"
            )
            failed = True
            continue

        printer(f"\n[PASS] {audio_path}")
        printer(
            "Feature differences: "
            f"mean={result['mean_absolute_difference']:.8g}, "
            f"max={result['maximum_absolute_difference']:.8g}"
        )
        response = result["wrapper_response"]
        printer(
            f"FastAPI prediction: {response['prediction']} "
            f"(confidence={response['confidence']:.6f}, "
            f"human={response['human_probability']:.6f}, "
            f"synthetic={response['synthetic_probability']:.6f})"
        )
        results.append((index, result))

    if failed:
        printer(
            "\nDiagnostic incomplete: at least one audio file failed a required check."
        )
        if labels is not None:
            printer("Ground-truth metrics skipped because not all files completed.")
        return 1

    if labels is None:
        printer(
            "\nNo ground-truth labels supplied; classification metrics were not computed."
        )
    else:
        predictions = [
            result["wrapper_response"]["prediction"]
            for _, result in results
        ]
        print_ground_truth_metrics(
            evaluate_ground_truth(predictions, labels),
            printer=printer,
        )
    return 0


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare training-style feature extraction with the FastAPI "
            "inference wrapper."
        )
    )
    parser.add_argument(
        "audio_files",
        nargs="*",
        type=Path,
        help="Audio files to diagnose; opens a multi-file picker if omitted.",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        type=normalize_label,
        choices=CLASS_NAMES,
        help=(
            "Verified ground-truth label for the corresponding audio file. "
            "Repeat once per file and in the same order."
        ),
    )
    return parser


def main(argv=None):
    arguments = build_argument_parser().parse_args(argv)
    used_file_picker = not arguments.audio_files
    audio_paths = arguments.audio_files or select_audio_files()
    if not audio_paths:
        print("No audio files were selected.")
        return 2

    labels = arguments.labels
    if labels is None and used_file_picker and sys.stdin.isatty():
        try:
            labels = prompt_ground_truth_labels(audio_paths)
        except (EOFError, GroundTruthError) as error:
            print(f"Ground-truth error: {error}")
            return 2
    else:
        try:
            labels = validate_ground_truth_labels(audio_paths, labels)
        except GroundTruthError as error:
            print(f"Ground-truth error: {error}")
            return 2

    try:
        model, metadata, columns, _ = load_artifacts()
    except Exception as error:
        print(f"Could not load or validate model artifacts: {error}")
        return 1

    print("\n========== SONICSENSE PIPELINE DIAGNOSTIC ==========")
    print("Model:", metadata["best_model"])
    print("Feature count:", len(columns))
    print("Model classes:", list(model.classes_))
    print("Class mapping:", EXPECTED_CLASS_MAPPING)
    return run_diagnostic(
        audio_paths,
        labels,
        model,
        metadata,
        columns,
    )


if __name__ == "__main__":
    sys.exit(main())
