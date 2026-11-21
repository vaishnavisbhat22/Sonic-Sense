"""Compare five examples per class from the held-out feature dataset."""

import json
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from extract_audio_features import feature_columns as extractor_feature_columns


MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"
REPO_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "audio_features_v2.csv"
METADATA_PATH = PROJECT_ROOT / "results" / "model_metadata.json"
MANIFEST_PATH = PROJECT_ROOT / "results" / "split_manifest.csv"
SAMPLE_COUNT_PER_CLASS = 5
CLASS_MAPPING = {0: "synthetic", 1: "human"}


def main():
    for path in (MODEL_PATH, METADATA_PATH, MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required inference input is missing: {path}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    metadata_data_path = Path(metadata.get("dataset", ""))
    data_path = REPO_DATA_PATH if REPO_DATA_PATH.is_file() else metadata_data_path
    if not data_path.is_file():
        raise FileNotFoundError(
            "Feature dataset not found in the repository or at the metadata path: "
            f"{REPO_DATA_PATH} | {metadata_data_path}"
        )
    if metadata.get("feature_columns") != extractor_feature_columns():
        raise ValueError("Metadata feature ordering differs from the extractor.")
    if metadata.get("xgboost_class_mapping") != {
        str(class_id): label for class_id, label in CLASS_MAPPING.items()
    }:
        raise ValueError("Metadata does not use the documented class mapping.")

    model = joblib.load(MODEL_PATH)
    model_classes = [int(class_id) for class_id in model.classes_]
    if set(model_classes) != set(CLASS_MAPPING):
        raise ValueError(f"Unexpected model classes: {model_classes}")

    dataset = pd.read_csv(data_path)
    manifest = pd.read_csv(MANIFEST_PATH)
    feature_names = metadata["feature_columns"]
    missing_columns = [
        column for column in ["filename", "label", *feature_names]
        if column not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if not {"filename", "split"}.issubset(manifest.columns):
        raise ValueError("Split manifest must contain filename and split columns.")

    test_filenames = set(
        manifest.loc[manifest["split"] == "test", "filename"].astype(str)
    )
    test_set = dataset[dataset["filename"].astype(str).isin(test_filenames)].copy()
    test_set["label"] = test_set["label"].astype(str).str.strip().str.lower()

    samples = []
    for label in ("human", "synthetic"):
        class_samples = test_set[test_set["label"] == label].head(
            SAMPLE_COUNT_PER_CLASS
        )
        if len(class_samples) != SAMPLE_COUNT_PER_CLASS:
            raise ValueError(
                f"Expected {SAMPLE_COUNT_PER_CLASS} test samples labeled {label}; "
                f"found {len(class_samples)}."
            )
        samples.append(class_samples)

    selected = pd.concat(samples, ignore_index=True)
    features = selected[feature_names]
    if features.isnull().any().any():
        raise ValueError("Selected dataset samples contain missing feature values.")

    predicted_ids = [int(value) for value in model.predict(features)]
    probabilities = model.predict_proba(features)
    if probabilities.shape != (len(selected), len(model_classes)):
        raise ValueError(
            f"Unexpected probability output shape: {probabilities.shape}"
        )

    print(
        f"Dataset inference from {data_path} "
        "(five test samples per actual class):"
    )
    for row_index, (_, row) in enumerate(selected.iterrows()):
        predicted_id = predicted_ids[row_index]
        predicted_label = CLASS_MAPPING[predicted_id]
        predicted_probability = probabilities[
            row_index, model_classes.index(predicted_id)
        ]
        human_probability = probabilities[
            row_index, model_classes.index(1)
        ]
        synthetic_probability = probabilities[
            row_index, model_classes.index(0)
        ]
        print(
            f"{row['filename']} | actual={row['label']} "
            f"| predicted={predicted_label} "
            f"| confidence={predicted_probability:.4f} "
            f"| human_probability={human_probability:.4f} "
            f"| synthetic_probability={synthetic_probability:.4f}"
        )


if __name__ == "__main__":
    main()
