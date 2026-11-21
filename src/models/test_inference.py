import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "audio_features_v2.csv"
METADATA_PATH = PROJECT_ROOT / "results" / "model_metadata.json"
MANIFEST_PATH = PROJECT_ROOT / "results" / "split_manifest.csv"

# Load metadata
with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

feature_columns = metadata["feature_columns"]

# Load trained model
print("\nLoading XGBoost model...")
model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# Load dataset and split manifest
df = pd.read_csv(DATA_PATH)
manifest = pd.read_csv(MANIFEST_PATH)

# Select only test samples
test_filenames = manifest.loc[
    manifest["split"] == "test", "filename"
]

test_df = df[df["filename"].isin(test_filenames)].copy()

# Match feature order used during training
X_test = test_df[feature_columns]

# Actual labels
y_actual = test_df["label"].astype(str).str.lower()

# Generate predictions
print("\nRunning inference...")

raw_predictions = model.predict(X_test)

label_mapping = {
    0: "synthetic",
    1: "human"
}

y_predicted = pd.Series(raw_predictions).astype(int).map(
    label_mapping
)

# Display sample predictions
print("\n========== SAMPLE PREDICTIONS ==========")

for i in range(min(10, len(X_test))):

    print(f"\nAudio: {test_df.iloc[i]['filename']}")
    print(f"Actual: {y_actual.iloc[i]}")
    print(f"Predicted: {y_predicted.iloc[i]}")

# Evaluate inference predictions
print("\n========== INFERENCE EVALUATION ==========")

accuracy = accuracy_score(y_actual, y_predicted)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_actual,
        y_predicted,
        labels=["human", "synthetic"],
        zero_division=0
    )
)

print("\nInference testing completed successfully!")