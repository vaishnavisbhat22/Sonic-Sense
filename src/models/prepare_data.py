import pandas as pd
import numpy as np
from pathlib import Path

# Dataset path
DATA_PATH = Path("data/processed/audio_features_v2.csv")


def validate_dataset():

    print("\n========== SONICSENSE DATA VALIDATION ==========\n")

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # Basic information
    print("1. DATASET DIMENSIONS")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Column information
    print("\n2. COLUMN INFORMATION")
    print(f"Filename column exists: {'filename' in df.columns}")
    print(f"Label column exists: {'label' in df.columns}")

    feature_columns = [
        col for col in df.columns
        if col not in ["filename", "label"]
    ]

    print(f"Numerical feature columns: {len(feature_columns)}")

    # Label distribution
    print("\n3. CLASS DISTRIBUTION")
    print(df["label"].value_counts(dropna=False))

    print("\nClass percentages:")
    print(df["label"].value_counts(normalize=True) * 100)

    # Missing values
    print("\n4. MISSING VALUES")
    print(f"Total missing values: {df.isnull().sum().sum()}")

    # Duplicate checks
    print("\n5. DUPLICATE CHECKS")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    if "filename" in df.columns:
        print(f"Duplicate filenames: {df['filename'].duplicated().sum()}")

    # Numeric validation
    print("\n6. NUMERICAL VALIDATION")

    numeric_df = df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    print(f"Non-numeric feature values: {numeric_df.isnull().sum().sum()}")
    print(f"Infinite values: {np.isinf(numeric_df).sum().sum()}")

    # Filename inspection
    print("\n7. FILENAME SAMPLES")
    if "filename" in df.columns:
        print(df["filename"].head(10).to_string(index=False))

    print("\n========== VALIDATION COMPLETED ==========\n")


if __name__ == "__main__":
    validate_dataset()