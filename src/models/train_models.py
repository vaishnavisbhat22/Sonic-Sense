import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Expected: <project>/src/models/train_models.py

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "audio_features_v2.csv"

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMN = "label"
FILENAME_COLUMN = "filename"

# Human is the positive class for evaluation.
POSITIVE_LABEL = "human"

CLASS_LABELS = ["human", "synthetic"]


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset():

    print("\n========== LOADING DATASET ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    required_columns = [LABEL_COLUMN, FILENAME_COLUMN]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' is missing."
            )

    # Separate features and labels.
    feature_columns = [
        column for column in df.columns
        if column not in [LABEL_COLUMN, FILENAME_COLUMN]
    ]

    X = df[feature_columns].copy()
    y = df[LABEL_COLUMN].astype(str).str.strip().str.lower()

    # Check labels.
    allowed_labels = {"human", "synthetic"}

    if not set(y.unique()).issubset(allowed_labels):
        raise ValueError(
            f"Unexpected labels found: {y.unique()}"
        )

    # Ensure all features are numeric.
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any():
        raise ValueError(
            "Missing or non-numeric values found in feature columns."
        )

    if np.isinf(X.to_numpy()).any():
        raise ValueError(
            "Infinite values found in feature columns."
        )

    if X.empty:
        raise ValueError("No feature columns found.")

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nNumber of features:", len(feature_columns))
    print("Missing values:", X.isnull().sum().sum())
    print("Infinite values:", np.isinf(X.to_numpy()).sum())

    return df, X, y, feature_columns


# ============================================================
# 3. SPLIT DATASET
# ============================================================

def split_dataset(df, X, y):

    print("\n========== SPLITTING DATASET ==========")

    # First split: 70% training, 30% temporary.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Second split: 15% validation, 15% testing.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nValidation class distribution:")
    print(y_val.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    # Save split manifest for reproducibility.
    split_manifest = pd.DataFrame({
        "filename": df.loc[X.index, FILENAME_COLUMN],
        "label": y,
        "split": "unused"
    })

    split_manifest.loc[X_train.index, "split"] = "train"
    split_manifest.loc[X_val.index, "split"] = "validation"
    split_manifest.loc[X_test.index, "split"] = "test"

    split_manifest.to_csv(
        RESULT_DIR / "split_manifest.csv",
        index=False
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 4. DEFINE MODELS
# ============================================================

def create_models(y_train):

    print("\n========== DEFINING MODELS ==========")

    human_count = int((y_train == "human").sum())
    synthetic_count = int((y_train == "synthetic").sum())

    # XGBoost uses numeric labels:
    # 0 = synthetic
    # 1 = human
    #
    # Therefore, negative/positive ratio is:
    # synthetic_count / human_count

    human_weight = synthetic_count / human_count

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE
            ))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                class_weight="balanced",
                max_iter=10000,
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=2.0,
            scale_pos_weight=human_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }

    print("Models defined:")
    for name in models:
        print("-", name)

    return models


# ============================================================
# 5. GET HUMAN-CLASS SCORES
# ============================================================

def get_human_scores(model, X_val, model_name):

    """
    Returns scores where higher values indicate a greater
    likelihood of the HUMAN class.

    This is important for calculating ROC-AUC correctly.
    """

    # XGBoost was trained using:
    # 0 = synthetic
    # 1 = human

    if model_name == "XGBoost":

        probabilities = model.predict_proba(X_val)

        # Probability of class 1 = human.
        return probabilities[:, 1]

    # Get the actual classifier inside a Pipeline.
    if isinstance(model, Pipeline):
        classifier = model.named_steps["classifier"]
    else:
        classifier = model

    classes = list(classifier.classes_)

    human_index = classes.index("human")

    # Models supporting predict_proba.
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_val)

        return probabilities[:, human_index]

    # LinearSVC does not provide predict_proba.
    # Its decision_function is positive for classes_[1].
    decision_scores = model.decision_function(X_val)

    if human_index == 1:
        return decision_scores

    return -decision_scores


# ============================================================
# 6. TRAIN AND EVALUATE MODELS ON VALIDATION DATA
# ============================================================

def train_and_evaluate(models, X_train, X_val, y_train, y_val):

    print("\n========== MODEL TRAINING ==========")

    validation_results = []
    validation_reports = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        start_time = time.time()

        # XGBoost requires numerical target labels.
        if name == "XGBoost":

            y_train_encoded = (
                y_train == "human"
            ).astype(int)

            model.fit(X_train, y_train_encoded)

            raw_predictions = model.predict(X_val)

            # Convert predictions back to original labels.
            predictions = pd.Series(
                raw_predictions,
                index=y_val.index
            ).round().astype(int).map({
                0: "synthetic",
                1: "human"
            })

        else:

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

        training_time = time.time() - start_time

        # Obtain correctly oriented human-class scores.
        scores = get_human_scores(
            model,
            X_val,
            name
        )

        # Validation metrics.
        accuracy = accuracy_score(y_val, predictions)

        precision = precision_score(
            y_val,
            predictions,
            pos_label="human",
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            pos_label="human",
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            pos_label="human",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        balanced_acc = balanced_accuracy_score(
            y_val,
            predictions
        )

        auc = roc_auc_score(
            (y_val == "human").astype(int),
            scores
        )

        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Human): {precision:.4f}")
        print(f"Recall (Human): {recall:.4f}")
        print(f"F1 (Human): {f1:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"ROC-AUC: {auc:.4f}")

        validation_results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision_Human": precision,
            "Recall_Human": recall,
            "F1_Human": f1,
            "Macro_F1": macro_f1,
            "Balanced_Accuracy": balanced_acc,
            "ROC_AUC": auc,
            "Training_Time_Seconds": training_time
        })

        # Classification report.
        report = classification_report(
            y_val,
            predictions,
            labels=CLASS_LABELS,
            target_names=CLASS_LABELS,
            zero_division=0
        )

        validation_reports[name] = report

        # Save confusion matrix.
        cm = confusion_matrix(
            y_val,
            predictions,
            labels=CLASS_LABELS
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual_Human", "Actual_Synthetic"],
            columns=["Predicted_Human", "Predicted_Synthetic"]
        )

        safe_name = name.lower().replace(" ", "_")

        cm_df.to_csv(
            RESULT_DIR / f"{safe_name}_validation_confusion_matrix.csv"
        )

        # Save trained model.
        joblib.dump(
            model,
            MODEL_DIR / f"{safe_name}.joblib"
        )

        print(f"{name} model saved.")

    # Save comparison results.
    results_df = pd.DataFrame(validation_results)

    results_df = results_df.sort_values(
        by="Macro_F1",
        ascending=False
    )

    results_df.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False
    )

    # Save classification reports.
    with open(
        RESULT_DIR / "classification_reports.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for name, report in validation_reports.items():

            file.write(f"\n{'=' * 60}\n")
            file.write(f"{name}\n")
            file.write(f"{'=' * 60}\n")
            file.write(report)
            file.write("\n")

    print("\n========== VALIDATION MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    return results_df, models


# ============================================================
# 7. SELECT BEST MODEL AND EVALUATE ON TEST DATA
# ============================================================

def evaluate_best_model(
    results_df,
    models,
    X_test,
    y_test,
    feature_columns
):

    print("\n========== BEST MODEL SELECTION ==========")

    # Select based on validation Macro F1.
    best_model_name = results_df.iloc[0]["Model"]

    best_model = models[best_model_name]

    print("Selected model:", best_model_name)

    # Test evaluation happens only after model selection.
    if best_model_name == "XGBoost":

        raw_predictions = best_model.predict(X_test)

        predictions = pd.Series(
            raw_predictions,
            index=y_test.index
        ).round().astype(int).map({
            0: "synthetic",
            1: "human"
        })

    else:

        predictions = best_model.predict(X_test)

    scores = get_human_scores(
        best_model,
        X_test,
        best_model_name
    )

    # Test metrics.
    test_metrics = {
        "Best_Model": best_model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision_Human": precision_score(
            y_test,
            predictions,
            pos_label="human",
            zero_division=0
        ),
        "Recall_Human": recall_score(
            y_test,
            predictions,
            pos_label="human",
            zero_division=0
        ),
        "F1_Human": f1_score(
            y_test,
            predictions,
            pos_label="human",
            zero_division=0
        ),
        "Macro_F1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_test,
            predictions
        ),
        "ROC_AUC": roc_auc_score(
            (y_test == "human").astype(int),
            scores
        )
    }

    print("\n========== FINAL TEST RESULTS ==========")

    for metric, value in test_metrics.items():

        if isinstance(value, (float, np.floating)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Save test metrics.
    with open(
        RESULT_DIR / "best_model_test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            test_metrics,
            file,
            indent=4
        )

    # Save test classification report.
    test_report = classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_LABELS,
        zero_division=0
    )

    with open(
        RESULT_DIR / "best_model_test_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(test_report)

    print("\nTest Classification Report:")
    print(test_report)

    # Save test confusion matrix.
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_LABELS
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Human", "Actual_Synthetic"],
        columns=["Predicted_Human", "Predicted_Synthetic"]
    )

    cm_df.to_csv(
        RESULT_DIR / "best_model_test_confusion_matrix.csv"
    )

    # Save metadata for Member 4.
    metadata = {
        "best_model": best_model_name,
        "model_file": (
            best_model_name.lower().replace(" ", "_") + ".joblib"
        ),
        "dataset": str(DATA_PATH),
        "number_of_features": len(feature_columns),
        "feature_columns": feature_columns,
        "class_labels": CLASS_LABELS,
        "positive_class": "human",
        "xgboost_class_mapping": {
            "0": "synthetic",
            "1": "human"
        },
        "test_metrics": test_metrics,
        "split": {
            "training": 0.70,
            "validation": 0.15,
            "testing": 0.15,
            "random_state": RANDOM_STATE
        },
        "split_limitation": (
            "Stratified random split used. Speaker/source identifiers "
            "were unavailable, so speaker-independent or source-independent "
            "evaluation cannot be claimed."
        )
    }

    with open(
        RESULT_DIR / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nBest model metadata saved.")

    return best_model_name


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("       SONICSENSE - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load dataset.
    df, X, y, feature_columns = load_dataset()

    # Split dataset.
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(df, X, y)

    # Create models.
    models = create_models(y_train)

    # Train and compare.
    results_df, trained_models = train_and_evaluate(
        models,
        X_train,
        X_val,
        y_train,
        y_val
    )

    # Select best model and evaluate on test data.
    best_model_name = evaluate_best_model(
        results_df,
        trained_models,
        X_test,
        y_test,
        feature_columns
    )

    print("\n")
    print("=" * 60)
    print("       SONICSENSE TRAINING COMPLETED")
    print("=" * 60)

    print("\nBest model:", best_model_name)

    print("\nGenerated folders:")
    print("Models:", MODEL_DIR)
    print("Results:", RESULT_DIR)

    print("\nImportant output files:")
    print("- results/model_comparison.csv")
    print("- results/best_model_test_metrics.json")
    print("- results/best_model_test_report.txt")
    print("- results/best_model_test_confusion_matrix.csv")
    print("- results/model_metadata.json")
    print("- results/split_manifest.csv")


if __name__ == "__main__":
    main()