"""Generate class-wise statistics, plots, missing-value, and outlier reports."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract_audio_features import DEFAULT_LABELS_FILE, DEFAULT_OUTPUT_FILE, load_labels


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EDA_DIR = PROJECT_ROOT / "eda" / "v2"
CLASSES = ("human", "synthetic")
PLOT_FEATURES = (
    "mfcc_01_mean",
    "gtcc_01_mean",
    "spectral_contrast_01_mean",
    "spectral_centroid_mean",
    "f0_mean_hz",
    "syllabic_rate_proxy",
)
OUTPUT_FILENAMES = (
    "eda_class_distribution.csv",
    "eda_class_distribution.png",
    "eda_feature_summary.csv",
    "eda_feature_distributions.png",
    "eda_mfcc_distributions.png",
    "eda_gtcc_distributions.png",
    "eda_spectral_comparison.png",
    "eda_prosodic_comparison.png",
    "eda_correlation_matrix.csv",
    "eda_correlation_heatmap.png",
    "eda_missing_values.csv",
    "eda_row_missing_values.csv",
    "eda_missing_values.png",
    "eda_outlier_summary.csv",
    "eda_outlier_summary.png",
)


def _feature_groups(columns):
    columns = list(columns)
    return {
        "mfcc": [name for name in columns if name.startswith("mfcc_")],
        "gtcc": [name for name in columns if name.startswith("gtcc_")],
        "spectral": [
            name for name in columns
            if name.startswith("spectral_")
            or name.startswith("zero_crossing_rate_")
            or name.startswith("rms_")
        ],
        "prosodic": [
            name for name in columns
            if name.startswith("f0_")
            or name in {"voiced_fraction", "syllabic_rate_proxy"}
        ],
    }


def _check_output_paths(output_dir):
    existing = [
        output_dir / name
        for name in OUTPUT_FILENAMES
        if (output_dir / name).exists()
    ]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing EDA output(s): "
            f"{paths}. Choose a new --output-dir to preserve these files."
        )


def _load_numeric_features(data, feature_columns):
    numeric_columns = {}
    missing_rows = []
    missing_summary = []

    for column in feature_columns:
        source = data[column]
        converted = pd.to_numeric(source, errors="coerce").astype(np.float32)
        parse_error = source.notna() & converted.isna()
        nonfinite = converted.notna() & ~np.isfinite(converted)
        values = converted.mask(nonfinite)
        missing = values.isna()
        numeric_columns[column] = values

        missing_summary.append({
            "feature": column,
            "row_count": len(data),
            "missing_count": int(missing.sum()),
            "missing_percentage": (
                float(missing.mean() * 100.0) if len(data) else 0.0
            ),
            "parse_error_count": int(parse_error.sum()),
            "nonfinite_count": int(nonfinite.sum()),
        })

    numeric = pd.DataFrame(numeric_columns, index=data.index)
    row_missing_counts = numeric.isna().sum(axis=1)
    for index in np.flatnonzero(row_missing_counts.to_numpy()):
        missing_rows.append({
            "filename": data.at[index, "filename"],
            "label": data.at[index, "label"],
            "missing_feature_count": int(row_missing_counts.iloc[index]),
            "missing_features": ", ".join(
                numeric.columns[numeric.iloc[index].isna()].tolist()
            ),
        })

    return numeric, pd.DataFrame(missing_summary), pd.DataFrame(
        missing_rows,
        columns=[
            "filename",
            "label",
            "missing_feature_count",
            "missing_features",
        ],
    )


def _save_class_distribution(class_summary, output_dir):
    class_summary.to_csv(output_dir / "eda_class_distribution.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(CLASSES))
    width = 0.36
    axis.bar(
        positions - width / 2,
        class_summary["labeled_rows"],
        width,
        label="Label CSV",
    )
    axis.bar(
        positions + width / 2,
        class_summary["feature_rows"],
        width,
        label="Feature CSV",
    )
    axis.set_xticks(positions, ["Human", "Synthetic"])
    axis.set_ylabel("Number of recordings")
    axis.set_title("Human vs Synthetic Class Distribution and Coverage")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "eda_class_distribution.png", dpi=160)
    plt.close(figure)


def _save_feature_summary(data, numeric, output_dir):
    summary_rows = []
    for label in ("all", *CLASSES):
        selected = np.ones(len(data), dtype=bool) if label == "all" else (
            data["label"].to_numpy() == label
        )
        for feature in numeric.columns:
            values = numeric.loc[selected, feature].dropna()
            summary_rows.append({
                "label": label,
                "feature": feature,
                "count": int(values.count()),
                "mean": float(values.mean()) if not values.empty else np.nan,
                "median": float(values.median()) if not values.empty else np.nan,
                "std": float(values.std()) if not values.empty else np.nan,
                "minimum": float(values.min()) if not values.empty else np.nan,
                "maximum": float(values.max()) if not values.empty else np.nan,
            })
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "eda_feature_summary.csv",
        index=False,
    )


def _save_distribution_plot(data, numeric, columns, path, title):
    if not columns:
        return False

    plot_columns = sorted(columns)
    columns_per_row = 4
    row_count = (len(plot_columns) + columns_per_row - 1) // columns_per_row
    figure, axes = plt.subplots(
        row_count,
        columns_per_row,
        figsize=(16, max(3.2 * row_count, 4)),
        squeeze=False,
    )
    colors = {"human": "#3978b8", "synthetic": "#d56b3f"}

    for axis, feature in zip(axes.flat, plot_columns):
        all_values = numeric[feature].dropna().to_numpy(dtype=np.float32)
        if all_values.size == 0:
            axis.set_title(feature.replace("_", " "))
            axis.text(0.5, 0.5, "No finite values", ha="center", va="center")
            axis.set_xlabel(feature.replace("_", " "))
            axis.set_ylabel("Density")
            continue

        lower, upper = np.quantile(all_values, [0.01, 0.99])
        if lower == upper:
            lower, upper = float(all_values.min()), float(all_values.max())
        if lower == upper:
            upper = lower + 1.0
        class_values = {
            label: numeric.loc[data["label"] == label, feature].dropna()
            for label in CLASSES
        }
        if any(
            not values.empty
            and not ((values >= lower) & (values <= upper)).any()
            for values in class_values.values()
        ):
            lower, upper = float(all_values.min()), float(all_values.max())
            if lower == upper:
                upper = lower + 1.0

        for label in CLASSES:
            values = class_values[label]
            if not values.empty:
                axis.hist(
                    values.to_numpy(dtype=np.float32),
                    bins=35,
                    range=(lower, upper),
                    density=True,
                    alpha=0.5,
                    color=colors[label],
                    label=label.title(),
                )
        axis.set_title(feature.replace("_", " "))
        axis.set_xlabel(feature.replace("_", " "))
        axis.set_ylabel("Density")
        axis.legend(fontsize=8)

    for axis in axes.flat[len(plot_columns):]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _save_group_comparison(data, numeric, columns, path, title):
    if not columns:
        return False

    plot_columns = sorted(columns)
    columns_per_row = 4
    row_count = (len(plot_columns) + columns_per_row - 1) // columns_per_row
    figure, axes = plt.subplots(
        row_count,
        columns_per_row,
        figsize=(16, max(3.2 * row_count, 4)),
        squeeze=False,
    )

    for axis, feature in zip(axes.flat, plot_columns):
        samples = [
            numeric.loc[data["label"] == label, feature].dropna().to_numpy(
                dtype=np.float32
            )
            for label in CLASSES
        ]
        if not any(values.size for values in samples):
            axis.set_title(feature.replace("_", " "))
            axis.text(0.5, 0.5, "No finite values", ha="center", va="center")
        else:
            axis.boxplot(
                samples,
                showfliers=False,
            )
            axis.set_xticks([1, 2], ["Human", "Synthetic"])
        axis.set_title(feature.replace("_", " "))
        axis.set_xlabel("Class")
        axis.set_ylabel("Feature value")
        axis.tick_params(axis="x", labelrotation=15)

    for axis in axes.flat[len(plot_columns):]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _save_correlation_heatmap(numeric, output_dir):
    correlation = numeric.corr(method="pearson", min_periods=2)
    correlation.to_csv(output_dir / "eda_correlation_matrix.csv")
    values = correlation.to_numpy(dtype=np.float32)
    masked = np.ma.masked_invalid(values)
    size = max(16, min(32, len(correlation.columns) * 0.22))
    figure, axis = plt.subplots(figsize=(size, size * 0.9))
    image = axis.imshow(
        masked,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    ticks = np.arange(len(correlation.columns))
    labels = [column.replace("_", " ") for column in correlation.columns]
    axis.set_xticks(ticks, labels, rotation=90, fontsize=5)
    axis.set_yticks(ticks, labels, fontsize=5)
    axis.set_xlabel("Feature")
    axis.set_ylabel("Feature")
    axis.set_title("Feature Correlation Heatmap (Pearson)")
    figure.colorbar(image, ax=axis, label="Correlation coefficient", shrink=0.75)
    figure.tight_layout()
    figure.savefig(output_dir / "eda_correlation_heatmap.png", dpi=180)
    plt.close(figure)


def _save_missing_analysis(missing_summary, row_missing, output_dir):
    missing_summary.to_csv(output_dir / "eda_missing_values.csv", index=False)
    row_missing.to_csv(output_dir / "eda_row_missing_values.csv", index=False)

    ordered = missing_summary.sort_values(
        "missing_percentage",
        ascending=True,
    )
    figure, axis = plt.subplots(
        figsize=(12, max(5, len(ordered) * 0.12))
    )
    axis.barh(
        ordered["feature"],
        ordered["missing_percentage"],
        color="#805ad5",
    )
    axis.set_xlabel("Missing or non-finite values (%)")
    axis.set_ylabel("Feature")
    axis.set_title("Missing Feature Values by Column")
    figure.tight_layout()
    figure.savefig(output_dir / "eda_missing_values.png", dpi=160)
    plt.close(figure)


def _save_outlier_analysis(data, numeric, output_dir):
    rows = []
    for label in CLASSES:
        class_mask = data["label"].to_numpy() == label
        for feature in numeric.columns:
            values = numeric.loc[class_mask, feature].dropna()
            if values.empty:
                rows.append({
                    "label": label,
                    "feature": feature,
                    "count": 0,
                    "q1": np.nan,
                    "q3": np.nan,
                    "iqr": np.nan,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                })
                continue

            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((values < lower) | (values > upper)).sum())
            rows.append({
                "label": label,
                "feature": feature,
                "count": int(values.count()),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": outlier_count,
                "outlier_percentage": (
                    float(outlier_count / values.count() * 100.0)
                    if values.count()
                    else 0.0
                ),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "eda_outlier_summary.csv", index=False)
    top = summary.sort_values(
        "outlier_percentage",
        ascending=False,
    ).groupby("label", sort=False).head(20)
    figure, axis = plt.subplots(figsize=(12, 8))
    features = list(dict.fromkeys(top["feature"].tolist()))
    positions = np.arange(len(features))
    width = 0.36
    for offset, label, color in (
        (-width / 2, "human", "#3978b8"),
        (width / 2, "synthetic", "#d56b3f"),
    ):
        class_rows = top.loc[top["label"] == label].set_index("feature")
        values = [
            float(class_rows.at[name, "outlier_percentage"])
            if name in class_rows.index
            else 0.0
            for name in features
        ]
        axis.bar(positions + offset, values, width, label=label.title(), color=color)
    axis.set_xticks(positions, [name.replace("_", " ") for name in features])
    axis.tick_params(axis="x", labelrotation=75)
    axis.set_xlabel("Feature (top 20 per class)")
    axis.set_ylabel("Outliers (%)")
    axis.set_title("IQR Outlier Rates by Class and Feature")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "eda_outlier_summary.png", dpi=160)
    plt.close(figure)


def analyze_features(features_file, labels_file, output_dir):
    features_file = Path(features_file)
    labels_file = Path(labels_file)
    output_dir = Path(output_dir)
    if not features_file.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {features_file}")

    _check_output_paths(output_dir)
    labels = load_labels(labels_file)

    # Read only identifiers and feature columns, then downcast numeric values.
    header = pd.read_csv(features_file, nrows=0).columns.tolist()
    required = {"filename", "label"}
    if not required.issubset(header):
        raise ValueError(f"{features_file} must include filename and label columns.")
    feature_columns = [column for column in header if column not in required]
    if not feature_columns:
        raise ValueError(f"No numeric feature columns found in {features_file}.")

    data = pd.read_csv(
        features_file,
        usecols=["filename", "label", *feature_columns],
        low_memory=False,
    )
    if data["filename"].isna().any() or data["filename"].duplicated().any():
        raise ValueError("Feature filenames must be present and unique.")

    data["label"] = data["label"].astype("string").str.strip().str.lower()
    invalid_labels = sorted(set(data["label"].dropna()) - set(CLASSES))
    if data["label"].isna().any() or invalid_labels:
        raise ValueError(f"Invalid or missing feature labels: {invalid_labels}")

    source_labels = data["filename"].map(labels)
    if source_labels.isna().any():
        raise ValueError(
            f"{int(source_labels.isna().sum())} feature row(s) have no matching source label."
        )
    mismatched = source_labels != data["label"]
    if mismatched.any():
        raise ValueError(
            f"{int(mismatched.sum())} feature row(s) have labels that disagree "
            "with the label CSV."
        )

    numeric, missing_summary, row_missing = _load_numeric_features(
        data,
        feature_columns,
    )
    groups = _feature_groups(numeric.columns)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_counts = pd.Series(labels, dtype="string").value_counts()
    feature_counts = data["label"].value_counts()
    class_summary = pd.DataFrame([
        {
            "label": label,
            "labeled_rows": int(label_counts.get(label, 0)),
            "feature_rows": int(feature_counts.get(label, 0)),
            "missing_feature_rows": int(
                label_counts.get(label, 0) - feature_counts.get(label, 0)
            ),
        }
        for label in CLASSES
    ])
    _save_class_distribution(class_summary, output_dir)
    _save_feature_summary(data, numeric, output_dir)

    _save_distribution_plot(
        data,
        numeric,
        [name for name in PLOT_FEATURES if name in numeric.columns],
        output_dir / "eda_feature_distributions.png",
        "Selected Feature Distributions by Class",
    )
    _save_distribution_plot(
        data,
        numeric,
        groups["mfcc"],
        output_dir / "eda_mfcc_distributions.png",
        "MFCC Distributions by Class (1st–99th Percentile Range)",
    )
    _save_distribution_plot(
        data,
        numeric,
        groups["gtcc"],
        output_dir / "eda_gtcc_distributions.png",
        "GTCC Distributions by Class (1st–99th Percentile Range)",
    )
    _save_group_comparison(
        data,
        numeric,
        groups["spectral"],
        output_dir / "eda_spectral_comparison.png",
        "Spectral Feature Comparison: Human vs Synthetic",
    )
    _save_group_comparison(
        data,
        numeric,
        groups["prosodic"],
        output_dir / "eda_prosodic_comparison.png",
        "Prosodic Feature Comparison: Human vs Synthetic",
    )
    _save_correlation_heatmap(numeric, output_dir)
    _save_missing_analysis(missing_summary, row_missing, output_dir)
    _save_outlier_analysis(data, numeric, output_dir)

    print(f"Labeled recordings: {len(labels):,}")
    print(f"Feature rows: {len(data):,}")
    print(f"Numeric features: {len(feature_columns):,}")
    print(f"Rows with missing/non-finite features: {len(row_missing):,}")
    print(f"EDA outputs: {output_dir}")
    if len(data) < len(labels):
        print(
            "Feature coverage is incomplete. Missing labeled recordings are "
            "reported in eda_class_distribution.csv."
        )
    return class_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate class-wise statistics and EDA plots for audio features."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EDA_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_features(args.features, args.labels, args.output_dir)


if __name__ == "__main__":
    main()
