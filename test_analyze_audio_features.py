import pandas as pd
import pytest

from analyze_audio_features import _feature_groups, analyze_features


def test_analyze_features_reports_class_coverage_and_creates_plots(tmp_path):
    labels_file = tmp_path / "labels.csv"
    labels_file.write_text(
        "filename,label\n"
        "human_1.wav,human\n"
        "human_2.wav,human\n"
        "synthetic_1.wav,synthetic\n"
        "synthetic_2.wav,synthetic\n",
        encoding="utf-8",
    )
    features_file = tmp_path / "features.csv"
    pd.DataFrame([
        {
            "filename": "human_1.wav",
            "label": "human",
            "mfcc_01_mean": 1.0,
            "gtcc_01_mean": 2.0,
            "spectral_contrast_01_mean": 0.1,
            "spectral_centroid_mean": 1000.0,
            "f0_mean_hz": 120.0,
            "voiced_fraction": 0.8,
            "syllabic_rate_proxy": 3.0,
        },
        {
            "filename": "synthetic_1.wav",
            "label": "synthetic",
            "mfcc_01_mean": 3.0,
            "gtcc_01_mean": 4.0,
            "spectral_contrast_01_mean": 0.2,
            "spectral_centroid_mean": 1500.0,
            "f0_mean_hz": 180.0,
            "voiced_fraction": 0.9,
            "syllabic_rate_proxy": 4.0,
        },
    ]).to_csv(features_file, index=False)

    output_dir = tmp_path / "eda"
    summary = analyze_features(features_file, labels_file, output_dir)

    assert summary["labeled_rows"].tolist() == [2, 2]
    assert summary["feature_rows"].tolist() == [1, 1]
    assert summary["missing_feature_rows"].tolist() == [1, 1]
    assert (output_dir / "eda_class_distribution.csv").is_file()
    assert (output_dir / "eda_feature_summary.csv").is_file()
    assert (output_dir / "eda_class_distribution.png").is_file()
    assert (output_dir / "eda_feature_distributions.png").is_file()
    assert (output_dir / "eda_mfcc_distributions.png").is_file()
    assert (output_dir / "eda_gtcc_distributions.png").is_file()
    assert (output_dir / "eda_spectral_comparison.png").is_file()
    assert (output_dir / "eda_prosodic_comparison.png").is_file()
    assert (output_dir / "eda_correlation_matrix.csv").is_file()
    assert (output_dir / "eda_correlation_heatmap.png").is_file()
    assert (output_dir / "eda_missing_values.csv").is_file()
    assert (output_dir / "eda_row_missing_values.csv").is_file()
    assert (output_dir / "eda_missing_values.png").is_file()
    assert (output_dir / "eda_outlier_summary.csv").is_file()
    assert (output_dir / "eda_outlier_summary.png").is_file()

    feature_summary = pd.read_csv(output_dir / "eda_feature_summary.csv")
    assert {"mean", "median", "std", "minimum", "maximum"}.issubset(
        feature_summary.columns
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_features(features_file, labels_file, output_dir)


def test_spectral_group_includes_zero_crossing_rate():
    groups = _feature_groups([
        "spectral_centroid_mean",
        "zero_crossing_rate_mean",
        "rms_mean",
        "f0_mean_hz",
    ])

    assert groups["spectral"] == [
        "spectral_centroid_mean",
        "zero_crossing_rate_mean",
        "rms_mean",
    ]
