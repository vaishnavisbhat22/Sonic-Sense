import pandas as pd

df = pd.read_csv("data/processed/audio_features_v2.csv")

# Separate classes
human = df[df["label"] == "human"]
synthetic = df[df["label"] == "synthetic"]

print("\n========== CLASS-WISE FEATURE ANALYSIS ==========")

print("\n1. CLASS COUNTS")
print(df["label"].value_counts())

# Selected features from different feature groups
selected_features = [
    "mfcc_01_mean",
    "mfcc_02_mean",
    "gtcc_01_mean",
    "spectral_centroid_mean",
    "spectral_rolloff_mean",
    "zero_crossing_rate_mean",
    "rms_mean",
    "f0_mean_hz",
    "f0_std_hz",
    "voiced_fraction",
    "syllabic_rate_proxy"
]

print("\n2. HUMAN VS SYNTHETIC FEATURE MEANS")

for feature in selected_features:

    human_mean = human[feature].mean()
    synthetic_mean = synthetic[feature].mean()

    difference = abs(human_mean - synthetic_mean)

    print(f"\nFeature: {feature}")
    print(f"Human mean: {human_mean:.4f}")
    print(f"Synthetic mean: {synthetic_mean:.4f}")
    print(f"Absolute difference: {difference:.4f}")

print("\n========== ANALYSIS COMPLETED ==========")