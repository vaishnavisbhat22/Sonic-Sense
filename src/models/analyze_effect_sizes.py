import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/audio_features_v2.csv")

human = df[df["label"] == "human"]
synthetic = df[df["label"] == "synthetic"]

features = df.drop(columns=["filename", "label"]).columns

results = []

for feature in features:

    h = human[feature]
    s = synthetic[feature]

    pooled_std = np.sqrt(
        (
            (len(h) - 1) * h.var()
            + (len(s) - 1) * s.var()
        )
        / (len(h) + len(s) - 2)
    )

    if pooled_std == 0:
        continue

    cohens_d = (s.mean() - h.mean()) / pooled_std

    results.append({
        "feature": feature,
        "human_mean": h.mean(),
        "synthetic_mean": s.mean(),
        "cohens_d": cohens_d,
        "absolute_effect": abs(cohens_d)
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="absolute_effect",
    ascending=False
)

print("\n========== TOP 20 FEATURES BY EFFECT SIZE ==========")

print(
    results_df.head(20).to_string(index=False)
)

print("\n========== EFFECT SIZE ANALYSIS COMPLETED ==========")

results_df.to_csv(
    "results/feature_effect_sizes.csv",
    index=False
)

print("\nResults saved to results/feature_effect_sizes.csv")