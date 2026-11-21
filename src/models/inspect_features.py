import pandas as pd
import numpy as np

DATA_PATH = "data/processed/audio_features_v2.csv"

df = pd.read_csv(DATA_PATH)

features = df.drop(columns=["filename", "label"])

print("\n========== FEATURE INSPECTION ==========")

# 1. Datatypes
print("\n1. FEATURE DATATYPES")
print(features.dtypes.value_counts())

# 2. Constant features
print("\n2. CONSTANT FEATURES")

constant = features.columns[
    features.nunique() <= 1
].tolist()

print(constant)
print("Total:", len(constant))

# 3. Feature statistics
print("\n3. FEATURE STATISTICS")
print(features.describe().T)

# 4. Near-zero F0 variation
print("\n4. NEAR-ZERO F0 STANDARD DEVIATION")

if "f0_std_hz" in features.columns:
    count = (features["f0_std_hz"] < 1e-6).sum()
    print("Near-zero recordings:", count)

# 5. Highly correlated features
print("\n5. HIGHLY CORRELATED FEATURE PAIRS")

corr = features.corr().abs()

upper_indices = np.triu_indices_from(corr, k=1)

correlation_values = corr.values[upper_indices]

pairs = []

for index, value in enumerate(correlation_values):

    if value > 0.95:

        row = upper_indices[0][index]
        col = upper_indices[1][index]

        pairs.append((
            corr.index[row],
            corr.columns[col],
            round(value, 4)
        ))

print("Pairs with correlation > 0.95:", len(pairs))

for pair in pairs[:30]:
    print(pair)

print("\n========== INSPECTION COMPLETED ==========")