# Member 2 to Member 3: Feature Dataset Handover

## 1. Project objective

SonicSense is a machine-learning project that studies whether an audio
recording contains **Human Speech** or **AI-Generated (Synthetic) Speech**.
The model receives numeric audio features and predicts one of these two
classes.

## 2. Dataset and final feature CSV

- **Input WAV folder:** `C:\SonicSense\audio_dataset`
- **Label file:** `C:\Sonic-Sense-main\audio_labels.csv`
- **Intended final feature file:** `C:\Sonic-Sense-main\audio_features_v2.csv`

**Handover status:** `audio_features_v2.csv` is ready for model training. It
contains 47,400 recordings and 134 numeric features. The verified class counts
are 10,700 `human` and 36,700 `synthetic`. Filenames and labels match
`audio_labels.csv`; there are no duplicate filenames, missing or non-finite
feature values, or recorded extraction failures. These checks do not represent
model performance or accuracy.

The existing `audio_features.csv` is an older, partial feature table. It
should not be substituted for the final expanded feature file without
confirming its schema and suitability for the agreed training workflow.

## 3. Meaning of the label column

The `label` column is the target that the model learns to predict:

- `human`: the recording is labeled as human speech.
- `synthetic`: the recording is labeled as AI-generated or synthetic speech.

Use the labels from the supplied label file and verify that each feature row's
filename maps to the same label. Labels are categories, not numeric feature
measurements.

## 4. Feature groups

The expanded feature extractor is designed to produce the following groups.
Most frame-based measurements are summarized by their mean and standard
deviation for each recording.

- **MFCC (Mel-Frequency Cepstral Coefficients):** Compact descriptors of the
  speech spectrum using Mel-spaced frequency bands. They summarize aspects of
  spectral-envelope shape.
- **GTCC (Gammatone Frequency Cepstral Coefficients):** Cepstral descriptors
  derived from gammatone-filter energies. They provide a complementary
  auditory-spaced view of spectral structure.
- **Spectral features:** Centroid, bandwidth, rolloff, flatness, contrast,
  zero-crossing rate, and RMS energy describe spectral balance, spread,
  high-frequency content, peak-to-valley contrast, waveform sign changes, and
  signal level.
- **Prosodic and timing-related features:** F0 describes estimated pitch;
  `voiced_fraction` is an energy-threshold activity estimate; and
  `syllabic_rate_proxy` is detected acoustic onsets per second. The latter two
  are proxies: they are not a true voicing detector or transcript-based
  syllable/speaking-rate measurements.

These features may describe class-related patterns, but none proves whether a
recording is synthetic. Model performance must be determined through
evaluation.

## 5. Load the final CSV with Pandas

The completed feature table can be loaded with Pandas:

```python
from pathlib import Path
import pandas as pd

feature_path = Path(r"C:\Sonic-Sense-main\audio_features_v2.csv")
data = pd.read_csv(feature_path)

print(data.shape)
print(data.columns.tolist())
print(data["label"].value_counts(dropna=False))
```

## 6. Separate X (features) and y (target)

Keep identifier columns such as `filename` out of the model inputs. The target
is `label`; all other columns should be numeric feature columns after
validation.

```python
target = "label"
identifier = "filename"

X = data.drop(columns=[target, identifier])
y = data[target]

assert X.select_dtypes(exclude="number").empty
assert y.notna().all()
assert set(y.unique()) <= {"human", "synthetic"}
```

Before fitting, confirm there are no missing, infinite, or non-numeric feature
values. Decide explicitly how to handle any invalid rows; do not silently
convert or discard them.

## 7. Why train-test splitting is necessary

The model must be evaluated on examples it did not use for fitting. A
train-test split reserves part of the labeled data for this evaluation. This
helps estimate performance on unseen examples and reveals when a model simply
memorizes its training data.

Use a stratified split when appropriate so both classes are represented in
each partition. If multiple segments can come from the same original
recording or speaker, consider a group-based split so related samples do not
appear in both training and test sets. The split design should match the
intended use of the detector.

## 8. Why feature scaling may be required

Features use different units and ranges: for example, F0 is measured in hertz,
while RMS and voiced fraction are much smaller values. Algorithms based on
distances, dot products, or gradient optimization (such as k-nearest
neighbors, support-vector machines, and logistic regression) can be affected
by these scale differences.

Scaling is often useful for those algorithms. Tree-based models generally do
not require feature scaling. If scaling is used, fit the scaler only on the
training data—preferably in a pipeline—and apply that fitted scaler to the
test data. This prevents information from the test set leaking into training.

## 9. Suggested evaluation metrics

Evaluate predictions on held-out data and report metrics for both classes,
with particular attention to the project's intended use.

- **Accuracy:** Fraction of all predictions that are correct. It can hide poor
  minority-class performance when labels are imbalanced.
- **Precision:** Of the examples predicted as a class, the fraction that
  actually belong to that class.
- **Recall:** Of the actual examples in a class, the fraction correctly found.
- **F1-score:** Harmonic mean of precision and recall for a class.
- **Confusion matrix:** Counts correct and incorrect predictions by actual
  class and predicted class, helping show which errors occur.

Report the class order and the positive class used for any binary metric.
Metrics should be calculated on the held-out test data, not the training data.
Do not claim results until the model has actually been trained and evaluated.

## 10. Checks before model training

1. Confirm extraction has finished and the final CSV exists.
2. Verify the row count and filename coverage against the intended WAV/label
   set; investigate missing files and extraction errors.
3. Check that filenames are present and unique, labels are valid, and each
   filename's label matches the label file.
4. Confirm the feature columns match the current extractor schema and exclude
   identifiers and target labels from `X`.
5. Check every feature for numeric type, missing values, and non-finite
   values; document any cleaning decision.
6. Inspect class counts and imbalance before selecting metrics or training
   options.
7. Check for duplicate or closely related recordings and choose a split that
   avoids leakage where possible.
8. Fit any preprocessing, imputation, or scaling only on the training data.
9. Keep the test data untouched until final evaluation.
10. Record the split method, random seed, feature set, model configuration,
    and evaluation results so the experiment can be reproduced.

## Current handover decision

The feature dataset is ready for Member 3. Use a stratified train-test split
so both classes are represented in training and testing. The class distribution
is imbalanced, so report metrics for each class, including precision, recall,
and F1-score, along with a confusion matrix and balanced accuracy. Consider a
group-aware split if multiple recordings or segments share a speaker or source
recording, to reduce leakage between training and test sets. The feature and
EDA checks do not establish detector accuracy; that must be measured on
held-out data.
