# SonicSense Exploratory Data Analysis Methodology

## Introduction

Exploratory Data Analysis (EDA) is used to inspect and summarize the extracted
audio features before model training. For SonicSense, EDA helps check class
coverage, understand feature distributions, identify missing or unusual
values, and see how features relate to one another.

EDA describes the dataset being examined. It does **not** prove that a
synthetic-speech detector is accurate. Detection performance must be measured
separately using an appropriate evaluation dataset and methodology.

## 1. Class distribution analysis: Human vs Synthetic

**Purpose:** Understand how many recordings belong to each class and whether
the feature dataset represents the labeled recordings.

**Method used:** Count the Human and Synthetic labels in the label file and in
the feature dataset. Compare those counts to identify class imbalance or
recordings that have labels but no extracted feature row.

**What the results can tell us:** The counts show whether one class is more
common and whether feature extraction has comparable coverage for both
classes. If the classes are imbalanced, this is important context for later
training and evaluation. The counts do not indicate how well a classifier will
perform.

## 2. Statistical analysis

**Purpose:** Summarize the values and ranges of each extracted feature.

**Method used:** For each feature, calculate the number of usable values, mean,
median, standard deviation, minimum, and maximum. These summaries can be
calculated for the full dataset and separately for Human and Synthetic
recordings. Missing and non-finite values should not be treated as valid
measurements.

**What the results can tell us:**

- **Mean** describes the arithmetic average and can be affected by extreme
  values.
- **Median** describes the middle value and is less sensitive to extreme
  values.
- **Standard deviation** describes how widely values vary around their mean.
- **Minimum and maximum** show the observed range and can help flag unusual
  values for further inspection.

These are descriptive summaries. They do not establish that a feature
separates the two classes reliably.

## 3. MFCC distribution analysis

**Purpose:** Examine the distributions of Mel-Frequency Cepstral Coefficients
(MFCCs) and compare their values between the two labels.

**Method used:** Plot the per-recording MFCC summary features separately for
Human and Synthetic examples. Distribution plots use a restricted display
range for readability; the underlying summary statistics should retain all
valid values.

**What the results can tell us:** The plots can show whether the classes have
similar or different distributions for particular MFCC summaries, and whether
those features have a wide spread or extreme values. MFCC distribution
differences are observations about this dataset, not proof of authenticity or
generalization to other speakers and recording conditions.

## 4. GTCC distribution analysis

**Purpose:** Examine Gammatone Frequency Cepstral Coefficients (GTCCs), which
provide a cepstral representation based on gammatone-filter channel energies.

**Method used:** Plot the per-recording GTCC summary features for Human and
Synthetic recordings, using the same class-wise comparison approach as for
MFCCs.

**What the results can tell us:** GTCC plots can show the spread and class-wise
patterns of the gammatone-based features. Comparing MFCC and GTCC distributions
helps describe how two different frequency-scale representations behave in
this dataset. It does not show that either representation is inherently more
accurate.

## 5. Spectral feature comparison

**Purpose:** Compare frequency and signal-shape descriptors between Human and
Synthetic recordings.

**Method used:** Compare features such as spectral centroid, bandwidth,
rolloff, flatness, contrast, zero-crossing rate, and RMS energy by class.
Class-wise box plots can show medians, interquartile ranges, and the spread of
the values; plotted outlier points may be hidden for clarity and are assessed
separately in the outlier analysis.

**What the results can tell us:** The plots can reveal differences in central
frequency, spectral spread, high-frequency energy, peak-to-valley contrast,
signal noisiness, or energy levels. They can also show overlap between the
classes and variation within each class. These features are affected by speech
content, speakers, microphones, codecs, and recording conditions as well as
the method used to generate speech.

## 6. Prosodic feature comparison

**Purpose:** Examine pitch-related and timing/activity features between the
two classes.

**Method used:** Compare F0 summaries, the feature named `voiced_fraction`,
and `syllabic_rate_proxy` by class. In the current implementation, F0 is
estimated with the YIN algorithm; `voiced_fraction` is based on an energy
threshold; and the onset-rate proxy counts detected acoustic onsets per
second.

**What the results can tell us:** These comparisons can describe pitch level
and variation, the fraction of frames above the energy threshold, and the rate
of detected acoustic events. Important limitations:

- `voiced_fraction` is not a true voiced/unvoiced speech detector; it is an
  energy-based activity estimate.
- `syllabic_rate_proxy` is not a transcript-based speaking rate or syllable
  count. An acoustic onset is not necessarily a syllable.
- Pitch and timing estimates can be affected by noise, silence, phonetic
  content, and estimator errors.

Therefore, these values should be interpreted as approximate acoustic
descriptors.

## 7. Correlation analysis using a heatmap

**Purpose:** Inspect pairwise linear relationships among numeric features.

**Method used:** Calculate Pearson correlation coefficients between feature
columns and display them as a heatmap. Coefficients range from -1 to +1:
positive values indicate that two features tend to increase together, negative
values indicate that one tends to decrease as the other increases, and values
near zero indicate little linear association in the observed data. Pairs with
insufficient valid data may have no displayed coefficient.

**What the results can tell us:** A heatmap can highlight groups of related
features or possible redundancy. Correlation does not establish causation,
does not capture every nonlinear relationship, and does not by itself identify
features that are useful for classification.

## 8. Missing value analysis

**Purpose:** Check whether feature values are absent or invalid and determine
how extensively the issue affects the dataset.

**Method used:** For each feature, count missing values and report the
percentage of rows affected. Values that cannot be parsed as numeric, as well
as non-finite values such as positive or negative infinity, are treated as
missing for this analysis. A row-level report can list the affected recording,
label, and missing features.

**What the results can tell us:** The summary shows which features or records
need attention and whether missingness is concentrated in one class. It does
not explain why values are missing. Investigate the source before deciding
whether to re-extract, exclude, or impute values; any such decision should be
documented.

## 9. Outlier detection using the IQR method

**Purpose:** Identify feature values that are unusually far from the central
range of their class.

**Method used:** For each feature within each class, calculate the first
quartile (Q1), third quartile (Q3), and interquartile range:

```text
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
```

Values below the lower bound or above the upper bound are counted as IQR
outliers. The analysis summarizes the number and percentage of outliers; it
does not automatically remove them.

**What the results can tell us:** Outlier summaries can identify features or
classes with unusually broad tails and help prioritize data-quality
investigation. An outlier may be a valid rare recording, a genuine speech
variation, a recording artifact, or a processing problem. The IQR rule alone
cannot distinguish among these causes.

## Reporting and interpretation

EDA results should be reported together with the feature extraction coverage,
missing-value counts, and known limitations of the measurements. Distribution
differences or correlations can motivate further testing, but they are not
evidence of detection accuracy. Accuracy and generalization require a separate
evaluation using suitable held-out data.
