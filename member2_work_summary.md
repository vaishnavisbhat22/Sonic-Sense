# Member 2 Work Summary: Feature Extraction and EDA

## Project role

As Member 2, my responsibility is to prepare labeled audio recordings for
analysis and model development. This includes checking audio and label
consistency, extracting acoustic features, examining those features through
Exploratory Data Analysis (EDA), and preparing a verified feature dataset for
Member 3.

The project objective is to study the classification of recordings as Human
Speech or AI-Generated (Synthetic) Speech. Acoustic features are measurements
for analysis; they do not prove that a recording belongs to either class.

## Dataset preparation and verification

The configured audio input folder is `C:\SonicSense\audio_dataset`, and the
project label mapping is stored in `audio_labels.csv`. The validation workflow
checks WAV readability and metadata such as sample rate, channel count,
duration, and whether filenames match labels.

The project also contains preprocessing metadata describing fixed-duration
segments. The paths in that metadata must be checked against the actual files
available to the team before assuming that those preprocessed segments are the
input to feature extraction. Source recordings may have variable lengths, and
the feature extractor does not itself split recordings into fixed-duration
segments.

## Feature extraction methodology

The implementation is in `extract_audio_features.py`. Audio is read as
floating-point samples, stereo channels are averaged to mono, and audio is
resampled to 16 kHz when necessary. Frame-based measurements use short
overlapping analysis windows. Their mean and standard deviation are stored as
recording-level summary features.

### MFCC and GTCC features

- **MFCC:** The extractor computes Mel-Frequency Cepstral Coefficients from
  Mel-scaled log spectral energies. Twenty coefficients are summarized by
  their mean and standard deviation.
- **GTCC:** The extractor filters audio through an ERB-spaced gammatone
  filterbank, computes log frame energies, and applies a discrete cosine
  transform. Twenty coefficients are summarized by their mean and standard
  deviation.

These are complementary compact descriptions of spectral shape. Their
distributions can be compared between labels, but neither feature family is
alone a definitive synthetic-speech detector.

### Spectral and prosodic features

The extraction code includes:

- **Spectral:** centroid, bandwidth, rolloff, flatness, contrast, zero-crossing
  rate, and RMS energy. Most are summarized using the mean and standard
  deviation across frames.
- **Prosodic and timing-related:** YIN-estimated fundamental frequency (F0)
  mean and standard deviation, an energy-threshold activity fraction named
  `voiced_fraction`, and an onset-per-second value named
  `syllabic_rate_proxy`.

The activity fraction is not a true voicing detector. The onset rate is not a
transcript-based syllable count or speaking rate. These values should be
reported as approximate acoustic descriptors.

## Exploratory Data Analysis activities

The EDA implementation in `analyze_audio_features.py` is designed to:

- Compare Human and Synthetic class counts and feature coverage.
- Calculate per-feature mean, median, standard deviation, minimum, and maximum.
- Plot MFCC and GTCC distributions by class.
- Compare spectral and prosodic features across classes.
- Calculate a Pearson correlation matrix and save a heatmap.
- Report missing, non-numeric, and non-finite values.
- Summarize potential outliers by class and feature using the 1.5 × IQR rule.

EDA describes patterns and data quality in the examined dataset. It does not
establish model accuracy or performance on unseen speakers, datasets, or
synthesis systems.

## Tools and libraries

The project uses:

- **Python** for the extraction and analysis workflow.
- **Librosa** for audio analysis, MFCCs, spectral measurements, YIN F0
  estimation, and onset detection.
- **SoundFile** for reading WAV audio.
- **SciPy** for gammatone filtering, the discrete cosine transform, and
  numerical operations.
- **NumPy** and **Pandas** for numerical processing, CSV input/output, and
  summaries.
- **Matplotlib** for EDA figures.
- **Pytest** for automated tests of feature extraction and EDA behavior.

## Output files

### Existing project outputs

- `audio_labels.csv`: filename-to-class mapping.
- `audio_features.csv`: older, partial feature table using the earlier feature
  schema.
- `audio_feature_errors.csv`: per-file feature extraction error report.
- `audio_quality_report.csv`: WAV metadata report.
- `audio_anomaly_report.csv`: previously identified short-audio entries.
- `audio_dataset_validation.csv`: per-file validation output; its coverage
  should be checked against the current audio inventory before relying on it.
- `preprocessing_metadata.csv` and `preprocessing_errors.csv`: preprocessing
  records and error report. Metadata entries should be checked against
  available audio files.

### EDA outputs

The original `eda` folder contains reports from the earlier feature table:

- `eda_class_distribution.csv`
- `eda_class_distribution.png`
- `eda_feature_summary.csv`
- `eda_feature_distributions.png`

The completed expanded-feature EDA is in `eda\v2`. It includes class
distribution, feature statistics, MFCC and GTCC distributions, spectral and
prosodic comparisons, a Pearson correlation matrix and heatmap, missing-value
reports, and IQR outlier summaries. Zero-crossing-rate features are included
in the spectral comparison. A refreshed corrected set is in `eda\v3`.

### Final outputs

The expanded feature table is saved as `audio_features_v2.csv`. The verified
table contains 47,400 recordings and 134 numeric features. Labels match the
label file, filenames are unique, and no missing or non-finite feature values
or extraction failures were found. Class counts are 10,700 Human and 36,700
Synthetic. The feature CSV is ready for Member 3.

## Challenges faced

- The project has an older feature CSV with a different schema from the
  expanded extractor output.
- Preprocessing metadata references paths that may not be available in the
  current workspace, so the exact input dataset stage needs confirmation.
- Source recordings can have variable durations, while preprocessing records
  describe fixed-duration segments.
- Feature extraction over the full dataset is computationally intensive and
  needs to complete before final EDA and model handoff.
- The features named `voiced_fraction` and `syllabic_rate_proxy` are
  approximations and require careful interpretation.
- Class imbalance and incomplete extraction coverage, if present, must be
  reported and addressed appropriately before model training.

## Current status: completed versus pending

### Completed or available

- 47,400 recordings were processed into `audio_features_v2.csv`.
- 134 numeric features were extracted, covering MFCC, GTCC, spectral, and
  prosodic/timing-related measurements.
- The extraction error report has zero recorded failures.
- The final feature table has zero duplicate filenames, missing labels,
  label mismatches, missing values, or non-finite feature values.
- Class distribution is 10,700 Human and 36,700 Synthetic.
- EDA reports and visualizations were generated from the expanded feature
  table in `eda\v2`; corrected spectral comparison outputs including
  zero-crossing rate are in `eda\v3`.
- The feature CSV is ready for handoff to Member 3.
- Methodology documents describe the feature extraction and EDA approaches.
- Automated tests cover the feature and EDA code paths.

### Pending or not yet verified

- Member 3's model training, validation, and performance reporting are outside
  Member 2's completed handoff and remain to be done.
- Detection accuracy or generalization has not been established by feature
  extraction or EDA.

## Limitations and future improvements

- Keep the distinction between the source WAV recordings and any
  preprocessed-segment metadata clear when describing the dataset.
- Treat IQR outliers as candidates for review, not automatic errors; valid
  speech can have unusual feature values.
- Consider a true voiced/unvoiced estimator and a transcript- or
  speech-recognition-based speaking-rate measure if those measurements are
  required.
- Evaluate any future model on held-out data designed to limit speaker or
  source leakage. EDA alone does not establish detection accuracy.

## Handover

The expanded feature CSV and its EDA reports are ready for Member 3. Use a
stratified train-test split so both classes are represented in each partition.
Because the class distribution is imbalanced, report per-class precision,
recall, and F1-score, as well as a confusion matrix and balanced accuracy.
Where recordings share a source recording or speaker, consider a
group-aware split to reduce data leakage. EDA does not establish model
accuracy.
