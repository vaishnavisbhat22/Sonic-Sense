# Member 2 Final Feature Dataset Verification Checklist

Use this checklist only after feature extraction has finished. The expected
input is `C:\Sonic-Sense-main\audio_features_v2.csv`, with 47,400 expected
WAV recordings and labels from `C:\Sonic-Sense-main\audio_labels.csv`.

This checklist is a verification procedure, not a report of completed
extraction. Do not mark an item complete until it has been checked against the
finished files. Do not run EDA or start model training until data-quality
issues are resolved or documented.

## 1. Confirm extraction has finished

- [ ] Confirm the extraction process has exited.
- [ ] Confirm `audio_features_v2.csv` exists and its size and modification time
  have stopped changing.
- [ ] Review `audio_feature_errors.csv` and note the number of failed files.
- [ ] If extraction was interrupted or failures need retrying, use the
  extractor's resume behavior, then repeat this checklist.

## 2. Verify row count and filename coverage

Expected: 47,400 feature records, with one unique row for every expected WAV.

- [ ] Count data rows in the feature CSV, excluding its header.
- [ ] Compare the row count with 47,400.
- [ ] Compare feature filenames with the filenames in the audio folder and
  label CSV.
- [ ] List any expected WAV filename missing from the feature CSV.
- [ ] List any feature filename that is not present in the expected WAV set.

A matching row count alone is not sufficient: different missing and extra
files can cancel each other out.

## 3. Check duplicate filenames

- [ ] Confirm the `filename` column exists.
- [ ] Confirm every filename is non-empty.
- [ ] Count duplicate filenames.
- [ ] Pass only if the duplicate count is zero.

## 4. Check missing and incorrect labels

- [ ] Confirm the `label` column exists and contains no blank values.
- [ ] Confirm all labels are exactly `human` or `synthetic` after the project's
  agreed normalization.
- [ ] Compare each feature row's label with the filename-to-label mapping in
  `audio_labels.csv`.
- [ ] Count and investigate any missing, invalid, or mismatched labels.
- [ ] Confirm that filenames in the feature CSV map to expected label records.

Do not silently guess a label for a missing or mismatched row.

## 5. Check missing, non-numeric, and infinite feature values

- [ ] Exclude only the identifier/target fields (`filename`, `label`) from
  numeric-feature validation.
- [ ] Confirm every expected feature column contains numeric values.
- [ ] Count missing values in each feature column and affected rows.
- [ ] Check for positive and negative infinity, as well as NaN.
- [ ] Investigate and document affected records; do not silently replace or
  drop values.
- [ ] Pass only when all remaining training rows have finite numeric feature
  values, or when any explicitly approved data-cleaning steps are documented.

## 6. Check extraction failures

- [ ] Confirm `audio_feature_errors.csv` exists and has the expected columns.
- [ ] Count its data rows, excluding the header.
- [ ] For each failed filename, verify it is not incorrectly counted as a
  successfully extracted feature row.
- [ ] Retry or document failures and reconcile them with the expected input
  list.
- [ ] Report the final number of successful, failed, and unresolved recordings.

## 7. Check feature schema

Confirm the CSV contains the current extractor's expected feature columns.
Column-name prefixes below are a quick check; use the current
`feature_columns()` definition in `extract_audio_features.py` as the schema
source of truth.

- [ ] **MFCC:** columns beginning with `mfcc_`.
- [ ] **GTCC:** columns beginning with `gtcc_`.
- [ ] **Spectral/time-domain:** centroid, bandwidth, rolloff, flatness,
  zero-crossing rate, RMS, and spectral-contrast columns.
- [ ] **Prosodic/timing:** `f0_mean_hz`, `f0_std_hz`, `voiced_fraction`, and
  `syllabic_rate_proxy`.
- [ ] Confirm no expected feature column is absent and no unexpected
  identifier/target fields have been included as model inputs.

The names `voiced_fraction` and `syllabic_rate_proxy` describe approximations:
the former is energy-threshold based, and the latter is an acoustic onset-rate
proxy, not transcript-based voicing or syllable counting.

## 8. Confirm both classes are present

- [ ] Count `human` rows in the feature CSV.
- [ ] Count `synthetic` rows in the feature CSV.
- [ ] Confirm both counts are greater than zero.
- [ ] Compare class counts with the label file and investigate coverage
  differences.

Do not assume the expected class proportions until counts have been measured.

## 9. Confirm Pandas can load and inspect the CSV

Run this after extraction is complete. It reads the final CSV and label file;
it does not alter them.

```powershell
cd C:\Sonic-Sense-main
python -c "import pandas as pd; from pathlib import Path; p=Path(r'audio_features_v2.csv'); df=pd.read_csv(p); print('shape:', df.shape); print('columns:', len(df.columns)); print('label counts:'); print(df['label'].value_counts(dropna=False)); print('duplicate filenames:', df['filename'].duplicated().sum()); print('missing filename:', df['filename'].isna().sum()); print('missing label:', df['label'].isna().sum())"
```

- [ ] Confirm the command completes without a parsing or decoding error.
- [ ] Confirm the reported shape and class counts are recorded in the handoff.
- [ ] Run separate checks for numeric conversion and finite values; successful
  CSV loading alone does not prove feature integrity.

## 10. Final sign-off for Member 3

- [ ] Row and filename coverage reconciled against the expected WAV and label
  sets.
- [ ] No unresolved duplicate filenames, missing labels, invalid labels, or
  label mismatches.
- [ ] Feature schema matches the current extractor.
- [ ] Missing/non-finite values and extraction errors are resolved or
  explicitly documented.
- [ ] Both classes are represented.
- [ ] Pandas load check passes.
- [ ] Final CSV path, row count, feature count, class counts, and known
  limitations are shared with Member 3.

Only mark the dataset ready for model training after the checks above have
been completed and the remaining exceptions have been agreed with the team.
