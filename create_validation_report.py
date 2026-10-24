from pathlib import Path
import pandas as pd

base = Path(r"E:\SonicSense")
data = base / "data"
metadata_path = data / "preprocessing_metadata.csv"
errors_path = data / "preprocessing_errors.csv"
report_path = data / "validation_report.txt"

df = pd.read_csv(metadata_path)
errors = (
    pd.read_csv(errors_path)
    if errors_path.exists() and errors_path.stat().st_size > 2
    else pd.DataFrame()
)

checks = {
    "Metadata contains rows": len(df) > 0,
    "All segments are 16 kHz": df["sample_rate"].eq(16000).all(),
    "All segments are mono": df["channels"].eq(1).all(),
    "All segments are 5 seconds": df["duration_seconds"].eq(5.0).all(),
    "All segments contain 80000 samples": df["segment_samples"].eq(80000).all(),
    "No missing processed file paths": df["processed_file"].notna().all(),
    "No duplicate segment IDs": df["segment_id"].is_unique,
    "No recorded preprocessing errors": errors.empty,
}

lines = [
    "SONICSENSE - DATASET VALIDATION REPORT",
    "=" * 42,
    f"Metadata rows: {len(df)}",
    f"Unique recordings: {df['sample_id'].nunique()}",
    f"Unique segments: {df['segment_id'].nunique()}",
    "",
    "LABEL COUNTS:",
    df["label"].value_counts().to_string(),
    "",
    "VALIDATION CHECKS:",
]

for name, passed in checks.items():
    lines.append(f"{'PASS' if passed else 'FAIL'} - {name}")

lines.extend([
    "",
    f"Recorded preprocessing errors: {len(errors)}",
    "Note: File existence and audio contents were not independently checked by this report."
])

report_path.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nReport saved to: {report_path}")
