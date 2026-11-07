import csv
import soundfile as sf
import numpy as np
from pathlib import Path

audio_folder = Path(r"E:\SonicSense\processed_audio")
output_file = Path(r"E:\SonicSense\audio_anomaly_report.csv")

short_count = 0
silent_count = 0
files = sorted(audio_folder.glob("D_*.wav"))

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename", "duration_seconds",
        "rms_level", "peak_level", "issue"
    ])

    for i, audio in enumerate(files, 1):
        data, sr = sf.read(str(audio), dtype="float32")

        duration = len(data) / sr
        rms = float(np.sqrt(np.mean(data ** 2)))
        peak = float(np.max(np.abs(data))) if len(data) else 0.0

        issues = []

        if duration < 1.0:
            issues.append("short_audio")
            short_count += 1

        if peak < 0.0001:
            issues.append("almost_silent")
            silent_count += 1

        if issues:
            writer.writerow([
                audio.name,
                round(duration, 3),
                round(rms, 6),
                round(peak, 6),
                ", ".join(issues)
            ])

        if i % 5000 == 0:
            print(f"Checked {i}/{len(files)} files")

print("\n========== AUDIO ANOMALY REPORT ==========")
print("Total files checked:", len(files))
print("Files shorter than 1 second:", short_count)
print("Almost silent files:", silent_count)
print("Report saved:", output_file)
