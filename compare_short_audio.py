import csv
import soundfile as sf
from pathlib import Path

flac_folder = Path(r"E:\SonicSense\extractedflac_D_aa\flac_D")
wav_folder = Path(r"E:\SonicSense\processed_audio")

anomaly_file = Path(r"E:\SonicSense\audio_anomaly_report.csv")
output_file = Path(r"E:\SonicSense\short_audio_comparison.csv")

import pandas as pd

anomalies = pd.read_csv(anomaly_file)
results = []

for _, row in anomalies.iterrows():
    name = row["filename"]
    flac_name = Path(name).with_suffix(".flac")

    flac_path = flac_folder / flac_name
    wav_path = wav_folder / name

    if not flac_path.exists():
        results.append([name, "FLAC missing", "", "", "CHECK"])
        continue

    try:
        flac_info = sf.info(str(flac_path))
        wav_info = sf.info(str(wav_path))

        flac_duration = flac_info.frames / flac_info.samplerate
        wav_duration = wav_info.frames / wav_info.samplerate

        difference = abs(flac_duration - wav_duration)
        status = "MATCH" if difference < 0.001 else "MISMATCH"

        results.append([
            name,
            round(flac_duration, 4),
            round(wav_duration, 4),
            round(difference, 6),
            status
        ])

    except Exception as e:
        results.append([name, "ERROR", "", "", str(e)])

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename", "FLAC_duration",
        "WAV_duration", "difference", "status"
    ])
    writer.writerows(results)

print("\n========== FLAC vs WAV COMPARISON ==========")
print("Files checked:", len(results))
print("Matching durations:", sum(r[4] == "MATCH" for r in results))
print("Mismatches:", sum(r[4] == "MISMATCH" for r in results))
print("Missing FLAC files:", sum(r[1] == "FLAC missing" for r in results))
print("Report:", output_file)

for result in results:
    if result[4] != "MATCH":
        print(result)
