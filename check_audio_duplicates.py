import hashlib
import csv
from pathlib import Path
from collections import defaultdict

audio_folder = Path(r"E:\SonicSense\processed_audio")
output_file = Path(r"E:\SonicSense\duplicate_audio_report.csv")

hash_groups = defaultdict(list)

wav_files = sorted(audio_folder.glob("D_*.wav"))

print("Checking exact audio duplicates...")
print("Total files:", len(wav_files))

for i, wav in enumerate(wav_files, start=1):
    sha256 = hashlib.sha256()

    with open(wav, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    hash_groups[sha256.hexdigest()].append(wav.name)

    if i % 1000 == 0:
        print(f"Processed {i}/{len(wav_files)} files")

duplicates = {
    file_hash: filenames
    for file_hash, filenames in hash_groups.items()
    if len(filenames) > 1
}

with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["sha256", "filename", "duplicate_count"])

    for file_hash, filenames in duplicates.items():
        for filename in filenames:
            writer.writerow([file_hash, filename, len(filenames)])

print("\n========== DUPLICATE REPORT ==========")
print("Total WAV files:", len(wav_files))
print("Unique audio contents:", len(hash_groups))
print("Duplicate groups:", len(duplicates))
print("Files involved in duplicates:", sum(len(v) for v in duplicates.values()))
print("Report saved:", output_file)
