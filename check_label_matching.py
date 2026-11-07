
from pathlib import Path
import re

protocol_folder = Path(
    r"E:\SonicSense\extractASVspoof5_protocols"
)

audio_folder = Path(
    r"E:\SonicSense\processed_audio"
)

# Read labels from all protocol TSV files
protocol_labels = {}

for tsv_file in protocol_folder.glob("*.tsv"):
    with open(tsv_file, "r", encoding="utf-8") as f:

        for line in f:
            columns = line.strip().split()

            if not columns:
                continue

            audio_ids = [
                x for x in columns
                if re.fullmatch(r"T_\d+", x)
            ]

            if not audio_ids:
                continue

            audio_id = audio_ids[0]
            number = audio_id.split("_")[1]

            if "spoof" in columns:
                label = "synthetic"
            elif "bonafide" in columns:
                label = "human"
            else:
                continue

            protocol_labels.setdefault(
                number, set()
            ).add(label)

# Check WAV files in main folder only
wav_files = list(audio_folder.glob("D_*.wav"))

matched = 0
unmatched = 0
conflicts = 0

for wav_file in wav_files:
    number = wav_file.stem.split("_")[1]

    labels = protocol_labels.get(number, set())

    if len(labels) == 1:
        matched += 1
    elif len(labels) == 0:
        unmatched += 1
    else:
        conflicts += 1

print("\n--- LABEL MATCHING REPORT ---")
print("WAV files:", len(wav_files))
print("Matched files:", matched)
print("Unmatched files:", unmatched)
print("Conflicting labels:", conflicts)
print("Protocol IDs with labels:", len(protocol_labels))
