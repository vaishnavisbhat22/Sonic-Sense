
from pathlib import Path
from collections import Counter

# Paths
protocol_file = Path(
    r"E:\SonicSense\extractASVspoof5_protocols\ASVspoof5.dev.track_1.tsv"
)

audio_folder = Path(r"E:\SonicSense\processed_audio")

# Read protocol labels
protocol_labels = {}
duplicate_ids = []

with open(protocol_file, "r") as file:
    for line in file:
        columns = line.strip().split()

        if len(columns) < 3:
            continue

        audio_id = columns[1]

        if "spoof" in columns:
            label = "synthetic"
        elif "bonafide" in columns:
            label = "human"
        else:
            continue

        if audio_id in protocol_labels:
            duplicate_ids.append(audio_id)

        protocol_labels[audio_id] = label

# Check WAV files
wav_files = list(audio_folder.glob("D_*.wav"))

matched = []
unmatched = []

for wav in wav_files:
    audio_id = wav.stem

    if audio_id in protocol_labels:
        matched.append((wav.name, protocol_labels[audio_id]))
    else:
        unmatched.append(wav.name)

# Report
print("\n" + "=" * 50)
print("LABEL MATCHING REPORT")
print("=" * 50)

print("WAV files:", len(wav_files))
print("Protocol IDs:", len(protocol_labels))
print("Matched files:", len(matched))
print("Unmatched files:", len(unmatched))
print("Duplicate protocol IDs:", len(duplicate_ids))

print("\nMatched label counts:")
print(Counter(label for _, label in matched))

print("\nFirst 10 matched files:")
for filename, label in matched[:10]:
    print(filename, "->", label)

print("\nFirst 10 unmatched files:")
for filename in unmatched[:10]:
    print(filename)
