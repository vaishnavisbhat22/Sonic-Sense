import csv
from pathlib import Path

protocol_file = Path(r"E:\SonicSense\extractASVspoof5_protocols\ASVspoof5.dev.track_1.tsv")
audio_folder = Path(r"E:\SonicSense\processed_audio")
output_file = Path(r"E:\SonicSense\audio_labels.csv")

labels = {}

with open(protocol_file, "r") as file:
    for line in file:
        columns = line.strip().split()

        if len(columns) < 9:
            continue

        audio_id = columns[1]

        if "spoof" in columns:
            labels[audio_id] = "synthetic"
        elif "bonafide" in columns:
            labels[audio_id] = "human"

count = 0

with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "label"])

    for wav in sorted(audio_folder.glob("D_*.wav")):
        if wav.stem in labels:
            writer.writerow([wav.name, labels[wav.stem]])
            count += 1

print("CSV created successfully!")
print("Total records:", count)
print("Saved at:", output_file)
