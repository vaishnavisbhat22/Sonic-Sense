import csv
import soundfile as sf
from pathlib import Path
from collections import Counter

audio_folder = Path(r"E:\SonicSense\processed_audio")
output_file = Path(r"E:\SonicSense\audio_quality_report.csv")

files = sorted(audio_folder.glob("D_*.wav"))

sample_rates = Counter()
channels_count = Counter()
subtypes = Counter()
bad_files = []
durations = []

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "sample_rate",
        "channels",
        "duration_seconds",
        "subtype"
    ])

    for i, audio in enumerate(files, 1):
        try:
            info = sf.info(str(audio))

            duration = info.frames / info.samplerate

            sample_rates[info.samplerate] += 1
            channels_count[info.channels] += 1
            subtypes[info.subtype] += 1
            durations.append(duration)

            writer.writerow([
                audio.name,
                info.samplerate,
                info.channels,
                round(duration, 3),
                info.subtype
            ])

        except Exception as e:
            bad_files.append((audio.name, str(e)))

        if i % 1000 == 0:
            print(f"Checked {i}/{len(files)} files")

print("\n========== AUDIO QUALITY REPORT ==========")
print("Total WAV files:", len(files))
print("Readable files:", len(files) - len(bad_files))
print("Unreadable files:", len(bad_files))
print("Sample rates:", sample_rates)
print("Channels:", channels_count)
print("Audio subtypes:", subtypes)

if durations:
    print("Minimum duration:", round(min(durations), 2), "seconds")
    print("Maximum duration:", round(max(durations), 2), "seconds")

print("Report saved:", output_file)

if bad_files:
    print("\nFirst 10 unreadable files:")
    for name, error in bad_files[:10]:
        print(name, error)
