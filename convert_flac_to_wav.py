from pathlib import Path
import soundfile as sf

input_folder = Path(r"E:\SonicSense\extractedflac_D_aa\flac_D")
output_folder = Path(r"E:\SonicSense\processed_audio")

output_folder.mkdir(parents=True, exist_ok=True)

count = 0

for flac_file in input_folder.glob("*.flac"):
    try:
        audio, sample_rate = sf.read(flac_file)

        wav_file = output_folder / (flac_file.stem + ".wav")

        sf.write(wav_file, audio, sample_rate, subtype="PCM_16")

        count += 1

        if count % 500 == 0:
            print(f"Converted {count} files")

    except Exception as e:
        print(f"Error: {flac_file.name}: {e}")

print(f"Conversion completed! Total: {count}")