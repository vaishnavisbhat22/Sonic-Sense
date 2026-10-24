
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_DIR = Path(r"E:\SonicSense")

AUDIO_DIR = PROJECT_DIR / "extractedflac_D_aa" / "flac_D"
PROTOCOL_FILE = (
    PROJECT_DIR
    / "extractASVspoof5_protocols"
    / "ASVspoof5.dev.track_1.tsv"
)

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "final_metadata.csv"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not AUDIO_DIR.is_dir():
        raise FileNotFoundError(f"Audio folder not found: {AUDIO_DIR}")

    if not PROTOCOL_FILE.is_file():
        raise FileNotFoundError(f"Protocol file not found: {PROTOCOL_FILE}")

    print("Reading protocol...")

    # The protocol is whitespace-separated and has no header.
    protocol = pd.read_csv(
        PROTOCOL_FILE,
        sep=r"\s+",
        header=None,
        dtype=str,
        engine="python",
    )

    if protocol.shape[1] < 9:
        raise ValueError(
            f"Expected at least 9 protocol columns, "
            f"but found {protocol.shape[1]}"
        )

    # Column 2 is the recording ID; column 9 is the label.
    protocol = protocol.iloc[:, [1, 8]].copy()
    protocol.columns = ["sample_id", "protocol_label"]

    protocol["protocol_label"] = (
        protocol["protocol_label"].str.lower().str.strip()
    )

    protocol = protocol[
        protocol["protocol_label"].isin(["bonafide", "spoof"])
    ].copy()

    # Convert ASVspoof labels to project labels.
    protocol["label"] = protocol["protocol_label"].map(
        {
            "bonafide": "human",
            "spoof": "synthetic",
        }
    )

    # Keep only FLAC files that exist in the audio directory.
    print("Indexing audio files...")
    audio_files = sorted(AUDIO_DIR.glob("*.flac"))

    audio = pd.DataFrame(
        {
            "sample_id": [path.stem for path in audio_files],
            "file_path": [str(path.resolve()) for path in audio_files],
        }
    )

    if audio.empty:
        raise RuntimeError(f"No FLAC files found in {AUDIO_DIR}")

    # Remove duplicate protocol IDs to avoid duplicate metadata rows.
    protocol = protocol.drop_duplicates(
        subset="sample_id",
        keep="first",
    )

    metadata = audio.merge(
        protocol,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    if metadata.empty:
        raise RuntimeError(
            "No audio files matched the protocol. "
            "Check the recording IDs and protocol columns."
        )

    metadata = metadata[
        [
            "sample_id",
            "file_path",
            "label",
            "protocol_label",
        ]
    ].sort_values("sample_id")

    metadata.to_csv(OUTPUT_FILE, index=False)

    print("\n--- METADATA SUMMARY ---")
    print(f"Audio files found: {len(audio):,}")
    print(f"Valid protocol labels: {len(protocol):,}")
    print(f"Matched recordings: {len(metadata):,}")
    print("\nLabel counts:")
    print(metadata["label"].value_counts().to_string())
    print(f"\nSaved metadata to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
