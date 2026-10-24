
"""
SONICSENSE - AUDIO PREPROCESSING

Purpose:
    Prepare ASVspoof 5 audio recordings for the SonicSense project.

Preprocessing steps:
    1. Load audio and convert it to mono at 16 kHz.
    2. Trim leading and trailing silence conservatively.
    3. Normalize peak amplitude to 0.95.
    4. Split audio into fixed 5-second segments.
    5. Pad the final incomplete segment with zeros.
    6. Save segments as 16-bit PCM WAV files.
    7. Validate saved audio and record processing errors.

Input:
    extractedflac_D_aa/flac_D/
    data/final_metadata.csv

Output:
    processed_audio/
    data/preprocessing_metadata.csv
    data/preprocessing_errors.csv

Note:
    Original audio files are not modified.
    No aggressive denoising is applied.
"""

from pathlib import Path
import math
import traceback

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Original audio directory
AUDIO_ROOT = PROJECT_ROOT / "extractedflac_D_aa" / "flac_D"

# Input metadata
INPUT_METADATA = PROJECT_ROOT / "data" / "final_metadata.csv"

# Processed audio output directory
OUTPUT_ROOT = PROJECT_ROOT / "processed_audio"

# Output reports
OUTPUT_METADATA = PROJECT_ROOT / "data" / "preprocessing_metadata.csv"
OUTPUT_ERRORS = PROJECT_ROOT / "data" / "preprocessing_errors.csv"

# Audio settings
TARGET_SAMPLE_RATE = 16000
SEGMENT_DURATION = 5.0
SEGMENT_SAMPLES = int(TARGET_SAMPLE_RATE * SEGMENT_DURATION)
NORMALIZATION_PEAK = 0.95
TRIM_TOP_DB = 40

# None processes all recordings.
MAX_FILES = None

# Print progress after every 100 recordings.
PROGRESS_INTERVAL = 100

# Validate existing segments instead of overwriting them.
SKIP_EXISTING_OUTPUTS = True


# ============================================================
# LOAD AND VALIDATE METADATA
# ============================================================

def load_metadata():
    """Load metadata and validate required columns."""

    if not INPUT_METADATA.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {INPUT_METADATA}"
        )

    metadata = pd.read_csv(INPUT_METADATA)

    required_columns = {"sample_id", "label"}
    missing = required_columns - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Missing required metadata columns: {sorted(missing)}"
        )

    metadata = metadata.dropna(
        subset=["sample_id", "label"]
    ).copy()

    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["label"] = (
        metadata["label"].astype(str).str.lower()
    )

    allowed_labels = {"human", "synthetic"}
    invalid_labels = set(metadata["label"]) - allowed_labels

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels: {sorted(invalid_labels)}. "
            f"Expected: {sorted(allowed_labels)}"
        )

    if metadata["sample_id"].duplicated().any():
        duplicates = metadata.loc[
            metadata["sample_id"].duplicated(),
            "sample_id"
        ].tolist()

        raise ValueError(
            "Duplicate sample IDs in metadata. Examples: "
            f"{duplicates[:10]}"
        )

    return metadata


# ============================================================
# FIND ORIGINAL AUDIO FILE
# ============================================================

def find_audio_file(sample_id):
    """Find the original FLAC recording by sample ID."""

    direct_path = AUDIO_ROOT / f"{sample_id}.flac"

    if direct_path.is_file():
        return direct_path

    # Search subdirectories if the file is not directly available.
    matches = list(AUDIO_ROOT.rglob(f"{sample_id}.flac"))

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"No FLAC recording found for sample ID: {sample_id}"
        )

    raise RuntimeError(
        f"Multiple files found for sample ID: {sample_id}"
    )


# ============================================================
# LOAD AND PREPROCESS AUDIO
# ============================================================

def load_and_preprocess(audio_path):
    """Load, trim, and normalize audio."""

    audio, sample_rate = librosa.load(
        str(audio_path),
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "Audio contains NaN or infinite values."
        )

    # Conservatively trim leading and trailing silence.
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=TRIM_TOP_DB
    )

    # Keep original audio if trimming leaves no useful signal.
    if (
        trimmed.size > 0
        and np.max(np.abs(trimmed)) > 1e-8
    ):
        audio = trimmed

    # Normalize peak amplitude to 0.95.
    # Leave silent audio unchanged.
    peak = float(np.max(np.abs(audio)))

    if peak > 1e-8:
        audio = audio * (NORMALIZATION_PEAK / peak)

    audio = np.asarray(audio, dtype=np.float32)

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "Preprocessed audio contains invalid values."
        )

    return audio, sample_rate


# ============================================================
# SPLIT AUDIO INTO 5-SECOND SEGMENTS
# ============================================================

def make_segments(audio):
    """Split audio and zero-pad the final incomplete segment."""

    if audio.size == 0:
        raise ValueError("Cannot segment empty audio.")

    number_of_segments = math.ceil(
        len(audio) / SEGMENT_SAMPLES
    )

    segments = []

    for index in range(number_of_segments):
        start = index * SEGMENT_SAMPLES
        end = start + SEGMENT_SAMPLES

        segment = audio[start:end]

        # Pad the final segment to exactly 5 seconds.
        if len(segment) < SEGMENT_SAMPLES:
            segment = np.pad(
                segment,
                (0, SEGMENT_SAMPLES - len(segment)),
                mode="constant"
            )

        segments.append(
            np.asarray(segment, dtype=np.float32)
        )

    return segments


# ============================================================
# VALIDATE SAVED AUDIO
# ============================================================

def validate_saved_audio(audio_path):
    """Validate sample rate, channels, duration, and audio values."""

    info = sf.info(str(audio_path))

    if info.samplerate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Incorrect sample rate: {info.samplerate}"
        )

    if info.channels != 1:
        raise ValueError(
            f"Incorrect channel count: {info.channels}"
        )

    if info.frames != SEGMENT_SAMPLES:
        raise ValueError(
            f"Incorrect frame count: {info.frames}"
        )

    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32"
    )

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            "Saved audio sample rate validation failed."
        )

    if audio.ndim != 1:
        raise ValueError("Saved audio is not mono.")

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "Saved audio contains non-finite values."
        )

    return info


# ============================================================
# PROCESS ONE RECORDING
# ============================================================

def process_recording(row):
    """Process one recording and return segment metadata."""

    sample_id = row["sample_id"]
    label = row["label"]

    audio_path = find_audio_file(sample_id)

    audio, sample_rate = load_and_preprocess(audio_path)
    segments = make_segments(audio)

    rows = []

    output_dir = OUTPUT_ROOT / label / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, segment in enumerate(segments, start=1):

        segment_name = f"{sample_id}_seg{index:03d}.wav"
        output_path = output_dir / segment_name

        if output_path.exists() and SKIP_EXISTING_OUTPUTS:
            # Existing output must pass validation.
            info = validate_saved_audio(output_path)

        else:
            sf.write(
                str(output_path),
                segment,
                TARGET_SAMPLE_RATE,
                subtype="PCM_16"
            )

            info = validate_saved_audio(output_path)

        rows.append({
            "sample_id": sample_id,
            "segment_id": segment_name,
            "label": label,
            "source_file": str(audio_path),
            "processed_file": str(output_path),
            "original_sample_rate": sample_rate,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_seconds": info.frames / info.samplerate,
            "segment_number": index,
            "segment_count_for_recording": len(segments),
            "segment_samples": info.frames,
            "trim_top_db": TRIM_TOP_DB,
            "normalization_peak": NORMALIZATION_PEAK,
            "denoising_applied": False
        })

    return rows


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Run preprocessing and save metadata and error reports."""

    if not AUDIO_ROOT.is_dir():
        raise FileNotFoundError(
            f"Audio directory not found: {AUDIO_ROOT}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_METADATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = load_metadata()

    if MAX_FILES is not None:
        metadata = metadata.head(MAX_FILES).copy()

    total = len(metadata)

    if total == 0:
        raise ValueError("No recordings found to process.")

    print("=" * 65)
    print("SONICSENSE AUDIO PREPROCESSING")
    print("=" * 65)
    print(f"Recordings to process: {total}")
    print(f"Target sample rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"Segment duration: {SEGMENT_DURATION} seconds")
    print(f"Progress interval: {PROGRESS_INTERVAL} recordings")
    print(f"Output folder: {OUTPUT_ROOT}")
    print("=" * 65)

    all_rows = []
    errors = []
    segments_created = 0

    for position, (_, row) in enumerate(
        metadata.iterrows(),
        start=1
    ):
        sample_id = str(row["sample_id"])

        try:
            segment_rows = process_recording(row)

            all_rows.extend(segment_rows)
            segments_created += len(segment_rows)

        except Exception as exc:
            errors.append({
                "sample_id": sample_id,
                "label": row["label"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc()
            })

        # Progress output: no repeated error count here.
        if (
            position % PROGRESS_INTERVAL == 0
            or position == total
        ):
            print(
                f"Progress: {position}/{total} recordings | "
                f"Segments created: {segments_created}"
            )

            # Save interim reports.
            pd.DataFrame(all_rows).to_csv(
                OUTPUT_METADATA,
                index=False
            )

            pd.DataFrame(errors).to_csv(
                OUTPUT_ERRORS,
                index=False
            )

    # Save final metadata and error reports.
    pd.DataFrame(all_rows).to_csv(
        OUTPUT_METADATA,
        index=False
    )

    pd.DataFrame(errors).to_csv(
        OUTPUT_ERRORS,
        index=False
    )

    # Final summary: error count appears only once.
    print("\n" + "=" * 65)
    print("PREPROCESSING FINISHED")
    print("=" * 65)
    print(f"Recordings attempted: {total}")
    print(f"Recordings with errors: {len(errors)}")
    print(f"Segments recorded in this run: {segments_created}")
    print(f"Metadata: {OUTPUT_METADATA}")
    print(f"Error report: {OUTPUT_ERRORS}")
    print("=" * 65)


if __name__ == "__main__":
    main()
