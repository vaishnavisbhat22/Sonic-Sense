"""Extract labeled, fixed-length acoustic feature vectors from audio files."""

import argparse
import csv
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import dct


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO_DIR = PROJECT_ROOT.parent / "SonicSense" / "audio_dataset"
DEFAULT_LABELS_FILE = PROJECT_ROOT / "audio_labels.csv"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "audio_features_v2.csv"

TARGET_SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MFCC = 20
N_MELS = 40
N_GTCC_FILTERS = 24
N_SPECTRAL_CONTRAST_BANDS = 6
PROGRESS_INTERVAL = 500


def feature_columns():
    """Return stable CSV columns for the extracted feature vector."""

    columns = []
    for index in range(1, N_MFCC + 1):
        columns.extend([f"mfcc_{index:02d}_mean", f"mfcc_{index:02d}_std"])

    for index in range(1, N_MFCC + 1):
        columns.extend([f"gtcc_{index:02d}_mean", f"gtcc_{index:02d}_std"])

    for name in (
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "zero_crossing_rate",
        "rms",
    ):
        columns.extend([f"{name}_mean", f"{name}_std"])

    for index in range(1, N_SPECTRAL_CONTRAST_BANDS + 2):
        columns.extend([
            f"spectral_contrast_{index:02d}_mean",
            f"spectral_contrast_{index:02d}_std",
        ])

    for index in range(1, 13):
        columns.extend([f"chroma_{index:02d}_mean", f"chroma_{index:02d}_std"])

    columns.extend([
        "f0_mean_hz",
        "f0_std_hz",
        "voiced_fraction",
        "syllabic_rate_proxy",
    ])
    return columns


@lru_cache(maxsize=1)
def _gammatone_filterbank():
    """Build ERB-spaced fourth-order gammatone filters for 16 kHz audio."""

    def hz_to_erb(frequency):
        return 21.4 * np.log10(1.0 + 0.00437 * frequency)

    def erb_to_hz(value):
        return (10.0 ** (value / 21.4) - 1.0) / 0.00437

    erb_centers = np.linspace(
        hz_to_erb(80.0),
        hz_to_erb(TARGET_SAMPLE_RATE / 2.0 - 100.0),
        N_GTCC_FILTERS,
    )
    center_frequencies = erb_to_hz(erb_centers)
    return tuple(
        signal.tf2sos(
            *signal.gammatone(
                float(center_frequency),
                "iir",
                fs=TARGET_SAMPLE_RATE,
            )
        )
        for center_frequency in center_frequencies
    )


def _extract_gtcc(audio):
    """Compute mean/std statistics for 20 gammatone cepstral coefficients."""

    log_energies = []
    for filter_coefficients in _gammatone_filterbank():
        filtered = signal.sosfilt(filter_coefficients, audio)
        frame_energy = librosa.feature.rms(
            y=filtered,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
        log_energies.append(np.log(np.maximum(frame_energy**2, 1e-12)))

    cepstra = dct(np.asarray(log_energies), type=2, axis=0, norm="ortho")
    return cepstra[:N_MFCC]


def _append_statistics(result, name, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]

    for index, row in enumerate(values, start=1):
        suffix = f"_{index:02d}" if values.shape[0] > 1 else ""
        result[f"{name}{suffix}_mean"] = float(np.mean(row))
        result[f"{name}{suffix}_std"] = float(np.std(row))


def extract_audio_features_from_signal(audio, sample_rate):
    """Compute the feature vector for decoded audio samples."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Audio is empty.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains non-finite samples.")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    elif audio.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio; got shape {audio.shape}.")

    if sample_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )
        sample_rate = TARGET_SAMPLE_RATE

    magnitude = np.abs(
        librosa.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
    )
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    gtcc = _extract_gtcc(audio)
    chroma = librosa.feature.chroma_stft(
        S=magnitude,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    spectral = {
        "spectral_centroid": librosa.feature.spectral_centroid(
            S=magnitude, sr=sample_rate
        ),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(
            S=magnitude, sr=sample_rate
        ),
        "spectral_rolloff": librosa.feature.spectral_rolloff(
            S=magnitude, sr=sample_rate
        ),
        "spectral_flatness": librosa.feature.spectral_flatness(S=magnitude),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(
            audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        ),
        "rms": librosa.feature.rms(
            S=magnitude,
            frame_length=N_FFT,
        ),
        "spectral_contrast": librosa.feature.spectral_contrast(
            S=magnitude,
            sr=sample_rate,
            n_bands=N_SPECTRAL_CONTRAST_BANDS,
        ),
    }
    f0 = librosa.yin(
        audio,
        fmin=60.0,
        fmax=500.0,
        sr=sample_rate,
        frame_length=1024,
        hop_length=HOP_LENGTH,
    )
    frame_rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=HOP_LENGTH,
    )[0]
    active_frames = frame_rms >= max(0.005, float(np.max(frame_rms)) * 0.03)
    voiced_f0 = f0[active_frames & np.isfinite(f0)]

    onset_envelope = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(magnitude + 1e-10),
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        backtrack=False,
        units="frames",
    )

    result = {}
    _append_statistics(result, "mfcc", mfcc)
    _append_statistics(result, "gtcc", gtcc)
    for name, values in spectral.items():
        _append_statistics(result, name, values)
    _append_statistics(result, "chroma", chroma)
    result["f0_mean_hz"] = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    result["f0_std_hz"] = float(np.std(voiced_f0)) if voiced_f0.size else 0.0
    result["voiced_fraction"] = (
        float(np.count_nonzero(active_frames) / active_frames.size)
        if active_frames.size
        else 0.0
    )
    duration_seconds = audio.size / sample_rate
    result["syllabic_rate_proxy"] = (
        float(onset_frames.size / duration_seconds) if duration_seconds else 0.0
    )

    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Feature extraction produced non-finite values.")

    return result


def extract_audio_features(audio_path):
    """Read an audio file and compute its feature vector."""

    audio, sample_rate = _read_audio(audio_path)
    return extract_audio_features_from_signal(audio, sample_rate)


def _read_audio(source):
    """Decode audio with SoundFile, falling back to PyAV for other containers."""

    if hasattr(source, "seek"):
        source.seek(0)

    try:
        return sf.read(
            source,
            dtype="float32",
            always_2d=False,
        )
    except RuntimeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return _read_audio_with_pyav(source)


def _read_audio_with_pyav(source):
    """Decode containers unsupported by SoundFile without changing sample rate."""

    import av

    channel_chunks = []
    sample_rate = None

    with av.open(source, mode="r") as container:
        audio_stream = next(
            (stream for stream in container.streams if stream.type == "audio"),
            None,
        )
        if audio_stream is None:
            raise ValueError("Audio stream not found.")

        for frame in container.decode(audio=audio_stream.index):
            if sample_rate is None:
                sample_rate = frame.sample_rate
            elif frame.sample_rate != sample_rate:
                raise ValueError("Audio sample rate changes within the stream.")

            channels = len(frame.layout.channels)
            samples = frame.to_ndarray()
            if frame.format.is_planar:
                samples = samples.T
            else:
                samples = samples.reshape(-1, channels)

            if np.issubdtype(samples.dtype, np.integer):
                limits = np.iinfo(samples.dtype)
                if limits.min < 0:
                    samples = samples.astype(np.float32) / max(
                        abs(limits.min),
                        limits.max,
                    )
                else:
                    midpoint = (limits.max + 1) / 2
                    samples = (samples.astype(np.float32) - midpoint) / midpoint
            else:
                samples = samples.astype(np.float32)

            channel_chunks.append(samples)

    if not channel_chunks or sample_rate is None:
        raise ValueError("Audio stream contains no decodable samples.")

    audio = np.concatenate(channel_chunks, axis=0)
    if audio.shape[1] == 1:
        audio = audio[:, 0]
    return audio, sample_rate


def extract_audio_features_from_bytes(audio_bytes):
    """Decode uploaded audio bytes and compute the same feature vector."""

    if not audio_bytes:
        raise ValueError("Audio data is empty.")

    audio, sample_rate = _read_audio(io.BytesIO(audio_bytes))
    return extract_audio_features_from_signal(audio, sample_rate)


def load_labels(labels_file):
    """Load and validate filename-to-label mappings."""

    labels = {}
    with labels_file.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required_columns = {"filename", "label"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{labels_file} must contain columns: filename,label"
            )

        for line_number, row in enumerate(reader, start=2):
            filename = (row["filename"] or "").strip()
            label = (row["label"] or "").strip().lower()
            if not filename:
                raise ValueError(f"Missing filename in {labels_file} at line {line_number}.")
            if label not in {"human", "synthetic"}:
                raise ValueError(
                    f"Invalid label {label!r} in {labels_file} at line {line_number}."
                )
            if filename in labels:
                raise ValueError(f"Duplicate filename in labels: {filename}")
            labels[filename] = label

    return labels


def _extract_labeled_file(job):
    path, label = job
    try:
        return path.name, label, extract_audio_features(path), ""
    except Exception as error:
        return path.name, label, None, f"{type(error).__name__}: {error}"


def _existing_filenames(output_file, expected_header):
    """Return completed recordings after checking a previous output schema."""

    if not output_file.exists():
        return set()

    with output_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected_header:
            raise ValueError(
                f"Existing output has an unexpected schema: {output_file}. "
                "Choose a new output path or use --overwrite."
            )
        filenames = set()
        for row in reader:
            filename = row.get("filename")
            if not filename:
                raise ValueError(f"Invalid existing output row in {output_file}.")
            if filename in filenames:
                raise ValueError(f"Duplicate filename in existing output: {filename}")
            filenames.add(filename)
    return filenames


def run_extraction(audio_dir, labels_file, output_file, errors_file, limit, workers, overwrite):
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not labels_file.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    if output_file.resolve() == errors_file.resolve():
        raise ValueError("Feature output and error report must use different paths.")
    if labels_file.resolve() in {output_file.resolve(), errors_file.resolve()}:
        raise ValueError("Output paths must not overwrite the labels file.")
    if workers < 1:
        raise ValueError("--workers must be at least 1.")

    labels = load_labels(labels_file)
    audio_files = sorted(audio_dir.glob("D_*.wav"))
    if not audio_files:
        raise RuntimeError(f"No D_*.wav files found in {audio_dir}")

    jobs = [(path, labels[path.name]) for path in audio_files if path.name in labels]
    unmatched_audio = len(audio_files) - len(jobs)
    labels_without_audio = len(set(labels) - {path.name for path in audio_files})
    if not jobs:
        raise RuntimeError("No audio files matched the label CSV.")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        jobs = jobs[:limit]

    header = ["filename", "label", *feature_columns()]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    errors_file.parent.mkdir(parents=True, exist_ok=True)

    if overwrite:
        completed = set()
        output_mode = "w"
    else:
        completed = _existing_filenames(output_file, header)
        output_mode = "a" if output_file.exists() else "w"

    pending = [job for job in jobs if job[0].name not in completed]
    started = time.monotonic()
    failed = 0
    errors_mode = "a" if errors_file.exists() and not overwrite else "w"

    with output_file.open(output_mode, newline="", encoding="utf-8") as output_stream, \
            errors_file.open(errors_mode, newline="", encoding="utf-8") as errors_stream:
        writer = csv.DictWriter(output_stream, fieldnames=header)
        error_writer = csv.writer(errors_stream)
        if output_mode == "w":
            writer.writeheader()
        if errors_mode == "w":
            error_writer.writerow(["filename", "label", "error"])

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, (filename, label, features, error) in enumerate(
                executor.map(_extract_labeled_file, pending, chunksize=8),
                start=1,
            ):
                if error:
                    failed += 1
                    error_writer.writerow([filename, label, error])
                else:
                    writer.writerow(
                        {"filename": filename, "label": label, **features}
                    )

                if index % PROGRESS_INTERVAL == 0 or index == len(pending):
                    output_stream.flush()
                    errors_stream.flush()
                    elapsed = time.monotonic() - started
                    rate = index / elapsed if elapsed else 0
                    remaining = (len(pending) - index) / rate if rate else 0
                    print(
                        f"Processed {index:,}/{len(pending):,} pending files "
                        f"({failed:,} failures); elapsed {elapsed / 60:.1f} min, "
                        f"estimated remaining {remaining / 60:.1f} min",
                        flush=True,
                    )

    total_completed = len(completed) + len(pending) - failed
    print(f"Audio files found: {len(audio_files):,}")
    print(f"Labeled files selected: {len(jobs):,}")
    print(f"Label rows without a WAV in this folder: {labels_without_audio:,}")
    print(f"Audio files without labels: {unmatched_audio:,}")
    print(f"Features available: {total_completed:,}")
    print(f"Failures this run: {failed:,}")
    print(f"Feature CSV: {output_file}")
    print(f"Error report: {errors_file}")

    if failed:
        raise RuntimeError(
            f"Feature extraction failed for {failed} file(s); see {errors_file}."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract labeled handcrafted audio features from WAV files."
    )
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--errors",
        type=Path,
        default=DEFAULT_OUTPUT_FILE.with_name("audio_feature_errors.csv"),
    )
    parser.add_argument("--limit", type=int, help="Process only the first N matched files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel worker processes (default: up to 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output instead of resuming it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_extraction(
        args.audio_dir,
        args.labels,
        args.output,
        args.errors,
        args.limit,
        args.workers,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
