# SonicSense Feature Extraction Methodology

## Purpose and scope

SonicSense represents each audio recording with numeric acoustic features for
exploration and supervised model training. These features describe different
properties of a recording; none of them alone proves that speech is human or
synthetic. Their usefulness must be measured with suitable evaluation data.

The current implementation is in `extract_audio_features.py`. It reads audio
as floating-point samples, converts stereo to mono by averaging channels, and
resamples to 16 kHz when needed. Most spectral and cepstral features use a
512-sample analysis window and a 160-sample hop (about 10 ms at 16 kHz).
Cepstral, spectral, chroma, and RMS frame measurements are summarized as a
mean and standard deviation for each recording. F0 and the other prosodic
quantities are stored as recording-level values.

The extractor writes one feature row per matched WAV file; it does not split a
recording into fixed-duration segments.

## Feature descriptions

### 1. MFCC — Mel-Frequency Cepstral Coefficients

- **What it measures:** A compact representation of the short-time spectral
  envelope. The spectrum is summarized using Mel-spaced frequency bands, their
  log energies are taken, and a discrete cosine transform (DCT) produces
  cepstral coefficients.
- **Information provided:** The coefficients describe broad spectral shape and
  how that shape changes over time. SonicSense computes 20 coefficients and
  stores each coefficient's mean and standard deviation over frames.
- **Why it may be useful:** Human vocal-tract resonances and synthesis or
  vocoder processing can produce different spectral-envelope patterns. MFCCs
  provide a compact way to compare those patterns, but they are not unique
  indicators of synthetic speech.

### 2. GTCC — Gammatone Frequency Cepstral Coefficients

- **What it measures:** Cepstral coefficients calculated from energy in
  gammatone-filter channels. Gammatone filters approximate aspects of human
  auditory frequency resolution.
- **Information provided:** The implementation uses 24 ERB-spaced gammatone
  filters from approximately 80 Hz to 7.9 kHz. It measures frame energy in
  each channel, takes log energy, applies a DCT, and retains 20 coefficients.
  Their means and standard deviations are saved.
- **Why it may be useful:** The auditory-spaced bands offer a different view of
  spectral structure than Mel-spaced MFCCs. Comparing MFCCs and GTCCs can show
  whether class differences persist under a different frequency scale; neither
  representation guarantees a reliable distinction by itself.

### 3. Spectral Centroid

- **What it measures:** The weighted average frequency of spectral energy in a
  frame.
- **Information provided:** A higher centroid generally corresponds to more
  energy at higher frequencies and is often described as a brighter sound.
  SonicSense stores frame-wise mean and standard deviation.
- **Why it may be useful:** Recording bandwidth, filtering, and synthesis
  artifacts can affect the distribution of high- versus low-frequency energy.
  The centroid is a broad descriptor and is also strongly affected by content
  and recording conditions.

### 4. Spectral Bandwidth

- **What it measures:** How far spectral energy is spread around the spectral
  centroid.
- **Information provided:** It summarizes whether a frame's energy is
  concentrated in a narrower frequency region or spread more widely.
  SonicSense stores frame-wise mean and standard deviation.
- **Why it may be useful:** Different synthesis, filtering, or compression
  processes may change spectral spread. The measure also varies naturally with
  phonemes and background noise, so it must be interpreted alongside other
  features.

### 5. Spectral Rolloff

- **What it measures:** The frequency below which a selected proportion of
  spectral energy is concentrated. Librosa's default rolloff proportion is
  used by the current implementation.
- **Information provided:** It gives a frame-level estimate of the upper
  frequency boundary containing most of the energy. SonicSense stores its mean
  and standard deviation.
- **Why it may be useful:** A rolloff shift can reflect changes to high
  frequencies caused by bandwidth limits, filtering, or synthesis. It does not
  identify the cause of the shift on its own.

### 6. Zero Crossing Rate

- **What it measures:** How often the waveform changes sign within a frame.
- **Information provided:** It is a time-domain indicator related to rapid
  oscillation and noisiness. SonicSense uses a 512-sample frame and 160-sample
  hop, then stores mean and standard deviation.
- **Why it may be useful:** Fricatives, noise, and some processing artifacts can
  change zero-crossing behavior. It is also sensitive to phonetic content and
  noise, so it is not a direct measure of speech authenticity.

### 7. Spectral Contrast

- **What it measures:** The difference between spectral peaks and valleys
  within frequency bands.
- **Information provided:** It describes how strongly spectral peaks stand out
  from nearby lower-energy regions. The implementation uses six octave-like
  bands plus the additional highest band returned by Librosa, and stores mean
  and standard deviation for each of the seven bands.
- **Why it may be useful:** Harmonic structure, spectral smoothing, and
  vocoder-like processing can affect peak-to-valley contrast. The measure is
  influenced by the sound being spoken and by recording conditions too.

### 8. Spectral Flatness

- **What it measures:** The ratio of the geometric mean to the arithmetic
  mean of spectral magnitudes in a frame.
- **Information provided:** Values nearer one indicate a more noise-like,
  spectrally even frame; lower values indicate a more concentrated or
  tone-like spectrum. SonicSense stores mean and standard deviation.
- **Why it may be useful:** Noise-like and harmonic components can differ in
  balance across recordings or synthesis methods. Speech naturally contains
  both kinds of sound, so flatness must be considered over many frames and with
  other features.

### 9. RMS Energy

- **What it measures:** Root-mean-square signal magnitude per frame.
- **Information provided:** It is a measure of short-time signal level.
  SonicSense stores mean and standard deviation of frame RMS.
- **Why it may be useful:** It can describe loudness variation, pauses, and
  amplitude consistency. These properties may reflect processing or
  normalization, but they are also affected by the speaker, microphone, and
  recording setup.

### 10. Fundamental Frequency (F0)

- **What it measures:** The estimated repetition rate of a periodic signal,
  corresponding to perceived pitch for voiced speech.
- **Information provided:** SonicSense uses Librosa's YIN estimator, searching
  from 60 to 500 Hz with a 1024-sample frame and 160-sample hop. It records
  mean and standard deviation of estimates in frames that pass its energy
  activity threshold.
- **Why it may be useful:** Pitch range and variation describe prosody, and
  some synthesis systems may produce different pitch patterns. Natural
  speakers also vary widely, and pitch estimates can be unreliable in
  unvoiced, noisy, or irregular sections.

### 11. Voiced Fraction

- **What it measures in this implementation:** The fraction of 1024-sample
  frames whose RMS energy is at least the greater of 0.005 or 3% of the
  recording's maximum frame RMS.
- **Information provided:** It approximates the proportion of frames above an
  amplitude activity threshold.
- **Why it may be useful:** Activity and pause patterns can contribute context
  when comparing recordings. **Despite its feature name, this calculation is
  not a true voicing detector**: an active frame may be unvoiced speech or
  noise, and a quiet voiced frame may fall below the threshold. Interpret it as
  an energy-based activity fraction, not a verified voiced/unvoiced ratio.

### 12. Onset Rate (syllabic-rate proxy)

- **What it measures:** The number of detected spectral onsets divided by the
  audio duration, in onsets per second.
- **Information provided:** Onsets are detected from a spectral-flux-like
  onset-strength envelope. The output is stored as `syllabic_rate_proxy`.
- **Why it may be useful:** Rhythm and timing patterns may provide context for
  comparing speech. However, an acoustic onset is not necessarily a syllable:
  the measure can count non-speech transients or miss syllabic boundaries.
  SonicSense does not perform transcription or syllable counting here.

## Interpretation and limitations

- These are acoustic descriptors, not stand-alone authenticity tests.
- Class differences in a dataset may come from speakers, microphones, codecs,
  languages, recording conditions, or dataset construction—not only from
  whether speech is synthetic.
- Mean and standard deviation summarize frame measurements and discard their
  original time order. They do not capture every temporal pattern.
- F0, activity, and onset estimates can be inaccurate for noisy, silent,
  clipped, very short, or non-speech recordings.
- The onset-rate proxy should not be reported as a measured speaking rate.
- The energy-threshold `voiced_fraction` should not be reported as a true
  voiced-frame detector without changing and validating the implementation.
- Detection performance must be established through evaluation on appropriate
  held-out data. This methodology makes no claim about accuracy.
