# SonicSense

SonicSense classifies an audio recording as **human** or **synthetic** using a
trained XGBoost model. This repository includes a FastAPI backend, a React/Vite
dashboard in `sonicsense-frontend/`, and the Streamlit client in `app.py`. The
backend reuses the project feature extractor in `extract_audio_features.py`
and does not retrain or modify the model.

## Requirements and installation

Use Python 3.12 or later. From the repository root, create and activate a
virtual environment, then install the backend and test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

The requirements include FastAPI/Uvicorn and multipart upload support,
Streamlit's HTTP client (`requests`), the audio stack (`librosa`, `soundfile`,
PyAV, NumPy, SciPy), model dependencies (XGBoost, scikit-learn, pandas, joblib),
and the test runner. PyAV provides a fallback decoder for M4A/MP4 audio;
SoundFile remains the primary decoder for formats it supports.

The backend expects the checked-in model and metadata at:

- `models/xgboost.joblib`
- `results/model_metadata.json`

The processed feature CSV is only needed for labeled dataset inference tests;
it is not needed to run the API.

## Start the backend

Run this command from the repository root:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`; interactive OpenAPI
documentation is at `http://127.0.0.1:8000/docs`.

### API

`GET /health` checks model readiness and returns the status and model name.

`POST /predict` accepts one multipart upload with field name `audio`. Supported
filename extensions are WAV, FLAC, MP3, OGG, and M4A; uploads are limited to
25 MiB. SoundFile decodes WAV, FLAC, MP3, and OGG; PyAV handles M4A/MP4.

Example:

```powershell
curl.exe -F "audio=@recording.wav" http://127.0.0.1:8000/predict
```

The response contains `prediction` (`human` or `synthetic`), `confidence`,
`human_probability`, `synthetic_probability`, `model_name`, and
`processing_time_ms`. Confidence and class probabilities are fractions from 0
to 1, not percentages. Class IDs remain `0 = synthetic` and `1 = human`;
probability positions are read using the model's actual `classes_` values.
Invalid file types and empty files return 4xx responses; undecodable audio
returns 422, unavailable artifacts return 503, and inference failures are
logged and returned as server errors without internal exception details.

The model expects exactly the 134 columns and order in
`results/model_metadata.json`. The API validates the metadata against the
shared feature extractor and the loaded model before serving predictions.

### Training and inference preprocessing

The model's 134-feature path is `extract_audio_features.py`, shared by feature
generation and uploaded-audio inference. It decodes samples, averages
multichannel audio to mono, and resamples to 16 kHz before calculating the
features. It does not denoise, peak-normalize, trim silence, or split uploads
into 5-second segments. `preprocess_audio.py` is a separate dataset utility
and is not invoked by this model's feature-generation or inference path.
Applying those extra steps at inference would change the model inputs and
requires retraining and evaluation first.

## Start the React frontend

From the repository root, enter the React/Vite project directory:

```powershell
Set-Location sonicsense-frontend
npm ci
Copy-Item .env.example .env
npm run dev
```

Open `http://localhost:5173`. Configure the backend URL in `.env` using
`VITE_API_BASE_URL=http://127.0.0.1:8000`. The backend allows this development
origin through CORS. Build the production assets with `npm run build`.

The React client supports audio upload and microphone recording. Recordings
are submitted as WAV using the same `POST /predict` API contract described
above.

## Start the optional Streamlit client

In a second terminal, with the virtual environment active:

```powershell
streamlit run app.py
```

The Streamlit interface sends uploads to `http://127.0.0.1:8000/predict` and
displays the returned class, confidence, class probabilities, model name, and
processing time. To use a different backend URL, set
`SONICSENSE_API_URL` to the full prediction URL before starting Streamlit:

```powershell
$env:SONICSENSE_API_URL = "http://127.0.0.1:8000/predict"
streamlit run app.py
```

## Testing

Run the backend and existing repository tests from the project root:

```powershell
python -m pytest -q
```

`backend/app/test_inference.py` exercises the HTTP API with generated test
audio, invalid/empty uploads, model readiness, and feature-schema validation.
`test_audio_decoder_parity.py` also checks mono/stereo MP3, OGG, and M4A
decoding. Generated test waveforms are pipeline tests, **not** labeled human or
synthetic speech recordings. From the React project directory, run
`npm run build` to type-check and build the frontend.

For five held-out feature rows of each actual class, run:

```powershell
python src\models\test_inference.py
```

That dataset inference command reports each sample's actual label, prediction,
and confidence. The feature CSV is not checked into this repository; the
script uses `data/processed/audio_features_v2.csv` when present, otherwise it
tries the dataset path recorded in the model metadata. It fails clearly if
neither exists. The model's recorded evaluation results are in `results/`;
they are not a guarantee of future or real-world performance.

## Known limitations

- Raw ASVspoof/FakeAVCeleb audio is not included in this checkout. The feature
  CSV referenced by the saved metadata is outside this repository, so the
  original dataset preprocessing cannot be reproduced here without obtaining
  the source data and updating its paths.
- CORS is restricted to local development origins. Before deployment, add the
  exact frontend origin to the backend allowlist; wildcard production CORS is
  not enabled.
- The saved evaluation used a stratified random split. Speaker/source
  identifiers were unavailable, so it does not establish speaker-independent
  or source-independent generalization.
- Performance can vary with codecs, microphones, noise, accents, speakers,
  and synthetic speech generators not represented in training data.
- The output probabilities are model scores and are not documented as
  calibrated probabilities.
- A predicted class is not proof of whether a recording is genuine. Treat the
  result as an automated estimate, not a definitive authenticity judgment.
