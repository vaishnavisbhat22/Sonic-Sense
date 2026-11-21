# SonicSense Frontend

React + Vite + TypeScript dashboard for the SonicSense synthetic speech detection project.

## 1. Requirements
- Node.js 18 or newer
- The SonicSense FastAPI backend

## 2. Install and run

```powershell
cd sonicsense-frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://localhost:5173`.

For production build:

```powershell
npm run build
npm run preview
```

## 3. Backend API contract

The frontend sends an audio file to:

`POST http://127.0.0.1:8000/predict`

Content type: `multipart/form-data`

Form field: `audio`

Expected JSON response:

```json
{
  "prediction": "synthetic",
  "confidence": 0.97,
  "human_probability": 0.03,
  "synthetic_probability": 0.97,
  "processing_time_ms": 142,
  "model_name": "XGBoost"
}
```

`confidence`, `human_probability`, and `synthetic_probability` are backend-calculated fractions from 0 to 1. Do not hardcode or simulate them.

### Critical model integration requirement

The trained XGBoost model expects **134 numerical features in the exact feature order recorded in `results/model_metadata.json`**. The API must use the same preprocessing and feature extraction implementation used to create `audio_features_v2.csv`. The model does not accept raw WAV/MP3/FLAC files directly.

The backend should:
1. Accept and validate the uploaded audio.
2. Apply the project's established audio preprocessing.
3. Extract the exact 134 model features, in the saved metadata order.
4. Load `models/xgboost.joblib` and predict.
5. Map numeric class `0` to `synthetic` and `1` to `human`.
6. Return the response schema above.

Do not build a new feature extractor by guessing feature formulas. Reuse Member 2's implementation.

## 4. Configure backend URL

Edit `.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Restart Vite after changing `.env`.

## 5. CORS

The FastAPI backend is configured to allow the Vite development origins
`http://localhost:5173` and `http://127.0.0.1:5173` for `GET`, `POST`, and
`OPTIONS` requests and the headers required for multipart uploads.

## 6. Notes
- The history shown is session-only and resets on refresh. Persistent history requires backend storage.
- Supported upload extensions: WAV, FLAC, MP3, OGG, and M4A; the backend validates and decodes uploads using the shared project audio pipeline.
- The frontend and backend enforce a 25 MiB upload limit.

## 7. Microphone recording

On the Analyze audio page, choose **Start recording**, allow microphone access,
then select **Stop recording**. Review the WAV playback, discard and re-record
if needed, or run detection on the recording. MediaRecorder controls recording;
Web Audio captures PCM and packages it as WAV so the upload does not depend on
the browser's support for decoding its MediaRecorder codec. The FastAPI backend
then applies its existing audio decoding and inference pipeline. Microphone
access requires a secure browser context (localhost is supported) and permission
from the user.

## 8. Testing

```powershell
npm run build
```

Run backend tests from the repository root using the backend's documented test
command. Browser microphone tests also require an available microphone or a
browser test environment that provides a media device.
