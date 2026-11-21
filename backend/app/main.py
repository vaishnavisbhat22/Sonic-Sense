"""FastAPI endpoints for SonicSense audio classification."""

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.inference import (
    AudioInputError,
    AudioProcessingError,
    FeatureSchemaError,
    InferenceExecutionError,
    MAX_UPLOAD_BYTES,
    ModelUnavailableError,
    load_model_and_metadata,
    predict_audio,
)


LOGGER = logging.getLogger(__name__)
app = FastAPI(title="SonicSense API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    """Report API and model readiness."""

    try:
        _, metadata = load_model_and_metadata()
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ok", "model_name": metadata["best_model"]}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    """Classify an uploaded audio recording."""

    try:
        audio_bytes = await audio.read(MAX_UPLOAD_BYTES + 1)
        return predict_audio(audio.filename, audio_bytes)
    except AudioInputError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=str(error),
        ) from error
    except AudioProcessingError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except InferenceExecutionError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except FeatureSchemaError as error:
        LOGGER.exception("Audio feature schema did not match the model.")
        raise HTTPException(
            status_code=500,
            detail="Audio features could not be matched to the detection model.",
        ) from error
    except Exception as error:
        LOGGER.exception("Unexpected error while handling an audio prediction.")
        raise HTTPException(
            status_code=500,
            detail="An unexpected prediction error occurred.",
        ) from error
    finally:
        await audio.close()
