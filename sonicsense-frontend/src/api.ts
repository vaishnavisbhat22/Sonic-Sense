import axios from "axios";
import type { ApiPrediction, Prediction } from "./types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
  timeout: 120000
});

/**
 * Backend contract:
 * POST /predict
 * multipart/form-data field: audio
 * JSON response example:
 * {
 *   "prediction": "synthetic",
 *   "confidence": 0.97,
 *   "human_probability": 0.03,
 *   "synthetic_probability": 0.97,
 *   "processing_time_ms": 142,
 *   "model_name": "XGBoost"
 * }
 *
 * The backend must perform the actual preprocessing and feature extraction
 * before calling the saved XGBoost model. The frontend never fabricates results.
 */
export async function analyzeAudio(file: File): Promise<Prediction> {
  const body = new FormData();
  body.append("audio", file);

  const { data } = await api.post<ApiPrediction>("/predict", body);

  const label = data.prediction;
  if (!["human", "synthetic"].includes(label)) {
    throw new Error("The API response did not contain a valid prediction label.");
  }
  const probabilities = [
    data.confidence,
    data.human_probability,
    data.synthetic_probability,
  ];
  if (
    probabilities.some(value => !Number.isFinite(value) || value < 0 || value > 1) ||
    !Number.isFinite(data.processing_time_ms) ||
    data.processing_time_ms < 0 ||
    typeof data.model_name !== "string"
  ) {
    throw new Error("The API response contains invalid prediction values.");
  }
  const predictedProbability = label === "human"
    ? data.human_probability
    : data.synthetic_probability;
  if (
    Math.abs(data.human_probability + data.synthetic_probability - 1) > 1e-5 ||
    Math.abs(data.confidence - predictedProbability) > 1e-5
  ) {
    throw new Error("The API response contains inconsistent probabilities.");
  }

  return {
    filename: file.name,
    label,
    confidence: data.confidence,
    probabilities: {
      human: data.human_probability,
      synthetic: data.synthetic_probability,
    },
    processing_time_ms: data.processing_time_ms,
    model: data.model_name,
    analyzed_at: new Date().toISOString()
  };
}
