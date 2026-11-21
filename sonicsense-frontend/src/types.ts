export type Prediction = {
  filename: string;
  label: "human" | "synthetic";
  confidence: number;
  probabilities: { human: number; synthetic: number };
  processing_time_ms?: number;
  model?: string;
  analyzed_at: string;
};

export type ApiPrediction = {
  prediction: "human" | "synthetic";
  confidence: number;
  human_probability: number;
  synthetic_probability: number;
  model_name: string;
  processing_time_ms: number;
};
