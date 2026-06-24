"""
FastAPI Backend for Sentiment Analysis
========================================
Serves sentiment predictions using a transformer model.

- If a fine-tuned model exists at MODEL_PATH (from running finetune_distilbert.py
  and copying the output folder here), it loads that.
- Otherwise, it falls back to a public pretrained sentiment model from
  HuggingFace, so the app works out of the box while you train your own.

Run with:
    pip install fastapi uvicorn transformers torch
    uvicorn app:app --reload --port 8000
"""

import os
import time

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Path to your own fine-tuned model (output of finetune_distilbert.py).
# If this folder doesn't exist, we fall back to a public pretrained model.
LOCAL_MODEL_PATH = "./sentiment-model-final"

# Public fallback: a DistilBERT fine-tuned on SST-2 (a standard sentiment
# benchmark). This means the app works immediately, even before you've run
# your own training script.
FALLBACK_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------
# LOAD MODEL (once, at startup)
# ----------------------------------------------------------------------------
if os.path.exists(LOCAL_MODEL_PATH):
    print(f"Loading YOUR fine-tuned model from {LOCAL_MODEL_PATH}")
    MODEL_SOURCE = "custom-fine-tuned"
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH).to(device)
else:
    print(f"No local model found at {LOCAL_MODEL_PATH}. "
          f"Falling back to public pretrained model: {FALLBACK_MODEL_NAME}")
    print("(Run training/finetune_distilbert.py and copy its output here "
          "to use your own fine-tuned model.)")
    MODEL_SOURCE = "pretrained-fallback"
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL_NAME).to(device)

model.eval()
LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}

# ----------------------------------------------------------------------------
# APP SETUP
# ----------------------------------------------------------------------------
app = FastAPI(title="Sentiment Analysis API")

# Allow the frontend (served from a different origin/file) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


class BatchTextInput(BaseModel):
    texts: list[str]


def predict_single(text: str) -> dict:
    start = time.time()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    pred_idx = int(torch.argmax(probs).item())
    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "text": text,
        "label": LABEL_MAP.get(pred_idx, str(pred_idx)),
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            LABEL_MAP.get(i, str(i)): round(float(p), 4) for i, p in enumerate(probs.tolist())
        },
        "latency_ms": latency_ms,
    }


@app.get("/")
def root():
    return {
        "status": "running",
        "model_source": MODEL_SOURCE,
        "device": device,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(input: TextInput):
    if not input.text.strip():
        return {"error": "Text cannot be empty"}
    return predict_single(input.text)


@app.post("/predict-batch")
def predict_batch(input: BatchTextInput):
    return {"results": [predict_single(t) for t in input.texts if t.strip()]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
