# Sentiment Reader — Transformer-based Sentiment Analysis

A sentiment analysis project using a fine-tuned **DistilBERT** model, served via
**FastAPI**, with a custom frontend.

---

## Why this project is structured this way (read this before your interview)

- **DistilBERT, not LSTM**: self-attention lets every word attend to every
  other word directly, regardless of distance — this matters for sentiment
  because the word that flips meaning ("but", "disappointing") is often far
  from what it modifies. DistilBERT is a distilled version of BERT — about
  40% smaller and 60% faster, while keeping ~97% of BERT's performance.
- **Fine-tuning, not training from scratch**: the model is pretrained on huge
  text corpora and already understands grammar and semantics. We only teach
  it the sentiment task on top of that, which needs far less data and compute
  than training a transformer from zero.
- **Fallback model**: the backend works immediately using a public pretrained
  sentiment model, even before you've run your own fine-tuning. This means you
  can demo it right away, then swap in your own fine-tuned model once it's
  ready — **you should actually do this part yourself** so you can speak to
  your own training run, your own data choices, and your own results.

---

## Project structure

```
sentiment-analyzer/
├── training/
│   └── finetune_distilbert.py   # Run this on Colab/Kaggle (free GPU) to train your own model
├── backend/
│   └── app.py                   # FastAPI server — serves predictions
├── frontend/
│   └── index.html               # Self-contained frontend (no build step needed)
├── requirements.txt
└── README.md
```

---

## How to run it

### 1. Backend (do this first)

```bash
cd backend
pip install -r ../requirements.txt
uvicorn app:app --reload --port 8000
```

This starts the API at `http://localhost:8000`. On first run, it'll
auto-download a public pretrained sentiment model (`distilbert-base-uncased-
finetuned-sst-2-english`) so it works without any training.

### 2. Frontend

Just open `frontend/index.html` directly in your browser (double-click it, or
use VS Code's "Open with Live Server"). It talks to the backend at
`http://localhost:8000` — change the `API_BASE` constant at the top of the
`<script>` tag if you deploy the backend elsewhere.

### 3. (Recommended) Fine-tune your own model

This is the step that makes the project genuinely yours, rather than "I wired
up someone else's model to a UI."

1. Open `training/finetune_distilbert.py` in Google Colab.
2. Runtime → Change runtime type → Hardware accelerator → **T4 GPU** (free tier).
3. Install dependencies (uncomment the `!pip install` line at the top).
4. Run all cells. By default it trains on a 5,000-example subset of IMDB for
   speed — bump this up (or swap in your own dataset, see below) for your
   real run.
5. Download the resulting `sentiment-model-final/` folder and place it inside
   `backend/` as `sentiment-model-final/` (matching `LOCAL_MODEL_PATH` in
   `app.py`).
6. Restart the backend — it will detect and load your fine-tuned model instead
   of the fallback.

#### Use your own data (strongly recommended — see the "Bring Your Own Data" section inside the training script)

Swap the IMDB dataset for something less generic: scraped product reviews,
app store reviews, or Hinglish/code-mixed text. Your CSV just needs `text`
and `label` (0 = negative, 1 = positive) columns. This is the single biggest
thing that separates this from a tutorial copy.

---

## API reference

**`POST /predict`**
```json
// Request
{ "text": "This product exceeded my expectations." }

// Response
{
  "text": "This product exceeded my expectations.",
  "label": "POSITIVE",
  "confidence": 0.9931,
  "probabilities": { "NEGATIVE": 0.0069, "POSITIVE": 0.9931 },
  "latency_ms": 14.2
}
```

**`POST /predict-batch`** — same shape, accepts `{ "texts": [...] }`, returns
a list of results.

**`GET /`** — health check, also reports whether you're running your own
fine-tuned model or the pretrained fallback.

---

## Things to actually understand before presenting this (likely interview questions)

1. **Why DistilBERT over a full BERT or an LSTM?** Speed/size tradeoff vs.
   accuracy; self-attention vs. sequential processing. (See top of this file.)
2. **What's "fine-tuning" vs "training from scratch"?** Be able to explain
   pretraining objectives (masked language modeling) vs. the downstream task
   head you added.
3. **How do you handle class imbalance** if your own dataset isn't 50/50?
   (`compute_metrics` in the training script reports precision/recall/F1, not
   just accuracy — know why accuracy alone is misleading on imbalanced data.)
4. **What happens with sarcasm or mixed sentiment?** Try the "mixed review"
   example chip in the UI and see how confident (or not) the model is —
   that's a great live demo moment in an interview.
5. **Why dynamic padding (`DataCollatorWithPadding`) instead of fixed-length
   padding?** Efficiency — you pad to the longest sequence *in the batch*,
   not to `MAX_LENGTH` every time.
6. **What would you change for production?** Batching requests, quantizing
   the model (e.g. ONNX/int8) for faster CPU inference, adding a confidence
   threshold for an "uncertain" class instead of forcing positive/negative on
   ambiguous text.

---

## Ideas to push this further (good "what would you add next" answers)

- Add a third **NEUTRAL** class, or move to 5-class (star rating) prediction.
- Add **aspect-based sentiment** (separately score "food", "service",
  "delivery" within one review) — meaningfully harder, less commonly attempted.
- Swap in **MuRIL** instead of DistilBERT if you go the Hinglish/code-mixed
  data route.
- Add a `/predict-batch` powered dashboard that ingests a CSV of reviews and
  shows sentiment trend over time.
