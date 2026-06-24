"""
Fine-tune DistilBERT for Sentiment Analysis
=============================================
Run this on Google Colab (free GPU: Runtime -> Change runtime type -> T4 GPU)
or Kaggle Notebooks.

This script fine-tunes a pretrained DistilBERT model on a sentiment dataset
(default: IMDB reviews, but swap in your own CSV easily - see "BRING YOUR OWN
DATA" section below for using Amazon/product reviews, Twitter data, etc.)

WHY DISTILBERT (be ready to explain this in interviews):
- It's a distilled (compressed) version of BERT - roughly 40% smaller, 60%
  faster, while retaining ~97% of BERT's performance on most NLP tasks.
- We fine-tune (adapt a pretrained model) rather than train from scratch,
  because the model already understands language structure/grammar/semantics
  from pretraining on huge corpora. We just teach it the sentiment task.

Install dependencies first (uncomment if running in Colab):
!pip install transformers datasets torch scikit-learn accelerate -q
"""

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2          # 0 = negative, 1 = positive. Change to 3+ for neutral/fine-grained.
MAX_LENGTH = 256        # truncate reviews longer than this many tokens
OUTPUT_DIR = "./sentiment-distilbert-finetuned"
BATCH_SIZE = 16
EPOCHS = 2              # 2-3 is usually enough for fine-tuning; more risks overfitting
LEARNING_RATE = 2e-5    # standard fine-tuning LR for BERT-family models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: No GPU detected. This will be very slow. "
          "In Colab: Runtime > Change runtime type > Hardware accelerator > T4 GPU")

# ----------------------------------------------------------------------------
# STEP 1: LOAD DATA
# ----------------------------------------------------------------------------
# Default: IMDB movie reviews (50k labeled examples, balanced classes)
print("Loading dataset...")
dataset = load_dataset("imdb")

# Use a subset for faster iteration while you're testing the pipeline.
# Comment these two lines out to train on the FULL dataset for your final run.
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(5000))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

# ----------------------------------------------------------------------------
# BRING YOUR OWN DATA (recommended - this is what makes the project "yours")
# ----------------------------------------------------------------------------
# Replace the load_dataset call above with your own CSV for a less-saturated
# project (e.g. scraped product reviews, Hinglish tweets, app store reviews).
# Your CSV needs two columns: "text" and "label" (0 = negative, 1 = positive).
#
# from datasets import Dataset
# import pandas as pd
# df = pd.read_csv("your_data.csv")
# full_dataset = Dataset.from_pandas(df)
# split = full_dataset.train_test_split(test_size=0.2, seed=42)
# dataset = {"train": split["train"], "test": split["test"]}
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# STEP 2: TOKENIZE
# ----------------------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        # No fixed padding here - the DataCollator pads dynamically per-batch,
        # which is more efficient than padding every example to MAX_LENGTH.
    )


print("Tokenizing...")
tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------------------------------------------------------
# STEP 3: LOAD MODEL
# ----------------------------------------------------------------------------
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
).to(device)


# ----------------------------------------------------------------------------
# STEP 4: METRICS
# ----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ----------------------------------------------------------------------------
# STEP 5: TRAIN
# ----------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,          # regularization to reduce overfitting
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    report_to="none",          # disable wandb/etc auto-logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# ----------------------------------------------------------------------------
# STEP 6: EVALUATE
# ----------------------------------------------------------------------------
print("\nFinal evaluation on test set:")
metrics = trainer.evaluate()
print(metrics)

# ----------------------------------------------------------------------------
# STEP 7: SAVE MODEL (download this folder from Colab, or push to HuggingFace Hub)
# ----------------------------------------------------------------------------
FINAL_MODEL_DIR = "./sentiment-model-final"
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"\nModel saved to {FINAL_MODEL_DIR}")
print("In Colab: zip this folder and download it, or push to HuggingFace Hub "
      "with model.push_to_hub('your-username/your-model-name')")

# ----------------------------------------------------------------------------
# QUICK SANITY CHECK
# ----------------------------------------------------------------------------
print("\nQuick sanity check on a few examples:")
test_sentences = [
    "This movie was absolutely fantastic, I loved every minute of it!",
    "Waste of time. The plot made no sense and the acting was terrible.",
    "It started promising but became disappointing by the end.",
]
model.eval()
for sent in test_sentences:
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    confidence = torch.softmax(logits, dim=-1).max().item()
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    print(f"  [{label} ({confidence:.1%})] {sent}")
