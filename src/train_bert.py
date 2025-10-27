import os
import json
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

# ---------------- Config ----------------
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
TRAIN_CSV = "data/processed/train.csv"
TEST_CSV = "data/processed/test.csv"
OUTPUT_DIR = "models/arabertv2-sentiment"
BATCH_SIZE = 8
NUM_EPOCHS = 4
MAX_LENGTH = 128
SEED = 42
# ----------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# 1. Load CSV into HuggingFace Dataset
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(
        examples["tweet"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

remove_cols = ["tweet"]
if "__index_level_0__" in train_ds.column_names:
    remove_cols.append("__index_level_0__")

train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=remove_cols)
test_ds = test_ds.map(preprocess_function, batched=True, remove_columns=remove_cols)

# 3. Data collator (dynamic padding)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 5. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
    }

# 6. TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to=[],  # disables wandb, tensorboard, etc.
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # حذف EarlyStoppingCallback نهائيًا
)


# 8. Train
trainer.train()

# 9. Save final artifacts
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 10. Final evaluation on test and save metrics
metrics = trainer.evaluate(eval_dataset=test_ds)
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("✅ Training complete. Model and tokenizer saved to:", OUTPUT_DIR)
print("📊 Metrics:", metrics)