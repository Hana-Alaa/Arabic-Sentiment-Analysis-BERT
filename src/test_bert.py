import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os

# ---------------- Config ----------------
MODEL_DIR = "models/arabertv2-sentiment"  # Path to the trained model directory
TEST_CSV = "data/processed/test.csv"       # Path to the test CSV file
MAX_LENGTH = 128                           # Maximum token length for tokenizer
# ----------------------------------------

# 1. Load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)          # Load tokenizer from model folder
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)  # Load model
model.eval()  # Set model to evaluation mode (disable training-specific layers like dropout)

# 2. Load test dataset
test_df = pd.read_csv(TEST_CSV)                   # Read CSV into pandas DataFrame
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))  # Convert DataFrame to HuggingFace Dataset

# 3. Preprocess function for tokenization
def preprocess_function(examples):
    return tokenizer(
        examples["tweet"],       # Tokenize the "tweet" column
        truncation=True,         # Truncate longer sequences
        max_length=MAX_LENGTH,   # Maximum sequence length
        padding=False            # Do not pad here, padding will be done dynamically
    )

# Remove unnecessary columns
remove_cols = ["tweet"]
if "__index_level_0__" in test_ds.column_names:
    remove_cols.append("__index_level_0__")

# Apply preprocessing to dataset
test_ds = test_ds.map(preprocess_function, batched=True, remove_columns=remove_cols)

# 4. Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. Prepare DataLoader
from torch.utils.data import DataLoader

test_loader = DataLoader(
    test_ds,
    batch_size=8,          # Number of samples per batch
    collate_fn=data_collator  # Use collator for dynamic padding
)

# 6. Prediction loop
all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient calculation for inference
    for batch in test_loader:
        labels = batch.pop("labels") if "labels" in batch else None  # Extract labels if exist
        batch = {k: v for k, v in batch.items()}                     # Prepare input batch
        outputs = model(**batch)                                     # Forward pass
        preds = torch.argmax(outputs.logits, dim=-1)                 # Get predicted class
        all_preds.extend(preds.cpu().numpy())                        # Collect predictions
        if labels is not None:
            all_labels.extend(labels.cpu().numpy())                 # Collect true labels

# 7. Evaluate performance if labels exist
if all_labels:
    acc = accuracy_score(all_labels, all_preds)  # Accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )  # Compute precision, recall, f1
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    print("📊 Test Metrics:", metrics)  # Print evaluation metrics
    # Save metrics to JSON
    with open(os.path.join(MODEL_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
else:
    print("⚠️ Test labels not found, only predictions saved.")

# 8. Save predictions
test_df["preds"] = all_preds  # Add predictions to DataFrame
test_df.to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False)  # Save CSV
print("✅ Test predictions saved to:", os.path.join(MODEL_DIR, "test_predictions.csv"))