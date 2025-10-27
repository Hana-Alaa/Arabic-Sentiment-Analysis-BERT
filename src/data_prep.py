import os
import pandas as pd

base_dir = "data/raw"
data = []

for label_folder in ["Positive", "Negative"]:
    folder_path = os.path.join(base_dir, label_folder)
    label = 1 if label_folder.lower() == "positive" else 0
    print(f"📂 Reading folder: {folder_path}, Label: {label}")
    for file in os.listdir(folder_path):
      if file.endswith(".txt"):
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
            if text:
                data.append({"tweet": text, "label": label})
            else:
                print(f"⚠️ Skipped empty file: {file_path}")


df = pd.DataFrame(data)
print(df["label"].value_counts())  
df.to_csv("data/processed/arabic_tweets_labeled.csv", index=False, encoding="utf-8")
print("✅ Saved as data/processed/arabic_tweets_labeled.csv")
