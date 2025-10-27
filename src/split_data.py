import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the cleaned dataset
df = pd.read_csv("data/processed/arabic_tweets_cleaned.csv")

# 2. Split the data (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# 3. Save the splits
train_df.to_csv("data/processed/train.csv", index=False, encoding="utf-8")
test_df.to_csv("data/processed/test.csv", index=False, encoding="utf-8")

# 4. Print dataset info
print("✅ Data successfully split and saved!")
print(f"🧠 Train set: {len(train_df)} samples")
print(f"🧪 Test set: {len(test_df)} samples\n")

# 5. Print label distribution
print("🔹 Train label distribution:")
print(train_df["label"].value_counts())

print("\n🔹 Test label distribution:")
print(test_df["label"].value_counts())
