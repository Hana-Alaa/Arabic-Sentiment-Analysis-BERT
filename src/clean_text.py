import pandas as pd
import re

# 1. Load the labeled tweets dataset
df = pd.read_csv("data/processed/arabic_tweets_labeled.csv")

# 2. Define a function to clean Arabic text
def clean_arabic_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove mentions and hashtags
    text = re.sub(r"[@#]\S+", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove Arabic diacritics (tashkeel)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # Remove emojis and non-Arabic symbols
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # Remove excessive repeated characters (like حلووووي → حلووي)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 3. Clean the text
df["clean_tweet"] = df["tweet"].astype(str).apply(clean_arabic_text)

# 4. Keep only non-empty tweets
df = df[df["clean_tweet"].str.strip() != ""]

# 5. Keep only relevant columns
df = df[["clean_tweet", "label"]].rename(columns={"clean_tweet": "tweet"})

# 6. Save the cleaned data
df.to_csv("data/processed/arabic_tweets_cleaned.csv", index=False, encoding="utf-8")

print("✅ Cleaned data saved successfully!")
print(f"🧹 Remaining tweets after cleaning: {len(df)}")
print(df.head())
