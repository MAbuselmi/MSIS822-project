# Phase 1: Project Setup & Data Acquisition
import pandas as pd
from datasets import load_dataset

# Loading dataset
print("Original dataset")
ds = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")
print("Available splits:", list(ds.keys()))

split_name = list(ds.keys())[0]
df = ds[split_name].to_pandas()
print("Original dataset shape:", df.shape)

# Save original data to raw dataset
raw_path = "data/raw/arabic_generated_abstracts.csv"
df.to_csv(raw_path, index=False, encoding="utf-8-sig")
print(f" Raw dataset saved to {raw_path}")


# binary dataset (text, label)
# Human
human = df[["original_abstract"]].copy()
human = human.rename(columns={"original_abstract": "text"})
human["label"] = 0  # 0 for Human

# AI
ai_cols = [
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract"
]

ai_list = []
for col in ai_cols:
    if col in df.columns:
        temp = df[[col]].copy()
        temp = temp.rename(columns={col: "text"})
        temp["label"] = 1  # 1 for AI
        ai_list.append(temp)

ai = pd.concat(ai_list, ignore_index=True)

# Combine
final_df = pd.concat([human, ai], ignore_index=True)
print("Combined dataset shape:", final_df.shape)
print(final_df.head())

# Save processed dataset
processed_path = "data/processed/phase1_dataset.csv"
final_df.to_csv(processed_path, index=False, encoding="utf-8-sig")
print(f"Processed dataset saved to {processed_path}")


# Check for missing values, duplicates, and inconsistencies
print("\nClass Distribution:")
print(final_df["label"].value_counts())

# Missing values
print("\nMissing values per column:")
print(final_df.isnull().sum())

# Duplicates values
print("\nDuplicate rows:", final_df.duplicated(subset=["text"]).sum())

# inconsistencies
# Check for empty strings
empty_texts = final_df[final_df["text"].str.strip() == ""]
print("\nEmpty texts found:", empty_texts.shape[0])

# Check for non-Arabic characters
non_arabic_texts = final_df[~final_df["text"].str.contains(r'[\u0600-\u06FF]', regex=True, na=False)]
print("Non-Arabic/mixed texts found:", non_arabic_texts.shape[0])

# Check for unusually short texts
short_texts = final_df[final_df["text"].str.split().str.len() < 5]
print("Unusually short texts found:", short_texts.shape[0])

# End of Phase 1
