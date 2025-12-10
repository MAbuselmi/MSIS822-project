import os
import pandas as pd
from datasets import load_dataset

# 1) Project Structure
#folders = [
#    "data/raw",
#    "data/processed",
#    "src",
#    "notebooks",
#    "models",
#    "results",
#    "reports",
#    ]

#files = {
# "src/preprocessing.py": "# Preprocessing functions will go here\n",
# "src/eda.py": "# EDA helper functions will go here\n",
# "notebooks/phase1.ipynb": "",
# "notebooks/phase2.ipynb": "",
# "notebooks/phase3.ipynb": "",
# "notebooks/phase4.ipynb": "",
# "notebooks/phase5.ipynb": "",
# "README.md": "# MSIS822 Project\n\nThis project detects AI-generated Arabic abstracts.\n",
# }

# 2) Create folders
# for folder in folders:
# os.makedirs(folder, exist_ok=True)
# print(f"Created folder: {folder}")

# 3) Create files
# for file_path, content in files.items():
# if not os.path.exists(file_path):
# with open(file_path, "w", encoding="utf-8") as f:
# f.write(content)
# print(f"Created file: {file_path}")
# else:
# print(f"File already exists: {file_path}")

# Phase1

ds = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")
print("Available splits:", list(ds.keys()))

dfs = []
for split in ds.keys():
    temp_df = ds[split].to_pandas()
    temp_df["split_name"] = split
    dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True)

print("Combined dataset shape:", df.shape)
df.head()

print("Columns:\n", df.columns.tolist())

print("\nDataFrame info:")
df.info()

raw_path = "data/raw/arabic_abstracts_raw.csv"
df.to_csv(raw_path, index=False, encoding="utf-8-sig")
print("Raw dataset saved to:", raw_path)

# Human abstracts
human = df[["original_abstract"]].copy()
human = human.rename(columns={"original_abstract": "text"})
human["label"] = 0  # 0 = Human

# AI abstracts
ai_cols = [
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract"
]

ai_frames = []
for col in ai_cols:
    if col in df.columns:
        temp = df[[col]].copy()
        temp = temp.rename(columns={col: "text"})
        temp["label"] = 1  # 1 = AI
        ai_frames.append(temp)

ai = pd.concat(ai_frames, ignore_index=True)

# Combine human + AI
final_df = pd.concat([human, ai], ignore_index=True)[["text", "label"]]

print("Final binary dataset shape:", final_df.shape)
final_df.head()

phase1_path = "data/processed/phase1_dataset.csv"
final_df.to_csv(phase1_path, index=False, encoding="utf-8-sig")
print("Phase 1 dataset saved to:", phase1_path)

print("Class distribution (counts):")
print(final_df["label"].value_counts())

print("\nClass distribution (proportions):")
print(final_df["label"].value_counts(normalize=True))

# Missing values
print("Missing values per column:")
print(final_df.isnull().sum())

# Duplicate texts
dup_count = final_df.duplicated(subset=["text"]).sum()
print("\nNumber of duplicate texts:", dup_count)

# Empty texts
empty_mask = final_df["text"].astype(str).str.strip() == ""
empty_count = empty_mask.sum()
print("Number of empty texts:", empty_count)

# Non-Arabic / mixed texts
non_arabic_mask = ~final_df["text"].astype(str).str.contains(r"[\u0600-\u06FF]", regex=True, na=False)
non_arabic_count = non_arabic_mask.sum()
print("Number of non-Arabic/mixed texts:", non_arabic_count)

# Very short texts
word_counts = final_df["text"].astype(str).str.split().str.len()
short_mask = word_counts < 5
short_count = short_mask.sum()
print("Number of unusually short texts (< 5 words):", short_count)