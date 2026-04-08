import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# =========================
# CONFIG
# =========================
INPUT_PATH = "../results/captum_attributions/captum_small_run/captum_results_2.csv"
OUTPUT_PATH = "../results/language_analysis/captum_tokens_summary.csv"

os.makedirs("../results/language_analysis", exist_ok=True)

# =========================
# LOAD
# =========================
df = pd.read_csv(INPUT_PATH)

# Remove unreliable attributions
df = df[df["delta_flagged"] == False]

# Rename for consistency
df = df.rename(columns={"sentence_id": "id"})

# =========================
# CLEAN TOKENS
# =========================
df = df[
    (df["token"].notna()) &
    (~df["token"].isin(["[CLS]", "[SEP]", ".", ","])) &
    (df["token"].str.len() > 2)
]

# Lowercase tokens for consistency
df["token"] = df["token"].str.lower()

# Remove stopwords
df = df[~df["token"].isin(ENGLISH_STOP_WORDS)]

# =========================
# IMPORTANCE
# =========================
df["importance"] = df["abs_attribution"]

# =========================
# AGGREGATE PER LABEL AND TOKEN
# =========================
results = []

for label in df["label"].unique():
    subset = df[df["label"] == label]
    total_sentences = subset["id"].nunique()

    token_stats = subset.groupby("token").agg(
        mean_importance=("importance", "mean"),
        frequency=("id", "nunique")
    )

    # Filter rare tokens (reduce noise)
    token_stats = token_stats[token_stats["frequency"] >= 3]

    # New scoring (less biased toward frequency)
    token_stats["weighted_importance"] = (
        token_stats["mean_importance"] * np.log1p(token_stats["frequency"])
    )

    token_stats = token_stats.sort_values("weighted_importance", ascending=False)

    for token, row in token_stats.iterrows():
        frequency_pct = row["frequency"] / total_sentences

        results.append({
            "label": label,
            "token": token,
            "mean_importance": row["mean_importance"],
            "frequency": row["frequency"],
            "frequency_pct": frequency_pct,
            "weighted_importance": row["weighted_importance"]
        })

# =========================
# SAVE
# =========================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved summarized token info to: {OUTPUT_PATH}")