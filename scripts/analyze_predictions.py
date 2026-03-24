"""
Post-hoc analysis of DistilBERT predictions.
Produces three CSVs used in the paper's discussion section:

  results/error_analysis.csv      — misclassified segments (FP + FN per label)
  results/label_by_speaker.csv    — mean label frequency per speaker
  results/label_by_year.csv       — mean label frequency per year (temporal trend)

Reads predictions from the most recent distilbert results folder so you
don't need to update paths manually.

Run:
    python scripts/analyze_predictions.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd

LABEL_COLS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]

# ── locate latest DistilBERT results ─────────────────────────────────────────
folders = sorted(glob.glob("results/distilbert_*/"))
if not folders:
    raise FileNotFoundError("No results/distilbert_* folder found. Run train_distilBERT.py first.")
results_dir = folders[-1]
print(f"Using results from: {results_dir}")

# ── load test predictions (model output) ─────────────────────────────────────
preds_df = pd.read_csv(os.path.join(results_dir, "test_predictions.csv"))

# ── load full annotated dataset to get ground-truth + metadata ───────────────
full_df = pd.read_csv("dataset/dataset_annotated_final.csv").dropna(subset=["text"])
full_df[LABEL_COLS] = full_df[LABEL_COLS].fillna(0).astype(int)

# ── align predictions with ground truth via text match ───────────────────────
# test_predictions.csv contains the text column — join on it
merged = preds_df.merge(
    full_df[["text", "speaker", "year"] + LABEL_COLS],
    on="text",
    how="left",
    suffixes=("_pred", "_true"),
)

os.makedirs("results", exist_ok=True)

# ── 1. Error analysis ─────────────────────────────────────────────────────────
errors = []
for col in LABEL_COLS:
    pred_col = f"{col}_pred"
    true_col = f"{col}_true"
    if pred_col not in merged.columns or true_col not in merged.columns:
        # fallback when suffixes weren't added (no conflict)
        pred_col = col
        true_col = col
        continue
    wrong = merged[merged[pred_col] != merged[true_col]].copy()
    for _, row in wrong.iterrows():
        errors.append({
            "label":    col,
            "error_type": "FP" if row[pred_col] == 1 else "FN",
            "pred":     int(row[pred_col]),
            "true":     int(row[true_col]),
            "speaker":  row.get("speaker", ""),
            "year":     row.get("year", ""),
            "text":     row["text"],
        })

# simpler fallback: use the raw pred columns vs columns named identically
if not errors:
    pred_cols = [c for c in merged.columns if c in LABEL_COLS and f"{c}_true" not in merged.columns]
    # after merge with suffixes, cols should be label_pred / label_true
    # handle the case where preds_df already had the label cols named plainly
    for col in LABEL_COLS:
        pc = f"{col}_pred" if f"{col}_pred" in merged.columns else col
        tc = f"{col}_true" if f"{col}_true" in merged.columns else col
        if pc == tc:
            continue
        wrong = merged[merged[pc] != merged[tc]].copy()
        for _, row in wrong.iterrows():
            errors.append({
                "label":      col,
                "error_type": "FP" if row[pc] == 1 else "FN",
                "pred":       int(row[pc]),
                "true":       int(row[tc]),
                "speaker":    row.get("speaker", ""),
                "year":       row.get("year", ""),
                "text":       row["text"],
            })

error_df = pd.DataFrame(errors)
error_df.to_csv("results/error_analysis.csv", index=False)
print(f"Error analysis: {len(error_df)} misclassifications across all labels")
if not error_df.empty:
    print(error_df.groupby(["label", "error_type"]).size().to_string())

# ── 2. Label frequency by speaker (ground truth on full dataset) ──────────────
if "speaker" in full_df.columns:
    by_speaker = (
        full_df.groupby("speaker")[LABEL_COLS]
        .agg(["mean", "sum", "count"])
    )
    # flatten multi-level columns
    by_speaker.columns = ["_".join(c) for c in by_speaker.columns]
    by_speaker = by_speaker.round(3).reset_index()
    by_speaker.to_csv("results/label_by_speaker.csv", index=False)
    print(f"\nLabel by speaker: {len(by_speaker)} speakers → results/label_by_speaker.csv")

    # compact summary (just means) for quick inspection
    means = full_df.groupby("speaker")[LABEL_COLS].mean().round(3)
    print("\nTop 10 speakers by total tactic use:")
    means["total"] = means.sum(axis=1)
    print(means.sort_values("total", ascending=False).head(10)[LABEL_COLS].to_string())
else:
    print("No 'speaker' column found — skipping speaker analysis.")

# ── 3. Label frequency by year (temporal trend) ──────────────────────────────
if "year" in full_df.columns:
    full_df["year"] = pd.to_numeric(full_df["year"], errors="coerce")
    by_year = (
        full_df.dropna(subset=["year"])
        .groupby("year")[LABEL_COLS]
        .mean()
        .round(3)
        .reset_index()
    )
    by_year.to_csv("results/label_by_year.csv", index=False)
    print(f"\nLabel by year: {len(by_year)} years → results/label_by_year.csv")
    print(by_year.tail(10).to_string(index=False))
else:
    print("No 'year' column found — skipping temporal analysis.")

print("\nDone. All analysis saved to results/")