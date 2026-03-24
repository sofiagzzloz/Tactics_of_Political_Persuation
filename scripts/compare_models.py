"""
Compares the latest baseline run against the latest DistilBERT run
and produces results/model_comparison.csv for the paper.

Automatically picks the most recent timestamped folder under results/
for each model type, so you don't need to update paths manually.

Run:
    python scripts/compare_models.py
"""

import os
import glob
import json
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

LABEL_COLS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]

# ── find most recent result folders ──────────────────────────────────────────
def latest_results(prefix: str) -> str:
    folders = sorted(glob.glob(f"results/{prefix}_*/"))
    if not folders:
        raise FileNotFoundError(
            f"No results folder matching results/{prefix}_* found. "
            f"Have you run the corresponding script yet?"
        )
    return folders[-1]

bert_dir     = latest_results("distilbert")
baseline_dir = latest_results("baseline")
print(f"DistilBERT results : {bert_dir}")
print(f"Baseline results   : {baseline_dir}")

with open(os.path.join(bert_dir,     "test_metrics.json")) as f:
    bert_agg = json.load(f)
with open(os.path.join(baseline_dir, "test_metrics.json")) as f:
    base = json.load(f)

# ── recompute per-label DistilBERT metrics from saved predictions ─────────────
# (train_distilBERT.py saves predictions but not per-label F1 to JSON)
bert_preds_path = os.path.join(bert_dir, "test_predictions.csv")
full_df   = pd.read_csv("dataset/dataset_annotated_final.csv").dropna(subset=["text"])
bert_preds = pd.read_csv(bert_preds_path)

# align ground truth via text match
merged = bert_preds.merge(
    full_df[["text"] + LABEL_COLS], on="text", how="left", suffixes=("_pred", "_true")
)

bert_per_label = {}
for col in LABEL_COLS:
    pc = f"{col}_pred" if f"{col}_pred" in merged.columns else col
    tc = f"{col}_true" if f"{col}_true" in merged.columns else col
    if pc == tc:
        # no suffix collision — pred and true share same name, can't distinguish
        # fall back to aggregate only
        bert_per_label[col] = {"f1": "n/a", "precision": "n/a", "recall": "n/a"}
        continue
    y_true = merged[tc].fillna(0).astype(int).values
    y_pred = merged[pc].fillna(0).astype(int).values
    bert_per_label[col] = {
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }

# ── build comparison table ────────────────────────────────────────────────────
rows = []
for col in LABEL_COLS:
    b_f1 = base[col]["f1"]
    d_f1 = bert_per_label[col]["f1"]
    delta = round(float(d_f1) - float(b_f1), 4) if d_f1 != "n/a" else "n/a"
    rows.append({
        "label":                col,
        "baseline_f1":          b_f1,
        "baseline_precision":   base[col]["precision"],
        "baseline_recall":      base[col]["recall"],
        "distilbert_f1":        d_f1,
        "distilbert_precision": bert_per_label[col]["precision"],
        "distilbert_recall":    bert_per_label[col]["recall"],
        "delta_f1":             delta,
    })

# summary rows
for avg in ("micro", "macro"):
    b_val = base.get(f"{avg}_f1", "n/a")
    d_val = bert_agg.get(f"eval_f1_{avg}", bert_agg.get(f"{avg}_f1", "n/a"))
    delta = round(float(d_val) - float(b_val), 4) if b_val != "n/a" and d_val != "n/a" else "n/a"
    rows.append({
        "label":                f"--- {avg.upper()} AVG ---",
        "baseline_f1":          b_val,
        "baseline_precision":   "",
        "baseline_recall":      "",
        "distilbert_f1":        d_val,
        "distilbert_precision": "",
        "distilbert_recall":    "",
        "delta_f1":             delta,
    })

table = pd.DataFrame(rows)
print("\n" + table[["label", "baseline_f1", "distilbert_f1", "delta_f1"]].to_string(index=False))

os.makedirs("results", exist_ok=True)
out_path = "results/model_comparison.csv"
table.to_csv(out_path, index=False)
print(f"\nFull table saved to {out_path}")