"""
Compares all models (Dummy, LR, SVM, DistilBERT) and saves:
  results/model_comparison.csv   — full per-label table

Automatically picks the most recent timestamped folder for each model.

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

LABEL_SHORT = {
    "emotion_appeal":    "Emotion",
    "authority_appeal":  "Authority",
    "polarization":      "Polarization",
    "presumption":       "Presumption",
    "exaggeration":      "Exaggeration",
    "rhetorical_framing":"Rhet. Framing",
}


# Helpers 
def latest(prefix):
    folders = sorted(glob.glob(f"results/{prefix}_*/"))
    if not folders:
        raise FileNotFoundError(f"No results/{prefix}_* folder. Run the script first.")
    return folders[-1]


def per_label_from_predictions(preds_csv):
    """Recompute per-label F1/P/R by joining predictions against ground truth."""
    full_df    = pd.read_csv("dataset/dataset_annotated_final.csv").dropna(subset=["text"])
    preds_df   = pd.read_csv(preds_csv)
    merged     = preds_df.merge(
        full_df[["text"] + LABEL_COLS], on="text", how="left", suffixes=("_pred", "_true")
    )
    out = {}
    for col in LABEL_COLS:
        pc = f"{col}_pred" if f"{col}_pred" in merged.columns else col
        tc = f"{col}_true" if f"{col}_true" in merged.columns else col
        if pc == tc:
            out[col] = {"f1": None, "precision": None, "recall": None}
            continue
        yt = merged[tc].fillna(0).astype(int).values
        yp = merged[pc].fillna(0).astype(int).values
        out[col] = {
            "f1":        round(f1_score(yt, yp, zero_division=0), 4),
            "precision": round(precision_score(yt, yp, zero_division=0), 4),
            "recall":    round(recall_score(yt, yp, zero_division=0), 4),
        }
    return out


def load_model(name, subpath=None):
    """Load metrics for one model. subpath handles baselines subdirs."""
    metrics_path = os.path.join(subpath or ".", "test_metrics.json")
    with open(metrics_path) as f:
        agg = json.load(f)
    preds_csv = os.path.join(subpath or ".", "test_predictions.csv")
    per_label = per_label_from_predictions(preds_csv)
    return name, agg, per_label


# Locate result folders 
bert_dir      = latest("distilbert")
baselines_dir = latest("baselines")

print(f"DistilBERT : {bert_dir}")
print(f"Baselines  : {baselines_dir}")

models = {}
models["DistilBERT"] = load_model(
    "DistilBERT", bert_dir
)
for model_key, display in [("dummy", "Dummy"), ("lr", "TF-IDF + LR"), ("svm", "TF-IDF + SVM")]:
    subdir = os.path.join(baselines_dir, model_key)
    if os.path.exists(subdir):
        models[display] = load_model(display, subdir)
    else:
        print(f"  Warning: {subdir} not found, skipping {display}")

# Build comparison table
rows = []
for col in LABEL_COLS:
    row = {"label": col}
    for display, (_, agg, per_label) in models.items():
        f1 = per_label[col]["f1"]
        row[f"{display}_f1"]        = f1
        row[f"{display}_precision"] = per_label[col]["precision"]
        row[f"{display}_recall"]    = per_label[col]["recall"]
    rows.append(row)

# Aggregate rows
for avg in ("micro", "macro"):
    row = {"label": f"--- {avg.upper()} AVG ---"}
    for display, (_, agg, _pl) in models.items():
        val = agg.get(f"eval_f1_{avg}", agg.get(f"{avg}_f1", None))
        row[f"{display}_f1"]        = val
        row[f"{display}_precision"] = ""
        row[f"{display}_recall"]    = ""
    rows.append(row)

table = pd.DataFrame(rows)

# Print summary 
model_names = list(models.keys())
f1_cols     = [f"{m}_f1" for m in model_names]
header      = f"{'Label':<22} " + "  ".join(f"{m:>14}" for m in model_names)
print("\n" + header)
print("-" * len(header))
for _, row in table.iterrows():
    vals = "  ".join(
        f"{row[c]:>14.4f}" if isinstance(row[c], float) else f"{'n/a':>14}"
        for c in f1_cols
    )
    print(f"{str(row['label']):<22} {vals}")

os.makedirs("results", exist_ok=True)
table.to_csv("results/model_comparison.csv", index=False)
print("\nSaved → results/model_comparison.csv")