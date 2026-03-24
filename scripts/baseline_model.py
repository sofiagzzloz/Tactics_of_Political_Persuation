"""
Baseline: TF-IDF (bigrams) + Logistic Regression (OneVsRest, multi-label)
Uses the same 70/15/15 train/val/test split as train_distilBERT.py (random_state=42).
Saves results to results/baseline_<timestamp>/ so they sit alongside the DistilBERT run.

Run:
    python scripts/baseline_model.py
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report
)

# ── config ──────────────────────────────────────────────────────────────────
DATA_PATH  = "dataset/dataset_annotated_final.csv"
RESULTS_DIR = f"results/baseline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_COLS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]

# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH).dropna(subset=["text"])
df[LABEL_COLS] = df[LABEL_COLS].fillna(0).astype(int)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
print("Label distribution (train % positive):")
print(train_df[LABEL_COLS].mean().round(3).to_string())

# ── vectorise ────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)
X_train = vectorizer.fit_transform(train_df["text"])
X_val   = vectorizer.transform(val_df["text"])
X_test  = vectorizer.transform(test_df["text"])

y_train = train_df[LABEL_COLS].values
y_val   = val_df[LABEL_COLS].values
y_test  = test_df[LABEL_COLS].values

# ── train ────────────────────────────────────────────────────────────────────
clf = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0),
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# ── threshold tuning on val set (mirrors train_distilBERT.py) ────────────────
val_probs = np.array(clf.predict_proba(X_val))  # shape (n_labels, n_samples, 2)
# OneVsRestClassifier.predict_proba returns (n_samples, n_labels) when n_labels > 1
val_probs_pos = clf.predict_proba(X_val)         # (n_samples, n_labels)

best_thresholds = []
for i, col in enumerate(LABEL_COLS):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (val_probs_pos[:, i] > t).astype(int)
        f1    = f1_score(y_val[:, i], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_thresholds.append(best_t)
    print(f"{col}: best threshold={best_t:.2f}, val_f1={best_f1:.3f}")

with open(f"{RESULTS_DIR}/thresholds.json", "w") as fh:
    json.dump(dict(zip(LABEL_COLS, best_thresholds)), fh, indent=4)

# ── evaluate on test set ──────────────────────────────────────────────────────
test_probs = clf.predict_proba(X_test)
final_preds = np.stack(
    [(test_probs[:, i] > best_thresholds[i]).astype(int) for i in range(len(LABEL_COLS))],
    axis=1,
)

per_label = {}
for i, col in enumerate(LABEL_COLS):
    per_label[col] = {
        "f1":        round(f1_score(y_test[:, i],     final_preds[:, i], zero_division=0), 4),
        "precision": round(precision_score(y_test[:, i], final_preds[:, i], zero_division=0), 4),
        "recall":    round(recall_score(y_test[:, i],    final_preds[:, i], zero_division=0), 4),
        "threshold": round(best_thresholds[i], 4),
    }

metrics = {
    **per_label,
    "micro_f1":        round(f1_score(y_test, final_preds, average="micro",      zero_division=0), 4),
    "macro_f1":        round(f1_score(y_test, final_preds, average="macro",      zero_division=0), 4),
    "micro_precision": round(precision_score(y_test, final_preds, average="micro", zero_division=0), 4),
    "micro_recall":    round(recall_score(y_test, final_preds, average="micro",    zero_division=0), 4),
}

print("\nTest results (with per-label thresholds):")
for col in LABEL_COLS:
    print(f"  {col}: F1={per_label[col]['f1']}  P={per_label[col]['precision']}  R={per_label[col]['recall']}")
print(f"  micro_f1={metrics['micro_f1']}  macro_f1={metrics['macro_f1']}")

# ── save everything ───────────────────────────────────────────────────────────
with open(f"{RESULTS_DIR}/test_metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=4)

preds_df = pd.DataFrame(final_preds, columns=LABEL_COLS)
preds_df["text"] = test_df["text"].values
preds_df.to_csv(f"{RESULTS_DIR}/test_predictions.csv", index=False)

config = {
    "model":       "TF-IDF + Logistic Regression (OneVsRest)",
    "vectorizer":  "TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True)",
    "classifier":  "LogisticRegression(class_weight=balanced, C=1.0)",
    "data":        DATA_PATH,
    "train_size":  len(train_df),
    "val_size":    len(val_df),
    "test_size":   len(test_df),
}
with open(f"{RESULTS_DIR}/config.json", "w") as fh:
    json.dump(config, fh, indent=4)

print(f"\nAll results saved to {RESULTS_DIR}/")