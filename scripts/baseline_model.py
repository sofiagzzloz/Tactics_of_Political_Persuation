"""
Baseline models: Dummy, TF-IDF + Logistic Regression, TF-IDF + LinearSVM
All three use the same 70/15/15 split as train_distilBERT.py (random_state=42).
Results saved to results/baselines_<timestamp>/ with one subfolder per model.

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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# Config
DATA_PATH   = "dataset/dataset_annotated_final.csv"
RESULTS_DIR = f"results/baselines_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_COLS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]

# Load & split 
df = pd.read_csv(DATA_PATH).dropna(subset=["text"])
df[LABEL_COLS] = df[LABEL_COLS].fillna(0).astype(int)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
print("\nLabel distribution (train % positive):")
print(train_df[LABEL_COLS].mean().round(3).to_string())

y_train = train_df[LABEL_COLS].values
y_val   = val_df[LABEL_COLS].values
y_test  = test_df[LABEL_COLS].values

# TF-IDF vectoriser (shared across LR and SVM)
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)
X_train = vectorizer.fit_transform(train_df["text"])
X_val   = vectorizer.transform(val_df["text"])
X_test  = vectorizer.transform(test_df["text"])


# Helpers
def tune_thresholds(clf, X_val, y_val):
    probs = clf.predict_proba(X_val)
    thresholds = []
    for i, col in enumerate(LABEL_COLS):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, i] > t).astype(int)
            f1    = f1_score(y_val[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
        print(f"  {col}: threshold={best_t:.2f}, val_f1={best_f1:.3f}")
    return thresholds


def evaluate_with_thresholds(clf, X_test, y_test, thresholds):
    probs = clf.predict_proba(X_test)
    final_preds = np.stack(
        [(probs[:, i] > thresholds[i]).astype(int) for i in range(len(LABEL_COLS))],
        axis=1,
    )
    return _build_metrics(y_test, final_preds, thresholds)


def evaluate_no_proba(clf, X_test, y_test):
    final_preds = clf.predict(X_test)
    return _build_metrics(y_test, final_preds, thresholds=None)


def _build_metrics(y_test, final_preds, thresholds):
    per_label = {}
    for i, col in enumerate(LABEL_COLS):
        entry = {
            "f1":        round(f1_score(y_test[:, i],        final_preds[:, i], zero_division=0), 4),
            "precision": round(precision_score(y_test[:, i], final_preds[:, i], zero_division=0), 4),
            "recall":    round(recall_score(y_test[:, i],     final_preds[:, i], zero_division=0), 4),
        }
        if thresholds:
            entry["threshold"] = round(thresholds[i], 4)
        per_label[col] = entry

    metrics = {
        **per_label,
        "micro_f1":        round(f1_score(y_test, final_preds, average="micro",      zero_division=0), 4),
        "macro_f1":        round(f1_score(y_test, final_preds, average="macro",      zero_division=0), 4),
        "micro_precision": round(precision_score(y_test, final_preds, average="micro", zero_division=0), 4),
        "micro_recall":    round(recall_score(y_test, final_preds, average="micro",    zero_division=0), 4),
    }
    return metrics, final_preds


def save_results(model_name, metrics, preds, test_texts, config, thresholds=None):
    subdir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(subdir, exist_ok=True)

    with open(os.path.join(subdir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(subdir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    if thresholds:
        with open(os.path.join(subdir, "thresholds.json"), "w") as f:
            json.dump(dict(zip(LABEL_COLS, thresholds)), f, indent=4)

    preds_df = pd.DataFrame(preds, columns=LABEL_COLS)
    preds_df["text"] = test_texts
    preds_df.to_csv(os.path.join(subdir, "test_predictions.csv"), index=False)

    print(f"  micro_f1={metrics['micro_f1']}  macro_f1={metrics['macro_f1']}")
    print(f"  Saved to {subdir}/")



# MODEL 1: Dummy (majority-class floor)
print("\n── Dummy classifier (majority class) ──")
dummy = OneVsRestClassifier(DummyClassifier(strategy="most_frequent"))
dummy.fit(X_train, y_train)
dummy_metrics, dummy_preds = evaluate_no_proba(dummy, X_test, y_test)
save_results("dummy", dummy_metrics, dummy_preds, test_df["text"].values, {
    "model": "DummyClassifier(strategy=most_frequent, OneVsRest)",
    "note":  "Predicts majority class per label — serves as performance floor",
})


# MODEL 2: TF-IDF + Logistic Regression
print("\n── TF-IDF + Logistic Regression ──")
lr = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0),
    n_jobs=-1,
)
lr.fit(X_train, y_train)
print("Threshold tuning on validation set:")
lr_thresholds = tune_thresholds(lr, X_val, y_val)
lr_metrics, lr_preds = evaluate_with_thresholds(lr, X_test, y_test, lr_thresholds)
save_results("lr", lr_metrics, lr_preds, test_df["text"].values, {
    "model":      "TF-IDF + Logistic Regression (OneVsRest)",
    "vectorizer": "TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True)",
    "classifier": "LogisticRegression(class_weight=balanced, C=1.0)",
}, thresholds=lr_thresholds)


# MODEL 3: TF-IDF + LinearSVM
print("\n── TF-IDF + LinearSVM ──")
# CalibratedClassifierCV wraps LinearSVC to produce probabilities for threshold tuning
svm = OneVsRestClassifier(
    CalibratedClassifierCV(LinearSVC(max_iter=2000, class_weight="balanced", C=1.0)),
    n_jobs=-1,
)
svm.fit(X_train, y_train)
print("Threshold tuning on validation set:")
svm_thresholds = tune_thresholds(svm, X_val, y_val)
svm_metrics, svm_preds = evaluate_with_thresholds(svm, X_test, y_test, svm_thresholds)
save_results("svm", svm_metrics, svm_preds, test_df["text"].values, {
    "model":      "TF-IDF + LinearSVM (OneVsRest)",
    "vectorizer": "TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True)",
    "classifier": "CalibratedClassifierCV(LinearSVC(class_weight=balanced, C=1.0))",
}, thresholds=svm_thresholds)

# Summary table 
print("\n── Summary ──")
print(f"{'Label':<22} {'Dummy F1':>10} {'LR F1':>10} {'SVM F1':>10}")
print("-" * 56)
for col in LABEL_COLS:
    print(
        f"{col:<22} "
        f"{dummy_metrics[col]['f1']:>10.4f} "
        f"{lr_metrics[col]['f1']:>10.4f} "
        f"{svm_metrics[col]['f1']:>10.4f}"
    )
print("-" * 56)
print(
    f"{'MICRO AVG':<22} "
    f"{dummy_metrics['micro_f1']:>10.4f} "
    f"{lr_metrics['micro_f1']:>10.4f} "
    f"{svm_metrics['micro_f1']:>10.4f}"
)
print(
    f"{'MACRO AVG':<22} "
    f"{dummy_metrics['macro_f1']:>10.4f} "
    f"{lr_metrics['macro_f1']:>10.4f} "
    f"{svm_metrics['macro_f1']:>10.4f}"
)
print(f"\nAll results saved to {RESULTS_DIR}/")