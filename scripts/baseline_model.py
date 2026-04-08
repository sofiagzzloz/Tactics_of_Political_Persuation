import os
import json
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    average_precision_score,
)

from scipy.special import expit


LABEL_COLUMNS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]

MODEL_NAME_MAP = {
    "dummy": "dummy",
    "logreg": "logreg_tfidf",
    "svm": "svm_tfidf",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_split(split_dir: Path, split_name: str) -> pd.DataFrame:
    path = split_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path)
    required_cols = {"text", *LABEL_COLUMNS}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def tune_thresholds(y_true: np.ndarray, y_scores: np.ndarray, labels: list[str]) -> dict:
    thresholds = {}
    for i, label in enumerate(labels):
        best_threshold = 0.5
        best_f1 = -1.0

        for thr in np.arange(0.10, 0.91, 0.05):
            y_pred = (y_scores[:, i] >= thr).astype(int)
            score = f1_score(y_true[:, i], y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(round(thr, 2))

        thresholds[label] = best_threshold
    return thresholds


def apply_thresholds(y_scores: np.ndarray, thresholds: dict, labels: list[str]) -> np.ndarray:
    y_pred = np.zeros_like(y_scores, dtype=int)
    for i, label in enumerate(labels):
        y_pred[:, i] = (y_scores[:, i] >= thresholds[label]).astype(int)
    return y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, labels: list[str]) -> tuple[dict, pd.DataFrame]:
    summary = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    rows = []
    for i, label in enumerate(labels):
        try:
            ap = float(average_precision_score(y_true[:, i], y_scores[:, i]))
        except ValueError:
            ap = 0.0

        rows.append(
            {
                "label": label,
                "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "average_precision": ap,
                "support": int(y_true[:, i].sum()),
            }
        )

    per_label_df = pd.DataFrame(rows)
    summary["macro_average_precision"] = float(per_label_df["average_precision"].mean())
    return summary, per_label_df


def get_model(model_type: str):
    if model_type == "dummy":
        return OneVsRestClassifier(
            DummyClassifier(strategy="prior")
        )

    if model_type == "logreg":
        return OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
            )
        )

    if model_type == "svm":
        return OneVsRestClassifier(
            LinearSVC(class_weight="balanced")
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
        if isinstance(scores, list):
            scores = np.column_stack([s[:, 1] for s in scores])
        return np.asarray(scores)

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        raw_scores = np.asarray(raw_scores)
        if raw_scores.ndim == 1:
            raw_scores = raw_scores[:, None]
        return expit(raw_scores)

    raise ValueError("Model does not support predict_proba or decision_function.")


def save_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: Path,
):
    out = pd.DataFrame({"text": df["text"].tolist()})

    for i, label in enumerate(labels):
        out[f"{label}_true"] = y_true[:, i]
        out[f"{label}_score"] = y_scores[:, i]
        out[f"{label}_pred"] = y_pred[:, i]

    out.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a baseline multilabel classifier.")
    parser.add_argument("--split-dir", type=str, required=True, help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--output-root", type=str, default="results", help="Root directory for output runs")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["dummy", "logreg", "svm"],
        help="Type of baseline model to train",
    )
    parser.add_argument("--max-features", type=int, default=20000, help="Max TF-IDF vocabulary size")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum ngram size for TF-IDF")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    output_root = Path(args.output_root)

    train_df = load_split(split_dir, "train")
    val_df = load_split(split_dir, "val")
    test_df = load_split(split_dir, "test")

    y_train = train_df[LABEL_COLUMNS].values.astype(int)
    y_val = val_df[LABEL_COLUMNS].values.astype(int)
    y_test = test_df[LABEL_COLUMNS].values.astype(int)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
    )

    X_train = vectorizer.fit_transform(train_df["text"].fillna(""))
    X_val = vectorizer.transform(val_df["text"].fillna(""))
    X_test = vectorizer.transform(test_df["text"].fillna(""))

    model = get_model(args.model_type)
    model.fit(X_train, y_train)

    val_scores = get_scores(model, X_val)
    thresholds = tune_thresholds(y_val, val_scores, LABEL_COLUMNS)

    test_scores = get_scores(model, X_test)
    test_pred = apply_thresholds(test_scores, thresholds, LABEL_COLUMNS)

    summary_metrics, per_label_df = compute_metrics(y_test, test_pred, test_scores, LABEL_COLUMNS)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = MODEL_NAME_MAP[args.model_type]
    run_dir = ensure_dir(output_root / f"{model_name}_{timestamp}")

    config = {
        "model_type": args.model_type,
        "model_name": model_name,
        "labels": LABEL_COLUMNS,
        "split_dir": str(split_dir),
        "vectorizer": {
            "max_features": args.max_features,
            "min_df": args.min_df,
            "ngram_range": [1, args.ngram_max],
        },
        "timestamp": timestamp,
    }

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(run_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)

    per_label_df.to_csv(run_dir / "per_label_metrics.csv", index=False)
    save_predictions(test_df, y_test, test_scores, test_pred, LABEL_COLUMNS, run_dir / "test_predictions.csv")

    print("\nSaved results to:", run_dir)
    print("\nSummary metrics:")
    for k, v in summary_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nThresholds:")
    for label, thr in thresholds.items():
        print(f"  {label}: {thr:.2f}")


if __name__ == "__main__":
    main()
