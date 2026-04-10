"""
Standardized hybrid baseline: TF-IDF sparse features + DistilBERT mean-pooled embeddings,
then a one-vs-rest logistic regression head.

This script also exports runnable artifacts for inference/Hugging Face model packaging.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from transformers import DistilBertModel, DistilBertTokenizerFast

from metrics_utils import LABEL_COLS, ensure_dir, load_json, save_standard_outputs, tune_thresholds_from_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standardized hybrid model.")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--encoder-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_splits(split_dir: str):
    split_path = Path(split_dir)
    train_df = pd.read_csv(split_path / "train.csv")
    val_df = pd.read_csv(split_path / "val.csv")
    test_df = pd.read_csv(split_path / "test.csv")
    metadata = load_json(split_path / "split_metadata.json")

    for frame in (train_df, val_df, test_df):
        frame["text"] = frame["text"].astype(str)
        for label in LABEL_COLS:
            frame[label] = frame[label].fillna(0).astype(int)

    return train_df, val_df, test_df, metadata


@torch.no_grad()
def get_embeddings(
    model: DistilBertModel,
    tokenizer: DistilBertTokenizerFast,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def export_runnable_artifacts(
    run_dir: Path,
    clf: OneVsRestClassifier,
    vectorizer: TfidfVectorizer,
    thresholds: dict[str, float],
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    artifacts_dir = ensure_dir(run_dir / "artifacts")

    joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")

    for idx, label in enumerate(LABEL_COLS):
        joblib.dump(clf.estimators_[idx], artifacts_dir / f"classifier_{label}.joblib")

    with open(artifacts_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    summary = {
        "micro_f1": round(float(f1_score(y_test, y_pred, average="micro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "micro_precision": round(float(precision_score(y_test, y_pred, average="micro", zero_division=0)), 4),
        "micro_recall": round(float(recall_score(y_test, y_pred, average="micro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
    }

    metadata = {
        "label_columns": LABEL_COLS,
        "bert_model_name": args.encoder_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "summary": summary,
    }

    with open(artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()
    train_df, val_df, test_df, split_metadata = load_splits(args.split_dir)

    y_train = train_df[LABEL_COLS].to_numpy(dtype=int)
    y_val = val_df[LABEL_COLS].to_numpy(dtype=int)
    y_test = test_df[LABEL_COLS].to_numpy(dtype=int)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    print("[1] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    x_train_tfidf = vectorizer.fit_transform(train_df["text"])
    x_val_tfidf = vectorizer.transform(val_df["text"])
    x_test_tfidf = vectorizer.transform(test_df["text"])

    print("[2] Extracting DistilBERT embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.encoder_name)
    encoder = DistilBertModel.from_pretrained(args.encoder_name).to(device)
    encoder.eval()

    x_train_bert = csr_matrix(
        get_embeddings(encoder, tokenizer, train_df["text"].tolist(), args.max_length, args.batch_size, device)
    )
    x_val_bert = csr_matrix(
        get_embeddings(encoder, tokenizer, val_df["text"].tolist(), args.max_length, args.batch_size, device)
    )
    x_test_bert = csr_matrix(
        get_embeddings(encoder, tokenizer, test_df["text"].tolist(), args.max_length, args.batch_size, device)
    )

    print("[3] Combining sparse features...")
    x_train = hstack([x_train_tfidf, x_train_bert], format="csr")
    x_val = hstack([x_val_tfidf, x_val_bert], format="csr")
    x_test = hstack([x_test_tfidf, x_test_bert], format="csr")

    print("[4] Training one-vs-rest logistic regression...")
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=args.seed),
        n_jobs=None,
    )
    clf.fit(x_train, y_train)

    val_prob = clf.predict_proba(x_val)
    thresholds = tune_thresholds_from_probs(y_val, val_prob, LABEL_COLS)
    threshold_arr = np.array([thresholds[label] for label in LABEL_COLS], dtype=float)

    test_prob = clf.predict_proba(x_test)
    test_pred = (test_prob >= threshold_arr.reshape(1, -1)).astype(int)

    run_dir = ensure_dir(Path(args.output_root) / f"hybrid_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    config = {
        "model_name": "hybrid",
        "split_dir": args.split_dir,
        "label_cols": LABEL_COLS,
        "threshold_policy": "per_label_val_optimized",
        "tfidf": {"max_features": 5000, "ngram_range": [1, 2], "sublinear_tf": True, "min_df": 2},
        "encoder_name": args.encoder_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    save_standard_outputs(
        run_dir,
        model_name="hybrid",
        config=config,
        split_metadata=split_metadata,
        test_df=test_df,
        y_true=y_test,
        y_pred=test_pred,
        y_prob=test_prob,
        label_cols=LABEL_COLS,
        thresholds=thresholds,
    )

    export_runnable_artifacts(
        run_dir=run_dir,
        clf=clf,
        vectorizer=vectorizer,
        thresholds=thresholds,
        args=args,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        y_test=y_test,
        y_pred=test_pred,
    )

    print(f"Saved hybrid results to {run_dir}")


if __name__ == "__main__":
    main()
