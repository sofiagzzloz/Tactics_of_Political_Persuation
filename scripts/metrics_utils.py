from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

LABEL_COLS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]

DEFAULT_THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def tune_thresholds_from_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_cols: Sequence[str] = LABEL_COLS,
    grid: Iterable[float] | None = None,
) -> dict[str, float]:
    grid = list(grid or DEFAULT_THRESHOLD_GRID)
    thresholds = {}
    for i, label in enumerate(label_cols):
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            preds = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[label] = best_t
    return thresholds


def thresholds_to_array(
    thresholds: dict[str, float] | Sequence[float],
    label_cols: Sequence[str] = LABEL_COLS,
) -> np.ndarray:
    if isinstance(thresholds, dict):
        return np.array([thresholds[label] for label in label_cols], dtype=float)
    arr = np.asarray(list(thresholds), dtype=float)
    if len(arr) != len(label_cols):
        raise ValueError("Threshold count does not match number of labels.")
    return arr


def apply_thresholds(
    y_prob: np.ndarray,
    thresholds: dict[str, float] | Sequence[float],
    label_cols: Sequence[str] = LABEL_COLS,
) -> np.ndarray:
    threshold_arr = thresholds_to_array(thresholds, label_cols)
    return (np.asarray(y_prob) >= threshold_arr.reshape(1, -1)).astype(int)


def build_per_label_metrics_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    label_cols: Sequence[str] = LABEL_COLS,
    thresholds: dict[str, float] | Sequence[float] | None = None,
) -> pd.DataFrame:
    rows = []
    threshold_arr = None if thresholds is None else thresholds_to_array(thresholds, label_cols)

    for i, label in enumerate(label_cols):
        row = {
            "label": label,
            "support_positive": int(np.sum(y_true[:, i] == 1)),
            "support_negative": int(np.sum(y_true[:, i] == 0)),
            "prevalence": float(np.mean(y_true[:, i])),
            "predicted_positive_rate": float(np.mean(y_pred[:, i])),
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "jaccard": float(jaccard_score(y_true[:, i], y_pred[:, i], zero_division=0)),
        }
        if y_prob is not None:
            row["average_precision"] = safe_average_precision(y_true[:, i], y_prob[:, i])
        if threshold_arr is not None:
            row["threshold"] = float(threshold_arr[i])
        rows.append(row)

    return pd.DataFrame(rows)


def build_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    label_cols: Sequence[str] = LABEL_COLS,
) -> dict[str, float]:
    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "label_cardinality_true": float(np.mean(np.sum(y_true, axis=1))),
        "label_cardinality_pred": float(np.mean(np.sum(y_pred, axis=1))),
    }
    if y_prob is not None:
        ap_values = [safe_average_precision(y_true[:, i], y_prob[:, i]) for i in range(len(label_cols))]
        valid_ap = [x for x in ap_values if not np.isnan(x)]
        metrics["macro_average_precision"] = float(np.mean(valid_ap)) if valid_ap else float("nan")
    return metrics


def build_predictions_df(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    label_cols: Sequence[str] = LABEL_COLS,
) -> pd.DataFrame:
    pred_df = test_df.copy().reset_index(drop=True)
    for i, label in enumerate(label_cols):
        pred_df[f"true_{label}"] = y_true[:, i].astype(int)
        pred_df[f"pred_{label}"] = y_pred[:, i].astype(int)
        if y_prob is not None:
            pred_df[f"prob_{label}"] = y_prob[:, i].astype(float)
    pred_df["num_true_labels"] = np.sum(y_true, axis=1)
    pred_df["num_pred_labels"] = np.sum(y_pred, axis=1)
    return pred_df


def save_standard_outputs(
    output_dir: str | os.PathLike,
    model_name: str,
    config: dict,
    split_metadata: dict | None,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    label_cols: Sequence[str] = LABEL_COLS,
    thresholds: dict[str, float] | Sequence[float] | None = None,
) -> None:
    out_dir = ensure_dir(output_dir)

    per_label_df = build_per_label_metrics_df(y_true, y_pred, y_prob, label_cols, thresholds)
    overall_metrics = build_overall_metrics(y_true, y_pred, y_prob, label_cols)
    overall_metrics["model_name"] = model_name

    serialisable_config = dict(config)
    if split_metadata is not None:
        serialisable_config["split_metadata"] = split_metadata

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(serialisable_config, f, indent=2)

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    if thresholds is not None:
        threshold_dict = thresholds if isinstance(thresholds, dict) else {
            label: float(thresholds[i]) for i, label in enumerate(label_cols)
        }
        with open(out_dir / "thresholds.json", "w", encoding="utf-8") as f:
            json.dump(threshold_dict, f, indent=2)

    per_label_df.to_csv(out_dir / "per_label_metrics.csv", index=False)
    preds_df = build_predictions_df(test_df, y_true, y_pred, y_prob, label_cols)
    preds_df.to_csv(out_dir / "test_predictions.csv", index=False)


def load_json(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
