from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_utils import LABEL_COLS, ensure_dir


try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except Exception:
    MultilabelStratifiedShuffleSplit = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed train/val/test splits for the project.")
    parser.add_argument("--data-path", default="dataset/dataset_annotated_final.csv")
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interpretation-sample-size", type=int, default=200)
    return parser.parse_args()


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["text"]).copy()
    for label in LABEL_COLS:
        df[label] = df[label].fillna(0).astype(int)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=int))
    return df.reset_index(drop=True)


def greedy_multilabel_split(df: pd.DataFrame, ratios: list[float], seed: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    y = df[LABEL_COLS].to_numpy(dtype=int)
    n = len(df)
    target_sizes = np.array([round(n * r) for r in ratios], dtype=int)
    while target_sizes.sum() < n:
        target_sizes[np.argmin(target_sizes)] += 1
    while target_sizes.sum() > n:
        target_sizes[np.argmax(target_sizes)] -= 1

    total_label_counts = y.sum(axis=0).astype(float)
    target_label_counts = np.outer(np.array(ratios, dtype=float), total_label_counts)

    order = np.arange(n)
    rng.shuffle(order)
    order = sorted(order, key=lambda idx: (int(y[idx].sum()), rng.random()), reverse=True)

    split_indices = [[] for _ in ratios]
    current_sizes = np.zeros(len(ratios), dtype=float)
    current_label_counts = np.zeros((len(ratios), len(LABEL_COLS)), dtype=float)

    for idx in order:
        row = y[idx]
        best_split = None
        best_cost = None
        for split_id in range(len(ratios)):
            if current_sizes[split_id] >= target_sizes[split_id]:
                continue

            new_size = current_sizes[split_id] + 1.0
            size_cost = ((new_size - target_sizes[split_id]) / max(target_sizes[split_id], 1.0)) ** 2
            before = np.sum(((current_label_counts[split_id] - target_label_counts[split_id]) / np.maximum(target_label_counts[split_id], 1.0)) ** 2)
            after_counts = current_label_counts[split_id] + row
            after = np.sum(((after_counts - target_label_counts[split_id]) / np.maximum(target_label_counts[split_id], 1.0)) ** 2)
            label_cost = after - before
            cardinality_bonus = -0.02 * row.sum()
            cost = size_cost + label_cost + cardinality_bonus
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = split_id

        if best_split is None:
            best_split = int(np.argmin(current_sizes / np.maximum(target_sizes, 1.0)))

        split_indices[best_split].append(idx)
        current_sizes[best_split] += 1.0
        current_label_counts[best_split] += row

    return [df.iloc[idxs].sample(frac=1.0, random_state=seed).reset_index(drop=True) for idxs in split_indices]


def iterative_split(df: pd.DataFrame, seed: int, train_size: float, val_size: float, test_size: float):
    if MultilabelStratifiedShuffleSplit is None:
        return greedy_multilabel_split(df, [train_size, val_size, test_size], seed)

    y = df[LABEL_COLS].to_numpy(dtype=int)
    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=seed)
    train_idx, temp_idx = next(first.split(df, y))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    temp_y = temp_df[LABEL_COLS].to_numpy(dtype=int)
    relative_test = test_size / (val_size + test_size)
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
    val_idx, test_idx = next(second.split(temp_df, temp_y))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    return [train_df, val_df, test_df]


def balanced_subset(df: pd.DataFrame, sample_n: int, seed: int) -> pd.DataFrame:
    if sample_n >= len(df):
        return df.copy().reset_index(drop=True)

    ratios = [sample_n / len(df), 1.0 - (sample_n / len(df))]
    subset_df, _ = greedy_multilabel_split(df, ratios, seed)
    return subset_df.reset_index(drop=True)


def summarise_split(df: pd.DataFrame) -> dict:
    summary = {"rows": int(len(df))}
    for label in LABEL_COLS:
        summary[label] = float(df[label].mean())
    summary["avg_labels_per_row"] = float(df[LABEL_COLS].sum(axis=1).mean())
    return summary


def main() -> None:
    args = parse_args()
    if not np.isclose(args.train_size + args.val_size + args.test_size, 1.0):
        raise ValueError("train/val/test sizes must sum to 1.0")

    out_dir = ensure_dir(args.output_dir)
    df = load_dataset(args.data_path)
    train_df, val_df, test_df = iterative_split(df, args.seed, args.train_size, args.val_size, args.test_size)

    interpretation_df = balanced_subset(test_df, args.interpretation_sample_size, args.seed)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    interpretation_df.to_csv(out_dir / "interpretation_subset.csv", index=False)

    metadata = {
        "data_path": args.data_path,
        "seed": args.seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "iterative_stratification_available": MultilabelStratifiedShuffleSplit is not None,
        "train_summary": summarise_split(train_df),
        "val_summary": summarise_split(val_df),
        "test_summary": summarise_split(test_df),
        "interpretation_subset_summary": summarise_split(interpretation_df),
    }
    with open(out_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved fixed splits to {out_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
