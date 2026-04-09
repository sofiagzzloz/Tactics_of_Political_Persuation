from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise token attributions from the standardised Captum output.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", default="results/language_analysis")
    parser.add_argument("--min-frequency", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--selection-reason", default="correct_positive")
    parser.add_argument("--keep-negative-attributions", action="store_true")
    return parser.parse_args()


SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", ".", ",", "'s", "``", "''"}


def clean_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["delta_flagged"] == False].copy()
    df = df[df["token"].notna()].copy()
    df["token"] = df["token"].astype(str).str.lower()
    df = df[~df["token"].isin(SPECIAL_TOKENS)]
    df = df[df["token"].str.len() > 2]
    df = df[~df["token"].isin(ENGLISH_STOP_WORDS)]
    return df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_path)
    if "selection_reason" in df.columns and args.selection_reason:
        df = df[df["selection_reason"] == args.selection_reason].copy()

    df = clean_tokens(df)
    if not args.keep_negative_attributions:
        df = df[df["attribution"] > 0].copy()

    df["importance"] = df["abs_attribution"]
    rows = []
    summary_rows = []

    for label, subset in df.groupby("label"):
        sentence_count = subset["row_id"].nunique() if "row_id" in subset.columns else subset["text"].nunique()
        token_stats = subset.groupby("token").agg(
            mean_importance=("importance", "mean"),
            median_importance=("importance", "median"),
            frequency=("text", "nunique"),
            avg_probability=("probability", "mean"),
        )
        token_stats = token_stats[token_stats["frequency"] >= args.min_frequency].copy()
        if token_stats.empty:
            continue
        token_stats["weighted_importance"] = token_stats["mean_importance"] * np.log1p(token_stats["frequency"])
        token_stats = token_stats.sort_values("weighted_importance", ascending=False)

        summary_rows.append(
            {
                "label": label,
                "rows_after_filtering": int(len(subset)),
                "sentences_after_filtering": int(sentence_count),
                "unique_tokens": int(token_stats.shape[0]),
                "mean_abs_attribution": float(subset["abs_attribution"].mean()),
                "mean_probability": float(subset["probability"].mean()) if "probability" in subset.columns else np.nan,
            }
        )

        for rank, (token, values) in enumerate(token_stats.head(args.top_k).iterrows(), start=1):
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "token": token,
                    "mean_importance": float(values["mean_importance"]),
                    "median_importance": float(values["median_importance"]),
                    "frequency": int(values["frequency"]),
                    "avg_probability": float(values["avg_probability"]),
                    "weighted_importance": float(values["weighted_importance"]),
                }
            )

    top_tokens_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    top_tokens_df.to_csv(output_dir / "top_tokens_per_label.csv", index=False)
    summary_df.to_csv(output_dir / "captum_summary_by_label.csv", index=False)

    print(f"Saved token summary to {output_dir / 'top_tokens_per_label.csv'}")
    print(f"Saved label summary to {output_dir / 'captum_summary_by_label.csv'}")


if __name__ == "__main__":
    main()
