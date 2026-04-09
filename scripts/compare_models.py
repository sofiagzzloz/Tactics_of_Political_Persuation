from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare standardised experiment outputs.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/model_comparison")
    return parser.parse_args()


def infer_model_tag(result_dir: Path) -> str:
    name = result_dir.name.lower()
    parent = result_dir.parent.name.lower()
    for candidate in [name, parent]:
        if candidate in {"dummy", "lr", "svm", "distilbert", "hybrid", "logreg_tfidf", "svm_tfidf"}:
            return candidate
    if name.startswith("distilbert_"):
        return "distilbert"
    if name.startswith("hybrid_"):
        return "hybrid"
    if name.startswith("dummy_"):
        return "dummy"
    if name.startswith("logreg_tfidf_"):
        return "logreg_tfidf"
    if name.startswith("svm_tfidf_"):
        return "svm_tfidf"
    return result_dir.name


def discover_runs(results_root: Path) -> list[dict]:
    runs = []
    for metrics_path in results_root.rglob("test_metrics.json"):
        result_dir = metrics_path.parent
        config_path = result_dir / "config.json"
        per_label_path = result_dir / "per_label_metrics.csv"
        if not config_path.exists() or not per_label_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        per_label_df = pd.read_csv(per_label_path)
        runs.append(
            {
                "result_dir": str(result_dir),
                "run_name": result_dir.parent.name if result_dir.parent != results_root else result_dir.name,
                "model_tag": config.get("model_name") or infer_model_tag(result_dir),
                "metrics": metrics,
                "config": config,
                "per_label_df": per_label_df,
            }
        )
    return runs


def overall_summary_df(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        rows.append(
            {
                "model": run["model_tag"],
                "run_name": run["run_name"],
                "result_dir": run["result_dir"],
                "micro_f1": metrics.get("micro_f1"),
                "macro_f1": metrics.get("macro_f1"),
                "samples_f1": metrics.get("samples_f1"),
                "micro_precision": metrics.get("micro_precision"),
                "micro_recall": metrics.get("micro_recall"),
                "macro_average_precision": metrics.get("macro_average_precision"),
                "subset_accuracy": metrics.get("subset_accuracy"),
                "hamming_loss": metrics.get("hamming_loss"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_overall(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in {"model", "run_name", "result_dir"}]
    agg = df.groupby("model")[numeric_cols].agg(["mean", "std", "count"])
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.to_flat_index()]
    return agg.reset_index().sort_values("macro_f1_mean", ascending=False)


def per_label_summary_df(runs: list[dict]) -> pd.DataFrame:
    parts = []
    for run in runs:
        df = run["per_label_df"].copy()
        df.insert(0, "model", run["model_tag"])
        df.insert(1, "run_name", run["run_name"])
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def aggregate_per_label(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [c for c in df.columns if c not in {"model", "run_name", "label"}]
    agg = df.groupby(["model", "label"])[numeric_cols].agg(["mean", "std", "count"])
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.to_flat_index()]
    return agg.reset_index().sort_values(["label", "f1_mean"], ascending=[True, False])


def build_uplift(per_label_df: pd.DataFrame, overall_df: pd.DataFrame):
    if "dummy" not in set(per_label_df["model"]):
        return pd.DataFrame(), pd.DataFrame()

    dummy_per_label = per_label_df[per_label_df["model"] == "dummy"][["label", "f1_mean"]].rename(columns={"f1_mean": "dummy_f1"})
    uplift_per_label = per_label_df.merge(dummy_per_label, on="label", how="left")
    uplift_per_label["f1_uplift_vs_dummy"] = uplift_per_label["f1_mean"] - uplift_per_label["dummy_f1"]

    dummy_overall = overall_df[overall_df["model"] == "dummy"]
    uplift_overall = overall_df.copy()
    if not dummy_overall.empty:
        dummy_macro = float(dummy_overall.iloc[0]["macro_f1_mean"])
        dummy_micro = float(dummy_overall.iloc[0]["micro_f1_mean"])
        uplift_overall["macro_f1_uplift_vs_dummy"] = uplift_overall["macro_f1_mean"] - dummy_macro
        uplift_overall["micro_f1_uplift_vs_dummy"] = uplift_overall["micro_f1_mean"] - dummy_micro
    return uplift_per_label, uplift_overall


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(results_root)
    if not runs:
        raise FileNotFoundError("No standardised runs found. Expected test_metrics.json + config.json + per_label_metrics.csv.")

    raw_overall = overall_summary_df(runs)
    raw_per_label = per_label_summary_df(runs)
    agg_overall = aggregate_overall(raw_overall)
    agg_per_label = aggregate_per_label(raw_per_label)
    uplift_per_label, uplift_overall = build_uplift(agg_per_label, agg_overall)

    raw_overall.to_csv(output_dir / "overall_runs_raw.csv", index=False)
    raw_per_label.to_csv(output_dir / "per_label_runs_raw.csv", index=False)
    agg_overall.to_csv(output_dir / "overall_summary.csv", index=False)
    agg_per_label.to_csv(output_dir / "per_label_summary.csv", index=False)
    uplift_per_label.to_csv(output_dir / "uplift_vs_dummy_per_label.csv", index=False)
    uplift_overall.to_csv(output_dir / "uplift_vs_dummy_overall.csv", index=False)

    print("Top models by macro_f1:")
    print(agg_overall[["model", "macro_f1_mean", "micro_f1_mean", "samples_f1_mean", "macro_average_precision_mean"]].to_string(index=False))
    print(f"\nSaved comparison tables to {output_dir}")


if __name__ == "__main__":
    main()
