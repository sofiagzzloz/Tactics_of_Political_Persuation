import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

MODEL_ORDER = [
    "DistilBERT",
    "Hybrid",
    "TF-IDF + LR",
    "TF-IDF + SVM",
    "Dummy",
]


def latest_dir(prefix: str, required_files: list[str] | None = None) -> Path:
    required_files = required_files or []
    matches = sorted(glob.glob(f"results/{prefix}_*/"))
    if not matches:
        raise FileNotFoundError(f"No folder found matching results/{prefix}_*/")

    for folder in reversed(matches):
        folder_path = Path(folder)
        if all((folder_path / req).exists() for req in required_files):
            return folder_path

    missing = ", ".join(required_files) if required_files else "<none>"
    raise FileNotFoundError(
        f"No results/{prefix}_* folder contains required files: {missing}"
    )


def per_label_from_predictions(preds_csv: Path) -> dict[str, dict[str, float]]:
    truth = pd.read_csv("dataset/dataset_annotated_final.csv").dropna(subset=["text"])
    preds = pd.read_csv(preds_csv)
    merged = preds.merge(
        truth[["text"] + LABEL_COLS],
        on="text",
        how="left",
        suffixes=("_pred", "_true"),
    )

    scores: dict[str, dict[str, float]] = {}
    for label in LABEL_COLS:
        pred_col = f"{label}_pred" if f"{label}_pred" in merged.columns else label
        true_col = f"{label}_true" if f"{label}_true" in merged.columns else label

        if pred_col == true_col:
            scores[label] = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
            continue

        y_true = merged[true_col].fillna(0).astype(int).values
        y_pred = merged[pred_col].fillna(0).astype(int).values

        scores[label] = {
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        }
    return scores


def load_distilbert() -> tuple[dict, dict[str, dict[str, float]]]:
    distil_dir = latest_dir("distilbert", required_files=["test_metrics.json", "test_predictions.csv"])
    with open(distil_dir / "test_metrics.json", "r", encoding="utf-8") as f:
        aggregate = json.load(f)
    per_label = per_label_from_predictions(distil_dir / "test_predictions.csv")
    return aggregate, per_label


def load_baseline_submodel(submodel: str) -> tuple[dict, dict[str, dict[str, float]]]:
    base_dir = latest_dir("baselines") / submodel
    with open(base_dir / "test_metrics.json", "r", encoding="utf-8") as f:
        aggregate = json.load(f)
    per_label = per_label_from_predictions(base_dir / "test_predictions.csv")
    return aggregate, per_label


def load_hybrid() -> tuple[dict, dict[str, dict[str, float]]]:
    hybrid_dir = latest_dir("hybrid")

    hybrid_csv = hybrid_dir / "hybrid_results.csv"
    if not hybrid_csv.exists():
        hybrid_csv = hybrid_dir / "results.csv"
    if not hybrid_csv.exists():
        raise FileNotFoundError("No hybrid_results.csv/results.csv found in latest hybrid folder")

    df = pd.read_csv(hybrid_csv)
    first_col = str(df.columns[0])
    if first_col.startswith("Unnamed"):
        df = df.rename(columns={first_col: "label"})

    if "label" not in df.columns:
        raise ValueError("Hybrid results CSV must include a label column")

    expected = {
        "test_f1": "f1",
        "test_precision": "precision",
        "test_recall": "recall",
    }
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Hybrid results CSV missing required column: {col}")

    per_label: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        if label not in LABEL_COLS:
            continue
        per_label[label] = {
            "f1": round(float(row["test_f1"]), 4),
            "precision": round(float(row["test_precision"]), 4),
            "recall": round(float(row["test_recall"]), 4),
        }

    missing = [label for label in LABEL_COLS if label not in per_label]
    if missing:
        raise ValueError(f"Hybrid results missing labels: {missing}")

    macro_f1 = round(float(df[df["label"].isin(LABEL_COLS)]["test_f1"].mean()), 4)
    macro_precision = round(float(df[df["label"].isin(LABEL_COLS)]["test_precision"].mean()), 4)
    macro_recall = round(float(df[df["label"].isin(LABEL_COLS)]["test_recall"].mean()), 4)

    summary_json = hybrid_dir / "results.json"
    micro_f1 = None
    if summary_json.exists():
        with open(summary_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            summary_block = payload.get("summary")
            if isinstance(summary_block, dict):
                micro_f1 = summary_block.get("micro_f1")
                macro_f1 = summary_block.get("macro_f1", macro_f1)
                macro_precision = summary_block.get("macro_precision", macro_precision)
                macro_recall = summary_block.get("macro_recall", macro_recall)
            else:
                # Backward compatibility if a flat summary was saved
                micro_f1 = payload.get("micro_f1")
                macro_f1 = payload.get("macro_f1", macro_f1)
                macro_precision = payload.get("macro_precision", macro_precision)
                macro_recall = payload.get("macro_recall", macro_recall)

    aggregate = {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1,
    }
    return aggregate, per_label


def model_bundle() -> dict[str, dict]:
    distil_agg, distil_label = load_distilbert()
    lr_agg, lr_label = load_baseline_submodel("lr")
    svm_agg, svm_label = load_baseline_submodel("svm")
    dummy_agg, dummy_label = load_baseline_submodel("dummy")
    hybrid_agg, hybrid_label = load_hybrid()

    return {
        "DistilBERT": {"aggregate": distil_agg, "per_label": distil_label},
        "Hybrid": {"aggregate": hybrid_agg, "per_label": hybrid_label},
        "TF-IDF + LR": {"aggregate": lr_agg, "per_label": lr_label},
        "TF-IDF + SVM": {"aggregate": svm_agg, "per_label": svm_label},
        "Dummy": {"aggregate": dummy_agg, "per_label": dummy_label},
    }


def build_tables(models: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_label_rows = []
    for model_name, payload in models.items():
        for label in LABEL_COLS:
            metrics = payload["per_label"][label]
            per_label_rows.append(
                {
                    "model": model_name,
                    "label": label,
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                }
            )

    per_label_df = pd.DataFrame(per_label_rows)

    summary_rows = []
    for model_name, payload in models.items():
        aggregate = payload["aggregate"]
        model_slice = per_label_df[per_label_df["model"] == model_name]
        macro_precision = round(float(model_slice["precision"].mean()), 4)
        macro_recall = round(float(model_slice["recall"].mean()), 4)

        micro_f1 = aggregate.get("eval_f1_micro")
        if micro_f1 is None:
            micro_f1 = aggregate.get("micro_f1")

        macro_f1 = aggregate.get("eval_f1_macro")
        if macro_f1 is None:
            macro_f1 = aggregate.get("macro_f1")

        summary_rows.append(
            {
                "model": model_name,
                "micro_f1": None if micro_f1 is None else round(float(micro_f1), 4),
                "macro_f1": None if macro_f1 is None else round(float(macro_f1), 4),
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["model"] = pd.Categorical(summary_df["model"], categories=MODEL_ORDER, ordered=True)
    summary_df = summary_df.sort_values("model").reset_index(drop=True)

    return summary_df, per_label_df


def plot_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(summary_df))
    bars = ax.bar(x, summary_df["macro_f1"].fillna(0), color="#4e79a7")

    for idx, val in enumerate(summary_df["macro_f1"].fillna(0)):
        ax.text(idx, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1")
    ax.set_title("Model comparison: Macro F1")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_label(per_label_df: pd.DataFrame, out_path: Path) -> None:
    pivot = per_label_df.pivot(index="label", columns="model", values="f1")
    pivot = pivot.reindex(index=LABEL_COLS)
    cols = [model for model in MODEL_ORDER if model in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(pivot.index))
    bar_width = 0.8 / len(cols)

    for idx, model in enumerate(cols):
        offsets = [i + (idx - (len(cols) - 1) / 2) * bar_width for i in x]
        ax.bar(offsets, pivot[model].values, width=bar_width * 0.95, label=model)

    ax.set_xticks(list(x))
    ax.set_xticklabels([label.replace("_", "\n") for label in pivot.index], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Per-label F1 across models")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path("results/comparisons")
    fig_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    models = model_bundle()
    summary_df, per_label_df = build_tables(models)

    summary_path = out_dir / "model_summary.csv"
    per_label_path = out_dir / "model_per_label_metrics.csv"
    leaderboard_path = out_dir / "model_leaderboard.csv"

    summary_df.to_csv(summary_path, index=False)
    per_label_df.to_csv(per_label_path, index=False)
    summary_df.sort_values("macro_f1", ascending=False).to_csv(leaderboard_path, index=False)

    plot_summary(summary_df, fig_dir / "model_macro_f1_comparison.png")
    plot_per_label(per_label_df, fig_dir / "model_per_label_f1_comparison.png")

    print(f"Saved {summary_path}")
    print(f"Saved {per_label_path}")
    print(f"Saved {leaderboard_path}")
    print("Saved results/figures/model_macro_f1_comparison.png")
    print("Saved results/figures/model_per_label_f1_comparison.png")


if __name__ == "__main__":
    main()