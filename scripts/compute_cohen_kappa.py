import argparse
from pathlib import Path

import pandas as pd

LABELS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]


def as_binary_int(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "no", "", "nan", "none", "<na>"}:
        return 0
    try:
        return 1 if int(float(text)) == 1 else 0
    except (ValueError, TypeError):
        return 0


def cohen_kappa_binary(human: pd.Series, model: pd.Series) -> float:
    h = human.map(as_binary_int)
    m = model.map(as_binary_int)

    n = len(h)
    if n == 0:
        return 0.0

    agree = (h == m).sum()
    p_o = agree / n

    p_h1 = (h == 1).sum() / n
    p_h0 = 1 - p_h1
    p_m1 = (m == 1).sum() / n
    p_m0 = 1 - p_m1

    p_e = (p_h1 * p_m1) + (p_h0 * p_m0)

    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0

    return (p_o - p_e) / (1 - p_e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa between human and model labels.")
    parser.add_argument("--human", required=True, help="Human-labeled CSV")
    parser.add_argument("--model", required=True, help="Model-labeled CSV")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--out", default="dataset/kappa_report.csv")
    args = parser.parse_args()

    human_path = Path(args.human)
    model_path = Path(args.model)

    if not human_path.exists() or not model_path.exists():
        raise SystemExit("Human or model CSV not found.")

    human_df = pd.read_csv(human_path)
    model_df = pd.read_csv(model_path)

    for label in LABELS:
        if label not in human_df.columns or label not in model_df.columns:
            raise SystemExit(f"Missing label column in input files: {label}")

    merged = human_df[[args.id_column] + LABELS].merge(
        model_df[[args.id_column] + LABELS],
        on=args.id_column,
        suffixes=("_human", "_model"),
        how="inner",
    )

    if merged.empty:
        raise SystemExit("No overlapping IDs found between human and model datasets.")

    rows = []
    for label in LABELS:
        kappa = cohen_kappa_binary(merged[f"{label}_human"], merged[f"{label}_model"])
        agreement = (
            merged[f"{label}_human"].map(as_binary_int) == merged[f"{label}_model"].map(as_binary_int)
        ).mean()
        rows.append({"label": label, "kappa": round(kappa, 4), "agreement": round(float(agreement), 4)})

    report_df = pd.DataFrame(rows)
    macro_kappa = report_df["kappa"].mean()
    macro_agreement = report_df["agreement"].mean()

    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)

    print(f"Compared rows: {len(merged)}")
    print(report_df.to_string(index=False))
    print(f"Macro kappa: {macro_kappa:.4f}")
    print(f"Macro agreement: {macro_agreement:.4f}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
