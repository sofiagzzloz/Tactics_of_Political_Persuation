import argparse
from pathlib import Path

import pandas as pd

LABEL_COLUMNS = {
    "emotional",
    "authority",
    "polarization",
    "presumption",
    "exaggeration",
    "framing",
}

REQUIRED_COLUMNS = {
    "id",
    "speech_id",
    "file_name",
    "segment_id",
    "text",
    "url",
    "speaker",
    "year",
} | LABEL_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset schema and labels.")
    parser.add_argument("--dataset", default="dataset/dataset.csv")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise SystemExit(f"Missing columns: {sorted(missing_columns)}")

    empty_text = df["text"].isna().sum() + (df["text"].astype(str).str.strip() == "").sum()
    if empty_text:
        raise SystemExit(f"Empty text rows detected: {empty_text}")

    duplicate_ids = df["id"].duplicated().sum()
    if duplicate_ids:
        raise SystemExit(f"Duplicate id values found: {duplicate_ids}")

    label_columns = sorted(LABEL_COLUMNS)
    non_empty_labels = 0
    for label in label_columns:
        series = df[label]
        non_empty_labels += (
            (~series.isna()) & (series.astype(str).str.strip() != "") & (series.astype(str).str.lower() != "nan")
        ).sum()

    print("Validation passed")
    print(f"Rows: {len(df)}")
    print(f"Label columns: {', '.join(label_columns)}")
    print(f"Non-empty labels: {non_empty_labels}")


if __name__ == "__main__":
    main()
