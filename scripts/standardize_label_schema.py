import argparse
from pathlib import Path

import pandas as pd

CANONICAL_LABELS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]

BASE_COLUMNS = [
    "id",
    "speech_id",
    "file_name",
    "segment_id",
    "text",
    "url",
    "speaker",
    "year",
]

LEGACY_TO_CANONICAL = {
    "emotional": "emotion_appeal",
    "authority": "authority_appeal",
    "framing": "rhetorical_framing",
}


def as_binary_string(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return "1"
    if text in {"0", "0.0", "false", "no", "", "nan", "none"}:
        return "0"
    try:
        return "1" if int(float(text)) == 1 else "0"
    except (ValueError, TypeError):
        return "0"


def standardize(df: pd.DataFrame, drop_legacy: bool) -> pd.DataFrame:
    for canonical in CANONICAL_LABELS:
        if canonical not in df.columns:
            df[canonical] = "0"

    for legacy, canonical in LEGACY_TO_CANONICAL.items():
        if legacy in df.columns:
            if canonical in df.columns:
                canonical_series = df[canonical].astype(str).fillna("")
                needs_fill = canonical_series.str.strip().isin(["", "nan", "None", "<NA>"])
                df.loc[needs_fill, canonical] = df.loc[needs_fill, legacy]
            else:
                df[canonical] = df[legacy]

    for canonical in CANONICAL_LABELS:
        df[canonical] = df[canonical].map(as_binary_string)

    if drop_legacy:
        removable = [column for column in LEGACY_TO_CANONICAL if column in df.columns]
        if removable:
            df = df.drop(columns=removable)

    ordered = [column for column in BASE_COLUMNS if column in df.columns] + CANONICAL_LABELS
    trailing = [
        column
        for column in df.columns
        if column not in ordered
    ]
    df = df[ordered + trailing]

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize dataset label schema to canonical columns.")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--drop-legacy", action="store_true", help="Drop legacy columns emotional/authority/framing")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = standardize(df, drop_legacy=args.drop_legacy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved standardized dataset: {output_path}")


if __name__ == "__main__":
    main()
