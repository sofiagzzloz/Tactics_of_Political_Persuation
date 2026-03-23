import argparse
import re
from pathlib import Path

import pandas as pd


def extract_year_from_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    match = re.search(r"/(19|20)\d{2}/", url)
    if match:
        return match.group(0).strip("/")
    return ""


def normalize_year(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return ""
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix year column using URL path year when available.")
    parser.add_argument("--input", default="dataset/dataset_annotated.csv")
    parser.add_argument("--output", default="dataset/dataset_annotated.csv")
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--year-column", default="year")
    parser.add_argument("--only-missing", action="store_true", help="Only fill empty years; do not overwrite mismatches")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.url_column not in df.columns or args.year_column not in df.columns:
        raise SystemExit(f"Columns not found: {args.url_column}, {args.year_column}")

    total = 0
    updated = 0

    for index in df.index:
        total += 1
        url_year = extract_year_from_url(str(df.at[index, args.url_column]))
        if not url_year:
            continue

        current_year = normalize_year(df.at[index, args.year_column])

        if args.only_missing:
            if current_year == "":
                df.at[index, args.year_column] = url_year
                updated += 1
        else:
            if current_year != url_year:
                df.at[index, args.year_column] = url_year
                updated += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Processed rows: {total}")
    print(f"Updated years: {updated}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
