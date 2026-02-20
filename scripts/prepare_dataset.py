import argparse
import csv
import re
from pathlib import Path

import pandas as pd

LABEL_COLUMNS = [
    "emotional",
    "authority",
    "polarization",
    "presumption",
    "exaggeration",
    "framing",
]


def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_paragraphs(text: str, min_length: int) -> list[str]:
    paragraphs = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_length]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        return [" ".join(sentences)] if sentences else []
    return [" ".join(sentences[i : i + chunk_size]) for i in range(0, len(sentences), chunk_size)]


def segment_text(text: str, min_length: int, chunk_size: int) -> list[str]:
    if chunk_size > 0:
        sentences = split_sentences(text)
        chunks = chunk_sentences(sentences, chunk_size)
        return [c for c in chunks if len(c) >= min_length]
    return split_paragraphs(text, min_length)


def load_metadata(metadata_path: Path) -> pd.DataFrame | None:
    if not metadata_path.exists():
        return None
    return pd.read_csv(metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, segment, and build speech datasets.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with raw .txt files")
    parser.add_argument("--clean-dir", default="data/cleaned", help="Directory for cleaned .txt files")
    parser.add_argument("--segmented-dir", default="data/segmented", help="Directory for segmented outputs")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory for dataset CSVs")
    parser.add_argument("--min-paragraph-length", type=int, default=50)
    parser.add_argument("--chunk-sentences", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--metadata-csv",
        default="data/metadata/speeches_metadata.csv",
        help="Optional metadata CSV with file_name column",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)
    segmented_dir = Path(args.segmented_dir)
    dataset_dir = Path(args.dataset_dir)

    clean_dir.mkdir(parents=True, exist_ok=True)
    segmented_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("*.txt"))
    if not raw_files:
        raise SystemExit("No raw .txt files found in raw directory.")

    metadata_df = load_metadata(Path(args.metadata_csv))
    rows: list[dict] = []

    for speech_index, raw_path in enumerate(raw_files, start=1):
        raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw_text)
        clean_path = clean_dir / raw_path.name
        clean_path.write_text(cleaned, encoding="utf-8")

        segments = segment_text(cleaned, args.min_paragraph_length, args.chunk_sentences)
        segmented_path = segmented_dir / raw_path.name
        segmented_path.write_text("\n\n".join(segments), encoding="utf-8")

        for segment_index, segment in enumerate(segments, start=1):
            rows.append(
                {
                    "id": f"{raw_path.stem}-{segment_index}",
                    "speech_id": speech_index,
                    "file_name": raw_path.name,
                    "segment_id": segment_index,
                    "text": segment,
                }
            )

    df = pd.DataFrame(rows)

    if metadata_df is not None and "file_name" in metadata_df.columns:
        df = df.merge(metadata_df, on="file_name", how="left")
    if "url" not in df.columns:
        df["url"] = ""
    if "speaker" not in df.columns:
        df["speaker"] = ""
    if "year" not in df.columns:
        df["year"] = ""

    if args.sample_size and args.sample_size > 0:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=args.random_seed)

    for label in LABEL_COLUMNS:
        df[label] = ""

    dataset_path = dataset_dir / "dataset.csv"
    df.to_csv(dataset_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Saved {len(df)} segments to {dataset_path}")


if __name__ == "__main__":
    main()
