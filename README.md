# NLP Dataset Pipeline (Speeches)

This repo builds a clean, segmented, multi-label annotation dataset from political speeches.

## Folder Structure

```
data/
  raw/                 # raw .txt speeches
  cleaned/             # cleaned .txt speeches
  segmented/           # segmented .txt files per speech
  urls/                # URL lists for optional downloading
  metadata/            # metadata CSVs

dataset/
  dataset.csv
```

## Quick Start

1) Put raw `.txt` files in `data/raw/` (manual copy works well).
2) Run the pipeline to clean, segment, and build datasets (default is 1 sentence per segment).

```bash
python scripts/prepare_dataset.py \
  --raw-dir data/raw \
  --clean-dir data/cleaned \
  --segmented-dir data/segmented \
  --dataset-dir dataset \
  --sample-size 1000
```

This produces:
- `dataset/dataset.csv` (with empty labels)

## Optional: Semi-Automated Download

Add URLs to `data/urls/presidential_urls.txt` or `data/urls/congressional_urls.txt` (one per line), then run:

```bash
python scripts/download_from_urls.py \
  --urls-file data/urls/presidential_urls.txt \
  --out-dir data/raw

# international sources
python scripts/download_from_urls.py \
  --urls-file data/urls/uk_urls.txt \
  --out-dir data/raw
python scripts/download_from_urls.py \
  --urls-file data/urls/canada_urls.txt \
  --out-dir data/raw
python scripts/download_from_urls.py \
  --urls-file data/urls/australia_urls.txt \
  --out-dir data/raw
```

## Optional Metadata

If you have metadata, put a CSV at `data/metadata/speeches_metadata.csv` with a `file_name` column
matching raw filenames and any extra columns you want (e.g., `speaker`, `year`, `speech_type`).
The pipeline will merge these columns automatically.

You can generate metadata automatically from the URL lists:

```bash
python scripts/extract_metadata.py \
  --urls data/urls/presidential_urls.txt data/urls/congressional_urls.txt \
        data/urls/uk_urls.txt data/urls/canada_urls.txt data/urls/australia_urls.txt \
  --out data/metadata/speeches_metadata.csv

## International URL Lists

Build UK, Canada, and Australia URL lists (official government sources):

```bash
python scripts/build_international_url_lists.py \
  --uk-pages 5 --uk-limit 100 \
  --canada-pages 5 --canada-limit 100 \
  --australia-pages 5 --australia-limit 100
```
```

## Label Columns

The annotation file includes:
- `emotional`
- `authority`
- `polarization`
- `presumption`
- `exaggeration`
- `framing`

Use `1` for present, `0` for absent.

## Ollama Auto-Annotation

Annotate rows using a local Ollama model (default: `gpt-oss:20b`):

```bash
python scripts/annotate_with_ollama.py \
  --input dataset/dataset.csv \
  --output dataset/dataset_annotated.csv \
  --model gpt-oss:20b \
  --max-rows 200
```

Options:
- `--start-row` and `--max-rows` for chunked annotation runs
- `--use-legacy-columns` to also mirror values into `emotional`, `authority`, `framing`
- `--dry-run` for pipeline validation without calling Ollama
