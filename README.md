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
```

## Optional Metadata

If you have metadata, put a CSV at `data/metadata/speeches_metadata.csv` with a `file_name` column
matching raw filenames and any extra columns you want (e.g., `speaker`, `year`, `speech_type`).
The pipeline will merge these columns automatically.

You can generate metadata automatically from the URL lists:

```bash
python scripts/extract_metadata.py \
  --urls data/urls/presidential_urls.txt data/urls/congressional_urls.txt \
  --out data/metadata/speeches_metadata.csv
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
