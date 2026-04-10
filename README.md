# Political Persuasion NLP Pipeline

This repository contains the full workflow for a political speech NLP project:
- dataset building and metadata extraction,
- automated annotation,
- model training/evaluation,
- figure/table generation,
- runnable hybrid model artifacts for reproducible inference.

## Project Outputs

- **Dataset artifacts** in `dataset/` (cleaned and labeled CSVs).
- **Model and analysis outputs** in `results/`.
- **Runnable hybrid model package** in each `results/hybrid_*/artifacts/` folder.

## Repository Structure

```text
data/                     # raw/cleaned/segmented text files, URLs, metadata
dataset/                  # final CSV datasets and kappa reports
results/                  # training runs, comparisons, figures, analyses
scripts/                  # all pipeline/training/evaluation utilities
splits/                   # fixed train/val/test splits
README.md
requirements.txt
```

## Environment Setup

Use the project virtual environment to avoid dependency mismatches.

```bash
git clone https://github.com/sofiagzzloz/Tactics_of_Political_Persuasion.git
cd Tactics_of_Political_Persuasion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducible End-to-End Workflow

### 1) Build dataset

```bash
python scripts/prepare_dataset.py \
  --raw-dir data/raw \
  --clean-dir data/cleaned \
  --segmented-dir data/segmented \
  --dataset-dir dataset \
  --sample-size 1000
```

### 2) (Optional) annotate with Ollama

```bash
python scripts/annotate_with_ollama.py \
  --input dataset/dataset.csv \
  --output dataset/dataset_annotated.csv \
  --model gpt-oss:20b \
  --ruleset balanced
```

### 3) Create fixed splits

```bash
python scripts/create_splits.py \
  --data-path dataset/dataset_annotated_final.csv \
  --output-dir splits \
  --seed 42
```

### 4) Train hybrid model + export runnable artifacts

```bash
python scripts/hybrid_model.py \
  --split-dir splits \
  --output-root results
```

### 5) Build comparison tables + figures

```bash
python scripts/compare_all_models.py
python scripts/generate_figures.py
```

## Label Taxonomy

Binary labels (`1` present, `0` absent):
- `emotion_appeal`
- `authority_appeal`
- `polarization`
- `presumption`
- `exaggeration`
- `rhetorical_framing`

## Runnable Hybrid Model (Inference)

After training, each run creates:

```text
results/hybrid_YYYYMMDD_HHMMSS/
  hybrid_results.csv
  results.json
  test_predictions.csv
  artifacts/
    tfidf_vectorizer.joblib
    classifier_<label>.joblib
    thresholds.json
    metadata.json
```

### Single-text inference

```bash
LATEST=$(ls -dt results/hybrid_* | head -n 1)
python scripts/infer_hybrid_model.py \
  --artifacts-dir "$LATEST/artifacts" \
  --text "We must act now to protect our families."
```

### CSV batch inference

```bash
LATEST=$(ls -dt results/hybrid_* | head -n 1)
python scripts/infer_hybrid_model.py \
  --artifacts-dir "$LATEST/artifacts" \
  --input-csv dataset/dataset_for_annotation.csv \
  --text-column text \
  --output results/hybrid_predictions.csv
```

## Key Report Artifacts

- `results/comparisons/model_leaderboard.csv`
- `results/comparisons/model_summary.csv`
- `results/figures/fig1_per_label_f1.png`
- `results/figures/fig2_overall_metrics.png`
- `results/figures/fig3_precision_recall.png`
- `results/figures/fig6_error_analysis.png`

## Hugging Face Repositories

- Dataset repo: `sofiagzzloz/political-persuasion-dataset`
- Model repo: `sofiagzzloz/political-persuasion-model`

## Notes for Reproducibility

- Always run scripts with the project `.venv` Python executable.
- The string `results/hybrid_YYYYMMDD_HHMMSS/...` is a placeholder; use an actual run folder (or `LATEST=$(ls -dt results/hybrid_* | head -n 1)`).
- For deterministic behavior, keep `--seed 42` and reuse `splits/`.
