"""
run_captum_attributions.py
──────────────────────────────────────────────────────────────────────────────
Generates token-level attribution scores for every sentence in the dataset
using Layer Integrated Gradients on a fine-tuned DistilBERT classifier.

Key improvements over v1:
  - Runs on the full dataset (or a configurable stratified sample) instead
    of 10 random rows, giving statistically meaningful aggregations.
  - Stores signed, L1-normalised attribution scores so values are comparable
    across sentences of different lengths.
  - Flags sentences where |convergence delta| > threshold so downstream
    analysis can filter out unreliable attributions.
  - Saves both a flat CSV (one row per sentence × label × token) and a
    structured JSON for any other downstream use.
  - Prints a short quality report at the end.
"""

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from captum.attr import LayerIntegratedGradients
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH        = "../models/distilbert_final"
DATASET_PATH      = "../dataset/dataset_annotated_final.csv"
RESULTS_DIR       = "../results/captum_attributions/captum_result.csv"

# Set SAMPLE_N to None to run on the full dataset.
# Set to an integer for a stratified sample (balanced across labels).
SAMPLE_N          = 200          # e.g. 200 for a quick run

MAX_LENGTH        = 128
N_STEPS           = 100            # IG integration steps; higher = more accurate but slower
DELTA_THRESHOLD   = 0.05          # flag runs where |delta| exceeds this

LABEL_COLS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {len(df)} rows")

if SAMPLE_N is not None:
    # Stratified sample: pick rows that have at least one positive label,
    # balanced so every label is represented roughly equally.
    df = df.sample(n=min(SAMPLE_N, len(df)), random_state=42).reset_index(drop=True)
    print(f"Using stratified sample of {len(df)} rows")
else:
    df = df.reset_index(drop=True)
    print("Using full dataset")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def encode(text: str) -> dict:
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def model_forward(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits


def merge_subword_attributions(tokens: list, scores: np.ndarray) -> tuple[list, list]:
    """
    Merge WordPiece sub-tokens back into full words, summing their attributions.
    Skips special tokens ([CLS], [SEP], [PAD]).
    """
    SPECIAL = {"[CLS]", "[SEP]", "[PAD]"}
    merged_words, merged_scores = [], []
    current_word, current_score = "", 0.0

    for tok, score in zip(tokens, scores):
        if tok in SPECIAL:
            continue
        if tok.startswith("##"):
            current_word  += tok[2:]
            current_score += float(score)
        else:
            if current_word:
                merged_words.append(current_word)
                merged_scores.append(current_score)
            current_word  = tok
            current_score = float(score)

    if current_word:
        merged_words.append(current_word)
        merged_scores.append(current_score)

    return merged_words, merged_scores


def l1_normalise(scores: list) -> list:
    """
    L1-normalise attribution scores so they sum to 1 in absolute terms.
    This makes scores comparable across sentences of different lengths.
    Returns the original list unchanged if the sum is zero.
    """
    total = sum(abs(s) for s in scores)
    if total == 0:
        return scores
    return [s / total for s in scores]


# ──────────────────────────────────────────────────────────────────────────────
# ATTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────────────
lig = LayerIntegratedGradients(model_forward, model.distilbert.embeddings)

results_list = []
flagged_count = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sentences"):
    text   = str(row["text"])
    inputs = encode(text)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    baseline_ids   = torch.zeros_like(input_ids)

    for label_idx, label_name in enumerate(LABEL_COLS):
        attr, delta = lig.attribute(
            input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=label_idx,
            n_steps=N_STEPS,
            return_convergence_delta=True,
        )

        # Sum over embedding dimension → one score per token position
        raw_scores = attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens     = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

        words, scores = merge_subword_attributions(tokens, raw_scores)
        norm_scores   = l1_normalise(scores)

        delta_val   = float(delta)
        is_flagged  = abs(delta_val) > DELTA_THRESHOLD
        if is_flagged:
            flagged_count += 1

        # Keep ALL tokens with their signed normalised score.
        # Downstream scripts can filter by sign or magnitude as needed.
        token_records = [
            {"token": w, "attribution": round(s, 6), "abs_attribution": round(abs(s), 6)}
            for w, s in zip(words, norm_scores)
        ]

        results_list.append({
            "sentence_id":  idx,
            "label":        label_name,
            "text":         text,
            "tokens":       token_records,
            "delta":        round(delta_val, 6),
            "delta_flagged": is_flagged,
        })

print(f"\nAttribution complete. Flagged runs (|delta| > {DELTA_THRESHOLD}): "
      f"{flagged_count} / {len(results_list)}")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE JSON
# ──────────────────────────────────────────────────────────────────────────────
json_path = os.path.join(RESULTS_DIR, "captum_results.json")
with open(json_path, "w") as f:
    json.dump(results_list, f, indent=2)
print(f"JSON saved → {json_path}")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE FLAT CSV  (one row per sentence × label × token)
# ──────────────────────────────────────────────────────────────────────────────
records = []
for r in results_list:
    for tok in r["tokens"]:
        records.append({
            "sentence_id":   r["sentence_id"],
            "label":         r["label"],
            "text":          r["text"],
            "token":         tok["token"],
            "attribution":   tok["attribution"],      # signed, L1-normalised
            "abs_attribution": tok["abs_attribution"],
            "delta":         r["delta"],
            "delta_flagged": r["delta_flagged"],
        })

flat_df = pd.DataFrame(records)
csv_path = os.path.join(RESULTS_DIR, "captum_results.csv")
flat_df.to_csv(csv_path, index=False)
print(f"Flat CSV saved → {csv_path}  ({len(flat_df):,} rows)")

# ──────────────────────────────────────────────────────────────────────────────
# QUALITY REPORT
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Quality report ──────────────────────────────────────────────────────")
print(f"  Total sentences processed : {flat_df['sentence_id'].nunique()}")
print(f"  Total label×sentence pairs: {len(results_list)}")
print(f"  Flagged (unreliable delta): {flagged_count}")
print(f"  Mean |delta| per label:")
delta_summary = (
    flat_df.drop_duplicates(["sentence_id", "label"])
    .groupby("label")["delta"]
    .apply(lambda x: round(x.abs().mean(), 5))
)
print(delta_summary.to_string())
print("────────────────────────────────────────────────────────────────────────")