"""
Standardised Captum attribution run for the fine-tuned DistilBERT model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from metrics_utils import LABEL_COLS, apply_thresholds, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Captum attributions on a fixed interpretation subset.")
    parser.add_argument("--model-dir", default="models/distilbert_final")
    parser.add_argument("--subset-path", default="splits/interpretation_subset.csv")
    parser.add_argument("--results-root", default="results/captum_attributions")
    parser.add_argument("--thresholds-path", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--delta-threshold", type=float, default=0.05)
    parser.add_argument(
        "--selection-policy",
        default="correct_positives",
        choices=["all", "predicted_positives", "correct_positives"],
    )
    return parser.parse_args()


def load_subset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["text"] = df["text"].astype(str)
    for label in LABEL_COLS:
        df[label] = df[label].fillna(0).astype(int)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=int))
    return df


@torch.no_grad()
def predict_probabilities(model, tokenizer, texts, max_length: int, batch_size: int, device: torch.device) -> np.ndarray:
    probs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits
        batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(batch_probs)
    return np.vstack(probs)


def merge_subword_attributions(tokens: list[str], scores: np.ndarray) -> tuple[list[str], list[float]]:
    special = {"[CLS]", "[SEP]", "[PAD]"}
    merged_tokens = []
    merged_scores = []
    current_token = ""
    current_score = 0.0

    for token, score in zip(tokens, scores):
        if token in special:
            continue
        if token.startswith("##"):
            current_token += token[2:]
            current_score += float(score)
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_scores.append(current_score)
            current_token = token
            current_score = float(score)

    if current_token:
        merged_tokens.append(current_token)
        merged_scores.append(current_score)
    return merged_tokens, merged_scores


def l1_normalise(scores: list[float]) -> list[float]:
    denom = sum(abs(x) for x in scores)
    if denom == 0:
        return list(scores)
    return [float(x) / denom for x in scores]


def resolve_thresholds(model_dir: Path, thresholds_path: str | None) -> dict[str, float]:
    if thresholds_path:
        return load_json(thresholds_path)
    candidate = model_dir / "thresholds.json"
    if not candidate.exists():
        raise FileNotFoundError("Could not find thresholds.json. Pass --thresholds-path explicitly.")
    return load_json(candidate)


def selection_reason(true_label: int, pred_label: int, policy: str) -> str | None:
    if policy == "all":
        return "all"
    if policy == "predicted_positives" and pred_label == 1:
        return "predicted_positive"
    if policy == "correct_positives" and pred_label == 1 and true_label == 1:
        return "correct_positive"
    return None


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    thresholds = resolve_thresholds(model_dir, args.thresholds_path)

    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model.eval()

    subset_df = load_subset(args.subset_path)
    probs = predict_probabilities(model, tokenizer, subset_df["text"].tolist(), args.max_length, args.batch_size, device)
    preds = apply_thresholds(probs, thresholds, LABEL_COLS)
    true_labels = subset_df[LABEL_COLS].to_numpy(dtype=int)

    lig = LayerIntegratedGradients(
        lambda input_ids, attention_mask: model(input_ids=input_ids, attention_mask=attention_mask).logits,
        model.distilbert.embeddings,
    )

    run_dir = Path(args.results_root) / f"captum_{args.selection_policy}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    json_records = []
    flat_records = []
    flagged_count = 0
    selected_pairs = 0

    for row_idx, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Sentences"):
        text = str(row["text"])
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        baseline_ids = torch.full_like(input_ids, fill_value=tokenizer.pad_token_id)

        for label_idx, label_name in enumerate(LABEL_COLS):
            true_label = int(true_labels[row_idx, label_idx])
            pred_label = int(preds[row_idx, label_idx])
            reason = selection_reason(true_label, pred_label, args.selection_policy)
            if reason is None:
                continue

            selected_pairs += 1
            attr, delta = lig.attribute(
                input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                target=label_idx,
                n_steps=args.n_steps,
                return_convergence_delta=True,
            )
            token_scores = attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            words, merged_scores = merge_subword_attributions(tokens, token_scores)
            norm_scores = l1_normalise(merged_scores)

            delta_value = float(delta)
            delta_flagged = abs(delta_value) > args.delta_threshold
            if delta_flagged:
                flagged_count += 1

            token_payload = []
            for word, score in zip(words, norm_scores):
                token_payload.append({
                    "token": word,
                    "attribution": round(float(score), 6),
                    "abs_attribution": round(abs(float(score)), 6),
                })
                flat_records.append(
                    {
                        "row_id": int(row["row_id"]),
                        "label": label_name,
                        "text": text,
                        "token": word,
                        "attribution": round(float(score), 6),
                        "abs_attribution": round(abs(float(score)), 6),
                        "probability": round(float(probs[row_idx, label_idx]), 6),
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "selection_reason": reason,
                        "delta": round(delta_value, 6),
                        "delta_flagged": delta_flagged,
                    }
                )

            json_records.append(
                {
                    "row_id": int(row["row_id"]),
                    "label": label_name,
                    "text": text,
                    "probability": round(float(probs[row_idx, label_idx]), 6),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "selection_reason": reason,
                    "delta": round(delta_value, 6),
                    "delta_flagged": delta_flagged,
                    "tokens": token_payload,
                }
            )

    flat_df = pd.DataFrame(flat_records)
    flat_df.to_csv(run_dir / "captum_results.csv", index=False)
    with open(run_dir / "captum_results.json", "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2)

    summary = {
        "subset_rows": int(len(subset_df)),
        "selection_policy": args.selection_policy,
        "selected_label_pairs": int(selected_pairs),
        "flagged_pairs": int(flagged_count),
        "flagged_rate": float(flagged_count / selected_pairs) if selected_pairs else 0.0,
        "n_steps": args.n_steps,
        "max_length": args.max_length,
        "delta_threshold": args.delta_threshold,
    }
    with open(run_dir / "captum_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved Captum artefacts to {run_dir}")


if __name__ == "__main__":
    main()
