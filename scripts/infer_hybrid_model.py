import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with exported hybrid model artifacts.")
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Path to artifacts directory produced by scripts/hybrid_model.py (contains .joblib/.json files)",
    )
    parser.add_argument("--text", default="", help="Single text segment to score")
    parser.add_argument("--input-csv", default="", help="Optional input CSV with a text column")
    parser.add_argument("--text-column", default="text", help="Text column name for --input-csv")
    parser.add_argument("--output", default="", help="Optional output CSV path when using --input-csv")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max sequence length")
    return parser.parse_args()


def load_artifacts(artifacts_dir: Path):
    metadata_path = artifacts_dir / "metadata.json"
    thresholds_path = artifacts_dir / "thresholds.json"
    vectorizer_path = artifacts_dir / "tfidf_vectorizer.joblib"

    if not metadata_path.exists() or not thresholds_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Missing required artifact files. Expected metadata.json, thresholds.json, tfidf_vectorizer.joblib"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    labels = metadata["label_columns"]
    vectorizer = joblib.load(vectorizer_path)

    classifiers = {}
    for label in labels:
        clf_path = artifacts_dir / f"classifier_{label}.joblib"
        if not clf_path.exists():
            raise FileNotFoundError(f"Missing classifier file: {clf_path}")
        classifiers[label] = joblib.load(clf_path)

    bert_model_name = metadata.get("bert_model_name", "distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = DistilBertModel.from_pretrained(bert_model_name)
    bert_model.eval()

    return labels, thresholds, vectorizer, classifiers, tokenizer, bert_model


def get_embeddings(texts: list[str], tokenizer: DistilBertTokenizerFast, model: DistilBertModel, max_length: int) -> np.ndarray:
    embeddings = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            mean_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        embeddings.append(mean_embedding)
    return np.array(embeddings)


def predict(
    texts: list[str],
    labels: list[str],
    thresholds: dict[str, float],
    vectorizer,
    classifiers: dict,
    tokenizer,
    bert_model,
    max_length: int,
) -> pd.DataFrame:
    tfidf_features = vectorizer.transform(texts).toarray()
    bert_embeddings = get_embeddings(texts, tokenizer, bert_model, max_length=max_length)
    features = np.concatenate([tfidf_features, bert_embeddings], axis=1)

    rows = []
    for row_idx, text in enumerate(texts):
        row = {"text": text}
        for label in labels:
            prob = float(classifiers[label].predict_proba(features[row_idx : row_idx + 1])[:, 1][0])
            pred = int(prob > float(thresholds[label]))
            row[f"{label}_prob"] = round(prob, 6)
            row[label] = pred
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts directory does not exist: {artifacts_dir}")

    if not args.text and not args.input_csv:
        raise SystemExit("Provide either --text or --input-csv")

    labels, thresholds, vectorizer, classifiers, tokenizer, bert_model = load_artifacts(artifacts_dir)

    if args.text:
        df = predict(
            texts=[args.text],
            labels=labels,
            thresholds=thresholds,
            vectorizer=vectorizer,
            classifiers=classifiers,
            tokenizer=tokenizer,
            bert_model=bert_model,
            max_length=args.max_length,
        )
        print(df.to_string(index=False))
        return

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    input_df = pd.read_csv(input_path)
    if args.text_column not in input_df.columns:
        raise SystemExit(f"Text column not found: {args.text_column}")

    texts = input_df[args.text_column].fillna("").astype(str).tolist()
    preds_df = predict(
        texts=texts,
        labels=labels,
        thresholds=thresholds,
        vectorizer=vectorizer,
        classifiers=classifiers,
        tokenizer=tokenizer,
        bert_model=bert_model,
        max_length=args.max_length,
    )

    merged = input_df.copy()
    for col in preds_df.columns:
        if col == "text":
            continue
        merged[col] = preds_df[col].values

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_hybrid_predictions.csv")
    merged.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
