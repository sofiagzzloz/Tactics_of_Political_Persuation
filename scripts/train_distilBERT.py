import argparse
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from packaging import version
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABEL_COLUMNS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_split(split_dir: Path, split_name: str) -> pd.DataFrame:
    path = split_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    required_cols = {"text", *LABEL_COLUMNS}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df["text"] = df["text"].fillna("").astype(str)
    return df


class MultiLabelTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["text"].tolist()
        self.labels = df[LABEL_COLUMNS].astype(int).values.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = [float(x) for x in self.labels[idx]]
        return encoded


class WeightedTrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**model_inputs)
        logits = outputs.logits

        labels = labels.to(logits.device).float()
        pos_weight = self.pos_weight.to(logits.device)

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_monitoring_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)

    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(labels, preds, average="samples", zero_division=0)),
    }


def tune_thresholds(y_true: np.ndarray, y_scores: np.ndarray, labels: list[str]) -> dict:
    thresholds = {}

    for i, label in enumerate(labels):
        best_threshold = 0.5
        best_f1 = -1.0

        for thr in np.arange(0.10, 0.91, 0.05):
            y_pred = (y_scores[:, i] >= thr).astype(int)
            score = f1_score(y_true[:, i], y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(round(thr, 2))

        thresholds[label] = best_threshold

    return thresholds


def apply_thresholds(y_scores: np.ndarray, thresholds: dict, labels: list[str]) -> np.ndarray:
    y_pred = np.zeros_like(y_scores, dtype=int)
    for i, label in enumerate(labels):
        y_pred[:, i] = (y_scores[:, i] >= thresholds[label]).astype(int)
    return y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, labels: list[str]):
    summary = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    rows = []
    for i, label in enumerate(labels):
        try:
            ap = float(average_precision_score(y_true[:, i], y_scores[:, i]))
        except ValueError:
            ap = 0.0

        rows.append(
            {
                "label": label,
                "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "average_precision": ap,
                "support": int(y_true[:, i].sum()),
            }
        )

    per_label_df = pd.DataFrame(rows)
    summary["macro_average_precision"] = float(per_label_df["average_precision"].mean())
    return summary, per_label_df


def save_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: Path,
):
    out = pd.DataFrame({"text": df["text"].tolist()})

    for i, label in enumerate(labels):
        out[f"{label}_true"] = y_true[:, i]
        out[f"{label}_score"] = y_scores[:, i]
        out[f"{label}_pred"] = y_pred[:, i]

    out.to_csv(output_path, index=False)


def get_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    positive = y_train.sum(axis=0).astype(float)
    negative = len(y_train) - positive
    pos_weight = np.divide(
        negative,
        np.clip(positive, a_min=1.0, a_max=None),
    )
    return torch.tensor(pos_weight, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT for multilabel persuasion classification.")
    parser.add_argument("--split-dir", type=str, required=True, help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--output-root", type=str, default="results", help="Root output directory")
    parser.add_argument("--model-output-dir", type=str, default="models/distilbert_final", help="Directory to save the final trained model")
    parser.add_argument("--pretrained-model", type=str, default="distilbert-base-uncased", help="HF model name")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    split_dir = Path(args.split_dir)
    output_root = Path(args.output_root)
    model_output_dir = ensure_dir(Path(args.model_output_dir))

    train_df = load_split(split_dir, "train")
    val_df = load_split(split_dir, "val")
    test_df = load_split(split_dir, "test")

    y_train = train_df[LABEL_COLUMNS].values.astype(int)
    y_val = val_df[LABEL_COLUMNS].values.astype(int)
    y_test = test_df[LABEL_COLUMNS].values.astype(int)

    pos_weight = get_pos_weight(y_train)
    prevalence = pd.Series(y_train.mean(axis=0), index=LABEL_COLUMNS)

    print(f"Using device: cpu")
    print(f"Per-label pos_weight: {pos_weight}")
    print("Training label prevalence:")
    for label, value in prevalence.items():
        print(f"{label:<20} {value:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
    )

    train_dataset = MultiLabelTextDataset(train_df, tokenizer, args.max_length)
    val_dataset = MultiLabelTextDataset(val_df, tokenizer, args.max_length)
    test_dataset = MultiLabelTextDataset(test_df, tokenizer, args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(output_root / f"distilbert_{timestamp}")
    checkpoint_dir = ensure_dir(run_dir / "hf_checkpoints")

    training_args_kwargs = dict(
        output_dir=str(checkpoint_dir),
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        use_cpu=True,
        seed=args.seed,
    )

    if version.parse(transformers.__version__) >= version.parse("4.46"):
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = dict(
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_monitoring_metrics,
    )

    if version.parse(transformers.__version__) >= version.parse("4.46"):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    trainer.train()

    val_output = trainer.predict(val_dataset)
    val_probs = sigmoid(val_output.predictions)
    thresholds = tune_thresholds(y_val, val_probs, LABEL_COLUMNS)

    test_output = trainer.predict(test_dataset)
    test_probs = sigmoid(test_output.predictions)
    test_pred = apply_thresholds(test_probs, thresholds, LABEL_COLUMNS)

    summary_metrics, per_label_df = compute_metrics(y_test, test_pred, test_probs, LABEL_COLUMNS)

    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    config = {
        "model_type": "distilbert",
        "model_name": "distilbert",
        "pretrained_model": args.pretrained_model,
        "labels": LABEL_COLUMNS,
        "split_dir": str(split_dir),
        "model_output_dir": str(model_output_dir),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "timestamp": timestamp,
        "transformers_version": transformers.__version__,
    }

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(run_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)

    with open(model_output_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    with open(model_output_dir / "label_list.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_COLUMNS, f, indent=2, ensure_ascii=False)

    per_label_df.to_csv(run_dir / "per_label_metrics.csv", index=False)
    save_predictions(test_df, y_test, test_probs, test_pred, LABEL_COLUMNS, run_dir / "test_predictions.csv")

    print("\nSaved results to:", run_dir)
    print("Saved model to:", model_output_dir)
    print("\nSummary metrics:")
    for k, v in summary_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nThresholds:")
    for label, thr in thresholds.items():
        print(f"  {label}: {thr:.2f}")


if __name__ == "__main__":
    main()
