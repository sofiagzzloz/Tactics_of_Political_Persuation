import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


df = pd.read_csv("dataset/dataset_annotated.csv")

label_cols = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing"
]
num_labels = len(label_cols)

# split
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df[label_cols].values.astype("float32")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

train_dataset = Dataset(train_df)
val_dataset = Dataset(val_df)
test_dataset = Dataset(test_df)

# problem type 
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification" 
)
model.to("cpu")  

# compute pos_weight for each label
pos_weight = torch.tensor(
    [(len(train_df) - train_df[col].sum()) / train_df[col].sum() for col in label_cols],
    dtype=torch.float32
)
print("Per-label pos_weight:", pos_weight)
print("Label distribution (% positive):")
print(train_df[label_cols].mean().round(3))

# custom trainer to use pos_weight in loss function
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.3).astype(int)  # was 0.5

    # this will allow us to see which labels are performing well and which are not
    per_label_f1 = {col: f1_score(labels[:, i], preds[:, i]) for i, col in enumerate(label_cols)}
    print("Per-label F1:", per_label_f1)

    return {
        "f1_micro": f1_score(labels, preds, average="micro"), #since we have class imbalance, micro will give us a better sense of overall performance
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="micro"),
        "recall": recall_score(labels, preds, average="micro"),
    }

training_args = TrainingArguments(
    output_dir="models/distilbert",
    num_train_epochs=5, # was 3, but we can increase it since we have a small dataset and are using a pre-trained model
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    use_cpu=True,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate(test_dataset)
print("Test results:", results)


""""
to find optimal thresholds for each label, we can predict on the validation set and then iterate over possible thresholds
this will help us understand if the default 0.5 threshold is appropriate for all labels or if some labels require a different 
threshold due to class imbalance or other factors
"""
val_preds = trainer.predict(val_dataset)
val_logits = val_preds.predictions
val_labels = val_preds.label_ids
val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()

best_thresholds = []
for i, col in enumerate(label_cols):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05): 
        preds = (val_probs[:, i] > t).astype(int)
        f1 = f1_score(val_labels[:, i], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_thresholds.append(best_t)
    print(f"{col}: best threshold={best_t:.2f}, f1={best_f1:.3f}")

# evluation 
test_preds = trainer.predict(test_dataset)
test_probs = torch.sigmoid(torch.tensor(test_preds.predictions)).numpy()
test_labels = test_preds.label_ids

final_preds = np.stack([
    (test_probs[:, i] > best_thresholds[i]).astype(int)
    for i in range(num_labels)
], axis=1)

print("\nFinal test results with per-label thresholds:")
for i, col in enumerate(label_cols):
    print(f"  {col}: f1={f1_score(test_labels[:, i], final_preds[:, i], zero_division=0):.3f}")
print(f"  f1_micro={f1_score(test_labels, final_preds, average='micro'):.3f}")
print(f"  f1_macro={f1_score(test_labels, final_preds, average='macro'):.3f}")

# save the model and tokenizer for later use in inference and interpretability
trainer.save_model("models/distilbert_final")
tokenizer.save_pretrained("models/distilbert_final")