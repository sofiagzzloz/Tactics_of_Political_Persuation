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
import os
import json
import time
import nlpaug.augmenter.word as naw

def augment_text(text, augmenter):
    return augmenter.augment(text)[0]

df = pd.read_csv("dataset/dataset_annotated_final.csv")

label_cols = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing"
]
num_labels = len(label_cols)

# Augment rare labels
# augmenter = naw.SynonymAug(aug_src='wordnet')
# rare_labels = ["authority_appeal", "polarization", "exaggeration"]
# augmented_rows = []

# for idx, row in df.iterrows():
#     for label in rare_labels:
#         if row[label] == 1:
#             augmented_text = augment_text(row["text"], augmenter)
#             new_row = row.copy()
#             new_row["text"] = augmented_text
#             augmented_rows.append(new_row)

# if augmented_rows:
#     df_aug = pd.DataFrame(augmented_rows)
#     df = pd.concat([df, df_aug], ignore_index=True)

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
    num_train_epochs=3, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    learning_rate=1e-5,
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


run_id = time.strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"results/distilbert_{run_id}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# save test metrics
with open(f"{RESULTS_DIR}/test_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

# save config and label names
config = {
    "model_name": "distilbert-base-uncased",
    "num_labels": num_labels,
    "label_cols": label_cols,
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 2e-5
}
with open(f"{RESULTS_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=4)

with open(f"{RESULTS_DIR}/labels.json", "w") as f:
    json.dump(label_cols, f, indent=4)


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

# save thresholds
with open(f"{RESULTS_DIR}/thresholds.json", "w") as f:
    json.dump(dict(zip(label_cols, best_thresholds)), f, indent=4)

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

# sivng predictions to CSV
preds_df = pd.DataFrame(final_preds, columns=label_cols)
preds_df["text"] = test_df["text"].values
preds_df.to_csv(f"{RESULTS_DIR}/test_predictions.csv", index=False)

# saving the model and tokenizer 
trainer.save_model("models/distilbert_final")
tokenizer.save_pretrained("models/distilbert_final")