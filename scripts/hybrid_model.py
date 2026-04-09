"""
Hybrid Model: Combines TF-IDF + DistilBERT embeddings
- TF-IDF captures explicit patterns (rare words, phrases)
- DistilBERT captures semantic meaning (context)
- Ensemble both for best of both worlds

Run:
    python scripts/hybrid_model.py
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import DistilBertTokenizerFast, DistilBertModel

# Config
DATA_PATH   = "dataset/dataset_annotated_final.csv"
RESULTS_DIR = f"results/hybrid_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_COLS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]

# Load & split
df = pd.read_csv(DATA_PATH).dropna(subset=["text"])
df[LABEL_COLS] = df[LABEL_COLS].fillna(0).astype(int)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

y_train = train_df[LABEL_COLS].values
y_val   = val_df[LABEL_COLS].values
y_test  = test_df[LABEL_COLS].values

# ============ FEATURE 1: TF-IDF ============
print("\n[1] Extracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)
X_tfidf_train = tfidf_vectorizer.fit_transform(train_df["text"]).toarray()
X_tfidf_val   = tfidf_vectorizer.transform(val_df["text"]).toarray()
X_tfidf_test  = tfidf_vectorizer.transform(test_df["text"]).toarray()
print(f"  TF-IDF features: {X_tfidf_train.shape}")

# ============ FEATURE 2: DistilBERT Embeddings ============
print("[2] Extracting DistilBERT embeddings...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.eval()  # Inference mode

def get_embeddings(texts):
    """Get mean pooled embeddings from DistilBERT"""
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling: average all token embeddings
            mean_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(mean_embedding[0])
    return np.array(embeddings)

X_bert_train = get_embeddings(train_df["text"].tolist())
X_bert_val   = get_embeddings(val_df["text"].tolist())
X_bert_test  = get_embeddings(test_df["text"].tolist())
print(f"  DistilBERT embeddings: {X_bert_train.shape}")

# ============ COMBINE FEATURES ============
print("[3] Combining features...")
X_train_hybrid = np.concatenate([X_tfidf_train, X_bert_train], axis=1)
X_val_hybrid   = np.concatenate([X_tfidf_val, X_bert_val], axis=1)
X_test_hybrid  = np.concatenate([X_tfidf_test, X_bert_test], axis=1)
print(f"  Combined features: {X_train_hybrid.shape}")

# ============ TRAIN CLASSIFIERS ============
print("[4] Training hybrid classifiers...")

results = {}
y_train_positive_rates = {label: float(y_train[:, idx].mean()) for idx, label in enumerate(LABEL_COLS)}
y_test_pred_matrix = np.zeros_like(y_test)
y_test_prob_matrix = np.zeros_like(y_test, dtype=float)
classifiers: dict[str, RandomForestClassifier] = {}

for label_idx, label_col in enumerate(LABEL_COLS):
    print(f"\n  Training {label_col}...")
    
    # Use RandomForest (handles mixed feature types well)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle imbalance
    )
    
    clf.fit(X_train_hybrid, y_train[:, label_idx])
    classifiers[label_col] = clf
    
    # Threshold tuning on validation set
    probs = clf.predict_proba(X_val_hybrid)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs > t).astype(int)
        f1 = f1_score(y_val[:, label_idx], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    
    # Evaluate on test set
    test_probs = clf.predict_proba(X_test_hybrid)[:, 1]
    test_preds = (test_probs > best_t).astype(int)
    y_test_pred_matrix[:, label_idx] = test_preds
    y_test_prob_matrix[:, label_idx] = test_probs
    
    test_f1 = f1_score(y_test[:, label_idx], test_preds, zero_division=0)
    test_precision = precision_score(y_test[:, label_idx], test_preds, zero_division=0)
    test_recall = recall_score(y_test[:, label_idx], test_preds, zero_division=0)
    
    results[label_col] = {
        "threshold": round(best_t, 2),
        "val_f1": round(best_f1, 4),
        "test_f1": round(test_f1, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
    }
    
    print(f"    Threshold: {best_t:.2f}, Test F1: {test_f1:.4f}")

# ============ SAVE RESULTS ============
print("\n[5] Saving results...")

artifacts_dir = os.path.join(RESULTS_DIR, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

micro_f1 = f1_score(y_test, y_test_pred_matrix, average="micro", zero_division=0)
macro_f1 = f1_score(y_test, y_test_pred_matrix, average="macro", zero_division=0)
micro_precision = precision_score(y_test, y_test_pred_matrix, average="micro", zero_division=0)
micro_recall = recall_score(y_test, y_test_pred_matrix, average="micro", zero_division=0)
macro_precision = precision_score(y_test, y_test_pred_matrix, average="macro", zero_division=0)
macro_recall = recall_score(y_test, y_test_pred_matrix, average="macro", zero_division=0)

summary = {
    "micro_f1": round(float(micro_f1), 4),
    "macro_f1": round(float(macro_f1), 4),
    "micro_precision": round(float(micro_precision), 4),
    "micro_recall": round(float(micro_recall), 4),
    "macro_precision": round(float(macro_precision), 4),
    "macro_recall": round(float(macro_recall), 4),
}

results_payload = {
    "summary": summary,
    "per_label": results,
}

with open(f"{RESULTS_DIR}/results.json", "w") as f:
    json.dump(results_payload, f, indent=2)

results_df = pd.DataFrame(results).T
results_df.index.name = "label"
results_df.to_csv(f"{RESULTS_DIR}/hybrid_results.csv")

thresholds = {label: float(results[label]["threshold"]) for label in LABEL_COLS}
with open(os.path.join(artifacts_dir, "thresholds.json"), "w") as f:
    json.dump(thresholds, f, indent=2)

metadata = {
    "label_columns": LABEL_COLS,
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": [1, 2],
    "bert_model_name": "distilbert-base-uncased",
    "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    "train_positive_rates": y_train_positive_rates,
    "summary": summary,
}
with open(os.path.join(artifacts_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

joblib.dump(tfidf_vectorizer, os.path.join(artifacts_dir, "tfidf_vectorizer.joblib"))
for label in LABEL_COLS:
    joblib.dump(classifiers[label], os.path.join(artifacts_dir, f"classifier_{label}.joblib"))

predictions_df = test_df[["text"]].reset_index(drop=True).copy()
for idx, label in enumerate(LABEL_COLS):
    predictions_df[label] = y_test_pred_matrix[:, idx].astype(int)
    predictions_df[f"{label}_prob"] = np.round(y_test_prob_matrix[:, idx], 6)
    predictions_df[f"{label}_true"] = y_test[:, idx].astype(int)
predictions_df.to_csv(f"{RESULTS_DIR}/test_predictions.csv", index=False)

print(f"\nResults saved to {RESULTS_DIR}/")
print(f"Artifacts saved to {artifacts_dir}/")
print("\nPer-label F1 scores:")
for label, metrics in results.items():
    print(f"  {label}: {metrics['test_f1']}")

print(f"\nMicro F1: {summary['micro_f1']:.4f}")
print(f"Macro F1: {summary['macro_f1']:.4f}")
