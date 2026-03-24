import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from captum.attr import LayerIntegratedGradients
import pandas as pd
import numpy as np
import os
import json

model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_final")
model.eval()
tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert_final")

label_cols = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing"
]

df = pd.read_csv("dataset/dataset_annotated.csv")
sample_df = df.sample(10, random_state=26)  

def encode(text):
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

def model_forward(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits  

lig = LayerIntegratedGradients(model_forward, model.distilbert.embeddings) 

RESULTS_DIR = "results/captum_attributions"
os.makedirs(RESULTS_DIR, exist_ok=True)

results_list = []

for idx, row in sample_df.iterrows():
    text = row["text"]
    inputs = encode(text)
    
    for label_idx, label_name in enumerate(label_cols):
        attr, delta = lig.attribute(
            inputs['input_ids'],
            baselines=torch.zeros_like(inputs['input_ids']),
            additional_forward_args=(inputs['attention_mask'],),
            target=label_idx,
            return_convergence_delta=True
        )
        
        word_importances = attr.sum(dim=-1).squeeze(0).detach().numpy()
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))

        # merge subwords and ignore special tokens
        merged_words = []
        merged_scores = []
        skip_special = {"[CLS]", "[SEP]", "[PAD]"}
        current_word = ""
        current_score = 0.0
        for tok, score in zip(tokens, word_importances):
            if tok in skip_special:
                continue
            if tok.startswith("##"):
                current_word += tok[2:]
                current_score += score
            else:
                if current_word:
                    merged_words.append(current_word)
                    merged_scores.append(current_score)
                current_word = tok
                current_score = score
        if current_word:
            merged_words.append(current_word)
            merged_scores.append(current_score)

        # select top 10 by absolute score
        top_tokens = sorted(
            zip(merged_words, merged_scores),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]

        results_list.append({
            "idx": idx,
            "label": label_name,
            "text": text,
            "top_tokens": [(tok, float(score)) for tok, score in top_tokens],
            "delta": float(delta)
        })

# save JSON
with open(os.path.join(RESULTS_DIR, "captum_results.json"), "w") as f:
    json.dump(results_list, f, indent=4)

# save CSV
records = []
for r in results_list:
    for tok, score in r["top_tokens"]:
        records.append({
            "idx": r["idx"],
            "label": r["label"],
            "text": r["text"],
            "token": tok,
            "score": score,
            "delta": r["delta"]
        })
pd.DataFrame(records).to_csv(os.path.join(RESULTS_DIR, "captum_results.csv"), index=False)