import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from captum.attr import LayerIntegratedGradients
import pandas as pd
import numpy as np


model = DistilBertForSequenceClassification.from_pretrained("models/distilbert/checkpoint-220")
model.eval()  
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

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

# helper functions: encode(), predict(), model_forward()
def encode(text):
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

def predict(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return torch.sigmoid(outputs.logits)  # returns plain tensor

# a wrpaper to get logits directly for captum
def model_forward(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits  

lig = LayerIntegratedGradients(model_forward, model.distilbert.embeddings) 

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
        
        # sum
        word_importances = attr.sum(dim=-1).squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
        

        top_tokens = sorted(
            zip(tokens, word_importances.detach().numpy()),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        print(f"\n--- Segment {idx}, Label: {label_name} ---")
        for tok, score in top_tokens:
            print(f"{tok}: {score:.4f}")