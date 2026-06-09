!pip install -q -U transformers --no-index --find-links /kaggle/input/workshop-packages-notebook
!pip install -q -U accelerate   --no-index --find-links /kaggle/input/workshop-packages-notebook
print('Done installing')


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/kaggle/input/roberta-fine-tuned-v1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
test_df = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
print('Done loading data')


def predict_proba(texts, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Make sure output is always a list, even if single item
        if probs.ndim == 0:
            probs = [probs.item()]
        elif probs.ndim == 1:
            probs = probs.tolist()

        all_probs.extend(probs)

    return all_probs


df = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")  # must have 'text' column
texts = df["text"].tolist()


probs = predict_proba(texts)

df["generated"] = probs
df = df[["id", "generated"]]
df.to_csv("submission.csv", index=False)

print('Done!!')

