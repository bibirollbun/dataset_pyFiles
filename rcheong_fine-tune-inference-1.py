from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
import torch.nn.functional as F
import os, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import numpy as np


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, os

MODEL_PATH = "/kaggle/input/inference-fine-tuned-mistral-1/transformers/default/1"

quant_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loaded fp16 model (no bitsandbytes needed).")



import pandas as pd
from tqdm import tqdm
import torch
test_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv") 

test_df["combined_text"] = (
    test_df["prompt"].fillna("") +
    "\n\nResponse A:\n" + test_df["response_a"].fillna("") +
    "\n\nResponse B:\n" + test_df["response_b"].fillna("")
)


import numpy as np
import pandas as pd
import torch

temperature = 1.5
batch_size = 16
max_length = 1536
n_aug = 3  # original + 2 cropped views
preds = []

quant_model.eval()

for i in range(0, len(test_df), batch_size):
    batch = test_df.iloc[i:i + batch_size]["combined_text"].tolist()
    all_probs = []

    for aug in range(n_aug):
        aug_batch = []
        if aug == 0:
            aug_batch = batch  # original text
        else:
            # very light random crop to simulate different truncations
            for text in batch:
                words = text.split()
                if len(words) > 120:
                    start = np.random.randint(0, len(words)//12 + 1)
                    end = len(words) - np.random.randint(0, len(words)//12 + 1)
                    aug_batch.append(" ".join(words[start:end]))
                else:
                    aug_batch.append(text)

        inputs = tokenizer(
            aug_batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(quant_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = quant_model(**inputs).logits
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    # average predictions across augmentations
    mean_probs = np.mean(np.stack(all_probs), axis=0)
    preds.extend(mean_probs)

preds = np.array(preds)

submission = pd.DataFrame({
    "id": test_df["id"],
    "winner_model_a": preds[:, 0],
    "winner_model_b": preds[:, 1],
    "winner_tie": preds[:, 2]
})

submission.to_csv("submission.csv", index=False)
print("submission.csv ready:", submission.shape)


# temperature = 1.5
# batch_size = 16
# preds = []

# for i in range(0, len(test_df), batch_size):
#     batch = test_df.iloc[i:i + batch_size]["combined_text"].tolist()
#     inputs = tokenizer(
#         batch,
#         truncation=True,
#         padding=True,
#         max_length=1536,
#         return_tensors="pt"
#     )
#     inputs = {k: v.to(quant_model.device) for k, v in inputs.items()}

#     with torch.no_grad():
#         logits = quant_model(**inputs).logits
#         logits = logits / temperature  # Apply temperature scaling
#         probs = torch.softmax(logits, dim=-1)
#         preds.extend(probs.cpu().tolist())

# preds = np.array(preds)
# submission = pd.DataFrame({
#     "id": test_df["id"],
#     "winner_model_a": preds[:, 0],
#     "winner_model_b": preds[:, 1],
#     "winner_tie": preds[:, 2]
# })

# submission.to_csv("submission.csv", index=False)
# print("submission.csv ready:", submission.shape)


# from sklearn.model_selection import train_test_split

# import pandas as pd
# from tqdm import tqdm
# import torch
# test_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/train.csv") 

# test_df["combined_text"] = (
#     test_df["prompt"].fillna("") +
#     "\n\nResponse A:\n" + test_df["response_a"].fillna("") +
#     "\n\nResponse B:\n" + test_df["response_b"].fillna("")
# )

# train_df, val_df = train_test_split(test_df, test_size=0.1, random_state=42, shuffle=True)



# val_labels = np.argmax(
#     val_df[["winner_model_a", "winner_model_b", "winner_tie"]].values,
#     axis=1
# )


# temperature = 1.5
# batch_size = 16
# preds = []

# for i in range(0, len(val_df), batch_size):
#     batch = val_df.iloc[i:i + batch_size]["combined_text"].tolist()
#     inputs = tokenizer(
#         batch,
#         truncation=True,
#         padding=True,
#         max_length=1536,
#         return_tensors="pt"
#     )
#     inputs = {k: v.to(quant_model.device) for k, v in inputs.items()}

#     with torch.no_grad():
#         logits = quant_model(**inputs).logits
#         logits = logits - logits.max(dim=-1, keepdim=True).values  # normalize
#         logits = logits / temperature
#         probs = torch.softmax(logits, dim=-1)
#         preds.extend(probs.cpu().tolist())

# preds = np.array(preds)

# from sklearn.metrics import log_loss

# for f in [1.0, 1.03, 1.05, 1.08, 1.1]:
#     adj = preds.copy()
#     adj[:, 2] *= f
#     adj = adj / adj.sum(axis=1, keepdims=True)
#     loss = log_loss(val_labels, adj)
#     print(f"Tie boost={f:.2f} → Log-loss={loss:.5f}")


# preds=[]
# batch_size=16
# for i in range(0, len(test_df), batch_size):
#     batch = test_df.iloc[i:i+batch_size]["combined_text"].tolist()
#     inputs = tokenizer(batch, truncation=True, padding=True, max_length=1024, return_tensors="pt")
#     inputs = {k: v.to(quant_model.device) for k, v in inputs.items()}

#     with torch.no_grad():
#         logits = quant_model(**inputs).logits
#         probs = torch.softmax(logits, dim=-1)
#         preds.extend(probs.cpu().tolist())

# preds = np.array(preds)
# submission = pd.DataFrame({
#     "id": test_df["id"],
#     "winner_model_a": preds[:, 0],
#     "winner_model_b": preds[:, 1],
#     "winner_tie": preds[:, 2]
# })
# submission.to_csv("submission.csv", index=False)
# print("submission.csv ready:", submission.shape)



# from sklearn.model_selection import train_test_split
# train_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/train.csv")

# train_df["combined_text"] = (
#     train_df["prompt"].fillna("") +
#     "\n\nResponse A:\n" + train_df["response_a"].fillna("") +
#     "\n\nResponse B:\n" + train_df["response_b"].fillna("")
# )

# # Create small validation split
# _, small_eval = train_test_split(
#     train_df,
#     test_size=0.01,  # 1% of data for quick eval
#     random_state=42,
#     shuffle=True
# )

# print("Eval size:", len(small_eval))

# batch_size = 16
# all_logits = []

# for i in tqdm(range(0, len(small_eval), batch_size)):
#     batch = small_eval.iloc[i:i+batch_size]["combined_text"].tolist()
#     inputs = tokenizer(
#         batch,
#         truncation=True,
#         padding=True,
#         max_length=1024,
#         return_tensors="pt"
#     )
#     inputs = {k: v.to(quant_model.device) for k, v in inputs.items()}

#     with torch.no_grad():
#         logits = quant_model(**inputs).logits
#         all_logits.append(logits.cpu())

# # Stack all logits
# logits = torch.cat(all_logits, dim=0)
# labels = small_eval["winner_model_a"].astype(int).to_numpy()  # <-- adjust this column name if needed

# print("Logits shape:", logits.shape, "Labels shape:", labels.shape)


# from sklearn.metrics import log_loss

# def evaluate_temps(logits, labels, temps=[0.7, 1.0, 1.2, 1.5, 2.0]):
#     for T in temps:
#         scaled_logits = logits / T
#         probs = torch.softmax(scaled_logits, dim=-1).numpy()

#         loss = log_loss(labels, probs, labels=[0, 1, 2])
#         print(f"T={T:.1f} → Log-loss={loss:.5f}")

# evaluate_temps(logits, labels)


