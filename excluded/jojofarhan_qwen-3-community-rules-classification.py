import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df= pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df.head(10)


df = df.drop(columns=['row_id', 'subreddit', 'positive_example_1','positive_example_2','negative_example_1','negative_example_2'])	
df.head()


df['clean'] = df['body'].str.replace(r"<[^>]+>", " ", regex=True)
df['clean'] = df['body'].str.lower()
df['clean'] = df['clean'].str.replace(r'http\S+|www.\S+', '', regex=True)
df['clean'] = df['clean'].str.replace(r'\s+', ' ', regex=True).str.strip()
df['clean'] = df['clean'].str.replace(r'\[.*?\]\(.*?\)', '', regex=True)
df['clean'] = df['clean'].str.replace(r'[^a-z\s]', ' ', regex=True)
df['clean'] = df['clean'].str.replace(r'\s+', ' ', regex=True).str.strip()

df = df[['clean','rule', 'rule_violation']]
df.head(10)


from tqdm import tqdm
import spacy

# Enable GPU
spacy.prefer_gpu()
print(f"Using GPU: {spacy.require_gpu()}")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


tqdm.pandas()

def remove_stopwords(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])

# Apply stopword removal
df['body'] = df['clean'].progress_apply(remove_stopwords)

# Drop old review column, keep clean text
df = df[['body','rule', 'rule_violation']]

df.head()


MODEL_NAME = "/kaggle/input/qwen2.5-coder/transformers/3b/1"   # Hugging Face Qwen embedding model
MAX_LEN = 512

device = "cuda" if torch.cuda.is_available() else "cpu"


train_df = df
# train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
train_df.head()


def build_prompt(row):
    return f"Rule: {row['rule']}\nComment: {row['body']}"

train_df["prompt"] = train_df.apply(build_prompt, axis=1)
test_df["prompt"] = test_df.apply(build_prompt, axis=1)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size].tolist()
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            # mean pooling
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(emb)
    return np.vstack(all_embeddings)


train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["prompt"],
    train_df["rule_violation"],
    stratify=train_df["rule_violation"],
    test_size=0.2,
    random_state=42
)


print("Encoding train...")
train_embeddings = get_embeddings(train_texts)
print("Encoding val...")
val_embeddings = get_embeddings(val_texts)


from xgboost import XGBClassifier
clf = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    use_label_encoder=False
)
clf.fit(train_embeddings, train_labels)


val_preds = clf.predict_proba(val_embeddings)[:, 1]
roc = roc_auc_score(val_labels, val_preds)
print("Validation ROC AUC:", roc)


fpr, tpr, thresholds = roc_curve(val_labels, val_preds)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc:.3f})", color="blue")
plt.plot([0,1], [0,1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Validation Set")
plt.legend()
plt.show()



# Turn probabilities into binary predictions (threshold = 0.5)
val_pred_labels = (val_preds >= 0.5).astype(int)

cm = confusion_matrix(val_labels, val_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])

plt.figure(figsize=(5,5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix on Validation Set")
plt.show()


print("Encoding test...")
test_embeddings = get_embeddings(test_df["prompt"])
test_preds = clf.predict_proba(test_embeddings)[:, 1]

test_df["rule_violation"] = test_preds


submission = test_df[["row_id", "rule_violation"]]
submission.to_csv("submission.csv", index=False)

print("Files in current directory:", os.listdir())
print(submission.head())

