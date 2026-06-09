# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


! pip install textstat langdetect


# =============== Core Libraries ===============
import os
import gc
import random
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.notebook import tqdm
import re

# =============== NLP & Feature Extraction ===============
import nltk
import spacy
import textstat
from textblob import TextBlob
import langdetect


# =============== Visualization ===============
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# =============== Transformers & Modeling ===============
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_scheduler
from torch.optim import AdamW # Corrected import

# =============== Sklearn & Classical ML ===============
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# =============== Utility ===============
import warnings
warnings.filterwarnings("ignore")


# ===============ğŸ”� Reproducibility ===============
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 42
seed_everything(SEED)

# ===============ğŸ“� Paths ===============
COMP_PATH = Path("/kaggle/input/fake-or-real-the-impostor-hunt")
TRAIN_PATH = COMP_PATH / "data" / "train"
TEST_PATH  = COMP_PATH / "data" / "test"
TRAIN_CSV  = COMP_PATH / "data" / "train.csv"

# ===============ğŸš€ Device ===============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on device: {DEVICE}")



# ===============ğŸ§¾ Logging ===============
from datetime import datetime

def print_log(message: str):
    time_str = datetime.now().strftime("%H:%M:%S")
    print(f"[{time_str}] {message}")

print_log("Environment setup complete.")









train_csv = pd.read_csv(TRAIN_PATH.parent / "train.csv")
print(f"Train Shape: {train_csv.shape}")
display(train_csv.head())



# Helper to read article file
def read_article(article_id, folder_path, file_num):
    # Construct the path to the specific file within the article directory
    file_path = folder_path / f"article_{str(article_id).zfill(4)}" / f"file_{file_num}.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Build full training dataframe with text pairs
def build_train_df(df_meta, folder_path):
    fake_articles, real_articles = [], []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        fake_id = row["id"]
        real_id = row["real_text_id"]
        # Read file_1.txt for fake articles and file_2.txt for real articles
        fake_text = read_article(fake_id, folder_path, 1)
        real_text = read_article(real_id, folder_path, 2)
        fake_articles.append(fake_text)
        real_articles.append(real_text)
    df = pd.DataFrame({
        "id": df_meta["id"],
        "real_text_id": df_meta["real_text_id"],
        "fake_text": fake_articles,
        "real_text": real_articles,
        "label": 1  # All samples are positive class (fake-real pair)
    })
    return df

train_df = build_train_df(train_csv, TRAIN_PATH)


# Code: Light Cleaning of Texts
def clean_text(txt):
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"\n", " ", txt)
    return txt.strip()

for col in ["fake_text", "real_text"]:
    train_df[col] = train_df[col].apply(clean_text)




# Code: Optional â€“ Save Cache
train_df.to_parquet("train_text_pairs.parquet", index=False)
print_log("Saved cleaned train dataframe with fake-real text pairs.")



#Code: Quick Preview
print(f"Train Pairs: {train_df.shape}")
display(train_df.head())



train_df["fake_len"] = train_df["fake_text"].apply(len)
train_df["real_len"] = train_df["real_text"].apply(len)

train_df["fake_words"] = train_df["fake_text"].apply(lambda x: len(x.split()))
train_df["real_words"] = train_df["real_text"].apply(lambda x: len(x.split()))

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(train_df["fake_len"], color="red", label="Fake", kde=True, ax=axs[0])
sns.histplot(train_df["real_len"], color="green", label="Real", kde=True, ax=axs[0])
axs[0].set_title("Character Length Distribution")
axs[0].legend()

sns.histplot(train_df["fake_words"], color="red", label="Fake", kde=True, ax=axs[1])
sns.histplot(train_df["real_words"], color="green", label="Real", kde=True, ax=axs[1])
axs[1].set_title("Word Count Distribution")
axs[1].legend()

plt.suptitle("Fake vs Real Text Length Distributions", fontsize=16)
plt.tight_layout()
plt.show()



train_df["fake_readability"] = train_df["fake_text"].apply(textstat.flesch_reading_ease)
train_df["real_readability"] = train_df["real_text"].apply(textstat.flesch_reading_ease)

plt.figure(figsize=(10,5))
sns.kdeplot(train_df["fake_readability"], label="Fake", shade=True, color="red")
sns.kdeplot(train_df["real_readability"], label="Real", shade=True, color="green")
plt.title("Flesch Reading Ease Score")
plt.xlabel("Score")
plt.legend()
plt.grid(True)
plt.show()



def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.subjectivity, blob.sentiment.polarity

train_df[["fake_subjectivity", "fake_polarity"]] = train_df["fake_text"].apply(get_sentiment).apply(pd.Series)
train_df[["real_subjectivity", "real_polarity"]] = train_df["real_text"].apply(get_sentiment).apply(pd.Series)

# Compare subjectivity
plt.figure(figsize=(10,5))
sns.kdeplot(train_df["fake_subjectivity"], label="Fake", color="red", shade=True)
sns.kdeplot(train_df["real_subjectivity"], label="Real", color="green", shade=True)
plt.title("Subjectivity Distribution (Fake vs Real)")
plt.xlabel("Subjectivity")
plt.legend()
plt.grid(True)
plt.show()



from langdetect import LangDetectException

def safe_lang_detect(text):
    try:
        return langdetect.detect(text)
    except LangDetectException:
        return "unknown"

train_df["fake_lang"] = train_df["fake_text"].apply(safe_lang_detect)
train_df["real_lang"] = train_df["real_text"].apply(safe_lang_detect)

lang_df = pd.DataFrame({
    "Fake": train_df["fake_lang"].value_counts(),
    "Real": train_df["real_lang"].value_counts()
}).fillna(0).astype(int)

lang_df.plot(kind="bar", figsize=(10,5), color=["red", "green"])
plt.title("Detected Language Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


from wordcloud import WordCloud

# Fake text WordCloud
fake_text_blob = " ".join(train_df["fake_text"])
wordcloud_fake = WordCloud(width=800, height=400, background_color="white").generate(fake_text_blob)

# Real text WordCloud
real_text_blob = " ".join(train_df["real_text"])
wordcloud_real = WordCloud(width=800, height=400, background_color="white").generate(real_text_blob)

fig, axs = plt.subplots(1, 2, figsize=(18, 8))
axs[0].imshow(wordcloud_fake, interpolation="bilinear")
axs[0].axis("off")
axs[0].set_title("Fake Text WordCloud", fontsize=16)

axs[1].imshow(wordcloud_real, interpolation="bilinear")
axs[1].axis("off")
axs[1].set_title("Real Text WordCloud", fontsize=16)

plt.tight_layout()
plt.show()



import spacy
nlp = spacy.load("en_core_web_sm")



def extract_classical_features(text):
    doc = nlp(text)  # Process text with spaCy
    blob = TextBlob(text)
    # Ensure words are spaCy tokens
    words = [token for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents) # Get sentences

    return {
        "char_len": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_len": np.mean([len(token.text) for token in words]) if words else 0,
        "syllable_count": textstat.syllable_count(text),
        "readability": textstat.flesch_reading_ease(text),
        "subjectivity": blob.sentiment.subjectivity,
        "polarity": blob.sentiment.polarity,
        "noun_count": len([token for token in doc if token.pos_ == "NOUN"]),
        "verb_count": len([token for token in doc if token.pos_ == "VERB"]),
    }



# Apply on both fake and real text
fake_feats_df = train_df["fake_text"].apply(extract_classical_features).apply(pd.Series).add_prefix("fake_")
real_feats_df = train_df["real_text"].apply(extract_classical_features).apply(pd.Series).add_prefix("real_")

# Concatenate features with original df
feature_df = pd.concat([train_df, fake_feats_df, real_feats_df], axis=1)


from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

def get_sentence_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = encoder(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# Compute embeddings
tqdm.pandas()
train_df["fake_emb"] = train_df["fake_text"].progress_apply(get_sentence_embedding)
train_df["real_emb"] = train_df["real_text"].progress_apply(get_sentence_embedding)



# Convert list-like embeddings to columns
fake_emb_df = pd.DataFrame(train_df["fake_emb"].tolist()).add_prefix("fake_emb_")
real_emb_df = pd.DataFrame(train_df["real_emb"].tolist()).add_prefix("real_emb_")

# Final feature matrix
final_features = pd.concat([fake_feats_df, real_feats_df, fake_emb_df, real_emb_df], axis=1)
final_labels = train_df["real_text_id"]

print(f"Feature shape: {final_features.shape}")
final_features.head()



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score



from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(final_features, final_labels, test_size=0.2, random_state=CFG.seed, stratify=final_labels)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")


class CFG:
    seed = 42
    n_folds = 5
    device = DEVICE
    input_dim = X_train.shape[1] # X_train is not defined yet
    output_dim = 2  # binary classification
    lr = 1e-4
    batch_size = 16
    epochs = 10


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='macro')
print(f"LogReg | Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")



class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, x):
        return self.model(x)




class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super(MLPClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(128, 2)  # Output logits for 2 classes
        )
    
    def forward(self, x):
        return self.model(x)



import torch.nn as nn

class FakeTextClassifier(nn.Module):
    def __init__(self, hidden_size=768):
        super(FakeTextClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size * 2, 1)  # For binary classification between text_1 and text_2
        self.dropout = nn.Dropout(0.3)

    def forward(self, text_1_emb, text_2_emb):
        # Concatenate both embeddings
        x = torch.cat([text_1_emb, text_2_emb], dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits.squeeze(1)  # [B]



class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, embed_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}



NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["real_text_id"])):
    train_df.loc[val_idx, "fold"] = fold

train_df.head()


# Assuming num_epochs is defined in CFG
num_epochs = CFG.epochs

# Assuming you have a Dataset and DataLoader setup for your training data
# Replace this with your actual DataLoader initialization
# For demonstration purposes, let's create a dummy DataLoader
from torch.utils.data import TensorDataset, DataLoader

# Assuming final_features and final_labels are already defined and are numpy arrays
# Convert numpy arrays to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long) # Assuming labels are integers

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)

print(f"Number of batches in train_dataloader: {len(train_dataloader)}")


# Example: assume total_steps is already calculated
total_steps = len(train_dataloader) * num_epochs  # replace with your actual values

# Optimizer: AdamW with weight decay
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

# Scheduler: Cosine decay with warmup
scheduler = get_scheduler(
    name="cosine",  # options: 'linear', 'cosine', 'polynomial'
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
    num_training_steps=total_steps
)


import torch

# Save model when best validation score is achieved
model_path = f"best_model_foldIs{fold}.pkl"
torch.save(model.state_dict(), model_path)
print(f"âœ… Model saved to {model_path}")



model.load_state_dict(torch.load(f"best_model_foldIs{fold}.pkl"))
model.to(device)
model.eval()



import os

test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test/"
test_ids = sorted([f for f in os.listdir(test_dir) if f.startswith("article_")])

# Load all articles (id is inferred from filename)
test_articles = {}
for fname in test_ids:
    aid = int(fname.split("_")[1])
    with open(os.path.join(test_dir, fname), "r", encoding="utf-8") as f:
        test_articles[aid] = f.read()

# Generate all possible (fake, real) pairs for each test ID
test_pairs = []
for i in range(0, len(test_articles), 2):
    fake_id = i
    real_id = i + 1
    fake_text = test_articles[fake_id]
    real_text = test_articles[real_id]
    
    test_pairs.append({
        "id": i // 2,
        "fake_text_id": fake_id,
        "real_text_id": real_id,
        "fake_text": fake_text,
        "real_text": real_text,
    })

test_df = pd.DataFrame(test_pairs)





