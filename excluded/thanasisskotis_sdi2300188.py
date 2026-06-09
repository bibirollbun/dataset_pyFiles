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


!pip install nltk contractions gensim ftfy torch-optimizer

import os
import re
import random
import string
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data
import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import emoji
import contractions
import ftfy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay)
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

nltk.download('all')
!unzip -o /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/

TRAIN_PATH      = "/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/train_dataset.csv"
VAL_PATH        = "/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/val_dataset.csv"
TEST_PATH       = "/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/test_dataset.csv"
SUBMISSION_PATH = "/kaggle/working/submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

random_seed = 37
np.random.seed(random_seed)
random.seed(random_seed)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

corrections = {r"\&quot\b": "", r"\&amp\b": "", r"\bgonna\b": "going to", r"\bgotta\b": "got to", r"\bwanna\b": "want to", r"\bain't\b": "is not", r"\bdunno\b": "do not know", r"\bya\b": "you", r"\bim\b": "i am", r"\bi'm\b": "i am", r"\blet's\b": "let us", r"\blemmie\b": "let me", r"\bbrb\b": "be right back", r"\bttyl\b": "talk to you later", r"\bsmh\b": "shaking my head", r"\bafaik\b": "as far as i know", r"\bomg\b": "oh my god", r"\bfyi\b": "for your information", r"\btbh\b": "to be honest", r"\bidk\b": "i do not know", r"\brly\b": "really", r"\bplz\b": "please", r"\bthx\b": "thanks", r"\bty\b": "thank you", r"\bnp\b": "no problem", r"\bbtw\b": "by the way", r"\blmao\b": "laughing my ass off", r"\blol\b": "laugh out loud", r"\brofl\b": "rolling on the floor laughing", r"\bikr\b": "i know right", r"\bcoz\b": "because", r"\bshud\b": "should", r"\bdoesnt\b": "does not", r"\bcant\b": "can not", r"\bwont\b": "will not", r"\bive\b": "i've", r"\bthru\b": "through", r"\bur\b": "your", r"\blolol\b": "lol", r"\bsumtimes\b": "sometimes", r"\bu\b": "you", r"\bminisota\b": "minnesota", r"\beated\b": "ate", r"\b2moro\b": "tomorrow", r"\btommorow\b": "tomorrow", r"\bwaitin\b": "waiting", r"\bbck\b": "back", r"\bnvm\b": "never mind", r"\bwheely\b": "wheeled", r"\bschbag\b": "schoolbag", r"\btwippl\b": "tweeps", r"\bkno\b": "know", r"\bcont\b": "continue", r"\bluv\b": "love", r"\b4get\b": "forget", r"\bdat\b": "that", r"\bya'll\b": "you all", r"\btho\b": "though", r"\bdis\b": "this", r"\bsrsly\b": "seriously", r"\bidc\b": "i do not care", r"\bimma\b": "i am going to", r"\bnite\b": "night", r"\bcya\b": "see you", r"\bdat's\b": "that is", r"\b4ever\b": "forever", r"\bimo\b": "in my opinion", r"\bimho\b": "in my humble opinion", r"\bwtf\b": "what the heck", r"\bwtg\b": "way to go", r"\bwbu\b": "what about you", r"\btysm\b": "thank you so much", r"\b2day\b": "today", r"\b2nite\b": "tonight", r"\bneva\b": "never", r"\bgud\b": "good", r"\bexcitd\b": "excited", r"\bpleez\b": "please", r"\bsorrye\b": "sorry", r"\btechincal\b": "technical", r"\brecieve\b": "receive", r"\bdefinately\b": "definitely", r"\bseperate\b": "separate", r"\boccured\b": "occurred", r"\buntill\b": "until", r"\bwich\b": "which", r"\bthier\b": "their", r"\bbeleive\b": "believe", r"\bcomming\b": "coming", r"\bgoverment\b": "government", r"\bpublically\b": "publicly", r"\breccomend\b": "recommend", r"\barguement\b": "argument", r"\bcalender\b": "calendar", r"\bconsciencious\b": "conscientious", r"\bembarass\b": "embarass", r"\bneccessary\b": "necessary", r"\bcommited\b": "committed", r"\boccassion\b": "occasion", r"\bverry\b": "very", r"\bsoooo\b": "so", r"\bexampel\b": "example", r"\breciept\b": "receipt", r"\bdoe\b": "though"}

contractions_expanded = {
    r"\bwhere're\b": "where are", r"\bhe'd\b": "he would", r"\bmustn't\b": "must not", r"\bwho'll\b": "who will", r"\bthey've\b": "they have", r"\bwhere'd\b": "where did", r"\bwhere'll\b": "where will", r"\bi'm\b": "I am", r"\bi'll've\b": "I will have", r"\bwhat've\b": "what have", r"\byou're\b": "you are", r"\bthey're\b": "they are", r"\bwhat'll\b": "what will", r"\bhow'll\b": "how will", r"\bhow're\b": "how are", r"\bhe'll've\b": "he will have", r"\bwho'd've\b": "who would have", r"\bwhat'd\b": "what did", r"\bthey'd\b": "they would", r"\bwhere'd've\b": "where did have", r"\bi'd've\b": "I would have", r"\byou'll've\b": "you will have", r"\bwhat's\b": "what is", r"\bthey'll\b": "they will", r"\bi've\b": "I have", r"\bwho're\b": "who are", r"\bhow's\b": "how is", r"\bweren't\b": "were not", r"\bwhat'll've\b": "what will have", r"\bthey'd've\b": "they would have", r"\bit'd\b": "it would", r"\bwho's\b": "who is", r"\bi'll\b": "I will", r"\bwhat'd've\b": "what did have", r"\bthey'll've\b": "they will have", r"\bthere's\b": "there is", r"\bwho've\b": "who have", r"\bhow'd\b": "how did", r"\bwhere's\b": "where is", r"\bmust've\b": "must have", r"\bwho'll've\b": "who will have", r"\bcan't\b": "can not", r"\bthat'll\b": "that will", r"\bwasn't\b": "was not", r"\bthey'll\b": "they will", r"\bi'd\b": "I would", r"\bshould've\b": "should have", r"\bthey'd've\b": "they would have", r"\bthat'd\b": "that would", r"\bit'd've\b": "it would have", r"\bit'll\b": "it will", r"\bhe'd've\b": "he would have", r"\bit'll've\b": "it will have", r"\bthey'd\b": "they would", r"\bit's\b": "it is", r"\bain't\b": "is not", r"\bthat'll've\b": "that will have", r"\bwhere'll've\b": "where will have", r"\bwhere'll\b": "where will", r"\byou've\b": "you have", r"\bwhat're\b": "what are", r"\bshouldn't\b": "should not", r"\bwho'd\b": "who would", r"\bthey're\b": "they are", r"\bthere'll've\b": "there will have", r"\bthere'll\b": "there will", r"\bthat'd've\b": "that would have", r"\bdoesn't\b": "does not", r"\bthere'd\b": "there would", r"\bthere'd've\b": "there would have", r"\bthat'd\b": "that would", r"\bwhat's\b": "what is", r"\bwho's\b": "who is", r"\bneedn't\b": "need not", r"\bshould've\b": "should have", r"\bhe's\b": "he is", r"\bdidn't\b": "did not", r"\byou'll\b": "you will", r"\byou'd\b": "you would", r"\byou'd've\b": "you would have", r"\bthere're\b": "there are", r"\blet's\b": "let us", r"\bthey'll've\b": "they will have", r"\bmightn't\b": "might not", r"\bmight've\b": "might have", r"\baint\b": "is not"
}

slang_dict = {
    r"\blmao\b": "laughing my ass off", r"\bhmu\b": "hit me up", r"\bbruh\b": "bro", r"\bwoudlnt\b": "would not", r"\bsmh\b": "shaking my head", r"\bgr8\b": "great", r"\bwont\b": "will not", r"\bl8r\b": "later", r"\bidk\b": "I don't know", r"\bttyl\b": "talk to you later", r"\bcya\b": "see you", r"\btbh\b": "to be honest", r"\bimo\b": "in my opinion", r"\brofl\b": "rolling on the floor laughing", r"\bhav\b": "have", r"\bfrm\b": "from", r"\bbff\b": "best friends forever", r"\bthx\b": "thanks", r"\bgtfo\b": "get the freak out", r"\bgtg\b": "got to go", r"\bjk\b": "just kidding", r"\bevery1\b": "everyone", r"\bu\b": "you", r"\bnp\b": "no problem", r"\byer\b": "you are", r"\bcuz\b": "because", r"\bbtw\b": "by the way", r"\bwtf\b": "what the heck", r"\br\b": "are", r"\bhols\b": "holidays", r"\basap\b": "as soon as possible", r"\by'all\b": "you all", r"\byay\b": "yes", r"\b2night\b": "tonight", r"\bfb\b": "facebook", r"\bhol\b": "holiday", r"\barent\b": "are not", r"\bnathing\b": "nothing", r"\blmfao\b": "laughing my freaking ass off", r"\bppl\b": "people", r"\bshouldnt\b": "should not", r"\bplz\b": "please", r"\btmi\b": "too much information", r"\bwbu\b": "what about you", r"\bgonna\b": "going to", r"\bsis\b": "sister", r"\bfyi\b": "for your information", r"\bomg\b": "oh my god", r"\bomfg\b": "oh my freaking god", r"\bbein\b": "being", r"\bda\b": "the", r"\bkno\b": "know", r"\bfml\b": "forget my life", r"\blemme\b": "let me", r"\brecieve\b": "receive", r"\bdefinately\b": "definitely", r"\boccured\b": "occurred", r"\bseperate\b": "separate", r"\buntill\b": "until", r"\bwich\b": "which", r"\bwoudl\b": "would", r"\bbeleive\b": "believe", r"\bthier\b": "their", r"\bgoverment\b": "government", r"\btommorow\b": "tomorrow", r"\benviroment\b": "environment", r"\baccomodate\b": "accommodate", r"\bpublically\b": "publicly", r"\bneccessary\b": "necessary", r"\bharrass\b": "harass"
}


def correct_spelling(text):
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def expand_contractions(text):
    for pattern, replacement in contractions_expanded.items():
        text = re.sub(pattern, replacement, text)
    return contractions.fix(text)


def expand_slang(text):
    for pattern, replacement in slang_dict.items():
        text = re.sub(rf"\b{pattern}\b", replacement, text, flags=re.IGNORECASE)
    return text


def reduce_repeated_letters(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def preprocess_text(text):
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "emailaddr", text)
    text = re.sub(r"\+?\d[\d\-\(\) ]{7,}\d", "phonenum", text)
    text = re.sub(r"&\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = ftfy.fix_text(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = reduce_repeated_letters(text)
    text = expand_slang(text)
    text = correct_spelling(text)
    text = expand_contractions(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens]


train_df["tokens"] = train_df["Text"].astype(str).apply(preprocess_text)
val_df["tokens"]   = val_df["Text"].astype(str).apply(preprocess_text)
test_df["tokens"]  = test_df["Text"].astype(str).apply(preprocess_text)

x_train, y_train = train_df["Text"], train_df["Label"]
x_val,   y_val   = val_df["Text"],   val_df["Label"]
x_test           = test_df["Text"]

# Convert GloVe format to word2vec format (required by KeyedVectors)
glove_input_file    = "/kaggle/input/pretrained-glove/glove.twitter.27B.200d.txt"
word2vec_output_file = "/kaggle/working/glove.twitter.27B.200d.w2v.txt"

if not os.path.exists(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)

word2vec = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
print(f"Loaded {len(word2vec.key_to_index)} word vectors.")


class TweetDataset(Dataset):
    def __init__(self, token_lists, labels, embeddings, max_len=40):
        self.token_lists   = token_lists
        self.labels        = labels
        self.embeddings    = embeddings
        self.max_len       = max_len
        self.embedding_dim = embeddings.vector_size
        # Learnable fallback for OOV tokens
        self.unk_embedding = nn.Parameter(torch.randn(self.embedding_dim), requires_grad=True)

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, idx):
        tokens = self.token_lists[idx] or ["<UNK>"]
        vectors = [
            self.embeddings[t] if t in self.embeddings else self.unk_embedding.detach().numpy()
            for t in tokens[:self.max_len]
        ]
        # Pad to max_len with zeros
        while len(vectors) < self.max_len:
            vectors.append(np.zeros(self.embedding_dim))
        X = torch.from_numpy(np.array(vectors)).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)
        return X, y


train_dataset = TweetDataset(train_df["tokens"].tolist(), train_df["Label"].tolist(), word2vec)
val_dataset   = TweetDataset(val_df["tokens"].tolist(),   val_df["Label"].tolist(),   word2vec)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=64)

train_dataset.training = True
val_dataset.training   = False

device = torch.device("cpu")


class FeedForwardNet(nn.Module):
    def __init__(self, dropout=0.15):
        super().__init__()
        self.fc1, self.bn1 = nn.Linear(200, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn2 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc3, self.bn3 = nn.Linear(256, 128), nn.BatchNorm1d(128)
        self.fc4, self.bn4 = nn.Linear(128,  64), nn.BatchNorm1d(64)
        self.fc5, self.bn5 = nn.Linear( 64,  32), nn.BatchNorm1d(32)
        self.output_layer  = nn.Linear(32, 1)
        # Residual projections to match dimensions across blocks
        self.res1 = nn.Linear(200, 512)
        self.res2 = nn.Linear(512, 256)
        self.res3 = nn.Linear(256, 128)
        self.res4 = nn.Linear(128,  64)
        self.res5 = nn.Linear( 64,  32)
        self.act     = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def _block(self, fc, bn, res, x):
        out = self.dropout(self.act(bn(fc(x))))
        return out + res(x)

    def forward(self, x):
        x = x.mean(dim=1)  # Average token embeddings into a sentence vector
        x = self._block(self.fc1, self.bn1, self.res1, x)
        x = self._block(self.fc2, self.bn2, self.res2, x)
        x = self._block(self.fc3, self.bn3, self.res3, x)
        x = self._block(self.fc4, self.bn4, self.res4, x)
        x = self._block(self.fc5, self.bn5, self.res5, x)
        return self.output_layer(x)


model     = FeedForwardNet().to(device)
loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=30)


def train_epoch(model, loader, optimizer, loss_func):
    model.train()
    losses = []
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        loss = loss_func(model(x_batch), y_batch.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            preds = (torch.sigmoid(model(x_batch.to(device))) > 0.5).float()
            y_true.extend(y_batch.numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return acc, p, r, f


def evaluate_loss(model, loader, loss_func):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            total += loss_func(model(x), y).item() * x.size(0)
    return total / len(loader.dataset)


EPOCHS         = 50
patience       = 15
best_acc       = 0
no_improvement = 0
best_model_path = "/kaggle/working/best_model.pth"
history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
           'val_loss':   [], 'val_acc':   [], 'val_f1':   []}

print(f"Training on {device}...")
for epoch in range(EPOCHS):
    train_loss                            = train_epoch(model, train_loader, optimizer, loss_func)
    train_acc, train_p, train_r, train_f1 = evaluate(model, train_loader)
    val_loss                              = evaluate_loss(model, val_loader, loss_func)
    val_acc, val_p, val_r, val_f1         = evaluate(model, val_loader)
    scheduler.step()

    for k, v in zip(history, [train_loss, train_acc, train_f1, val_loss, val_acc, val_f1]):
        history[k].append(v)

    print(f"Epoch {epoch+1}/{EPOCHS}:")
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, P={train_p:.4f}, R={train_r:.4f}, F1={train_f1:.4f}")
    print(f"  Val:   Loss={val_loss:.4f},   Acc={val_acc:.4f},   P={val_p:.4f},   R={val_r:.4f},   F1={val_f1:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  Best model saved (val acc={val_acc:.4f})")
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load(best_model_path))
model.eval()

plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'],   label='Val Loss')
plt.title('Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_f1'], label='Train F1')
plt.plot(history['val_f1'],   label='Val F1')
plt.title('F1 over Epochs'); plt.xlabel('Epoch'); plt.ylabel('F1'); plt.legend()
plt.tight_layout()
plt.savefig('/kaggle/working/training_history.png')
plt.show()


def get_preds_and_labels(model, loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            probs = torch.sigmoid(model(x_batch.to(device))).cpu().numpy().flatten()
            y_true.extend(y_batch.numpy().flatten())
            y_prob.extend(probs)
    return np.array(y_true), np.array(y_prob)


y_true_val, y_prob_val = get_preds_and_labels(model, val_loader)
fpr, tpr, _ = roc_curve(y_true_val, y_prob_val)
roc_auc     = auc(fpr, tpr)
y_pred_val  = (y_prob_val > 0.5).astype(int)

plt.figure(figsize=(18, 8))
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss'); plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history['train_acc'], label='Train Acc'); plt.plot(history['val_acc'], label='Val Acc')
plt.ylim(0.45, 0.85); plt.yticks(np.arange(0.45, 0.86, 0.02))
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history['train_f1'], label='Train F1'); plt.plot(history['val_f1'], label='Val F1')
plt.ylim(0.65, 0.7); plt.yticks(np.arange(0.71, 0.86, 0.01))
plt.title('F1'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("/kaggle/working/separated_metric_plots.png"); plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
ConfusionMatrixDisplay(confusion_matrix(y_true_val, y_pred_val)).plot(ax=plt.gca(), colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig("/kaggle/working/final_evaluation_plots.png"); plt.show()

test_dataset = TweetDataset(test_df["tokens"].tolist(), [0]*len(test_df), word2vec)
test_loader  = DataLoader(test_dataset, batch_size=64)

model.load_state_dict(torch.load(best_model_path))
model.eval()

test_preds = []
with torch.no_grad():
    for x_batch, _ in test_loader:
        probs = torch.sigmoid(model(x_batch.to(device))).cpu().numpy().flatten()
        test_preds.extend((probs > 0.5).astype(int))

id_column  = "ID" if "ID" in test_df.columns else "Id"
submission = pd.DataFrame({id_column: test_df[id_column], "Label": test_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")


####################
######EXPERIMENTS###
###################

'''
### 1-6 Layer experiments ###

nn.Linear(512, 256)
nn.Linear(256, 256)
nn.Linear(256, 128)
nn.Linear(128, 64)
nn.Linear(64, 32)
nn.Linear(32, 1)



### Activation Functions ###

nn.ReLU()
nn.SELU()
nn.ELU()
nn.GELU()
nn.Leaky_Relu()


### Loss Functions ###

nn.MSE(reduction = "mean")
nn.MSE(reduction = "sum")
nn.BCELoss()
nn.BCEWithLogitsLoss()


class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.02):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(preds, targets)

loss_func = BCEWithLogitsLossWithSmoothing(smoothing)

nn.CrossEntropyLoss()



### Optimizers and Learning Rate ###


torch.optim.Adam(model.parameters(), lr = 1e-3)
torch.optim.AdamW(model.parameters(), lr = 1e-3)
torch.optim.AdamW(model.parameters(), lr = 5e-4)
torch.optim.RMSprop(model.parameters(), lr = 5e-4)
torch.optim.RMSprop(model.parameters(), lr = 1e-3)
torch.optim.Adagrad(model.parameters(), lr = 0.01)
torch.optim.Adagrad(model.parameters(), lr = 0.05)


### Dropout ###

nn.Dropout(0.1)
nn.Dropout(0.15)
nn.Dropout(0.2)



### Normalization ###

nn.BatchNorm1d(hidden_dim) with batch_size = 32/64/128 and nn.ELU()

nn.LayerNorm(hidden_dim) with batch_size = 32/64/128 and nn.GELU()

nn.LayerNorm(hidden_dim) with batch_size = 64 and nn.ELU()



### Weight Decay ###

torch.optim.RMSprop(model.parameters(), lr = 5e-5)
torch.optim.RMSprop(model.parameters(), lr = 1e-5)
torch.optim.RMSprop(model.parameters(), lr = 5e-4)
torch.optim.RMSprop(model.parameters(), lr = 1e-4)


### Early Stopping ###

patience = 10 
patience = 15
patience = 20


### Lookahead ###

base_optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-4, weight_decay=1e-5)
optimizer = Lookahead(base_optimizer)


### Scheduler ###

CosineAnnealingLR(optimizer, T_max=40)
CosineAnnealingLR(optimizer, T_max=50)
StepLR(step_size = 10, gamma = 0.1)
ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)


### Skip Connections ###

self.res1 = nn.Linear(200, 512)  # Project input to match fc1 output
self.res2 = nn.Linear(512, 256)
self.res3 = nn.Linear(256, 128)
self.res4 = nn.Linear(128, 64)
self.res5 = nn.Linear(64, 32)

x = out + self.res(residual)


### Input Pooling ###

x_mean = x.mean(dim=1)
x_max = x.max(dim=1).values
x_std = x.std(dim=1)
x = torch.cat([x_mean, x_max], dim=1)
x = torch.cat([x_mean x_std], dim=1)
x = torch.cat([x_mean, x_max, x_std], dim=1)


### Token Sequence Length ###

max_len = 25
max_len = 40
max_len = 50


### The gloves tested are from the datasets in the input ###



'''

