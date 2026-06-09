import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


import pandas as pd
import numpy as np
final_df = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
train = pd.read_csv('/kaggle/input/10042-ishan-creating-folds-lmsys/train_5folds.csv')
final_df.head()


final_df['text'] = 'User prompt: ' +final_df['prompt'] +  '\n\nModel A :\n' + final_df['response_a'] +'\n\n--------\n\nModel B:\n'  + final_df['response_b']
print(final_df['text'][0])
train['text'] = 'User prompt: ' + train['prompt'] +  '\n\nModel A :\n' + train['response_a'] +'\n\n--------\n\nModel B:\n'  + train['response_b']


print(len(final_df))
final_df.head()


final_texts = final_df['text'].values
batch_size = 8
num_classes = 3
kfolds = 5


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.gru(x)
        output = self.fc(h_n[-1])  # use final hidden state
        logits = self.fc2(output)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
def encode(sentence):
    return torch.tensor([vocab.get(w, 1) for w in sentence])

def predict(text, model):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(encoded)
        probs = F.softmax(logits, dim=1)
        return probs


final_df['winner_model_a'] = 0.0
final_df['winner_model_b'] = 0.0
final_df['winner_tie'] = 0.0

epoches_to_use = [0,0,0,0,0]

for kfold in range(kfolds):
    print(f"Prediction fold: {kfold}")

    test_texts = train[train['kfold'] == kfold]['text'].values
    train_texts = train[train['kfold'] != kfold]['text'].values

    test_tokenized = [t.split() for t in test_texts]
    train_tokenized = [t.split() for t in train_texts]

    vocab = {"<pad>": 0, "<unk>": 1}
    for word in Counter(w for sent in test_tokenized for w in sent):
        vocab[word] = len(vocab)
    for word in Counter(w for sent in train_tokenized for w in sent):
        vocab[word] = len(vocab)

    model_path = f"/kaggle/input/10042-ishan-gru-training-lmsys/gru_classifier_kfold_{kfold}_epoch_{epoches_to_use[kfold]}.pth"
    model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)
    model_loaded.load_state_dict(torch.load(model_path, map_location=device))
    model_loaded.eval()

    for i, text in enumerate(final_texts):
        probs = predict(text, model_loaded)
        final_df.at[i, 'winner_model_a'] += float(probs[0][0]) / kfolds
        final_df.at[i, 'winner_model_b'] += float(probs[0][1]) / kfolds
        final_df.at[i, 'winner_tie'] += float(probs[0][2]) / kfolds


final_df[['id','winner_model_a','winner_model_b','winner_tie']].to_csv('submission.csv', index=False)
print("Done")




