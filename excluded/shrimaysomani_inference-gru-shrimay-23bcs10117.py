import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.optim as optim


import pandas as pd
import numpy as np
final_df = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
train = pd.read_csv('/kaggle/input/lmsys-kfolds/train_5folds.csv')
final_df.head()


final_df['text'] = 'User prompt: ' +final_df['prompt'] +  '\n\nModel A :\n' + final_df['response_a'] +'\n\n--------\n\nModel B:\n'  + final_df['response_b']
print(final_df['text'][0])
train['text'] = 'User prompt: ' + train['prompt'] +  '\n\nModel A :\n' + train['response_a'] +'\n\n--------\n\nModel B:\n'  + train['response_b']


print(len(final_df))
final_df.head()


final_texts = final_df['text'].values

kfold = 0
test_texts = train[train['kfold']==kfold]['text'].values
train_texts = train[train['kfold']!=kfold]['text'].values

len(test_texts)+len(train_texts)
len(test_texts)+len(train_texts)


batch_size = 8
num_classes = 3

# Tokenize
test_tokenized = [t.split() for t in test_texts]
train_tokenized = [t.split() for t in train_texts]
print(len(test_tokenized)+len(train_tokenized))

# Build vocabulary
vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in test_tokenized for w in sent):
    vocab[word] = len(vocab)

for word in Counter(w for sent in train_tokenized for w in sent):
    vocab[word] = len(vocab)

def encode(sentence):
    return torch.tensor([vocab.get(w, 1) for w in sentence])



import torch.nn as nn

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
        output = self.fc(h_n[-1])
        logits = self.fc2(output)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model_loaded = GRUClassifier(vocab_size=len(vocab))
model_loaded.load_state_dict(torch.load("/kaggle/input/training-gru-shrimay-23bcs10117/gru_classifier_0.pth", map_location=device))
model_loaded = model_loaded.to(device)
model_loaded.eval()



import torch.nn.functional as F

def predict(text):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model_loaded(encoded)
        probs = F.softmax(logits, dim=1)
        return probs

class_0_prob = []
class_1_prob = []
class_2_prob = []

for text in final_texts:
    ans = predict(text)
    class_0_prob.append(float(ans[0][0]))
    class_1_prob.append(float(ans[0][1]))
    class_2_prob.append(float(ans[0][2]))



final_df['winner_model_a']=class_0_prob
final_df['winner_model_b']=class_1_prob
final_df['winner_tie']=class_2_prob
final_df[['id','winner_model_a','winner_model_b','winner_tie']].head()


final_df[['id','winner_model_a','winner_model_b','winner_tie']].to_csv('submission.csv',index=False)

