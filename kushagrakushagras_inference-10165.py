# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as param # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


final_df = param.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
train = param.read_csv('/kaggle/input/5folds-train/5folds_trains.csv')
final_df.head()


final_df['text'] = 'User prompt: ' +final_df['prompt'] +  '\n\nModel A :\n' + final_df['response_a'] +'\n\n--------\n\nModel B:\n'  + final_df['response_b']
print(final_df['text'][0])
train['text'] = 'User prompt: ' + train['prompt'] +  '\n\nModel A :\n' + train['response_a'] +'\n\n--------\n\nModel B:\n'  + train['response_b']


final_texts = final_df['text'].values
batch_size = 8
num_classes = 3
kfolds = 5


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=3):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.gru_layer = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.projection_layer = nn.Linear(hidden_dim, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        output, h_n = self.gru_layer(x)
        h_last = h_n[-1]
        x = self.projection_layer(h_last)
        logits = self.output_layer(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
def encode(sentence):
    return torch.tensor([vocab.get(w, 1) for w in sentence])

def predict(text):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0)
    with torch.no_grad():
        logits = model_loaded(encoded)
        # pred = torch.argmax(logits, dim=1).item()
        probs = F.softmax(logits, dim=1)
        # print(probs)
        return probs


epoches_to_use = [0,0,0,0,0] 
class_0_probs = []
class_1_probs = []
class_2_probs = []
for kfold in range(kfolds):
    print(f"prediction fold: {kfold}")
    test_texts = train[train['kfold']==kfold]['text'].values
    train_texts = train[train['kfold']!=kfold]['text'].values
    
    len(test_texts)+len(train_texts)
    len(test_texts)+len(train_texts)
    
    test_tokenized = [t.split() for t in test_texts]
    train_tokenized = [t.split() for t in train_texts]
    print(len(test_tokenized)+len(train_tokenized))
    
    vocab = {"<pad>": 0, "<unk>": 1}
    for word in Counter(w for sent in test_tokenized for w in sent):
        vocab[word] = len(vocab)
    
    for word in Counter(w for sent in train_tokenized for w in sent):
        vocab[word] = len(vocab)
    model_loaded = GRUClassifier(vocab_size=len(vocab))  # same init as before
    model_loaded.load_state_dict(torch.load(f"/kaggle/input/training-10165/gru_classifier_fold_{kfold}_epoch_{epoches_to_use[kfold]}.pth", map_location=device))
    model_loaded.eval()

    

    class_0_prob = []
    class_1_prob = []
    class_2_prob = []
    for text in final_texts:
        ans = predict(text)
        class_0_prob.append(float(ans[0][0]))
        class_1_prob.append(float(ans[0][1]))
        class_2_prob.append(float(ans[0][2]))

    class_0_probs.append(class_0_prob)
    class_1_probs.append(class_1_prob)
    class_2_probs.append(class_2_prob)



final_df['winner_model_a']=np.sum(np.array(class_0_probs), axis=0)/5
final_df['winner_model_b']=np.sum(np.array(class_1_probs), axis=0)/5
final_df['winner_tie']=np.sum(np.array(class_2_probs), axis=0)/5
final_df[['id','winner_model_a','winner_model_b','winner_tie']].head()


final_df[['id','winner_model_a','winner_model_b','winner_tie']].to_csv('submission.csv',index=False)







