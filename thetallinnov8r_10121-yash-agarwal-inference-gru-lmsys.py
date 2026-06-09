import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


train = pd.read_csv('/kaggle/input/10121-yash-agarwal-kfolds-lmsys/train_5folds.csv')
test_df = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')

train['text'] = (
    'User prompt: ' + train['prompt'] +
    '\n\nModel A :\n' + train['response_a'] +
    '\n\n--------\n\nModel B:\n' + train['response_b']
)

test_df['text'] = (
    'User prompt: ' + test_df['prompt'] +
    '\n\nModel A :\n' + test_df['response_a'] +
    '\n\n--------\n\nModel B:\n' + test_df['response_b']
)

test_texts = test_df['text'].values


all_texts_tokenized = [t.split() for t in train['text'].values]

vocab = {"<pad>": 0, "<unk>": 1}
word_counts = Counter(word for sent in all_texts_tokenized for word in sent)

for word in word_counts:
    vocab[word] = len(vocab)

vocab_size = len(vocab)
print("Global vocab size:", vocab_size)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        out = self.fc(h_n[-1])
        return self.fc2(out)


def encode(tokens):
    return torch.tensor([vocab.get(w, 1) for w in tokens]).unsqueeze(0).to(device)


FOLDS = 5
fold_predictions = []

for k in range(FOLDS):
    print(f"\n======================================")
    print(f" Running GRU inference for Fold {k}")
    print(f"======================================")

    model_path = f"/kaggle/input/10121-yash-agarwal-training-gru-lmsys/best_gru_fold{k}.pth"
    
    model = GRUClassifier(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    fold_probs = []

    for text in tqdm(test_texts):
        tokens = text.split()
        inp = encode(tokens)

        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        fold_probs.append(probs)

    fold_predictions.append(fold_probs)


fold_predictions = np.array(fold_predictions)
final_probs = fold_predictions.mean(axis=0)

test_df['winner_model_a'] = final_probs[:,0]
test_df['winner_model_b'] = final_probs[:,1]
test_df['winner_tie']     = final_probs[:,2]

test_df[['id','winner_model_a','winner_model_b','winner_tie']].to_csv("submission.csv", index=False)
print("\n submission.csv created successfully!")

