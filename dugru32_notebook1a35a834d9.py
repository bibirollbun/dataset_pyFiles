import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# Ορισμός συσκευής
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: " + str(device))

# --- Φόρτωση Δεδομένων ΓΙΑ ΤΟ TWITTER DATASET ---
data_dir = "/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/"
train_df = pd.read_csv(f"{data_dir}train_dataset.csv")
val_df = pd.read_csv(f"{data_dir}val_dataset.csv")
# Δεν υπάρχει ξεχωριστό test set σε αυτό το σημείο, θα χρησιμοποιήσουμε το val_df για "test" όπως στο αρχικό παράδειγμα
test_set = val_df.copy() # Δημιουργούμε ένα αντίγραφο του val_df για να μιμηθούμε το test set


# --- Προετοιμασία Δεδομένων ---
porter_stemmer = PorterStemmer()

# Δημιουργία 'tokenized_text' και 'stemmed_tokens' στα train_df και val_df ξεχωριστά
train_df['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in train_df['Text']]
train_df['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in train_df['tokenized_text']]

val_df['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in val_df['Text']]
val_df['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in val_df['tokenized_text']]

print("Columns in train_df before split:")
print(train_df.columns)
print("\nColumns in val_df before split:")
print(val_df.columns)


# --- Train Test Split (Χρήση του val_df ως test) ---
def split_train_test(train_df, test_df, shuffle_state=True):
    X_train = train_df[['stemmed_tokens']]
    Y_train = train_df['Label']
    X_test = test_df[['stemmed_tokens']]
    Y_test = test_df['Label']

    print("Value counts for Train sentiments")
    print(Y_train.value_counts())
    print("Value counts for Test sentiments")
    print(Y_test.value_counts())
    print(type(X_train))
    print(type(Y_train))
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.to_frame()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
    print(X_train.head())
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = split_train_test(train_df, val_df)

# --- Word2Vec Μοντέλο ---
# Συνένωση stemmed tokens από train και val για εκπαίδευση Word2Vec
all_stemmed_tokens = pd.concat([train_df['stemmed_tokens'], val_df['stemmed_tokens']], ignore_index=True)

size = 500
window = 3
min_count = 1
workers = 3
sg = 0
OUTPUT_FOLDER = '/kaggle/working/'

def make_word2vec_model(all_stemmed_tokens, padding, sg, min_count, vector_size, workers, window):
    if padding:
        temp_df = list(all_stemmed_tokens)
        temp_df.append(['pad'])
        word2vec_file = OUTPUT_FOLDER + 'twitter_data' + '_PAD.model'
    w2v_model = Word2Vec(sentences=temp_df, min_count=min_count, vector_size=vector_size, workers=workers, window=window, sg=sg)
    w2v_model.save(word2vec_file)
    return w2v_model, word2vec_file

size = 500 
vector_size = size 
w2vmodel, word2vec_file = make_word2vec_model(all_stemmed_tokens, padding=True, sg=sg, min_count=min_count, vector_size=vector_size, workers=workers, window=window)


# --- Padding ---
max_sen_len = train_df.stemmed_tokens.map(len).max()
padding_idx = w2vmodel.wv.key_to_index.get('pad')
if padding_idx is None:
    padding_idx = 0 # Fallback if 'pad' is not in vocab for some reason

def make_word2vec_vector_cnn(sentence):
    padded_X = [padding_idx for i in range(max_sen_len)]
    i = 0
    for word in sentence:
        if word not in w2vmodel.wv:
            padded_X[i] = 0
        else:
            padded_X[i] = w2vmodel.wv.key_to_index.get(word, 0)
        i += 1
    return torch.tensor(padded_X, dtype=torch.long, device=device).view(1, -1)

# --- CNN Classifier Model Creation (Παρόμοιο με το αρχικό παράδειγμα) ---
EMBEDDING_SIZE = 500
NUM_FILTERS = 10

class CnnTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, window_sizes=(1, 2, 3, 5)):
        super(CnnTextClassifier, self).__init__()
        w2vmodel_loaded = gensim.models.KeyedVectors.load(OUTPUT_FOLDER + 'twitter_data_PAD.model')
        weights = w2vmodel_loaded.wv
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=w2vmodel_loaded.wv.key_to_index.get('pad', 0))
        self.convs = nn.ModuleList([
            nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])
        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.unsqueeze(x, 1)
        xs = []
        for conv in self.convs:
            x2 = torch.tanh(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        return probs

NUM_CLASSES = 2
VOCAB_SIZE = len(w2vmodel.wv)
print(VOCAB_SIZE)
cnn_model = CnnTextClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
cnn_model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001)
num_epochs = 10

# --- Model Train  ---
loss_file_name = OUTPUT_FOLDER + 'twitter_cnn_loss_with_padding.csv'
f = open(loss_file_name, 'w')
f.write('iter, loss\n')
losses = []
cnn_model.train()
for epoch in range(num_epochs):
    print("Epoch" + str(epoch + 1))
    train_loss = 0
    for index, row in X_train.iterrows():
        cnn_model.zero_grad()
        bow_vec = make_word2vec_vector_cnn(row['stemmed_tokens'])
        probs = cnn_model(bow_vec)
        # ΧΡΗΣΙΜΟΠΟΙΗΣΕ Y_train ΓΙΑ ΤΗΝ ΕΤΙΚΕΤΑ
        target = torch.tensor([Y_train['Label'][index]], dtype=torch.long, device=device)
        loss = loss_function(probs, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'train_loss : {train_loss / len(X_train)}')
    print("Epoch ran :" + str(epoch + 1))
    f.write(str((epoch + 1)) + "," + str(train_loss / len(X_train)) + "\n")
    train_loss = 0
torch.save(cnn_model, OUTPUT_FOLDER + 'twitter_cnn_model_with_padding.pth')
f.close()

# --- Evaluation
bow_cnn_predictions = []
original_lables_cnn_bow = []
cnn_model.eval()
with torch.no_grad():
    for index, row in X_test.iterrows():
        bow_vec = make_word2vec_vector_cnn(row['stemmed_tokens'])
        probs = cnn_model(bow_vec)
        _, predicted = torch.max(probs.data, 1)
        bow_cnn_predictions.append(predicted.cpu().numpy()[0])
        # ΧΡΗΣΙΜΟΠΟΙΗΣΕ Y_test ΓΙΑ ΤΙΣ ΠΡΑΓΜΑΤΙΚΕΣ ΕΤΙΚΕΤΕΣ
        original_lables_cnn_bow.append(Y_test['Label'][index])

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
print(confusion_matrix(original_lables_cnn_bow, bow_cnn_predictions))
print(classification_report(original_lables_cnn_bow, bow_cnn_predictions))

# Plot της confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(original_lables_cnn_bow, bow_cnn_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - CNN Model')
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + 'confusion_matrix_heatmap.pdf')
plt.close()


# Υπολογισμός των επιπλέον μετρικών
accuracy = accuracy_score(original_lables_cnn_bow, bow_cnn_predictions)
precision = precision_score(original_lables_cnn_bow, bow_cnn_predictions)
recall = recall_score(original_lables_cnn_bow, bow_cnn_predictions)
f1 = f1_score(original_lables_cnn_bow, bow_cnn_predictions)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

loss_df = pd.read_csv(loss_file_name)
plt_loss = loss_df[' loss'].plot()
fig = plt_loss.get_figure()
fig.savefig(OUTPUT_FOLDER + 'twitter_cnn_loss_plot.pdf')

