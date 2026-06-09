import re
import string
import zipfile

import emoji
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm


training_path = r"/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
with zipfile.ZipFile(training_path) as train_zip:
    with train_zip.open("train.csv") as csv:
        training_data = pd.read_csv(csv)

training_data.head()



sample_path = r"/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"
with zipfile.ZipFile(sample_path) as z:
    with z.open("sample_submission.csv") as csv:
        sample_data = pd.read_csv(csv)

sample_data.head()


test_path = r"/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
with zipfile.ZipFile(test_path) as z:
    with z.open("test.csv") as csv:
        test_data = pd.read_csv(csv)

test_data.head()


test_labels_path = r"/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip"
with zipfile.ZipFile(test_labels_path) as z:
    with z.open("test_labels.csv") as csv:
        test_labels_data = pd.read_csv(csv)

test_labels_data.head()


punc = string.punctuation
punc.replace('#', '')
punc.replace('!', '')
punc.replace('?', '')
punc = punc + "∞θ÷α•à−β∅³π‘₹´°£€\×™√²—"

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}


stpwds = stopwords.words('english')

nlp = spacy.load("en_core_web_sm")

time_zone_abbreviations = [
        "UTC", "GMT", "EST", "CST", "PST", "MST",
        "EDT", "CDT", "PDT", "MDT", "CET", "EET",
        "WET", "AEST", "ACST", "AWST", "HST",
        "AKST", "IST", "JST", "KST", "NZST"
    ]

patterns = [
    r'\\[nrtbfv\\]',         # \n, \t ..etc
    '<.*?>',                 # Html tags
    r'https?://\S+|www\.\S+',# Links
    r'\ufeff',               # BOM characters
    r'^[^a-zA-Z0-9]+$',      # Non-alphanumeric tokens
    r'ｗｗｗ．\S+',            # Full-width URLs
    r'[\uf700-\uf7ff]',      # Unicode private-use chars
    r'^[－—…]+$',            # Special punctuation-only tokens
    r'[︵︶]'                # CJK parentheses
]

def preprocess(text):
    for regex in patterns:
        text = re.sub(regex, '', text)
    text = text.translate(str.maketrans(punc, ' ' * len(punc)))
    text = ' '.join(word for word in text.split() if word not in time_zone_abbreviations)
    text = ' '.join(word for word in text.split() if word not in stpwds)
    text = ' '.join(chat_words.get(word.lower(), word) for word in text.split())
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


comments = list(training_data["comment_text"])
train_iter = iter(comments)

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        cleaned_text = preprocess(text)
        tokens = [
            token for token in tokenizer(cleaned_text)
            if 1 < len(token) < 25
        ]
        yield tokens

# Build vocabulary with size limit
vocab = build_vocab_from_iterator(
    yield_tokens(train_iter),
    specials=["<pad>", "<unk>"],
    max_tokens=30002  # 30K + 2 special tokens for unkown tokens and padding
)
vocab.set_default_index(vocab["<unk>"])
PAD_IDX = vocab['<pad>']

print(f"Final vocabulary size: {len(vocab)}")
print("Sample valid tokens:", [t for t in list(vocab.get_itos())[2:12]])


torch.save(vocab, "vocab.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def text_pipeline(text):
    return [
        vocab[token] if token in vocab else vocab['<unk>'] 
        for token in tokenizer(text)
    ]

def label_pipeline(labels):
    return torch.FloatTensor(labels)




from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

class PaddedDataset(Dataset):
    def __init__(self, df, vocab, max_length=None):
        self.df = df
        self.vocab = vocab
        self.max_length = max_length
        self.label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['comment_text']
        labels = self.df.iloc[idx][self.label_cols].values.astype(float)
        
        # Tokenize and numericalize
        tokens = tokenizer(preprocess(text))
        if self.max_length:
            tokens = tokens[:self.max_length]
        numericalized = [self.vocab[token] for token in tokens]
        
        return torch.tensor(numericalized, dtype=torch.long), torch.tensor(labels, dtype=torch.float)


def collate_batch(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
     # Filter invalid sequences (length <=0)
    valid_mask = lengths > 0
    if not valid_mask.all():
        texts = [t for t, valid in zip(texts, valid_mask) if valid]
        labels = [l for l, valid in zip(labels, valid_mask) if valid]
        lengths = lengths[valid_mask]
    
    # Add fallback for empty batch
    if len(texts) == 0:
        return torch.zeros((1,1), dtype=torch.long), torch.zeros((1,6)), torch.tensor([1])
    # Pad sequences to match longest in batch
    padded_texts = torch.nn.utils.rnn.pad_sequence(
        texts, 
        batch_first=True, 
        padding_value=PAD_IDX
    )
    
    return padded_texts, torch.stack(labels), lengths


BATCH_SIZE = 64
MAX_SEQ_LEN = 256

dataset = PaddedDataset(training_data, vocab, max_length=MAX_SEQ_LEN)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_batch,
    pin_memory=True,  # Faster data transfer to GPU
    num_workers=2     # Parallel data loading
)


class lstm(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted = False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        out = self.fc1(hidden[-1])
        return torch.sigmoid(out)


class BIDirectional_lstm(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout =  nn.Dropout(p = 0.3)
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted = False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc1(hidden_output)
        out = self.dropout(out)
        return torch.sigmoid(out)


def train_model(model, train_loader, val_loader, epochs, learning_rate, filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for texts, labels, lengths in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            # Move data to device
            texts, labels = texts.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).all(dim=1).sum().item()
            total += labels.size(0)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for texts, labels, lengths in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts, lengths)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).all(dim=1).sum().item()
                val_total += labels.size(0)
        
        # Epoch Statistics
        train_loss /= len(train_loader)
        train_acc = correct / total
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename + '.pth')
    
    print("Training complete!")



train_df, val_df = train_test_split(training_data.iloc[:, 1:], test_size=0.2)

train_dataset = PaddedDataset(train_df, vocab, max_length=256)
val_dataset = PaddedDataset(val_df, vocab, max_length=256)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_batch,
    pin_memory=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    collate_fn=collate_batch,
    pin_memory=True,
    num_workers=4
)


    


lstm_model = lstm(
    vocab_size=len(vocab),
    embed_dim=50,
    hidden_dim=256,
    output_dim=6,
    pad_idx=PAD_IDX
)

train_model(lstm_model, train_loader, val_loader, 5, 0.001, "lstm")


BIDirectional_model = BIDirectional_lstm(
    vocab_size=len(vocab),
    embed_dim=50,
    hidden_dim=256,
    output_dim=6,
    pad_idx=PAD_IDX
)

train_model(BIDirectional_model, train_loader, val_loader, 7, 0.0005, "bidirctional_lstm")


def load_glove_from_file(glove_file):
    word_to_vec = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.split()
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]], dtype=np.float32)
            word_to_vec[word] = vector
    return word_to_vec

# 1. Load your GloVe file
glove_path = r"/kaggle/input/glove-embeddings/glove.6B.100d.txt"  # Update with your path
glove_vectors = load_glove_from_file(glove_path)

# 2. Create embedding matrix aligned with your vocabulary
def create_embedding_matrix(vocab, embedding_dim=100):
    vocab_size = len(vocab)
    weights = torch.zeros(vocab_size, embedding_dim)
    
    for word, idx in vocab.get_stoi().items():
        if word in glove_vectors:
            weights[idx] = torch.tensor(glove_vectors[word])
        elif word == "<pad>":
            weights[idx] = torch.zeros(embedding_dim)  # Pad token
        else:
            # Initialize unknown words randomly
            weights[idx] = torch.randn(embedding_dim) * 0.25
            
    return weights



class BI_lstm_GloVe_model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, output_dim):
        super().__init__()
        # Initialize with GloVe weights
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted = False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(hidden_output)
        return torch.sigmoid(out)


embedding_dim = 100  # matches GloVe dimension
vocab_size = len(vocab)
bi_lstm_glove_model = BI_lstm_GloVe_model(
    vocab_size,
    embedding_dim, 
    hidden_dim = 256,
    pad_idx = PAD_IDX,
    output_dim = 6
)

# Create embedding matrix
weights = create_embedding_matrix(vocab, 100)
# Assign to model
bi_lstm_glove_model.embedding = nn.Embedding.from_pretrained(weights, freeze=False)


train_model(bi_lstm_glove_model, train_loader, val_loader, 10, 0.0005, "bi_lstm_glove")


class Improved_BI_LSTM_GloVe(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # Enhanced Architecture
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                           num_layers=2,              # Stacked LSTMs
                           bidirectional=True, 
                           batch_first=True,
                           dropout=0.3)               # Inter-layer dropout
        
        self.attention = nn.Linear(hidden_dim * 2, 1) # Simple attention mechanism
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)     # Batch normalization
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),                          # Increased dropout
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize with kaiming normal for better convergence
        for layer in [self.attention, *self.fc]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, text, lengths):
        # Embedding with dropout
        embedded = F.dropout(self.embedding(text), p=0.2, training=self.training)
        
        # Packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM with 2 layers
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(output), dim=1)
        context_vector = torch.sum(attention_weights * output, dim=1)
        
        # Batch norm + FC
        context_vector = self.bn1(context_vector)
        return self.fc(context_vector)


def train_model(model, train_loader, val_loader, epochs, learning_rate, filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for texts, labels, lengths in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            # Move data to device
            texts, labels = texts.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).all(dim=1).sum().item()
            total += labels.size(0)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for texts, labels, lengths in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts, lengths)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).all(dim=1).sum().item()
                val_total += labels.size(0)
        
        # Epoch Statistics
        train_loss /= len(train_loader)
        train_acc = correct / total
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename + '.pth')
    
    print("Training complete!")


final_model = Improved_BI_LSTM_GloVe(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=256,
    pad_idx=PAD_IDX,
    output_dim=6
)
final_model.embedding.weight.data.copy_(weights)
final_model.embedding.weight.requires_grad = True
train_model(final_model, train_loader, val_loader, 10, 0.0001, "final")


ev_data = pd.concat([test_data,test_labels_data.iloc[:,1:]], axis=1)
# dropping -1 rows, these rows weren't used for evaluation models in the competetion and marked with -1 
ev_data = ev_data[ev_data['toxic']!= -1]
ev_data


eval_dataset = PaddedDataset(ev_data, vocab, max_length=256)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=512,
    collate_fn=collate_batch,
    pin_memory=True,
    num_workers=4
)


def calc_roc(model):
    model.eval()
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for texts, labels, lengths in eval_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts, lengths)
            
            # Store batch results
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    # Concatenate all batches
    all_labels = np.concatenate(all_labels, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    
    # Calculate ROC-AUC for each class
    roc_scores = []
    for col in range(6):  # the original evaluation method is to take ROC-AUC scores average for the 6 classed
        if np.sum(all_labels[:, col]) > 0:
            roc = roc_auc_score(all_labels[:, col], all_outputs[:, col])
            roc_scores.append(roc)
    
    # Return average
    return np.mean(roc_scores)


lstm_model = lstm_model = lstm(
    vocab_size=len(vocab),
    embed_dim=50,
    hidden_dim=256,
    output_dim=6,
    pad_idx=PAD_IDX
).to(device)

bi_lstm_model = BIDirectional_model = BIDirectional_lstm(
    vocab_size=len(vocab),
    embed_dim=50,
    hidden_dim=256,
    output_dim=6,
    pad_idx=PAD_IDX
).to(device)

vocab_size = len(vocab)
bi_lstm_glove_model = BI_lstm_GloVe_model(
    vocab_size,
    100, 
    hidden_dim = 256,
    pad_idx = PAD_IDX,
    output_dim = 6
).to(device)

final_model = Improved_BI_LSTM_GloVe(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=256,
    pad_idx=PAD_IDX,
    output_dim=6
).to(device)

lstm_model.load_state_dict(torch.load("lstm.pth"))
bi_lstm_model.load_state_dict(torch.load("bidirctional_lstm.pth"))
bi_lstm_glove_model.load_state_dict(torch.load("bi_lstm_glove.pth"))
final_model.load_state_dict(torch.load("final.pth"))

print("lstm model roc-auc ", calc_roc(lstm_model))
print("BiDirectional lstm model roc-auc ", calc_roc(bi_lstm_model))
print("BiDirectional lstm with pretrained embedding model roc-auc ", calc_roc(bi_lstm_glove_model))
print("stacked Bidirectional lstm model with pretrained embedding roc-auc ", calc_roc(final_model))




