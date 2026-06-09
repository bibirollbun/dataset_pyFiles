import numpy as np
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score 


try:
    df = pd.read_csv("/kaggle/input/comments-classification/Dataset/train.csv")
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

    # Calculate the length of each comment (number of words)
    train_df['comment_length'] = train_df['comment_text'].str.lower().str.strip().str.split().str.len()
    
    print("--- Descriptive Statistics for Comment Length ---")

    stats = train_df['comment_length'].describe(percentiles=[.50, .90, .95, .99])
    print(stats)
    print("-" * 45)

    print("\nGenerating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 7))

    # Plot the main histogram, focusing the range for better visibility
    plt.hist(train_df['comment_length'], bins=100, edgecolor='black', alpha=0.7, range=(0, 400))

    plt.title('Distribution of Comment Lengths in Training Data', fontsize=16)
    plt.xlabel('Number of Words (Tokens)', fontsize=12)
    plt.ylabel('Frequency (Number of Comments)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.show()

except FileNotFoundError:
    print("Error: 'train.csv' not found. Please make sure the file is in the correct directory.")


import pandas as pd
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
# Use sklearn for a clean initial split
from sklearn.model_selection import train_test_split

df = pd.read_csv("t/kaggle/input/comments-classification/Dataset/train.csv")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Original df size: {len(df)}")
print(f"Training df size: {len(train_df)}")
print(f"Validation df size: {len(val_df)}")

train_df['comment_text'] = train_df['comment_text'].str.lower().str.strip()
tokenized_comments = [comment.split() for comment in train_df['comment_text']]
all_tokens = [t for comment in tokenized_comments for t in comment]
token_counts = Counter(all_tokens)

TOP_K = 50000
most_common_tokens = [tok for tok, _ in token_counts.most_common(TOP_K)]
vocab = {tok: idx+2 for idx, tok in enumerate(most_common_tokens)}
vocab['[PAD]'] = 0
vocab['[UNK]'] = 1
vocab['[CLS]'] = len(vocab)

print("Reduced vocab size:", len(vocab))


class CommentDataset(Dataset):
    def __init__(self, dataframe, vocab, max_len):
        self.comments = dataframe['comment_text'].tolist()
        self.labels = dataframe['psychotic_depression'].tolist()
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, idx):
        comment = self.comments[idx]
        label = self.labels[idx]

        # Tokenize, numericalize, and pad/truncate the single comment
        tokens = comment.split()
        token_ids = [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]
        cls_token_id = self.vocab['[CLS]']
        token_ids = [cls_token_id] + token_ids
        
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['[PAD]']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        
        return torch.tensor(token_ids), torch.tensor(label, dtype=torch.float32)


d_model = 512
h = 8               # number of attention heads
vocab_size = len(vocab)  # reduced vocab
max_len = 100        # max tokens per comment
dropout = 0.3
batch_size = 16 
lr = 1e-5
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


val_df['comment_text'] = val_df['comment_text'].str.lower().str.strip()
from torch.utils.data import DataLoader, WeightedRandomSampler

# compute class counts
class_counts = train_df["psychotic_depression"].value_counts().to_dict()
neg_count = class_counts[0]
pos_count = class_counts[1]

# compute sample weights 
weights = [1/neg_count if label == 0 else 1/pos_count for label in train_df["psychotic_depression"]]

# create sampler
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
train_dataset = CommentDataset(train_df, vocab, MAX_LEN)
val_dataset = CommentDataset(val_df, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



def positional_encodings(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]  #[[0], [1], [2], ..., [seq_len - 1]]
    dims = np.arange(d_model)[np.newaxis, :] # [[0, 1, 2, ..., d_model - 1]]
    
    angles_input = pos/(np.power(10000, (2*(dims//2))/d_model))
    pos_encodings = np.zeros((seq_len, d_model))
    pos_encodings[:, 0::2] = np.sin(angles_input[:, 0::2])
    pos_encodings[:, 1::2] = np.cos(angles_input[:, 1::2])

    return torch.tensor(pos_encodings, dtype=torch.float32)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h 
        
        #Linear projections for Q, K, V
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        #Final linear projection
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        #Project Q, K, V
        Q = self.Wq(x)  
        K = self.Wk(x)
        V = self.Wv(x)
        
        # Split into h heads
        Q = Q.view(batch_size, seq_len, self.h, self.d_k).transpose(1,2)  
        K = K.view(batch_size, seq_len, self.h, self.d_k).transpose(1,2)
        V = V.view(batch_size, seq_len, self.h, self.d_k).transpose(1,2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)  
        attn = torch.matmul(scores.softmax(dim=-1), V)
        
        # Concatenate heads
        attn = attn.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model) 
        
        # Final projection
        out = self.W_o(attn)  # (batch, seq_len, d_model)
        return out



class FeedFwd(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model*2)
        self.linear2 = nn.Linear(2*d_model, d_model)
    
    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffd = FeedFwd(d_model)       
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.mha(x))  
        x = self.norm2(x + self.ffd(x))  

        return x


class Model(nn.Module):
    def __init__(self, vocab_size, d_model, h, max_len=512, dropout=0.2):
        super().__init__()
        self.d_model = d_model

        self.embeddings = nn.Embedding(vocab_size, d_model)

        self.block1 = EncoderBlock(d_model, h)
        self.block2 = EncoderBlock(d_model, h)
        self.block3 = EncoderBlock(d_model, h)

        self.ffd_block = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(2*d_model, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(d_model, 1)   
        )

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embeddings(x) * (self.d_model ** 0.5)
        pos_enc = positional_encodings(x.size(1), self.d_model).to(x.device)  # (seq_len, d_model)
        x = x + pos_enc.unsqueeze(0)  # broadcast over batch


        # pass through encoder blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x[:,0,:]   

        x = self.ffd_block(x)
        return x



model = Model(vocab_size, d_model, h, max_len, dropout).to(device)
neg_samples = df['psychotic_depression'].value_counts()[0]
pos_samples = df['psychotic_depression'].value_counts()[1]
weight = torch.tensor([neg_samples / pos_samples], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)



optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)


for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

    for batch_tokens, batch_labels in train_progress_bar:
        batch_tokens = batch_tokens.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_tokens).squeeze(1)
        loss = criterion(outputs, batch_labels)
        
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * batch_tokens.size(0)
        train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_dataset)
    
    model.eval()
    total_val_loss = 0
    all_labels = []
    all_preds = []
    
    val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

    with torch.no_grad():
        for batch_tokens, batch_labels in val_progress_bar:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_tokens).squeeze(1)
            loss = criterion(outputs, batch_labels)
            
            total_val_loss += loss.item() * batch_tokens.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.4).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            val_progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
    avg_val_loss = total_val_loss / len(val_dataset)
    
    # Calculate F1 score from all collected predictions and labels
    val_f1 = f1_score(all_labels, all_preds)
    
    # Print summary for the epoch
    print(f"Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")


torch.save({
            'model_state_dict': model.state_dict(),
            }, 'model_params.pth')
    


model = Model(vocab_size, d_model, h, max_len, dropout).to(device)
checkpoint = torch.load('model_params.pth')
model.load_state_dict(checkpoint['model_state_dict'])


test_df = pd.read_csv("/kaggle/input/comments-classification/Dataset/test.csv")
test_df['comment_text'] = test_df['comment_text'].str.lower().str.strip()
test_df['psychotic_depression'] = 0 
test_dataset = CommentDataset(test_df, vocab, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval() # IMPORTANT: Set the model to evaluation mode

# 2. Run the inference loop
all_predictions = []
with torch.no_grad(): # Disable gradient calculations
    for batch_tokens, _ in tqdm(test_loader, desc="Generating Predictions"):
        batch_tokens = batch_tokens.to(device)
        
        # Get model outputs (logits)
        outputs = model(batch_tokens).squeeze(1)
        
        # Convert logits to probabilities and then to binary predictions
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.4).int() # Using a 0.5 threshold
        
        # Store the predictions
        all_predictions.extend(preds.cpu().numpy())

# Add the predictions as a new column
test_df['psychotic_depression'] = all_predictions

# Save the final results to a CSV file
# You might only need the ID and the prediction columns for submission
submission_df = test_df[['id', 'psychotic_depression']] # Adjust columns as needed
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(submission_df.head())




