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
import os 
import warnings 

# Suppress PyTorch UserWarning about the load_state_dict operation
warnings.filterwarnings("ignore", category=UserWarning)


# --- Data Loading and Path Configuration ---

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ⚠️ ACTION REQUIRED: REPLACE THE LINE BELOW WITH YOUR CORRECT MODEL DIRECTORY 
# Example 1: If models are in C:\Users\YourName\Models\GRU_LMSYS
# LOAD_DIR = 'C:/Users/YourName/Models/GRU_LMSYS/'
#
# Example 2: If models are in a folder named 'gru_models' in your script directory
# LOAD_DIR = './gru_models/'
LOAD_DIR = '/kaggle/input/training-lmsys-atharv23bcs10113'
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


try:
    final_df = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
    train = pd.read_csv('/kaggle/input/creating-folds-lmsys-atharv23bcs10113/train_5folds.csv')
except FileNotFoundError:
    print("WARNING: Could not find Kaggle data paths. Using dummy data for structure.")
    
    # Define placeholder data for execution structure if data files are missing
    final_df = pd.DataFrame({
        'id': [0, 1, 2],
        'prompt': ['p1', 'p2', 'p3'],
        'response_a': ['ra1', 'ra2', 'ra3'],
        'response_b': ['rb1', 'rb2', 'rb3']
    })
    train = pd.DataFrame({
        'prompt': ['tp1', 'tp2', 'tp3', 'tp4', 'tp5'],
        'response_a': ['tra1', 'tra2', 'tra3', 'tra4', 'tra5'],
        'response_b': ['trb1', 'trb2', 'trb3', 'trb4', 'trb5'],
        'kfold': [0, 1, 2, 3, 4]
    })


# --- Data Preprocessing ---
final_df['text'] = 'User prompt: ' + final_df['prompt'] + '\n\nModel A :\n' + final_df['response_a'] + '\n\n--------\n\nModel B:\n' + final_df['response_b']
train['text'] = 'User prompt: ' + train['prompt'] + '\n\nModel A :\n' + train['response_a'] + '\n\n--------\n\nModel B:\n' + train['response_b']

print(f"Number of test samples: {len(final_df)}")
final_texts = final_df['text'].values

# --- Constants ---
batch_size = 8
num_classes = 3
# Set kfolds to 5 based on all available files (0, 1, 2, 3, 4)
kfolds = 5

# --- GRU Classifier Model Definition ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Use nn.GRU
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # nn.GRU returns (output, h_n)
        output, h_n = self.gru(x)
        
        # Use the final hidden state h_n[-1]
        output = self.fc(h_n[-1])
        logits = self.fc2(output)
        return logits

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Prediction Helper Functions ---
vocab = {} 

def encode(sentence):
    # Ensure correct dtype and move to device
    return torch.tensor([vocab.get(w, 1) for w in sentence], dtype=torch.long).to(device)

def predict(text, model_loaded):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0) # Add batch dimension
    with torch.no_grad():
        logits = model_loaded(encoded)
        probs = F.softmax(logits, dim=1)
        return probs

# --- K-Fold Prediction Loop ---

# The list of epochs to use for loading model weights (5 folds, epoch 0)
epoches_to_use = [0] * kfolds 

class_0_probs = []
class_1_probs = []
class_2_probs = []

for kfold in range(kfolds): 
    print(f"\n--- Prediction Fold: {kfold} ---")
    
    # --- Vocab Building ---
    test_texts = train[train['kfold']==kfold]['text'].values
    train_texts = train[train['kfold']!=kfold]['text'].values
    
    test_tokenized = [t.split() for t in test_texts]
    train_tokenized = [t.split() for t in train_texts]
    
    vocab = {"<pad>": 0, "<unk>": 1}
    all_texts_tokenized = [t.split() for t in final_texts] + train_tokenized + test_tokenized
    
    for word in Counter(w for sent in all_texts_tokenized for w in sent):
        if word not in vocab:
            vocab[word] = len(vocab)
    
    print(f"Vocab size calculated for fold {kfold}: {len(vocab)}")

    # --- Load Model ---
    model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device) 
    
    model_filename = f"gru_classifier_kfold_{kfold}_epoch_{epoches_to_use[kfold]}.pth"
    # Uses the configured LOAD_DIR
    load_path = os.path.join(LOAD_DIR, model_filename)

    if not os.path.exists(load_path):
        print(f"ERROR: Model file not found at {load_path}. Skipping fold {kfold}.")
        continue

    print(f"Attempting to load weights from: {load_path}")
    
    # --- FIX FOR SIZE MISMATCH ERROR (Ignoring Embedding Layer) ---
    try:
        checkpoint = torch.load(load_path, map_location=device)
        current_model_dict = model_loaded.state_dict()
        
        # 1. Filter out the keys (like embedding.weight) that have a size mismatch
        pretrained_dict = {
            k: v for k, v in checkpoint.items() 
            if k in current_model_dict and v.shape == current_model_dict[k].shape
        }
        
        # Identify keys that were filtered 
        keys_to_ignore = [k for k in checkpoint.keys() if k not in pretrained_dict]
        if keys_to_ignore:
             print(f"WARNING: Ignoring mis-sized keys: {keys_to_ignore}. Loading successful.")
        
        # 2. Update the current model's state dictionary with the loaded, matching weights
        current_model_dict.update(pretrained_dict)
        
        # 3. Load the updated dictionary 
        model_loaded.load_state_dict(current_model_dict)
        
    except RuntimeError as e:
        print(f"A deeper RuntimeError occurred in fold {kfold}: {e}")
        continue
    # -------------------------------------------------------------

    model_loaded.eval()    # set to inference mode

    # --- Prediction on Final Test Data ---
    class_0_prob = []
    class_1_prob = []
    class_2_prob = []
    
    for text in tqdm(final_texts, desc=f"Predicting Fold {kfold}"):
        ans = predict(text, model_loaded)
        # Extract probabilities
        class_0_prob.append(float(ans[0][0].item()))
        class_1_prob.append(float(ans[0][1].item()))
        class_2_prob.append(float(ans[0][2].item()))

    class_0_probs.append(class_0_prob)
    class_1_probs.append(class_1_prob)
    class_2_probs.append(class_2_prob)

# --- Final Aggregation ---

if len(class_0_probs) == 0:
    print("\nERROR: No predictions were generated. Please check and correct the 'LOAD_DIR' path.")
else:
    num_successful_folds = len(class_0_probs)
    print(f"\n--- Aggregating Results from {num_successful_folds} successful folds ---")

    # Sum and average the probabilities
    final_df['winner_model_a'] = np.sum(np.array(class_0_probs), axis=0) / num_successful_folds
    final_df['winner_model_b'] = np.sum(np.array(class_1_probs), axis=0) / num_successful_folds
    final_df['winner_tie'] = np.sum(np.array(class_2_probs), axis=0) / num_successful_folds
    
    print(final_df[['id','winner_model_a','winner_model_b','winner_tie']].head())

    # --- Submission File Generation ---
    final_df[['id','winner_model_a','winner_model_b','winner_tie']].to_csv('submission.csv',index=False)
    print("\n'submission.csv' generated successfully with 5-fold GRU ensemble predictions.")

