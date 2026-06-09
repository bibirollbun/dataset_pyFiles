# =========================================================================================
# AIRR-ML-25: Adaptive Immune Profiling Challenge - Professional Winning Solution
# Model: Gated Attention-based Multiple Instance Learning (MIL) with Deep Sequence Encoding
# =========================================================================================

"""
SOLUTION OVERVIEW:
This solution utilizes a Deep Learning approach tailored for Multiple Instance Learning (MIL).
In this context, a patient's Repertoire is a "Bag", and the immune sequences are "Instances".

1.  **Sequence Encoder:** Each amino acid sequence (junction_aa) is embedded and processed 
    via a Bi-directional GRU to capture structural/functional context.
2.  **Gated Attention Mechanism:** (Ilse et al., 2018) The network learns an 'attention weight' 
    for every sequence in a repertoire. High weights indicate sequences highly correlated 
    with the label (Disease).
3.  **Aggregation:** Sequence representations are aggregated into a single 'Repertoire Vector'
    using the learned attention weights.
4.  **Classification:** A final classifier predicts the immune state (Healthy/Disease) based 
    on the Repertoire Vector.

Strengths:
-   **Task 1 (Prediction):** High accuracy by leveraging deep sequence features.
-   **Task 2 (Interpretation):** The attention weights directly provide the "importance score" 
    required to rank the contributing sequences.
"""
# =========================================================================================
# AIRR-ML-25: Professional Solution - FINAL VERSION (Auto-Path & Bug Fix)
# =========================================================================================

import os
import sys
import numpy as np
import pandas as pd
import glob
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import random
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

# --- Reproducibility Setup ---
SEED = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"âœ… Using device: {DEVICE}")


# =========================================================================================
# 1. ROBUST PATH CONFIGURATION
# =========================================================================================

BASE_DIR = "/kaggle/input/adaptive-immune-profiling-challenge-2025"

# 1. Detect Train/Test Directories (Handles 'train' vs 'train_datasets' naming convention)
if os.path.exists(os.path.join(BASE_DIR, "train_datasets")):
    TRAIN_DIR = os.path.join(BASE_DIR, "train_datasets")
    TEST_DIR = os.path.join(BASE_DIR, "test_datasets")
else:
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    TEST_DIR = os.path.join(BASE_DIR, "test")

print(f"ğŸ“‚ Detected Repertoires Directory: {TRAIN_DIR}")

# 2. Detect Metadata File (Crucial Fix for FileNotFoundError)
METADATA_PATH = None
# Check all common Kaggle path structures
potential_paths = [
    os.path.join(BASE_DIR, "metadata.csv"),
    os.path.join(TRAIN_DIR, "metadata.csv"),
]
for p in potential_paths:
    if os.path.exists(p):
        METADATA_PATH = p
        break

if METADATA_PATH is None:
    # Fallback search if standard paths fail
    found_metas = glob.glob(os.path.join(BASE_DIR, "**", "metadata.csv"), recursive=True)
    if found_metas:
        METADATA_PATH = found_metas[0]

if METADATA_PATH is None:
    # Raise error if still not found, but we catch it later
    print(f"â�Œ CRITICAL ERROR: metadata.csv NOT FOUND in {BASE_DIR} structure.")
else:
    print(f"âœ… Found Metadata Path: {METADATA_PATH}")


# =========================================================================================
# 2. Data Processing & Helper Functions
# =========================================================================================

# Amino Acid Vocabulary for encoding
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)} # 0 reserved for padding
VOCAB_SIZE = len(AA_VOCAB) + 1
MAX_SEQ_LEN = 30 

def encode_sequence(seq, max_len=MAX_SEQ_LEN):
    """Encodes amino acid string to integer list."""
    if pd.isna(seq): return [0] * max_len
    seq = seq[:max_len]
    encoded = [AA_TO_INT.get(aa, 0) for aa in seq]
    padding = [0] * (max_len - len(encoded))
    return encoded + padding

class AIRRDataset(Dataset):
    """Dataset for MIL, handling variable-sized bags (Repertoires)."""
    def __init__(self, repertoires_data, labels_map=None, is_train=True, max_seqs_per_bag=10000):
        self.repertoires_data = repertoires_data
        self.rep_ids = list(repertoires_data.keys())
        self.labels_map = labels_map
        self.is_train = is_train
        self.max_seqs_per_bag = max_seqs_per_bag

    def __len__(self):
        return len(self.rep_ids)

    def __getitem__(self, idx):
        rep_id = self.rep_ids[idx]
        df = self.repertoires_data[rep_id]
        
        # Subsampling for memory efficiency and regularization
        if self.is_train and len(df) > self.max_seqs_per_bag:
            df = df.sample(n=self.max_seqs_per_bag, random_state=SEED)
            
        sequences = [encode_sequence(seq) for seq in df['junction_aa'].values]
        seq_tensor = torch.tensor(sequences, dtype=torch.long)
        
        # Get Label
        label = torch.tensor(0.0, dtype=torch.float)
        if self.is_train and self.labels_map:
            val = self.labels_map.get(str(rep_id))
            if val is not None:
                label = torch.tensor(val, dtype=torch.float)
                
        # Return seqs, label, ID (as string), and raw sequences/V/J calls
        return seq_tensor, label, str(rep_id), df[['junction_aa', 'v_call', 'j_call']].reset_index(drop=True)

def collate_bags(batch):
    """Custom collate function for DataLoader."""
    seqs, labels, rep_ids, raw_dfs = zip(*batch)
    labels = torch.stack(labels)
    return seqs, labels, rep_ids, raw_dfs


# =========================================================================================
# 3. Model Architecture: Gated Attention MIL
# =========================================================================================

class AttentionMILModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, mlp_dim=128):
        super().__init__()
        
        # 1. Sequence Encoder (Instance Encoder)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.seq_encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        seq_out_dim = hidden_dim * 2

        # 2. Gated Attention Mechanism
        self.attention_V = nn.Sequential(nn.Linear(seq_out_dim, mlp_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(seq_out_dim, mlp_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(mlp_dim, 1)

        # 3. Bag Classifier
        self.classifier = nn.Sequential(
            nn.Linear(seq_out_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(mlp_dim, 1)
        )

    def forward(self, bag_seqs):
        # Feature Extraction
        embedded = self.embedding(bag_seqs) 
        _, hidden = self.seq_encoder(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1) # (N, Dim)

        # Attention Scores
        A = self.attention_weights(self.attention_V(hidden) * self.attention_U(hidden)) # (N, 1)
        A = torch.softmax(torch.transpose(A, 1, 0), dim=1) # (1, N)

        # Aggregation
        bag_rep = torch.mm(A, hidden) # (1, Dim)

        # Classification
        return self.classifier(bag_rep).squeeze(1), A.squeeze(0)


# =========================================================================================
# 4. Predictor Engine (Handles Data Loading and Training)
# =========================================================================================

class ImmuneStatePredictor:
    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.train_metadata = {}

    def _load_files(self, directory):
        """Loads all parquet files in a directory recursively."""
        reps = {}
        meta_list = []
        # Robust recursive search for parquets
        files = glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True)
        
        for f in tqdm(files, desc="Loading Parquets"):
            try:
                rep_id = os.path.basename(f).replace('.parquet', '')
                dataset_id = os.path.basename(os.path.dirname(f))
                
                df = pd.read_parquet(f)
                # Keep only essential columns to save memory
                if 'junction_aa' in df.columns:
                    reps[rep_id] = df[['junction_aa', 'v_call', 'j_call']]
                    meta_list.append({'repertoire_id': rep_id, 'dataset_id': dataset_id})
            except: pass
        return reps, pd.DataFrame(meta_list)

    def fit(self, train_dir, meta_path):
        """Loads data, initializes model, and starts training."""
        if not meta_path or not os.path.exists(meta_path):
            raise FileNotFoundError("Metadata file missing. Cannot train.")
            
        print("\n--- Starting Training Process ---")
        train_reps, train_meta = self._load_files(train_dir)
        if not train_reps: raise ValueError("No training repertoires found! Check TRAIN_DIR path.")
        
        # Load Labels
        labels_df = pd.read_csv(meta_path)
        labels_df['repertoire_id'] = labels_df['repertoire_id'].astype(str)
        labels_map = dict(zip(labels_df['repertoire_id'], labels_df['label']))
        
        self.train_metadata = {'repertoires': train_reps, 'meta': train_meta}
        
        ds = AIRRDataset(train_reps, labels_map, is_train=True)
        # Use batch_size=1 for standard MIL training
        loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_bags, num_workers=2)
        
        # Model and Optimizer setup
        self.model = AttentionMILModel(VOCAB_SIZE).to(self.device)
        opt = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        
        self.model.train()
        EPOCHS = 8 
        for epoch in range(EPOCHS):
            total_loss = 0
            for seqs, labels, _, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                # Since batch_size=1, we take the first element from the list
                seqs, labels = seqs[0].to(self.device), labels[0].to(self.device)
                
                opt.zero_grad()
                logits, _ = self.model(seqs)
                loss = crit(logits, labels)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/len(loader):.4f}")
        print("--- Training Completed ---")

    def predict(self, test_dir):
        """Performs inference on test data (Task 1)."""
        print("\n--- Phase 2: Predicting Test Set ---")
        test_reps, _ = self._load_files(test_dir)
        loader = DataLoader(AIRRDataset(test_reps, is_train=False), batch_size=1, collate_fn=collate_bags, num_workers=2)
        
        preds = {}
        self.model.eval()
        with torch.no_grad():
            for seqs, _, rep_ids, _ in tqdm(loader, desc="Inference"):
                logits, _ = self.model(seqs[0].to(self.device))
                preds[rep_ids[0]] = torch.sigmoid(logits).item()
        return preds

    def interpret(self):
        """Extracts attention scores for sequence ranking (Task 2)."""
        print("\n--- Phase 3: Interpreting Sequences (Attention Scores) ---")
        loader = DataLoader(AIRRDataset(self.train_metadata['repertoires'], is_train=False), 
                            batch_size=1, collate_fn=collate_bags, num_workers=2)
        scores = {} # {dataset_id: { (junc, v, j): max_score }}

        self.model.eval()
        with torch.no_grad():
            for seqs, _, rep_ids, dfs in tqdm(loader, desc="Scanning Attention"):
                
                # Get dataset_id for the current repertoire
                ds_row = self.train_metadata['meta'][self.train_metadata['meta']['repertoire_id'] == rep_ids[0]]
                if ds_row.empty: continue
                ds_id = ds_row['dataset_id'].values[0]
                if ds_id not in scores: scores[ds_id] = {}
                
                _, attn = self.model(seqs[0].to(self.device))
                attn = attn.cpu().numpy()
                df = dfs[0] # Raw dataframe for the bag
                
                # Store the max attention score for each unique sequence
                for i, r in df.iterrows():
                    key = (r['junction_aa'], r['v_call'], r['j_call'])
                    if key not in scores[ds_id] or attn[i] > scores[ds_id][key]:
                        scores[ds_id][key] = attn[i]
        
        # Rank top 50k per dataset
        rows = []
        print("Sorting and ranking top 50,000 sequences...")
        for ds, data in scores.items():
            sorted_seqs = sorted(data.items(), key=lambda x: x[1], reverse=True)[:50000]
            for rank, (k, s) in enumerate(sorted_seqs, 1):
                rows.append({'dataset_id': ds, 'junction_aa': k[0], 'v_call': k[1], 'j_call': k[2], 'rank': rank})
        return pd.DataFrame(rows)


# =========================================================================================
# 5. Execution Pipeline (FINAL)
# =========================================================================================

# Initialize
predictor = ImmuneStatePredictor()

try:
    # A. Train
    if METADATA_PATH:
        predictor.fit(TRAIN_DIR, METADATA_PATH)
        
        # B. Task 1 Predictions
        preds = predictor.predict(TEST_DIR)
        df1 = pd.DataFrame(list(preds.items()), columns=['repertoire_id', 'probability'])
        # Fill required columns with placeholders
        for c in ['dataset_id', 'junction_aa', 'v_call', 'j_call', 'rank']: df1[c] = -999.0
        
        # C. Task 2 Interpretation
        df2 = predictor.interpret()
        df2['repertoire_id'] = "dummy_id" # Required placeholder
        df2['probability'] = -999.0
        df2 = df2[df1.columns] # Ensure column order matches df1
        
        # D. Submission
        final = pd.concat([df1, df2], ignore_index=True)
        final = final.astype({'repertoire_id': str, 'dataset_id': str, 'junction_aa': str, 
                              'v_call': str, 'j_call': str, 'probability': float, 'rank': float})
        final.to_csv("submission.csv", index=False)
        
        print("\n--- Final Submission Summary ---")
        print(f"âœ… Success! Submission saved to submission.csv")
        print(f"Final shape: {final.shape}")
        print("Head (Predictions):")
        print(final[final['repertoire_id'] != 'dummy_id'].head())
        print("Head (Ranked Sequences):")
        print(final[final['repertoire_id'] == 'dummy_id'].head())
        
    else:
        # Fallback if metadata was not found (should be caught by the check above)
        print("â�Œ ERROR: Metadata file not found. Submission cannot be generated.")

except Exception as e:
    # Catch any runtime errors gracefully without crashing the kernel
    print(f"\nâ�Œ CRITICAL EXECUTION FAILURE: {e}")

