# =========================================================================================
# AIRR-ML-25: Professional Solution - Production V.5 (Fixing Pickling & Final Structure)
# =========================================================================================

import os
import numpy as np
import pandas as pd
import glob
from tqdm.auto import tqdm 
from tqdm.contrib.concurrent import process_map # Using process_map for I/O bound speed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import random
import warnings

# --- Reproducibility & Environment Setup ---
SEED = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(SEED)
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"âœ… Using device: {DEVICE}")


# =========================================================================================
# 1. PATHS, CONSTANTS & DYNAMIC ENCODING MAPS
# =========================================================================================

BASE_DIR = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
TRAIN_DIR = os.path.join(BASE_DIR, "train_datasets")
TEST_DIR = os.path.join(BASE_DIR, "test_datasets")

# Sequence/Amino Acid Constants
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)} 
VOCAB_SIZE = len(AA_VOCAB) + 1
MAX_SEQ_LEN = 30 
MAX_SEQS_PER_BAG = 10000 

# Dynamic Gene Call Encoding Maps (Global access for initialization)
V_CALLS_MAP = {}
J_CALLS_MAP = {}
V_VOCAB_SIZE = 1 
J_VOCAB_SIZE = 1 

# Metadata Detection
METADATA_PATH = None
found_metas = glob.glob(os.path.join(BASE_DIR, "**", "metadata.csv"), recursive=True)
if found_metas:
    METADATA_PATH = found_metas[0] 
print(f"âœ… Found Metadata Path: {METADATA_PATH if METADATA_PATH else 'â�Œ NOT FOUND'}")


# =========================================================================================
# 2. DATA PROCESSING AND DATASET
# =========================================================================================

def get_gene_id(gene_call, gene_map, is_v_call):
    """Maps gene call strings to integer IDs dynamically (Used for sequential TRAIN encoding)."""
    global V_VOCAB_SIZE, J_VOCAB_SIZE
    if pd.isna(gene_call) or gene_call == "":
        return 0 
    
    gene = gene_call.split('*')[0].split(',')[0].strip() 
    
    if gene not in gene_map:
        if is_v_call:
            gene_map[gene] = V_VOCAB_SIZE
            V_VOCAB_SIZE += 1
            return V_VOCAB_SIZE - 1
        else:
            gene_map[gene] = J_VOCAB_SIZE
            J_VOCAB_SIZE += 1
            return J_VOCAB_SIZE - 1
    return gene_map[gene]

def encode_sequence(seq, max_len=MAX_SEQ_LEN):
    """Encodes amino acid string to integer list."""
    if pd.isna(seq): return [0] * max_len
    seq = seq[:max_len]
    encoded = [AA_TO_INT.get(aa, 0) for aa in seq]
    padding = [0] * (max_len - len(encoded))
    return encoded + padding

# --- Standalone Parallel Helper Function (CRITICAL FIX: Must be global for pickling) ---
def _process_single_file_global(f, is_train_dir, v_map_train, j_map_train):
    """Helper function for parallel loading of a single TSV file (Global Scope)."""
    try:
        rep_id = os.path.basename(f).replace('.tsv', '')
        df = pd.read_csv(f, sep='\t')
        required_cols = ['junction_aa', 'v_call', 'j_call']
        
        if not all(col in df.columns for col in required_cols):
            return None, None 

        df = df[required_cols].dropna(subset=['junction_aa'])
        
        if not is_train_dir:
            # Use fixed maps from training (Test/Inference)
            # Create mappers using the maps passed from the main process
            v_call_mapper = lambda x: v_map_train.get(x.split('*')[0].split(',')[0].strip(), 0) if pd.notna(x) else 0
            j_call_mapper = lambda x: j_map_train.get(x.split('*')[0].split(',')[0].strip(), 0) if pd.notna(x) else 0
            df['v_call_id'] = df['v_call'].apply(v_call_mapper)
            df['j_call_id'] = df['j_call'].apply(j_call_mapper)
            
            # For Test, return the dataframe with IDs already added
            return rep_id, df
        
        # For Train, return the raw data frame for sequential encoding later
        return rep_id, df 
        
    except Exception as e:
        # print(f"Warning: Failed to load file {f}. Error: {e}. Skipping.")
        return None, None


class AIRRDataset(Dataset):
    """Dataset for MIL, handling variable-sized bags (Repertoires)."""
    def __init__(self, rep_ids, repertoires_data, labels_map=None, is_train=True):
        self.rep_ids = rep_ids
        self.repertoires_data = repertoires_data
        self.labels_map = labels_map
        self.is_train = is_train

    def __len__(self):
        return len(self.rep_ids)

    def __getitem__(self, idx):
        rep_id = self.rep_ids[idx]
        df = self.repertoires_data[rep_id]
        
        if self.is_train and len(df) > MAX_SEQS_PER_BAG:
            df = df.sample(n=MAX_SEQS_PER_BAG, random_state=SEED)
            
        sequences = [encode_sequence(seq) for seq in df['junction_aa'].values]
        seq_tensor = torch.tensor(sequences, dtype=torch.long)
        
        v_tensor = torch.tensor(df['v_call_id'].values, dtype=torch.long)
        j_tensor = torch.tensor(df['j_call_id'].values, dtype=torch.long)
        
        label = torch.tensor(0.0, dtype=torch.float)
        if self.labels_map:
            val = self.labels_map.get(str(rep_id))
            if val is not None:
                label = torch.tensor(val, dtype=torch.float)
                
        raw_df = df[['junction_aa', 'v_call', 'j_call']].reset_index(drop=True)
                
        return seq_tensor, v_tensor, j_tensor, label, str(rep_id), raw_df

def collate_bags(batch):
    """Custom collate function for DataLoader (Batch size = 1 is standard for this MIL implementation)."""
    seqs, v_calls, j_calls, labels, rep_ids, raw_dfs = zip(*batch)
    labels = torch.stack(labels)
    return seqs, v_calls, j_calls, labels, rep_ids, raw_dfs


# =========================================================================================
# 3. MODEL ARCHITECTURE: Gated Attention MIL (with V/J Features)
# =========================================================================================

class AttentionMILModel(nn.Module):
    """
    Multiple Instance Learning model with Gated Attention, integrating
    CDR3 sequence features (GRU) and V/J gene call features (Embeddings).
    """
    def __init__(self, vocab_size, v_vocab_size, j_vocab_size, 
                 embedding_dim=64, hidden_dim=128, mlp_dim=128):
        super().__init__()
        
        # 1. Sequence Encoder (Instance Encoder)
        self.seq_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.seq_encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        seq_out_dim = hidden_dim * 2 
        
        # 2. V/J Embeddings (External Features)
        self.v_embedding = nn.Embedding(v_vocab_size, embedding_dim // 2, padding_idx=0)
        self.j_embedding = nn.Embedding(j_vocab_size, embedding_dim // 2, padding_idx=0)
        vj_out_dim = embedding_dim
        
        # Total Feature Dimension after concatenation
        total_feature_dim = seq_out_dim + vj_out_dim 

        # 3. Gated Attention Mechanism 
        self.attention_V = nn.Sequential(nn.Linear(total_feature_dim, mlp_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(total_feature_dim, mlp_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(mlp_dim, 1)

        # 4. Bag Classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(mlp_dim, 1)
        )
        
        # 

    def forward(self, bag_seqs, bag_v_calls, bag_j_calls):
        # 1. Sequence Features (CDR3)
        embedded_seq = self.seq_embedding(bag_seqs) 
        _, hidden_seq = self.seq_encoder(embedded_seq)
        hidden_seq = torch.cat((hidden_seq[-2], hidden_seq[-1]), dim=1) 

        # 2. V/J Features
        embedded_v = self.v_embedding(bag_v_calls) 
        embedded_j = self.j_embedding(bag_j_calls) 
        
        # 3. Concatenate all features (Multimodal Fusion)
        instance_features = torch.cat((hidden_seq, embedded_v, embedded_j), dim=1)

        # 4. Attention Scores 
        A = self.attention_weights(self.attention_V(instance_features) * self.attention_U(instance_features)) 
        A = torch.softmax(torch.transpose(A, 1, 0), dim=1) 

        # 5. Aggregation (Weighted Average)
        bag_rep = torch.mm(A, instance_features) 
        
        # 6. Classification
        return self.classifier(bag_rep).squeeze(1), A.squeeze(0)


# =========================================================================================
# 4. PREDICTOR ENGINE (TRAINING & INFERENCE)
# =========================================================================================

class ImmuneStatePredictor:
    def __init__(self):
        self.device = DEVICE 
        self.model = None
        self.train_data = {}
        # Maps are initialized globally, but used locally for consistency during training
        self.v_map = V_CALLS_MAP.copy() 
        self.j_map = J_CALLS_MAP.copy()
        self.v_vocab_size = V_VOCAB_SIZE
        self.j_vocab_size = J_VOCAB_SIZE
        self.best_val_loss = float('inf')
        self.best_model_weights = None


    def _load_files(self, directory, is_train_dir):
        """Loads all repertoire data files (.tsv) using parallel processing (process_map)."""
        files = glob.glob(os.path.join(directory, "**", "*.tsv"), recursive=True)
        print(f"ğŸ”� Found {len(files)} repertoire files (.tsv) via deep scan in {directory}")

        # Pass instance maps to the global helper function for TEST data encoding
        # CRITICAL FIX: Use the global function directly
        results = process_map(
            lambda f: _process_single_file_global(f, is_train_dir, self.v_map, self.j_map), 
            files, 
            max_workers=os.cpu_count() * 2 if os.cpu_count() else 4,
            chunksize=8,
            desc="Loading TSV Files"
        )
        
        reps = {}
        for rep_id, df in results:
            if rep_id:
                reps[rep_id] = df

        if is_train_dir:
            # Sequential Gene Encoding Post-Load (Mandatory for dynamic map creation)
            global V_CALLS_MAP, J_CALLS_MAP, V_VOCAB_SIZE, J_VOCAB_SIZE
            V_CALLS_MAP = {}
            J_CALLS_MAP = {}
            V_VOCAB_SIZE = 1 
            J_VOCAB_SIZE = 1 
            
            # Recalculate V/J maps and add IDs to DataFrames sequentially
            print("Encoding V/J genes sequentially after parallel loading...")
            for rep_id, df in tqdm(reps.items(), desc="Sequential Encoding"):
                df['v_call_id'] = df['v_call'].apply(lambda x: get_gene_id(x, V_CALLS_MAP, True))
                df['j_call_id'] = df['j_call'].apply(lambda x: get_gene_id(x, J_CALLS_MAP, False))
            
            # Update instance attributes
            self.v_map = V_CALLS_MAP.copy()
            self.j_map = J_CALLS_MAP.copy()
            self.v_vocab_size = V_VOCAB_SIZE
            self.j_vocab_size = J_VOCAB_SIZE
            
        return reps, pd.DataFrame()


    def fit(self, train_dir, meta_path):
        """Loads data, initializes model, and starts training with Validation and Early Stopping."""
        print("\n--- Starting Training Process ---")
        train_reps, _ = self._load_files(train_dir, is_train_dir=True)
        if not train_reps: raise ValueError("No training repertoires found.")
        
        print(f"ğŸ“Š V/J Vocab Size: V={self.v_vocab_size}, J={self.j_vocab_size}")
        
        # Load Labels
        labels_df = pd.read_csv(meta_path)
        labels_df['repertoire_id'] = labels_df['repertoire_id'].astype(str)
        
        if 'label_positive' not in labels_df.columns:
            raise KeyError(f"Expected 'label_positive' column not found in metadata. Available columns: {labels_df.columns.tolist()}")
            
        labels_map = dict(zip(labels_df['repertoire_id'], labels_df['label_positive']))
        self.train_data['repertoires'] = train_reps

        # --- Train/Validation Split ---
        all_rep_ids = list(train_reps.keys())
        train_ids, val_ids = train_test_split(all_rep_ids, test_size=0.2, random_state=SEED)
        
        train_ds = AIRRDataset(train_ids, train_reps, labels_map, is_train=True)
        val_ds = AIRRDataset(val_ids, train_reps, labels_map, is_train=False) 
        
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_bags, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_bags, num_workers=2)
        
        # Model, Optimizer, and Scheduler Setup
        self.model = AttentionMILModel(VOCAB_SIZE, self.v_vocab_size, self.j_vocab_size).to(self.device)
        opt = optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, T_max=20, eta_min=1e-7) 
        crit = nn.BCEWithLogitsLoss()
        
        PATIENCE = 5
        epochs_no_improve = 0
        EPOCHS = 30 

        for epoch in range(EPOCHS):
            # Training Loop
            total_loss = 0
            self.model.train()
            for seqs, v_calls, j_calls, labels, _, _ in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}"):
                
                seqs, v_calls, j_calls = seqs[0].to(self.device), v_calls[0].to(self.device), j_calls[0].to(self.device)
                labels = labels[0].to(self.device).view(-1) # CRITICAL FIX: Reshape labels to [1]
                
                opt.zero_grad()
                logits, _ = self.model(seqs, v_calls, j_calls)
                loss = crit(logits, labels)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            scheduler.step()

            # Validation Loop
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for seqs, v_calls, j_calls, labels, _, _ in val_loader:
                    seqs, v_calls, j_calls = seqs[0].to(self.device), v_calls[0].to(self.device), j_calls[0].to(self.device)
                    labels = labels[0].to(self.device).view(-1) # CRITICAL FIX: Reshape labels to [1]
                    
                    logits, _ = self.model(seqs, v_calls, j_calls)
                    val_loss += crit(logits, labels).item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} finished. Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            # Early Stopping Check
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_weights = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == PATIENCE:
                    print(f"ğŸ›‘ Early stopping triggered after {epoch+1} epochs.")
                    break
        
        # Load best model weights for inference
        if self.best_model_weights:
            self.model.load_state_dict(self.best_model_weights)
            print("âœ… Loaded best model weights.")

        print("--- Training Completed ---")


    def predict(self, test_dir):
        """Performs inference on test data (Task 1)."""
        print("\n--- Phase 2: Predicting Test Set ---")
        test_reps, _ = self._load_files(test_dir, is_train_dir=False) 
        
        loader = DataLoader(AIRRDataset(list(test_reps.keys()), test_reps, is_train=False), 
                            batch_size=1, collate_fn=collate_bags, num_workers=2)
        
        preds = {}
        self.model.eval()
        with torch.no_grad():
            for seqs, v_calls, j_calls, _, rep_ids, _ in tqdm(loader, desc="Inference"):
                seqs, v_calls, j_calls = seqs[0].to(self.device), v_calls[0].to(self.device), j_calls[0].to(self.device)
                logits, _ = self.model(seqs, v_calls, j_calls)
                preds[rep_ids[0]] = torch.sigmoid(logits).item()
        return preds

    def interpret(self):
        """Extracts attention scores for sequence ranking (Task 2)."""
        print("\n--- Phase 3: Interpreting Sequences (Attention Scores) ---")
        
        rep_id_to_dataset_id = {}
        # NOTE: Using self.train_data['repertoires'] keys here assumes the training data was loaded successfully
        for rep_id in self.train_data['repertoires'].keys():
            try:
                # Assuming TRAIN_DIR is defined globally
                full_path = glob.glob(os.path.join(TRAIN_DIR, "**", f"{rep_id}.tsv"), recursive=True)[0]
                dataset_id = os.path.basename(os.path.dirname(os.path.dirname(full_path)))
                rep_id_to_dataset_id[rep_id] = dataset_id
            except:
                 rep_id_to_dataset_id[rep_id] = "unknown_dataset" 


        loader = DataLoader(AIRRDataset(list(self.train_data['repertoires'].keys()), self.train_data['repertoires'], is_train=False), 
                            batch_size=1, collate_fn=collate_bags, num_workers=2)
        scores = {} # {dataset_id: { (junc, v, j): max_score }}

        self.model.eval()
        with torch.no_grad():
            for seqs, v_calls, j_calls, _, rep_ids, dfs in tqdm(loader, desc="Scanning Attention"):
                
                rep_id = rep_ids[0]
                ds_id = rep_id_to_dataset_id.get(rep_id)
                if not ds_id or ds_id == "unknown_dataset": continue
                
                if ds_id not in scores: scores[ds_id] = {}
                
                seqs, v_calls, j_calls = seqs[0].to(self.device), v_calls[0].to(self.device), j_calls[0].to(self.device)
                _, attn = self.model(seqs, v_calls, j_calls)
                attn = attn.cpu().numpy()
                df = dfs[0] 
                
                # Store the max attention score for each unique sequence/V/J combination
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
# 5. EXECUTION PIPELINE (FINAL SUBMISSION)
# =========================================================================================

# Initialize
predictor = ImmuneStatePredictor()

try:
    if METADATA_PATH:
        # A. Train
        predictor.fit(TRAIN_DIR, METADATA_PATH)
        
        # B. Task 1 Predictions
        preds = predictor.predict(TEST_DIR)
        df1 = pd.DataFrame(list(preds.items()), columns=['repertoire_id', 'probability'])
        
        # Prepare df1 for concatenation with dummy values for Task 2 columns
        for c in ['dataset_id', 'junction_aa', 'v_call', 'j_call', 'rank']: df1[c] = -999.0 
        
        # C. Task 2 Interpretation
        df2 = predictor.interpret()
        
        # Prepare df2 for concatenation with dummy values for Task 1 columns
        df2['repertoire_id'] = "dummy_id"
        df2['probability'] = -999.0
        df2 = df2[df1.columns] # Ensure column order matches df1
        
        # D. Submission
        final = pd.concat([df1, df2], ignore_index=True)
        
        # Ensure correct data types for final submission file
        final = final.astype({'repertoire_id': str, 'dataset_id': str, 'junction_aa': str, 
                              'v_call': str, 'j_call': str, 'probability': float, 'rank': float})
        final.to_csv("submission.csv", index=False)
        
        print("\n--- Final Submission Summary ---")
        print(f"âœ… Success! Submission saved to submission.csv")
        print(f"Final shape: {final.shape}")
        
    else:
        print("â�Œ ERROR: Metadata file not found. Submission cannot be generated.")

except Exception as e:
    print(f"\nâ�Œ CRITICAL EXECUTION FAILURE: {e}")

