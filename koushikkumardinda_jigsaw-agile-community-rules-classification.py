# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ====================================================================
# JIGSAW AGENT: COMPLETE OPTIMIZED SCRIPT (FINAL CORRECTION)
# ====================================================================

# --- Setup and Imports ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
MODEL_PATH = '/kaggle/input/pretrain_model.torch/pytorch/default/6' 
MAX_LEN = 256
BATCH_SIZE = 32
N_FOLDS = 5
EPOCHS = 4
LEARNING_RATE = 2e-5
SEED = 42

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- CRITICAL DEBUGGING & SAFETY ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(SEED)
print(f"Using device: {device}")


# ====================================================================
# TASK 1: MODEL & DATASET DEFINITION 
# ====================================================================

class JigsawDataset(Dataset):
    """Handles hybrid input with corrected subreddit index safety."""
    def __init__(self, df, tokenizer, max_len, subreddit_map):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.subreddit_map = subreddit_map
        self.num_subreddits_base = len(subreddit_map) 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        body = str(row['body'])
        rule = str(row['rule'])
        subreddit = str(row['subreddit']) # Ensure subreddit is string
        
        # Contextual Concatenation
        text = self.tokenizer.sep_token.join([rule, body])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # --- Subreddit Index Safety (Corrected logic) ---
        subreddit_idx = self.subreddit_map.get(subreddit)
        
        if subreddit_idx is None:
            # Map ANY unseen subreddit to the reserved index N
            subreddit_idx = self.num_subreddits_base 
            
        
        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'subreddit_idx': torch.tensor(subreddit_idx, dtype=torch.long)
        }

        if 'rule_violation' in row:
            inputs['labels'] = torch.tensor(row['rule_violation'], dtype=torch.float)

        return inputs


class ContextFusedModel(nn.Module):
    """
    Accepts an already loaded and resized LLM object to ensure tokenizer alignment.
    """
    def __init__(self, llm_model, num_subreddits, hidden_size=768):
        super().__init__()
        
        # --- FIX: Receive pre-loaded model instead of loading inside __init__ ---
        self.llm = llm_model
        llm_output_size = self.llm.config.hidden_size
        
        SUBREDDIT_EMBED_DIM = hidden_size // 8
        # num_subreddits here is the base count, +1 handles the unseen category
        self.subreddit_embedding = nn.Embedding(num_subreddits + 1, SUBREDDIT_EMBED_DIM) 
        
        combined_feature_size = llm_output_size + SUBREDDIT_EMBED_DIM
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(combined_feature_size, 1) 
        )

    def forward(self, input_ids, attention_mask, subreddit_idx):
        llm_output = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_token_output = llm_output[:, 0, :]
        
        subreddit_embed = self.subreddit_embedding(subreddit_idx)
        
        fused_features = torch.cat((cls_token_output, subreddit_embed), dim=1)
        
        logits = self.classifier(fused_features)
        
        return logits.flatten()


# ====================================================================
# TASK 2: TRAINING, EVALUATION, AND ENSEMBLE
# ====================================================================


def train_model(model, data_loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        subreddit_idx = batch['subreddit_idx'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, subreddit_idx=subreddit_idx)
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            subreddit_idx = batch['subreddit_idx'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, subreddit_idx=subreddit_idx)
            preds = torch.sigmoid(outputs).cpu().numpy() 
            all_preds.extend(preds)
    return np.array(all_preds)


def run_cv_ensemble(train_df, test_df, tokenizer, llm_model, num_subreddits):
    
    train_df['stratify_group'] = train_df['rule'].astype(str) + '_' + train_df['rule_violation'].astype(str)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train_df))
    test_preds_list = []
    
    # Build map based only on the training data set structure
    all_train_subreddits = sorted(train_df['subreddit'].unique())
    subreddit_map = {sub: i for i, sub in enumerate(all_train_subreddits)}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['stratify_group'])):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Datasets/Loaders
        train_dataset = JigsawDataset(fold_train_df, tokenizer, MAX_LEN, subreddit_map)
        val_dataset = JigsawDataset(fold_val_df, tokenizer, MAX_LEN, subreddit_map)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

        # Model Initialization: Pass the pre-loaded LLM object
        model = ContextFusedModel(llm_model, num_subreddits).to(device)
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE) 
        
        total_steps = len(train_loader) * EPOCHS 
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        loss_fn = BCEWithLogitsLoss().to(device)
        
        best_val_auc = -1.0
        model_save_path = f'best_model_fold_{fold}.pth'
        
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_model(model, train_loader, optimizer, scheduler, loss_fn)
            val_preds = evaluate_model(model, val_loader)
            
            val_auc = roc_auc_score(fold_val_df['rule_violation'], val_preds)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), model_save_path)
                
        # Load best for final OOF/Test prediction
        model.load_state_dict(torch.load(model_save_path))
        oof_preds[val_idx] = evaluate_model(model, val_loader)
        
        # Test set prediction
        test_dataset = JigsawDataset(test_df, tokenizer, MAX_LEN, subreddit_map)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 4, shuffle=False)
        test_preds = evaluate_model(model, test_loader)
        test_preds_list.append(test_preds)

        # Clean up
        del model; torch.cuda.empty_cache()
        os.remove(model_save_path)


    final_test_preds = np.mean(test_preds_list, axis=0)
    
    oof_auc = roc_auc_score(train_df['rule_violation'], oof_preds)
    print(f"\n--- FOLDING COMPLETE ---")
    print(f"Final OOF AUC (CV Score): {oof_auc:.5f}")
    
    return final_test_preds


# ====================================================================
# TASK 3: EXECUTION AND SUBMISSION (Model Pre-loading)
# ====================================================================

def main():
    # --- Data Loading ---
    TRAIN_PATH = '/kaggle/input/jigsaw-agile-community-rules/train.csv'
    TEST_PATH = '/kaggle/input/jigsaw-agile-community-rules/test.csv'
    SAMPLE_SUB_PATH = '/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv'
    
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        sample_submission = pd.read_csv(SAMPLE_SUB_PATH)
    except FileNotFoundError:
        print("Error loading data. Check Kaggle input paths.")
        return

    # --- Global Model and Tokenizer Initialization (CRITICAL FIX) ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 1. Handle Special Tokens (Ensure Consistent Vocabulary)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    
    # 2. Load Base Model
    llm_model = AutoModel.from_pretrained(MODEL_PATH)
    
    # 3. Resize Model Embeddings to match Tokenizer size (Prevents CUDA assert)
    if len(tokenizer) != llm_model.get_input_embeddings().num_embeddings:
        print(f"Resizing LLM embeddings from {llm_model.get_input_embeddings().num_embeddings} to {len(tokenizer)}")
        llm_model.resize_token_embeddings(len(tokenizer))
    
    # Determine number of unique subreddits (for ContextFusedModel initialization)
    all_train_subreddits = sorted(train_df['subreddit'].unique())
    num_subreddits = len(all_train_subreddits)
    
    # --- Execute Training/Ensemble ---
    final_predictions = run_cv_ensemble(
        train_df, test_df, tokenizer, llm_model, num_subreddits
    )
    
    # Clean up model after all folds are done
    del llm_model
    torch.cuda.empty_cache()

    # --- Submission File Generation ---
    submission_df = sample_submission.copy()
    submission_df['row_id'] = test_df['row_id'] 
    submission_df['rule_violation'] = final_predictions
    
    submission_df['rule_violation'] = np.clip(submission_df['rule_violation'], 1e-15, 1 - 1e-15)
    
    submission_df.to_csv('submission.csv', index=False)
    print("\nâœ… Submission file 'submission.csv' successfully created. Ready for submission! ðŸš€")

if __name__ == '__main__':
    main()

