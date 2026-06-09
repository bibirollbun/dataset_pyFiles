# ======================================================================
# QWEN 14B GPTQ-INT4 - COMPLETE PIPELINE WITH META-STACKING
# ======================================================================

"""
STRATEGY: Single 14B Model with Calibration

Why skip 7B:
- 7B showed poor performance (60% AUC, only 11 unique predictions)
- 14B should give 87-90% AUC on its own
- Skip ensemble, focus on single strong model

Expected:
- Training time: 6-8 hours
- Expected AUC: 87-90% (calibrated)
- Competitive for top 20-30%
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("QWEN 14B GPTQ-INT4 - SINGLE MODEL APPROACH")
print("="*70)

import sys
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("QWEN 14B GPTQ-INT4 - OFFLINE MODE")
print("="*70)

# ======================================================================
# STEP 1: INSTALL DEPENDENCIES FROM OFFLINE DATASET
# ======================================================================

print("\n[Step 1/10] Installing dependencies from offline dataset...")
print("-"*70)

# UPDATE THIS PATH to match your Kaggle dataset
DEPENDENCIES_PATH = "/kaggle/input/model-offline/kaggle_gptq_offline_package/wheels"

print(f"Installing from: {DEPENDENCIES_PATH}")

# Verify path exists
print("\nVerifying dataset path...")
!ls -la {DEPENDENCIES_PATH} | head -20

# Install dependencies
print("\n" + "-"*70)
print("Installing packages...")
print("-"*70)

print("\n[1/4] Installing gekko...")
!pip install --no-index --no-deps --find-links {DEPENDENCIES_PATH} gekko -q

print("\n[2/4] Installing rouge packages...")
!pip install --no-index --no-deps --find-links {DEPENDENCIES_PATH} rouge rouge-score absl-py six nltk -q

print("\n[3/4] Installing optimum...")
!pip install --no-index --no-deps --find-links {DEPENDENCIES_PATH} optimum -q

print("\n[4/4] Installing auto-gptq...")
!pip install --no-index --no-deps --find-links {DEPENDENCIES_PATH} auto-gptq -q

print("\n" + "-"*70)
print("âœ“ Installation commands completed")
print("-"*70)

# ======================================================================
# STEP 2: VERIFY INSTALLATIONS
# ======================================================================

print("\n[Step 2/10] Verifying installations...")
print("-"*70)

installation_success = True

# Verify gekko
try:
    import gekko
    print(f"  âœ“ gekko imported successfully")
except ImportError as e:
    print(f"  âœ— gekko import failed: {e}")
    installation_success = False

# Verify auto-gptq
try:
    import auto_gptq
    print(f"  âœ“ auto-gptq version: {auto_gptq.__version__}")
except ImportError as e:
    print(f"  âœ— auto-gptq import failed: {e}")
    installation_success = False

# Verify optimum
try:
    from optimum.gptq import GPTQQuantizer
    print(f"  âœ“ optimum.gptq imported successfully")
except ImportError as e:
    print(f"  âœ— optimum import failed: {e}")
    installation_success = False

if not installation_success:
    print("\nâ�Œ Installation failed. Please check errors above.")
    sys.exit(1)

print("\nâœ“ All dependencies verified!")
print("="*70)

# ======================================================================
# STEP 2: IMPORT LIBRARIES
# ======================================================================

print("\n[Step 2/8] Importing libraries...")
print("-"*70)

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import gc
import re
import time
from datetime import datetime

print("âœ“ Libraries imported")
print("="*70)

# ======================================================================
# STEP 3: CONFIGURATION
# ======================================================================

class Config:
    # Data paths
    TRAIN_PATH = '/kaggle/input/jigsaw-agile-community-rules/train.csv'
    TEST_PATH = '/kaggle/input/jigsaw-agile-community-rules/test.csv'
    WORK_DIR = '/kaggle/working/'
    
    # Model path - UPDATE THIS
    MODEL_14B_PATH = "/kaggle/input/qwen2.5/transformers/14b-instruct-gptq-int4/1"
    
    # Inference settings - OPTIMIZED
    MAX_NEW_TOKENS = 8
    TEMPERATURE = 0.3
    
    # Calibration
    USE_CALIBRATION = True
    USE_META_STACKING = True  # Will train meta-model on 14B predictions
    SEED = 42

config = Config()

print(f"\n[Step 3/8] Configuration")
print("-"*70)
print(f"âœ“ Model: {config.MODEL_14B_PATH}")
print(f"âœ“ Quantization: GPTQ Int4")
print(f"âœ“ Expected AUC: 87-90%")
print(f"âœ“ Meta-stacking: Enabled")
print("="*70)


# ======================================================================
# STEP 4: LOAD DATA
# ======================================================================

print(f"\n[Step 4/8] Loading data...")
print("-"*70)

train_df = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)

print(f"âœ“ Train: {len(train_df)} samples")
print(f"âœ“ Test: {len(test_df)} samples")
print(f"âœ“ Violation rate: {train_df['rule_violation'].mean():.1%}")
print("="*70)

# ======================================================================
# STEP 5: PROMPT AND INFERENCE FUNCTIONS
# ======================================================================

print(f"\n[Step 5/8] Defining Prompt and Inference Functions...")
print("-"*70)

def create_prompt(row):
    """
    Optimized prompt for Qwen 14B.
    """
    prompt = f"""Analyze if this Reddit comment violates a subreddit rule.

**Subreddit**: r/{row['subreddit']}
**Rule Being Enforced**: {row['rule']}

**Clear Violations of This Rule**:
- "{row['positive_example_1']}"
- "{row['positive_example_2']}"

**Acceptable Comments (NOT violations)**:
- "{row['negative_example_1']}"
- "{row['negative_example_2']}"

**Comment Under Review**:
"{row['body']}"

**Your Task**: 
Compare this comment to the violation examples. Does it break the same rule in a similar way? Consider the specific subreddit context and cultural norms of r/{row['subreddit']}.

Output a probability from 0.0 (clearly allowed) to 1.0 (clear violation). Be precise - use the full range (e.g., 0.15, 0.73, 0.92).

**Probability**:"""
    
    return prompt

def get_violation_probability(row, model, tokenizer):
    """
    Optimized inference with timing.
    Returns: (probability, inference_time_seconds)
    """
    start_time = time.time()
    
    try:
        prompt = create_prompt(row)
        
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1536
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Extract probability
        numbers = re.findall(r'\b(?:0?\.\d+|1\.0|[01])\b', response)
        
        if numbers:
            prob = float(numbers[0])
            prob = max(0.0, min(1.0, prob))
        else:
            prob = 0.5
        
        inference_time = time.time() - start_time
        return prob, inference_time
    
    except Exception as e:
        inference_time = time.time() - start_time
        return 0.5, inference_time

print(f"âœ“ Functions created: create_prompt, get_violation_probability")
print("="*70)


# ======================================================================
# STEP 6: LOAD QWEN 14B MODEL
# ======================================================================

# Note: Renamed to [Step 5/8] in output as per original script's print statements
print(f"\n[Step 5/8] Loading QWEN 14B (GPTQ INT4)") 
print("="*70)

# Clear memory
gc.collect()
torch.cuda.empty_cache()

# Check GPU
if not torch.cuda.is_available():
    print("â�Œ No GPU detected! Enable T4 x2 in Kaggle settings.")
    sys.exit(1)

num_gpus = torch.cuda.device_count()
print(f"âœ“ {num_gpus} GPU(s) available")

for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")

# Load tokenizer
print("\nLoading tokenizer...")
start_load = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_14B_PATH,
    trust_remote_code=True
)

print(f"âœ“ Tokenizer loaded ({time.time() - start_load:.1f}s)")

# Load model (may use 1 or 2 GPUs)
print("\nLoading 14B model...")
print("(This may take 2-3 minutes...)")
start_load = time.time()

model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_14B_PATH,
    device_map="auto",  # Auto-distribute across available GPUs
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    use_cache=True
)

model.eval()

load_time = time.time() - start_load
print(f"âœ“ Model loaded ({load_time:.1f}s)")

# Check memory distribution
print("\nGPU Memory Usage:")
print("-"*70)
total_mem_used = 0
for i in range(num_gpus):
    mem_used = torch.cuda.memory_allocated(i) / 1e9
    mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
    total_mem_used += mem_used
    if mem_used > 0.1:
        print(f"  GPU {i}: {mem_used:.2f}GB / {mem_total:.2f}GB ({mem_used/mem_total*100:.1f}%)")

print(f"  Total: {total_mem_used:.2f}GB")

# Quick speed test
print("\n[Quick Speed Test - 5 samples]")
print("-"*70)

test_times = []
for idx in range(5):
    row = train_df.iloc[idx]
    prob, inf_time = get_violation_probability(row, model, tokenizer)
    test_times.append(inf_time)
    print(f"  Sample {idx+1}: {inf_time:.2f}s (pred={prob:.3f}, actual={row['rule_violation']})")

avg_time = np.mean(test_times)
total_est = (avg_time * len(train_df)) / 60

print(f"\nâœ“ Average: {avg_time:.2f}s per sample")
print(f"âœ“ Estimated total time: {total_est:.0f} minutes (~{total_est/60:.1f} hours)")

if avg_time > 15:
    print("\nâš ï¸�  Slower than expected (>15s/sample)")
    print("    This may take longer than estimated")
elif avg_time > 10:
    print("\nâš ï¸�  On the slower side (10-15s/sample)")
else:
    print("\nâœ… Good speed!")

print("="*70)


# ======================================================================
# EMBEDDINGS + XGBOOST APPROACH (MOST RELIABLE)
# ======================================================================

print("="*70)
print("EMBEDDINGS + XGBOOST PIPELINE")
print("="*70)

def get_embeddings(text, rule, model, tokenizer):
    """Extract embeddings from 14B model."""
    # Combine text and rule
    prompt = f"Rule: {rule}\n\nComment: {text}"
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    )
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use last hidden state, mean pooling
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        embeddings = hidden_states.mean(dim=1)  # (batch, hidden_dim)
    
    return embeddings[0].cpu().numpy()

# ======================================================================
# STEP 1: EXTRACT EMBEDDINGS (TAKES ~1 HOUR)
# ======================================================================

print("\n[Step 1/3] Extracting embeddings from training data...")
print("-"*70)

train_embeddings = []

start_time = time.time()

for idx in range(len(train_df)):
    row = train_df.iloc[idx]
    
    embedding = get_embeddings(row['body'], row['rule'], model, tokenizer)
    train_embeddings.append(embedding)
    
    if (idx + 1) % 100 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = (len(train_df) - idx - 1) * avg_time
        
        print(f"  Progress: {idx+1}/{len(train_df)} | "
              f"Elapsed: {elapsed/60:.1f}min | "
              f"ETA: {remaining/60:.1f}min")
        sys.stdout.flush()
    
    # Memory cleanup
    if idx % 200 == 0 and idx > 0:
        gc.collect()
        torch.cuda.empty_cache()

train_embeddings = np.array(train_embeddings)

print(f"\nâœ“ Training embeddings extracted: {train_embeddings.shape}")
print(f"  Time: {(time.time() - start_time)/60:.1f} minutes")

# ======================================================================
# STEP 2: EXTRACT TEST EMBEDDINGS
# ======================================================================

print("\n[Step 2/3] Extracting embeddings from test data...")
print("-"*70)

test_embeddings = []

for idx in range(len(test_df)):
    row = test_df.iloc[idx]
    
    embedding = get_embeddings(row['body'], row['rule'], model, tokenizer)
    test_embeddings.append(embedding)
    
    if (idx + 1) % 50 == 0 or (idx + 1) == len(test_df):
        print(f"  Progress: {idx+1}/{len(test_df)}")
        sys.stdout.flush()
    
    if idx % 100 == 0 and idx > 0:
        gc.collect()
        torch.cuda.empty_cache()

test_embeddings = np.array(test_embeddings)

print(f"\nâœ“ Test embeddings extracted: {test_embeddings.shape}")

# Clear model from memory - no longer needed
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()

print("\nâœ“ Model cleared from memory")

# ======================================================================
# STEP 3: TRAIN XGBOOST CLASSIFIER
# ======================================================================

print("\n[Step 3/3] Training XGBoost on embeddings...")
print("-"*70)

import xgboost as xgb

# Prepare data
X_train = train_embeddings
y_train = train_df['rule_violation'].values

X_test = test_embeddings

# Train XGBoost with cross-validation
from sklearn.model_selection import StratifiedKFold

cv_scores = []
test_predictions_cv = np.zeros(len(test_df))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTraining with 5-fold CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFold {fold+1}/5:")
    
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'tree_method': 'gpu_hist',  # Use GPU
        'gpu_id': 0
    }
    
    # Train
    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model_xgb = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Predict on validation
    val_pred = model_xgb.predict(dval)
    fold_auc = roc_auc_score(y_fold_val, val_pred)
    cv_scores.append(fold_auc)
    
    print(f"  Validation AUC: {fold_auc:.4f}")
    
    # Predict on test
    dtest = xgb.DMatrix(X_test)
    test_predictions_cv += model_xgb.predict(dtest) / 5

# Average CV score
mean_cv_auc = np.mean(cv_scores)
std_cv_auc = np.std(cv_scores)

print(f"\n{'='*70}")
print("XGBOOST RESULTS:")
print("="*70)
print(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean CV AUC: {mean_cv_auc:.4f} Â± {std_cv_auc:.4f}")
print("="*70)




# ======================================================================
# ENHANCED FEATURES + XGBOOST
# ======================================================================

print("="*70)
print("ADDING ENHANCED FEATURES")
print("="*70)

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# ======================================================================
# STEP 1: CREATE ADDITIONAL FEATURES
# ======================================================================

print("\n[Creating additional features...]")
print("-"*70)

def create_text_features(df):
    """Create text-based features."""
    features = pd.DataFrame()
    
    # Length features
    features['body_length'] = df['body'].str.len()
    features['body_words'] = df['body'].str.split().str.len()
    features['rule_length'] = df['rule'].str.len()
    
    # Punctuation features
    features['exclamation_count'] = df['body'].str.count('!')
    features['question_count'] = df['body'].str.count(r'\?')
    features['caps_ratio'] = df['body'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    
    # URL and link features
    features['has_url'] = df['body'].str.contains(r'http|www', case=False).astype(int)
    features['has_link'] = df['body'].str.contains(r'\[.*\]\(.*\)', case=False).astype(int)
    
    # Example similarity (simple keyword matching)
    features['matches_positive_1'] = df.apply(
        lambda row: any(word.lower() in row['body'].lower() 
                       for word in str(row['positive_example_1']).split()[:5]), 
        axis=1
    ).astype(int)
    
    features['matches_positive_2'] = df.apply(
        lambda row: any(word.lower() in row['body'].lower() 
                       for word in str(row['positive_example_2']).split()[:5]), 
        axis=1
    ).astype(int)
    
    return features.fillna(0).values

print("Creating text features for train...")
train_text_features = create_text_features(train_df)
print(f"  Train text features: {train_text_features.shape}")

print("Creating text features for test...")
test_text_features = create_text_features(test_df)
print(f"  Test text features: {test_text_features.shape}")


# ======================================================================
# STEP 2: CREATE TF-IDF FEATURES
# ======================================================================

print("\n[Creating TF-IDF features...]")
print("-"*70)

# TF-IDF on comment body
tfidf_body = TfidfVectorizer(
    max_features=500,
    min_df=3,
    max_df=0.8,
    ngram_range=(1, 2),
    sublinear_tf=True
)

print("Fitting TF-IDF on body...")
train_tfidf_body = tfidf_body.fit_transform(train_df['body'])
test_tfidf_body = tfidf_body.transform(test_df['body'])
print(f"  Body TF-IDF shape: {train_tfidf_body.shape}")

# TF-IDF on rule text
tfidf_rule = TfidfVectorizer(
    max_features=100,
    min_df=2,
    ngram_range=(1, 2)
)

print("Fitting TF-IDF on rules...")
train_tfidf_rule = tfidf_rule.fit_transform(train_df['rule'])
test_tfidf_rule = tfidf_rule.transform(test_df['rule'])
print(f"  Rule TF-IDF shape: {train_tfidf_rule.shape}")

# ======================================================================
# STEP 3: COMBINE ALL FEATURES (FIXED)
# ======================================================================

print("\n[Combining all features...]")
print("-"*70)

# Convert embeddings to float32 first, THEN to sparse
train_embeddings_float32 = train_embeddings.astype(np.float32)
test_embeddings_float32 = test_embeddings.astype(np.float32)

train_embeddings_sparse = csr_matrix(train_embeddings_float32)
test_embeddings_sparse = csr_matrix(test_embeddings_float32)

train_text_features_sparse = csr_matrix(train_text_features)
test_text_features_sparse = csr_matrix(test_text_features)

# Combine: Embeddings + Text Features + TF-IDF
X_train_combined = hstack([
    train_embeddings_sparse,
    train_text_features_sparse,
    train_tfidf_body,
    train_tfidf_rule
])

X_test_combined = hstack([
    test_embeddings_sparse,
    test_text_features_sparse,
    test_tfidf_body,
    test_tfidf_rule
])

print(f"\nâœ“ Combined features shape:")
print(f"  Train: {X_train_combined.shape}")
print(f"  Test: {X_test_combined.shape}")
# ======================================================================
# STEP 4: TRAIN XGBOOST WITH COMBINED FEATURES
# ======================================================================

print("\n[Training XGBoost with enhanced features...]")
print("="*70)

y_train = train_df['rule_violation'].values

cv_scores_enhanced = []
test_predictions_enhanced = np.zeros(len(test_df))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTraining with 5-fold CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train)):
    print(f"\nFold {fold+1}/5:")
    
    X_fold_train = X_train_combined[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_combined[val_idx]
    y_fold_val = y_train[val_idx]
    
    # Enhanced XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 7,  # Deeper tree
        'learning_rate': 0.03,  # Lower learning rate
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42,
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    }
    
    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model_xgb = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,  # More rounds
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=50
    )
    
    # Predict
    val_pred = model_xgb.predict(dval)
    fold_auc = roc_auc_score(y_fold_val, val_pred)
    cv_scores_enhanced.append(fold_auc)
    
    print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")
    
    # Test predictions
    dtest = xgb.DMatrix(X_test_combined)
    test_predictions_enhanced += model_xgb.predict(dtest) / 5

mean_cv_enhanced = np.mean(cv_scores_enhanced)
std_cv_enhanced = np.std(cv_scores_enhanced)

print(f"\n{'='*70}")
print("ENHANCED XGBOOST RESULTS:")
print("="*70)
print(f"  Previous (embeddings only): 0.8335")
print(f"  Enhanced (all features):    {mean_cv_enhanced:.4f}")
print(f"  Improvement:                +{(mean_cv_enhanced - 0.8335)*100:.2f}pp")
print(f"  Std:                        {std_cv_enhanced:.4f}")
print("="*70)



# ======================================================================
# ADD LIGHTGBM ENSEMBLE
# ======================================================================

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

print("\n" + "="*70)
print("TRAINING LIGHTGBM ENSEMBLE")
print("="*70)

cv_scores_lgb = []

# ======================================================================
# FIX: INITIALIZE THE TEST PREDICTION ARRAY HERE
# ======================================================================
# Assumes 'test_df' or 'X_test_combined' is already loaded and has the correct shape
test_predictions_lgb = np.zeros(len(test_df))
# ======================================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTraining LightGBM with 5-fold CV...")

# (The rest of your loop code follows...)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train)):
    print(f"\nFold {fold+1}/5:")
    
    X_fold_train = X_train_combined[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_combined[val_idx]
    y_fold_val = y_train[val_idx]
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 7,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'seed': 42,
        'verbose': -1
    }
    
    dtrain = lgb.Dataset(X_fold_train, label=y_fold_train)
    dval = lgb.Dataset(X_fold_val, label=y_fold_val, reference=dtrain)
    
    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(50)
        ]
    )
    
    # Validate
    val_pred = model_lgb.predict(X_fold_val)
    fold_auc = roc_auc_score(y_fold_val, val_pred)
    cv_scores_lgb.append(fold_auc)
    
    print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")
    
    # Test predictions (This line will now work)
    test_predictions_lgb += model_lgb.predict(X_test_combined) / 5

mean_cv_lgb = np.mean(cv_scores_lgb)

print(f"\n{'='*70}")
print("LIGHTGBM RESULTS:")
print("="*70)
print(f"  Mean CV AUC: {mean_cv_lgb:.4f} Â± {np.std(cv_scores_lgb):.4f}")
print("="*70)


# ======================================================================
# ENSEMBLE: XGBoost + LightGBM
# ======================================================================

print("\n" + "="*70)
print("CREATING ENSEMBLE")
print("="*70)

# Try different ensemble weights
ensemble_weights = [
    (0.5, 0.5, "50-50"),
    (0.6, 0.4, "60-40 XGB"),
    (0.4, 0.6, "40-60 LGB"),
    (0.7, 0.3, "70-30 XGB")
]

best_weight = None
best_ensemble_auc = 0

# Mocking missing variables for context
# test_predictions_enhanced is assumed to be the prediction array from your XGBoost model
# Example:
# test_predictions_enhanced = np.random.rand(len(test_df)) 

print("\nTesting ensemble weights on validation folds...")

for w_xgb, w_lgb, name in ensemble_weights:
    # We need to recalculate CV predictions for XGBoost (quick version)
    # For now, let's use the test predictions
    
    test_ensemble = w_xgb * test_predictions_enhanced + w_lgb * test_predictions_lgb
    
    # Estimate performance (use average of individual models as proxy)
    # The 0.8368 value is the hardcoded CV score from your previous XGBoost model
    estimated_auc = w_xgb * 0.8368 + w_lgb * mean_cv_lgb
    
    print(f"  {name}: Estimated ~{estimated_auc:.4f}")
    
    if estimated_auc > best_ensemble_auc:
        best_ensemble_auc = estimated_auc
        best_weight = (w_xgb, w_lgb, name)
        best_test_ensemble = test_ensemble

print(f"\nâœ“ Best ensemble: {best_weight[2]}")
print(f"  Estimated AUC: {best_ensemble_auc:.4f}")


# ======================================================================
# FINAL COMPARISON:
# ======================================================================
import pandas as pd # Adding import for submission section

print("\n" + "="*70)
print("FINAL COMPARISON:")
print("="*70)
print(f"  Embeddings only:       0.8335")
print(f"  Enhanced XGBoost:      0.8368")
print(f"  LightGBM:              {mean_cv_lgb:.4f}")
print(f"  XGB+LGB Ensemble:      ~{best_ensemble_auc:.4f}")
print("="*70)


# ======================================================================
# NEURAL NETWORK MODEL
# ======================================================================

print("\n" + "="*70)
print("TRAINING NEURAL NETWORK")
print("="*70)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

# ======================================================================
# STEP 1: PREPARE DATA FOR NEURAL NETWORK
# ======================================================================

print("\n[Step 1/4] Preparing data for neural network...")
print("-"*70)

# Mocking missing variables for context
# X_train_combined, X_test_combined are assumed to be sparse matrices
# y_train, test_df are assumed to be loaded
# test_predictions_enhanced, test_predictions_lgb are assumed to exist
# from previous steps.

# Convert sparse matrices to dense (NN needs dense input)
print("Converting sparse matrices to dense...")
X_train_dense = X_train_combined.toarray().astype(np.float32)
X_test_dense = X_test_combined.toarray().astype(np.float32)

print(f"âœ“ Dense shape: {X_train_dense.shape}")

# Standardize features for NN
scaler_nn = StandardScaler()
X_train_scaled = scaler_nn.fit_transform(X_train_dense)
X_test_scaled = scaler_nn.transform(X_test_dense)

print(f"âœ“ Features scaled")


# ======================================================================
# STEP 2: DEFINE NEURAL NETWORK ARCHITECTURE
# ======================================================================

print("\n[Step 2/4] Defining neural network architecture...")
print("-"*70)

class BinaryClassificationNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64]):
        super(BinaryClassificationNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

input_dim = X_train_scaled.shape[1]
print(f"âœ“ Neural network architecture:")
print(f"  Input dimension: {input_dim}")
print(f"  Hidden layers: [512, 256, 128, 64]")
print(f"  Output: 1 (binary classification)")
print(f"  Dropout: 0.3")
print(f"  Total parameters: ~{(input_dim*512 + 512*256 + 256*128 + 128*64 + 64*1):,}")


# ======================================================================
# STEP 3: TRAIN NEURAL NETWORK WITH CROSS-VALIDATION
# ======================================================================

print("\n[Step 3/4] Training neural network with 5-fold CV...")
print("="*70)

# Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nâœ“ Using device: {device}")

cv_scores_nn = []
test_predictions_nn = np.zeros(len(test_df))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"\n{'='*70}")
    print(f"Fold {fold+1}/5")
    print("="*70)
    
    X_fold_train = X_train_scaled[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_scaled[val_idx]
    y_fold_val = y_train[val_idx]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_fold_train)
    y_train_tensor = torch.FloatTensor(y_fold_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_fold_val)
    y_val_tensor = torch.FloatTensor(y_fold_val).reshape(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model_nn = BinaryClassificationNN(input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_nn.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )
    
    # Training loop
    n_epochs = 50
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model_nn.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model_nn(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model_nn.eval()
        with torch.no_grad():
            X_val_device = X_val_tensor.to(device)
            val_outputs = model_nn(X_val_device).cpu().numpy()
            val_auc = roc_auc_score(y_fold_val, val_outputs)
        
        scheduler.step(val_auc)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val AUC: {val_auc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model weights
            best_model_state = model_nn.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model_nn.load_state_dict(best_model_state)
    
    # Final validation AUC
    model_nn.eval()
    with torch.no_grad():
        X_val_device = X_val_tensor.to(device)
        val_pred = model_nn(X_val_device).cpu().numpy().flatten()
        fold_auc = roc_auc_score(y_fold_val, val_pred)
    
    cv_scores_nn.append(fold_auc)
    print(f"\n  âœ“ Fold {fold+1} Best AUC: {fold_auc:.4f}")
    
    # Test predictions
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    with torch.no_grad():
        test_pred = model_nn(X_test_tensor).cpu().numpy().flatten()
        test_predictions_nn += test_pred / 5
    
    # Clear GPU memory
    del model_nn, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
    torch.cuda.empty_cache()

mean_cv_nn = np.mean(cv_scores_nn)
std_cv_nn = np.std(cv_scores_nn)

print(f"\n{'='*70}")
print("NEURAL NETWORK RESULTS:")
print("="*70)
print(f"  CV Fold AUCs: {[f'{s:.4f}' for s in cv_scores_nn]}")
print(f"  Mean CV: {mean_cv_nn:.4f} Â± {std_cv_nn:.4f}")
print("="*70)


# ======================================================================
# STEP 4: ENSEMBLE WITH XGB, LGB, AND NN
# ======================================================================

print("\n[Step 4/4] Creating 3-model ensemble (XGB + LGB + NN)...")
print("="*70)

# Test different ensemble weights
ensemble_configs = [
    (0.33, 0.33, 0.34, "Equal weight"),
    (0.4, 0.3, 0.3, "40-30-30 XGB focus"),
    (0.35, 0.35, 0.3, "35-35-30 Boost focus"),
    (0.3, 0.3, 0.4, "30-30-40 NN focus"),
    (0.5, 0.25, 0.25, "50-25-25 XGB heavy"),
]

best_ensemble_config = None
best_ensemble_score = 0

print("\nTesting ensemble configurations:")
print("-"*70)

# The CV scores for XGB (0.8368) and LGB (0.8351) are hardcoded 
# based on your previous runs.
XGB_SCORE = 0.8368
LGB_SCORE = 0.8351

for w_xgb, w_lgb, w_nn, name in ensemble_configs:
    # Estimate performance (weighted average of CV scores)
    estimated_auc = w_xgb * XGB_SCORE + w_lgb * LGB_SCORE + w_nn * mean_cv_nn
    
    print(f"  {name:20s}: Estimated ~{estimated_auc:.4f}")
    
    if estimated_auc > best_ensemble_score:
        best_ensemble_score = estimated_auc
        best_ensemble_config = (w_xgb, w_lgb, w_nn, name)

w_xgb, w_lgb, w_nn, best_name = best_ensemble_config

# Create best ensemble
test_predictions_3model = (
    w_xgb * test_predictions_enhanced + 
    w_lgb * test_predictions_lgb + 
    w_nn * test_predictions_nn
)

print(f"\nâœ“ Best ensemble: {best_name}")
print(f"  Weights: XGB={w_xgb:.2f}, LGB={w_lgb:.2f}, NN={w_nn:.2f}")
print(f"  Estimated AUC: {best_ensemble_score:.4f}")


# ======================================================================
# META-STACKING ON XGB + LGB ENSEMBLE (BEST APPROACH)
# ======================================================================

print("\n" + "="*70)
print("META-STACKING ON XGB + LGB ENSEMBLE")
print("="*70)

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ======================================================================
# STEP 1: GENERATE OUT-OF-FOLD PREDICTIONS
# ======================================================================

print("\n[Step 1/3] Generating out-of-fold predictions...")
print("-"*70)

# Mocking missing variables for context
# train_df, test_df, X_train_combined, y_train, X_test_combined 
# are assumed to be loaded.

# Initialize OOF arrays
oof_predictions_xgb = np.zeros(len(train_df))
oof_predictions_lgb = np.zeros(len(train_df))

# Initialize test prediction arrays
test_predictions_xgb_final = np.zeros(len(test_df))
test_predictions_lgb_final = np.zeros(len(test_df))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTraining XGBoost + LightGBM for OOF predictions...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train)):
    print(f"\nFold {fold+1}/5:")
    
    X_fold_train = X_train_combined[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_combined[val_idx]
    y_fold_val = y_train[val_idx]
    
    # ========== XGBoost ==========
    params_xgb = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 3, 'gamma': 0.1,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'seed': 42,
        'tree_method': 'gpu_hist', 'gpu_id': 0
    }
    
    dtrain_xgb = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dval_xgb = xgb.DMatrix(X_fold_val, label=y_fold_val)
    
    model_xgb = xgb.train(
        params_xgb, dtrain_xgb, num_boost_round=1000,
        evals=[(dtrain_xgb, 'train'), (dval_xgb, 'val')],
        early_stopping_rounds=100, verbose_eval=False
    )
    
    # OOF predictions
    oof_predictions_xgb[val_idx] = model_xgb.predict(dval_xgb)
    
    # Test predictions
    dtest_xgb = xgb.DMatrix(X_test_combined)
    test_predictions_xgb_final += model_xgb.predict(dtest_xgb) / 5
    
    # ========== LightGBM ==========
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 7,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
        'seed': 42, 'verbose': -1
    }
    
    dtrain_lgb = lgb.Dataset(X_fold_train, label=y_fold_train)
    dval_lgb = lgb.Dataset(X_fold_val, label=y_fold_val, reference=dtrain_lgb)
    
    model_lgb = lgb.train(
        lgb_params, dtrain_lgb, num_boost_round=1000,
        valid_sets=[dtrain_lgb, dval_lgb],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    # OOF predictions
    oof_predictions_lgb[val_idx] = model_lgb.predict(X_fold_val)
    
    # Test predictions
    test_predictions_lgb_final += model_lgb.predict(X_test_combined) / 5
    
    # Fold performance
    xgb_auc = roc_auc_score(y_fold_val, oof_predictions_xgb[val_idx])
    lgb_auc = roc_auc_score(y_fold_val, oof_predictions_lgb[val_idx])
    
    print(f"  XGBoost AUC: {xgb_auc:.4f}")
    print(f"  LightGBM AUC: {lgb_auc:.4f}")

# Overall OOF AUC
oof_auc_xgb = roc_auc_score(y_train, oof_predictions_xgb)
oof_auc_lgb = roc_auc_score(y_train, oof_predictions_lgb)

# Simple ensemble baseline (using the 60-40 weights from your previous script)
oof_ensemble_simple = 0.6 * oof_predictions_xgb + 0.4 * oof_predictions_lgb
oof_auc_simple = roc_auc_score(y_train, oof_ensemble_simple)

print(f"\n{'='*70}")
print("OOF RESULTS:")
print("="*70)
print(f"  XGBoost OOF:     {oof_auc_xgb:.4f}")
print(f"  LightGBM OOF:    {oof_auc_lgb:.4f}")
print(f"  Simple Ensemble: {oof_auc_simple:.4f}")
print("="*70)


# ======================================================================
# STEP 2: CREATE META-FEATURES
# ======================================================================

print("\n[Step 2/3] Creating meta-features...")
print("-"*70)

def create_meta_features_advanced(pred_xgb, pred_lgb, df):
    """
    Create advanced meta-features from base model predictions.
    """
    meta_features = pd.DataFrame()
    
    # Base predictions
    meta_features['xgb'] = pred_xgb
    meta_features['lgb'] = pred_lgb
    
    # Ensemble features
    meta_features['avg'] = (pred_xgb + pred_lgb) / 2
    meta_features['max'] = np.maximum(pred_xgb, pred_lgb)
    meta_features['min'] = np.minimum(pred_xgb, pred_lgb)
    meta_features['diff'] = np.abs(pred_xgb - pred_lgb)
    meta_features['product'] = pred_xgb * pred_lgb
    
    # Polynomial features
    meta_features['xgb_squared'] = pred_xgb ** 2
    meta_features['lgb_squared'] = pred_lgb ** 2
    meta_features['avg_squared'] = meta_features['avg'] ** 2
    
    # Confidence features (agreement)
    meta_features['agreement_high'] = ((pred_xgb > 0.7) & (pred_lgb > 0.7)).astype(int)
    meta_features['agreement_low'] = ((pred_xgb < 0.3) & (pred_lgb < 0.3)).astype(int)
    meta_features['disagreement'] = (np.abs(pred_xgb - pred_lgb) > 0.3).astype(int)
    
    # Interaction with text features (assuming 'body' column exists)
    meta_features['avg_x_length'] = meta_features['avg'] * df['body'].str.len()
    meta_features['diff_x_length'] = meta_features['diff'] * df['body'].str.len()
    
    return meta_features.values

# Create meta-features
X_meta_train = create_meta_features_advanced(
    oof_predictions_xgb, 
    oof_predictions_lgb, 
    train_df
)

X_meta_test = create_meta_features_advanced(
    test_predictions_xgb_final,
    test_predictions_lgb_final,
    test_df
)

y_meta_train = y_train

print(f"âœ“ Meta-features created:")
print(f"  Train: {X_meta_train.shape}")
print(f"  Test: {X_meta_test.shape}")


# ======================================================================
# STEP 3: TRAIN META-MODELS
# ======================================================================

print("\n[Step 3/3] Training meta-models...")
print("="*70)

# We'll try multiple meta-models and ensemble them
meta_predictions_oof = {}
meta_predictions_test = {}

# ---------- Meta-Model 1: Logistic Regression ----------
print("\n[Meta-Model 1: Logistic Regression]")

oof_lr = np.zeros(len(train_df))
test_lr = np.zeros(len(test_df))
cv_scores_lr = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_train, y_meta_train)):
    X_fold_train = X_meta_train[train_idx]
    y_fold_train = y_meta_train[train_idx]
    X_fold_val = X_meta_train[val_idx]
    y_fold_val = y_meta_train[val_idx]
    
    scaler = StandardScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = scaler.transform(X_fold_val)
    
    meta_lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0,
        class_weight='balanced'
    )
    
    meta_lr.fit(X_fold_train_scaled, y_fold_train)
    
    oof_lr[val_idx] = meta_lr.predict_proba(X_fold_val_scaled)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, oof_lr[val_idx])
    cv_scores_lr.append(fold_auc)
    
    X_meta_test_scaled = scaler.transform(X_meta_test)
    test_lr += meta_lr.predict_proba(X_meta_test_scaled)[:, 1] / 5

auc_lr = roc_auc_score(y_meta_train, oof_lr)
print(f"  CV AUC: {np.mean(cv_scores_lr):.4f} Â± {np.std(cv_scores_lr):.4f}")
print(f"  OOF AUC: {auc_lr:.4f}")

meta_predictions_oof['lr'] = oof_lr
meta_predictions_test['lr'] = test_lr

# ---------- Meta-Model 2: Ridge Regression ----------
print("\n[Meta-Model 2: Ridge Regression]")

oof_ridge = np.zeros(len(train_df))
test_ridge = np.zeros(len(test_df))
cv_scores_ridge = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_train, y_meta_train)):
    X_fold_train = X_meta_train[train_idx]
    y_fold_train = y_meta_train[train_idx]
    X_fold_val = X_meta_train[val_idx]
    y_fold_val = y_meta_train[val_idx]
    
    scaler = StandardScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = scaler.transform(X_fold_val)
    
    meta_ridge = Ridge(alpha=1.0, random_state=42)
    
    meta_ridge.fit(X_fold_train_scaled, y_fold_train)
    
    oof_ridge[val_idx] = meta_ridge.predict(X_fold_val_scaled)
    oof_ridge[val_idx] = np.clip(oof_ridge[val_idx], 0, 1)  # Clip to [0,1]
    
    fold_auc = roc_auc_score(y_fold_val, oof_ridge[val_idx])
    cv_scores_ridge.append(fold_auc)
    
    X_meta_test_scaled = scaler.transform(X_meta_test)
    test_ridge += np.clip(meta_ridge.predict(X_meta_test_scaled), 0, 1) / 5

auc_ridge = roc_auc_score(y_meta_train, oof_ridge)
print(f"  CV AUC: {np.mean(cv_scores_ridge):.4f} Â± {np.std(cv_scores_ridge):.4f}")
print(f"  OOF AUC: {auc_ridge:.4f}")

meta_predictions_oof['ridge'] = oof_ridge
meta_predictions_test['ridge'] = test_ridge


# ======================================================================
# META-MODEL COMPARISON
# ======================================================================

print("\n" + "="*70)
print("META-MODEL COMPARISON:")
print("="*70)

results = {
    'Simple Ensemble': oof_auc_simple,
    'Meta-LR': auc_lr,
    'Meta-Ridge': auc_ridge
}

for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    improvement = (score - oof_auc_simple) * 100
    print(f"  {name:20s}: {score:.4f} ({improvement:+.2f}pp)")

# Choose best
best_meta = max(results, key=results.get)
best_score = results[best_meta]

print(f"\nâœ… Best approach: {best_meta} ({best_score:.4f})")

# Select final predictions
if best_meta == 'Meta-LR':
    final_test_predictions = test_lr
elif best_meta == 'Meta-Ridge':
    final_test_predictions = test_ridge
else:
    # Fallback to simple ensemble if it's somehow the best
    final_test_predictions = 0.6 * test_predictions_xgb_final + 0.4 * test_predictions_lgb_final


# ======================================================================
# TRYING ISOTONIC CALIBRATION
# ======================================================================
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

print("="*70)
print("TRYING ISOTONIC CALIBRATION")
print("="*70)

# Generate OOF predictions for calibration
# These oof_... variables are from the meta-stacking script
oof_ensemble = 0.6 * oof_predictions_xgb + 0.4 * oof_predictions_lgb

# Per-rule calibration
calibrators = {}
calibrated_oof = np.zeros(len(train_df))

print("\nTraining calibrators for each 'rule'...")

for rule in train_df['rule'].unique():
    rule_mask = train_df['rule'] == rule
    
    if rule_mask.sum() >= 10:  # Need minimum samples to fit
        calibrator = IsotonicRegression(out_of_bounds='clip')
        
        # Fit on OOF predictions vs. true labels for this rule
        calibrator.fit(
            oof_ensemble[rule_mask],
            y_train[rule_mask]
        )
        
        calibrators[rule] = calibrator
        calibrated_oof[rule_mask] = calibrator.predict(oof_ensemble[rule_mask])
    else:
        # Not enough data, use original prediction
        calibrated_oof[rule_mask] = oof_ensemble[rule_mask]

print(f"âœ“ Trained {len(calibrators)} rule-specific calibrators.")


# Check improvement
auc_before = roc_auc_score(y_train, oof_ensemble)
auc_after = roc_auc_score(y_train, calibrated_oof)

print(f"\nCalibration Results:")
print(f"  Before: {auc_before:.4f}")
print(f"  After:  {auc_after:.4f}")
print(f"  Change: {(auc_after - auc_before)*100:+.2f}pp")

# ======================================================================
# DECIDE WHETHER TO USE CALIBRATED PREDICTIONS
# ======================================================================

if auc_after > auc_before:
    print("\nâœ… Calibration helps! Applying to test set...")
    
    # (Code to apply to test set and submit will go here)

else:
    print("\nâš ï¸�  Calibration doesn't help - use original ensemble or meta-model")
    print("="*70)


if auc_after > auc_before:
    
    # ==================================================================
    # STEP 1: APPLY CALIBRATION TO TEST SET
    # ==================================================================
    
    # Create the test ensemble using the same weights and base models
    # Note: Using the test predictions from the OOF generation step
    test_ensemble = 0.6 * test_predictions_xgb_final + 0.4 * test_predictions_lgb_final
    test_calibrated = np.zeros(len(test_df))
    
    print("Applying rule-specific calibration to test data...")
    
    for idx, row in test_df.iterrows():
        rule = row['rule']
        if rule in calibrators:
            # Apply the specific calibrator for this rule
            test_calibrated[idx] = calibrators[rule].predict([test_ensemble[idx]])[0]
        else:
            # Rule not seen or too rare, use original prediction
            test_calibrated[idx] = test_ensemble[idx]
    
    print("âœ“ Test set calibrated.")

    # (Submission code will follow)

# (The else block from Section 2 handles the other case)


# ======================================================================
# ADVANCED FEATURE ENGINEERING
# ======================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

print("="*70)
print("ADVANCED FEATURE ENGINEERING")
print("="*70)

def create_advanced_features(df):
    """Create sophisticated text and interaction features."""
    
    features = pd.DataFrame()
    
    # Mocking missing variables for context
    # train_df, test_df, X_train_combined, y_train, X_test_combined
    # are assumed to be loaded.
    
    # ========== Text Pattern Features ==========
    print("\n[1/6] Text pattern features...")
    
    # Toxicity indicators
    features['caps_words'] = df['body'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper()))
    features['caps_ratio'] = df['body'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
    
    # Punctuation patterns
    features['exclamation_ratio'] = df['body'].str.count('!') / df['body'].str.len()
    features['question_ratio'] = df['body'].str.count(r'\?') / df['body'].str.len()
    features['ellipsis_count'] = df['body'].str.count(r'\.\.+')
    
    # Profanity and intensity markers
    features['repeated_chars'] = df['body'].apply(lambda x: sum(1 for i in range(len(str(x))-2) if str(x)[i] == str(x)[i+1] == str(x)[i+2]))
    features['repeated_punctuation'] = df['body'].str.count(r'[!?]{2,}')
    
    # ========== Semantic Features ==========
    print("[2/6] Semantic features...")
    
    # Word overlap with examples
    def jaccard_similarity(text1, text2):
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        if len(words1) == 0 or len(words2) == 0:
            return 0
        return len(words1 & words2) / len(words1 | words2)
    
    features['similarity_pos1'] = df.apply(
        lambda row: jaccard_similarity(row['body'], row['positive_example_1']), axis=1
    )
    features['similarity_pos2'] = df.apply(
        lambda row: jaccard_similarity(row['body'], row['positive_example_2']), axis=1
    )
    features['similarity_neg1'] = df.apply(
        lambda row: jaccard_similarity(row['body'], row['negative_example_1']), axis=1
    )
    features['similarity_neg2'] = df.apply(
        lambda row: jaccard_similarity(row['body'], row['negative_example_2']), axis=1
    )
    
    features['max_pos_similarity'] = features[['similarity_pos1', 'similarity_pos2']].max(axis=1)
    features['max_neg_similarity'] = features[['similarity_neg1', 'similarity_neg2']].max(axis=1)
    features['similarity_diff'] = features['max_pos_similarity'] - features['max_neg_similarity']
    
    # ========== Structural Features ==========
    print("[3/6] Structural features...")
    
    # Length features
    features['body_length'] = df['body'].str.len()
    features['body_words'] = df['body'].str.split().str.len()
    features['avg_word_length'] = features['body_length'] / features['body_words'].replace(0, 1)
    features['rule_length'] = df['rule'].str.len()
    features['rule_words'] = df['rule'].str.split().str.len()
    
    # Sentence structure
    features['sentence_count'] = df['body'].str.count(r'[.!?]+') + 1
    features['avg_sentence_length'] = features['body_words'] / features['sentence_count']
    
    # ========== URL and Link Features ==========
    print("[4/6] URL and link features...")
    
    features['has_url'] = df['body'].str.contains(r'http[s]?://|www\.', case=False, regex=True).astype(int)
    features['url_count'] = df['body'].str.count(r'http[s]?://|www\.')
    features['has_markdown_link'] = df['body'].str.contains(r'\[.*\]\(.*\)', regex=True).astype(int)
    features['markdown_link_count'] = df['body'].str.count(r'\[.*\]\(.*\)')
    
    # ========== Special Character Features ==========
    print("[5/6] Special character features...")
    
    features['special_char_ratio'] = df['body'].apply(
        lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace()) / max(len(str(x)), 1)
    )
    features['digit_ratio'] = df['body'].apply(lambda x: sum(1 for c in str(x) if c.isdigit()) / max(len(str(x)), 1))
    features['emoji_count'] = df['body'].apply(lambda x: sum(1 for c in str(x) if ord(c) > 127))
    
    # ========== Subreddit-Specific Features ==========
    print("[6/6] Subreddit features...")
    
    # Encode subreddit
    le_subreddit = LabelEncoder()
    features['subreddit_encoded'] = le_subreddit.fit_transform(df['subreddit'])
    
    # Subreddit statistics
    subreddit_stats = df.groupby('subreddit').size()
    features['subreddit_frequency'] = df['subreddit'].map(subreddit_stats)
    
    print("\nâœ“ Advanced features created")
    
    return features.fillna(0)


# Create advanced features
print("\nCreating advanced features for train...")
train_advanced = create_advanced_features(train_df)

print("\nCreating advanced features for test...")
test_advanced = create_advanced_features(test_df)

print(f"\nâœ“ Advanced features shape: {train_advanced.shape}")

# ======================================================================
# COMBINE WITH EXISTING FEATURES
# ======================================================================

print("\n[Combining all features...]")
print("-"*70)

# Convert to sparse
train_advanced_sparse = csr_matrix(train_advanced.values.astype(np.float32))
test_advanced_sparse = csr_matrix(test_advanced.values.astype(np.float32))

# Combine: Embeddings + Text Features + TF-IDF + Advanced Features
X_train_super = hstack([
    X_train_combined,  # Your existing features
    train_advanced_sparse
])

X_test_super = hstack([
    X_test_combined,
    test_advanced_sparse
])

print(f"\nâœ“ Super combined features:")
print(f"  Train: {X_train_super.shape}")
print(f"  Test: {X_test_super.shape}")


# ======================================================================
# RETRAIN XGB + LGB WITH SUPER FEATURES
# ======================================================================

print("\n[Training with super features...]")
print("="*70)

oof_xgb_super = np.zeros(len(train_df))
oof_lgb_super = np.zeros(len(train_df))
test_xgb_super = np.zeros(len(test_df))
test_lgb_super = np.zeros(len(test_df))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_super, y_train)):
    print(f"\nFold {fold+1}/5:")
    
    X_fold_train = X_train_super[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_super[val_idx]
    y_fold_val = y_train[val_idx]
    
    # XGBoost
    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
    
    params_xgb = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 7,
        'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 3, 'gamma': 0.1, 'reg_alpha': 0.1,
        'reg_lambda': 1.0, 'seed': 42, 'tree_method': 'gpu_hist', 'gpu_id': 0
    }
    
    model_xgb = xgb.train(
        params_xgb, dtrain, num_boost_round=1000,
        evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=False
    )
    
    oof_xgb_super[val_idx] = model_xgb.predict(dval)
    test_xgb_super += model_xgb.predict(xgb.DMatrix(X_test_super)) / 5
    
    # LightGBM
    dtrain_lgb = lgb.Dataset(X_fold_train, label=y_fold_train)
    dval_lgb = lgb.Dataset(X_fold_val, label=y_fold_val, reference=dtrain_lgb)
    
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.03,
        'num_leaves': 31, 'max_depth': 7, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'device': 'gpu',
        'seed': 42, 'verbose': -1
    }

    model_lgb = lgb.train(
        lgb_params,
        dtrain_lgb, num_boost_round=1000,
        valid_sets=[dval_lgb], 
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    oof_lgb_super[val_idx] = model_lgb.predict(X_fold_val)
    test_lgb_super += model_lgb.predict(X_test_super) / 5
    
    xgb_auc = roc_auc_score(y_fold_val, oof_xgb_super[val_idx])
    lgb_auc = roc_auc_score(y_fold_val, oof_lgb_super[val_idx])
    print(f"  XGB: {xgb_auc:.4f}, LGB: {lgb_auc:.4f}")


# Check improvement
auc_xgb_super = roc_auc_score(y_train, oof_xgb_super)
auc_lgb_super = roc_auc_score(y_train, oof_lgb_super)
ensemble_super = 0.6 * oof_xgb_super + 0.4 * oof_lgb_super
auc_ensemble_super = roc_auc_score(y_train, ensemble_super)

# Hardcoded previous scores for comparison
PREV_XGB = 0.8368
PREV_LGB = 0.8351
PREV_ENSEMBLE = 0.8373

print(f"\n{'='*70}")
print("SUPER FEATURES RESULTS:")
print("="*70)
print(f"  Previous XGB:        {PREV_XGB:.4f}")
print(f"  Super XGB:           {auc_xgb_super:.4f} ({(auc_xgb_super-PREV_XGB)*100:+.2f}pp)")
print(f"  Previous LGB:        {PREV_LGB:.4f}")
print(f"  Super LGB:           {auc_lgb_super:.4f} ({(auc_lgb_super-PREV_LGB)*100:+.2f}pp)")
print(f"  Previous Ensemble:   {PREV_ENSEMBLE:.4f}")
print(f"  Super Ensemble:      {auc_ensemble_super:.4f} ({(auc_ensemble_super-PREV_ENSEMBLE)*100:+.2f}pp)")

if auc_ensemble_super > PREV_ENSEMBLE:
    print(f"\nâœ… Super features improve performance!")
    
    # (Calibration and Submission logic follows)

else:
    print(f"\nâš ï¸�  Super features don't improve - stick with current best")
    print("="*70)


if auc_ensemble_super > PREV_ENSEMBLE:
    
    # ==================================================================
    # APPLY CALIBRATION
    # ==================================================================
    print("\n[Applying calibration to super ensemble...]")
    
    # Train Calibrators
    calibrators_super = {}
    for rule in train_df['rule'].unique():
        rule_mask = train_df['rule'] == rule
        if rule_mask.sum() >= 10:
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(ensemble_super[rule_mask], y_train[rule_mask])
            calibrators_super[rule] = calibrator
    
    # Calibrate OOF predictions (to get final OOF score)
    ensemble_super_cal = np.zeros(len(train_df))
    for idx, row in train_df.iterrows():
        rule = row['rule']
        if rule in calibrators_super:
            ensemble_super_cal[idx] = calibrators_super[rule].predict([ensemble_super[idx]])[0]
        else:
            ensemble_super_cal[idx] = ensemble_super[idx]
    
    auc_super_cal = roc_auc_score(y_train, ensemble_super_cal)
    
    print(f"  Before calibration: {auc_ensemble_super:.4f}")
    print(f"  After calibration:  {auc_super_cal:.4f} ({(auc_super_cal-auc_ensemble_super)*100:+.2f}pp)")
    
    # Apply to Test Set
    print("\n[Applying calibration to super test set...]")
    test_ensemble_super = 0.6 * test_xgb_super + 0.4 * test_lgb_super
    test_super_cal = np.zeros(len(test_df))
    
    for idx, row in test_df.iterrows():
        rule = row['rule']
        if rule in calibrators_super:
            test_super_cal[idx] = calibrators_super[rule].predict([test_ensemble_super[idx]])[0]
        else:
            test_super_cal[idx] = test_ensemble_super[idx]
            
    print("âœ“ Calibration applied.")

    # (Submission logic follows)


# ======================================================================
# ADD CATBOOST TO ENSEMBLE
# ======================================================================

print("="*70)
print("ADDING CATBOOST TO ENSEMBLE")
print("="*70)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

try:
    import catboost as cb
    print("âœ“ CatBoost available")
except:
    print("Installing CatBoost...")
    !pip install catboost -q
    import catboost as cb

# ======================================================================
# TRAIN CATBOOST WITH CV
# ======================================================================

print("\n[Training CatBoost with 5-fold CV...]")
print("-"*70)

# Mocking missing variables for context
# train_df, test_df, X_train_super, y_train, X_test_super
# oof_xgb_super, oof_lgb_super, test_xgb_super, test_lgb_super
# are all assumed to be loaded from previous steps.

oof_cat = np.zeros(len(train_df))
test_cat = np.zeros(len(test_df))
cv_scores_cat = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_super, y_train)):
    print(f"\nFold {fold+1}/5:")
    
    X_fold_train = X_train_super[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_super[val_idx]
    y_fold_val = y_train[val_idx]
    
    # CatBoost parameters
    model_cat = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=7,
        loss_function='Logloss',
        eval_metric='AUC',
        early_stopping_rounds=100,
        task_type='GPU',
        devices='0',
        random_seed=42,
        verbose=False
    )
    
    # Train
    model_cat.fit(
        X_fold_train, y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        use_best_model=True
    )
    
    # OOF predictions
    oof_cat[val_idx] = model_cat.predict_proba(X_fold_val)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, oof_cat[val_idx])
    cv_scores_cat.append(fold_auc)
    
    print(f"  CatBoost AUC: {fold_auc:.4f}")
    
    # Test predictions
    test_cat += model_cat.predict_proba(X_test_super)[:, 1] / 5

auc_cat = roc_auc_score(y_train, oof_cat)

print(f"\n{'='*70}")
print("CATBOOST RESULTS:")
print("="*70)
print(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores_cat]}")
print(f"  Mean CV: {np.mean(cv_scores_cat):.4f} Â± {np.std(cv_scores_cat):.4f}")
print(f"  OOF AUC: {auc_cat:.4f}")
print("="*70)


# ======================================================================
# CREATE 3-MODEL ENSEMBLE (XGB + LGB + CAT)
# ======================================================================

print("\n[Creating 3-model ensemble...]")
print("-"*70)

# Test different weight combinations
ensemble_configs = [
    (0.4, 0.3, 0.3, "40-30-30"),
    (0.5, 0.25, 0.25, "50-25-25 XGB heavy"),
    (0.33, 0.33, 0.34, "Equal weight"),
    (0.35, 0.35, 0.3, "35-35-30 Boost focus"),
    (0.3, 0.3, 0.4, "30-30-40 CAT focus"),
]

best_config = None
best_score = 0

print("\nTesting ensemble weights:")
for w_xgb, w_lgb, w_cat, name in ensemble_configs:
    oof_ensemble = w_xgb * oof_xgb_super + w_lgb * oof_lgb_super + w_cat * oof_cat
    auc = roc_auc_score(y_train, oof_ensemble)
    
    print(f"  {name:25s}: {auc:.4f}")
    
    if auc > best_score:
        best_score = auc
        best_config = (w_xgb, w_lgb, w_cat, name)

w_xgb, w_lgb, w_cat, best_name = best_config

print(f"\nâœ“ Best weights: {best_name}")
print(f"  XGB={w_xgb:.2f}, LGB={w_lgb:.2f}, CAT={w_cat:.2f}")
print(f"  OOF AUC: {best_score:.4f}")

# Create 3-model ensemble
oof_3model = w_xgb * oof_xgb_super + w_lgb * oof_lgb_super + w_cat * oof_cat
test_3model = w_xgb * test_xgb_super + w_lgb * test_lgb_super + w_cat * test_cat


# ======================================================================
# ADVANCED CALIBRATION: PER-SUBREDDIT-RULE + MULTIPLE METHODS
# ======================================================================

print("="*70)
print("ADVANCED CALIBRATION STRATEGIES")
print("="*70)

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Mocking missing variables for context
# oof_3model, test_3model, train_df, test_df, y_train are all assumed to
# be loaded from the previous (CatBoost) step.

# ======================================================================
# STRATEGY 1: FINE-GRAINED PER-SUBREDDIT-RULE CALIBRATION
# ======================================================================

print("\n[Strategy 1: Per-Subreddit-Rule Calibration]")
print("-"*70)

def fine_grained_calibration(oof_preds, test_preds, train_df, test_df, y_train, min_samples=5):
    """
    Calibrate at the most granular level: subreddit + rule combination.
    Falls back to rule-only if insufficient samples.
    """
    
    calibrators = {}
    oof_calibrated = np.zeros(len(train_df))
    test_calibrated = np.zeros(len(test_df))
    
    # Statistics
    subreddit_rule_count = 0
    rule_only_count = 0
    no_calibration_count = 0
    
    # Get unique subreddit-rule combinations
    train_df_temp = train_df.copy()
    train_df_temp['oof_pred'] = oof_preds
    
    for subreddit in train_df_temp['subreddit'].unique():
        for rule in train_df_temp[train_df_temp['subreddit'] == subreddit]['rule'].unique():
            
            # Try subreddit + rule calibration
            mask_train = (train_df_temp['subreddit'] == subreddit) & (train_df_temp['rule'] == rule)
            
            key = (subreddit, rule)
            
            if mask_train.sum() >= min_samples:
                # Enough samples for subreddit-rule calibration
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(
                    oof_preds[mask_train],
                    y_train[mask_train]
                )
                calibrators[key] = calibrator
                oof_calibrated[mask_train] = calibrator.predict(oof_preds[mask_train])
                subreddit_rule_count += 1
                
            else:
                # Fall back to rule-only calibration
                mask_train_rule = (train_df_temp['rule'] == rule)
                
                if mask_train_rule.sum() >= min_samples:
                    key_rule = ('*', rule)  # Wildcard for subreddit
                    
                    if key_rule not in calibrators:
                        calibrator = IsotonicRegression(out_of_bounds='clip')
                        calibrator.fit(
                            oof_preds[mask_train_rule],
                            y_train[mask_train_rule]
                        )
                        calibrators[key_rule] = calibrator
                        rule_only_count += 1
                    
                    oof_calibrated[mask_train] = calibrators[key_rule].predict(oof_preds[mask_train])
                else:
                    # No calibration
                    oof_calibrated[mask_train] = oof_preds[mask_train]
                    no_calibration_count += 1
    
    # Apply to test set
    for idx, row in test_df.iterrows():
        subreddit = row['subreddit']
        rule = row['rule']
        
        key = (subreddit, rule)
        key_rule = ('*', rule)
        
        if key in calibrators:
            test_calibrated[idx] = calibrators[key].predict([test_preds[idx]])[0]
        elif key_rule in calibrators:
            test_calibrated[idx] = calibrators[key_rule].predict([test_preds[idx]])[0]
        else:
            test_calibrated[idx] = test_preds[idx]
    
    print(f"  Subreddit-Rule calibrators: {subreddit_rule_count}")
    print(f"  Rule-only calibrators: {rule_only_count}")
    print(f"  No calibration: {no_calibration_count}")
    
    return oof_calibrated, test_calibrated, calibrators

# Apply fine-grained calibration
oof_fine_grained, test_fine_grained, calibrators_fine = fine_grained_calibration(
    oof_3model, test_3model, train_df, test_df, y_train
)

auc_fine_grained = roc_auc_score(y_train, oof_fine_grained)

print(f"\nâœ“ Fine-grained calibration AUC: {auc_fine_grained:.4f}")


# ======================================================================
# STRATEGY 2: MULTIPLE CALIBRATION METHODS
# ======================================================================

print("\n[Strategy 2: Multiple Calibration Methods]")
print("-"*70)

def platt_scaling(oof_preds, test_preds, y_train):
    """
    Platt scaling: fit a logistic regression on predictions.
    """
    oof_preds_2d = oof_preds.reshape(-1, 1)
    test_preds_2d = test_preds.reshape(-1, 1)
    
    platt_model = LogisticRegression(random_state=42, max_iter=1000)
    platt_model.fit(oof_preds_2d, y_train)
    
    oof_calibrated = platt_model.predict_proba(oof_preds_2d)[:, 1]
    test_calibrated = platt_model.predict_proba(test_preds_2d)[:, 1]
    
    return oof_calibrated, test_calibrated

def beta_calibration(oof_preds, test_preds, y_train, bins=20):
    """
    Beta calibration: map predictions to empirical probabilities using binning.
    """
    oof_calibrated = np.zeros(len(oof_preds))
    test_calibrated = np.zeros(len(test_preds))
    
    # Create bins
    bin_edges = np.linspace(0, 1, bins + 1)
    
    for i in range(bins):
        bin_mask = (oof_preds >= bin_edges[i]) & (oof_preds < bin_edges[i + 1])
        
        if i == bins - 1:  # Last bin includes right edge
            bin_mask = (oof_preds >= bin_edges[i]) & (oof_preds <= bin_edges[i + 1])
        
        if bin_mask.sum() > 0:
            # Calculate empirical probability for this bin
            empirical_prob = y_train[bin_mask].mean()
            oof_calibrated[bin_mask] = empirical_prob
            
            # Apply to test
            test_bin_mask = (test_preds >= bin_edges[i]) & (test_preds < bin_edges[i + 1])
            if i == bins - 1:
                test_bin_mask = (test_preds >= bin_edges[i]) & (test_preds <= bin_edges[i + 1])
            test_calibrated[test_bin_mask] = empirical_prob
    
    # Handle any test predictions that didn't fall into a bin (e.g., if bin was empty)
    # This is a fallback to prevent 0s
    uncalibrated_mask = (test_calibrated == 0)
    test_calibrated[uncalibrated_mask] = test_preds[uncalibrated_mask]

    return oof_calibrated, test_calibrated

# Method 1: Isotonic Regression (per rule - simple)
print("\n[Method 1: Isotonic Regression (per-rule)]")
oof_isotonic = np.zeros(len(train_df))
test_isotonic = np.zeros(len(test_df))

calibrators_isotonic = {}
for rule in train_df['rule'].unique():
    rule_mask = train_df['rule'] == rule
    
    if rule_mask.sum() >= 10:
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(oof_3model[rule_mask], y_train[rule_mask])
        calibrators_isotonic[rule] = calibrator
        oof_isotonic[rule_mask] = calibrator.predict(oof_3model[rule_mask])
    else:
        oof_isotonic[rule_mask] = oof_3model[rule_mask]

# Apply to test
for idx, row in test_df.iterrows():
    rule = row['rule']
    if rule in calibrators_isotonic:
        test_isotonic[idx] = calibrators_isotonic[rule].predict([test_3model[idx]])[0]
    else:
        test_isotonic[idx] = test_3model[idx]

auc_isotonic = roc_auc_score(y_train, oof_isotonic)
print(f"  AUC: {auc_isotonic:.4f}")

# Method 2: Platt Scaling
print("\n[Method 2: Platt Scaling]")
oof_platt, test_platt = platt_scaling(oof_3model, test_3model, y_train)
auc_platt = roc_auc_score(y_train, oof_platt)
print(f"  AUC: {auc_platt:.4f}")

# Method 3: Beta Calibration
print("\n[Method 3: Beta Calibration (Binning)]")
oof_beta, test_beta = beta_calibration(oof_3model, test_3model, y_train, bins=20)
auc_beta = roc_auc_score(y_train, oof_beta)
print(f"  AUC: {auc_beta:.4f}")


# ======================================================================
# STRATEGY 3: BLEND CALIBRATION METHODS
# ======================================================================

print("\n[Strategy 3: Blending Calibration Methods]")
print("="*70)

# Collect all calibration methods
calibration_methods = {
    'Fine-grained': (oof_fine_grained, test_fine_grained, auc_fine_grained),
    'Isotonic': (oof_isotonic, test_isotonic, auc_isotonic),
    'Platt': (oof_platt, test_platt, auc_platt),
    'Beta': (oof_beta, test_beta, auc_beta),
}

# Display all methods
print("\nAll calibration methods:")
print("-"*70)
for name, (oof, test, auc) in sorted(calibration_methods.items(), key=lambda x: x[1][2], reverse=True):
    print(f"  {name:20s}: {auc:.4f}")

# Find optimal blend weights using optimization
def blend_objective(weights):
    """Objective function for blending calibration methods."""
    weights = weights / weights.sum()  # Normalize
    
    blended_oof = sum(w * oof for w, (oof, _, _) in zip(weights, calibration_methods.values()))
    auc = roc_auc_score(y_train, blended_oof)
    
    return -auc  # Negative for minimization

# Optimize blend weights
n_methods = len(calibration_methods)
initial_weights = np.ones(n_methods) / n_methods

result = minimize(
    blend_objective,
    x0=initial_weights,
    method='SLSQP',
    bounds=[(0, 1)] * n_methods,
    constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
)

optimal_weights = result.x
optimal_auc = -result.fun

print(f"\nâœ“ Optimal blend weights:")
for name, weight in zip(calibration_methods.keys(), optimal_weights):
    print(f"  {name:20s}: {weight:.3f}")

print(f"\nâœ“ Blended OOF AUC: {optimal_auc:.4f}")

# Create blended predictions
oof_blended = sum(w * oof for w, (oof, _, _) in zip(optimal_weights, calibration_methods.values()))
test_blended = sum(w * test for w, (_, test, _) in zip(optimal_weights, calibration_methods.values()))


# ======================================================================
# COMPARE ALL STRATEGIES
# ======================================================================

print("\n" + "="*70)
print("CALIBRATION STRATEGY COMPARISON")
print("="*70)

strategies = {
    'Uncalibrated 3-model': roc_auc_score(y_train, oof_3model),
    'Fine-grained (subreddit+rule)': auc_fine_grained,
    'Isotonic (per-rule)': auc_isotonic,
    'Platt Scaling': auc_platt,
    'Beta Calibration': auc_beta,
    'Optimal Blend': optimal_auc
}

best_strategy = max(strategies, key=strategies.get)
best_auc = strategies[best_strategy]

for name, auc in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
    improvement = (auc - strategies['Uncalibrated 3-model']) * 100
    marker = "âœ…" if name == best_strategy else "  "
    print(f"{marker} {name:35s}: {auc:.4f} ({improvement:+.2f}pp)")

# Map strategy name to the corresponding test predictions
strategy_predictions = {
    'Uncalibrated 3-model': test_3model,
    'Fine-grained (subreddit+rule)': test_fine_grained,
    'Isotonic (per-rule)': test_isotonic,
    'Platt Scaling': test_platt,
    'Beta Calibration': test_beta,
    'Optimal Blend': test_blended
}

# Get the test predictions for the best strategy
best_test_preds = strategy_predictions[best_strategy]


# ======================================================================
# FINAL VALIDATION & SANITY CHECKS
# ======================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

print("="*70)
print("FINAL VALIDATION & SANITY CHECKS")
print("="*70)

# Mocking missing variables for context
# test_fine_grained, test_blended, test_isotonic, train_df, test_df, y_train,
# oof_fine_grained, auc_fine_grained are assumed to be loaded from previous steps.

# Load the best predictions
best_test_preds = test_fine_grained

print("\n[1] Prediction Statistics:")
print("-"*70)
print(f"  Count:    {len(best_test_preds)}")
print(f"  Mean:     {best_test_preds.mean():.4f}")
print(f"  Median:   {np.median(best_test_preds):.4f}")
print(f"  Std:      {best_test_preds.std():.4f}")
print(f"  Min:      {best_test_preds.min():.4f}")
print(f"  Max:      {best_test_preds.max():.4f}")

# Check for any issues
print("\n[2] Quality Checks:")
print("-"*70)

# Check for NaNs
nan_count = np.isnan(best_test_preds).sum()
print(f"  NaN values: {nan_count}")
if nan_count > 0:
    print("  âš ï¸�  WARNING: Found NaN values! Filling with 0.5...")
    best_test_preds = np.nan_to_num(best_test_preds, nan=0.5)

# Check range
out_of_range = ((best_test_preds < 0) | (best_test_preds > 1)).sum()
print(f"  Out of [0,1] range: {out_of_range}")
if out_of_range > 0:
    print("  âš ï¸�  WARNING: Clipping to [0,1]...")
    best_test_preds = np.clip(best_test_preds, 0, 1)

# Check for extreme concentrations
very_low = (best_test_preds < 0.1).sum()
very_high = (best_test_preds > 0.9).sum()
print(f"  Very confident (>0.9 or <0.1): {very_low + very_high}/{len(best_test_preds)} ({(very_low+very_high)/len(best_test_preds)*100:.1f}%)")


# Distribution analysis
print("\n[3] Prediction Distribution:")
print("-"*70)
bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

for low, high in bins:
    count = np.sum((best_test_preds >= low) & (best_test_preds < high))
    # Handle last bin inclusion
    if high == 1.0:
        count = np.sum((best_test_preds >= low) & (best_test_preds <= high))
        
    pct = count/len(best_test_preds)*100
    bar = "â–ˆ" * int(pct / 3)
    print(f"  [{low:.1f}-{high:.1f}): {count:4d} ({pct:5.1f}%) {bar}")

# Compare train distribution with test
print("\n[4] Train vs Test Distribution:")
print("-"*70)

train_mean = y_train.mean()
test_mean = best_test_preds.mean()

print(f"  Training violation rate: {train_mean:.4f}")
print(f"  Test predicted rate:     {test_mean:.4f}")
print(f"  Difference:              {abs(train_mean - test_mean):.4f}")

if abs(train_mean - test_mean) > 0.1:
    print("  âš ï¸�  WARNING: Large distribution shift!")
else:
    print("  âœ“ Distributions are reasonably similar")


# ======================================================================
# SAVE SUBMISSION.CSV DIRECTLY TO KAGGLE OUTPUT
# ======================================================================

import pandas as pd
import os

print("="*70)
print("CREATING SUBMISSION FOR KAGGLE")
print("="*70)

# Define the output path (Kaggle working directory)
output_path = '/kaggle/working/submission.csv'

# Create submission DataFrame
# Use your best predictions (adjust variable name if different)
submission = pd.DataFrame({
    'row_id': test_df.index,
    'rule_violation': test_fine_grained  # Change this to your best predictions variable
})

# Save directly to Kaggle output
submission.to_csv(output_path, index=False)

print(f"\nâœ… Submission saved to: {output_path}")

# Verify
sub_check = pd.read_csv(output_path)

print(f"\nğŸ“Š Submission Details:")
print(f"   Rows: {len(sub_check)}")
print(f"   Columns: {list(sub_check.columns)}")
print(f"   Mean: {sub_check['rule_violation'].mean():.4f}")
print(f"   Std:  {sub_check['rule_violation'].std():.4f}")

print(f"\n   First 5 rows:")
print(sub_check.head().to_string(index=False))

# Validation
checks = [
    ('Correct columns', list(sub_check.columns) == ['row_id', 'rule_violation']),
    ('Correct row count', len(sub_check) == len(test_df)),
    ('No NaN values', not sub_check['rule_violation'].isna().any()),
    ('Values in [0,1]', sub_check['rule_violation'].between(0, 1).all())
]

print(f"\nâœ… Validation:")
all_passed = True
for check_name, passed in checks:
    status = "âœ…" if passed else "â�Œ"
    print(f"   {status} {check_name}")
    if not passed:
        all_passed = False

if all_passed:
    print(f"\n{'='*70}")
    print("ğŸš€ READY TO SUBMIT!")
    print("="*70)
    print(f"\n  Click 'Save Version' â†’ 'Save & Run All'")
    print(f"  Then submit from Output tab")
else:
    print(f"\nâ�Œ Fix validation errors above!")

print("="*70)


# ======================================================================
# SAVE SUBMISSION.CSV DIRECTLY TO KAGGLE OUTPUT
# ======================================================================
import pandas as pd
import os
print("="*70)
print("CREATING SUBMISSION FOR KAGGLE")
print("="*70)

# Define the output path (Kaggle working directory)
output_path = '/kaggle/working/submission.csv'

# Create submission DataFrame
# IMPORTANT: Use test_df['row_id'] or test_df.index depending on your data structure
# If 'row_id' is a column in test_df, use test_df['row_id']
# If 'row_id' is the index, use test_df.index

# Check if 'row_id' exists as a column in test_df
if 'row_id' in test_df.columns:
    row_ids = test_df['row_id'].values
    print("âœ… Using 'row_id' column from test_df")
else:
    row_ids = test_df.index.values
    print("âœ… Using index from test_df")

submission = pd.DataFrame({
    'row_id': row_ids,
    'rule_violation': test_fine_grained  # Change this to your best predictions variable
})

# Save directly to Kaggle output
submission.to_csv(output_path, index=False)
print(f"\nâœ… Submission saved to: {output_path}")

# Verify
sub_check = pd.read_csv(output_path)
print(f"\nğŸ“Š Submission Details:")
print(f"   Rows: {len(sub_check)}")
print(f"   Columns: {list(sub_check.columns)}")
print(f"   Mean: {sub_check['rule_violation'].mean():.4f}")
print(f"   Std:  {sub_check['rule_violation'].std():.4f}")
print(f"\n   First 10 rows:")
print(sub_check.head(10).to_string(index=False))
print(f"\n   Row ID range: {sub_check['row_id'].min()} to {sub_check['row_id'].max()}")

# Validation
checks = [
    ('Correct columns', list(sub_check.columns) == ['row_id', 'rule_violation']),
    ('Correct row count', len(sub_check) == len(test_df)),
    ('No NaN values', not sub_check['rule_violation'].isna().any()),
    ('Values in [0,1]', sub_check['rule_violation'].between(0, 1).all()),
    ('Row IDs start correctly', sub_check['row_id'].min() >= 2029)  # Added check
]

print(f"\nâœ… Validation:")
all_passed = True
for check_name, passed in checks:
    status = "âœ…" if passed else "â�Œ"
    print(f"   {status} {check_name}")
    if not passed:
        all_passed = False

if all_passed:
    print(f"\n{'='*70}")
    print("ğŸš€ READY TO SUBMIT!")
    print("="*70)
    print(f"\n  Click 'Save Version' â†’ 'Save & Run All'")
    print(f"  Then submit from Output tab")
else:
    print(f"\nâ�Œ Fix validation errors above!")
    
print("="*70)


# ======================================================================
# DIAGNOSTIC CHECK FOR 0.9117 SCORE (FIXED)
# ======================================================================
import numpy as np
from sklearn.metrics import roc_auc_score

print("="*70)
print("VALIDATING 0.9117 SCORE - OVERFITTING CHECK")
print("="*70)

# 1. Check what predictions achieved 0.9117
print("\n1ï¸�âƒ£ SCORE BREAKDOWN:")
print(f"   Fine-grained calibrated score: 0.9117")
print(f"   Uncalibrated 3-model score: 0.8478")
print(f"   Difference: +{0.9117 - 0.8478:.4f} ({((0.9117 - 0.8478)/0.8478)*100:.2f}% improvement)")

# 2. Check dataframe columns first
print("\n2ï¸�âƒ£ DATAFRAME INFO:")
if 'train_df' in locals():
    print(f"   Train shape: {train_df.shape}")
    print(f"   Columns: {list(train_df.columns)}")
else:
    print("   âš ï¸�  train_df not found")

# 3. Calibration granularity check (with correct column names)
print("\n3ï¸�âƒ£ CALIBRATION GRANULARITY:")
if 'train_df' in locals():
    # Check what columns exist for grouping
    possible_cols = ['subreddit', 'rule', 'rule_id']
    group_cols = []
    
    for col in possible_cols:
        if col in train_df.columns:
            group_cols.append(col)
            print(f"   âœ“ Found column: {col}")
    
    if len(group_cols) >= 2:
        # Use first two available grouping columns
        n_groups = train_df.groupby(group_cols[:2]).size()
        print(f"\n   Grouping by: {group_cols[:2]}")
        print(f"   Total groups: {len(n_groups)}")
        print(f"   Min samples per group: {n_groups.min()}")
        print(f"   Max samples per group: {n_groups.max()}")
        print(f"   Median samples per group: {n_groups.median():.0f}")
        print(f"\n   Groups with <10 samples: {(n_groups < 10).sum()}")
        print(f"   Groups with <5 samples: {(n_groups < 5).sum()}")
        print(f"   Groups with <3 samples: {(n_groups < 3).sum()}")
        
        print(f"\n   Smallest 10 groups:")
        print(n_groups.nsmallest(10))
    else:
        print(f"   âš ï¸�  Not enough grouping columns found")

# 4. Prediction distribution check
print("\n4ï¸�âƒ£ PREDICTION DISTRIBUTION:")
if 'oof_calibrated_fine' in locals():
    print(f"   Shape: {oof_calibrated_fine.shape}")
    print(f"   Min: {oof_calibrated_fine.min():.6f}")
    print(f"   Max: {oof_calibrated_fine.max():.6f}")
    print(f"   Mean: {oof_calibrated_fine.mean():.4f}")
    print(f"   Std: {oof_calibrated_fine.std():.4f}")
    
    # Check for suspicious patterns
    n_zeros = (oof_calibrated_fine == 0).sum()
    n_ones = (oof_calibrated_fine == 1).sum()
    n_near_zero = (oof_calibrated_fine < 0.01).sum()
    n_near_one = (oof_calibrated_fine > 0.99).sum()
    
    print(f"\n   Extreme predictions:")
    print(f"   Exactly 0: {n_zeros} ({n_zeros/len(oof_calibrated_fine)*100:.1f}%)")
    print(f"   Exactly 1: {n_ones} ({n_ones/len(oof_calibrated_fine)*100:.1f}%)")
    print(f"   Near 0 (<0.01): {n_near_zero} ({n_near_zero/len(oof_calibrated_fine)*100:.1f}%)")
    print(f"   Near 1 (>0.99): {n_near_one} ({n_near_one/len(oof_calibrated_fine)*100:.1f}%)")
    
    if n_zeros > 100 or n_ones > 100:
        print("\n   âš ï¸�  WARNING: Many extreme predictions!")
        print("   This suggests overfitting in calibration")
else:
    print("   âš ï¸�  oof_calibrated_fine not found")
    # Check what OOF variables exist
    oof_vars = [var for var in dir() if 'oof' in var.lower()]
    print(f"   Available OOF variables: {oof_vars[:10]}")

# 5. Compare with other calibration methods
print("\n5ï¸�âƒ£ CALIBRATION METHOD COMPARISON:")
scores = {
    'Fine-grained (subreddit+rule)': 0.9117,
    'Optimal Blend': 0.8771,
    'Isotonic (per-rule)': 0.8566,
    'Beta Calibration': 0.8496,
    'Uncalibrated 3-model': 0.8478,
}

for method, score in scores.items():
    diff = score - 0.8478
    print(f"   {method:35s}: {score:.4f} (+{diff:.4f})")

print("\n" + "="*70)
print("ğŸ�¯ CRITICAL QUESTION")
print("="*70)
print("\nHave you submitted to the leaderboard?")
print("   â€¢ If YES: What's your PUBLIC LB score?")
print("   â€¢ If NO: SUBMIT NOW to validate!")
print("\nIf LB score drops significantly (e.g., to 0.85),")
print("the 0.9117 is overfit. Use Optimal Blend (0.8771) instead.")
print("="*70)


# ======================================================================
# SAFE SUBMISSION - OPTIMAL BLEND (0.8771 OOF)
# ======================================================================
import pandas as pd

print("="*70)
print("CREATING SAFE SUBMISSION - OPTIMAL BLEND")
print("="*70)

output_path = '/kaggle/working/submission.csv'

# Check if row_id exists in test_df
if 'row_id' in test_df.columns:
    row_ids = test_df['row_id'].values
    print("âœ… Using 'row_id' column from test_df")
else:
    row_ids = test_df.index.values
    print("âœ… Using index from test_df")

# Use your Optimal Blend predictions
submission = pd.DataFrame({
    'row_id': row_ids,
    'rule_violation': test_blended  # Your optimal blend
})

# Save submission
submission.to_csv(output_path, index=False)
print(f"\nâœ… Submission saved to: {output_path}")

# Verify
sub_check = pd.read_csv(output_path)
print(f"\nğŸ“Š Submission Details:")
print(f"   Rows: {len(sub_check)}")
print(f"   Expected LB: ~0.87-0.88 (based on 0.8771 OOF)")
print(f"   Mean: {sub_check['rule_violation'].mean():.4f}")
print(f"   Std:  {sub_check['rule_violation'].std():.4f}")
print(f"\n   First 10 rows:")
print(sub_check.head(10).to_string(index=False))

print("\n" + "="*70)
print("ğŸš€ SUBMIT THIS FIRST - IT'S YOUR SAFE BASELINE")
print("="*70)

