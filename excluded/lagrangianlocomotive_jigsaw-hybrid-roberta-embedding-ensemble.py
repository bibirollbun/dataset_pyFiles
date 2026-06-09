# Pin versions (Kaggle env may vary; log for comparison)
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
import transformers
print(f'Transformers: {transformers.__version__}')
import sentence_transformers
print(f'Sentence Transformers: {sentence_transformers.__version__}')

# Log Kaggle env vars
import os
kaggle_vars = {k: v for k, v in os.environ.items() if k.startswith('KAGGLE_')}
print('Kaggle Env Vars:', kaggle_vars)

# Check and log dataset presence
datasets = [
    '/kaggle/input/jigsaw-agile-community-rules',
    '/kaggle/input/all-roberta-large-v1',
    '/kaggle/input/jigsaw-roberta-fine-tuned'
]
for ds in datasets:
    exists = os.path.exists(ds)
    print(f'Dataset {ds}: {"Present" if exists else "Missing"}')
    if exists:
        files = os.listdir(ds)[:5]  # Log first 5 files
        print(f'  Sample files: {files}')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

# Paths
INPUT_DIR = '/kaggle/input/jigsaw-agile-community-rules'
WORKING_DIR = '/kaggle/working'


# Load data
train_df = pd.read_csv(f'{INPUT_DIR}/train.csv')
test_df = pd.read_csv(f'{INPUT_DIR}/test.csv')
sample_df = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')

print(f'Train: {len(train_df)}, Test: {len(test_df)}, Sample: {len(sample_df)}')

# For simplicity, use 'body' as text
train_texts = train_df['body'].fillna('').tolist()
test_texts = test_df['body'].fillna('').tolist()
train_labels = train_df['rule_violation'].values


# Data audit: Label distribution
print('Overall label distribution:')
print(train_df['rule_violation'].value_counts())
print('\nLabel distribution by rule:')
rule_dist = train_df.groupby('rule')['rule_violation'].value_counts().unstack().fillna(0)
print(rule_dist)
print('\nLabel distribution by subreddit:')
subreddit_dist = train_df.groupby('subreddit')['rule_violation'].value_counts().unstack().fillna(0)
print(subreddit_dist.head(10))  # Top 10 subreddits

# Export summaries
rule_dist.to_csv('/kaggle/working/rule_distribution.csv')
subreddit_dist.to_csv('/kaggle/working/subreddit_distribution.csv')
print('Summaries exported to /kaggle/working/')


# Fold-based logistic ensemble
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

# Feature enrichment (ensure defined)
try:
    train_features_scaled
except NameError:
    # Ensure model is defined
    try:
        model
    except NameError:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('/kaggle/input/all-roberta-large-v1/all-roberta-large-v1')
    
    # Encode texts if not defined
    try:
        train_embeddings
    except NameError:
        print('Encoding train...')
        train_embeddings = model.encode(train_texts, batch_size=8, show_progress_bar=True)  # Reduced for memory
        print('Encoding test...')
        test_embeddings = model.encode(test_texts, batch_size=8, show_progress_bar=True)  # Reduced for memory
    
    # Rule embeddings
    unique_rules = train_df['rule'].unique()
    rule_texts = [rule for rule in unique_rules]
    rule_embeddings = model.encode(rule_texts, batch_size=8, show_progress_bar=True)  # Reduced for memory
    rule_emb_dict = {rule: emb for rule, emb in zip(unique_rules, rule_embeddings)}

    # Metadata scalars
    def extract_features(df):
        features = []
        for _, row in df.iterrows():
            body = row['body']
            length = len(body)
            url_count = body.count('http')
            uppercase_ratio = sum(1 for c in body if c.isupper()) / max(1, len(body))
            subreddit_freq = train_df['subreddit'].value_counts().get(row['subreddit'], 0) / len(train_df)
            features.append([length, url_count, uppercase_ratio, subreddit_freq])
        return np.array(features)

    train_meta = extract_features(train_df)
    test_meta = extract_features(test_df)

    # Combine embeddings and metadata
    train_rule_emb = np.array([rule_emb_dict[row['rule']] for _, row in train_df.iterrows()])
    test_rule_emb = np.array([rule_emb_dict.get(row['rule'], np.zeros(768)) for _, row in test_df.iterrows()])

    train_features = np.hstack([train_embeddings, train_rule_emb, train_meta])
    test_features = np.hstack([test_embeddings, test_rule_emb, test_meta])

    # Scale
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

# Stratified CV
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store fold models and predictions
fold_models = []
cv_scores = []
fold_reports = []
val_predictions = np.zeros(len(train_labels))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_features_scaled, train_labels)):
    print(f'Fold {fold+1}:')
    X_train_fold = train_features_scaled[train_idx]
    y_train_fold = train_labels[train_idx]
    X_val_fold = train_features_scaled[val_idx]
    y_val_fold = train_labels[val_idx]
    
    # Tune LR
    param_grid = {'C': [0.1, 1.0], 'class_weight': ['balanced']}
    lr = LogisticRegression(random_state=42, max_iter=1000)
    grid = GridSearchCV(lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_fold, y_train_fold)
    
    best_lr = grid.best_estimator_
    fold_models.append(best_lr)
    
    # Predict on val
    val_probs = best_lr.predict_proba(X_val_fold)[:, 1]
    auc = roc_auc_score(y_val_fold, val_probs)
    cv_scores.append(auc)
    val_predictions[val_idx] = val_probs
    
    print(f'  Best params: {grid.best_params_}, AUC: {auc:.3f}')
    fold_reports.append({'fold': fold+1, 'val_size': len(val_idx), 'pos_rate': y_val_fold.mean(), 'auc': auc})

print(f'\nEnsemble CV AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}')
# Export
pd.DataFrame(fold_reports).to_csv('/kaggle/working/cv_fold_reports.csv', index=False)

print(f'Models trained, ready for test prediction after feature scaling')


# Feature enrichment (rule embeddings already done in fold cell)
# Metadata scalars
def extract_features(df):
    features = []
    for _, row in df.iterrows():
        body = row['body']
        length = len(body)
        url_count = body.count('http')
        uppercase_ratio = sum(1 for c in body if c.isupper()) / max(1, len(body))
        subreddit_freq = train_df['subreddit'].value_counts().get(row['subreddit'], 0) / len(train_df)
        features.append([length, url_count, uppercase_ratio, subreddit_freq])
    return np.array(features)

train_meta = extract_features(train_df)
test_meta = extract_features(test_df)

# Combine embeddings and metadata
train_rule_emb = np.array([rule_emb_dict[row['rule']] for _, row in train_df.iterrows()])
test_rule_emb = np.array([rule_emb_dict.get(row['rule'], np.zeros(768)) for _, row in test_df.iterrows()])  # Fallback for unseen rules

train_features = np.hstack([train_embeddings, train_rule_emb, train_meta])
test_features = np.hstack([test_embeddings, test_rule_emb, test_meta])

print(f'Enriched train features shape: {train_features.shape}')
print(f'Enriched test features shape: {test_features.shape}')

# Persist preprocessor (simple scaler)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

print('Features scaled and ready')


# Test predictions from logistic ensemble
test_probs = np.mean([model.predict_proba(test_features_scaled)[:, 1] for model in fold_models], axis=0)
print(f'Test predictions averaged from {len(fold_models)} models')


# RoBERTa Inference
import torch
import logging
import math
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Add file handler to write logs to output (avoid duplicates)
if not logger.handlers:
    file_handler = logging.FileHandler('/kaggle/working/roberta_inference.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

logger.info('Starting RoBERTa inference pipeline')
try:
    logger.info('Loading RoBERTa tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/jigsaw-roberta-fine-tuned', local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained('/kaggle/input/jigsaw-roberta-fine-tuned', local_files_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    model.to(device)
    model.eval()
    logger.info('Model loaded and set to eval mode')
    
    # Log memory and GPU usage
    if device.type == 'cuda':
        logger.info(f'GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB')
        logger.info(f'GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB')
    else:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f'CPU memory used: {memory.used / 1024**2:.2f} MB / {memory.total / 1024**2:.2f} MB')
    
    logger.info('Model setup complete')
except Exception as e:
    logger.error(f'Error during model setup: {e}')
    raise

def predict_roberta(texts, batch_size=16):
    logger.info(f'Starting prediction on {len(texts)} texts with batch_size {batch_size}')
    probs = []
    total_batches = math.ceil(len(texts) / batch_size)
    start_time = time.time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.debug(f'Batch {i//batch_size + 1} input shape: {inputs["input_ids"].shape}')
            with torch.no_grad():
                outputs = model(**inputs)
                batch_probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
                probs.extend(batch_probs)
            logger.info(f'Processed batch {i//batch_size + 1}/{total_batches}')
        except Exception as e:
            logger.error(f'Error processing batch {i//batch_size + 1}: {e}')
            raise
    elapsed = time.time() - start_time
    logger.info(f'Prediction complete in {elapsed:.2f}s, generated {len(probs)} probabilities')
    return probs

import time
roberta_probs = predict_roberta(test_texts)
logger.info(f'RoBERTa probs shape: {len(roberta_probs)}, sample: {roberta_probs[:5]}')

# Ensemble: average with logistic probs
ensemble_probs = (test_probs + roberta_probs) / 2
logger.info(f'Ensemble probs shape: {len(ensemble_probs)}, sample: {ensemble_probs[:5]}')
logger.info('RoBERTa inference pipeline completed successfully')


# Submission
print(f'Final ensemble_probs length: {len(ensemble_probs)}, sample_df length: {len(sample_df)}')
if len(ensemble_probs) != len(sample_df):
    print('Length mismatch, using fallback 0.5')
    ensemble_probs = [0.5] * len(sample_df)
import numpy as np
ensemble_probs = np.clip(ensemble_probs, 0, 1)  # Ensure valid range
if np.isnan(ensemble_probs).any():
    print('NaN detected, replacing with 0.5')
    ensemble_probs = np.nan_to_num(ensemble_probs, nan=0.5)
sample_df['rule_violation'] = ensemble_probs
sample_df.to_csv('/kaggle/working/submission.csv', index=False)
print('Ensemble submission saved')

