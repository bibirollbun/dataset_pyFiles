# ==============================================================================
# 1. SETUP & IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm.auto import tqdm
import re
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
tqdm.pandas(desc="Processing")
print("Setup Complete. Notebook is running in OFFLINE mode.")

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
class CFG:
    ppl_model_path = '/kaggle/input/distilgpt2-offline-for-impostor/distilgpt2_offline'

    lgbm_params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'random_state': 42,
        'n_estimators': 2000, 'learning_rate': 0.02, 'num_leaves': 10,
        'min_child_samples': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'verbosity': -1, 'n_jobs': -1
    }
    n_folds = 5

# ==============================================================================
# 3. LOAD & PREPARE DATA
# ==============================================================================
BASE_PATH = Path('/kaggle/input/fake-or-real-the-impostor-hunt')
DATA_PATH = BASE_PATH / 'data'
TRAIN_PATH = DATA_PATH / 'train'
TEST_PATH = DATA_PATH / 'test'
TRAIN_CSV_PATH = DATA_PATH / 'train.csv'
train_df = pd.read_csv(TRAIN_CSV_PATH)

def load_texts(sample_id, base_path):
    article_dir_name = f"article_{sample_id:04d}" 
    article_path = base_path / article_dir_name
    try:
        with open(article_path / 'file_1.txt', 'r', encoding='utf-8') as f: text1 = f.read()
        with open(article_path / 'file_2.txt', 'r', encoding='utf-8') as f: text2 = f.read()
        return text1, text2
    except FileNotFoundError: return None, None

print("Loading training text data...")
texts = train_df['id'].progress_apply(lambda sample_id: load_texts(sample_id, TRAIN_PATH))
train_df[['text_1', 'text_2']] = pd.DataFrame(texts.tolist(), index=train_df.index)
train_df['label'] = train_df['real_text_id'].map({1: 0, 2: 1})
print("Data loaded.")

# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================
print("Initializing feature engineering pipeline from local files...")
device = "cuda" if torch.cuda.is_available() else "cpu"

ppl_model = AutoModelForCausalLM.from_pretrained(CFG.ppl_model_path).to(device)
ppl_tokenizer = AutoTokenizer.from_pretrained(CFG.ppl_model_path)
print(f"Perplexity model loaded from '{CFG.ppl_model_path}' to {device}.")

def calculate_perplexity(text, model, tokenizer):
    if pd.isna(text) or len(text) == 0: return np.nan
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)
    max_length = model.config.n_positions
    stride = 512
    nlls = []
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - begin_loc
        if trg_len == 0: continue
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            log_likelihood = outputs.loss * trg_len
        nlls.append(log_likelihood)
        
    if not nlls: return np.nan
    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return ppl.item()

def create_all_features(text, model, tokenizer):
    if pd.isna(text):
        return {'char_count': 0, 'word_count': 0, 'sentence_count': 0, 'avg_word_length': 0, 'avg_sentence_length': 0, 'unique_word_count': 0, 'ttr': 0, 'digit_count': 0, 'uppercase_word_count': 0, 'perplexity': np.nan}
    words = text.split()
    word_count = len(words)
    sentence_count = len(re.findall(r'[.!?]+', text))
    features = {
        'char_count': len(text),
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0,
        'unique_word_count': len(set(words)),
        'ttr': len(set(words)) / word_count if word_count > 0 else 0,
        'digit_count': sum(c.isdigit() for c in text),
        'uppercase_word_count': sum(1 for w in words if w.isupper() and len(w) > 1),
        'perplexity': calculate_perplexity(text, model, tokenizer)
    }
    return features

print("Generating features for training data...")
all_texts_train = pd.concat([train_df['text_1'], train_df['text_2']], ignore_index=True)
all_features_train = all_texts_train.progress_apply(lambda x: pd.Series(create_all_features(x, ppl_model, ppl_tokenizer)))

num_train_samples = len(train_df)
features_1 = all_features_train.iloc[:num_train_samples].reset_index(drop=True)
features_2 = all_features_train.iloc[num_train_samples:].reset_index(drop=True)
feature_diff = (features_1 - features_2).fillna(0)
feature_diff.columns = [f'diff_{col}' for col in feature_diff.columns]
y = train_df['label']
print("Training features are ready.")

# ==============================================================================
# 5. TRAIN OOF ENSEMBLE MODEL
# ==============================================================================
print("Training LightGBM model with OOF stacking...")

oof_preds = np.zeros(len(train_df))
skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=42)
models = []
for fold, (train_idx, valid_idx) in enumerate(skf.split(feature_diff, y)):
    X_train, y_train = feature_diff.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = feature_diff.iloc[valid_idx], y.iloc[valid_idx]

    model = lgb.LGBMClassifier(**CFG.lgbm_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )
    
    models.append(model)
    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]

    print(f"Fold {fold + 1} done.")

# Optional: Evaluate OOF predictions
oof_accuracy = accuracy_score(y, oof_preds > 0.5)
oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF Accuracy: {oof_accuracy:.4f}")
print(f"OOF AUC: {oof_auc:.4f}")

# ==============================================================================
# 6. GENERATE FINAL SUBMISSION (OOF ENSEMBLE)
# ==============================================================================
print("Generating ensemble submission using OOF models...")
test_ids = sorted([int(d.name.split('_')[1]) for d in TEST_PATH.iterdir() if d.is_dir()])
submission_data = []

for test_id in tqdm(test_ids, desc="Processing Test Set"):
    text1, text2 = load_texts(test_id, TEST_PATH)
    if text1 is not None:
        f1 = create_all_features(text1, ppl_model, ppl_tokenizer)
        f2 = create_all_features(text2, ppl_model, ppl_tokenizer)
        test_diff = (pd.Series(f1) - pd.Series(f2)).fillna(0).values.reshape(1, -1)
        
        # Average predictions from all models
        test_pred_proba = np.mean([model.predict_proba(test_diff)[:, 1][0] for model in models])
        prediction = 2 if test_pred_proba > 0.5 else 1
        submission_data.append({'id': test_id, 'real_text_id': prediction})

submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully using OOF ensemble!")
display(submission_df.head())


