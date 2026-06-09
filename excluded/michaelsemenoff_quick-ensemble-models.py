# Data manipulation
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import platform

# Machine Learning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ğŸ”¥ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.backends.cudnn.benchmark = True

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("\nâœ… Libraries imported successfully!")


# Load datasets
train_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')
sample_sub = pd.read_csv('/kaggle/input/rmit-hackathon-2025/sample_submission.csv')

# Create binary target
train_df['target'] = (train_df['label'] == 'jailbreak').astype(int)

print(f"Training set: {train_df.shape}")
print(f"Test set: {test_df.shape}")
print(f"\nClass distribution:")
print(train_df['target'].value_counts())
print(f"\nClass balance: {train_df['target'].value_counts(normalize=True).round(3)}")

# Display sample
display(train_df.head())


def create_features(df):
    """Create engineered features based on EDA insights"""
    features = pd.DataFrame()
    
    # Basic length features
    features['char_count'] = df['text'].str.len()
    features['word_count'] = df['text'].str.split().str.len()
    features['sentence_count'] = df['text'].apply(lambda x: len(sent_tokenize(str(x))))
    features['avg_word_length'] = features['char_count'] / (features['word_count'] + 1)
    
    # Jailbreak keyword indicators (from EDA)
    jailbreak_keywords = [
        'imagine', 'pretend', 'hypothetical', 'fictional', 'scenario',
        'ignore', 'bypass', 'override', 'disregard', 'forget',
        'guidelines', 'restrictions', 'constraints', 'rules',
        'character', 'roleplay', 'narrative', 'story', 'alternate',
        'research', 'academic', 'study', 'educational', 'project',
        'developer mode', 'enabled', 'answer ans', 'content question'
    ]
    
    for keyword in jailbreak_keywords[:15]:  # Top 15 to avoid too many features
        features[f'has_{keyword.replace(" ", "_")}'] = df['text'].str.lower().str.contains(keyword, regex=False).astype(int)
    
    # Complexity metrics
    features['punctuation_count'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c in string.punctuation))
    features['question_count'] = df['text'].str.count(r'\?')
    features['quote_count'] = df['text'].apply(lambda x: str(x).count('"') + str(x).count("'"))
    features['bracket_count'] = df['text'].apply(lambda x: str(x).count('(') + str(x).count('[') + str(x).count('{'))
    features['comma_count'] = df['text'].str.count(',')
    
    # Special characters
    features['has_json'] = df['text'].apply(lambda x: int('{' in str(x) and '}' in str(x) and ':' in str(x)))
    features['has_newlines'] = df['text'].str.contains('\n').astype(int)
    features['has_code_block'] = df['text'].str.contains('```').astype(int)
    features['capital_ratio'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
    features['digit_ratio'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isdigit()) / (len(str(x)) + 1))
    
    # Lexical diversity
    features['lexical_diversity'] = df['text'].apply(
        lambda x: len(set(w.lower() for w in word_tokenize(str(x)) if w.isalnum())) / (features.loc[features.index == df[df['text'] == x].index[0], 'word_count'].values[0] + 1) if len(word_tokenize(str(x))) > 0 else 0
    )
    
    return features

print("Creating features for training set...")
train_features = create_features(train_df)
print(f"Training features shape: {train_features.shape}")

print("\nCreating features for test set...")
test_features = create_features(test_df)
print(f"Test features shape: {test_features.shape}")

print("\nâœ… Feature engineering complete!")
print("\nFeature list:")
for i, col in enumerate(train_features.columns, 1):
    print(f"{i:2d}. {col}")


print("=" * 80)
print("MODEL 1: TF-IDF + Logistic Regression (Baseline)")
print("=" * 80)

# TF-IDF vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),  # unigrams, bigrams, trigrams
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_test_tfidf = tfidf.transform(test_df['text'])
y_train = train_df['target'].values

print(f"TF-IDF features shape: {X_train_tfidf.shape}")

# Train with cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = []
oof_predictions = np.zeros(len(train_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tfidf, y_train), 1):
    X_tr, X_val = X_train_tfidf[train_idx], X_train_tfidf[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Train model
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, class_weight='balanced')
    model.fit(X_tr, y_tr)
    
    # Predict probabilities
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_predictions[val_idx] = val_preds
    
    # Calculate ROC-AUC
    score = roc_auc_score(y_val, val_preds)
    cv_scores.append(score)
    print(f"Fold {fold}: ROC-AUC = {score:.6f}")

print(f"\n{'='*80}")
print(f"Mean CV ROC-AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
print(f"{'='*80}")

# Train final model on full data
lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, class_weight='balanced')
lr_model.fit(X_train_tfidf, y_train)

# Predictions for test set
lr_test_preds = lr_model.predict_proba(X_test_tfidf)[:, 1]

print(f"\nâœ… Baseline model trained!")
print(f"Test predictions range: [{lr_test_preds.min():.4f}, {lr_test_preds.max():.4f}]")


print("=" * 80)
print("MODEL 2: XGBoost with Engineered Features")
print("=" * 80)

# Create compact TF-IDF features (fewer features for XGBoost)
tfidf_compact = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9
)

X_train_tfidf_compact = tfidf_compact.fit_transform(train_df['text']).toarray()
X_test_tfidf_compact = tfidf_compact.transform(test_df['text']).toarray()

# Combine with engineered features
X_train_combined = np.hstack([train_features.values, X_train_tfidf_compact])
X_test_combined = np.hstack([test_features.values, X_test_tfidf_compact])

print(f"Combined features shape: {X_train_combined.shape}")

# Scale features
scaler = StandardScaler()
X_train_combined = scaler.fit_transform(X_train_combined)
X_test_combined = scaler.transform(X_test_combined)

# XGBoost with cross-validation
xgb_params = {
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1.5,  # Handle class imbalance
    'eval_metric': 'auc',
    'random_state': SEED,
    'tree_method': 'hist',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

cv_scores_xgb = []
oof_predictions_xgb = np.zeros(len(train_df))
feature_importance = np.zeros(X_train_combined.shape[1])

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train), 1):
    X_tr, X_val = X_train_combined[train_idx], X_train_combined[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Train model
    model = XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict probabilities
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_predictions_xgb[val_idx] = val_preds
    
    # Accumulate feature importance
    feature_importance += model.feature_importances_
    
    # Calculate ROC-AUC
    score = roc_auc_score(y_val, val_preds)
    cv_scores_xgb.append(score)
    print(f"Fold {fold}: ROC-AUC = {score:.6f}")

print(f"\n{'='*80}")
print(f"Mean CV ROC-AUC: {np.mean(cv_scores_xgb):.6f} (+/- {np.std(cv_scores_xgb):.6f})")
print(f"{'='*80}")

# Train final model on full data
xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(X_train_combined, y_train, verbose=False)

# Predictions for test set
xgb_test_preds = xgb_model.predict_proba(X_test_combined)[:, 1]

print(f"\nâœ… XGBoost model trained!")
print(f"Test predictions range: [{xgb_test_preds.min():.4f}, {xgb_test_preds.max():.4f}]")

# Feature importance (top 20)
feature_names = list(train_features.columns) + [f'tfidf_{i}' for i in range(X_train_tfidf_compact.shape[1])]
feature_importance = feature_importance / len(cv_scores_xgb)
top_features_idx = np.argsort(feature_importance)[-20:]

print("\nğŸ“Š Top 20 Most Important Features:")
for idx in reversed(top_features_idx):
    if idx < len(train_features.columns):
        print(f"  {feature_names[idx]:30s}: {feature_importance[idx]:.4f}")


print("=" * 80)
print("MODEL 3: DistilBERT Fine-tuning (GPU-Accelerated)")
print("=" * 80)

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else 0
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Training function

def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, use_amp):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast(enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Get predictions
        logits = outputs.logits.detach()
        preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / max(len(dataloader), 1)
    roc_auc = roc_auc_score(true_labels, predictions)
    
    return avg_loss, roc_auc

# Validation function

def eval_model(model, dataloader, device, use_amp):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    roc_auc = roc_auc_score(true_labels, predictions)
    return predictions, roc_auc

# Initialize tokenizer with manual snapshot fallback to avoid chat template fetch errors
import os
MODEL_NAME = 'distilbert-base-uncased'
print(f"Loading tokenizer: {MODEL_NAME}")

tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    print("Tokenizer downloaded directly from Hugging Face.")
except Exception as e:
    if 'additional_chat_templates' in str(e):
        print("Encountered additional_chat_templates 404. Using manual snapshot download without chat templates...")
        from huggingface_hub import snapshot_download

        cache_base = os.path.join(os.getcwd(), 'hf_cache', MODEL_NAME.replace('/', '__'), 'tokenizer')
        allow_patterns = [
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.txt',
            'vocab.json',
            'special_tokens_map.json',
            'added_tokens.json',
            'merges.txt'
        ]

        cache_dir = snapshot_download(
            repo_id=MODEL_NAME,
            repo_type='model',
            local_dir=cache_base,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            ignore_patterns=['*.safetensors', '*.bin', '*.onnx', '*.tflite', '*.h5', '*.msgpack']
        )

        tokenizer = AutoTokenizer.from_pretrained(cache_dir, use_fast=True, local_files_only=True)
        tokenizer.chat_template = None
        print(f"Tokenizer loaded from cached snapshot at {cache_dir}.")
    else:
        raise e

print(f"âœ… Tokenizer loaded successfully")
print(f"Using model: {MODEL_NAME}")
print(f"Device: {device}")

# Hyperparameters
MAX_LENGTH = 256
BATCH_SIZE = 16 if torch.cuda.is_available() else 8
EPOCHS = 3
LEARNING_RATE = 2e-5
USE_AMP = torch.cuda.is_available()
NUM_WORKERS = 0 if platform.system().lower().startswith('win') else 2
PIN_MEMORY = torch.cuda.is_available()

print(f"\nHyperparameters:")
print(f"  Max Length: {MAX_LENGTH}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  AMP Enabled: {USE_AMP}")
print(f"  DataLoader Workers: {NUM_WORKERS}")

common_loader_kwargs = {
    'batch_size': BATCH_SIZE,
    'pin_memory': PIN_MEMORY,
    'num_workers': NUM_WORKERS
}


# Use 3-fold CV for BERT (faster training)
skf_bert = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
cv_scores_bert = []
oof_predictions_bert = np.zeros(len(train_df))

common_loader_kwargs = {
    'batch_size': BATCH_SIZE,
    'pin_memory': PIN_MEMORY,
    'num_workers': NUM_WORKERS
}

for fold, (train_idx, val_idx) in enumerate(skf_bert.split(train_df, y_train), 1):
    print(f"\n{'='*80}")
    print(f"FOLD {fold}/3")
    print(f"{'='*80}")
    
    # Prepare datasets
    train_texts = train_df.iloc[train_idx]['text'].values
    train_labels = y_train[train_idx]
    val_texts = train_df.iloc[val_idx]['text'].values
    val_labels = y_train[val_idx]
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    model.to(device)
    
    # Optimizer, scheduler, and scaler for AMP
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=USE_AMP)
    
    # Training loop
    best_val_auc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, scheduler, device, scaler, USE_AMP)
        val_preds, val_auc = eval_model(model, val_loader, device, USE_AMP)
        
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # Store predictions
            oof_predictions_bert[val_idx] = val_preds
    
    cv_scores_bert.append(best_val_auc)
    print(f"\nFold {fold} Best ROC-AUC: {best_val_auc:.6f}")
    
    # Clean up GPU memory
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\n{'='*80}")
print(f"DistilBERT Mean CV ROC-AUC: {np.mean(cv_scores_bert):.6f} (+/- {np.std(cv_scores_bert):.6f})")
print(f"{'='*80}")


print("\n" + "="*80)
print("Training final DistilBERT model on full training data...")
print("="*80)

# Prepare full training dataset
full_train_dataset = TextDataset(train_df['text'].values, y_train, tokenizer, MAX_LENGTH)
full_train_loader = DataLoader(full_train_dataset, shuffle=True, **common_loader_kwargs)

# Prepare test dataset
test_dataset = TextDataset(test_df['text'].values, [0]*len(test_df), tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

# Initialize final model
final_model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    problem_type="single_label_classification"
)
final_model.to(device)

# Optimizer, scheduler, and scaler
optimizer = AdamW(final_model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(full_train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
final_scaler = GradScaler(enabled=USE_AMP)

# Training loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    train_loss, train_auc = train_epoch(final_model, full_train_loader, optimizer, scheduler, device, final_scaler, USE_AMP)
    print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")

# Predict on test set
print("\nGenerating predictions for test set...")
final_model.eval()
bert_test_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with autocast(enabled=USE_AMP):
            outputs = final_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        bert_test_preds.extend(preds)

bert_test_preds = np.array(bert_test_preds)

print(f"\nâœ… DistilBERT model trained and predictions generated!")
print(f"Test predictions range: [{bert_test_preds.min():.4f}, {bert_test_preds.max():.4f}]")

# Clean up
del final_model, optimizer, scheduler, final_scaler
torch.cuda.empty_cache() if torch.cuda.is_available() else None


print("=" * 80)
print("MODEL COMPARISON & ENSEMBLE")
print("=" * 80)

# Compare CV scores
results_df = pd.DataFrame({
    'Model': ['TF-IDF + Logistic Regression', 'XGBoost + Features', 'DistilBERT'],
    'CV ROC-AUC': [
        np.mean(cv_scores),
        np.mean(cv_scores_xgb),
        np.mean(cv_scores_bert)
    ],
    'Std': [
        np.std(cv_scores),
        np.std(cv_scores_xgb),
        np.std(cv_scores_bert)
    ]
})

print("\nCross-Validation Results:")
print("-" * 80)
display(results_df)

# Visualize model performance
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
ax.bar(x, results_df['CV ROC-AUC'], yerr=results_df['Std'], 
       capsize=5, alpha=0.7, color=['coral', 'skyblue', 'lightgreen'])
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('ROC-AUC Score', fontsize=12)
ax.set_title('Model Performance Comparison (Cross-Validation)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.set_ylim([0.8, 1.0])
ax.grid(True, alpha=0.3, axis='y')

for i, (auc, std) in enumerate(zip(results_df['CV ROC-AUC'], results_df['Std'])):
    ax.text(i, auc + std + 0.005, f'{auc:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Create ensemble predictions (weighted average)
print("\n" + "=" * 80)
print("ENSEMBLE STRATEGY")
print("=" * 80)

# Calculate optimal weights based on CV performance
weights = np.array([
    np.mean(cv_scores),
    np.mean(cv_scores_xgb),
    np.mean(cv_scores_bert)
])
weights = weights / weights.sum()  # Normalize to sum to 1

print("\nEnsemble Weights (based on CV performance):")
for model, weight in zip(results_df['Model'], weights):
    print(f"  {model:35s}: {weight:.4f}")

# Create ensemble predictions
ensemble_test_preds = (
    weights[0] * lr_test_preds +
    weights[1] * xgb_test_preds +
    weights[2] * bert_test_preds
)

print(f"\nâœ… Ensemble predictions created!")
print(f"Ensemble predictions range: [{ensemble_test_preds.min():.4f}, {ensemble_test_preds.max():.4f}]")

# Distribution of predictions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(lr_test_preds, bins=50, alpha=0.7, color='coral', edgecolor='black')
axes[0, 0].set_title('Logistic Regression Predictions', fontweight='bold')
axes[0, 0].set_xlabel('Predicted Probability')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(xgb_test_preds, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 1].set_title('XGBoost Predictions', fontweight='bold')
axes[0, 1].set_xlabel('Predicted Probability')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(bert_test_preds, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('DistilBERT Predictions', fontweight='bold')
axes[1, 0].set_xlabel('Predicted Probability')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(ensemble_test_preds, bins=50, alpha=0.7, color='gold', edgecolor='black')
axes[1, 1].set_title('Ensemble Predictions', fontweight='bold')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


print("=" * 80)
print("GENERATING SUBMISSION FILES")
print("=" * 80)

# Create submission dataframes
submissions = {
    'logistic_regression': lr_test_preds,
    'xgboost': xgb_test_preds,
    'distilbert': bert_test_preds,
    'ensemble': ensemble_test_preds
}

for model_name, predictions in submissions.items():
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'TARGET': predictions
    })
    
    filename = f'submission_{model_name}.csv'
    submission.to_csv(filename, index=False)
    print(f"âœ… Saved: {filename}")
    
    # Display sample
    print(f"\nSample from {model_name}:")
    display(submission.head(10))

print("\n" + "=" * 80)
print("SUBMISSION STATISTICS")
print("=" * 80)

stats_df = pd.DataFrame({
    'Model': list(submissions.keys()),
    'Mean': [np.mean(pred) for pred in submissions.values()],
    'Std': [np.std(pred) for pred in submissions.values()],
    'Min': [np.min(pred) for pred in submissions.values()],
    'Max': [np.max(pred) for pred in submissions.values()],
    'Median': [np.median(pred) for pred in submissions.values()]
})

display(stats_df)

print("\n" + "=" * 80)
print("ğŸ�¯ RECOMMENDATION")
print("=" * 80)
print("\nBased on cross-validation performance:")
best_model_idx = np.argmax([np.mean(cv_scores), np.mean(cv_scores_xgb), np.mean(cv_scores_bert)])
best_model_name = results_df.iloc[best_model_idx]['Model']
print(f"  Best Single Model: {best_model_name} (CV ROC-AUC: {results_df.iloc[best_model_idx]['CV ROC-AUC']:.6f})")
print(f"  Recommended Submission: submission_ensemble.csv")
print(f"  Ensemble combines strengths of all models for robust predictions!")

print("\n" + "=" * 80)
print("âœ… ALL SUBMISSIONS READY!")
print("=" * 80)


print("=" * 80)
print("MODEL INSIGHTS & PREDICTIONS ANALYSIS")
print("=" * 80)

# Analyze prediction confidence
ensemble_confident_jailbreak = np.sum(ensemble_test_preds > 0.8)
ensemble_confident_benign = np.sum(ensemble_test_preds < 0.2)
ensemble_uncertain = np.sum((ensemble_test_preds >= 0.2) & (ensemble_test_preds <= 0.8))

print(f"\nEnsemble Prediction Confidence:")
print(f"  Confident Jailbreak (>0.8): {ensemble_confident_jailbreak} ({ensemble_confident_jailbreak/len(test_df)*100:.1f}%)")
print(f"  Confident Benign (<0.2):    {ensemble_confident_benign} ({ensemble_confident_benign/len(test_df)*100:.1f}%)")
print(f"  Uncertain (0.2-0.8):        {ensemble_uncertain} ({ensemble_uncertain/len(test_df)*100:.1f}%)")

# Model agreement analysis
model_agreement = np.zeros(len(test_df))
predictions_matrix = np.column_stack([lr_test_preds, xgb_test_preds, bert_test_preds])

for i in range(len(test_df)):
    # Check if all models agree on direction (all >0.5 or all <0.5)
    preds = predictions_matrix[i]
    if (preds > 0.5).all() or (preds < 0.5).all():
        model_agreement[i] = 1

agreement_rate = model_agreement.mean()
print(f"\nModel Agreement Rate: {agreement_rate*100:.1f}%")
print(f"  All models agree: {int(agreement_rate * len(test_df))} samples")
print(f"  Models disagree:  {int((1-agreement_rate) * len(test_df))} samples")

# Examples of high confidence predictions
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)

# Get some examples
high_jailbreak_idx = np.where(ensemble_test_preds > 0.9)[0][:3]
high_benign_idx = np.where(ensemble_test_preds < 0.1)[0][:3]
uncertain_idx = np.where((ensemble_test_preds > 0.45) & (ensemble_test_preds < 0.55))[0][:3]

print("\nğŸš¨ HIGH CONFIDENCE JAILBREAK (Ensemble > 0.9):")
for idx in high_jailbreak_idx:
    print(f"\nID: {test_df.iloc[idx]['Id']}")
    print(f"Ensemble: {ensemble_test_preds[idx]:.4f} | LR: {lr_test_preds[idx]:.4f} | XGB: {xgb_test_preds[idx]:.4f} | BERT: {bert_test_preds[idx]:.4f}")
    print(f"Text: {test_df.iloc[idx]['text'][:200]}...")

print("\n" + "-" * 80)
print("\nâœ… HIGH CONFIDENCE BENIGN (Ensemble < 0.1):")
for idx in high_benign_idx:
    print(f"\nID: {test_df.iloc[idx]['Id']}")
    print(f"Ensemble: {ensemble_test_preds[idx]:.4f} | LR: {lr_test_preds[idx]:.4f} | XGB: {xgb_test_preds[idx]:.4f} | BERT: {bert_test_preds[idx]:.4f}")
    print(f"Text: {test_df.iloc[idx]['text'][:200]}...")

print("\n" + "-" * 80)
print("\nâ�“ UNCERTAIN PREDICTIONS (Ensemble ~0.5):")
for idx in uncertain_idx:
    print(f"\nID: {test_df.iloc[idx]['Id']}")
    print(f"Ensemble: {ensemble_test_preds[idx]:.4f} | LR: {lr_test_preds[idx]:.4f} | XGB: {xgb_test_preds[idx]:.4f} | BERT: {bert_test_preds[idx]:.4f}")
    print(f"Text: {test_df.iloc[idx]['text'][:200]}...")

# Correlation between model predictions
print("\n" + "=" * 80)
print("MODEL PREDICTIONS CORRELATION")
print("=" * 80)

pred_corr = np.corrcoef([lr_test_preds, xgb_test_preds, bert_test_preds])
pred_corr_df = pd.DataFrame(
    pred_corr,
    index=['Logistic Reg', 'XGBoost', 'DistilBERT'],
    columns=['Logistic Reg', 'XGBoost', 'DistilBERT']
)

print("\nCorrelation Matrix:")
display(pred_corr_df)

plt.figure(figsize=(8, 6))
sns.heatmap(pred_corr_df, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0.5, vmin=0, vmax=1, square=True, linewidths=1)
plt.title('Correlation Between Model Predictions', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

print("\nğŸ’¡ High correlation indicates models learn similar patterns.")
print("   Lower correlation suggests diverse approaches that benefit ensembling.")

