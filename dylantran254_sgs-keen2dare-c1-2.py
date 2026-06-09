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


# Load and explore the training data
df = pd.read_csv('/kaggle/input/rmit-genai/train.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nLabel distribution (%):")
print(df['label'].value_counts(normalize=True) * 100)


# ============================================================================
# Challenge 1
# ============================================================================

# ============================================================================
# CHALLENGE 1: FUNDAMENTALS - JAILBREAK DETECTION
# Exploratory Data Analysis + Baseline Models
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, classification_report, 
                             confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# Config
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("CHALLENGE 1: FUNDAMENTALS - JAILBREAK DETECTION")
print("=" * 80)
print("\nğŸ�¯ Tasks:")
print("   1. Load and explore the dataset")
print("   2. Perform Exploratory Data Analysis (EDA)")
print("   3. Build baseline model (TF-IDF + Logistic Regression)")
print("   4. Generate first submission")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: DATA LOADING")
print("=" * 80)

train_csv = '/kaggle/input/rmit-genai/train.csv'
test_csv = '/kaggle/input/rmit-genai/test.csv'

# Load data
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

print(f"âœ“ Train shape: {train_df.shape}")
print(f"âœ“ Test shape: {test_df.shape}")

print(f"\nğŸ“‹ Train columns: {train_df.columns.tolist()}")
print(f"ğŸ“‹ Test columns: {test_df.columns.tolist()}")

print(f"\nğŸ“Š First 3 train samples:")
print(train_df.head(3))

print(f"\nğŸ“Š First 3 test samples:")
print(test_df.head(3))

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# 2.1 Basic Statistics
print("\n[2.1] Dataset Overview")
print("-" * 40)
print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")
print(f"\nTrain data info:")
print(train_df.info())

# 2.2 Target Distribution
print("\n[2.2] Target Distribution")
print("-" * 40)
label_counts = train_df['label'].value_counts()
print(label_counts)
print(f"\nClass balance:")
for label, count in label_counts.items():
    print(f"   {label}: {count} ({count/len(train_df)*100:.1f}%)")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
label_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Pie chart
axes[1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.3 Text Length Analysis
print("\n[2.3] Text Length Analysis")
print("-" * 40)

train_df['text_length'] = train_df['text'].str.len()
train_df['word_count'] = train_df['text'].str.split().str.len()

print("\nText Length Statistics by Class:")
print(train_df.groupby('label')[['text_length', 'word_count']].describe())

# Visualize text length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label in train_df['label'].unique():
    data = train_df[train_df['label'] == label]['text_length']
    axes[0].hist(data, bins=50, alpha=0.6, label=label)

axes[0].set_title('Text Length Distribution by Class', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Text Length (characters)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].set_xlim(0, 1000)

# Box plot
train_df.boxplot(column='text_length', by='label', ax=axes[1])
axes[1].set_title('Text Length Box Plot by Class', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Text Length (characters)')
plt.suptitle('')

plt.tight_layout()
plt.savefig('text_length_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.4 Word Count Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label in train_df['label'].unique():
    data = train_df[train_df['label'] == label]['word_count']
    axes[0].hist(data, bins=50, alpha=0.6, label=label)

axes[0].set_title('Word Count Distribution by Class', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].set_xlim(0, 200)

# Box plot
train_df.boxplot(column='word_count', by='label', ax=axes[1])
axes[1].set_title('Word Count Box Plot by Class', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Word Count')
plt.suptitle('')

plt.tight_layout()
plt.savefig('word_count_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.5 Sample Texts
print("\n[2.5] Sample Texts from Each Class")
print("-" * 40)

for label in train_df['label'].unique():
    print(f"\n{'='*60}")
    print(f"LABEL: {label.upper()}")
    print('='*60)
    samples = train_df[train_df['label'] == label].sample(3, random_state=RANDOM_SEED)
    for idx, row in samples.iterrows():
        print(f"\nSample {idx}:")
        print(f"Text: {row['text'][:200]}...")
        print(f"Length: {row['text_length']} chars, {row['word_count']} words")

# 2.6 Missing Values
print("\n[2.6] Missing Values Check")
print("-" * 40)
print("Train data:")
print(train_df.isnull().sum())
print("\nTest data:")
print(test_df.isnull().sum())

# 2.7 Common Words Analysis
print("\n[2.7] Most Common Words by Class")
print("-" * 40)

from collections import Counter
import re

def get_top_words(texts, n=20):
    """Extract top N most common words"""
    all_words = []
    for text in texts:
        # Simple tokenization
        words = re.findall(r'\b\w+\b', str(text).lower())
        # Filter out very short words
        words = [w for w in words if len(w) > 2]
        all_words.extend(words)
    return Counter(all_words).most_common(n)

for label in train_df['label'].unique():
    texts = train_df[train_df['label'] == label]['text']
    top_words = get_top_words(texts, n=15)
    
    print(f"\n{label.upper()} - Top 15 words:")
    for word, count in top_words:
        print(f"   {word}: {count}")

# Visualize top words
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, label in enumerate(train_df['label'].unique()):
    texts = train_df[train_df['label'] == label]['text']
    top_words = get_top_words(texts, n=15)
    
    words, counts = zip(*top_words)
    axes[idx].barh(words, counts, color='#3498db' if label == 'benign' else '#e74c3c')
    axes[idx].set_title(f'Top 15 Words - {label.upper()}', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Frequency')
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.savefig('top_words_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: DATA PREPROCESSING")
print("=" * 80)

def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = text.replace('\x00', '')  # Remove null characters
    return text

# Preprocess
train_df['text_clean'] = train_df['text'].apply(preprocess_text)
test_df['text_clean'] = test_df['text'].apply(preprocess_text)

# Create binary target
train_df['target'] = (train_df['label'] == 'jailbreak').astype(int)

print(f"âœ“ Preprocessed {len(train_df)} train and {len(test_df)} test samples")
print(f"\nâœ“ Target encoding:")
print(f"   benign â†’ 0")
print(f"   jailbreak â†’ 1")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    train_df['text_clean'], 
    train_df['target'],
    test_size=0.2, 
    random_state=RANDOM_SEED, 
    stratify=train_df['target']
)

print(f"\nâœ“ Train/Val split:")
print(f"   Train: {len(X_train)} samples")
print(f"   Val:   {len(X_val)} samples")
print(f"\nâœ“ Class distribution:")
print(f"   Train: {pd.Series(y_train).value_counts().to_dict()}")
print(f"   Val:   {pd.Series(y_val).value_counts().to_dict()}")

# ============================================================================
# STEP 4: BASELINE MODEL - TF-IDF + LOGISTIC REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: BASELINE MODEL - TF-IDF + LOGISTIC REGRESSION")
print("=" * 80)

# 4.1 TF-IDF Vectorization
print("\n[4.1] TF-IDF Vectorization")
print("-" * 40)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # unigrams and bigrams
    min_df=2,
    max_df=0.9,
    strip_accents='unicode',
    lowercase=True,
    stop_words='english'
)

print("TF-IDF parameters:")
print(f"   max_features: 5000")
print(f"   ngram_range: (1, 2)")
print(f"   min_df: 2")
print(f"   max_df: 0.9")

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(test_df['text_clean'])

print(f"\nâœ“ TF-IDF shapes:")
print(f"   Train: {X_train_tfidf.shape}")
print(f"   Val:   {X_val_tfidf.shape}")
print(f"   Test:  {X_test_tfidf.shape}")

# 4.2 Train Logistic Regression
print("\n[4.2] Training Logistic Regression")
print("-" * 40)

clf = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_SEED,
    class_weight='balanced',
    C=1.0,
    solver='liblinear'
)

clf.fit(X_train_tfidf, y_train)

print("âœ“ Model trained successfully")

# 4.3 Predictions
y_train_pred = clf.predict(X_train_tfidf)
y_train_proba = clf.predict_proba(X_train_tfidf)[:, 1]

y_val_pred = clf.predict(X_val_tfidf)
y_val_proba = clf.predict_proba(X_val_tfidf)[:, 1]

test_proba = clf.predict_proba(X_test_tfidf)[:, 1]

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MODEL EVALUATION")
print("=" * 80)

# 5.1 Performance Metrics
print("\n[5.1] Performance Metrics")
print("-" * 40)

train_auc = roc_auc_score(y_train, y_train_proba)
val_auc = roc_auc_score(y_val, y_val_proba)

print(f"\nğŸ“Š ROC-AUC Scores:")
print(f"   Train AUC: {train_auc:.4f}")
print(f"   Val AUC:   {val_auc:.4f}")
print(f"   Difference: {abs(train_auc - val_auc):.4f}")

print(f"\nğŸ“Š Classification Report (Validation):")
print(classification_report(y_val, y_val_pred, 
                          target_names=['Benign', 'Jailbreak']))

# 5.2 Confusion Matrix
print("\n[5.2] Confusion Matrix")
print("-" * 40)

cm = confusion_matrix(y_val, y_val_pred)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Jailbreak'],
            yticklabels=['Benign', 'Jailbreak'])
plt.title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.3 ROC Curve
print("\n[5.3] ROC Curve")
print("-" * 40)

fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Validation Set', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.4 Prediction Distribution
print("\n[5.4] Prediction Distribution")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Validation predictions
for label in [0, 1]:
    mask = y_val == label
    axes[0].hist(y_val_proba[mask], bins=50, alpha=0.6, 
                label=f'True: {"Benign" if label == 0 else "Jailbreak"}')

axes[0].set_title('Validation Prediction Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Probability (Jailbreak)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')

# Test predictions
axes[1].hist(test_proba, bins=50, color='skyblue', edgecolor='black')
axes[1].set_title('Test Prediction Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Probability (Jailbreak)')
axes[1].set_ylabel('Frequency')
axes[1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nTest predictions summary:")
print(f"   Mean: {test_proba.mean():.4f}")
print(f"   Std:  {test_proba.std():.4f}")
print(f"   Min:  {test_proba.min():.4f}")
print(f"   Max:  {test_proba.max():.4f}")

# 5.5 Feature Importance
print("\n[5.5] Top Predictive Features")
print("-" * 40)

# Get feature names and coefficients
feature_names = tfidf.get_feature_names_out()
coefficients = clf.coef_[0]

# Top features for jailbreak class
top_jailbreak_idx = np.argsort(coefficients)[-20:]
print("\nTop 20 features indicating JAILBREAK:")
for idx in reversed(top_jailbreak_idx):
    print(f"   {feature_names[idx]}: {coefficients[idx]:.4f}")

# Top features for benign class
top_benign_idx = np.argsort(coefficients)[:20]
print("\nTop 20 features indicating BENIGN:")
for idx in top_benign_idx:
    print(f"   {feature_names[idx]}: {coefficients[idx]:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Jailbreak features
top_jail_features = [feature_names[i] for i in reversed(top_jailbreak_idx[:15])]
top_jail_coefs = [coefficients[i] for i in reversed(top_jailbreak_idx[:15])]
axes[0].barh(top_jail_features, top_jail_coefs, color='#e74c3c')
axes[0].set_title('Top 15 Jailbreak Indicators', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Coefficient')

# Benign features
top_benign_features = [feature_names[i] for i in top_benign_idx[:15]]
top_benign_coefs = [coefficients[i] for i in top_benign_idx[:15]]
axes[1].barh(top_benign_features, top_benign_coefs, color='#2ecc71')
axes[1].set_title('Top 15 Benign Indicators', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Coefficient')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 6: CREATE SUBMISSION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: CREATE SUBMISSION FILE")
print("=" * 80)

# Create submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'target': test_proba
})

# Validate
assert len(submission) == len(test_df), "Row count mismatch"
assert submission['target'].between(0, 1).all(), "Target must be between 0 and 1"
assert not submission.isnull().any().any(), "Contains null values"

# Save
submission.to_csv('submission.csv', index=False)

print(f"\nâœ… SUBMISSION CREATED")
print(f"   File: submission.csv")
print(f"   Rows: {len(submission)}")
print(f"   Columns: {submission.columns.tolist()}")

print(f"\nğŸ“Š Submission Preview:")
print(submission.head(10))

print(f"\nğŸ“ˆ Submission Statistics:")
print(submission['target'].describe())

print(f"\nğŸ“¦ Prediction Distribution:")
print(f"   < 0.3 (likely benign):     {(submission['target'] < 0.3).sum()}")
print(f"   0.3-0.7 (uncertain):       {((submission['target'] >= 0.3) & (submission['target'] <= 0.7)).sum()}")
print(f"   > 0.7 (likely jailbreak):  {(submission['target'] > 0.7).sum()}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("âœ… CHALLENGE 1 COMPLETED!")
print("=" * 80)
print("\nğŸ“‹ Summary:")
print(f"   âœ“ EDA completed with visualizations")
print(f"   âœ“ Baseline model trained (TF-IDF + Logistic Regression)")
print(f"   âœ“ Validation AUC: {val_auc:.4f}")
print(f"   âœ“ Submission file created: submission.csv")
print("\nğŸ�¯ Next Steps:")
print("   1. Upload submission.csv to Kaggle")
print("   2. Earn Challenge 1 marks (20%)")
print("   3. Proceed to Challenge 2 for advanced models")
print("=" * 80)


# ============================================================================
# ROBUST MULTILINGUAL MODELS - REVISED ENSEMBLE
# Models: XLM-RoBERTa-base + DeBERTa-v3-base
# Primary: Local models | Fallback: HF Inference API
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, 
                             precision_score, recall_score, classification_report)
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

# Config
sns.set_style('whitegrid')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("ROBUST MULTILINGUAL JAILBREAK DETECTOR - REVISED")
print("=" * 80)
print("\nğŸ�¯ ENSEMBLE MODELS:")
print("   1. XLM-RoBERTa-base (multilingual)")
print("   2. DeBERTa-v3-base (strong performance)")
print("\nâš™ï¸� STRATEGY:")
print("   â€¢ Try loading models locally (faster)")
print("   â€¢ If download fails â†’ Use HF Inference API")
print("\nâš™ï¸� KAGGLE SETUP:")
print("   â€¢ Enable Internet in Settings")
print("   â€¢ (Optional) Add HF_TOKEN secret for API fallback")
print("=" * 80)

start_time = time.time()

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: DATA LOADING")
print("=" * 80)

train_csv = '/kaggle/input/rmit-genai/train.csv'
test_csv = '/kaggle/input/rmit-genai/test.csv'

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    return str(text).strip().replace('\x00', '')

try:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    train_df['text_clean'] = train_df['text'].apply(preprocess_text)
    test_df['text_clean'] = test_df['text'].apply(preprocess_text)
    train_df['target'] = (train_df['label'] == 'jailbreak').astype(int)
    
    print(f"âœ“ Loaded: {len(train_df)} train, {len(test_df)} test")
    print(f"âœ“ Classes: {train_df['target'].value_counts().to_dict()}")
    
except Exception as e:
    print(f"â�Œ Error loading data: {e}")
    raise

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    train_df['text_clean'], train_df['target'],
    test_size=0.2, random_state=RANDOM_SEED, stratify=train_df['target']
)

print(f"âœ“ Split: {len(X_train)} train / {len(X_val)} val")

results = []

# ============================================================================
# MODEL 1: XLM-RoBERTa-base (with fallbacks)
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 1: XLM-RoBERTa-base")
print("=" * 80)

model1_start = time.time()
xlm_success = False

# Try Method 1: Local with resume capability
try:
    print("\n[Method 1] Loading XLM-RoBERTa locally...")
    
    import torch
    from transformers import XLMRobertaTokenizer, XLMRobertaModel
    
    # Set cache directory
    cache_dir = '/kaggle/working/models_cache/xlm'
    os.makedirs(cache_dir, exist_ok=True)
    
    print("   Downloading tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        'FacebookAI/xlm-roberta-base',
        cache_dir=cache_dir,
        resume_download=True,
        force_download=False
    )
    
    print("   Downloading model (this may take 2-3 min)...")
    model = XLMRobertaModel.from_pretrained(
        'FacebookAI/xlm-roberta-base',
        cache_dir=cache_dir,
        resume_download=True,
        force_download=False
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"âœ“ Model loaded on {device}")
    
    # Extract embeddings
    def get_embeddings_xlm(texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, 
                             max_length=128, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_emb)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"      {i+len(batch)}/{len(texts)} processed...")
        
        return np.vstack(embeddings)
    
    print("\n   Extracting embeddings...")
    train_emb = get_embeddings_xlm(X_train.tolist())
    val_emb = get_embeddings_xlm(X_val.tolist())
    test_emb = get_embeddings_xlm(test_df['text_clean'].tolist())
    
    print(f"âœ“ Embeddings: train={train_emb.shape}, val={val_emb.shape}")
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, 
                            class_weight='balanced', C=1.0)
    clf.fit(train_emb, y_train)
    
    y_train_proba = clf.predict_proba(train_emb)[:, 1]
    y_val_proba = clf.predict_proba(val_emb)[:, 1]
    test_proba = clf.predict_proba(test_emb)[:, 1]
    
    xlm_success = True
    print("âœ“ LOCAL METHOD SUCCESS")
    
except Exception as e:
    print(f"\nâ�Œ Local method failed: {str(e)[:150]}")
    print("\n[Method 2] Trying HF Inference API...")
    
    # Try Method 2: HF Inference API
    try:
        if "HF_TOKEN" not in os.environ:
            raise ValueError("HF_TOKEN not found. Add it in Kaggle Secrets")
        
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=os.environ["HF_TOKEN"])
        
        def get_api_embeddings(texts, batch_size=5):
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    result = client.feature_extraction(
                        text[:512],
                        model="FacebookAI/xlm-roberta-base"
                    )
                    embeddings.append(np.mean(result, axis=0))
                except:
                    embeddings.append(np.zeros(768))
                
                if (i + 1) % 50 == 0:
                    print(f"      {i+1}/{len(texts)} via API...")
                    time.sleep(1)  # Rate limit
            
            return np.array(embeddings)
        
        print("   Getting embeddings via API (slower)...")
        train_emb = get_api_embeddings(X_train.tolist())
        val_emb = get_api_embeddings(X_val.tolist())
        test_emb = get_api_embeddings(test_df['text_clean'].tolist())
        
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, 
                                class_weight='balanced')
        clf.fit(train_emb, y_train)
        
        y_train_proba = clf.predict_proba(train_emb)[:, 1]
        y_val_proba = clf.predict_proba(val_emb)[:, 1]
        test_proba = clf.predict_proba(test_emb)[:, 1]
        
        xlm_success = True
        print("âœ“ API METHOD SUCCESS")
        
    except Exception as e2:
        print(f"\nâ�Œ API method also failed: {str(e2)[:150]}")
        print("   âš ï¸� XLM-RoBERTa skipped")
        
        y_train_proba = np.zeros(len(y_train))
        y_val_proba = np.zeros(len(y_val))
        test_proba = np.zeros(len(test_df))

# Save XLM results
if xlm_success:
    y_train_pred = (y_train_proba > 0.5).astype(int)
    y_val_pred = (y_val_proba > 0.5).astype(int)
    
    results.append({
        'Model': 'XLM-RoBERTa-base',
        'Train_AUC': roc_auc_score(y_train, y_train_proba),
        'Val_AUC': roc_auc_score(y_val, y_val_proba),
        'Val_F1': f1_score(y_val, y_val_pred),
        'Precision': precision_score(y_val, y_val_pred),
        'Recall': recall_score(y_val, y_val_pred),
        'Time': time.time() - model1_start,
        'predictions': test_proba,
        'val_proba': y_val_proba
    })
    
    print(f"\nâœ… XLM-RoBERTa Results:")
    print(f"   Val AUC: {results[-1]['Val_AUC']:.4f}")
    print(f"   Val F1:  {results[-1]['Val_F1']:.4f}")
    print(f"   Time:    {results[-1]['Time']/60:.1f} min")


# ============================================================================
# MODEL 2: DeBERTa-v3 (Try Large, Fallback to Base)
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 2: DeBERTa-v3")
print("=" * 80)

model2_start = time.time()
deberta_success = False
deberta_model_name = None  # Initialize

# Try DeBERTa-v3-LARGE first
try:
    print("\n[Method 1] Loading DeBERTa-v3-LARGE locally...")
    
    import torch
    from transformers import DebertaV2Tokenizer, DebertaV2Model
    
    cache_dir = '/kaggle/working/models_cache/deberta_large'
    os.makedirs(cache_dir, exist_ok=True)
    
    print("   Downloading tokenizer...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(
        'microsoft/deberta-v3-large',
        cache_dir=cache_dir,
        resume_download=True,
        force_download=False
    )
    
    print("   Downloading model (this may take 5-7 min)...")
    model = DebertaV2Model.from_pretrained(
        'microsoft/deberta-v3-large',
        cache_dir=cache_dir,
        resume_download=True,
        force_download=False
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"âœ“ Model loaded on {device}")
    
    # Extract embeddings
    def get_embeddings_deberta(texts, batch_size=8):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, 
                             max_length=256, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_emb)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"      {i+len(batch)}/{len(texts)} processed...")
        
        return np.vstack(embeddings)
    
    print("\n   Extracting embeddings...")
    train_emb = get_embeddings_deberta(X_train.tolist())
    val_emb = get_embeddings_deberta(X_val.tolist())
    test_emb = get_embeddings_deberta(test_df['text_clean'].tolist())
    
    print(f"âœ“ Embeddings: train={train_emb.shape}, val={val_emb.shape}")
    
    # Train classifier
    clf = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED, 
        class_weight='balanced',
        C=1.0,
        solver='lbfgs'
    )
    clf.fit(train_emb, y_train)
    
    y_train_proba = clf.predict_proba(train_emb)[:, 1]
    y_val_proba = clf.predict_proba(val_emb)[:, 1]
    test_proba = clf.predict_proba(test_emb)[:, 1]
    
    deberta_success = True
    deberta_model_name = 'DeBERTa-v3-LARGE'
    print("âœ“ LARGE MODEL SUCCESS")
    
except Exception as e:
    print(f"\nâ�Œ DeBERTa-LARGE failed: {str(e)[:150]}")
    print("\n[Method 2] Falling back to DeBERTa-v3-BASE...")
    
    # FALLBACK TO BASE MODEL (properly indented inside except block)
    try:
        import torch
        from transformers import DebertaV2Tokenizer, DebertaV2Model
        
        cache_dir = '/kaggle/working/models_cache/deberta_base'
        os.makedirs(cache_dir, exist_ok=True)
        
        print("   Downloading tokenizer...")
        tokenizer = DebertaV2Tokenizer.from_pretrained(
            'microsoft/deberta-v3-base',
            cache_dir=cache_dir,
            resume_download=True,
            force_download=False
        )
        
        print("   Downloading model (this may take 2-3 min)...")
        model = DebertaV2Model.from_pretrained(
            'microsoft/deberta-v3-base',
            cache_dir=cache_dir,
            resume_download=True,
            force_download=False
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        print(f"âœ“ BASE model loaded on {device}")
        
        # Extract embeddings
        def get_embeddings_deberta_base(texts, batch_size=16):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, 
                                 max_length=128, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(batch_emb)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"      {i+len(batch)}/{len(texts)} processed...")
            
            return np.vstack(embeddings)
        
        print("\n   Extracting embeddings...")
        train_emb = get_embeddings_deberta_base(X_train.tolist())
        val_emb = get_embeddings_deberta_base(X_val.tolist())
        test_emb = get_embeddings_deberta_base(test_df['text_clean'].tolist())
        
        print(f"âœ“ Embeddings: train={train_emb.shape}, val={val_emb.shape}")
        
        # Train classifier
        clf = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED, 
            class_weight='balanced',
            C=1.0
        )
        clf.fit(train_emb, y_train)
        
        y_train_proba = clf.predict_proba(train_emb)[:, 1]
        y_val_proba = clf.predict_proba(val_emb)[:, 1]
        test_proba = clf.predict_proba(test_emb)[:, 1]
        
        deberta_success = True
        deberta_model_name = 'DeBERTa-v3-base'
        print("âœ“ BASE MODEL SUCCESS")
        
    except Exception as e2:
        print(f"\nâ�Œ DeBERTa-BASE also failed: {str(e2)[:150]}")
        print("   âš ï¸� Skipping DeBERTa entirely")
        deberta_success = False

# Save DeBERTa results (if successful)
if deberta_success:
    y_train_pred = (y_train_proba > 0.5).astype(int)
    y_val_pred = (y_val_proba > 0.5).astype(int)
    
    results.append({
        'Model': deberta_model_name,
        'Train_AUC': roc_auc_score(y_train, y_train_proba),
        'Val_AUC': roc_auc_score(y_val, y_val_proba),
        'Val_F1': f1_score(y_val, y_val_pred),
        'Precision': precision_score(y_val, y_val_pred),
        'Recall': recall_score(y_val, y_val_pred),
        'Time': time.time() - model2_start,
        'predictions': test_proba,
        'val_proba': y_val_proba
    })
    
    print(f"\nâœ… {deberta_model_name} Results:")
    print(f"   Val AUC: {results[-1]['Val_AUC']:.4f}")
    print(f"   Val F1:  {results[-1]['Val_F1']:.4f}")
    print(f"   Time:    {results[-1]['Time']/60:.1f} min")
else:
    print("\nâš ï¸� DeBERTa skipped - continuing with other models")
    
# ============================================================================
# MODEL 3: TF-IDF Baseline (CRITICAL - Best performing!)
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 3: TF-IDF + Logistic Regression")
print("=" * 80)

model3_start = time.time()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased from 5000
        ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents='unicode',
        lowercase=True,
        analyzer='word'
    )
    
    train_tfidf = vectorizer.fit_transform(X_train)
    val_tfidf = vectorizer.transform(X_val)
    test_tfidf = vectorizer.transform(test_df['text_clean'])
    
    clf = LogisticRegression(
        max_iter=2000,  # Increased
        random_state=RANDOM_SEED,
        class_weight='balanced',
        C=2.0,  # Higher regularization
        solver='saga'  # Better for large datasets
    )
    clf.fit(train_tfidf, y_train)
    
    y_train_proba = clf.predict_proba(train_tfidf)[:, 1]
    y_val_proba = clf.predict_proba(val_tfidf)[:, 1]
    test_proba = clf.predict_proba(test_tfidf)[:, 1]
    
    y_train_pred = (y_train_proba > 0.5).astype(int)
    y_val_pred = (y_val_proba > 0.5).astype(int)
    
    results.append({
        'Model': 'TF-IDF Enhanced',
        'Train_AUC': roc_auc_score(y_train, y_train_proba),
        'Val_AUC': roc_auc_score(y_val, y_val_proba),
        'Val_F1': f1_score(y_val, y_val_pred),
        'Precision': precision_score(y_val, y_val_pred),
        'Recall': recall_score(y_val, y_val_pred),
        'Time': time.time() - model3_start,
        'predictions': test_proba,
        'val_proba': y_val_proba
    })
    
    print(f"âœ“ TF-IDF trained in {results[-1]['Time']:.1f}s")
    print(f"   Val AUC: {results[-1]['Val_AUC']:.4f}")
    print(f"   Val F1:  {results[-1]['Val_F1']:.4f}")
    
except Exception as e:
    print(f"â�Œ TF-IDF failed: {e}")

# ============================================================================
# ENSEMBLE & SUBMISSION
# ============================================================================

print("\n" + "=" * 80)
print("CREATING ENSEMBLE & SUBMISSION")
print("=" * 80)

if len(results) == 0:
    print("â�Œ All models failed! Check internet connection and try again.")
    raise RuntimeError("No models trained successfully")

# Sort by validation AUC
results_df = pd.DataFrame([{k: v for k, v in r.items() 
                            if k not in ['predictions', 'val_proba']} 
                           for r in results])
results_df = results_df.sort_values('Val_AUC', ascending=False)

print("\nğŸ“Š Model Performance:")
print(results_df[['Model', 'Val_AUC', 'Val_F1', 'Precision', 'Recall', 'Time']].to_string(index=False))

# Create ensemble if multiple models succeeded
if len(results) >= 2:
    print("\n[Creating Power-Weighted Ensemble]")
    
    # Extract validation AUCs
    val_aucs = np.array([r['Val_AUC'] for r in results])
    
    # Strategy 1: Power weighting (emphasize best models)
    power = 3  # Cube the AUC scores
    weights_power = np.power(val_aucs, power)
    weights_power = weights_power / weights_power.sum()
    
    # Strategy 2: Rank-based weighting
    ranks = np.argsort(np.argsort(val_aucs)) + 1
    weights_rank = ranks / ranks.sum()
    
    # Strategy 3: Original AUC weighting
    weights_auc = val_aucs / val_aucs.sum()
    
    print("\nğŸ“Š Weighting Strategies:")
    print(f"{'Model':<30} {'Val AUC':<12} {'PowerÂ³':<12} {'Rank':<12} {'AUC':<12}")
    print("-" * 78)
    for i, r in enumerate(results):
        print(f"{r['Model']:<30} {val_aucs[i]:<12.4f} {weights_power[i]:<12.4f} {weights_rank[i]:<12.4f} {weights_auc[i]:<12.4f}")
    
    # Test all strategies
    strategies = {
        'PowerÂ³': weights_power,
        'Rank': weights_rank,
        'AUC': weights_auc
    }
    
    best_strategy = None
    best_val_auc = 0
    
    for strategy_name, weights in strategies.items():
        ensemble_val_proba = sum(w * r['val_proba'] for w, r in zip(weights, results))
        ensemble_val_auc = roc_auc_score(y_val, ensemble_val_proba)
        
        print(f"\n   {strategy_name} Ensemble Val AUC: {ensemble_val_auc:.4f}")
        
        if ensemble_val_auc > best_val_auc:
            best_val_auc = ensemble_val_auc
            best_strategy = (strategy_name, weights)
    
    # Use best strategy
    strategy_name, weights = best_strategy
    ensemble_proba = sum(w * r['predictions'] for w, r in zip(weights, results))
    
    print(f"\nâœ“ Using {strategy_name} weighting (Val AUC: {best_val_auc:.4f})")
    
    # Compare to best single model
    best_single_auc = max(r['Val_AUC'] for r in results)
    
    if best_val_auc >= best_single_auc:
        final_predictions = ensemble_proba
        model_name = f"{strategy_name} Ensemble"
        print(f"   â†’ Ensemble wins! (+{best_val_auc - best_single_auc:.4f})")
    else:
        # Use best single model
        best_idx = np.argmax([r['Val_AUC'] for r in results])
        final_predictions = results[best_idx]['predictions']
        model_name = results[best_idx]['Model']
        print(f"   â†’ Using best single: {model_name}")
    
elif len(results) == 1:
    final_predictions = results[0]['predictions']
    model_name = results[0]['Model']
    print(f"\nâœ“ Using single model: {model_name}")
else:
    raise RuntimeError("No models succeeded")

# Create submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'target': final_predictions
})

submission.to_csv('submission.csv', index=False)

print("\n" + "=" * 80)
print("âœ… SUBMISSION CREATED")
print("=" * 80)
print(f"   File: submission.csv")
print(f"   Rows: {len(submission)}")
print(f"   Model: {model_name}")
print(f"   Target range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
print(f"   Total time: {(time.time() - start_time)/60:.1f} minutes")
print("=" * 80)

# Display summary statistics
print("\nğŸ“ˆ Prediction Statistics:")
print(f"   Mean: {final_predictions.mean():.4f}")
print(f"   Std:  {final_predictions.std():.4f}")
print(f"   Distribution:")
print(f"   â€¢ < 0.3 (likely benign):   {(final_predictions < 0.3).sum()}")
print(f"   â€¢ 0.3-0.7 (uncertain):     {((final_predictions >= 0.3) & (final_predictions <= 0.7)).sum()}")
print(f"   â€¢ > 0.7 (likely jailbreak): {(final_predictions > 0.7).sum()}")

print("\nâœ¨ Submission ready for upload!")


