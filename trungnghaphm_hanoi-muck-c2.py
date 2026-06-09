# ============================================================
# ğŸ�¯ OFFLINE 0.97+ TARGET | Runtime: ~10-12 mins
# Strategy: 30+ Jailbreak Features + Expanded TF-IDF + Ensemble
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    raise ImportError("XGBoost required! Install: pip install xgboost")

# ============================================================
# ğŸ“� Text Preprocessing
# ============================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================
# ğŸ”§ ENHANCED JAILBREAK FEATURES (30+ features)
# ============================================================
def extract_advanced_features(df, text_col='text'):
    """Extract comprehensive jailbreak detection features"""
    features = pd.DataFrame()
    
    # Basic stats
    features['text_len'] = df[text_col].str.len()
    features['word_count'] = df[text_col].str.split().str.len()
    features['avg_word_len'] = features['text_len'] / (features['word_count'] + 1)
    features['char_word_ratio'] = features['text_len'] / (features['word_count'] + 1)
    
    # CRITICAL JAILBREAK PATTERNS
    features['roleplay'] = df[text_col].str.contains(
        r"\b(pretend|imagine|roleplay|act as|you are now|you're now)\b", 
        case=False, na=False, regex=True
    ).astype(int)
    
    features['instruction_override'] = df[text_col].str.contains(
        r"\b(ignore|forget|disregard|override|bypass|disable)\b.*\b(previous|instructions|rules|guidelines|constraints)\b",
        case=False, na=False, regex=True
    ).astype(int)
    
    features['system_prompt'] = df[text_col].str.contains(
        r"\b(system prompt|system message|original instructions|base instructions)\b",
        case=False, na=False, regex=True
    ).astype(int)
    
    features['hypothetical'] = df[text_col].str.contains(
        r"\b(hypothetically|in theory|what if|suppose|assuming)\b",
        case=False, na=False, regex=True
    ).astype(int)
    
    features['permission'] = df[text_col].str.contains(
        r"\b(can you|could you|would you|please help|assist me)\b",
        case=False, na=False, regex=True
    ).astype(int)
    
    features['code_mode'] = df[text_col].str.contains(
        r"\b(code mode|developer mode|debug mode|admin mode|root access)\b",
        case=False, na=False, regex=True
    ).astype(int)
    
    features['jailbreak_terms'] = df[text_col].str.contains(
        r"\b(jailbreak|dan|do anything now|unrestricted|uncensored)\b",
        case=False, na=False, regex=True
    ).astype(int)
    
    # Character patterns
    features['upper_count'] = df[text_col].str.count(r'[A-Z]')
    features['upper_ratio'] = features['upper_count'] / (features['text_len'] + 1)
    features['digit_count'] = df[text_col].str.count(r'\d')
    features['digit_ratio'] = features['digit_count'] / (features['text_len'] + 1)
    
    # Punctuation
    features['exclamation'] = df[text_col].str.count('!')
    features['question'] = df[text_col].str.count(r'\?')
    features['period'] = df[text_col].str.count(r'\.')
    features['comma'] = df[text_col].str.count(',')
    features['punctuation_total'] = df[text_col].str.count(r'[.,;:!?]')
    features['punctuation_ratio'] = features['punctuation_total'] / (features['text_len'] + 1)
    features['caps_words'] = df[text_col].str.findall(r'\b[A-Z]{2,}\b').str.len()
    features['repeated_chars'] = df[text_col].str.count(r'(.)\1{2,}')
    
    # Special chars
    features['mention_count'] = df[text_col].str.count(r'@')
    features['hashtag_count'] = df[text_col].str.count(r'#')
    features['url_count'] = df[text_col].str.count(r'http|www')
    
    # Diversity
    features['unique_ratio'] = df[text_col].apply(
        lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1)
    )
    features['stopword_ratio'] = df[text_col].str.count(r'\b(the|a|an|in|on|at|to|for)\b') / (features['word_count'] + 1)
    
    # Sentence structure
    features['sentence_count'] = df[text_col].str.count(r'[.!?]+') + 1
    features['avg_sentence_len'] = features['word_count'] / features['sentence_count']
    
    # Toxicity
    features['negative_words'] = df[text_col].str.count(r'\b(hate|kill|bad|worst|terrible|awful|stupid)\b')
    features['profanity_proxy'] = df[text_col].str.count(r'[*#@]{2,}')
    
    features = features.fillna(0)
    return features

# ============================================================
# 1ï¸�âƒ£ Load Data
# ============================================================
print("ğŸ“‚ Loading data...")
train = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")
test = pd.read_csv("/kaggle/input/rmit-hackathon-2025/test.csv")
sample = pd.read_csv("/kaggle/input/rmit-hackathon-2025/sample_submission.csv")

print(f"Train: {train.shape} | Test: {test.shape}")
print(f"Labels: {train['label'].value_counts(normalize=True).to_dict()}")

# ============================================================
# ğŸ”„ Encode Labels
# ============================================================
if train["label"].dtype == 'object':
    label_encoder = LabelEncoder()
    train["label_encoded"] = label_encoder.fit_transform(train["label"])
    label_col = "label_encoded"
else:
    label_col = "label"

# ============================================================
# 2ï¸�âƒ£ Feature Engineering
# ============================================================
print("\nğŸ”§ Engineering features...")
train["text_clean"] = train["text"].apply(clean_text)
test["text_clean"] = test["text"].apply(clean_text)

train_features = extract_advanced_features(train, 'text')
test_features = extract_advanced_features(test, 'text')
print(f"   âœ“ Created {train_features.shape[1]} jailbreak features")

# ============================================================
# 3ï¸�âƒ£ Train/Val Split
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    train["text_clean"], train[label_col], 
    test_size=0.2, random_state=42, stratify=train[label_col]
)

feat_train, feat_val = train_test_split(
    train_features, test_size=0.2, random_state=42, stratify=train[label_col]
)

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

# ============================================================
# 4ï¸�âƒ£ EXPANDED TF-IDF (80k features)
# ============================================================
print("\nğŸ“Š Vectorizing text...")

# Word-level TF-IDF (expanded)
word_tfidf = TfidfVectorizer(
    max_features=60000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents='unicode'
)

# Character-level TF-IDF
char_tfidf = TfidfVectorizer(
    max_features=20000,
    analyzer='char',
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True
)

X_train_word = word_tfidf.fit_transform(X_train)
X_val_word = word_tfidf.transform(X_val)
X_train_char = char_tfidf.fit_transform(X_train)
X_val_char = char_tfidf.transform(X_val)

# Combine ALL features
X_train_comb = hstack([
    X_train_word,
    X_train_char,
    csr_matrix(feat_train.values)
])

X_val_comb = hstack([
    X_val_word,
    X_val_char,
    csr_matrix(feat_val.values)
])

print(f"   âœ“ Combined features: {X_train_comb.shape[1]:,}")

# ============================================================
# 5ï¸�âƒ£ TRAIN 3 MODELS (Ensemble)
# ============================================================
print("\nğŸ�¯ Training ensemble...\n")
val_predictions = []
val_scores = []
models = []

# Model 1: Logistic Regression (fast baseline)
print("Training Logistic Regression...")
lr = LogisticRegression(C=2.0, max_iter=500, n_jobs=-1, random_state=42)
lr.fit(X_train_comb, y_train)
lr_val = lr.predict_proba(X_val_comb)[:, 1]
lr_score = roc_auc_score(y_val, lr_val)
print(f"   âœ“ LR Val ROC-AUC: {lr_score:.5f}")
val_predictions.append(lr_val)
val_scores.append(lr_score)
models.append(('LR', lr))

# Model 2: XGBoost (deep & wide)
print("\nTraining XGBoost #1...")
xgb1 = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=9,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1.5,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='auc'
)
xgb1.fit(X_train_comb, y_train, verbose=False)
xgb1_val = xgb1.predict_proba(X_val_comb)[:, 1]
xgb1_score = roc_auc_score(y_val, xgb1_val)
print(f"   âœ“ XGB1 Val ROC-AUC: {xgb1_score:.5f}")
val_predictions.append(xgb1_val)
val_scores.append(xgb1_score)
models.append(('XGB1', xgb1))

# Model 3: XGBoost (shallower, more trees)
print("\nTraining XGBoost #2...")
xgb2 = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=1.0,
    min_child_weight=5,
    reg_alpha=0.2,
    reg_lambda=1.5,
    random_state=43,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='auc'
)
xgb2.fit(X_train_comb, y_train, verbose=False)
xgb2_val = xgb2.predict_proba(X_val_comb)[:, 1]
xgb2_score = roc_auc_score(y_val, xgb2_val)
print(f"   âœ“ XGB2 Val ROC-AUC: {xgb2_score:.5f}")
val_predictions.append(xgb2_val)
val_scores.append(xgb2_score)
models.append(('XGB2', xgb2))

# ============================================================
# 6ï¸�âƒ£ SMART ENSEMBLE WEIGHTING
# ============================================================
# Weight by validation scores (better models get more weight)
total_score = sum(val_scores)
weights = [s / total_score for s in val_scores]

print(f"\nâš–ï¸� Ensemble Weights:")
for (name, _), w, s in zip(models, weights, val_scores):
    print(f"   {name}: {w:.3f} (score: {s:.5f})")

# Calculate weighted ensemble
ensemble_val = sum(w * pred for w, pred in zip(weights, val_predictions))
ensemble_score = roc_auc_score(y_val, ensemble_val)
print(f"\nâœ… ENSEMBLE Val ROC-AUC: {ensemble_score:.5f} ğŸš€")

# ============================================================
# 7ï¸�âƒ£ Train on Full Data & Predict
# ============================================================
print("\nğŸš€ Training on full data...")

X_full_word = word_tfidf.fit_transform(train["text_clean"])
X_test_word = word_tfidf.transform(test["text_clean"])
X_full_char = char_tfidf.fit_transform(train["text_clean"])
X_test_char = char_tfidf.transform(test["text_clean"])

X_full_comb = hstack([X_full_word, X_full_char, csr_matrix(train_features.values)])
X_test_comb = hstack([X_test_word, X_test_char, csr_matrix(test_features.values)])

test_preds = []
for name, model in models:
    print(f"   Training {name}...")
    if 'XGB' in name:
        model.fit(X_full_comb, train[label_col], verbose=False)
    else:
        model.fit(X_full_comb, train[label_col])
    pred = model.predict_proba(X_test_comb)[:, 1]
    test_preds.append(pred)

# Weighted ensemble
final_preds = sum(w * pred for w, pred in zip(weights, test_preds))

print(f"\nâœ… Predictions ready!")
print(f"   Range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
print(f"   Mean: {final_preds.mean():.4f}")


# ============================================================
# âœ… Create Submission
# ============================================================

submission = pd.DataFrame({
    sample.columns[0]: test["Id"],
    sample.columns[1]: final_preds.astype(float)
})

# Validation
assert submission.shape[0] == sample.shape[0], "â�Œ Row mismatch!"
assert not submission.isna().any().any(), "â�Œ NaN found!"

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created âœ…")
print(f"Rows: {len(submission):,}\n")
print(submission.head(10))

