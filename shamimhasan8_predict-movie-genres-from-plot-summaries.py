# ============================================================================
# ADVANCED MOVIE GENRE PREDICTION - COMPLETE WINNING SOLUTION
# ============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn. model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

import json
from collections import Counter, defaultdict
from itertools import combinations
from scipy.sparse import hstack, csr_matrix
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

train_df = pd.read_csv('/kaggle/input/predict-movie-genres-from-plot-summaries/train.csv')
test_df = pd.read_csv('/kaggle/input/predict-movie-genres-from-plot-summaries/test.csv')
genres_df = pd.read_csv('/kaggle/input/predict-movie-genres-from-plot-summaries/movies_genres.csv')

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Total genres: {len(genres_df)}")

# ============================================================================
# SECTION 2: DATA PREPROCESSING & PARSING
# ============================================================================

print("\n" + "=" * 80)
print("PARSING GENRE IDS")
print("=" * 80)

def parse_genre_ids(genre_str):
    """Parse genre IDs from string format"""
    if pd.isna(genre_str):
        return []
    genre_str = str(genre_str).strip()
    if genre_str.startswith('['):
        genre_str = genre_str. replace('[', '').replace(']', '').replace("'", "")
    try:
        return [int(g. strip()) for g in genre_str. split() if g.strip().isdigit()]
    except:
        return []

train_df['genre_list'] = train_df['genre_ids'].apply(parse_genre_ids)
train_df['num_genres'] = train_df['genre_list'].apply(len)

print(f"Average genres per movie: {train_df['num_genres'].mean():.2f}")
print(f"Movies with at least one genre: {(train_df['num_genres'] > 0).sum()}")

# ============================================================================
# SECTION 3: ADVANCED TEXT PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("TEXT PREPROCESSING")
print("=" * 80)

class AdvancedTextPreprocessor: 
    """Advanced text preprocessing with domain-specific features"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = text.replace('"', ' ').replace("'", ' ')
        
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def extract_linguistic_features(self, text):
        """Extract linguistic patterns"""
        if pd.isna(text):
            return {}
        
        sentences = sent_tokenize(str(text))
        tokens = word_tokenize(str(text).lower())
        
        features = {
            'sentence_count': len(sentences),
            'avg_sentence_length': len(tokens) / max(len(sentences), 1),
            'word_count': len(tokens),
            'dialogue_presence': str(text).count('"'),
            'exclamation_count': str(text).count('!'),
            'question_count': str(text).count('?'),
            'quote_count': str(text).count('"') + str(text).count("'"),
            'uppercase_words': sum(1 for word in str(text).split() if word.isupper()),
        }
        
        return features

preprocessor = AdvancedTextPreprocessor()

print("Preprocessing training data...")
train_df['overview_processed'] = train_df['overview']. apply(preprocessor.preprocess)
test_df['overview_processed'] = test_df['overview'].apply(preprocessor.preprocess)

# Extract linguistic features
print("Extracting linguistic features...")
linguistic_features_train = train_df['overview']. apply(preprocessor.extract_linguistic_features)
linguistic_features_test = test_df['overview']. apply(preprocessor.extract_linguistic_features)

linguistic_df_train = pd.DataFrame(list(linguistic_features_train))
linguistic_df_test = pd.DataFrame(list(linguistic_features_test))

print(f"Linguistic features shape: {linguistic_df_train. shape}")

# ============================================================================
# SECTION 4: MULTI-LEVEL FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VECTORIZED FEATURES")
print("=" * 80)

# TF-IDF with bigrams
print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    stop_words='english'
)

X_tfidf_train = tfidf_vectorizer.fit_transform(train_df['overview_processed'])
X_tfidf_test = tfidf_vectorizer.transform(test_df['overview_processed'])
print(f"TF-IDF shape: {X_tfidf_train.shape}")

# Character n-grams
print("Creating character n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 4),
    max_features=2000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_char_train = char_vectorizer.fit_transform(train_df['overview'])
X_char_test = char_vectorizer.transform(test_df['overview'])
print(f"Character n-gram shape: {X_char_train.shape}")

# Word-level n-grams (bigrams and trigrams)
print("Creating word n-gram features...")
ngram_vectorizer = TfidfVectorizer(
    ngram_range=(2, 3),
    max_features=3000,
    min_df=2,
    max_df=0.90,
    sublinear_tf=True,
    stop_words='english'
)

X_ngram_train = ngram_vectorizer. fit_transform(train_df['overview_processed'])
X_ngram_test = ngram_vectorizer.transform(test_df['overview_processed'])
print(f"Word n-gram shape: {X_ngram_train.shape}")

# Combine all features
print("Combining features...")
features_train = [X_tfidf_train, X_char_train, X_ngram_train]
features_test = [X_tfidf_test, X_char_test, X_ngram_test]

# Add linguistic features
if linguistic_df_train.shape[1] > 0:
    scaler = StandardScaler()
    linguistic_scaled_train = scaler.fit_transform(linguistic_df_train. values)
    linguistic_scaled_test = scaler.transform(linguistic_df_test.values)
    features_train.append(csr_matrix(linguistic_scaled_train))
    features_test.append(csr_matrix(linguistic_scaled_test))

X_train_combined = hstack(features_train, format='csr')
X_test_combined = hstack(features_test, format='csr')

print(f"Total combined features - Train: {X_train_combined. shape}, Test: {X_test_combined.shape}")

# ============================================================================
# SECTION 5: MULTI-LABEL PREPARATION
# ============================================================================

print("\n" + "=" * 80)
print("PREPARING MULTI-LABEL TARGET")
print("=" * 80)

mlb = MultiLabelBinarizer(classes=sorted(genres_df['id'].unique()))
y_train = mlb.fit_transform(train_df['genre_list'])

print(f"Multi-label matrix shape: {y_train.shape}")
print(f"Genre distribution:")
for idx, genre_id in enumerate(mlb.classes_):
    genre_name = genres_df[genres_df['id'] == genre_id]['name'].values[0]
    count = int(y_train[: , idx].sum())
    print(f"{genre_name:20s}:{count:5d}samples")

# ============================================================================
# SECTION 6: TRAIN-VALIDATION SPLIT WITH STRATIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("SPLITTING DATA")
print("=" * 80)

def create_stratification_key(genre_list):
    if len(genre_list) == 0:
        return 'none'
    return '_'.join(map(str, sorted(genre_list)[: 2]))

strat_keys = train_df['genre_list'].apply(create_stratification_key)

X_train, X_val, y_train_split, y_val, idx_train, idx_val = train_test_split(
    X_train_combined, y_train, range(len(train_df)),
    test_size=0.20,
    random_state=42,
    stratify=strat_keys
)

print(f"Training set:  {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# ============================================================================
# SECTION 7: ENSEMBLE MODEL TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING ENSEMBLE MODELS")
print("=" * 80)

models = {}

# Model 1: Logistic Regression
print("\n[1/5] Training Logistic Regression...")
lr_model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=2000,
        C=0.5,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    ),
    n_jobs=-1
)
lr_model.fit(X_train, y_train_split)
y_val_lr = lr_model.predict_proba(X_val)
y_test_lr = lr_model.predict_proba(X_test_combined)
models['lr'] = (y_val_lr, y_test_lr)
print("✓ Logistic Regression trained")

# Model 2: Linear SVM
print("[2/5] Training Linear SVM...")
svm_model = OneVsRestClassifier(
    LinearSVC(
        max_iter=3000,
        C=0.3,
        class_weight='balanced',
        random_state=42,
        dual=False,
        verbose=0
    ),
    n_jobs=-1
)
svm_model.fit(X_train, y_train_split)
y_val_svm = svm_model. decision_function(X_val)
y_test_svm = svm_model.decision_function(X_test_combined)
scaler_svm = StandardScaler()
y_val_svm = scaler_svm.fit_transform(y_val_svm)
y_test_svm = scaler_svm. transform(y_test_svm)
y_val_svm = np.clip(y_val_svm, 0, 1)
y_test_svm = np.clip(y_test_svm, 0, 1)
models['svm'] = (y_val_svm, y_test_svm)
print("✓ Linear SVM trained")

# Model 3: Random Forest
print("[3/5] Training Random Forest...")
rf_model = OneVsRestClassifier(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    n_jobs=-1
)
rf_model.fit(X_train, y_train_split)
y_val_rf = rf_model. predict_proba(X_val)
y_test_rf = rf_model.predict_proba(X_test_combined)
models['rf'] = (y_val_rf, y_test_rf)
print("✓ Random Forest trained")

# Model 4: XGBoost
print("[4/5] Training XGBoost...")
xgb_model = OneVsRestClassifier(
    xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=5,
        tree_method='hist',
        verbosity=0
    ),
    n_jobs=-1
)
xgb_model.fit(X_train, y_train_split)
y_val_xgb = xgb_model.predict_proba(X_val)
y_test_xgb = xgb_model.predict_proba(X_test_combined)
models['xgb'] = (y_val_xgb, y_test_xgb)
print("✓ XGBoost trained")

# Model 5: LightGBM
print("[5/5] Training LightGBM...")
lgb_model = OneVsRestClassifier(
    lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.08,
        num_leaves=40,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        is_unbalance=True,
        verbose=-1
    ),
    n_jobs=-1
)
lgb_model.fit(X_train, y_train_split)
y_val_lgb = lgb_model.predict_proba(X_val)
y_test_lgb = lgb_model.predict_proba(X_test_combined)
models['lgb'] = (y_val_lgb, y_test_lgb)
print("✓ LightGBM trained")

# ============================================================================
# SECTION 8: COMPUTE INTELLIGENT ENSEMBLE WEIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING ENSEMBLE WEIGHTS")
print("=" * 80)

def calculate_per_genre_f1(y_pred_proba, y_true, threshold=0.5):
    """Calculate F1 score for each genre"""
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    f1_scores = []
    
    for genre_idx in range(y_true.shape[1]):
        f1 = f1_score(y_true[:, genre_idx], y_pred_binary[:, genre_idx], zero_division=0)
        f1_scores.append(f1)
    
    return np.array(f1_scores)

# Calculate F1 scores for each model
f1_scores_all = {}
for model_name, (y_val_pred, _) in models.items():
    f1_scores_all[model_name] = calculate_per_genre_f1(y_val_pred, y_val, threshold=0.5)

# Compute genre-specific weights based on F1 scores
genre_weights = np.zeros((len(mlb.classes_), 5))
model_names_list = ['lr', 'svm', 'rf', 'xgb', 'lgb']

for genre_idx in range(len(mlb.classes_)):
    f1_scores = np.array([f1_scores_all[name][genre_idx] for name in model_names_list])
    total_f1 = f1_scores.sum()
    
    if total_f1 > 0:
        genre_weights[genre_idx] = f1_scores / total_f1
    else:
        # Equal weights if all F1 scores are zero
        genre_weights[genre_idx] = np.ones(5) / 5

print("Ensemble weights computed per genre")

# ============================================================================
# SECTION 9: CREATE WEIGHTED ENSEMBLE PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING WEIGHTED ENSEMBLE")
print("=" * 80)

y_val_ensemble = np.zeros_like(models['lr'][0], dtype=np.float32)
y_test_ensemble = np.zeros_like(models['lr'][1], dtype=np.float32)

for genre_idx in range(len(mlb.classes_)):
    weights = genre_weights[genre_idx]
    
    y_val_ensemble[: , genre_idx] = (
        weights[0] * models['lr'][0][:, genre_idx] +
        weights[1] * models['svm'][0][:, genre_idx] +
        weights[2] * models['rf'][0][:, genre_idx] +
        weights[3] * models['xgb'][0][:, genre_idx] +
        weights[4] * models['lgb'][0][:, genre_idx]
    )
    
    y_test_ensemble[:, genre_idx] = (
        weights[0] * models['lr'][1][:, genre_idx] +
        weights[1] * models['svm'][1][: , genre_idx] +
        weights[2] * models['rf'][1][:, genre_idx] +
        weights[3] * models['xgb'][1][:, genre_idx] +
        weights[4] * models['lgb'][1][:, genre_idx]
    )

print(f"Ensemble predictions - Val: {y_val_ensemble. shape}, Test: {y_test_ensemble.shape}")

# ============================================================================
# SECTION 10: THRESHOLD OPTIMIZATION (CRITICAL!)
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZING THRESHOLDS FOR MACRO F1")
print("=" * 80)

def macro_f1_objective(thresholds, y_val_proba, y_val_true):
    """Calculate negative macro F1 score for optimization"""
    f1_scores = []
    
    for genre_idx in range(y_val_true.shape[1]):
        y_pred = (y_val_proba[: , genre_idx] >= thresholds[genre_idx]).astype(int)
        y_true = y_val_true[:, genre_idx]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    return -np.mean(f1_scores)  # Negative because we're minimizing

initial_thresholds = np.full(len(mlb.classes_), 0.5)

result = minimize(
    lambda t: macro_f1_objective(t, y_val_ensemble, y_val),
    initial_thresholds,
    method='Nelder-Mead',
    options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5}
)

optimal_thresholds = result.x
optimal_macro_f1 = -result.fun

print(f"✓ Optimization complete!")
print(f"Optimal Macro F1 (Validation): {optimal_macro_f1:.4f}")
print(f"\nOptimal thresholds per genre:")
for idx, genre_id in enumerate(mlb.classes_):
    genre_name = genres_df[genres_df['id'] == genre_id]['name'].values[0]
    print(f"{genre_name:20s}:{optimal_thresholds[idx]:.3f}")

# ============================================================================
# SECTION 11: GENERATE FINAL PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING FINAL PREDICTIONS")
print("=" * 80)

y_test_binary = np.zeros_like(y_test_ensemble, dtype=int)

for genre_idx in range(len(mlb.classes_)):
    y_test_binary[:, genre_idx] = (y_test_ensemble[:, genre_idx] >= optimal_thresholds[genre_idx]).astype(int)

# Convert predictions back to genre IDs
predictions = []
for movie_idx in range(len(test_df)):
    predicted_genres = []
    
    for genre_idx, genre_id in enumerate(mlb. classes_):
        if y_test_binary[movie_idx, genre_idx] == 1:
            predicted_genres.append(str(int(genre_id)))
    
    predictions.append(' '.join(predicted_genres) if predicted_genres else '0')

test_df['genre_ids'] = predictions

# ============================================================================
# SECTION 12: CREATE SUBMISSION FILE
# ============================================================================

print("\n" + "=" * 80)
print("CREATING SUBMISSION FILE")
print("=" * 80)

submission_df = test_df[['movie_id', 'genre_ids']].copy()
submission_df. to_csv('submission.csv', index=False)

print("✓ Submission file saved: submission.csv")
print(f"\nSample predictions:")
print(submission_df.head(20))

# ============================================================================
# SECTION 13: VALIDATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION METRICS")
print("=" * 80)

f1_per_genre = []
precision_per_genre = []
recall_per_genre = []

print("\nPer-Genre Performance:")
for genre_idx, genre_id in enumerate(mlb.classes_):
    y_pred = (y_val_ensemble[:, genre_idx] >= optimal_thresholds[genre_idx]).astype(int)
    y_true = y_val[: , genre_idx]
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    f1_per_genre.append(f1)
    precision_per_genre.append(precision)
    recall_per_genre. append(recall)
    
    genre_name = genres_df[genres_df['id'] == genre_id]['name'].values[0]
    print(f"{genre_name:20s}|F1:{f1:.3f}|Precision:{precision:.3f}|Recall:{recall:.3f}")

macro_f1_final = np.mean(f1_per_genre)
macro_precision = np.mean(precision_per_genre)
macro_recall = np.mean(recall_per_genre)

print(f"\n{'='*70}")
print(f"MACRO F1:{macro_f1_final:.4f}|Precision:{macro_precision:.4f}|Recall:{macro_recall:.4f}")
print(f"{'='*70}")

print("\n✓ Solution complete!")

