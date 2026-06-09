# Standard library imports
import os
import re
import zlib
import math
from collections import Counter
from difflib import SequenceMatcher

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Machine Learning & Optimization
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Natural Language Processing (NLP)
import nltk
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Quietly download NLTK resources to keep the notebook output clean
for resource in ['stopwords', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'punkt']:
    nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words('english'))


# Pre-compiling Regex patterns for performance
REGISTRY = {
    'numerical': re.compile(r'\b\d+([.,]\d+)?\b'),
    'years': re.compile(r'\b(19[6-9]\d|20[0-3]\d)\b'),
    'acronyms': re.compile(r'\b[A-Z]{2,}\b'),
    'measurements': re.compile(r'\b(km|m|cm|mm|Hz|kHz|MHz|GHz|s|ms|K|째C|째F|AU|AU:)?\b', re.I)
}


def load_dataset(data_dir, real_text_mapping=None, is_train=True):
    """
    Parses directory structure to build a structured DataFrame.
    
    Args:
        data_dir (str): Path to the train or test directory.
        real_text_mapping (pd.DataFrame): The CSV containing labels (only for training).
        is_train (bool): Flag to handle label assignment.
    """
    dataset = []
    
    # Identify article directories and ignore hidden files
    articles = [a for a in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, a))]
    
    for article in articles:
        article_path = os.path.join(data_dir, article)
        f1_path = os.path.join(article_path, 'file_1.txt')
        f2_path = os.path.join(article_path, 'file_2.txt')
        
        # Safe reading with context managers
        try:
            with open(f1_path, 'r', encoding='utf-8') as f1, open(f2_path, 'r', encoding='utf-8') as f2:
                file_1_content = f1.read()
                file_2_content = f2.read()
            
            entry = {
                'article_folder': article,
                'file_1': file_1_content,
                'file_2': file_2_content,
            }
            
            # Map labels if in training mode
            if is_train and real_text_mapping is not None:
                # Extract numerical ID using regex for robustness
                match = re.search(r'\d+', article)
                if match:
                    article_id = int(match.group())
                    # Retrieve the correct file ID (1 or 2) from the mapping
                    label = real_text_mapping.loc[real_text_mapping['id'] == article_id, 'real_text_id'].values[0]
                    entry['real_text_id'] = label
            
            dataset.append(entry)
            
        except FileNotFoundError:
            print(f"Warning: Files missing in {article_path}")
            continue

    return pd.DataFrame(dataset)

# Define paths (Kaggle environment)
BASE_PATH = '/kaggle/input/fake-or-real-the-impostor-hunt/data/'
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')

# Execute Loading
train_mapping = pd.read_csv(TRAIN_CSV)
train_df = load_dataset(os.path.join(BASE_PATH, 'train'), real_text_mapping=train_mapping, is_train=True)
test_df = load_dataset(os.path.join(BASE_PATH, 'test'), is_train=False)

print(f"Data successfully loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}")


train_df.head()


# Isolating the target variable
y = train_df['real_text_id']

# Dropping the target from the feature set
X = train_df.drop(columns=['real_text_id'])

print(f"Feature set shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")


X


def shannon_entropy(tokens):
    """
    Computes the Shannon Entropy of the text to measure information density.
    Higher entropy often suggests a more diverse vocabulary.
    """
    if not tokens:
        return 0.0
    freqs = Counter(tokens)
    total = sum(freqs.values())
    return -sum((c/total) * math.log2(c/total) for c in freqs.values())

def compute_text_stats(text):
    """
    Comprehensive stylometric feature extractor.
    """
    # Type-safety check
    text = text if isinstance(text, str) else ""
    
    # Tokenization
    tokens = word_tokenize(text)
    sents = sent_tokenize(text)
    n_words = len(tokens)
    n_sents = max(1, len(sents))
    chars = len(text)
    
    # 1. Lexical Diversity
    uniques = len(set([w.lower() for w in tokens]))
    ttr = uniques / max(1, n_words) # Type-Token Ratio
    hapax = sum(1 for w, c in Counter(tokens).items() if c == 1)
    
    # 2. Syntax & Punctuation
    punctuation = {
        "n_commas": text.count(','),
        "n_semicol": text.count(';'),
        "n_colon": text.count(':'),
        "n_dash": text.count('-'),
        "n_paren": text.count('(') + text.count(')'),
        "n_quote": text.count('"') + text.count("'"),
        "n_ellipsis": text.count('...')
    }
    
    # 3. Stylistic Ratios
    stop_ratio = sum(1 for w in tokens if w.lower() in STOPWORDS) / max(1, n_words)
    uppercase_ratio = sum(1 for ch in text if ch.isupper()) / max(1, chars)
    
    # 4. Technical Markers (using pre-defined REGISTRY)
    n_numbers = len(REGISTRY['numerical'].findall(text))
    n_years = len(REGISTRY['years'].findall(text))
    allcaps_words = len(REGISTRY['acronyms'].findall(text))
    n_measure = len(REGISTRY['measurements'].findall(text))
    
    # 5. Information Theory & Complexity
    # Using zlib for compression ratio as a proxy for text complexity
    encoded_text = text.encode('utf-8')
    comp_ratio = len(zlib.compress(encoded_text)) / max(1, len(encoded_text))
    entropy = shannon_entropy([w.lower() for w in tokens])
    
    # 6. Readability Analysis
    try:
        flesch = textstat.flesch_reading_ease(text)
        smog = textstat.smog_index(text)
        ari = textstat.automated_readability_index(text)
    except Exception:
        flesch = smog = ari = 0.0

    # 7. Part-of-Speech (POS) Analysis
    pos_tags = nltk.pos_tag(tokens)
    pos_counts = Counter([p[:2] for w, p in pos_tags]) # Simplified tags (NN, VB, JJ, RB)
    
    # Ratios
    noun_ratio = pos_counts['NN'] / max(1, n_words)
    verb_ratio = pos_counts['VB'] / max(1, n_words)
    adj_ratio  = pos_counts['JJ'] / max(1, n_words)
    adv_ratio  = pos_counts['RB'] / max(1, n_words)

    # 8. Linguistic Hedges (Markers of uncertainty)
    hedges = ["may", "might", "could", "likely", "suggest", "possible", "probable", "estimate"]
    hedge_hits = sum(text.lower().count(h) for h in hedges) / max(1, n_words)

    # Consolidate features
    stats = {
        "n_words": n_words,
        "n_chars": chars,
        "n_sents": n_sents,
        "avg_word_len": np.mean([len(w) for w in tokens]) if tokens else 0.0,
        "ttr": ttr,
        "hapax": hapax,
        "stop_ratio": stop_ratio,
        "uppercase_ratio": uppercase_ratio,
        "comp_ratio": comp_ratio,
        "entropy": entropy,
        "flesch": flesch,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "hedge_hits": hedge_hits,
        **punctuation # Unpacking punctuation dict
    }
    
    return stats


# Splitting the dataset
# We use a stratified approach (implicit here) to maintain class balance
X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Recommended: ensures both sets have the same % of each class
)

print(f"Splitting complete:")
print(f"   - Training samples: {len(X_train)}")
print(f"   - Validation samples: {len(X_val)}")


def prepare_feature_matrix(df):
    """
    Transforms a DataFrame of text pairs into a comparative feature matrix.
    """
    # Extract features for both files
    features_1 = np.array([list(compute_text_stats(t).values()) for t in df["file_1"]])
    features_2 = np.array([list(compute_text_stats(t).values()) for t in df["file_2"]])
    
    # Compute differentials (The "Comparative" trick)
    diff_12 = features_1 - features_2
    diff_21 = features_2 - features_1
    
    # Horizontal stacking of all feature sets
    return np.hstack([features_1, features_2, diff_12, diff_21])

# Processing all splits
print("Transforming text into stylometric matrices...")
X_train_final = prepare_feature_matrix(X_train)
X_val_final   = prepare_feature_matrix(X_val)
X_test_final  = prepare_feature_matrix(test_df)

# Target encoding: 1 if file_1 is real, 0 if file_2 is real
# This creates a clear binary classification objective
y_train_binary = (y_train == 1).astype(int)
y_val_binary   = (y_val == 1).astype(int)

print(f"Final Feature Matrix Shape: {X_train_final.shape}")


from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train_final)

# Transform validation and test data using the training parameters
# DO NOT use fit_transform here to avoid data leakage
X_val_scaled = scaler.transform(X_val_final)
X_test_scaled = scaler.transform(X_test_final)

print("Scaling complete.")
print(f"Mean of first feature (Train): {X_train_scaled[:, 0].mean():.2f}")
print(f"Mean of first feature (Val):   {X_val_scaled[:, 0].mean():.2f}")


import optuna
from sklearn.ensemble import ExtraTreesClassifier

def pairwise_accuracy_score(y_true, y_proba):
    """
    Computes accuracy based on the 0.5 probability threshold.
    """
    y_pred = (y_proba > 0.5).astype(int)
    return accuracy_score(y_true, y_pred)

def objective(trial):
    """
    Optuna objective function for tuning ExtraTreesClassifier.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 10, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Initialize and train the model
    model = ExtraTreesClassifier(**params)
    model.fit(X_train_scaled, y_train_binary)
    
    # Evaluate using Pairwise Accuracy
    proba_val = model.predict_proba(X_val_scaled)[:, 1]
    return pairwise_accuracy_score(y_val_binary, proba_val)

# Execute the study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print(f"Best Parameters: {study.best_params}")
print(f"Best Validation Accuracy: {study.best_value:.4f}")


# Retraining with optimized parameters
best_params = study.best_params
final_model = ExtraTreesClassifier(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train_scaled, y_train_binary)

# Visualizing Feature Importance
import matplotlib.pyplot as plt

# Extracting feature names (grouped by file_1, file_2, and differentials)
sample_stats = compute_text_stats("sample")
base_names = list(sample_stats.keys())
full_feature_names = base_names + [f"{n}_v2" for n in base_names] + \
                     [f"diff_{n}" for n in base_names] + [f"diff_inv_{n}" for n in base_names]

importances = final_model.feature_importances_
indices = np.argsort(importances)[-15:] # Top 15 features

plt.figure(figsize=(10, 7))
plt.title('Top 15 Most Discriminative Stylometric Features')
plt.barh(range(len(indices)), importances[indices], align='center', color='#2ecc71')
plt.yticks(range(len(indices)), [full_feature_names[i] for i in indices])
plt.xlabel('Relative Gini Importance')
plt.tight_layout()
plt.show()


final_model


# Concatenate Train and Validation to get 100% of the data
X_full_final = np.vstack([X_train_scaled, X_val_scaled])
y_full_binary = np.concatenate([y_train_binary, y_val_binary])

print(f"Final training on {X_full_final.shape[0]} samples...")


# Retraining with Best Parameters
final_model_full = ExtraTreesClassifier(**study.best_params, random_state=42, n_jobs=-1)
final_model_full.fit(X_full_final, y_full_binary)

# Inference on Test Set
proba_test_full = final_model_full.predict_proba(X_test_scaled)[:, 1]
test_preds_full = (proba_test_full < 0.5).astype(int) + 1

test_ids = test_df["article_folder"].str.extract(r"article_(\d+)")[0].astype(int)

# Creating the final submission
submission_full = pd.DataFrame({
    "id": test_ids,
    "real_text_id": test_preds_full
}).sort_values("id")

submission_full.to_csv("submission.csv", index=False)
print("Full-data submission file generated!")

