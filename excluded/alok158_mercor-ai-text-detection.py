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


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


d=pd.read_csv("/kaggle/input/mercor-ai-detection/train.csv")
d.info()


from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "/kaggle/input/modernbert/transformers/base/2"

# Add use_fast=False to bypass the error
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("\nModel and tokenizer loaded successfully! ğŸ�‰")


import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from collections import Counter
import warnings
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Download NLTK Data ---
print("Downloading NLTK data (punkt)...")
nltk.download('punkt', quiet=True)

print("Downloading NLTK POS tagger (averaged_perceptron_tagger)...")
# Download the main package
nltk.download('averaged_perceptron_tagger') 

print("Downloading NLTK POS tagger (averaged_perceptron_tagger_eng) as requested by error...")
# Download the specific alias the error is looking for
nltk.download('averaged_perceptron_tagger_eng')

print("NLTK data downloads complete.")

# --- 2. Define Feature Engineering Function ---

def add_advanced_features(df):
    """
    Applies a suite of advanced linguistic features to the dataframe
    (limited to NLTK and base Python due to environment constraints).
    """
    print("Starting feature calculation for dataframe...")
    
    # --- Base Features (from your notebook) ---
    print("Calculating base features...")
    # Ensure all inputs are strings before processing
    df['answer_str'] = df['answer'].astype(str)
    df['char_count'] = df['answer_str'].apply(len)
    df['word_count'] = df['answer_str'].apply(lambda x: len(x.split()))
    df['avg_word_len'] = df['char_count'] / (df['word_count'] + 1e-6)

    # --- Tokenization (do this once) ---
    print("Tokenizing texts (NLTK)...")
    df['words'] = df['answer_str'].apply(word_tokenize)
    df['sentences'] = df['answer_str'].apply(sent_tokenize)
    df['num_sentences'] = df['sentences'].apply(len)

    # --- Linguistic Features ---
    print("Calculating linguistic features...")
    # TTR (Type-Token Ratio)
    df['ttr'] = df['words'].apply(lambda w: len(set(w)) / (len(w) + 1e-6) if len(w) > 0 else 0)
    # Std. Deviation of Sentence Length (in words)
    df['std_sent_len'] = df['sentences'].apply(lambda s: np.std([len(word_tokenize(sent)) for sent in s]) if len(s) > 0 else 0)
    # Std. Deviation of Word Length
    df['std_word_len'] = df['words'].apply(lambda w: np.std([len(word) for word in w]) if len(w) > 0 else 0)

    # --- Punctuation Frequency ---
    print("Calculating punctuation frequency...")
    df['comma_freq'] = df['answer_str'].apply(lambda x: x.count(',')) / (df['word_count'] + 1e-6)
    df['semicolon_freq'] = df['answer_str'].apply(lambda x: x.count(';')) / (df['word_count'] + 1e-6)
    df['question_freq'] = df['answer_str'].apply(lambda x: x.count('?')) / (df['word_count'] + 1e-6)

    # --- Part-of-Speech (POS) Tag Features ---
    print("Calculating Part-of-Speech (POS) tag features...")
    # This is the line that was failing
    df['pos_tags'] = df['words'].apply(pos_tag)
    
    def count_pos(tags, tag_prefix):
        return sum(1 for _, tag in tags if tag.startswith(tag_prefix))
        
    df['noun_freq'] = df['pos_tags'].apply(lambda tags: count_pos(tags, 'NN')) / (df['word_count'] + 1e-6)
    df['adj_freq'] = df['pos_tags'].apply(lambda tags: count_pos(tags, 'JJ')) / (df['word_count'] + 1e-6)
    df['verb_freq'] = df['pos_tags'].apply(lambda tags: count_pos(tags, 'VB')) / (df['word_count'] + 1e-6)
    df['adv_freq'] = df['pos_tags'].apply(lambda tags: count_pos(tags, 'RB')) / (df['word_count'] + 1e-6)
    df['pronoun_freq'] = df['pos_tags'].apply(lambda tags: count_pos(tags, 'PRP')) / (df['word_count'] + 1e-6)

    print("Feature calculation complete.")
    return df

# --- 3. Load Data and Apply Features ---
print("Loading original data...")
# These paths are from your notebook and will work in your Kaggle environment
train_df = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
test_df = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')

print("\n--- Augmenting Training Data ---")
train_augmented = add_advanced_features(train_df.copy())

print("\n--- Augmenting Test Data ---")
test_augmented = add_advanced_features(test_df.copy())

# --- 4. Define Feature List and Save ---

# This is the new list you will use in your notebook
all_numerical_features = [
    # Base
    'char_count', 'word_count', 'avg_word_len', 'num_sentences',
    # Linguistic
    'ttr', 'std_sent_len', 'std_word_len',
    # Punctuation
    'comma_freq', 'semicolon_freq', 'question_freq',
    # POS
    'noun_freq', 'adj_freq', 'verb_freq', 'adv_freq', 'pronoun_freq',
]

# Clean up intermediate columns before saving
cols_to_drop = ['answer_str', 'words', 'sentences', 'pos_tags']
train_augmented = train_augmented.drop(columns=cols_to_drop, errors='ignore')
test_augmented = test_augmented.drop(columns=cols_to_drop, errors='ignore')

print("\n--- Augmented Training Data Info ---")
# Check for NaN/Inf values and impute
train_augmented.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in all_numerical_features:
    if train_augmented[col].isna().any():
        median_val = train_augmented[col].median()
        if pd.isna(median_val):
            median_val = 0 # Fallback if all NaNs
        print(f"Imputing NaNs in '{col}' with median value: {median_val}")
        train_augmented[col] = train_augmented[col].fillna(median_val)
        test_augmented[col] = test_augmented[col].fillna(median_val) # Use train median for test

train_augmented.info()

print("\n--- Augmented Training Data Head (New Features) ---")
print(train_augmented[all_numerical_features].head())

print("\nSaving augmented data to CSV...")
# These will save to your /kaggle/working/ directory
train_augmented.to_csv("train_augmented.csv", index=False)
test_augmented.to_csv("test_augmented.csv", index=False)
print("Saved 'train_augmented.csv' and 'test_augmented.csv'")


import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from scipy.sparse import hstack
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load and Prepare Data ---
print("Loading and preparing AUGMENTED data...")
# Load the new files you just created
train = pd.read_csv('train_augmented.csv')
test = pd.read_csv('test_augmented.csv')
sample_submission = pd.read_csv('/kaggle/input/mercor-ai-detection/sample_submission.csv')

# --- TF-IDF Vectorization (same as before) ---
# We still use TF-IDF on the 'answer' column
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
train_tfidf = tfidf.fit_transform(train['answer'])
test_tfidf = tfidf.transform(test['answer'])

# --- Define NEW Numerical Feature List ---
# This is the new, expanded list of features
numerical_features = [
    'char_count', 'word_count', 'avg_word_len', 'num_sentences',
    'ttr', 'std_sent_len', 'std_word_len',
    'comma_freq', 'semicolon_freq', 'question_freq',
    'noun_freq', 'adj_freq', 'verb_freq', 'adv_freq', 'pronoun_freq'
]

train_numerical = train[numerical_features].values
test_numerical = test[numerical_features].values

# --- Combine Features (same as before) ---
X_train = hstack([train_tfidf, train_numerical]).tocsr()
X_test = hstack([test_tfidf, test_numerical]).tocsr()
y_train = train['is_cheating']

# --- 2. Define the Optuna Objective Function ---
def objective(trial):
    # Define hyperparameter search space for each base model
    lr_c = trial.suggest_float('lr_c', 1e-3, 1e2, log=True)
    nb_alpha = trial.suggest_float('nb_alpha', 1e-3, 1.0, log=True)
    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 300),
        'max_depth': trial.suggest_int('lgbm_max_depth', 3, 10),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Create the estimators with the suggested hyperparameters
    estimators = [
        ('lr', LogisticRegression(C=lr_c, random_state=42)),
        ('nb', MultinomialNB(alpha=nb_alpha)),
        ('lgbm', lgb.LGBMClassifier(**lgbm_params))
    ]

    # Create the stacking model
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        # --- THIS LINE IS NOW FIXED ---
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Inner CV for stacking
    )

    # --- Evaluate the model using robust cross-validation ---
    # We use RepeatedStratifiedKFold for a more stable score on this small dataset
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return score.mean()

# --- 3. Run Optuna Study ---
print("Starting Optuna hyperparameter optimization with new features...")
study = optuna.create_study(direction='maximize', study_name='stacking-optimization-v2')
study.optimize(objective, n_trials=30) # 30 trials to balance speed and results

print(f"Optimization finished! Best trial ROC AUC (Repeated CV): {study.best_value:.5f}")
print("Best hyperparameters found:")
print(study.best_params)

# --- 4. Train Final Model with Best Hyperparameters ---
print("\nTraining final model with the best hyperparameters...")

# Get the best params from the study
best_params = study.best_params
best_estimators = [
    ('lr', LogisticRegression(C=best_params['lr_c'], random_state=42)),
    ('nb', MultinomialNB(alpha=best_params['nb_alpha'])),
    ('lgbm', lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=best_params['lgbm_n_estimators'],
        learning_rate=best_params['lgbm_learning_rate'],
        num_leaves=best_params['lgbm_num_leaves'],
        max_depth=best_params['lgbm_max_depth'],
        random_state=42
    ))
]

final_stacking_model = StackingClassifier(
    estimators=best_estimators,
    final_estimator=LogisticRegression(),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

# Fit the final model on all training data
final_stacking_model.fit(X_train, y_train)
print("Final model training complete!")

# --- 5. Print Training ROC AUC and Generate Submission ---
# Predict on the training data to get the training ROC AUC score
train_preds = final_stacking_model.predict_proba(X_train)[:, 1]
train_roc_auc = roc_auc_score(y_train, train_preds)

print(f"\nROC AUC Score on the entire training set: {train_roc_auc:.5f}")
print("Note: The cross-validated score from the Optuna study is a more reliable estimate of performance on unseen data.")

# Generate predictions for the test set
print("\nGenerating test set predictions...")
test_probabilities = final_stacking_model.predict_proba(X_test)[:, 1]

# Create the submission file
submission_df = pd.DataFrame({'id': test['id'], 'is_cheating': test_probabilities})
submission_df.to_csv('submission_stacking_augmented.csv', index=False)

print("Augmented stacking submission file created successfully!")


import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from collections import Counter
import warnings
import re
import lightgbm as lgb
from scipy.sparse import hstack
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB

# --- 0. Suppress Warnings ---
warnings.filterwarnings('ignore')

# --- 1. Download NLTK Data ---
print("Downloading NLTK data (punkt, averaged_perceptron_tagger)...")
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
# This one is for the 'eng' alias fix
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except:
    print("Could not download 'averaged_perceptron_tagger_eng', continuing...")
print("NLTK data downloaded.")

# --- 2. Define Feature Engineering Function ---
def add_advanced_features(df):
    """
    Applies a suite of advanced linguistic features to the dataframe.
    """
    print(f"Starting feature calculation for dataframe with {len(df)} rows...")
    
    # Base Features
    df['answer_str'] = df['answer'].astype(str)
    df['char_count'] = df['answer_str'].apply(len)
    df['word_count'] = df['answer_str'].apply(lambda x: len(x.split()))
    df['avg_word_len'] = df['char_count'] / (df['word_count'] + 1e-6)

    # Tokenization
    df['words'] = df['answer_str'].apply(word_tokenize)
    df['sentences'] = df['answer_str'].apply(sent_tokenize)
    df['num_sentences'] = df['sentences'].apply(len)

    # Linguistic Features
    df['ttr'] = df['words'].apply(lambda w: len(set(w)) / (len(w) + 1e-6) if len(w) > 0 else 0)
    df['std_sent_len'] = df['sentences'].apply(lambda s: np.std([len(word_tokenize(sent)) for sent in s]) if len(s) > 0 else 0)
    df['std_word_len'] = df['words'].apply(lambda w: np.std([len(word) for word in w]) if len(w) > 0 else 0)

    # Punctuation Frequency
    df['comma_freq'] = df['answer_str'].apply(lambda x: x.count(',')) / (df['word_count'] + 1e-6)
    df['semicolon_freq'] = df['answer_str'].apply(lambda x: x.count(';')) / (df['word_count'] + 1e-6)
    df['question_freq'] = df['answer_str'].apply(lambda x: x.count('?')) / (df['word_count'] + 1e-6)

    # Part-of-Speech (POS) Tag Features
    df['pos_tags'] = df['words'].apply(pos_tag)
    
    def count_pos(tags, tag_prefix):
        return sum(1 for _, tag in tags if tag.startswith(tag_prefix))
        
    df['noun_freq'] = df['pos_tags'].apply(lambda tags: count_pos(tags, 'NN')) / (df['word_count'] + 1e-6)
    df['adj_freq'] = df['pos_tags'].apply(lambda tags: count_pos(tags, 'JJ')) / (df['word_count'] + 1e-6)
    df['verb_freq'] = df['pos_tags'].apply(lambda tags: count_pos(tags, 'VB')) / (df['word_count'] + 1e-6)
    df['adv_freq'] = df['pos_tags'].apply(lambda tags: count_pos(tags, 'RB')) / (df['word_count'] + 1e-6)
    df['pronoun_freq'] = df['pos_tags'].apply(lambda tags: count_pos(tags, 'PRP')) / (df['word_count'] + 1e-6)

    print("Feature calculation complete.")
    return df

# --- 3. Load ORIGINAL Data and Apply Features ---
print("Loading original data...")
train_df = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
test_df = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')
sample_submission = pd.read_csv('/kaggle/input/mercor-ai-detection/sample_submission.csv')

print("\n--- Augmenting Training Data ---")
train = add_advanced_features(train_df)

print("\n--- Augmenting Test Data ---")
test = add_advanced_features(test_df)

# --- 4. TF-IDF and Feature Combination ---
print("\nRunning TF-IDF vectorization...")
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
train_tfidf = tfidf.fit_transform(train['answer'])
test_tfidf = tfidf.transform(test['answer'])

# Define NEW Numerical Feature List
numerical_features = [
    'char_count', 'word_count', 'avg_word_len', 'num_sentences',
    'ttr', 'std_sent_len', 'std_word_len',
    'comma_freq', 'semicolon_freq', 'question_freq',
    'noun_freq', 'adj_freq', 'verb_freq', 'adv_freq', 'pronoun_freq'
]

# Impute NaNs just in case (e.g., from std dev on single-word answers)
for col in numerical_features:
    if train[col].isna().any():
        median_val = train[col].median()
        if pd.isna(median_val): median_val = 0
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

train_numerical = train[numerical_features].values
test_numerical = test[numerical_features].values

# Combine Features
print("Combining TF-IDF and numerical features...")
X_train = hstack([train_tfidf, train_numerical]).tocsr()
X_test = hstack([test_tfidf, test_numerical]).tocsr()
y_train = train['is_cheating']

# --- 5. Define Best Hyperparameters (from your original notebook) ---
print("Using best hyperparameters from your original notebook (Trial 3)...")
best_params = {
    'lr_c': 84.3593448813156, 
    'nb_alpha': 0.00763077988606349, 
    'lgbm_n_estimators': 112, 
    'lgbm_learning_rate': 0.14895455070202243, 
    'lgbm_num_leaves': 29, 
    'lgbm_max_depth': 9
}

# --- 6. Define the Stacking Model ---
best_estimators = [
    ('lr', LogisticRegression(C=best_params['lr_c'], random_state=42)),
    ('nb', MultinomialNB(alpha=best_params['nb_alpha'])),
    ('lgbm', lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=best_params['lgbm_n_estimators'],
        learning_rate=best_params['lgbm_learning_rate'],
        num_leaves=best_params['lgbm_num_leaves'],
        max_depth=best_params['lgbm_max_depth'],
        random_state=42,
        verbose=-1
    ))
]

final_stacking_model = StackingClassifier(
    estimators=best_estimators,
    final_estimator=LogisticRegression(),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

# --- 7. Run "Quick Test" Cross-Validation ---
print("\nRunning 'Quick Test' with Repeated Cross-Validation...")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(final_stacking_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"\n--- Quick Test Result ---")
print(f"Mean ROC AUC with new features: {np.mean(scores):.5f}")
print(f"Std Dev of ROC AUC: {np.std(scores):.5f}")
print("Compare this to your original 0.973 CV score (from 5-fold CV).")

# --- 8. Train Final Model and Generate Submission ---
print("\nTraining final model on all data...")
final_stacking_model.fit(X_train, y_train)
print("Final model training complete!")

# (Optional) Check training ROC AUC
# train_preds = final_stacking_model.predict_proba(X_train)[:, 1]
# train_roc_auc = roc_auc_score(y_train, train_preds)
# print(f"ROC AUC Score on the entire training set: {train_roc_auc:.5f}")

# Generate predictions for the test set
print("\nGenerating test set predictions...")
test_probabilities = final_stacking_model.predict_proba(X_test)[:, 1]

# Create the submission file
submission_df = pd.DataFrame({'id': test['id'], 'is_cheating': test_probabilities})
submission_df.to_csv('submission_stacking_augmented.csv', index=False)

print("\nAugmented stacking submission file created successfully!")
print("Done.")


d=pd.read_csv("/kaggle/working/submission_stacking_augmented.csv")
d.info()


# Complete, robust pipeline: TF-IDF + LightGBM Optuna tuning + Normalization + SHAP
# Run in one cell (Kaggle/Colab). Adjust paths / n_trials as needed.

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import nltk
# ensure both variants of tagger are available for different NLTK versions
for pkg in ["punkt", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ML & NLP libs
import optuna
import shap
import lightgbm as lgb

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

# ----------------------------
# 1) Feature engineering
# ----------------------------
def safe_pos_tag(tokens):
    """Return POS tags for tokens; on failure return empty list."""
    try:
        return pos_tag(tokens)
    except Exception:
        # return placeholder tags (all NN) to avoid crashes
        return [(t, "NN") for t in tokens]

def add_advanced_features(df, text_col="answer"):
    df = df.copy()
    df["answer_str"] = df[text_col].astype(str)

    # Basic counts
    df["char_count"] = df["answer_str"].apply(len)
    df["word_count"] = df["answer_str"].apply(lambda x: len(x.split()))
    df["avg_word_len"] = df["char_count"] / (df["word_count"] + 1e-6)

    # Tokenization
    df["words"] = df["answer_str"].apply(lambda x: word_tokenize(x) if isinstance(x, str) and x.strip() != "" else [])
    df["sentences"] = df["answer_str"].apply(lambda x: sent_tokenize(x) if isinstance(x, str) and x.strip() != "" else [])
    df["num_sentences"] = df["sentences"].apply(len)

    # Lexical / stylistic features
    df["ttr"] = df["words"].apply(lambda w: len(set(w)) / (len(w) + 1e-6) if len(w) > 0 else 0)
    df["std_sent_len"] = df["sentences"].apply(lambda s: np.std([len(word_tokenize(sent)) for sent in s]) if len(s) > 0 else 0)
    df["std_word_len"] = df["words"].apply(lambda w: np.std([len(word) for word in w]) if len(w) > 0 else 0)

    df["comma_freq"] = df["answer_str"].apply(lambda x: x.count(",")) / (df["word_count"] + 1e-6)
    df["semicolon_freq"] = df["answer_str"].apply(lambda x: x.count(";")) / (df["word_count"] + 1e-6)
    df["question_freq"] = df["answer_str"].apply(lambda x: x.count("?")) / (df["word_count"] + 1e-6)

    # POS tagging (safe)
    df["pos_tags"] = df["words"].apply(lambda tokens: safe_pos_tag(tokens) if len(tokens) > 0 else [])

    def count_pos(tags, tag_prefix):
        return sum(1 for _, tag in tags if tag.startswith(tag_prefix))

    df["noun_freq"] = df["pos_tags"].apply(lambda tags: count_pos(tags, "NN")) / (df["word_count"] + 1e-6)
    df["adj_freq"] = df["pos_tags"].apply(lambda tags: count_pos(tags, "JJ")) / (df["word_count"] + 1e-6)
    df["verb_freq"] = df["pos_tags"].apply(lambda tags: count_pos(tags, "VB")) / (df["word_count"] + 1e-6)
    df["adv_freq"] = df["pos_tags"].apply(lambda tags: count_pos(tags, "RB")) / (df["word_count"] + 1e-6)
    df["pronoun_freq"] = df["pos_tags"].apply(lambda tags: count_pos(tags, "PRP")) / (df["word_count"] + 1e-6)

    # drop intermediate large columns if you want to save memory (uncomment)
    # df = df.drop(columns=["words", "sentences", "pos_tags"])
    return df

# ----------------------------
# 2) Load data (modify paths if needed)
# ----------------------------
TRAIN_PATH = "/kaggle/input/mercor-ai-detection/train.csv"
TEST_PATH  = "/kaggle/input/mercor-ai-detection/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/mercor-ai-detection/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"Train rows: {len(train)}, Test rows: {len(test)}")

train = add_advanced_features(train)
test  = add_advanced_features(test)

numerical_features = [
    "char_count","word_count","avg_word_len","num_sentences",
    "ttr","std_sent_len","std_word_len",
    "comma_freq","semicolon_freq","question_freq",
    "noun_freq","adj_freq","verb_freq","adv_freq","pronoun_freq"
]

# Fill NaNs (safe)
train[numerical_features] = train[numerical_features].fillna(0)
test[numerical_features]  = test[numerical_features].fillna(0)

# ----------------------------
# 3) Unified Optuna search: TF-IDF + LightGBM
# ----------------------------
# set n_trials smaller for quick runs; increase for better results
N_TRIALS = 20
RANDOM_SEED = 42

def objective(trial):
    # --- TF-IDF search space
    max_features = trial.suggest_int("max_features", 3000, 12000)
    ngram_choice = trial.suggest_categorical("ngram_range", [(1,1),(1,2),(1,3)])
    sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])
    min_df = trial.suggest_int("min_df", 1, 3)
    max_df = trial.suggest_float("max_df", 0.8, 1.0)

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_choice,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df
    )

    X_tfidf = tfidf.fit_transform(train["answer"].astype(str))

    # normalize numerical features
    scaler_local = StandardScaler(with_mean=False)
    X_num = scaler_local.fit_transform(train[numerical_features].values)

    X = hstack([X_tfidf, X_num])
    y = train["is_cheating"].values

    # --- LightGBM search space
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 60),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "random_state": RANDOM_SEED,
        "verbose": -1
    }

    base_models = [
        ("lr", LogisticRegression(C=84.36, random_state=RANDOM_SEED, max_iter=300)),
        ("nb", MultinomialNB(alpha=0.0076)),
        ("lgbm", lgb.LGBMClassifier(**lgb_params))
    ]
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=300),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=1
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))

print("ğŸ”� Running Optuna study (TF-IDF + LightGBM). This may take a while...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("âœ… Optuna best value (ROC AUC):", study.best_value)
print("âœ… Optuna best params:")
print(study.best_params)

# Save study summary and trials
os.makedirs("optuna_artifacts", exist_ok=True)
study.trials_dataframe().to_csv("optuna_artifacts/optuna_trials.csv", index=False)
joblib.dump(study, "optuna_artifacts/optuna_study.pkl")

# ----------------------------
# 4) Build final pipeline using best params
# ----------------------------
best = study.best_params

best_tfidf = TfidfVectorizer(
    max_features = best["max_features"],
    ngram_range  = best["ngram_range"],
    sublinear_tf = best["sublinear_tf"],
    min_df       = best["min_df"],
    max_df       = best["max_df"]
)
print("Fitting best TF-IDF on full training data...")
train_tfidf = best_tfidf.fit_transform(train["answer"].astype(str))
test_tfidf  = best_tfidf.transform(test["answer"].astype(str))

scaler = StandardScaler(with_mean=False)
train_num_scaled = scaler.fit_transform(train[numerical_features].values)
test_num_scaled  = scaler.transform(test[numerical_features].values)

X_train = hstack([train_tfidf, train_num_scaled]).tocsr()
X_test  = hstack([test_tfidf, test_num_scaled]).tocsr()
y_train = train["is_cheating"].values

# Prepare best LightGBM params (pull from study)
best_lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": best["learning_rate"],
    "num_leaves": int(best["num_leaves"]),
    "max_depth": int(best["max_depth"]),
    "n_estimators": int(best["n_estimators"]),
    "min_child_samples": int(best["min_child_samples"]),
    "random_state": RANDOM_SEED,
    "verbose": -1
}

base_models = [
    ("lr", LogisticRegression(C=84.36, random_state=RANDOM_SEED, max_iter=300)),
    ("nb", MultinomialNB(alpha=0.0076)),
    ("lgbm", lgb.LGBMClassifier(**best_lgb_params))
]
final_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=300),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
    n_jobs=-1
)

# Cross-validate final stacking model for a quick check
print("Running cross-validation on final stacking model (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(final_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"Final stacking CV ROC AUC: {np.mean(cv_scores):.5f} Â± {np.std(cv_scores):.5f}")

# Fit final model on full training set
print("Training final stacking model on all training data...")
final_model.fit(X_train, y_train)
joblib.dump(final_model, "final_stacking_model.pkl")
joblib.dump(best_tfidf, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "numerical_scaler.pkl")
print("Models and artifacts saved to disk.")

# ----------------------------
# 5) SHAP interpretation for LightGBM model inside the stack
# ----------------------------
print("\nğŸ”� Computing SHAP values (LightGBM inside the stacking classifier)...")
try:
    lgb_model = final_model.named_estimators_["lgbm"]
except Exception as e:
    raise RuntimeError("Could not find 'lgbm' in final_model.named_estimators_. Ensure estimator name is correct.") from e

# Sample a small dense subset (SHAP is expensive)
sample_size = min(200, X_train.shape[0])
if hasattr(X_train, "toarray"):
    sample_X = X_train[:sample_size].toarray()
else:
    sample_X = np.array(X_train[:sample_size])

# Build TreeExplainer
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(sample_X)

# shap_values can be list (per-class) for classification; pick positive class
if isinstance(shap_values, list) and len(shap_values) >= 2:
    shap_vals_pos = shap_values[1]
else:
    shap_vals_pos = shap_values

# Feature names: TF-IDF feature names + numerical features
tfidf_names = list(best_tfidf.get_feature_names_out())
feature_names = tfidf_names + numerical_features
feature_names = feature_names[: sample_X.shape[1] ]  # align lengths

# Plot summary (will render in notebook)
print("Showing SHAP summary plot (top features)...")
shap.summary_plot(shap_vals_pos, sample_X, feature_names=feature_names, max_display=20, show=True)

# Save numeric SHAP values for later analysis (optional)
np.save("optuna_artifacts/shap_values.npy", shap_vals_pos)
pd.DataFrame(shap_vals_pos, columns=feature_names[:shap_vals_pos.shape[1]]).head(5).to_csv("optuna_artifacts/shap_sample_head.csv", index=False)

# ----------------------------
# 6) Generate submission
# ----------------------------
print("Generating submission predictions for the test set...")
test_preds = final_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({"id": test["id"], "is_cheating": test_preds})
submission.to_csv("submission_stacking_optuna_full.csv", index=False)
print("Saved submission -> submission_stacking_optuna_full.csv")
print("All done.")





