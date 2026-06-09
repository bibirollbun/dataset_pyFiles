import pandas as pd
import numpy as np
import re
import warnings
import gc

# --- 1. IMPORTS & SETUP ---

# Install and import necessary libraries
!pip install -q textstat transformers accelerate
import textstat
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams

# NLTK setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Advanced models
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

# Transformer & PyTorch
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.utils import logging
from tqdm.auto import tqdm

# Set logging to error only to avoid spam
logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running on {device} ---")


# --- 2. CONFIGURATION ---
class CFG:
    SEED = 42
    N_SPLITS = 10
    INPUT_PATH = "/kaggle/input/mercor-ai-detection/"

    # Tier 1: Feature Config
    EMBED_MODEL_PATH = "microsoft/deberta-v3-base"
    PPL_MODEL_PATH = "distilgpt2"
    PCA_DIM = 64 # Dimensionality reduction for embeddings

    # TF-IDF Config
    TFIDF_WORD_MAX = 2000
    TFIDF_CHAR_MAX = 1000

# Set random seed for reproducibility
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.SEED)

# --- 3. TIER 1: ADVANCED FEATURE ENGINEERING ---

# Initialize text processing tools once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def advanced_text_preprocessing(text):
    """Cleans and preprocesses text for TF-IDF"""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Keep only letters and space
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

def get_ngram_diversity(text, n):
    """Calculates n-gram diversity."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0
    tokens = text.lower().split()
    n_grams = list(ngrams(tokens, n))
    if len(n_grams) == 0:
        return 0
    return len(set(n_grams)) / len(n_grams)

def extract_linguistic_features(df, perplexity_features):
    """
    Extracts advanced linguistic, readability, and n-gram features.
    """
    print("Extracting linguistic features...")
    features_list = []

    # Pre-process topics for faster alignment checks
    processed_topics = {topic: set(advanced_text_preprocessing(str(topic)).split())
                        for topic in df['topic'].unique()}

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Linguistic Features"):
        text = str(row['answer'])

        # Basic stats
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)

        features = {}
        features['word_count'] = word_count
        features['sentence_count'] = sentence_count
        features['avg_sentence_length'] = word_count / max(sentence_count, 1)
        features['avg_word_length'] = char_count / max(word_count, 1)

        # Linguistic Diversity (Your insight)
        if word_count > 0:
            features['lexical_diversity'] = len(set(words)) / word_count
        else:
            features['lexical_diversity'] = 0

        # Punctuation and structure
        features['punctuation_count'] = len(re.findall(r'[.,!?;:]', text))
        features['capital_ratio'] = sum(1 for char in text if char.isupper()) / max(char_count, 1)

        # Structural Consistency (Your insight)
        if sentence_count > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            features['sentence_length_variance'] = np.var(sentence_lengths)
        else:
            features['sentence_length_variance'] = 0

        # Topic Alignment (Your insight)
        topic_words = processed_topics.get(row['topic'], set())
        text_words = set(advanced_text_preprocessing(text).split())
        if len(text_words) > 0:
            features['topic_coherence'] = len(topic_words.intersection(text_words)) / len(text_words)
        else:
            features['topic_coherence'] = 0

        # Readability Scores (textstat)
        try:
            features['flesch_kincaid'] = textstat.flesch_kincaid_grade(text)
            features['gunning_fog'] = textstat.gunning_fog(text)
        except:
            features['flesch_kincaid'] = 0
            features['gunning_fog'] = 0

        # NEW: N-gram Diversity
        features['bigram_diversity'] = get_ngram_diversity(text, 2)
        features['trigram_diversity'] = get_ngram_diversity(text, 3)

        features_list.append(features)

    linguistic_df = pd.DataFrame(features_list).fillna(0)

    # Add perplexity features
    linguistic_df['perplexity'] = perplexity_features

    return linguistic_df

def get_perplexity_features(texts, model_name, batch_size=16):
    """Calculates perplexity for a list of texts using a causal LM."""
    print(f"Calculating perplexity with {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: # Added check for pad token
        tokenizer.pad_token = tokenizer.eos_token

    perplexities = []
    # Ensure texts is a numpy array for slicing
    if not isinstance(texts, np.ndarray):
         texts = np.array(texts)

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Perplexity ({model_name})"):
        batch_np = texts[i:i+batch_size]
        batch_list = batch_np.tolist() # Convert NumPy array slice to Python list
        if not batch_list: continue # Skip empty batches

        inputs = tokenizer(batch_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss # Mean loss for the batch
            if loss is not None:
                ppl = torch.exp(loss).cpu().item() # Use .item() for single float value
                perplexities.extend([ppl] * len(batch_list))
            else:
                 perplexities.extend([np.nan] * len(batch_list)) # Handle no loss case

    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
    # Ensure correct length and handle NaNs
    final_perplexities = np.array(perplexities[:len(texts)])
    if np.isnan(final_perplexities).any():
         median_ppl = np.nanmedian(final_perplexities)
         # Use 0 if all are NaN (edge case)
         final_perplexities = np.nan_to_num(final_perplexities, nan=median_ppl if not np.isnan(median_ppl) else 0)
    return final_perplexities


def mean_pooling(model_output, attention_mask):
    """Pools the last hidden state using the attention mask."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Clamp sum to avoid division by zero
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_transformer_embeddings(texts, model_name, batch_size=32):
    """Generates transformer embeddings for a list of texts."""
    print(f"Getting embeddings with {model_name}...")
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_embeddings = []
     # Ensure texts is a numpy array for slicing
    if not isinstance(texts, np.ndarray):
         texts = np.array(texts)

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embeddings ({model_name})"):
        batch_np = texts[i:i+batch_size]
        batch_list = batch_np.tolist() # Convert NumPy array slice to Python list
        if not batch_list: continue # Skip empty batches

        inputs = tokenizer(batch_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = mean_pooling(outputs, inputs['attention_mask'])
            all_embeddings.append(batch_embeddings.cpu().numpy())

    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(all_embeddings)

# --- 4. MODEL DEFINITIONS ---
def get_models():
    """Returns a dictionary of our tuned ensemble models."""

    # --- START OF CHANGE ---
    # Reverted to original parameters that scored 0.99743
    xgb_params = {
        'n_estimators': 1000, 'max_depth': 4, 'learning_rate': 0.03, # Original depth
        'subsample': 0.8, 'colsample_bytree': 0.7, #'gamma': 0.1, # Removed gamma
        'random_state': CFG.SEED,
        'eval_metric': 'auc', 'tree_method': 'hist', 'early_stopping_rounds': 100
    }

    lgbm_params = {
        'n_estimators': 1000, 'max_depth': 4, 'learning_rate': 0.03, # Original depth
        'num_leaves': 15, # Original leaves
        'subsample': 0.8, 'colsample_bytree': 0.7,
        #'reg_alpha': 0.1, # Removed L1
        #'reg_lambda': 0.1, # Removed L2
        'random_state': CFG.SEED, 'verbose': -1, 'n_jobs': -1,
        'callbacks': [lgb.early_stopping(100, verbose=False)]
    }

    cat_params = {
        'iterations': 1000, 'depth': 4, 'learning_rate': 0.03, # Original depth
        #'l2_leaf_reg': 3, # Removed L2
        'random_seed': CFG.SEED, 'eval_metric': 'AUC',
        'verbose': 0, 'early_stopping_rounds': 100
    }
    # --- END OF CHANGE ---

    models = {
        'lgbm': LGBMClassifier(**lgbm_params),
        'xgb': xgb.XGBClassifier(**xgb_params),
        'cat': CatBoostClassifier(**cat_params)
    }
    return models

# --- 5. MAIN EXECUTION (CV + STACKING) ---
def main():
    """Main function to run the CV, Stacking, and create a submission."""

    print("Loading data...")
    train_df = pd.read_csv(CFG.INPUT_PATH + "train.csv")
    test_df = pd.read_csv(CFG.INPUT_PATH + "test.csv")

    # --- TIER 1: Pre-compute all features ---

    # 1. Perplexity Features
    train_ppl = get_perplexity_features(train_df['answer'].values, CFG.PPL_MODEL_PATH)
    test_ppl = get_perplexity_features(test_df['answer'].values, CFG.PPL_MODEL_PATH)

    # 2. Transformer Embeddings
    train_embed = get_transformer_embeddings(train_df['answer'].values, CFG.EMBED_MODEL_PATH)
    test_embed = get_transformer_embeddings(test_df['answer'].values, CFG.EMBED_MODEL_PATH)

    # 3. PCA on Embeddings
    print(f"Running PCA for dimensionality reduction to {CFG.PCA_DIM}...")
    pca = PCA(n_components=CFG.PCA_DIM, random_state=CFG.SEED)
    train_embed_pca = pca.fit_transform(train_embed)
    test_embed_pca = pca.transform(test_embed)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # 4. Linguistic Features (now includes perplexity)
    train_linguistic_features_df = extract_linguistic_features(train_df, train_ppl)
    test_linguistic_features_df = extract_linguistic_features(test_df, test_ppl)

    # 5. Text Preprocessing for TF-IDF
    print("Preprocessing text for TF-IDF...")
    train_df['processed_answer'] = train_df['answer'].apply(advanced_text_preprocessing)
    test_df['processed_answer'] = test_df['answer'].apply(advanced_text_preprocessing)

    # 6. Topic Encoding
    print("Encoding 'topic' feature...")
    all_topics = pd.concat([train_df['topic'], test_df['topic']]).astype(str).unique()
    le = LabelEncoder()
    le.fit(all_topics)
    train_linguistic_features_df['topic_encoded'] = le.transform(train_df['topic'].astype(str))
    test_linguistic_features_df['topic_encoded'] = le.transform(test_df['topic'].astype(str))

    # --- CV & Model Training ---
    skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)

    # Arrays to store predictions for TIER 2 Stacking
    oof_meta_features = np.zeros((len(train_df), len(get_models())))
    test_meta_features = np.zeros((len(test_df), len(get_models())))

    fold_scores = []
    y_train = train_df['is_cheating'].values

    print(f"\n--- Starting {CFG.N_SPLITS}-Fold Cross-Validation ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
        print(f"--- Fold {fold + 1}/{CFG.N_SPLITS} ---")

        # --- Create fold-specific data ---
        X_train_fold_df, X_val_fold_df = train_df.iloc[train_idx], train_df.iloc[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Get pre-computed features
        X_train_linguistic = train_linguistic_features_df.iloc[train_idx].values
        X_val_linguistic = train_linguistic_features_df.iloc[val_idx].values
        X_train_embed = train_embed_pca[train_idx]
        X_val_embed = train_embed_pca[val_idx]

        # --- TF-IDF (Fitted ONLY on train fold) ---
        print("Fitting vectorizers...")
        tfidf_word = TfidfVectorizer(max_features=CFG.TFIDF_WORD_MAX, ngram_range=(1, 3))
        X_train_tfidf_word = tfidf_word.fit_transform(X_train_fold_df['processed_answer']).toarray()
        X_val_tfidf_word = tfidf_word.transform(X_val_fold_df['processed_answer']).toarray()
        X_test_tfidf_word = tfidf_word.transform(test_df['processed_answer']).toarray()

        tfidf_char = TfidfVectorizer(max_features=CFG.TFIDF_CHAR_MAX, ngram_range=(2, 5), analyzer='char')
        X_train_tfidf_char = tfidf_char.fit_transform(X_train_fold_df['answer']).toarray()
        X_val_tfidf_char = tfidf_char.transform(X_val_fold_df['answer']).toarray()
        X_test_tfidf_char = tfidf_char.transform(test_df['answer']).toarray()

        # --- Combine all features ---
        X_train_full = np.hstack([X_train_linguistic, X_train_embed, X_train_tfidf_word, X_train_tfidf_char])
        X_val_full = np.hstack([X_val_linguistic, X_val_embed, X_val_tfidf_word, X_val_tfidf_char])
        # Construct X_test_full carefully to ensure column order matches train/val
        X_test_full = np.hstack([test_linguistic_features_df.values, test_embed_pca, X_test_tfidf_word, X_test_tfidf_char])


        del X_train_tfidf_word, X_val_tfidf_word, X_train_tfidf_char, X_val_tfidf_char
        gc.collect()

        # --- Train Models ---
        print("Training models...")
        models = get_models()
        fold_val_preds = []

        for i, (name, model) in enumerate(models.items()):
            print(f"  Training {name}...")
            if name in ['lgbm', 'xgb', 'cat']: # Added catboost here
                model.fit(X_train_full, y_train_fold, eval_set=[(X_val_full, y_val_fold)])
            else: # Should not happen with current models, but safe practice
                model.fit(X_train_full, y_train_fold)

            val_pred = model.predict_proba(X_val_full)[:, 1]
            test_pred = model.predict_proba(X_test_full)[:, 1]

            # Store preds for stacking
            oof_meta_features[val_idx, i] = val_pred
            test_meta_features[:, i] += test_pred / CFG.N_SPLITS # Average test preds across folds
            fold_val_preds.append(val_pred)

        # --- Score Fold Ensemble ---
        avg_val_pred = np.mean(fold_val_preds, axis=0) # Simple average for fold score
        score = roc_auc_score(y_val_fold, avg_val_pred)
        fold_scores.append(score)
        print(f"Fold {fold + 1} Simple Avg AUC: {score:.6f}")

    # --- Final CV Score (Simple Avg) ---
    final_cv_score_simple = roc_auc_score(y_train, np.mean(oof_meta_features, axis=1))
    print("\n==============================================")
    print(f"FINAL CV SCORE (Simple Avg): {final_cv_score_simple:.6f}")
    print(f"Fold Scores: {[round(f, 4) for f in fold_scores]}")
    print("==============================================")

    # --- TIER 2: STACKING ENSEMBLE ---
    print("\n--- Training Tier 2 Stacking Meta-Model ---")

    # Use the OOF predictions from all models as features
    # Keeping C=1.0 as per Plan A Step 2
    meta_model = LogisticRegression(random_state=CFG.SEED, C=1.0)

    # Get OOF score for the stacked model
    skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
    oof_stack_preds = np.zeros(len(train_df))
    for fold, (train_idx, val_idx) in enumerate(skf_meta.split(oof_meta_features, y_train)):
        # Keeping C=1.0 as per Plan A Step 2
        meta_model_fold = LogisticRegression(random_state=CFG.SEED, C=1.0)
        meta_model_fold.fit(oof_meta_features[train_idx], y_train[train_idx])
        oof_stack_preds[val_idx] = meta_model_fold.predict_proba(oof_meta_features[val_idx])[:, 1]

    final_cv_score_stacked = roc_auc_score(y_train, oof_stack_preds)
    print(f"FINAL CV SCORE (Stacked): {final_cv_score_stacked:.6f}")

    # Train final meta-model on all OOF features
    meta_model.fit(oof_meta_features, y_train)

    # Predict on test features
    final_test_preds = meta_model.predict_proba(test_meta_features)[:, 1]

    # --- Create Submission ---
    print("\nCreating final submission (stacked)...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'is_cheating': final_test_preds
    })
    submission.to_csv('submission.csv', index=False)

    print("submission.csv created successfully!")
    print(submission.head())
    print(f"\nFinal Stacked Prediction Statistics:")
    print(submission['is_cheating'].describe())

# --- Run main execution ---
if __name__ == "__main__":
    main()

