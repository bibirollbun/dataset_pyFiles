import numpy as np
import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
import textstat
import re
from collections import Counter
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
from pathlib import Path
import os
import sys
import pickle
import joblib
warnings.filterwarnings('ignore')

# NLTK downloads (run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

ROOT_DIR = "../"
SEED = 42
N_SPLITS = 10


# Detect Kaggle environment
IS_KAGGLE = os.path.exists('/kaggle')
print(f"Running on Kaggle: {IS_KAGGLE}")

# Set offline model paths
if IS_KAGGLE:
    # Kaggle dataset path (adjust according to actual dataset name)
    OFFLINE_MODELS_PATH = Path("/kaggle/input/math-misconception-offline-models")
else:
    # Local environment path
    OFFLINE_MODELS_PATH = Path("./offline_models")

print(f"Offline models path: {OFFLINE_MODELS_PATH}")
print(f"Models path exists: {OFFLINE_MODELS_PATH.exists()}")

# Check available models
AVAILABLE_MODELS = []
if OFFLINE_MODELS_PATH.exists():
    AVAILABLE_MODELS = [d.name for d in OFFLINE_MODELS_PATH.iterdir() if d.is_dir()]
    print(f"Available offline models: {AVAILABLE_MODELS}")
else:
    print("âš ï¸�  No offline models found. Will use TF-IDF + Meta features only.")

# Check Transformers availability
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"âœ… Transformers available. Using device: {DEVICE}")
except ImportError:
    print("âš ï¸�  Transformers not available. Using TF-IDF + Meta features only.")
    DEVICE = 'cpu'

# Determine feature types to use
USE_BERT_EMBEDDINGS = TRANSFORMERS_AVAILABLE and len(AVAILABLE_MODELS) > 0
print(f"\nğŸ�¯ Feature configuration:")
print(f"   BERT embeddings: {'âœ…' if USE_BERT_EMBEDDINGS else 'â�Œ'}")
print(f"   TF-IDF features: âœ…")
print(f"   Meta features: âœ…")


def load_offline_model(model_priority=['distilbert-base-uncased', 'bert-base-uncased']):
    """Load offline models according to priority order"""
    
    if not USE_BERT_EMBEDDINGS:
        return None, None
    
    for model_name in model_priority:
        model_dir = OFFLINE_MODELS_PATH / model_name.replace("/", "_")
        
        if model_dir.exists():
            try:
                print(f"Loading model from: {model_dir}")
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModel.from_pretrained(model_dir).to(DEVICE)
                model.eval()
                print(f"âœ… Successfully loaded {model_name}")
                return tokenizer, model
            except Exception as e:
                print(f"â�Œ Failed to load {model_name}: {e}")
                continue
    
    print("âš ï¸�  No models could be loaded. Falling back to TF-IDF only.")
    return None, None

def get_bert_embeddings_offline(texts, tokenizer, model, max_length=512, batch_size=16):
    """Generate BERT embeddings in offline environment"""
    
    if tokenizer is None or model is None:
        return None
    
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenization
            try:
                encoded = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Send to GPU
                encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
                
                # Model inference
                outputs = model(**encoded)
                
                # Use [CLS] token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"Processed {i + len(batch_texts)} / {len(texts)} texts")
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                # Fill with zero vectors on error
                dummy_embedding = np.zeros((len(batch_texts), 768))  # BERT base dimension
                embeddings.extend(dummy_embedding)
    
    return np.array(embeddings) if embeddings else None

# Load models
print("ğŸ”„ Loading offline models...")
tokenizer, bert_model = load_offline_model()

if tokenizer is not None:
    print(f"âœ… Model loaded successfully")
    print(f"   Vocab size: {len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else 'unknown'}")
    print(f"   Max length: {tokenizer.model_max_length}")
else:
    print("â�Œ No BERT model available - using enhanced TF-IDF features")


# Load data
train = pl.read_csv(f"{ROOT_DIR}input/map-charting-student-math-misunderstandings/train.csv")
test = pl.read_csv(f"{ROOT_DIR}input/map-charting-student-math-misunderstandings/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {train.columns}")
print(f"\nCategory distribution:")
print(train['Category'].value_counts().sort('count', descending=True))


# Preprocess Misconception
train = train.with_columns([
    pl.col('Misconception').fill_null('NA').cast(pl.Utf8).alias('Misconception')
])

# Create target_cat
train = train.with_columns([
    (pl.col('Category') + ":" + pl.col('Misconception')).alias('target_cat')
])

print(f"Misconception distribution (top 10):")
print(train['Misconception'].value_counts().sort('count', descending=True).head(10))
print(f"\nTotal unique misconceptions: {train['Misconception'].n_unique()}")


# Create Category mapping
category_counts = train['Category'].value_counts().sort('count', descending=True)
map_target1 = {row['Category']: idx for idx, row in enumerate(category_counts.iter_rows(named=True))}

# Create Misconception mapping
misconception_counts = train['Misconception'].value_counts().sort('count', descending=True)
map_target2 = {row['Misconception']: idx for idx, row in enumerate(misconception_counts.iter_rows(named=True))}

# Create target1 and target2
train = train.with_columns([
    pl.col('Category').map_elements(lambda x: map_target1.get(x, -1), return_dtype=pl.Int64).alias('target1'),
    pl.col('Misconception').map_elements(lambda x: map_target2.get(x, -1), return_dtype=pl.Int64).alias('target2')
])

print(f"Category classes: {len(map_target1)}")
print(f"Misconception classes: {len(map_target2)}")


# Create sentence
def create_sentence(row):
    return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\nExplanation: {row['StudentExplanation']}"

train = train.with_columns([
    pl.struct(['QuestionText', 'MC_Answer', 'StudentExplanation']).map_elements(
        create_sentence, return_dtype=pl.Utf8
    ).alias('sentence')
])

test = test.with_columns([
    pl.struct(['QuestionText', 'MC_Answer', 'StudentExplanation']).map_elements(
        create_sentence, return_dtype=pl.Utf8
    ).alias('sentence')
])

print("Sample sentence:")
print(train['sentence'][0][:200] + "...")

# Create list of all texts
all_sentences = train['sentence'].to_list() + test['sentence'].to_list()
train_sentences = train['sentence'].to_list()
test_sentences = test['sentence'].to_list()


# Generate BERT embeddings
train_bert = None
test_bert = None

if USE_BERT_EMBEDDINGS and tokenizer is not None:
    print("ğŸ”„ Generating BERT embeddings...")
    try:
        # Generate embeddings for all texts
        all_embeddings = get_bert_embeddings_offline(
            all_sentences, tokenizer, bert_model, 
            batch_size=8 if DEVICE == 'cuda' else 4
        )
        
        if all_embeddings is not None:
            # Split into train/test
            train_bert = all_embeddings[:len(train)]
            test_bert = all_embeddings[len(train):]
            print(f"âœ… BERT embeddings generated successfully")
            print(f"   Train shape: {train_bert.shape}")
            print(f"   Test shape: {test_bert.shape}")
        else:
            print("â�Œ Failed to generate BERT embeddings")
            
    except Exception as e:
        print(f"â�Œ Error generating BERT embeddings: {e}")
        train_bert = None
        test_bert = None
else:
    print("â�­ï¸�  Skipping BERT embeddings (not available)")


# Generate more TF-IDF features when BERT is not available
print("ğŸ”„ Creating enhanced TF-IDF features...")

all_sentences_df = pd.concat([
    train.select('sentence').to_pandas(),
    test.select('sentence').to_pandas()
])

tfidf_features = []

# TF-IDF 1: Word-level (1,3)-grams
tfidf1 = TfidfVectorizer(
    stop_words='english', ngram_range=(1, 3), analyzer='word',
    max_df=0.95, min_df=2, max_features=12000 if not USE_BERT_EMBEDDINGS else 8000
)
tfidf1.fit(all_sentences_df['sentence'])
train_tfidf1 = tfidf1.transform(train_sentences)
test_tfidf1 = tfidf1.transform(test_sentences)
tfidf_features.append((train_tfidf1, test_tfidf1, "word_1-3gram"))

# TF-IDF 2: Character-level (3,6)-grams
tfidf2 = TfidfVectorizer(
    ngram_range=(3, 6), analyzer='char',
    max_df=0.95, min_df=2, max_features=8000 if not USE_BERT_EMBEDDINGS else 5000
)
tfidf2.fit(all_sentences_df['sentence'])
train_tfidf2 = tfidf2.transform(train_sentences)
test_tfidf2 = tfidf2.transform(test_sentences)
tfidf_features.append((train_tfidf2, test_tfidf2, "char_3-6gram"))

# TF-IDF 3: Word-level (1,2)-grams with higher min_df
tfidf3 = TfidfVectorizer(
    stop_words='english', ngram_range=(1, 2), analyzer='word',
    max_df=0.90, min_df=5, max_features=10000 if not USE_BERT_EMBEDDINGS else 6000
)
tfidf3.fit(all_sentences_df['sentence'])
train_tfidf3 = tfidf3.transform(train_sentences)
test_tfidf3 = tfidf3.transform(test_sentences)
tfidf_features.append((train_tfidf3, test_tfidf3, "word_1-2gram_filtered"))

# Additional TF-IDF features when BERT is not available
if not USE_BERT_EMBEDDINGS:
    print("ğŸ”„ Adding extra TF-IDF features (BERT not available)...")
    
    # TF-IDF 4: Character-level (4,8)-grams
    tfidf4 = TfidfVectorizer(
        ngram_range=(4, 8), analyzer='char',
        max_df=0.90, min_df=3, max_features=6000
    )
    tfidf4.fit(all_sentences_df['sentence'])
    train_tfidf4 = tfidf4.transform(train_sentences)
    test_tfidf4 = tfidf4.transform(test_sentences)
    tfidf_features.append((train_tfidf4, test_tfidf4, "char_4-8gram"))
    
    # TF-IDF 5: Word-level (2,4)-grams
    tfidf5 = TfidfVectorizer(
        stop_words='english', ngram_range=(2, 4), analyzer='word',
        max_df=0.85, min_df=3, max_features=8000
    )
    tfidf5.fit(all_sentences_df['sentence'])
    train_tfidf5 = tfidf5.transform(train_sentences)
    test_tfidf5 = tfidf5.transform(test_sentences)
    tfidf_features.append((train_tfidf5, test_tfidf5, "word_2-4gram"))

# Combine TF-IDF features
train_tfidf_list = [feat[0] for feat in tfidf_features]
test_tfidf_list = [feat[1] for feat in tfidf_features]

train_tfidf_combined = hstack(train_tfidf_list)
test_tfidf_combined = hstack(test_tfidf_list)

print(f"Combined TF-IDF shape: {train_tfidf_combined.shape}")
for i, (_, _, name) in enumerate(tfidf_features):
    print(f"  {name}: {train_tfidf_list[i].shape}")

# Dimensionality reduction with SVD
svd_components = 400 if not USE_BERT_EMBEDDINGS else 300
svd = TruncatedSVD(n_components=svd_components, random_state=SEED)
train_tfidf_reduced = svd.fit_transform(train_tfidf_combined)
test_tfidf_reduced = svd.transform(test_tfidf_combined)

print(f"Reduced TF-IDF shape: {train_tfidf_reduced.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")


def extract_comprehensive_features(df):
    """Extract comprehensive meta features"""
    
    # Basic text length features
    df = df.with_columns([
        pl.col('QuestionText').str.len_chars().alias('question_len'),
        pl.col('MC_Answer').str.len_chars().alias('answer_len'),
        pl.col('StudentExplanation').str.len_chars().alias('explanation_len'),
        pl.col('sentence').str.len_chars().alias('total_len')
    ])
    
    # Word count features
    df = df.with_columns([
        pl.col('QuestionText').str.split(' ').list.len().alias('question_words'),
        pl.col('MC_Answer').str.split(' ').list.len().alias('answer_words'),
        pl.col('StudentExplanation').str.split(' ').list.len().alias('explanation_words')
    ])
    
    # Ratio features
    df = df.with_columns([
        (pl.col('answer_len') / (pl.col('question_len') + 1)).alias('answer_question_len_ratio'),
        (pl.col('explanation_len') / (pl.col('question_len') + 1)).alias('explanation_question_len_ratio'),
        (pl.col('explanation_words') / (pl.col('question_words') + 1)).alias('explanation_question_words_ratio'),
        (pl.col('answer_words') / (pl.col('explanation_words') + 1)).alias('answer_explanation_words_ratio')
    ])
    
    return df

# Add basic meta features
train = extract_comprehensive_features(train)
test = extract_comprehensive_features(test)

basic_meta_cols = [
    'question_len', 'answer_len', 'explanation_len', 'total_len',
    'question_words', 'answer_words', 'explanation_words',
    'answer_question_len_ratio', 'explanation_question_len_ratio', 
    'explanation_question_words_ratio', 'answer_explanation_words_ratio'
]

print("Basic meta features:")
print(train.select(basic_meta_cols).head())


def extract_advanced_nlp_features(texts):
    """Extract advanced NLP features"""
    
    sia = SentimentIntensityAnalyzer()
    features = []
    
    for text in texts:
        if pd.isna(text) or text == '':
            text = 'empty'
            
        # Readability metrics
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            flesch_kincaid = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
            automated_readability = textstat.automated_readability_index(text)
        except:
            flesch_score = flesch_kincaid = gunning_fog = automated_readability = 0
        
        # Sentiment analysis
        sentiment_scores = sia.polarity_scores(text)
        
        # Character-level features
        num_uppercase = sum(1 for c in text if c.isupper())
        num_lowercase = sum(1 for c in text if c.islower())
        num_digits = sum(1 for c in text if c.isdigit())
        num_spaces = text.count(' ')
        num_punctuation = sum(1 for c in text if c in '.,!?;:')
        num_special_chars = sum(1 for c in text if c in '()[]{}"\'') 
        
        # Math-specific features
        num_equations = len(re.findall(r'\d+\s*[+\-*/=]\s*\d+', text))
        num_fractions = len(re.findall(r'\d+/\d+', text))
        num_percentages = len(re.findall(r'\d+%', text))
        num_decimals = len(re.findall(r'\d+\.\d+', text))
        num_parentheses = text.count('(') + text.count(')')
        
        # Question/Answer patterns
        num_question_marks = text.count('?')
        num_exclamation = text.count('!')
        
        # Vocabulary features
        words = text.lower().split()
        unique_words = len(set(words))
        vocab_diversity = unique_words / (len(words) + 1)
        
        # Word length statistics
        word_lengths = [len(word) for word in words]
        avg_word_len = np.mean(word_lengths) if word_lengths else 0
        max_word_len = max(word_lengths) if word_lengths else 0
        min_word_len = min(word_lengths) if word_lengths else 0
        
        # Sentence structure
        num_sentences = len(re.split(r'[.!?]+', text))
        avg_sentence_len = len(words) / max(num_sentences, 1)
        
        # Common math terms
        math_terms = ['calculate', 'solve', 'equation', 'formula', 'answer', 'result', 
                     'number', 'plus', 'minus', 'multiply', 'divide', 'equal']
        num_math_terms = sum(1 for term in math_terms if term in text.lower())
        
        features.append([
            flesch_score, flesch_kincaid, gunning_fog, automated_readability,
            sentiment_scores['compound'], sentiment_scores['pos'], 
            sentiment_scores['neu'], sentiment_scores['neg'],
            num_uppercase, num_lowercase, num_digits, num_spaces,
            num_punctuation, num_special_chars,
            num_equations, num_fractions, num_percentages, num_decimals, num_parentheses,
            num_question_marks, num_exclamation,
            unique_words, vocab_diversity, avg_word_len, max_word_len, min_word_len,
            num_sentences, avg_sentence_len, num_math_terms
        ])
    
    return np.array(features)

# Generate advanced NLP features
print("ğŸ”„ Extracting advanced NLP features...")
train_advanced = extract_advanced_nlp_features(train_sentences)
test_advanced = extract_advanced_nlp_features(test_sentences)

advanced_feature_names = [
    'flesch_score', 'flesch_kincaid', 'gunning_fog', 'automated_readability',
    'sentiment_compound', 'sentiment_pos', 'sentiment_neu', 'sentiment_neg',
    'num_uppercase', 'num_lowercase', 'num_digits', 'num_spaces',
    'num_punctuation', 'num_special_chars',
    'num_equations', 'num_fractions', 'num_percentages', 'num_decimals', 'num_parentheses',
    'num_question_marks', 'num_exclamation',
    'unique_words', 'vocab_diversity', 'avg_word_len', 'max_word_len', 'min_word_len',
    'num_sentences', 'avg_sentence_len', 'num_math_terms'
]

print(f"Advanced NLP features shape - Train: {train_advanced.shape}, Test: {test_advanced.shape}")


# Get basic meta features
train_basic_meta = train.select(basic_meta_cols).to_numpy().astype(np.float32)
test_basic_meta = test.select(basic_meta_cols).to_numpy().astype(np.float32)

# Combine all features
feature_components = [
    (train_tfidf_reduced, test_tfidf_reduced, "TF-IDF (SVD)"),
    (train_basic_meta, test_basic_meta, "Basic Meta"),
    (train_advanced, test_advanced, "Advanced NLP")
]

# Add BERT if available
if USE_BERT_EMBEDDINGS and train_bert is not None:
    feature_components.insert(1, (train_bert, test_bert, "BERT Embeddings"))

# Combine features
train_features_list = [comp[0] for comp in feature_components]
test_features_list = [comp[1] for comp in feature_components]

train_features = np.hstack(train_features_list)
test_features = np.hstack(test_features_list)

print(f"\nğŸ�¯ Final feature dimensions:")
start_idx = 0
for train_feat, test_feat, name in feature_components:
    print(f"   {name}: {train_feat.shape[1]}")
    start_idx += train_feat.shape[1]
print(f"   Total: {train_features.shape[1]}")

# Feature normalization
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

print(f"\nâœ… Features prepared:")
print(f"   Train: {train_features_scaled.shape}")
print(f"   Test: {test_features_scaled.shape}")


# Ensemble learning for Category prediction
print("ğŸ”„ Training Category classification models...")

train_target1 = train['target1'].to_numpy()
n_classes1 = len(map_target1)

# Arrays to store prediction results
ytrain1_lr = np.zeros((len(train), n_classes1))
ytrain1_lgb = np.zeros((len(train), n_classes1))
ytrain1_xgb = np.zeros((len(train), n_classes1))
ytrain1_rf = np.zeros((len(train), n_classes1))

ytest1_lr = np.zeros((len(test), n_classes1))
ytest1_lgb = np.zeros((len(test), n_classes1))
ytest1_xgb = np.zeros((len(test), n_classes1))
ytest1_rf = np.zeros((len(test), n_classes1))

# Store trained models
category_models = {'lr': [], 'lgb': [], 'xgb': [], 'rf': []}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(skf.split(train_features_scaled, train_target1)):
    print(f"Category Fold {fold+1}/{N_SPLITS}")
    
    X_train_fold, X_valid_fold = train_features_scaled[train_idx], train_features_scaled[valid_idx]
    y_train_fold, y_valid_fold = train_target1[train_idx], train_target1[valid_idx]
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED, n_jobs=-1)
    lr_model.fit(X_train_fold, y_train_fold)
    ytrain1_lr[valid_idx] = lr_model.predict_proba(X_valid_fold)
    ytest1_lr += lr_model.predict_proba(test_features_scaled) / N_SPLITS
    category_models['lr'].append(lr_model)
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        max_depth=8, random_state=SEED, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )
    ytrain1_lgb[valid_idx] = lgb_model.predict_proba(X_valid_fold)
    ytest1_lgb += lgb_model.predict_proba(test_features_scaled) / N_SPLITS
    category_models['lgb'].append(lgb_model)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=SEED, n_jobs=-1, eval_metric='mlogloss'
    )
    xgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        early_stopping_rounds=30, verbose=False
    )
    ytrain1_xgb[valid_idx] = xgb_model.predict_proba(X_valid_fold)
    ytest1_xgb += xgb_model.predict_proba(test_features_scaled) / N_SPLITS
    category_models['xgb'].append(xgb_model)
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1
    )
    rf_model.fit(X_train_fold, y_train_fold)
    ytrain1_rf[valid_idx] = rf_model.predict_proba(X_valid_fold)
    ytest1_rf += rf_model.predict_proba(test_features_scaled) / N_SPLITS
    category_models['rf'].append(rf_model)

# Ensemble (weighted average)
if USE_BERT_EMBEDDINGS:
    # Weights for BERT-enabled case
    ytrain1 = 0.3 * ytrain1_lr + 0.35 * ytrain1_lgb + 0.25 * ytrain1_xgb + 0.1 * ytrain1_rf
    ytest1 = 0.3 * ytest1_lr + 0.35 * ytest1_lgb + 0.25 * ytest1_xgb + 0.1 * ytest1_rf
else:
    # Weights for BERT-disabled case (emphasize diversity)
    ytrain1 = 0.25 * ytrain1_lr + 0.35 * ytrain1_lgb + 0.25 * ytrain1_xgb + 0.15 * ytrain1_rf
    ytest1 = 0.25 * ytest1_lr + 0.35 * ytest1_lgb + 0.25 * ytest1_xgb + 0.15 * ytest1_rf

# Evaluate Category classification
category_acc = accuracy_score(train_target1, np.argmax(ytrain1, axis=1))
category_f1 = f1_score(train_target1, np.argmax(ytrain1, axis=1), average='weighted')

print(f"\nâœ… Category Results:")
print(f"   Accuracy: {category_acc:.4f}")
print(f"   F1-score: {category_f1:.4f}")


# Ensemble learning for Misconception prediction
print("ğŸ”„ Training Misconception classification models...")

train_target2 = train['target2'].to_numpy()
n_classes2 = len(map_target2)

# Arrays to store prediction results
ytrain2_lr = np.zeros((len(train), n_classes2))
ytrain2_lgb = np.zeros((len(train), n_classes2))
ytrain2_xgb = np.zeros((len(train), n_classes2))
ytrain2_rf = np.zeros((len(train), n_classes2))

ytest2_lr = np.zeros((len(test), n_classes2))
ytest2_lgb = np.zeros((len(test), n_classes2))
ytest2_xgb = np.zeros((len(test), n_classes2))
ytest2_rf = np.zeros((len(test), n_classes2))

# Store trained models
misconception_models = {'lr': [], 'lgb': [], 'xgb': [], 'rf': []}

for fold, (train_idx, valid_idx) in enumerate(skf.split(train_features_scaled, train_target2)):
    print(f"Misconception Fold {fold+1}/{N_SPLITS}")
    
    X_train_fold, X_valid_fold = train_features_scaled[train_idx], train_features_scaled[valid_idx]
    y_train_fold, y_valid_fold = train_target2[train_idx], train_target2[valid_idx]
    
    # Logistic Regression with class balancing
    lr_model = LogisticRegression(
        max_iter=1500, C=0.5, class_weight='balanced', 
        random_state=SEED, n_jobs=-1
    )
    lr_model.fit(X_train_fold, y_train_fold)
    ytrain2_lr[valid_idx] = lr_model.predict_proba(X_valid_fold)
    ytest2_lr += lr_model.predict_proba(test_features_scaled) / N_SPLITS
    misconception_models['lr'].append(lr_model)
    
    # LightGBM with class balancing
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=50,
        max_depth=10, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, class_weight='balanced',
        random_state=SEED, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)]
    )
    ytrain2_lgb[valid_idx] = lgb_model.predict_proba(X_valid_fold)
    ytest2_lgb += lgb_model.predict_proba(test_features_scaled) / N_SPLITS
    misconception_models['lgb'].append(lgb_model)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, eval_metric='mlogloss'
    )
    xgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        early_stopping_rounds=40, verbose=False
    )
    ytrain2_xgb[valid_idx] = xgb_model.predict_proba(X_valid_fold)
    ytest2_xgb += xgb_model.predict_proba(test_features_scaled) / N_SPLITS
    misconception_models['xgb'].append(xgb_model)
    
    # Random Forest with balanced class weights
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight='balanced',
        random_state=SEED, n_jobs=-1
    )
    rf_model.fit(X_train_fold, y_train_fold)
    ytrain2_rf[valid_idx] = rf_model.predict_proba(X_valid_fold)
    ytest2_rf += rf_model.predict_proba(test_features_scaled) / N_SPLITS
    misconception_models['rf'].append(rf_model)

# Ensemble (weighted average)
if USE_BERT_EMBEDDINGS:
    # Weights for BERT-enabled case
    ytrain2 = 0.4 * ytrain2_lr + 0.3 * ytrain2_lgb + 0.2 * ytrain2_xgb + 0.1 * ytrain2_rf
    ytest2 = 0.4 * ytest2_lr + 0.3 * ytest2_lgb + 0.2 * ytest2_xgb + 0.1 * ytest2_rf
else:
    # Weights for BERT-disabled case (emphasize LightGBM)
    ytrain2 = 0.35 * ytrain2_lr + 0.35 * ytrain2_lgb + 0.2 * ytrain2_xgb + 0.1 * ytrain2_rf
    ytest2 = 0.35 * ytest2_lr + 0.35 * ytest2_lgb + 0.2 * ytest2_xgb + 0.1 * ytest2_rf

# Evaluate Misconception classification
misconception_acc = accuracy_score(train_target2, np.argmax(ytrain2, axis=1))
misconception_f1 = f1_score(train_target2, np.argmax(ytrain2, axis=1), average='weighted')

print(f"\nâœ… Misconception Results:")
print(f"   Accuracy: {misconception_acc:.4f}")
print(f"   F1-score: {misconception_f1:.4f}")


# Create inverse mappings
map_inverse1 = {v: k for k, v in map_target1.items()}
map_inverse2 = {v: k for k, v in map_target2.items()}

# Generate predictions for validation data (for MAP@3 evaluation)
ytrain2_filtered = ytrain2.copy()
ytrain2_filtered[:, 0] = 0  # Set NA class probability to 0

predicted1_train = np.argsort(-ytrain1, axis=1)[:, :3]
predicted2_train = np.argsort(-ytrain2_filtered, axis=1)[:, :3]

predict_train = []
for i in range(len(predicted1_train)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1_train[i, j]]
        p2 = map_inverse2[predicted2_train[i, j]]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2)
        else:
            pred.append(p1 + ":NA")
    predict_train.append(pred)

# Calculate MAP@3
def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score += 1.
        elif t == p[1]:
            score += 1/2
        elif t == p[2]:
            score += 1/3
    return score / len(target_list)

train_target_cat = train['target_cat'].to_list()
map3_score = map3(train_target_cat, predict_train)

print(f"\nğŸ�¯ Final Validation Results:")
print(f"   Acc@1: {np.mean([train_target_cat[i] == predict_train[i][0] for i in range(len(predict_train))]):.4f}")
print(f"   Acc@2: {np.mean([train_target_cat[i] in predict_train[i][:2] for i in range(len(predict_train))]):.4f}")
print(f"   Acc@3: {np.mean([train_target_cat[i] in predict_train[i] for i in range(len(predict_train))]):.4f}")
print(f"   MAP@3: {map3_score:.4f}")

# Summary of feature types used
print(f"\nğŸ“Š Feature Summary:")
print(f"   BERT embeddings: {'âœ… Used' if USE_BERT_EMBEDDINGS and train_bert is not None else 'â�Œ Not used'}")
print(f"   TF-IDF features: âœ… Used ({len(tfidf_features)} variants)")
print(f"   Meta features: âœ… Used ({len(basic_meta_cols) + len(advanced_feature_names)} features)")
print(f"   Total features: {train_features.shape[1]}")


# Save all necessary components for inference
print("ğŸ”„ Saving trained models and feature pipeline...")

# Create models directory
models_dir = Path('./trained_models')
models_dir.mkdir(exist_ok=True)

# Save TF-IDF vectorizers
tfidf_vectorizers = {
    'tfidf1': tfidf1,
    'tfidf2': tfidf2,
    'tfidf3': tfidf3
}

# Add extra vectorizers if available
if not USE_BERT_EMBEDDINGS:
    tfidf_vectorizers['tfidf4'] = tfidf4
    tfidf_vectorizers['tfidf5'] = tfidf5

with open(models_dir / 'tfidf_vectorizers.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizers, f)

# Save SVD transformer
with open(models_dir / 'svd_transformer.pkl', 'wb') as f:
    pickle.dump(svd, f)

# Save feature scaler
with open(models_dir / 'feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save target mappings
target_mappings = {
    'map_target1': map_target1,
    'map_target2': map_target2,
    'map_inverse1': map_inverse1,
    'map_inverse2': map_inverse2
}
with open(models_dir / 'target_mappings.pkl', 'wb') as f:
    pickle.dump(target_mappings, f)

# Save trained models
with open(models_dir / 'category_models.pkl', 'wb') as f:
    pickle.dump(category_models, f)

with open(models_dir / 'misconception_models.pkl', 'wb') as f:
    pickle.dump(misconception_models, f)

# Save feature configuration
feature_config = {
    'USE_BERT_EMBEDDINGS': USE_BERT_EMBEDDINGS,
    'basic_meta_cols': basic_meta_cols,
    'advanced_feature_names': advanced_feature_names,
    'tfidf_feature_names': [feat[2] for feat in tfidf_features],
    'svd_components': svd_components,
    'n_classes1': n_classes1,
    'n_classes2': n_classes2
}
with open(models_dir / 'feature_config.pkl', 'wb') as f:
    pickle.dump(feature_config, f)

# Save BERT model info if available
if USE_BERT_EMBEDDINGS and train_bert is not None:
    bert_info = {
        'model_available': True,
        'embedding_dim': train_bert.shape[1],
        'train_bert': train_bert,
        'test_bert': test_bert
    }
else:
    bert_info = {'model_available': False}
    
with open(models_dir / 'bert_info.pkl', 'wb') as f:
    pickle.dump(bert_info, f)

print(f"âœ… All models and feature pipeline saved to {models_dir}")
print(f"ğŸ“� Saved files:")
for file in sorted(models_dir.glob('*.pkl')):
    print(f"   - {file.name}")

print(f"\nğŸ’¾ Training Summary:")
print(f"   Category Accuracy: {category_acc:.4f}")
print(f"   Misconception Accuracy: {misconception_acc:.4f}")
print(f"   Final MAP@3: {map3_score:.4f}")
print(f"   Models ready for inference notebook!")

