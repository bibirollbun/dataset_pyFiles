# Core libraries
import numpy as np
import polars as pl
import pandas as pd
import warnings
from pathlib import Path
import gc
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import xgboost as xgb

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Text processing
import re
from collections import Counter
import textstat

# Configuration
ROOT_DIR = "../"
SEED = 42
N_SPLITS = 5
N_TRIALS = 1  # Number of Optuna trials

# Check GPU availability for XGBoost
try:
    import cupy
    GPU_AVAILABLE = True
    print("âœ… GPU available for XGBoost")
except ImportError:
    GPU_AVAILABLE = False
    print("âš ï¸�  GPU not available, using CPU for XGBoost")

print(f"Configuration: SEED={SEED}, N_SPLITS={N_SPLITS}, N_TRIALS={N_TRIALS}")


# Load datasets
print("Loading datasets...")
train = pl.read_csv(f"{ROOT_DIR}input/map-charting-student-math-misunderstandings/train.csv")
test = pl.read_csv(f"{ROOT_DIR}input/map-charting-student-math-misunderstandings/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {train.columns}")

# Basic data exploration
print("\nCategory distribution:")
print(train['Category'].value_counts().sort('count', descending=True))

print("\nMisconception distribution (top 10):")
display(train['Misconception'].value_counts().sort('count', descending=True).head(10))


# Handle missing values and create target variables
print("Preprocessing targets...")

# Fill null misconceptions
train = train.with_columns([
    pl.col('Misconception').fill_null('NA').cast(pl.Utf8).alias('Misconception')
])

# Create combined target
train = train.with_columns([
    (pl.col('Category') + ":" + pl.col('Misconception')).alias('target_cat')
])

# Create mapping for categories
category_counts = train['Category'].value_counts().sort('count', descending=True)
map_target1 = {row['Category']: idx for idx, row in enumerate(category_counts.iter_rows(named=True))}

# Create mapping for misconceptions
misconception_counts = train['Misconception'].value_counts().sort('count', descending=True)
map_target2 = {row['Misconception']: idx for idx, row in enumerate(misconception_counts.iter_rows(named=True))}

# Apply mappings
train = train.with_columns([
    pl.col('Category').map_elements(lambda x: map_target1.get(x, -1), return_dtype=pl.Int64).alias('target1'),
    pl.col('Misconception').map_elements(lambda x: map_target2.get(x, -1), return_dtype=pl.Int64).alias('target2')
])

print(f"Number of categories: {len(map_target1)}")
print(f"Number of misconceptions: {len(map_target2)}")


# Create concatenated sentence
def create_sentence(row):
    """Combine question, answer, and explanation into a single text."""
    return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\nExplanation: {row['StudentExplanation']}"

# Apply to both datasets
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


def extract_text_complexity_features(df):
    """Extract comprehensive text complexity and length features."""
    
    # Basic length features
    df = df.with_columns([
        # Character counts
        pl.col('QuestionText').str.len_chars().alias('question_char_len'),
        pl.col('MC_Answer').str.len_chars().alias('answer_char_len'),
        pl.col('StudentExplanation').str.len_chars().alias('explanation_char_len'),
        pl.col('sentence').str.len_chars().alias('total_char_len'),
        
        # Word counts
        pl.col('QuestionText').str.split(' ').list.len().alias('question_word_len'),
        pl.col('MC_Answer').str.split(' ').list.len().alias('answer_word_len'),
        pl.col('StudentExplanation').str.split(' ').list.len().alias('explanation_word_len'),
        
        # Sentence counts (approximated by periods, exclamation marks, question marks)
        (pl.col('QuestionText').str.count_matches(r'[.!?]') + 1).alias('question_sent_count'),
        (pl.col('StudentExplanation').str.count_matches(r'[.!?]') + 1).alias('explanation_sent_count'),
    ])
    
    # Ratio features
    df = df.with_columns([
        # Length ratios
        (pl.col('answer_char_len') / (pl.col('question_char_len') + 1)).alias('answer_question_char_ratio'),
        (pl.col('explanation_char_len') / (pl.col('question_char_len') + 1)).alias('explanation_question_char_ratio'),
        (pl.col('explanation_char_len') / (pl.col('answer_char_len') + 1)).alias('explanation_answer_char_ratio'),
        
        # Word ratios
        (pl.col('answer_word_len') / (pl.col('question_word_len') + 1)).alias('answer_question_word_ratio'),
        (pl.col('explanation_word_len') / (pl.col('question_word_len') + 1)).alias('explanation_question_word_ratio'),
        
        # Average word length
        (pl.col('question_char_len') / (pl.col('question_word_len') + 1)).alias('avg_question_word_len'),
        (pl.col('answer_char_len') / (pl.col('answer_word_len') + 1)).alias('avg_answer_word_len'),
        (pl.col('explanation_char_len') / (pl.col('explanation_word_len') + 1)).alias('avg_explanation_word_len'),
    ])
    
    return df

# Apply to both datasets
print("Extracting text complexity features...")
train = extract_text_complexity_features(train)
test = extract_text_complexity_features(test)

# Define basic feature columns
basic_feature_cols = [
    'question_char_len', 'answer_char_len', 'explanation_char_len', 'total_char_len',
    'question_word_len', 'answer_word_len', 'explanation_word_len',
    'question_sent_count', 'explanation_sent_count',
    'answer_question_char_ratio', 'explanation_question_char_ratio', 'explanation_answer_char_ratio',
    'answer_question_word_ratio', 'explanation_question_word_ratio',
    'avg_question_word_len', 'avg_answer_word_len', 'avg_explanation_word_len'
]

print(f"Basic features shape: {len(basic_feature_cols)}")
print("Sample features:")
display(train.select(basic_feature_cols[:5]).head())


def extract_vocabulary_complexity(texts):
    """Extract vocabulary complexity features from texts."""
    features = []
    
    for text in texts:
        if pd.isna(text) or text == '':
            text = 'empty'
        
        # Basic text statistics
        words = text.lower().split()
        word_set = set(words)
        
        # Vocabulary diversity
        vocab_diversity = len(word_set) / max(len(words), 1)
        
        # Readability scores
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            flesch_kincaid = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
        except:
            flesch_score = flesch_kincaid = gunning_fog = 0
        
        # Character-level complexity
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        punctuation_ratio = sum(1 for c in text if c in '.,!?;:') / max(len(text), 1)
        
        # Mathematical content indicators
        math_operators = len(re.findall(r'[+\-*/=<>]', text))
        numbers = len(re.findall(r'\b\d+\b', text))
        fractions = len(re.findall(r'\d+/\d+', text))
        percentages = len(re.findall(r'\d+%', text))
        
        # Word frequency analysis
        word_freq = Counter(words)
        max_word_freq = max(word_freq.values()) if word_freq else 0
        unique_words_ratio = len([w for w in word_freq if word_freq[w] == 1]) / max(len(words), 1)
        
        # Question and explanation patterns
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        
        features.append([
            vocab_diversity, flesch_score, flesch_kincaid, gunning_fog,
            uppercase_ratio, digit_ratio, punctuation_ratio,
            math_operators, numbers, fractions, percentages,
            max_word_freq, unique_words_ratio,
            question_marks, exclamation_marks
        ])
    
    return np.array(features)

# Extract vocabulary complexity features
print("Extracting vocabulary complexity features...")
train_vocab_features = extract_vocabulary_complexity(train['sentence'].to_list())
test_vocab_features = extract_vocabulary_complexity(test['sentence'].to_list())

vocab_feature_names = [
    'vocab_diversity', 'flesch_score', 'flesch_kincaid', 'gunning_fog',
    'uppercase_ratio', 'digit_ratio', 'punctuation_ratio',
    'math_operators', 'numbers', 'fractions', 'percentages',
    'max_word_freq', 'unique_words_ratio',
    'question_marks', 'exclamation_marks'
]

print(f"Vocabulary features shape: {train_vocab_features.shape}")
print(f"Feature names: {vocab_feature_names}")


def extract_similarity_features(df):
    """Extract similarity scores between different text components."""
    
    # Create TF-IDF vectorizer for similarity calculation
    tfidf_sim = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    
    # Fit on all text components
    all_texts = (df['QuestionText'].to_list() + 
                df['MC_Answer'].to_list() + 
                df['StudentExplanation'].to_list())
    tfidf_sim.fit(all_texts)
    
    # Calculate similarities
    similarities = []
    
    for i in range(len(df)):
        row = df.row(i, named=True)
        
        # Get TF-IDF vectors
        question_vec = tfidf_sim.transform([row['QuestionText']])
        answer_vec = tfidf_sim.transform([row['MC_Answer']])
        explanation_vec = tfidf_sim.transform([row['StudentExplanation']])
        
        # Calculate cosine similarities
        question_answer_sim = cosine_similarity(question_vec, answer_vec)[0, 0]
        question_explanation_sim = cosine_similarity(question_vec, explanation_vec)[0, 0]
        answer_explanation_sim = cosine_similarity(answer_vec, explanation_vec)[0, 0]
        
        # Text overlap features (simple word overlap)
        question_words = set(row['QuestionText'].lower().split())
        answer_words = set(row['MC_Answer'].lower().split())
        explanation_words = set(row['StudentExplanation'].lower().split())
        
        question_answer_overlap = len(question_words & answer_words) / max(len(question_words | answer_words), 1)
        question_explanation_overlap = len(question_words & explanation_words) / max(len(question_words | explanation_words), 1)
        answer_explanation_overlap = len(answer_words & explanation_words) / max(len(answer_words | explanation_words), 1)
        
        similarities.append([
            question_answer_sim, question_explanation_sim, answer_explanation_sim,
            question_answer_overlap, question_explanation_overlap, answer_explanation_overlap
        ])
    
    return np.array(similarities)

# Extract similarity features
print("Extracting similarity features...")
train_similarity_features = extract_similarity_features(train)
test_similarity_features = extract_similarity_features(test)

similarity_feature_names = [
    'question_answer_cosine_sim', 'question_explanation_cosine_sim', 'answer_explanation_cosine_sim',
    'question_answer_word_overlap', 'question_explanation_word_overlap', 'answer_explanation_word_overlap'
]

print(f"Similarity features shape: {train_similarity_features.shape}")
print(f"Feature names: {similarity_feature_names}")


# Create multiple TF-IDF feature sets
print("Creating TF-IDF features...")

# Combine all sentences for fitting
all_sentences = pd.concat([
    train.select('sentence').to_pandas(),
    test.select('sentence').to_pandas()
])

# TF-IDF 1: Word-level (1,3)-grams for categories
tfidf1 = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 3), 
    analyzer='word',
    max_df=0.95, 
    min_df=2, 
    max_features=12000
)
tfidf1.fit(all_sentences['sentence'])
train_tfidf1 = tfidf1.transform(train['sentence'].to_pandas())
test_tfidf1 = tfidf1.transform(test['sentence'].to_pandas())

# TF-IDF 2: Character-level (4,6)-grams
tfidf2 = TfidfVectorizer(
    ngram_range=(4, 6), 
    analyzer='char',
    max_df=0.95, 
    min_df=2, 
    max_features=5000
)
tfidf2.fit(all_sentences['sentence'])
train_tfidf2 = tfidf2.transform(train['sentence'].to_pandas())
test_tfidf2 = tfidf2.transform(test['sentence'].to_pandas())

# TF-IDF 3: Specialized for misconceptions
tfidf_misc = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 2), 
    analyzer='word',
    max_df=0.90, 
    min_df=2, 
    max_features=15000
)
tfidf_misc.fit(all_sentences['sentence'])
train_tfidf_misc = tfidf_misc.transform(train['sentence'].to_pandas())
test_tfidf_misc = tfidf_misc.transform(test['sentence'].to_pandas())

# Combine TF-IDF features for categories
train_embeddings = hstack([train_tfidf1, train_tfidf2])
test_embeddings = hstack([test_tfidf1, test_tfidf2])

print(f'Category TF-IDF shape: {train_embeddings.shape}')
print(f'Misconception TF-IDF shape: {train_tfidf_misc.shape}')

# Clean up memory
del all_sentences
gc.collect()


def create_lightgbm_objective(X, y, task_type='category'):
    """Create Optuna objective function for LightGBM hyperparameter optimization."""
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': SEED,
            'verbose': -1
        }
        
        if task_type == 'misconception':
            params['class_weight'] = 'balanced'
        
        # Cross-validation
        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # Reduced folds for speed
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val_fold)
            score = accuracy_score(y_val_fold, y_pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    return objective

def create_xgboost_objective(X, y, task_type='category'):
    """Create Optuna objective function for XGBoost hyperparameter optimization."""
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y)),
            'eval_metric': 'mlogloss',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': SEED,
            'n_jobs': -1
        }
        
        if GPU_AVAILABLE:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
        
        # Cross-validation
        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            y_pred = model.predict(X_val_fold)
            score = accuracy_score(y_val_fold, y_pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    return objective

print("Objective functions created successfully!")


# Prepare data for optimization
print("Preparing data for hyperparameter optimization...")

# Combine all features for categories
train_basic_features = train.select(basic_feature_cols).to_numpy().astype(np.float32)
train_category_features = np.hstack([
    train_embeddings.toarray(),
    train_basic_features,
    train_vocab_features,
    train_similarity_features
])

train_target1 = train['target1'].to_numpy()

print(f"Category features shape: {train_category_features.shape}")
print(f"Category target shape: {train_target1.shape}")


# Optimize LightGBM for categories
print("Optimizing LightGBM for category classification...")

study_lgb_cat = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED)
)

lgb_objective_cat = create_lightgbm_objective(train_category_features, train_target1, 'category')
study_lgb_cat.optimize(lgb_objective_cat, n_trials=N_TRIALS//2)  # Reduced trials for time

best_lgb_params_cat = study_lgb_cat.best_params
print(f"Best LightGBM Category Score: {study_lgb_cat.best_value:.4f}")
print(f"Best LightGBM Category Params: {best_lgb_params_cat}")


# Optimize XGBoost for categories
print("Optimizing XGBoost for category classification...")

study_xgb_cat = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED)
)

xgb_objective_cat = create_xgboost_objective(train_category_features, train_target1, 'category')
study_xgb_cat.optimize(xgb_objective_cat, n_trials=N_TRIALS//2)

best_xgb_params_cat = study_xgb_cat.best_params
print(f"Best XGBoost Category Score: {study_xgb_cat.best_value:.4f}")
print(f"Best XGBoost Category Params: {best_xgb_params_cat}")

# Clean up memory
del train_category_features
gc.collect()


# Prepare data for misconception optimization
print("Preparing data for misconception optimization...")

test_basic_features = test.select(basic_feature_cols).to_numpy().astype(np.float32)

# Combine features for misconceptions
train_misconception_features = np.hstack([
    train_tfidf_misc.toarray(),
    train_basic_features,
    train_vocab_features,
    train_similarity_features
])

train_target2 = train['target2'].to_numpy()

print(f"Misconception features shape: {train_misconception_features.shape}")
print(f"Misconception target shape: {train_target2.shape}")


# Optimize LightGBM for misconceptions
print("Optimizing LightGBM for misconception classification...")

study_lgb_misc = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED)
)

lgb_objective_misc = create_lightgbm_objective(train_misconception_features, train_target2, 'misconception')
study_lgb_misc.optimize(lgb_objective_misc, n_trials=N_TRIALS//2)

best_lgb_params_misc = study_lgb_misc.best_params
print(f"Best LightGBM Misconception Score: {study_lgb_misc.best_value:.4f}")
print(f"Best LightGBM Misconception Params: {best_lgb_params_misc}")


# Optimize XGBoost for misconceptions
print("Optimizing XGBoost for misconception classification...")

study_xgb_misc = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED)
)

xgb_objective_misc = create_xgboost_objective(train_misconception_features, train_target2, 'misconception')
study_xgb_misc.optimize(xgb_objective_misc, n_trials=N_TRIALS//2)

best_xgb_params_misc = study_xgb_misc.best_params
print(f"Best XGBoost Misconception Score: {study_xgb_misc.best_value:.4f}")
print(f"Best XGBoost Misconception Params: {best_xgb_params_misc}")


# Recreate category features for final training
print("Preparing final category features...")

train_basic_features = train.select(basic_feature_cols).to_numpy().astype(np.float32)
test_basic_features = test.select(basic_feature_cols).to_numpy().astype(np.float32)

train_category_features = np.hstack([
    train_embeddings.toarray(),
    train_basic_features,
    train_vocab_features,
    train_similarity_features
])

test_category_features = np.hstack([
    test_embeddings.toarray(),
    test_basic_features,
    test_vocab_features,
    test_similarity_features
])

print(f"Final category features - Train: {train_category_features.shape}, Test: {test_category_features.shape}")


# Train category models with optimized parameters
print("Training optimized category classification models...")

# Initialize prediction arrays
n_classes1 = len(map_target1)
ytrain1_lr = np.zeros((len(train), n_classes1))
ytrain1_lgb = np.zeros((len(train), n_classes1))
ytrain1_xgb = np.zeros((len(train), n_classes1))

ytest1_lr = np.zeros((len(test), n_classes1))
ytest1_lgb = np.zeros((len(test), n_classes1))
ytest1_xgb = np.zeros((len(test), n_classes1))

# Cross-validation training
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(skf.split(train_category_features, train_target1)):
    print(f"Category Fold {fold+1}/{N_SPLITS}")
    
    X_train_fold = train_category_features[train_idx]
    X_valid_fold = train_category_features[valid_idx]
    y_train_fold = train_target1[train_idx]
    y_valid_fold = train_target1[valid_idx]
    
    # Logistic Regression (baseline)
    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED, n_jobs=-1)
    lr_model.fit(X_train_fold, y_train_fold)
    ytrain1_lr[valid_idx] = lr_model.predict_proba(X_valid_fold)
    ytest1_lr += lr_model.predict_proba(test_category_features) / N_SPLITS
    
    # Optimized LightGBM
    lgb_params = best_lgb_params_cat.copy()
    lgb_params.update({
        'objective': 'multiclass',
        'num_class': n_classes1,
        'metric': 'multi_logloss',
        'random_state': SEED,
        'verbose': -1
    })
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    ytrain1_lgb[valid_idx] = lgb_model.predict_proba(X_valid_fold)
    ytest1_lgb += lgb_model.predict_proba(test_category_features) / N_SPLITS
    
    # Optimized XGBoost
    xgb_params = best_xgb_params_cat.copy()
    xgb_params.update({
        'objective': 'multi:softprob',
        'num_class': n_classes1,
        'eval_metric': 'mlogloss',
        'random_state': SEED,
        'n_jobs': -1
    })
    
    if GPU_AVAILABLE:
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['gpu_id'] = 0
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        early_stopping_rounds=20,
        verbose=False
    )
    ytrain1_xgb[valid_idx] = xgb_model.predict_proba(X_valid_fold)
    ytest1_xgb += xgb_model.predict_proba(test_category_features) / N_SPLITS

print("Category models training completed!")

# Clean up memory
del train_category_features, test_category_features
gc.collect()


# Prepare final misconception features
print("Preparing final misconception features...")

test_misconception_features = np.hstack([
    test_tfidf_misc.toarray(),
    test_basic_features,
    test_vocab_features,
    test_similarity_features
])

print(f"Final misconception features - Train: {train_misconception_features.shape}, Test: {test_misconception_features.shape}")


# Train misconception models with optimized parameters
print("Training optimized misconception classification models...")

# Initialize prediction arrays
n_classes2 = len(map_target2)
ytrain2_lr = np.zeros((len(train), n_classes2))
ytrain2_lgb = np.zeros((len(train), n_classes2))
ytrain2_xgb = np.zeros((len(train), n_classes2))

ytest2_lr = np.zeros((len(test), n_classes2))
ytest2_lgb = np.zeros((len(test), n_classes2))
ytest2_xgb = np.zeros((len(test), n_classes2))

# Cross-validation training
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_misconception_features, train_target2)):
    print(f"Misconception Fold {fold+1}/{N_SPLITS}")
    
    X_train_fold = train_misconception_features[train_idx]
    X_valid_fold = train_misconception_features[valid_idx]
    y_train_fold = train_target2[train_idx]
    y_valid_fold = train_target2[valid_idx]
    
    # Logistic Regression with class balancing
    lr_model = LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        C=0.5, 
        random_state=SEED,
        n_jobs=-1
    )
    lr_model.fit(X_train_fold, y_train_fold)
    ytrain2_lr[valid_idx] = lr_model.predict_proba(X_valid_fold)
    ytest2_lr += lr_model.predict_proba(test_misconception_features) / N_SPLITS
    
    # Optimized LightGBM with class balancing
    lgb_params = best_lgb_params_misc.copy()
    lgb_params.update({
        'objective': 'multiclass',
        'num_class': n_classes2,
        'metric': 'multi_logloss',
        'class_weight': 'balanced',
        'random_state': SEED,
        'verbose': -1
    })
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )
    ytrain2_lgb[valid_idx] = lgb_model.predict_proba(X_valid_fold)
    ytest2_lgb += lgb_model.predict_proba(test_misconception_features) / N_SPLITS
    
    # Optimized XGBoost
    xgb_params = best_xgb_params_misc.copy()
    xgb_params.update({
        'objective': 'multi:softprob',
        'num_class': n_classes2,
        'eval_metric': 'mlogloss',
        'random_state': SEED,
        'n_jobs': -1
    })
    
    if GPU_AVAILABLE:
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['gpu_id'] = 0
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        early_stopping_rounds=30,
        verbose=False
    )
    ytrain2_xgb[valid_idx] = xgb_model.predict_proba(X_valid_fold)
    ytest2_xgb += xgb_model.predict_proba(test_misconception_features) / N_SPLITS

print("Misconception models training completed!")

# Clean up memory
del train_misconception_features, test_misconception_features
gc.collect()


def optimize_ensemble_weights(predictions_list, targets, method='optuna'):
    """Optimize ensemble weights using Optuna."""
    
    def ensemble_objective(trial):
        # Suggest weights
        weights = []
        for i in range(len(predictions_list)):
            weight = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Calculate ensemble predictions
        ensemble_pred = np.zeros_like(predictions_list[0])
        for i, pred in enumerate(predictions_list):
            ensemble_pred += weights[i] * pred
        
        # Calculate accuracy
        y_pred = np.argmax(ensemble_pred, axis=1)
        accuracy = accuracy_score(targets, y_pred)
        
        return accuracy
    
    # Optimize weights
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    study.optimize(ensemble_objective, n_trials=100)
    
    # Get best weights
    best_weights = []
    for i in range(len(predictions_list)):
        best_weights.append(study.best_params[f'weight_{i}'])
    
    best_weights = np.array(best_weights)
    best_weights = best_weights / best_weights.sum()
    
    return best_weights, study.best_value

print("Ensemble optimization function defined!")


# Optimize category ensemble weights
print("Optimizing category ensemble weights...")

category_predictions = [ytrain1_lr, ytrain1_lgb, ytrain1_xgb]
category_weights, category_score = optimize_ensemble_weights(category_predictions, train_target1)

print(f"Optimized category weights: LR={category_weights[0]:.3f}, LGB={category_weights[1]:.3f}, XGB={category_weights[2]:.3f}")
print(f"Category ensemble accuracy: {category_score:.4f}")

# Create optimized category ensemble
ytrain1_ensemble = (
    category_weights[0] * ytrain1_lr +
    category_weights[1] * ytrain1_lgb +
    category_weights[2] * ytrain1_xgb
)

ytest1_ensemble = (
    category_weights[0] * ytest1_lr +
    category_weights[1] * ytest1_lgb +
    category_weights[2] * ytest1_xgb
)


# Optimize misconception ensemble weights
print("Optimizing misconception ensemble weights...")

misconception_predictions = [ytrain2_lr, ytrain2_lgb, ytrain2_xgb]
misconception_weights, misconception_score = optimize_ensemble_weights(misconception_predictions, train_target2)

print(f"Optimized misconception weights: LR={misconception_weights[0]:.3f}, LGB={misconception_weights[1]:.3f}, XGB={misconception_weights[2]:.3f}")
print(f"Misconception ensemble accuracy: {misconception_score:.4f}")

# Create optimized misconception ensemble
ytrain2_ensemble = (
    misconception_weights[0] * ytrain2_lr +
    misconception_weights[1] * ytrain2_lgb +
    misconception_weights[2] * ytrain2_xgb
)

ytest2_ensemble = (
    misconception_weights[0] * ytest2_lr +
    misconception_weights[1] * ytest2_lgb +
    misconception_weights[2] * ytest2_xgb
)


# Individual model performance
print("=== Individual Model Performance ===")

# Category models
cat_lr_acc = accuracy_score(train_target1, np.argmax(ytrain1_lr, axis=1))
cat_lgb_acc = accuracy_score(train_target1, np.argmax(ytrain1_lgb, axis=1))
cat_xgb_acc = accuracy_score(train_target1, np.argmax(ytrain1_xgb, axis=1))
cat_ensemble_acc = accuracy_score(train_target1, np.argmax(ytrain1_ensemble, axis=1))

print(f"Category Classification:")
print(f"  Logistic Regression: {cat_lr_acc:.4f}")
print(f"  LightGBM: {cat_lgb_acc:.4f}")
print(f"  XGBoost: {cat_xgb_acc:.4f}")
print(f"  Ensemble: {cat_ensemble_acc:.4f}")

# Misconception models
misc_lr_acc = accuracy_score(train_target2, np.argmax(ytrain2_lr, axis=1))
misc_lgb_acc = accuracy_score(train_target2, np.argmax(ytrain2_lgb, axis=1))
misc_xgb_acc = accuracy_score(train_target2, np.argmax(ytrain2_xgb, axis=1))
misc_ensemble_acc = accuracy_score(train_target2, np.argmax(ytrain2_ensemble, axis=1))

print(f"\nMisconception Classification:")
print(f"  Logistic Regression: {misc_lr_acc:.4f}")
print(f"  LightGBM: {misc_lgb_acc:.4f}")
print(f"  XGBoost: {misc_xgb_acc:.4f}")
print(f"  Ensemble: {misc_ensemble_acc:.4f}")

# F1 scores
cat_f1 = f1_score(train_target1, np.argmax(ytrain1_ensemble, axis=1), average='weighted')
misc_f1 = f1_score(train_target2, np.argmax(ytrain2_ensemble, axis=1), average='weighted')

print(f"\n=== F1 Scores (Weighted) ===")
print(f"Category F1: {cat_f1:.4f}")
print(f"Misconception F1: {misc_f1:.4f}")


# Create inverse mappings
map_inverse1 = {v: k for k, v in map_target1.items()}
map_inverse2 = {v: k for k, v in map_target2.items()}

# Generate validation predictions for MAP@3 evaluation
ytrain2_filtered = ytrain2_ensemble.copy()
ytrain2_filtered[:, 0] = 0  # Set NA class probability to 0

predicted1_train = np.argsort(-ytrain1_ensemble, axis=1)[:, :3]
predicted2_train = np.argsort(-ytrain2_filtered, axis=1)[:, :3]

train_predictions = []
for i in range(len(predicted1_train)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1_train[i, j]]
        p2 = map_inverse2[predicted2_train[i, j]]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2)
        else:
            pred.append(p1 + ":NA")
    train_predictions.append(pred)

# Calculate MAP@3 score
def map3(target_list, pred_list):
    """Calculate Mean Average Precision at 3."""
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
map3_score = map3(train_target_cat, train_predictions)

print("=== Final Validation Results ===")
print(f"Acc@1: {np.mean([train_target_cat[i] == train_predictions[i][0] for i in range(len(train_predictions))]):.4f}")
print(f"Acc@2: {np.mean([train_target_cat[i] in train_predictions[i][:2] for i in range(len(train_predictions))]):.4f}")
print(f"Acc@3: {np.mean([train_target_cat[i] in train_predictions[i] for i in range(len(train_predictions))]):.4f}")
print(f"MAP@3: {map3_score:.4f}")


# Generate test predictions
print("Generating test predictions...")

ytest2_filtered = ytest2_ensemble.copy()
ytest2_filtered[:, 0] = 0  # Set NA class probability to 0

predicted1_test = np.argsort(-ytest1_ensemble, axis=1)[:, :3]
predicted2_test = np.argsort(-ytest2_filtered, axis=1)[:, :3]

test_predictions = []
for i in range(len(predicted1_test)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1_test[i, j]]
        p2 = map_inverse2[predicted2_test[i, j]]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2)
        else:
            pred.append(p1 + ":NA")
    test_predictions.append(" ".join(pred))

# Create submission file
sub = pl.read_csv(f"{ROOT_DIR}input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub = sub.with_columns([
    pl.Series('Category:Misconception', test_predictions)
])

submission_filename = "submission.csv"
sub.write_csv(submission_filename)

print(f"\nâœ… Submission file '{submission_filename}' created successfully!")
print(f"ğŸ“Š Sample predictions:")
print(sub.head())


print("\n" + "="*60)
print("ğŸ�¯ ENHANCED ML PIPELINE PERFORMANCE SUMMARY")
print("="*60)

print(f"\nğŸ“Š Model Performance:")
print(f"   Category Accuracy: {cat_ensemble_acc:.4f}")
print(f"   Misconception Accuracy: {misc_ensemble_acc:.4f}")
print(f"   Final MAP@3 Score: {map3_score:.4f}")

print(f"\nğŸ”§ Optimization Results:")
print(f"   LightGBM Category Score: {study_lgb_cat.best_value:.4f}")
print(f"   XGBoost Category Score: {study_xgb_cat.best_value:.4f}")
print(f"   LightGBM Misconception Score: {study_lgb_misc.best_value:.4f}")
print(f"   XGBoost Misconception Score: {study_xgb_misc.best_value:.4f}")

print(f"\nâš™ï¸�  Ensemble Weights:")
print(f"   Category: LR={category_weights[0]:.3f}, LGB={category_weights[1]:.3f}, XGB={category_weights[2]:.3f}")
print(f"   Misconception: LR={misconception_weights[0]:.3f}, LGB={misconception_weights[1]:.3f}, XGB={misconception_weights[2]:.3f}")

print(f"\nğŸš€ Technical Features:")
print(f"   GPU Acceleration: {'âœ… Enabled' if GPU_AVAILABLE else 'â�Œ Disabled'}")
print(f"   Optuna Trials: {N_TRIALS}")
print(f"   Cross-Validation Folds: {N_SPLITS}")
print(f"   Advanced Features: Text Length + Vocabulary Complexity + Similarity Scores")

total_features = (len(basic_feature_cols) + 
                 len(vocab_feature_names) + 
                 len(similarity_feature_names) + 
                 train_embeddings.shape[1] + 
                 train_tfidf_misc.shape[1])

print(f"\nğŸ“ˆ Feature Engineering:")
print(f"   Basic Text Features: {len(basic_feature_cols)}")
print(f"   Vocabulary Complexity Features: {len(vocab_feature_names)}")
print(f"   Similarity Features: {len(similarity_feature_names)}")
print(f"   TF-IDF Features: {train_embeddings.shape[1] + train_tfidf_misc.shape[1]}")
print(f"   Total Features: {total_features}")

print(f"\nğŸ’¾ Output:")
print(f"   Submission File: {submission_filename}")
print(f"   Ready for Competition: âœ…")

print("="*60)

