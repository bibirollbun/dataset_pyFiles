import numpy as np
import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import textstat
import re
from scipy.sparse import hstack
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
from pathlib import Path
import os
import pickle
warnings.filterwarnings('ignore')

# NLTK downloads (run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

ROOT_DIR = "../"
MODELS_DIR = Path('/kaggle/input/trained-models-multi-level-text-analysis/trained_models')

print(f"Models directory: {MODELS_DIR}")
print(f"Models directory exists: {MODELS_DIR.exists()}")


print("ğŸ”„ Loading trained models and feature pipeline...")

# Load feature configuration
with open(MODELS_DIR / 'feature_config.pkl', 'rb') as f:
    feature_config = pickle.load(f)

USE_BERT_EMBEDDINGS = feature_config['USE_BERT_EMBEDDINGS']
basic_meta_cols = feature_config['basic_meta_cols']
advanced_feature_names = feature_config['advanced_feature_names']
tfidf_feature_names = feature_config['tfidf_feature_names']
svd_components = feature_config['svd_components']
n_classes1 = feature_config['n_classes1']
n_classes2 = feature_config['n_classes2']

print(f"Feature configuration loaded:")
print(f"   BERT embeddings: {'âœ…' if USE_BERT_EMBEDDINGS else 'â�Œ'}")
print(f"   Basic meta features: {len(basic_meta_cols)}")
print(f"   Advanced NLP features: {len(advanced_feature_names)}")
print(f"   TF-IDF variants: {len(tfidf_feature_names)}")

# Load TF-IDF vectorizers
with open(MODELS_DIR / 'tfidf_vectorizers.pkl', 'rb') as f:
    tfidf_vectorizers = pickle.load(f)

# Load SVD transformer
with open(MODELS_DIR / 'svd_transformer.pkl', 'rb') as f:
    svd = pickle.load(f)

# Load feature scaler
with open(MODELS_DIR / 'feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load target mappings
with open(MODELS_DIR / 'target_mappings.pkl', 'rb') as f:
    target_mappings = pickle.load(f)

map_target1 = target_mappings['map_target1']
map_target2 = target_mappings['map_target2']
map_inverse1 = target_mappings['map_inverse1']
map_inverse2 = target_mappings['map_inverse2']

# Load trained models
with open(MODELS_DIR / 'category_models.pkl', 'rb') as f:
    category_models = pickle.load(f)

with open(MODELS_DIR / 'misconception_models.pkl', 'rb') as f:
    misconception_models = pickle.load(f)

# Load BERT info
with open(MODELS_DIR / 'bert_info.pkl', 'rb') as f:
    bert_info = pickle.load(f)

test_bert = None
if bert_info['model_available']:
    test_bert = bert_info['test_bert']
    print(f"   BERT embeddings loaded: {test_bert.shape}")

print(f"âœ… All models and feature pipeline loaded successfully!")
print(f"   Category models: {len(category_models['lr'])} folds")
print(f"   Misconception models: {len(misconception_models['lr'])} folds")


# Load test data
test = pl.read_csv(f"{ROOT_DIR}input/map-charting-student-math-misunderstandings/test.csv")

print(f"Test shape: {test.shape}")
print(f"Test columns: {test.columns}")

# Create sentence (same as training)
def create_sentence(row):
    return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\nExplanation: {row['StudentExplanation']}"

test = test.with_columns([
    pl.struct(['QuestionText', 'MC_Answer', 'StudentExplanation']).map_elements(
        create_sentence, return_dtype=pl.Utf8
    ).alias('sentence')
])

test_sentences = test['sentence'].to_list()

print("Sample test sentence:")
print(test_sentences[0][:200] + "...")


print("ğŸ”„ Generating TF-IDF features for test data...")

# Generate TF-IDF features using pre-trained vectorizers
test_tfidf_features = []

for vectorizer_name in ['tfidf1', 'tfidf2', 'tfidf3']:
    if vectorizer_name in tfidf_vectorizers:
        vectorizer = tfidf_vectorizers[vectorizer_name]
        test_tfidf = vectorizer.transform(test_sentences)
        test_tfidf_features.append(test_tfidf)
        print(f"   {vectorizer_name}: {test_tfidf.shape}")

# Add extra vectorizers if available
for vectorizer_name in ['tfidf4', 'tfidf5']:
    if vectorizer_name in tfidf_vectorizers:
        vectorizer = tfidf_vectorizers[vectorizer_name]
        test_tfidf = vectorizer.transform(test_sentences)
        test_tfidf_features.append(test_tfidf)
        print(f"   {vectorizer_name}: {test_tfidf.shape}")

# Combine TF-IDF features
test_tfidf_combined = hstack(test_tfidf_features)
print(f"Combined TF-IDF shape: {test_tfidf_combined.shape}")

# Apply SVD reduction
test_tfidf_reduced = svd.transform(test_tfidf_combined)
print(f"Reduced TF-IDF shape: {test_tfidf_reduced.shape}")


def extract_comprehensive_features(df):
    """Extract comprehensive meta features (same as training)"""
    
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

def extract_advanced_nlp_features(texts):
    """Extract advanced NLP features (same as training)"""
    
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

print("ğŸ”„ Generating meta features for test data...")

# Add basic meta features
test = extract_comprehensive_features(test)
test_basic_meta = test.select(basic_meta_cols).to_numpy().astype(np.float32)

# Generate advanced NLP features
test_advanced = extract_advanced_nlp_features(test_sentences)

print(f"Basic meta features shape: {test_basic_meta.shape}")
print(f"Advanced NLP features shape: {test_advanced.shape}")


# Combine all features (same order as training)
feature_components = [
    (test_tfidf_reduced, "TF-IDF (SVD)"),
    (test_basic_meta, "Basic Meta"),
    (test_advanced, "Advanced NLP")
]

# Add BERT if available
if USE_BERT_EMBEDDINGS and test_bert is not None:
    feature_components.insert(1, (test_bert, "BERT Embeddings"))

# Combine features
test_features_list = [comp[0] for comp in feature_components]
test_features = np.hstack(test_features_list)

print(f"\nğŸ�¯ Final test feature dimensions:")
for test_feat, name in feature_components:
    print(f"   {name}: {test_feat.shape[1]}")
print(f"   Total: {test_features.shape[1]}")

# Apply feature normalization (using pre-fitted scaler)
test_features_scaled = scaler.transform(test_features)

print(f"\nâœ… Test features prepared: {test_features_scaled.shape}")


print("ğŸ”„ Running inference for Category classification...")

# Initialize prediction arrays
ytest1_lr = np.zeros((len(test), n_classes1))
ytest1_lgb = np.zeros((len(test), n_classes1))
ytest1_xgb = np.zeros((len(test), n_classes1))
ytest1_rf = np.zeros((len(test), n_classes1))

n_folds = len(category_models['lr'])

# Run inference with all trained folds
for fold in range(n_folds):
    print(f"Category inference fold {fold+1}/{n_folds}")
    
    # Logistic Regression
    lr_model = category_models['lr'][fold]
    ytest1_lr += lr_model.predict_proba(test_features_scaled) / n_folds
    
    # LightGBM
    lgb_model = category_models['lgb'][fold]
    ytest1_lgb += lgb_model.predict_proba(test_features_scaled) / n_folds
    
    # XGBoost
    xgb_model = category_models['xgb'][fold]
    ytest1_xgb += xgb_model.predict_proba(test_features_scaled) / n_folds
    
    # Random Forest
    rf_model = category_models['rf'][fold]
    ytest1_rf += rf_model.predict_proba(test_features_scaled) / n_folds

# Ensemble (same weights as training)
if USE_BERT_EMBEDDINGS:
    ytest1 = 0.3 * ytest1_lr + 0.35 * ytest1_lgb + 0.25 * ytest1_xgb + 0.1 * ytest1_rf
else:
    ytest1 = 0.25 * ytest1_lr + 0.35 * ytest1_lgb + 0.25 * ytest1_xgb + 0.15 * ytest1_rf

print(f"âœ… Category classification completed: {ytest1.shape}")


print("ğŸ”„ Running inference for Misconception classification...")

# Initialize prediction arrays
ytest2_lr = np.zeros((len(test), n_classes2))
ytest2_lgb = np.zeros((len(test), n_classes2))
ytest2_xgb = np.zeros((len(test), n_classes2))
ytest2_rf = np.zeros((len(test), n_classes2))

# Run inference with all trained folds
for fold in range(n_folds):
    print(f"Misconception inference fold {fold+1}/{n_folds}")
    
    # Logistic Regression
    lr_model = misconception_models['lr'][fold]
    ytest2_lr += lr_model.predict_proba(test_features_scaled) / n_folds
    
    # LightGBM
    lgb_model = misconception_models['lgb'][fold]
    ytest2_lgb += lgb_model.predict_proba(test_features_scaled) / n_folds
    
    # XGBoost
    xgb_model = misconception_models['xgb'][fold]
    ytest2_xgb += xgb_model.predict_proba(test_features_scaled) / n_folds
    
    # Random Forest
    rf_model = misconception_models['rf'][fold]
    ytest2_rf += rf_model.predict_proba(test_features_scaled) / n_folds

# Ensemble (same weights as training)
if USE_BERT_EMBEDDINGS:
    ytest2 = 0.3 * ytest2_lr + 0.3 * ytest2_lgb + 0.3 * ytest2_xgb + 0.1 * ytest2_rf
else:
    ytest2 = 0.35 * ytest2_lr + 0.35 * ytest2_lgb + 0.2 * ytest2_xgb + 0.1 * ytest2_rf

print(f"âœ… Misconception classification completed: {ytest2.shape}")


print("ğŸ”„ Generating final predictions...")

# Filter misconception predictions (set NA class probability to 0)
ytest2_filtered = ytest2.copy()
ytest2_filtered[:, 0] = 0  # Set NA class probability to 0

# Get top 3 predictions for each classification task
predicted1_test = np.argsort(-ytest1, axis=1)[:, :3]
predicted2_test = np.argsort(-ytest2_filtered, axis=1)[:, :3]

# Create final predictions
predict_test = []
for i in range(len(predicted1_test)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1_test[i, j]]
        p2 = map_inverse2[predicted2_test[i, j]]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2)
        else:
            pred.append(p1 + ":NA")
    predict_test.append(" ".join(pred))

print(f"âœ… Generated {len(predict_test)} predictions")

# Create submission file
submission_filename = "submission.csv"

sub = pl.read_csv(f"{ROOT_DIR}input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub = sub.with_columns([
    pl.Series('Category:Misconception', predict_test)
])

sub.write_csv(submission_filename)
print(f"\nâœ… Submission file '{submission_filename}' created successfully!")
print(f"ğŸ“‹ Sample predictions:")
print(sub.head())

# Show prediction confidence
top1_confidence_cat = np.mean(np.max(ytest1, axis=1))
top1_confidence_misc = np.mean(np.max(ytest2_filtered, axis=1))

print(f"\nğŸ“Š Prediction Statistics:")
print(f"   Average top-1 confidence (Category): {top1_confidence_cat:.4f}")
print(f"   Average top-1 confidence (Misconception): {top1_confidence_misc:.4f}")
print(f"   Feature configuration: {'BERT + TF-IDF + Meta' if USE_BERT_EMBEDDINGS else 'TF-IDF + Meta'}")


print("\n" + "="*60)
print("ğŸ�¯ INFERENCE COMPLETED - EXECUTION SUMMARY")
print("="*60)

print(f"\nğŸ“Š Model Configuration:")
if USE_BERT_EMBEDDINGS:
    print(f"   âœ… BERT embeddings loaded from training")
else:
    print(f"   â�Œ BERT embeddings not used")
print(f"   âœ… TF-IDF features ({len(tfidf_feature_names)} variants)")
print(f"   âœ… Meta features ({len(basic_meta_cols) + len(advanced_feature_names)} total)")
print(f"   âœ… {n_folds}-fold ensemble (LR + LGB + XGB + RF)")

print(f"\nğŸ“ˆ Inference Results:")
print(f"   Test samples processed: {len(test)}")
print(f"   Category classes: {n_classes1}")
print(f"   Misconception classes: {n_classes2}")
print(f"   Predictions per sample: 3")

print(f"\nğŸ“� Output:")
print(f"   Submission file: {submission_filename}")
print(f"   Ready for Kaggle submission: âœ…")

print(f"\nğŸ’¡ Performance Notes:")
print(f"   - Inference completed using pre-trained models")
print(f"   - Feature pipeline identical to training")
print(f"   - No internet connection required")
print(f"   - Fast inference suitable for competition time limits")

print("="*60)

