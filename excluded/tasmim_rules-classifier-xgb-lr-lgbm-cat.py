import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings


warnings.filterwarnings('ignore')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')


# Basic EDA
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print("\nTraining data columns:")
print(train_df.columns)
print("\nTraining data head:")
print(train_df.head())
print("\nTarget distribution:")
print(train_df['rule_violation'].value_counts(normalize=True))
print("\nUnique rules in training:")
print(train_df['rule'].value_counts())
print("\nUnique subreddits in training:")
print(train_df['subreddit'].value_counts())


# Create basic stopwords list (instead of downloading from NLTK)
basic_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}


# Simple lemmatization replacement (without NLTK)
def simple_lemmatize(word):
    # Basic verb conjugations
    if word.endswith('ing'):
        return word[:-3]
    elif word.endswith('ed'):
        return word[:-2]
    elif word.endswith('es'):
        return word[:-2]
    elif word.endswith('s'):
        return word[:-1]
    return word


# Text preprocessing without NLTK
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '&gt;'
    text = re.sub(r'\@\w+|\&gt;', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize and clean
    tokens = text.split()
    tokens = [simple_lemmatize(word) for word in tokens if word not in basic_stopwords]
    
    return ' '.join(tokens)

# Apply cleaning to both train and test
print("Cleaning text data...")
train_df['cleaned_body'] = train_df['body'].apply(clean_text)
test_df['cleaned_body'] = test_df['body'].apply(clean_text)


# # Text preprocessing
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Remove URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
#     # Remove user @ references and '&gt;'
#     text = re.sub(r'\@\w+|\&gt;', '', text)
    
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
    
#     # Remove numbers
#     text = re.sub(r'\d+', '', text)
    
#     # Tokenize and lemmatize
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
#     return ' '.join(tokens)

# # Apply cleaning to both train and test
# train_df['cleaned_body'] = train_df['body'].apply(clean_text)
# test_df['cleaned_body'] = test_df['body'].apply(clean_text)


# Feature engineering
def basic_text_features(text):
    features = {}
    
    # Word count
    features['word_count'] = len(text.split())
    
    # Character count
    features['char_count'] = len(text)
    
    # Average word length
    if features['word_count'] > 0:
        features['avg_word_length'] = features['char_count'] / features['word_count']
    else:
        features['avg_word_length'] = 0
    
    # Count of uppercase letters
    features['uppercase_count'] = sum(1 for c in text if c.isupper())
    
    # Count of special characters
    features['special_char_count'] = sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    return features

# Apply feature engineering
train_features = train_df['cleaned_body'].apply(basic_text_features).apply(pd.Series)
test_features = test_df['cleaned_body'].apply(basic_text_features).apply(pd.Series)


# Combine features with original data
train_df = pd.concat([train_df, train_features], axis=1)
test_df = pd.concat([test_df, test_features], axis=1)

# Prepare data for modeling
X = train_df[['cleaned_body', 'subreddit', 'word_count', 'char_count', 'avg_word_length', 
              'uppercase_count', 'special_char_count']]
y = train_df['rule_violation']

X_test = test_df[['cleaned_body', 'subreddit', 'word_count', 'char_count', 'avg_word_length', 
                  'uppercase_count', 'special_char_count']]


# Split into train and validation sets
print(f"train test split")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing for different feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), 'cleaned_body'),
        ('subreddit', OneHotEncoder(handle_unknown='ignore'), ['subreddit']),
        ('num', 'passthrough', ['word_count', 'char_count', 'avg_word_length', 
                                'uppercase_count', 'special_char_count'])
    ])

# Model 1: Logistic Regression with TF-IDF and features
print(f"Logistic regression training")
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', CalibratedClassifierCV(LogisticRegression(max_iter=1000)))
])

lr_pipeline.fit(X_train, y_train)
lr_val_preds = lr_pipeline.predict_proba(X_val)[:, 1]
lr_auc = roc_auc_score(y_val, lr_val_preds)
print(f"Logistic Regression AUC: {lr_auc:.4f}")

# Model 2: XGBoost with TF-IDF and features
print(f"XGB training")
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    ))
])

xgb_pipeline.fit(X_train, y_train)
xgb_val_preds = xgb_pipeline.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_val_preds)
print(f"XGBoost AUC: {xgb_auc:.4f}")

# Model 3: LightGBM with TF-IDF and features
print(f"LightGBM training")
lgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        metric='auc'
    ))
])

lgb_pipeline.fit(X_train, y_train)
lgb_val_preds = lgb_pipeline.predict_proba(X_val)[:, 1]
lgb_auc = roc_auc_score(y_val, lgb_val_preds)
print(f"LightGBM AUC: {lgb_auc:.4f}")

# Model 4: CatBoost with text features
# CatBoost can handle text features directly, so we'll use a different approach
print(f"Catboost training")
cat_features = ['subreddit']
text_features = ['cleaned_body']

# For CatBoost, we'll combine all features
train_pool = cb.Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_features,
    text_features=text_features
)

val_pool = cb.Pool(
    data=X_val,
    label=y_val,
    cat_features=cat_features,
    text_features=text_features
)

cat_model = cb.CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=0
)

cat_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=20, verbose=0)
cat_val_preds = cat_model.predict_proba(val_pool)[:, 1]
cat_auc = roc_auc_score(y_val, cat_val_preds)
print(f"CatBoost AUC: {cat_auc:.4f}")


# Ensemble predictions (simple average)
ensemble_val_preds = (lr_val_preds + xgb_val_preds + lgb_val_preds + cat_val_preds) / 4
ensemble_auc = roc_auc_score(y_val, ensemble_val_preds)
print(f"Ensemble AUC: {ensemble_auc:.4f}")

# Train final models on full training data
print("\nTraining final models on full data...")

# Logistic Regression
lr_pipeline.fit(X, y)

# XGBoost
xgb_pipeline.fit(X, y)

# LightGBM
lgb_pipeline.fit(X, y)

# CatBoost
full_pool = cb.Pool(
    data=X,
    label=y,
    cat_features=cat_features,
    text_features=text_features
)
cat_model.fit(full_pool, verbose=0)

# Generate test predictions
print("\nGenerating test predictions...")

# Logistic Regression
lr_test_preds = lr_pipeline.predict_proba(X_test)[:, 1]

# XGBoost
xgb_test_preds = xgb_pipeline.predict_proba(X_test)[:, 1]

# LightGBM
lgb_test_preds = lgb_pipeline.predict_proba(X_test)[:, 1]

# CatBoost
test_pool = cb.Pool(
    data=X_test,
    cat_features=cat_features,
    text_features=text_features
)
cat_test_preds = cat_model.predict_proba(test_pool)[:, 1]

# Ensemble predictions (weighted average based on validation performance)
weights = {
    'lr': lr_auc,
    'xgb': xgb_auc,
    'lgb': lgb_auc,
    'cat': cat_auc
}
total_weight = sum(weights.values())
weighted_ensemble_test_preds = (
    weights['lr'] * lr_test_preds +
    weights['xgb'] * xgb_test_preds +
    weights['lgb'] * lgb_test_preds +
    weights['cat'] * cat_test_preds
) / total_weight


# Create submission file
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': weighted_ensemble_test_preds
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")


submission.head()




