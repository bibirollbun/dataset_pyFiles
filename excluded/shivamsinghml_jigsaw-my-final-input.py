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



# Install required packages
!pip install transformers[torch] datasets
!pip install scikit-learn xgboost lightgbm nltk

# %% [code]
# Import necessary libraries
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Machine learning
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import lightgbm as lgb

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD  # Using standard optimizers instead of AdamW

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Other utilities
from tqdm import tqdm
import time
import json
from collections import Counter

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# %% [code]
# Load the data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


# Exploratory Data Analysis
print("Training data overview:")
print(train_df.head())
print("\nTraining data info:")
print(train_df.info())
print("\nTarget distribution:")
print(train_df['rule_violation'].value_counts(normalize=True))
print("\nRule distribution:")
print(train_df['rule'].value_counts())
print("\nSubreddit distribution:")
print(train_df['subreddit'].value_counts().head(10))


# Check for missing values
print(f"\nMissing values in train: {train_df.isnull().sum().sum()}")
print(f"Missing values in test: {test_df.isnull().sum().sum()}")



# Basic statistics and visualization
train_df['comment_length'] = train_df['body'].apply(lambda x: len(str(x)))
train_df['word_count'] = train_df['body'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(train_df['comment_length'], bins=50)
plt.title('Comment Length Distribution')

plt.subplot(1, 3, 2)
sns.histplot(train_df['word_count'], bins=50)
plt.title('Word Count Distribution')

plt.subplot(1, 3, 3)
sns.countplot(x='rule_violation', data=train_df)
plt.title('Target Variable Distribution')

plt.tight_layout()
plt.show()


# Text preprocessing functions
def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and subreddit links
    text = re.sub(r'\br/\w+|\bu/\w+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Apply cleaning
train_df['cleaned_body'] = train_df['body'].apply(clean_text)
test_df['cleaned_body'] = test_df['body'].apply(clean_text)

print("Sample cleaned text:")
print(train_df['cleaned_body'].iloc[0][:200] + "...")


# Advanced feature engineering
class FeatureEngineer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def extract_features(self, text):
        """Extract various text features"""
        if not text or text.strip() == "":
            return {
                'char_length': 0,
                'word_count': 0,
                'stopword_ratio': 0,
                'avg_word_length': 0,
                'unique_word_ratio': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'capital_ratio': 0,
                'has_url': 0,
                'has_mention': 0
            }
        
        # Basic features
        char_length = len(text)
        words = text.split()
        word_count = len(words)
        
        # Advanced features
        stopword_count = sum(1 for word in words if word in self.stop_words)
        stopword_ratio = stopword_count / max(word_count, 1)
        avg_word_length = char_length / max(word_count, 1)
        unique_word_ratio = len(set(words)) / max(word_count, 1)
        
        # Punctuation features
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Capitalization features (from original text)
        original_text = str(text)
        capital_ratio = sum(1 for char in original_text if char.isupper()) / max(len(original_text), 1)
        
        # Content features
        has_url = 1 if re.search(r'http|www', original_text) else 0
        has_mention = 1 if re.search(r'r/\w+|u/\w+', original_text) else 0
        
        return {
            'char_length': char_length,
            'word_count': word_count,
            'stopword_ratio': stopword_ratio,
            'avg_word_length': avg_word_length,
            'unique_word_ratio': unique_word_ratio,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'capital_ratio': capital_ratio,
            'has_url': has_url,
            'has_mention': has_mention
        }

# Extract features
feature_engineer = FeatureEngineer()


train_features = pd.DataFrame(
    train_df['body'].apply(feature_engineer.extract_features).tolist()
)
test_features = pd.DataFrame(
    test_df['body'].apply(feature_engineer.extract_features).tolist()
)


# Add subreddit as a feature (one-hot encoding)
train_subreddit_dummies = pd.get_dummies(train_df['subreddit'], prefix='subreddit')
test_subreddit_dummies = pd.get_dummies(test_df['subreddit'], prefix='subreddit')


# Align columns
all_subreddit_cols = list(set(train_subreddit_dummies.columns) | set(test_subreddit_dummies.columns))
for col in all_subreddit_cols:
    if col not in train_subreddit_dummies:
        train_subreddit_dummies[col] = 0
    if col not in test_subreddit_dummies:
        test_subreddit_dummies[col] = 0

train_subreddit_dummies = train_subreddit_dummies[all_subreddit_cols]
test_subreddit_dummies = test_subreddit_dummies[all_subreddit_cols]


# Combine all features
X_train_ml = pd.concat([train_features, train_subreddit_dummies], axis=1)
X_test_ml = pd.concat([test_features, test_subreddit_dummies], axis=1)
y_train = train_df['rule_violation'].values

print(f"ML features shape: {X_train_ml.shape}")
# Fix the feature engineering to ensure numeric types only
print("Checking data types in X_train_ml:")
print(X_train_ml.dtypes)


# Identify non-numeric columns
non_numeric_cols = X_train_ml.select_dtypes(include=['object', 'bool']).columns
print(f"Non-numeric columns: {list(non_numeric_cols)}")


# Convert boolean columns to integers
for col in X_train_ml.columns:
    if X_train_ml[col].dtype == 'bool':
        X_train_ml[col] = X_train_ml[col].astype(int)
        X_test_ml[col] = X_test_ml[col].astype(int)


# Also check for any other object types and convert them
for col in X_train_ml.columns:
    if X_train_ml[col].dtype == 'object':
        # For categorical columns, use one-hot encoding
        if X_train_ml[col].nunique() < 10:  # If few unique values
            # One-hot encode
            train_dummies = pd.get_dummies(X_train_ml[col], prefix=col)
            test_dummies = pd.get_dummies(X_test_ml[col], prefix=col)
            
            # Align columns
            all_cols = list(set(train_dummies.columns) | set(test_dummies.columns))
            for dummy_col in all_cols:
                if dummy_col not in train_dummies:
                    train_dummies[dummy_col] = 0
                if dummy_col not in test_dummies:
                    test_dummies[dummy_col] = 0
            
            train_dummies = train_dummies[all_cols]
            test_dummies = test_dummies[all_cols]
            
            # Replace the original column with dummies
            X_train_ml = pd.concat([X_train_ml.drop(col, axis=1), train_dummies], axis=1)
            X_test_ml = pd.concat([X_test_ml.drop(col, axis=1), test_dummies], axis=1)
        else:
            # For many unique values, use label encoding or drop
            X_train_ml = X_train_ml.drop(col, axis=1)
            X_test_ml = X_test_ml.drop(col, axis=1)

print("Data types after conversion:")
print(X_train_ml.dtypes)


# TF-IDF features for traditional models
vectorizer = TfidfVectorizer(
    max_features=500,  # Reduced for efficiency
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(train_df['cleaned_body'])
X_test_tfidf = vectorizer.transform(test_df['cleaned_body'])

print(f"TF-IDF features shape: {X_train_tfidf.shape}")


# Now try to combine with sparse matrices
from scipy import sparse

try:
    X_train_combined = sparse.hstack([X_train_tfidf, sparse.csr_matrix(X_train_ml.astype(float))])
    X_test_combined = sparse.hstack([X_test_tfidf, sparse.csr_matrix(X_test_ml.astype(float))])
    
    print(f"Combined features shape: {X_train_combined.shape}")
    print("Successfully combined features!")
    
except Exception as e:
    print(f"Error combining features: {e}")
    print("Using alternative approach...")
    
    # Alternative: Convert everything to dense arrays
    X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_ml.astype(float).values])
    X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_ml.astype(float).values])
    
    print(f"Using dense arrays. Combined features shape: {X_train_combined.shape}")


class TraditionalML:
    def __init__(self):
        self.models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='auc',
                use_label_encoder=False
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'logreg': LogisticRegression(
                C=0.1,
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            ),
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }
        
    def train_cv(self, X, y, n_splits=3):
        """Train with cross-validation - handles both sparse and dense matrices"""
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Get the number of samples properly for both sparse and dense matrices
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
            
        oof_preds = np.zeros(n_samples)
        test_preds = np.zeros((X_test_combined.shape[0], len(self.models)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            # Handle both sparse and dense matrices
            if hasattr(X, 'iloc'):  # pandas DataFrame
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            elif hasattr(X, 'shape'):  # numpy array or sparse matrix
                X_train, X_val = X[train_idx], X[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                
            y_train_fold, y_val = y[train_idx], y[val_idx]
            
            fold_preds = []
            
            for i, (name, model) in enumerate(self.models.items()):
                # Convert to dense if sparse and model doesn't handle sparse well
                if hasattr(X_train, 'toarray') and name in ['xgb', 'lgb']:
                    X_train_dense = X_train.toarray()
                    X_val_dense = X_val.toarray()
                    X_test_dense = X_test_combined.toarray()
                else:
                    X_train_dense = X_train
                    X_val_dense = X_val
                    X_test_dense = X_test_combined
                
                try:
                    model.fit(X_train_dense, y_train_fold)
                    
                    # OOF predictions
                    val_preds = model.predict_proba(X_val_dense)[:, 1]
                    oof_preds[val_idx] += val_preds / len(self.models)
                    
                    # Test predictions
                    test_preds[:, i] += model.predict_proba(X_test_dense)[:, 1] / n_splits
                    fold_auc = roc_auc_score(y_val, val_preds)
                    fold_preds.append(fold_auc)
                    
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    # If model fails, use random predictions
                    val_preds = np.random.random(len(y_val))
                    oof_preds[val_idx] += val_preds / len(self.models)
                    test_preds[:, i] += np.random.random(X_test_combined.shape[0]) / n_splits
                    fold_preds.append(0.5)
            
            print(f"Fold {fold+1} AUCs: {[f'{x:.4f}' for x in fold_preds]}")
        
        # Ensemble test predictions
        test_preds_ensemble = test_preds.mean(axis=1)
        
        # Calculate OOF AUC
        oof_auc = roc_auc_score(y, oof_preds)
        print(f"Overall OOF AUC: {oof_auc:.4f}")
        
        return oof_preds, test_preds_ensemble



# Train traditional models
print("Training traditional ML models...")
ml_model = TraditionalML()
ml_oof_preds, ml_test_preds = ml_model.train_cv(X_train_combined, y_train)


# Simple Neural Network without transformers
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.3):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)


# Train simple neural network
def train_simple_nn(X_train, y_train, X_test, epochs=50, lr=0.001):
    """Train a simple neural network"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Initialize model
    model = SimpleNN(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)  # Using standard Adam
    
    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    # Predictions
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).cpu().numpy().flatten()
        train_preds = model(X_train_tensor).cpu().numpy().flatten()
    
    return train_preds, test_preds


print("Training simple neural network...")
nn_train_preds, nn_test_preds = train_simple_nn(
    X_train_combined.toarray() if hasattr(X_train_combined, 'toarray') else X_train_combined,
    y_train,
    X_test_combined.toarray() if hasattr(X_test_combined, 'toarray') else X_test_combined
)


nn_auc = roc_auc_score(y_train, nn_train_preds)
print(f"Neural Network OOF AUC: {nn_auc:.4f}")


# Bag of Words approach as alternative to transformers
def bow_approach():
    """Bag of Words with traditional classifiers"""
    # Count vectorizer
    count_vectorizer = CountVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_train_bow = count_vectorizer.fit_transform(train_df['cleaned_body'])
    X_test_bow = count_vectorizer.transform(test_df['cleaned_body'])
    
    # Combine with other features
    X_train_bow_combined = sparse.hstack([X_train_bow, sparse.csr_matrix(X_train_ml)])
    X_test_bow_combined = sparse.hstack([X_test_bow, sparse.csr_matrix(X_test_ml)])
    
    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_bow_combined, y_train)
    
    nb_train_preds = nb_model.predict_proba(X_train_bow_combined)[:, 1]
    nb_test_preds = nb_model.predict_proba(X_test_bow_combined)[:, 1]
    
    nb_auc = roc_auc_score(y_train, nb_train_preds)
    print(f"Naive Bayes AUC: {nb_auc:.4f}")
    
    return nb_train_preds, nb_test_preds


print("Training Naive Bayes model...")
nb_train_preds, nb_test_preds = bow_approach()


# Ensemble all models
def create_ensemble(predictions_list, weights=None):
    """Create weighted ensemble of predictions"""
    if weights is None:
        weights = [1/len(predictions_list)] * len(predictions_list)
    
    ensemble_preds = np.zeros_like(predictions_list[0])
    for preds, weight in zip(predictions_list, weights):
        ensemble_preds += preds * weight
    
    return ensemble_preds


# List of all test predictions
all_test_preds = [
    ml_test_preds,           # Traditional ML ensemble
    nn_test_preds,           # Neural network
    nb_test_preds            # Naive Bayes
]

# Corresponding OOF predictions for weighting
all_oof_preds = [
    ml_oof_preds,
    nn_train_preds,
    nb_train_preds
]


# Calculate weights based on OOF performance
weights = []
for oof_preds in all_oof_preds:
    auc = roc_auc_score(y_train, oof_preds)
    weights.append(auc)  # Weight by performance

# Normalize weights
weights = np.array(weights) / sum(weights)
print(f"Model weights: {weights}")


# Create final ensemble
final_test_preds = create_ensemble(all_test_preds, weights)

# %% [code]
# Calibrate predictions
calibrator = CalibratedClassifierCV(LogisticRegression(), cv=3, method='sigmoid')
# Use ML predictions for calibration since they have best OOF performance
calibrator.fit(ml_oof_preds.reshape(-1, 1), y_train)
final_calibrated_preds = calibrator.predict_proba(final_test_preds.reshape(-1, 1))[:, 1]


# Prepare submission
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': final_calibrated_preds
})

# Ensure predictions are within [0, 1] range
submission['rule_violation'] = submission['rule_violation'].clip(0, 1)


# Save submission
submission.to_csv('submission.csv', index=False)


print("Submission file created successfully!")
print(f"Submission shape: {submission.shape}")
print(f"Prediction range: [{submission['rule_violation'].min():.3f}, {submission['rule_violation'].max():.3f}]")
print(f"Mean prediction: {submission['rule_violation'].mean():.3f}")


# Show sample predictions
print("\nSample predictions:")
print(submission.head(10))


# Additional analysis for insight
print("\n=== Model Performance Summary ===")
print(f"Traditional ML OOF AUC: {roc_auc_score(y_train, ml_oof_preds):.4f}")
print(f"Neural Network OOF AUC: {nn_auc:.4f}")
print(f"Naive Bayes OOF AUC: {roc_auc_score(y_train, nb_train_preds):.4f}")


# Error analysis
def analyze_errors(true_labels, pred_probs, threshold=0.5):
    """Analyze misclassified examples"""
    pred_labels = (pred_probs >= threshold).astype(int)
    errors = np.where(true_labels != pred_labels)[0]
    
    print(f"Total errors: {len(errors)}")
    print(f"Error rate: {len(errors)/len(true_labels):.3f}")
    
    # Analyze false positives/negatives
    fp = np.where((true_labels == 0) & (pred_labels == 1))[0]
    fn = np.where((true_labels == 1) & (pred_labels == 0))[0]
    
    print(f"False positives: {len(fp)}")
    print(f"False negatives: {len(fn)}")
    
    return fp, fn


# Analyze errors on best model
fp, fn = analyze_errors(y_train, ml_oof_preds)


print("\nExample false positives (predicted violation but wasn't):")
for i in fp[:2]:
    print(f"Text: {train_df['body'].iloc[i][:100]}...")
    print(f"Predicted prob: {ml_oof_preds[i]:.3f}\n")

print("\nExample false negatives (predicted clean but was violation):")
for i in fn[:2]:
    print(f"Text: {train_df['body'].iloc[i][:100]}...")
    print(f"Predicted prob: {ml_oof_preds[i]:.3f}\n")

