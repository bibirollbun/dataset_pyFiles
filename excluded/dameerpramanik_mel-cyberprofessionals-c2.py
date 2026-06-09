import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')
sample_submission = pd.read_csv('/kaggle/input/rmit-hackathon-2025/sample_submission.csv')

print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTraining data info:")
print(train_df.info())
print("\nLabel distribution:")
print(train_df['label'].value_counts())


# Data Exploration and Analysis
def explore_data(df):
    print("First few samples:")
    print(df.head())
    print("\nLabel distribution:")
    print(df['label'].value_counts(normalize=True))
    
    # Text length analysis
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    return df

train_df = explore_data(train_df)

# Visualize text characteristics
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
train_df['label'].value_counts().plot(kind='bar')
plt.title('Label Distribution')

plt.subplot(1, 3, 2)
train_df.groupby('label')['text_length'].plot(kind='kde', legend=True)
plt.title('Text Length by Label')

plt.subplot(1, 3, 3)
train_df.groupby('label')['word_count'].plot(kind='kde', legend=True)
plt.title('Word Count by Label')

plt.tight_layout()
plt.show()


# Advanced Text Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Download NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AdvancedTextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation, and stem
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and token not in self.punctuation]
        
        return ' '.join(tokens)

preprocessor = AdvancedTextPreprocessor()

print("Original text example:", train_df['text'].iloc[0])
print("Preprocessed text example:", preprocessor.preprocess_text(train_df['text'].iloc[0]))


# Feature Engineering
def extract_features(df):
    df = df.copy()
    
    # Basic text features
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['avg_word_length'] = df['text_length'] / df['word_count']
    df['unique_word_ratio'] = df['text'].apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)
    
    # Special character features (common in jailbreak prompts)
    df['special_char_count'] = df['text'].apply(lambda x: len(re.findall(r'[^\w\s]', str(x))))
    df['uppercase_ratio'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
    
    # Suspicious pattern indicators
    jailbreak_keywords = ['ignore', 'bypass', 'override', 'jailbreak', 'hack', 'exploit', 
                         'security', 'filter', 'restriction', 'unauthorized', 'malicious']
    
    for keyword in jailbreak_keywords:
        df[f'contains_{keyword}'] = df['text'].str.contains(keyword, case=False, na=False).astype(int)
    
    return df

# Apply feature engineering
train_df_featured = extract_features(train_df)
test_df_featured = extract_features(test_df)

print("Additional features created:")
print([col for col in train_df_featured.columns if col not in ['id', 'text', 'label']])


# Prepare data for modeling
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Prepare text data
X_text = train_df['text'].apply(preprocessor.preprocess_text)
X_test_text = test_df['text'].apply(preprocessor.preprocess_text)

# Prepare numeric features
numeric_features = ['text_length', 'word_count', 'avg_word_length', 'unique_word_ratio', 
                   'special_char_count', 'uppercase_ratio'] + \
                  [f'contains_{keyword}' for keyword in ['ignore', 'bypass', 'override', 'jailbreak']]

X_numeric = train_df_featured[numeric_features]
X_test_numeric = test_df_featured[numeric_features]

# Target variable
y = (train_df['label'] == 'jailbreak').astype(int)

print("Target distribution:")
print(y.value_counts())


# Ensemble Model with Multiple Approaches
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Approach 1: TF-IDF with Multiple Classifiers
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.8
)

X_tfidf = tfidf_vectorizer.fit_transform(X_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

print("TF-IDF features shape:", X_tfidf.shape)

# Combine TF-IDF with numeric features
from scipy.sparse import hstack

X_combined = hstack([X_tfidf, X_numeric.values])
X_test_combined = hstack([X_test_tfidf, X_test_numeric.values])

print("Combined features shape:", X_combined.shape)


# Model Training with Cross-Validation
from sklearn.model_selection import StratifiedKFold

# Define models
models = {
    'xgboost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    ),
    'lightgbm': LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    'logistic': LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}

print("Performing cross-validation...")
for name, model in models.items():
    if name == 'logistic':
        # Use only TF-IDF features for logistic regression
        scores = cross_val_score(model, X_tfidf, y, cv=cv, scoring='roc_auc')
    else:
        scores = cross_val_score(model, X_combined, y, cv=cv, scoring='roc_auc')
    
    cv_scores[name] = scores
    print(f"{name:12} CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")


# Train Final Ensemble Model
# Use the best performing model or create an ensemble

# Train XGBoost (usually performs well on text data)
final_model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

print("Training final model...")
final_model.fit(X_combined, y)

# Make predictions on test set
test_predictions = final_model.predict_proba(X_test_combined)[:, 1]

print("Prediction statistics:")
print(f"Min probability: {test_predictions.min():.4f}")
print(f"Max probability: {test_predictions.max():.4f}")
print(f"Mean probability: {test_predictions.mean():.4f}")


# Create Submission File
submission = pd.DataFrame({
    'Id': test_df['Id'],          # Capital 'I'
    'TARGET': test_predictions    # All caps
})

# Ensure probabilities are within [0, 1] range
submission['TARGET'] = submission['TARGET'].clip(0, 1)

print("Submission sample:")
print(submission.head(10))

print(f"\nSubmission shape: {submission.shape}")
print(f"TARGET range: [{submission['TARGET'].min():.4f}, {submission['TARGET'].max():.4f}]")

# Save submission
submission_filename = '/kaggle/working/submission.csv'
submission.to_csv(submission_filename, index=False)
print(f"\nSubmission saved as: {submission_filename}")

# Verify submission
print("\nVerifying submission file...")
verify_df = pd.read_csv(submission_filename)
print(f"Verified shape: {verify_df.shape}")
print(f"Verified TARGET range: [{verify_df['TARGET'].min():.4f}, {verify_df['TARGET'].max():.4f}]")


