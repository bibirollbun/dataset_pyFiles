# -*- coding: utf-8 -*-
"""
MAP_Charting_Preprocessing_and_Features.py

This notebook handles data loading, text preprocessing, feature engineering,
and target encoding for the "Map Charting Student Math Misunderstandings" competition.
It downloads necessary NLTK data and saves all processed artifacts for offline use
in a subsequent model training notebook.
"""

import numpy as np
import pandas as pd
import cudf # Included for consistency, though not heavily used for Pandas operations here
import cuml # Included for consistency, though not heavily used for Pandas operations here
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
from nltk.stem import WordNetLemmatizer
import nltk
from scipy import sparse # For saving sparse matrices
import os # For creating directories and managing paths
import pickle # For saving Python objects like LabelEncoder and dictionaries

import warnings
warnings.filterwarnings('ignore')

print("--- Starting Preprocessing and Feature Generation Notebook ---")

# --- NLTK Data Download (Requires Internet) ---
# Ensure NLTK data is downloaded. This part runs in the notebook with internet.
# The downloaded data will be implicitly saved with the notebook's output.
try:
    nltk.data.find('corpora/wordnet')
    print("'wordnet' already found.")
except LookupError:
    print("Downloading 'wordnet' NLTK corpus...")
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
    print("'omw-1.4' already found.")
except LookupError:
    print("Downloading 'omw-1.4' NLTK corpus...")
    nltk.download('omw-1.4', quiet=True)

print("NLTK data setup complete.")

# --- Text Cleaning and Feature Extraction Functions ---

def advanced_clean(text):
    """Advanced text cleaning."""
    # Detect math patterns (e.g., "1/2", "\frac{a}{b}") and replace with a token
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'FRAC_\1_\2', text)

    # Basic cleaning: remove newlines, extra spaces, and most punctuation
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_]', '', text) # Keep alphanumeric, spaces, and underscore

    return text.strip().lower()

def extract_math_features(text):
    """Extract mathematical features."""
    features = {}
    # Count occurrences of fraction tokens
    features['frac_count'] = len(re.findall(r'FRAC_\d+_\d+|\\frac', text))
    # Count standalone numbers
    features['number_count'] = len(re.findall(r'\b\d+\b', text))
    # Count common mathematical operators
    features['operator_count'] = len(re.findall(r'[\+\-\*\/\=]', text))
    return features

def fast_lemmatize(text):
    """Lemmatize words using WordNetLemmatizer."""
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def create_numeric_features(df):
    """Create basic numeric features (lengths, ratios, math features)."""
    # Basic length features
    df['mc_answer_len'] = df['MC_Answer'].astype(str).str.len()
    df['explanation_len'] = df['StudentExplanation'].astype(str).str.len()
    df['question_len'] = df['QuestionText'].astype(str).str.len()

    # Ratio feature
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)

    # Extract mathematical features from QuestionText and MC_Answer
    for col in ['QuestionText', 'MC_Answer']:
        math_features = df[col].apply(extract_math_features).apply(pd.Series)
        prefix = 'mc_' if col == 'MC_Answer' else ''
        math_features.columns = [f'{prefix}{c}' for c in math_features.columns]
        df = pd.concat([df, math_features], axis=1)
    return df

# --- Data Loading and Initial Preparation ---
print("--- Loading Raw Data ---")
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

# Fill missing 'Misconception' values with 'NA' and ensure string type
train['Misconception'] = train['Misconception'].fillna('NA').astype(str)
# Create a combined target category
train['target_cat'] = train.apply(lambda x: x['Category'] + ":" + x['Misconception'], axis=1)

print(f"Raw Train shape: {train.shape}, Raw Test shape: {test.shape}")

# --- Target Encoding ---
print("\n--- Encoding Targets ---")
le_target = LabelEncoder()
train['target_encoded'] = le_target.fit_transform(train['target_cat'])
target_classes = le_target.classes_ # Store class names for inverse transformation later
n_classes = len(target_classes)
print(f"Number of target classes: {n_classes}")

# Create mappings for Category and Misconception to numerical targets (for MAP@3 reconstruction)
map_target1 = {category: i for i, category in enumerate(train['Category'].unique())}
map_target2 = {misconception: i for i, misconception in enumerate(train['Misconception'].unique())}

# --- Feature Engineering (Numeric) ---
print("\n--- Creating Numeric Features ---")
train = create_numeric_features(train)
test = create_numeric_features(test)

# Define numeric features to be used
numeric_features = [
    'mc_answer_len', 'explanation_len', 'question_len',
    'explanation_to_question_ratio', 'frac_count', 'number_count',
    'operator_count', 'mc_frac_count', 'mc_number_count',
    'mc_operator_count'
]
# Filter to ensure only existing columns are selected
numeric_features = [f for f in numeric_features if f in train.columns]

# Extract and fill NaNs for numeric features
X_numeric_train = train[numeric_features].fillna(0).values
X_numeric_test = test[numeric_features].fillna(0).values
print(f"Train numeric features shape: {X_numeric_train.shape}")
print(f"Test numeric features shape: {X_numeric_test.shape}")


# --- Text Preprocessing and TF-IDF Vectorization ---
print("\n--- Processing Text and Creating TF-IDF Features ---")

# Combine relevant text columns into a single 'sentence' feature
train['combined_text'] = (
    "Question: " + train['QuestionText'].astype(str) +
    " Answer: " + train['MC_Answer'].astype(str) +
    " Explanation: " + train['StudentExplanation'].astype(str)
)
test['combined_text'] = (
    "Question: " + test['QuestionText'].astype(str) +
    " Answer: " + test['MC_Answer'].astype(str) +
    " Explanation: " + test['StudentExplanation'].astype(str)
)

# Apply advanced cleaning and lemmatization
train['cleaned_text'] = train['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
test['cleaned_text'] = test['combined_text'].apply(advanced_clean).apply(fast_lemmatize)

print("Sample cleaned sentence (train):", train['cleaned_text'].iloc[0])

# Initialize TF-IDF Vectorizer
tfidf_model = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),
    max_df=0.95,
    min_df=2,
    max_features=5000 # Limiting features for efficiency
)

# Fit TF-IDF on combined train and test text to ensure consistent vocabulary
all_text_for_tfidf = pd.concat([train['cleaned_text'], test['cleaned_text']])
tfidf_model.fit(all_text_for_tfidf)

# Transform text into sparse TF-IDF matrices
train_tfidf = tfidf_model.transform(train['cleaned_text'])
test_tfidf = tfidf_model.transform(test['cleaned_text'])

print(f'Train TF-IDF sparse shape: {train_tfidf.shape}')
print(f'Test TF-IDF sparse shape: {test_tfidf.shape}')

# --- Saving Processed Data for Offline Notebook ---
output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)
print(f"\n--- Saving Processed Data to {output_dir} ---")

# Save sparse TF-IDF matrices
sparse.save_npz(os.path.join(output_dir, 'train_tfidf.npz'), train_tfidf)
sparse.save_npz(os.path.join(output_dir, 'test_tfidf.npz'), test_tfidf)

# Save numeric features (as pandas DataFrame then to CSV)
pd.DataFrame(X_numeric_train, columns=numeric_features).to_csv(
    os.path.join(output_dir, 'train_numeric_features.csv'), index=False
)
pd.DataFrame(X_numeric_test, columns=numeric_features).to_csv(
    os.path.join(output_dir, 'test_numeric_features.csv'), index=False
)

# Save target encoded labels
train['target_encoded'].to_csv(os.path.join(output_dir, 'train_target_encoded.csv'), index=False)
train['target_cat'].to_csv(os.path.join(output_dir, 'train_target_cat.csv'), index=False) # For MAP@3 eval

# Save LabelEncoder and mapping dictionaries
with open(os.path.join(output_dir, 'le_target.pkl'), 'wb') as f:
    pickle.dump(le_target, f)
with open(os.path.join(output_dir, 'target_classes.pkl'), 'wb') as f:
    pickle.dump(target_classes, f)
with open(os.path.join(output_dir, 'map_target1.pkl'), 'wb') as f:
    pickle.dump(map_target1, f)
with open(os.path.join(output_dir, 'map_target2.pkl'), 'wb') as f:
    pickle.dump(map_target2, f)

print("All processed data and mappings saved successfully!")
print("Please save this notebook's output as a new Kaggle dataset (e.g., 'map-charting-processed-data').")
print("Then, add this new dataset as an input to your next notebook for model training.")

