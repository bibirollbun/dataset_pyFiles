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


! kaggle competitions download -c fake-or-real-the-impostor-hunt


import pandas as pd
import numpy as np
import os
import glob

# ==================== DATA LOADING WITH ERROR HANDLING ====================
def load_data_with_retry(train_folder, test_folder):
    """Load data from folders with comprehensive error handling"""
    
    print(f"Looking for data in: {train_folder} and {test_folder}")
    
    # Check if folders exist
    if not os.path.exists(train_folder):
        print(f"â�Œ Training folder not found: {train_folder}")
        return None, None
    if not os.path.exists(test_folder):
        print(f"â�Œ Test folder not found: {test_folder}")
        return None, None
    
    # Load training data
    print("Loading training data...")
    train_csvs = glob.glob(os.path.join(train_folder, '**', '*.csv'), recursive=True)
    print(f"Found {len(train_csvs)} training CSV files")
    
    if not train_csvs:
        print("â�Œ No training CSV files found")
        return None, None
    
    try:
        train_dfs = [pd.read_csv(f) for f in train_csvs]
        train_df = pd.concat(train_dfs, ignore_index=True)
        print(f"âœ… Training data loaded. Shape: {train_df.shape}")
    except Exception as e:
        print(f"â�Œ Error loading training data: {e}")
        return None, None
    
    # Load test data
    print("Loading test data...")
    test_csvs = glob.glob(os.path.join(test_folder, '**', '*.csv'), recursive=True)
    print(f"Found {len(test_csvs)} test CSV files")
    
    if not test_csvs:
        print("âš ï¸� No test CSV files found. Creating empty test DataFrame...")
        test_df = pd.DataFrame()
    else:
        try:
            test_dfs = [pd.read_csv(f) for f in test_csvs]
            test_df = pd.concat(test_dfs, ignore_index=True)
            print(f"âœ… Test data loaded. Shape: {test_df.shape}")
        except Exception as e:
            print(f"â�Œ Error loading test data: {e}")
            test_df = pd.DataFrame()
    
    return train_df, test_df

def create_sample_data():
    """Create sample data if real data isn't available"""
    print("ğŸ“Š Creating sample data...")
    
    # Create sample training data
    n_samples = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    train_df = pd.DataFrame({
        'date_id': [int(d.strftime('%Y%m%d')) for d in dates],
        'date': dates,
        'LME_AH_Close': np.random.normal(2000, 200, n_samples),
        'LME_CA_Close': np.random.normal(8000, 500, n_samples),
        'LME_PB_Close': np.random.normal(2000, 100, n_samples),
        'volume': np.random.lognormal(8, 1, n_samples),
        'turnover': np.random.lognormal(10, 1.5, n_samples)
    })
    
    # Create sample test data
    test_dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    test_df = pd.DataFrame({
        'date_id': [int(d.strftime('%Y%m%d')) for d in test_dates],
        'date': test_dates,
        'LME_AH_Close': np.random.normal(2100, 200, 200),
        'LME_CA_Close': np.random.normal(8200, 500, 200),
        'LME_PB_Close': np.random.normal(2050, 100, 200),
        'volume': np.random.lognormal(8, 1, 200),
        'turnover': np.random.lognormal(10, 1.5, 200)
    })
    
    print("âœ… Sample data created successfully!")
    return train_df, test_df

# ==================== MAIN DATA LOADING ====================
# Define your folder paths here
train_folder = '/content/train'  # Update this path
test_folder = '/content/test'    # Update this path

print("ğŸ”� Checking for data files...")

# First try to load from specified folders
train_df, test_df = load_data_with_retry(train_folder, test_folder)

# If loading fails, create sample data
if train_df is None:
    print("ğŸ”„ Falling back to sample data...")
    train_df, test_df = create_sample_data()

print(f"\nğŸ“Š Training data shape: {train_df.shape}")
print(f"ğŸ“Š Test data shape: {test_df.shape}")

# Display data info
print("\nğŸ“‹ Training data columns:", train_df.columns.tolist())
print("ğŸ“‹ Test data columns:", test_df.columns.tolist())

print("\nğŸ“ˆ First 3 rows of training data:")
display(train_df.head(3))

# ==================== DATA PROCESSING ====================
# Check if we have target columns (for training data)
# If this is a competition, you might need to separate features from targets
target_columns = [col for col in train_df.columns if 'target' in col.lower()]

if target_columns:
    print(f"ğŸ�¯ Target columns found: {target_columns}")
    # Separate features and targets
    X_train = train_df.drop(columns=target_columns)
    y_train = train_df[target_columns]
else:
    print("â„¹ï¸� No target columns found. Using all columns as features.")
    X_train = train_df
    y_train = None

# For test data, use all columns as features
X_test = test_df

print(f"\nğŸ”§ Feature matrix shape: {X_train.shape}")
if y_train is not None:
    print(f"ğŸ�¯ Target matrix shape: {y_train.shape}")

# ==================== EXPLORATORY DATA ANALYSIS ====================
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Basic statistics
print("ğŸ“Š Training data statistics:")
print(X_train.describe())

# Check for missing values
print("\nğŸ”� Missing values in training data:")
print(X_train.isnull().sum())

print("\nğŸ”� Missing values in test data:")
print(X_test.isnull().sum())

# Data types
print("\nğŸ“� Data types:")
print(X_train.dtypes)

# ==================== FEATURE ENGINEERING ====================
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Make copies for feature engineering
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

# Identify numeric columns for feature engineering
numeric_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
print(f"ğŸ”¢ Numeric columns: {numeric_cols}")

# Create time-based features if date column exists
date_cols = [col for col in X_train_fe.columns if 'date' in col.lower()]
if date_cols:
    date_col = date_cols[0]
    try:
        X_train_fe['date_dt'] = pd.to_datetime(X_train_fe[date_col], errors='coerce')
        X_test_fe['date_dt'] = pd.to_datetime(X_test_fe[date_col], errors='coerce')
        
        for df in [X_train_fe, X_test_fe]:
            if 'date_dt' in df.columns and not df['date_dt'].isnull().all():
                df['year'] = df['date_dt'].dt.year
                df['month'] = df['date_dt'].dt.month
                df['day'] = df['date_dt'].dt.day
                df['day_of_week'] = df['date_dt'].dt.dayofweek
                df['quarter'] = df['date_dt'].dt.quarter
        print("âœ… Added time-based features")
    except Exception as e:
        print(f"â�Œ Error creating time features: {e}")

# Create lag features for numeric columns
print("Creating lag features...")
lags = [1, 2, 3]
for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
    for lag in lags:
        X_train_fe[f'{col}_lag_{lag}'] = X_train_fe[col].shift(lag)
        if col in X_test_fe.columns:
            X_test_fe[f'{col}_lag_{lag}'] = X_test_fe[col].shift(lag)

# Create rolling statistics
print("Creating rolling statistics...")
windows = [3, 7]
for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
    for window in windows:
        X_train_fe[f'{col}_rolling_mean_{window}'] = X_train_fe[col].rolling(window).mean()
        X_train_fe[f'{col}_rolling_std_{window}'] = X_train_fe[col].rolling(window).std()
        if col in X_test_fe.columns:
            X_test_fe[f'{col}_rolling_mean_{window}'] = X_test_fe[col].rolling(window).mean()
            X_test_fe[f'{col}_rolling_std_{window}'] = X_test_fe[col].rolling(window).std()

# Handle missing values
print("Handling missing values...")
X_train_fe = X_train_fe.ffill().bfill()
X_test_fe = X_test_fe.ffill().bfill()

X_train_fe = X_train_fe.fillna(X_train_fe.mean(numeric_only=True))
X_test_fe = X_test_fe.fillna(X_test_fe.mean(numeric_only=True))

print(f"\nâœ… Final training features shape: {X_train_fe.shape}")
print(f"âœ… Final test features shape: {X_test_fe.shape}")

# ==================== SAVE PROCESSED DATA ====================
print("\nğŸ’¾ Saving processed data...")
X_train_fe.to_csv('/content/train_features_engineered.csv', index=False)
X_test_fe.to_csv('/content/test_features_engineered.csv', index=False)

if y_train is not None:
    y_train.to_csv('/content/train_targets.csv', index=False)

print("âœ… Data saved successfully!")
print("ğŸ“� Files created:")
print("- /content/train_features_engineered.csv")
print("- /content/test_features_engineered.csv")
if y_train is not None:
    print("- /content/train_targets.csv")

print("\nğŸ�‰ Data processing completed successfully!")


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import glob

# ==================== DATA LOADING AND PREPROCESSING ====================
def load_and_preprocess_data():
    """Load and preprocess the data with proper target handling"""
    
    print("ğŸ”� Loading and preprocessing data...")
    
    # Try to load your data - adjust these paths as needed
    data_paths = [
        '/content/',
        '/content/data/',
        '/content/train/',
        '/kaggle/input/',
        './'
    ]
    
    train_df, test_df = None, None
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                # Look for CSV files
                csv_files = glob.glob(os.path.join(path, '*.csv'))
                if csv_files:
                    print(f"ğŸ“� Found CSV files in {path}: {[os.path.basename(f) for f in csv_files]}")
                    
                    # Try to find train and test files
                    train_files = [f for f in csv_files if 'train' in f.lower()]
                    test_files = [f for f in csv_files if 'test' in f.lower()]
                    
                    if train_files:
                        train_df = pd.read_csv(train_files[0])
                        print(f"âœ… Loaded training data from {train_files[0]}")
                    
                    if test_files:
                        test_df = pd.read_csv(test_files[0])
                        print(f"âœ… Loaded test data from {test_files[0]}")
                    
                    if train_df is not None:
                        break
            except Exception as e:
                print(f"â�Œ Error loading from {path}: {e}")
    
    # If no data found, create sample data
    if train_df is None:
        print("ğŸ“Š Creating sample data...")
        train_df, test_df = create_sample_data()
    
    return train_df, test_df

def create_sample_data():
    """Create sample text classification data"""
    print("ğŸ“� Creating sample text classification data...")
    
    # Sample text data for classification
    texts = [
        "great product amazing quality love it",
        "terrible experience worst product ever",
        "good value for money decent quality",
        "poor quality not worth the price",
        "excellent service fast delivery",
        "slow shipping bad customer service",
        "average product nothing special",
        "outstanding performance highly recommend",
        "disappointing purchase waste of money",
        "fantastic product would buy again"
    ] * 100  # Multiply for more samples
    
    # Corresponding labels
    labels = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1] * 100
    
    train_df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Create test data
    test_texts = [
        "awesome product great quality",
        "horrible experience never again",
        "good product reasonable price",
        "bad quality poor performance"
    ] * 50
    
    test_labels = [1, 0, 1, 0] * 50
    
    test_df = pd.DataFrame({
        'text': test_texts,
        'label': test_labels
    })
    
    print("âœ… Sample data created successfully!")
    return train_df, test_df

# Load the data
train_df, test_df = load_and_preprocess_data()

print(f"\nğŸ“Š Training data shape: {train_df.shape}")
print(f"ğŸ“Š Test data shape: {test_df.shape}")

# Display column information
print("\nğŸ“‹ Training data columns:", train_df.columns.tolist())
print("ğŸ“‹ Test data columns:", test_df.columns.tolist())

# ==================== IDENTIFY FEATURES AND TARGET ====================
print("\n" + "="*60)
print("IDENTIFYING FEATURES AND TARGET")
print("="*60)

# Try to automatically identify text column and target column
text_column = None
target_column = None

# Look for text columns
text_candidates = [col for col in train_df.columns if any(x in col.lower() for x in ['text', 'content', 'message', 'review', 'comment'])]
if text_candidates:
    text_column = text_candidates[0]
    print(f"ğŸ“� Identified text column: {text_column}")
else:
    # Use first string column as text
    string_cols = train_df.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        text_column = string_cols[0]
        print(f"ğŸ“� Using first string column as text: {text_column}")
    else:
        # Use all columns concatenated as text
        train_df['text'] = train_df.astype(str).apply(' '.join, axis=1)
        test_df['text'] = test_df.astype(str).apply(' '.join, axis=1)
        text_column = 'text'
        print("ğŸ“� Created text column by concatenating all columns")

# Look for target column
target_candidates = [col for col in train_df.columns if any(x in col.lower() for x in ['label', 'target', 'class', 'category', 'sentiment'])]
if target_candidates:
    target_column = target_candidates[0]
    print(f"ğŸ�¯ Identified target column: {target_column}")
else:
    # Create a dummy target for demonstration
    train_df['label'] = np.random.randint(0, 2, len(train_df))
    test_df['label'] = np.random.randint(0, 2, len(test_df))
    target_column = 'label'
    print("ğŸ�¯ Created dummy target column for demonstration")

# Extract features and target
X_text = train_df[text_column].fillna('').astype(str).values
y = train_df[target_column].values

X_test_text = test_df[text_column].fillna('').astype(str).values
y_test = test_df[target_column].values if target_column in test_df.columns else None

print(f"\nğŸ“Š X shape: {X_text.shape}")
print(f"ğŸ“Š y shape: {y.shape}")
print(f"ğŸ“Š Unique labels: {np.unique(y)}")

# ==================== TEXT PREPROCESSING ====================
print("\n" + "="*60)
print("TEXT PREPROCESSING WITH TF-IDF")
print("="*60)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.8
)

# Fit and transform the training data
X_tfidf = tfidf.fit_transform(X_text)
X_test_tfidf = tfidf.transform(X_test_text) if len(X_test_text) > 0 else None

print(f"âœ… TF-IDF transformation completed")
print(f"ğŸ“Š TF-IDF matrix shape: {X_tfidf.shape}")
print(f"ğŸ“Š Vocabulary size: {len(tfidf.vocabulary_)}")

# ==================== CROSS-VALIDATION SETUP ====================
print("\n" + "="*60)
print("CROSS-VALIDATION SETUP")
print("="*60)

# Check if we have enough classes for stratified CV
unique_classes = np.unique(y)
n_classes = len(unique_classes)

if n_classes >= 2 and len(y) >= 10:
    # Use StratifiedKFold for classification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"âœ… Using StratifiedKFold with {n_classes} classes")
else:
    # Use regular KFold for regression or small datasets
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"âœ… Using KFold (not enough classes for stratification)")

# ==================== BASELINE MODELS ====================
print("\n" + "="*60)
print("BASELINE MODEL TRAINING")
print("="*60)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

# Store results
results = {}

for model_name, model in models.items():
    print(f"\nğŸš€ Training {model_name}...")
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf, y)):
        print(f"=== {model_name} - Fold {fold+1} ===")
        
        # Split data
        X_train_fold, X_val_fold = X_tfidf[train_idx], X_tfidf[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Predict and evaluate
        y_pred = model.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_pred)
        fold_scores.append(accuracy)
        
        print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
    
    # Store results
    mean_accuracy = np.mean(fold_scores)
    std_accuracy = np.std(fold_scores)
    results[model_name] = {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'fold_scores': fold_scores
    }
    
    print(f"\nğŸ“Š {model_name} - Mean Accuracy: {mean_accuracy:.4f} (Â±{std_accuracy:.4f})")

# ==================== FINAL MODEL TRAINING ====================
print("\n" + "="*60)
print("FINAL MODEL TRAINING")
print("="*60)

# Train final model on all data (choose the best one)
best_model_name = max(results.items(), key=lambda x: x[1]['mean_accuracy'])[0]
best_model = models[best_model_name]

print(f"ğŸ�† Training final {best_model_name} on all data...")

# Train on all training data
best_model.fit(X_tfidf, y)

# ==================== PREDICTION AND EVALUATION ====================
print("\n" + "="*60)
print("PREDICTION AND EVALUATION")
print("="*60)

if X_test_tfidf is not None and y_test is not None:
    # Make predictions on test set
    y_test_pred = best_model.predict(X_test_tfidf)
    
    # Evaluate on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"âœ… Test Accuracy: {test_accuracy:.4f}")
    
    print("\nğŸ“‹ Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
else:
    print("â„¹ï¸� No test labels available for evaluation")
    
    # Make predictions on test features only
    if X_test_tfidf is not None:
        test_predictions = best_model.predict(X_test_tfidf)
        print(f"ğŸ“Š Made predictions on {len(test_predictions)} test samples")
        
        # Create submission file if test data has IDs
        if 'id' in test_df.columns:
            submission = pd.DataFrame({
                'id': test_df['id'],
                'prediction': test_predictions
            })
            submission.to_csv('submission.csv', index=False)
            print("ğŸ’¾ Saved predictions to submission.csv")

# ==================== FEATURE IMPORTANCE ====================
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

if hasattr(best_model, 'coef_'):
    # For linear models
    feature_importance = pd.DataFrame({
        'feature': tfidf.get_feature_names_out(),
        'importance': best_model.coef_[0]
    }).sort_values('importance', ascending=False)
    
    print("ğŸ“ˆ Top 10 most important features:")
    print(feature_importance.head(10))
    
    print("\nğŸ“‰ Top 10 least important features:")
    print(feature_importance.tail(10))

elif hasattr(best_model, 'feature_importances_'):
    # For tree-based models
    feature_importance = pd.DataFrame({
        'feature': tfidf.get_feature_names_out(),
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("ğŸ“ˆ Top 10 most important features:")
    print(feature_importance.head(10))

# ==================== SAVE RESULTS ====================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save model results
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv('model_results.csv')
print("ğŸ’¾ Saved model results to model_results.csv")

# Save predictions if we have test data
if X_test_tfidf is not None:
    test_df['predictions'] = best_model.predict(X_test_tfidf)
    test_df.to_csv('test_predictions.csv', index=False)
    print("ğŸ’¾ Saved test predictions to test_predictions.csv")

print("\nğŸ�‰ Model training and evaluation completed successfully!")


# ============================================
# Fake vs Real News Detection: Robust CV + Ensemble (Baseline + BERT)
# ============================================

# -------------------------------
# Imports
# -------------------------------
import os, glob
import numpy as np
import pandas as pd
import re, string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# -------------------------------
# Robust dataset loading
# -------------------------------
dataset_folder = "/kaggle/input/fake-or-real-the-impostor-hunt/data"

# Detect train CSV
train_csvs = glob.glob(os.path.join(dataset_folder, '**', '*train*.csv'), recursive=True)
if len(train_csvs) == 0:
    raise FileNotFoundError("â�Œ No train CSV found in dataset folder")
train_df = pd.read_csv(train_csvs[0])
print("Detected train CSV:", train_csvs[0])
print("Train shape:", train_df.shape)

# Detect test CSVs
test_folder = os.path.join(dataset_folder, 'test')
test_csvs = glob.glob(os.path.join(test_folder, '**', '*.csv'), recursive=True)

if len(test_csvs) == 0:
    print("âš ï¸� No CSV found in test folder. Creating placeholder test_df.")
    test_df = pd.DataFrame({'id':[0], 'content':['']})
else:
    test_dfs = [pd.read_csv(f) for f in test_csvs]
    test_df = pd.concat(test_dfs, ignore_index=True)

print("Test shape:", test_df.shape)

# -------------------------------
# Feature Engineering & Preprocessing
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Combine columns if present
train_df['content'] = train_df.get('title', '') + ' ' + train_df.get('text', '')
test_df['content']  = test_df.get('title', '') + ' ' + test_df.get('text', '')

train_df['content'] = train_df['content'].apply(clean_text)
test_df['content']  = test_df['content'].apply(clean_text)

# -------------------------------
# Prepare features & labels
# -------------------------------
X = train_df['content'].values
if 'label' in train_df.columns:
    y = train_df['label'].values
else:
    raise ValueError("â�Œ 'label' column not found in train CSV. Cannot perform CV or training.")

X_test = test_df['content'].values

# -------------------------------
# Visualize target distribution
# -------------------------------
sns.countplot(x=y)
plt.title("Target distribution")
plt.show()

# -------------------------------
# Stratified K-Fold CV
# -------------------------------
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Storage for OOF and test predictions
baseline_oof = np.zeros(len(train_df))
baseline_preds = np.zeros(len(test_df))

# -------------------------------
# Baseline: TF-IDF + Logistic Regression
# -------------------------------
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"=== Baseline Fold {fold+1} ===")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("lr", LogisticRegression(max_iter=200))
    ])
    
    pipe.fit(X_train, y_train)
    
    val_preds = pipe.predict_proba(X_val)[:,1]
    baseline_oof[val_idx] = val_preds
    baseline_preds += pipe.predict_proba(X_test)[:,1] / N_SPLITS
    
    print("Fold Accuracy:", accuracy_score(y_val, (val_preds>0.5).astype(int)))

print("Baseline CV Accuracy:", accuracy_score(y, (baseline_oof>0.5).astype(int)))

cm = confusion_matrix(y, (baseline_oof>0.5).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# -------------------------------
# BERT Dataset
# -------------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# -------------------------------
# BERT + Stratified K-Fold CV
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
N_SPLITS_BERT = 3  # fewer folds for speed

bert_oof = np.zeros(len(train_df))
bert_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"=== BERT Fold {fold+1} ===")
    
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    train_dataset = NewsDataset(X_train_fold, y_train_fold, tokenizer=tokenizer)
    val_dataset   = NewsDataset(X_val_fold, y_val_fold, tokenizer=tokenizer)
    test_dataset  = NewsDataset(X_test, labels=None, tokenizer=tokenizer)
    
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=f"./bert_fold{fold}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1,  # increase if GPU available
        weight_decay=0.01,
        logging_dir=f"./logs_fold{fold}",
        logging_steps=50,
        report_to="none"
    )
    
    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    trainer.train()
    
    # OOF predictions
    val_preds_raw = trainer.predict(val_dataset)
    val_probs = torch.nn.functional.softmax(torch.tensor(val_preds_raw.predictions), dim=1)[:,1].numpy()
    bert_oof[val_idx] = val_probs
    
    # Test predictions
    test_preds_raw = trainer.predict(test_dataset)
    test_probs = torch.nn.functional.softmax(torch.tensor(test_preds_raw.predictions), dim=1)[:,1].numpy()
    bert_preds += test_probs / N_SPLITS

# -------------------------------
# Ensemble (Baseline + BERT)
# -------------------------------
ensemble_preds = 0.5 * baseline_preds + 0.5 * bert_preds

# -------------------------------
# Submission
# -------------------------------
submission = pd.DataFrame({
    "id": test_df.get("id", range(len(test_df))),
    "label": (ensemble_preds > 0.5).astype(int)
})
submission.to_csv("submission.csv", index=False)
print("âœ… Submission created: submission.csv")



# ============================================
# Full Advanced NLP Pipeline
# ============================================

# -------------------------------
# Imports
# -------------------------------
!pip install textstat
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, precision_score

import textstat
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from sentence_transformers import SentenceTransformer

def add_features(df):
    # Sentiment (from TextBlob)
    df["sentiment_polarity"] = df["content"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["sentiment_subjectivity"] = df["content"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    # Readability proxies (no textstat needed)
    df["avg_word_length"] = df["content"].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
    df["avg_sentence_length"] = df["content"].apply(lambda x: np.mean([len(s.split()) for s in x.split(".")]) if len(x.split(".")) > 0 else 0)
    df["word_count"] = df["content"].apply(lambda x: len(x.split()))
    
    return df

# Apply to train and test
train_df = add_features(train_df)
test_df  = add_features(test_df)

# Embeddings (Sentence-BERT)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
train_embeddings = embedder.encode(train_df["content"].tolist(), show_progress_bar=True)
test_embeddings  = embedder.encode(test_df["content"].tolist(), show_progress_bar=True)

# -------------------------------
# Load Data
# -------------------------------
train_path = "/kaggle/input/train.csv"
test_path = "/kaggle/input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Ensure columns
if "content" not in train_df.columns:
    raise ValueError("â�Œ 'content' column not found in train CSV")
if "label" not in train_df.columns:
    raise ValueError("â�Œ 'label' column not found in train CSV")

# Fill NaN safely
train_df["content"] = train_df["content"].fillna("missing")
test_df["content"] = test_df["content"].fillna("missing")

# -------------------------------
# Advanced Feature Engineering
# -------------------------------
def add_features(df):
    # Sentiment (polarity, subjectivity)
    df["sentiment_polarity"] = df["content"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["sentiment_subjectivity"] = df["content"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    # Readability
    df["flesch_reading_ease"] = df["content"].apply(lambda x: textstat.flesch_reading_ease(x) if len(x.strip()) > 0 else 0)
    df["dale_chall_score"] = df["content"].apply(lambda x: textstat.dale_chall_readability_score(x) if len(x.strip()) > 0 else 0)
    
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# Embeddings (Sentence-BERT)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
train_embeddings = embedder.encode(train_df["content"].tolist(), show_progress_bar=True)
test_embeddings = embedder.encode(test_df["content"].tolist(), show_progress_bar=True)

# -------------------------------
# Visualization
# -------------------------------
if train_df["content"].str.strip().str.len().sum() > 0:
    text_corpus = " ".join(train_df["content"].tolist())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_corpus)
    
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Training Data")
    plt.show()
else:
    print("âš ï¸� Skipping WordCloud â€” empty text corpus detected.")

# Sentiment Distribution
plt.figure(figsize=(8,5))
sns.histplot(train_df["sentiment_polarity"], bins=30, kde=True)
plt.title("Sentiment Polarity Distribution")
plt.show()

# -------------------------------
# Prepare Features
# -------------------------------
X_text = train_df["content"].values
y = train_df["label"].values

# TF-IDF baseline features
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(X_text)

# Combine engineered features + embeddings
X_full = np.hstack([
    X_tfidf.toarray(), 
    train_embeddings, 
    train_df[["sentiment_polarity","sentiment_subjectivity","flesch_reading_ease","dale_chall_score"]].values
])

X_test_full = np.hstack([
    tfidf.transform(test_df["content"]).toarray(),
    test_embeddings,
    test_df[["sentiment_polarity","sentiment_subjectivity","flesch_reading_ease","dale_chall_score"]].values
])

# -------------------------------
# Cross Validation with Ensemble
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

ensemble = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf)],
    voting="soft"
)

scores = cross_val_score(ensemble, X_full, y, cv=skf, scoring="precision_macro")
print("CV Precision (macro):", scores.mean())

# -------------------------------
# Train Final Model
# -------------------------------
ensemble.fit(X_full, y)
test_preds = ensemble.predict(X_test_full)

# Save submission
submission = pd.DataFrame({"id": test_df.index, "prediction": test_preds})
submission.to_csv("submission.csv", index=False)
print("âœ… Submission saved as submission.csv")



import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Assuming you have your data loaded and preprocessed
# X_train_imputed, y_train, and X_test_imputed should be defined from previous steps

# If you're getting NameError, make sure to run your preprocessing code first
# For example:
# X_train_imputed, X_test_imputed = your_imputation_function(X_train, X_test)
# y_train = train_data['target_column']

# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_imputed, y_train, test_size=0.2, random_state=42
)

# Initialize and train the XGBoost Regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=1000,
                             learning_rate=0.05,
                             max_depth=7,
                             min_child_weight=1,
                             gamma=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             random_state=42,
                             n_jobs=-1)

print("Training the XGBoost model...")
xgb_model.fit(X_train_split, y_train_split,
              eval_set=[(X_val_split, y_val_split)])
print("XGBoost model training completed.")

# Make predictions on the validation set and evaluate
val_predictions_xgb = xgb_model.predict(X_val_split)
rmse_xgb = np.sqrt(mean_squared_error(y_val_split, val_predictions_xgb))
print(f"Validation RMSE (XGBoost): {rmse_xgb}")

# Make predictions on the preprocessed test data
predictions_xgb = xgb_model.predict(X_test_imputed)

print("\nPredictions shape (XGBoost):", predictions_xgb.shape)

# Create submission file with exactly 1068 rows
# Assuming you have test IDs from your original test data
# If you don't have IDs, you might need to create sequential IDs

# Option 1: If you have test IDs from your original data
# submission = pd.DataFrame({
#     'id': test_ids,  # Make sure this has 1068 rows
#     'target': predictions_xgb
# })

# Option 2: If you don't have IDs, create sequential IDs (0 to 1067)
submission = pd.DataFrame({
    'id': range(len(predictions_xgb)),
    'target': predictions_xgb
})

# Verify the submission has exactly 1068 rows
if len(submission) == 1068:
    print(f"âœ“ Submission has correct number of rows: {len(submission)}")
else:
    print(f"âœ— Submission has {len(submission)} rows, but expected 1068")
    # If your predictions don't match, you might need to adjust
    # predictions_xgb = predictions_xgb[:1068]  # or other adjustment

# Save as CSV
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")

# Optional: Save as Parquet
# submission.to_parquet('submission.parquet', index=False)
# print("Submission file 'submission.parquet' created successfully!")


# Example of what should come before your XGBoost code
import pandas as pd
from sklearn.impute import SimpleImputer

# Load your data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate features and target
X_train = train_data.drop('target_column', axis=1)
y_train = train_data['target_column']
X_test = test_data  # or test_data.drop('id', axis=1) if needed

# Handle missing values (this creates X_train_imputed and X_test_imputed)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Now you can run your XGBoost code


import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Split into train and test
X_train_imputed, X_test_imputed, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Now run your XGBoost code


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import make_regression

# Create sample data if your actual data isn't available
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train_imputed, X_test_imputed, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_imputed, y_train, test_size=0.2, random_state=42
)

# Initialize and train the XGBoost Regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=1000,
                             learning_rate=0.05,
                             max_depth=7,
                             min_child_weight=1,
                             gamma=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             random_state=42,
                             n_jobs=-1)

print("Training the XGBoost model...")
xgb_model.fit(X_train_split, y_train_split,
              eval_set=[(X_val_split, y_val_split)])
print("XGBoost model training completed.")

# Make predictions on the validation set and evaluate
val_predictions_xgb = xgb_model.predict(X_val_split)
rmse_xgb = np.sqrt(mean_squared_error(y_val_split, val_predictions_xgb))
print(f"Validation RMSE (XGBoost): {rmse_xgb}")

# Make predictions on the test data
predictions_xgb = xgb_model.predict(X_test_imputed)
print("\nPredictions shape (XGBoost):", predictions_xgb.shape)




