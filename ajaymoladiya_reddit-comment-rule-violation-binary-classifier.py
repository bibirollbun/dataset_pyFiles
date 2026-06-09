# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)



# Load the datasets
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

print("=== DATASET OVERVIEW ===")
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")



print(f"\n=== TRAINING DATA COLUMNS ===")
print(train_df.columns.tolist())


print(f"\n=== TARGET DISTRIBUTION ===")
print(train_df['rule_violation'].value_counts())
print(f"Positive class ratio: {train_df['rule_violation'].mean():.3f}")


print(f"\n=== UNIQUE RULES IN TRAINING ===")
for i, rule in enumerate(train_df['rule'].unique(), 1):
    print(f"{i}. {rule}")


print(f"\n=== UNIQUE RULES IN TEST ===")
for i, rule in enumerate(test_df['rule'].unique(), 1):
    print(f"{i}. {rule}")


# Examine sample data
print("=== SAMPLE TRAINING DATA ===")
print(train_df[['body', 'rule', 'rule_violation']].head(3))

print("\n=== SAMPLE TEST DATA ===")
print(test_df[['body', 'rule']].head(3))



# Check for missing values
print(f"\n=== MISSING VALUES ===")
print("Training data:")
print(train_df.isnull().sum())
print("\nTest data:")
print(test_df.isnull().sum())


# Analyze text length distribution
train_df['body_length'] = train_df['body'].str.len()
test_df['body_length'] = test_df['body'].str.len()

print(f"\n=== TEXT LENGTH STATISTICS ===")
print("Training data body length:")
print(train_df['body_length'].describe())
print("\nTest data body length:")
print(test_df['body_length'].describe())


# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

# Apply preprocessing
train_df['body_clean'] = train_df['body'].apply(preprocess_text)
test_df['body_clean'] = test_df['body'].apply(preprocess_text)

print(f"Sample cleaned text: {train_df['body_clean'].iloc[0][:200]}...")



# Create additional features
def create_features(df):
    """Create additional features from text and rule information"""
    
    # Text-based features
    df['word_count'] = df['body_clean'].str.split().str.len()
    df['char_count'] = df['body_clean'].str.len()
    df['sentence_count'] = df['body_clean'].str.count(r'[.!?]+')
    df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)
    
    # Punctuation and special characters
    df['exclamation_count'] = df['body_clean'].str.count('!')
    df['question_count'] = df['body_clean'].str.count('\?')
    df['caps_ratio'] = df['body_clean'].str.count(r'[A-Z]') / (df['char_count'] + 1)
    
    # URL and link indicators
    df['has_url'] = df['body'].str.contains(r'http|www', case=False, na=False).astype(int)
    df['has_email'] = df['body'].str.contains(r'@', case=False, na=False).astype(int)
    
    # Rule-based features
    df['rule_length'] = df['rule'].str.len()
    df['rule_word_count'] = df['rule'].str.split().str.len()
    
    # Subreddit encoding (simple label encoding)
    df['subreddit_encoded'] = pd.Categorical(df['subreddit']).codes
    
    return df

# Apply feature engineering
train_df = create_features(train_df)
test_df = create_features(test_df)

print(f"New features created: {[col for col in train_df.columns if col not in ['row_id', 'body', 'rule', 'subreddit', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2', 'rule_violation', 'body_clean']]}")



# Prepare text data for vectorization
# Combine comment text with rule description for better context
train_df['combined_text'] = train_df['body_clean'] + ' ' + train_df['rule']
test_df['combined_text'] = test_df['body_clean'] + ' ' + test_df['rule']

# Initialize vectorizers
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.95
)



# Fit and transform training data
X_text_train = tfidf_vectorizer.fit_transform(train_df['combined_text'])
X_text_test = tfidf_vectorizer.transform(test_df['combined_text'])

print(f"Text vectorization completed!")
print(f"TF-IDF matrix shape (train): {X_text_train.shape}")
print(f"TF-IDF matrix shape (test): {X_text_test.shape}")


# Prepare numerical features
numerical_features = [
    'word_count', 'char_count', 'sentence_count', 'avg_word_length',
    'exclamation_count', 'question_count', 'caps_ratio', 'has_url', 'has_email',
    'rule_length', 'rule_word_count', 'subreddit_encoded'
]

X_num_train = train_df[numerical_features].values
X_num_test = test_df[numerical_features].values

print(f"Numerical features shape (train): {X_num_train.shape}")
print(f"Numerical features shape (test): {X_num_test.shape}")


# Combine text and numerical features
from scipy.sparse import hstack

X_train_combined = hstack([X_text_train, X_num_train])
X_test_combined = hstack([X_text_test, X_num_test])

print(f"Combined features shape (train): {X_train_combined.shape}")
print(f"Combined features shape (test): {X_test_combined.shape}")

# Prepare target variable
y_train = train_df['rule_violation'].values

print(f"Target variable shape: {y_train.shape}")
print(f"Target distribution: {np.bincount(y_train)}")


# Split training data for validation
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_combined, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Training target distribution: {np.bincount(y_train_split)}")
print(f"Validation target distribution: {np.bincount(y_val)}")

# Initialize multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True, kernel='linear')
}

# Train and evaluate models
model_scores = {}
trained_models = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    
    # Train model
    model.fit(X_train, y_train_split)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    # Calculate AUC score
    auc_score = roc_auc_score(y_val, y_pred_proba)
    model_scores[name] = auc_score
    
    # Store trained model
    trained_models[name] = model
    
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_val, y_pred))

# Display model comparison
print(f"\n=== MODEL COMPARISON ===")
for name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.4f}")

# Select best model
best_model_name = max(model_scores, key=model_scores.get)
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name} with AUC: {model_scores[best_model_name]:.4f}")



# Cross-validation with the best model
print(f"\n=== CROSS-VALIDATION WITH {best_model_name.upper()} ===")

# Retrain best model on full training data
best_model.fit(X_train_combined, y_train)

# Perform cross-validation
cv_scores = cross_val_score(best_model, X_train_combined, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n=== FEATURE IMPORTANCE ===")
    feature_names = list(tfidf_vectorizer.get_feature_names_out()) + numerical_features
    
    # Get top 20 most important features
    importance_scores = best_model.feature_importances_
    top_indices = np.argsort(importance_scores)[-20:][::-1]
    
    print("Top 20 most important features:")
    for i, idx in enumerate(top_indices):
        if idx < len(feature_names):
            print(f"{i+1:2d}. {feature_names[idx]:30s} {importance_scores[idx]:.4f}")
        else:
            print(f"{i+1:2d}. Numerical feature {idx-len(feature_names):2d} {importance_scores[idx]:.4f}")



# Generate predictions for test set
print("=== GENERATING TEST PREDICTIONS ===")

# Make predictions on test set
test_predictions = best_model.predict_proba(X_test_combined)[:, 1]

print(f"Test predictions shape: {test_predictions.shape}")
print(f"Prediction statistics:")
print(f"  Min: {test_predictions.min():.4f}")
print(f"  Max: {test_predictions.max():.4f}")
print(f"  Mean: {test_predictions.mean():.4f}")
print(f"  Std: {test_predictions.std():.4f}")

# Create submission file
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': test_predictions
})

print(f"\nSubmission file shape: {submission.shape}")
print(f"Sample predictions:")
print(submission.head(10))

# Save submission file
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission file saved as 'submission.csv'")

# Display prediction distribution
plt.figure(figsize=(10, 6))
plt.hist(test_predictions, bins=50, alpha=0.7, edgecolor='black')
plt.title('Distribution of Test Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()



# Analyze predictions by rule type
print("=== ANALYSIS BY RULE TYPE ===")

# Get unique rules in test set
test_rules = test_df['rule'].unique()
print(f"Number of unique rules in test set: {len(test_rules)}")

# Analyze predictions for each rule
rule_analysis = []
for rule in test_rules:
    rule_mask = test_df['rule'] == rule
    rule_predictions = test_predictions[rule_mask]
    
    rule_analysis.append({
        'rule': rule,
        'count': rule_mask.sum(),
        'mean_prediction': rule_predictions.mean(),
        'std_prediction': rule_predictions.std(),
        'min_prediction': rule_predictions.min(),
        'max_prediction': rule_predictions.max()
    })

rule_df = pd.DataFrame(rule_analysis)
rule_df = rule_df.sort_values('mean_prediction', ascending=False)

print("\nRule-wise prediction statistics:")
print(rule_df.to_string(index=False))

# Visualize rule-wise predictions
plt.figure(figsize=(12, 8))
plt.barh(range(len(rule_df)), rule_df['mean_prediction'])
plt.yticks(range(len(rule_df)), [rule[:50] + '...' if len(rule) > 50 else rule for rule in rule_df['rule']])
plt.xlabel('Mean Predicted Probability')
plt.title('Mean Prediction Probability by Rule')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# Analyze predictions by subreddit
print("=== ANALYSIS BY SUBREDDIT ===")

subreddit_analysis = []
for subreddit in test_df['subreddit'].unique():
    subreddit_mask = test_df['subreddit'] == subreddit
    subreddit_predictions = test_predictions[subreddit_mask]
    
    subreddit_analysis.append({
        'subreddit': subreddit,
        'count': subreddit_mask.sum(),
        'mean_prediction': subreddit_predictions.mean(),
        'std_prediction': subreddit_predictions.std()
    })

subreddit_df = pd.DataFrame(subreddit_analysis)
subreddit_df = subreddit_df.sort_values('mean_prediction', ascending=False)

print("\nSubreddit-wise prediction statistics:")
print(subreddit_df.to_string(index=False))

# Visualize subreddit-wise predictions
plt.figure(figsize=(10, 6))
plt.bar(range(len(subreddit_df)), subreddit_df['mean_prediction'])
plt.xticks(range(len(subreddit_df)), subreddit_df['subreddit'], rotation=45)
plt.ylabel('Mean Predicted Probability')
plt.title('Mean Prediction Probability by Subreddit')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


