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


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# 1. DATA LOADING
print("=== LOADING DATA ===")

# Define file paths
train_path = '/kaggle/input/task-1-clickbait-detection-msci-641-s-25/train.jsonl'
val_path = '/kaggle/input/task-1-clickbait-detection-msci-641-s-25/val.jsonl'
test_path = '/kaggle/input/task-1-clickbait-detection-msci-641-s-25/test.jsonl'

# Load JSONL files
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

train_df = load_jsonl(train_path)
val_df = load_jsonl(val_path)
test_df = load_jsonl(test_path)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# 2. FIX TAGS COLUMN
print("\n=== FIXING TAGS COLUMN ===")

# Extract string from list - tags are stored as ['phrase'] instead of 'phrase'
train_df['tags'] = train_df['tags'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
val_df['tags'] = val_df['tags'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

print("Tags fixed. Unique tags:", train_df['tags'].unique())

# 3. EXPLORATORY DATA ANALYSIS
print("\n=== EXPLORATORY DATA ANALYSIS ===")

# Check columns
print("\nTraining data columns:")
print(train_df.columns.tolist())

# Check for missing values in important columns
print("\nMissing values in key columns:")
key_columns = ['postText', 'targetTitle', 'targetParagraphs', 'tags']
for col in key_columns:
    missing = train_df[col].isnull().sum()
    print(f"{col}: {missing}")

# Distribution of spoiler types
print("\nSpoiler type distribution in training data:")
print(train_df['tags'].value_counts())
print("\nPercentage distribution:")
print(train_df['tags'].value_counts(normalize=True).round(4) * 100)

# Visualize spoiler type distribution
plt.figure(figsize=(10, 6))
train_df['tags'].value_counts().plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Distribution of Spoiler Types in Training Data', fontsize=14)
plt.xlabel('Spoiler Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
for i, v in enumerate(train_df['tags'].value_counts()):
    plt.text(i, v + 10, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# Analyze text lengths
print("\n=== ANALYZING TEXT LENGTHS ===")

# Calculate lengths safely
train_df['postText_length'] = train_df['postText'].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)
train_df['targetTitle_length'] = train_df['targetTitle'].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)
train_df['targetParagraphs_length'] = train_df['targetParagraphs'].apply(
    lambda x: sum(len(str(p).split()) for p in x) if isinstance(x, list) else 0
)

# Plot text length distributions by spoiler type
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

columns = ['postText_length', 'targetTitle_length', 'targetParagraphs_length']
colors = {'phrase': '#1f77b4', 'passage': '#ff7f0e', 'multi': '#2ca02c'}

for idx, col in enumerate(columns):
    for tag in sorted(train_df['tags'].unique()):
        data = train_df[train_df['tags'] == tag][col]
        axes[idx].hist(data, alpha=0.6, label=tag, bins=30, color=colors[tag])
    axes[idx].set_xlabel(col.replace('_', ' ').title())
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].set_title(f'Distribution of {col.replace("_", " ").title()}')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical summary by spoiler type
print("\nMean text lengths by spoiler type:")
mean_summary = train_df.groupby('tags')[columns].mean().round(1)
print(mean_summary)

# 4. FEATURE ENGINEERING
print("\n=== FEATURE ENGINEERING ===")

def create_text_features(df):
    """Create combined text features from the dataframe"""
    features = []
    
    for idx, row in df.iterrows():
        # Get post text
        post_text = str(row['postText']) if pd.notna(row['postText']) else ''
        
        # Get target title
        target_title = str(row['targetTitle']) if pd.notna(row['targetTitle']) else ''
        
        # Get target paragraphs
        if isinstance(row['targetParagraphs'], list):
            target_paragraphs = ' '.join([str(p) for p in row['targetParagraphs']])
        else:
            target_paragraphs = str(row['targetParagraphs']) if pd.notna(row['targetParagraphs']) else ''
        
        # Combine all text with separators
        combined_text = f"{post_text} [TITLE] {target_title} [CONTENT] {target_paragraphs}"
        features.append(combined_text)
    
    return features

def create_meta_features(df):
    """Create numerical meta features"""
    meta_features = pd.DataFrame()
    
    # Text length features (words)
    meta_features['post_word_count'] = df['postText'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    meta_features['title_word_count'] = df['targetTitle'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    meta_features['content_word_count'] = df['targetParagraphs'].apply(
        lambda x: sum(len(str(p).split()) for p in x) if isinstance(x, list) else 0
    )
    
    # Character length features
    meta_features['post_char_count'] = df['postText'].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    meta_features['title_char_count'] = df['targetTitle'].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    
    # Punctuation features
    meta_features['post_question_marks'] = df['postText'].apply(
        lambda x: str(x).count('?') if pd.notna(x) else 0
    )
    meta_features['post_exclamation_marks'] = df['postText'].apply(
        lambda x: str(x).count('!') if pd.notna(x) else 0
    )
    meta_features['post_ellipsis'] = df['postText'].apply(
        lambda x: str(x).count('...') if pd.notna(x) else 0
    )
    
    # Paragraph count
    meta_features['num_paragraphs'] = df['targetParagraphs'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    
    # Ratios
    meta_features['post_to_content_ratio'] = (
        meta_features['post_word_count'] / (meta_features['content_word_count'] + 1)
    ).fillna(0)
    
    # Average words per paragraph
    meta_features['avg_words_per_paragraph'] = (
        meta_features['content_word_count'] / (meta_features['num_paragraphs'] + 1)
    ).fillna(0)
    
    return meta_features

# Create text features
print("Creating text features...")
train_text_features = create_text_features(train_df)
val_text_features = create_text_features(val_df)
test_text_features = create_text_features(test_df)

# Create meta features
print("Creating meta features...")
train_meta = create_meta_features(train_df)
val_meta = create_meta_features(val_df)
test_meta = create_meta_features(test_df)

# Create labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['tags'])
val_labels = label_encoder.transform(val_df['tags'])

print(f"\nLabel encoding mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label} -> {i}")

# 5. TEXT VECTORIZATION
print("\n=== TEXT VECTORIZATION ===")

# TF-IDF Vectorization
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    stop_words='english',
    strip_accents='unicode'
)

X_train_text = tfidf.fit_transform(train_text_features)
X_val_text = tfidf.transform(val_text_features)
X_test_text = tfidf.transform(test_text_features)

# Normalize meta features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_meta_scaled = scaler.fit_transform(train_meta)
val_meta_scaled = scaler.transform(val_meta)
test_meta_scaled = scaler.transform(test_meta)

# Combine text and meta features
X_train = hstack([X_train_text, train_meta_scaled])
X_val = hstack([X_val_text, val_meta_scaled])
X_test = hstack([X_test_text, test_meta_scaled])

print(f"Final feature shapes:")
print(f"  Train: {X_train.shape}")
print(f"  Val: {X_val.shape}")
print(f"  Test: {X_test.shape}")

# 6. MODEL TRAINING
print("\n=== MODEL TRAINING ===")

# Define models
models = {
    'Logistic Regression': LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(
        model, X_train, train_labels,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro',
        n_jobs=-1
    )
    cv_results[name] = cv_scores
    print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train and evaluate on validation set
print("\n=== VALIDATION SET EVALUATION ===")
best_model = None
best_model_name = None
best_f1 = 0

for name, model in models.items():
    print(f"\n{name}:")
    
    # Train
    model.fit(X_train, train_labels)
    
    # Predict
    val_pred = model.predict(X_val)
    
    # Calculate metrics
    f1 = f1_score(val_labels, val_pred, average='macro')
    
    print(f"Macro F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        val_labels, val_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # Track best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} with F1={best_f1:.4f}")

# Confusion matrix for best model
val_pred_best = best_model.predict(X_val)
cm = confusion_matrix(val_labels, val_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# 7. FINAL MODEL TRAINING
print("\n=== TRAINING FINAL MODEL ===")

# Combine train and validation data
combined_text = train_text_features + val_text_features
combined_meta = np.vstack([train_meta, val_meta])
combined_labels = np.concatenate([train_labels, val_labels])

# Re-fit TF-IDF on combined data
tfidf_final = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    stop_words='english',
    strip_accents='unicode'
)

X_combined_text = tfidf_final.fit_transform(combined_text)
X_test_final_text = tfidf_final.transform(test_text_features)

# Re-fit scaler on combined data
scaler_final = StandardScaler()
combined_meta_scaled = scaler_final.fit_transform(combined_meta)
test_meta_scaled_final = scaler_final.transform(test_meta)

# Combine features
X_combined = hstack([X_combined_text, combined_meta_scaled])
X_test_final = hstack([X_test_final_text, test_meta_scaled_final])

# Train final model (using best model type)
if best_model_name == 'Logistic Regression':
    final_model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
else:
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

print(f"Training {best_model_name} on combined dataset...")
final_model.fit(X_combined, combined_labels)

# 8. GENERATE PREDICTIONS
print("\n=== GENERATING PREDICTIONS ===")

# Make predictions
test_predictions = final_model.predict(X_test_final)
test_predictions_proba = final_model.predict_proba(X_test_final)

# Convert to labels
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Create submission
submission = pd.DataFrame({
    'id': range(len(test_df)),
    'spoilerType': test_predictions_labels
})

# Show prediction distribution
print("\nPrediction distribution:")
pred_counts = submission['spoilerType'].value_counts()
print(pred_counts)
print("\nPercentage distribution:")
print((pred_counts / len(submission) * 100).round(2))

# Compare with training distribution
print("\nComparison with training distribution:")
train_dist = train_df['tags'].value_counts(normalize=True) * 100
pred_dist = submission['spoilerType'].value_counts(normalize=True) * 100
comparison = pd.DataFrame({
    'Training %': train_dist.round(2),
    'Predictions %': pred_dist.round(2)
})
print(comparison)

# Save predictions
submission.to_csv('prediction_task1.csv', index=False)
print(f"\nPredictions saved to 'prediction_task1.csv'")

# Show sample predictions
print("\nFirst 20 predictions:")
print(submission.head(20))

# Show confidence scores for first 10 predictions
print("\nConfidence scores for first 10 predictions:")
for i in range(min(10, len(test_predictions_proba))):
    probs = test_predictions_proba[i]
    pred_label = test_predictions_labels[i]
    confidence = max(probs) * 100
    print(f"ID {i}: {pred_label} (confidence: {confidence:.1f}%)")

print("\n=== PIPELINE COMPLETE ===")
print(f"Final model: {type(final_model).__name__}")
print(f"Best validation F1 score: {best_f1:.4f}")
print(f"Features used: {X_train.shape[1]} total ({X_train_text.shape[1]} text + {train_meta.shape[1]} meta)")

