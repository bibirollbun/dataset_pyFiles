# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# Load the data
train_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')
sample_submission = pd.read_csv('/kaggle/input/rmit-hackathon-2025/sample_submission.csv')

print("âœ… Data loaded successfully!")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nFirst few rows of training data:")
train_df.head()


# Convert labels to binary (if not already)
# jailbreak = 1, benign = 0
train_df['target'] = (train_df['label'] == 'jailbreak').astype(int)

# Check class distribution
print("ğŸ�¯ Class Distribution:")
print(train_df['label'].value_counts())
print(f"\nPercentage:")
print(train_df['label'].value_counts(normalize=True) * 100)

# Visualize class distribution
plt.figure(figsize=(8, 5))
train_df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution: Benign vs Jailbreak', fontsize=14, fontweight='bold')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Analyze text length
train_df['text_length'] = train_df['text'].str.len()

print("ğŸ“� Text Length Statistics:")
print(train_df.groupby('label')['text_length'].describe())

# Visualize text length distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(train_df[train_df['label'] == 'benign']['text_length'], bins=50, alpha=0.7, label='Benign', color='green')
plt.hist(train_df[train_df['label'] == 'jailbreak']['text_length'], bins=50, alpha=0.7, label='Jailbreak', color='red')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Text Length Distribution by Class')
plt.legend()

plt.subplot(1, 2, 2)
train_df.boxplot(column='text_length', by='label', ax=plt.gca())
plt.suptitle('')
plt.title('Text Length by Class (Boxplot)')
plt.xlabel('Label')
plt.ylabel('Text Length')

plt.tight_layout()
plt.show()


# Split data for validation
X = train_df['text']
y = train_df['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"âœ… Data split complete!")
print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")


# Create TF-IDF features
print("ğŸ”„ Creating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
    min_df=2,
    max_df=0.9,
    strip_accents='unicode',
    lowercase=True,
    analyzer='word',
    token_pattern=r'\w{1,}',
    sublinear_tf=True
)

# Fit and transform training data
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

print(f"âœ… TF-IDF features created!")
print(f"Feature matrix shape: {X_train_tfidf.shape}")


# Train Logistic Regression model
print("ğŸ”„ Training Logistic Regression model...")

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=1.0,
    solver='saga',
    class_weight='balanced'
)

lr_model.fit(X_train_tfidf, y_train)

print("âœ… Model trained successfully!")


# Evaluate on validation set
y_val_pred_proba = lr_model.predict_proba(X_val_tfidf)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred_proba)

print(f"ğŸ“Š Validation Results:")
print(f"ROC-AUC Score: {val_auc:.4f}")

# Get predictions for classification report
y_val_pred = lr_model.predict(X_val_tfidf)
print(f"\nğŸ“ˆ Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Benign', 'Jailbreak']))

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Jailbreak'], yticklabels=['Benign', 'Jailbreak'])
plt.title('Confusion Matrix - TF-IDF + Logistic Regression', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()


# Retrain on full training data
print("ğŸ”„ Retraining on full training data...")

X_full_tfidf = tfidf.fit_transform(train_df['text'])
lr_model.fit(X_full_tfidf, train_df['target'])

print("âœ… Model retrained on full data!")


# Generate predictions for test set
print("ğŸ”„ Generating predictions for test set...")

X_test_tfidf = tfidf.transform(test_df['text'])
test_predictions = lr_model.predict_proba(X_test_tfidf)[:, 1]

print(f"âœ… Predictions generated!")
print(f"Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
print(f"Mean prediction: {test_predictions.mean():.4f}")


# Create submission file
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'TARGET': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file created: submission.csv")
print("\nFirst few predictions:")
print(submission.head(10))




