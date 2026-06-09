import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)



# Define paths
TRAIN_LABELS_PATH = '/kaggle/input/make-data-count-finding-data-references/train_labels.csv'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/make-data-count-finding-data-references/sample_submission.csv'
TRAIN_PDF_PATH = '/kaggle/input/make-data-count-finding-data-references/train/PDF'
TRAIN_XML_PATH = '/kaggle/input/make-data-count-finding-data-references/train/XML'
TEST_PDF_PATH = '/kaggle/input/make-data-count-finding-data-references/test/PDF'
TEST_XML_PATH = '/kaggle/input/make-data-count-finding-data-references/test/XML'



# Load data
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# Display head of train labels
print("Train Labels Head:")
train_labels.head()


# Basic Statistics and Exploration
print("\n--- EDA ---")
print("\nTrain Labels Info:")
train_labels.info()


print("\nTrain Labels Describe:")
train_labels.describe(include='all')


print("\nTrain Labels Value Counts (Type):")
print(train_labels['type'].value_counts())


# Data Visualization
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Subplot 1: Distribution of 'type'
plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
ax1 = sns.countplot(data=train_labels, x='type', palette="viridis")
plt.title('Distribution of Data Types', fontsize=14, fontweight='bold', color='darkblue')
plt.xlabel('Data Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.gca().set_facecolor('#f0f0f0')

# Add value annotations
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Subplot 2: Distribution of 'article_id'
plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
article_counts = train_labels['article_id'].value_counts()
sns.histplot(article_counts, bins=30, kde=True, color='skyblue')
plt.title('Distribution of Article IDs', fontsize=14, fontweight='bold', color='darkblue')
plt.xlabel('Number of References', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.gca().set_facecolor('#f0f0f0')

# Subplot 3: Length of dataset_id
plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
dataset_id_lengths = train_labels['dataset_id'].str.len()
sns.histplot(dataset_id_lengths, bins=30, kde=True, color='lightcoral')
plt.title('Distribution of Dataset ID Lengths', fontsize=14, fontweight='bold', color='darkblue')
plt.xlabel('Length of Dataset ID', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.gca().set_facecolor('#f0f0f0')

plt.tight_layout()
plt.show()

# Feature Engineering (Example: Length of dataset_id)
train_labels['dataset_id_length'] = train_labels['dataset_id'].str.len()

# Visualizing the distribution of dataset_id_length
plt.figure(figsize=(8, 6))
sns.histplot(train_labels['dataset_id_length'], bins=30, kde=True, color='mediumseagreen')
plt.title('Distribution of Dataset ID Lengths', fontsize=14, fontweight='bold', color='darkblue')
plt.xlabel('Length of Dataset ID', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.gca().set_facecolor('#f0f0f0')
plt.show()

# Feature Engineering : DOI Related Feature
def is_doi(dataset_id):
    return 1 if 'doi.org' in dataset_id else 0

train_labels['is_doi'] = train_labels['dataset_id'].apply(is_doi)

# Visualization of the new feature
plt.figure(figsize=(6, 4))
sns.countplot(data=train_labels, x='is_doi', palette='coolwarm')
plt.title('Distribution of DOI Presence', fontsize=14, fontweight='bold', color='darkblue')
plt.xlabel('Is DOI Present', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['No', 'Yes'])
plt.gca().set_facecolor('#f0f0f0')

# Add value annotations
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.show()



# Prepare data for model training
X = train_labels[['dataset_id_length', 'is_doi']]  # Features
y = train_labels['type']  # Target variable

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded)



# Train XGBoost model

# Define XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',  # Specify for multi-class classification
    num_class=len(label_encoder.classes_),  # Number of classes
    eval_metric='mlogloss',  # Metric for multi-class classification
    random_state=SEED,
    use_label_encoder=False # Suppress a warning
)



# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = xgb_model.predict(X_val)

# Convert predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred_val)

# Calculate F1 score
f1 = f1_score(label_encoder.inverse_transform(y_val), y_pred_labels, average='weighted')
print(f"Validation F1 Score: {f1}")


# 1. Create dummy features for the test set (as we don't have real test data)
# In a real scenario, we would extract these features from the test data
sample_submission['dataset_id_length'] = sample_submission['dataset_id'].str.len()
sample_submission['is_doi'] = sample_submission['dataset_id'].apply(is_doi)

# Prepare test data for prediction
X_test = sample_submission[['dataset_id_length', 'is_doi']]

# Make predictions on the test set
y_pred_test = xgb_model.predict(X_test)

# Convert predictions back to original labels
sample_submission['type'] = label_encoder.inverse_transform(y_pred_test)



# Create submission file
submission = sample_submission[['row_id', 'article_id', 'dataset_id', 'type']]

# Display head of submission file
print("\nSubmission Head:")
print(submission.head())

# Save submission file
submission.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")

