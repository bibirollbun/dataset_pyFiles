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
import warnings

# Import models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Import utilities
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
# This step loads the data you have available in your Kaggle input directory.
print("Step 1: Loading data...")
# The file paths are set for the standard Kaggle environment
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# --- 2. Advanced Feature Engineering ---
# Creating new features is key to unlocking higher performance.
print("\nStep 2: Performing advanced feature engineering...")
def feature_engineer(df):
    """Creates new, insightful features to improve model performance."""
    # Convert binary text to numbers first
    for col in ['Stage_fear', 'Drained_after_socializing']:
        if col in df.columns:
            # Check if the column is not already numeric
            if df[col].dtype == 'object':
                df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Fill missing values before creating new features
    # This ensures calculations don't fail due to NaNs
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            elif df[col].dtype == 'object':
                 df[col].fillna(df[col].mode()[0], inplace=True)

    # Interaction Features: Capture relationships between variables
    df['social_engagement'] = df['Social_event_attendance'] * df['Post_frequency']
    df['friend_circle_stability'] = df['Friends_circle_size'] / (df['Time_spent_Alone'] + 1e-6) # Add epsilon to avoid division by zero

    # Statistical Features: Summarize behavioral patterns
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    df['social_mean'] = df[numerical_cols].mean(axis=1)
    df['social_std'] = df[numerical_cols].std(axis=1)

    return df

train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)


# --- 3. Preprocessing ---
print("\nStep 3: Preprocessing data for modeling...")
# Encode the target variable 'Personality' into 1s and 0s
train_df['Personality'] = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

# Align columns to ensure the test set has the exact same features as the training set
train_labels = train_df['Personality']
train_ids = train_df['id']
test_ids = test_df['id']

# Drop unnecessary columns
train_df = train_df.drop(columns=['id', 'Personality'])
test_df = test_df.drop(columns=['id'])

# Ensure feature consistency and order
features = [col for col in train_df.columns if col in test_df.columns]
X = train_df[features]
X_test = test_df[features]


# --- 4. Stacking Ensemble Training ---
print("\nStep 4: Training the Stacking Ensemble with 5-fold cross-validation...")

# Setup 5-fold cross-validation
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Define the three powerful base models
base_models = {
    'xgb': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'lgbm': LGBMClassifier(random_state=42),
    'cat': CatBoostClassifier(random_state=42, verbose=0)
}

# Store the predictions from the base models (these will be the features for our meta-model)
oof_meta_features = np.zeros((len(X), len(base_models)))
test_meta_features = np.zeros((len(X_test), len(base_models)))

# Loop through each model to train it and get its predictions
for i, (model_name, model) in enumerate(base_models.items()):
    print(f"  Training base model: {model_name}...")
    # This inner loop trains the model on 4 folds and predicts on the 5th, repeating 5 times
    for fold_, (train_idx, val_idx) in enumerate(folds.split(X, train_labels)):
        X_train, y_train = X.iloc[train_idx], train_labels.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], train_labels.iloc[val_idx]

        model.fit(X_train, y_train)

        # Predict on the validation fold (these are "out-of-fold" predictions)
        oof_meta_features[val_idx, i] = model.predict_proba(X_val)[:, 1]

        # Predict on the test set and average the predictions over the 5 folds
        test_meta_features[:, i] += model.predict_proba(X_test)[:, 1] / NFOLDS


# --- 5. Meta-Model Training ---
print("\nStep 5: Training the Meta-Model to blend predictions...")

# Train the final meta-model on the predictions from our base models
meta_model = LogisticRegression(random_state=42)
meta_model.fit(oof_meta_features, train_labels)

# Make final predictions using the trained meta-model
final_predictions_proba = meta_model.predict_proba(test_meta_features)[:, 1]
# Convert probabilities to final class labels (0 or 1)
final_predictions = (final_predictions_proba > 0.5).astype(int)


# --- 6. Create Submission File ---
print("\nStep 6: Creating the final submission file...")
submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})
# Convert numeric predictions back to text labels ('Extrovert'/'Introvert')
submission_df['Personality'] = submission_df['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission_df.to_csv('submission.csv', index=False)

print("\n✅ Submission file 'submission.csv' has been created successfully!")
print("Top 5 rows of your submission:")
print(submission_df.head())

