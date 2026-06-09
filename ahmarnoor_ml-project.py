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


# PLAsTiCC Astronomical Classification - Phase 1
# EC452 Machine Learning Semester Project
!pip install lightgbm

!pip install xgboost

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data from Kaggle dataset path
meta_path = '/kaggle/input/PLAsTiCC-2018/training_set_metadata.csv'
lc_path = '/kaggle/input/PLAsTiCC-2018/training_set.csv'

train_meta = pd.read_csv(meta_path)
train_lc = pd.read_csv(lc_path)

# Data Exploration
print("\nMetadata shape:", train_meta.shape)
print("Light curves shape:", train_lc.shape)

# Enhanced Feature Engineering
def create_features(df):
    grouped = df.groupby('object_id')

    # Basic flux features
    features = grouped['flux'].agg(['mean', 'std', 'min', 'max', 'skew', 'median']).add_prefix('flux_')
    features['flux_range'] = features['flux_max'] - features['flux_min']
    features['flux_ratio'] = (features['flux_max'] - features['flux_mean']) / (features['flux_mean'] - features['flux_min'] + 1e-6)

    # Passband-specific features
    passbands = sorted(df['passband'].unique())
    for band in passbands:
        band_group = df[df['passband'] == band].groupby('object_id')
        features[f'flux_mean_band_{band}'] = band_group['flux'].mean()
        features[f'flux_std_band_{band}'] = band_group['flux'].std()
        features[f'flux_amp_band_{band}'] = band_group['flux'].max() - band_group['flux'].min()

    # Time features
    features['mjd_span'] = grouped['mjd'].max() - grouped['mjd'].min()
    features['mjd_density'] = grouped.size() / (features['mjd_span'] + 1e-6)

    # Detection features
    features['detection_rate'] = grouped['detected'].mean()
    features['detected_count'] = grouped['detected'].sum()

    # Flux error features
    features['flux_err_ratio'] = grouped['flux_err'].mean() / (grouped['flux'].std() + 1e-6)

    return features

print("\nCreating features...")
train_features = create_features(train_lc)

# Merge data
train = train_meta.merge(train_features, left_on='object_id', right_index=True, how='left')

# Handle missing values
num_imputer = SimpleImputer(strategy='median')
numerical_cols = train.select_dtypes(include=np.number).columns
train[numerical_cols] = num_imputer.fit_transform(train[numerical_cols])

# Feature selection
train = train.drop(['object_id'], axis=1)

# New features
train['hostgal_photoz_uncertainty'] = train['hostgal_photoz_err'] / (train['hostgal_photoz'] + 1e-6)
train['specz_photoz_diff'] = np.abs(train['hostgal_specz'] - train['hostgal_photoz'])

# Prepare data
X = train.drop('target', axis=1)
y = train['target']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)

# Class weights
class_counts = np.bincount(y_encoded)
class_weights = {i: sum(class_counts)/class_counts[i] for i in range(n_classes)}
sample_weights = np.array([class_weights[y] for y in y_encoded])

# Train-test split
X_train, X_val, y_train, y_val, weights_train, _ = train_test_split(
    X, y_encoded, sample_weights, test_size=0.15,
    stratify=y_encoded, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Model configuration
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    ),

    'XGBoost': XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=n_classes,
        random_state=42,
        n_jobs=-1
    )
}

# Train models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == 'XGBoost':
        model.fit(X_train_scaled, y_train,
                  sample_weight=weights_train,
                  eval_set=[(X_val_scaled, y_val)],
                  verbose=False)
    else:
        model.fit(X_train_scaled, y_train,
                  sample_weight=weights_train)

    probs = model.predict_proba(X_val_scaled)
    loss = log_loss(y_val, probs)
    results[name] = loss
    print(f"{name} Validation Log Loss: {loss:.4f}")

# Results
print("\nModel Performance:")
for name, loss in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name}: {loss:.4f}")

# Feature Importance
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    plt.figure(figsize=(12, 8))
    importances.sort_values().tail(20).plot.barh()
    plt.title(f'{best_model_name} Feature Importance')
    plt.show()

# Save artifacts (optional in Kaggle environment, paths can be adjusted)
joblib.dump(best_model, f'best_{best_model_name.lower()}_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Confusion Matrix
y_pred = best_model.predict(X_val_scaled)
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_.astype(str))
fig, ax = plt.subplots(figsize=(15, 15))
disp.plot(ax=ax, xticks_rotation=90)
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_.astype(str)))



import os
import glob
import gc
import pandas as pd
import numpy as np

# Load test metadata
test_meta = pd.read_csv('/kaggle/input/PLAsTiCC-2018/test_set_metadata.csv')

# Collect ONLY the 11 test batch files (exclude test_set.csv)
batch_files = sorted(glob.glob('/kaggle/input/PLAsTiCC-2018/test_set_batch*.csv'))

# List to hold predictions
submission_parts = []

# Features used in training
trained_features = X.columns.tolist()

# Prediction function
def process_test_batch(batch_df):
    # Create features for this batch's light curves
    test_features = create_features(batch_df)

    # Select metadata rows for these object_ids
    meta = test_meta[test_meta['object_id'].isin(test_features.index)].copy()
    meta['hostgal_photoz_uncertainty'] = meta['hostgal_photoz_err'] / (meta['hostgal_photoz'] + 1e-6)
    meta['specz_photoz_diff'] = np.abs(meta['hostgal_specz'] - meta['hostgal_photoz'])

    # Merge features and metadata
    test_full = meta.merge(test_features, left_on='object_id', right_index=True, how='left')

    # Fill missing values
    test_full = test_full.fillna(0)

    # Add missing columns to match training features
    for col in trained_features:
        if col not in test_full.columns:
            test_full[col] = 0

    # Reorder columns to match training features
    test_full = test_full[trained_features]

    object_ids = meta['object_id']
    test_scaled = scaler.transform(test_full)

    # Predict probabilities
    preds = best_model.predict_proba(test_scaled)

    # Build predictions DataFrame
    df_preds = pd.DataFrame(preds, columns=le.classes_.astype(str))
    df_preds.insert(0, 'object_id', object_ids.values)

    # Add the required '99' class with zero probabilities
    df_preds['99'] = 0.0

    return df_preds

print("\nðŸš€ Starting prediction on the 11 test batches only...\n")

for i, batch_file in enumerate(batch_files):
    print(f"Processing: {os.path.basename(batch_file)} ({i+1}/{len(batch_files)})")
    batch_df = pd.read_csv(batch_file)
    batch_preds = process_test_batch(batch_df)
    submission_parts.append(batch_preds)
    del batch_df, batch_preds
    gc.collect()

# Combine predictions from all batches
final_submission = pd.concat(submission_parts, ignore_index=True)

# Save submission file
final_submission.to_csv('submission.csv', index=False)
print("\nâœ… Saved submission.csv with shape:", final_submission.shape)

# Fix column names to match sample_submission format
# Convert '6.0' â†’ 'class_6'
final_submission.columns = ['object_id'] + [f'class_{int(float(c))}' for c in final_submission.columns[1:]]

# Reorder columns to exactly match sample_submission.csv
sample_sub_path = '/kaggle/input/PLAsTiCC-2018/sample_submission.csv'
sample_cols = pd.read_csv(sample_sub_path, nrows=1).columns.tolist()
final_submission = final_submission[sample_cols]

# Save corrected submission
final_submission.to_csv('submission.csv', index=False)
print("\nâœ… Final submission.csv saved with correct column names and order.")




