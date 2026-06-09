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


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier, Pool


import warnings
warnings.filterwarnings('ignore')

# Set some display options for better visualization
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

print("Libraries imported successfully!")


# Load the data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
#df_orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

print(f"Training data shape: {df_train.shape}")
print(f"Test data shape: {df_test.shape}")
#print(f"Original dataset shape: {df_orig.shape}")
print(f"\nTarget distribution:")
print(df_train['loan_paid_back'].value_counts(normalize=True))


print("Training Data Head:")
display(df_train.head())

print("\nTraining Data Info:")
print(df_train.info())

print("\nMissing Values in Train Data:")
print(df_train.isnull().sum())

print("\nMissing Values in Test Data:")
print(df_test.isnull().sum())

print('\nDescriptive statistics for numerical columns') 
display(df_train.describe())


# Distribution of the target variable 'accident_risk'
plt.figure(figsize=(10, 6))
sns.countplot(x='loan_paid_back', data=df_train, palette='pastel', edgecolor='black')
plt.title('Distribution of Loan Payback')
plt.xlabel('Loan Payback')
plt.ylabel('Count')
plt.show()


categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)


# compact view of categorical features vs the target
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()
cmap = plt.get_cmap('magma')
colors = cmap([0.9, 0.66, 0.33])
target = 'loan_paid_back'

for i, col in enumerate(categorical_features):
    grouped = df_train.groupby(col)[target].mean()
    axes[i].bar(grouped.index.astype(str), grouped.values, color=colors)
    axes[i].set_ylabel(f'Mean {target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.show()


numerical_features = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'loan_paid_back']]
print(numerical_features)


# Loop through all numerical features
for col in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # --- Left: Distribution (Histogram + KDE) ---
    sns.histplot(df_train[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f"Distribution of {col}", fontsize=12)
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frequency")
    
    # --- Right: Boxplot (Outliers) ---
    sns.boxplot(x=df_train[col], ax=axes[1], color='lightcoral')
    axes[1].set_title(f"Boxplot of {col}", fontsize=12)
    axes[1].set_xlabel(col)
    
    # Clean layout
    plt.tight_layout()
    plt.show()





# def create_frequency_features(df, df_test, cat_cols, num_cols):
#     """
#     Add frequency and binning features efficiently.

#     - For each categorical column, create <col>_freq = how often each value appears in train data.
#     - For numeric columns, split values into 5, 10, 15 quantile bins.
    
#     Parameters:
#     - df: Training DataFrame
#     - df_test: Test DataFrame
#     - cat_cols: List of categorical column names for frequency encoding
#     - num_cols: List of numerical column names for binning
#     """
#     # Pre-allocate DataFrames for new features to avoid fragmentation
#     freq_features_train = pd.DataFrame(index=df.index)
#     freq_features_test = pd.DataFrame(index=df_test.index)
#     bin_features_train = pd.DataFrame(index=df.index)
#     bin_features_test = pd.DataFrame(index=df_test.index)

#     # --- Frequency encoding for categorical columns ---
#     for col in cat_cols:
#         freq = df[col].value_counts()
#         freq_features_train[f"{col}_freq"] = df[col].map(freq)
#         # Fill unseen categories in test set with 0 (they have zero frequency in train)
#         freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(0)

#     # --- Quantile binning for numeric columns ---
#     for col in num_cols:
#         for q in [5, 10, 15]:
#             try:
#                 train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
#                 bin_features_train[f"{col}_bin{q}"] = train_bins
#                 # Use pd.cut for test data using train's bins
#                 bin_features_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True).fillna(0)
#             except Exception:
#                 # This can happen if a column has too few unique values
#                 bin_features_train[f"{col}_bin{q}"] = 0
#                 bin_features_test[f"{col}_bin{q}"] = 0

#     # Concatenate all new features at once
#     df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
#     df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)

#     return df, df_test


target_col = 'loan_paid_back'

numerical_features = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'loan_paid_back']]
print(numerical_features)

categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)


# df_train, df_test = create_frequency_features(df_train, df_test, 
#                                               cat_cols=categorical_features, 
#                                               num_cols=numerical_features)


y = df_train[target_col]
test_ids = df_test['id']

df_train_fe = df_train.drop(['id', target_col], axis=1)
df_test_fe = df_test.drop(['id'], axis=1)


X = df_train_fe.copy()
X_test = df_test_fe.copy()


print(f"Final training data shape: {X.shape}")
print(f"Final test data shape: {X_test.shape}")


# numerical_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
# numerical_features = [col for col in numerical_features if col not in ['id', 'loan_paid_back']]
# categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()


catboost_params = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 5,
    'l2_leaf_reg': 3,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'od_type': 'Iter',      # Use early stopping
    'od_wait': 100,         # Stop after 100 rounds of no improvement
    'grow_policy': 'Lossguide',
    'thread_count': -1,
    'verbose': False,       # Suppress training output
    'verbose': 100,
    'allow_writing_files': False
}


# from catboost import CatBoostClassifier, Pool
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score

## 1. Setup Cross-Validation
N_SPLITS = 5 # You can change this    
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Create arrays to store results
val_scores = []
oof_preds = np.zeros(len(X_test)) # Out-of-fold predictions for the test set
# oof_train_preds = np.zeros(len(X)) # Optional: To store validation preds

## 2. Start the Cross-Validation Loop
# (This continues from your provided loop structure)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/{N_SPLITS}")
    print(f"{'='*60}")
    
    # Get fold data (using your variables)
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # --- Inside the loop ---
    
    # 1. Initialize the model
    # We pass the params dictionary directly
    model = CatBoostClassifier(**catboost_params)
    
    # 2. Fit the model
    # We provide the categorical features list
    # The validation set is used for early stopping
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              cat_features=categorical_features
              #use_best_model=True
             )
    
    # 3. Make predictions on the validation set
    # We use predict_proba to get probabilities for AUC
    val_preds = model.predict_proba(X_val)[:, 1]
    
    # 4. Score the model
    fold_auc = roc_auc_score(y_val, val_preds)
    val_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")
    
    # 5. Make predictions on the test set
    # We average the predictions from each fold
    oof_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    # Optional: Store validation predictions
    # oof_train_preds[val_idx] = val_preds


# Report Final Score
print(f"\n{'='*60}")
print(f"Overall CV AUC: {np.mean(val_scores):.5f} +/- {np.std(val_scores):.5f}")
print(f"{'='*60}")


# Create Submission File
submission_df = pd.DataFrame({'id': test_ids, target_col: oof_preds})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print(submission_df.head())

