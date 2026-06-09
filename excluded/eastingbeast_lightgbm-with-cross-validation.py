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


# Basic data manipulation
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning: preprocessing, model selection, metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score

# Tree-based models (excellent for tabular data)
import lightgbm as lgb

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("âœ… Libraries imported successfully!")


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# Display shapes
print(f"âœ… Training set: {train.shape}")
print(f"âœ… Testing set:  {test.shape}")

# Show first 5 rows of training data
train.head()


# === 1. Data Types ===
print("ğŸ“� Data Types of Features:")
print(train.dtypes)
print("\n" + "="*50 + "\n")

# === 2. Missing Values ===
print("ğŸ§© Missing Values (Training Set):")
missing = train.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("âœ… No missing values detected.")
else:
    print(missing)
print("\n" + "="*50 + "\n")

# === 3. Unique Values for Categorical Columns (Safely) ===
categorical_cols = train.select_dtypes(include='object').columns.tolist()

print("ğŸ�·ï¸�  Unique Values in Categorical Features:")
for col in categorical_cols:
    unique_vals = sorted(train[col].astype(str).unique())
    print(f"{col}: {train[col].nunique()} unique values â†’ {unique_vals}")
print("\n" + "="*50 + "\n")

# === 4. Numerical Summary Statistics ===
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in numerical_cols:
    numerical_cols.remove('id')  # Remove 'id' from numerical features if present
if 'y' in numerical_cols:
    numerical_cols.remove('y')   # Keep target separate

print("ğŸ“ˆ Summary Statistics (Numerical Features - Excluding id & y):")
display(train[numerical_cols].describe().T)

# === 5. Target Distribution ===
print("ğŸ�¯ Target Distribution (y):")
target_counts = train['y'].value_counts().sort_index()
print(target_counts)
print("\nPercentage of '1's (subscribed): {:.2f}%".format(100 * train['y'].mean()))


# Set up figure styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# === 1. Target Distribution ===
axes[0,0].pie(train['y'].value_counts(), labels=['No (0)', 'Yes (1)'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
axes[0,0].set_title("Target Distribution (y): Subscription Rate", fontsize=14)

# === 2. Age vs. y ===
sns.boxplot(data=train, x='y', y='age', ax=axes[0,1])
axes[0,1].set_title("Age vs. Subscription (y)")

# === 3. Balance vs. y ===
sns.boxplot(data=train, x='y', y='balance', ax=axes[1,0])
axes[1,0].set_title("Balance vs. Subscription (y)")

# === 4. Duration vs. y ===
sns.boxplot(data=train, x='y', y='duration', ax=axes[1,1])
axes[1,1].set_title("Call Duration vs. Subscription (y)")

plt.tight_layout()
plt.show()


# Set up figure styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# === 1. Job vs. Subscription Rate ===
job_subscription_rate = train.groupby('job')['y'].mean().sort_values()
sns.barplot(x=job_subscription_rate.values, y=job_subscription_rate.index, ax=axes[0,0], palette='viridis')
axes[0,0].set_title("Subscription Rate by Job", fontsize=14)
axes[0,0].set_xlabel("Subscription Rate")

# === 2. Education vs. Subscription Rate ===
edu_subscription_rate = train.groupby('education')['y'].mean().sort_values()
sns.barplot(x=edu_subscription_rate.values, y=edu_subscription_rate.index, ax=axes[0,1], palette='viridis')
axes[0,1].set_title("Subscription Rate by Education Level", fontsize=14)
axes[0,1].set_xlabel("Subscription Rate")

# === 3. Contact Method vs. Subscription Rate ===
contact_subscription_rate = train.groupby('contact')['y'].mean().sort_values()
sns.barplot(x=contact_subscription_rate.values, y=contact_subscription_rate.index, ax=axes[1,0], palette='viridis')
axes[1,0].set_title("Subscription Rate by Contact Method", fontsize=14)
axes[1,0].set_xlabel("Subscription Rate")

# === 4. Previous Campaign Outcome vs. Subscription Rate ===
poutcome_subscription_rate = train.groupby('poutcome')['y'].mean().sort_values()
sns.barplot(x=poutcome_subscription_rate.values, y=poutcome_subscription_rate.index, ax=axes[1,1], palette='viridis')
axes[1,1].set_title("Subscription Rate by Previous Campaign Outcome", fontsize=14)
axes[1,1].set_xlabel("Subscription Rate")

plt.tight_layout()
plt.show()


# Import libraries
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

print("ğŸ”„ Starting preprocessing...")

# === 1. Feature Engineering (Do this FIRST on train and test) ===
train = train.copy()
test = test.copy()

# New interaction features
train['balance_per_campaign'] = train['balance'] / (train['campaign'] + 1)
test['balance_per_campaign'] = test['balance'] / (test['campaign'] + 1)

train['duration_per_campaign'] = train['duration'] / (train['campaign'] + 1)
test['duration_per_campaign'] = test['duration'] / (test['campaign'] + 1)

# Log transform highly skewed features
train['log_balance'] = np.log1p(train['balance'])
test['log_balance'] = np.log1p(test['balance'])

# Bin age into groups (optional)
train['age_bin'] = pd.cut(train['age'], bins=5, labels=False)
test['age_bin'] = pd.cut(test['age'], bins=5, labels=False)

# Update lists of features
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
                  'balance_per_campaign', 'duration_per_campaign', 'log_balance', 'age_bin']

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Remove 'education' from one-hot list since we handle it separately
onehot_features = [col for col in categorical_cols if col != 'education']

# === 2. Define Encoders ===
# Ordinal Encoder for 'education' â€” now includes 'unknown'
ordinal_encoder = OrdinalEncoder(
    categories=[['primary', 'secondary', 'tertiary', 'unknown']],
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

# One-Hot Encoder for other categoricals
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# === 3. Build ColumnTransformer (Only Once!) ===
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),           # All numerical features
        ('ord', ordinal_encoder, ['education']),           # Ordinal encode education
        ('cat', onehot_encoder, onehot_features)          # One-hot rest
    ],
    remainder='drop'  # Drop any unlisted columns
)

# === 4. Prepare X and y ===
X_train = train.drop(['id', 'y'], axis=1)
y_train = train['y']
X_test = test.drop('id', axis=1)

# === 5. Fit and Transform ===
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("âœ… Preprocessing complete!")
print("Training shape after preprocessing:", X_train_processed.shape)
print("Test shape after preprocessing:", X_test_processed.shape)


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

# === 1. Set up cross-validation ===
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# === 2. LightGBM Parameters ===
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'baging_fraction': 0.8,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'min_child_samples': 20,
    'verbose': -1,
    'random_state': 42
}

# === 3. Training setup ===
oof_preds = np.zeros(X_train_processed.shape[0])
test_preds = np.zeros(X_test_processed.shape[0])
cv_scores = []
best_iterations = []  # âœ… Store best iteration from each fold

print("ğŸš€ Starting Cross-Validation...\n")

# === 4. Train one fold at a time ===
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_train)):
    print(f"ğŸ�‹ï¸�â€�â™‚ï¸� Training Fold {fold + 1}/{n_splits}")
    
    X_tr, X_val = X_train_processed[train_idx], X_train_processed[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    model = lgb.train(
        params=lgb_params,
        train_set=train_set,
        num_boost_round=10000,
        valid_sets=[train_set, val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(100)
        ]
    )
    
    # Save predictions and score
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    
    fold_auc = roc_auc_score(y_val, val_pred)
    cv_scores.append(fold_auc)
    best_iterations.append(model.best_iteration)  # âœ… Save best iteration
    
    print(f"âœ… Fold {fold + 1} AUC: {fold_auc:.6f}, Best Iteration: {model.best_iteration}\n")

# === 5. Final CV Score ===
mean_cv_auc = np.mean(cv_scores)
std_cv_auc = np.std(cv_scores)
print(f"ğŸ�‰ Mean CV AUC: {mean_cv_auc:.6f} Â± {std_cv_auc:.6f}")
print(f"ğŸ“Œ OOF ROC AUC: {roc_auc_score(y_train, oof_preds):.6f}")
print(f"ğŸ“Š Average Best Iteration: {int(np.mean(best_iterations))}")

# === 6. Predict on Test Set ===
print("\nğŸ”® Generating final test predictions...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_train)):
    X_tr, y_tr = X_train_processed[train_idx], y_train.iloc[train_idx]
    
    train_set = lgb.Dataset(X_tr, label=y_tr)
    
    # Use stored best iteration from this fold
    num_rounds = best_iterations[fold]
    
    model = lgb.train(
        params=lgb_params,
        train_set=train_set,
        num_boost_round=num_rounds,
        callbacks=[
            lgb.log_evaluation(False)  # Silent
        ]
    )
    
    test_preds += model.predict(X_test_processed, num_iteration=num_rounds) / n_splits

print("âœ… Final test predictions ready!")


# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'y': test_preds  # Already averaged across 5 folds
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

# Display first few rows
print("ğŸ�‰ Submission file created: 'submission.csv'")
print(f"ğŸ“Š Shape: {submission.shape}")
print(submission.head())

