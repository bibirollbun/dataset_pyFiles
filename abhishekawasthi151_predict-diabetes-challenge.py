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


import os

# 0 = all logs, 1 = filter info, 2 = filter warnings, 3 = filter errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import math
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# --- LOAD EXTERNAL DATA ---
df_orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')




print("Training Data Head:")
df_train.head()


print("\nTraining Data Info:")
df_train.info()


print("\nMissing Values in Train Data:")
print(df_train.isnull().sum())


df_train.head()


print("\nMissing Values in Test Data:")
print(df_test.isnull().sum())


df_train.head()


# Descriptive statistics for numerical columns
df_train.describe()


# Distribution of the target variable 'accident_risk'
plt.figure(figsize=(10, 6))
sns.countplot(x='diagnosed_diabetes', data=df_train, palette='pastel', edgecolor='black')
plt.title('Distribution of Diagnosed Diabetes')
plt.xlabel('Diagnosed Diabetes')
plt.ylabel('Count')
plt.show()


categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)


# A more compact view of categorical features vs the target
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()
cmap = plt.get_cmap('magma')
colors = cmap([0.9, 0.66, 0.33])
target = 'diagnosed_diabetes'

for i, col in enumerate(categorical_features):
    grouped = df_train.groupby(col)[target].mean()
    axes[i].bar(grouped.index.astype(str), grouped.values, color=colors)
    axes[i].set_ylabel(f'Mean {target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.show()


numerical_features = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'diagnosed_diabetes']]
print(numerical_features)


# (Histogram/KDE) and the outlier check (Boxplot)
# Filter numerical features
numerical_features = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'diagnosed_diabetes']]

# --- Configuration for the Grid ---
features_per_row = 3
total_features = len(numerical_features)
n_rows = math.ceil(total_features / features_per_row)

# We need 2 columns per feature (1 for Hist, 1 for Box), so n_cols = features_per_row * 2
fig, axes = plt.subplots(n_rows, features_per_row * 2, figsize=(20, 4 * n_rows))
axes = axes.flatten() # Flatten to make indexing easier

for i, col in enumerate(numerical_features):
    # Calculate the exact spots for this feature in the flattened grid
    # Each feature takes up 2 spots: index 2*i and 2*i + 1
    hist_idx = i * 2
    box_idx = i * 2 + 1
    
    # --- Plot 1: Distribution (Histogram + KDE) ---
    sns.histplot(df_train[col], kde=True, ax=axes[hist_idx], color='skyblue')
    axes[hist_idx].set_title(f"{col} Dist", fontsize=10)
    axes[hist_idx].set_xlabel('')
    axes[hist_idx].set_ylabel('') # Save space
    
    # --- Plot 2: Boxplot (Outliers) ---
    sns.boxplot(x=df_train[col], ax=axes[box_idx], color='lightcoral')
    axes[box_idx].set_title(f"{col} Box", fontsize=10)
    axes[box_idx].set_xlabel('')
    
# Hide any unused subplots (if features aren't a perfect multiple of 3)
for j in range(len(numerical_features) * 2, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


# Violin Plot
n_cols = 3
n_rows = math.ceil(len(numerical_features) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    sns.violinplot(x=df_train[col], ax=axes[i], color='mediumpurple')
    axes[i].set_title(col)

# Hide empty plots
for j in range(len(numerical_features), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


# add 'id' column in the original dataset
df_orig['id'] = df_orig.index


# ensure original dataset has the same columns as the training dataset
df_orig = df_orig[df_train.columns.to_list()]


df_train.shape


df_train = df_train.merge(df_orig, how='outer')
df_train.shape


# External Target Encoding using External Dataset (100k records)

def add_external_encoding(train_df, test_df, ext_df, target_col='diagnosed_diabetes'):
    print("Adding External Encodings...")
    
    # Identify columns that exist in both datasets (excluding ID and Target)
    common_cols = [c for c in train_df.columns 
                   if c in ext_df.columns 
                   and c not in ['id', target_col]]
    
    # Create temporary copies to avoid SettingWithCopy warnings
    tr = train_df.copy()
    te = test_df.copy()
    
    for col in common_cols:
        # 1. MEAN ENCODING (The "Risk Score")
        # Calculate avg target for each value in the EXTERNAL dataset
        mapping = ext_df.groupby(col)[target_col].mean()
        
        # Map this "risk" to our competition data
        tr[f'ext_mean_{col}'] = tr[col].map(mapping)
        te[f'ext_mean_{col}'] = te[col].map(mapping)
        
        # 2. FREQUENCY ENCODING (How common is this value?)
        # Sometimes rare values in external data imply different risks
        counts = ext_df.groupby(col).size()
        
        # Use log(count) to normalize the range
        tr[f'ext_cnt_{col}'] = np.log1p(tr[col].map(counts).fillna(0))
        te[f'ext_cnt_{col}'] = np.log1p(te[col].map(counts).fillna(0))
        
        # Handle cases where a value in Train doesn't exist in External
        # We fill with the global average from external data
        global_mean = ext_df[target_col].mean()
        tr[f'ext_mean_{col}'] = tr[f'ext_mean_{col}'].fillna(global_mean)
        te[f'ext_mean_{col}'] = te[f'ext_mean_{col}'].fillna(global_mean)

    print(f"Added {len(common_cols)*2} new external features.")
    return tr, te


df_train, df_test = add_external_encoding(df_train, df_test, df_orig)


df_train


X_train = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
X_test = df_test.drop(['id'], axis=1)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")


# # Separate target
# y_train = df_train['diagnosed_diabetes'].values
# X_train = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
# X_test = df_test.drop(['id'], axis=1)

# print(f"Train shape: {X_train.shape}")
# print(f"Test shape: {X_test.shape}")


# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]


print(f"\nCategorical columns: {categorical_cols}")
print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"\nNumerical columns: {numerical_cols}")

target = df_train['diagnosed_diabetes']
X_raw = X_train.drop(['diagnosed_diabetes', 'id'], axis=1, errors='ignore')
X_test_raw = X_test.drop(['id'], axis=1, errors='ignore')

# --- A) Create Label Encoded Version (For XGBoost & LightGBM) ---
print("Creating Label Encoded dataset for XGB/LGBM...")
X_le = X_raw.copy()
X_test_le = X_test_raw.copy()

for col in X_le.columns:
    if col in categorical_cols or X_le[col].dtype == 'object':
        le = LabelEncoder()
        # Handle NaNs before encoding to prevent crash
        X_le[col] = X_le[col].fillna("MISSING").astype(str)
        X_test_le[col] = X_test_le[col].fillna("MISSING").astype(str)
        
        # Fit on combined to cover all categories
        full_data = pd.concat([X_le[col], X_test_le[col]])
        le.fit(full_data)
        X_le[col] = le.transform(X_le[col])
        X_test_le[col] = le.transform(X_test_le[col])

# --- B) Create Native Categorical Version (For CatBoost) ---
print("Creating Native Categorical dataset for CatBoost...")
X_cat = X_raw.copy()
X_test_cat = X_test_raw.copy()

# CatBoost likes NaNs in categories to be filled with a string
for col in categorical_cols:
    if col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna("Missing").astype(str)
        X_test_cat[col] = X_test_cat[col].fillna("Missing").astype(str)


## parameter hypertuned with optuna but just for baseline |LightGBM|XGBoost|CatBoost
## I should do it again 

xgb_params ={
    'n_estimators': 2000, # We control this via early stopping
    'early_stopping_rounds': 50,
    'booster': 'gbtree',
    'tree_method': 'hist',     # Fast training
    'eval_metric': 'logloss',
    'learning_rate': 0.010586281318793418, 
    'max_depth': 5, 
    'subsample': 0.9419910623833896, 
    'colsample_bytree': 0.5244058847875112, 
    'min_child_weight': 7, 
    'reg_alpha': 0.00015151084454479046, 
    'reg_lambda': 2.161158791085214e-08, 
    'gamma': 2.240078485583776e-07}


lgb_params = {
    'n_estimators': 2000,
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.04151567000333162, 
    'num_leaves': 93, 'max_depth': 3, 
    'min_child_samples': 97, 
    'subsample': 0.8336810469662667, 
    'colsample_bytree': 0.5021699121748862, 
    'reg_alpha': 0.015640727219830758, 
    'reg_lambda': 1.374990603296636e-06
}



cat_params = {'iterations': 2000,
    'eval_metric': 'AUC',
    'verbose': 0,
    'task_type': 'CPU', # Change to 'GPU' if available
    'cat_features': [c for c in categorical_cols if c in X_cat.columns],
 'learning_rate': 0.08141363864155182, 
 'depth': 4, 
 'l2_leaf_reg': 2.721242066354407, 
 'random_strength': 0.3197413721687479, 
 'subsample': 0.8585190651619243}


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# OOF (Out of Fold) Predictions for CV scoring
oof_xgb = np.zeros(len(X_le))
oof_lgb = np.zeros(len(X_le))
oof_cat = np.zeros(len(X_cat))

# Test Predictions
pred_xgb = np.zeros(len(X_test_le))
pred_lgb = np.zeros(len(X_test_le))
pred_cat = np.zeros(len(X_test_cat))

print(f"\n{'='*20} Starting Cross-Validation {'='*20}")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_le, target)):
    
    # --- XGBoost & LightGBM (Use Label Encoded Data) ---
    X_tr_le, X_val_le = X_le.iloc[train_idx], X_le.iloc[val_idx]
    y_tr, y_val = target.iloc[train_idx], target.iloc[val_idx]
    
    # XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr_le, y_tr, eval_set=[(X_val_le, y_val)], verbose=500)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val_le)[:, 1]
    pred_xgb += model_xgb.predict_proba(X_test_le)[:, 1] / 5
    
    # LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_tr_le, y_tr, eval_set=[(X_val_le, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val_le)[:, 1]
    pred_lgb += model_lgb.predict_proba(X_test_le)[:, 1] / 5
    
    # --- CatBoost (Use Native Data) ---
    X_tr_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), early_stopping_rounds=50)
    oof_cat[val_idx] = model_cat.predict_proba(X_val_cat)[:, 1]
    pred_cat += model_cat.predict_proba(X_test_cat)[:, 1] / 5
    
    print(f"Fold {fold+1} done.")





from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# --- BLENDING STRATEGY 1: SCIPY OPTIMIZER (Gradient Descent) ---
print(f"\n{'='*20} STRATEGY 1: SCIPY MINIMIZE {'='*20}")

def minimize_auc(weights):
    # Normalize weights so they sum to 1.0
    w = np.abs(weights)
    w = w / w.sum()
    
    # Create the blended OOF prediction (3 Models)
    final_oof = (w[0] * oof_xgb + 
                 w[1] * oof_lgb + 
                 w[2] * oof_cat)
    
    return -roc_auc_score(target, final_oof)

# Starting guess: Equal weights
initial_weights = [0.33, 0.33, 0.33]
bounds = [(0, 1)] * 3

# Run optimization
result = minimize(minimize_auc, initial_weights, bounds=bounds, method='SLSQP')

# Extract results
weights_scipy = np.abs(result.x) / np.abs(result.x).sum()
auc_scipy = -result.fun
pred_scipy = (weights_scipy[0] * pred_xgb + 
              weights_scipy[1] * pred_lgb + 
              weights_scipy[2] * pred_cat)

print(f"Scipy Weights: XGB: {weights_scipy[0]:.4f}, LGB: {weights_scipy[1]:.4f}, CAT: {weights_scipy[2]:.4f}")
print(f"Scipy Blend AUC: {auc_scipy:.6f}")


# --- 4. BLENDING STRATEGY 2: RANDOM SEARCH (Brute Force) ---
print(f"\n{'='*20} STRATEGY 2: RANDOM SEARCH {'='*20}")

# Lists for iteration
oof_list = [oof_xgb, oof_lgb, oof_cat]
pred_list = [pred_xgb, pred_lgb, pred_cat]

best_auc_random = 0
best_weights_random = [0.33, 0.33, 0.33]

# Try 5000 random combinations
np.random.seed(42) # Reproducibility
for i in range(5000):
    # Generate 3 random weights that sum to 1
    weights = np.random.dirichlet(np.ones(3), size=1)[0]
    
    # Fast weighted average
    current_oof = (weights[0] * oof_xgb + 
                   weights[1] * oof_lgb + 
                   weights[2] * oof_cat)
    
    current_auc = roc_auc_score(target, current_oof)
    
    if current_auc > best_auc_random:
        best_auc_random = current_auc
        best_weights_random = weights

pred_random = (best_weights_random[0] * pred_xgb + 
               best_weights_random[1] * pred_lgb + 
               best_weights_random[2] * pred_cat)

print(f"Random Weights: XGB: {best_weights_random[0]:.4f}, LGB: {best_weights_random[1]:.4f}, CAT: {best_weights_random[2]:.4f}")
print(f"Random Blend AUC: {best_auc_random:.6f}")
  

# --- 5. FINAL COMPARISON & SELECTION ---

# Single Model Scores
auc_xgb = roc_auc_score(target, oof_xgb)
auc_lgb = roc_auc_score(target, oof_lgb)
auc_cat = roc_auc_score(target, oof_cat)

# Simple Average
oof_simple = (oof_xgb + oof_lgb + oof_cat) / 3
auc_simple = roc_auc_score(target, oof_simple)

# Rank Blend
oof_rank = (pd.Series(oof_xgb).rank(pct=True) + 
            pd.Series(oof_lgb).rank(pct=True) + 
            pd.Series(oof_cat).rank(pct=True)) / 3
auc_rank = roc_auc_score(target, oof_rank)

# Store results
results = {
    'XGBoost': {'score': auc_xgb, 'preds': pred_xgb},
    'LightGBM': {'score': auc_lgb, 'preds': pred_lgb},
    'CatBoost': {'score': auc_cat, 'preds': pred_cat},
    'Simple_Avg': {'score': auc_simple, 'preds': (pred_xgb + pred_lgb + pred_cat) / 3},
    'Rank_Blend': {'score': auc_rank, 
                   'preds': (pd.Series(pred_xgb).rank(pct=True) + 
                             pd.Series(pred_lgb).rank(pct=True) + 
                             pd.Series(pred_cat).rank(pct=True)) / 3},
    'Scipy_Blend': {'score': auc_scipy, 'preds': pred_scipy},
    'Random_Blend': {'score': best_auc_random, 'preds': pred_random}
}

# Print Summary
print(f"\n{'='*10} FINAL SCOREBOARD {'='*10}")
print(f"XGBoost:       {auc_xgb:.6f}")
print(f"LightGBM:      {auc_lgb:.6f}")
print(f"CatBoost:      {auc_cat:.6f}")
print(f"Simple Avg:    {auc_simple:.6f}")
print(f"Rank Blend:    {auc_rank:.6f}")
print(f"Scipy Blend:   {auc_scipy:.6f}")
print(f"Random Blend:  {best_auc_random:.6f}")
print(f"{'='*40}")


# --- VISUALIZATION ---

# Prepare Data
model_names = ['XGB', 'LGBM', 'CatBoost', 'Simple', 'Rank', 'Scipy', 'Random']
model_scores = [auc_xgb, auc_lgb, auc_cat, auc_simple, auc_rank, auc_scipy, best_auc_random]

# Create DataFrame
df_results = pd.DataFrame({'Model': model_names, 'AUC': model_scores})
df_results = df_results.sort_values(by='AUC', ascending=True)

# Plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Highlight winner
colors = ['#bdc3c7' if x < df_results['AUC'].max() else '#2ecc71' for x in df_results['AUC']]

ax = sns.barplot(x='AUC', y='Model', data=df_results, palette=colors)

for i, v in enumerate(df_results['AUC']):
    ax.text(v, i, f' {v:.6f}', va='center', fontweight='bold', color='#2c3e50')

# Zoom x-axis
min_score = min(model_scores)
max_score = max(model_scores)
margin = (max_score - min_score) * 0.2
plt.xlim(min_score - margin, max_score + margin)

plt.title('Leaderboard: 3 models , Scipy , Random Search Blend', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('ROC AUC Score', fontsize=12)
plt.ylabel('')
plt.tight_layout()
plt.show()


# --- SUBMISSION ---
# Automatically pick the best one
best_method = max(results, key=lambda x: results[x]['score'])
best_score = results[best_method]['score']
final_predictions = results[best_method]['preds']

print(f"\n✅ Best Strategy: {best_method} with CV: {best_score:.6f}")
print("Generating submission file...")

submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': np.clip(final_predictions, 0.001, 0.999) # Clip is good practice
})

submission.to_csv('submission.csv', index=False)
print("Saved to 'submission.csv'")



# corr_df = pd.DataFrame({
#     'XGB': oof_xgb, 'LGB': oof_lgb, 
#     'CAT': oof_cat, 'NN': oof_nn
# }).corr()
# print(corr_df)




