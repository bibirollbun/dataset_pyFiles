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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Machine Learning
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Configuration
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)

# --- Load Data (Kaggle Input Paths) ---
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
SUBMISSION_PATH = "/kaggle/input/playground-series-s5e12/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"âœ… Data Loaded.")
print(f"Train Shape: {train.shape}")
print(f"Test Shape:  {test.shape}")


# Calculate correlation matrix
corr = train.corr(numeric_only=True)
target_corr = corr[['diagnosed_diabetes']].sort_values(by='diagnosed_diabetes', ascending=False)

# Plot Heatmap
plt.figure(figsize=(6, 10))
sns.heatmap(target_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation with Diagnosed Diabetes")
plt.show()


# Create 10 bins for visualization
train_viz = train.copy()
train_viz['bmi_bin'] = pd.qcut(train_viz['bmi'], q=10, labels=False)
train_viz['bp_bin'] = pd.qcut(train_viz['systolic_bp'], q=10, labels=False)

# Calculate mean risk per bin
bmi_risk = train_viz.groupby('bmi_bin')['diagnosed_diabetes'].mean()
bp_risk = train_viz.groupby('bp_bin')['diagnosed_diabetes'].mean()

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# BMI Plot
sns.lineplot(x=bmi_risk.index, y=bmi_risk.values, ax=axes[0], marker='o', color='blue')
axes[0].set_title('Diabetes Risk by BMI Decile')
axes[0].set_xlabel('BMI Bin (0=Low, 9=High)')
axes[0].set_ylabel('Diabetes Probability')

# BP Plot
sns.lineplot(x=bp_risk.index, y=bp_risk.values, ax=axes[1], marker='o', color='red')
axes[1].set_title('Diabetes Risk by Systolic BP Decile')
axes[1].set_xlabel('BP Bin (0=Low, 9=High)')
axes[1].set_ylabel('Diabetes Probability')

plt.show()


def create_features(df):
    df = df.copy()
    
    # --- 1. Interactions with Family History ---
    # Helps trees find specific thresholds for high-risk family groups
    df['bmi_x_family'] = df['bmi'] * df['family_history_diabetes']
    df['age_x_family'] = df['age'] * df['family_history_diabetes']
    
    # --- 2. Medical Ratios ---
    # LH Ratio (LDL / HDL) -> Higher is worse
    # Added epsilon (1e-5) to avoid division by zero
    df['lh_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    
    # Obesity vs Activity Ratio
    df['bmi_activity_ratio'] = df['bmi'] / (df['physical_activity_minutes_per_week'] + 1)
    
    return df

# Apply features
train = create_features(train)
test = create_features(test)


# 1. Prepare Bins for Encoding
for df in [train, test]:
    # Age Groups (10-year bins)
    df['age_bin'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 200], labels=False, right=False)
    
    # Quantile Bins (10 bins)
    df['tri_bin'] = pd.qcut(df['triglycerides'], q=10, labels=False)
    df['bmi_q'] = pd.qcut(df['bmi'], q=10, labels=False)
    df['bp_q'] = pd.qcut(df['systolic_bp'], q=10, labels=False)
    
    # Interaction Category (BMI Rank + BP Rank)
    df['bmi_bp_combo'] = df['bmi_q'].astype(str) + '_' + df['bp_q'].astype(str)

# Columns to Target Encode
te_features = ['age_bin', 'tri_bin', 'bmi_bp_combo']
new_te_cols = ['age_group_te', 'triglycerides_te', 'bmi_bp_te']

# Initialize
for col in new_te_cols:
    train[col] = 0
    test[col] = 0

# 2. Perform Encoding
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- A. Encode Train (Out-of-Fold) ---
for tr_idx, val_idx in kf.split(train, train['diagnosed_diabetes']):
    X_tr, X_val = train.iloc[tr_idx], train.iloc[val_idx]
    
    for i, orig_col in enumerate(te_features):
        target_col = new_te_cols[i]
        # Calculate mean on training folds only
        map_dict = X_tr.groupby(orig_col)['diagnosed_diabetes'].mean()
        # Map to validation fold
        train.loc[val_idx, target_col] = X_val[orig_col].map(map_dict)

# --- B. Encode Test (Global) ---
for i, orig_col in enumerate(te_features):
    target_col = new_te_cols[i]
    # Use full training data stats
    global_map = train.groupby(orig_col)['diagnosed_diabetes'].mean()
    test[target_col] = test[orig_col].map(global_map)
    # Fill missing values with global mean
    test[target_col] = test[target_col].fillna(train['diagnosed_diabetes'].mean())

print("âœ… Target Encoding Completed.")


# Drop intermediate columns and ID
drop_cols = [
    'id', 'diagnosed_diabetes',
    'age_bin', 'tri_bin', 'bmi_q', 'bp_q', 'bmi_bp_combo',
    'triglycerides' # Replaced by Target Encoded version
]

# Select features
actual_drop = [c for c in drop_cols if c in train.columns]
features = [c for c in train.columns if c not in actual_drop]

# Label Encoding for strings (Gender, Ethnicity, etc.)
cat_cols = train[features].select_dtypes(include=['object']).columns.tolist()

le = LabelEncoder()
for col in cat_cols:
    # Fit on both datasets to handle all categories
    full_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_data)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

print(f"Final Feature Count: {len(features)}")


y = train['diagnosed_diabetes']
X = train[features]
X_test = test[features]

# Predictions containers
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

# CV Strategy
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


print("\nâš¡ Training LightGBM...")
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'n_estimators': 10000,
    'num_leaves': 31,
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    
    callbacks = [
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(0)
    ]
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=callbacks)
    
    # Predict
    oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
    pred_lgb += model.predict_proba(X_test)[:, 1] / 5
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.5f}")


print("\nğŸ�± Training CatBoost...")
cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.03,
    'iterations': 10000,
    'depth': 6,
    'random_seed': 42,
    'verbose': 0,
    'allow_writing_files': False
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(**cat_params)
    
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
    
    oof_cat[val_idx] = model.predict_proba(X_val)[:, 1]
    pred_cat += model.predict_proba(X_test)[:, 1] / 5
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, oof_cat[val_idx]):.5f}")


# --- Final Scores ---
auc_lgb = roc_auc_score(y, oof_lgb)
auc_cat = roc_auc_score(y, oof_cat)

# Simple Blending (50/50)
final_oof = 0.5 * oof_lgb + 0.5 * oof_cat
auc_ensemble = roc_auc_score(y, final_oof)

print(f"\n======== RESULTS ========")
print(f"LightGBM AUC : {auc_lgb:.5f}")
print(f"CatBoost AUC : {auc_cat:.5f}")
print(f"Ensemble AUC : {auc_ensemble:.5f} (+{auc_ensemble - max(auc_lgb, auc_cat):.5f})")
print(f"=========================")

# --- Create Submission ---
final_test_pred = 0.5 * pred_lgb + 0.5 * pred_cat

submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': final_test_pred
})

submission.to_csv('submission_ensemble.csv', index=False)
print("âœ… submission_ensemble.csv created successfully!")

