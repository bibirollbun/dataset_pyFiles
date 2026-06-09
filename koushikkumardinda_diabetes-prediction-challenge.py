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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# --- 1. Configuration & Setup ---
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120 
pd.set_option('display.max_columns', None)

print(">>> Loading Data...")
# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Drop ID column (non-predictive)
train_df = train_df.drop('id', axis=1)
test_ids = test_df['id'] # Save for submission
test_df = test_df.drop('id', axis=1)

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape:  {test_df.shape}")

# --- 2. Feature Identification ---
target_col = 'diagnosed_diabetes'

# Identify columns automatically
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove target from numerical list if present
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

print(f"\nCategorical Features: {categorical_cols}")
print(f"Numerical Features:   {numerical_cols}")

# --- 3. Target Distribution Analysis ---
plt.figure(figsize=(6, 4))
ax = sns.countplot(x=target_col, data=train_df, palette='viridis')
plt.title('Target Distribution: Diagnosed Diabetes', fontweight='bold')
plt.xlabel('Diagnosis (0=No, 1=Yes)')

# Add percentage labels
total = len(train_df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')
plt.show()

# --- 4. Categorical Feature Analysis (Bar Charts) ---
if categorical_cols:
    print("\n>>> Analyzing Categorical Features...")
    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        # Plot counts grouped by Target
        sns.countplot(x=col, hue=target_col, data=train_df, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{col} Distribution by Target')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(title='Diabetes')
    
    # Remove empty subplots
    for i in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()

# --- 5. Numerical Feature Drift (Train vs Test) ---
# Check if Test data looks different from Train data
print("\n>>> Analyzing Numerical Drift (Train vs Test)...")
n_cols = 3
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.kdeplot(train_df[col], label='Train', fill=True, alpha=0.3, ax=axes[i], color='blue')
    sns.kdeplot(test_df[col], label='Test', fill=True, alpha=0.3, ax=axes[i], color='orange')
    axes[i].set_title(f'{col} Distribution')
    axes[i].legend()

for i in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[i])
    
plt.tight_layout()
plt.show()

# --- 6. Numerical Features vs Target (Box Plots) ---
print("\n>>> Analyzing Numerical Features Separation...")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.boxplot(x=target_col, y=col, data=train_df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} vs Target')

for i in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# --- 7. Correlation Matrix (Fixed for Categoricals) ---
print("\n>>> Generatng Correlation Matrix...")

# Create a copy for encoding so we don't affect original data
train_encoded = train_df.copy()
le = LabelEncoder()

# Encode Categorical columns to numbers (0, 1, 2...)
for col in categorical_cols:
    # Handle potential NaNs in categorical columns just in case
    train_encoded[col] = train_encoded[col].fillna("Missing")
    train_encoded[col] = le.fit_transform(train_encoded[col].astype(str))

# Calculate Correlation
corr_matrix = train_encoded.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Heatmap (Encoded)', fontsize=16)
plt.show()

# --- 8. Missing Values Check ---
print("\n>>> Missing Values Summary:")
missing_train = train_df.isnull().sum()
if missing_train.sum() == 0:
    print("No missing values in Train set.")
else:
    print(missing_train[missing_train > 0])


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

# --- 1. Configuration ---
warnings.filterwarnings('ignore')
FOLDS = 5
SEED = 42

print(">>> Loading Data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

# --- 2. Preprocessing & Encoding ---
# We identify categorical features and encode them into integers (0, 1, 2...)
# XGBoost can handle these well as long as they are numbers.

target_col = 'diagnosed_diabetes'
drop_cols = ['id', target_col]
features = [col for col in train.columns if col not in drop_cols]

# Combine for consistent encoding
all_data = pd.concat([train[features], test[features]], axis=0)
cat_cols = all_data.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Encoding Categorical Features: {cat_cols}")

for col in cat_cols:
    le = LabelEncoder()
    # Fit on all data to ensure no unseen labels in test
    le.fit(all_data[col].astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# --- 3. The Validation Rig (Stratified K-Fold) ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []

print(f"\n>>> Starting XGBoost Training ({FOLDS} Folds)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[target_col])):
    X_train, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
    y_train, y_val = train.iloc[train_idx][target_col], train.iloc[val_idx][target_col]
    
    # XGBoost Parameters (Standard Baseline)
    # We use 'auc' as the metric to align with the competition goal
    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=SEED,
        n_jobs=-1,
        device="cuda" if pd.Series([0]).dtype == "float16" else "cpu" # Auto-detect GPU if avail
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    
    # Predict
    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(test[features])[:, 1]
    
    # Store
    oof_preds[val_idx] = val_pred
    test_preds += test_pred / FOLDS
    
    # Score
    auc = roc_auc_score(y_val, val_pred)
    scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.5f} | Best Iteration: {model.best_iteration}")

# --- 4. Results & Submission ---
overall_auc = roc_auc_score(train[target_col], oof_preds)
print(f"\n>>> Overall OOF AUC: {overall_auc:.5f}")
print(f">>> Average Fold AUC: {np.mean(scores):.5f}")

# Create Submission File
submission['diagnosed_diabetes'] = test_preds
submission.to_csv('submission.csv', index=False)
print(">>> submission.csv created successfully!")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Define the Engineering Function ---
def engineer_features(df):
    df_eng = df.copy()
    
    # A. Domain Knowledge Ratios (Metabolic Proxies)
    # ----------------------------------------------
    
    # 1. Insulin Resistance Proxy (Glucose * Insulin)
    # High Glucose + High Insulin = Body is struggling to process sugar
    if 'Glucose' in df_eng.columns and 'Insulin' in df_eng.columns:
        df_eng['Insulin_Glucose_Ratio'] = df_eng['Insulin'] / (df_eng['Glucose'] + 1)
        df_eng['HOMA_Proxy'] = df_eng['Insulin'] * df_eng['Glucose'] 

    # 2. BMI Context (Weight vs Height/Age interactions)
    if 'BMI' in df_eng.columns:
        if 'Age' in df_eng.columns:
            # Impact of BMI gets worse as you age
            df_eng['BMI_Age_Interaction'] = df_eng['BMI'] * df_eng['Age'] 
        
        if 'SkinThickness' in df_eng.columns:
            # Body Fat Distribution
            df_eng['BMI_SkinThickness_Ratio'] = df_eng['BMI'] / (df_eng['SkinThickness'] + 1)

    # 3. Blood Pressure Context
    if 'BloodPressure' in df_eng.columns and 'Age' in df_eng.columns:
        df_eng['BP_per_Age'] = df_eng['BloodPressure'] / (df_eng['Age'] + 1)

    # B. Mathematical Transformations (Handling Skew)
    # -----------------------------------------------
    # Insulin and DiabetesPedigreeFunction often have "long tails" (outliers).
    # Log-transforming them pulls outliers back in, helping tree models find splits.
    
    skewed_cols = ['Insulin', 'DiabetesPedigreeFunction', 'Age']
    for col in skewed_cols:
        if col in df_eng.columns:
            df_eng[f'Log_{col}'] = np.log1p(df_eng[col])

    # C. Binning (Non-Linear Categories)
    # ----------------------------------
    # Sometimes being "Old" is a risk factor, regardless of exact year.
    if 'Age' in df_eng.columns:
        df_eng['Age_Group'] = pd.cut(df_eng['Age'], bins=[0, 25, 45, 65, 100], labels=[0, 1, 2, 3]).astype(int)

    if 'Glucose' in df_eng.columns:
        # Medical thresholds for Prediabetes/Diabetes
        df_eng['Glucose_Risk'] = pd.cut(df_eng['Glucose'], bins=[-1, 100, 140, 200, 1000], labels=[0, 1, 2, 3]).astype(int)

    return df_eng

# --- 2. Apply to Data ---
print(">>> Applying Feature Engineering...")

# Reload fresh to be safe
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv').drop('id', axis=1)
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv').drop('id', axis=1)

# Apply
train_eng = engineer_features(train)
test_eng = engineer_features(test)

print(f"Original Feature Count: {train.shape[1]}")
print(f"New Feature Count:      {train_eng.shape[1]}")
print(f"New Features Created:   {list(set(train_eng.columns) - set(train.columns))}")

# --- 3. Quick Correlation Check on New Features ---
# Did we create anything useful? Check correlation with Target.
new_cols = list(set(train_eng.columns) - set(train.columns))
target_col = 'diagnosed_diabetes'

# Encode if necessary for correlation
corr_check = train_eng[new_cols + [target_col]].copy()
# Simple encoding for the check
for col in corr_check.columns:
    if corr_check[col].dtype == 'object':
         corr_check[col] =  pd.factorize(corr_check[col])[0]

plt.figure(figsize=(10, 8))
sns.heatmap(corr_check.corr()[[target_col]].sort_values(by=target_col, ascending=False), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation of Engineered Features with Target")
plt.show()


import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

# --- 1. Setup ---
warnings.filterwarnings('ignore')

# Reload Data to ensure a clean slate
print(">>> Loading & Engineering Data...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')

# --- Re-Apply Feature Engineering (From previous step) ---
# (Pasting the logic here to ensure this script runs standalone)
def engineer_features(df):
    df_eng = df.copy()
    if 'Glucose' in df_eng.columns and 'Insulin' in df_eng.columns:
        df_eng['Insulin_Glucose_Ratio'] = df_eng['Insulin'] / (df_eng['Glucose'] + 1)
        df_eng['HOMA_Proxy'] = df_eng['Insulin'] * df_eng['Glucose'] 
    if 'BMI' in df_eng.columns and 'Age' in df_eng.columns:
        df_eng['BMI_Age_Interaction'] = df_eng['BMI'] * df_eng['Age'] 
    if 'BloodPressure' in df_eng.columns and 'Age' in df_eng.columns:
        df_eng['BP_per_Age'] = df_eng['BloodPressure'] / (df_eng['Age'] + 1)
    return df_eng

train_eng = engineer_features(train_df)

# --- 2. THE FIX: Strict Preprocessing for XGBoost ---
# This block fixes the "ValueError: DataFrame.dtypes..." by converting ALL strings to Ints
print(">>> Encoding Categorical Columns...")

# Identify target and drop ID
target_col = 'diagnosed_diabetes'
drop_cols = ['id', target_col]
features = [col for col in train_eng.columns if col not in drop_cols]

# Identify columns that are text/objects (e.g., Gender, Ethnicity)
object_cols = train_eng[features].select_dtypes(include=['object', 'category']).columns.tolist()
print(f"   Encoding: {object_cols}")

# Apply Label Encoding (String -> Int)
le = LabelEncoder()
for col in object_cols:
    # Handle NaNs just in case, convert to string, then encode
    train_eng[col] = train_eng[col].fillna("Missing").astype(str)
    train_eng[col] = le.fit_transform(train_eng[col])

# Final Check: Ensure X contains ONLY numbers
X = train_eng[features]
y = train_eng[target_col]

# Verify no objects remain
if X.select_dtypes(include=['object']).shape[1] > 0:
    raise ValueError("Error: Object columns still exist!")
else:
    print(">>> Data clean. All columns are numeric.")

# --- 3. Define the Objective Function ---
def objective(trial):
    params = {
        'n_estimators': 3000,
        # Optimization Search Space
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        
        # Fixed Boilerplate
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_jobs': -1,
        'random_state': 42,
        'tree_method': 'hist', # 'gpu_hist' if you have GPU enabled
        'device': "cuda" if pd.Series([0]).dtype == "float16" else "cpu"
    }

    # Cross-Validation Loop
    cv_scores = []
    # Using 3 folds for speed during tuning, 5 or 10 for final training
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(cv_scores)

# --- 4. Run Optimization ---
print(">>> Starting Optuna Optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20) 

print("\n" + "="*40)
print(f"BEST AUC: {study.best_value:.5f}")
print("BEST PARAMS:")
for key, value in study.best_params.items():
    print(f"   {key}: {value}")
print("="*40)


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

# --- 1. Configuration ---
warnings.filterwarnings('ignore')
FOLDS = 10  # Grandmasters often use 10 folds for final subs to reduce variance
SEED = 42

# --- REPLACE THESE WITH YOUR OPTUNA BEST PARAMS ---
best_params = {
    'n_estimators': 5000,          # Increased for final training
    'learning_rate': 0.01,         # Decreased for better generalization
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1,         # Adjust if you found class imbalance
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': SEED,
    'tree_method': 'hist',
    'device': "cuda" if pd.Series([0]).dtype == "float16" else "cpu"
}

# --- 2. Load & Engineer Data ---
print(">>> Loading Data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

# Drop ID
train = train.drop('id', axis=1)
test_ids = test['id'] # Keep for file writing
test = test.drop('id', axis=1)

# Feature Engineering Function (Must be identical to Optuna step)
def engineer_features(df):
    df_eng = df.copy()
    if 'Glucose' in df_eng.columns and 'Insulin' in df_eng.columns:
        df_eng['Insulin_Glucose_Ratio'] = df_eng['Insulin'] / (df_eng['Glucose'] + 1)
        df_eng['HOMA_Proxy'] = df_eng['Insulin'] * df_eng['Glucose'] 
    if 'BMI' in df_eng.columns and 'Age' in df_eng.columns:
        df_eng['BMI_Age_Interaction'] = df_eng['BMI'] * df_eng['Age'] 
    if 'BloodPressure' in df_eng.columns and 'Age' in df_eng.columns:
        df_eng['BP_per_Age'] = df_eng['BloodPressure'] / (df_eng['Age'] + 1)
    return df_eng

print(">>> Engineering Features...")
train = engineer_features(train)
test = engineer_features(test)

# --- 3. Encoding (The Fix) ---
print(">>> Encoding Categorical Data...")
target_col = 'diagnosed_diabetes'
features = [col for col in train.columns if col != target_col]

# Combine for consistent encoding
all_data = pd.concat([train[features], test[features]], axis=0)
object_cols = all_data.select_dtypes(include=['object', 'category']).columns.tolist()

for col in object_cols:
    le = LabelEncoder()
    le.fit(all_data[col].astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X = train[features]
y = train[target_col]
X_test = test[features]

# --- 4. Stratified K-Fold Training ---
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []

print(f"\n>>> Starting Training with {FOLDS} Folds...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**best_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    
    # Predict (Probabilities of class 1)
    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred_fold = model.predict_proba(X_test)[:, 1]
    
    # Store OOF and Test predictions
    oof_preds[val_idx] = val_pred
    test_preds += test_pred_fold / FOLDS
    
    auc = roc_auc_score(y_val, val_pred)
    scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.5f}")

# --- 5. Output Results ---
overall_auc = roc_auc_score(y, oof_preds)
print(f"\n>>> Overall CV AUC: {overall_auc:.5f}")
print(f">>> Average Fold AUC: {np.mean(scores):.5f}")

# Save Submission
submission['diagnosed_diabetes'] = test_preds
submission.to_csv('submission.csv', index=False)
print(">>> submission.csv saved!")

# Save OOF Predictions (For Stacking later)
oof_df = pd.DataFrame({'id': range(len(oof_preds)), 'pred': oof_preds, 'target': y})
oof_df.to_csv('xgb_oof.csv', index=False)
print(">>> xgb_oof.csv saved (Keep this for Ensembling!)")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

# --- 1. Configuration ---
warnings.filterwarnings('ignore')
FOLDS = 10
SEED = 42

# --- 2. Load & Engineer Data (Must match XGBoost exactly) ---
print(">>> Loading Data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

def engineer_features(df):
    df_eng = df.copy()
    if 'Glucose' in df_eng.columns and 'Insulin' in df_eng.columns:
        df_eng['Insulin_Glucose_Ratio'] = df_eng['Insulin'] / (df_eng['Glucose'] + 1)
        df_eng['HOMA_Proxy'] = df_eng['Insulin'] * df_eng['Glucose'] 
    if 'BMI' in df_eng.columns and 'Age' in df_eng.columns:
        df_eng['BMI_Age_Interaction'] = df_eng['BMI'] * df_eng['Age'] 
    if 'BloodPressure' in df_eng.columns and 'Age' in df_eng.columns:
        df_eng['BP_per_Age'] = df_eng['BloodPressure'] / (df_eng['Age'] + 1)
    return df_eng

print(">>> Engineering Features...")
train = engineer_features(train)
test = engineer_features(test)

# --- 3. Encoding ---
# LightGBM handles integers well, but we need to track WHICH cols are categorical
target_col = 'diagnosed_diabetes'
features = [col for col in train.columns if col != target_col]

all_data = pd.concat([train[features], test[features]], axis=0)
object_cols = all_data.select_dtypes(include=['object', 'category']).columns.tolist()

for col in object_cols:
    le = LabelEncoder()
    le.fit(all_data[col].astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X = train[features]
y = train[target_col]
X_test = test[features]

# --- 4. LightGBM Training ---
# Grandmaster Tip: LightGBM is faster, so we can use more estimators
lgbm_params = {
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'max_depth': 8,
    'num_leaves': 31, # Important LGBM parameter
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': -1,
    'random_state': SEED,
    'verbose': -1
}

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []

print(f"\n>>> Starting LightGBM Training ({FOLDS} Folds)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # We explicitly tell LightGBM which columns are categorical for better splits
    model = lgb.LGBMClassifier(**lgbm_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        categorical_feature=object_cols, # Native handling!
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / FOLDS
    oof_preds[val_idx] = val_pred
    
    auc = roc_auc_score(y_val, val_pred)
    scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.5f}")

print(f"\n>>> LightGBM Overall AUC: {roc_auc_score(y, oof_preds):.5f}")

# Save results
submission['diagnosed_diabetes'] = test_preds
submission.to_csv('lgbm_submission.csv', index=False)

oof_df = pd.DataFrame({'id': range(len(oof_preds)), 'pred': oof_preds, 'target': y})
oof_df.to_csv('lgbm_oof.csv', index=False)
print(">>> LightGBM files saved successfully!")


# --- ENSEMBLE SCRIPT ---
import pandas as pd

# Load the submissions
xgb_sub = pd.read_csv('submission.csv') # Generated in Step 3
lgbm_sub = pd.read_csv('lgbm_submission.csv') # Generated just now

# Define Weights (Sum must be 1.0)
# Adjust these based on which model had the higher "Overall AUC" in your notebooks
w_xgb = 0.6
w_lgbm = 0.4

print(f"Blending: {w_xgb} * XGBoost + {w_lgbm} * LightGBM")

# Blend
ensemble_pred = (xgb_sub['diagnosed_diabetes'] * w_xgb) + (lgbm_sub['diagnosed_diabetes'] * w_lgbm)

# Create Final File
ensemble_sub = xgb_sub.copy()
ensemble_sub['diagnosed_diabetes'] = ensemble_pred
ensemble_sub.to_csv('ensemble_submission.csv', index=False)

print(">>> ensemble_submission.csv created! Submit this file.")

