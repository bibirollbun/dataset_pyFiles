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


# Paths
train_path = '/kaggle/input/playground-series-s5e6/train.csv'

# Load the data
df_train = pd.read_csv(train_path)

# Quick preview
print("Train shape:", df_train.shape)

df_train.head()


# Drop the 'id' column
df_train.drop(columns='id', inplace=True)

# Check updated shapes
print("âœ… Train shape:", df_train.shape)


# Replace spaces with underscores in column names
df_train.columns = df_train.columns.str.replace(' ', '_')

# Re-identify numerical and categorical columns
num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Print updated results
print("ğŸ“Š Numerical columns:")
print(num_cols)

print("\nğŸ”¤ Categorical columns:")
print(cat_cols)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns (excluding target)
cat_cols = ['Soil_Type', 'Crop_Type']

# Encode features
feature_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    feature_encoders[col] = le

# Encode target
label_encoder = LabelEncoder()
df_train['Fertilizer_Name'] = label_encoder.fit_transform(df_train['Fertilizer_Name'])

target_col='Fertilizer_Name'
# âœ… Prepare features (X) and target (y)
X = df_train.drop(target_col, axis=1)
y = df_train[target_col]

# âœ… Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# âœ… Print shapes
print("âœ… Data split complete:")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape:   {y_val.shape}")


# âœ… MAP@3 function
def mapk(y_true, y_pred_probs, k=3):
    topk = np.argsort(y_pred_probs, axis=1)[:, -k:][:, ::-1]
    score = 0.0
    for i, true_label in enumerate(y_true):
        if true_label in topk[i]:
            rank = np.where(topk[i] == true_label)[0][0]
            score += 1.0 / (rank + 1)
    return score / len(y_true)




import xgboost as xgb

# Parameters from the best Optuna trial
xgb_params = {
    'max_depth': 9,
    'learning_rate': 0.0596454809828514,
    'n_estimators': 209,
    'gamma': 0.8556650746626432,
    'subsample': 0.6455815589842786,
    'colsample_bytree': 0.5886047456705095,
    'reg_alpha': 0.5825789845674176,
    'reg_lambda': 0.04084905094136837,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'tree_method': 'gpu_hist',  # Use 'hist' if you donâ€™t have a GPU
    'verbosity': 0,
    'num_class': len(y_train.unique()),  # Important: set number of classes correctly
    'use_label_encoder': False
}

# Fit the model
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train)

# Predict probabilities for the validation set
y_pred_probs = xgb_model.predict_proba(X_val)

# Compute MAP@3
map3_score = mapk(y_val, y_pred_probs, k=3)
print(f"ğŸ“Š MAP@3 on validation set xgb: {map3_score:.4f}")



from catboost import CatBoostClassifier

# âœ… Define best CatBoost parameters (with GPU)
cat_params = {
    'iterations': 1000,
    'learning_rate': 0.030531370968286817,
    'depth': 7,
    'l2_leaf_reg': 0.18485172766501892,
    'random_strength': 0.47173616768677434,
    'bagging_temperature': 0.3743755561933306,
    'border_count': 179,
    'grow_policy': 'Depthwise',
    'auto_class_weights': 'SqrtBalanced',
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'verbose':0,
    'random_seed': 42,
    'task_type': 'GPU',     # âœ… GPU support
    'devices': '0'          # GPU device index (default is '0')
}

# âœ… Train model
cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

# âœ… Predict class probabilities
y_pred_probs = cat_model.predict_proba(X_val)

# âœ… Evaluate
map3_score = mapk(y_val.values, y_pred_probs, k=3)
print(f"ğŸ“Š MAP@3 on validation set (CatBoost GPU): {map3_score:.4f}")



import lightgbm as lgb# âœ… Best hyperparameters
# âœ… LightGBM best parameters (with verbosity and row-wise optimization)
lgb_params = {
    'learning_rate': 0.18356898403249025,
    'num_leaves': 61,
    'max_depth': 5,
    'min_child_samples': 73,
    'subsample': 0.5963113528610424,
    'colsample_bytree': 0.6609825784097408,
    'reg_alpha': 0.2883943429846648,
    'reg_lambda': 0.44167045605849575,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'verbosity': -1,                 # âœ… Suppress info and warnings
    'force_row_wise': True,         # âœ… Use row-wise for speed
    'num_class': len(np.unique(y_train)),  # Make sure to set this correctly
    'random_state': 42
}

# âœ… Train LightGBM model
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# âœ… Predict probabilities
y_pred_probs = lgb_model.predict_proba(X_val)

# âœ… Evaluate MAP@3
map3_score = mapk(y_val.values, y_pred_probs, k=3)
print(f"ğŸ“Š MAP@3 on validation set (LightGBM): {map3_score:.4f}")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from xgboost import XGBClassifier

# Holdout split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)
    map3_train = mapk(y_train, train_probs, k=3)
    map3_val = mapk(y_val, val_probs, k=3)
    gap = map3_train - map3_val
    return train_probs, val_probs, map3_train, map3_val, gap

# Train base models
xgb_model = XGBClassifier(**xgb_params)
_, xgb_val_probs, map3_xgb_train, map3_xgb_val, gap_xgb = evaluate_model(xgb_model, X_train, y_train, X_val, y_val)

cat_model = CatBoostClassifier(**cat_params)
_, cat_val_probs, map3_cat_train, map3_cat_val, gap_cat = evaluate_model(cat_model, X_train, y_train, X_val, y_val)

lgb_model = lgb.LGBMClassifier(**lgb_params)
_, lgb_val_probs, map3_lgb_train, map3_lgb_val, gap_lgb = evaluate_model(lgb_model, X_train, y_train, X_val, y_val)

print(f"ğŸ“¦ XGBoost    | Train MAP@3: {map3_xgb_train:.4f} | Val MAP@3: {map3_xgb_val:.4f} | Gap: {gap_xgb:.4f}")
print(f"ğŸ�± CatBoost   | Train MAP@3: {map3_cat_train:.4f} | Val MAP@3: {map3_cat_val:.4f} | Gap: {gap_cat:.4f}")
print(f"ğŸ’¡ LightGBM   | Train MAP@3: {map3_lgb_train:.4f} | Val MAP@3: {map3_lgb_val:.4f} | Gap: {gap_lgb:.4f}")

# -------------------------------
# Step 4: Average Ensemble (Equal)
# -------------------------------
avg_probs = (xgb_val_probs + cat_val_probs + lgb_val_probs) / 3.0
map3_avg = mapk(y_val, avg_probs, k=3)
print(f"ğŸ“Š Equal Averaging Ensemble MAP@3: {map3_avg:.4f}")

# -----------------------------------------
# Step 5: Weighted Average (Inverse Gap)
# -----------------------------------------
eps = 1e-6
gaps = np.array([gap_xgb, gap_cat, gap_lgb])
inv_gaps = 1 / (gaps + eps)
weights = inv_gaps / inv_gaps.sum()

weighted_avg_probs = (
    weights[0] * xgb_val_probs +
    weights[1] * cat_val_probs +
    weights[2] * lgb_val_probs
)

map3_weighted = mapk(y_val, weighted_avg_probs, k=3)
print(f"ğŸ“Š Weighted Average Ensemble MAP@3: {map3_weighted:.4f}")
print(f"ğŸ“� Weights: XGB={weights[0]:.3f}, Cat={weights[1]:.3f}, LGB={weights[2]:.3f}")

# --------------------------------------------------
# Step 6: Ridge Regression Meta-Ensemble (Stacking)
# --------------------------------------------------
# Use validation probs as features for Ridge
val_features = np.hstack((xgb_val_probs, cat_val_probs, lgb_val_probs))

# Ridge ClassifierCV (built-in CV to select alpha)
ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7), cv=5)
ridge.fit(val_features, y_val)
val_preds_ridge = ridge.decision_function(val_features)

# Softmax to convert to probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

ridge_probs = softmax(val_preds_ridge)
map3_ridge = mapk(y_val, ridge_probs, k=3)
print(f"ğŸ”— Ridge Meta-Ensemble MAP@3: {map3_ridge:.4f}")

