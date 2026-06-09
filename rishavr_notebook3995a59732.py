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


# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

# Verify it loaded by looking at the shape
print("Train shape:", train.shape)
print("Test shape:", test.shape)


# Check for missing values in the training set
print(train.isnull().sum())


train.info()


from sklearn.preprocessing import OrdinalEncoder

# 1. Identify the text columns automatically
cat_cols = train.select_dtypes(include=['object']).columns
print("Encoding columns:", cat_cols)

# 2. Setup the Encoder (It learns the mapping from the train set)
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 3. Transform the columns into numbers
train[cat_cols] = encoder.fit_transform(train[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

# 4. Check the results
print(train[cat_cols].head())


# 1. Define X (Features) - Drop the target and the ID
# axis=1 means we are dropping 'columns', not rows
X = train.drop(['diagnosed_diabetes', 'id'], axis=1)

# 2. Define y (Target) - Just the answer column
y = train['diagnosed_diabetes']

# 3. Check the results
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import numpy as np

# 1. Prepare the Test Data (Drop ID so it matches X)
# We save the IDs first because we need them for the final file!
test_ids = test['id']
X_test = test.drop('id', axis=1)

# 2. Setup Cross-Validation (5 Folds)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Create arrays to store our predictions
oof_preds = np.zeros(len(X))     # "Out Of Fold" predictions (for checking accuracy)
test_preds = np.zeros(len(X_test)) # Predictions for the final submission
scores = []

print("Starting training... (This might take a minute)")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # Split data into Train and Validation for this fold
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Initialize the Model (XGBoost)
    model = XGBClassifier(
        n_estimators=500,        # Number of "trees" in the forest
        learning_rate=0.05,      # How fast it learns
        max_depth=6,             # How complex each tree can be
        eval_metric='auc',       # The metric we care about
        early_stopping_rounds=50,# Stop if it stops improving
        n_jobs=-1,               # Use all CPU cores
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Predict on Validation set (to see how well we did)
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    # Predict on the real Test set (for submission)
    test_preds += model.predict_proba(X_test)[:, 1] / 5 # Average over 5 folds
    
    # Calculate score
    score = roc_auc_score(y_val, val_preds)
    scores.append(score)
    print(f"Fold {fold+1} AUC: {score:.5f}")

print(f"\nAverage CV AUC: {np.mean(scores):.5f}")


# 1. Create a DataFrame with the ID and our Predictions
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_preds
})

# 2. Save it to a CSV file (index=False removes the row numbers)
submission.to_csv('submission.csv', index=False)

print("Submission file saved! check the output section.")
print(submission.head())


# 1. Create the new "Super Clue" in both datasets
train['Age_x_BMI'] = train['age'] * train['bmi']
test['Age_x_BMI'] = test['age'] * test['bmi']

# 2. Re-define X to include this new column
# (We have to do this so the model knows to look at the new column)
X = train.drop(['diagnosed_diabetes', 'id'], axis=1)
X_test = test.drop('id', axis=1)

print("New feature created!")
print("New shape of X:", X.shape)


# 1. Clean up: Remove the "bad" feature if it exists
if 'Age_x_BMI' in train.columns:
    train = train.drop('Age_x_BMI', axis=1)
    test = test.drop('Age_x_BMI', axis=1)

# 2. Create the "Medical Feature" (Pulse Pressure)
train['Pulse_Pressure'] = train['systolic_bp'] - train['diastolic_bp']
test['Pulse_Pressure'] = test['systolic_bp'] - test['diastolic_bp']

# 3. Define X again
X = train.drop(['diagnosed_diabetes', 'id'], axis=1)
X_test = test.drop('id', axis=1)

print("Training with Pulse_Pressure...")

# 4. Retrain
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        eval_metric='auc',
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    val_preds = model.predict_proba(X_val)[:, 1]
    scores.append(roc_auc_score(y_val, val_preds))
    test_preds += model.predict_proba(X_test)[:, 1] / 5

print(f"Baseline Score: 0.72448")
print(f"New Score:      {np.mean(scores):.5f}")


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# 1. Clean up: Remove the previous failed feature
if 'Pulse_Pressure' in train.columns:
    train = train.drop('Pulse_Pressure', axis=1)
    test = test.drop('Pulse_Pressure', axis=1)

# 2. Create the "Binning" Feature (Age Groups)
# We cut the ages into bins: 0-19, 20-29, 30-39, etc.
# labels=False gives them numbers (0, 1, 2...) directly
train['Age_Group'] = pd.cut(train['age'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 100], labels=False)
test['Age_Group'] = pd.cut(test['age'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 100], labels=False)

# 3. Define X again
X = train.drop(['diagnosed_diabetes', 'id'], axis=1)
X_test = test.drop('id', axis=1)

print("Training with Age_Group...")

# 4. Retrain
oof_preds = np.zeros(len(X))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        eval_metric='auc',
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    val_preds = model.predict_proba(X_val)[:, 1]
    scores.append(roc_auc_score(y_val, val_preds))

print(f"Baseline Score: 0.72448")
print(f"New Score:      {np.mean(scores):.5f}")


from lightgbm import LGBMClassifier

# 1. Setup arrays to store predictions
lgbm_oof = np.zeros(len(X))
lgbm_test_preds = np.zeros(len(X_test))
lgbm_scores = []

print("Training LightGBM...")

# 2. Train on the same 5 Folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Initialize LightGBM
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,        # Controls tree complexity
        random_state=42,
        n_jobs=-1,
        verbose=-1            # Keeps it quiet
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[] # older versions might need early stopping here, but let's keep it simple
    )
    
    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    lgbm_oof[val_idx] = val_preds
    lgbm_scores.append(roc_auc_score(y_val, val_preds))
    
    # Add to test predictions
    lgbm_test_preds += model.predict_proba(X_test)[:, 1] / 5

print(f"LightGBM Average CV AUC: {np.mean(lgbm_scores):.5f}")


# 1. Average the predictions (50% XGBoost + 50% LightGBM)
ensemble_preds = (test_preds + lgbm_test_preds) / 2

# 2. Create the final submission file
submission['diagnosed_diabetes'] = ensemble_preds
submission.to_csv('submission_ensemble.csv', index=False)

print("Ensemble submission saved successfully!")
print(submission.head())


# 1. Load your two different submission files
file_1 = pd.read_csv('submission.csv')          # The original XGBoost
file_2 = pd.read_csv('submission_ensemble.csv') # The Blend (XGB + LightGBM)

# 2. Check if the predictions are identical
are_same = file_1['diagnosed_diabetes'].equals(file_2['diagnosed_diabetes'])

print(f"Are the files exactly the same? {are_same}")

if not are_same:
    # If they are different, print the average difference
    diff = np.mean(np.abs(file_1['diagnosed_diabetes'] - file_2['diagnosed_diabetes']))
    print(f"Average difference per patient: {diff:.5f}")
    
    # Show the first 5 rows side-by-side
    comparison = pd.DataFrame({
        'Original': file_1['diagnosed_diabetes'],
        'Ensemble': file_2['diagnosed_diabetes']
    })
    print("\nFirst 5 rows comparison:")
    print(comparison.head())


# --- STEP 1: RELOAD & RESET ---
# We reload to ensure we are starting fresh with a clean slate
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# --- STEP 2: ADVANCED MEDICAL FEATURE ENGINEERING ---

# 1. Mean Arterial Pressure (MAP) - A critical medical vital sign
train['MAP'] = (train['systolic_bp'] + (2 * train['diastolic_bp'])) / 3
test['MAP'] = (test['systolic_bp'] + (2 * test['diastolic_bp'])) / 3

# 2. Cholesterol Ratios (The "Atherogenic Index")
# (We add 1e-5 to avoid dividing by zero, just in case)
train['Chol_Ratio'] = train['ldl_cholesterol'] / (train['hdl_cholesterol'] + 1e-5)
test['Chol_Ratio'] = test['ldl_cholesterol'] / (test['hdl_cholesterol'] + 1e-5)

# 3. Visceral Fat Proxy (Interaction of BMI and Waist)
train['Visceral_Fat'] = train['bmi'] * train['waist_to_hip_ratio']
test['Visceral_Fat'] = test['bmi'] * test['waist_to_hip_ratio']

# 4. Log Transform Skewed Data (Triglycerides often have extreme outliers)
train['Log_Triglycerides'] = np.log1p(train['triglycerides'])
test['Log_Triglycerides'] = np.log1p(test['triglycerides'])

print("Grandmaster Features Created.")

# --- STEP 3: PREPARE DATA ---
# Re-encode and define X/y
from sklearn.preprocessing import OrdinalEncoder
cat_cols = train.select_dtypes(include=['object']).columns
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train[cat_cols] = encoder.fit_transform(train[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

X = train.drop(['diagnosed_diabetes', 'id'], axis=1)
y = train['diagnosed_diabetes']
test_ids = test['id']
X_test = test.drop('id', axis=1)

print(f"New Feature Count: {X.shape[1]}")


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier # The new challenger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # Increased to 10 Folds for precision

# Arrays to store predictions
xgb_preds = np.zeros(len(X_test))
lgbm_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

print("--- Starting The Trinity Training ---")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 1. XGBoost
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=8, 
                        subsample=0.7, colsample_bytree=0.7, n_jobs=-1, random_state=42, early_stopping_rounds=50)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds += xgb.predict_proba(X_test)[:, 1] / 10
    
    # 2. LightGBM
    lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=60, 
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[])
    lgbm_preds += lgbm.predict_proba(X_test)[:, 1] / 10
    
    # 3. CatBoost (The Secret Weapon)
    cat = CatBoostClassifier(n_estimators=1000, learning_rate=0.03, depth=6, 
                             verbose=0, random_state=42, allow_writing_files=False)
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_preds += cat.predict_proba(X_test)[:, 1] / 10
    
    print(f"Fold {fold+1}/10 Complete.")

# --- BLENDING ---
# We give slightly more weight to CatBoost as it's often more stable on medical data
final_preds = (0.34 * xgb_preds) + (0.33 * lgbm_preds) + (0.33 * cat_preds)

submission = pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': final_preds})
submission.to_csv('submission_trinity.csv', index=False)

print("Trinity Model Submission Saved!")




