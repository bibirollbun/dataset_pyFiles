import pandas as pd
import numpy as np
import os
import time
import warnings
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

# --- Suppress Warnings for Cleaner Output ---
# This new line specifically quiets the logs from the Optuna library.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# This specifically targets the XGBoost UserWarning you mentioned.
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')


# --- 1. Data Loading and Preparation ---
print("--- Loading and Preparing Data ---")
# Load competition data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# --- Data Augmentation ---
try:
    original = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
    original.rename(columns={
        'Temparature': 'Temparature', 'Humidity ':'Humidity', 'Moisture':'Moisture', 
        'Soil Type':'Soil Type', 'Crop Type':'Crop Type', 'Nitrogen':'Nitrogen', 
        'Potassium':'Potassium', 'Phosphorous':'Phosphorous', 'Fertilizer Name':'Fertilizer Name'
    }, inplace=True)
    train = pd.concat([train, original], ignore_index=True)
    train.drop_duplicates(inplace=True)
    print("Successfully loaded and concatenated original dataset.")
except FileNotFoundError:
    print("Original dataset not found. Proceeding with competition data only.")

# --- ADDING PROVEN FEATURE ---
print("Creating 'Soil_Crop_Interaction' feature.")
for df in [train, test]:
    df['Soil_Crop_Interaction'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)


# Drop ID columns
if 'id' in train.columns:
    train.drop('id', axis=1, inplace=True)
test_ids = test['id']
if 'id' in test.columns:
    test.drop('id', axis=1, inplace=True)

# --- Categorical Encoding (Label Encoding for XGBoost) ---
cat_cols = ['Soil Type', 'Crop Type', 'Soil_Crop_Interaction']
for col in cat_cols:
    le = LabelEncoder()
    combined_data = pd.concat([train[col], test[col]]).astype(str)
    le.fit(combined_data)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    
target_encoder = LabelEncoder()
train["Fertilizer Name"] = target_encoder.fit_transform(train["Fertilizer Name"])

X = train.drop(columns=["Fertilizer Name"])
y = train["Fertilizer Name"]
X = X.astype({col: 'category' for col in cat_cols})
test = test.astype({col: 'category' for col in cat_cols})

print("Data preparation complete.")

# --- 2. Optuna Hyperparameter Optimization ---
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def mapk(actual, predicted, k=3):
    actual_wrapped = [[a] for a in actual]
    return np.mean([
        len(set(p[:k]) & set(a)) / min(len(a), k) for a, p in zip(actual_wrapped, predicted)
    ])

def objective(trial):
    params = {
        "objective": "multi:softprob",
        "num_class": len(np.unique(y)),
        "eval_metric": "mlogloss",
        "device": "cuda",
        "tree_method": "hist",
        "enable_categorical": True,
        "early_stopping_rounds": 50, # CORRECT: Moved parameter here
        "n_estimators": trial.suggest_int("n_estimators", 500, 4000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5),
    }

    model = xgb.XGBClassifier(**params)
    # CORRECT: early_stopping_rounds removed from .fit()
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    probs = model.predict_proba(X_valid)
    top3_preds = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    
    return mapk(y_valid.values, top3_preds)

print("\n--- Starting Optuna Optimization ---")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25, timeout=7200)

print("Optimization finished.")
print("Best trial:")
best_params = study.best_params
print(f"  Value (MAP@3): {study.best_value}")
print("  Params: ")
for key, value in best_params.items():
    print(f"    {key}: {value}")

# --- 3. Final Model Training with Best Parameters ---
print("\n--- Training Final Model with Best Parameters using 5-Fold CV ---")
final_params = best_params
# Make sure these essential parameters are in the final dictionary
final_params.update({
    "objective": "multi:softprob",
    "num_class": len(np.unique(y)),
    "eval_metric": "mlogloss",
    "device": "cuda",
    "tree_method": "hist",
    "enable_categorical": True,
    # CORRECT: Also moved parameter here for the final training
    "early_stopping_rounds": 50,
})

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
test_predictions = np.zeros((len(test), len(np.unique(y))))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"{'='*10} Fold {fold} {'='*10}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = XGBClassifier(**final_params)
    # CORRECT: early_stopping_rounds removed from .fit()
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
    
    test_predictions += model.predict_proba(test) / FOLDS

# --- 4. Create Submission File ---
print("\n--- Creating Submission File ---")
top_3_preds_indices = np.argsort(test_predictions, axis=1)[:, -3:][:, ::-1]
top_3_preds_labels = target_encoder.inverse_transform(top_3_preds_indices.ravel()).reshape(top_3_preds_indices.shape)

submission['Fertilizer Name'] = [' '.join(row) for row in top_3_preds_labels]
submission.to_csv('submission.csv', index=False)

print("✅ Submission file 'submission.csv' created successfully!")
display(submission.head())



