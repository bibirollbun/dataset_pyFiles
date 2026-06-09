import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, 
                             confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split, cross_val_score

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Formatting
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape:  {test_df.shape}")


print("\n--- Columns in Training Set ---")
print(train_df.columns.tolist())

print("\n--- Target Distribution ---")
target_col = 'efs'

print(train_df[target_col].value_counts(normalize=True))

plt.figure(figsize=(6, 4))
sns.countplot(x=train_df[target_col])
plt.title(f"Distribution of Target: {target_col}")
plt.show()


train_df.info()


train_df.head()


def clean_feature_names(df):
    clean_cols = []
    for col in df.columns:
        # Convert to string first
        new_col = str(col)
        
        # Replace common JSON special characters with underscore
        # We target: [ ] < > : , " ' { }
        new_col = re.sub(r'[\[\]<>\:, "\'\{\}]', '_', new_col)
        
        # Clean up double underscores (optional, just makes it prettier)
        new_col = re.sub(r'__+', '_', new_col)
        
        clean_cols.append(new_col)
        
    df.columns = clean_cols
    return df


def preprocess(df_train, df_test):
    # Target
    y = df_train['efs']

    # IDs
    test_IDs = test_df['ID']

    # Features to Drop
    drop_cols_train = ['ID', 'efs', 'efs_time']
    drop_cols_test = ['ID']
    
    df_train = df_train.drop(columns=drop_cols_train, errors='ignore')
    df_test = df_test.drop(columns=drop_cols_test, errors='ignore')
    
    print(f"Initial Feature Shape: {df_train.shape}")

    # Feature Engineering
    # Age Difference (Donor Age - Patient Age)
    def add_age_diff(df):
        d_age = df['donor_age'].fillna(df['donor_age'].median())
        p_age = df['age_at_hct'].fillna(df['age_at_hct'].median())
        df['age_diff'] = d_age - p_age
        return df
    
    df_train = add_age_diff(df_train)
    df_test = add_age_diff(df_test)
    
    # Identify Numerical vs Categorical
    num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df_train.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numerical Cols: {len(num_cols)}")
    print(f"Categorical Cols: {len(cat_cols)}")
    
    # Imputation
    # Numeric -> Median
    imp_num = SimpleImputer(strategy='median')
    df_train[num_cols] = imp_num.fit_transform(df_train[num_cols])
    df_test[num_cols] = imp_num.transform(df_test[num_cols])
    
    # Categorical -> "Missing" (New category)
    imp_cat = SimpleImputer(strategy='constant', fill_value='Missing')
    df_train[cat_cols] = imp_cat.fit_transform(df_train[cat_cols])
    df_test[cat_cols] = imp_cat.transform(df_test[cat_cols])
    
    # One-Hot Encoding
    ntrain = df_train.shape[0]
    combined = pd.concat([df_train, df_test], axis=0)
    
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    
    # Split back
    df_train_final = combined_encoded[:ntrain]
    df_test_final = combined_encoded[ntrain:]
    
    # Scaling
    # Required for Logistic Regression to converge
    scaler = StandardScaler()
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train_final), columns=df_train_final.columns)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test_final), columns=df_test_final.columns)

    df_train_final = clean_feature_names(df_train_scaled)
    df_test_final = clean_feature_names(df_test_scaled)
    
    return df_train_final, df_test_final, y, test_IDs

# Execute
train_df_prep, test_df_prep, y, test_IDs = preprocess(train_df, test_df)

print("\nâœ… Data Processed Successfully!")
print(f"Final Train Shape: {train_df_prep.shape}")
print(f"Final Test Shape:  {test_df_prep.shape}")


train_df_prep.head()


train_df_prep.info()


X_train, X_test, y_train, y_test = train_test_split(train_df_prep, y, test_size=0.2, random_state=42, stratify=y)

print(f"Local Train Shape: {X_train.shape}")
print(f"Local Val Shape:   {X_test.shape}")


def objective(trial, model_name):
    # Depending on the model name, suggest different hyperparameters
    if model_name == "Logistic Regression":
        params = {
            'C': trial.suggest_float('C', 0.001, 10.0, log=True),
            'solver': 'liblinear',
            'random_state': 42
        }
        model = LogisticRegression(**params)
        
    elif model_name == "Decision Tree":
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
        model = DecisionTreeClassifier(**params)
        
    elif model_name == "Random Forest":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        
    elif model_name == "Gradient Boosting":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'random_state': 42
        }
        model = GradientBoostingClassifier(**params)

    score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return score

# Optimization & Evaluation Loop
model_names = [
    "Logistic Regression", 
    "Decision Tree", 
    "Random Forest", 
    "Gradient Boosting"
]

results = {}
best_models = {}
best_params_preset = {
    "Logistic Regression": {'C': 0.0034834383104194854}, 
    "Decision Tree": {'max_depth': 6, 'min_samples_split': 3, 'min_samples_leaf': 10}, 
    "Random Forest": {'n_estimators': 189, 'max_depth': 20, 'min_samples_split': 5}, 
    "Gradient Boosting": {'n_estimators': 172, 'learning_rate': 0.051543329187906745, 'max_depth': 4}
}
print("\nğŸš€ STARTING HYPERPARAMETER TUNING & EVALUATION...")

for name in model_names:
    print(f"\n{'='*40}")
    print(f"ğŸ”¹ Tuning {name}...")

    preset_params = best_params_preset.get(name)

    if preset_params is None:
        start_time = time.time()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, name), n_trials=10)
        
        tuning_time = time.time() - start_time
        
        print(f"   Best Params: {study.best_params}")
        print(f"   Tuning Time: {tuning_time:.2f} seconds")
        
        best_params = study.best_params
    else:
        print("   âœ… Using Preset Params (Skipping Optuna)")
        best_params = preset_params
        print(f"   Params: {best_params}")

    # --- Train Best Model ---
    # Re-instantiate the model with best params
    if name == "Logistic Regression":
        best_model = LogisticRegression(**best_params, solver='liblinear', random_state=42)
    elif name == "Decision Tree":
        best_model = DecisionTreeClassifier(**best_params, random_state=42)
    elif name == "Random Forest":
        best_model = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)
    elif name == "Gradient Boosting":
        best_model = GradientBoostingClassifier(**best_params, random_state=42)
        
    best_model.fit(X_train, y_train)
    best_models[name] = best_model
    
    # Evaluate on Test Set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[name] = auc
    
    print(f"\nğŸ“Š {name} Results:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   ROC-AUC:   {auc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   Confusion Matrix:\n{cm}")

# Identify the best model
winner = max(results, key=results.get)
print(f"\n{'='*40}")
print(f"ğŸ�† BEST MODEL: {winner}")
print(f"   Best AUC: {results[winner]:.4f}")
print(f"{'='*40}")


def objective_advanced(trial, model_name):
    if model_name == "XGBoost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        model = xgb.XGBClassifier(**params)
        
    elif model_name == "LightGBM":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        model = lgb.LGBMClassifier(**params)
        
    elif model_name == "CatBoost":
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': 42,
            'verbose': 0
        }
        model = CatBoostClassifier(**params)

    # Cross-validation tuning
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

# --- Optimization Loop ---
advanced_models = ["XGBoost", "LightGBM", "CatBoost"]
advanced_results = {}
best_advanced_models = {}
best_advanced_params_preset = {
    "XGBoost": {'n_estimators': 449, 'learning_rate': 0.08807182965977357, 'max_depth': 3, 'subsample': 0.6853222890966244, 'colsample_bytree': 0.9368160787289258},
    "LightGBM": {'n_estimators': 356, 'learning_rate': 0.027901918953983702, 'num_leaves': 30, 'max_depth': 12, 'min_child_samples': 49},
    "CatBoost": {'iterations': 495, 'learning_rate': 0.039364391861413175, 'depth': 5, 'l2_leaf_reg': 6.2101135754851935}
}
print("\nğŸš€ STARTING ADVANCED MODEL TUNING...")

for name in advanced_models:
    print(f"\n{'='*40}")
    print(f"ğŸ”¹ Tuning {name}...")

    preset_params = best_advanced_params_preset.get(name)

    if preset_params is None:
        start_time = time.time()
        
        # Optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_advanced(trial, name), n_trials=20)
        
        tuning_time = time.time() - start_time
        print(f"   Best Params: {study.best_params}")
        print(f"   Tuning Time: {tuning_time:.2f} seconds")
        
        # Retrain Best Version
        best_params = study.best_params
    else:
        print("   âœ… Using Preset Params (Skipping Optuna)")
        best_params = preset_params
        print(f"   Params: {best_params}")
    
    if name == "XGBoost":
        best_model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1)
    elif name == "LightGBM":
        best_model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbosity=-1)
    elif name == "CatBoost":
        best_model = CatBoostClassifier(**best_params, random_seed=42, verbose=0)
        
    best_model.fit(X_train, y_train)
    best_advanced_models[name] = best_model
    
    # 3. Evaluate
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    advanced_results[name] = auc
    
    print(f"\nğŸ“Š {name} Results:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   ROC-AUC:   {auc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   Confusion Matrix:\n{cm}")

# Find the Grand Champion
winner_advanced = max(advanced_results, key=advanced_results.get)
print(f"\n{'='*40}")
print(f"ğŸ�† ADVANCED CHAMPION: {winner_advanced}")
print(f"   Best AUC: {advanced_results[winner_advanced]:.4f}")
print(f"{'='*40}")


CHOSEN_MODEL_NAME = "XGBoost" 
SUBMISSION_FILENAME = "submission.csv"

print(f"ğŸš€ Preparing Submission with: {CHOSEN_MODEL_NAME}")


print("   Recovering full dataset...")
train_df_prep, test_df_prep, y, test_ids = preprocess(train_df, test_df)

print(f"   Full Training Shape: {train_df_prep.shape}")
print(f"   Full Test Shape:     {test_df_prep.shape}")

try:
    if CHOSEN_MODEL_NAME in best_advanced_models:
        model = best_advanced_models[CHOSEN_MODEL_NAME]
        best_params = model.get_params()
    elif CHOSEN_MODEL_NAME in best_models:
        model = best_models[CHOSEN_MODEL_NAME]
        best_params = model.get_params()
    else:
        raise ValueError(f"Model '{CHOSEN_MODEL_NAME}' not found in results!")
        
    print(f"   Loaded best parameters for {CHOSEN_MODEL_NAME}")
    
except NameError:
    print("âš ï¸� 'best_advanced_models' dict not found.")
    # Fallback: Define default params if previous step wasn't run
    best_params = {'random_state': 42}

# RE-TRAIN ON FULL DATA & PREDICT
print("   Training on 100% of data...")

# Instantiate a fresh model with the best parameters
if "Logistic Regression" in CHOSEN_MODEL_NAME:
    final_model = LogisticRegression(**best_params)
    
elif "Decision Tree" in CHOSEN_MODEL_NAME:
    final_model = DecisionTreeClassifier(**best_params)
    
elif "Random Forest" in CHOSEN_MODEL_NAME:
    final_model = RandomForestClassifier(**best_params)
    
elif "Gradient Boosting" in CHOSEN_MODEL_NAME:
    final_model = GradientBoostingClassifier(**best_params)

elif "XGBoost" in CHOSEN_MODEL_NAME:
    final_model = xgb.XGBClassifier(**best_params)

elif "LightGBM" in CHOSEN_MODEL_NAME:
    final_model = lgb.LGBMClassifier(**best_params)

elif "CatBoost" in CHOSEN_MODEL_NAME:
    final_model = CatBoostClassifier(**best_params)

# fit
final_model.fit(train_df_prep, y)

# predict
final_preds = final_model.predict_proba(test_df_prep)[:, 1]

# save
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'prediction': final_preds
})

submission.to_csv(SUBMISSION_FILENAME, index=False)
print(f"âœ… Submission saved to: {SUBMISSION_FILENAME}")
print(f"   Shape: {submission.shape}")
print(f"   First 5 predictions:\n{submission.head()}")

