from IPython.display import IFrame
IFrame("https://www.youtube.com/embed/pcT_bpOXa-M", width="100%", height=400)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid",palette="deep",font_scale=1.1,rc={'axes.spines.right': False, 'axes.spines.top': False})

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=3, suppress=True)
plt.rcParams['axes.formatter.use_mathtext'] = True


train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train_df.shape


train_df.info()


numerical_cols = train_df.select_dtypes(include=['number']).columns

# 3. Plot: Loop through the columns and plot distributions
# We determine the grid size dynamically based on the number of columns
num_plots = len(numerical_cols)
rows = (num_plots + 1) // 2  # Calculate required rows for a 2-column grid

plt.figure(figsize=(10, 4 * rows))

for i, col in enumerate(numerical_cols):
    plt.subplot(rows, 2, i + 1)
    # kde=True adds the smooth density line over the bars
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


numerical_to_boolean = ["cardiovascular_history","hypertension_history","family_history_diabetes"]
train_df[numerical_to_boolean] = train_df[numerical_to_boolean].astype('category')
test_df[numerical_to_boolean] = test_df[numerical_to_boolean].astype('category')


skewed_to_gauss = ['physical_activity_minutes_per_week']
train_df[skewed_to_gauss] = np.log1p(train_df[skewed_to_gauss])
test_df[skewed_to_gauss] = np.log1p(test_df[skewed_to_gauss])


# 2. Filter: Select only categorical columns (object or category)
cat_cols = train_df.select_dtypes(include=['object', 'category']).columns

# 3. Plotting
# Calculate grid size (e.g., 2 columns wide)
n_cols = 2
n_rows = (len(cat_cols) + n_cols - 1) // n_cols

plt.figure(figsize=(14, 4 * n_rows))

for i, col in enumerate(cat_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    
    # sns.countplot automatically counts the occurrences
    # y=col makes it horizontal (easier to read labels)
    # order=... sorts the bars from largest to smallest
    sns.countplot(
        data=train_df, 
        y=col, 
        order=train_df[col].value_counts().index, 
        palette='viridis'
    )
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


object_cols = train_df.select_dtypes(include=['object']).columns.tolist()

print(f"Converting columns: {object_cols}")

# 2. Convert to Category
# We loop through them to ensure the category 'universe' is consistent
for col in object_cols:
    # A. Get the union of unique values from both Train and Test
    # This prevents errors if Test has a value that Train doesn't have
    unique_values = set(train_df[col].unique()) | set(test_df[col].unique())
    unique_values = sorted(list(unique_values))
    
    # B. Define the specific categorical type
    cat_type = pd.CategoricalDtype(categories=unique_values, ordered=False)
    
    # C. Apply to both dataframes
    train_df[col] = train_df[col].astype(cat_type)
    test_df[col] = test_df[col].astype(cat_type)


import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Turn this to True if you have selected a GPU Accelerator in Kaggle
USE_GPU = False 

MODELS_TO_TEST = ['xgboost', 'lightgbm', 'catboost']
TARGET_COL = 'diagnosed_diabetes'
ID_COL = 'id'

# OPTIMIZATION SETTINGS (Faster)
N_FOLDS_OPT = 3      # Use fewer folds for finding params to save time
N_TRIALS = 15        # Number of hyperparam combinations to try
MAX_OPT_SAMPLES = 100000 # Limit rows for Optuna to speed up search (e.g. 100k)

# SUBMISSION SETTINGS (High Quality)
N_FOLDS_SUB = 5      # Use more folds for the final stable prediction
RANDOM_SEED = 42

study_results = {}

print("--- Preprocessing: Aligning Categories ---")
object_cols = train_df.select_dtypes(include=['object']).columns.tolist()

for col in object_cols:
    unique_vals = set(train_df[col].unique()) | set(test_df[col].unique())
    unique_vals = sorted(list(unique_vals))
    cat_type = pd.CategoricalDtype(categories=unique_vals, ordered=False)
    train_df[col] = train_df[col].astype(cat_type)
    test_df[col] = test_df[col].astype(cat_type)

X = train_df.drop(columns=[TARGET_COL, ID_COL])
y = train_df[TARGET_COL]
X_submission = test_df.drop(columns=[ID_COL])

cat_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype.name == 'category']

if len(X) > MAX_OPT_SAMPLES:
    print(f"\n[INFO] Dataset is large ({len(X)} rows). Subsampling to {MAX_OPT_SAMPLES} for Optuna tuning.")
    X_opt, _, y_opt, _ = train_test_split(X, y, train_size=MAX_OPT_SAMPLES, stratify=y, random_state=RANDOM_SEED)
else:
    print(f"\n[INFO] Using full dataset ({len(X)} rows) for Optuna tuning.")
    X_opt, y_opt = X, y


def run_cv_with_early_stopping(model_name, params, X, y, n_folds):
    """
    Runs Cross-Validation manually to enable Early Stopping.
    This is much faster than cross_val_score.
    """
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # --- Initialize Model ---
        if model_name == 'xgboost':
            model = xgb.XGBClassifier(**params)
            # XGBoost Early Stopping
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
        elif model_name == 'lightgbm':
            model = lgb.LGBMClassifier(**params)
            # LightGBM Early Stopping (callbacks in newer versions)
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=callbacks)

        elif model_name == 'catboost':
            model = cb.CatBoostClassifier(**params)
            # CatBoost Early Stopping
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

        # Predict
        preds = model.predict_proba(X_val)[:, 1]
        
        # Calculate AUC for this fold
        try:
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y_val, preds)
            fold_scores.append(score)
        except:
            fold_scores.append(0.5) # Fallback

    return np.mean(fold_scores)



for model_name in MODELS_TO_TEST:
    print(f"\n============================================")
    print(f"  Optimizing: {model_name.upper()}")
    print(f"============================================")

    def objective(trial):
        
        if model_name == 'xgboost':
            params = {
                'n_estimators': 1500, # High cap, let early stopping decide
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'objective': 'binary:logistic',
                'tree_method': 'gpu_hist' if USE_GPU else 'hist', # GPU ACTIVATION
                'enable_categorical': True,
                'n_jobs': -1,
                'random_state': RANDOM_SEED,
                'eval_metric': 'auc',
                'early_stopping_rounds': 50 # Parameter for the XGB wrapper
            }
            
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': 1500,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'binary',
                'device': 'gpu' if USE_GPU else 'cpu', # GPU ACTIVATION
                'n_jobs': -1,
                'verbosity': -1,
                'random_state': RANDOM_SEED
            }

        elif model_name == 'catboost':
            params = {
                'iterations': 1500,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'loss_function': 'Logloss',
                'task_type': 'GPU' if USE_GPU else 'CPU', # GPU ACTIVATION
                'thread_count': -1,
                'random_seed': RANDOM_SEED,
                'cat_features': cat_features_indices
            }

        avg_score = run_cv_with_early_stopping(model_name, params, X_opt, y_opt, n_folds=N_FOLDS_OPT)
        return avg_score

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)
    
    study_results[model_name] = {
        'best_score': study.best_value,
        'best_params': study.best_params
    }
    print(f"  > Best AUC for {model_name}: {study.best_value:.4f}")


print("\n--- Generating Comparison Plot ---")

model_names = list(study_results.keys())
scores = [study_results[m]['best_score'] for m in model_names]

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=scores, palette='viridis')
plt.title('Model Comparison: Best CV AUC Score', fontsize=15)
plt.ylabel('AUC Score')
plt.ylim(min(scores) - 0.05, max(scores) + 0.05) # Zoom in on the top
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on bars
for i, v in enumerate(scores):
    plt.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png') # Save plot
plt.show()




import shap  

print("\n============================================")
print("  TRAINING BEST MODEL & SUBMISSION")
print("============================================")

best_model_name = max(study_results, key=lambda k: study_results[k]['best_score'])
best_score = study_results[best_model_name]['best_score']
best_params = study_results[best_model_name]['best_params']

print(f"\n[WINNER] The best performing model is: {best_model_name.upper()}")
print(f"         Optuna CV AUC: {best_score:.4f}")
print("         Proceeding to train on full dataset with K-Fold Averaging...")

kf = StratifiedKFold(n_splits=N_FOLDS_SUB, shuffle=True, random_state=RANDOM_SEED)

if best_model_name == 'xgboost':
    best_params.update({
        'n_estimators': 1500, 'objective': 'binary:logistic', 'eval_metric': 'auc', 
        'tree_method': 'gpu_hist' if USE_GPU else 'hist', 'enable_categorical': True, 
        'early_stopping_rounds': 50
    })
    ModelClass = xgb.XGBClassifier
elif best_model_name == 'lightgbm':
    best_params.update({
        'n_estimators': 1500, 'objective': 'binary', 'device': 'gpu' if USE_GPU else 'cpu', 
        'verbosity': -1
    })
    ModelClass = lgb.LGBMClassifier
elif best_model_name == 'catboost':
    best_params.update({
        'iterations': 1500, 'loss_function': 'Logloss', 'task_type': 'GPU' if USE_GPU else 'CPU', 
        'cat_features': cat_features_indices
    })
    ModelClass = cb.CatBoostClassifier

fold_preds = []
feature_importances = np.zeros(len(X.columns))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"  Training Fold {fold+1}/{N_FOLDS_SUB}...")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = ModelClass(**best_params)
    
    if best_model_name == 'lightgbm':
         model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', 
                   callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    elif best_model_name == 'catboost':
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    elif best_model_name == 'xgboost':
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    probs = model.predict_proba(X_submission)[:, 1]
    fold_preds.append(probs)

    try:
         feature_importances += model.feature_importances_ / N_FOLDS_SUB
    except AttributeError:
         pass
         
    if fold == N_FOLDS_SUB - 1:
        print(f"  -> Calculating SHAP values on fold {fold+1} validation set...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            
            if isinstance(shap_values, list):
                shap_matrix = shap_values[1]
            else:
                shap_matrix = shap_values
                
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_matrix, X_val, show=False)
            plt.title(f'SHAP Summary (Directional Importance) - {best_model_name.upper()}')
            plt.tight_layout()
            plt.savefig('shap_summary_best_model.png')
            plt.show()
            print("  -> Saved SHAP Summary Plot (shap_summary_best_model.png)")
            
        except Exception as e:
            print(f"  [WARNING] Could not generate SHAP plot: {e}")

avg_preds = np.mean(fold_preds, axis=0)




# Save Submission
filename = "submission.csv" # Standard name since it's the winner
sub = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: avg_preds})
sub.to_csv(filename, index=False)
print(f"  -> Saved Final Submission: {filename}")




print(f"  -> Plotting standard feature importance...")
fi_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
fi_df = fi_df.sort_values(by='importance', ascending=False).head(20) # Top 20 features

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=fi_df, palette='viridis')
plt.title(f'Top 20 Feature Importance (Magnitude) - {best_model_name.upper()}')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f'feature_importance_best_model.png')
print("  -> Saved Standard Feature Importance Plot")
plt.show()




