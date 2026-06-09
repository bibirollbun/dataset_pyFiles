import pandas as pd
df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df_external = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')

# Map external target to binary with extra weight indicator
df_external['y'] = df_external['y'].map({'yes': 1, 'no': 0})

# Indexing
df_train.set_index("id", inplace=True)
df_test.set_index("id", inplace=True)

df_combined = pd.concat([df_train, df_external], ignore_index=True)


# Columns to list
target = 'y'
numerical_cols = df_train.select_dtypes(include='int64').columns.drop(target)
categorical_cols = df_train.select_dtypes(include='object').columns

# Cardinalities
low_cardinality = ['poutcome',
                   'contact',
                   'loan',
                   'housing',
                   'default',
                   'education',
                   'marital'
                  ]

high_cardinality = ['month',
                    'job']


from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

# LabelEncoder for high-cardinality
df_combined = df_combined.copy()

label_encoders = {}

for col in high_cardinality:
    le = LabelEncoder()
    df_combined[col] = le.fit_transform(df_combined[col].astype(str))
    label_encoders[col] = le

# Apply stored encoders to df_test
for col in high_cardinality:
    le = label_encoders[col]
    df_test[col] = le.transform(df_test[col].astype(str))

# X and y dataframes
X = df_combined.drop(columns=['y'])
y = df_combined['y']

# Handle class imbalance
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(scale_pos_weight)


# Preprocessor for categorical columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(transformers=[
    ('low_card', OneHotEncoder(handle_unknown='ignore'), low_cardinality),
], remainder='passthrough')


import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
import time

# Speed optimizations for large datasets
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_lightgbm_fast(X, y, preprocessor, n_trials=30, cv_folds=3):
    """Fast LightGBM optimization for large datasets"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 375, 563),  # ±20%
            'learning_rate': trial.suggest_float('learning_rate', 0.0436, 0.0654, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 160, 240),
            'max_depth': trial.suggest_int('max_depth', 10, 16),
            'min_child_samples': trial.suggest_int('min_child_samples', 37, 57),
            'subsample': trial.suggest_float('subsample', 0.611, 0.917),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.643, 0.965),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.00163, 0.00244, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.565, 0.847),
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }
        
        model = LGBMClassifier(**params)
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])
        
        # Use fewer CV folds for speed
        cv_scores = cross_val_score(
            pipeline, X, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=1
        )
        
        return cv_scores.mean()
    
    # Add pruner for early stopping of bad trials
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

def optimize_catboost_fast(X, y, preprocessor, n_trials=30, cv_folds=3):
    """Fast CatBoost optimization"""
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 360, 540),
            'learning_rate': trial.suggest_float('learning_rate', 0.124, 0.186, log=True),
            'depth': trial.suggest_int('depth', 6, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.05, 4.58),
            'border_count': trial.suggest_int('border_count', 102, 154),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.209, 0.315),
            'random_strength': trial.suggest_float('random_strength', 0.017, 0.026),
            'verbose': 0,
            'random_state': 42,
            'thread_count': -1,
            'allow_writing_files': False
        }
        
        model = CatBoostClassifier(task_type="GPU", devices="0", **params)
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])
        
        cv_scores = cross_val_score(
            pipeline, X, y,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=1
        )
        
        return cv_scores.mean()
    
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

def optimize_xgboost_fast(X, y, preprocessor, n_trials=30, cv_folds=3):
    """Fast XGBoost optimization"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 460, 690),
            'learning_rate': trial.suggest_float('learning_rate', 0.091, 0.136, log=True),
            'max_depth': trial.suggest_int('max_depth', 8, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 7, 11),
            'subsample': trial.suggest_float('subsample', 0.643, 0.965),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.314, 0.471),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.236, 0.354),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.998, 1.498),
            'gamma': trial.suggest_float('gamma', 1.286, 1.931),
            'use_label_encoder': False,
            'eval_metric': 'aucpr',
            'tree_method': 'gpu_hist',
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = XGBClassifier(**params)
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])
        
        cv_scores = cross_val_score(
            pipeline, X, y,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=1
        )
        
        return cv_scores.mean()
    
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

def progressive_optimization(X, y, preprocessor):
    """
    Progressive optimization strategy:
    1. Quick optimization first (few trials, 3-fold CV)
    2. Fine-tune best model with more trials
    """
    
    print("=== PHASE 1: Quick Optimization (30 trials, 3-fold CV) ===")
    start_time = time.time()
    
    # Quick optimization
    studies = {}
    
    print("Optimizing LightGBM...")
    lgbm_start = time.time()
    studies['lgbm'] = optimize_lightgbm_fast(X, y, preprocessor, n_trials=30, cv_folds=3)
    lgbm_time = time.time() - lgbm_start
    print(f"LightGBM done in {lgbm_time/60:.1f} minutes. Best score: {studies['lgbm'].best_value:.4f}")
    
    print("Optimizing CatBoost...")
    cat_start = time.time()
    studies['catboost'] = optimize_catboost_fast(X, y, preprocessor, n_trials=30, cv_folds=3)
    cat_time = time.time() - cat_start
    print(f"CatBoost done in {cat_time/60:.1f} minutes. Best score: {studies['catboost'].best_value:.4f}")
    
    print("Optimizing XGBoost...")
    xgb_start = time.time()
    studies['xgboost'] = optimize_xgboost_fast(X, y, preprocessor, n_trials=30, cv_folds=3)
    xgb_time = time.time() - xgb_start
    print(f"XGBoost done in {xgb_time/60:.1f} minutes. Best score: {studies['xgboost'].best_value:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nPhase 1 total time: {total_time/60:.1f} minutes")
    
    # Find best performing model
    best_model = max(studies.keys(), key=lambda k: studies[k].best_value)
    print(f"Best model: {best_model} with score {studies[best_model].best_value:.4f}")
    
    return studies, best_model

# Usage example
def run_optimization():
    """Run the optimization with timing"""
    
    print("Starting hyperparameter optimization...")
    print(f"Dataset size: {len(X):,} rows, {X.shape[1]} features")
    
    # Run progressive optimization
    studies, best_model = progressive_optimization(X, y, preprocessor)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    
    for name, study in studies.items():
        print(f"\n{name.upper()}:")
        print(f"  Best Score: {study.best_value:.4f}")
        print(f"  Best Params: {study.best_params}")
    
    return studies

# Run
#studies = run_optimization()


import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Create optimized models using best parameters from Optuna with GPU acceleration

warnings.filterwarnings("ignore")

xgb_clf = XGBClassifier(
    n_estimators=647,
    learning_rate=0.09114756913639371,
    max_depth=12,
    min_child_weight=10,
    subsample=0.761789863074996,
    colsample_bytree=0.4446584971388443,
    reg_alpha=0.31411679571900897,
    reg_lambda=1.3204905555684525,
    gamma=1.3593858855013405,
    tree_method='gpu_hist',
    gpu_id=0,
    use_label_encoder=False,
    eval_metric='logloss'
)

lgb_clf = LGBMClassifier(
    n_estimators=545,
    learning_rate=0.050021327454551366,
    num_leaves=193,
    max_depth=15,
    min_child_samples=54,
    subsample=0.8060876973621491,
    colsample_bytree=0.6449670145917592,
    reg_alpha=0.0023620286601468052,
    reg_lambda=0.8463753324549448,
    device='gpu',
    verbose=-1
)

cat_clf = CatBoostClassifier(
    iterations=515,
    learning_rate=0.17472914139061765,
    depth=8,
    l2_leaf_reg=3.582130167868121,
    border_count=146,
    bagging_temperature=0.2801692174344982,
    random_strength=0.02075363936076622,
    task_type="GPU",
    devices="0",
    verbose=0,
    #cat_features=low_cardinality+high_cardinality,
    allow_writing_files=False
)

# Base models list
base_models = [
    ('xgb', xgb_clf),
    ('lgb', lgb_clf),
    #('cat', cat_clf)
]

# Cross-validation setup
n_splits = 10  # 5
n_repeats = 5  # 3
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Initialize arrays for meta-features
meta_X = np.zeros((len(X), len(base_models)))
meta_test = np.zeros((len(df_test), len(base_models)))

print(f"\nTraining ensemble with {n_splits}-fold CV, {n_repeats} repeats...")

# Train each base model
for i, (name, model) in enumerate(base_models):
    print(f"\nTraining base model: {name}")
    
    # Arrays to track out-of-fold predictions
    oof = np.zeros(len(X))
    oof_counts = np.zeros(len(X))
    test_fold_preds = []
    
    # Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        print(f"  Fold {fold_idx + 1}/{n_splits * n_repeats}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        
        # Create and fit pipeline
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        
        # Get out-of-fold predictions
        val_preds = pipeline.predict_proba(X_val)[:, 1]
        oof[val_idx] += val_preds
        oof_counts[val_idx] += 1
        
        # Get test predictions for this fold
        test_preds = pipeline.predict_proba(df_test)[:, 1]
        test_fold_preds.append(test_preds)
    
    # Average predictions properly
    meta_X[:, i] = oof / oof_counts
    meta_test[:, i] = np.mean(test_fold_preds, axis=0)
    
    # Show individual model performance
    oof_score = roc_auc_score(y, meta_X[:, i])
    print(f"  OOF ROC AUC ({name}): {oof_score:.4f}")

print(f"\nTraining meta-learner...")

# Train meta-learner with regularization to prevent extreme weights
meta_model = LogisticRegression(
    C=0.1,
    random_state=42,
    max_iter=1000
)
meta_model.fit(meta_X, y)

# Get final predictions
stacked_preds = meta_model.predict_proba(meta_test)[:, 1]

# Calculate final performance
final_oof_preds = meta_model.predict_proba(meta_X)[:, 1]
final_auc = roc_auc_score(y, final_oof_preds)
print(f"\nFinal Meta-model ROC AUC (OOF): {final_auc:.4f}")

# Show model weights (should be more balanced now)
print(f"\nOptimized Meta-model weights:")
for i, (name, _) in enumerate(base_models):
    print(f"  {name}: {meta_model.coef_[0][i]:.4f}")
print(f"  Intercept: {meta_model.intercept_[0]:.4f}")

print(f"\nCreating submission file...")

# Create submission DataFrame
# Use 'id' column if it exists, otherwise use index
if 'id' in df_test.columns:
    submission_ids = df_test['id']
else:
    submission_ids = df_test.index

submission = pd.DataFrame({
    'id': submission_ids,
    'y': stacked_preds
})

# Check for any issues
if submission['y'].isna().sum() > 0:
    print(f"Warning: {submission['y'].isna().sum()} NaN values found in predictions!")
    submission['y'] = submission['y'].fillna(submission['y'].mean())

# Save submission file
filename = 'optimized_stacked_submission3.csv'
submission.to_csv(filename, index=False)

print(f"\n" + "="*50)
print("SUBMISSION READY!")
print(f"File: {filename}")
print(f"Final AUC: {final_auc:.4f}")
print("="*50)

print(f"\nFiles created:")
print(f"{filename} - Ensemble")


# Create submission DataFrame
# Use 'id' column if it exists, otherwise use index
if 'id' in df_test.columns:
    submission_ids = df_test['id']
else:
    submission_ids = df_test.index

submission = pd.DataFrame({
    'id': submission_ids,
    'y': stacked_preds
})

# Check for any issues
if submission['y'].isna().sum() > 0:
    print(f"Warning: {submission['y'].isna().sum()} NaN values found in predictions!")
    submission['y'] = submission['y'].fillna(submission['y'].mean())

# Save submission file
filename = 'optimized_stacked_submission4.csv'
submission.to_csv(filename, index=False)

print(f"\n" + "="*50)
print("SUBMISSION READY!")
print(f"File: {filename}")
print(f"Final AUC: {final_auc:.4f}")
print("="*50)

print(f"\nFiles created:")
print(f"{filename} - Ensemble")




