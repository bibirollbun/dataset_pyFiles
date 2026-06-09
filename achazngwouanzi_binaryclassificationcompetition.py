# --- Basic Libraries ---
import pandas as pd
import numpy as np

# --- Modeling ---
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# --- Utilities ---
import gc
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


# Define file paths
COMP_PATH = '/kaggle/input/playground-series-s5e8/'
ORIG_PATH = '/kaggle/input/bank-marketing-dataset-full/'

# Load competition data
df_train = pd.read_csv(COMP_PATH + 'train.csv')
df_test = pd.read_csv(COMP_PATH + 'test.csv')
df_submission = pd.read_csv(COMP_PATH + 'sample_submission.csv')

# Load original data
df_orig = pd.read_csv(ORIG_PATH + 'bank-full.csv', sep=';')

# --- Create the 'is_original' flag ---
df_train['is_original'] = 0
df_orig['is_original'] = 1

print(f"Competition Train Shape: {df_train.shape}")
print(f"Competition Test Shape:  {df_test.shape}")
print(f"Original Data Shape:     {df_orig.shape}")


# --- Harmonize & Combine ---
# Drop the 'id' column
df_train = df_train.drop('id', axis=1)
df_test_ids = df_test['id']
df_test = df_test.drop('id', axis=1)

# Add 'is_original' flag to the test set as well (it's synthetic)
df_test['is_original'] = 0

# Map the target variable in the original dataset to 0/1
df_orig['y'] = df_orig['y'].map({'yes': 1, 'no': 0})

# Combine original data with competition training data
train_full = pd.concat([df_train, df_orig], ignore_index=True)

# Combine all data for unified preprocessing
combined_df = pd.concat([train_full.drop('y', axis=1), df_test], ignore_index=True)

print(f"Full Training Data Shape: {train_full.shape}")
print(f"Combined DataFrame Shape for Preprocessing: {combined_df.shape}")


# --- Create New Features ---

# Interaction features
combined_df['balance_per_age'] = combined_df['balance'] / (combined_df['age'] + 1)
combined_df['duration_per_campaign'] = combined_df['duration'] / (combined_df['campaign'] + 1)

# Indicator features
combined_df['no_previous_contact'] = (combined_df['pdays'] == -1).astype(int)

print("New features created successfully.")
combined_df.head()


# --- Memory Optimization Function ---
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else: df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

# Apply memory optimization
combined_df = reduce_mem_usage(combined_df)

# --- Label Encoding ---
categorical_cols = combined_df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])

print("\nMemory optimization and Label Encoding complete.")


# --- Prepare data for modeling ---
X = combined_df[:len(train_full)]
X_test = combined_df[len(train_full):]
y = train_full['y']

# --- Model Training ---
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(X_test.shape[0])

# LightGBM parameters with GPU enabled
# Best parameters found by the 40-trial Optuna study
# lgb_params = {
#     'objective': 'binary',
#     'metric': 'auc',
#     'boosting_type': 'gbdt',
#     'device': 'gpu',
#     'random_state': 42,
    
#     # Your NEW proven parameters:
#     'n_estimators': 2264,
#     'learning_rate': 0.0795263602880664,
#     'num_leaves': 138,
#     'max_depth': 10,
#     'subsample': 0.5773625282947411,
#     'colsample_bytree': 0.5458486122385823,
#     'reg_alpha': 4.635636727152218,
#     'reg_lambda': 9.374318255299915e-06,
    
#     # Other good parameters to keep
#     'n_jobs': -1,
#     'verbose': -1,
# }

# Best parameters found by the 80-trial Optuna study
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'random_state': 42,
    
    # Your NEWEST proven parameters:
    'n_estimators': 1422,
    'learning_rate': 0.07923375968167197,
    'num_leaves': 139,
    'max_depth': 11,
    'subsample': 0.6493187541279558,
    'colsample_bytree': 0.5910591112540586,
    'reg_alpha': 4.544481324932982,
    'reg_lambda': 8.830446222772213e-07,
    
    # Other good parameters to keep
    'n_jobs': -1,
    'verbose': -1,
}

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    sub_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits
    
    fold_auc = roc_auc_score(y_valid, oof_preds[valid_idx])
    print(f"Fold {n_fold+1} AUC: {fold_auc:.5f}")
    
    del model, X_train, y_train, X_valid, y_valid
    gc.collect()

cv_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall CV AUC: {cv_auc:.5f}")


df_submission['y'] = sub_preds
df_submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
df_submission.head()


# !pip install lightgbm --quiet


# pip install optuna-integration[lightgbm]


# # Cellule de Tuning pour 40 essais
# import optuna

# # 1. DÃ‰FINIR LA FONCTION OBJECTIVE
# # Optuna essaiera de maximiser le score retournÃ© par cette fonction.
# def objective(trial):
    
#     # DÃ©finir l'espace de recherche des hyperparamÃ¨tres
#     params = {
#         'objective': 'binary',
#         'metric': 'auc',
#         'boosting_type': 'gbdt',
#         'device': 'gpu',
#         'random_state': 42,
#         'n_jobs': -1,
#         'verbose': -1,
#         'n_estimators': trial.suggest_int('n_estimators', 800, 2500),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#         'max_depth': trial.suggest_int('max_depth', 5, 12),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
#     }

#     # Validation croisÃ©e Ã  l'intÃ©rieur de l'essai
#     NFOLDS = 5
#     folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
#     fold_scores = []
    
#     for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
#         X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#         X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

#         model = lgb.LGBMClassifier(**params)
        
#         # Le "Pruning" est activÃ© ici
#         pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
        
#         model.fit(X_train, y_train,
#                   eval_set=[(X_valid, y_valid)],
#                   eval_metric='auc',
#                   callbacks=[lgb.early_stopping(100, verbose=False), pruning_callback])

#         preds = model.predict_proba(X_valid)[:, 1]
#         auc_score = roc_auc_score(y_valid, preds)
#         fold_scores.append(auc_score)

#     # Retourner la moyenne des scores des 5 folds
#     return np.mean(fold_scores)


# # 2. LANCER L'Ã‰TUDE
# # CrÃ©e un "pruner" pour arrÃªter les mauvais essais
# pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
# study = optuna.create_study(direction='maximize', pruner=pruner)

# # Lancer 40 essais avec un timeout de 2 heures (7200 secondes)
# study.optimize(objective, n_trials=80, timeout=14400)


# # 3. AFFICHER LES RÃ‰SULTATS
# print("\n" + "="*50)
# print(f"Best CV AUC from study: {study.best_value}")
# print("Best LGBM parameters found:")
# print(study.best_params)
# print("="*50)




