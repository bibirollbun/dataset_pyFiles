# Cell 1 — Environment & imports
import os
import random
import gc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.impute import SimpleImputer

# Optional - for stacking meta NN
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# Optional hyperparameter tuning
import optuna

print("Versions:", "numpy", np.__version__, "pandas", pd.__version__)



# Cell 2 — Paths and constants
TRAIN_PATH = '/kaggle/input/playground-series-s5e11/train.csv'
TEST_PATH  = '/kaggle/input/playground-series-s5e11/test.csv'
OUT_PATH   = '/kaggle/working/submission_ensemble.csv'

SEED = 42
N_FOLDS = 5
SHUFFLE = True
VERBOSE = 100
N_JOBS = -1

# Set seeds
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything(SEED)



# Cell 3 — Load
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train.shape, test.shape



# Cell 4 — Column lists (adjust if schema differs)
ID_COL = 'id'
TARGET = 'loan_paid_back'
categorical_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
num_cols = [c for c in train.columns if c not in [ID_COL, TARGET] + categorical_cols]
print("Numerical:", num_cols)
print("Categorical:", categorical_cols)



# Cell 5 — Basic cleaning & consistent encoding utilities
# Fill missing for categoricals with 'missing' and numeric with median
for c in categorical_cols:
    train[c] = train[c].fillna('missing').astype(str)
    test[c]  = test[c].fillna('missing').astype(str)

num_imputer = SimpleImputer(strategy='median')
train_num = pd.DataFrame(num_imputer.fit_transform(train[num_cols]), columns=num_cols)
test_num  = pd.DataFrame(num_imputer.transform(test[num_cols]), columns=num_cols)

# keep copies
train_proc = train.copy()
test_proc  = test.copy()
train_proc[num_cols] = train_num
test_proc[num_cols] = test_num



# Cell 6 — Feature engineering (high-impact)
def add_features(df):
    # Ratios & interactions
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1e-9)
    df['income_per_credit'] = df['annual_income'] / (df['credit_score'] + 1e-9)
    df['interest_per_loan'] = df['interest_rate'] * df['loan_amount']
    df['credit_sq'] = df['credit_score'] ** 2
    df['dti_income'] = df['debt_to_income_ratio'] / (df['annual_income'] + 1e-9)
    # Clip / replace inf/nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

train_proc = add_features(train_proc)
test_proc  = add_features(test_proc)

# Add group aggregations (target-statistics) using out-of-fold safe method later during training
engineered_num_cols = [c for c in train_proc.columns if c not in [ID_COL, TARGET] + categorical_cols]
print("Engineered numeric columns count:", len(engineered_num_cols))



# Cell 7 — Target encoding with KFold (OOF-safe) for categorical columns
# We'll create new columns: te_<col>
def target_encode_oof(train_df, target_col, cat_cols, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=SHUFFLE, random_state=seed)
    oof = pd.DataFrame(index=train_df.index)
    global_means = train_df[target_col].mean()
    for col in cat_cols:
        oof_col = np.zeros(len(train_df))
        for tr_idx, val_idx in skf.split(train_df, train_df[target_col]):
            tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            stats = tr.groupby(col)[target_col].agg(['mean','count'])
            # smoothing
            smooth = (stats['count'] * stats['mean'] + 10 * global_means) / (stats['count'] + 10)
            val_vals = train_df.iloc[val_idx][col].map(smooth).fillna(global_means).values
            oof_col[val_idx] = val_vals
        oof[f'te_{col}'] = oof_col
    return oof

# create OOF encodings for train
te_oof = target_encode_oof(train_proc, TARGET, categorical_cols, n_splits=N_FOLDS, seed=SEED)
train_proc = pd.concat([train_proc.reset_index(drop=True), te_oof.reset_index(drop=True)], axis=1)

# For test, map using full-train statistics
full_stats = {}
global_mean = train_proc[TARGET].mean()
for col in categorical_cols:
    stats = train_proc.groupby(col)[TARGET].agg(['mean','count'])
    smooth = (stats['count'] * stats['mean'] + 10 * global_mean) / (stats['count'] + 10)
    full_stats[col] = smooth
    test_proc[f'te_{col}'] = test_proc[col].map(smooth).fillna(global_mean)

print("Added target-encoded features for categorical columns.")



# Cell 8 — Label encoding for catboost or for tree models we can pass categorial indices
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_cat_arr = ord_enc.fit_transform(train_proc[categorical_cols]).astype(int)
test_cat_arr = ord_enc.transform(test_proc[categorical_cols]).astype(int)

# Convert to category dtype for catboost (or keep arrays for other models)
for i, col in enumerate(categorical_cols):
    train_proc[f'le_{col}'] = train_cat_arr[:, i]
    test_proc[f'le_{col}'] = test_cat_arr[:, i]



# Cell 9 — Final feature list
# numeric features: engineered_num_cols + te_ columns + label encoded columns
te_cols = [f'te_{c}' for c in categorical_cols]
le_cols = [f'le_{c}' for c in categorical_cols]

feature_cols = num_cols + ['loan_to_income', 'income_per_credit', 'interest_per_loan', 'credit_sq', 'dti_income'] + te_cols + le_cols
# remove duplicates if any
feature_cols = [c for i,c in enumerate(feature_cols) if c not in feature_cols[:i]]
print("Total features:", len(feature_cols))



# Cell 10 — Prepare arrays
X = train_proc[feature_cols].values
y = train_proc[TARGET].values
X_test = test_proc[feature_cols].values

# Scale numeric features for models that benefit (we'll scale only the numeric sub-block for meta model)
scaler = StandardScaler()
# identify numeric indices to scale (numerics include num_cols + engineered numeric)
numeric_for_scaler = [c for c in feature_cols if c in train_proc.select_dtypes(include=[np.number]).columns]
X_scaled_for_meta = scaler.fit_transform(train_proc[numeric_for_scaler].values)
X_test_scaled_for_meta = scaler.transform(test_proc[numeric_for_scaler].values)



# Cell 11 — Helpers: CV train for LightGBM, XGBoost, CatBoost (output oof preds + test preds)
def train_lgb(X, y, X_test, feature_names, categorical_feature_indices=None, n_splits=N_FOLDS, seed=SEED):
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    models = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=SHUFFLE, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"LGB Fold {fold+1}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        lgb_train = lgb.Dataset(X_tr, y_tr, feature_name=feature_names, categorical_feature=categorical_feature_indices)
        lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=feature_names, categorical_feature=categorical_feature_indices)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 64,
            'min_data_in_leaf': 100,
            'feature_fraction': 0.8,
            'bagging_freq': 1,
            'bagging_fraction': 0.8,
            'lambda_l1': 0.2,
            'lambda_l2': 0.2,
            'seed': seed + fold,
            'verbosity': -1,
            'n_jobs': N_JOBS
        }

        # Updated: callbacks instead of verbose_eval
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(VERBOSE)]
        )
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof[val_idx] = val_pred
        preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        models.append(model)
        print("Fold ROC AUC:", roc_auc_score(y_val, val_pred))
    print("LGB OOF ROC AUC:", roc_auc_score(y, oof))
    return oof, preds, models

def train_xgb(X, y, X_test, feature_names, n_splits=N_FOLDS, seed=SEED):
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    models = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=SHUFFLE, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"XGB Fold {fold+1}")
        dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx], feature_names=feature_names)
        dval   = xgb.DMatrix(X[val_idx], label=y[val_idx], feature_names=feature_names)
        dtest  = xgb.DMatrix(X_test, feature_names=feature_names)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'alpha': 0.0,
            'seed': seed+fold,
            'verbosity': 0
        }
        watchlist = [(dtrain, 'train'), (dval, 'valid')]
        bst = xgb.train(
            params, dtrain, num_boost_round=5000, evals=watchlist, 
            early_stopping_rounds=100, verbose_eval=VERBOSE
        )
        # Corrected line in train_xgb
        val_pred = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
        oof[val_idx] = val_pred
        preds += bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1)) / n_splits

        models.append(bst)
        print("Fold ROC AUC:", roc_auc_score(y[val_idx], val_pred))
    print("XGB OOF ROC AUC:", roc_auc_score(y, oof))
    return oof, preds, models

def train_cat(X_df, y, X_test_df, cat_cols_names, n_splits=N_FOLDS, seed=SEED):
    oof = np.zeros(len(X_df))
    preds = np.zeros(len(X_test_df))
    models = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=SHUFFLE, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_df, y)):
        print(f"CatBoost Fold {fold+1}")
        X_tr, X_val = X_df.iloc[tr_idx], X_df.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            eval_metric='AUC',
            random_seed=seed+fold,
            od_type='Iter',
            early_stopping_rounds=100,
            verbose=VERBOSE
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_cols_names)
        val_pred = model.predict_proba(X_val)[:,1]
        oof[val_idx] = val_pred
        preds += model.predict_proba(X_test_df)[:,1] / n_splits
        models.append(model)
        print("Fold ROC AUC:", roc_auc_score(y_val, val_pred))
    print("CatBoost OOF ROC AUC:", roc_auc_score(y, oof))
    return oof, preds, models



# Cell 12 — Train models 

# For CatBoost we'll feed DataFrames and pass categorical names
X_df = train_proc[feature_cols].copy()
X_test_df = test_proc[feature_cols].copy()

cat_feature_names_for_cb = [
    c for c in feature_cols 
    if c.startswith('le_') or c in categorical_cols
]

# LightGBM / XGBoost use numpy arrays
feature_names = feature_cols

# Train LightGBM
oof_lgb, pred_lgb, models_lgb = train_lgb(
    X, y, X_test,
    feature_names,
    categorical_feature_indices=None
)

# Train XGBoost
oof_xgb, pred_xgb, models_xgb = train_xgb(
    X, y, X_test,
    feature_names
)

# Train CatBoost (with DataFrame)
oof_cat, pred_cat, models_cat = train_cat(
    X_df, y, X_test_df,
    cat_feature_names_for_cb
)



# Cell 13 — Build stacking dataset (out-of-fold predictions)
# Create OOF matrix (n_samples x n_models), and test preds matrix
oof_stack = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
test_stack = np.vstack([pred_lgb, pred_xgb, pred_cat]).T

print("Stack shapes:", oof_stack.shape, test_stack.shape)
print("Base OOF ROC AUCs:", roc_auc_score(y, oof_lgb), roc_auc_score(y, oof_xgb), roc_auc_score(y, oof_cat))



# Cell 14 — Train meta-model: simple Logistic Regression + optional NN
# Logistic Regression meta (fast, robust)
meta_clf = LogisticRegression(max_iter=2000)
meta_clf.fit(oof_stack, y)
meta_oof_pred = meta_clf.predict_proba(oof_stack)[:,1]
meta_test_pred = meta_clf.predict_proba(test_stack)[:,1]
print("Meta ROC AUC on OOF:", roc_auc_score(y, meta_oof_pred))

# Optionally, build a small NN meta-model for further gains
def train_meta_nn(oof_train, y_train, oof_test, epochs=50, batch_size=1024):
    inp = layers.Input(shape=(oof_train.shape[1],))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = models.Model(inp, out)
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    m.fit(oof_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=2)
    return m

# Uncomment to train NN meta
# meta_nn = train_meta_nn(oof_stack, y, test_stack, epochs=50)
# meta_nn_oof = meta_nn.predict(oof_stack).ravel()
# meta_nn_test = meta_nn.predict(test_stack).ravel()
# print("Meta NN OOF AUC:", roc_auc_score(y, meta_nn_oof))



# Cell 15 — Final ensemble: weighted average of base preds + meta pred
# Simple strategy: give meta a high weight if it's better
# Compute base ensemble simple average
base_average_oof = (oof_lgb + oof_xgb + oof_cat) / 3
base_average_test = (pred_lgb + pred_xgb + pred_cat) / 3
print("Base avg OOF AUC:", roc_auc_score(y, base_average_oof))

# Weighted blend: weigh models by their OOF AUC performance
scores = np.array([roc_auc_score(y, oof_lgb), roc_auc_score(y, oof_xgb), roc_auc_score(y, oof_cat)])
weights = scores / scores.sum()
print("Base model weights:", weights)

ensemble_test_preds = weights[0]*pred_lgb + weights[1]*pred_xgb + weights[2]*pred_cat
# combine with meta
final_test_preds = 0.6 * meta_test_pred + 0.4 * ensemble_test_preds  # adjust blending weights as needed

# Evaluate OOF for final blended with same formula
final_oof_preds = 0.6 * meta_oof_pred + 0.4 * base_average_oof
print("Final blended OOF AUC:", roc_auc_score(y, final_oof_preds))
print("Final blended OOF Accuracy (threshold 0.5):", accuracy_score(y, (final_oof_preds>=0.5).astype(int)))



# Cell 16 — Create and save submission
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: final_test_preds  # probabilities; if required, convert to binary with >=0.5
})
submission.to_csv(OUT_PATH, index=False)
print("Saved submission to:", OUT_PATH)
submission.head()



# Cell 17 — Optional: Optuna tuning example for LightGBM (one function)
# NOTE: Run only if you have time and resources. It will tune on a single fold or CV.
def optuna_lgb_cv(trial, X, y):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 500),
        'max_bin': trial.suggest_int('max_bin', 64, 512),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'seed': SEED
    }
    cv = lgb.cv(param, lgb.Dataset(X, label=y), nfold=3, stratified=True, early_stopping_rounds=50, metrics='auc', seed=SEED)
    return max(cv['auc-mean'])

# Example optuna usage (adjust n_trials)
# study = optuna.create_study(direction='maximize')
# study.optimize(lambda trial: optuna_lgb_cv(trial, X, y), n_trials=30)
# print("Best params:", study.best_params)


