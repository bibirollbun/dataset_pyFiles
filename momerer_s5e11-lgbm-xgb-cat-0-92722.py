import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression # NEW: For Stacking
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from scipy.optimize import minimize
import gc

# Set display options for debugging
pd.set_option('display.max_columns', None)

# =================================================================================
# Load Data
# =================================================================================
print("Loading data...")
try:
    train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
    orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
except FileNotFoundError:
    print("Data files not found. Please check file paths.")
    # Fallback for local execution (optional)
    # ...

print(f'Initial Train Shape: {train.shape}')
print(f'Initial Test Shape: {test.shape}')
print(f'Initial Orig Shape: {orig.shape}')

TARGET = 'loan_paid_back'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
NUMS = [col for col in BASE if col not in CATS]

# =================================================================================
# Feature Engineering Part 0 - Domain-Specific Features
# =================================================================================
print("\nCreating domain-specific features...")
DOMAIN = []
for df in [train, test, orig]:
    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1e-6)
    if 'credit_score' in df.columns and 'loan_amount' in df.columns:
        df['credit_score_to_loan_ratio'] = df['credit_score'] / (df['loan_amount'] + 1e-6)
    if 'annual_income' in df.columns:
        df['income_per_month'] = df['annual_income'] / 12
    if 'loan_amount' in df.columns and 'loan_term_months' in df.columns:
        df['loan_per_month'] = df['loan_amount'] / (df['loan_term_months'] + 1e-6)
        if 'income_per_month' in df.columns:
            df['income_after_loan'] = df['income_per_month'] - df['loan_per_month']
    if 'credit_score' in df.columns and 'annual_income' in df.columns:
        df['credit_score_x_income'] = df['credit_score'] * df['annual_income']
    if 'loan_term_months' in df.columns and 'credit_score' in df.columns:
        df['loan_term_x_credit_score'] = df['loan_term_months'] * df['credit_score']

DOMAIN = ['loan_to_income_ratio', 'loan_per_month', 'income_per_month', 
            'income_after_loan', 'credit_score_x_income', 'credit_score_to_loan_ratio',
            'loan_term_x_credit_score']
print(f'{len(DOMAIN)} DOMAIN Features created.')


# =================================================================================
# Feature Engineering Part 1 - Aggregation Features (EXPANDED)
# =================================================================================
print("\nCreating aggregation features...")
AGG = []
# NEW: Expanded list of numerical features to aggregate
AGG_NUMS = [
    'annual_income', 'loan_amount', 'credit_score', 'loan_term_months',
    'loan_to_income_ratio', 'loan_per_month', 'income_after_loan', 'credit_score_to_loan_ratio'
]
AGG_CATS = ['grade_subgrade', 'employment_status', 'education_level', 'loan_purpose', 'marital_status']

for df in [train, test, orig]:
    for cat in AGG_CATS:
        if cat not in df.columns: continue
        for num in AGG_NUMS:
            if num not in df.columns: continue
            
            interaction_name = f'{cat}_x_{num}'
            
            # Mean
            new_col_mean = f'agg_mean_{interaction_name}'
            if new_col_mean not in AGG: AGG.append(new_col_mean)
            mean_map = df.groupby(cat)[num].transform('mean')
            df[new_col_mean] = mean_map
            
            # Std
            new_col_std = f'agg_std_{interaction_name}'
            if new_col_std not in AGG: AGG.append(new_col_std)
            std_map = df.groupby(cat)[num].transform('std')
            df[new_col_std] = std_map

            # Diff from mean
            new_col_diff = f'agg_diff_{interaction_name}'
            if new_col_diff not in AGG: AGG.append(new_col_diff)
            df[new_col_diff] = df[num] - df[new_col_mean]

print(f'{len(AGG)} AGG Features created (unique).')


# =================================================================================
# Feature Engineering Part 2 - Bigram Interactions
# =================================================================================
print("\nCreating bigram interaction features...")
INTER = []
TE_BASE = [col for col in BASE if col not in ['annual_income', 'loan_amount']]
for col1, col2 in combinations(TE_BASE, 2):
    new_col_name = f'{col1}_{col2}'
    INTER.append(new_col_name)
    for df in [train, test, orig]:
        if col1 in df.columns and col2 in df.columns:
            df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
print(f'{len(INTER)} INTER Features created.')


# =================================================================================
# Feature Engineering Part 3 - Rounding Features
# =================================================================================
print("\nCreating rounding features...")
ROUND = []
rounding_levels = {'1s': 0, '10s': -1}
for col in ['annual_income', 'loan_amount']:
    for suffix, level in rounding_levels.items():
        new_col_name = f'{col}_ROUND_{suffix}'
        ROUND.append(new_col_name)
        for df in [train, test, orig]:
            if col in df.columns:
                df[new_col_name] = df[col].round(level).astype(int)
print(f'{len(ROUND)} ROUND Features created.')


# =================================================================================
# Feature Engineering Part 4 - ORIG Features
# =================================================================================
print("\nCreating ORIG (external data) features...")
ORIG = []
if 'loan_paid_back' not in orig.columns:
    print("WARNING: 'loan_paid_back' target column not in 'orig' dataset. Skipping ORIG features.")
else:
    for col in BASE:
        if col in orig.columns:
            mean_map = orig.groupby(col)[TARGET].mean()
            new_mean_col_name = f"orig_mean_{col}"
            mean_map.name = new_mean_col_name
            train = train.merge(mean_map, on=col, how='left')
            test = test.merge(mean_map, on=col, how='left')
            orig = orig.merge(mean_map, on=col, how='left')
            ORIG.append(new_mean_col_name)
            
            count_map = orig.groupby(col).size().reset_index(name=f"orig_count_{col}")
            new_count_col_name = f"orig_count_{col}"
            train = train.merge(count_map, on=col, how='left')
            test = test.merge(count_map, on=col, how='left')
            orig = orig.merge(count_map, on=col, how='left')
            ORIG.append(new_count_col_name)
print(f'{len(ORIG)} ORIG Features created.')


# =================================================================================
# Setup
# =================================================================================
print("\nSetting up data for training...")
FEATURES = BASE + ORIG + INTER + ROUND + DOMAIN + AGG

common_features_train = [col for col in FEATURES if col in train.columns]
common_features_orig = [col for col in FEATURES if col in orig.columns]
common_features_test = [col for col in FEATURES if col in test.columns]

FINAL_FEATURES = list(set(common_features_train) & set(common_features_orig) & set(common_features_test))
print(f'Total common features used: {len(FINAL_FEATURES)}')

orig_train = orig[common_features_orig + [TARGET]].copy()
train_data = train[common_features_train + [TARGET, 'id']].copy()

orig_train = orig_train[FINAL_FEATURES + [TARGET]]
train_data = train_data[FINAL_FEATURES + [TARGET, 'id']]
X_test_final = test[FINAL_FEATURES].copy()

# Concatenate train and orig data
X = pd.concat([train_data[FINAL_FEATURES], orig_train[FINAL_FEATURES]], ignore_index=True)
y = pd.concat([train_data[TARGET], orig_train[TARGET]], ignore_index=True)

train_ids = train_data['id']
oof_preds_shape = len(train_data) # To slice OOFs later

print(f'Combined X Shape: {X.shape}')
print(f'Combined y Shape: {y.shape}')
print(f'Final Test X Shape: {X_test_final.shape}')

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
CATS_FINAL = [col for col in CATS if col in FINAL_FEATURES]

# NEW: Tuned Parameters (Deeper trees)
params_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8, # From 7 to 8
    'colsample_bytree': 0.4,
    'subsample': 0.6,
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'early_stopping_rounds': 200,
    'random_state': 42,
    'n_jobs': -1,
    'enable_categorical': True,
    'device': 'cuda',
}
params_cat = {
    'iterations': 10000,
    'learning_rate': 0.01,
    'depth': 8, # From 7 to 8
    'l2_leaf_reg': 2.5,
    'random_strength': 1.0,
    'bagging_temperature': 0.5,
    'border_count': 128,
    'task_type': 'GPU',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 200,
}
params_lgbm = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'num_leaves': 48, # From 31 to 48
    'max_depth': 8, # From 7 to 8
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'device': 'gpu',
}


# =================================================================================
# Target Encoder Class
# (No changes)
# =================================================================================
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}
    def fit(self, X, y):
        temp_df = X.copy()
        temp_df['target'] = y
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                if col not in temp_df.columns: continue
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        return self
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            if col not in X.columns: continue
            for agg_func in self.aggs:
                if agg_func not in self.mappings_.get(col, {}): continue
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
        if self.drop_original:
            cols_to_drop = [c for c in self.cols_to_encode if c in X_transformed.columns]
            X_transformed.drop(columns=cols_to_drop, inplace=True)
        return X_transformed
    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train
            for col in self.cols_to_encode:
                if col not in temp_df_train.columns: continue
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    fold_global_stat = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        m = self.smooth
                        if self.smooth == 'auto':
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0 and not pd.isna(avg_variance_within):
                                m = avg_variance_within / variance_between
                            else:
                                m = 0
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
        if self.drop_original:
            cols_to_drop = [c for c in self.cols_to_encode if c in X_transformed.columns]
            X_transformed.drop(columns=cols_to_drop, inplace=True)
        return X_transformed


# =================================================================================
# Train Level 1 Models (XGB, CAT, LGBM)
# =================================================================================
print('\n' + '='*80)
print('TRAINING Level 1 Models (XGB + CAT + LGBM) on Combined Data')
print('='*80)

oof_preds_xgb = np.zeros(len(X))
test_preds_xgb = np.zeros(len(X_test_final))
oof_preds_cat = np.zeros(len(X))
test_preds_cat = np.zeros(len(X_test_final))
oof_preds_lgbm = np.zeros(len(X))
test_preds_lgbm = np.zeros(len(X_test_final))

TE_INTER_FINAL = [col for col in INTER if col in FINAL_FEATURES]
TE_ROUND_FINAL = [col for col in ROUND if col in FINAL_FEATURES]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'\n--- Fold {fold}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test_fold = X_test_final.copy()

    # --- Preprocessing: Target Encoding ---
    # Apply TE per-fold to prevent leakage
    TE = TargetEncoder(cols_to_encode=TE_INTER_FINAL, cv=5, smooth=1.0, aggs=['mean'], drop_original=True)
    X_train = TE.fit_transform(X_train, y_train)
    X_val = TE.transform(X_val)
    X_test_fold = TE.transform(X_test_fold)

    TE2 = TargetEncoder(cols_to_encode=TE_ROUND_FINAL, cv=5, smooth=1.0, aggs=['mean'], drop_original=False)
    X_train = TE2.fit_transform(X_train, y_train)
    X_val = TE2.transform(X_val)
    X_test_fold = TE2.transform(X_test_fold)
    
    TE3 = TargetEncoder(cols_to_encode=CATS_FINAL, cv=5, smooth=1.0, aggs=['mean'], drop_original=False)
    X_train = TE3.fit_transform(X_train, y_train)
    X_val = TE3.transform(X_val)
    X_test_fold = TE3.transform(X_test_fold)
    
    # Fill NA values that might have been created by agg features
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test_fold = X_test_fold.fillna(0)

    # --- Model 1: XGBoost ---
    print("Training XGBoost...")
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = X_test_fold.copy()
    
    for col in CATS_FINAL:
        if col in X_train_xgb.columns:
            X_train_xgb[col] = X_train_xgb[col].astype('category')
            X_val_xgb[col] = X_val_xgb[col].astype('category')
            X_test_xgb[col] = X_test_xgb[col].astype('category')
    
    model_xgb = XGBClassifier(**params_xgb)
    model_xgb.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)
    
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_val_xgb)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test_xgb)[:, 1] / N_SPLITS
    fold_score_xgb = roc_auc_score(y_val, oof_preds_xgb[val_idx])
    print(f'XGBoost AUC: {fold_score_xgb:.5f}')
    del model_xgb, X_train_xgb, X_val_xgb, X_test_xgb
    gc.collect()
    
    # --- Model 2: CatBoost ---
    print("Training CatBoost...")
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    X_test_cat = X_test_fold.copy()
    
    for cat in CATS_FINAL:
        if cat in X_train_cat.columns:
            X_train_cat[cat] = X_train_cat[cat].astype(str)
            X_val_cat[cat] = X_val_cat[cat].astype(str)
            X_test_cat[cat] = X_test_cat[cat].astype(str)
    
    cat_features_indices = [i for i, col in enumerate(X_train_cat.columns) if col in CATS_FINAL]
    
    train_pool = Pool(X_train_cat, y_train, cat_features=cat_features_indices)
    valid_pool = Pool(X_val_cat, y_val, cat_features=cat_features_indices)
    test_pool = Pool(X_test_cat, cat_features=cat_features_indices)
    
    model_cat = CatBoostClassifier(**params_cat)
    model_cat.fit(train_pool, eval_set=valid_pool)
    
    oof_preds_cat[val_idx] = model_cat.predict_proba(valid_pool)[:, 1]
    test_preds_cat += model_cat.predict_proba(test_pool)[:, 1] / N_SPLITS
    fold_score_cat = roc_auc_score(y_val, oof_preds_cat[val_idx])
    print(f'CatBoost AUC: {fold_score_cat:.5f}')
    del model_cat, X_train_cat, X_val_cat, X_test_cat, train_pool, valid_pool, test_pool
    gc.collect()
    
    # --- Model 3: LightGBM ---
    print("Training LightGBM...")
    X_train_lgbm = X_train.copy()
    X_val_lgbm = X_val.copy()
    X_test_lgbm = X_test_fold.copy()
    
    for col in CATS_FINAL:
        if col in X_train_lgbm.columns:
            X_train_lgbm[col] = X_train_lgbm[col].astype('category')
            X_val_lgbm[col] = X_val_lgbm[col].astype('category')
            X_test_lgbm[col] = X_test_lgbm[col].astype('category')
    
    model_lgbm = LGBMClassifier(**params_lgbm)
    model_lgbm.fit(X_train_lgbm, y_train,
                   eval_set=[(X_val_lgbm, y_val)],
                   eval_metric='auc',
                   callbacks=[
                       early_stopping(200, verbose=False),
                       log_evaluation(period=0) # Suppress logs
                   ])
    
    oof_preds_lgbm[val_idx] = model_lgbm.predict_proba(X_val_lgbm)[:, 1]
    test_preds_lgbm += model_lgbm.predict_proba(X_test_lgbm)[:, 1] / N_SPLITS
    fold_score_lgbm = roc_auc_score(y_val, oof_preds_lgbm[val_idx])
    print(f'LightGBM AUC: {fold_score_lgbm:.5f}')
    del model_lgbm, X_train_lgbm, X_val_lgbm, X_test_lgbm
    gc.collect()
    

cv_xgb = roc_auc_score(y, oof_preds_xgb)
cv_cat = roc_auc_score(y, oof_preds_cat)
cv_lgbm = roc_auc_score(y, oof_preds_lgbm)
print(f'\n{"="*80}')
print('Level 1 Model OOF Scores (on Combined Data):')
print(f'XGBoost Overall CV: {cv_xgb:.5f}')
print(f'CatBoost Overall CV: {cv_cat:.5f}')
print(f'LightGBM Overall CV: {cv_lgbm:.5f}')


# =================================================================================
# Ensemble & Save Submission
# NEW: Level 2 Stacking with Meta-Model
# =================================================================================
print('\n' + '='*80)
print('ENSEMBLE (Level 2 Stacking with Meta-Model)')
print('='*80)

# 1. Create Meta-Training Data (OOF Predictions)
X_meta_train = np.vstack([oof_preds_xgb, oof_preds_cat, oof_preds_lgbm]).T
y_meta_train = y # Target is the same combined 'y'

# 2. Create Meta-Test Data (Averaged Test Predictions)
X_meta_test = np.vstack([test_preds_xgb, test_preds_cat, test_preds_lgbm]).T

# 3. Train the Meta-Model
print("Training meta-model (Logistic Regression)...")
# Using a simple, regularized model to prevent overfitting the OOFs
meta_model = LogisticRegression(C=0.1, solver='lbfgs', random_state=42, n_jobs=-1)
meta_model.fit(X_meta_train, y_meta_train)
print("Meta-model trained.")

# 4. Get Final Predictions
oof_preds = meta_model.predict_proba(X_meta_train)[:, 1]
test_preds = meta_model.predict_proba(X_meta_test)[:, 1]

# --- Final Scores ---
overall_auc = roc_auc_score(y, oof_preds)
print(f'\nFinal Ensemble CV (Combined): {overall_auc:.5f}')

# Slice the OOF predictions to match the original 'train' data size
oof_preds_orig_train = oof_preds[:oof_preds_shape]
y_orig_train = y.iloc[:oof_preds_shape]
overall_auc_orig_train = roc_auc_score(y_orig_train, oof_preds_orig_train)
print(f'Final Ensemble CV (Original Train Only): {overall_auc_orig_train:.5f}')

print(f'Expected LB (Estimate): {overall_auc_orig_train + 0.00016:.5f} - {overall_auc_orig_train + 0.00022:.5f}')
print('='*80)

# Save
print("Saving submission and OOF files...")
submission_df = pd.DataFrame({'id': test['id'], TARGET: test_preds})
submission_df.to_csv('submission.csv', index=False)

oof_df = pd.DataFrame({'id': train_ids, TARGET: oof_preds_orig_train})
oof_df.to_csv(f'oof_predictions_{overall_auc_orig_train:.5f}.csv', index=False)

print(f'\n✅ submission.csv saved')
print(f'✅ oof_predictions_{overall_auc_orig_train:.5f}.csv saved')
print("\nProcess completed successfully.")

