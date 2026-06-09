

%load_ext cudf.pandas
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import gc
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from pandas.errors import PerformanceWarning
from sklearn.metrics import roc_auc_score
from itertools import combinations
from xgboost import XGBClassifier
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=PerformanceWarning)

# ====================================================================
# DATA PREPARATION AND FEATURE DEFINITIONS
# ====================================================================

TARGET = 'y'
NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Load datasets: competition train/test + original dataset for augmentation
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')
orig = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', delimiter=';')

# Convert target to binary (yes=1, no=0) for original dataset
orig['y'] = orig['y'].replace({'yes': 1, 'no': 0})

# Convert categorical columns to category dtype for memory efficiency and XGBoost optimization
train[CATS] = train[CATS].astype('category')
test[CATS] = test[CATS].astype('category')
orig[CATS] = orig[CATS].astype('category')

# ====================================================================
# ADVANCED FEATURE ENGINEERING: PAIRWISE COMBINATIONS
# ====================================================================

TE_columns = []
columns = NUMS + CATS

print("Creating pairwise feature combinations...")
for r in [2]:
    for cols in tqdm(list(combinations(columns, r))):
        name = '-'.join(cols)

        # Create combination features by concatenating string representations
        train[name] = train[cols[0]].astype(str)
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype(str)

        test[name] = test[cols[0]].astype(str)
        for col in cols[1:]:
            test[name] = test[name] + '_' + test[col].astype(str)

        orig[name] = orig[cols[0]].astype(str)
        for col in cols[1:]:
            orig[name] = orig[name] + '_' + orig[col].astype(str)
        
        # Apply consistent encoding across datasets
        combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        orig[name] = combined[len(train) + len(test):]

        TE_columns.append(name)

FEATURES = train.columns.tolist()
FEATURES.remove(TARGET)

# ====================================================================
# ADVANCED TARGET ENCODING WITH REGULARIZATION
# ====================================================================

def target_encode_advanced(train, valid, test, col, target=TARGET, kfold=10, smooth=4):
    """
    Advanced target encoding with cross-validation to prevent overfitting
    """
    train['kfold'] = ((train.index) % kfold)
    col_name = '_'.join(col)
    train[f'TE_MEAN_' + col_name] = 0.
    
    np.random.seed(42)
    
    for i in range(kfold):
        df_tmp = train[train['kfold'] != i]
        mn = train[target].mean()
        
        df_tmp = df_tmp[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
        df_tmp.columns = col + ['mean', 'count']
        df_tmp['TE_tmp'] = ((df_tmp['mean'] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
        
        df_tmp_m = train[col + ['kfold', f'TE_MEAN_' + col_name]].merge(df_tmp, how='left', left_on=col, right_on=col)
        df_tmp_m.loc[df_tmp_m['kfold'] == i, f'TE_MEAN_' + col_name] = df_tmp_m.loc[df_tmp_m['kfold'] == i, 'TE_tmp']
        train[f'TE_MEAN_' + col_name] = df_tmp_m[f'TE_MEAN_' + col_name].fillna(mn).values

    # Encode validation and test sets
    df_tmp = train[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
    mn = train[target].mean()
    df_tmp.columns = col + ['mean', 'count']
    df_tmp['TE_tmp'] = ((df_tmp['mean'] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
    
    df_tmp_m = valid[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    valid[f'TE_MEAN_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    valid[f'TE_MEAN_' + col_name] = valid[f'TE_MEAN_' + col_name].astype('float32')

    df_tmp_m = test[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    test[f'TE_MEAN_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    test[f'TE_MEAN_' + col_name] = test[f'TE_MEAN_' + col_name].astype('float32')

    train = train.drop('kfold', axis=1)
    train[f'TE_MEAN_' + col_name] = train[f'TE_MEAN_' + col_name].astype('float32')

    return (train, valid, test)

def count_encode(train, valid, test, col):
    """Count encoding with log transformation"""
    counts = train[col].value_counts()
    
    train[f'CE_{col}'] = np.log1p(train[col].map(counts))
    valid[f'CE_{col}'] = np.log1p(valid[col].map(counts).fillna(0))
    test[f'CE_{col}'] = np.log1p(test[col].map(counts).fillna(0))
    
    return (train, valid, test)

# ====================================================================
# CROSS-VALIDATION WITH IMPROVED EFFICIENCY
# ====================================================================

oof = np.zeros(len(train))
pred = np.zeros(len(test))

rskf = RepeatedStratifiedKFold(n_splits=6, n_repeats=2, random_state=42)
fold_scores = []
models = []

print("\nStarting cross-validation training...")
for idx, (train_idx, val_idx) in enumerate(rskf.split(train, train[TARGET])):
    print(f"\n=== Fold {idx + 1}/10 ===")
    
    X_train, X_val = train.loc[train_idx, FEATURES], train.loc[val_idx, FEATURES]
    y_train, y_val = train.loc[train_idx, TARGET], train.loc[val_idx, TARGET]
    X_test = test.copy()

    # Data augmentation
    X_train_orig = pd.concat([X_train, orig[FEATURES], orig[FEATURES]])
    y_train_orig = pd.concat([y_train, orig[TARGET], orig[TARGET]])

    # Apply feature encoding
    for col in tqdm(TE_columns, desc="Feature Encoding"):
        X_train_orig, X_val, X_test = target_encode_advanced(
            pd.concat([X_train_orig, y_train_orig], axis=1), X_val, X_test, [col], smooth=2
        )
        X_train_orig = X_train_orig.drop(TARGET, axis=1)
        
        X_train_orig, X_val, X_test = count_encode(X_train_orig, X_val, X_test, col)
        
        # Clean up original categorical features
        X_train_orig = X_train_orig.drop(col, axis=1)
        X_val = X_val.drop(col, axis=1)
        X_test = X_test.drop(col, axis=1)
    
    # ====================================================================
    # XGBOOST PARAMETERS
    # ====================================================================
    
    parameters_xgboost = {
        'n_estimators': 8000,         
        'max_leaves': 127,            
        'min_child_weight': 1.5,     
        'max_depth': 100,               
        'grow_policy': 'lossguide',   
        'learning_rate': 0.018,      
        'tree_method': 'hist',        
        'subsample': 0.85,            
        'colsample_bylevel': 0.7,     
        'colsample_bytree': 0.75,       
        'colsample_bynode': 0.85,     
        'sampling_method': 'gradient_based',  
        'reg_alpha': 2.5,             
        'reg_lambda': 0.8,            
        'enable_categorical': True,    
        'max_cat_to_onehot': 1,       
        'device': 'cuda',            
        'n_jobs': -1,                 
        'random_state': 42 + idx,     
        'verbosity': 0,               
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    
    model = XGBClassifier(**parameters_xgboost)
    
    model.fit(
        X_train_orig, y_train_orig, 
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=300
    )
    
    # Generate predictions (no scaling needed for AUC!)
    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    
    oof[val_idx] += val_pred
    pred += test_pred
    
    fold_score = roc_auc_score(y_val, val_pred)
    fold_scores.append(fold_score)
    print(f'Fold {idx + 1}: {fold_score:.6f}')
    
    models.append(model)
    
    # Memory cleanup
    del X_train_orig, X_val, y_train_orig, y_val, X_test
    gc.collect()

# ====================================================================
# FINAL PREDICTIONS - SIMPLIFIED WITHOUT UNNECESSARY SCALING
# ====================================================================

# Normalize OOF predictions (handle overlapping folds)
fold_counts = np.zeros(len(train))
for idx, (train_idx, val_idx) in enumerate(rskf.split(train, train[TARGET])):
    fold_counts[val_idx] += 1
oof = oof / fold_counts

# Average test predictions
pred /= 10

print(f'\nFinal CV AUC: {roc_auc_score(train[TARGET], oof):.6f}')
print(f'CV AUC std: {np.std(fold_scores):.6f}')

# ====================================================================
# SUBMISSION - NO UNNECESSARY OPTIMIZATION
# ====================================================================

submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission['y'] = pred  # Direct use without scaling
submission.to_csv('submission.csv', index=False)

# Save predictions for ensembling
pd.DataFrame({'xgb_oof': oof}).to_csv('xgb_oof.csv', index=False)
pd.DataFrame({'xgb_pred': pred}).to_csv('xgb_pred.csv', index=False)

print("\nTraining complete!")
print("Files saved:")
print("- submission.csv")
print("- xgb_oof.csv (out-of-fold predictions)")  
print("- xgb_pred.csv (test predictions)")










