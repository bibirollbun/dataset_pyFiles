import kagglehub

# Download latest version
path = kagglehub.dataset_download("sushant097/bank-marketing-dataset-full")

print("Path to dataset files:", path)


!pip install --upgrade lightgbm
from lightgbm import early_stopping


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
import random
import os

# ========================== CONFIG & SEEDING ==========================

SEED = 42
N_SPLITS = 6
TARGET = 'y'
PATH_TRAIN = '/kaggle/input/playground-series-s5e8/train.csv'
PATH_TEST = '/kaggle/input/playground-series-s5e8/test.csv'
PATH_SUB = '/kaggle/input/playground-series-s5e8/sample_submission.csv'
PATH_BANK = '/kaggle/input/bank-marketing-dataset-full/bank-full.csv'

def set_global_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_global_seed(SEED)

# ========================== DATA LOADING ==========================

def load_data():
    train = pd.read_csv(PATH_TRAIN, index_col='id')
    test = pd.read_csv(PATH_TEST, index_col='id')
    sample_sub = pd.read_csv(PATH_SUB)
    bank = pd.read_csv(PATH_BANK, sep=';')
    return train, test, sample_sub, bank

# ========================== DATA PREPROCESSING ==========================

def preprocess_data(train, test, bank, target, seed=SEED):
    bank[target] = (bank[target] == 'yes').astype(int)
    
    # Align columns for concat
    missing_cols = set(train.columns) - set(bank.columns)
    for col in missing_cols:
        bank[col] = np.nan
    bank = bank[train.columns]  # align order

    train = pd.concat([train, bank], ignore_index=True)
    train = train.drop_duplicates()

    # Features
    cat_features = train.select_dtypes('object').columns.tolist()
    num_features = [c for c in train.select_dtypes(np.number).columns if c != target]
    
    # Fill NA
    for df in [train, test]:
        df[cat_features] = df[cat_features].fillna('NaN')
        df[num_features] = df[num_features].fillna(df[num_features].median())
    
    # Ordinal encoding
    encoder = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(pd.concat([train[cat_features], test[cat_features]], ignore_index=True))
    train[cat_features] = encoder.transform(train[cat_features])
    test[cat_features]  = encoder.transform(test[cat_features])

    X = train.drop(target, axis=1)
    y = train[target].values
    X_test = test.copy()
    return X, y, X_test, cat_features

# ========================== MODEL DEFINITIONS ==========================

def get_models(seed=SEED):
    models = {
        'XGB': XGBClassifier(
            tree_method='hist', objective='binary:logistic', eval_metric='auc',
            n_estimators=10000, learning_rate=0.0078,
            max_depth=12, reg_lambda=1.43, reg_alpha=5.63, subsample=0.94,
            colsample_bytree=0.71, random_state=seed, verbosity=0,
            early_stopping_rounds=100  # ← here!
        ),
        'LGBM': LGBMClassifier(
            objective='binary', metric='auc', n_estimators=10000, 
            learning_rate=0.0173, max_depth=18, num_leaves=402,
            min_child_samples=97, subsample=0.55, colsample_bytree=0.55,
            reg_alpha=0.018, reg_lambda=5.73, random_state=seed
        ),
    }
    return models

# ========================== TRAINING & OOF PREDICTIONS ==========================

def fit_and_predict(X, y, X_test, cat_features, n_splits=N_SPLITS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = get_models(seed)
    oof_preds = {}
    test_preds = {}

    for name, model in models.items():
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        print(f"\nTraining {name}...")
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
            X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            this_model = clone(model)
            # LightGBM
            if name == "LGBM":
                this_model.fit(
                    X_trn, y_trn,
                    eval_set=[(X_val, y_val)],
                    categorical_feature=cat_features,
                    callbacks=[early_stopping(stopping_rounds=100)],
                )

            # XGBoost
            elif name == "XGB":
                this_model.fit(
                    X_trn, y_trn,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

            else:
                this_model.fit(X_trn, y_trn)
            
            oof[val_idx] = this_model.predict_proba(X_val)[:, 1]
            test_pred += this_model.predict_proba(X_test)[:, 1] / n_splits

            print(f"  Fold {fold+1} AUC: {roc_auc_score(y_val, oof[val_idx]):.5f}")
        
        auc = roc_auc_score(y, oof)
        print(f"{name} OOF AUC: {auc:.5f}")
        oof_preds[name] = oof
        test_preds[name] = test_pred
    return oof_preds, test_preds

# ========================== STACKING ==========================

def stack_and_predict(oof_preds, test_preds, y, seed=SEED):
    # Stack OOF and test predictions for meta-model
    meta_X = np.column_stack([oof_preds[k] for k in oof_preds])
    meta_X_test = np.column_stack([test_preds[k] for k in test_preds])

    meta_model = LogisticRegression(C=0.1, random_state=seed, max_iter=1000)
    meta_model.fit(meta_X, y)
    final_preds = meta_model.predict_proba(meta_X_test)[:, 1]
    return final_preds

# ========================== SUBMISSION ==========================

def make_submission(submission, preds, filename='submission.csv'):
    submission[TARGET] = preds
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")

# ========================== MAIN PIPELINE ==========================

def main():
    train, test, sample_sub, bank = load_data()
    X, y, X_test, cat_features = preprocess_data(train, test, bank, TARGET)
    oof_preds, test_preds = fit_and_predict(X, y, X_test, cat_features)
    final_preds = stack_and_predict(oof_preds, test_preds, y)
    make_submission(sample_sub, final_preds)

if __name__ == '__main__':
    main()


