import os
import sys

import numpy as np
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action="ignore", category=PerformanceWarning)

import time
from datetime import datetime

import matplotlib.pyplot as plt

from collections import OrderedDict
from itertools import combinations, permutations, chain

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, log_loss
from scipy.optimize import minimize
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostClassifier

# import torch
# import torch.nn as nn

import optuna


def on_kaggle():
    return "KAGGLE_URL_BASE" in os.environ or os.path.exists("/kaggle/input")

if on_kaggle():
    DATA_DIR = "/kaggle/input/playground-series-s5e7"
else:
    CWD = os.getcwd()
    DATA_DIR = os.path.join(os.getcwd(), "data")

PATH_TO_TRAIN = os.path.join(DATA_DIR, "train.csv")
PATH_TO_TEST = os.path.join(DATA_DIR, "test.csv")
PATH_TO_SS = os.path.join(DATA_DIR, "sample_submission.csv")

df_train = pd.read_csv(PATH_TO_TRAIN)
df_test = pd.read_csv(PATH_TO_TEST)
df_ss = pd.read_csv(PATH_TO_SS)

TARGET = "Personality"


def data_overview(df, target_col="Personality"):
    print("\n=== ALL VARIABLES: DTYPE, Nunique, N_null, Sample ===")
    for col in df.columns:
        print(f"{col:25}  {str(df[col].dtype):10}   nunique={df[col].nunique():4}   missing={df[col].isnull().sum():4}    Example: {df[col].iloc[0]}")
    
    print("\n=== DESCRIBE (Numerics) ===")
    display(df.describe())
    
    print("\n=== Nulls per column ===")
    print(df.isnull().sum())
    
    print("\n=== Most Frequent Values (all cols) ===")
    for col in df.columns:
        most_common = df[col].mode()[0]
        print(f"{col:25}: {most_common}")
    
    # For numeric columns only, show classwise mean
    print("\n=== Classwise Mean (target) ===")
    for col in df.columns:
        if col not in [target_col, 'id'] and np.issubdtype(df[col].dtype, np.number):
            print(f"{col}")
            print(df.groupby(target_col)[col].mean())
            print("-"*24)

# data_overview(df_train, target_col=TARGET)


def plot_logloss_history(logloss_history, figsize=(8,5), title="OOF Logloss per Model Run", y_log=False):
    plt.figure(figsize=figsize)
    n = len(logloss_history)
    plt.plot(range(1, n+1), logloss_history, marker='o', linestyle='-')
    plt.xlabel("Model Run", fontsize=12)
    plt.ylabel("Running OOF Logloss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(range(1, n+1))
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    if y_log:
        plt.yscale('log')
    plt.tight_layout()
    plt.show()

def print_metrics(y_true, oof_preds, target_enc, verbose=False):
    y_pred = np.argmax(oof_preds, axis=1)
    print("-"*60)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Log loss:", log_loss(y_true, oof_preds))
    print("-"*60)
    if verbose:
        print(classification_report(y_true, y_pred, target_names=target_enc.classes_))
    
def print_number_of_features(df_train, X, id_col="id", target_col=TARGET):
    original_features = set(df_train.columns) - {id_col, target_col}
    final_features = set(X.columns)
    engineered_features = final_features - original_features

    print(f"Number of original features: {len(original_features)}")
    print(f"Number of engineered features: {len(engineered_features)}")
    print(f"Total number of features: {len(X.columns)}")
    # print("Engineered features:", engineered_features)

def plot_top_N_features(fi_df, N=30):
    plt.figure(figsize=(8, max(6, N//5)))
    plt.barh(fi_df.feature.head(N)[::-1], fi_df.importance.head(N)[::-1], color="skyblue")
    plt.xlabel('Importance')
    plt.title(f'Top {N} CatBoost Feature Importances')
    plt.tight_layout()
    plt.show()    
    
def plot_top_N_features_horizontal(fi_df, N=10):
    plt.figure(figsize=(max(8, N//2), 6))
    plt.bar(fi_df.feature.head(N), fi_df.importance.head(N), color="skyblue")
    plt.ylabel('Importance')
    plt.title(f'Top {N} CatBoost Feature Importances')
    plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_all_features(fi_df):
    plt.figure(figsize=(10,3))
    plt.plot(fi_df.importance.values, marker='o')
    plt.title('Feature importances ranked (descending)')
    plt.ylabel('Importance')
    plt.xlabel('Rank')
    plt.grid()
    plt.show()

def create_submission(df_test, test_preds, target_enc, filename='submission.csv'):
    y_test_pred = target_enc.inverse_transform(np.argmax(test_preds, axis=1))
    sub = df_test[['id']].copy()
    sub['Personality'] = y_test_pred
    sub.to_csv(filename, index=False)
    print(f"Submission file saved as: {filename}")
    return sub


def preprocess(
    df_train,
    df_test,
    target_col="Personality",
    id_col="id",
    encoder_type="ordinal",
    engineered_features=False
):
    import numpy as np
    import pandas as pd
    from itertools import permutations, combinations
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
    from sklearn.impute import SimpleImputer
    from collections import OrderedDict

    # -------- TRACKERS --------
    cat_features = []
    bin_features = []
    group_bin_features = []
    cross_features = []
    num_features = []

    # --- Helper: parse engineered_features
    FEATURE_KEYS = [
        'pairwise', 'log', 'freq', 'cross', 'binning', 'grp_stats', 'row_stats',
        'rank', 'ntile', 'cat_inter_freq', 'outlier', 'pca'
    ]
    if isinstance(engineered_features, bool):
        use_features = FEATURE_KEYS if engineered_features else []
    else:
        use_features = list(engineered_features)

    # Remove id/target columns BEFORE combining (avoid leakage!)
    # y = df_train[target_col].copy()
    y = pd.Series(df_train[target_col].values, name=target_col)
    df_train = df_train.drop([id_col, target_col], axis=1)
    df_test = df_test.drop([id_col], axis=1)

    n_train = len(df_train)
    data_all = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)

    # --- Identify base features
    fe_cat = data_all.select_dtypes(include=['object', 'category']).columns.tolist()
    fe_num = data_all.select_dtypes(include=[np.number]).columns.tolist()
    cat_features.extend(fe_cat)
    num_features.extend(fe_num)

    data_all[fe_cat] = data_all[fe_cat].fillna('NA')
    data_all[fe_num] = data_all[fe_num].fillna(0)

    epsilon = 1e-3

    # ==============================
    #  ENGINEERED FEATURES SECTIONS
    # ==============================
    # --- 1. Pairwise numerical features ---
    if 'pairwise' in use_features:
        for col1, col2 in permutations(fe_num, 2):
            if col1 == col2: continue
            data_all[f"{col1}_minus_{col2}"] = data_all[col1] - data_all[col2]
            data_all[f"{col1}_div_{col2}"] = data_all[col1] / (data_all[col2] + epsilon)

    # --- 2. Logarithmic/sqrt transformations ---
    if 'log' in use_features:
        for col in fe_num:
            data_all[f"{col}_log"] = np.log(data_all[col] + 1e-6)
            data_all[f"{col}_sqrt"] = np.sqrt(np.clip(data_all[col], 0, None))

    # --- 3. Frequency/count encoding for categoricals ---
    if 'freq' in use_features:
        for col in fe_cat:
            vc = data_all[col].value_counts()
            data_all[f"{col}_count"] = data_all[col].map(vc)
            data_all[f"{col}_freq"] = data_all[col].map(vc) / len(data_all)

    # --- 4. Categorical cross features ("feature crosses") ---
    if 'cross' in use_features:
        for col1, col2 in combinations(fe_cat, 2):
            cross_name = f"{col1}__X__{col2}"
            data_all[cross_name] = data_all[col1].astype(str) + "_" + data_all[col2].astype(str)
            cross_features.append(cross_name)

    # --- 5. Numeric binning + cross-binning ---
    if 'binning' in use_features:
        for col in fe_num:
            for q in [2, 3, 4, 5, 7, 10, 20]:  # More bins for richer features
                bin_name = f"{col}_bin{q}"
                data_all[bin_name] = pd.qcut(data_all[col], q, duplicates='drop', labels=False)
                bin_features.append(bin_name)
        for cat in fe_cat:
            for col in fe_num:
                binname = f"{col}_bin_grpby_{cat}"
                data_all[binname] = data_all.groupby(cat)[col].transform(
                    lambda x: pd.qcut(x, 4, duplicates='drop', labels=False)
                )
                group_bin_features.append(binname)

    # --- 6. Rowwise statistics ---
    if 'row_stats' in use_features:
        data_all["num_nan"] = data_all[fe_num].isna().sum(axis=1)
        data_all["cat_nan"] = data_all[fe_cat].isna().sum(axis=1)
        data_all["num_sum"] = data_all[fe_num].sum(axis=1)
        data_all["num_mean"] = data_all[fe_num].mean(axis=1)
        data_all["num_nan_frac"] = data_all[fe_num].isna().mean(axis=1)
        data_all["cat_nan_frac"] = data_all[fe_cat].isna().mean(axis=1)
        # New:
        data_all["num_row_std"] = data_all[fe_num].std(axis=1)
        data_all["num_row_min"] = data_all[fe_num].min(axis=1)
        data_all["num_row_max"] = data_all[fe_num].max(axis=1)
        data_all["num_row_range"] = data_all["num_row_max"] - data_all["num_row_min"]

    # --- 7. Groupby statistics (per numeric, per cat) ---
    if 'grp_stats' in use_features:
        for cat in fe_cat:
            for num in fe_num:
                group = data_all.groupby(cat)[num]
                data_all[f"{num}_grpby_{cat}_mean"] = group.transform("mean")
                data_all[f"{num}_grpby_{cat}_std"] = group.transform("std")
                data_all[f"{num}_grpby_{cat}_min"] = group.transform("min")
                data_all[f"{num}_grpby_{cat}_max"] = group.transform("max")
                data_all[f"{num}_grpby_{cat}_median"] = group.transform("median")
                data_all[f"{num}_grpby_{cat}_skew"] = group.transform("skew")
                data_all[f"{num}_grpby_{cat}_count"] = group.transform("count")
                data_all[f"{num}_grpby_{cat}_q25"] = group.transform(lambda x: x.quantile(0.25))
                data_all[f"{num}_grpby_{cat}_q75"] = group.transform(lambda x: x.quantile(0.75))

    # --- 8. Rank Features (NEW) ---
    if 'rank' in use_features:
        for col in fe_num:
            data_all[f"{col}_rank"] = data_all[col].rank(method="average") / len(data_all)

    # --- 10. Quantile/Ntile Features ---
    if 'ntile' in use_features:
        for col in fe_num:
            data_all[f"{col}_ntile10"] = pd.qcut(data_all[col], 10, labels=False, duplicates='drop')

    # --- 11. Interaction Counts for Cats (NEW) ---
    if 'cat_inter_freq' in use_features:
        if len(fe_cat) >= 2:
            for c1, c2 in combinations(fe_cat, 2):
                freq = data_all.groupby([c1, c2]).size()
                name = f"{c1}__{c2}_freq"
                data_all[name] = data_all[[c1, c2]].apply(lambda x: freq.get(tuple(x), 0), axis=1)
    # --- 12. Outlier Flags (NEW) ---
    if 'outlier' in use_features:
        for col in fe_num:
            q1, q3 = data_all[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            data_all[f"{col}_is_outlier"] = ((data_all[col] < (q1 - 1.5*iqr)) | (data_all[col] > (q3 + 1.5*iqr))).astype(int)

    # --- 13. PCA Features ---
    if 'pca' in use_features and len(fe_num) > 2:
        from sklearn.decomposition import PCA
        n_pca = min(3, len(fe_num))
        comps = PCA(n_components=n_pca).fit_transform(data_all[fe_num])
        for i in range(n_pca):
            data_all[f"pca{i+1}"] = comps[:, i]
    # ==============================
    #  END OF ENGINEERED FEATURES SECTIONS
    # ==============================


    # Identify all available pseudo-categorical features, including bins/crosses
    pseudo_cats = cat_features + bin_features + group_bin_features + cross_features
    pseudo_cats = list(OrderedDict.fromkeys([f for f in pseudo_cats if f in data_all.columns]))  # safety


    # --------- SPLIT BACK -----------
    X = data_all.iloc[:n_train].reset_index(drop=True)
    X_test = data_all.iloc[n_train:].reset_index(drop=True)

    # ---------------------------------
    #  ENCODING: Ordinal for ALL pseudo-cats
    # ---------------------------------
    all_pseudo_cat = cat_features + bin_features + group_bin_features + cross_features
    all_pseudo_cat = list(OrderedDict.fromkeys([f for f in all_pseudo_cat if f in X.columns]))
    for c in all_pseudo_cat:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    if encoder_type == "ordinal":
        enc = OrdinalEncoder()
        X_all = pd.concat([X, X_test], axis=0, ignore_index=True)
        X_all[all_pseudo_cat] = enc.fit_transform(X_all[all_pseudo_cat])
        X = X_all.iloc[:len(X)].reset_index(drop=True)
        X_test = X_all.iloc[len(X):].reset_index(drop=True)
    else:
        raise NotImplementedError("Only ordinal implemented")

    # Encode target
    target_enc = LabelEncoder()
    # y_enc = target_enc.fit_transform(y)
    y_enc = pd.Series(target_enc.fit_transform(y), index=y.index, name=target_col)

    # Ensure all columns numeric
    X = X.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

    # Return all feature trackers for downstream use
    trackers = {
        "cat_features": cat_features,
        "bin_features": bin_features,
        "group_bin_features": group_bin_features,
        "cross_features": cross_features,
        "num_features": num_features,
        "all_pseudo_cat": all_pseudo_cat,
    }
    return X, y_enc, X_test, target_enc, trackers


def add_cross_features(df, feature_lists, degree=2, max_unique=200, base_keys=None):
    """
    Adds crosses of given degree for the feature sets specified by base_keys in feature_lists.
    """
    new_features = []
    if base_keys is None:
        base_keys = ['cat_features', 'bin_features', 'group_bin_features', 'cross_features']
    cross_bases = []
    for key in base_keys:
        cross_bases += feature_lists.get(key, [])
    cross_bases = list(dict.fromkeys(cross_bases))
    for comb in combinations(cross_bases, degree):
        colname = "__X__".join(comb)
        crossed = df[list(comb)].astype(str).agg("_".join, axis=1)
        if crossed.nunique() <= max_unique:
            df[colname] = crossed
            new_features.append(colname)
    return new_features


def add_bins(df, numerics, bin_sizes=[2, 3, 5, 10, 20], prefix="bin"):
    """Add quantile bins for each column in numerics."""
    new_bins = []
    for col in numerics:
        col_data = df[col]
        for b in bin_sizes:
            bname = f"{col}_{prefix}{b}"
            try:
                df[bname] = pd.qcut(col_data, b, labels=False, duplicates='drop')
                new_bins.append(bname)
            except ValueError:
                # Not enough unique values, skip this bin size
                continue
    return new_bins


def add_categorical_encodings(
    X, y, X_test, cat_cols,
    encodings=('label', 'mean', 'median', 'min', 'max', 'nunique', 'count'),
    n_splits=5,
    smoothing=10,
):
    """
    For each categorical column, add label/target/count encodings:
        - label: ordinal (label) encoding
        - mean, median, min, max: target encodings
        - nunique: #unique target classes
        - count: category frequency
    Returns: columns to add to X and X_test
    """
    X_enc = pd.DataFrame(index=X.index)
    X_test_enc = pd.DataFrame(index=X_test.index)
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
    import warnings

    if 'label' in encodings:
        # LABEL ENCODING
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_enc[[c+'_le' for c in cat_cols]] = le.fit_transform(X[cat_cols])
        X_test_enc[[c+'_le' for c in cat_cols]] = le.transform(X_test[cat_cols])

    if any(e in encodings for e in ['mean','median','min','max','nunique']):
        # Out-of-fold TE for mean, median, min, max, nunique
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for col in cat_cols:
            global_mean = y.mean()
            for agg in ['mean','median','min','max','nunique']:
                colname = f'{col}_te_{agg}'
                X_enc[colname] = np.nan
                X_test_enc[colname] = 0
                for train_idx, val_idx in kf.split(X):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                    # For nunique in classification, counts number of unique y observed per cat value
                    if agg == 'nunique':
                        stat = y_tr.groupby(X_tr[col]).nunique()
                        fill = y_tr.nunique()
                    else:
                        stat = getattr(y_tr.groupby(X_tr[col]), agg)()
                        fill = y_tr.agg(agg)
                    X_enc.iloc[val_idx, X_enc.columns.get_loc(colname)] = X_val[col].map(stat).fillna(fill).values
                # On test, use full train stats
                if agg == 'nunique':
                    stat = y.groupby(X[col]).nunique()
                    fill = y.nunique()
                else:
                    stat = getattr(y.groupby(X[col]), agg)()
                    fill = y.agg(agg)
                X_test_enc[colname] = X_test[col].map(stat).fillna(fill).values

    if 'count' in encodings:
        for col in cat_cols:
            vc = X[col].value_counts()
            X_enc[f'{col}_ce'] = X[col].map(vc).fillna(0)
            X_test_enc[f'{col}_ce'] = X_test[col].map(vc).fillna(0)

    return X_enc, X_test_enc


def xgboost_cv(
    X, y_enc, params, N_SEEDS=5, N_FOLDS=5, X_test=None, verbose=True
):
    n_classes = len(np.unique(y_enc))
    is_binary = n_classes == 2
    n_samples = len(X)
    oof_preds = np.zeros((n_samples, n_classes))
    oof_counts = np.zeros(n_samples, dtype=int)

    if X_test is not None:
        n_test = len(X_test)
        test_preds = np.zeros((n_test, n_classes))
    else:
        test_preds = None

    logloss_history = []
    model_count = 0

    # Impute (XGBoost can't handle NaN)
    imputer = SimpleImputer(strategy='median')
    X_array = imputer.fit_transform(X)
    if X_test is not None:
        X_test_array = imputer.transform(X_test)

    for seed in range(N_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_array, y_enc)):
            model_count += 1
            params_this = params.copy()
            params_this['random_state'] = seed * 1000 + fold

            clf = xgb.XGBClassifier(**params_this)
            clf.fit(
                X_array[train_idx], y_enc[train_idx],
                eval_set=[(X_array[val_idx], y_enc[val_idx])],
                verbose=0
            )
            # Predict on validation set
            val_pred_proba = clf.predict_proba(X_array[val_idx])
            # --- Handle possible shape issues if binary ---
            if is_binary and val_pred_proba.shape[1] == 1:
                # Should not occur with proper XGBClassifier
                val_pred_proba = np.concatenate([1-val_pred_proba, val_pred_proba], axis=1)
            elif is_binary and val_pred_proba.shape[1] == 2:
                pass  # Standard
            elif val_pred_proba.shape[1] != n_classes:
                raise ValueError(f"pred.shape={val_pred_proba.shape}, expected {n_classes} columns")

            oof_preds[val_idx] += val_pred_proba
            oof_counts[val_idx] += 1

            # For test set ensemble
            if X_test is not None:
                test_pred_proba = clf.predict_proba(X_test_array)
                if is_binary and test_pred_proba.shape[1] == 1:
                    test_pred_proba = np.concatenate([1-test_pred_proba, test_pred_proba], axis=1)
                elif is_binary and test_pred_proba.shape[1] == 2:
                    pass
                elif test_pred_proba.shape[1] != n_classes:
                    raise ValueError(f"test_pred.shape={test_pred_proba.shape}, expected {n_classes} columns")
                test_preds += test_pred_proba

            # Metrics: OOF logloss for all currently predicted indices
            mask = oof_counts > 0
            running_oof = np.zeros_like(oof_preds)
            # Average across number of predictions for each row
            running_oof[mask] = oof_preds[mask] / oof_counts[mask, None]
            
            # Normalize each row to sum to 1 (prevents logloss warning)
            running_oof[mask] = np.clip(running_oof[mask], 1e-10, 1-1e-10)
            row_sums = running_oof[mask].sum(axis=1, keepdims=True)
            running_oof[mask] = running_oof[mask] / row_sums
            # Now running_oof[mask] rows are guaranteed to sum to 1.

            logl = log_loss(y_enc[mask], running_oof[mask])  # No more warning!
            if verbose:
                acc = accuracy_score(y_enc[mask], np.argmax(running_oof[mask], axis=1))
                print(f"Model {model_count:02d}/{N_SEEDS*N_FOLDS:02d} | Seed: {seed+1:<2d} | "
                      f"Fold: {fold+1:<2d} | OOF Accuracy: {acc:.5f} | OOF Logloss: {logl:.5f}")
            logloss_history.append(logl)

    # Final normalization: combine all seeds/folds OOF
    mask = oof_counts > 0
    oof_preds_final = np.zeros_like(oof_preds)
    oof_preds_final[mask] = oof_preds[mask] / oof_counts[mask, None]
    
    oof_preds_final[mask] = np.clip(oof_preds_final[mask], 1e-10, 1-1e-10)
    row_sum = oof_preds_final[mask].sum(axis=1, keepdims=True)
    oof_preds_final[mask] = oof_preds_final[mask] / row_sum

    probsums = oof_preds_final[mask].sum(axis=1)
    if not np.allclose(probsums, 1, atol=1e-7):
        print("WARNING: Some row(s) not summing to 1 after normalization!", probsums)

    if X_test is not None and model_count > 0:
        test_preds /= model_count
        # For test_preds: always normalize/clipping too
        test_preds = np.clip(test_preds, 1e-10, 1-1e-10)
        row_sum_test = test_preds.sum(axis=1, keepdims=True)
        test_preds = test_preds / row_sum_test

    return oof_preds_final, test_preds, logloss_history


# Perform test of train and inference with "standard" engineered features
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_leaves': 25,
    'min_child_weight': 0.0034,
    'learning_rate': 0.005,
    'n_estimators': 10000,
    'subsample': 0.8,
    'colsample_bytree': 0.87,
    'colsample_bylevel': 0.83,
    'reg_alpha': 0.0029,
    'reg_lambda': 27.1,
    'tree_method': 'hist',
    'early_stopping_rounds': 50
}

# FEATURE_KEYS for reference
FEATURE_KEYS = [
        'pairwise', 'log', 'freq', 'cross', 'binning', 'grp_stats', 'row_stats',
        'rank', 'ntile', 'cat_inter_freq', 'outlier', 'pca'
    ]
X, y_enc, X_test, target_enc, trackers = preprocess(df_train, df_test, engineered_features=True)
print_number_of_features(df_train, X)

oof_preds, test_preds, logloss_history = xgboost_cv(X, y_enc, xgb_params, N_SEEDS=3, N_FOLDS=5, X_test=X_test, verbose=True)


print_metrics(y_enc, oof_preds, target_enc)
plot_logloss_history(logloss_history, y_log=False)
submission = create_submission(df_test, test_preds, target_enc)
submission.head()


####################
# 1. PREPROCESS
####################
X, y_enc, X_test, target_enc, trackers = preprocess(df_train, df_test, engineered_features=True)

####################
# 2. SELECT BASE CATEGORICALS FOR ENCODING
####################
# You can also add crossing features here! For now: all pseudo-cats and their bins/grouped
cat_cols = (
    trackers['cat_features'] +
    trackers['bin_features'] +
    trackers['group_bin_features'] +
    trackers['cross_features']
)
cat_cols = [c for c in cat_cols if c in X.columns and X[c].nunique() < 200]

# Force string dtype for all base cats (important for non-numeric counts/TE)
for c in cat_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)

####################
# 3. MULTI-HEAD ENCODINGS: label, CE, TE-mean, TE-median, TE-min, TE-max, TE-nunique
####################

def multihead_cat_encodings(X, y, X_test, cat_cols, n_splits=5):
    # Will create columns: col_le, col_ce, col_te_mean, ..., for each col in cat_cols
    X_enc = pd.DataFrame(index=X.index)
    X_test_enc = pd.DataFrame(index=X_test.index)

    # Label encoding (ordinal)
    enc_le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_enc[[c + '_le' for c in cat_cols]] = enc_le.fit_transform(X[cat_cols])
    X_test_enc[[c + '_le' for c in cat_cols]] = enc_le.transform(X_test[cat_cols])

    # Count encoding (frequency)
    for c in cat_cols:
        vc = X[c].value_counts()
        X_enc[c + '_ce'] = X[c].map(vc).fillna(0)
        X_test_enc[c + '_ce'] = X_test[c].map(vc).fillna(0)

    # OOF target encodings (mean, median, min, max, nunique)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    agg_list = ['mean','median','min','max','nunique']
    for c in cat_cols:
        for agg in agg_list:
            cname = f'{c}_te_{agg}'
            X_enc[cname] = np.nan
            X_test_enc[cname] = 0
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                if agg == 'nunique':
                    stat = y_tr.groupby(X_tr[c]).nunique()
                    fill = y_tr.nunique()
                else:
                    stat = getattr(y_tr.groupby(X_tr[c]), agg)()
                    fill = y_tr.agg(agg)
                X_enc.iloc[val_idx, X_enc.columns.get_loc(cname)] = X_val[c].map(stat).fillna(fill).values
            # On test set
            if agg == 'nunique':
                stat = y.groupby(X[c]).nunique()
                fill = y.nunique()
            else:
                stat = getattr(y.groupby(X[c]), agg)()
                fill = y.agg(agg)
            X_test_enc[cname] = X_test[c].map(stat).fillna(fill).values

    return X_enc, X_test_enc

# Make sure y_enc is a Series
if not isinstance(y_enc, pd.Series):
    y_enc = pd.Series(y_enc, name="target")

X_enc, X_test_enc = multihead_cat_encodings(X, y_enc, X_test, cat_cols, n_splits=5)

# Combine with X:
X_mega = pd.concat([X, X_enc], axis=1)
X_test_mega = pd.concat([X_test, X_test_enc], axis=1)

####################
# 4. FEATURE SETS FOR ABLATION
####################
# Classical/engineered only
engineered_cols = [c for c in X.columns if not (c.endswith('_te_mean') or c.endswith('_te_median') or c.endswith('_te_min') or c.endswith('_te_max') or c.endswith('_te_nunique') or c.endswith('_le') or c.endswith('_ce'))]
X_engineered = X[engineered_cols]
X_test_engineered = X_test[engineered_cols]

# Only TE/encoding features
te_feat_cols = [c for c in X_mega.columns if (c.endswith('_te_mean') or c.endswith('_te_median') or c.endswith('_te_min') or c.endswith('_te_max') or c.endswith('_te_nunique') or c.endswith('_le') or c.endswith('_ce'))]
X_te_feats = X_mega[te_feat_cols]
X_test_te_feats = X_test_mega[te_feat_cols]

# All features (classical + engineered + all encodings)
X_all = X_mega
X_test_all = X_test_mega

####################
# 5. PRINT FEATURE STATS
####################
print(f"Classical/engineered-only features: {len(X_engineered.columns)}")
print(f"Multi-encoded TE features: {len(X_te_feats.columns)}")
print(f"ALL features: {len(X_all.columns)}")

####################
# 6. MODELING WITH YOUR xgboost_cv
####################
print("-- Classical/engineered only --")
oof_eng, test_eng, hist_eng = xgboost_cv(
    X_engineered, y_enc, xgb_params, N_SEEDS=3, N_FOLDS=5, X_test=X_test_engineered, verbose=True
)
print("Logloss (eng):", np.mean(hist_eng))

print("-- Only TE/multi-encoded features --")
oof_te, test_te, hist_te = xgboost_cv(
    X_te_feats, y_enc, xgb_params, N_SEEDS=3, N_FOLDS=5, X_test=X_test_te_feats, verbose=True
)
print("Logloss (te):", np.mean(hist_te))

print("-- All features --")
oof_all, test_all, hist_all = xgboost_cv(
    X_all, y_enc, xgb_params, N_SEEDS=3, N_FOLDS=5, X_test=X_test_all, verbose=True
)
print("Logloss (all):", np.mean(hist_all))


print_metrics(y_enc, oof_preds, target_enc)
plot_logloss_history(hist_all, y_log=False)
submission = create_submission(df_test, test_all, target_enc)
submission.head()


def optuna_xgboost_objective(trial, X, y_enc):
    start = time.time()
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 4),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "n_jobs": -1,
        "objective": "multi:softprob" if len(np.unique(y_enc)) > 2 else "binary:logistic",
        "eval_metric": "logloss",
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        # "random_state": 42,  # Set in xgboost_cv
        'early_stopping_rounds': trial.suggest_int("early_stopping_rounds", 20, 50)
    }
    if trial.number == 0:
        print_number_of_features(df_train, X)

    oof_preds, _, _ = xgboost_cv(X, y_enc, params, N_SEEDS=2, N_FOLDS=5, X_test=None, verbose=0)
    score = log_loss(y_enc, oof_preds)
    acc = accuracy_score(y_enc, oof_preds.argmax(axis=1))
    end = time.time()
    elapsed = end - start

    tstring = datetime.now().strftime('%H:%M:%S')
    print(f"[{tstring}] Trial {trial.number+1 if hasattr(trial,'number') else '?'}: "
          f"Logloss={score:.5f}, Accuracy={acc:.5f}, "
          f"Time: {elapsed:.2f}s, Params: {params}")
    return score

study = optuna.create_study(direction="minimize")
RUN_OPTUNA = False

if RUN_OPTUNA:
    study.optimize(lambda trial: optuna_xgboost_objective(trial, X_all, y_enc), n_trials=1000)
    best_params = study.best_params.copy()
    print("Best params:", study.best_params)
    print("Best logloss:", study.best_value)
else:
    # Established best_params from my own optimisation study performed locally
    best_params = {
        'n_estimators': 5324,
        'learning_rate': 0.1885703879967384,
        'max_depth': 4,
        'subsample': 0.766193779962779,
        'colsample_bytree': 0.8669102182411459,
        'gamma': 2.249829704055976,
        'reg_alpha': 3.3819275721779194,
        'reg_lambda': 0.8170308707958724,
        'min_child_weight': 3.2546808134220346,
        'early_stopping_rounds': 48
    }


oof_preds, test_preds, logloss_history = xgboost_cv(X_all, y_enc, best_params, N_SEEDS=10, N_FOLDS=10, X_test=X_test_all, verbose=True)

print_metrics(y_enc, oof_preds, target_enc)
plot_logloss_history(logloss_history)

submission = create_submission(df_test, test_preds, target_enc, filename='submission.csv')
submission.head()




