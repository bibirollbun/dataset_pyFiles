%load_ext cudf.pandas

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# import optuna

import warnings
warnings.filterwarnings('ignore')


TARGET = 'loan_paid_back'
TRAIN  = '/kaggle/input/playground-series-s5e11/train.csv'
TEST   = '/kaggle/input/playground-series-s5e11/test.csv'
SUB    = '/kaggle/input/playground-series-s5e11/sample_submission.csv'
ORG    = '/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv'
OOF    = '/kaggle/input/ps5e11-experiment-tracking/oof13.npy'
PREDS  = '/kaggle/input/ps5e11-experiment-tracking/preds13.npy'
V      =  15


train = pd.read_csv(TRAIN,index_col='id')
test  = pd.read_csv(TEST,index_col='id')

org   = pd.read_csv(ORG)
org   = org[train.columns.to_list()]

print(train.shape)
print(test.shape)
print(org.shape)

display(train.head())
display(test.head())
display(org.head())


NUMS = [col for col in test.columns if train[col].dtype in ['float64','int64']]
CATS = [col for col in test.columns if train[col].dtype in ['O']]

print(f" We have {len(NUMS)} numerical columns and  {len(CATS)} categorical columns")
print(NUMS)
print(CATS)


oof   = np.load(OOF)
preds = np.load(PREDS)

train['y'] = oof
test['y']  = preds

display(train.head())
display(test.head())


cols = NUMS+CATS
def create_frequency_features(df, df_test):
    """
    Add frequency and binning features efficiently.

    - For each categorical column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5, 10, 15 quantile bins.
    """
    # Pre-allocate DataFrames for new features to avoid fragmentation
    freq_features_train = pd.DataFrame(index=df.index)
    freq_features_test = pd.DataFrame(index=df_test.index)
    bin_features_train = pd.DataFrame(index=df.index)
    bin_features_test = pd.DataFrame(index=df_test.index)

    for col in cols:
        # --- Frequency encoding ---
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
        freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

        # --- Quantile binning for numeric columns ---
        if col in NUMS:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    bin_features_train[f"{col}_bin{q}"] = train_bins
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    bin_features_train[f"{col}_bin{q}"] = 0
                    bin_features_test[f"{col}_bin{q}"] = 0

    # Concatenate all new features at once
    df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
    df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)

    return df, df_test

train,test = create_frequency_features(train,test)


def add_subgrade_feature(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the numeric subgrade value from the alphanumeric string column."""

    train = train.copy()
    test = test.copy()
    train['subgrade'] = train['grade_subgrade'].str[1:].astype(int)
    test['subgrade']  = test['grade_subgrade'].str[1:].astype(int)
    train['grade']    = train['grade_subgrade'].str[0].astype('category')
    test['grade']     = test['grade_subgrade'].str[0].astype('category')
    return train, test

train,test = add_subgrade_feature(train,test)


from itertools import combinations

INTER = []
BASE = NUMS+CATS

TE_BASE = [col for col in BASE if col not in ['annual_income', 'loan_amount']]
print(TE_BASE)
for col1, col2 in combinations(TE_BASE, 2):
    new_col_name = f'{col1}_{col2}'
    INTER.append(new_col_name)
    for df in [train, test]:
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
        df[new_col_name] = df[new_col_name].astype('category')
        
print(f'{len(INTER)} INTER Features created.')


ROUND = []

rounding_levels = {
    '1s': 0,   
    '10s': -1,
    '100s':-2
}

for col in ['annual_income', 'loan_amount']:
    for suffix, level in rounding_levels.items():
        new_col_name = f'{col}_ROUND_{suffix}'
        ROUND.append(new_col_name)
        
        for df in [train, test]:
            df[new_col_name] = df[col].round(level).astype(int)

print(f'{len(ROUND)} ROUND Features created.')


ORIG = []

for col in BASE:
    # MEAN
    mean_map = org.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # COUNT
    new_count_col_name = f"orig_count_{col}"
    count_map = org.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

print(f'{len(ORIG)} ORIG Features created.')


FEATURES = CATS + NUMS + ORIG + INTER + ROUND
print(len(FEATURES), 'Features.')
print(FEATURES)


from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder that supports multiple aggregation functions,
    internal cross-validation for leakage prevention, and smoothing.

    Parameters
    ----------
    cols_to_encode : list of str
        List of column names to be target encoded.

    aggs : list of str, default=['mean']
        List of aggregation functions to apply. Any function accepted by
        pandas' `.agg()` method is supported, such as:
        'mean', 'std', 'var', 'min', 'max', 'skew', 'nunique', 
        'count', 'sum', 'median'.
        Smoothing is applied only to the 'mean' aggregation.

    cv : int, default=5
        Number of folds for cross-validation in fit_transform.

    smooth : float or 'auto', default='auto'
        The smoothing parameter `m`. A larger value puts more weight on the 
        global mean. If 'auto', an empirical Bayes estimate is used.
        
    drop_original : bool, default=False
        If True, the original columns to be encoded are dropped.
    """
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        """
        Learn mappings from the entire dataset.
        These mappings are used for the transform method on validation/test data.
        """
        temp_df = X.copy()
        temp_df['target'] = y

        # Learn global statistics for each aggregation
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        # Learn category-specific mappings
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        
        return self

    def transform(self, X):
        """
        Apply learned mappings to the data.
        Unseen categories are filled with global statistics.
        """
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
        
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed

    def fit_transform(self, X, y):
        """
        Fit and transform the data using internal cross-validation to prevent leakage.
        """
        # First, fit on the entire dataset to get global mappings for transform method
        self.fit(X, y)

        # Initialize an empty DataFrame to store encoded features
        encoded_features = pd.DataFrame(index=X.index)
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                # --- Calculate mappings only on the training part of the fold ---
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    
                    # Calculate global stat for this fold
                    fold_global_stat = y_train.agg(agg_func)
                    
                    # Calculate category stats for this fold
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    # --- Apply smoothing only for 'mean' aggregation ---
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        
                        m = self.smooth
                        if self.smooth == 'auto':
                            # Empirical Bayes smoothing
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0  # No smoothing if no variance between groups
                        
                        # Apply smoothing formula
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    
                    # Store encoded values for the validation fold
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        # Merge with original DataFrame
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
            
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed


combined = pd.concat([train,test],axis=0,ignore_index=True)
for col in CATS:
    if col not in test.columns:
        continue
    combined[col] = combined[col].astype('category')
# for col in INTER:
#     combined[col],_ = pd.factorize(combined[col])

train = combined[:len(train)]
test  = combined[-len(test):]


# X = train[FEATURES]
# y = train[TARGET]
# y_= train['y']


# import optuna
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import xgboost as xgb
# import numpy as np
# import pandas as pd


# def objective(trial, X, y_true, y_pred_prev=None, n_splits=5, seed=42):
#     """
#     Objective for Optuna that tunes residual-based XGBoost using 5-fold CV.
#     Evaluates RMSE on actual target (not residuals).

#     Parameters
#     ----------
#     X : pd.DataFrame or np.ndarray
#         Feature matrix
#     y_true : np.ndarray
#         True target values
#     y_pred_prev : np.ndarray or None
#         Previous model predictions (for residual correction)
#     n_splits : int
#         Number of CV folds
#     """
#     # --- Residual setup ---
#     if y_pred_prev is None:
#         y_pred_prev = np.zeros_like(y_true)

#     # --- Hyperparameter search space ---
#     param = {
#         "objective": "reg:squarederror",
#         "eval_metric": "rmse",
#         "tree_method": "hist",
#         "device": "cuda",
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "min_child_weight": trial.suggest_float("min_child_weight", 1, 15),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
#         "gamma": trial.suggest_float("gamma", 0.0, 5.0),
#         "max_leaves": trial.suggest_int("max_leaves", 4, 32),
#     }

#     # --- 5-Fold CV ---
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     rmse_scores = []

#     for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
#         X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
#         y_train, y_valid = y_true[train_idx], y_true[valid_idx]
#         y_pred_prev_train, y_pred_prev_valid = y_pred_prev[train_idx], y_pred_prev[valid_idx]

#         # Residuals for training
#         y_resid_train = y_train - y_pred_prev_train
#         y_resid_valid = y_valid - y_pred_prev_valid

#         dtrain = xgb.DMatrix(X_train, label=y_resid_train, enable_categorical=True)
#         dvalid = xgb.DMatrix(X_valid, label=y_resid_valid, enable_categorical=True)

#         # Train residual model
#         model = xgb.train(
#             params=param,
#             dtrain=dtrain,
#             num_boost_round=2000,
#             evals=[(dtrain, "train"), (dvalid, "valid")],
#             early_stopping_rounds=100,
#             verbose_eval=False,
#         )

#         # Predict residuals
#         preds_resid = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))

#         # Update cumulative predictions
#         preds_total = y_pred_prev_valid + preds_resid

#         # Compute RMSE on true y_valid
#         fold_rmse = mean_squared_error(y_valid, preds_total, squared=False)
#         rmse_scores.append(fold_rmse)

#     # --- Aggregate fold scores ---
#     mean_rmse = np.mean(rmse_scores)
#     print(f"Trial {trial.number}: CV RMSE = {mean_rmse:.5f}")

#     # Optuna minimizes RMSE directly
#     return mean_rmse


# def tune_xgb_with_optuna(X, y_true, y_pred_prev=None, n_trials=10, timeout=900, n_splits=5, seed=42):
#     """
#     Run Optuna hyperparameter tuning with 5-fold CV.
#     """
#     if y_pred_prev is None:
#         y_pred_prev = np.zeros(len(y_true))

#     print(f"Starting Optuna tuning with {n_splits}-fold CV on residuals...")
#     study = optuna.create_study(direction="minimize", study_name="xgb_residual_cv_tuning")
#     study.optimize(
#         lambda trial: objective(trial, X, y_true, y_pred_prev, n_splits=n_splits, seed=seed),
#         n_trials=n_trials,
#         timeout=timeout,
#         show_progress_bar=True
#     )

#     # --- Report best result ---
#     trial = study.best_trial
#     print("\nBest trial summary:")
#     print(f"  • Best RMSE: {trial.value:.6f}")
#     print(f"  • Best parameters: {trial.params}")
#     return trial.params, trial.value



# tune_xgb_with_optuna(X,y,y_)


y_ = train[TARGET]-train['y']
y_


from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
import pandas as pd


FOLDS = 5
SEED = 42

# params = {
#     'learning_rate': 0.06186797779567245,
#     'max_depth': 4,
#     'min_child_weight': 2.471191153712657,
#     'subsample': 0.9204358577490471,
#     'colsample_bytree': 0.9726618445009412,
#     'reg_alpha': 0.7561497329237483,
#     'reg_lambda': 2.3328153899545514,
#     'gamma': 2.835177499102379,
#     'max_leaves': 8,
#     "objective": "reg:squarederror",
#     "eval_metric": "rmse",
#     "tree_method": "hist",
#     "device": "cuda",
#     'random_state':SEED
# }

params = {
    "objective": "reg:squarederror",   
    "eval_metric": "rmse",             
    "learning_rate": 0.01,
    "max_depth": 6,                    
    "subsample": 0.9,
    "colsample_bytree": 0.6,
    "seed": SEED,
    "device": "cuda",
}

oof = np.zeros(len(train))
preds = np.zeros(len(test))
# kf = KFold(n_splits = FOLDS,random_state=SEED,shuffle=True)
X = train[FEATURES]
y = train[TARGET]


%%time

oof = np.zeros(len(X))
preds = np.zeros(len(test))
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    
    print(f"\n{'#'*25}")
    print(f"### FOLD : {fold+1} ###")
    print(f"{'#'*25}\n")
    
    # ------------------------
    # Data split
    # ------------------------
    x_train, x_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_train, y_valid = y.iloc[train_idx].copy(), y.iloc[valid_idx].copy()
    x_test = test[FEATURES].copy()

    # meta targets (previous predictions)
    y_train_base = train.iloc[train_idx]['y'].values
    y_valid_base = train.iloc[valid_idx]['y'].values
    y_test_base = test['y'].values

    # ------------------------
    # Target Encoding
    # ------------------------
    TE = TargetEncoder(cols_to_encode=INTER, cv=5, smooth=1.0, aggs=['mean'], drop_original=True)
    x_train = TE.fit_transform(x_train, y_train)
    x_valid = TE.transform(x_valid)
    x_test  = TE.transform(x_test)

    TE2 = TargetEncoder(cols_to_encode=ROUND, cv=5, smooth=1.0, aggs=['mean'], drop_original=True)
    x_train = TE2.fit_transform(x_train, y_train)
    x_valid = TE2.transform(x_valid)
    x_test  = TE2.transform(x_test)

    # ------------------------
    # Train on residuals
    # ------------------------
    y_train_resid = y_train - y_train_base
    y_valid_resid = y_valid - y_valid_base

    dtrain = xgb.DMatrix(x_train, label=y_train_resid, enable_categorical=True)
    dval   = xgb.DMatrix(x_valid, label=y_valid_resid, enable_categorical=True)
    dtest  = xgb.DMatrix(x_test, enable_categorical=True)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=1000
    )

    # ------------------------
    # Predict residuals, correct by adding base predictions
    # ------------------------
    valid_resid_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_resid_pred  = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    oof[valid_idx] = valid_resid_pred + y_valid_base
    preds += (test_resid_pred + y_test_base) / FOLDS

    # ------------------------
    # Compute per-fold RMSE on corrected predictions
    # ------------------------
    fold_rmse = mean_squared_error(
        y_valid, oof[valid_idx], squared=False
    )
    print(f"FOLD {fold+1} RMSE: {fold_rmse:.5f}")

# ------------------------
# Overall CV evaluation
# ------------------------
y_true_final = y.to_numpy()
y_pred_final = oof
oof_rmse = mean_squared_error(y_true_final, y_pred_final, squared=False)

print("\n" + "="*50)
print(f"OOF RMSE: {oof_rmse:.5f}")
print("="*50)


np.save(f'oof{V}.npy', oof)
np.save(f'preds{V}.npy', preds)

# mlflow.log_artifact(f'oof{cfg.V}.npy')
# mlflow.log_artifact(f'preds{cfg.V}.npy')


sub = pd.read_csv(SUB)
sub[TARGET] = preds
sub.to_csv(f'PS5E11_V{V}.csv',index=False)
display(sub.head())

