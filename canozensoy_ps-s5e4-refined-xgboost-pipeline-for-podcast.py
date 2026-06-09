# Improved XGBoost Model for Podcast Listening Time Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import gc
from tqdm import tqdm
from itertools import combinations
from collections import Counter
from colorama import Fore, Style
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, callback

warnings.filterwarnings("ignore")


# === Load Data ===
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# === Optimize Data Types ===
for col in train_df.select_dtypes(include='float64').columns:
    train_df[col] = train_df[col].astype('float32')
for col in train_df.select_dtypes(include='int64').columns:
    train_df[col] = train_df[col].astype('int32')


# === Fill Missing Values ===
train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].median(), inplace=True)
train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
test_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
test_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)


# === Feature Lists ===
target = ["Listening_Time_minutes"]
features = [c for c in train_df.columns if c not in target]
cats = [c for c in features if train_df[c].dtype == "object"]


# === Pairwise Encoding ===
encoded_columns = []
encode_columns = ['Episode_Length_minutes', 'Episode_Title', 'Host_Popularity_percentage', 
                  'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
pair_size = [2, 3, 4]

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r)), desc=f"Generating {r}-way encodings"):
        new_col = '_'.join(cols)
        train_df[new_col] = train_df[list(cols)].astype(str).agg('_'.join, axis=1).astype('category')
        test_df[new_col] = test_df[list(cols)].astype(str).agg('_'.join, axis=1).astype('category')
        encoded_columns.append(new_col)


# === Update Feature List ===
features = [c for c in train_df.columns if c not in target]
cats = [c for c in features if train_df[c].dtype == "category"]


# === Ensure All Object Columns Are Categorical ===
for col in features:
    if train_df[col].dtype == "object":
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")


# === Target Encoding Function ===
def target_encode(train_df, test_df, col, target, stats='mean', prefix='TE'):
    col_name = f"{prefix}_{col}"
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    agg = train_df.groupby(col)[target].agg(stats)
    if isinstance(agg, pd.DataFrame):
        agg = agg.iloc[:, 0]
    test_df[col_name] = test_df[col].map(agg)
    test_df[col_name].fillna(agg.mean(), inplace=True)
    return test_df


# === XGBoost Parameters ===
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'max_depth': 10,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.8,
    'reg_lambda': 4,
    'seed': 42,
    'enable_categorical': True,
    'tree_method': 'hist',
    'device': 'gpu'
}


# === Progress Callback ===
class TQDMCallback(callback.TrainingCallback):
    def __init__(self, total, fold_num=None):
        self.pbar = tqdm(total=total, desc=f"Training Fold {fold_num}", leave=True)
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        if epoch + 1 == self.pbar.total:
            self.pbar.close()
        return False


# === Cross Validation Training ===
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds_xgb = np.zeros(len(train_df))
test_preds_xgb = np.zeros(len(test_df))

gc.collect()

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    print(Fore.GREEN + f"### Fold {fold+1} ###" + Style.RESET_ALL)

    X_train = train_df.loc[train_idx, features + target].reset_index(drop=True)
    y_train = X_train[target]
    X_valid = train_df.loc[valid_idx, features].reset_index(drop=True)
    y_valid = train_df.loc[valid_idx, target].reset_index(drop=True)
    X_test = test_df[features].reset_index(drop=True)

    # Nested KFold Target Encoding
    kf2 = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for train_idx2, valid_idx2 in kf2.split(X_train):
        train2 = X_train.iloc[train_idx2].copy()
        valid2 = X_train.iloc[valid_idx2][features].copy()

        for col in tqdm(encoded_columns, total=len(encoded_columns), desc=f"Second KFold Encoding"):
            te_col = f'TE_{col}'
            valid2 = target_encode(train2, valid2, col, target, stats='mean', prefix="TE")
            X_train.loc[valid_idx2, te_col] = valid2[te_col].values

        del train2, valid2
        gc.collect()

    for col in encoded_columns:
        X_valid = target_encode(X_train, X_valid, col, target, stats='mean', prefix="TE")
        X_test = target_encode(X_train, X_test, col, target, stats='mean', prefix="TE")

    te_cols = [f'TE_{col}' for col in encoded_columns]

    X_train.drop(columns=encoded_columns + target, inplace=True)
    X_valid.drop(columns=encoded_columns, inplace=True)
    X_test.drop(columns=encoded_columns, inplace=True)

    model = XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=150,
        verbose=0,
        callbacks=[TQDMCallback(total=xgb_params['n_estimators'], fold_num=fold+1)]
    )

    oof_preds_xgb[valid_idx] = model.predict(X_valid)
    test_preds_xgb += model.predict(X_test) / FOLDS

    gc.collect()


# === Evaluation ===
rmse = np.sqrt(mean_squared_error(train_df[target], oof_preds_xgb))
print(Fore.CYAN + f"Validation RMSE: {rmse}" + Style.RESET_ALL)


# === Submission ===
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sub["Listening_Time_minutes"] = test_preds_xgb
sub.to_csv("submission.csv", index=False)
print(Fore.YELLOW + "submission.csv file has been saved." + Style.RESET_ALL)


