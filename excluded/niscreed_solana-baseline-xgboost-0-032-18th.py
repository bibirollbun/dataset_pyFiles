# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pd.set_option('display.max_columns', 100)
FOLDS = 10


# Used for TargetEncoder.
!pip install scikit-learn==1.6.0 >> /dev/null 2>&1


print("Loading Train & Test sets.")
df_train = pd.read_csv("/kaggle/input/solana-skill-sprint-memcoin-movement-prediction/train.csv", index_col="Unnamed: 0")
df_test = pd.read_csv("/kaggle/input/solana-skill-sprint-memcoin-movement-prediction/test_unlabeled.csv", index_col="Unnamed: 0")
print("Loading extra token info.")
df_info1 = pd.read_csv("/kaggle/input/pump-fun-graduation-february-2025/dune_token_info.csv")
df_info2 = pd.read_csv("/kaggle/input/pump-fun-graduation-february-2025/dune_token_info_v2.csv")
print("Loading df features.")
df_features = pd.read_csv("/kaggle/input/generate-chunk-data-baseline/chunk_data_aggregated.csv")
for column_name in df_features.columns:
    if column_name in ["base_coin"]:
        continue
    elif df_features[column_name].dtype == "float64":
        df_features[column_name] = df_features[column_name].astype("float32")
if "tx_count_y" in df_features:
    df_features = df_features.rename(columns={"tx_count_x": "tx_count"})
    df_features = df_features.drop(columns=["tx_count_y"])
print("Loading df on chain.")
df_onchain = pd.read_csv("/kaggle/input/generate-chunk-data-baseline/info_on_chain_grouped.csv")


df_onchain


df_features


categorical_column_names = ["mint"]
numerical_column_names = ["slot_min"]
column_name_to_predict = "has_graduated"


df_info = pd.concat([df_info1, df_info2])
df_info = df_info.drop(columns=["name", "token_uri"])
tx_counts = df_info['init_tx'].value_counts()
df_info['init_tx_frequency'] = df_info['init_tx'].map(tx_counts)
df_info = df_info.drop_duplicates()
df_info.head(3)


# Merging with train data.
df_train_with_extra = df_train.merge(
    df_info,
    left_on="mint",
    right_on="token_mint_address",
    how="left",
)
df_train_with_extra = df_train_with_extra.drop(columns=["token_mint_address"])
assert len(df_train) == len(df_train_with_extra)
df_train_with_extra.info()


# Merging with test data.
df_test_with_extra = df_test.merge(
    df_info,
    left_on="mint",
    right_on="token_mint_address",
    how="left",
)
df_test_with_extra = df_test_with_extra.drop(columns=["token_mint_address"])
assert len(df_test) == len(df_test_with_extra)
df_test_with_extra.info()


def fillna(df: pd.DataFrame) -> pd.DataFrame:
    """Filling na values.

    :param df: Entry DataFrame.
    :return pd.DataFrame: output DataFrame with filled columns.
    :notes:
    - This is a simple version, doing it with more accuracy after EDA must improve the logloss.
    """
    # Copying.
    df_filled = df.copy()
    # Iterating over columns.
    for column_name in df_filled.columns:
        if column_name in [column_name_to_predict]:
            continue
        elif column_name == "is_valid":
            value = df_filled[column_name].mode()[0]
        elif df_filled[column_name].dtype in ["category", "object"]:
            value = df_filled[column_name].mode()[0]
        elif df_filled[column_name].dtype in ["float32", "float64"]:
            value = df_filled[column_name].mean()
        elif df_filled[column_name].dtype in ["int8", "int16", "int32", "int64"]:
            value = df_filled[column_name].mode()[0]
        else:
            raise TypeError(f"{column_name}: {df_filled[column_name].dtype}.")
        df_filled[column_name] = df_filled[column_name].fillna(value=value)
    # Returning filled df.
    return df_filled
    

def optimize(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize the column types to improve the memory usage.

    :param df: Entry DataFrame.
    :return pd.DataFrame: memory optimized DataFrame.
    """
    # Copying
    new_df = df.copy()
    # df_train.slot_min.max() < info.max
    for column_name in df.columns:
        if column_name in ["mint", "init_tx"]:
            continue
        elif column_name in ["is_valid"]:
            new_df[column_name] = new_df[column_name].map({True: 1, False: 0}).astype("int8")
        elif column_name in ["dev_balance"]:
            new_df[column_name] = new_df[column_name].astype("float64")
        elif column_name in ["tx_idx", "bundle_size", "init_tx_frequency"]:
            new_df[column_name] = new_df[column_name].astype("int16")
        elif column_name in ["slot", "slot_min"]:
            new_df[column_name] = new_df[column_name].astype("int32")
        elif column_name in ["slot_graduated", "gas_used", "bundled_buys"]:
            new_df[column_name] = new_df[column_name].astype("float32")
        elif column_name in ["has_graduated", "has_slot_graduated"]:
            new_df[column_name] = new_df[column_name].astype("int8")
        elif column_name in ["created_at", "block_time"]:
            new_df[column_name] = pd.to_datetime(new_df[column_name])
        elif column_name in [
            "decimals", "symbol", "creator", "bundled_buys_count", "creation_ix_index",
            "direct_pf_invocation",	"amount_of_instructions", "amount_of_lookup_reads", "amount_of_lookup_writes",
            "pf_program_index", "symbol"
        ]:
            new_df[column_name] = new_df[column_name].astype("category")
        else:
            raise ValueError(f"{column_name}")
    return new_df

df_train_with_extra_filled = fillna(df=df_train_with_extra)
df_test_with_extra_filled = fillna(df=df_test_with_extra)
df_train_with_extra_filled_opti = optimize(df=df_train_with_extra_filled)
df_test_with_extra_filled_opti = optimize(df=df_test_with_extra_filled)
assert len(df_train_with_extra_filled_opti) == len(df_train)
assert len(df_test_with_extra_filled_opti) == len(df_test)
df_train_with_extra_filled_opti.info()


df_train_with_extra_filled_opti


from typing import List, Text, Dict, Union, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler

def scale_and_encoder_features(
        df: pd.DataFrame,
        skip_column_names: List[Text],
    ) -> pd.DataFrame:
    """Scale the features.

    :param df: Entry DataFrame.
    :param skip_column_names: column names to skip while scaling. 
    :return pd.DataFrame: ML input ready DataFrame.
    """
    # Creating a copy.
    scaled_and_encoded_df = df.copy()
    # Iterating over features.
    for column_name in scaled_and_encoded_df.columns:
        # Preparing encoder & scaler.
        if column_name in skip_column_names:
            continue
        elif column_name in ["init_tx"]:
            enc = LabelEncoder()
            scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]].values.ravel())
        elif scaled_and_encoded_df[column_name].dtype in ["category", "int8", "bool"]:
            enc = LabelEncoder()
            scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]].values.ravel())
        elif scaled_and_encoded_df[column_name].dtype in ["int32", "float32", "int16", "float64", "int64"]:
            enc = StandardScaler()
            scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]])
        elif scaled_and_encoded_df[column_name].dtype in ["datetime64[ns, UTC]", "datetime64[ns]"]:
            scaled_and_encoded_df[column_name] = scaled_and_encoded_df[column_name].astype('int64') // 10**9
            enc = StandardScaler()
            scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]])
        else:
            raise TypeError(f"{column_name} - {scaled_and_encoded_df[column_name].dtype}.")
    return scaled_and_encoded_df

df_train_scale = scale_and_encoder_features(df=df_train_with_extra_filled_opti, skip_column_names=["mint"])
df_test_scale = scale_and_encoder_features(df=df_test_with_extra_filled_opti, skip_column_names=["mint"])
df_test_scale.head(3)


feature_column_names = list(set(df_test_scale.columns).difference(set(["mint"])))
feature_column_names


df_train_extended = df_train_scale.merge(
    df_features,
    left_on="mint",
    right_on="base_coin",
    how="left",
).drop(columns=["base_coin"])
assert len(df_train) == len(df_train_extended), f"Fail merging: {len(df_test)} vs {len(df_test_extended)}."
df_test_extended = df_test_scale.merge(
    df_features,
    left_on="mint",
    right_on="base_coin",
    how="left",
).drop(columns=["base_coin"])
assert len(df_test) == len(df_test_extended), f"Fail merging: {len(df_test)} vs {len(df_test_extended)}."
df_train_extended.info()


df_train_extended2 = df_train_extended.merge(
    df_onchain,
    on="mint",
    how="left",
)
assert len(df_train) == len(df_train_extended2), f"Fail merging: {len(df_test)} vs {len(df_train_extended2)}."
df_test_extended2 = df_test_extended.merge(
    df_onchain,
    on="mint",
    how="left",
)
assert len(df_test) == len(df_test_extended2), f"Fail merging: {len(df_test)} vs {len(df_test_extended2)}."
df_test_extended2.info()


model_column_names = (
    feature_column_names
    + df_features.columns.tolist()[1:]
    + df_onchain.columns.tolist()[1:]
)
for col in [
    'max_version',
    'cnt_tx_idx',
    'cnt_slot',
    'nu_tx_idx',
    'nu_slot',
    'slot_std',
    'is_valid',
    'avg_bundle_size',
    'buy_count',
    'count',
    'decimals',
    'net_direction',
    'slot_diff_first',
    'sell_count',
    'slot_span']:
    model_column_names.remove(col)


target_encoder_columns = [
    "most_common_symbol", "tx_count", "unique_wallets", 
    # "bundle_structure", "curve_address", 
]


# Imports
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
)
from sklearn.preprocessing import TargetEncoder
import datetime as dt

# Setting KFold
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Preparing outputs
oof_cat = np.zeros(len(df_train_extended))
pred_cat = np.zeros(len(df_test_extended))

cat_params = {
    'random_seed': 42,
    'verbose': False,
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'learning_rate': 0.01,
    'depth': 8,
    'l2_leaf_reg': 3,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'early_stopping_rounds': 100,
    'iterations': 2400,
}
start_time = dt.datetime.now()

# OUTER K FOLD
for i, (train_index, test_index) in enumerate(kf.split(df_train_extended, df_train[column_name_to_predict])):
    X_train = df_train_extended2.loc[train_index, model_column_names].reset_index(drop=True).copy()
    y_train = df_train_extended2.loc[train_index, column_name_to_predict]

    X_valid = df_train_extended2.loc[test_index, model_column_names].reset_index(drop=True).copy()
    y_valid = df_train_extended2.loc[test_index, column_name_to_predict]

    X_test = df_test_extended2.loc[:, model_column_names].reset_index(drop=True).copy()

    for column_name in target_encoder_columns:
        TE = TargetEncoder()
        X_train[column_name] = TE.fit_transform(X_train[[column_name]], y_train)
        X_valid[column_name] = TE.transform(X_valid[[column_name]])
        X_test[column_name] = TE.transform(X_test[[column_name]])


    # TRAIN SVC MODEL
    model = CatBoostClassifier(**cat_params)
    
    # TRAIN MODEL
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100)
    valid_score = log_loss(y_valid, model.predict_proba(X_valid)[:, 1])
    print(f"# Fold {i+1} - score is: {100*valid_score:.2f}%. It took {(dt.datetime.now() - start_time).seconds / 60:.2f} minutes.")

    # PREDICT OOF AND TEST
    oof_cat[test_index] += model.predict_proba(X_valid)[:, 1]
    pred_cat += model.predict_proba(X_test)[:,1]

pred_cat /= FOLDS

score = log_loss(df_train_extended2[column_name_to_predict], oof_cat)
print(f"Score: {100*score:.2f}% - it took {(dt.datetime.now() - start_time).seconds / 60:.2f} minutes.")
print("To beat  3.76% LogLoss with 7 folds.")

df_test[column_name_to_predict] = pred_cat
df_test[["mint", column_name_to_predict]].to_csv("submission_cat.csv", index=False, sep=",")


import matplotlib.pyplot as plt
import seaborn as sns

# Last model info
plt.figure(figsize=(13,9))
abs_feature_importance = abs(model.feature_importances_.reshape(-1))
sns.barplot(
    x="feature_importance",
    y="feature_names",
    data=pd.DataFrame({
        "feature_importance": abs_feature_importance,
        "feature_names": model_column_names,
    }).sort_values(by=["feature_importance"], ascending=False),
)
plt.title("Feature Importance from CAT")
plt.show()


plt.figure(figsize=(13,9))
plt.title(column_name_to_predict)
plt.xlabel('true', fontsize=12)
plt.ylabel('pred', fontsize=12)
plt.scatter(df_train_extended2[column_name_to_predict], oof_cat, alpha=0.3)


# Import
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
# from sklearn.preprocessing import TargetEncoder
import datetime as dt

# Setting KFold
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Preparing outputs
oof_xgb = np.zeros(len(df_train_extended))
pred_xgb = np.zeros(len(df_test_extended))

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.01, # learning_rate
    'max_depth': 8,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'min_child_weight': 2,
    'gamma': 0.1,
    'lambda': 0, # L2 reg
    'alpha': 1, # L1 reg
    'seed': 42,
    'tree_method': 'hist',
    'n_estimators': 1500, 
    'early_stopping_rounds': 100,
}
start_time = dt.datetime.now()

# OUTER K FOLD
for i, (train_index, test_index) in enumerate(kf.split(df_train_extended, df_train[column_name_to_predict])):
    X_train = df_train_extended2.loc[train_index, model_column_names].reset_index(drop=True).copy()
    y_train = df_train_extended2.loc[train_index, column_name_to_predict]

    X_valid = df_train_extended2.loc[test_index, model_column_names].reset_index(drop=True).copy()
    y_valid = df_train_extended2.loc[test_index, column_name_to_predict]

    X_test = df_test_extended2.loc[:, model_column_names].reset_index(drop=True).copy()

    # TRAIN SVC MODEL
    model = XGBClassifier(**xgb_params)
    
    # TRAIN MODEL
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    valid_score = log_loss(y_valid, model.predict_proba(X_valid)[:, 1])
    print(f"# Fold {i+1} - score is: {100*valid_score:.2f}%. It took {(dt.datetime.now() - start_time).seconds / 60:.2f} minutes.")

    # PREDICT OOF AND TEST
    oof_xgb[test_index] += model.predict_proba(X_valid)[:, 1]
    pred_xgb += model.predict_proba(X_test)[:,1]

pred_xgb /= FOLDS

score = log_loss(df_train_extended2[column_name_to_predict], oof_xgb)
print(f"Score: {100*score:.2f}% - it took {(dt.datetime.now() - start_time).seconds / 60:.2f} minutes.")
print("To beat 3.74% LogLoss with 7 folds.")

df_test[column_name_to_predict] = pred_xgb
df_test[["mint", column_name_to_predict]].to_csv("submission_xgb.csv", index=False, sep=",")


import matplotlib.pyplot as plt
import seaborn as sns

# Last model info
plt.figure(figsize=(13,9))
abs_feature_importance = abs(model.feature_importances_.reshape(-1))
sns.barplot(
    x="feature_importance",
    y="feature_names",
    data=pd.DataFrame({
        "feature_importance": abs_feature_importance,
        "feature_names": model_column_names,
    }).sort_values(by=["feature_importance"], ascending=False),
)
plt.title("Feature Importance from XGB")
plt.show()


# Import
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
import datetime as dt

# Setting KFold
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Preparing outputs
oof_lgbm = np.zeros(len(df_train_extended))
pred_lgbm = np.zeros(len(df_test_extended))

lgbm_params = {
    'objective': 'binary',
    'metric': 'logloss',
    # 'boosting_type': 'gbdt',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'num_leaves': 30,
    'max_depth': 7,
    'lambda_l1': 0.5, # reg_alpha
    'lambda_l2': 0.08, # reg_lambda
    'colsample_bytree': 0.7,
    'subsample': 0.56,
    'min_child_samples': 90,
    'seed': 42,
    'verbose': -1,
}
start_time = dt.datetime.now()

# OUTER K FOLD
for i, (train_index, test_index) in enumerate(kf.split(df_train_extended, df_train[column_name_to_predict])):
    X_train = df_train_extended2.loc[train_index, model_column_names].reset_index(drop=True).copy()
    y_train = df_train_extended2.loc[train_index, column_name_to_predict]

    X_valid = df_train_extended2.loc[test_index, model_column_names].reset_index(drop=True).copy()
    y_valid = df_train_extended2.loc[test_index, column_name_to_predict]

    X_test = df_test_extended2.loc[:, model_column_names].reset_index(drop=True).copy()

    for column_name in target_encoder_columns:
        TE = TargetEncoder()
        X_train[column_name] = TE.fit_transform(X_train[[column_name]], y_train)
        X_valid[column_name] = TE.transform(X_valid[[column_name]])
        X_test[column_name] = TE.transform(X_test[[column_name]])
    
    # TRAIN SVC MODEL
    model = LGBMClassifier(**lgbm_params)
    
    # TRAIN MODEL
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    valid_score = log_loss(y_valid, model.predict_proba(X_valid)[:, 1])
    print(f"# Fold {i+1} - score is: {100*valid_score:.2f}%. It took {(dt.datetime.now() - start_time).seconds / 60:.2f} minutes.")

    # PREDICT OOF AND TEST
    oof_lgbm[test_index] += model.predict_proba(X_valid)[:, 1]
    pred_lgbm += model.predict_proba(X_test)[:,1]

pred_lgbm /= FOLDS

score = log_loss(df_train_extended2[column_name_to_predict], oof_lgbm)
print(f"Score: {100*score:.2f}% - it took {(dt.datetime.now() - start_time).seconds / 60:.2f} minutes.")
print("To beat 3.76% LogLoss with 7 folds.")
df_test[column_name_to_predict] = pred_lgbm
df_test[["mint", column_name_to_predict]].to_csv("submission_lgbm.csv", index=False, sep=",")


import matplotlib.pyplot as plt
import seaborn as sns

# Last model info
plt.figure(figsize=(13,9))
abs_feature_importance = abs(model.feature_importances_.reshape(-1))
sns.barplot(
    x="feature_importance",
    y="feature_names",
    data=pd.DataFrame({
        "feature_importance": abs_feature_importance,
        "feature_names": model_column_names,
    }).sort_values(by=["feature_importance"], ascending=False),
)
plt.title("Feature Importance from LGBM")
plt.show()


final_oof = (oof_cat + oof_xgb + oof_lgbm) / 3
score = log_loss(df_train_extended2[column_name_to_predict], final_oof)
print(f"Score: {100*score:.2f}%. Meaning with 3. To beat 3.73%.")
final_pred = (pred_cat + pred_xgb + pred_lgbm) / 3
df_test[column_name_to_predict] = final_pred
df_test[["mint", column_name_to_predict]].to_csv("submission_mean3.csv", index=False, sep=",")


final_oof = (oof_lgbm + oof_xgb) / 2
score = log_loss(df_train_extended2[column_name_to_predict], final_oof)
print(f"Score: {100*score:.2f}%. Meaning with 2. To beat 3.73%.")
final_pred = (pred_lgbm + pred_xgb) / 2
df_test[column_name_to_predict] = final_pred
df_test[["mint", column_name_to_predict]].to_csv("submission_mean2.csv", index=False, sep=",")


oof_train = np.column_stack([oof_xgb, oof_lgbm, oof_cat])
preds = np.column_stack([pred_xgb, pred_lgbm, pred_cat])

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

total_logloss = 0
stackings_preds = 0
oof_final= np.zeros(len(oof_train))
new_folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
y = df_train[[column_name_to_predict]].values
start_time = dt.datetime.now()

for i, (train_index, valid_index) in enumerate(new_folds.split(oof_train, y)):
    x_train = oof_train[train_index]
    y_train = y[train_index]
    x_valid = oof_train[valid_index]
    y_valid = y[valid_index]

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=120,
        criterion="log_loss",
        max_depth=4,
    )
    model.fit(x_train, y_train.ravel())
    oof_final[valid_index] += model.predict_proba(x_valid)[:, 1]
    stackings_preds += model.predict_proba(preds)[:, 1]
    logloss = log_loss(y_valid, oof_final[valid_index])
    total_logloss += logloss / FOLDS
    print(f"Fold: {i+1} - Log_Loss : {100*logloss:.2f}%. It took {(dt.datetime.now() - start_time).seconds / 60:.2f} minutes.")

stackings_preds /= FOLDS

print(f"Total LogLoss score : {100*total_logloss:.2f}% - to beat 3.73% with {FOLDS} folds.")
print(f"It took {(dt.datetime.now() - start_time).seconds / 60:.2f} minutes.")


df_test[column_name_to_predict] = stackings_preds
df_test[["mint", column_name_to_predict]].to_csv("submission.csv", index=False, sep=",")

