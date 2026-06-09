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


import numpy as np
import pandas as pd
import gc
import lightgbm as lgb

# Load Data
train_df = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test_df = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")

# Extract TARGET
y = train_df["TARGET"].copy()
train_df.drop(columns=["TARGET"], inplace=True)

# Load Auxiliary Data
bureau = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
bureau_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau_balance.csv")
pos_cash = pd.read_csv("/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv")
credit_card = pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")
installments = pd.read_csv("/kaggle/input/home-credit-default-risk/installments_payments.csv")

# Fix Bureau Aggregation
bureau_balance_agg = bureau_balance.groupby("SK_ID_BUREAU").agg({"MONTHS_BALANCE": ["min", "max", "mean"]})
bureau_balance_agg.columns = ["BB_" + "_".join(col).upper() for col in bureau_balance_agg.columns]
bureau_balance_agg.reset_index(inplace=True)

bureau = bureau.merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")
del bureau_balance, bureau_balance_agg
gc.collect()

if "SK_ID_CURR" not in bureau.columns:
    raise KeyError("'SK_ID_CURR' is missing from the bureau dataset")

bureau_agg = bureau.groupby("SK_ID_CURR").agg({
    "DAYS_CREDIT": ["min", "max", "mean"],
    "CREDIT_DAY_OVERDUE": ["max", "mean"],
    "AMT_CREDIT_SUM": ["sum", "mean"],
    "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
    "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean"]
})
bureau_agg.columns = ["BURO_" + "_".join(col).upper() for col in bureau_agg.columns]
bureau_agg.reset_index(inplace=True)

# Merge with Train & Test
train_df = train_df.merge(bureau_agg, on="SK_ID_CURR", how="left")
test_df = test_df.merge(bureau_agg, on="SK_ID_CURR", how="left")

del bureau, bureau_agg
gc.collect()


# Feature Engineering for Other Datasets
for df, prefix in zip([pos_cash, credit_card, installments], ["POS", "CC", "INS"]):
    df_agg = df.groupby("SK_ID_CURR").agg({col: ["min", "max", "mean"] for col in df.columns if df[col].dtype != "object"})
    df_agg.columns = [prefix + "_" + "_".join(col).upper() for col in df_agg.columns]
    df_agg.reset_index(inplace=True)
    train_df = train_df.merge(df_agg, on="SK_ID_CURR", how="left")
    test_df = test_df.merge(df_agg, on="SK_ID_CURR", how="left")
    del df, df_agg
    gc.collect()

# Encode Categorical Features
obj_cols = train_df.select_dtypes(include=["object"]).columns
for col in obj_cols:
    train_df[col], _ = pd.factorize(train_df[col])
    test_df[col], _ = pd.factorize(test_df[col])
#split test and train data
df_full = pd.concat([train_df, test_df], axis=0, ignore_index=True)
train_len = len(train_df)
train_df = df_full.iloc[:train_len, :]
test_df = df_full.iloc[train_len:, :]

del df_full
gc.collect()

# Fill NaN Values
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# Drop ID columns or non-feature columns if needed
drop_cols = ["SK_ID_CURR", "SK_ID_BUREAU"]  # Might not exist in the merged data, so ignore errors
X = train_df.drop(columns=drop_cols, errors="ignore")
X_test = test_df.drop(columns=drop_cols, errors="ignore")

print("X shape:", X.shape, "| y shape:", y.shape)
print("X_test shape:", X_test.shape)


#Train LightGBM Model
train_data = lgb.Dataset(X, label=y)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "reg_alpha": 0.1, 
    "reg_lambda": 0.1,
    "learning_rate": 0.05,
    "num_leaves": 32,
    "feature_fraction": 0.8,
    "subsample": 0.8,
    "random_state": 42
}

# For simplicity, we're not using a validation set here
model = lgb.train(params, train_data, num_boost_round=500)

#Prediction and Submission

y_pred = model.predict(X_test)

submission = pd.DataFrame({
    "SK_ID_CURR": test_df["SK_ID_CURR"],
    "TARGET": y_pred
})

submission.to_csv("submission_lgb.csv", index=False)
print("Submission file is created!")

