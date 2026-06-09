# =====================================
# 1. Install / Import Libraries
# =====================================
!pip install lightgbm --quiet

import numpy as np
import pandas as pd
import lightgbm as lgb
import gc  # for garbage collection (useful with large datasets)
from sklearn.preprocessing import LabelEncoder

# =====================================
# 2. Load Main Data
# =====================================
train_data = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test_data  = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)

# Extract TARGET and drop from train
target = train_data["TARGET"].copy()
train_data.drop(columns=["TARGET"], inplace=True)

# =====================================
# 3. Load Auxiliary Datasets
# =====================================
bureau_data = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
bureau_balance_data = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau_balance.csv")
pos_cash_data = pd.read_csv("/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv")
credit_card_data = pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")
installments_data = pd.read_csv("/kaggle/input/home-credit-default-risk/installments_payments.csv")

print("Bureau shape:", bureau_data.shape)
print("Bureau Balance shape:", bureau_balance_data.shape)
print("POS_CASH shape:", pos_cash_data.shape)
print("Credit Card shape:", credit_card_data.shape)
print("Installments shape:", installments_data.shape)

# =====================================
# 4. Feature Engineering Functions
# =====================================

def aggregate_bureau_data(bureau_data, bureau_balance_data):
    bureau_balance_agg = bureau_balance_data.groupby("SK_ID_BUREAU").agg({
        "MONTHS_BALANCE": ["min", "max", "mean"]
    })

    bureau_balance_agg.columns = ["BB_" + "_".join(col).upper() for col in bureau_balance_agg.columns.ravel()]
    bureau_balance_agg.reset_index(inplace=True)

    bureau_data = bureau_data.merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")

    bureau_agg = bureau_data.groupby("SK_ID_CURR").agg({
        "DAYS_CREDIT": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["sum", "mean"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean"],
        "BB_MONTHS_BALANCE_MIN": ["min", "mean"],
        "BB_MONTHS_BALANCE_MAX": ["max", "mean"]
    })

    bureau_agg.columns = ["BURO_" + "_".join(col).upper() for col in bureau_agg.columns.ravel()]
    bureau_agg.reset_index(inplace=True)
    return bureau_agg

def aggregate_pos_cash_data(pos_cash_data):
    pos_agg = pos_cash_data.groupby("SK_ID_CURR").agg({
        "MONTHS_BALANCE": ["min", "max", "mean"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"]
    })

    pos_agg.columns = ["POS_" + "_".join(col).upper() for col in pos_agg.columns.ravel()]
    pos_agg.reset_index(inplace=True)
    return pos_agg

def aggregate_credit_card_data(credit_card_data):
    cc_agg = credit_card_data.groupby("SK_ID_CURR").agg({
        "MONTHS_BALANCE": ["min", "max", "mean"],
        "AMT_BALANCE": ["sum", "mean", "max"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
        "SK_DPD": ["max", "mean"]
    })

    cc_agg.columns = ["CC_" + "_".join(col).upper() for col in cc_agg.columns.ravel()]
    cc_agg.reset_index(inplace=True)
    return cc_agg

def aggregate_installments_data(installments_data):
    install_agg = installments_data.groupby("SK_ID_CURR").agg({
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "AMT_INSTALMENT": ["sum", "mean", "max"],
        "AMT_PAYMENT": ["sum", "mean", "min"],
        "DAYS_ENTRY_PAYMENT": ["min", "max", "mean"]
    })

    install_agg.columns = ["INS_" + "_".join(col).upper() for col in install_agg.columns.ravel()]
    install_agg.reset_index(inplace=True)
    return install_agg

# =====================================
# 5. Feature Aggregation and Merging
# =====================================
train_data = train_data.merge(aggregate_bureau_data(bureau_data, bureau_balance_data), on="SK_ID_CURR", how="left")
test_data  = test_data.merge(aggregate_bureau_data(bureau_data, bureau_balance_data), on="SK_ID_CURR", how="left")

train_data = train_data.merge(aggregate_pos_cash_data(pos_cash_data), on="SK_ID_CURR", how="left")
test_data  = test_data.merge(aggregate_pos_cash_data(pos_cash_data), on="SK_ID_CURR", how="left")

train_data = train_data.merge(aggregate_credit_card_data(credit_card_data), on="SK_ID_CURR", how="left")
test_data  = test_data.merge(aggregate_credit_card_data(credit_card_data), on="SK_ID_CURR", how="left")

train_data = train_data.merge(aggregate_installments_data(installments_data), on="SK_ID_CURR", how="left")
test_data  = test_data.merge(aggregate_installments_data(installments_data), on="SK_ID_CURR", how="left")

# =====================================
# 6. Handle Categorical Columns
# =====================================
combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

categorical_columns = combined_data.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    le = LabelEncoder()
    combined_data[column] = combined_data[column].astype(str)
    combined_data[column] = le.fit_transform(combined_data[column])

train_data = combined_data.iloc[:len(target), :].copy()
test_data = combined_data.iloc[len(target):, :].copy()

# =====================================
# 7. Model Building and Training
# =====================================
train_features = train_data.drop(columns=["SK_ID_CURR", "SK_ID_BUREAU"], errors="ignore")
test_features = test_data.drop(columns=["SK_ID_CURR", "SK_ID_BUREAU"], errors="ignore")

train_dataset = lgb.Dataset(train_features, label=target)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 32,
    "feature_fraction": 0.8,
    "subsample": 0.8,
    "random_state": 42
}

model = lgb.train(params, train_dataset, num_boost_round=500)

# =====================================
# 8. Prediction & Submission
# =====================================
predictions = model.predict(test_features)

submission = pd.DataFrame({
    "SK_ID_CURR": test_data["SK_ID_CURR"],
    "TARGET": predictions
})

submission.to_csv("submission_lgb.csv", index=False)
print("Submission file 'submission_lgb.csv' created! Upload it to Kaggle for scoring.")





