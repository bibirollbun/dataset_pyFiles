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


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """
    Down‑cast float64→float32 and int64→int32/16 to cut RAM ~50 %.
    Does NOT affect object / category columns.
    """
    for col in df.columns:
        t = df[col].dtype
        if t.kind in "iuf":                                          
            df[col] = pd.to_numeric(
                df[col],
                downcast="float" if t.kind == "f" else "integer"
            )
    return df



DATA="/kaggle/input/home-credit-default-risk"
app_train=reduce_mem(pd.read_csv(f"{DATA}/application_train.csv"))
app_test  = reduce_mem(pd.read_csv(f"{DATA}/application_test.csv"))

bureau        = reduce_mem(pd.read_csv(f"{DATA}/bureau.csv"))
bureau_bal    = reduce_mem(pd.read_csv(f"{DATA}/bureau_balance.csv"))
prev          = reduce_mem(pd.read_csv(f"{DATA}/previous_application.csv"))
inst_payments = reduce_mem(pd.read_csv(f"{DATA}/installments_payments.csv"))
pos_cash      = reduce_mem(pd.read_csv(f"{DATA}/POS_CASH_balance.csv"))
credit_card   = reduce_mem(pd.read_csv(f"{DATA}/credit_card_balance.csv"))


bureau.head()




def make_bureau_features(bur: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
   
    bb_agg = (bb
              .groupby("SK_ID_BUREAU")
              .agg(months_min=("MONTHS_BALANCE", "min"),
                   months_max=("MONTHS_BALANCE", "max"),
                   status_last=("STATUS", "last"))
              .reset_index())
    bur = bur.merge(bb_agg, on="SK_ID_BUREAU", how="left")

   
    num_cols = ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT"]

    agg_dict = {c: ["min", "max", "mean"] for c in num_cols} | {
        "CREDIT_ACTIVE": lambda s: (s == "Active").sum(),
        "CREDIT_DAY_OVERDUE": ["max"],
        "CREDIT_TYPE": "nunique"
    }

    bur_agg = bur.groupby("SK_ID_CURR").agg(agg_dict)
    bur_agg.columns = ["BUR_" + "_".join(col).upper() for col in bur_agg.columns]
    return bur_agg.reset_index()




def make_prev_app_features(prev: pd.DataFrame) -> pd.DataFrame:
    agg = (prev
           .groupby("SK_ID_CURR")
           .agg(PREV_APP_CNT=("SK_ID_PREV", "count"),
                PREV_APPROVED=("NAME_CONTRACT_STATUS", lambda s: (s == "Approved").sum()),
                PREV_REFUSED=("NAME_CONTRACT_STATUS",  lambda s: (s == "Refused").sum()),
                PREV_AMT_APP_MEAN=("AMT_APPLICATION", "mean"),
                PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean")))
    agg["PREV_APPROVAL_RATIO"] = agg.PREV_APPROVED / agg.PREV_APP_CNT
    return agg.reset_index()




def make_installment_features(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Build applicant‑level features from installments_payments.csv.

    Parameters
    ----------
    inst : pd.DataFrame
        Raw installments_payments table.

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with engineered features.
    """
    
    inst["PAY_DIFF"] = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]

    # b) Positive  = paid late   (bad)
    #    Negative  = paid early  (good)
    inst["PAY_DELAY_DAYS"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]

   
    agg = (
        inst
        .groupby("SK_ID_CURR")
        .agg(
            INST_CNT=("SK_ID_PREV", "count"),               
            INST_PAY_DIFF_MEAN=("PAY_DIFF", "mean"),
            INST_PAY_DIFF_MIN=("PAY_DIFF", "min"),
            INST_PAY_DIFF_MAX=("PAY_DIFF", "max"),
            INST_DELAY_MEAN=("PAY_DELAY_DAYS", "mean"),
            INST_DELAY_MAX=("PAY_DELAY_DAYS", "max"),
            INST_TOTAL_PAID=("AMT_PAYMENT", "sum"),
            INST_TOTAL_DUE=("AMT_INSTALMENT", "sum"),
        )
    )

    
    agg["INST_PAY_RATIO"] = agg["INST_TOTAL_PAID"] / agg["INST_TOTAL_DUE"]

    return agg.reset_index()




def make_pos_cc_features(pos: pd.DataFrame, cc: pd.DataFrame) -> pd.DataFrame:
    pos_agg = (pos
               .groupby("SK_ID_CURR")
               .agg(POS_LOAN_CNT=("SK_ID_PREV", "nunique"),
                    POS_SK_DPD_MAX=("SK_DPD", "max")))
    cc_agg = (cc
              .groupby("SK_ID_CURR")
              .agg(CC_BAL_MEAN=("AMT_BALANCE", "mean"),
                   CC_SK_DPD_MEAN=("SK_DPD", "mean")))
    return pos_agg.join(cc_agg, how="outer")



bureau_feat = make_bureau_features(bureau, bureau_bal)
prev_feat   = make_prev_app_features(prev)
inst_feat   = make_installment_features(inst_payments)
poscc_feat  = make_pos_cc_features(pos_cash, credit_card)

for feat in [bureau_feat, prev_feat, inst_feat, poscc_feat]:
    app_train = app_train.merge(feat, on="SK_ID_CURR", how="left")
    app_test  = app_test.merge(feat,  on="SK_ID_CURR", how="left")

# Replace NaNs (appearing when an applicant lacks a table) with 0
app_train.fillna(0, inplace=True)
app_test.fillna(0, inplace=True)



app_train.head()


TARGET = "TARGET"
cat_cols=[c for c in app_train.columns if app_train[c].dtype=="object"]
num_cols=[c for c in app_train.columns if c not in cat_cols + ["SK_ID_CURR",TARGET]]


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


app_train[num_cols] = app_train[num_cols].replace([np.inf, -np.inf], np.nan)
app_test[num_cols]  = app_test[num_cols].replace([np.inf, -np.inf], np.nan)

for c in num_cols:
    app_train[c] = pd.to_numeric(app_train[c], errors="coerce")
    app_test[c]  = pd.to_numeric(app_test[c],  errors="coerce")

app_train[num_cols].fillna(0, inplace=True)
app_test[num_cols].fillna(0,  inplace=True)



scaler = StandardScaler()

app_train[num_cols] = scaler.fit_transform(app_train[num_cols])
app_test[num_cols]  = scaler.transform(app_test[num_cols])


# concat so we get identical dummy columns for train and test
full_cat = pd.concat([app_train[cat_cols], app_test[cat_cols]], axis=0)

dummies = pd.get_dummies(full_cat, dummy_na=False)

# split back
n_train = len(app_train)
train_dummies = dummies.iloc[:n_train].reset_index(drop=True)
test_dummies  = dummies.iloc[n_train:].reset_index(drop=True)

# drop raw categorical cols and add dummies
app_train = pd.concat([app_train.drop(columns=cat_cols).reset_index(drop=True),
                       train_dummies], axis=1)
app_test  = pd.concat([app_test.drop(columns=cat_cols).reset_index(drop=True),
                       test_dummies],  axis=1)



X_train = app_train.drop(columns=[TARGET, "SK_ID_CURR"])
y_train = app_train[TARGET]

X_test  = app_test.drop(columns=["SK_ID_CURR"])


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_train_filled = imputer.fit_transform(X_train)



rf = RandomForestClassifier(n_estimators=600, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(rf, X_train_filled, y_train, cv=cv,
                        method="predict_proba", n_jobs=-1)[:, 1]

print("CV AUC:", roc_auc_score(y_train, oof))

rf.fit(X_train_filled, y_train)
test_pred = rf.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "SK_ID_CURR": app_test["SK_ID_CURR"],
    "TARGET":     test_pred
})
submission.to_csv("submission.csv", index=False)




