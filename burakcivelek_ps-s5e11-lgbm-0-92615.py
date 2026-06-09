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


from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

TARGET = "loan_paid_back"


def create_frequency_and_bins(train, test, cols, num_cols):
    new_train = pd.DataFrame(index=train.index)
    new_test = pd.DataFrame(index=test.index)

    for col in cols:
        # Frequency
        freq = train[col].value_counts()
        new_train[f"{col}_freq"] = train[col].map(freq).astype(float)
        new_test[f"{col}_freq"] = test[col].map(freq).astype(float).fillna(freq.mean())

        # Quantile bins
        if col in num_cols:
            for q in [5, 10, 15]:
                try:
                    tr_bins, bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    new_train[f"{col}_bin{q}"] = tr_bins.astype(float)
                    new_test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True).astype(float)
                except Exception:
                    new_train[f"{col}_bin{q}"] = 0.0
                    new_test[f"{col}_bin{q}"] = 0.0
    return new_train, new_test

def to_categoricals(df, cat_cols):
    for col in cat_cols:
        if df[col].dtype != "category":
            df[col] = df[col].astype("category")
    return df

def target_encoding(train, test, cols, target_col):
    te_train = pd.DataFrame(index=train.index)
    te_test = pd.DataFrame(index=test.index)
    kf = StratifiedKFold(n_splits=7, random_state=42, shuffle=True)

    for col in cols:
        oof = np.zeros(len(train))
        for train_idx, val_idx in kf.split(train, train[target_col]):
            X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
            mean_map = X_tr.groupby(col)[target_col].mean()
            oof[val_idx] = X_val[col].map(mean_map)

        te_train[f"te_{col}"] = oof
        
        # Global mapping for test
        global_mean = train.groupby(col)[target_col].mean()
        te_test[f"te_{col}"] = test[col].map(global_mean)

    return te_train, te_test

# Feature Interaction utils
def multiplicative_interaction(df, col1, col2):
    df[f'{col1}X{col2}'] = df[col1] * df[col2]
    return df

def divisive_interaction(df, col1, col2, eps=1e-6):
    df[f'{col1}/{col2}'] = df[col1] / (df[col2] + eps)
    return df


train_df["grade"] = train_df["grade_subgrade"].str[0]
test_df["grade"] = test_df["grade_subgrade"].str[0]

train_df['subgrade'] = train_df['grade_subgrade'].str[1:].astype(int)
test_df['subgrade']  = test_df['grade_subgrade'].str[1:].astype(int)

train_df.drop(columns=["grade_subgrade"], inplace=True)
test_df.drop(columns=["grade_subgrade"], inplace=True)

# Interaction Features
multiplicative_interaction(train_df, "loan_amount", "interest_rate")
multiplicative_interaction(test_df, "loan_amount", "interest_rate")
divisive_interaction(train_df, "loan_amount", "annual_income")
divisive_interaction(test_df, "loan_amount", "annual_income")

# Feature lists
cols = [col for col in train_df.columns if col not in [TARGET, "id"]]
num_cols = [col for col in cols if train_df[col].dtype in ["int64","float64"]]
cat_cols = [col for col in cols if train_df[col].dtype in ["object","category"]]

# Target Encoding on all columns
te_train, te_test = target_encoding(train_df, test_df, cols, TARGET)

# Frequency + Quantile-bin encodings
fq_train, fq_test = create_frequency_and_bins(train_df, test_df, cols=cols, num_cols=num_cols)

# Concat original + encoding
f_train = pd.concat([train_df[cols], te_train, fq_train], axis=1)
f_test  = pd.concat([test_df[cols], te_test, fq_test], axis=1)

# Convert to categorical
f_train = to_categoricals(f_train, cat_cols)
f_test  = to_categoricals(f_test, cat_cols)

# Update cat_cols
cat_cols = f_train.select_dtypes(include=["object","category"]).columns.tolist()


lgb_model = lgb.LGBMClassifier(
    n_estimators=10000,
    learning_rate=0.015,
    objective="binary",
    metric="auc",
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth=-1,
    seed=42,
    n_jobs=-1,
    device_type="gpu",  # If your LightGBM was not compiled with GPU support, this will throw an error. Change to "cpu" if needed.
    reg_lambda=1,
)

lgb_model.fit(
    X=f_train,
    y=train_df[TARGET],
    categorical_feature=cat_cols
)

test_preds  = lgb_model.predict_proba(f_test)[:,1]

submission = pd.DataFrame({
    "id": test_df["id"],
    "loan_paid_back": test_preds
})

submission.to_csv("./submission.csv", index=False)
print("submission saved.")

