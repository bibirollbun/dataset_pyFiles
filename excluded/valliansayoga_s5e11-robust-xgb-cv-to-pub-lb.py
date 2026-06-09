import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv").drop("id", axis=1)
train.head()


train.isna().sum()


from math import ceil


cat_cols = train.select_dtypes(object).columns

ncols = 3
nrows = ceil(len(cat_cols) / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, layout="constrained", 
                         figsize=[ncols*5, nrows*5])
axes = axes.flatten()
for col, ax in zip(cat_cols, axes):
    sns.countplot(y=train[col], ax=ax, hue=train.loan_paid_back.astype(int))


def preprocess(df, train=True, categories=None):
    higher_ed = [
        "Bachelor's",
        "Master's",
        "PhD",
    ]
    df["has_higher_ed"] = df.education_level.isin(higher_ed).astype(int)
    
    productive_loan = [
        "Business",
        "Education",
    ]
    df["productive_loan"] = df.loan_purpose.isin(higher_ed).astype(int)
    df["has_income"] = (~df.employment_status.eq('Student')).astype(int)

    if train:
        grade = sorted(df.grade_subgrade.unique(), reverse=True)
        df.grade_subgrade = pd.Categorical(df.grade_subgrade, categories=grade, ordered=True)

        categories = dict()
        for col in cat_cols:
            df[col] = pd.Categorical(df[col])
            categories[col] = df[col].dtype
        return df, categories
    else:
        for col in cat_cols:
            df[col] = df[col].astype(categories[col])
    return df

train, categories = preprocess(train)


target = "loan_paid_back"
X = train.drop(target, axis=1)
y = train[target]


import xgboost as xgb

dtrain = xgb.DMatrix(X, y, enable_categorical=True)


params = {
    # --- Global Configuration ---
    "verbosity": 1,            # 0 (silent), 1 (warning), 2 (info), 3 (debug)
    "nthread": 4,           # int >= 1 or None

    # --- General Parameters ---
    "booster": "gbtree",       # "gbtree", "gblinear", "dart"
    "device": "cuda",           # "cpu", "cuda", "cuda:<ordinal>", "gpu", "gpu:<ordinal>"
    "eta": 0.05,                # range: [0,1]
    "gamma": 0.1,                # range: [0,∞)
    "max_depth": 6,            # int >= 0
    "min_child_weight": 1.5,     # float >= 0
    "max_delta_step": 1,       # float >= 0
    "subsample": 0.8,          # range: (0,1]
    "sampling_method": "gradient_based",  # "uniform", "gradient_based"
    "colsample_bytree": 0.9,   # range: (0,1]
    "colsample_bylevel": 0.9,  # range: (0,1]
    "colsample_bynode": 0.9,   # range: (0,1]
    "lambda": 5,             # range: [0,∞)
    "alpha": 1.5,              # range: [0,∞)
    "tree_method": "auto",     # "auto", "exact", "approx", "hist", "gpu_hist"
    "grow_policy": "lossguide",# "depthwise", "lossguide"
    "max_leaves": 0,           # int >= 0
    "max_bin": 256,            # int >= 2
    "num_parallel_tree": 2,    # int >= 1

    "objective": "binary:logistic",  # see below for full list
    "eval_metric": "auc",        # list[str] or str
    "seed": 0,                  # int
}

num_round = 5000
verbose_eval = 100
print("running cross validation")
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
result = xgb.cv(
    params,
    dtrain,
    num_round,
    stratified=True, 
    nfold=5,
    metrics={"auc"},
    seed=0,
    early_stopping_rounds=20,
    callbacks=[xgb.callback.EvaluationMonitor(period=verbose_eval, show_stdv=True)],
)


result["train-auc-mean"].plot(label="train-auc-mean")
result["test-auc-mean"].plot(label="test-auc-mean")
plt.legend()


max_round = result.shape[0]
max_round


model = xgb.train(params, dtrain, num_boost_round=max_round, verbose_eval=100)


test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
test.head()


test = preprocess(test, train=False, categories=categories)


dtest = xgb.DMatrix(test[X.columns], enable_categorical=True)


test[target] = model.predict(dtest)


test[["id", target]].to_csv("submission.csv", index=False)


!head submission.csv

