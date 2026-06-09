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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")


df_train.head()


import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr = df_train.corr(numeric_only=True)

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr,
    annot=True,          # show correlation values
    fmt=".2f",           # 2 decimal places
    cmap="coolwarm",     # color palette
    center=0,            # 0 in the middle of color scale
    linewidths=0.5
)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()


y = df_train["BeatsPerMinute"]              # Target column
X = df_train.drop(columns=["BeatsPerMinute"])  # All other columns as features


import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cat_feats = [c for c in X_train.columns if str(X_train[c].dtype) == "category"]

dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats or "auto")
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, categorical_feature=cat_feats or "auto")

params = {
    "objective": "regression",
    "metric": "rmse",             
    "learning_rate": 0.03,
    "num_leaves": 64,           
    "max_depth": -1,             
    "min_data_in_leaf": 50,        
    "feature_fraction": 0.9,      
    "bagging_fraction": 0.9,   
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "min_gain_to_split": 0.0,
    "max_bin": 255,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=10000,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=2000, verbose=False),
        lgb.log_evaluation(period=1000),
    ],
)

print("Best iteration:", model.best_iteration)



from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score

lgbm = LGBMRegressor(
    objective="regression",
    learning_rate=0.03,
    n_estimators=10000,      
    num_leaves=64,
    max_depth=-1,
    min_child_samples=50,    
    subsample=0.9,      
    subsample_freq=1,   
    colsample_bytree=0.9,      
    reg_alpha=0.0,         
    reg_lambda=1.0, 
    random_state=42,
    n_jobs=-1,
)

# Early stopping requires a validation set:
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(200, verbose=False)]
)

# Cross-validation example (no early stopping here)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lgbm.set_params(n_estimators=lgbm.best_iteration_ or 1000),
                         X, y, cv=cv, scoring="neg_root_mean_squared_error")
print("CV RMSE:", (-scores).mean())



df_test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# Generate predictions
preds = model.predict(df_test)

# Create submission dataframe
submission = pd.DataFrame({
    "id": df_test["id"], 
    "BeatsPerMinute": preds
})

# Export to CSV
submission.to_csv("submission.csv", index=False)
submission.head()

