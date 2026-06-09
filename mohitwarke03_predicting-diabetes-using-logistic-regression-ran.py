%load_ext autoreload
%autoreload 2

import warnings
warnings.filterwarnings("ignore")


import os
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def setup(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 25)
    return seed

seed = setup()



def running_in_kaggle():
    return os.path.isdir("/kaggle/input")

DATA_DIR = Path("/kaggle/input/playground-series-s5e12") if running_in_kaggle() else Path("data")

# Load train
train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df.head()



test_df = pd.read_csv(DATA_DIR / "test.csv")
test_df.head()


print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

train_df.info()
train_df.describe().T


TARGET = "diagnosed_diabetes"

y = train_df[TARGET]
X = train_df.drop(columns=[TARGET])
X_test = test_df.copy()


cat_cols = []
num_cols = []

for col in X.columns:
    if X[col].dtype == "object":
        cat_cols.append(col)
    else:
        num_cols.append(col)

cat_cols, num_cols


encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])


scaler = StandardScaler()

X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=seed, stratify=y
)

print("Train size:", X_train.shape, "Val size:", X_val.shape)


log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train, y_train)

log_val_pred = log_reg.predict_proba(X_val)[:, 1]
log_auc = roc_auc_score(y_val, log_val_pred)

print("Logistic Regression AUC:", log_auc)


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=seed
)
rf.fit(X_train, y_train)

rf_val_pred = rf.predict_proba(X_val)[:, 1]
rf_auc = roc_auc_score(y_val, rf_val_pred)

print("Random Forest AUC:", rf_auc)


lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "max_depth": -1,
    "num_leaves": 32,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "verbose": -1,
}

lgb_model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=2000,
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(100)
    ]
)

lgb_val_pred = lgb_model.predict(X_val)
lgb_auc = roc_auc_score(y_val, lgb_val_pred)

print("LightGBM AUC:", lgb_auc)


print("LR AUC :", log_auc)
print("RF AUC :", rf_auc)
print("LGB AUC:", lgb_auc)

best_model = None

if lgb_auc == max(log_auc, rf_auc, lgb_auc):
    best_model = "lightgbm"
elif rf_auc == max(log_auc, rf_auc, lgb_auc):
    best_model = "rf"
else:
    best_model = "lr"

print("Best Model =", best_model)



if best_model == "lr":
    final_model = LogisticRegression(max_iter=2000)
    final_model.fit(X, y)
    test_pred = final_model.predict_proba(X_test)[:, 1]

elif best_model == "rf":
    final_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=seed
    )
    final_model.fit(X, y)
    test_pred = final_model.predict_proba(X_test)[:, 1]

else:
    final_model = lgb.train(
        params,
        lgb.Dataset(X, y),
        num_boost_round=lgb_model.best_iteration
    )
    test_pred = final_model.predict(X_test)


sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub[TARGET] = test_pred
sub.head()




sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")


if best_model == "rf":
    importances = rf.feature_importances_
    feat = X.columns

elif best_model == "lightgbm":
    importances = final_model.feature_importance()
    feat = X.columns

else:
    importances = log_reg.coef_[0]
    feat = X.columns

fi_df = pd.DataFrame({"Feature": feat, "Importance": importances})
fi_df.sort_values("Importance", ascending=False).head(20)


sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
sample[TARGET] = test_pred
sample.to_csv("/kaggle/working/submission.csv", index=False)




