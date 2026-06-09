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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import roc_auc_score



data=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
print("data row x column:", data.shape)
print(data.head())


# model using XGBoost
X=data.copy()
X.pop("id")
y=X.pop("diagnosed_diabetes")

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False),
     make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

# Train/validation split
X_train,X_val,y_train,y_val = train_test_split(
    X,y,test_size = 0.2,random_state = 42,stratify = y
)


import xgboost as xgb
from sklearn.metrics import roc_auc_score

model_xgb = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",      # fast for large data
    random_state=42,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])
)

model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

preds = model_xgb.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, preds)
print("Validation AUC:", auc)




# model using LightGBM
X=data.copy()
X.pop("id")
y=X.pop("diagnosed_diabetes")

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns:", categorical_cols)

for col in categorical_cols:
    X[col] = X[col].astype("category")



# Train/validation split
X_train,X_val,y_train,y_val = train_test_split(
    X,y,test_size = 0.2,random_state = 42,stratify = y
)

import lightgbm as lgb
from lightgbm import LGBMClassifier


model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    categorical_feature=categorical_cols,
    callbacks=[
        lgb.log_evaluation(period=50)
    ]
)

val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)



# tuning experiments 1
model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.06,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    categorical_feature=categorical_cols,
    callbacks=[
        lgb.log_evaluation(period=50)
    ]
)

val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)


# tuning experiments 2
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    categorical_feature=categorical_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)


# final model LightGBM
X_full = X.copy()
y_full = y.copy()
categorical_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns:", categorical_cols)

for col in categorical_cols:
    X_full[col] = X_full[col].astype("category")

lgb_model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model.fit(
    X_full, y_full,
    categorical_feature=categorical_cols,
)


X_full.shape


df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
X_test=df_test.copy()
X_test.pop("id")
print(X_test.shape)
categorical_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_cols:
    X_test[col] = X_test[col].astype("category")


test_pred = lgb_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    "id": df_test["id"],
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission_1.csv", index=False)

print("Your submission was successfully saved!")


import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val, plot_type="bar")
shap.summary_plot(shap_values, X_val)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_var = np.var(shap_values, axis=0)
shap_df = pd.DataFrame({
    "feature": X_val.columns,
    "mean_abs_shap": mean_abs_shap,
    "var_shap": shap_var
}).sort_values(by="mean_abs_shap", ascending=False)

print("\nTop 20 important features:")
print(shap_df.head(20))


low_impact_threshold = 0.05 # adjust based on your data
# high_variance_threshold = np.percentile(shap_var, 75)

features_to_consider_drop = shap_df[
    shap_df["mean_abs_shap"] < low_impact_threshold 
]["feature"].tolist()

print("\nFeatures suggested for pruning:")
print(features_to_consider_drop)


X=data.copy()
X.pop("id")
y=X.pop("diagnosed_diabetes")

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns:", categorical_cols)

for col in categorical_cols:
    X[col] = X[col].astype("category")



# Train/validation split
X_train,X_val,y_train,y_val = train_test_split(
    X,y,test_size = 0.2,random_state = 42,stratify = y
)

X_train_pruned = X_train.drop(columns=features_to_consider_drop)
X_valid_pruned = X_val.drop(columns=features_to_consider_drop)


model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train_pruned , y_train,
    eval_set=[(X_valid_pruned, y_val)],
    eval_metric="auc",
    categorical_feature= [c for c in categorical_cols if c in X_train_pruned.columns],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)
val_pred = model.predict_proba(X_valid_pruned)[:, 1]
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)


# Use 5-fold CV Optuna objective
import optuna
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X=data.copy()
X.pop("id")
y=X.pop("diagnosed_diabetes")

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns:", categorical_cols)

for col in categorical_cols:
    X[col] = X[col].astype("category")


def objective(trial):

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "verbosity": -1,
        "seed": 42
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

       
        for col in categorical_cols:
            X_val[col] = X_val[col].cat.set_categories(
                X_train[col].cat.categories
            )

        train_data = lgb.Dataset(
            X_train,
            y_train,
            categorical_feature=categorical_cols,
            free_raw_data=False
        )

        val_data = lgb.Dataset(
            X_val,
            y_val,
            categorical_feature=categorical_cols,
            free_raw_data=False
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        aucs.append(roc_auc_score(y_val, preds))

    return np.mean(aucs)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best CV AUC:", study.best_value)
print("Best params:", study.best_params)


best_params = study.best_params
best_params.update({
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "seed": 42
})

final_model = lgb.LGBMClassifier(
    **best_params,
    n_estimators=2000
)

final_model.fit(
    X,
    y,
    categorical_feature=categorical_cols
)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
X_test=df_test.copy()
X_test.pop("id")
print(X_test.shape)
categorical_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_cols:
    X_test[col] = X_test[col].astype("category")

test_pred = final_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    "id": df_test["id"],
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission.csv", index=False)

print("Your submission was successfully saved!")


# Try another model
from catboost import CatBoostClassifier

X=data.copy()
X.pop("id")
y=X.pop("diagnosed_diabetes")

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns:", categorical_cols)

for col in categorical_cols:
    X[col] = X[col].astype("category")



# Train/validation split
X_train,X_val,y_train,y_val = train_test_split(
    X,y,test_size = 0.2,random_state = 42,stratify = y
)
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    cat_features=categorical_cols,
    early_stopping_rounds=100
)

val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)




