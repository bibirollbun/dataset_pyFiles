# did this categorization manually :)

# train.columns

num_cols = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides','family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'diagnosed_diabetes']

oe_cols = ['education_level', 'income_level'] # ordinal encoded columns : as values have hierarchy
ohe_cols = ['gender', 'ethnicity', 'smoking_status', 'employment_status'] # one hot encoded columns


# load libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

# changing settings for better visibility
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)  # Show all columns

# read files
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

train.head(5)


# these above questions can be answered before doing the split 
# let's do the ohe and ordinal encoding first in that case
from sklearn.preprocessing import OneHotEncoder

ohe_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

ohe_enc_cols = pd.DataFrame(
    ohe_enc.fit_transform(train[ohe_cols]),
    columns = ohe_enc.get_feature_names_out()
)


from sklearn.preprocessing import OrdinalEncoder

education_level_cat = ['No formal', 'Highschool', 'Graduate', 'Postgraduate']
income_level_cat = ['Low','Lower-Middle', 'Middle','Upper-Middle', 'High']

oe_enc = OrdinalEncoder(categories = [education_level_cat, income_level_cat])

oe_enc_cols = pd.DataFrame(
    oe_enc.fit_transform(train[oe_cols]),
    columns = oe_enc.get_feature_names_out()
)


# pushing the data back to the train dataset
train = pd.concat([train.drop(columns = ohe_cols + oe_cols), oe_enc_cols, ohe_enc_cols], axis = 1)

# pushing diagnosed_diabeetes column at last
train['diagnosed_diabetes'] = train.pop('diagnosed_diabetes')


# looking for missing data 
# for col in train.columns :
#     print(f"missing data for {col} is : {train[col].isna().sum()/len(train)}")

# GPT suggested code for better aesthetics
print("Missing Data Summary".center(100, "="))

for col, frac in train.isna().mean().items():
    print(f"{col:<25} : {frac:>6.2%}")

print("=" * 100)



# ----------------------------
# Blood pressure features
# ----------------------------
train["pulse_pressure"] = train["systolic_bp"] - train["diastolic_bp"]
train["map"] = (2 * train["diastolic_bp"] + train["systolic_bp"]) / 3

# ----------------------------
# Lipid ratios
# ----------------------------
train["tg_hdl_ratio"] = train["triglycerides"] / train["hdl_cholesterol"]
train["ldl_hdl_ratio"] = train["ldl_cholesterol"] / train["hdl_cholesterol"]
train["totalchol_hdl_ratio"] = train["cholesterol_total"] / train["hdl_cholesterol"]
train["non_hdl_cholesterol"] = train["cholesterol_total"] - train["hdl_cholesterol"]

# ----------------------------
# Log transforms (heavy-tailed)
# ----------------------------
train["log_triglycerides"] = np.log1p(train["triglycerides"])
train["log_cholesterol_total"] = np.log1p(train["cholesterol_total"])

# ----------------------------
# Age anchors
# ----------------------------
train["age_over_40"] = (train["age"] >= 40).astype(int)
train["age_over_50"] = (train["age"] >= 50).astype(int)



# since there's no missing data we move on to the next steps to see the distribution 
# we are only keeping variables which are continuous
cont_var = ['age','alcohol_consumption_per_week', 'diet_score',
           'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
           'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
           'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
           'triglycerides']

# creating a long dataset which can be fed to FacetGrid function from the seaborn library
# another tweak suggested by GPT
# https://www.kaggle.com/tilii7 was my inspiration for drawing these beautiful charts 
# i myself had an ungly ass code piece
long_df = train.melt(
    id_vars="diagnosed_diabetes",
    value_vars=cont_var,
    var_name="feature",
    value_name="value"
)


# viewing the distriubution for the above variables
sns.set(style="whitegrid")

g = sns.FacetGrid(
    long_df,
    col="feature",
    col_wrap=3,
    hue="diagnosed_diabetes",
    sharex=False,
    sharey=False,
    height=3
)

g.map(sns.kdeplot, "value", common_norm=False)
g.add_legend()
plt.show()


# Calculate the correlation matrix
corr_matrix = train.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(25, 15))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu', center=0, linewidths=0.5, fmt=".2f")
plt.show()


# # doing the data split
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(train.drop(columns =['diagnosed_diabetes']), 
#                  train['diagnosed_diabetes'], 
#                  train_size = 0.8, 
#                  stratify = train['diagnosed_diabetes'])

X_train, y_train = train.drop(columns =['diagnosed_diabetes']),train['diagnosed_diabetes']


# import numpy as np
# import optuna
# import xgboost as xgb

# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold


# def objective(trial):

#     params = {
#         "objective": "binary:logistic",
#         "eval_metric": "auc",

#         # ---- GPU settings ----
#         "tree_method": "gpu_hist",
#         "predictor": "gpu_predictor",
#         "device": "cuda",

#         # ---- hyperparams ----
#         "n_estimators": trial.suggest_int("n_estimators", 150, 600),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
#         "max_depth": trial.suggest_int("max_depth", 3, 5),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),

#         "random_state": 42,
#     }

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     fold_scores = []

#     for train_idx, val_idx in skf.split(X_train, y_train):
#         X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         model = xgb.XGBClassifier(**params)

#         model.fit(
#             X_tr,
#             y_tr,
#             eval_set=[(X_val, y_val)],
#             early_stopping_rounds=50,
#             verbose=False,
#         )

#         val_preds = model.predict_proba(X_val)[:, 1]
#         fold_scores.append(roc_auc_score(y_val, val_preds))

#     return np.mean(fold_scores)



# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=1)


# print("Best CV AUC:", study.best_value)
# print("Best params:")
# for k, v in study.best_params.items():
#     print(f"  {k}: {v}")


# best_model = xgb.XGBClassifier(
#     **study.best_params,
#     objective="binary:logistic",
#     eval_metric="auc",
#     tree_method="gpu_hist",
#     predictor="gpu_predictor",
#     device="cuda",
#     random_state=42,
# )

# best_model.fit(X_train, y_train)

# test_preds = best_model.predict_proba(X_train)[:, 1]
# test_auc = roc_auc_score(y_train, test_preds)

# print(f"Final Test AUC: {test_auc:.4f}")


# # best params for xgboost 
# Best params:
#   n_estimators: 239
#   learning_rate: 0.09581679117443968
#   max_depth: 5
#   min_child_weight: 18
#   subsample: 0.7226511673146256
#   colsample_bytree: 0.823448150557
#   reg_alpha: 2.283381643663481
#   reg_lambda: 2.8440841480847707
# Final Test AUC: 0.7304


# # code for light gbm
# import numpy as np
# import optuna
# import lightgbm as lgb

# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold


# def objective(trial):

#     max_depth = trial.suggest_int("max_depth", 3, 5)

#     params = {
#         "objective": "binary",
#         "metric": "auc",

#         # ---- GPU settings ----
#         "device": "gpu",
#         "gpu_platform_id": 0,
#         "gpu_device_id": 0,

#         # ---- hyperparams ----
#         "n_estimators": trial.suggest_int("n_estimators", 150, 600),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
#         "max_depth": max_depth,
#         "num_leaves": trial.suggest_int(
#             "num_leaves", 2 ** max_depth, 2 ** (max_depth + 1)
#         ),
#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#         "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
#         "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 5.0),

#         "verbosity": -1,
#         "seed": 42,
#     }

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     fold_scores = []

#     for train_idx, val_idx in skf.split(X_train, y_train):
#         X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         model = lgb.LGBMClassifier(**params)

#         model.fit(
#             X_tr,
#             y_tr,
#             eval_set=[(X_val, y_val)],
#             eval_metric="auc",
#             callbacks=[lgb.early_stopping(50)],
#         )

#         val_preds = model.predict_proba(X_val)[:, 1]
#         fold_scores.append(roc_auc_score(y_val, val_preds))

#     return np.mean(fold_scores)


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

# print("Best CV AUC:", study.best_value)
# print("Best params:")
# for k, v in study.best_params.items():
#     print(f"  {k}: {v}")


# best_model = lgb.LGBMClassifier(
#     **study.best_params,
#     objective="binary",
#     metric="auc",
#     device="gpu",
#     seed=42,
# )

# best_model.fit(X_train, y_train)

# train_preds = best_model.predict_proba(X_train)[:, 1]
# train_auc = roc_auc_score(y_train, train_preds)

# print(f"Final Train AUC: {train_auc:.4f}")


# # best params for lightgbm
# Best params:
#   max_depth: 5
#   n_estimators: 581
#   learning_rate: 0.09162799904204544
#   num_leaves: 57
#   min_child_samples: 78
#   bagging_fraction: 0.9168034182166901
#   feature_fraction: 0.7790074516971525
#   lambda_l1: 2.129653490992313
#   lambda_l2: 0.8645343646620951
# Final Train AUC: 0.7435


import numpy as np
import optuna
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def objective(trial):

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
    
        # ---- GPU ----
        "task_type": "GPU",
        "devices": "0",
    
        # ---- bootstrap ----
        "bootstrap_type": "Bernoulli",
    
        # ---- hyperparams ----
        "iterations": trial.suggest_int("iterations", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    
        # ---- stability ----
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = CatBoostClassifier(**params)

        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            use_best_model=True,
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        fold_scores.append(roc_auc_score(y_val, val_preds))

    return np.mean(fold_scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

print("Best CV AUC:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")


best_model = CatBoostClassifier(
    **study.best_params,
    loss_function="Logloss",
    eval_metric="AUC",
    task_type="GPU",
    devices="0",

    # THIS WAS MISSING
    bootstrap_type="Bernoulli",

    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)


best_model.fit(X_train, y_train)

train_preds = best_model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, train_preds)

print(f"Final Train AUC: {train_auc:.4f}")


ohe_enc_cols = pd.DataFrame(
    ohe_enc.transform(test[ohe_cols]),
    columns = ohe_enc.get_feature_names_out()
)

oe_enc_cols = pd.DataFrame(
    oe_enc.transform(test[oe_cols]),
    columns = oe_enc.get_feature_names_out()
)

# pushing the data back to the train dataset
test = pd.concat([test.drop(columns = ohe_cols + oe_cols), oe_enc_cols, ohe_enc_cols], axis = 1)

# ----------------------------
# Blood pressure features
# ----------------------------
test["pulse_pressure"] = test["systolic_bp"] - test["diastolic_bp"]
test["map"] = (2 * test["diastolic_bp"] + test["systolic_bp"]) / 3

# ----------------------------
# Lipid ratios
# ----------------------------
test["tg_hdl_ratio"] = test["triglycerides"] / test["hdl_cholesterol"]
test["ldl_hdl_ratio"] = test["ldl_cholesterol"] / test["hdl_cholesterol"]
test["totalchol_hdl_ratio"] = test["cholesterol_total"] / test["hdl_cholesterol"]
test["non_hdl_cholesterol"] = test["cholesterol_total"] - test["hdl_cholesterol"]

# ----------------------------
# Log transforms (heavy-tailed)
# ----------------------------
test["log_triglycerides"] = np.log1p(test["triglycerides"])
test["log_cholesterol_total"] = np.log1p(test["cholesterol_total"])

# ----------------------------
# Age anchors
# ----------------------------
test["age_over_40"] = (test["age"] >= 40).astype(int)
test["age_over_50"] = (test["age"] >= 50).astype(int)


preds = best_model.predict_proba(test)[:, 1]

sub_data = pd.concat([test['id'],pd.Series(preds, name ='diagnosed_diabetes')], axis = 1)

sub_data.to_csv('submission.csv',index=False)

