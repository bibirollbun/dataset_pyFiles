

from IPython.display import display, Markdown
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import math
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd 
import scipy.stats as ss
import seaborn as sns
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import optuna.visualization as vis
import matplotlib.pyplot as plt
import catboost as cb



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Decide between local or kaggle cloud storage         
KAGGLE_ENV = 'kaggle' in os.listdir('/')
data_path = '/kaggle/input' if KAGGLE_ENV else '../kaggle/input'

# This is a good idea to work only locally. But If you wanna ran your NB also at kaggle... this is not working.
# # Pull the dataset from kaggle, it is concat dataset train + original dataset
# dataset_name = 'dantheshark/s4-e11-train-concat'
# if KAGGLE_ENV:
#     kaggle.api.dataset_download_files(dataset_name, path="../kaggle/input/", unzip=True)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 


# # Load the data
# train_original = pd.read_csv(data_path + '/playground-series-s4e11/train.csv')
# test_original = pd.read_csv(data_path + '/playground-series-s4e11/test.csv')
# sample_submission = pd.read_csv(data_path + '/playground-series-s4e11/sample_submission.csv')

# original_data = pd.read_csv(data_path + '/depression-surveydataset-for-analysis/final_depression_dataset_1.csv')

# train_concat_data = pd.read_csv(data_path + '/s4-e11-train-concat/s4-e11-train-concat.csv')
# train_final_data = pd.read_csv(data_path + '/s4-e11-train-concat-final/s4-e11-train-concat-final.csv')

# test_concat_data = pd.read_csv(data_path + '/s4-e11-test-concat/s4-e11-test-concat.csv')
# test_final_data = pd.read_csv(data_path + '/s4-e11-test-concat-final/s4-e11-test-concat-final.csv')
submission_template = pd.read_csv(data_path + '/playground-series-s5e3/sample_submission.csv')


# Load the eda data
test_final_data = pd.read_csv(data_path + '/playground-series-s5e3-test-final' + '/playground-series-s5e3-test-final.csv')
train_final_data = pd.read_csv(data_path + '/playground-series-s5e3-train-final' + '/playground-series-s5e3-train-final.csv')

# # Original Data
train_original = pd.read_csv(data_path + '/playground-series-s5e3' + '/train.csv')
test_original = pd.read_csv(data_path + '/playground-series-s5e3' + '/test.csv')
# sample_submission = pd.read_csv(data_path + competition_name+ '/sample_submission.csv')
# original_data = pd.read_csv(data_path  + '/rainfall-prediction-using-machine-learning/Rainfall.csv')


X = train_final_data.drop(columns=["rainfall"])
y = train_final_data["rainfall"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
X_test_scaled = scaler.transform(test_final_data)

# DMatrix for XGBoost, better performance (speed, memory usage compare to pandas and numpy)
D_train = xgb.DMatrix(X_train_scaled, label=y_train)
D_holdout = xgb.DMatrix(X_holdout_scaled)
D_kaggle = xgb.DMatrix(X_test_scaled)
D_train_cv = xgb.DMatrix(X_train_scaled, label=y_train)

def objective(trial):
    params = {
        "objective": "binary:logistic", # binary classification
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2), # (eta) step size shrinkage used in update to prevents overfitting
        "max_depth": trial.suggest_int("max_depth", 3, 10), # maximum depth of a tree, increase depth will increase model complexity
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), # fraction of samples to be used for each tree, lower values prevent overfitting
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), # fraction of features to be used for each tree, lower values prevent overfitting
        "gamma": trial.suggest_float("gamma", 0.0, 0.5), # minimum loss reduction required to make a further partition on a leaf node, higher values prevent overfitting
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0), # L1 regularization, lasso, delete irrelevant features
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0), # L2 regularization, ridge, prevent overfitting
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),# min sum of hessian in leaf node
        "max_delta_step": trial.suggest_float("max_delta_step", 0, 10), # max delta step we allow each tree weight estimation to be
        "sampling_method": trial.suggest_categorical("sampling_method", ["uniform"]), # random sample is uniform
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]), # depthwise: grow uniformly, lossguide: grow based on loss, better for bigger datasets
        "scale_pos_weight": (y_train.value_counts()[0] / y_train.value_counts()[1]) ** 0.3, # weight of positive class in binary classification
        "tree_method": "hist", # histogram-based algorithm, faster and uses less memory
        "device": "cuda",# GPU acceleration
        "eval_metric": "auc"# AUC metric for binary classification, needed metric from kaggle competition
    }

    num_boost_round = trial.suggest_int("num_boost_round", 100, 600) # number of boosting rounds, number of trees to build, everytime different set of hyperparameters

    D_train_cv = xgb.DMatrix(X_train_scaled, label=y_train)

    cv_results = xgb.cv(
        params=params,
        dtrain=D_train_cv,
        num_boost_round=num_boost_round,
        nfold=5,
        stratified=True,
        early_stopping_rounds=50, # stop training if no improvement in 50 rounds
        metrics="auc",
        seed=42,
        verbose_eval=False,
        as_pandas=True
    )

    return cv_results["test-auc-mean"].max()

# optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

best_params = study.best_trial.params
num_boost_round = best_params.pop("num_boost_round")
print("Best hyperparameters:", best_params)

# use the best hyperparameters to train the final model
D_final_train = xgb.DMatrix(X_train_scaled, label=y_train)


final_model = xgb.train( # better than fit, because we can use DMatrix and have more control (evals is not possible with fit)
    best_params,
    D_final_train,
    num_boost_round=num_boost_round,
    evals=[(D_final_train, "train")],
    verbose_eval=False
)

# evaluate the model on the holdout set
D_holdout = xgb.DMatrix(X_holdout)
y_pred_proba = final_model.predict(D_holdout)
auc_holdout = roc_auc_score(y_holdout, y_pred_proba)
print(f"Holdout ROC-AUC: {auc_holdout:.5f}")

# save the model
MODEL_PATH = "best_xgb_model.pkl"
joblib.dump(final_model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")


y_pred_kaggle = final_model.predict(D_kaggle)
submission_template["rainfall"] = y_pred_kaggle  
submission_template.to_csv("submission_WIN.csv", index=False)
print("Submission file saved as submission_WIN.csv")



# split the data into train and holdout sets
X = train_final_data.drop(columns=["rainfall"])
y = train_final_data["rainfall"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
X_test_scaled = scaler.transform(test_final_data)

xgb_selector_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    n_estimators=300,
    random_state=42,
    tree_method="hist",
    device="cuda"
)
xgb_selector_model.fit(X_train_scaled, y_train)

selector = SelectFromModel(xgb_selector_model, threshold="median", prefit=True)
X_train_sel = selector.transform(X_train_scaled)
X_holdout_sel = selector.transform(X_holdout_scaled)
X_test_sel = selector.transform(X_test_scaled)

selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected features:", list(selected_features))


lgbm = lgb.LGBMClassifier(n_estimators=300, random_state=42)
xgbm = xgb.XGBClassifier(n_estimators=300, random_state=42, tree_method="hist", device="cuda", use_label_encoder=False)
cat = cb.CatBoostClassifier(n_estimators=300, verbose=0, random_state=42)

lgbm.fit(X_train_sel, y_train)
xgbm.fit(X_train_sel, y_train)
cat.fit(X_train_sel, y_train)

# get probabilities for the holdout set
lgbm_probs = lgbm.predict_proba(X_holdout_sel)[:, 1]
xgbm_probs = xgbm.predict_proba(X_holdout_sel)[:, 1]
cat_probs = cat.predict_proba(X_holdout_sel)[:, 1]


# stacking
stacked_train = np.vstack([lgbm_probs, xgbm_probs, cat_probs]).T
meta_model = LogisticRegression()
meta_model.fit(stacked_train, y_holdout)

# roc auc on holdout set
stacked_pred = meta_model.predict_proba(stacked_train)[:, 1]
auc = roc_auc_score(y_holdout, stacked_pred)
print(f"Meta-Model ROC-AUC on Holdout: {auc:.5f}")



# get probabilities for the test set
lgbm_test_probs = lgbm.predict_proba(X_test_sel)[:, 1]
xgbm_test_probs = xgbm.predict_proba(X_test_sel)[:, 1]
cat_test_probs = cat.predict_proba(X_test_sel)[:, 1]

stacked_test = np.vstack([lgbm_test_probs, xgbm_test_probs, cat_test_probs]).T
kaggle_preds = meta_model.predict_proba(stacked_test)[:, 1]

submission_template["rainfall"] = kaggle_preds
submission_template.to_csv("submission_stack_sel.csv", index=False)
print("Submission saved as submission_stack_sel.csv")


# temperature range: max - min
train_final_data["temp_range"] = train_final_data["maxtemp"] - train_final_data["mintemp"]
test_final_data["temp_range"] = test_final_data["maxtemp"] - test_final_data["mintemp"]

# humidity ratio: humidity / dewpoint
train_final_data["humidity_ratio"] = train_final_data["humidity"] / (train_final_data["dewpoint"] + 1e-5)
test_final_data["humidity_ratio"] = test_final_data["humidity"] / (test_final_data["dewpoint"] + 1e-5)

# humidity and sunshine interaction
train_final_data["humid_sun"] = train_final_data["humidity"] * train_final_data["sunshine"]
test_final_data["humid_sun"] = test_final_data["humidity"] * test_final_data["sunshine"]

# day cycles in sin/cosine
train_final_data["day_sin"] = np.sin(2 * np.pi * train_final_data["day"] / 31)
train_final_data["day_cos"] = np.cos(2 * np.pi * train_final_data["day"] / 31)
test_final_data["day_sin"] = np.sin(2 * np.pi * test_final_data["day"] / 31)
test_final_data["day_cos"] = np.cos(2 * np.pi * test_final_data["day"] / 31)



X = train_final_data.drop(columns=["rainfall"])
y = train_final_data["rainfall"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
X_test_scaled = scaler.transform(test_final_data)

# feature selection
xgb_selector_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    n_estimators=300,
    random_state=42,
    tree_method="hist",
    device="cuda"
)
xgb_selector_model.fit(X_train_scaled, y_train)

selector = SelectFromModel(xgb_selector_model, threshold="median", prefit=True)
X_train_sel = selector.transform(X_train_scaled)
X_holdout_sel = selector.transform(X_holdout_scaled)
X_test_sel = selector.transform(X_test_scaled)

selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected features:", list(selected_features))

lgbm = lgb.LGBMClassifier(n_estimators=300, random_state=42)
xgbm = xgb.XGBClassifier(n_estimators=300, random_state=42, tree_method="hist", device="cuda", use_label_encoder=False)
cat = cb.CatBoostClassifier(n_estimators=300, verbose=0, random_state=42)

lgbm.fit(X_train_sel, y_train)
xgbm.fit(X_train_sel, y_train)
cat.fit(X_train_sel, y_train)

# probabilities for the holdout set
lgbm_probs = lgbm.predict_proba(X_holdout_sel)[:, 1]
xgbm_probs = xgbm.predict_proba(X_holdout_sel)[:, 1]
cat_probs = cat.predict_proba(X_holdout_sel)[:, 1]

# training the meta-model
stacked_train = np.vstack([lgbm_probs, xgbm_probs, cat_probs]).T
meta_model = LogisticRegression()
meta_model.fit(stacked_train, y_holdout)

# ROC Auc on holdout set
stacked_pred = meta_model.predict_proba(stacked_train)[:, 1]
auc = roc_auc_score(y_holdout, stacked_pred)
print(f"Meta-Model ROC-AUC on Holdout: {auc:.5f}")


# get probabilities for the test set
lgbm_test_probs = lgbm.predict_proba(X_test_sel)[:, 1]
xgbm_test_probs = xgbm.predict_proba(X_test_sel)[:, 1]
cat_test_probs = cat.predict_proba(X_test_sel)[:, 1]

stacked_test = np.vstack([lgbm_test_probs, xgbm_test_probs, cat_test_probs]).T
kaggle_preds = meta_model.predict_proba(stacked_test)[:, 1]

submission_template["rainfall"] = kaggle_preds
submission_template.to_csv("submission_stack_sel_feature.csv", index=False)
print("Submission saved as submission_stack_sel_feature.csv")


# --- Feature Engineering ---
train_final_data["temp_range"] = train_final_data["maxtemp"] - train_final_data["mintemp"]
test_final_data["temp_range"] = test_final_data["maxtemp"] - test_final_data["mintemp"]

train_final_data["humidity_ratio"] = train_final_data["humidity"] / (train_final_data["dewpoint"] + 1e-5)
test_final_data["humidity_ratio"] = test_final_data["humidity"] / (test_final_data["dewpoint"] + 1e-5)

train_final_data["humid_sun"] = train_final_data["humidity"] * train_final_data["sunshine"]
test_final_data["humid_sun"] = test_final_data["humidity"] * test_final_data["sunshine"]

train_final_data["day_sin"] = np.sin(2 * np.pi * train_final_data["day"] / 31)
train_final_data["day_cos"] = np.cos(2 * np.pi * train_final_data["day"] / 31)
test_final_data["day_sin"] = np.sin(2 * np.pi * test_final_data["day"] / 31)
test_final_data["day_cos"] = np.cos(2 * np.pi * test_final_data["day"] / 31)

# Split and Scale Data
X = train_final_data.drop(columns=["rainfall"])
y = train_final_data["rainfall"]
X_test = test_final_data.copy()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# --- Cross-Validation Setup ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
oof_preds_cat = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

# --- Base-Models Training ---
for fold, (train_idx, valid_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"Fold {fold + 1}")

    X_train_fold, y_train_fold = X_scaled[train_idx], y.iloc[train_idx]
    X_valid_fold, y_valid_fold = X_scaled[valid_idx], y.iloc[valid_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=300, random_state=42)
    lgb_model.fit(X_train_fold, y_train_fold)
    oof_preds_lgb[valid_idx] = lgb_model.predict_proba(X_valid_fold)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=300, random_state=42, tree_method="hist", device="cuda", use_label_encoder=False)
    xgb_model.fit(X_train_fold, y_train_fold)
    oof_preds_xgb[valid_idx] = xgb_model.predict_proba(X_valid_fold)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits

    # CatBoost
    cat_model = cb.CatBoostClassifier(n_estimators=300, verbose=0, random_state=42)
    cat_model.fit(X_train_fold, y_train_fold)
    oof_preds_cat[valid_idx] = cat_model.predict_proba(X_valid_fold)[:, 1]
    test_preds_cat += cat_model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits

# --- Meta-Model Training (auf Out-of-Fold Predictions) ---
stacked_oof = np.vstack([oof_preds_lgb, oof_preds_xgb, oof_preds_cat]).T
stacked_test = np.vstack([test_preds_lgb, test_preds_xgb, test_preds_cat]).T

meta_model = LogisticRegression()
meta_model.fit(stacked_oof, y)

# --- Evaluation ---
meta_preds_holdout = meta_model.predict_proba(stacked_oof)[:, 1]
auc = roc_auc_score(y, meta_preds_holdout)
print(f"Meta-Model ROC-AUC on Full Training Set (CV): {auc:.5f}")

# --- Final Submission ---
kaggle_preds = meta_model.predict_proba(stacked_test)[:, 1]
submission_template["rainfall"] = kaggle_preds
submission_template.to_csv("submission_stack_cv_fe2.csv", index=False)
print("Submission saved as submission_stack_cv_fe2.csv")



# Feature Engineering
def add_features(df):
    df["temp_range"] = df["maxtemp"] - df["mintemp"]
    df["humidity_ratio"] = df["humidity"] / (df["dewpoint"] + 1e-5)
    df["humid_sun"] = df["humidity"] * df["sunshine"]
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    return df

train_final_data = add_features(train_final_data)
test_final_data = add_features(test_final_data)

# Split and Scale Data
X = train_final_data.drop(columns=["rainfall"])
y = train_final_data["rainfall"]
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
X_test_scaled = scaler.transform(test_final_data)

# Feature Selection with XGBoost
xgb_selector_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    n_estimators=300,
    random_state=42,
    tree_method="hist",
    device="cuda"
)
xgb_selector_model.fit(X_train_scaled, y_train)

selector = SelectFromModel(xgb_selector_model, threshold="median", prefit=True)
X_train_sel = selector.transform(X_train_scaled)
X_holdout_sel = selector.transform(X_holdout_scaled)
X_test_sel = selector.transform(X_test_scaled)

selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected features:", list(selected_features))

# Train Base Models
lgbm = lgb.LGBMClassifier(n_estimators=300, random_state=42)
xgbm = xgb.XGBClassifier(n_estimators=300, random_state=42, tree_method="hist", device="cuda", use_label_encoder=False)
cat = cb.CatBoostClassifier(n_estimators=300, verbose=0, random_state=42)

lgbm.fit(X_train_sel, y_train)
xgbm.fit(X_train_sel, y_train)
cat.fit(X_train_sel, y_train)

# Get Probabilities for Holdout Set
lgbm_probs = lgbm.predict_proba(X_holdout_sel)[:, 1]
xgbm_probs = xgbm.predict_proba(X_holdout_sel)[:, 1]
cat_probs = cat.predict_proba(X_holdout_sel)[:, 1]

# train the meta-model
stacked_train = np.vstack([lgbm_probs, xgbm_probs, cat_probs]).T
meta_model = LogisticRegression()
meta_model.fit(stacked_train, y_holdout)

# ROC AUC on holdout set
stacked_pred = meta_model.predict_proba(stacked_train)[:, 1]
auc = roc_auc_score(y_holdout, stacked_pred)
print(f"Meta-Model ROC-AUC on Holdout: {auc:.5f}")

# Permutation Importance
result = permutation_importance(
    meta_model, stacked_train, y_holdout,
    n_repeats=20, random_state=42, scoring="roc_auc"
)

# Visualize Permutation Importance
feature_names = ["LightGBM", "XGBoost", "CatBoost"]
importances = result.importances_mean
stds = result.importances_std
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 4))
plt.title("Permutation Importance (Meta-Model)")
plt.bar(range(len(importances)), importances[indices], yerr=stds[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
plt.ylabel("Mean decrease in ROC-AUC")
plt.tight_layout()
plt.show()

# Final Prediction at Testset
lgbm_test_probs = lgbm.predict_proba(X_test_sel)[:, 1]
xgbm_test_probs = xgbm.predict_proba(X_test_sel)[:, 1]
cat_test_probs = cat.predict_proba(X_test_sel)[:, 1]

stacked_test = np.vstack([lgbm_test_probs, xgbm_test_probs, cat_test_probs]).T
kaggle_preds = meta_model.predict_proba(stacked_test)[:, 1]

submission_template["rainfall"] = kaggle_preds
submission_template.to_csv("submission_stack_fs_cv_pi.csv", index=False)
print("Submission saved as submission_stack_fs_cv_pi.csv")


# The inputs of the meta-model (probabilities of the base models)
stacked_train = np.vstack([lgbm_probs, xgbm_probs, cat_probs]).T

# Permutation Importance at Holdout-Set
result = permutation_importance(
    meta_model,                # Meta-Modell (Logistic Regression)
    stacked_train,             # Input  (Base-Model-Probs)
    y_holdout,                 # target
    n_repeats=30,
    random_state=42,
    scoring="roc_auc"
)

importances = result.importances_mean
stds = result.importances_std
feature_names = ["LGBM", "XGBoost", "CatBoost"]

# Plot
plt.figure(figsize=(6, 4))
plt.barh(feature_names, importances, xerr=stds)
plt.xlabel("Mean decrease in ROC-AUC")
plt.title("Permutation Importance (Meta-Model)")
plt.tight_layout()
plt.show()



# Feature Engineering
train_final_data["temp_range"] = train_final_data["maxtemp"] - train_final_data["mintemp"]
test_final_data["temp_range"] = test_final_data["maxtemp"] - test_final_data["mintemp"]

train_final_data["humidity_ratio"] = train_final_data["humidity"] / (train_final_data["dewpoint"] + 1e-5)
test_final_data["humidity_ratio"] = test_final_data["humidity"] / (test_final_data["dewpoint"] + 1e-5)

train_final_data["humid_sun"] = train_final_data["humidity"] * train_final_data["sunshine"]
test_final_data["humid_sun"] = test_final_data["humidity"] * test_final_data["sunshine"]

train_final_data["day_sin"] = np.sin(2 * np.pi * train_final_data["day"] / 31)
train_final_data["day_cos"] = np.cos(2 * np.pi * train_final_data["day"] / 31)
test_final_data["day_sin"] = np.sin(2 * np.pi * test_final_data["day"] / 31)
test_final_data["day_cos"] = np.cos(2 * np.pi * test_final_data["day"] / 31)

# Prepare Data for Training
X = train_final_data.drop(columns=["rainfall"])
y = train_final_data["rainfall"]
X_test = test_final_data.copy()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"Fold {fold + 1}")
    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = cb.CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
        loss_function="Logloss"
    )
    model.fit(X_train_fold, y_train_fold)

    oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
    test_preds += model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits

auc = roc_auc_score(y, oof_preds)
print(f"CV ROC-AUC: {auc:.5f}")

# Feature Importances
model.fit(X_scaled, y)
importances = model.get_feature_importance()
feature_names = X.columns
sorted_idx = np.argsort(importances)

plt.figure(figsize=(8, 6))
plt.barh(feature_names[sorted_idx], importances[sorted_idx])
plt.title("CatBoost Feature Importances")
plt.tight_layout()
plt.show()

# Submission
submission_template["rainfall"] = test_preds
submission_template.to_csv("submission_catboost_cv.csv", index=False)
print("Submission saved as submission_catboost_cv.csv")



# feature engineering
train_final_data["temp_range"] = train_final_data["maxtemp"] - train_final_data["mintemp"]
test_final_data["temp_range"] = test_final_data["maxtemp"] - test_final_data["mintemp"]
train_final_data["humidity_ratio"] = train_final_data["humidity"] / (train_final_data["dewpoint"] + 1e-5)
test_final_data["humidity_ratio"] = test_final_data["humidity"] / (test_final_data["dewpoint"] + 1e-5)
train_final_data["humid_sun"] = train_final_data["humidity"] * train_final_data["sunshine"]
test_final_data["humid_sun"] = test_final_data["humidity"] * test_final_data["sunshine"]
train_final_data["day_sin"] = np.sin(2 * np.pi * train_final_data["day"] / 31)
test_final_data["day_sin"] = np.sin(2 * np.pi * test_final_data["day"] / 31)
train_final_data["day_cos"] = np.cos(2 * np.pi * train_final_data["day"] / 31)
test_final_data["day_cos"] = np.cos(2 * np.pi * test_final_data["day"] / 31)

X = train_final_data.drop(columns=["rainfall"])
y = train_final_data["rainfall"]
X_test = test_final_data.copy()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Prepare for OOF stacking
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_meta = np.zeros(len(X))
lgb_test_preds = []
xgb_test_preds = []
cat_test_preds = []

# Folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"Fold {fold + 1}")
    X_train, y_train = X_scaled[train_idx], y.iloc[train_idx]
    X_val, y_val = X_scaled[val_idx], y.iloc[val_idx]

    # Base-Models
    model_lgb = lgb.LGBMClassifier(n_estimators=300, random_state=42)
    model_xgb = xgb.XGBClassifier(n_estimators=300, random_state=42, tree_method="hist", device="cuda", use_label_encoder=False, eval_metric="auc")
    model_cat = cb.CatBoostClassifier(n_estimators=300, verbose=0, random_state=42)

    model_lgb.fit(X_train, y_train)
    model_xgb.fit(X_train, y_train)
    model_cat.fit(X_train, y_train)

    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]

    lgb_test_preds.append(model_lgb.predict_proba(X_test_scaled)[:, 1])
    xgb_test_preds.append(model_xgb.predict_proba(X_test_scaled)[:, 1])
    cat_test_preds.append(model_cat.predict_proba(X_test_scaled)[:, 1])

# Meta-Model
stacked_features = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
meta_model = XGBClassifier(n_estimators=200, random_state=42, tree_method="hist", device="cuda")
meta_model.fit(stacked_features, y)
oof_meta_preds = meta_model.predict_proba(stacked_features)[:, 1]

print("CV ROC-AUC:", round(roc_auc_score(y, oof_meta_preds), 5))

# Final Test Prediction
final_lgb = np.mean(lgb_test_preds, axis=0)
final_xgb = np.mean(xgb_test_preds, axis=0)
final_cat = np.mean(cat_test_preds, axis=0)
stacked_test = np.vstack([final_lgb, final_xgb, final_cat]).T
final_preds = meta_model.predict_proba(stacked_test)[:, 1]

# Save Submission
submission_template["rainfall"] = final_preds
submission_template.to_csv("submission_stack_oof.csv", index=False)
print("Submission saved as submission_stack_oof.csv")


# New features
train_final_data["temp_diff"] = train_final_data["maxtemp"] - train_final_data["mintemp"]
test_final_data["temp_diff"] = test_final_data["maxtemp"] - test_final_data["mintemp"]
train_final_data["humidity_dew"] = train_final_data["humidity"] / (train_final_data["dewpoint"] + 1e-5)
test_final_data["humidity_dew"] = test_final_data["humidity"] / (test_final_data["dewpoint"] + 1e-5)

X = train_final_data.drop(columns=["rainfall"])
y = train_final_data["rainfall"]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

result = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=42, scoring="roc_auc")
importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
selected = importances.head(10).index.tolist()
print("Top 10 Features:", selected)


# keep the selected features
X = train_final_data[selected]
X_test = test_final_data[selected]
y = train_final_data["rainfall"]


X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
X_test_scaled = scaler.transform(X_test)

# train base models
lgbm = lgb.LGBMClassifier(n_estimators=300, random_state=42)
xgbm = xgb.XGBClassifier(n_estimators=300, random_state=42, tree_method="hist", device="cuda", use_label_encoder=False)
cat = cb.CatBoostClassifier(n_estimators=300, verbose=0, random_state=42)

lgbm.fit(X_train_scaled, y_train)
xgbm.fit(X_train_scaled, y_train)
cat.fit(X_train_scaled, y_train)

# prob for holdout
lgbm_probs = lgbm.predict_proba(X_holdout_scaled)[:, 1]
xgbm_probs = xgbm.predict_proba(X_holdout_scaled)[:, 1]
cat_probs = cat.predict_proba(X_holdout_scaled)[:, 1]

# Meta-Modell (LogReg or LightGBM)
stacked_train = np.vstack([lgbm_probs, xgbm_probs, cat_probs]).T
meta_model = LogisticRegression()
meta_model.fit(stacked_train, y_holdout)

# Prediction at Holdout
stacked_pred = meta_model.predict_proba(stacked_train)[:, 1]
print("Meta-Model ROC-AUC on Holdout:", roc_auc_score(y_holdout, stacked_pred))


lgbm_test = lgbm.predict_proba(X_test_scaled)[:, 1]
xgbm_test = xgbm.predict_proba(X_test_scaled)[:, 1]
cat_test = cat.predict_proba(X_test_scaled)[:, 1]

stacked_test = np.vstack([lgbm_test, xgbm_test, cat_test]).T
submission_template["rainfall"] = meta_model.predict_proba(stacked_test)[:, 1]
submission_template.to_csv("submission_perm_stack.csv", index=False)



# Proabilities for lgbm, xgbm, cat
stacked_train = np.vstack([lgbm_probs, xgbm_probs, cat_probs]).T

# target values
y_meta = y_holdout

# Optuna Objective
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method": "hist",
        "device": "cuda"
    }

    dtrain = xgb.DMatrix(stacked_train, label=y_meta)
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=300,
        nfold=5,
        stratified=True,
        early_stopping_rounds=30,
        metrics="auc",
        seed=42,
        verbose_eval=False,
        as_pandas=True
    )

    return cv_results["test-auc-mean"].max()

# start study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

print("Best ROC-AUC:", study.best_value)
print("Best params:", study.best_trial.params)

# final model study
best_params = study.best_trial.params
dtrain_final = xgb.DMatrix(stacked_train, label=y_meta)
meta_model_xgb = xgb.train(best_params, dtrain_final, num_boost_round=300)


stacked_test = np.vstack([lgbm_test_probs, xgbm_test_probs, cat_test_probs]).T
dtest = xgb.DMatrix(stacked_test)
kaggle_preds = meta_model_xgb.predict(dtest)

# Submission
submission_template["rainfall"] = kaggle_preds
submission_template.to_csv("submission_stack_xgb_optuna_cv.csv", index=False)
print("Submission saved as submission_stack_xgb_optuna_cv.csv")



def optuna_lgbm_objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        lgb_train = lgb.Dataset(X_t, label=y_t)
        lgb_valid = lgb.Dataset(X_v, label=y_v)

        model = lgb.train(params, lgb_train, valid_sets=[lgb_valid], verbose_eval=False, early_stopping_rounds=50)
        preds = model.predict(X_v)
        aucs.append(roc_auc_score(y_v, preds))

    return np.mean(aucs)



X_full = train_final_data.drop(columns=["rainfall"])
y_full = train_final_data["rainfall"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
X_test_scaled = scaler.transform(test_final_data)


# Example with RidgeClassifier as Meta-Model:
meta_model = RidgeClassifier()
meta_model.fit(stacked_train, y_holdout)
pred = meta_model.decision_function(stacked_train)
print("Meta ROC-AUC Ridge:", roc_auc_score(y_holdout, pred))

# Example for CatBoost as Meta-Model:
meta_cat = CatBoostClassifier(verbose=0)
meta_cat.fit(stacked_train, y_holdout)
pred = meta_cat.predict_proba(stacked_train)[:, 1]
print("Meta ROC-AUC CatBoost:", roc_auc_score(y_holdout, pred))

# Example for LightGBM as Meta-Model:
meta_lgb = LGBMClassifier()
meta_lgb.fit(stacked_train, y_holdout)
pred = meta_lgb.predict_proba(stacked_train)[:, 1]
print("Meta ROC-AUC LightGBM:", roc_auc_score(y_holdout, pred))

# Use the best meta-model (e.g., CatBoostClassifier in this case)
best_meta_model = meta_cat  # Replace with the best performing model
kaggle_preds = best_meta_model.predict_proba(stacked_test)[:, 1]

# Create the submission file
submission_template["rainfall"] = kaggle_preds
submission_template.to_csv("submission_stack_best_meta.csv", index=False)
print("Submission saved as submission_stack_best_meta.csv")


# Feature Engineering
train_final_data["temp_diff"] = train_final_data["maxtemp"] - train_final_data["mintemp"]
test_final_data["temp_diff"] = test_final_data["maxtemp"] - test_final_data["mintemp"]

train_final_data["humidity_dew"] = train_final_data["humidity"] / (train_final_data["dewpoint"] + 1e-5)
test_final_data["humidity_dew"] = test_final_data["humidity"] / (test_final_data["dewpoint"] + 1e-5)

train_final_data["humid_sun"] = train_final_data["humidity"] * train_final_data["sunshine"]
test_final_data["humid_sun"] = test_final_data["humidity"] * test_final_data["sunshine"]

train_final_data["day_sin"] = np.sin(2 * np.pi * train_final_data["day"] / 31)
train_final_data["day_cos"] = np.cos(2 * np.pi * train_final_data["day"] / 31)
test_final_data["day_sin"] = np.sin(2 * np.pi * test_final_data["day"] / 31)
test_final_data["day_cos"] = np.cos(2 * np.pi * test_final_data["day"] / 31)

# Split + Scaling
X = train_final_data.drop(columns=["rainfall"])
y = train_final_data["rainfall"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
X_test_scaled = scaler.transform(test_final_data)

# Models
lgbm = lgb.LGBMClassifier(n_estimators=300, random_state=42)
xgbm = xgb.XGBClassifier(n_estimators=300, random_state=42, tree_method="hist", device="cuda", use_label_encoder=False)
cat = cb.CatBoostClassifier(n_estimators=300, verbose=0, random_state=42)

# Training
lgbm.fit(X_train_scaled, y_train)
xgbm.fit(X_train_scaled, y_train)
cat.fit(X_train_scaled, y_train)

# Probabilities for Holdout Set
lgbm_probs = lgbm.predict_proba(X_holdout_scaled)[:, 1]
xgbm_probs = xgbm.predict_proba(X_holdout_scaled)[:, 1]
cat_probs = cat.predict_proba(X_holdout_scaled)[:, 1]

# Blending
blend = 0.3 * lgbm_probs + 0.3 * xgbm_probs + 0.4 * cat_probs
auc = roc_auc_score(y_holdout, blend)
print(f"Blended AUC (Holdout): {auc:.5f}")

# test set probabilities
lgbm_test_probs = lgbm.predict_proba(X_test_scaled)[:, 1]
xgbm_test_probs = xgbm.predict_proba(X_test_scaled)[:, 1]
cat_test_probs = cat.predict_proba(X_test_scaled)[:, 1]

blend_test = 0.3 * lgbm_test_probs + 0.3 * xgbm_test_probs + 0.4 * cat_test_probs

# Submission
submission_template["rainfall"] = blend_test
submission_template.to_csv("submission_blend.csv", index=False)
print("Submission saved as submission_blend.csv")


def feature_engineering(df):
    df["temp_diff"] = df["maxtemp"] - df["mintemp"]
    df["humidity_dew"] = df["humidity"] / (df["dewpoint"] + 1e-5)
    df["humid_sun"] = df["humidity"] * df["sunshine"]
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["wind_dir_rad"] = np.deg2rad(df["winddirection"])
    df["wind_x"] = df["windspeed"] * np.cos(df["wind_dir_rad"])
    df["wind_y"] = df["windspeed"] * np.sin(df["wind_dir_rad"])
    return df

train_final_data = feature_engineering(train_final_data)
test_final_data = feature_engineering(test_final_data)

X_full = train_final_data.drop(columns=["rainfall"])
y_full = train_final_data["rainfall"]
X_test_full = test_final_data.copy()

scaler = RobustScaler()
X_full_scaled = scaler.fit_transform(X_full)
X_test_scaled = scaler.transform(X_test_full)

def objective_lgbm(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100)
    }

    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, valid_idx in skf.split(X_full_scaled, y_full):
        X_train_fold = X_full_scaled[train_idx]
        y_train_fold = y_full.iloc[train_idx]
        X_valid_fold = X_full_scaled[valid_idx]
        y_valid_fold = y_full.iloc[valid_idx]

        model = lgb.LGBMClassifier(**params, n_estimators=300, random_state=42)
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict_proba(X_valid_fold)[:, 1]
        fold_auc = roc_auc_score(y_valid_fold, preds)
        scores.append(fold_auc)

    return np.mean(scores)

study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=200)

print("Best trial:", study_lgbm.best_trial.value)
print("Best params:", study_lgbm.best_params)


best_params = study_lgbm.best_params

# Final LightGBM instance with the found parameters
final_lgb = lgb.LGBMClassifier(**best_params, n_estimators=300, random_state=42)
final_lgb.fit(X_full_scaled, y_full)

# Test-Prediction
test_preds = final_lgb.predict_proba(X_test_scaled)[:, 1]
submission_template["rainfall"] = test_preds
submission_template.to_csv("submission_lgb_optuna.csv", index=False)
print("Saved submission_lgb_optuna.csv")



def feature_engineering(df):
    df["temp_diff"] = df["maxtemp"] - df["mintemp"]
    df["humidity_dew"] = df["humidity"] / (df["dewpoint"] + 1e-5)
    df["humid_sun"] = df["humidity"] * df["sunshine"]
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["wind_dir_rad"] = np.deg2rad(df["winddirection"])
    df["wind_x"] = df["windspeed"] * np.cos(df["wind_dir_rad"])
    df["wind_y"] = df["windspeed"] * np.sin(df["wind_dir_rad"])
    return df

train_final_data = feature_engineering(train_final_data)
test_final_data = feature_engineering(test_final_data)

X_all = train_final_data.drop(columns=["rainfall"])
y_all = train_final_data["rainfall"]
X_test = test_final_data.copy()

scaler = RobustScaler()
X_all_scaled = scaler.fit_transform(X_all)
X_test_scaled = scaler.transform(X_test)


N_TRIALS = 200

# --- LightGBM Tuning ---
def objective_lgbm(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100)
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in skf.split(X_all_scaled, y_all):
        X_t, y_t = X_all_scaled[train_idx], y_all.iloc[train_idx]
        X_v, y_v = X_all_scaled[valid_idx], y_all.iloc[valid_idx]
        model = lgb.LGBMClassifier(**params, n_estimators=300, random_state=42)
        model.fit(X_t, y_t)
        preds = model.predict_proba(X_v)[:, 1]
        scores.append(roc_auc_score(y_v, preds))
    return np.mean(scores)

study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS)
best_lgbm_params = study_lgbm.best_params
print("[Optuna] Best LGBM:", best_lgbm_params)

# --- XGBoost Tuning ---
def objective_xgb(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "use_label_encoder": False,
        "tree_method": "hist",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in skf.split(X_all_scaled, y_all):
        X_t, y_t = X_all_scaled[train_idx], y_all.iloc[train_idx]
        X_v, y_v = X_all_scaled[valid_idx], y_all.iloc[valid_idx]
        model = xgb.XGBClassifier(**params, n_estimators=300, random_state=42)
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        preds = model.predict_proba(X_v)[:, 1]
        scores.append(roc_auc_score(y_v, preds))
    return np.mean(scores)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)
best_xgb_params = study_xgb.best_params
print("[Optuna] Best XGB:", best_xgb_params)

# --- CatBoost Tuning ---
def objective_cat(trial):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in skf.split(X_all_scaled, y_all):
        X_t, y_t = X_all_scaled[train_idx], y_all.iloc[train_idx]
        X_v, y_v = X_all_scaled[valid_idx], y_all.iloc[valid_idx]

        model = cb.CatBoostClassifier(**params, n_estimators=300, verbose=0)
        model.fit(X_t, y_t, eval_set=(X_v, y_v))
        preds = model.predict_proba(X_v)[:, 1]
        scores.append(roc_auc_score(y_v, preds))
    return np.mean(scores)

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=N_TRIALS)
best_cat_params = study_cat.best_params
print("[Optuna] Best Cat:", best_cat_params)


N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_all))
oof_xgb = np.zeros(len(X_all))
oof_cat = np.zeros(len(X_all))

models_lgb = []
models_xgb = []
models_cat = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all_scaled, y_all)):
    print(f"Fold {fold+1}")
    X_t, X_v = X_all_scaled[train_idx], X_all_scaled[val_idx]
    y_t, y_v = y_all.iloc[train_idx], y_all.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**best_lgbm_params, n_estimators=300, random_state=42)
    lgb_model.fit(X_t, y_t)
    oof_lgb[val_idx] = lgb_model.predict_proba(X_v)[:, 1]
    models_lgb.append(lgb_model)

    # XGBoost
    xgb_model = xgb.XGBClassifier(**best_xgb_params, n_estimators=300, random_state=42, use_label_encoder=False, eval_metric='auc')
    xgb_model.fit(X_t, y_t)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_v)[:, 1]
    models_xgb.append(xgb_model)

    # CatBoost
    cat_model = cb.CatBoostClassifier(**best_cat_params, n_estimators=300, random_state=42, verbose=0)
    cat_model.fit(X_t, y_t)
    oof_cat[val_idx] = cat_model.predict_proba(X_v)[:, 1]
    models_cat.append(cat_model)

print("OOF-LGB AUC:", roc_auc_score(y_all, oof_lgb))
print("OOF-XGB AUC:", roc_auc_score(y_all, oof_xgb))
print("OOF-CAT AUC:", roc_auc_score(y_all, oof_cat))

# Stacking

stacked_train = np.vstack([oof_lgb, oof_xgb, oof_cat]).T

meta_model = cb.CatBoostClassifier(random_state=42, verbose=0)
meta_model.fit(stacked_train, y_all)
meta_preds_oof = meta_model.predict_proba(stacked_train)[:, 1]
print("Meta OOF AUC:", roc_auc_score(y_all, meta_preds_oof))

########################################
# 4) Final Models on Full-Data + Test-Pred
########################################

# Base-Models retrain on FULL data
final_lgb = lgb.LGBMClassifier(**best_lgbm_params, n_estimators=300, random_state=42)
final_lgb.fit(X_all_scaled, y_all)
test_pred_lgb = final_lgb.predict_proba(X_test_scaled)[:, 1]

final_xgb = xgb.XGBClassifier(**best_xgb_params, n_estimators=300, random_state=42, use_label_encoder=False, eval_metric='auc')
final_xgb.fit(X_all_scaled, y_all)
test_pred_xgb = final_xgb.predict_proba(X_test_scaled)[:, 1]

final_cat = cb.CatBoostClassifier(**best_cat_params, n_estimators=300, random_state=42, verbose=0)
final_cat.fit(X_all_scaled, y_all)
test_pred_cat = final_cat.predict_proba(X_test_scaled)[:, 1]

# Stacking at Test
stacked_test = np.vstack([test_pred_lgb, test_pred_xgb, test_pred_cat]).T
final_preds = meta_model.predict_proba(stacked_test)[:, 1]

submission_template["rainfall"] = final_preds
submission_template.to_csv("submission_endboss.csv", index=False)
print("Submission saved as submission_endboss.csv")

