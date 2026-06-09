import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import optuna


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
numeric_df = df.select_dtypes(include=["number"])
numeric_df["ldl_hdl_ratio"] = numeric_df["ldl_cholesterol"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["cholesterol_ratio"] = numeric_df["cholesterol_total"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["activity_bmi"] = numeric_df["physical_activity_minutes_per_week"] / (numeric_df["bmi"] + 1e-6)
numeric_df["age_bmi"] = numeric_df["age"] * numeric_df["bmi"]
numeric_df["age_activity"] = numeric_df["age"] * numeric_df["physical_activity_minutes_per_week"]
numeric_df["age_triglycerides"] = numeric_df["age"] * numeric_df["triglycerides"]
numeric_df["high_bmi"] = (numeric_df["bmi"] > 30).astype(int)
numeric_df["high_triglycerides"] = (numeric_df["triglycerides"] > 150).astype(int)
numeric_df["insulin_resistance_proxy"] = (numeric_df["bmi"] * numeric_df["triglycerides"] / (numeric_df["physical_activity_minutes_per_week"] + 1))
numeric_df["metabolic_risk_score"] = (
    0.4 * numeric_df["bmi"] +
    0.3 * numeric_df["triglycerides"] +
    0.2 * numeric_df["age"] -
    0.3 * numeric_df["physical_activity_minutes_per_week"]
)
numeric_df.head()


features = [
    "family_history_diabetes",
    "physical_activity_minutes_per_week",
    "activity_bmi",
    "age_bmi",
    "age_triglycerides",
    "age_activity",
    "age",
    "ldl_hdl_ratio",
    "triglycerides",
    "cholesterol_ratio",
    "bmi",
    "insulin_resistance_proxy",
    "metabolic_risk_score"
]
X = numeric_df[features]
y = numeric_df["diagnosed_diabetes"]
features2 = [
    "family_history_diabetes",
    "physical_activity_minutes_per_week",
    "activity_bmi",
    "age_bmi",
    "age_triglycerides",
    "age_activity",
    "age",
    "ldl_hdl_ratio",
    "triglycerides",
    "cholesterol_ratio",
    "bmi"
]
X2 = numeric_df[features2]
y2 = numeric_df["diagnosed_diabetes"]


proxy_lgbm_model1 = lgb.LGBMClassifier(**{'learning_rate': 0.0643034598947598,
 'num_leaves': 76,
 'max_depth': 13,
 'min_data_in_leaf': 171,
 'subsample': 0.8410325148146744,
 'colsample_bytree': 0.5597106246510247,
 'n_estimators': 436,
 'reg_alpha': 8.364491539122616,
 'reg_lambda': 7.535851957156131,
 'min_gain_to_split': 0.4657421507730968,
 'verbosity':-1,
 'random_state':777})
proxy_lgbm_model2 = lgb.LGBMClassifier(**{'learning_rate': 0.0643034598947598,
 'num_leaves': 76,
 'max_depth': 13,
 'min_data_in_leaf': 171,
 'subsample': 0.8410325148146744,
 'colsample_bytree': 0.5597106246510247,
 'n_estimators': 436,
 'reg_alpha': 8.364491539122616,
 'reg_lambda': 7.535851957156131,
 'min_gain_to_split': 0.4657421507730968,
 'verbosity':-1,
 'random_state':2025})


proxy_lgbm_model1.fit(X, y) 
proxy_lgbm_model2.fit(X, y) 


lgbm_model1 = lgb.LGBMClassifier(**{'learning_rate': 0.07183521969292042,
 'num_leaves': 69,
 'max_depth': 5,
 'min_data_in_leaf': 155,
 'subsample': 0.6038402866038635,
 'colsample_bytree': 0.7556756642569674,
 'n_estimators': 762,
 'reg_alpha': 1.7469018618683667,
 'reg_lambda': 5.229358219612677,
 'min_gain_to_split': 0.11624493766312131,
 'verbosity':-1,
 'random_state':42})
lgbm_model2 = lgb.LGBMClassifier(**{'learning_rate': 0.07183521969292042,
 'num_leaves': 69,
 'max_depth': 5,
 'min_data_in_leaf': 155,
 'subsample': 0.6038402866038635,
 'colsample_bytree': 0.7556756642569674,
 'n_estimators': 762,
 'reg_alpha': 1.7469018618683667,
 'reg_lambda': 5.229358219612677,
 'min_gain_to_split': 0.11624493766312131,
 'verbosity':-1,
 'random_state':1337})


lgbm_model1.fit(X2, y2) 
lgbm_model2.fit(X2, y2) 


hbc = HistGradientBoostingClassifier(**{'learning_rate': 0.04129003713472085,
 'max_depth': 7,
 'max_iter': 863,
 'min_samples_leaf': 49,
 'l2_regularization': 0.159201740522381,
 'max_bins': 237,
 'random_state':777})
hbc.fit(X, y) 


test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
numeric_df = test.select_dtypes(include=["number"])
numeric_df["ldl_hdl_ratio"] = numeric_df["ldl_cholesterol"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["cholesterol_ratio"] = numeric_df["cholesterol_total"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["activity_bmi"] = numeric_df["physical_activity_minutes_per_week"] / (numeric_df["bmi"] + 1e-6)
numeric_df["age_bmi"] = numeric_df["age"] * numeric_df["bmi"]
numeric_df["age_activity"] = numeric_df["age"] * numeric_df["physical_activity_minutes_per_week"]
numeric_df["age_triglycerides"] = numeric_df["age"] * numeric_df["triglycerides"]
numeric_df["high_bmi"] = (numeric_df["bmi"] > 30).astype(int)
numeric_df["high_triglycerides"] = (numeric_df["triglycerides"] > 150).astype(int)
numeric_df["insulin_resistance_proxy"] = (numeric_df["bmi"] * numeric_df["triglycerides"] / (numeric_df["physical_activity_minutes_per_week"] + 1))
numeric_df["metabolic_risk_score"] = (
    0.4 * numeric_df["bmi"] +
    0.3 * numeric_df["triglycerides"] +
    0.2 * numeric_df["age"] -
    0.3 * numeric_df["physical_activity_minutes_per_week"]
)
full = numeric_df


X1 = full[features]
X2 = full[features2]
y_score1 = proxy_lgbm_model1.predict_proba(X1)[:,1]
y_score2 = proxy_lgbm_model2.predict_proba(X1)[:,1]
y_score3 = lgbm_model1.predict_proba(X2)[:,1]
y_score4 = lgbm_model2.predict_proba(X2)[:,1]
y_score5 = hbc.predict_proba(X1)[:,1]
ensemble_pred = (
    0.30 * y_score1 +
    0.25 * y_score2 +
    0.20 * y_score3 +
    0.15 * y_score4 +
    0.10 * y_score5
)
test['diagnosed_diabetes'] = ensemble_pred
submit = test[['id','diagnosed_diabetes']]
submit = submit.set_index('id')
submit.to_csv('submission.csv')
submit

