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


import pandas as pd

from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


import numpy as np
TRAIN_CSV_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TARGET_COL = "diagnosed_diabetes"
RANDOM_STATE = 42
N_FOLDS = 5

df = pd.read_csv(TRAIN_CSV_PATH)


df_fe = df.copy()

df_fe['bmi_age'] = df_fe['bmi'] * df_fe['age']
df_fe['waist_to_hip_age'] = df_fe['waist_to_hip_ratio'] / df_fe['age']
df_fe['bp_ratio'] = df_fe['systolic_bp'] / df_fe['diastolic_bp']

df_fe['family_bmi'] = df_fe['family_history_diabetes'] * df_fe['bmi']
df_fe['age_physical_activity'] = df_fe['age'] * df_fe['physical_activity_minutes_per_week']

df_fe['age_bin'] = pd.cut(df_fe['age'], bins=[0,30,45,60,120], labels=['<30','30-45','45-60','60+'])
df_fe['bmi_cat'] = pd.cut(df_fe['bmi'], bins=[0,18.5,24.9,29.9,100], labels=['Underweight','Normal','Overweight','Obese'])

df_fe['cholesterol_risk_score'] = df_fe['cholesterol_total'] + df_fe['ldl_cholesterol'] - df_fe['hdl_cholesterol']
df_fe['cardio_risk_score'] = df_fe['hypertension_history'] + df_fe['cardiovascular_history']

df_fe['income_encoded'] = OrdinalEncoder(categories=[['Low','Lower-Middle','Middle','Upper-Middle','High']]).fit_transform(df_fe[['income_level']])
df_fe['age_bin_encoded'] = OrdinalEncoder(categories=[['<30','30-45','45-60','60+']]).fit_transform(df_fe[['age_bin']])
df_fe['bmi_cat_encoded'] = OrdinalEncoder(categories=[['Underweight','Normal','Overweight','Obese']]).fit_transform(df_fe[['bmi_cat']])

df_fe = pd.get_dummies(df_fe, columns=['gender','smoking_status','education_level','employment_status'], drop_first=False, dtype=int)

eth_mean = df_fe.groupby('ethnicity')[TARGET_COL].mean()
df_fe['ethnicity_encoded'] = df_fe['ethnicity'].map(eth_mean)
df_fe.drop(columns=['ethnicity','age','bmi','waist_to_hip_ratio','systolic_bp','diastolic_bp',
                    'cholesterol_total','ldl_cholesterol','hdl_cholesterol',
                    'hypertension_history','cardiovascular_history',
                    'income_level','age_bin','bmi_cat'], inplace=True)

X = df_fe.drop(columns=[TARGET_COL]).astype(float)
y = df_fe[TARGET_COL].values


# =====================================================
# fast_gpu_stacked_model.py
# =====================================================

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------- CONFIG ----------------
N_FOLDS = 5
RANDOM_STATE = 42
MAX_ITER_CAT = 2000
MAX_ITER_LGB = 2000

# ---------------- DATA ----------------
# Assuming X, y are already prepared
# X = df_fe.drop(columns=[TARGET_COL]).astype(float)
# y = df_fe[TARGET_COL].values

# ---------------- STACKING CV ----------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_cat = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
best_iter_cat = 0
best_iter_lgb = 0

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{N_FOLDS}")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # ----- CATBOOST GPU -----
    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    cat = CatBoostClassifier(
        iterations=MAX_ITER_CAT,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=8,
        bagging_temperature=0.8,
        eval_metric='AUC',
        task_type='GPU',          # ✅ GPU
        devices='0:1',            # GPU devices if multiple available
        random_seed=RANDOM_STATE,
        verbose=0,
        class_weights=[1, scale_pos_weight],
        early_stopping_rounds=30
    )
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]
    best_iter_cat = max(best_iter_cat, cat.best_iteration_)

    # ----- LIGHTGBM GPU -----
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'num_leaves': 48,
        'min_data_in_leaf': 40,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbosity': -1,
        'seed': RANDOM_STATE,
        'device': 'gpu',   # ✅ GPU
    }

    lgb_tr = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_ds = lgb.Dataset(X_val, label=y_val, reference=lgb_tr)

    lgb_model = lgb.train(
        lgb_params,
        lgb_tr,
        num_boost_round=MAX_ITER_LGB,
        valid_sets=[lgb_val_ds],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=0)  # disables printing
        ]
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    best_iter_lgb = max(best_iter_lgb, lgb_model.best_iteration)

# ----- META MODEL -----
meta_X = np.column_stack([oof_cat, oof_lgb])
meta_model = LogisticRegression()
meta_model.fit(meta_X, y)
val_auc = roc_auc_score(y, meta_model.predict_proba(meta_X)[:, 1])
print(f"✅ Stacking done. OOF AUC: {val_auc:.5f}")
print(f"Best iterations → CatBoost: {best_iter_cat}, LightGBM: {best_iter_lgb}")

# ----- FINAL FULL MODEL -----
# CatBoost full GPU
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
cat_full = CatBoostClassifier(
    iterations=best_iter_cat,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=8,
    bagging_temperature=0.8,
    eval_metric='AUC',
    task_type='GPU',
    devices='0:1',
    random_seed=RANDOM_STATE,
    verbose=200,
    class_weights=[1, scale_pos_weight]
)
cat_full.fit(X, y)

# LightGBM full GPU
lgb_full = lgb.Dataset(X, label=y)
lgb_full_model = lgb.train(
    lgb_params,
    lgb_full,
    num_boost_round=best_iter_lgb
)

# ----- FINAL PREDICTION FUNCTION -----
# def blended_predict_proba(X_input):
#     return (cat_full.predict_proba(X_input)[:, 1] * 0.5 +  # you can adjust weights later
#             lgb_full_model.predict(X_input) * 0.5)



def stacked_predict_proba(X_input):
    cat_pred = cat_full.predict_proba(X_input)[:,1]
    lgb_pred = lgb_full_model.predict(X_input)
    meta_features = np.column_stack([cat_pred, lgb_pred])
    return meta_model.predict_proba(meta_features)[:,1]

print("✅ Full stacked model ready.")


# ---------------- SAVE FINAL MODELS ----------------
cat_full.save_model("catboost_full.cbm")
lgb_full_model.save_model("lightgbm_full.txt")  # save LightGBM booster

# Save meta-model and features
import pickle

with open("stack_meta.pkl", "wb") as f:
    pickle.dump(meta_model, f)

with open("features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Models and meta saved successfully.")



# # =====================================================
# # predict_test_and_submit_stacking.py
# # =====================================================

# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# from sklearn.preprocessing import OrdinalEncoder
# import pickle

# # ---------------- PATHS ----------------
# TEST_CSV_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
# CAT_MODEL_PATH = "/content/catboost_full.cbm"
# LGB_MODEL_PATH = "/content/lightgbm_full.txt"
# META_MODEL_PATH = "/content/stack_meta.pkl"  # meta-model (e.g., LogisticRegression)
# FEATURES_PATH = "/content/features.pkl"
# SUBMISSION_PATH = "/content/drive/MyDrive/KaggleSubmissions/submission10.csv"

# # ---------------- LOAD META & FEATURES ----------------
# with open(META_MODEL_PATH, "rb") as f:
#     meta_model = pickle.load(f)

# with open(FEATURES_PATH, "rb") as f:
#     FEATURES = pickle.load(f)

# # ---------------- LOAD BASE MODELS ----------------
# cat_model = CatBoostClassifier()
# cat_model.load_model(CAT_MODEL_PATH)

# lgb_model = lgb.Booster(model_file=LGB_MODEL_PATH)

# # ---------------- READ TEST ----------------
# df = pd.read_csv(TEST_CSV_PATH)

# # ---------------- FEATURE ENGINEERING ----------------
# df_fe = df.copy()
# df_fe['bmi_age'] = df_fe['bmi'] * df_fe['age']
# df_fe['waist_to_hip_age'] = df_fe['waist_to_hip_ratio'] / df_fe['age']
# df_fe['bp_ratio'] = df_fe['systolic_bp'] / df_fe['diastolic_bp']
# df_fe['family_bmi'] = df_fe['family_history_diabetes'] * df_fe['bmi']
# df_fe['age_physical_activity'] = df_fe['age'] * df_fe['physical_activity_minutes_per_week']

# df_fe['age_bin'] = pd.cut(
#     df_fe['age'],
#     bins=[0, 30, 45, 60, 120],
#     labels=['<30', '30-45', '45-60', '60+']
# )

# df_fe['bmi_cat'] = pd.cut(
#     df_fe['bmi'],
#     bins=[0, 18.5, 24.9, 29.9, 100],
#     labels=['Underweight', 'Normal', 'Overweight', 'Obese']
# )

# df_fe['cholesterol_risk_score'] = df_fe['cholesterol_total'] + df_fe['ldl_cholesterol'] - df_fe['hdl_cholesterol']
# df_fe['cardio_risk_score'] = df_fe['hypertension_history'] + df_fe['cardiovascular_history']

# df_fe['income_encoded'] = OrdinalEncoder(
#     categories=[['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']]
# ).fit_transform(df_fe[['income_level']])

# df_fe['age_bin_encoded'] = OrdinalEncoder(
#     categories=[['<30', '30-45', '45-60', '60+']]
# ).fit_transform(df_fe[['age_bin']])

# df_fe['bmi_cat_encoded'] = OrdinalEncoder(
#     categories=[['Underweight', 'Normal', 'Overweight', 'Obese']]
# ).fit_transform(df_fe[['bmi_cat']])

# df_fe = pd.get_dummies(
#     df_fe,
#     columns=['gender', 'smoking_status', 'education_level', 'employment_status'],
#     drop_first=False,
#     dtype=int
# )

# # safe fallback for ethnicity
# df_fe['ethnicity_encoded'] = 0.0

# # ---------------- DROP RAW ----------------
# df_fe.drop(
#     columns=[
#         'ethnicity', 'age', 'bmi', 'waist_to_hip_ratio',
#         'systolic_bp', 'diastolic_bp',
#         'cholesterol_total', 'ldl_cholesterol', 'hdl_cholesterol',
#         'hypertension_history', 'cardiovascular_history',
#         'income_level', 'age_bin', 'bmi_cat'
#     ],
#     inplace=True,
#     errors='ignore'
# )

# # ---------------- ALIGN FEATURES ----------------
# for col in FEATURES:
#     if col not in df_fe:
#         df_fe[col] = 0

# df_fe = df_fe[FEATURES].astype(float)

# # ---------------- STACKED PREDICTIONS ----------------
# cat_pred = cat_model.predict_proba(df_fe)[:, 1]
# lgb_pred = lgb_model.predict(df_fe)

# meta_features = np.column_stack([cat_pred, lgb_pred])
# final_pred = meta_model.predict_proba(meta_features)[:, 1]

# # ---------------- SUBMISSION ----------------
# submission = pd.DataFrame({
#     "id": df["id"],  # adjust if Kaggle uses a different column
#     "diagnosed_diabetes": final_pred
# })

# submission.to_csv(SUBMISSION_PATH, index=False)
# print("✅ Submission file created:", SUBMISSION_PATH)


