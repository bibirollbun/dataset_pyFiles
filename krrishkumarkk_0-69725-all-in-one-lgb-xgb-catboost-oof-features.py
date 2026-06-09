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
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


df=df.drop("id",axis=1)
# df=df.drop("smoking_status",axis=1)


df.describe


X = df.drop("diagnosed_diabetes", axis=1)
y = df["diagnosed_diabetes"]

X = X.copy()



#have commented some features to resuce complexity
def feature_engineering(X):
    X = X.copy()

    # X["age_over_45"] = (X["age"] >= 45).astype(int)
    X["bmi_obese"] = (X["bmi"] >= 30).astype(int)
    X["bmi_overweight"] = ((X["bmi"] >= 25) & (X["bmi"] < 30)).astype(int)
    X["central_obesity"] = (X["waist_to_hip_ratio"] > 0.9).astype(int)

    X["hypertension_risk"] = (
        (X["systolic_bp"] >= 140) | (X["diastolic_bp"] >= 90)
    ).astype(int)

    X["chol_hdl_ratio"] = X["cholesterol_total"] / X["hdl_cholesterol"]
    X["ldl_hdl_ratio"] = X["ldl_cholesterol"] / X["hdl_cholesterol"]
    X["triglyceride_hdl_ratio"] = X["triglycerides"] / X["hdl_cholesterol"]

    X["sedentary_lifestyle"] = (X["physical_activity_minutes_per_week"] < 150).astype(int)
    X["poor_sleep"] = (X["sleep_hours_per_day"] < 6).astype(int)
    X["high_screen_time"] = (X["screen_time_hours_per_day"] > 6).astype(int)

    # X["diet_activity_interaction"] = X["diet_score"] * X["physical_activity_minutes_per_week"]
    # X["genetic_lifestyle_risk"] = (
    #     X["family_history_diabetes"] * X["bmi"] * X["sedentary_lifestyle"]
    # )

    return X



X = feature_engineering(X)


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


WORKING WITH CAT COLUMNS


num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols=X.select_dtypes(include=["object"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ohe = OneHotEncoder(drop="first", sparse_output=False)


transformer = ColumnTransformer(
    transformers=[
        ("cat", ohe, cat_cols)
        # ("edu", oe1, ["education_level"]),
        # ("inc", oe2, ["income_level"]),
        # ("sc",sc,num_cols),
        # ("pca",pca,num_cols)
    ],
    remainder="passthrough"
)



#trail of ordinal encodeer, standard acaler and pca

# oe1 = OrdinalEncoder(
#     categories=[["Postgraduate", "Graduate", "Highschool", "No formal"]]
# )

# oe2 = OrdinalEncoder(
#     categories=[["High", "Upper-Middle", "Middle", "Lower-Middle", "Low"]]
# )
# sc=StandardScaler()
# pca=PCA(n_components=None)



X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#outlier manangement

# from sklearn.ensemble import IsolationForest
# import numpy as np

# iso = IsolationForest(
#     n_estimators=300,
#     contamination=0.02,   # 2% outliers (safe start)
#     random_state=42,
#     n_jobs=-1
# )
# outlier_labels = iso.fit_predict(X_final_transformed)

# # Keep only inliers
# mask = outlier_labels == 1

# X_final_clean = X_final_transformed[mask]
# y_final_clean = y_final.iloc[mask]


# Combine train + validation
X_final = pd.concat([X_train, X_test], axis=0)
y_final = pd.concat([y_train, y_test], axis=0)

X_final_transformed = transformer.fit_transform(X_final)







test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

test_ids = test["id"]
test = test.drop(["id"], axis=1)

test = feature_engineering(test)



test_transformed = transformer.transform(test)


#model trials

# lr = LogisticRegression(
#     max_iter=500,
#     class_weight="balanced"
# )
# model = LogisticRegression(
#     penalty="elasticnet",
#     solver="saga",
#     l1_ratio=0.4,
#     max_iter=4000,
#     class_weight="balanced"
# )


# model = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=12,
#     min_samples_leaf=10,
#     class_weight="balanced",
#     random_state=42
# )

# model = lgb.LGBMClassifier(
#     n_estimators=600,
#     learning_rate=0.05,
#     num_leaves=31,
#     max_depth=-1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     class_weight="balanced",
#     random_state=42
# )





#OOF 

# lg.fit(X_final_transformed, y_final)
# from sklearn.model_selection import StratifiedKFold
# n_folds = 5
# skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# oof_preds = np.zeros((X_final_transformed.shape[0], 3))
# test_preds = np.zeros((test_transformed.shape[0], 3))


# lgb_model = lgb.LGBMClassifier(
#     n_estimators=1200,
#     learning_rate=0.03,
#     num_leaves=64,
#     max_depth=-1,
#     min_child_samples=25,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     reg_alpha=0.5,
#     reg_lambda=0.8,
#     class_weight="balanced",
#     random_state=42
# )

# # lgb_model.fit(X_final_transformed,y_final)

# # from xgboost import XGBClassifier

# # xgb_model = XGBClassifier(
# #     n_estimators=1200,
# #     learning_rate=0.03,
# #     max_depth=6,
# #     subsample=0.8,
# #     colsample_bytree=0.8,
#     reg_alpha=0.5,
#     reg_lambda=1.0,
#     scale_pos_weight=1.65,
#     eval_metric="logloss",
#     random_state=42
# )
# # xgb_model.fit(X_final_transformed,y_final)

# # from catboost import CatBoostClassifier

# # cat_model = CatBoostClassifier(
# #     iterations=1200,
# #     learning_rate=0.03,
# #     depth=6,
# #     l2_leaf_reg=6,
# #     loss_function="Logloss",
# #     eval_metric="AUC",
# #     random_seed=42,
# #     verbose=0
# # )
# # # cat_model.fit(X_final_transformed,y_final)

# for fold, (train_idx, val_idx) in enumerate(skf.split(X_final_transformed, y_final)):
#     print(f"\nFold {fold+1}")

#     X_tr = X_final_transformed[train_idx]
#     y_tr = y_final.iloc[train_idx]

#     X_val = X_final_transformed[val_idx]
#     y_val = y_final.iloc[val_idx]

#     lgb_model.fit(X_tr,y_tr)
#     oof_preds[val_idx, 0] = lgb_model.predict_proba(X_val)[:, 1]
#     test_preds[:, 0] += lgb_model.predict_proba(test_transformed)[:, 1] / n_folds

#     xgb_model.fit(X_tr, y_tr)
#     oof_preds[val_idx, 1] = xgb_model.predict_proba(X_val)[:, 1]
#     test_preds[:, 1] += xgb_model.predict_proba(test_transformed)[:, 1] / n_folds

#     cat_model.fit(X_tr, y_tr)
#     oof_preds[val_idx, 2] = cat_model.predict_proba(X_val)[:, 1]
#     test_preds[:, 2] += cat_model.predict_proba(test_transformed)[:, 1] / n_folds


# meta_model = LogisticRegression(max_iter=2000,solver="lbfgs")
# meta_model.fit(oof_preds, y_final)

# final_test_preds = meta_model.predict_proba(test_preds)[:, 1]


#lgbm using mutliple seed

seeds = [42, 100, 202, 777, 999]
test_preds = []

for seed in seeds:
    model = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=seed
    )
    model.fit(X_final_transformed, y_final)
    test_preds.append(model.predict_proba(test_transformed)[:, 1])

final_pred = np.mean(test_preds, axis=0)



















# lgb_pred=lgb_model.predict_proba(test_transformed)[:, 1]
# cat_pred=cat_model.predict_proba(test_transformed)[:, 1]
# xgb_pred=cat_model.predict_proba(test_transformed)[:, 1]


# Simple blending

# final_probs = 0.55 * lgb_pred + 0.25 * xgb_pred + 0.2 * cat_pred

# final_pred = (
#     0.55 * lgb_pred +
#     0.30 * final_test_preds +
#     0.15 * cat_pred
# )


# print(X_train_transformed.shape)
# print(y.value_counts())



submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": final_pred
})

submission.to_csv("submission.csv", index=False)

print("✅ Submission file created")
print(submission.head())


