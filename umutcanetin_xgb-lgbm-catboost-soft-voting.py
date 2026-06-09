import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


df_train.columns


df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


def pipeline(df=df_train):
    res = df.copy()

    # Polynomial features

    res["age**2"] = res["age"] ** 2

    # Binning

    res["smoke"] = (res["smoking_status"] != "Never").astype(int)
    res["isRetired"] = (res["employment_status"] == "Retired").astype(int)
    res["isMan"] = (res["gender"] == "Male").astype(int)

    # Cutting

    median = res["physical_activity_minutes_per_week"].median()
    res["physicalCutted"] = res["physical_activity_minutes_per_week"].apply(
        lambda activity: 1 if activity >= median else 0
    )

    # Interactions

    res["alcohol*age"] = res["alcohol_consumption_per_week"] * res["age"]
    res["activity*age"] = res["physical_activity_minutes_per_week"] * res["age"]
    res["activity*age*diet"] = (
        res["physical_activity_minutes_per_week"] * res["age"] * res["diet_score"]
    )
    res["bmi*age"] = res["bmi"] * res["age"]
    res["bmi*activity"] = res["bmi"] * res["physical_activity_minutes_per_week"]

    # Medicine specific feature engineering

    res["pulse_pressure"] = res["systolic_bp"] - res["diastolic_bp"]
    res["MAP"] = res["diastolic_bp"] + (res["pulse_pressure"] / 3)

    res["cholesterol_ratio"] = res["cholesterol_total"] / (
        res["hdl_cholesterol"] + 1e-6
    )
    res["ldl_hdl_ratio"] = res["ldl_cholesterol"] / (res["hdl_cholesterol"] + 1e-6)

    has_high_bmi = (res["bmi"] >= 30).astype(int)
    has_high_bp = ((res["systolic_bp"] >= 130) | (res["diastolic_bp"] >= 85)).astype(
        int
    )
    has_high_trig = (res["triglycerides"] >= 150).astype(int)
    has_low_hdl = (res["hdl_cholesterol"] < 40).astype(int)

    res["metabolic_score"] = has_high_bmi + has_high_bp + has_high_trig + has_low_hdl

    # Encode ethnicity

    res["isHispanic"] = (res["ethnicity"] == "Hispanic").astype(int)
    res["isWhite"] = (res["ethnicity"] == "White").astype(int)
    res["isAsian"] = (res["ethnicity"] == "Asian").astype(int)
    res["isBlack"] = (res["ethnicity"] == "Black").astype(int)

    # Dropped

    toDropped = [
        "ethnicity",
        "gender",
        "sleep_hours_per_day",
        "screen_time_hours_per_day",
        "income_level",
        "education_level",
        "alcohol_consumption_per_week",
        "smoking_status",
        "employment_status",
    ]
    res = res.drop(columns=toDropped)

    return res


df_train = pipeline(df_train)
df_test = pipeline(df_test)


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    verbosity=1,  # silent
    reg_alpha=0,
    reg_lambda=0.5,
    random_state=42,
    scale_pos_weight=0.6,
    eval_metric="logloss",
)

X = df_train.drop(columns="diagnosed_diabetes")
y = df_train["diagnosed_diabetes"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print(f"Score: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))


kaggle_pred = model.predict_proba(df_test)[:, 1]
submission = pd.DataFrame({"id": df_test["id"], "diagnosed_diabetes": kaggle_pred})
submission.to_csv("xgb2.csv", index=False)


import lightgbm as lgb


model = lgb.LGBMClassifier(
    objective="binary",
    metric="binary_logloss",
    n_estimators=1000,
    learning_rate=0.025,
    random_state=42,
    scale_pos_weight=0.6,
)

model.fit(X_train, y_train)

kaggle_pred = model.predict_proba(df_test)[:, 1]
submission = pd.DataFrame({"id": df_test["id"], "diagnosed_diabetes": kaggle_pred})
submission.to_csv("lgbm2.csv", index=False)


from catboost import CatBoostClassifier

params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "rsm": 0.8,
    "early_stopping_rounds": 50,
    "verbose": 100,
    "scale_pos_weight": 0.6,
}

model = CatBoostClassifier(**params)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    use_best_model=True,
)


kaggle_pred = model.predict_proba(df_test)[:, 1]
submission = pd.DataFrame({"id": df_test["id"], "diagnosed_diabetes": kaggle_pred})
submission.to_csv("catboost1.csv", index=False)


cat = pd.read_csv("catboost1.csv")["diagnosed_diabetes"]
lgbm = pd.read_csv("lgbm2.csv")["diagnosed_diabetes"]
xgb = pd.read_csv("xgb2.csv")["diagnosed_diabetes"]

ensambled = cat * 0.35 + lgbm * 0.35 + xgb * 0.3

submission = pd.DataFrame({"id": df_test["id"], "diagnosed_diabetes": ensambled})
submission.to_csv("submission.csv", index=False)




