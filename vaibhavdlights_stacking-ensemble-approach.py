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
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')


eps = 1e-6

df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
df["bp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + eps)
df["map"] = (2 * df["diastolic_bp"] + df["systolic_bp"]) / 3.0

df["chol_ratio_ldl_hdl"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + eps)
df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + eps)
df["non_hdl_chol"] = df["cholesterol_total"] - df["hdl_cholesterol"]

df["bmi_age"] = df["bmi"] * df["age"]
df["whr_bmi"] = df["waist_to_hip_ratio"] * df["bmi"]

df["activity_bmi"] = df["physical_activity_minutes_per_week"] / (df["bmi"] + 1.0)
df["sleep_screen_ratio"] = df["sleep_hours_per_day"] / (df["screen_time_hours_per_day"] + 1.0)
df["alcohol_age"] = df["alcohol_consumption_per_week"] * df["age"]

df["hr_sbp"] = df["heart_rate"] * df["systolic_bp"]

df["log_triglycerides"] = np.log1p(df["triglycerides"])
df["log_ldl"] = np.log1p(df["ldl_cholesterol"])
df["log_chol_total"] = np.log1p(df["cholesterol_total"])


train_df = df.iloc[0:50000]
test_df = df.iloc[50000:70000]
train_df.drop(columns='id', inplace=True)
test_df.drop(columns='id', inplace=True)
test_df.shape


cat_col = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

train_df = pd.get_dummies(train_df, columns=cat_col, drop_first=False, dtype=int)
test_df = pd.get_dummies(test_df, columns=cat_col, drop_first=False, dtype=int)


X_train = train_df.drop(columns=['diagnosed_diabetes'])
y_train = train_df['diagnosed_diabetes']
X_test = test_df.drop(columns=['diagnosed_diabetes'])
y_test = test_df['diagnosed_diabetes']


from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def roc_auc_plot(X_test, y_test, model):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)

    history_path = "history.txt"
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        last_idx = int(lines[-1].split(",", 1)[0]) if lines else -1
    except FileNotFoundError:
        last_idx = -1

    next_idx = last_idx + 1
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(f"{next_idx},{auc:.5f}\n")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.5f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return auc


gb = GradientBoostingClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    random_state=0,
)

lgbm = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
)

xgb = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=0,
    n_jobs=-1,
)

cat = CatBoostClassifier(
    iterations=4000,
    learning_rate=0.02,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=0,
    verbose=0,
)

stack = StackingClassifier(
    estimators=[("gb", gb), ("lgbm", lgbm), ("xgb", xgb), ("cat", cat)],
    final_estimator=LogisticRegression(max_iter=4000, penalty='l2', C=0.3),
    stack_method="predict_proba",
    passthrough=True,
    # cv=5,
    n_jobs=-1
)

model = Pipeline([
    ("stack", stack)
])

model.fit(X_train, y_train)


roc_auc_plot(X_test, y_test, model)


def make_submission(model, test_csv_path, cat_col, out_path="submission.csv",
                    id_col="id", target_col="diagnosed_diabetes"):
    """
    Reads test CSV, applies one-hot encoding, predicts probabilities, and writes submission CSV.

    IMPORTANT: This assumes your model was trained on data processed the SAME way
    (same get_dummies columns order). If you used a sklearn Pipeline with OneHotEncoder,
    you should NOT use get_dummies here—just pass the raw dataframe to the pipeline.
    """
    subm = pd.read_csv(test_csv_path)

    if id_col not in subm.columns:
        raise ValueError(f"'{id_col}' column not found in {test_csv_path}")

    X = subm.drop(columns=[id_col])

    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=cat_col, drop_first=False, dtype=int)

    # Predict probability of class 1
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model has no predict_proba(). Use a model/pipeline that supports it "
                             "or adapt to decision_function.")
    pred = model.predict_proba(X)[:, 1]

    submission = pd.DataFrame({
        id_col: subm[id_col],
        target_col: pred
    })

    submission.to_csv(out_path, index=False)
    return submission


# submission = make_submission(model, "test.csv", cat_col, out_path="submission.csv")




