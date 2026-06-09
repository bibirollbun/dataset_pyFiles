import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import shap
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train["Personality"] = train["Personality"].map({"Introvert": 0, "Extrovert": 1})


X = train.drop(columns=["id", "Personality"])
y = train["Personality"]
X_test = test.drop(columns=["id"])

non_numeric_cols = X.select_dtypes(include='object').columns
for col in non_numeric_cols:
    X[col] = X[col].astype('category').cat.codes
    X_test[col] = X_test[col].astype('category').cat.codes

X["psych_sum"] = X.select_dtypes(include='number').sum(axis=1)
X_test["psych_sum"] = X_test.select_dtypes(include='number').sum(axis=1)


models = [
    ('lgb', LGBMClassifier(n_estimators=200, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('cat', CatBoostClassifier(verbose=0, iterations=200, random_state=42))
]

stacking_model = StackingClassifier(
    estimators=models,
    final_estimator=LGBMClassifier(n_estimators=100),
    cv=5,
    passthrough=True
)

stacking_model.fit(X, y)


# 6. PREDICT & SUBMIT
preds = stacking_model.predict(X_test)
sample_submission["Personality"] = np.where(preds == 1, "Extrovert", "Introvert")
sample_submission.to_csv("submission.csv", index=False)



explainer = shap.Explainer(stacking_model.named_estimators_['lgb'], X)
shap_values = explainer(X[:100])
shap.plots.beeswarm(shap_values)

