import pandas as pd
import numpy as np
import os 
import time 
import math
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
import plotly.io as pio
import plotly.subplots as sp
from scipy.stats import skew, kurtosis, zscore
from scipy.stats import chi2_contingency
import plotly.figure_factory as ff  
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
import xgboost as xgb


warnings.filterwarnings('ignore')
sns.set(style='darkgrid')
pio.renderers.default = 'iframe_connected'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train.head()


train.describe().round(2).T


train.isnull().sum()


num_col = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history']

cat_col = ['gender', 'ethnicity', 'education_level', 'income_level',
       'smoking_status', 'employment_status']

target_col = 'diagnosed_diabetes'


corr_mat = train[num_col].corr()

plt.figure(figsize=(14,12))
sns.heatmap(
    corr_mat,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)
plt.title("Correlation Matrix (Numerical Variables)")
plt.tight_layout()
plt.show()


X = train.drop("diagnosed_diabetes", axis=1)
y = train["diagnosed_diabetes"]


cat_cols = X.select_dtypes(include="object").columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype("category").cat.codes
    test[col] = test[col].astype("category").cat.codes


N_FOLDS = 5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
pred_lgb = np.zeros(len(test))
pred_xgb = np.zeros(len(test))


# --- 4. XGBoost Loop ---
xgb_params = dict(
    n_estimators=3000,    # 3000 Trees
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    enable_categorical=True,
    tree_method="gpu_hist",  
    device="cuda",
)
print("Training XGBoost...")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    # Early stopping in fit()
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=100
    )

    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:,1]
    pred_xgb += xgb_model.predict_proba(test)[:,1] / N_FOLDS

print(f"XGB OOF AUC: {roc_auc_score(y, oof_xgb):.5f}")


test_pred = 0.5 * pred_lgb + 0.5 * pred_xgb

submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")




