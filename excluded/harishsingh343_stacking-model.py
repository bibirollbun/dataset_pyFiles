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
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer    
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
!pip install category-encoders
from category_encoders import TargetEncoder



df_train  = pd.read_csv(r"/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e12/test.csv')
test = df_test.copy()


X = df_train.drop(columns=['diagnosed_diabetes', 'id'])
y = df_train['diagnosed_diabetes']
df_test = df_test.drop(columns= ['id'])


def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal weight'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'
X['bmi_category'] = X['bmi'].apply(bmi_category)
df_test['bmi_category'] = df_test['bmi'].apply(bmi_category)

df_test['medical_history_count'] = df_test[['cardiovascular_history', 'family_history_diabetes', 'hypertension_history']].sum(axis=1)
X['medical_history_count'] = X[['cardiovascular_history', 'family_history_diabetes', 'hypertension_history']].sum(axis=1)

X['ldl_hdl_ratio'] = X['ldl_cholesterol'] / (X['hdl_cholesterol']+1)
df_test['ldl_hdl_ratio'] = df_test['ldl_cholesterol'] / (df_test['hdl_cholesterol']+1)

X['cholesterol_ratio'] = X['cholesterol_total'] / (X['hdl_cholesterol']+1)
df_test['cholesterol_ratio'] = df_test['cholesterol_total'] / (df_test['hdl_cholesterol']+1)

df_test['Hypertension_Risk'] = ((df_test['systolic_bp'] >= 130) | (df_test['diastolic_bp'] >= 80)).astype(int)
X['Hypertension_Risk'] = ((X['systolic_bp'] >= 130) | (X['diastolic_bp'] >= 80)).astype(int)

X["age_bmi"] = X["age"] * X["bmi"]
df_test["age_bmi"] = df_test["age"] * df_test["bmi"]


numerical_cols =    X.select_dtypes(include=['int64', 'float64']).columns.tolist()



categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
binary_cols = ['family_history_diabetes','cardiovascular_history','hypertension_history','medical_history_count','Hypertension_Risk']



enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
imp = SimpleImputer(strategy="median")

X[categorical_cols] = enc.fit_transform(X[categorical_cols])
df_test[categorical_cols] = enc.transform(df_test[categorical_cols])

X[numerical_cols] = imp.fit_transform(X[numerical_cols])
df_test[numerical_cols] = imp.transform(df_test[numerical_cols])


'''
te = TargetEncoder(cols=categorical_cols)

X_selected[categorical_cols] = te.fit_transform(
    X_selected[categorical_cols], y
)


df_test[categorical_cols] = te.transform(
    df_test[categorical_cols]
)
'''


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

pred_lgb = np.zeros(len(df_test))
pred_cat = np.zeros(len(df_test))
pred_xgb = np.zeros(len(df_test))


for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n===== FOLD {fold+1} / 5 =====")

    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    
    lgb = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=48,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        reg_alpha=1.0,
        reg_lambda=1.0,
        verbosity=-1,
        class_weight="balanced"
    )
    lgb.fit(X_train, y_train)

    oof_lgb[val_idx] = lgb.predict_proba(X_valid)[:, 1]
    pred_lgb += lgb.predict_proba(df_test)[:, 1] / kf.n_splits

   
    cat = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=8,
        verbose=False,
        random_seed=42
    )
    cat.fit(X_train, y_train)

    oof_cat[val_idx] = cat.predict_proba(X_valid)[:, 1]
    pred_cat += cat.predict_proba(df_test)[:, 1] / kf.n_splits

  
    xgb = XGBClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        tree_method="hist"
    )
    xgb.fit(X_train, y_train)

    oof_xgb[val_idx] = xgb.predict_proba(X_valid)[:, 1]
    pred_xgb += xgb.predict_proba(df_test)[:, 1] / kf.n_splits


oof_blend = 0.4 * oof_lgb + 0.35 * oof_cat + 0.25 * oof_xgb
pred_blend = 0.4 * pred_lgb + 0.35 * pred_cat + 0.25 * pred_xgb


print("\nLightGBM ROC:", roc_auc_score(y, oof_lgb))
print("CatBoost ROC:", roc_auc_score(y, oof_cat))
print("XGBoost ROC:", roc_auc_score(y, oof_xgb))
print("Blended ROC:", roc_auc_score(y, oof_blend))


stack_train = np.vstack([oof_lgb, oof_cat, oof_xgb]).T
stack_test  = np.vstack([pred_lgb, pred_cat, pred_xgb]).T

meta_model = LogisticRegression(max_iter=2000)
meta_model.fit(stack_train, y)

pred_final = meta_model.predict_proba(stack_test)[:, 1]

print("\nFinal Stacked ROC:",
      roc_auc_score(y, meta_model.predict_proba(stack_train)[:, 1]))


submission_data = pd.DataFrame({
        "id": test['id'],
        "diagnosed_diabetes": pred_final
})
submission_data.to_csv('submission.csv', index = False)

