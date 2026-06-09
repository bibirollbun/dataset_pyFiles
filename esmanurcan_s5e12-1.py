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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid")



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

train.head()



print("Train shape:", train.shape)
print("Test shape:", test.shape)

train.info()



train.isna().sum().sort_values(ascending=False)



sns.countplot(x=train["diagnosed_diabetes"])
plt.title("Target Distribution")
plt.show()

train["diagnosed_diabetes"].value_counts(normalize=True)



categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = train.select_dtypes(exclude=["object"]).columns.tolist()

numeric_cols.remove("diagnosed_diabetes")  # target numeric ama feature değil

categorical_cols, numeric_cols[:5]



from sklearn.preprocessing import LabelEncoder

train_df = train.copy()
test_df = test.copy()

categorical_cols = [
    'gender', 'ethnicity', 'education_level',
    'income_level', 'smoking_status', 'employment_status'
]

le = LabelEncoder()

for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])



X = train_df.drop(["diagnosed_diabetes", "id"], axis=1)
y = train_df["diagnosed_diabetes"]



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape, X_val.shape



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Validation tahmini (probability)
val_preds = model.predict_proba(X_val)[:, 1]

# AUC hesaplama
auc = roc_auc_score(y_val, val_preds)
auc



from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb_model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist"
)

xgb_model.fit(X_train, y_train)

val_preds_xgb = xgb_model.predict_proba(X_val)[:, 1]
auc_xgb = roc_auc_score(y_val, val_preds_xgb)
auc_xgb



import lightgbm as lgb
from sklearn.metrics import roc_auc_score

lgb_model = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
)

lgb_model.fit(X_train, y_train)

val_preds_lgb = lgb_model.predict_proba(X_val)[:, 1]
auc_lgb = roc_auc_score(y_val, val_preds_lgb)
auc_lgb



# === FEATURE ENGINEERING ===

for d in [train_df, test_df]:

    d['bp_ratio'] = d['systolic_bp'] / d['diastolic_bp']
    d['chol_ratio'] = d['ldl_cholesterol'] / d['hdl_cholesterol']
    d['tg_hdl_ratio'] = d['triglycerides'] / d['hdl_cholesterol']
    d['obesity_index'] = d['bmi'] * d['waist_to_hip_ratio']

    d['sedentary_index'] = d['screen_time_hours_per_day'] / (d['physical_activity_minutes_per_week'] + 1)
    d['wellness_index'] = d['diet_score'] + d['sleep_hours_per_day']

    # Risk flags
    d['high_bp_flag'] = ((d['systolic_bp'] > 130) | (d['diastolic_bp'] > 85)).astype(int)
    d['high_ldl_flag'] = (d['ldl_cholesterol'] > 130).astype(int)
    d['high_bmi_flag'] = (d['bmi'] > 30).astype(int)



X_train = train_df.drop("diagnosed_diabetes", axis=1)
y_train = train_df["diagnosed_diabetes"]

X_test = test_df.copy()
train_df.head()



from sklearn.metrics import roc_auc_score

features = [col for col in train_df.columns if col not in ['diagnosed_diabetes', 'id']]
X_train = train_df[features]
y_train = train_df['diagnosed_diabetes']

X_test = test_df[features]

params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'learning_rate': 0.03,
    'num_leaves': 50,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50
}

lgb_train = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, lgb_train, num_boost_round=1000)

train_pred = model.predict(X_train)
auc = roc_auc_score(y_train, train_pred)
auc


test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': test_pred
})

submission.to_csv('submission.csv', index=False)
submission.head()


