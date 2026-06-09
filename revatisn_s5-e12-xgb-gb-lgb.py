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


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train.info()


test.info()


TARGET = 'diagnosed_diabetes'
CATS = ['gender','ethnicity','education_level','income_level',
        'smoking_status','employment_status']
NUMS = ['age','alcohol_consumption_per_week','physical_activity_minutes_per_week',
        'diet_score','sleep_hours_per_day','screen_time_hours_per_day','bmi',
        'waist_to_hip_ratio','systolic_bp','diastolic_bp','heart_rate',
        'cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides',
        'family_history_diabetes','hypertension_history','cardiovascular_history']


import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import numpy as np

X = train[CATS + NUMS]
y = train[TARGET]
X_test = test[CATS + NUMS]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATS)
    ]
)

X_tr_p   = preprocess.fit_transform(X_tr)
X_val_p  = preprocess.transform(X_val)
X_test_p = preprocess.transform(X_test)

# XGBoost model
xgb_clf = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)
xgb_clf.fit(X_tr_p, y_tr)
xgb_val_pred = xgb_clf.predict_proba(X_val_p)[:, 1]
xgb_test_pred = xgb_clf.predict_proba(X_test_p)[:, 1]


# Gradient Boosting Classifier (sklearn)
gbc_clf = GradientBoostingClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
gbc_clf.fit(X_tr_p, y_tr)
gbc_val_pred = gbc_clf.predict_proba(X_val_p)[:, 1]
gbc_test_pred = gbc_clf.predict_proba(X_test_p)[:, 1]

# Prepare stacking input features (validation predictions)
stack_X = np.column_stack((xgb_val_pred, gbc_val_pred))

# Train logistic regression meta-model
meta_clf = LogisticRegression(random_state=42)
meta_clf.fit(stack_X, y_val)

# Predict on validation set using meta-model
stack_val_pred = meta_clf.predict_proba(stack_X)[:, 1]

# Predict on test set using meta-model
stack_test_X = np.column_stack((xgb_test_pred, gbc_test_pred))
stack_test_pred = meta_clf.predict_proba(stack_test_X)[:, 1]

print("Stacking Ensemble AUC:", roc_auc_score(y_val, stack_val_pred))

# Create submission
sub = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': stack_test_pred
})

sub.to_csv('submission_stacking_lgbm_xgb_gbc.csv', index=False)


