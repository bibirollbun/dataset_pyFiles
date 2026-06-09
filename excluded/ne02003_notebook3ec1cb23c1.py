import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from lightgbm import LGBMClassifier
import os


os.listdir('/kaggle/input')



DATASET_FOLDER = 'playground-series-s5e12'
TRAIN_PATH = f'/kaggle/input/{DATASET_FOLDER}/train.csv'
TEST_PATH  = f'/kaggle/input/{DATASET_FOLDER}/test.csv'
SUB_PATH   = f'/kaggle/input/{DATASET_FOLDER}/sample_submission.csv'




train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)




X_train = train.drop(['id', 'diagnosed_diabetes'], axis=1)
y_train = train['diagnosed_diabetes']
X_test  = test.drop(['id'], axis=1)




cat_cols = X_train.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col]  = le.transform(X_test[col].astype(str))




scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)




xgb_model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)




lgb_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    class_weight='balanced',
    random_state=42
)
lgb_model.fit(X_train, y_train)




final_pred = (
    xgb_model.predict_proba(X_test)[:,1] +
    lgb_model.predict_proba(X_test)[:,1]
) / 2




submission = pd.read_csv(SUB_PATH)
submission['diagnosed_diabetes'] = final_pred
submission.to_csv('submission.csv', index=False)
submission.head()


