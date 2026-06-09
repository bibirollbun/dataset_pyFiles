import os
import shutil
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")


SEED = 42
train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')


train.info()


categorical_cols = []

for column in train.columns:
    if train[column].dtype == "object":
        categorical_cols.append(column)

categorical_cols


ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train[categorical_cols] = ordinal_encoder.fit_transform(train[categorical_cols])
test[categorical_cols] = ordinal_encoder.transform(test[categorical_cols])


label_encoder = LabelEncoder()
encode_target = 'Depression'

train[encode_target] = label_encoder.fit_transform(train[encode_target])


features = train.columns[1:-1]
target = 'Depression'

X_train, X_test, y_train, y_test = train_test_split(train[features], train[target], random_state=SEED)


len(X_train), len(X_test)


cat_model = CatBoostClassifier(
    task_type="GPU",
    devices='0:1',
    verbose=0, # to avoid looking at long logs
    random_state=SEED
)
cat_model.fit(X_train, y_train)


cat_preds = cat_model.predict(X_test)
cat_acc = accuracy_score(cat_preds, y_test)

print(f'Accuracy on CatBoost: {cat_acc:.2f}')


cat_result = cat_model.predict(test[features])


submission = pd.DataFrame({
    'id': test['id'],
    'Depression': cat_result
})

submission.to_csv('CAT_result.csv', index=False)
submission.head()


xgb_model = XGBClassifier(
    device='cuda',
    random_state=SEED,
)
xgb_model.fit(X_train, y_train)


xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(xgb_preds, y_test)

print(f'Accuracy on XGBoost: {xgb_acc:.2f}')


XGBoost_result = xgb_model.predict(test[features])


submission = pd.DataFrame({
    'id': test['id'],
    'Depression': XGBoost_result
})

submission.to_csv('XGB_result.csv', index=False)
submission.head()


lgbm_model = LGBMClassifier(
    verbose=-1,
    random_state=SEED,
)
lgbm_model.fit(X_train, y_train)


lgbm_preds = lgbm_model.predict(X_test)
lgbm_acc = accuracy_score(lgbm_preds, y_test)

print(f'Accuracy on LightGBM: {lgbm_acc:.2f}')


lgbm_result = lgbm_model.predict(test[features])


submission = pd.DataFrame({
    'id': test['id'],
    'Depression': lgbm_result
})

submission.to_csv('lgbm_result.csv', index=False)
submission.head()


cat_model = CatBoostClassifier(task_type="GPU", devices='0:1', verbose=0, random_state=SEED)
xgb_model = XGBClassifier(device='cuda', random_state=SEED)
lgbm_model = LGBMClassifier(verbose=-1, random_state=SEED)

# Creating an ensemble with VotingClassifier
ensemble = VotingClassifier(
    estimators=[
        ('xgboost', xgb_model),
        ('catboost', cat_model),
        ('lightgbm', lgbm_model)
    ],
    voting='soft'  # soft voting for probabilistic predictions
)

ensemble.fit(X_train, y_train)


ensemble_preds = ensemble.predict(X_test)
ensemble_acc = accuracy_score(ensemble_preds, y_test)

print(f'Accuracy on ensemble: {ensemble_acc:.2f}')


ensemble_result = ensemble.predict(test[features])


submission = pd.DataFrame({
    'id': test['id'],
    'Depression': ensemble_result
})

submission.to_csv('ensemble_result.csv', index=False)
submission.head()




