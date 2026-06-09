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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import optuna
import lightgbm as lgb

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# Fill numerical columns
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in num_cols:
    median_val = train[col].median()
    mean_val = train[col].mean()
    if abs(train[col].skew()) > 1:
        train[col].fillna(median_val, inplace=True)
        test[col].fillna(test[col].median(), inplace=True)
    else:
        train[col].fillna(mean_val, inplace=True)
        test[col].fillna(test[col].mean(), inplace=True)

# Fill categorical columns using mode
cat_cols = ['Stage_fear', 'Drained_after_socializing']
for col in cat_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)

# Encode labels
le = LabelEncoder()
train['Personality_encoded'] = le.fit_transform(train['Personality'])

# Encode Stage_fear and Drained_after_socializing
le_stage = LabelEncoder()
le_drained = LabelEncoder()
train['Stage_fear'] = le_stage.fit_transform(train['Stage_fear'])
train['Drained_after_socializing'] = le_drained.fit_transform(train['Drained_after_socializing'])
test['Stage_fear'] = le_stage.transform(test['Stage_fear'])
test['Drained_after_socializing'] = le_drained.transform(test['Drained_after_socializing'])

# Train/test split
X = train.drop(columns=['id', 'Personality', 'Personality_encoded'])
y = train['Personality_encoded']
X_test = test.drop(columns=['id'])
id_test = test['id']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LightGBM tuning
def objective_lgbm(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    return scores.mean()

print(" Tuning LightGBM with Optuna...")
study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=50)

best_params = study_lgbm.best_params
best_params.update({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1
})

print(" Training best LightGBM model...")
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train, y_train)

# Predict and decode
preds_encoded = final_model.predict(X_test)
preds_labels = le.inverse_transform(preds_encoded)

# Save submission
submission = pd.DataFrame({'id': id_test, 'Personality': preds_labels})
submission.to_csv('submission.csv', index=False)
print(" submission.csv created successfully!")





