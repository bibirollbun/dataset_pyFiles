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
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score


print("Loading data...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape:  {test_df.shape}")


def apply_optimal_features(df):
    df = df.copy()
    

    if df['grade_subgrade'].dtype == 'object':
        df["grade_number"] = df["grade_subgrade"].str.extract(r'(\d+)').astype(float).fillna(0)
        
        grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        df["grade_letter_score"] = df["grade_subgrade"].str[0].map(grade_map).fillna(4).astype(int)


    for column in ['annual_income', 'loan_amount']:
        df[f'{column}_ROUND_100s'] = df[column].round(-2)  
        
    return df

print("Applying feature engineering...")
train_fe = apply_optimal_features(train_df)
test_fe = apply_optimal_features(test_df)


target = 'loan_paid_back'
drop_cols = ['id', target]
features = [c for c in train_fe.columns if c not in drop_cols]

cat_cols = train_fe[features].select_dtypes(include=['object']).columns.tolist()

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_fe[cat_cols] = encoder.fit_transform(train_fe[cat_cols].astype(str))
test_fe[cat_cols] = encoder.transform(test_fe[cat_cols].astype(str))

X = train_fe[features]
y = train_fe[target]
X_test = test_fe[features]


param_dist = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5]
}

clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist', 
    random_state=42,
    n_jobs=-1
)

print("\nStarting Hyperparameter Tuning (this may take a few minutes)...")
search = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_dist,
    n_iter=20, 
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42
)
search.fit(X, y)

print(f"Best Parameters Found: {search.best_params_}")
print(f"Best CV Score: {search.best_score_:.5f}")


best_model = search.best_estimator_
best_model.set_params(n_estimators=2000, learning_rate=0.01) 
best_model.fit(X, y)


print("Generating predictions...")
test_preds = best_model.predict_proba(X_test)[:, 1]


submission['loan_paid_back'] = test_preds
submission.to_csv('submission_tuned.csv', index=False)
print("Success! 'submission_tuned.csv' created.")

