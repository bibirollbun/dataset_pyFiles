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
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import time
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')

test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

sample = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train.columns


features = [
     'annual_income', 'debt_to_income_ratio', 'credit_score',
       'loan_amount', 'interest_rate', 'gender', 'marital_status',
       'education_level', 'employment_status', 'loan_purpose',
       'grade_subgrade'
]
target_col = 'loan_paid_back'   

X = train[features].copy()
y = train[target_col].copy()
X_test = test[features].copy()


cat_cols = ['employment_status', 'loan_purpose','gender', 'marital_status','education_level',
            'grade_subgrade']
num_cols = [c for c in features if c not in cat_cols]

num_imputer = SimpleImputer(strategy='median')

cat_imputer = SimpleImputer(strategy='most_frequent')

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

for c in cat_cols:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')

print("Class distribution:\n", y.value_counts(normalize=True))


X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=RANDOM_STATE)


base_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE

}


print("\n--- Quick baseline train with callbacks (fast check) ---")
quick_params = base_params.copy()
quick_params.update({'n_estimators': 1000, 'n_jobs': -1, 'verbose': -1})
quick_model = LGBMClassifier(**quick_params)

t0 = time.time()
quick_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[
        early_stopping(stopping_rounds=50),   
        log_evaluation(period=0)
    ]
)
t1 = time.time()
val_pred = quick_model.predict_proba(X_val)[:, 1]
print(f"Quick baseline AUC: {roc_auc_score(y_val, val_pred):.6f}")
print("Quick best_iteration:", quick_model.best_iteration_)
print(f"Quick train time: {(t1-t0):.1f} sec")


print("\n--- Running RandomizedSearchCV (reduced and faster) ---")

param_dist = {
    'num_leaves': [31, 50, 70, 100],
    'learning_rate': [0.01, 0.02, 0.03, 0.05],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0.0, 0.1, 0.5],
    'reg_lambda': [0.0, 0.5, 1.0],
    'min_child_samples': [20, 40, 60]
}

search_estimator = LGBMClassifier(objective='binary', n_estimators=1000,
                                 random_state=RANDOM_STATE, n_jobs=1)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  

search = RandomizedSearchCV(
    estimator=search_estimator,
    param_distributions=param_dist,
    n_iter=10,              
    scoring='roc_auc',
    cv=cv,
    random_state=RANDOM_STATE,
    verbose=2,
    n_jobs=-1,
    return_train_score=False
)

t0 = time.time()
search.fit(X, y)   
t1 = time.time()
print(f"RandomizedSearchCV finished in {(t1-t0)/60:.2f} minutes")
print("Best CV AUC:", search.best_score_)
print("Best params:", search.best_params_)

best_params = search.best_params_.copy()


final_params = base_params.copy()
final_params.update(best_params)
final_params.update({'n_estimators': 4000, 'n_jobs': -1, 'verbose': -1})

final_model = LGBMClassifier(**final_params)

X_full_tr, X_full_val, y_full_tr, y_full_val = train_test_split(
    X, y, test_size=0.05, stratify=y, random_state=RANDOM_STATE
)

print("\n--- Training final model with early stopping callbacks ---")
t0 = time.time()
final_model.fit(
    X_full_tr, y_full_tr,
    eval_set=[(X_full_val, y_full_val)],
    eval_metric='auc',
    callbacks=[
        early_stopping(stopping_rounds=200),
        log_evaluation(period=50)
    ]
)
t1 = time.time()
print("Final best_iteration:", final_model.best_iteration_)
print(f"Final train time: {(t1-t0)/60:.2f} minutes")


test_preds = final_model.predict_proba(X_test)[:, 1]
print("Prediction made successfully!")


submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission saved: submission.csv")

