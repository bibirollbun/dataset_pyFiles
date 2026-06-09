import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


train_path = '/kaggle/input/playground-series-s5e11/train.csv'
test_path = '/kaggle/input/playground-series-s5e11/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


raw = train.copy()

train_cat = raw.copy()
train_log = raw.copy()
test_cat = test.copy()
test_log = test.copy()


train_cat['new_education_level'] = train_cat['education_level'].astype('category').cat.codes
train_cat['new_grade_subgrade'] = train_cat['grade_subgrade'].astype('category').cat.codes

mean_target = train_cat.groupby('loan_purpose')['loan_paid_back'].mean()
train_cat['new_loan_purpose'] = train_cat['loan_purpose'].map(mean_target)

train_cat = pd.get_dummies(train_cat, columns=['gender', 'marital_status', 'employment_status'], drop_first=False)

cols_to_drop = ['education_level', 'grade_subgrade', 'loan_purpose']
train_cat.drop(columns=cols_to_drop, inplace=True)

test_cat['new_education_level'] = test_cat['education_level'].astype('category').cat.codes
test_cat['new_grade_subgrade'] = test_cat['grade_subgrade'].astype('category').cat.codes
test_cat['new_loan_purpose'] = test_cat['loan_purpose'].map(mean_target)  # use TRAIN map
test_cat = pd.get_dummies(test_cat, columns=['gender','marital_status','employment_status'], drop_first=False)
test_cat.drop(columns=['education_level','grade_subgrade','loan_purpose'], inplace=True)




X_cat = train_cat.drop('loan_paid_back', axis=1)
y_cat = train_cat['loan_paid_back']

test_cat = test_cat.copy()
test_ids = test['id']


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X_cat, y_cat,
    test_size=0.20,
    random_state=42,
    stratify=y_cat
)



from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.015,   # recommended for long training
    depth=6,
    l2_leaf_reg=8,
    loss_function='Logloss',
    eval_metric='AUC',
    subsample=0.8,
    random_seed=42,
    verbose=300
)



cat_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=500,
    verbose=0
)


cat_pred = cat_model.predict_proba(test_cat)[:, 1]

import pandas as pd

submission = pd.DataFrame({
    'id': test_ids,
    'prediction': cat_pred
})

submission.to_csv("catboost.csv", index=False)

print("Saved → catboost.csv")



train_log["annual_income"] = np.log1p(train_log["annual_income"])
train_log["debt_to_income_ratio"] = np.log1p(train_log["debt_to_income_ratio"])

train_log = pd.get_dummies(train_log, 
                           columns=['gender','marital_status','education_level',
                                    'employment_status','loan_purpose',
                                    'grade_subgrade'],
                           drop_first=False)

test_log["annual_income"] = np.log1p(test_log["annual_income"])
test_log["debt_to_income_ratio"] = np.log1p(test_log["debt_to_income_ratio"])
test_log = pd.get_dummies(test_log, 
                           columns=['gender','marital_status','education_level',
                                    'employment_status','loan_purpose',
                                    'grade_subgrade'],
                           drop_first=False)


X_log = train_log.drop("loan_paid_back", axis=1)
y_log = train_log["loan_paid_back"]

test_log = test_log.copy()
test_ids = test['id']


from sklearn.model_selection import train_test_split

X_train_log, X_valid_log, y_train_log, y_valid_log = train_test_split(
    X_log, y_log,
    test_size=0.20,
    random_state=42,
    stratify=y_log
)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_log_scaled = scaler.fit_transform(X_train_log)
X_valid_log_scaled = scaler.transform(X_valid_log)
test_log_scaled = scaler.transform(test_log)



from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(
    penalty='l2',
    C=0.8,               # slightly stronger regularization improves Kaggle score
    max_iter=5000,
    solver='lbfgs',
    n_jobs=-1
)

log_model.fit(X_train_log_scaled, y_train_log)



log_pred = log_model.predict_proba(test_log)[:, 1]

import pandas as pd

submission = pd.DataFrame({
    'id': test_ids,
    'prediction': log_pred
})

submission.to_csv("logreg.csv", index=False)

print("Saved → logreg.csv")



import pandas as pd

# Load submissions
cat = pd.read_csv("catboost.csv")
log = pd.read_csv("logreg.csv")

# Check for same IDs
assert all(cat['id'] == log['id']), "ID mismatch in submissions!"

# Weighted blend
final_pred = 0.90 * cat['prediction'] + 0.10 * log['prediction']

# Save blended submission
blend = pd.DataFrame({
    "id": cat['id'],
    "prediction": final_pred
})

blend.to_csv("submission.csv", index=False)

print("Saved → submission.csv")


