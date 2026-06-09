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
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression


train = pd.read_csv(r'/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e11/test.csv')
original = pd.read_csv(r'/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')


train.head(2)


original.head(2)


common_columns = list(set(train.columns).intersection(set(original.columns)))


common_columns


combine = pd.concat([original[common_columns], train[common_columns]], ignore_index = True)


for df in [combine,test]:
    df['loan_income_ratio'] = df['loan_amount'] / df['annual_income']
    df['loan_rate'] = (df['interest_rate']/100) * df['loan_amount']
    df['debt'] = df['annual_income'] * df['debt_to_income_ratio']
    df['loan_rate_debt_ratio'] = df['loan_rate'] / df['debt']
    df['credit_score_debt_ratio'] = df['credit_score'] / df['debt']
    df['debt_per_loan_income_ratio'] = df['debt'] * df['loan_income_ratio'] 
    df['debt_to_income_per_interest'] = df['debt_to_income_ratio'] * df['interest_rate']
    df['loan_amount_per_credit_score'] = df['loan_amount'] * df['credit_score']
    df['annual_income_per_inter_rate'] = df['annual_income'] * df['interest_rate']
    df['loan_amount_over_debt'] = df['loan_amount'] / df['debt']
    df['loan_amt_over_annu_inc'] = df['loan_amount'] /df['annual_income']
    df['cat1'] = df['employment_status'] + df['marital_status']
    df['cat2'] = df['marital_status'] + df['education_level']
    df['cat3'] = df['loan_purpose'] + df['employment_status']
    df['cat4'] = df['grade_subgrade'] + df['loan_purpose']
    df['cat5'] = df['gender'] + df['employment_status']
    




combine['loan_paid_back'].value_counts() / combine.shape[0]


from sklearn.utils import resample


X = combine.drop('loan_paid_back',axis = 1)
y = combine['loan_paid_back']


X_majority = X[y == 0]
y_majority = y[y == 0]
X_minority = X[y == 1]
y_minority = y[y == 1]


X_minority_oversampled, y_minority_oversampled = resample(X_minority, y_minority , 
                                                          #replace=True, # sample with replacement
                                                          n_samples=len(X_majority), # match majority class
                                                          random_state=42)


X_oversampled = pd.concat([X_majority, X_minority_oversampled])
y_oversampled = pd.concat([y_majority, y_minority_oversampled])


sum(y_oversampled) / len(y_oversampled)


assert X_oversampled.shape[0] == len(y_oversampled)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


num_columns = X_oversampled.select_dtypes(np.number).columns
cat_columns = X_oversampled.select_dtypes(include=["object", "category"]).columns


num_columns


cat_columns


col_transformer = ColumnTransformer(
    transformers = [
        ('num',StandardScaler(),num_columns),
        ('cat',OneHotEncoder(sparse_output = False),cat_columns)
    ]
    
)


X_oversampled_transf = pd.DataFrame(col_transformer.fit_transform(X_oversampled), columns = col_transformer.get_feature_names_out())


X_oversampled_transf.index = y_oversampled.index


X_oversampled_transf.head(2)


y_oversampled.head(2)


X_train, X_test, y_train, y_test = train_test_split(X_oversampled_transf, y_oversampled , test_size =  0.2, random_state = 42)


X_train.head()


y_train.head()


assert X_train.shape[0] == len(y_train)
assert X_test.shape[0] == len(y_test)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


model1 = xgb.XGBClassifier(
    n_estimators=10000,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc"   
)


assert (X_train.index == y_train.index).all()


auc_scores_model1 = []

for train_idx, val_idx in skf.split(X_train, y_train):
    X_train_skf, X_val_skf = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_skf, y_val_skf = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model1.fit(X_train_skf, y_train_skf)
    y_pred = model1.predict_proba(X_val_skf)[:,1]

    auc = roc_auc_score(y_val_skf, y_pred)
    auc_scores_model1.append(auc)


print(f"Mean auc scores model1:  {np.mean(auc_scores_model1)}")


model2 = CatBoostClassifier(
    iterations=10000,               
    depth=5,
    learning_rate=0.01,
    border_count=244,
    eval_metric="AUC",
    loss_function="Logloss",
    random_seed=42,
    verbose=0,
    early_stopping_rounds=300)


auc_scores_model2 = []

for train_idx, val_idx in skf.split(X_train, y_train):
    X_train_skf, X_val_skf = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_skf, y_val_skf = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model2.fit(X_train_skf, y_train_skf)
    y_pred = model1.predict_proba(X_val_skf)[:,1]

    auc = roc_auc_score(y_val_skf, y_pred)
    auc_scores_model2.append(auc)


print(f"Mean auc scores model2:  {np.mean(auc_scores_model2)}")


model3 = LogisticRegression(
        solver="liblinear",
        penalty="l1",
        max_iter=1000,
        n_jobs=-1
    )


auc_scores_model3 = []

for train_idx, val_idx in skf.split(X_train, y_train):
    X_train_skf, X_val_skf = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_skf, y_val_skf = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model3.fit(X_train_skf, y_train_skf)
    y_pred = model3.predict_proba(X_val_skf)[:,1]

    auc = roc_auc_score(y_val_skf, y_pred)
    auc_scores_model3.append(auc)


print(f"Mean auc scores model3:  {np.mean(auc_scores_model3)}")


models_scores = [np.mean(auc_scores_model1), np.mean(auc_scores_model2), np.mean(auc_scores_model3)]


np.mean(models_scores)


tot_scores = 0
for i in models_scores:
    tot_scores += 1/i



models_weights = [(1/i)/tot_scores for i in models_scores]


sum(models_weights)


y_preds = models_weights[0] * model1.predict_proba(X_test)[:,1] + models_weights[1] * model2.predict_proba(X_test)[:,1] + models_weights[2] * model3.predict_proba(X_test)[:,1]


print(f"AUC total models:  {roc_auc_score(y_test, y_preds)}")


test_transformed = pd.DataFrame(col_transformer.fit_transform(test), columns = col_transformer.get_feature_names_out())


test_preds = models_weights[0] * model1.predict_proba(test_transformed)[:,1] + models_weights[1] * model2.predict_proba(test_transformed)[:,1] + models_weights[2] * model3.predict_proba(test_transformed)[:,1]


submission = pd.DataFrame({'id': test['id'], 'loan_paid_back': test_preds} )


submission.head()


submission.to_csv('submission.csv', index=False)
print("Submission created")

