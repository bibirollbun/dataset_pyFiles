import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline , Pipeline
from sklearn.compose import ColumnTransformer

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier



train_df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
train_df.head()


test_df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-test.csv')
test_df.head()


train_df.isnull().sum()


test_df.isnull().sum()


train_df.describe()


train_df['MonthlyIncomeBin'] = pd.qcut(
    train_df['MonthlyIncome'], q=10, duplicates='drop'
)


train_df.head()


sns.scatterplot(data=train_df, x = 'MonthlyIncome' , y ='age',hue='SeriousDlqin2yrs')
plt.show()


plt.figure(figsize=(18,10))
sns.boxplot(data=train_df, x = 'MonthlyIncomeBin' , y ='age',hue='SeriousDlqin2yrs')
plt.show()


train_df.groupby(['MonthlyIncomeBin'])['SeriousDlqin2yrs'].mean()


train_df['NumberOfDependents'].unique()


plt.figure(figsize=(18,10))
sns.relplot(data=train_df, x = 'MonthlyIncome' , y ='NumberOfDependents',hue='SeriousDlqin2yrs')
plt.show()


sns.relplot(data=train_df, x = 'NumberOfOpenCreditLinesAndLoans' , y ='NumberOfDependents',hue='SeriousDlqin2yrs')
plt.show()


train_df.groupby('NumberOfDependents')['SeriousDlqin2yrs'].mean()


train_df.groupby('NumberOfOpenCreditLinesAndLoans')['SeriousDlqin2yrs'].mean()


train_df['TotalDelinquency'] = train_df['NumberOfTimes90DaysLate'] + train_df['NumberOfTime30-59DaysPastDueNotWorse'] + train_df['NumberOfTime60-89DaysPastDueNotWorse']
test_df['TotalDelinquency'] = test_df['NumberOfTimes90DaysLate'] + test_df['NumberOfTime30-59DaysPastDueNotWorse'] + test_df['NumberOfTime60-89DaysPastDueNotWorse']

train_df['DebtPerCreditLine'] = train_df['DebtRatio'] / (train_df['NumberOfOpenCreditLinesAndLoans'] + 1)
test_df['DebtPerCreditLine'] = test_df['DebtRatio'] / (test_df['NumberOfOpenCreditLinesAndLoans'] + 1)


train_df.isnull().sum()


## Standerization columns
stand_col = ['NumberOfTime30-59DaysPastDueNotWorse','age','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse']

## missing values columns
missing_val_col = ['MonthlyIncome','NumberOfDependents']




standerization_pipeline = Pipeline([
    ('standerization',StandardScaler())
])

missing_val_pipeline = Pipeline([
    ('age_missing_val',SimpleImputer(strategy='median')),
    ('age_standerization',StandardScaler())
])


col_preprocessing = ColumnTransformer(
    transformers=[
        ('missing',missing_val_pipeline,missing_val_col),
        ('standerization',standerization_pipeline,stand_col)
    ],
    remainder = 'passthrough'
)


test_df.isnull().sum()


## X and Y and splitting into train test split


train_df.drop(columns=['MonthlyIncomeBin'], inplace=True)

x = train_df.drop(columns=['SeriousDlqin2yrs'])
y = train_df[['SeriousDlqin2yrs']]




x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)


pipe_lr = Pipeline([
    ('preprocessing',col_preprocessing),
    ('model',LogisticRegression())
])


param_grid_lr = {
    'model__C': [0.001,0.01, 0.1, 1, 10],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear', 'saga'],
    'model__max_iter': [500, 1000,1500]
}


gridSearchLR = RandomizedSearchCV(
    estimator=pipe_lr,
    param_distributions=param_grid_lr,
    n_iter=30,        
    cv=5,        
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)


gridSearchLR.fit(x_train ,y_train.values.ravel())


gridSearchLR.score(x_test,y_test.values.ravel())


best_lr = gridSearchLR.best_estimator_
scores = cross_val_score(best_lr, x, y.values.ravel(), cv=5)
print(scores)
print(np.average(scores))


gridSearchLR.best_params_


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    pipe_lr,
    x, y.values.ravel(),
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

print("CV AUC scores:", scores)
print("Mean CV AUC:", scores.mean())


pipe_xgb = Pipeline([
    ('preprocessing', col_preprocessing),
    ('model', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',  
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    ))
])


param_grid_xgb = {
    'model__n_estimators': [300, 500, 800],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 4, 5, 6],
    'model__subsample': [0.7, 0.8, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 1.0],
    'model__gamma': [0, 0.1, 0.3],
    'model__min_child_weight': [1, 3, 5]
}


gridSearchXGB = RandomizedSearchCV(
    estimator=pipe_xgb,
    param_distributions=param_grid_xgb,
    n_iter=40,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)


gridSearchXGB.fit(x_train, y_train.values.ravel())


gridSearchXGB.score(x_test, y_test.values.ravel())


best_xg = gridSearchXGB.best_estimator_
scores = cross_val_score(best_xg, x, y.values.ravel(), cv=5)
print(scores)
print(np.average(scores))


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    pipe_xgb,
    x, y.values.ravel(),
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

print("CV AUC scores:", scores)
print("Mean CV AUC:", scores.mean())


gridSearchXGB.best_params_


best_lr_params=  {'model__solver': 'saga',
 'model__penalty': 'l1',
 'model__max_iter': 1500,
 'model__C': 1}


pipe_lr.set_params(**best_lr_params)


pipe_lr.fit(x, y.values.ravel())


y_test_pred = pipe_lr.predict(test_df)


y_test_pred_prob = pipe_lr.predict_proba(test_df)[:, 1]


test_df.head()


test_ID = test_df['Unnamed: 0']
X_test = test_df.drop(columns=['Unnamed: 0'])


submission = pd.DataFrame({
    "Id": test_ID,
    "Probability": pipe_lr.predict_proba(test_df)[:, 1]
})

submission.to_csv("submission.csv", index=False)
print("Submission file created correctly for Kaggle!")


# best_xgb_params = {'model__subsample': 0.8,
#  'model__n_estimators': 300,
#  'model__min_child_weight': 5,
#  'model__max_depth': 4,
#  'model__learning_rate': 0.05,
#  'model__gamma': 0.1,
#  'model__colsample_bytree': 0.7}
best_xgb_params = {
'model__subsample': 0.8,
 'model__n_estimators': 300,
 'model__min_child_weight': 5,
 'model__max_depth': 3,
 'model__learning_rate': 0.05,
 'model__gamma': 0.3,
 'model__colsample_bytree': 0.8}


pipe_xgb.set_params(**best_lr_params)


pipe_xgb.fit(x, y.values.ravel())


y_test_pred_prob = pipe_xgb.predict_proba(test_df)[:, 1]


submission = pd.DataFrame({
    "Id": test_ID,
    "Probability": pipe_xgb.predict_proba(test_df)[:, 1]
})

submission.to_csv("submission_xgb.csv", index=False)
print("Submission file created correctly for Kaggle!")




