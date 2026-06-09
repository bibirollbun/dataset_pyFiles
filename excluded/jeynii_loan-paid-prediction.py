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


!pip install lightgbm



import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier





train_dataset = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_dataset = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample =pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


train_dataset.head()


def preprocessing_data(dataset):
    
    
    dataset['gender'] = dataset['gender'].map({
        'Male':1, 'male':1,
        'Female':0, 'female':0
        }).fillna(-1).astype(int)
    
    education = {
        'High School': 1,
        "Bachelor's": 2,
        "Master's": 3,
        'PhD': 4,
        'Other': 0
    }
    dataset['education_level'] = dataset['education_level'].map(education).fillna(0).astype(int)
    
    emp = {
        'Self-employed': 0,
        'Employed': 1,
        'Unemployed': 2,
        'Retired': 3,
        'Student': 4
    }
    dataset['employment_status'] = dataset['employment_status'].map(emp).fillna(-1).astype(int)
    
    encoded_df = dataset.copy()
    for col in encoded_df.select_dtypes(include='object').columns:
        encoded_df[col] = encoded_df[col].astype('category').cat.codes
    
    return encoded_df



def features(dataset):
    dataset['interest_to_credit'] = dataset['interest_rate'] / dataset['credit_score']
    dataset['interest_to_credit'] = (
        dataset['interest_to_credit']
        .replace([float('inf'), -float('inf')], np.nan)
        .fillna(dataset['interest_to_credit'].median())
    )

    dataset['loan_to_income'] = dataset['loan_amount'] / dataset['annual_income']
    dataset['loan_to_income'] = (
        dataset['loan_to_income']
        .replace([float('inf'), -float('inf')], np.nan)
        .fillna(dataset['loan_to_income'].median())
    )
    
    return dataset



encod_df=preprocessing_data(train_dataset)


encod_df=features(encod_df)


encod_df.head()




correlation = encod_df.corr()['loan_paid_back'].sort_values(ascending=False)
print(correlation)



sns.heatmap(encod_df.corr() ,annot=True,fmt=".2f")
plt.show()


def add_ratio_emp_status_to_grade_subgrade(dataset):
    combo_counts = dataset.groupby(['employment_status', 'grade_subgrade']).size().reset_index(name='count')
    grade_counts = dataset.groupby('grade_subgrade').size().reset_index(name='grade_count')
    combo = combo_counts.merge(grade_counts, on='grade_subgrade')
    combo['employment_grade_ratio'] = combo['count'] / combo['grade_count']
    dataset = dataset.merge(combo[['employment_status', 'grade_subgrade','employment_grade_ratio']],
              on=['employment_status', 'grade_subgrade'], how='left')
    return dataset



encod_df=add_ratio_emp_status_to_grade_subgrade(encod_df)



y=train_dataset[['loan_paid_back']]


x_train,x_test,y_train,y_test= train_test_split(encod_df,y,test_size=0.2,random_state=42)



feature_model = lgb.LGBMClassifier()
feature_model.fit(x_train, y_train.squeeze())
importance = pd.DataFrame({
    'feature': x_train.columns,
    'importance': feature_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))



X=encod_df[['id','credit_score','interest_to_credit','debt_to_income_ratio','interest_rate','annual_income','loan_to_income','loan_amount',"employment_grade_ratio"]]



scaler=StandardScaler()
X=scaler.fit_transform(X)


x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)


y_train.squeeze()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_reg = LogisticRegression(max_iter=500)
param_log = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
grid_log = GridSearchCV(log_reg, param_log, cv=skf, scoring='roc_auc', n_jobs=-1)
grid_log.fit(x_train, y_train.squeeze())
best_log = grid_log.best_estimator_



from lightgbm import LGBMClassifier


lgbm = LGBMClassifier(random_state=42)
param_lgbm = {
    'num_leaves': [15, 31],
    'max_depth': [5, 10],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}
grid_lgbm = GridSearchCV(lgbm, param_lgbm, cv=skf, scoring='roc_auc', n_jobs=-1)
grid_lgbm.fit(x_train, y_train.squeeze())
best_lgbm = grid_lgbm.best_estimator_



prob_log = best_log.predict_proba(x_test)[:, 1]
pred_log = (prob_log >= 0.5).astype(int)
prob_lgbm = best_lgbm.predict_proba(x_test)[:, 1]
pred_lgbm=(prob_lgbm >= 0.5).astype(int)

print("Ensemble Accuracy log:", accuracy_score(y_test, pred_log))
print("Ensemble ROC-AUC log:", roc_auc_score(y_test, pred_log))
print("Ensemble Accuracy lgbm:", accuracy_score(y_test, pred_lgbm))
print("Ensemble ROC-AUC lgbm:", roc_auc_score(y_test, pred_lgbm))



final_pred_prob = (prob_log + prob_lgbm) / 2

final_pred = (final_pred_prob >= 0.5).astype(int)

print("Ensemble Accuracy:", accuracy_score(y_test, final_pred))
print("Ensemble ROC-AUC:", roc_auc_score(y_test, final_pred_prob))




final_pred_prob = 0.4*prob_log + 0.6*prob_lgbm

final_pred = (final_pred_prob >= 0.5).astype(int)

print("Ensemble Accuracy:", accuracy_score(y_test, final_pred))
print("Ensemble ROC-AUC:", roc_auc_score(y_test, final_pred_prob))



from sklearn.ensemble import StackingClassifier

stack_model = StackingClassifier(
    estimators=[
        ('log_reg', best_log),
        ('lgbm', best_lgbm)
    ],
    final_estimator=LogisticRegression(),
    cv=skf,
    n_jobs=-1
)

stack_model.fit(x_train, y_train.squeeze())
stack_pred = stack_model.predict(x_test)

print("Stacking Accuracy:", accuracy_score(y_test, stack_pred))
print("Stacking ROC-AUC:", roc_auc_score(y_test, stack_model.predict_proba(x_test)[:,1]))



encoded_test=preprocessing_data(test_dataset)
encoded_test=features(encoded_test)
encoded_test=add_ratio_emp_status_to_grade_subgrade(encoded_test)



encoded_test.head()


test_final=encoded_test[['id','credit_score','interest_to_credit','debt_to_income_ratio','interest_rate','annual_income','loan_to_income','loan_amount',"employment_grade_ratio"]]

test_final = scaler.transform(test_final)



test_pred_final = stack_model.predict(test_final)

submission = pd.DataFrame({
    'id': test_dataset['id'],
    'loan_paid_back': test_pred_final.astype(int)
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")


submission.head()




