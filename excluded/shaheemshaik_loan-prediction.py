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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

#data
train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

#showcasing data
train.info()
train.head()


#numerical features from data
#income vs loan-amount
train['income_to_loan_ratio']=train['annual_income']/(train['loan_amount']+1)
test['income_to_loan_ratio']=test['annual_income']/(test['loan_amount']+1)

#debt risk calculation
train['risk_score']=train['debt_to_income_ratio']*train['interest_rate']
test['risk_score']=test['debt_to_income_ratio']*test['interest_rate']

#credit and income combined feature
train['credit_income_ratio']=train['credit_score']/(train['annual_income']+1)
test['credit_income_ratio']=test['credit_score']/(test['annual_income']+1)

#loan to interest rate
train['loan_amount_times_interest']=train['loan_amount']*train['interest_rate']
test['loan_amount_times_interest']=test['loan_amount']*test['interest_rate']


#categorical features from data
#split grade_subgrade into grade and subgrade
train['grade']=train['grade_subgrade'].str[0]
train['subgrade']=train['grade_subgrade'].str[1].astype(int)
test['grade']=test['grade_subgrade'].str[0]
test['subgrade']=test['grade_subgrade'].str[1].astype(int)

#One-hot encode categorical columns
categorical_cols=['gender','marital_status','education_level','employment_status','loan_purpose','grade']
train=pd.get_dummies(train, columns=categorical_cols)
test=pd.get_dummies(test, columns=categorical_cols)


#drop unnecessary columns
X=train.drop(['id','loan_paid_back','grade_subgrade'],axis=1)
y=train['loan_paid_back']
X_test=test.drop(['id','grade_subgrade'],axis=1,errors='ignore')

#train-test-splitting of data
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


#ML Model - XGBoost
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

#fit data
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=50
)

#prediction
y_val_pred=model.predict_proba(X_val)[:, 1]

#ROC-AUC
auc=roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC-AUC: {auc:.4f}")


#feature importance graph
import matplotlib.pyplot as plt
xgb.plot_importance(model, max_num_features=20)
plt.show()


#submission
submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': model.predict_proba(X_test)[:, 1]
})
submission.to_csv('loanpredsubmission.csv', index=False)

