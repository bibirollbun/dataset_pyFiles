import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import seaborn.objects as so


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
train.drop('id', axis=1 , inplace=True )#


print("train：")
#print(train.isnull().sum()) 
#print(train.columns)
print(train.shape)
#print(train.info())
print(train.nunique())
#print(train.select_dtypes(include='object').describe())
#train.describe().round(2)
#print("test：")
#print(test.isnull().sum()) 
#duplicate_rows=train[train.duplicated()]
categorical_cols = ['gender', 'marital_status' , 'education_level', 'employment_status','loan_purpose','grade_subgrade']
#numerical_cols = train.select_dtypes(exclude=['object'])
numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']


correlation = train.select_dtypes(exclude='object').corr()
correlation.style.background_gradient(cmap='viridis')


for feature in categorical_cols:
    (
        so.Plot(train, y=feature  , alpha="loan_paid_back")
        .add(so.Bar(), so.Hist(), so.Dodge())
        .label(x=None, y=feature, title=f"Count of {feature}" )
        .layout(size=(8, 5))
        #.scale(x=so.Continuous().label(rotation=90))
        .show()
    )   


X = train.drop('loan_paid_back', axis=1 )#, inplace=True
y = train['loan_paid_back']

encoder = OrdinalEncoder()
scaler = StandardScaler()

X_cat = encoder.fit_transform(X[categorical_cols])
#for col in enumerate(categorical_cols):
#    X[col] = encoder.fit_transform(X[col])
#X_cat=pd.DataFrame(X[col],columns=['gender', 'marital_status' , 'education_level', 'employment_status','loan_purpose','grade_subgrade'])

X_num = scaler.fit_transform(X[numerical_cols])

X_processed = np.hstack([X_cat, X_num])

X_test_cat = encoder.transform(test[categorical_cols])
X_test_num = scaler.transform(test[numerical_cols])

X_test_processed = np.hstack([X_test_cat, X_test_num])

X_processed = pd.DataFrame(X_processed, index=X.index, columns=X.columns)
#x_train, x_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


params = {
           'n_estimators': 1000,
           'max_depth': 10, 
           'min_samples_leaf': 33, 
           'subsample': 0.81, 
           'learning_rate': 0.0064, 
           'lambda_l1': 1.2991459277687692e-05, 
           'lambda_l2': 0.0007304768170358017,
           'objective': 'binary',  # Changed to binary
           'metric': 'auc',  # Changed to binary error
           'boosting_type': 'gbdt',
           'random_state': 42,
           'verbose': -1
       }

oof_preds = np.zeros(len(X_processed))
test_preds = np.zeros(len(X_test_processed))

n=5
skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)

model=LGBMClassifier(**params)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y), 1):
    print(f'--- Fold {fold}/{n} ---')
    
    X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
       
    model.fit(X_train, y_train)
    
    y_pred_val = model.predict_proba(X_val)[:,1]
    fold_test_preds = model.predict_proba(X_test_processed)[:,1]
    
    oof_preds[val_idx] = y_pred_val
    
    fold_score = roc_auc_score(y_val, y_pred_val)
    
    print(f'Fold {fold} AUC: {fold_score:.4f}')
    
    test_preds += fold_test_preds / n 

overall_auc = roc_auc_score(y, oof_preds)
print('#'*10)
print(f'Overall OOF AUC: {overall_auc:.4f}')
print('#'*10)


sub.loan_paid_back=test_preds
sub.to_csv("submission.csv", index=False)
sub.head()




