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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import OneHotEncoder,RobustScaler,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


ed_level = {'No formal':0,'Highschool':1,'Graduate':2,'Postgraduate':3}

df['education_level'] = df['education_level'].map(ed_level)
test['education_level'] = test['education_level'].map(ed_level)

smoke = {'Never':0,'Current':2,'Former':1}

df['smoking_status'] = df['smoking_status'].map(smoke)
test['smoking_status'] = test['smoking_status'].map(smoke)


income = {'Low':0,'Middle':1,'Lower-Middle':2,'Upper-Middle':3,'High':4}

df['income_level'] = df['income_level'].map(income)
test['income_level'] = test['income_level'].map(income)

employ = {'Employed':0,'Retired':1,'Student':2,'Unemployed':3}

df['employment_status'] = df['employment_status'].map(employ)
test['employment_status'] = test['employment_status'].map(employ)


df['MAP'] = df['diastolic_bp'] + 0.333*(df['systolic_bp']-df['diastolic_bp'])
test['MAP'] = test['diastolic_bp'] + 0.333*(test['systolic_bp']-test['diastolic_bp'])

df['chol_hdl_ratio'] = df['cholesterol_total']/df['hdl_cholesterol']
test['chol_hdl_ratio'] = test['cholesterol_total']/test['hdl_cholesterol']

df['ldl_hdl_ratio'] = df['ldl_cholesterol']/df['hdl_cholesterol']
test['ldl_hdl_ratio'] = test['ldl_cholesterol']/test['hdl_cholesterol']

df['tg_hdl_ratio'] = df['triglycerides']/df['hdl_cholesterol']
test['tg_hdl_ratio'] = test['triglycerides']/test['hdl_cholesterol']

df['age_bmi'] = df['age']*df['bmi']
test['age_bmi'] = test['age']*test['bmi']

df['age_waist_hip'] = df['age']*df['waist_to_hip_ratio']
test['age_waist_hip'] = test['age']*test['waist_to_hip_ratio']




to_log = [
    'alcohol_consumption_per_week',
    'ldl_hdl_ratio',
    'chol_hdl_ratio',
    'tg_hdl_ratio',
    'physical_activity_minutes_per_week'
]

for col in to_log:
    df[col] = np.log1p(df[col])
    test[col] = np.log1p(df[col])


X = df.drop(columns=(['diagnosed_diabetes','id']))
y =df['diagnosed_diabetes']


num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols =X.select_dtypes(include=['category','object']).columns


for col in num_cols:
    X[col] = df[col].clip(df[col].quantile(0.01),df[col].quantile(0.99))
    test[col] = test[col].clip(test[col].quantile(0.01),test[col].quantile(0.99))
    


# Run this in your training environment
bounds_dict = {}
for col in num_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    bounds_dict[col] = {"lower": lower, "upper": upper}

print(bounds_dict)


num_pipe = Pipeline([
    ('scale',StandardScaler())
])

cat_pipe = Pipeline([
    ('encode',OneHotEncoder(handle_unknown='ignore'))
])

preprocess =ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])


X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=4)
from sklearn.metrics import roc_auc_score


# !pip install catboost
from catboost import CatBoostClassifier


model_cat = Pipeline([
    ('pre',preprocess),
    ('algo',CatBoostClassifier(loss_function='Logloss',eval_metric='AUC',iterations=1500,learning_rate=0.03
                               ,depth=12,l2_leaf_reg=6,bootstrap_type='Bayesian',min_data_in_leaf=50,verbose=0))
])

model_cat.fit(X_train,y_train)
y_pred_prob = model_cat.predict_proba(X_valid)[:,1]
.

y_pred_train_prob = model_cat.predict_proba(X_train)[:,1]
print('validation score',roc_auc_score(y_valid,y_pred_prob))
print('training score',roc_auc_score(y_train,y_pred_train_prob))


from sklearn.metrics import roc_curve
y_pred_prob = model_cat.predict_proba(X_valid)[:,1]
fpr,tpr,thresh = roc_curve(y_valid,y_pred_prob)
plt.plot(fpr,tpr,color='blue',label='Roc Auc curve')
plt.plot([0,1],[0,1],color='red',linestyle='--',label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')


from sklearn.metrics import roc_curve
# y_pred_prob = model_xgb.predict_proba(X_valid)[:,1]
# fpr,tpr,thresh = roc_curve(y_valid,y_pred_prob)
# plt.plot(fpr,tpr,color='blue',label='Roc Auc curve')
# plt.plot([0,1],[0,1],color='red',linestyle='--',label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')


test_pred = model_cat.predict_proba(test)[:,1]
submit = pd.DataFrame({'id':test['id'],'diagnosed_diabetes':test_pred})
submit.to_csv('submission.csv',index=False)


import pickle

pickle_model_path = '/kaggle/working/model.pkl'
with open(pickle_model_path,'wb')as f:
    pickle.dump(model_cat,f)




