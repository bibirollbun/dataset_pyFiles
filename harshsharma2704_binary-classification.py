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
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


mapping = {'yes':1,'no':0}
edu_order = {'unknown':0,'primary':1,'secondary':2,'tertiary':3}
month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

df['default'] = df['default'].map(mapping)
df['housing'] = df['housing'].map(mapping)
df['loan'] = df['loan'].map(mapping)
df['education'] = df['education'].map(edu_order)
df['month'] = df['month'].map(month_map)
df['sin_month'] = np.sin(2*np.pi*df['month']/12)
df['cos_month'] = np.cos(2*np.pi*df['month']/12)
df['was_contacted'] = df['pdays'].apply(lambda x:0 if x==-1 else 1)

test['default'] = test['default'].map(mapping)
test['housing'] = test['housing'].map(mapping)
test['loan'] = test['loan'].map(mapping)
test['education'] = test['education'].map(edu_order)
test['month'] = test['month'].map(month_map)
test['sin_month'] = np.sin(2*np.pi*test['month']/12)
test['cos_month'] = np.cos(2*np.pi*test['month']/12)
test['was_contacted'] = test['pdays'].apply(lambda x:0 if x==-1 else 1)



X = df.drop(columns=['y','id','month','pdays'])
y = df['y']


X.isna().mean()*100


num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(include=['object','category']).columns


df.isna().sum()


X.isna().sum()


for i in num_cols:
    lower = X[i].quantile(0.1)
    upper = X[i].quantile(0.9)
    X[i].clip(lower = lower,upper = upper)

for i in num_cols:
    lower_test = test[i].quantile(0.1)
    upper_test = test[i].quantile(0.9)
    test[i].clip(lower=lower_test,upper = upper_test)


for  i in num_cols:
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    sns.boxplot(x=X[i])
    plt.subplot(122)
    sns.kdeplot(x=X[i])


for i in num_cols:
    print(i,X[i].value_counts().head(10))
    print('-'*30)


for i in cat_cols:
    plt.figure()
    sns.countplot(x=X[i])


# sns.scatterplot(x='age',y='balance',hue='housing',data=X)
sns.scatterplot(x='age',y='pdays',hue='y',data=df)


from sklearn.metrics import roc_auc_score,classification_report,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,RobustScaler,StandardScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier


X_train,X_valid,y_train,y_valid= train_test_split(X,y,test_size=0.2,random_state=20)

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='mean')),
    # ('scale',PowerTransformer(method='yeo-johnson')),
    ('scaler',RobustScaler())
])

cat_pipe = Pipeline([
    ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
])

preprocessing = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])


model_lr = Pipeline([
    ('pre',preprocessing),
    ('algo',LogisticRegression(max_iter=850,fit_intercept=True,class_weight='balanced',solver='saga'))
])
model_lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_valid)
y_pred_train = model_lr.predict(X_train)
print('validation score',roc_auc_score(y_valid,y_pred_lr))
print('training  score',roc_auc_score(y_train,y_pred_train))


model_gb = Pipeline([
    ('pre',preprocessing),
    ('algo',HistGradientBoostingClassifier(max_iter=1050,learning_rate=0.01,max_depth=5))
])

model_gb.fit(X_train,y_train)
y_pred = model_gb.predict(X_valid)
y_pred_train = model_gb.predict(X_train)
print('validation score',roc_auc_score(y_valid,y_pred))
print('training score',roc_auc_score(y_train,y_pred_train))


print(classification_report(y_valid,y_pred))


# from sklearn.model_selection import GridSearchCV

# params = {
#     'algo__max_iter':[800,850,900,950,1000],
#     'algo__learning_rate':[0.01,0.02,0.1,0.2],
#     'algo__max_depth':[8,10,12,15]
# }

# search = GridSearchCV(param_grid=params,cv=5,scoring='roc_auc',estimator=model_gb,n_jobs=-1)
# search.fit(X_train,y_train)

# print('best parameters',search.best_params_)
# print('best score',search.best_score_)


model_lgb = Pipeline([
    ('pre',preprocessing),
    ('algo',LGBMClassifier(n_estimators=820,learning_rate=0.01,max_depth=15,verbose=-1))
])

model_lgb.fit(X_train,y_train)
y_pred_lgb = model_lgb.predict(X_valid)
y_pred_train = model_lgb.predict(X_train)
print('validation score',roc_auc_score(y_valid,y_pred_lgb))
print('training score',roc_auc_score(y_train,y_pred_train))


print(classification_report(y_valid,y_pred_lgb))


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
estimators = [
    ('gbc',SVC(C=0.6,kernel='linear',degree=6,max_iter=1650)),
    ('lgbm',LGBMClassifier(n_estimators=1020,learning_rate=0.15,max_depth=8,verbose=-1))
]
 
model_vote  = Pipeline([
    ('preprocess',preprocessing),
    ('algo',VotingClassifier(estimators=estimators,n_jobs=-1,voting='hard'))
])

model_vote.fit(X_train,y_train)
y_pred_vote = model_vote.predict(X_valid)
y_pred_train =  model_vote.predict(X_train)
print('validation score',roc_auc_score(y_valid,y_pred_vote))
print('training score',roc_auc_score(y_train,y_pred_train))


print(classification_report(y_valid,y_pred))


model_xgb = Pipeline([
    ('pre',preprocessing),
    ('algo',XGBClassifier(learning_rate=0.12,max_depth=8,gamma=0.35,n_estimators=1450,subsample=0.85,min_child_weight=5))
])

model_xgb.fit(X_train,y_train)
y_pred = model_xgb.predict(X_valid)
y_pred_train = model_xgb.predict(X_train)
print('validation score',roc_auc_score(y_valid,y_pred))
print('training score',roc_auc_score(y_train,y_pred_train))


print(classification_report(y_valid,y_pred))


y_pred_test = model_xgb.predict(test)


submit = pd.DataFrame({'id':test['id'],'y':y_pred_test})
submit.to_csv('submission.csv',index=False)

