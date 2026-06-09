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


data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.metrics import average_precision_score as aps
import seaborn as sns
from sklearn.ensemble import StackingClassifier,RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression



print("null in data",data.isna().mean()*100)
print()
print("data duplicate",data.duplicated().sum())
print()
print('null in test',test.isna().mean()*100)
print()
print('duplicates in test',test.duplicated().sum())


data.describe()


data.head()
# data.shape


X=data.drop(columns=['id','Fertilizer Name'])
test_id = test['id']
test = test.drop(columns=['id'])
y = data['Fertilizer Name']
le = LabelEncoder()
y = le.fit_transform(y)
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include='object').columns



sns.histplot(y,kde=True)



X[num_cols].corr()


data.head()


plt.figure(figsize=(20,8))
plt.subplot(231)
data['Crop Type'].value_counts().plot(kind='pie')

plt.figure(figsize=(20,8))
plt.subplot(232)
data['Soil Type'].value_counts().plot(kind='pie')

plt.figure(figsize=(20,8))
plt.subplot(233)
data['Fertilizer Name'].value_counts().plot(kind='pie')


for i in num_cols:
    plt.figure(figsize=(15,8))
    plt.subplot(231)
    sns.kdeplot(data[i])
    plt.subplot(232)
    sns.histplot(data[i],bins=50)
   


for i in num_cols:
    plt.figure(figsize=(15,8))
    plt.subplot(221)
    sns.kdeplot(data[i])
    plt.subplot(222)
    sns.histplot(data[i],bins=100)


plt.figure(figsize=(15,8))
sns.countplot(x=data['Crop Type'],hue=data['Soil Type'])


plt.figure(figsize=(15,8))
sns.countplot(x=data['Crop Type'],hue=data['Fertilizer Name'])

plt.figure(figsize=(15,8))
sns.countplot(x=data['Soil Type'],hue=data['Fertilizer Name'])



X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=20)
import lightgbm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.preprocessing import label_binarize 
from sklearn.metrics import average_precision_score
from sklearn.impute import SimpleImputer

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='mean')),
    ('scale',StandardScaler())
])

cat_pipe = Pipeline([
    ('ohe',OneHotEncoder(handle_unknown='ignore'))
    
])

preprocess = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])

model_lgbm = Pipeline([
    ('preprocess',preprocess),
    ('algo',LGBMClassifier(learning_rate=0.42,max_depth=5,verbosity=-1))
])

model_xgb = Pipeline([
    ('preprocess',preprocess),
    ('algo',XGBClassifier(learning_rate=0.5,max_depth=12,reg_lambda=10,alpha=6,subsample=0.8,colsample_bytree=0.3,min_child_weight=2))
])

estimators = [
    ('rf',RandomForestClassifier(n_estimators=350,max_depth=8,n_jobs=-1)),
    ('lr',LogisticRegression(max_iter=550,n_jobs=-1))]

stratified = StratifiedKFold(n_splits=5,shuffle=True)

stack = Pipeline([
    ('preprocess',preprocess),
    ('algo',StackingClassifier(estimators=estimators,cv=stratified))
])


def mapk(y_true,y_pred,k=3):
    score=0.0
    for true,pred in zip(y_true,y_pred):
         for i in range(k):
            if pred[i] == true:
                score+= 1.0 / (i + 1)
                break
    return score / len(y_true)


model_xgb.fit(X_train,y_train)
y_valid_probs = model_xgb.predict_proba(X_valid)
top3_preds = np.argsort(y_valid_probs,axis=1)[:,-3:][:,::-1]
map3_score = mapk(y_valid,top3_preds,k=3)
map3_score


y_train_prob=model_xgb.predict_proba(X_train)
top3_train_preds = np.argsort(y_train_prob,axis=1)[:,-3:][:,::-1]
map3_train_score = mapk(y_train,top3_train_preds,k=3)
map3_train_score


# stack.fit(X_train,y_train)
# y_valid_probs = stack.predict_proba(X_valid)
# top3_preds = np.argsort(y_valid_probs,axis=1)[:,-3:][:,::-1]
# map3_score = mapk(y_valid,top3_preds,k=3)
# map3_score


# y_train_prob=stack.predict_proba(X_train)
# top3_train_preds = np.argsort(y_train_prob,axis=1)[:,-3:][:,::-1]
# map3_train_score = mapk(y_train,top3_train_preds,k=3)
# map3_train_score


test_preds = model_xgb.predict_proba(test)
top3_test_preds = np.argsort(test_preds,axis=1)[:,-3:][:,::-1]
fertilizer_preds = le.inverse_transform(top3_test_preds.ravel()).reshape(top3_test_preds.shape)


# from sklearn.model_selection import RandomizedSearchCV
# params= {
#     'algo__estimators__rf__n_estimators':[600,700,750,800],
#     'algo__estimators__rf__max_depth':[15,18,22],
#     'algo__estimators__lr__max_iter':[500,700,850,900]
# }
# search = RandomizedSearchCV(stack,param_distributions=params,cv=5,scoring='r2')
# search.fit(X_train,y_train)
# print('best paramters',search.best_params_)
# print('best score',search.best_score_)


output = pd.DataFrame({'id':test_id,'Fertilizer Name':[' '.join(row) for row in fertilizer_preds.astype(str)]})
output.to_csv('submission.csv',index=False)
submit = pd.read_csv('/kaggle/working/submission.csv')


submit.head()

