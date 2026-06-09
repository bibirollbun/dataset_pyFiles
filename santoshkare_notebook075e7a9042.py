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
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn import svm

from warnings import filterwarnings
filterwarnings('ignore')


train_path = '/kaggle/input/beat-xgboost-with-neural-nets-smoker-status/train.csv'
test_path = '/kaggle/input/beat-xgboost-with-neural-nets-smoker-status/test.csv'
sample_path = '/kaggle/input/beat-xgboost-with-neural-nets-smoker-status/sample_submission.csv'


df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample = pd.read_csv(sample_path)


print(df_train.shape)
print(df_test.shape)
print(df_sample.shape)



df_train.head()


df_test.head()


df_train.info()


df_train.describe()


df_train["smoking"].value_counts()


df_train["smoking"].value_counts().plot(kind = 'pie',autopct='%1.1f%%')


df_train.isnull().sum()


right_skewed={}
left_skewed={}
normal={}

for feature in df_train.columns:
    s=stats.skew(df_train[feature])
    if(s > 0):
        right_skewed[feature]=round(s,2)
    elif(s < 0):
        left_skewed[feature]=round(s,2)
    else:
        normal[feature]=round(s,2)


right_skewed


left_skewed


normal


df_train.drop(columns="id",axis=1,inplace=True)


df_train.columns


len(df_train.columns)


plt.figure(figsize=(16,12))
for feature, i in zip(df_train.columns,range(1,24)):
    plt.subplot(5,5,i)
    df_train.boxplot(column=feature)
    plt.tight_layout() 


plt.figure(figsize=(12,8))
sns.heatmap(df_train.corr(),annot=True)


plt.figure(figsize=(25,20))
for feature,i in zip(df_train.columns,range(1,24)):
    plt.subplot(5,5,i)
    sns.distplot(df_train[feature],kde=True)
    plt.tight_layout()


x=df_train.drop("smoking",axis=1)
y=df_train["smoking"]


x.shape,y.shape



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state = 56,
                                                 stratify=y)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


model_lr=LogisticRegression(random_state=11,n_jobs=-1)


model_lr.fit(x_train,y_train)


y_train_predict_lr=model_lr.predict(x_train)

y_test_predict_lr=model_lr.predict(x_test)


y_test_prob_predict_lr = model_lr.predict_proba(x_test)[:,1]


roc_auc_score_lr = roc_auc_score(y_test,y_test_prob_predict_lr)


print(classification_report(y_test,y_test_predict_lr))


model_rfc = RandomForestClassifier(n_jobs=-1,random_state=11,class_weight='balanced')
model_rfc.fit(x_train,y_train)



y_train_predict_rfc=model_rfc.predict(x_train)


y_test_predict_rfc=model_rfc.predict(x_test)


y_test_prob_predict_rfc = model_rfc.predict_proba(x_test)[:,1]


roc_auc_score_rfc = roc_auc_score(y_test,y_test_prob_predict_rfc)


print(classification_report(y_test,y_test_predict_rfc))


import xgboost


model_xgb = xgboost.XGBClassifier()
model_xgb.fit(x_train,y_train)


y_train_predict_xgb=model_xgb.predict(x_train)


y_test_predict_xgb=model_xgb.predict(x_test)


y_test_prob_predict_xgb = model_xgb.predict_proba(x_test)[:,1]
roc_auc_score_xgb = roc_auc_score(y_test,y_test_prob_predict_xgb)


print(classification_report(y_test,y_test_predict_xgb))


from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import make_scorer, roc_auc_score


kf = KFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scorer = make_scorer(roc_auc_score)


n_estimators = [25,50,75,100]
scores_list = []
for n in n_estimators:
    model_xgb.set_params(n_estimators = n)
    scores = cross_val_score(estimator=model_xgb,X=x_train,y=y_train, 
                             scoring=roc_auc_scorer, cv=kf)
    
    
    mean_score = scores.mean()
    scores_list.append(mean_score)


plt.figure(figsize=(12,6))
plt.plot(n_estimators,scores_list, marker='o')
plt.title('ROC-AUC vs n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('ROC-AUC Score')
plt.grid(True)
plt.show()


model_xgb_cv = xgboost.XGBClassifier(n_estimators = 200,max_depth = 5,learning_rate = 0.2,
                                     subsample = 1.0,colsample_bytree=1.0)
model_xgb_cv.fit(x_train,y_train)

y_train_predict_xgb_cv=model_xgb_cv.predict(x_train)
y_test_predict_xgb_cv=model_xgb_cv.predict(x_test)


y_test_prob_predict_xgb_cv = model_xgb_cv.predict_proba(x_test)[:,1]
roc_auc_score_xgb_cv = roc_auc_score(y_test,y_test_prob_predict_xgb_cv)


roc_auc_score_xgb_cv


print(f"roc_auc score logistic regression : {roc_auc_score_lr}")
print(f"roc_auc score random forest : {roc_auc_score_rfc}")
print(f"roc_auc score xgboost : {roc_auc_score_xgb}")
print(f"roc_auc score xgboost cv : {roc_auc_score_xgb_cv}")


df_test.set_index(keys='id',drop = True,inplace = True)



df_test_scaled = sc.transform(df_test)


df_test_predict = model_xgb_cv.predict(df_test_scaled)
df_test_prob_predict = model_xgb_cv.predict_proba(df_test_scaled)[:,1]


df_test_prob_predict


df_test.index.values


df_results = pd.DataFrame({'id' : df_test.index.values,'smoking':df_test_prob_predict})
df_results

