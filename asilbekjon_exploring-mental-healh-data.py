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


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from xgboost import XGBClassifier


train=pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
sample=pd.read_csv('/kaggle/input/playground-series-s4e11/sample_submission.csv')



train.info()


train.describe()


to_healthy= ['More Healthy','Pratham', 'Gender', 'BSc','3','1.0','Hormonal','Electrician','Mihir','M.Tech','Vegas',
               'Male','Indoor','Class 12','Yes']
train['Dietary Habits'] = train['Dietary Habits'].replace(to_healthy, 'Healthy')
moderate=['2','Less than Healthy','Less Healthy']
train['Dietary Habits'] = train['Dietary Habits'].replace(moderate, 'Moderate')
train['Dietary Habits'] = train['Dietary Habits'].replace(['No Healthy','No'], 'Unhealthy')


train['Depression'].value_counts()


# class distribution
clss_dis = train['Depression'].value_counts()/len(train)*100

#plotting
plt.figure(figsize=(7,7))
plt.pie(clss_dis, labels=clss_dis.index, autopct='%1.2f%%')

plt.title('Class Distribution')
plt.show()

sns.countplot(x='Depression', data=train)
plt.title('Class Distribution')
plt.show()



fig, ax = plt.subplots(2, 2, figsize=(20, 10))
sns.histplot(x='CGPA', bins=30, data=train, hue='Depression', ax=ax[0, 0])
sns.histplot(x='Work/Study Hours', data=train, bins=30, hue='Depression', ax=ax[0, 1])
sns.histplot(x='Age', bins=60, data=train, hue='Depression', ax=ax[1, 0])

sns.countplot(x='Job Satisfaction', data=train, hue='Depression', ax=ax[1, 1])

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(20, 10))
sns.countplot(x='Gender', hue='Depression', data=train, ax=ax[0,0])
sns.countplot(x='Working Professional or Student', hue='Depression', data=train, ax=ax[0,1])
sns.countplot(x='Dietary Habits', hue='Depression', data=train, ax=ax[1,1])
sns.countplot(x='Have you ever had suicidal thoughts ?', hue='Depression', data=train, ax=ax[1,0])
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(20, 10))
sns.countplot(x='Academic Pressure', hue='Depression', data=train, ax=ax[0,0])
sns.countplot(x='Work Pressure', hue='Depression', data=train, ax=ax[0,1])
sns.countplot(x='Study Satisfaction', hue='Depression', data=train, ax=ax[1,1])
sns.countplot(x='Job Satisfaction', hue='Depression', data=train, ax=ax[1,0])
plt.show()


df=train.drop(['Name','id','City','Profession','Sleep Duration','Degree'],axis=1)


df.ffill(inplace=True)


df.bfill(inplace=True)


df.isnull().sum()


df.Depression.value_counts()


from sklearn.utils import resample
maj_class = df[df['Depression'] == 0]
min_class = df[df['Depression'] == 1]
# Oversample the minority class
min_oversampled = resample(min_class,
                                replace=True,
                                n_samples=len(maj_class),
                                random_state=42)
# Combine the majority class with the oversampled minority class
df1 = pd.concat([maj_class, min_oversampled])


# Check the new class distribution after oversampling
data_dist = df1['Depression'].value_counts()
print("\nNew Class Distribution After Oversampling:\n", data_dist)


numeric=['Age','Academic Pressure','Work Pressure','CGPA','Study Satisfaction','Job Satisfaction','Work/Study Hours','Financial Stress']
categorical=['Gender','Working Professional or Student','Dietary Habits','Have you ever had suicidal thoughts ?','Family History of Mental Illness']
#numeric pipeline
num_pipeline=Pipeline([
    ('std_scaler',StandardScaler())
])
#categorical pipeline
cat_pipeline=Pipeline([
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])
#full pipeline
full_pipeline=ColumnTransformer([
    ('num',num_pipeline,numeric),
    ('cat',cat_pipeline,categorical)
])


X=df1.drop('Depression',axis=1)
y=df1['Depression'].copy()


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)


X_train_prepared=full_pipeline.fit_transform(X_train)
X_test_prepared=full_pipeline.transform(X_test)


RF=RandomForestClassifier()
RF.fit(X_train_prepared,y_train)
y_predict=RF.predict(X_test_prepared)
y_proba = RF.predict_proba(X_test_prepared)
y_proba_clipped=np.clip(y_proba, 1e-15, 1 - 1e-15)
logloss = log_loss(y_test,y_proba_clipped)
print(classification_report(y_test,y_predict))
print('Accuracy',accuracy_score(y_test,y_predict))
print('Log Loss:', logloss)


DT=DecisionTreeClassifier()
DT.fit(X_train_prepared,y_train)
y_predict_1=DT.predict(X_test_prepared)
y_proba_1 = DT.predict_proba(X_test_prepared)
y_proba_clipped_1=np.clip(y_proba_1, 1e-15, 1 - 1e-15)
logloss_1 = log_loss(y_test, y_proba_clipped_1)
print(classification_report(y_test,y_predict_1))
print('Accuracy',accuracy_score(y_test,y_predict_1))
print('Log Loss:', logloss_1)


GB=GradientBoostingClassifier()
GB.fit(X_train_prepared,y_train)
y_predict_2=GB.predict(X_test_prepared)
y_proba_2= GB.predict_proba(X_test_prepared)
y_proba_clipped_2=np.clip(y_proba_2, 1e-15, 1 - 1e-15)
logloss_2= log_loss(y_test, y_proba_clipped_2)
print(classification_report(y_test,y_predict_2))
print('Accuracy',accuracy_score(y_test,y_predict_2))
print('Log Loss:', logloss_2)


XGB=XGBClassifier(use_label_encoder=False, eval_metric='logloss')
XGB.fit(X_train_prepared,y_train)
y_predict_3 = XGB.predict(X_test_prepared)
y_proba_3= XGB.predict_proba(X_test_prepared)
y_proba_clipped_3=np.clip(y_proba_3, 1e-15, 1 - 1e-15)
logloss_3= log_loss(y_test, y_proba_clipped_3)
print(classification_report(y_test, y_predict_3))
print("Accuracy:", accuracy_score(y_test,y_predict_3))
print('Log Loss:', logloss_3)


test=pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')
test


test_id=test.id


test.drop(['Name','id','City','Profession','Sleep Duration','Degree'],axis=1,inplace=True)


test.info()


to_healthy1= ['More Healthy','Pratham', 'Gender', 'BSc','3','1.0','Hormonal','Electrician','Mihir','M.Tech','Vegas',
               'Male','Indoor','Class 12','Yes','Prachi','Mealy','Resistant','MCA','5 Healthy','Academic','Educational','Soham','Vivaan','Raghav','Naina','Kolkata']
test['Dietary Habits'] = test['Dietary Habits'].replace(to_healthy1, 'Healthy')
moderate1=['2','Less than Healthy','Less Healthy']
test['Dietary Habits'] = test['Dietary Habits'].replace(moderate1, 'Moderate')
test['Dietary Habits'] = test['Dietary Habits'].replace(['No Healthy','No','5 Unhealthy'], 'Unhealthy')


test.ffill(inplace=True)
test.bfill(inplace=True)


test.info()


test_prepared=full_pipeline.transform(test)
y_test_predict=RF.predict(test_prepared)
submission=pd.DataFrame({'id':test_id,'predict':y_test_predict})
submission.to_csv('Submission_mental_health1.csv',index=False)




