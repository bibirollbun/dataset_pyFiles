import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random 
import math
import os 
import socket
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import xgboost as xgb



import time

from sklearn import preprocessing,metrics
from sklearn.metrics import classification_report,mean_squared_error,confusion_matrix,f1_score,accuracy_score,precision_score,recall_score,auc,roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
from collections import Counter


df_ie = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_ie_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df_ie.isnull().sum()



df_ie_test.isnull().sum()


plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
sns.histplot(data=df_ie,x='Time_spent_Alone',kde=True)
plt.xlabel('Time_spent_Alone')


plt.subplot(2,2,2)
sns.histplot(data=df_ie,x='Social_event_attendance',kde=True)
plt.xlabel('Social_event_attendance')


plt.subplot(2,2,3)
sns.histplot(data=df_ie,x='Friends_circle_size',kde=True)
plt.xlabel('Friends_circle_size')


plt.subplot(2,2,4)
sns.histplot(data=df_ie,x='Post_frequency',kde=True)
plt.xlabel('Post_frequency')


plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
sns.histplot(data=df_ie_test,x='Time_spent_Alone',kde=True)
plt.xlabel('Time_spent_Alone')


plt.subplot(2,2,2)
sns.histplot(data=df_ie_test,x='Social_event_attendance',kde=True)
plt.xlabel('Social_event_attendance')


plt.subplot(2,2,3)
sns.histplot(data=df_ie_test,x='Friends_circle_size',kde=True)
plt.xlabel('Friends_circle_size')


plt.subplot(2,2,4)
sns.histplot(data=df_ie_test,x='Post_frequency',kde=True)
plt.xlabel('Post_frequency')


plt.figure(figsize=(8,6))

plt.subplot(1,1,1)
sns.countplot(data=df_ie, x='Personality')
plt.title('Distribution of Personality')
plt.show()

plt.subplot(1,2,2)
sns.scatterplot(data=df_ie, x='Time_spent_Alone', y='Friends_circle_size', hue='Personality')
plt.title('Time Spent Alone vs Friends Circle Size by Personality')
plt.show()


df_ie_test.head()


df_ie['Time_spent_Alone'] = df_ie['Time_spent_Alone'].fillna(df_ie['Time_spent_Alone'].median())
df_ie['Social_event_attendance'] = df_ie['Social_event_attendance'].fillna(df_ie['Social_event_attendance'].median())
df_ie['Friends_circle_size'] = df_ie['Friends_circle_size'].fillna(df_ie['Friends_circle_size'].median())
df_ie['Post_frequency'] = df_ie['Post_frequency'].fillna(df_ie['Post_frequency'].median())



df_ie['Going_outside'] = df_ie['Going_outside'].fillna(df_ie['Going_outside'].median())



cat_cols = ['Stage_fear', 'Drained_after_socializing']
df_ie[cat_cols] = df_ie[cat_cols].apply(LabelEncoder().fit_transform)
#df_ie = LabelEncoder().fit_transform(df_ie)


df_ie.head()


df_ie_test.head()


df_ie_test['Time_spent_Alone'] = df_ie_test['Time_spent_Alone'].fillna(df_ie_test['Time_spent_Alone'].median())
df_ie_test['Post_frequency'] = df_ie_test['Post_frequency'].fillna(df_ie_test['Post_frequency'].median())




df_ie_test['Social_event_attendance'] = df_ie_test['Social_event_attendance'].fillna(df_ie_test['Social_event_attendance'].median())
df_ie_test['Going_outside'] = df_ie_test['Going_outside'].fillna(df_ie_test['Going_outside'].median())
df_ie_test['Friends_circle_size'] = df_ie_test['Friends_circle_size'].fillna(df_ie_test['Friends_circle_size'].median())


df_ie_test.isnull().sum()


le_stage_fear = LabelEncoder()
le_drained = LabelEncoder()


df_ie_test['Stage_fear'] = df_ie_test['Stage_fear'].fillna(df_ie_test['Stage_fear'].mode()[0])
df_ie_test['Drained_after_socializing'] = df_ie_test['Drained_after_socializing'].fillna(df_ie_test['Drained_after_socializing'].mode()[0])

df_ie_test['Stage_fear'] = le_stage_fear.fit(['No', 'Yes']).transform(df_ie_test['Stage_fear'])
df_ie_test['Drained_after_socializing'] = le_drained.fit(['No', 'Yes']).transform(df_ie_test['Drained_after_socializing'])


X = df_ie.drop(columns=['id','Personality'])
y = df_ie['Personality']


y.head()


y = y.map({'Extrovert': 1, 'Introvert': 0})



y.head()


X.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("LGBM Accuracy:", accuracy)


model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=200,       # Set a fixed number of trees
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


model.fit(X_train, y_train)



y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))


X_test_1 = df_ie_test.drop(columns=['id'])



predictions = model.predict(X_test_1)



labels = pd.Series(predictions).map({1: 'Extrovert', 0: 'Introvert'})



labels


submission = pd.DataFrame({
    'id': df_ie_test['id'],
    'Personality': labels
})
submission.to_csv('submission_xgboost.csv', index=False)


submission.head()




