import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv',index_col=0)
train.head()


test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv',index_col=0)
test.head()


train.info()


train.describe()


plt.figure(figsize=(12,10))
sns.barplot(data=train,x='job',y='balance',palette='rainbow')
plt.xticks(rotation=30)


sns.distplot(train['age'])


plt.figure(figsize=(10,8))
sns.distplot(train['balance'],color='red')


sns.distplot(train['duration'],color='green')


sns.countplot(data=train,x='y')


plt.figure(figsize=(12,10))
sns.countplot(data=train,x='job')
plt.xticks(rotation=30)


sns.countplot(data=train,x='marital')


sns.countplot(data=train,x='education')


sns.countplot(data=train,x='default',hue='y')


sns.countplot(data=train,x='housing',hue='y')


sns.countplot(data=train,x='loan',hue='y')


sns.countplot(data=train,x='contact')


sns.countplot(data=train,x='poutcome')


sns.heatmap(train.corr(numeric_only=True),annot=True,cmap='coolwarm')


cat_columns=[col for col in train.columns if train[col].dtype=='object']


cat_columns


from sklearn.preprocessing import OneHotEncoder


encoder=OneHotEncoder(drop='first',sparse_output=False)
encoded_train=encoder.fit_transform(train[cat_columns])
encoded_test=encoder.transform(test[cat_columns])


encoded_cols=encoder.get_feature_names_out()


encoded_train=pd.DataFrame(encoded_train,columns=encoded_cols,index=train.index)
encoded_test=pd.DataFrame(encoded_test,columns=encoded_cols,index=test.index)


encoded_train=pd.concat([train,encoded_train],axis=1)
encoded_test=pd.concat([test,encoded_test],axis=1)
encoded_train.drop(cat_columns,axis=1,inplace=True)
encoded_test.drop(cat_columns,axis=1,inplace=True)
encoded_train.head()


from sklearn.model_selection import train_test_split


X=encoded_train.drop('y',axis=1)
y=encoded_train['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


from sklearn.preprocessing import StandardScaler


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression


log_model=LogisticRegression()
log_model.fit(X_train,y_train)
pred=log_model.predict(X_test)
pred_probs=log_model.predict_proba(X_test)[:,1]


from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


roc_auc_score(y_test,pred_probs)


from sklearn.tree import DecisionTreeClassifier


dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred=dtree.predict(X_test)
pred_probs=dtree.predict_proba(X_test)[:,1]


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


roc_auc_score(y_test,pred_probs)


from sklearn.ensemble import RandomForestClassifier


rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
pred=rfc.predict(X_test)
pred_probs=rfc.predict_proba(X_test)[:,1]


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


roc_auc_score(y_test,pred_probs)


from xgboost import XGBClassifier


xgb=XGBClassifier()
xgb.fit(X_train,y_train)
pred=xgb.predict(X_test)
pred_probs=xgb.predict_proba(X_test)[:,1]


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


roc_auc_score(y_test,pred_probs)


from lightgbm import LGBMClassifier


lgbm=LGBMClassifier()
lgbm.fit(X_train,y_train)
pred=lgbm.predict(X_test)
pred_probs=lgbm.predict_proba(X_test)[:,1]


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


roc_auc_score(y_test,pred_probs)


xgb=XGBClassifier(n_estimators=300)
xgb.fit(X_train,y_train)
pred=xgb.predict(X_test)
pred_probs=xgb.predict_proba(X_test)[:,1]


print(classification_report(y_test,pred))


sns.heatmap(confusion_matrix(y_test,pred),cmap='viridis',annot=True)


fpr, tpr, thresholds = roc_curve(y_test,pred_probs)
plt.plot([0,1],[0,1],'--')
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')


roc_auc_score(y_test,pred_probs)


scaler=StandardScaler()
train_X=scaler.fit_transform(X)
test_X=scaler.transform(encoded_test)


xgb=XGBClassifier(n_estimators=300)
xgb.fit(train_X,y)
pred_probs=xgb.predict_proba(test_X)[:,1]


final=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


final['y']=pred_probs


final.to_csv('final.csv',index=False)

