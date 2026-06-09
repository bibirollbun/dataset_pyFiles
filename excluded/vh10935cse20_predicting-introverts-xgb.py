import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.head(3)


train.info()


#Checking for Null Values
train.isna().sum()


train.describe().T


train.dtypes


test.head(3)


test.info()


#Checking for null values
test.isna().sum()


test.describe().T


numeric=['int64','float64']
cat=['object']
for i in train:
    if (train[i].dtypes in numeric) and (train[i].isna().sum()!=0):
        mean_num=train[i].mean()
        train[i]=train[i].fillna(mean_num)
    elif (train[i].dtypes in cat) and (train[i].isna().sum()!=0):
        mode_cat=train[i].mode()[0]
        train[i]=train[i].fillna(mode_cat)
    else:
        print("Column doesn't match the above criteria",i)


train.isna().sum()


numeric=['int64','float64']
cat=['object']
for i in test:
    if (test[i].dtypes in numeric) and (test[i].isna().sum()!=0):
        mean_num=test[i].mean()
        test[i]=test[i].fillna(mean_num)
    elif (test[i].dtypes in cat) and (test[i].isna().sum()!=0):
        mode_cat=test[i].mode()[0]
        test[i]=test[i].fillna(mode_cat)
    else:
        print("Column doesn't match the above criteria",i)


test.isna().sum()


cat_cols=['Stage_fear','Drained_after_socializing']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


for i in cat_cols:
    train[i]=le.fit_transform(train[i])
    test[i]=le.transform(test[i])


X=train.drop(columns=['id','Personality'])
y=train['Personality']

X_test=test.drop(columns=['id'])


train['Stage_fear']


model=LabelEncoder()
y=model.fit_transform(y)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)


xgb=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train,y_train)


y_pred=xgb.predict(X_val)


accuracy_score(y_val, y_pred)


y_test_pred=xgb.predict(X_test)


y_test_labels = model.inverse_transform(y_test_pred)

submission = test[['id']].copy()
submission['Personality'] = y_test_labels

submission.to_csv('submission.csv', index=False)


submission

