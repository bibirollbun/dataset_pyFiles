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


import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df.head()


train_df = pd.get_dummies(train_df,columns = ['Stage_fear','Drained_after_socializing'],drop_first = True)


train_df.head()


test_df = pd.get_dummies(test_df,columns = ['Stage_fear','Drained_after_socializing'],drop_first = True)


test_df.head()


train_df['Stage_fear_Yes'] = train_df['Stage_fear_Yes'].replace({False:0,True:1})
test_df['Stage_fear_Yes'] = test_df['Stage_fear_Yes'].replace({False:0,True:1})


train_df.head()


train_df['Drained_after_socializing_Yes'] = train_df['Drained_after_socializing_Yes'].replace({False:0,True:1})
test_df['Drained_after_socializing_Yes'] = test_df['Drained_after_socializing_Yes'].replace({False:0,True:1})


train_df.head()


train_df.isnull().sum()


test_df.isnull().sum()


X =train_df.drop(columns = 'Personality',axis = 1)
y = train_df['Personality']


X.corr()


from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


imputer = IterativeImputer(random_state=42)

# Fit and transform the data
train_df_imputed = imputer.fit_transform(X)

# Convert back to DataFrame with original column names
train_df_imputed = pd.DataFrame(train_df_imputed, columns=X.columns)

train_df_imputed.head()


test_imputed = imputer.transform(test_df)
test_df_imputed = pd.DataFrame(test_imputed, columns=test_df.columns)
test_df_imputed.head()


from sklearn.preprocessing import StandardScaler


cols_to_scale = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size',
                'Post_frequency']


scaler = StandardScaler()

# Fit and transform the data
scaled_array = scaler.fit_transform(train_df_imputed[cols_to_scale])

# Convert to DataFrame
train_df_imputed[cols_to_scale] = pd.DataFrame(scaled_array, columns=cols_to_scale)


train_df_imputed.head()


# Fit and transform the data
scaled_array = scaler.transform(test_df_imputed[cols_to_scale])

# Convert to DataFrame
test_df_imputed[cols_to_scale] = pd.DataFrame(scaled_array, columns=cols_to_scale)


test_df_imputed.head()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_df_imputed,y,test_size = 0.25,random_state = 42)


X_train.shape,X_test.shape


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


lgr_model = LogisticRegression()


lgr_model.fit(X_train,y_train)


y_pred  = lgr_model.predict(X_test)


print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


y_pred_lgr = lgr_model.predict(test_df_imputed)


lgr_df = pd.DataFrame({
    'id' : test_df_imputed['id'],
    'Participation' : y_pred_lgr
})


lgr_df.head()


lgr_df['Participation'].value_counts()


from sklearn.svm import SVC


model_svc = SVC()


model_svc.fit(X_train,y_train)


y_pred_svc  = model_svc.predict(X_test)


print(accuracy_score(y_test,y_pred_svc))
print(classification_report(y_test,y_pred_svc))


from sklearn.ensemble import RandomForestClassifier


rfc_model = RandomForestClassifier()


rfc_model.fit(X_train,y_train)


y_pred_rfc = rfc_model.predict(X_test)


print(accuracy_score(y_test,y_pred_rfc))
print(classification_report(y_test,y_pred_rfc))


test_df_rfc = rfc_model.predict(test_df_imputed)


rfc_df = pd.DataFrame({
    'id':test_df_imputed['id'],
    'Participation' : test_df_rfc
})


rfc_df.head()


rfc_df['Participation'].value_counts()


from xgboost import XGBClassifier


xgb_model = XGBClassifier()


y_train = y_train.replace({'Extrovert':1,'Introvert':0})


# y_train


xgb_model.fit(X_train,y_train)


y_pred_xgb = xgb_model.predict(X_test)


y_test = y_test.replace({'Extrovert':1,'Introvert':0})


print(accuracy_score(y_test,y_pred_xgb))
print(classification_report(y_test,y_pred_xgb))


xgb_pred_test = xgb_model.predict(test_df_imputed)


xgb_pred_test


xgb_df =pd.DataFrame({
    'id':test_df_imputed['id'],
    'Participation' : xgb_pred_test
})


xgb_df.head()


xgb_df['Participation'] = xgb_df['Participation'].replace({1:'Extrovert',0:'Introvert'})


xgb_df.head()


xgb_df['id'] = xgb_df['id'].astype(int)


xgb_df.head()


xgb_df.to_csv('XGB_MODEL.csv',index = False)


import lightgbm as lgb


lgb_model = lgb.LGBMClassifier(random_state=42)

# Train the model
lgb_model.fit(X_train, y_train)

# Predict
y_pred = lgb_model.predict(X_test)

# Evaluate
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred))
print("LightGBM Classification Report:\n", classification_report(y_test, y_pred))


lgb_model_test = lgb_model.predict(test_df_imputed)


lgb_df_test = pd.DataFrame({
    'id' : test_df_imputed['id'],
    'Personality':lgb_model_test
})


lgb_df_test['Personality']  = lgb_df_test['Personality'].replace({1:'Extrovert',0:'Introvert'})


lgb_df_test.head()


lgb_df_test['Personality'].value_counts()


lgb_df_test.to_csv('lgb_without_tunning.csv')




