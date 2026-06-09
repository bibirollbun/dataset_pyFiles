# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')
s_sub = pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')


train.head()


missing_total = train.isnull().sum()
missing_percent = (missing_total/len(train))*100
missing_df = pd.DataFrame({'missing total':missing_total,'missing percent':missing_percent})
missing_df


train.info()


train.drop('Surname',axis=1,inplace=True)
test.drop('Surname',axis=1,inplace=True)


train.drop(['id','CustomerId'], inplace= True, axis = 1)
test.drop(['id','CustomerId'], inplace= True, axis = 1)


def geo_encoder(df):
    encoder = OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore')

    encoder.fit(df[['Geography']])
    df_encoded = encoder.transform(df[['Geography']])
    df_encoded_df = pd.DataFrame(df_encoded, columns = encoder.get_feature_names_out(['Geography']),index=df.index)

    df = pd.concat([df.drop('Geography', axis=1), df_encoded_df], axis=1)

    
    for col in encoder.get_feature_names_out(['Geography']):
        df[col] = df[col].astype(int)
    return df

train = geo_encoder(train)
test = geo_encoder(test)



mapping = {'Male':0,'Female':1}
train['Gender'] = train['Gender'].map(mapping).astype(int)
test['Gender'] = test['Gender'].map(mapping).astype(int)



train['combined_info'] = (train['NumOfProducts']+train['HasCrCard'])*train['IsActiveMember']

train['activity/balance'] = np.ones(len(train))
train.loc[(train['IsActiveMember']==0) & (train['Balance']==0), 'activity/balance'] = 0
train['activity/balance'] = train['activity/balance'].astype(int)

test['combined_info'] = (test['NumOfProducts']+test['HasCrCard'])*test['IsActiveMember']

test['activity/balance'] = np.ones(len(test))
test.loc[(test['IsActiveMember']==0) & (test['Balance']==0), 'activity/balance'] = 0
test['activity/balance'] = test['activity/balance'].astype(int)



X = train.drop('Exited',axis = 1)
y = train['Exited']

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.3)


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix



# xgb_model = XGBClassifier()
# xgb_model.fit(X_train, y_train)
# print("ROC AUC of XGBoost:", roc_auc_score(xgb_model.predict(X_val), y_val))


# test_id = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')
# submission = pd.DataFrame()
# submission["id"] = test_id.index
# submission["Exited"] = xgb_model.predict_proba(test)[:,1]

# submission.to_csv("submission.csv",header=True,index=False)
# submission



s_sub


xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="auc",
    early_stopping_rounds=50      
)


xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=20
)


y_val_pred = xgb_model.predict_proba(X_val)[:, 1]
print("ROC AUC of XGBoost:", roc_auc_score(y_val, y_val_pred))




test_id = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')

submission = pd.DataFrame({
    "id":     test_id["id"], 
    "Exited": xgb_model.predict_proba(test)[:,1]
})
submission.to_csv("submission.csv", index=False)
submission.head()




submission.to_csv('/kaggle/working/submission.csv', index=False)




submission




