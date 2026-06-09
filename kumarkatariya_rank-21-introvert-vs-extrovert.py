# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test_df.head()

train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv') 
train_datasert = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
train_dataset = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')

train_df.tail()


train_datasert = train_datasert.rename(columns = {'Personality':'match_p'}).drop_duplicates(['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing','Friends_circle_size','Post_frequency'])
train_datasert.head()


train_datasert.shape


train_dataset = train_dataset.rename(columns= {'Personality':'match_p'}).drop_duplicates(['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing','Friends_circle_size','Post_frequency'])
train_dataset.head()


# now lets concate both external datasets

external_df = pd.concat([train_datasert,train_dataset],ignore_index=True)
external_df.drop_duplicates(['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing','Friends_circle_size','Post_frequency'],inplace=True)
 


train_df.head()


merge_on = ['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing','Friends_circle_size','Post_frequency']


train_df = train_df.merge(external_df,on=merge_on,how='left')
test_df = test_df.merge(external_df,on=merge_on,how='left')


train_df.head()

# removing id col

train_df.drop('id',axis=1,inplace=True)


test_df.head()


train_df.head()


target = train_df['Personality']
train_df.drop('Personality',axis=1,inplace=True)
num_cols = train_df.select_dtypes(include=['float64']).columns
cat_cols = train_df.select_dtypes(include=['object']).columns


from sklearn.impute import SimpleImputer 

SI = SimpleImputer(strategy='median')

train_df[num_cols] = SI.fit_transform(train_df[num_cols])
test_df[num_cols] = SI.transform(test_df[num_cols])


SI = SimpleImputer(strategy='most_frequent')

train_df[cat_cols] = SI.fit_transform(train_df[cat_cols])
test_df[cat_cols] = SI.transform(test_df[cat_cols])


train_df.head()


# now lets do one hot encoding to see whether we are getting the better score or not 


from sklearn.preprocessing import OneHotEncoder 

encoder = OneHotEncoder(drop ='first',sparse_output=False) 

enc_array = (encoder.fit_transform(train_df[cat_cols]))
enc_array_test = encoder.transform(test_df[cat_cols])
enc_df = pd.DataFrame(enc_array,columns = encoder.get_feature_names_out(cat_cols),index = train_df.index)
enc_df_test = pd.DataFrame(enc_array_test,columns = encoder.get_feature_names_out(cat_cols),index=test_df.index)


train_df.drop(columns = cat_cols,inplace=True,axis=1)
test_df.drop(columns=cat_cols,inplace=True,axis=1)


 


train_df.head()

train_df = pd.concat([train_df,enc_df],axis=1)
test_df = pd.concat([test_df,enc_df_test],axis=1)
train_df.head()
test_df.head()


test_ids = test_df['id']
test_ids


test_df.drop('id',axis=1,inplace=True)


X = train_df
y = target



y = y.map({'Extrovert':0,'Introvert':1})


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,log_loss

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
 


params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}


from sklearn.model_selection import RepeatedStratifiedKFold 

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

N_SPLITS = 5
N_REPEATS = 3
skf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

# CV only on training set!
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=50, verbose_eval=False)

    oof_preds[val_idx] += model.predict(dval) / N_REPEATS
    test_preds += model.predict(dtest) / (N_REPEATS * N_SPLITS)
 


print(oof_preds[val_idx].shape)
print(model.predict(dval).shape)


model.get_score(importance_type='gain')


submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


ll = log_loss(y_train, oof_preds)
cv_acc = accuracy_score(y_train, oof_preds > 0.35)
print(f"Cross-Validation log loss: {ll:.4f}, accuracy: {cv_acc:.4f}")

# Create submission
final_preds = pd.Series((test_preds > 0.35).astype(int))
submission["Personality"] = final_preds.map({0:'Extrovert',1:'Introvert'})
submission.to_csv("submission.csv", index=False)
submission.head()














 


 




