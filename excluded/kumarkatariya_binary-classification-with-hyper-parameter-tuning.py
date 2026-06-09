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


train_df =  pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train_df.head()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_df.head()


train_df.duplicated().sum()


train_df.isna().sum()


test_ids = test_df['id']


test_df.drop('id',axis=1,inplace=True)


train_ids = train_df['id'] 


train_df.drop('id',axis=1,inplace=True)


train_df.head()


target = train_df['y']


train_df.drop('y',axis=1,inplace=True)


train_df.head()


train_df.info()


num_cols = train_df.select_dtypes(include=['int64']).columns
cat_cols = train_df.select_dtypes(include=['object']).columns


from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] =  le.transform(test_df[col]) 


train_df.head()


test_df.head()


X = train_df
y = target


# import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
from sklearn.metrics import roc_auc_score
import numpy as np 

rfk = StratifiedKFold(n_splits=5)
auc_scores = []
for train_idx,val_idx in rfk.split(X,y):
    X_train,X_valid = X.iloc[train_idx],X.iloc[val_idx]
    y_train,y_valid = y.iloc[train_idx],y.iloc[val_idx]
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train,y_train)
    y_pred = rf.predict_proba(X_valid)[:,1]
    auc = roc_auc_score(y_valid,y_pred)
    auc_scores.append(auc)
print(np.mean(auc_scores))     


test_pred = rf.predict_proba(test_df)[:,1]
submission = pd.DataFrame({'id':test_ids,
             'y':test_pred})



submission.to_csv('submission.csv',index=False)


# best_params = study.best_trial.params
# best_params['objective'] = 'binary:logistic'
# best_params['verbosity'] = 0

# # Convert full train set to DMatrix
# dtrain_full = xgb.DMatrix(X, label=y)

# # Train the final model
# final_model = xgb.train(best_params, dtrain_full, num_boost_round=100)

# # STEP 2: Predict probabilities on test data
# # Make sure test_df is already preprocessed just like X
# dtest = xgb.DMatrix(test_df)
# test_preds = final_model.predict(dtest)  # output will be probability of class 1




# STEP 3: Prepare submission file
# submission = pd.DataFrame({
#     'id': test_ids,     # Use the ID column from test set
#     'y': test_preds     # These are the predicted probabilities
# })

# # Save to CSV
# submission.to_csv("submission.csv", index=False)














