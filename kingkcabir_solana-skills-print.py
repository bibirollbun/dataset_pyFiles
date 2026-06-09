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


skill_submit = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/sample_submission.csv')
skill_test = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/test_unlabeled.csv')
skill_train = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/train.csv')


skill_test.head(2)


skill_submit.head(2)


class get_summary:
    def __init__(self, x):
        self.x = x
    def data_set(self):
        #checks for duplicate
        duplicate = self.x.duplicated().any()
        #drop duplicates 
        if duplicate == True:
            self.x.drop_duplicates(inplace=True)
            self.x.reset_index(drop=True)
        #checks for empty values
        null = self.x.isna().sum().any()
        #missing values
        total_missing = self.x.isnull().sum().sum()
        #data types
        data_type = self.x.dtypes
        #shape
        shapes = self.x.shape
        return f"Duplicate: {duplicate}\nNull: {null}\nMissing_value: {total_missing}\nTypes:\n{data_type}\nShape: {shapes}"
     #missing values
    def total_missing(self):
        missing_vals = self.x.isnull().sum()
        cols_with_missing = missing_vals[missing_vals > 0]
        return cols_with_missing.to_dict()
print(f"Training dataset:\n{get_summary(skill_train).data_set()}\nTest dataset:\n{get_summary(skill_test).data_set()}")
print(f"columns with missing values train\n{get_summary(skill_train).total_missing()}\ncolumns with missing values test\n{get_summary(skill_test).total_missing()}")


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
                    
import hashlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 


skill_train.head(3)


#dropping the column with NaN values
skill_train.drop("slot_graduated", axis=1, inplace=True)
skill_train[5:10]


X = skill_train.drop('has_graduated', axis=1)


#encoding
def encode_(data):
    data['mint_hashed'] = data['mint'].apply(lambda x: int(hashlib.sha256(str(x).encode()).hexdigest(), 16) % (10 ** 8))

    if 'is_valid' in data.columns: 
        label_enc = LabelEncoder()
        data['is_valid'] = label_enc.fit_transform(data['is_valid'])
    else:
        raise KeyError("The 'is_valid' column is missing in the DataFrame.")

    return data.head(4)

encode_(X)    


y = LabelEncoder().fit_transform(skill_train['has_graduated'])
y


X.drop('mint', axis=1, inplace=True)
X.head(2)


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=12)


import xgboost as xgb
from sklearn.metrics import log_loss

train_data = xgb.DMatrix(X_train, label=y_train)
valid_data = xgb.DMatrix(X_val, label=y_val)
params = {
         'num_leaves': 30,
         'learning_rate': 1,
         'n_estimators': 1000,
         'random_state': 12,
         'n_jobs': -1,
         'objective': 'binary:logistic',
         'eval_metric': 'logloss'
}
                              
num_round = 200
bst = xgb.train(params, train_data, num_round,
                evals=[(train_data, 'train'), (valid_data, 'val')])


y_pred = bst.predict(xgb.DMatrix(X_val))
y_pred_bin = (y_pred > 0.5).astype(int)
y_loss = log_loss(y_val, y_pred_bin)
print(y_loss)


encode_(skill_test)


skill_test.drop('mint', axis=1, inplace=True)


test_pred = bst.predict(xgb.DMatrix(skill_test))


submission = pd.DataFrame({'mint': skill_submit['mint'], 'has_graduated': test_pred})
submission.head(3)


submission.to_csv("submission.csv", index=False)

