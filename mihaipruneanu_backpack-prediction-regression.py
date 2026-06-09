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

#Load data
PATH = '/kaggle/input/playground-series-s5e2/'
train = pd.read_csv(PATH + 'train.csv')
test  = pd.read_csv(PATH + 'test.csv')

train.head()


def smart_fill(df, num_cols, cat_cols):
    df = df.copy()
    # numeric
    for col in num_cols:
        df[col + '_was_nan'] = df[col].isna().astype(int)
        med = df[col].median()
        df[col].fillna(med, inplace=True)
    # categorical
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown').str.strip()
    return df
print("Operation successfull")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


PATH   = '/kaggle/input/playground-series-s5e2/'   
TRAIN  = 'train.csv'
TEST   = 'test.csv'

train = pd.read_csv(PATH + TRAIN)
test  = pd.read_csv(PATH + TEST)

TARGET  = 'Price'    
ID_COL  = 'id'        

num_cols = train.select_dtypes(include=['int64', 'float64']).columns.drop([TARGET, ID_COL])
cat_cols = train.select_dtypes(include=['object', 'category']).columns

#Fill missing values
def fill_missing(df):
    df = df.copy()
    # numeric columns
    for col in num_cols:
        df[col + '_was_nan'] = df[col].isna().astype(int)
        med = df[col].median()
        df[col] = df[col].fillna(med)
    # categorical columns
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown').str.strip()
    return df

train_f = fill_missing(train)
test_f  = fill_missing(test)

#one-hot enconding
train_enc = pd.get_dummies(train_f, columns=cat_cols, drop_first=True)
test_enc  = pd.get_dummies(test_f,  columns=cat_cols, drop_first=True)

train_enc, test_enc = train_enc.align(test_enc, axis=1, fill_value=0)

X = train_enc.drop([TARGET, ID_COL], axis=1)
y = train_enc[TARGET]

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
test_enc[num_cols] = scaler.transform(test_enc[num_cols])

#train validation and split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)

linreg = LinearRegression()
linreg.fit(X_tr, y_tr)

val_pred = linreg.predict(X_val)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f'Validation RMSE : {rmse:.3f}')

#retrain on full data and predict
linreg.fit(X, y)

test_features = test_enc.drop([ID_COL, TARGET], axis=1, errors='ignore')
test_enc[TARGET] = linreg.predict(test_features)

#build the submission file
submission = test_enc[[ID_COL, TARGET]]
submission.to_csv('submission.csv', index=False)
print('submission.csv saved')


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

PATH        = '/kaggle/input/playground-series-s5e2/'
TRAIN       = 'train.csv'
TRAIN_EXTRA = 'training_extra.csv'
TEST        = 'test.csv'
TARGET      = 'Price'
ID_COL      = 'id'

train       = pd.read_csv(PATH + TRAIN)
train_extra = pd.read_csv(PATH + TRAIN_EXTRA)
test        = pd.read_csv(PATH + TEST)

num_cols = train.select_dtypes(include=['int64', 'float64']).columns.drop([TARGET, ID_COL])
cat_cols = train.select_dtypes(include=['object', 'category']).columns

def fill_missing(df):
    df = df.copy()
    for col in num_cols:
        df[col + '_was_nan'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown').str.strip()
    return df

train_f       = fill_missing(train)
train_extra_f = fill_missing(train_extra)
test_f        = fill_missing(test)

train_enc       = pd.get_dummies(train_f,       columns=cat_cols, drop_first=True)
train_extra_enc = pd.get_dummies(train_extra_f, columns=cat_cols, drop_first=True)
test_enc        = pd.get_dummies(test_f,        columns=cat_cols, drop_first=True)

# align columns
all_cols = train_enc.columns.union(train_extra_enc.columns).union(test_enc.columns)
train_enc       = train_enc.reindex(columns=all_cols, fill_value=0)
train_extra_enc = train_extra_enc.reindex(columns=all_cols, fill_value=0)
test_enc        = test_enc.reindex(columns=all_cols, fill_value=0)

#split features and target
X_orig  = train_enc.drop([TARGET, ID_COL], axis=1)
y_orig  = train_enc[TARGET]
X_extra = train_extra_enc.drop([TARGET, ID_COL], axis=1)
y_extra = train_extra_enc[TARGET]
X_test  = test_enc.drop([TARGET, ID_COL], axis=1, errors='ignore')

scaler = StandardScaler()
scaler.fit(pd.concat([X_orig, X_extra])[num_cols])
X_orig[num_cols]  = scaler.transform(X_orig[num_cols])
X_extra[num_cols] = scaler.transform(X_extra[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])


X_full = pd.concat([X_orig, X_extra], axis=0).reset_index(drop=True)
y_full = pd.concat([y_orig, y_extra], axis=0).reset_index(drop=True)
print(f'Combined training samples: {X_full.shape[0]}')

#train and validate on the new dataset
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42
)
linreg = LinearRegression()
linreg.fit(X_tr, y_tr)
val_pred = linreg.predict(X_val)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f'Validation RMSE on combined data: {rmse:.3f}')

linreg.fit(X_full, y_full)
test_preds = linreg.predict(X_test)

# build 
submission = pd.DataFrame({
    ID_COL: test_enc[ID_COL],
    TARGET: test_preds
})
submission.to_csv('submission_full.csv', index=False)
print('submission_full.csv saved')





