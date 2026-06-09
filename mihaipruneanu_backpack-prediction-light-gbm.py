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


PATH = '/kaggle/input/playground-series-s5e2/'
train_main = pd.read_csv(PATH + 'train.csv')
test  = pd.read_csv(PATH + 'test.csv')
train_extra = pd.read_csv(PATH + 'training_extra.csv')

train = pd.concat([train_main, train_extra], axis=0, ignore_index=True)


#save test IDs and drop 'id'
test_ids = test['id']
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

#identify target and columns
TARGET   = 'Price'
num_cols = train.select_dtypes(include=['int64','float64']).columns.drop(TARGET)
cat_cols = train.select_dtypes(include=['object']).columns

#fill and one-hot encode
def preprocess(df):
    df = df.copy()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown').astype(str).str.strip()
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

#preprocessing
train_p = preprocess(train)
test_p  = preprocess(test)

#align features(exclude target)
X, X_test = train_p.drop(TARGET, axis=1), test_p
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

y = train_p[TARGET]
print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")


from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor,early_stopping,log_evaluation
from sklearn.metrics import mean_squared_error

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[
        early_stopping(50),
        log_evaluation(0)
    ]
)

y_pred = model.predict(X_val)
print("Validation RMSE:", mean_squared_error(y_val, y_pred, squared=False))



import pandas as pd
ID_COL = "id"        
TARGET = "Price" 


model.fit(X, y)


preds = model.predict(X_test)

submission = pd.DataFrame({
    ID_COL:   test_ids,   
    TARGET:   preds       
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved:", submission.shape)





