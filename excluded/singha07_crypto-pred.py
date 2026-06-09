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
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


test_df.head(5)


train_df.head(5)


train_df.info()


import numpy as np

for col in train_df.columns:
    col_type = train_df[col].dtype

    if col_type == 'int64':
        # Check min/max to choose int32 or int16
        if train_df[col].min() > np.iinfo(np.int16).min and train_df[col].max() < np.iinfo(np.int16).max:
            train_df[col] = train_df[col].astype(np.int16)
        else:
            train_df[col] = train_df[col].astype(np.int32)

    elif col_type == 'float64':
        train_df[col] = train_df[col].astype(np.float32)



train_df.info()


import numpy as np

for col in test_df.columns:
    col_type = test_df[col].dtype

    if col_type == 'int64':
        # Check min/max to choose int32 or int16
        if test_df[col].min() > np.iinfo(np.int16).min and test_df[col].max() < np.iinfo(np.int16).max:
            test_df[col] = test_df[col].astype(np.int16)
        else:
            test_df[col] = test_df[col].astype(np.int32)

    elif col_type == 'float64':
        test_df[col] = test_df[col].astype(np.float32)



train_df.shape


train_df.isna().sum()[train_df.isna().sum() > 0]



test_df.isna().sum()[test_df.isna().sum() > 0]



train_df.select_dtypes(include='object').columns


train_df = train_df.reset_index(drop=True)



train_df.head()


test_df = test_df.reset_index()



ids = test_df['ID'].copy()
test_df.drop(columns = ['ID'],inplace = True)



from sklearn.model_selection import train_test_split

X = train_df.drop(columns = ['label'])
y = train_df['label']


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


import numpy as np

# Check for infinite values
print(np.isinf(X).sum())  # count of infinite values in training features

# Check for very large values (e.g., >1e10)
print((X > 1e10).sum())
print((X < -1e10).sum())



X.replace([np.inf, -np.inf], np.nan, inplace=True)
# x_test.replace([np.inf, -np.inf], np.nan, inplace=True)



test_df.replace([np.inf, -np.inf], np.nan, inplace=True)


# x_train.fillna(method='ffill', inplace=True)  # forward fill
# x_test.fillna(method='ffill', inplace=True)
test_df.fillna(method='ffill', inplace=True)
X.fillna(method='ffill', inplace=True)


assert not np.isinf(X).any().any(), "Still have inf in X_train"
# assert not np.isinf(x_test).any().any(), "Still have inf in X_val"


x_train.shape


from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    # max_depth=6,
    tree_method='gpu_hist',
    # subsample=0.8,
    # colsample_bytree=0.8
)

xgb_model.fit(X,y)




test_df.drop(columns = ['label'],inplace = True)


batch_size = 10000
predictions = []

for i in range(0, len(test_df), batch_size):
    batch_pred = xgb_model.predict(test_df.iloc[i:i+batch_size])
    predictions.extend(batch_pred)

predictions = np.array(predictions)



predictions


submission = pd.DataFrame({
    'ID' : ids,
    'prediction': predictions
})


submission


submission.to_csv('submission2.csv', index=False)


