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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('/kaggle/input/car-price-prediction-khazani-ahmed/train (1).csv')
df_test = pd.read_csv('/kaggle/input/car-price-prediction-khazani-ahmed/test (1).csv')
test_y = pd.read_csv('/kaggle/input/car-price-prediction-khazani-ahmed/sample_submission (2).csv')
train_y = df_train.pop('price')


names_train = df_train['CarName'].str.split().str[0]
df_train.pop('CarName')
names_test = df_test['CarName'].str.split().str[0]
df_test.pop('CarName')


df_train


ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
mms = MinMaxScaler()


def filter_data(dataframe: pd.DataFrame, flag=True) -> pd.DataFrame:
    df = dataframe.copy()
    # 
    
    df.set_index('car_ID', inplace=True)

    categorical_cols = df.select_dtypes('object').columns

    numeric_cols = df.select_dtypes('number').columns

    if flag:
        ohe.fit(df[categorical_cols])
        mms.fit(df[numeric_cols])
    ohe_data = ohe.transform(df[categorical_cols])
    ohe_df = pd.DataFrame(ohe_data, index=df.index)

    scaled_data = mms.transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=df.index)

    result_df = pd.concat([ohe_df, scaled_df], axis=1)
    result_df.columns = result_df.columns.astype(str)
    return result_df


X_train = filter_data(df_train)
X_test = filter_data(df_test, False)


model = Ridge(alpha=0.01)
model.fit(X_train, train_y)
predictions = model.predict(X_test)


predictions = pd.Series(predictions, name='price')
df_test = pd.concat([df_test, predictions, names_test], axis=1)
df_train = pd.concat([df_train, train_y, names_train], axis=1)


name_cost = (df_train.groupby('CarName')
             .agg({'price': 'mean'})
             .sort_values(by='price', ascending=False))
name_cost_pred = (df_test.groupby('CarName')
                  .agg({'price': 'mean'})
                  .sort_values(by='price', ascending=False))


plt.figure(figsize=(12, 6))
sns.barplot(data=name_cost.T)
plt.title('Train Data')
plt.xlabel('Mark')
plt.ylabel('Mean cost')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(data=name_cost_pred.T)
plt.title('Pred Data')
plt.xlabel('Mark')
plt.ylabel('Mean cost')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

