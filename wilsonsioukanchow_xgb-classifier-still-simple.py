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


df_sub = pd.read_csv(r'/kaggle/input/playground-series-s5e7/sample_submission.csv')
df_sub


df_train = pd.read_csv(r'/kaggle/input/playground-series-s5e7/train.csv')
"""
in here there was an error in this dataset, "RuntimeWarning: invalid value encountered in greater
  has_large_values = (abs_vals > 1e6).any()"

"""
df_train


df_train_1 = df_train.copy()
list_of_yes_or_no = ['Stage_fear', 'Drained_after_socializing']
for column in list_of_yes_or_no:
    print(column)
    df_train_1[column] = df_train_1[column].map({'Yes': 1, 'No': 0}).fillna(-1)


df_train_not_categorical = df_train_1.drop(columns=list_of_yes_or_no)
df_train_not_categorical = df_train_not_categorical.drop(columns=['id'])
df_train_not_categorical = df_train_not_categorical.fillna(-1)
df_train_not_categorical


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def plot_quick(df):
    df.select_dtypes(include=[np.number]).hist(figsize=(15, 10), bins=300)
    plt.tight_layout()
    plt.show()

plot_quick(df_train_not_categorical)


df_train = pd.read_csv(r'/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e7/test.csv')

list_of_yes_or_no = ['Stage_fear', 'Drained_after_socializing']
for column in list_of_yes_or_no:
    print(column)
    df_train[column] = df_train[column].map({'Yes': 1, 'No': 0})
    df_test[column] = df_test[column].map({'Yes': 1, 'No': 0})
df_train = df_train.fillna(-1)
df_test = df_test.fillna(-1)


#not this time!
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = df_train.drop(columns=['id', 'Personality'])
y_train = df_train['Personality'].map({'Extrovert': 0, 'Introvert': 1})
id_train = df_train['id'] # For this submission, i need this

X_test = df_test.drop(columns=['id'])
id_test = df_test['id']


import xgboost as xgb
model_xgb = xgb.XGBClassifier()
model_xgb.fit(X_train, y_train)
predictions = model_xgb.predict(X_test)

results_df = pd.DataFrame()
results_df['id'] = id_test
results_df['Personality'] = predictions
results_df['Personality'] = results_df['Personality'].map({0: 'Extrovert', 1:'Introvert'})
results_df


results_df.to_csv('submission.csv', index=False)


for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

