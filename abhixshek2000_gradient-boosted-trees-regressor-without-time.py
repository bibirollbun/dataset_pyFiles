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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_df.shape, test_df.shape


train_df.head()


train_df.info()


test_df.info()


train_df.describe(include="all")


train_df['year'] = pd.to_datetime(train_df.date).dt.year


train_df.columns


train_df = train_df.loc[~train_df['num_sold'].isnull(), ['country', 'store', 'product', 'year', 'num_sold']]


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_percentage_error


X = train_df[['country', 'store', 'product']]
y = train_df['num_sold'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


gbrt = make_pipeline(OneHotEncoder(handle_unknown="ignore"), GradientBoostingRegressor(max_depth=5, random_state=42))


gbrt


gbrt.fit(X_train, y_train)


y_preds = gbrt.predict(X_test)


y_preds[:4]


mean_absolute_percentage_error(y_test, y_preds)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

def submission_generation(test_df, model):
    ids = test_df['id']
    test_df = test_df.drop("id", axis=1)
    
    test_df['year'] = pd.to_datetime(test_df['date']).dt.year

    test_preds = model.predict(test_df[['country', 'store', 'product']])
    submission_df = pd.DataFrame({'id': ids, 'num_sold': test_preds})
    submission_df.to_csv('submission.csv', index=False)


submission_generation(test_df, gbrt)




