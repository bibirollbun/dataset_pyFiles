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


import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
data.head()
test_data = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


data = data.drop('id', axis=1)


data.shape


data.info()


corr_data = data.corr()


sns.heatmap(corr_data, annot=True, cmap='viridis')


X_train = data.drop('BeatsPerMinute', axis=1)
y_train = data['BeatsPerMinute']


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('ss', StandardScaler()),
    ('model', LinearRegression()),
])

pipe.fit(X_train, y_train)

pipe.named_steps['model'].coef_


test_data.head()


test_data.shape


X_test = test_data.drop('id', axis=1)

predictions = pipe.predict(X_test)
predictions.shape


submission_data = {
    'id': test_data['id'],
    'BeatsPerMinute': predictions
}

submission_df = pd.DataFrame(submission_data)


submission_df.head()


submission_df.to_csv('/kaggle/working/submission.csv', index=False)




