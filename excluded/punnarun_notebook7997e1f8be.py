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


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('/kaggle/input/cro-ds-test-condominium-price-prediction-th/train.csv')
df_test = pd.read_csv('/kaggle/input/cro-ds-test-condominium-price-prediction-th/test.csv')


# data cleaning
missing_train = df.isnull().sum() / len(df) * 100
valid_features = []

for col in df.columns:
    if col == 'target': 
        continue
    if missing_train[col] < 50: 
        valid_features.append(col)


features = []
for col in valid_features:
    if col in df_test.columns: 
        if df[col].dtype in ['int64', 'float64']:
            features.append(col)


for i in features:
    median_value = df[i].median()
    df[i] = df[i].fillna(median_value)
        
    median_value = df_test[i].median()
    df_test[i] = df_test[i].fillna(median_value)

df = df.fillna(0)
df_test = df_test.fillna(0)


# feature engineer
data = df[features].copy()
target = df['target']

log_target = np.log(target)

engineered_features = []

facility_cols = [col for col in features if col.startswith('facility_')]
if facility_cols:
    data.loc[:, 'facility_score'] = data[facility_cols].sum(axis=1)
    df_test.loc[:, 'facility_score'] = df_test[facility_cols].sum(axis=1)
    engineered_features.append('facility_score')

all_features = features + engineered_features
df_test_subset = df_test[all_features].copy()


# train and predict
rf_model = RandomForestRegressor()

rf_model.fit(data[all_features], log_target)
predictions = rf_model.predict(df_test_subset)

submission = pd.DataFrame({
    'Id': df_test['Id'],
    'Predicted': predictions 
})

submission.to_csv('/kaggle/working/submission_rf99.csv', index=False)

