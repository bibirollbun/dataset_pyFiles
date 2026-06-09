# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


dataset = pd.read_csv('../input/playground-series-s5e4/train.csv')


dataset.columns


y_train = dataset.loc[: , ['Listening_Time_minutes']]
dataset = dataset.drop(['Listening_Time_minutes' ,'id'], axis = 1)


dataset['Episode_Title'] = dataset['Episode_Title'].str.extract('(\d+)').astype(int)


# Get list of categorical variables
s = (dataset.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


dataset.isnull().sum()


dataset['Number_of_Ads'].fillna(0, inplace=True)


# Group-wise imputation (better)
dataset['Episode_Length_minutes'] = dataset.groupby('Podcast_Name')['Episode_Length_minutes']\
                               .transform(lambda x: x.fillna(x.median()))


dataset['Guest_Popularity_percentage'] = dataset.groupby('Podcast_Name')['Guest_Popularity_percentage']\
                               .transform(lambda x: x.fillna(x.median()))


dataset.isnull().sum()


numeric_data = dataset.select_dtypes(exclude=['object'])

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_data = encoder.fit_transform(dataset[object_cols])
encoded_feature_names = encoder.get_feature_names_out(object_cols)
encoded_dataset = pd.concat([
    pd.DataFrame(encoded_data, columns=encoded_feature_names, index=dataset.index),
    numeric_data
], axis=1)


encoded_dataset.head()


encoded_dataset.loc[0:5 ,['Episode_Title']]


X_train = np.array(encoded_dataset)


y_train = np.array(y_train)
y_train[0 : 1]
y_train.shape


X_train[0:1]


my_model = XGBRegressor(n_estimators = 2000 , learning_rate = 0.005 , n_jobs=4,)
my_model.fit(X_train, y_train)


predictions = my_model.predict(X_train)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_train)))


test_dataset = pd.read_csv('../input/playground-series-s5e4/test.csv')


test_dataset.head()


ids = test_dataset.loc[: , 'id']
ids = ids.tolist()


test_dataset = test_dataset.drop(['id'], axis = 1)


test_dataset['Episode_Title'] = test_dataset['Episode_Title'].str.extract('(\d+)').astype(int)


# Get list of categorical variables
s = (test_dataset.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


test_dataset.isnull().sum()


# Group-wise imputation (better)
test_dataset['Episode_Length_minutes'] = test_dataset.groupby('Podcast_Name')['Episode_Length_minutes']\
                               .transform(lambda x: x.fillna(x.median()))


test_dataset['Guest_Popularity_percentage'] = test_dataset.groupby('Podcast_Name')['Guest_Popularity_percentage']\
                               .transform(lambda x: x.fillna(x.median()))


test_dataset.isnull().sum()


numeric_test_data = test_dataset.select_dtypes(exclude=['object'])

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_data = encoder.fit_transform(test_dataset[object_cols])
encoded_feature_names = encoder.get_feature_names_out(object_cols)
encoded_test_dataset = pd.concat([
    pd.DataFrame(encoded_data, columns=encoded_feature_names, index=test_dataset.index),
    numeric_test_data
], axis=1)


encoded_test_dataset.columns


X_test = np.array(encoded_test_dataset)


predictions = my_model.predict(X_test)



predictions.shape


predictions= predictions.tolist()


submission = pd.DataFrame({
   'id' : ids,
    'Listening_Time_minutes':  predictions
});
submission.to_csv('sample_submissson.csv' , index=False)




