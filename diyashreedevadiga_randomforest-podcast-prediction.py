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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
train_df.head()


train_df.isnull().sum()


train_df = train_df.dropna(subset='Number_of_Ads')
test_df = test_df.dropna(subset='Number_of_Ads')


guest_median_train = train_df['Guest_Popularity_percentage'].median()
train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(guest_median_train)

guest_median_test = test_df['Guest_Popularity_percentage'].median()
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(guest_median_test)


ep_median_train = train_df['Episode_Length_minutes'].median()
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(ep_median_train)

ep_median_test = test_df['Episode_Length_minutes'].median()
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(ep_median_test)


train_df.info()
test_df.info()


df1 = train_df.copy()
df2 = test_df.copy()
from sklearn.preprocessing import LabelEncoder
l_model = LabelEncoder()
column = ['Genre','Publication_Day','Publication_Time','Episode_Sentiment']
for c in column:
    df1[c] = l_model.fit_transform(df1[c].astype(str))
    df2[c] = l_model.fit_transform(df2[c].astype(str))

print("Done")


X_test_org = df2.copy()
df1.drop(columns='id',inplace=True)
df2.drop(columns='id',inplace=True)
df1.info()


numeric_df = df1.select_dtypes(include=['float64','int64'])
corr_matrix = numeric_df.corr()
corr_matrix


target = ['Listening_Time_minutes']
y_train = df1[target]
X_train = df1.drop(columns=['Listening_Time_minutes','Podcast_Name','Episode_Title'])

X_test = df2.drop(columns=['Podcast_Name','Episode_Title'])


X_train


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
model = RandomForestRegressor()
model.fit(X_train,y_train)
y_pred= model.predict(X_train)
reg_mse = mean_squared_error(y_train,y_pred)
reg_rmse = np.sqrt(reg_mse)
reg_rmse


y_pred_test= model.predict(X_test)


output = pd.DataFrame({
    'id': X_test_org['id'],
    'Predicted_Listening_Time_minutes': y_pred_test
})

output.to_csv('submission.csv', index=False)


output.head()




