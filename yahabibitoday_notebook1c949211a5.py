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


train_df=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
train_df.head()



test_df=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_df.head()


import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error


test_ids = test_df['id']



X = train_df.drop('Listening_Time_minutes', axis=1)
y = train_df['Listening_Time_minutes']


# Fill Number_of_Ads with mode
X['Number_of_Ads'] = X['Number_of_Ads'].fillna(X['Number_of_Ads'].mode()[0])
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].fillna(test_df['Number_of_Ads'].mode()[0])

# Fill Guest_Popularity_percentage with mean
X['Guest_Popularity_percentage'] = X['Guest_Popularity_percentage'].fillna(X['Guest_Popularity_percentage'].mean())
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].mean())






# print(test_df.isnull().sum())
cat_cols = ['Podcast_Name','Episode_Title','Genre','Publication_Day', 'Publication_Time', 'Episode_Sentiment']


X = pd.get_dummies(X, columns=cat_cols)
test_df = pd.get_dummies(test_df, columns=cat_cols)



X, test_df = X.align(test_df, join='left', axis=1, fill_value=0)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)



y_valid_pred = model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_valid_pred)
print("Validation MAE:", mae)



y_pred=model.predict(test_df)

submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': y_pred
})


submission.to_csv('submission12.csv',index=False)



pdf=pd.read_csv('/kaggle/working/submission12.csv')
pdf.head()

