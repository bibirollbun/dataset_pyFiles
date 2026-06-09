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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df.head()


df_clean = df.dropna().copy()


df_clean = df.dropna().copy()
df_clean = df_clean[df_clean['Listening_Time_minutes'] !=0]

df_clean = df_clean.drop(columns=['id'])


df_clean=df_clean[df_clean['Guest_Popularity_percentage'] != 0]


df_clean.head()


new_df = df_clean


label_encoder = LabelEncoder()
categorical_columns = ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]

for col in categorical_columns:
    new_df[col] = label_encoder.fit_transform(new_df[col])

new_df["Episode_Title"] = new_df["Episode_Title"].astype(str)

new_df["Episode_Title"] = new_df["Episode_Title"].str.extract(r"(\d+)").astype(int)


new_df.head()


# Create a new column for the ratio between Listening_Time_minutes and Episode_Length_minutes
new_df['listening_to_episode_ratio'] = new_df['Listening_Time_minutes'] / new_df['Episode_Length_minutes']

# Filter rows where the ratio is greater than or equal to 0.30
new_df = new_df[new_df['listening_to_episode_ratio'] >= 0.20]

# Optionally, drop the 'listening_to_episode_ratio' column if you no longer need it
new_df = new_df.drop(columns=['listening_to_episode_ratio'])


new_df.count()


#train model with this data only to fill others data 
X = new_df.drop(columns=['Listening_Time_minutes'],axis=1)
y = new_df['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=.2)
model = XGBRegressor(random_state=0)
model.fit(X_train,y_train)
score = model.score(X_test,y_test)
score


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


np.sqrt(mse)




