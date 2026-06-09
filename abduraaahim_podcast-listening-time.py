# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=0)
train_df.head(10)


train_df.shape


train_df.isna().sum()


train_df.info()


guest_popularity_mean = np.round(train_df['Guest_Popularity_percentage'].median())
episode_length_mean = np.round(train_df['Episode_Length_minutes'].median())

print(f'{guest_popularity_mean=}')
print(f'{episode_length_mean=}')


train_df.fillna({'Guest_Popularity_percentage':guest_popularity_mean, 'Episode_Length_minutes':episode_length_mean}, inplace=True)


train_df.dropna(inplace=True)


train_df.isna().sum()


train_df.sample(5)


print(f"genres: \n{train_df['Genre'].unique()}\n")
print(f"publication day: \n{train_df['Publication_Day'].unique()}\n")
print(f"publication time: \n{train_df['Publication_Time'].unique()}\n")
print(f"publication sentiment: \n{train_df['Episode_Sentiment'].unique()}")


genres = {'True Crime':0,'Comedy':1,'Education':2,'Technology':3, 
          'Health':4,'News':5,'Music':6,'Sports':7,'Business':8,'Lifestyle':9}

publication_day = {'Thursday':0,'Saturday':1,'Tuesday':2,'Monday':3,
                   'Sunday':4,'Wednesday':5,'Friday':6,}

publication_time = {'Night':0,'Afternoon':1,'Evening':2,'Morning':3}

Episode_Sentiment = {'Positive':0, 'Negative':1, 'Neutral':2}


train_df['Genre'] = train_df['Genre'].map(genres)
train_df['Publication_Day'] = train_df['Publication_Day'].map(publication_day)
train_df['Publication_Time'] = train_df['Publication_Time'].map(publication_time)
train_df['Episode_Sentiment'] = train_df['Episode_Sentiment'].map(Episode_Sentiment)


train_df.head()


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
encoder = OrdinalEncoder()
train_df[['Episode_Title']] = encoder.fit_transform(train_df[['Episode_Title']])


train_df.head()


X = train_df.drop('Listening_Time_minutes', axis=1)
y = train_df['Listening_Time_minutes']


X.drop('Podcast_Name', axis=1, inplace=True)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_prepared = scaler.fit_transform(X)


X.shape, X_prepared.shape


X_prepared_df = pd.DataFrame(X_prepared, columns=X.columns, index=X.index)
X_prepared_df


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


lr_predict = linear_model.predict(X_test)


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual')
plt.plot(lr_predict, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()


plt.hist(lr_predict, bins=100)
plt.title("Distribution of Predictions")
plt.show()




