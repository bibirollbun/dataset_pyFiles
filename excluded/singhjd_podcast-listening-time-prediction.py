# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)


train_df.head()


test_df.head()


train_df.info()


train_df.isna().mean()*100


train_df['Episode_Length_minutes']=train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].mean())
train_df['Guest_Popularity_percentage']=train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].mean())


train_df.dropna(inplace=True)


train_df.isna().mean()*100


test_df.isna().mean()*100


test_df['Episode_Length_minutes']=test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].mean())
test_df['Guest_Popularity_percentage']=test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].mean())


test_df.isna().mean()*100


test_ids=test_df['id']


train_df['Podcast_Name'].unique()


train_df['Podcast_Name'].value_counts()


podcast_name_grouping=train_df.groupby(['Podcast_Name'], as_index=False).agg({'Listening_Time_minutes':'mean'}).sort_values(by=['Listening_Time_minutes'], ascending=False)
podcast_name_grouping['Listening_Time_minutes']=podcast_name_grouping['Listening_Time_minutes'].apply(lambda x: round(x, 2))


top_10_most_listend_podcast=podcast_name_grouping.head(10)
least_10_most_listend_podcast=podcast_name_grouping.tail(10)


plt.figure(figsize=(10, 6))
ax=sns.barplot(x='Podcast_Name', y='Listening_Time_minutes', data=top_10_most_listend_podcast)
for bars in ax.containers:
    ax.bar_label(bars)
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(10, 6))
ax=sns.barplot(x='Podcast_Name', y='Listening_Time_minutes', data=least_10_most_listend_podcast)
for bars in ax.containers:
    ax.bar_label(bars)
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(10, 6))
ax=sns.countplot(x='Publication_Time', data=train_df)
for bars in ax.containers:
    ax.bar_label(bars)
plt.show()


plt.figure(figsize=(10, 6))
ax=sns.countplot(x='Publication_Day', data=train_df)
for bars in ax.containers:
    ax.bar_label(bars)
plt.show()


plt.figure(figsize=(10, 6))
ax=sns.countplot(x='Genre', data=train_df)
for bars in ax.containers:
    ax.bar_label(bars)
plt.show()


host_popularity_grouping=train_df.groupby(['Podcast_Name'], as_index=False).agg({'Host_Popularity_percentage':'mean'}).sort_values(by=['Host_Popularity_percentage'], ascending=False)


top_10_host_popularity=host_popularity_grouping.head(10)
least_10_host_popularity=host_popularity_grouping.tail(10)


top_10_host_popularity['Host_Popularity_percentage']=top_10_host_popularity['Host_Popularity_percentage'].apply(lambda x: round(x, 2))
least_10_host_popularity['Host_Popularity_percentage']=least_10_host_popularity['Host_Popularity_percentage'].apply(lambda x: round(x, 2))


ax=sns.barplot(x='Podcast_Name', y='Host_Popularity_percentage', data=top_10_host_popularity)
for bars in ax.containers:
    plt.bar_label(bars)
plt.xticks(rotation=45)
plt.show()


ax=sns.barplot(x='Podcast_Name', y='Host_Popularity_percentage', data=least_10_host_popularity)
for bars in ax.containers:
    plt.bar_label(bars)
plt.xticks(rotation=45)
plt.show()


plt.pie(train_df['Episode_Sentiment'].value_counts(), labels=train_df['Episode_Sentiment'].value_counts().index, autopct='%1.1f%%')
plt.show()


train_df.drop(columns=['id'], inplace=True)
test_df.drop(columns=['id'], inplace=True)


object_columns=[]
for i in train_df.select_dtypes(include=['object']):
    object_columns.append(i)

object_columns


le=LabelEncoder()
le_Podcast_Name=LabelEncoder()
le_Episode_Title=LabelEncoder()
le_Genre=LabelEncoder()
le_Publication_Day=LabelEncoder()
le_Publication_Time=LabelEncoder()
le_Episode_Sentiment=LabelEncoder()


#Training Data

train_df['Podcast_Name']=le_Podcast_Name.fit_transform(train_df['Podcast_Name'])
train_df['Episode_Title']=le_Episode_Title.fit_transform(train_df['Episode_Title'])
train_df['Genre']=le_Genre.fit_transform(train_df['Genre'])
train_df['Publication_Day']=le_Publication_Day.fit_transform(train_df['Publication_Day'])
train_df['Publication_Time']=le_Publication_Time.fit_transform(train_df['Publication_Time'])
train_df['Episode_Sentiment']=le_Episode_Sentiment.fit_transform(train_df['Episode_Sentiment'])

#Testing data

test_df['Podcast_Name']=le_Podcast_Name.fit_transform(test_df['Podcast_Name'])
test_df['Episode_Title']=le_Episode_Title.fit_transform(test_df['Episode_Title'])
test_df['Genre']=le_Genre.fit_transform(test_df['Genre'])
test_df['Publication_Day']=le_Publication_Day.fit_transform(test_df['Publication_Day'])
test_df['Publication_Time']=le_Publication_Time.fit_transform(test_df['Publication_Time'])
test_df['Episode_Sentiment']=le_Episode_Sentiment.fit_transform(test_df['Episode_Sentiment'])


print("Publication_Time Mapping:")
for idx, value in enumerate(le_Publication_Time.classes_):
    print(f"{value}: {idx}")


train_df.head()


correlation = train_df.corr()['Listening_Time_minutes'].drop('Listening_Time_minutes').sort_values(ascending=False)

# Plot correlation
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation.index, y=correlation.values)
plt.xticks(rotation=45)
plt.title("Feature Correlation with Listening Time")
plt.show()


train_df.drop(columns=['Genre', 'Podcast_Name', 'Publication_Day'], inplace=True)
test_df.drop(columns=['Genre', 'Podcast_Name', 'Publication_Day'], inplace=True)


X=train_df.drop(columns=['Listening_Time_minutes'])
y=train_df['Listening_Time_minutes']


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)


print("X_train shape:", X_train.shape)
print("x_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


linear_model=LinearRegression()
linear_model.fit(X_train, y_train)


linear_model_pred=linear_model.predict(X_test)


print("Mean abolsute error:", mean_absolute_error(y_test, linear_model_pred))
print("Mean squared error:", mean_squared_error(y_test, linear_model_pred))
print("Root mean abolsute error:", np.sqrt(mean_absolute_error(y_test, linear_model_pred)))


prediction1=linear_model.predict(test_df)


submission1=pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': prediction1})
submission1=submission1.to_csv('submission1')
print("Submission1 file created")


train_df.head()




