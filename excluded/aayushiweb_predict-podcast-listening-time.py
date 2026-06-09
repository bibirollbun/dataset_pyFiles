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


train=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
train.head()


test=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test.head()


train.info()


test.info()





median_epi=train['Episode_Length_minutes'].median()
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(median_epi)


train.info()


medi_epi= train["Guest_Popularity_percentage"].median()
train["Guest_Popularity_percentage"]=train["Guest_Popularity_percentage"].fillna(medi_epi)


medi_epi= train["Number_of_Ads"].mode()
train["Number_of_Ads"]=train["Number_of_Ads"].fillna(medi_epi)


train.info()


median_epi=test['Episode_Length_minutes'].median()
test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(median_epi)


medi_epi= test["Guest_Popularity_percentage"].median()
test["Guest_Popularity_percentage"]=test["Guest_Popularity_percentage"].fillna(medi_epi)





train.head()


from sklearn.preprocessing import OneHotEncoder
genres = train[['Genre']]  

encoder = OneHotEncoder(sparse_output=False, drop='first') 
encoded_genres = encoder.fit_transform(genres)


genre_encoded_df = pd.DataFrame(
    encoded_genres,
    columns=encoder.get_feature_names_out(['Genre']))
genre_encoded_df=genre_encoded_df.astype(int)
genre_encoded_df.head()


train.reset_index(drop=True, inplace=True)
genre_encoded_df.reset_index(drop=True, inplace=True)


train_con = pd.concat([train, genre_encoded_df], axis=1)
print("\nFinal DataFrame:")
train_con.head()


train_publication=pd.get_dummies(train["Publication_Day"],dtype=int)

train_publication


train_pub_time= pd.get_dummies(train["Publication_Time"],dtype=int)
train_pub_time


train_publication.reset_index(drop=True, inplace=True)
train_pub_time.reset_index(drop=True, inplace=True)
train_con.reset_index(drop=True,inplace=True)


train_conc=pd.concat([train_con,train_publication,train_pub_time],axis=1)
train_conc.head()


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()






train_conc["Episode_Sentiment"]=encoder.fit_transform(train_conc['Episode_Sentiment'])



train_conc=train_conc.drop(columns=["Genre","Publication_Day","Publication_Time"])
train_conc.info()


train_conc.info()



train_conc.head()


freq_map = train_conc["Podcast_Name"].value_counts(normalize=True).to_dict()
train_conc["Podcast_Name_Freq"] = train_conc["Podcast_Name"].map(freq_map)


train_conc["Podcast_Name_Freq"]



train_conc['episode_number'] = train_conc['Episode_Title'].str.extract('(\d+)').astype(int)
train_conc['episode_number']


train_conc


train_conc.drop(columns=['Podcast_Name', 'Episode_Title'], inplace=True)


train_conc.info()


test.info()



test["Episode_Sentiment"]=encoder.fit_transform(test['Episode_Sentiment'])


freq_map = test["Podcast_Name"].value_counts(normalize=True).to_dict()
test["Podcast_Name_Freq"] = test["Podcast_Name"].map(freq_map)


test['episode_number'] = test['Episode_Title'].str.extract('(\d+)').astype(int)
test['episode_number']


test.info()


test_day_publication=pd.get_dummies(test["Publication_Day"],dtype=int)

test_day_publication


test_publication=pd.get_dummies(test["Publication_Time"],dtype=int)

test_publication


test_publication.reset_index(drop=True, inplace=True)
test_day_publication.reset_index(drop=True, inplace=True)
test.reset_index(drop=True,inplace=True)





genres = test[['Genre']]  

encoder = OneHotEncoder(sparse_output=False, drop='first') 
encoded_genres = encoder.fit_transform(genres)


genre_encoded_df = pd.DataFrame(
    encoded_genres,
    columns=encoder.get_feature_names_out(['Genre']))
genre_encoded_df=genre_encoded_df.astype(int)
genre_encoded_df.head()


genre_encoded_df.reset_index(drop=True,inplace=True)
test.reset_index(drop=True,inplace=True)



test_conc=pd.concat([test,test_publication,test_day_publication,genre_encoded_df],axis=1)
test_conc.head()


test_conc.info()


test_conc.drop(columns=['Podcast_Name', 'Episode_Title','Genre'], inplace=True)


test_conc.drop(columns=['Publication_Day', 'Publication_Time'], inplace=True)


test_conc['Number_of_Ads'] = test_conc['Number_of_Ads'].astype(int)


mean_value = train_conc['Number_of_Ads'].mean()
train_conc['Number_of_Ads'].fillna(mean_value, inplace=True)
train_conc['Number_of_Ads'] = train_conc['Number_of_Ads'].astype(int)


test_conc.info()


test_conc.shape


train_conc.shape


train_conc.info()





import seaborn as sns
import matplotlib.pyplot as plt



numerical_columns = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes', 'Podcast_Name_Freq', 'episode_number']



plt.figure(figsize=(14, 8))
sns.boxplot(data=train_conc[numerical_columns])
plt.title('Boxplot of Numerical Features')
plt.xticks(rotation=90)
plt.show()


correlation_matrix = train_conc[numerical_columns].corr()

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# Plotting the sentiment distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Episode_Sentiment', data=train_conc)
plt.title('Distribution of Episode Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# Day of the week distribution
days_of_week = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']

plt.figure(figsize=(10, 6))
train_conc[days_of_week].sum().plot(kind='bar', color='lightgreen')
plt.title('Distribution of Podcasts by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


from sklearn.preprocessing import StandardScaler


X_train = train_conc.drop(columns=['Listening_Time_minutes'])
y_train = train_conc['Listening_Time_minutes']

X_test = test_conc


X_test = X_test[X_train.columns]




scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train)


X_test_scaled = scaler.transform(X_test)



X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


X_train_scaled 
X_test_scaled







