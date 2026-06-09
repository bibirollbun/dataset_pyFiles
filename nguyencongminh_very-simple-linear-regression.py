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
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.model_selection import train_test_split


train_dataset = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_dataset = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_dataset


train_dataset['train'] = 1
test_dataset['train'] = 0



combined_dataset = pd.concat([train_dataset, test_dataset])


train_dataset[train_dataset['Guest_Popularity_percentage'].isna()]


combined_dataset = combined_dataset.drop(columns = ['Podcast_Name', 'Episode_Title', 'Publication_Day'])


combined_dataset 





train_dataset[train_dataset['Number_of_Ads'].isna()]


test_dataset[test_dataset['Number_of_Ads'].isna()]


test_dataset[test_dataset['Episode_Length_minutes'].isna()]


train_dataset[train_dataset['Episode_Sentiment'].isnull()]


train_dataset[train_dataset['Listening_Time_minutes'].isna()]


train_dataset[train_dataset['Publication_Time'].isna()]


list_of_genre = combined_dataset['Genre'].unique()


list_of_genre


# listForPublicDay = combined_dataset['Publication_Day'].unique()
# len(listForPublicDay)


listForPublicTime = combined_dataset['Publication_Time'].unique()


listForPublicTime


listForNumberOfAds = combined_dataset['Number_of_Ads'].unique()


listForNumberOfAds


listForSentiment = train_dataset['Episode_Sentiment'].unique()


# calculate average Episode_Length_minutes based on the genre
mean_of_episode_minute = []
for genre in list_of_genre:
    mean_needed = combined_dataset[combined_dataset['Genre'] == genre]['Episode_Length_minutes'].mean()
    mean_of_episode_minute.append(mean_needed)


mean_of_episode_minute


#calculate average Guest_Popularity_percentage based on the genre
mean_of_guest_popularity = []
for genre in list_of_genre:
    mean_needed = combined_dataset[combined_dataset['Genre'] == genre]['Guest_Popularity_percentage'].mean()
    mean_of_guest_popularity.append(mean_needed)
mean_of_guest_popularity


# calculate average num_of_ads based on the genre 
mean_of_num_of_ads = []
for genre in list_of_genre:
    combined_needed = combined_dataset[combined_dataset['Genre'] == genre]['Number_of_Ads'].mean()
    mean_of_num_of_ads.append(mean_needed)
mean_of_num_of_ads


# calculate average Host_Popularity_percentage based on genre	
mean_of_host_popularity = []
for genre in list_of_genre:
    mean_needed = combined_dataset[combined_dataset['Genre'] == genre]['Host_Popularity_percentage'].mean()
    mean_of_host_popularity.append(mean_needed)
mean_of_host_popularity


# filling the nan value by mean group by genre
combined_dataset['Episode_Length_minutes'] = combined_dataset['Episode_Length_minutes'].fillna(
    combined_dataset['Genre'].map(dict(zip(list_of_genre, mean_of_episode_minute)))
)

combined_dataset['Guest_Popularity_percentage'] = combined_dataset['Guest_Popularity_percentage'].fillna(
    combined_dataset['Genre'].map(dict(zip(list_of_genre, mean_of_guest_popularity)))
)

combined_dataset['Number_of_Ads'] = combined_dataset['Number_of_Ads'].fillna(
    train_dataset['Genre'].map(dict(zip(list_of_genre, mean_of_num_of_ads)))
)

combined_dataset['Host_Popularity_percentage'] = combined_dataset['Host_Popularity_percentage'].fillna(
    combined_dataset['Genre'].map(dict(zip(list_of_genre, mean_of_host_popularity)))
)



combined_dataset


# feature engineering for better regression
combined_dataset['Guest_Host_Ratio'] = combined_dataset['Guest_Popularity_percentage'] / (combined_dataset['Host_Popularity_percentage'] + 1e-6)


combined_dataset


combined_dataset['Ads_Length_Ratio'] = combined_dataset['Number_of_Ads'] / (combined_dataset['Episode_Length_minutes'] + 1e-6)


combined_dataset


list_of_genre


dict2genre = {}

for i in range(len(list_of_genre)):
    dict2genre[list_of_genre[i]] = i

# dict2publicationDay = {}
# # for i in range(len(listForPublicDay)):
# #     dict2publicationDay[listForPublicDay[i]] = i
dict2publicationTime = {}
for i in range(len(listForPublicTime)):
    dict2publicationTime[listForPublicTime[i]] = i

dict2sentiment = {}
for i in range(len(listForSentiment)):
    dict2sentiment[listForSentiment[i]] = i


# x['Genre'] = x['Genre'].map(dict2genre)
# # x['Publication_Day'] = x['Publication_Day'].map(dict2publicationDay)
# x['Publication_Time'] = x['Publication_Time'].map(dict2publicationTime)
# x['Episode_Sentiment'] = x['Episode_Sentiment'].map(dict2sentiment)


combined_dataset['Genre'] = combined_dataset['Genre'].map(dict2genre)
# combined_dataset['Publication_Day'] = train_dataset['Publication_Day'].map(dict2publicationDay)
combined_dataset['Publication_Time'] = combined_dataset['Publication_Time'].map(dict2publicationTime)
combined_dataset['Episode_Sentiment'] = combined_dataset['Episode_Sentiment'].map(dict2sentiment)


dict2genre


combined_dataset


# split back to train and test
train_dataset = combined_dataset[combined_dataset['train'] == 1]
test_dataset = combined_dataset[combined_dataset['train'] == 0]


train_dataset


test_dataset


x_train = train_dataset[['Episode_Length_minutes', 'Genre', 'Host_Popularity_percentage', 'Publication_Time', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Guest_Host_Ratio', 'Ads_Length_Ratio']]
y_train = train_dataset['Listening_Time_minutes']


x_train


# x_train[x_train['Ads_Length_Ratio'].isna()]
# x_train[x_train['Ads_Length_Ratio'].isna()]['Ads_Length_Ratio'] = 0


# x_train[x_train['Ads_Length_Ratio'].isna()]



# x = x[['Episode_Length_minutes', 'Number_of_Ads']]


x_train = x_train.to_numpy()
y_train = y_train.to_numpy()


x_train





# feature_for_episodelength = x[:, 1]
# import matplotlib.pyplot as plt


# # Scatter plot
# plt.scatter(feature_for_episodelength, y, color='blue', label='Data Points')
# plt.xlabel("Feature (X)")
# plt.ylabel("Target (y)")
# plt.title("Scatter Plot of episode Length vs. Target")
# plt.legend()
# plt.show()


# feature_for_episodelength = x[:, 2]
# import matplotlib.pyplot as plt


# # Scatter plot
# plt.scatter(feature_for_episodelength, y, color='blue', label='Data Points')
# plt.xlabel("Feature (X)")
# plt.ylabel("Target (y)")
# plt.title("Scatter Plot of genre and Target")
# plt.legend()
# plt.show()



# feature_for_episodelength = x[:, 3]
# import matplotlib.pyplot as plt


# # Scatter plot
# plt.scatter(feature_for_episodelength, y, color='blue', label='Data Points')
# plt.xlabel("Feature (X)")
# plt.ylabel("Target (y)")
# plt.title("Scatter Plot of Host Popular Percentage vs. Target")
# plt.legend()
# plt.show()



# # Publication_Day
# feature_for_episodelength = x[:, 4]
# import matplotlib.pyplot as plt


# # Scatter plot
# plt.scatter(feature_for_episodelength, y, color='blue', label='Data Points')
# plt.xlabel("Feature (X)")
# plt.ylabel("Target (y)")
# plt.title("Scatter Plot of publication day vs. Target")
# plt.legend()
# plt.show()



# # Publication_Time
# feature_for_episodelength = x[:, 5]
# import matplotlib.pyplot as plt


# # Scatter plot
# plt.scatter(feature_for_episodelength, y, color='blue', label='Data Points')
# plt.xlabel("Feature (X)")
# plt.ylabel("Target (y)")
# plt.title("Scatter Plot of publication time vs. Target")
# plt.legend()
# plt.show()



# # Guest_Popularity_percentage

# feature_for_episodelength = x[:, 6]
# import matplotlib.pyplot as plt


# # Scatter plot
# plt.scatter(feature_for_episodelength, y, color='blue', label='Data Points')
# plt.xlabel("Feature (X)")
# plt.ylabel("Target (y)")
# plt.title("Scatter Plot of guest popularity percentage vs. Target")
# plt.legend()
# plt.show()



# # Number_of_Ads
# feature_for_episodelength = x[:, 7]
# import matplotlib.pyplot as plt


# # Scatter plot
# plt.scatter(feature_for_episodelength, y, color='blue', label='Data Points')
# plt.xlabel("Feature (X)")
# plt.ylabel("Target (y)")
# plt.title("Scatter Plot of number of ads vs. Target")
# plt.legend()
# plt.show()



# # Episode_Sentiment
# feature_for_episodelength = x[:, 8]
# import matplotlib.pyplot as plt


# # Scatter plot
# plt.scatter(feature_for_episodelength, y, color='blue', label='Data Points')
# plt.xlabel("Feature (X)")
# plt.ylabel("Target (y)")
# plt.title("Scatter Plot of sentiment vs. Target")
# plt.legend()
# plt.show()



# feature_for_episodelength = x[:, 0]
# import matplotlib.pyplot as plt


# # Scatter plot
# plt.scatter(feature_for_episodelength, y, color='blue', label='Data Points')
# plt.xlabel("Feature (X)")
# plt.ylabel("Target (y)")
# plt.title("Scatter Plot of episode title vs. Target")
# plt.legend()
# plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# # Compute correlation matrix
# train_dataset = train_dataset.drop(columns=['id', 'Podcast_Name'])

# corr_matrix = train_dataset.corr()

# # Plot heatmap
# plt.figure(figsize=(6,4))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Feature Correlation Heatmap")
# plt.show()



y_train


import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error



# Initialize and fit the linear regression model
model = LinearRegression() 
model.fit(x_train, y_train)

# Predict the output
y_pred = model.predict(x_train)

# Calculate RMSE manually
rmse = np.sqrt(mean_squared_error(y_train, y_pred))

# Print results
print(f"Model Coefficients: {model.coef_}, Intercept: {model.intercept_}")
print(f"RMSE: {rmse}")



# submission
test_dataset
test_id = test_dataset['id'].unique()



x_test = test_dataset[['Episode_Length_minutes', 'Genre', 'Host_Popularity_percentage', 'Publication_Time', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Guest_Host_Ratio', 'Ads_Length_Ratio']]



x_test


x_test



x_test = x_test.to_numpy()


y_test_pred = model.predict(x_test)


y_test_pred


submit_df = pd.DataFrame({
    'id': test_id,
    'Listening_Time_minutes': y_test_pred
})



submit_df.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
submission




