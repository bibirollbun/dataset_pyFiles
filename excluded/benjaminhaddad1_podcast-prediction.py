# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import xgboost as xgb
from xgboost import XGBClassifier
#imports xgboost 
from sklearn.metrics import accuracy_score, classification_report


import matplotlib.pyplot as plt


train_set = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
train_set.describe()
#looks at data based on purely numerical catagories


 test_set = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


episode_number = train_set['Episode_Title']
episode_numbers = episode_number.str.extract(r'(\d+)', expand=False)


podcast_map = {
    "Athlete's Arena": 0,
    "Brain Boost": 1,
    "Business Briefs": 2,
    "Business Insights": 3,
    "Comedy Corner": 4,
    "Crime Chronicles": 5,
    "Criminal Minds": 6,
    "Current Affairs": 7,
    "Daily Digest": 8,
    "Detective Diaries": 9,
    "Digital Digest": 10,
    "Educational Nuggets": 11,
    "Fashion Forward": 12,
    "Finance Focus": 13,
    "Fitness First": 14,
    "Funny Folks": 15,
    "Gadget Geek": 16,
    "Game Day": 17,
    "Global News": 18,
    "Health Hour": 19,
    "Healthy Living": 20,
    "Home & Living": 21,
    "Humor Hub": 22,
    "Innovators": 23,
    "Joke Junction": 24,
    "Laugh Line": 25,
    "Learning Lab": 26,
    "Life Lessons": 27,
    "Lifestyle Lounge": 28,"Market Masters": 29,"Melody Mix": 30,"Mind & Body": 31,
    "Money Matters": 32,"Music Matters": 33,"Mystery Matters": 34,"News Roundup": 35,
    "Sound Waves": 36,"Sport Spot": 37,"Sports Central": 38,"Sports Weekly": 39,
    "Study Sessions": 40,"Style Guide": 41,"Tech Talks": 42,"Tech Trends": 43,
    "True Crime Stories": 44,"Tune Time": 45,"Wellness Wave": 46,"World Watch": 47
}



#maps out non-numerical values to numerical data
names_set = train_set['Podcast_Name'].map(podcast_map).astype('category')
test_names_set = test_set['Podcast_Name'].map(podcast_map).astype('category')

genre_order = {'Business':0, 'Comedy':1,'Education':2,'Health' :3,'Lifestyle':4,'Music': 5,
               'News':6, 'Sports':7,'Technology':8,'True Crime':9}
podcast_genre = train_set['Genre'].map(genre_order).astype('category')
test_podcast_genre = test_set['Genre'].map(genre_order).astype('category')

day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
             'Friday': 4, 'Saturday': 5, 'Sunday': 6}
day_published = train_set['Publication_Day'].map(day_order).astype('category')

test_day_published = test_set['Publication_Day'].map(day_order).astype('category')

time_order = {'Morning':0,'Afternoon':1,'Evening':2,'Night':3}
time_published = train_set['Publication_Time'].map(time_order).astype('category')

test_time_published = test_set['Publication_Time'].map(time_order).astype('category')

sentiment_order = {'Negative': 0, 'Neutral':1, 'Positive': 2}

episode_sentiment = train_set['Episode_Sentiment'].map(sentiment_order).astype('category')
test_episode_sentiment = test_set['Episode_Sentiment'].map(sentiment_order).astype('category')




length_minutes = train_set['Episode_Length_minutes']
number_ads = train_set['Number_of_Ads'].astype('category')
test_number_ads = test_set['Number_of_Ads'].astype('category')
episode_ids = train_set['id']



result_popularity = train_set['Host_Popularity_percentage'] + train_set['Guest_Popularity_percentage']
result_popularity = result_popularity.fillna(train_set['Host_Popularity_percentage'] )
result_popularity.head()


# Fills in the blank values in train set for episode lengths part 1 creating the averages for each show
avg_completion_df = train_set.groupby('Podcast_Name').apply(
    lambda group: pd.Series({
        'avg_completion_ratio': (
            group.dropna(subset=['Episode_Length_minutes'])['Listening_Time_minutes'] / 
            group.dropna(subset=['Episode_Length_minutes'])['Episode_Length_minutes']
        ).mean(),
        'total_time_listened': group['Listening_Time_minutes'].sum(),
        'avg_episode_length': group['Episode_Length_minutes'].mean(skipna=True),
    })
).reset_index()



# Fills in the blank values in train set for episode lengths part 2 adding the averaged values to a new dataframe
avg_completion_df['imputed_length'] = avg_completion_df['avg_completion_ratio'] * avg_completion_df['avg_episode_length']

train_set_filled = train_set.copy()

train_set_filled = train_set_filled.merge(
    avg_completion_df[['Podcast_Name', 'imputed_length']], 
    on='Podcast_Name', 
    how='left'
)


#fills in new values into the length_minutes section
train_set_filled['Episode_Length_minutes'] = train_set_filled['Episode_Length_minutes'].fillna(
    train_set_filled['imputed_length']
)
median_length = train_set_filled['Episode_Length_minutes'].median()
train_set_filled['Episode_Length_minutes'] = train_set_filled['Episode_Length_minutes'].fillna(median_length)
length_minutes = train_set_filled['Episode_Length_minutes'].astype('category')


#just gets rid of the temporary data set
del train_set_filled
gc.collect()


#I can put the number of ads in because there aren't half adds
X = pd.DataFrame({
    'Genre': podcast_genre,
    'Publication_Day': day_published,
    'Publication_Time': time_published,
    'Episode_Sentiment': episode_sentiment,
    'Number_of_Ads': number_ads,
    'Podcast_Name': names_set
})
y = length_minutes
#using 42 as it is the answer to the universe and 0.35 because the data is large enough
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)


# For checking null values in categorical data
print("NaN values in y_train:", y_train.isna().sum())

# For cleaning your target variable
# 1. First identify rows without missing values
good_indices = ~y_train.isna()

# 2. Filter your data
X_train_clean = X_train[good_indices]
y_train_clean = y_train[good_indices]


if hasattr(y_train, 'replace'):  # For pandas Series
    y_train_clean = y_train.replace([np.inf, -np.inf], np.nan)
else:  # For numpy arrays
    y_train_clean = np.where(np.isinf(y_train), np.nan, y_train)

# Drop rows with NaN in target
valid_mask = ~pd.isna(y_train_clean)
X_train_clean = X_train[valid_mask]
y_train_clean = y_train_clean[valid_mask]

print(f"Remaining rows after cleaning: {len(y_train_clean)}")



y_train.describe()


X_train.describe()


#trains model non numerical data
catagorical_model = xgb.XGBRegressor(tree_method="hist", enable_categorical=True, device="cuda")


catagorical_model.fit(X_train_clean, y_train_clean)


test_X_nonnumerical = pd.DataFrame({
    'Genre': test_podcast_genre,
    'Publication_Day': test_day_published,
    'Publication_Time': test_time_published,
    'Episode_Sentiment': test_episode_sentiment,
    'Number_of_Ads': test_number_ads,
    'Podcast_Name':test_names_set
})
y_pred_test = catagorical_model.predict(test_X_nonnumerical)



predictions = pd.DataFrame()


train_set.describe()


time_listened = train_set['Listening_Time_minutes']

randomForestTrain_X = pd.DataFrame({'Episode_Length_minutes':length_minutes,
                                  'Result_Popularity':result_popularity,
                                  'Episode_Numbers':episode_numbers,
                                 })
randomForestTrain_Y = time_listened
 #episode_numbers, length_minutes, result_popularity, number_ads


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=250)
model.fit(randomForestTrain_X, randomForestTrain_Y)

