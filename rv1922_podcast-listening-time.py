import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from itertools import combinations
from scipy.stats import gmean, hmean
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
h2o.init()


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train.head()


cat_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']


genre_mapping = {
    'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4,
    'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9
}

publication_day_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

publication_time_mapping = {
    'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3
}

episode_sentiment_mapping = {
    'Positive': 0, 'Negative': 1, 'Neutral': 2
}


podcast_name_mapping = {
    'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3,
    'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7,
    'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12,
    'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17,
    'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21,
    'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25,
    "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30,
    'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34,
    'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38,
    'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42,
    'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46,
    'Tune Time': 47
}


train["Episode_Title"] = train["Episode_Title"].str.replace("Episode ", "", regex=False).astype(int)
test["Episode_Title"] = test["Episode_Title"].str.replace("Episode ", "", regex=False).astype(int)


train['Genre'] = train['Genre'].map(genre_mapping)
test['Genre'] = test['Genre'].map(genre_mapping)

train['Publication_Day'] = train['Publication_Day'].map(publication_day_mapping)
test['Publication_Day'] = test['Publication_Day'].map(publication_day_mapping)

train['Publication_Time'] = train['Publication_Time'].map(publication_time_mapping)
test['Publication_Time'] = test['Publication_Time'].map(publication_time_mapping)

train['Episode_Sentiment'] = train['Episode_Sentiment'].map(episode_sentiment_mapping)
test['Episode_Sentiment'] = test['Episode_Sentiment'].map(episode_sentiment_mapping)

train['Podcast_Name'] = train['Podcast_Name'].map(podcast_name_mapping)
test['Podcast_Name'] = test['Podcast_Name'].map(podcast_name_mapping)


train.head()


train = h2o.H2OFrame(train)


test = h2o.H2OFrame(test)


aml = H2OAutoML(max_runtime_secs=2000,seed=42)
aml.train(y='Listening_Time_minutes', training_frame=train)


leaderboard = aml.leaderboard
print(leaderboard)
best_model = aml.leader
print(best_model)


test = h2o.H2OFrame(test)


predictions = best_model.predict(test)
predictions_df = predictions.as_data_frame()


submission['Listening_Time_minutes'] =(predictions_df['predict'].values)


submission.head()
submission.to_csv('submission.csv', index=False)
print("File Saved!!")


submission.head()

