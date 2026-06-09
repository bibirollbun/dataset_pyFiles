# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col=0)
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col=0)

# Merge train + test
merge_test = test.copy()
merge_test['Listening_Time_minutes'] = np.nan
full_data = pd.concat([train, merge_test])

# Podcast Name Word Features
ignore_words = ['&']
replace_word = {'Sports': 'Sport', 'Waves': 'Wave', 'Minds': 'Mind', 'Healthy': 'Health', 'Lifestyle': 'Life', "Athlete's": "Athlete"}
words = set([replace_word.get(w, w) for name in full_data['Podcast_Name'].unique() for w in name.split() if w not in ignore_words])

for w in tqdm(words):
    full_data[f'Podcast_Name_Contains_{w}'] = full_data['Podcast_Name'].str.contains(w)

# Dummy Encoding for Podcast Name
full_data['Podcast_Name_Dummy'] = full_data['Podcast_Name']
full_data = pd.get_dummies(full_data, columns=['Podcast_Name_Dummy'])

# Episode Number Features
full_data['Episode_Num'] = full_data['Episode_Title'].str[8:].astype(int)
for st in ['quantile', 'uniform', 'kmeans']:
    binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy=st)
    full_data[f'EpNum_{st}_5bin'] = binner.fit_transform(full_data[['Episode_Num']])
full_data['EpNum_ordinal_encoded'] = OrdinalEncoder().fit_transform(full_data[['Episode_Num']])

# Episode Length Features
full_data['Podcast_Episode_Length_mean'] = full_data.groupby('Podcast_Name')['Episode_Length_minutes'].transform('mean')
full_data['Podcast_Episode_Length_median'] = full_data.groupby('Podcast_Name')['Episode_Length_minutes'].transform('median')

full_data.loc[101637, 'Episode_Length_minutes'] = 125
full_data.loc[804434, 'Episode_Length_minutes'] = 132
full_data.loc[806597, 'Episode_Length_minutes'] = 146

podcasts_grouped = full_data.groupby('Podcast_Name')['Episode_Length_minutes']
full_data['EpLenMins_median_filled'] = podcasts_grouped.transform(lambda x: x.fillna(x.median()))
full_data['Episode_Length_minutes'] = podcasts_grouped.transform(lambda x: x.fillna(x.mean()))

for st in ['quantile', 'uniform', 'kmeans']:
    for bn in [5, 10, 20]:
        binner = KBinsDiscretizer(n_bins=bn, encode='ordinal', strategy=st)
        full_data[f'EpLenMins_{st}_{bn}bin'] = binner.fit_transform(full_data[['Episode_Length_minutes']])

full_data['EpLenMins_ordinal_encoded'] = OrdinalEncoder().fit_transform(full_data[['Episode_Length_minutes']])
full_data['EpLenMins_log2'] = np.log2(1 + full_data['Episode_Length_minutes'])
full_data['EpLenMins_sqrt'] = np.sqrt(1 + full_data['Episode_Length_minutes'])

# Genre Encoding
full_data = pd.get_dummies(full_data, columns=['Genre'])

# Host Popularity
full_data['Podcast_Host_Popularity_mean'] = full_data.groupby('Podcast_Name')['Host_Popularity_percentage'].transform('mean')
full_data['Podcast_Host_Popularity_median'] = full_data.groupby('Podcast_Name')['Host_Popularity_percentage'].transform('median')

for st in ['quantile', 'uniform', 'kmeans']:
    for bn in [5, 10, 20]:
        binner = KBinsDiscretizer(n_bins=bn, encode='ordinal', strategy=st)
        full_data[f'HostPopPrcnt_{st}_{bn}bin'] = binner.fit_transform(full_data[['Host_Popularity_percentage']])

full_data['HostPopPrcnt_ordinal_encoded'] = OrdinalEncoder().fit_transform(full_data[['Host_Popularity_percentage']])

# Publication Day Features
full_data['Is_Weekend'] = full_data['Publication_Day'].isin(['Saturday', 'Sunday'])
full_data['Publication_Day_Dummy'] = full_data['Publication_Day']
full_data = pd.get_dummies(full_data, columns=['Publication_Day_Dummy'])
full_data['Publication_Day'] = full_data['Publication_Day'].map({'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7})

# Publication Time Features
full_data['Publication_Time_Dummy'] = full_data['Publication_Time']
full_data = pd.get_dummies(full_data, columns=['Publication_Time_Dummy'])
full_data['Publication_Time'] = full_data['Publication_Time'].map({'Morning':1, 'Afternoon':2, 'Evening':3, 'Night':4})

# Guest Popularity
full_data['Have_Guest'] = ~full_data['Guest_Popularity_percentage'].isna()
full_data['Podcast_Guest_Popularity_mean'] = full_data.groupby('Podcast_Name')['Guest_Popularity_percentage'].transform('mean')
full_data['Podcast_Guest_Popularity_median'] = full_data.groupby('Podcast_Name')['Guest_Popularity_percentage'].transform('median')

podcasts_grouped = full_data.groupby('Podcast_Name')['Guest_Popularity_percentage']
full_data['Guest_Popularity_percentage'] = podcasts_grouped.transform(lambda x: x.fillna(x.mean()))
full_data['HostPopPrcnt_median_filled'] = podcasts_grouped.transform(lambda x: x.fillna(x.median()))

for st in ['quantile', 'uniform', 'kmeans']:
    for bn in [5, 10, 20]:
        binner = KBinsDiscretizer(n_bins=bn, encode='ordinal', strategy=st)
        full_data[f'GuestPopPrcnt_{st}_{bn}bin'] = binner.fit_transform(full_data[['Guest_Popularity_percentage']])

full_data['GuestPopPrcnt_ordinal_encoded'] = OrdinalEncoder().fit_transform(full_data[['Guest_Popularity_percentage']])

# Ads Feature Engineering
num_of_ads = full_data['Number_of_Ads']
full_data['Number_of_Ads'] = full_data['Number_of_Ads'].fillna(0)
full_data.loc[num_of_ads > 3, 'Number_of_Ads'] = 4

full_data['Have_Ads'] = full_data['Number_of_Ads'] > 0
full_data['Ads_One'] = full_data['Number_of_Ads'] == 1
full_data['Ads_Two'] = full_data['Number_of_Ads'] == 2
full_data['Ads_Three'] = full_data['Number_of_Ads'] == 3
full_data['Ads_Four'] = full_data['Number_of_Ads'] == 4
full_data['Ads_atleast_One'] = full_data['Number_of_Ads'] >= 1
full_data['Ads_atleast_Two'] = full_data['Number_of_Ads'] >= 2
full_data['Ads_atleast_Three'] = full_data['Number_of_Ads'] >= 3

# Episode Sentiment
full_data['Sentiment_Num'] = full_data['Episode_Sentiment'].map({'Negative': -1, 'Neutral': 0, 'Positive': 1})
full_data = pd.get_dummies(full_data, columns=['Episode_Sentiment'])

full_data['Podcast_Sentiment_Num_mean'] = full_data.groupby('Podcast_Name')['Sentiment_Num'].transform('mean')
full_data['Podcast_Sentiment_Num_median'] = full_data.groupby('Podcast_Name')['Sentiment_Num'].transform('median')

# Target Aggregation
full_data['Podcast_target_mean'] = full_data.groupby('Podcast_Name')['Listening_Time_minutes'].transform('mean')
full_data['Podcast_target_median'] = full_data.groupby('Podcast_Name')['Listening_Time_minutes'].transform('median')

# Separate back to Train/Test
train_features = full_data.loc[train.index].drop(['Podcast_Name', 'Episode_Title', 'Listening_Time_minutes'], axis=1)
train_target = full_data.loc[train.index, 'Listening_Time_minutes']
test_features = full_data.loc[test.index].drop(['Podcast_Name', 'Episode_Title', 'Listening_Time_minutes'], axis=1)

# Train/Validation Split
X_train, X_valid, y_train, y_valid = train_test_split(train_features, train_target, test_size=0.2, random_state=42)

# Model
model = LGBMRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=100)])

# Validation Prediction
val_preds = model.predict(X_valid)
print("Validation RMSE:", mean_squared_error(y_valid, val_preds, squared=False))

# Final Train on Full Data
model_final = LGBMRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
model_final.fit(train_features, train_target)

# Test Prediction
test_preds = model_final.predict(test_features)

# Submission
submission = sample_submission.copy()
submission['Listening_Time_minutes'] = test_preds
submission.to_csv('submission.csv')

