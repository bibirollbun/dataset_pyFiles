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


import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.ticker import NullFormatter

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as rf

import warnings
warnings.filterwarnings('ignore')


# tf.random.set_seed(42)
np.random.seed(42)


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train_df = train.copy()
test_df = test.copy()


train_df.head(3)


test_df.head(3)


submission.head(2)


train_df.info()


test_df.info()


train_df.isnull().sum()


test_df.isnull().sum()


train_df.describe().T


test_df.describe().T


# Outlier: Episode_Length_minutes

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
train_df.Episode_Length_minutes.plot.box(ax=ax[0])
test_df.Episode_Length_minutes.plot.box(ax=ax[1])


# Check for duplicates in the datasets

print(f'Training data duplicates: {train_df.duplicated().sum()}')
print(f'Test data duplicates    :  {test_df.duplicated().sum()}')


# Count number of distinct elements in specified axis

print('Training data:')
for col in train_df.columns:
    print(' ', col, ':', train_df[col].nunique())

print('')
print('Test data:')
for col in test_df.columns:
    print(' ', col, ':', test_df[col].nunique())


# Check for difference values between training data and test data.

print('Values of Podcast_Name:     ', set(train_df.Podcast_Name.unique()) == set(test_df.Podcast_Name.unique()))
print('Values of Episode_Title:    ', set(train_df.Episode_Title.unique()) == set(test_df.Episode_Title.unique()))
print('Values of Genre:            ', set(train_df.Genre.unique()) == set(test_df.Genre.unique()))
print('Values of Publication_Day:  ', set(train_df.Publication_Day.unique()) == set(test_df.Publication_Day.unique()))
print('Values of Publication_Time: ', set(train_df.Publication_Time.unique()) == set(test_df.Publication_Time.unique()))
print('Values of Number_of_Ads:    ', set(train_df.Number_of_Ads.unique()) == set(test_df.Number_of_Ads.unique()))
print('Values of Episode_Sentiment:', set(train_df.Episode_Sentiment.unique()) == set(test_df.Episode_Sentiment.unique()))



# Number of types of 'Number_of_Ads'  train:12, test:6 
#                                       ... Pre-Roll, Mid-Roll, Post-Roll, (Bumper Ad ?)
# Counts of unique values of 'Number_of_Ads'

print('Training data')
print(train_df.Number_of_Ads.value_counts(dropna=False), '\n')
print('Test data')
print(test_df.Number_of_Ads.value_counts(dropna=False))


# Unique values in specified axis. 

podcast_names = list(train_df.Podcast_Name.unique())
episode_titles = list(train_df.Episode_Title.unique())
genres = list(train_df.Genre.unique())

print('Number of unique Podcast_Name: 48\n', sorted(podcast_names), '\n')
print('Number of unique Episode_Title: 100\n', sorted(episode_titles), '\n')
print('Number of unique Genre: 10\n', sorted(genres))


# Unique values in specific columns. 

public_days = list(train_df.Publication_Day.unique())
public_times = list(train_df.Publication_Time.unique())
episode_sentis = list(train_df.Episode_Sentiment.unique())

print('Number of unique Publication_Day:   7\n', public_days, '\n')
print('Number of unique Publication_Time:  4\n', public_times, '\n')
print('Number of unique Episode_Sentiment: 3\n', episode_sentis, '\n')


# Count null and non-null values

print('----- Training data -----')
print('Episode_Length_minutes are null:     ', 
      train_df.loc[(train_df.Episode_Length_minutes.isnull()) & (~train_df.Guest_Popularity_percentage.isnull()), :].shape[0])
print('Guest_Popularity_percentage are null:',
      train_df.loc[(~train_df.Episode_Length_minutes.isnull()) & (train_df.Guest_Popularity_percentage.isnull()),:].shape[0])
print('Both null:                           ',
      train_df.loc[(train_df.Episode_Length_minutes.isnull()) & (train_df.Guest_Popularity_percentage.isnull()), :].shape[0])
print('Non-null:                            ',
      train_df.loc[(~train_df.Episode_Length_minutes.isnull()) & (~train_df.Guest_Popularity_percentage.isnull()),:].shape[0], '\n')

print('----- Test data -----')
print('Episode_Length_minutes are null     :',
      test_df.loc[(test_df.Episode_Length_minutes.isnull()) & (~test_df.Guest_Popularity_percentage.isnull()), :].shape[0])
print('Guest_Popularity_percentage are null:', 
      test_df.loc[(~test_df.Episode_Length_minutes.isnull()) & (test_df.Guest_Popularity_percentage.isnull()),:].shape[0])
print('Both null:                           ',
      test_df.loc[(test_df.Episode_Length_minutes.isnull()) & (test_df.Guest_Popularity_percentage.isnull()), :].shape[0])
print('Non-null:                            ',
      test_df.loc[(~test_df.Episode_Length_minutes.isnull()) & (~test_df.Guest_Popularity_percentage.isnull()),:].shape[0])



train_df.columns


# Add features: 'null_count', 'null_kinds'

train_df['null_count'] = train_df.isnull().sum(axis=1).astype('int8')
train_df['null_kinds'] = 0
train_df.loc[(train_df.null_count == 1) & (train_df.Episode_Length_minutes.isnull()), 'null_kinds'] = 1
train_df.loc[(train_df.null_count == 1) & (train_df.Guest_Popularity_percentage.isnull()), 'null_kinds'] = 2
train_df.loc[(train_df.null_count == 2), 'null_kinds'] = 3

test_df['null_count'] = test_df.isnull().sum(axis=1).astype('int8')
test_df['null_kinds'] = 0
test_df.loc[(test_df.null_count == 1) & (test_df.Episode_Length_minutes.isnull()), 'null_kinds'] = 1
test_df.loc[(test_df.null_count == 1) & (test_df.Guest_Popularity_percentage.isnull()), 'null_kinds'] = 2
test_df.loc[(test_df.null_count == 2), 'null_kinds'] = 3



# Ratio: missing values

print('Training data')
print('Count of missing values = 0:', np.round(len(train_df.query('null_count == 0')) / len(train_df), 2) * 100, '%')
print('Count of missing values = 1:', np.round(len(train_df.query('null_count == 1')) / len(train_df), 2) * 100, '%')
print('Count of missing values = 2:', np.round(len(train_df.query('null_count == 2')) / len(train_df), 2) * 100, '%')

print('Test data')
print('Count of missing values = 0:', np.round(len(test_df.query('null_count == 0')) / len(test_df), 2) * 100, '%')
print('Count of missing values = 1:', np.round(len(test_df.query('null_count == 1')) / len(test_df), 2) * 100, '%')
print('Count of missing values = 2:', np.round(len(test_df.query('null_count == 2')) / len(test_df), 2) * 100, '%')



# Ratio: missing values of 'Episode_Length_minutes', 'Guest_Popularity_percentage'

print('Training data')
print('Count of missing values are zero:', np.round(len(train_df.query('null_kinds == 0')) / len(train_df), 2) * 100, '%')
print('Episode_Length_minutes:           ', np.round(len(train_df.query('null_kinds == 1')) / len(train_df), 2) * 100, '%')
print('Guest_Popularity_percentage:     ', np.round(len(train_df.query('null_kinds == 2')) / len(train_df), 2) * 100, '%')
print('Both:                             ', np.round(len(train_df.query('null_kinds == 3')) / len(train_df), 2) * 100, '%', '\n')
print('Test data')
print('Count of missing values are zero:', np.round(len(test_df.query('null_count == 0')) / len(test_df), 2) * 100, '%')
print('Episode_Length_minutes:          ', np.round(len(test_df.query('null_count == 1')) / len(test_df), 2) * 100, '%')
print('Guest_Popularity_percentage:      ', np.round(len(test_df.query('null_count == 2')) / len(test_df), 2) * 100, '%')
print('Both:                             ', np.round(len(test_df.query('null_kinds == 3')) / len(test_df), 2) * 100, '%')



# Which is one, MCAR (Missing Completely At Random), MAR (Missing At Random), or MNAR (Missing Not At Random?

# Categorical features: 'Podcast_Name', 'Episode_Title', 'Genre'

all_df = pd.concat([train_df, test_df], axis=0)

print('Ratio: missing values')
print('Training data:', np.round(np.sum(train_df['null_count'], axis=0) / len(train_df), 2) * 100, '%')
print('Test data    :', np.round(np.sum(test_df['null_count'], axis=0) / len(test_df), 2) * 100, '%')

tr_null_count_df = train_df.groupby(['Podcast_Name', 'Episode_Title', 'Genre'])['null_count'].count().reset_index()
te_null_count_df = test_df.groupby(['Podcast_Name', 'Episode_Title', 'Genre'])['null_count'].count().reset_index()

tr_non_null_df = train_df.loc[train_df.null_count == 0]
te_non_null_df = test_df.loc[test_df.null_count == 0]

print('Training data')
print(train_df.null_count.value_counts(), '\n')

print('Training data')
print('Each features have missing values')
print('Podcast_Name nunique: ', tr_null_count_df.loc[tr_null_count_df.null_count > 1]['Podcast_Name'].value_counts().count())
print('Episode_Title nunique:', tr_null_count_df.loc[tr_null_count_df.null_count > 1]['Episode_Title'].value_counts().count())
print('Genre nunique:        ', tr_null_count_df.loc[tr_null_count_df.null_count > 1]['Genre'].value_counts().count())
print('Each features have not missing values')
print('Podcast_Name nunique: ', tr_non_null_df['Podcast_Name'].nunique())
print('Episode_Title nunique:', tr_non_null_df['Episode_Title'].nunique())
print('Genre nunique:        ', tr_non_null_df['Genre'].nunique(), '\n')

print('Test data')
print(test_df.null_count.value_counts(), '\n')
print('Test data')
print('Each features have missing values')
print('Podcast_Name nunique: ', te_null_count_df.loc[te_null_count_df.null_count > 1]['Podcast_Name'].value_counts().count())
print('Episode_Title nunique:', te_null_count_df.loc[te_null_count_df.null_count > 1]['Episode_Title'].value_counts().count())
print('Genre nunique:        ', te_null_count_df.loc[te_null_count_df.null_count > 1]['Genre'].value_counts().count())
print('Each features have not missing values')
print('Podcast_Name nunique: ', te_non_null_df['Podcast_Name'].nunique())
print('Episode_Title nunique:', te_non_null_df['Episode_Title'].nunique())
print('Genre nunique:        ', te_non_null_df['Genre'].nunique())



cross_1 = pd.pivot_table(
    data=train_df,
    values='Listening_Time_minutes',
    aggfunc='sum',
    index='Podcast_Name',
    columns='null_count', 
    margins=True,
)
print(cross_1)


cross_df = cross_1[:-1]
cross_df.columns = ['null_0', 'null_1', 'null_2', 'all']
cross_df['ratio_0_minus_all'] = cross_df['null_0'] / cross_df['all']
cross_df['ratio_1_minus_all'] = cross_df['null_1'] / cross_df['all']
cross_df['ratio_2_minus_all'] = cross_df['null_2'] / cross_df['all']
cross_df[['ratio_0_minus_all', 'ratio_1_minus_all', 'ratio_2_minus_all']].describe().T


cross_df = cross_df.reindex().reset_index()
cross_df[['Podcast_Name', 'ratio_0_minus_all', 'ratio_1_minus_all', 'ratio_2_minus_all']]


fig, ax = plt.subplots(3, 1, figsize=(20, 20))

sns.barplot(data=cross_df, x='Podcast_Name', y='ratio_0_minus_all', ax=ax[0])
ax[0].tick_params(axis='x', labelrotation=45)
ax[0].set_title('Count of missing values: 0')
ax[0].axhline(y=cross_df.ratio_0_minus_all.mean(), label='mean')
ax[0].set_ylim(0.0, 0.8)
ax[0].legend()

sns.barplot(data=cross_df, x='Podcast_Name', y='ratio_1_minus_all', ax=ax[1])
ax[1].tick_params(axis='x', labelrotation=45)
ax[1].set_title('Count of missing values: 1')
ax[1].axhline(y=cross_df.ratio_1_minus_all.mean(), label='mean')
ax[1].set_ylim(0.0, 0.8)
ax[1].legend()

sns.barplot(data=cross_df, x='Podcast_Name', y='ratio_2_minus_all', ax=ax[2])
ax[2].tick_params(axis='x', labelrotation=45)
ax[2].set_title('Count of missing values: 2')
ax[2].axhline(y=cross_df.ratio_2_minus_all.mean(), label='mean')
ax[2].set_ylim(0.0, 0.8)
ax[2].legend()

plt.tight_layout()
plt.show()


plt.clf()
plt.close()





# Training data, Test data: 'Episode_Length_minutes','Host_Popularity_percentage', 
#                           'Guest_Popularity_percentage', 'Number_of_Ads'
# Outlier

num_cols = ['Episode_Length_minutes',
            'Host_Popularity_percentage', 
            'Guest_Popularity_percentage', 
            'Number_of_Ads']

all_df = pd.concat([train_df, test_df], axis=0)

all_p99 = all_df[num_cols].quantile(0.99)
all_p01 = all_df[num_cols].quantile(0.01)

train_df[num_cols] = train_df[num_cols].clip(all_p01, all_p99, axis=1)
test_df[num_cols] = test_df[num_cols].clip(all_p01, all_p99, axis=1)



# groups: 'Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time'
# Podcast_Name : 48, Episode_Title : 100, Genre : 10, Publication_Day : 7, Publication_Time : 4

train_num_groups = train_df.groupby(['Podcast_Name', 'Episode_Title', 'Genre',
                                     'Publication_Day', 'Publication_Time']).ngroups
test_num_groups = test_df.groupby(['Podcast_Name', 'Episode_Title', 'Genre',
                                   'Publication_Day', 'Publication_Time']).ngroups

print('Count of all combination   :', len(podcast_names) * len(episode_titles) * len(genres) * len(public_days) * len(public_times))
print('training data num of groups:', train_num_groups)
print('test data num of groups    :', test_num_groups)


# Compute medians for each groups (exclude 'Number_of_Ads')
# Replacing missing data with substituted values

def fill_missing_value(data: pd.DataFrame, lists: list,
                       groups: list, column_name: list,
                       agg: str = 'median') -> pd.DataFrame:
    for col in column_name:
        
        df = data[lists].groupby(groups)[col].agg(agg)
        group_df = df.reindex().reset_index()
        merge_df = data.merge(group_df, on=groups, how='left')

        col_y = f'{col}_y'
        
        if col == 'Number_of_Ads':
            data.loc[data[col] > 4.0, col] = round(merge_df[col_y])
        else:
            data.loc[data[col].isnull(), col] = merge_df[col_y]

    return data


# groups: 'Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time'
# replace values: 'Episode_Length_minutes', 'Guest_Popularity_percentage'

train_df = fill_missing_value(train_df,
                        lists=['Podcast_Name', 'Episode_Title', 'Genre',
                               'Publication_Day', 'Publication_Time',
                              'Episode_Length_minutes', 'Guest_Popularity_percentage'],
                        groups=['Podcast_Name', 'Episode_Title',
                                'Genre', 'Publication_Day', 'Publication_Time'], 
                        column_name=['Episode_Length_minutes', 'Guest_Popularity_percentage'])

test_df = fill_missing_value(test_df,
                             lists=['Podcast_Name', 'Episode_Title', 'Genre',
                                   'Publication_Day', 'Publication_Time',
                                   'Episode_Length_minutes', 'Guest_Popularity_percentage'],
                             groups=['Podcast_Name', 'Episode_Title',
                                     'Genre', 'Publication_Day', 'Publication_Time'], 
                             column_name=['Episode_Length_minutes', 'Guest_Popularity_percentage'])


# groups: 'Podcast_Name', 'Episode_Title', 'Genre'
# replace values: 'Episode_Length_minutes', 'Guest_Popularity_percentage'

train_df = fill_missing_value(train_df,
                              lists=['Podcast_Name', 'Episode_Title', 'Genre',
                                     'Episode_Length_minutes', 'Guest_Popularity_percentage'],
                              groups=['Podcast_Name', 'Episode_Title','Genre'], 
                              column_name=['Episode_Length_minutes', 'Guest_Popularity_percentage'])

test_df = fill_missing_value(test_df,
                             lists=['Podcast_Name', 'Episode_Title', 'Genre',
                                    'Episode_Length_minutes', 'Guest_Popularity_percentage'],
                             groups=['Podcast_Name', 'Episode_Title','Genre'], 
                             column_name=['Episode_Length_minutes', 'Guest_Popularity_percentage'])


# groups: 'Podcast_Name', 'Episode_Title', 'Publication_Day', 'Publication_Time'
# replace values: 'Episode_Length_minutes', 'Guest_Popularity_percentage'

train_df = fill_missing_value(train_df,
                              lists=['Podcast_Name', 'Episode_Title', 
                                     'Publication_Day', 'Publication_Time',
                                     'Episode_Length_minutes', 'Guest_Popularity_percentage'],
                              groups=['Podcast_Name', 'Episode_Title',
                                      'Publication_Day', 'Publication_Time'], 
                              column_name=['Episode_Length_minutes', 'Guest_Popularity_percentage'])

test_df = fill_missing_value(test_df,
                             lists=['Podcast_Name', 'Episode_Title', 
                                    'Publication_Day', 'Publication_Time',
                                    'Episode_Length_minutes', 'Guest_Popularity_percentage'],
                             groups=['Podcast_Name', 'Episode_Title',
                                    'Publication_Day', 'Publication_Time'], 
                             column_name=['Episode_Length_minutes', 'Guest_Popularity_percentage'])


# groups: 'Podcast_Name', 'Episode_Title'
# replace values: 'Episode_Length_minutes', 'Guest_Popularity_percentage'

train_df = fill_missing_value(train_df,
                              lists=['Podcast_Name', 'Episode_Title', 
                                     'Episode_Length_minutes', 'Guest_Popularity_percentage'],
                              groups=['Podcast_Name', 'Episode_Title'], 
                              column_name=['Episode_Length_minutes', 'Guest_Popularity_percentage'])

test_df = fill_missing_value(test_df,
                             lists=['Podcast_Name', 'Episode_Title', 
                                    'Episode_Length_minutes', 'Guest_Popularity_percentage'],
                             groups=['Podcast_Name', 'Episode_Title'], 
                             column_name=['Episode_Length_minutes', 'Guest_Popularity_percentage'])


print(train_df.isnull().sum()) 
print(test_df.isnull().sum())


train_df.describe().T


test_df.describe().T


# Episode_Length_minutes
epi_target_mean = train_df.groupby('Episode_Length_minutes')['Listening_Time_minutes'].agg('mean').reindex().reset_index()
epi_target_means = []
epi_bins = []

for bin in np.arange(0.05, 1.0, 0.1):
    epi_bin = train_df['Episode_Length_minutes'].quantile(bin)
    epi_target_bin = epi_target_mean.loc[epi_target_mean.Episode_Length_minutes.eq(epi_bin), 
                                         'Listening_Time_minutes'].values
    
    epi_target_means.append(epi_target_bin[0])
    epi_bins.append(epi_bin)
    
epi_bin_df = pd.DataFrame(
    {'Episode_Length_minutes_bin': epi_bins,
     'Listening_Time_minutes_mean': epi_target_means})


# Host_Popularity_percentage
host_target_mean = train_df.groupby('Host_Popularity_percentage')['Listening_Time_minutes'].agg('mean').reindex().reset_index()
host_target_means = []
host_bins = []

for bin in np.arange(0.05, 1.0, 0.1): 
    host_bin = train_df['Host_Popularity_percentage'].quantile(bin)
    host_target_bin = host_target_mean.loc[host_target_mean.Host_Popularity_percentage.eq(host_bin), 
                                           'Listening_Time_minutes'].values
    host_target_means.append(host_target_bin[0])
    host_bins.append(host_bin)
    
host_bin_df = pd.DataFrame(
    {'Host_Popularity_percentage_bin': host_bins,
     'Listening_Time_minutes_mean': host_target_means})


# Guest_Popularity_percentage
gue_target_mean = train_df.groupby('Guest_Popularity_percentage')['Listening_Time_minutes'].agg('mean').reindex().reset_index()
gue_target_means = []
gue_bins = []

for bin in np.arange(0.05, 1.0, 0.1): 
    gue_bin = train_df['Guest_Popularity_percentage'].quantile(bin)
    gue_target_bin = gue_target_mean.loc[gue_target_mean.Guest_Popularity_percentage.eq(round(gue_bin), 1), 
                                        'Listening_Time_minutes'].values
    gue_target_means.append(gue_target_bin[0])
    gue_bins.append(gue_bin)
    
gue_bin_df = pd.DataFrame(
    {'Guest_Popularity_percentage_bin': gue_bins,
     'Listening_Time_minutes_mean': gue_target_means})

# Number_of_Ads
ads_target_mean = train_df.groupby('Number_of_Ads')['Listening_Time_minutes'].agg('mean').reindex().reset_index()
ads_target_means = []
ads_bins = []

for bin in np.arange(0.5, 1.0, 0.2):
    ads_bin = train_df['Number_of_Ads'].quantile(bin)
    ads_target_bin = ads_target_mean.loc[ads_target_mean.Number_of_Ads.eq(ads_bin), 'Listening_Time_minutes'].values
    ads_target_means.append(ads_target_bin[0])
    ads_bins.append(ads_bin)
    
ads_bin_df = pd.DataFrame(
    {'Number_of_Ads_bin': ads_bins,
     'Listening_Time_minutes_mean': ads_target_means})



fig, ax = plt.subplots(4, 2, figsize=(18, 10))

# Episode_Length_minutes
sns.kdeplot(train_df.Episode_Length_minutes, ax=ax[0][0])
sns.histplot(data=train_df, x='Episode_Length_minutes', ax=ax[0][0],
             label='Train', stat='density')
ax[0][0].set_xticks(np.arange(0, 121, 20), [0, 20, 40, 60, 80, 100, 120])
ax[0][0].set_xlim(0, 130)
ax[0][0].set_title('Episode_Length_minutes')
ax[0][0].legend()

sns.lineplot(data=epi_bin_df,
             x='Episode_Length_minutes_bin', y='Listening_Time_minutes_mean', ax=ax[0][1],
             marker='o')
ax[0][1].set_xticks(np.array(epi_bin_df.Episode_Length_minutes_bin), 
                    np.array(epi_bin_df.Episode_Length_minutes_bin))
ax[0][1].set_title('Episode_Length_minutes')
ax[0][1].grid()

# Host_Popularity_percentage
sns.kdeplot(train_df.Host_Popularity_percentage, ax=ax[1][0])
sns.histplot(data=train_df, x='Host_Popularity_percentage', ax=ax[1][0],
             label='Train', stat='density')
ax[1][0].set_xticks(np.arange(20, 121, 20), [20, 40, 60, 80, 100, 120])
ax[1][0].set_xlim(10, 110)
ax[1][0].set_title('Host_Popularity_percentage')
ax[1][0].legend()

sns.lineplot(data=host_bin_df,
             x='Host_Popularity_percentage_bin', y='Listening_Time_minutes_mean', ax=ax[1][1],
             marker='o')
ax[1][1].set_xticks(np.array(host_bin_df.Host_Popularity_percentage_bin), 
                    np.array(host_bin_df.Host_Popularity_percentage_bin))
ax[1][1].set_title('Host_Popularity_percentage')
ax[1][1].grid()

# Guest_Popularity_percentage
sns.kdeplot(train_df.Guest_Popularity_percentage, ax=ax[2][0])
sns.histplot(data=train_df, x='Guest_Popularity_percentage', ax=ax[2][0],
             label='Train', stat='density')
ax[2][0].set_xticks(np.arange(0, 121, 20), [0, 20, 40, 60, 80, 100, 120])
ax[2][0].set_title('Guest_Popularity_percentage')
ax[2][0].legend()

sns.lineplot(data=gue_bin_df,
             x='Guest_Popularity_percentage_bin', y='Listening_Time_minutes_mean', ax=ax[2][1],
             marker='o')
ax[2][1].set_xticks(np.array(gue_bin_df.Guest_Popularity_percentage_bin), 
                    np.round(gue_bin_df.Guest_Popularity_percentage_bin, 2))
ax[2][1].set_title('Guest_Popularity_percentage')
ax[2][1].grid()


# Number_of_Ads
sns.kdeplot(train_df.Number_of_Ads, ax=ax[3][0])
sns.histplot(data=train_df, x='Number_of_Ads', ax=ax[3][0],
             label='Train', stat='density')
ax[3][0].set_xticks(np.arange(0.0, 4.1, 1.0), [0.0, 1.0, 2.0, 3.0, 4.0])
ax[3][0].set_xlim(-1.0, 4.0)
ax[3][0].set_title('Number_of_Ads')
ax[3][0].legend()

sns.lineplot(data=ads_bin_df,
             x='Number_of_Ads_bin', y='Listening_Time_minutes_mean', ax=ax[3][1],
             marker='o')
ax[3][1].set_xticks(np.array(ads_bin_df.Number_of_Ads_bin), 
                    np.array(ads_bin_df.Number_of_Ads_bin))
ax[3][1].set_title('Number_of_Ads')
ax[3][1].grid()

plt.tight_layout()
plt.show()


# Add features: 'senti_by_ad'

senti_mapping = {
    'Positive': 0, 
    'Negative': 2, 
    'Neutral': 1, 
}

train_df['Episode_Sentiment'] = train_df['Episode_Sentiment'].map(senti_mapping)
test_df['Episode_Sentiment'] = test_df['Episode_Sentiment'].map(senti_mapping)


# Heatmaps

cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
        'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Listening_Time_minutes']

sns.set(font_scale=0.6)
plt.figure(figsize=(8,6))
sns.heatmap(train_df[cols].corr(), linewidth=0.2,
            annot=True, annot_kws={"fontsize": 10}, cmap='Greens'
)
plt.show()


plt.clf()
plt.close()


# Publication_Day
day_target_means = []
days = []

for day in public_days:
    day_target = train_df.loc[train_df.Publication_Day.eq(day), 'Listening_Time_minutes'].mean()
    
    day_target_means.append(day_target)
    days.append(day)
    
day_df = pd.DataFrame(
    {'Publication_Day': days,
     'Listening_Time_minutes_mean': day_target_means})

# Publication_Time
time_target_means = []
times = []

for time in public_times:
    time_target = train_df.loc[train_df.Publication_Time.eq(time), 'Listening_Time_minutes'].mean()
    
    time_target_means.append(time_target)
    times.append(time)
    
time_df = pd.DataFrame(
    {'Publication_Time': times,
     'Listening_Time_minutes_mean': time_target_means})

# 'Episode_Sentiment'
senti_target_means = []
sentis = []

for senti in [0,2,1]:
    senti_target = train_df.loc[train_df.Episode_Sentiment.eq(senti), 'Listening_Time_minutes'].mean()
    
    senti_target_means.append(senti_target)
    sentis.append(senti)
    
senti_df = pd.DataFrame(
    {'Episode_Sentiment': sentis,
     'Listening_Time_minutes_mean': senti_target_means})


public_days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday']
public_times = ['Morning', 'Afternoon',  'Evening', 'Night']
episode_sentis = ['Positive', 'Neutral', 'Negative']

fig, ax = plt.subplots(3, 2, figsize=(18, 10))

# Publication_Day
sns.countplot(
    data=train_df, x='Publication_Day', 
    ax=ax[0][0]
)
ax[0][0].set_xticks(np.arange(0, 7, 1), public_days)
ax[0][0].set_title('Publication_Day')

sns.lineplot(data=day_df,
             x='Publication_Day', y='Listening_Time_minutes_mean', ax=ax[0][1],
             marker='o')
ax[0][1].set_xticks(np.arange(0, 7, 1), public_days)
ax[0][1].set_title('Publication_Day')
ax[0][1].grid()

# Publication_Time
sns.countplot(
    data=train_df, x='Publication_Time', 
    ax=ax[1][0]
)
ax[1][0].set_xticks(np.arange(0, 4, 1), public_times)
ax[1][0].set_title('Publication_Time')

sns.lineplot(data=time_df,
             x='Publication_Time', y='Listening_Time_minutes_mean', ax=ax[1][1],
             marker='o')
ax[1][1].set_xticks(np.arange(0, 4, 1), public_times)
ax[1][1].set_title('Publication_Time')
ax[1][1].grid()

# Episode_Sentiment
sns.countplot(
    data=train_df, x='Episode_Sentiment', 
    ax=ax[2][0]
)
ax[2][0].set_xticks(np.arange(0, 3, 1), episode_sentis)
ax[2][0].set_title('Episode_Sentiment')

sns.lineplot(data=senti_df,
             x='Episode_Sentiment', y='Listening_Time_minutes_mean', ax=ax[2][1],
             marker='o')
ax[2][1].set_xticks(np.arange(0, 3, 1), episode_sentis)
ax[2][1].set_title('Episode_Sentiment')
ax[2][1].grid()

plt.tight_layout()
plt.show()


plt.clf()
plt.close()


# Plot each features 
# Number of targets values equal to zero

target_zero = train_df.loc[train_df.Listening_Time_minutes.eq(0.0)]

fig = plt.figure(figsize=(20, 40))
gs = fig.add_gridspec(5, 2, width_ratios=(1,1))

ax1 = fig.add_subplot(gs[0, 0:]) 
ax2 = fig.add_subplot(gs[1, 0:]) 
ax3 = fig.add_subplot(gs[2, 0:])
ax4 = fig.add_subplot(gs[3, 0])
ax5 = fig.add_subplot(gs[3, 1])
ax6 = fig.add_subplot(gs[4, 0])
ax7 = fig.add_subplot(gs[4, 1])

# Podcast_Name
sns.countplot(
    data=target_zero, x="Podcast_Name",
    ax=ax1
)
ax1.bar_label(ax1.containers[0], size=14)
ax1.set_ylabel("Number of\nzero target")
ax1.set_ylim(0, 750)
ax1.set_title("Podcast_Name with zero target")
ax1.tick_params(axis='x', labelrotation=45)

# Episode_Title
sns.countplot(
    data=target_zero, x='Episode_Title',
    ax=ax2
)
ax2.bar_label(ax2.containers[0], size=12)
ax2.set_ylim(0, 600)
ax2.set_ylabel('Number of\nzero Target')
ax2.set_title('Episode_Title with zero Target')
ax2.tick_params(axis='x', labelrotation=45)

# Genre
sns.countplot(
    data=target_zero, x='Genre',
    ax=ax3
)
ax3.bar_label(ax3.containers[0], size=14)
ax3.set_ylim(0, 1500)
ax3.set_ylabel('Number of\nzero Target')
ax3.set_title('Genre with zero Target')
ax3.tick_params(axis='x', labelrotation=45)

# Publication_Day
sns.countplot(
    data=target_zero, x='Publication_Day',
    ax=ax4
)
ax4.bar_label(ax4.containers[0], size=14)
ax4.set_ylim(0, 2000)
ax4.set_ylabel('Number of\nzero Target')
ax4.set_title('Publication_Day with zero Target')

# Publication_Time
sns.countplot(
    data=target_zero, x='Publication_Time',
    ax=ax5
)
ax5.bar_label(ax5.containers[0], size=14)
ax5.set_ylim(0, 3100)
ax5.set_ylabel('Number of\nzero Target')
ax5.set_title('Publication_Time with zero Target')

# number_of_Ads
sns.countplot(
    data=target_zero, x='Number_of_Ads',
    ax=ax6
)

ax6.bar_label(ax6.containers[0], size=14)
ax6.set_ylabel('Number of\nzero Target')
ax6.set_title('Number_of_Ads with zero Target')



# Episode_Sentiment
sns.countplot(
    data=target_zero, x='Episode_Sentiment',
    ax=ax7
)
ax7.bar_label(ax7.containers[0], size=14)
ax7.set_xticks(np.arange(0, 3, 1), episode_sentis)
ax7.set_ylim(0, 5500)
ax7.set_ylabel('Number of\nzero Target')
ax7.set_title('Episode_Sentiment with zero Target')

plt.tight_layout()
plt.show()


plt.clf()
plt.close()


# Plot each features 
# Number of targets values not equal to zero

target_non_zero = train_df.loc[train_df.Listening_Time_minutes.ne(0.0)]

fig = plt.figure(figsize=(20, 40))
gs = fig.add_gridspec(5, 2, width_ratios=(1,1))

ax1 = fig.add_subplot(gs[0, 0:]) 
ax2 = fig.add_subplot(gs[1, 0:]) 
ax3 = fig.add_subplot(gs[2, 0:])
ax4 = fig.add_subplot(gs[3, 0])
ax5 = fig.add_subplot(gs[3, 1])
ax6 = fig.add_subplot(gs[4, 0])
ax7 = fig.add_subplot(gs[4, 1])

# Podcast_Name
sns.countplot(
    data=target_non_zero, x="Podcast_Name",
    ax=ax1
)
ax1.bar_label(ax1.containers[0], size=10)
ax1.set_ylabel("Number of\nnon zero target")
# ax1.set_ylim(0, 9000)
ax1.set_title("Podcast_Name with non zero target")
ax1.tick_params(axis='x', labelrotation=45)

# Episode_Title
sns.countplot(
    data=target_non_zero, x='Episode_Title',
    ax=ax2
)
ax2.bar_label(ax2.containers[0], size=12)
# ax2.set_ylim(0, 9000)
ax2.set_ylabel('Number of\nnon zero Target')
ax2.set_title('Episode_Title with non zero Target')
ax2.tick_params(axis='x', labelrotation=45)

# Genre
sns.countplot(
    data=target_non_zero, x='Genre',
    ax=ax3
)
ax3.bar_label(ax3.containers[0], size=14)
# ax3.set_ylim(0, 9000)
ax3.set_ylabel('Number of\nnon zero Target')
ax3.set_title('Genre with non zero Target')
ax3.tick_params(axis='x', labelrotation=45)

# Publication_Day
sns.countplot(
    data=target_non_zero, x='Publication_Day',
    ax=ax4
)
ax4.bar_label(ax4.containers[0], size=14)
# ax4.set_ylim(0, 9000)
ax4.set_ylabel('Number of\nnon zero Target')
ax4.set_title('Publication_Day with non zero Target')

# Publication_Time
sns.countplot(
    data=target_non_zero, x='Publication_Time',
    ax=ax5
)
ax5.bar_label(ax5.containers[0], size=14)
# ax5.set_ylim(0, 9000)
ax5.set_ylabel('Number of\nnon zero Target')
ax5.set_title('Publication_Time with non nonzero Target')

# number_of_Ads
sns.countplot(
    data=target_non_zero, x='Number_of_Ads',
    ax=ax6
)

ax6.bar_label(ax6.containers[0], size=14)
ax6.set_ylabel('Number of\nnon zero Target')
ax6.set_title('Number_of_Ads with non zero Target')



# Episode_Sentiment
sns.countplot(
    data=target_non_zero, x='Episode_Sentiment',
    ax=ax7
)
ax7.bar_label(ax7.containers[0], size=14)
ax7.set_xticks(np.arange(0, 3, 1), episode_sentis)
# ax7.set_ylim(0, 9000)
ax7.set_ylabel('Number of\nnon_zero Target')
ax7.set_title('Episode_Sentiment with non zero Target')

plt.tight_layout()
plt.show()


plt.clf()
plt.close()


train_df['Target_zero'] = train_df.Listening_Time_minutes.values

train_df['Target_zero'] = train_df.Target_zero.apply(lambda x: 1 if x == 0 else 0)

train_df.head(3)


# Plot each features 
# Number of targets values

fig = plt.figure(figsize=(20, 40))
gs = fig.add_gridspec(5, 2, width_ratios=(1,1))

ax1 = fig.add_subplot(gs[0, 0:]) 
ax2 = fig.add_subplot(gs[1, 0:]) 
ax3 = fig.add_subplot(gs[2, 0:])
ax4 = fig.add_subplot(gs[3, 0])
ax5 = fig.add_subplot(gs[3, 1])
ax6 = fig.add_subplot(gs[4, 0])
ax7 = fig.add_subplot(gs[4, 1])

# Podcast_Name
sns.countplot(
    data=train_df, x="Podcast_Name",
    ax=ax1, hue='Target_zero'
)
ax1.bar_label(ax1.containers[0], size=10)
ax1.set_ylabel("Compare Number of\ntarget non zero and zero")
# ax1.set_ylim(0, 750)
ax1.set_title("Podcast_Name with target")
ax1.tick_params(axis='x', labelrotation=45)

# Episode_Title
sns.countplot(
    data=train_df, x='Episode_Title',
    ax=ax2, hue='Target_zero'
)
ax2.bar_label(ax2.containers[0], size=12)
# ax2.set_ylim(0, 600)
ax2.set_ylabel('Compare Number of\ntarget non zero and zero')
ax2.set_title('Episode_Title with Target')
ax2.tick_params(axis='x', labelrotation=45)

# Genre
sns.countplot(
    data=train_df, x='Genre',
    ax=ax3, hue='Target_zero'
)
ax3.bar_label(ax3.containers[0], size=14)
# ax3.set_ylim(0, 1500)
ax3.set_ylabel('Compare Number of\ntarget non zero and zero')
ax3.set_title('Genre with Target')
ax3.tick_params(axis='x', labelrotation=45)

# Publication_Day
sns.countplot(
    data=train_df, x='Publication_Day',
    ax=ax4, hue='Target_zero'
)
ax4.bar_label(ax4.containers[0], size=14)
# ax4.set_ylim(0, 2000)
ax4.set_ylabel('Compare Number of\ntarget non zero and zero')
ax4.set_title('Publication_Day with Target')

# Publication_Time
sns.countplot(
    data=train_df, x='Publication_Time',
    ax=ax5, hue='Target_zero'
)
ax5.bar_label(ax5.containers[0], size=14)
# ax5.set_ylim(0, 3100)
ax5.set_ylabel('Compare Number of\ntarget non zero and zero')
ax5.set_title('Publication_Time with Target')

# number_of_Ads
sns.countplot(
    data=train_df, x='Number_of_Ads',
    ax=ax6, hue='Target_zero'
)

ax6.bar_label(ax6.containers[0], size=14)
ax6.set_ylabel('Compare Number of\ntarget non zero and zero')
ax6.set_title('Number_of_Ads with Target')



# Episode_Sentiment
sns.countplot(
    data=train_df, x='Episode_Sentiment',
    ax=ax7, hue='Target_zero'
)
ax7.bar_label(ax7.containers[0], size=14)
ax7.set_xticks(np.arange(0, 3, 1), episode_sentis)
# ax7.set_ylim(0, 5500)
ax7.set_ylabel('Compare Number of\ntarget non zero and zero')
ax7.set_title('Episode_Sentiment with Target')

plt.tight_layout()
plt.show()


plt.clf()
plt.close()


# Distribusion of Listening_Time_minutes
# Podcast_Name

plt.rcParams['font.size'] = 5
fig, ax = plt.subplots(3, 2, figsize=(12, 10))

cols_1 = podcast_names[:8]
sns.boxplot(data=train_df.loc[train_df.Podcast_Name.isin(cols_1)],
            x='Podcast_Name', y='Listening_Time_minutes',
            hue='Podcast_Name', ax=ax[0][0])
ax[0][0].legend(loc='upper right', fontsize='small')
ax[0][0].tick_params(axis='x', labelrotation=45)

cols_2 = podcast_names[8:16]
sns.boxplot(data=train_df.loc[train_df.Podcast_Name.isin(cols_2)],
            x='Podcast_Name', y='Listening_Time_minutes',
            hue='Podcast_Name', ax=ax[0][1])
ax[0][1].legend(loc='upper right', fontsize='small')
ax[0][1].tick_params(axis='x', labelrotation=45)

cols_3 = podcast_names[16:24]
sns.boxplot(data=train_df.loc[train_df.Podcast_Name.isin(cols_3)],
            x='Podcast_Name', y='Listening_Time_minutes',
            hue='Podcast_Name', ax=ax[1][0])
ax[1][0].legend(loc='upper right', fontsize='small')
ax[1][0].tick_params(axis='x', labelrotation=45)

cols_4 = podcast_names[24:32]
sns.boxplot(data=train_df.loc[train_df.Podcast_Name.isin(cols_4)],
            x='Podcast_Name', y='Listening_Time_minutes',
            hue='Podcast_Name', ax=ax[1][1])
ax[1][1].legend(loc='upper right', fontsize='small')
ax[1][1].tick_params(axis='x', labelrotation=45)

cols_5 = podcast_names[32:40]
sns.boxplot(data=train_df.loc[train_df.Podcast_Name.isin(cols_5)],
            x='Podcast_Name', y='Listening_Time_minutes',
            hue='Podcast_Name', ax=ax[2][0])
ax[2][0].legend(loc='upper right', fontsize='small')
ax[2][0].tick_params(axis='x', labelrotation=45)

cols_6 = podcast_names[40:]
sns.boxplot(data=train_df.loc[train_df.Podcast_Name.isin(cols_6)],
            x='Podcast_Name', y='Listening_Time_minutes',
            hue='Podcast_Name', ax=ax[2][1])
ax[2][1].legend(loc='upper right', fontsize='small')
ax[2][1].tick_params(axis='x', labelrotation=45)

plt.tight_layout()
plt.show()


# Distribusion of Listening_Time_minutes
# Genre

plt.rcParams['font.size'] = 6
fig, ax = plt.subplots(figsize=(12, 3))

sns.boxplot(data=train_df.loc[train_df.Genre.isin(genres)],
            x='Genre', y='Listening_Time_minutes',
            hue='Genre', ax=ax)
plt.legend(loc='best')
plt.show()


plt.clf()
plt.close()


# Target encoding of categorical features

cat_cols = ['Podcast_Name', 'Episode_Title', 'Genre',
            'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for c in cat_cols:
    # Caluculate the target average of training data
    data_tmp = pd.DataFrame({c: train_df[c], 
                             'target': train_df['Listening_Time_minutes']})
    target_mean = data_tmp.groupby(c)['target'].mean()

    c_encode = f'{c}_encode'
    # Replace values: test data
    test_df[c_encode] = test_df[c].map(target_mean)

    # Temporary storage
    tmp = np.repeat(np.nan, train_df.shape[0])

    # Split training data
    kfolds = 4
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

    # Use out-of-hold to avoid data leakage
    for idx_1, idx_2 in kf.split(train_df):
        target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        tmp[idx_2] = train_df[c].iloc[idx_2].map(target_mean)

    # Replace with converted values
    train_df[c_encode] = tmp


fig, ax = plt.subplots(3, 2, figsize=(18, 10))

# Podcast_Name_encode
sns.kdeplot(train_df.Podcast_Name_encode, ax=ax[0][0], color='orange')
sns.histplot(data=train_df, x='Podcast_Name_encode', ax=ax[0][0],
             label='Train', stat='density')
ax[0][0].set_title('Podcast_Name_encode')
ax[0][0].legend()

# Episode_Title_encode
sns.kdeplot(train_df.Episode_Title_encode, ax=ax[0][1], color='orange')
sns.histplot(data=train_df, x='Episode_Title_encode', ax=ax[0][1],
             label='Train', stat='density')
ax[0][1].set_title('Episode_Title_encode')
ax[0][1].legend()

# Genre_encode
sns.kdeplot(train_df.Genre_encode, ax=ax[1][0], color='orange')
sns.histplot(data=train_df, x='Genre_encode', ax=ax[1][0],
             label='Train', stat='density')
ax[1][0].set_title('Genre_encode')
ax[1][0].legend()

# Publication_Day_encode
sns.kdeplot(train_df.Publication_Day_encode, ax=ax[1][1], color='orange')
sns.histplot(data=train_df, x='Publication_Day_encode', ax=ax[1][1],
             label='Train', stat='density')
ax[1][1].set_title('Publication_Day_encode')
ax[1][1].legend()

# Publication_Time_encode
sns.kdeplot(train_df.Publication_Time_encode, ax=ax[2][0], color='orange')
sns.histplot(data=train_df, x='Publication_Time_encode', ax=ax[2][0],
             label='Train', stat='density')
ax[2][0].set_title('Publication_Time_encode')
ax[2][0].legend()

# Episode_Sentiment_encode
sns.kdeplot(train_df.Episode_Sentiment_encode, ax=ax[2][1], color='orange')
sns.histplot(data=train_df, x='Episode_Sentiment_encode', ax=ax[2][1],
             label='Train', stat='density')
ax[2][1].set_title('Episode_Sentiment_encode')
ax[2][1].legend()

plt.tight_layout()
plt.show()


plt.clf()
plt.close()

