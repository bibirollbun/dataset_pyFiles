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
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.ticker import NullFormatter

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
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
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from feature_engine.creation import CyclicalFeatures

import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.impute import KNNImputer

import warnings
warnings.filterwarnings('ignore')

# tf.random.set_seed(42)
np.random.seed(42)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# Add features: 'is_null', 'null_type'

train_df['is_null'] = train_df.isnull().sum(axis=1).astype('int8')
test_df['is_null'] = test_df.isnull().sum(axis=1).astype('int8')

train_df['null_type'] = 0
train_df.loc[train_df.Episode_Length_minutes.isnull(), 'null_type'] = 1
train_df.loc[train_df.Guest_Popularity_percentage.isnull(), 'null_type'] = 2
train_df.loc[(train_df.Episode_Length_minutes.isnull()) & (train_df.Guest_Popularity_percentage.isnull()), 'null_type'] = 3

test_df['null_type'] = 0
test_df.loc[test_df.Episode_Length_minutes.isnull(), 'null_type'] = 1
test_df.loc[test_df.Guest_Popularity_percentage.isnull(), 'null_type'] = 2
test_df.loc[(test_df.Episode_Length_minutes.isnull()) & (test_df.Guest_Popularity_percentage.isnull()), 'null_type'] = 3

train_df.null_type = train_df.null_type.astype('int8')
test_df.null_type = test_df.null_type.astype('int8')



all_df = pd.concat([train_df, test_df], axis=0)

# Check the percentage of zero
target_zero = all_df.loc[all_df.Listening_Time_minutes.eq(0.0)]
target_non_zero = all_df.loc[all_df.Listening_Time_minutes.ne(0.0)]

(target_zero.groupby(['Podcast_Name', 'Episode_Title',
                      'Publication_Day', 
                      'Publication_Time'])['Podcast_Name'].count()/target_non_zero.groupby(['Podcast_Name', 'Episode_Title', 
                                                                                            'Publication_Day', 
                                                                                            'Publication_Time'])['Podcast_Name'].count()).sort_values(ascending=False).head(60)



# Check: targets are over 100 when 'Episode_Length_minutes' is null
train_df.loc[(train_df.null_type == 1) & 
             (train_df['Listening_Time_minutes'] > 100), ['Podcast_Name',
                                                          'Episode_Title', 
                                                          'Publication_Day', 'Publication_Time']].value_counts().head(60)


# Check: targets are over 100 when 'Episode_Length_minutes' is null
train_df.loc[(train_df.null_type == 1) & 
             (train_df['Listening_Time_minutes'] > 100), ['Podcast_Name',
                                                          'Episode_Title', 
                                                          'Publication_Day', 'Publication_Time']].value_counts().head(65)


# Add features: 'epi_null_100_over'

# Podcast_Name         Episode_Title  Publication_Day  Publication_Time
# Mind & Body          Episode 72     Wednesday        Evening             5
# Lifestyle Lounge     Episode 31     Thursday         Evening             4
# Game Day             Episode 99     Sunday           Night               4
# Finance Focus        Episode 7      Monday           Afternoon           4
# Tech Trends          Episode 20     Friday           Night               4
# Tech Trends          Episode 31     Monday           Morning             3
# Music Matters        Episode 84     Sunday           Evening             3
# Comedy Corner        Episode 25     Sunday           Morning             3
# Health Hour          Episode 49     Saturday         Morning             3
# Brain Boost          Episode 25     Sunday           Evening             3
# Healthy Living       Episode 37     Saturday         Evening             3
# Funny Folks          Episode 8      Monday           Night               2
# Gadget Geek          Episode 7      Monday           Afternoon           2
# Mind & Body          Episode 69     Friday           Evening             2
# Business Insights    Episode 61     Friday           Afternoon           2
# Mystery Matters      Episode 30     Saturday         Afternoon           2
# Athlete's Arena      Episode 84     Friday           Morning             2
# Educational Nuggets  Episode 35     Saturday         Morning             2
# Wellness Wave        Episode 14     Friday           Afternoon           2
# Athlete's Arena      Episode 84     Saturday         Morning             2

epi_null_list = [
    ['Mind & Body', 'Episode 72', 'Wednesday', 'Evening'],
    ['Lifestyle Lounge', 'Episode 31', 'Thursday', 'Evening'],
    ['Game Day', 'Episode 99', 'Sunday', 'Night'],
    ['Finance Focus', 'Episode 7', 'Monday', 'Afternoon'],
    ['Tech Trends', 'Episode 20', 'Friday', 'Night'],
    ['Tech Trends', 'Episode 31', 'Monday', 'Morning'],
    ['Music Matters', 'Episode 84', 'Sunday', 'Evening'],
    ['Comedy Corner', 'Episode 25', 'Sunday', 'Morning'],
    ['Health Hour', 'Episode 49', 'Saturday', 'Morning'],
    ['Brain Boost', 'Episode 25', 'Sunday', 'Evening'],
    ['Healthy Living', 'Episode 37', 'Saturday', 'Evening'],
    ['Funny Folks', 'Episode 8', 'Monday', 'Night'],
    ['Gadget Geek', 'Episode 7', 'Monday', 'Afternoon'], 
    ['Mind & Body', 'Episode 69', 'Friday', 'Evening'],
    ['Business Insights', 'Episode 61', 'Friday', 'Afternoon'],
    ['Mystery Matters', 'Episode 30', 'Saturday', 'Afternoon'],
    ["Athlete's Arena", 'Episode 84', 'Friday', 'Morning'],
    ['Educational Nuggets', 'Episode 35', 'Saturday', 'Morning'],
    ['Wellness Wave', 'Episode 14', 'Friday', 'Afternoon'],
    ["Athlete's Arena", 'Episode 84', 'Saturday', 'Morning'],
]
all_df['epi_null_100_over'] = 0

for i, list in enumerate(epi_null_list):
    name = list[0]
    title = list[1]
    day = list[2]
    time = list[3]
    all_df.loc[(all_df.null_type == 1) & 
               (all_df.Listening_Time_minutes.ge(100)) &
               (all_df.Podcast_Name.eq(name)) & 
               (all_df.Episode_Title.eq(title)) &
               (all_df.Publication_Day.eq(day)) &
               (all_df.Publication_Time.eq(time)),  'epi_null_100_over'] = 1

# Mind & Body          Episode 74     Wednesday        Evening             2
# Sound Waves          Episode 15     Monday           Night               2
# Home & Living        Episode 94     Thursday         Evening             2
# Sports Weekly        Episode 16     Saturday         Evening             2
# Tune Time            Episode 73     Monday           Evening             2
# Music Matters        Episode 90     Saturday         Evening             2
# Funny Folks          Episode 83     Friday           Afternoon           2
# Humor Hub            Episode 4      Tuesday          Evening             2
# Current Affairs      Episode 46     Sunday           Morning             2
# Melody Mix           Episode 58     Saturday         Evening             2
# Humor Hub            Episode 54     Thursday         Evening             2
# Business Briefs      Episode 61     Wednesday        Morning             2
# Gadget Geek          Episode 59     Thursday         Night               2
# Detective Diaries    Episode 53     Saturday         Afternoon           2
# True Crime Stories   Episode 17     Sunday           Morning             2
# Business Insights    Episode 44     Saturday         Evening             2
# Game Day             Episode 56     Saturday         Evening             2
# Mystery Matters      Episode 65     Sunday           Afternoon           2
# Athlete's Arena      Episode 22     Wednesday        Evening             2
# Funny Folks          Episode 98     Monday           Night               2
# Money Matters        Episode 72     Saturday         Afternoon           2
# Laugh Line           Episode 64     Tuesday          Afternoon           2
# Fashion Forward      Episode 33     Sunday           Afternoon           2
# Mind & Body          Episode 30     Saturday         Afternoon           2
# Mystery Matters      Episode 83     Friday           Afternoon           2
# Funny Folks          Episode 65     Monday           Night               2
# Business Insights    Episode 34     Wednesday        Night               2
# Mystery Matters      Episode 58     Friday           Evening             2
# Daily Digest         Episode 67     Saturday         Evening             2
# Learning Lab         Episode 33     Tuesday          Morning             2

epi_null_list_2 = [
    ['Mind & Body', 'Episode 74', 'Wednesday', 'Evening'],
    ['Sound Waves', 'Episode 15', 'Monday', 'Night'],
    ['Home & Living', 'Episode 94', 'Thursday', 'Evening'],
    ['Sports Weekly', 'Episode 16', 'Saturday', 'Evening'],
    ['Tune Time', 'Episode 73', 'Monday', 'Evening'],
    ['Music Matters', 'Episode 90', 'Saturday', 'Evening'],
    ['Funny Folks', 'Episode 83', 'Friday', 'Afternoon'],
    ['Humor Hub', 'Episode 4', 'Tuesday', 'Evening'],
    ['Current Affairs', 'Episode 46', 'Sunday', 'Morning'],
    ['Melody Mix', 'Episode 58', 'Saturday', 'Evening'],
    ['Humor Hub', 'Episode 54', 'Thursday', 'Evening'],
    ['Business Briefs', 'Episode 61', 'Wednesday', 'Morning'],
    ['Gadget Geek', 'Episode 59', 'Thursday', 'Night'],
    ['Detective Diaries', 'Episode 53', 'Saturday', 'Afternoon'],
    ['True Crime Stories', 'Episode 17', 'Sunday', 'Morning'],
    ['Business Insights', 'Episode 44', 'Saturday', 'Evening'],
    ['Game Day', 'Episode 56', 'Saturday', 'Evening'],
    ['Mystery Matters', 'Episode 65', 'Sunday', 'Afternoon'],
    ["Athlete's Arena", 'Episode 22', 'Wednesday', 'Evening'],
    ['Funny Folks', 'Episode 98', 'Monday', 'Night'],
    ['Money Matters', 'Episode 72', 'Saturday', 'Afternoon'],
    ['Laugh Line', 'Episode 64', 'Tuesday', 'Afternoon'],
    ['Fashion Forward', 'Episode 33', 'Sunday', 'Afternoon'],
    ['Mind & Body', 'Episode 30', 'Saturday', 'Afternoon'],
    ['Mystery Matters', 'Episode 83', 'Friday', 'Afternoon'],
    ['Funny Folks', 'Episode 65', 'Monday', 'Night'],
    ['Business Insights', 'Episode 34', 'Wednesday', 'Night'],
    ['Mystery Matters', 'Episode 58', 'Friday', 'Evening'],
    ['Daily Digest', 'Episode 67', 'Saturday', 'Evening'],
    ['Learning Lab', 'Episode 33', 'Tuesday', 'Morning'],
]

for i, list in enumerate(epi_null_list_2):
    name = list[0]
    title = list[1]
    day = list[2]
    time = list[3]
    all_df.loc[(all_df.null_type == 1) & 
               (all_df.Listening_Time_minutes.ge(100)) &
               (all_df.Podcast_Name.eq(name)) & 
               (all_df.Episode_Title.eq(title)) &
               (all_df.Publication_Day.eq(day)) &
               (all_df.Publication_Time.eq(time)),  'epi_null_100_over'] = 1

# Market Masters       Episode 33     Thursday         Afternoon           2
# Athlete's Arena      Episode 51     Friday           Evening             2
# Funny Folks          Episode 99     Monday           Morning             2
# Finance Focus        Episode 25     Monday           Afternoon           2
# Wellness Wave        Episode 59     Thursday         Night               2
# Finance Focus        Episode 14     Saturday         Night               2
# Fashion Forward      Episode 79     Friday           Afternoon           2
# Funny Folks          Episode 31     Thursday         Night               2
# Gadget Geek          Episode 98     Thursday         Afternoon           2
# Crime Chronicles     Episode 28     Tuesday          Evening             2
# Market Masters    Episode 62     Monday           Morning             2
# Crime Chronicles  Episode 95     Tuesday          Evening             2
# Fashion Forward   Episode 79     Friday           Night               2
# Tech Talks        Episode 61     Saturday         Morning             2

epi_null_list_3 = [
    ['Market Masters', 'Episode 33', 'Thursday', 'Afternoon'],
    ["Athlete's Arena", 'Episode 51', 'Friday', 'Evening'],
    ['Funny Folks', 'Episode 99', 'Monday', 'Morning'],
    ['Finance Focus', 'Episode 25', 'Monday', 'Afternoon'],
    ['Wellness Wave', 'Episode 59', 'Thursday', 'Night'],
    ['Finance Focus', 'Episode 14', 'Saturday', 'Night'],
    ['Fashion Forward', 'Episode 79', 'Friday', 'Afternoon'],
    ['Funny Folks', 'Episode 31', 'Thursday', 'Night'],
    ['Gadget Geek', 'Episode 98', 'Thursday', 'Afternoon'],
    ['Crime Chronicles', 'Episode 28', 'Tuesday', 'Evening'],
    ['Market Masters', 'Episode 62', 'Monday', 'Morning'],
    ['Crime Chronicles', 'Episode 95', 'Tuesday', 'Evening'],
    ['Fashion Forward', 'Episode 79', 'Friday', 'Night'],
    ['Tech Talks', 'Episode 61', 'Saturday', 'Morning'],
]

for i, list in enumerate(epi_null_list_3):
    name = list[0]
    title = list[1]
    day = list[2]
    time = list[3]
    all_df.loc[(all_df.null_type == 1) & 
               (all_df.Listening_Time_minutes.ge(100)) &
               (all_df.Podcast_Name.eq(name)) & 
               (all_df.Episode_Title.eq(title)) &
               (all_df.Publication_Day.eq(day)) &
               (all_df.Publication_Time.eq(time)),  'epi_null_100_over'] = 1

all_df['epi_null_100_over'] = all_df['epi_null_100_over'].astype('int8')


all_df['epi_null_100_over'].value_counts()


# Add features: 'Genre_type'

all_df['Genre_type'] = ''

# Podcast_Name.eq('Tech Talks')  Technology  22815 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Tech Talks') & 
           all_df.Genre.eq('Technology'), 'Genre_type'] = 'Tech_0'
all_df.loc[all_df.Podcast_Name.eq('Tech Talks') & 
           all_df.Genre.ne('Technology'), 'Genre_type'] = 'Tech_1'

# Podcast_Name.eq('Digital Digest') Technology  16135 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Digital Digest') & 
           all_df.Genre.eq('Technology'), 'Genre_type'] = 'Tech_0'
all_df.loc[all_df.Podcast_Name.eq('Digital Digest') & 
           all_df.Genre.ne('Technology'), 'Genre_type'] = 'Tech_1'

# Podcast_Name.eq('Tech Trends') Technology  19534 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Tech Trends') & 
           all_df.Genre.eq('Technology'), 'Genre_type'] = 'Tech_0'
all_df.loc[all_df.Podcast_Name.eq('Tech Trends') & 
           all_df.Genre.ne('Technology'), 'Genre_type'] = 'Tech_1'

# Podcast_Name.eq('Innovators') Technology  12890 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Innovators') & 
           all_df.Genre.eq('Technology'), 'Genre_type'] = 'Tech_0'
all_df.loc[all_df.Podcast_Name.eq('Innovators') & 
           all_df.Genre.ne('Technology'), 'Genre_type'] = 'Tech_1'

# Podcast_Name.eq('Gadget Geek') Technology  14731 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Gadget Geek') & 
           all_df.Genre.eq('Technology'), 'Genre_type'] = 'Tech_0'
all_df.loc[all_df.Podcast_Name.eq('Gadget Geek') & 
           all_df.Genre.ne('Technology'), 'Genre_type'] = 'Tech_1'

# Podcast_Name.eq('Sport Spot') Sports  14756 ->  0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Sport Spot') & 
           all_df.Genre.eq('Sports'), 'Genre_type'] = 'Sport_0'
all_df.loc[all_df.Podcast_Name.eq('Sport Spot') & 
           all_df.Genre.ne('Sports'), 'Genre_type'] = 'Sport_1'

# Podcast_Name.eq('Sports Weekly') Sports  20048 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Sports Weekly') & 
           all_df.Genre.eq('Sports'), 'Genre_type'] = 'Sport_0'
all_df.loc[all_df.Podcast_Name.eq('Sports Weekly') & 
           all_df.Genre.ne('Sports'), 'Genre_type'] = 'Sport_1'

# Podcast_Name.eq("Athlete's Arena") Sports  17256 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq("Athlete's Arena") & 
           all_df.Genre.eq('Sports'), 'Genre_type'] = 'Sport_0'
all_df.loc[all_df.Podcast_Name.eq("Athlete's Arena") & 
           all_df.Genre.ne('Sports'), 'Genre_type'] = 'Sport_1'

# Podcast_Name.eq('Sports Central') Sports  16185 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Sports Central') & 
           all_df.Genre.eq('Sports'), 'Genre_type'] = 'Sport_0'
all_df.loc[all_df.Podcast_Name.eq('Sports Central') & 
           all_df.Genre.ne('Sports'), 'Genre_type'] = 'Sport_1'

# Podcast_Name.eq('Game Day') Sports  19195 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Game Day') & 
           all_df.Genre.eq('Sports'), 'Genre_type'] = 'Sport_0'
all_df.loc[all_df.Podcast_Name.eq('Game Day') & 
           all_df.Genre.ne('Sports'), 'Genre_type'] = 'Sport_1'

# Podcast_Name.eq('Business Briefs') Business  17004 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Business Briefs') & 
           all_df.Genre.eq('Business'), 'Genre_type'] = 'Business_0'
all_df.loc[all_df.Podcast_Name.eq('Business Briefs') & 
           all_df.Genre.ne('Business'), 'Genre_type'] = 'Business_1'

# Podcast_Name.eq('Business Insights') Business  19471 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Business Insights') & 
           all_df.Genre.eq('Business'), 'Genre_type'] = 'Business_0'
all_df.loc[all_df.Podcast_Name.eq('Business Insights') & 
           all_df.Genre.ne('Business'), 'Genre_type'] = 'Business_1'

# Podcast_Name.eq('Market Masters') Business   13056 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Market Masters') & 
           all_df.Genre.eq('Business'), 'Genre_type'] = 'Business_0'
all_df.loc[all_df.Podcast_Name.eq('Market Masters') & 
           all_df.Genre.ne('Business'), 'Genre_type'] = 'Business_1'

# Podcast_Name.eq('Finance Focus') Business  17569  -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Finance Focus') & 
           all_df.Genre.eq('Business'), 'Genre_type'] = 'Business_0'
all_df.loc[all_df.Podcast_Name.eq('Finance Focus') & 
           all_df.Genre.ne('Business'), 'Genre_type'] = 'Business_1'

# Podcast_Name.eq('Money Matters') Business   13341 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Money Matters') & 
           all_df.Genre.eq('Business'), 'Genre_type'] = 'Business_0'
all_df.loc[all_df.Podcast_Name.eq('Money Matters') & 
           all_df.Genre.ne('Business'), 'Genre_type'] = 'Business_1'

# Podcast_Name.eq('Brain Boost') Education  11504 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Brain Boost') & 
           all_df.Genre.eq('Education'), 'Genre_type'] = 'Education_0'
all_df.loc[all_df.Podcast_Name.eq('Brain Boost') & 
           all_df.Genre.ne('Education'), 'Genre_type'] = 'Education_1'

# Podcast_Name.eq('Educational Nuggets') Education  12231 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Educational Nuggets') & 
           all_df.Genre.eq('Education'), 'Genre_type'] = 'Education_0'
all_df.loc[all_df.Podcast_Name.eq('Educational Nuggets') & 
           all_df.Genre.ne('Education'), 'Genre_type'] = 'Education_1'

# Podcast_Name.eq('Learning Lab') Education 12271 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Learning Lab') & 
           all_df.Genre.eq('Education'), 'Genre_type'] = 'Education_0'
all_df.loc[all_df.Podcast_Name.eq('Learning Lab') & 
           all_df.Genre.ne('Education'), 'Genre_type'] = 'Education_1'

# Podcast_Name.eq('Study Sessions') Education 13006 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Study Sessions') & 
           all_df.Genre.eq('Education'), 'Genre_type'] = 'Education_0'
all_df.loc[all_df.Podcast_Name.eq('Study Sessions') & 
           all_df.Genre.ne('Education'), 'Genre_type'] = 'Education_1'

# Podcast_Name.eq('Healthy Living') Health  12176 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Healthy Living') & 
           all_df.Genre.eq('Health'), 'Genre_type'] = 'Health_0'
all_df.loc[all_df.Podcast_Name.eq('Healthy Living') & 
           all_df.Genre.ne('Health'), 'Genre_type'] = 'Health_1'

# Podcast_Name.eq('Health Hour') Health  11111 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Health Hour') & 
           all_df.Genre.eq('Health'), 'Genre_type'] = 'Health_0'
all_df.loc[all_df.Podcast_Name.eq('Health Hour') & 
           all_df.Genre.ne('Health'), 'Genre_type'] = 'Health_1'

# Podcast_Name.eq('Wellness Wave') Health  14967 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Wellness Wave') & 
           all_df.Genre.eq('Health'), 'Genre_type'] = 'Health_0'
all_df.loc[all_df.Podcast_Name.eq('Wellness Wave') & 
           all_df.Genre.ne('Health'), 'Genre_type'] = 'Health_1'

# Podcast_Name.eq('Mind & Body') Health 13612 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Mind & Body') & 
           all_df.Genre.eq('Health'), 'Genre_type'] = 'Health_0'
all_df.loc[all_df.Podcast_Name.eq('Mind & Body') & 
           all_df.Genre.ne('Health'), 'Genre_type'] = 'Health_1'

# Podcast_Name.eq('Fitness First') Health 19434 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Fitness First') & 
           all_df.Genre.eq('Health'), 'Genre_type'] = 'Health_0'
all_df.loc[all_df.Podcast_Name.eq('Fitness First') & 
           all_df.Genre.ne('Health'), 'Genre_type'] = 'Health_1'

# Podcast_Name.eq('Funny Folks') Comedy 19599 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Funny Folks') & 
           all_df.Genre.eq('Comedy'), 'Genre_type'] = 'Comedy_0'
all_df.loc[all_df.Podcast_Name.eq('Funny Folks') & 
           all_df.Genre.ne('Comedy'), 'Genre_type'] = 'Comedy_1'

# Podcast_Name.eq('Laugh Line') Comedy 14619 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Laugh Line') & 
           all_df.Genre.eq('Comedy'), 'Genre_type'] = 'Comedy_0'
all_df.loc[all_df.Podcast_Name.eq('Laugh Line') & 
           all_df.Genre.ne('Comedy'), 'Genre_type'] = 'Comedy_1'

# Podcast_Name.eq('Comedy Corner') Comedy 15924 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Comedy Corner') & 
           all_df.Genre.eq('Comedy'), 'Genre_type'] = 'Comedy_0'
all_df.loc[all_df.Podcast_Name.eq('Comedy Corner') & 
           all_df.Genre.ne('Comedy'), 'Genre_type'] = 'Comedy_1'

# Podcast_Name.eq('Humor Hub') Comedy 16083 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Humor Hub') & 
           all_df.Genre.eq('Comedy'), 'Genre_type'] = 'Comedy_0'
all_df.loc[all_df.Podcast_Name.eq('Humor Hub') & 
           all_df.Genre.ne('Comedy'), 'Genre_type'] = 'Comedy_1'

# Podcast_Name.eq('Joke Junction') Comedy 15039 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Joke Junction') & 
           all_df.Genre.eq('Comedy'), 'Genre_type'] = 'Comedy_0'
all_df.loc[all_df.Podcast_Name.eq('Joke Junction') & 
           all_df.Genre.ne('Comedy'), 'Genre_type'] = 'Comedy_1'

# Podcast_Name.eq('Sound Waves') Music 13899 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Sound Waves') & 
           all_df.Genre.eq('Music'), 'Genre_type'] = 'Music_0'
all_df.loc[all_df.Podcast_Name.eq('Sound Waves') & 
           all_df.Genre.ne('Music'), 'Genre_type'] = 'Music_1'

# Podcast_Name.eq('Melody Mix') Music 18855 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Melody Mix') & 
           all_df.Genre.eq('Music'), 'Genre_type'] = 'Music_0'
all_df.loc[all_df.Podcast_Name.eq('Melody Mix') & 
           all_df.Genre.ne('Music'), 'Genre_type'] = 'Music_1'

# Podcast_Name.eq('Music Matters') Music 12648 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Music Matters') & 
           all_df.Genre.eq('Music'), 'Genre_type'] = 'Music_0'
all_df.loc[all_df.Podcast_Name.eq('Music Matters') & 
           all_df.Genre.ne('Music'), 'Genre_type'] = 'Music_1'

# Podcast_Name.eq('Tune Time') Music 17199 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Tune Time') & 
           all_df.Genre.eq('Music'), 'Genre_type'] = 'Music_0'
all_df.loc[all_df.Podcast_Name.eq('Tune Time') & 
           all_df.Genre.ne('Music'), 'Genre_type'] = 'Music_1'

# Podcast_Name.eq('Fashion Forward') Lifestyle  17249 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Fashion Forward') & 
           all_df.Genre.eq('Lifestyle'), 'Genre_type'] = 'Lifestyle_0'
all_df.loc[all_df.Podcast_Name.eq('Fashion Forward') & 
           all_df.Genre.ne('Lifestyle'), 'Genre_type'] = 'Lifestyle_1'

# Podcast_Name.eq('Style Guide') Lifestyle  19305 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Style Guide') & 
           all_df.Genre.eq('Lifestyle'), 'Genre_type'] = 'Lifestyle_0'
all_df.loc[all_df.Podcast_Name.eq('Style Guide') & 
           all_df.Genre.ne('Lifestyle'), 'Genre_type'] = 'Lifestyle_1'

# Podcast_Name.eq('Life Lessons') Lifestyle   14436 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Life Lessons') & 
           all_df.Genre.eq('Lifestyle'), 'Genre_type'] = 'Lifestyle_0'
all_df.loc[all_df.Podcast_Name.eq('Life Lessons') & 
           all_df.Genre.ne('Lifestyle'), 'Genre_type'] = 'Lifestyle_1'

# Podcast_Name.eq('Lifestyle Lounge') Lifestyle  16641 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Lifestyle Lounge') & 
           all_df.Genre.eq('Lifestyle'), 'Genre_type'] = 'Lifestyle_0'
all_df.loc[all_df.Podcast_Name.eq('Lifestyle Lounge') & 
           all_df.Genre.ne('Lifestyle'), 'Genre_type'] = 'Lifestyle_1'

# Podcast_Name.eq('Home & Living') Lifestyle  14650 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Home & Living') & 
           all_df.Genre.eq('Lifestyle'), 'Genre_type'] = 'Lifestyle_0'
all_df.loc[all_df.Podcast_Name.eq('Home & Living') & 
           all_df.Genre.ne('Lifestyle'), 'Genre_type'] = 'Lifestyle_1'

# Podcast_Name.eq('World Watch') News 13997 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('World Watch') & 
           all_df.Genre.eq('News'), 'Genre_type'] = 'News_0'
all_df.loc[all_df.Podcast_Name.eq('World Watch') & 
           all_df.Genre.ne('News'), 'Genre_type'] = 'News_1'

# Podcast_Name.eq('Current Affairs') News 13086 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Current Affairs') & 
           all_df.Genre.eq('News'), 'Genre_type'] = 'News_0'
all_df.loc[all_df.Podcast_Name.eq('Current Affairs') & 
           all_df.Genre.ne('News'), 'Genre_type'] = 'News_1'

# Podcast_Name.eq('Global News') News 13627 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Global News') & 
           all_df.Genre.eq('News'), 'Genre_type'] = 'News_0'
all_df.loc[all_df.Podcast_Name.eq('Global News') & 
           all_df.Genre.ne('News'), 'Genre_type'] = 'News_1'

# Podcast_Name.eq('News Roundup') News 9165 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('News Roundup') & 
           all_df.Genre.eq('News'), 'Genre_type'] = 'News_0'
all_df.loc[all_df.Podcast_Name.eq('News Roundup') & 
           all_df.Genre.ne('News'), 'Genre_type'] = 'News_1'

# Podcast_Name.eq('Daily Digest')  News   13357 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Daily Digest') & 
           all_df.Genre.eq('News'), 'Genre_type'] = 'News_0'
all_df.loc[all_df.Podcast_Name.eq('Daily Digest') & 
           all_df.Genre.ne('News'), 'Genre_type'] = 'News_1'

# Podcast_Name.eq('Detective Diaries') True Crime  17417 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Detective Diaries') & 
           all_df.Genre.eq('True Crime'), 'Genre_type'] = 'True_Crime_0'
all_df.loc[all_df.Podcast_Name.eq('Detective Diaries') & 
           all_df.Genre.ne('True Crime'), 'Genre_type'] = 'True_Crime_1'

# Podcast_Name.eq('Mystery Matters') True Crime  15976 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Mystery Matters') & 
           all_df.Genre.eq('True Crime'), 'Genre_type'] = 'True_Crime_0'
all_df.loc[all_df.Podcast_Name.eq('Mystery Matters') & 
           all_df.Genre.ne('True Crime'), 'Genre_type'] = 'True_Crime_1'

# Podcast_Name.eq('True Crime Stories') True Crime  16361 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('True Crime Stories') & 
           all_df.Genre.eq('True Crime'), 'Genre_type'] = 'True_Crime_0'
all_df.loc[all_df.Podcast_Name.eq('True Crime Stories') & 
           all_df.Genre.ne('True Crime'), 'Genre_type'] = 'True_Crime_1'

# Podcast_Name.eq('Crime Chronicles') True Crime  17336 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Crime Chronicles') & 
           all_df.Genre.eq('True Crime'), 'Genre_type'] = 'True_Crime_0'
all_df.loc[all_df.Podcast_Name.eq('Crime Chronicles') & 
           all_df.Genre.ne('True Crime'), 'Genre_type'] = 'True_Crime_1'

# Podcast_Name.eq('Criminal Minds') True Crime 17697 -> 0 other -> 1
all_df.loc[all_df.Podcast_Name.eq('Criminal Minds') & 
           all_df.Genre.eq('True Crime'), 'Genre_type'] = 'True_Crime_0'
all_df.loc[all_df.Podcast_Name.eq('Criminal Minds') & 
           all_df.Genre.ne('True Crime'), 'Genre_type'] = 'True_Crime_1'

# all_df = all_df.drop(['Genre'], axis=1)


# Add features: 'day_time'

all_df['day_time'] = all_df['Publication_Day'] + '_' + all_df['Publication_Time']

day_time_mapping = {
    'Sunday_Morning': 0,
    'Sunday_Afternoon': 1,    
    'Sunday_Evening': 2,
    'Sunday_Night': 3,
    'Monday_Morning': 4,
    'Monday_Afternoon': 5,
    'Monday_Evening': 6,
    'Monday_Night': 7,
    'Tuesday_Morning': 8,
    'Tuesday_Afternoon': 9,    
    'Tuesday_Evening': 10,
    'Tuesday_Night': 11,
    'Wednesday_Morning': 12,
    'Wednesday_Afternoon': 13,
    'Wednesday_Evening': 14,
    'Wednesday_Night': 15, 
    'Thursday_Morning': 16,
    'Thursday_Afternoon': 17,
    'Thursday_Evening': 18,
    'Thursday_Night': 19,
    'Friday_Morning': 20, 
    'Friday_Afternoon':21, 
    'Friday_Evening': 22,
    'Friday_Night': 23, 
    'Saturday_Morning': 24,
    'Saturday_Afternoon': 25, 
    'Saturday_Evening': 26,
    'Saturday_Night': 27
}

all_df['day_time'] = all_df['day_time'].map(day_time_mapping)
all_df['day_time'] = all_df['day_time'].astype('int8')

# all_df = all_df.drop(['Publication_Day', 'Publication_Time'], axis=1)


train_df = all_df.loc[~all_df.Listening_Time_minutes.isnull()]
test_df = all_df.loc[all_df.Listening_Time_minutes.isnull()]
test_df = test_df.drop('Listening_Time_minutes', axis=1)

print(train_df.shape, test_df.shape)
del all_df


# Target encoding of categorical features

cat_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 
            'Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Genre_type']
            
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
    train_df[c_encode] = train_df[c_encode].astype('float32')


all_df = pd.concat([train_df, test_df], axis=0)


# Unique values in specified axis. 
# Sort in ascending order total value of target encoding

podcast_names = all_df.groupby('Podcast_Name')['Podcast_Name_encode'].sum().sort_values().index
genres = all_df.groupby('Genre')['Genre_encode'].sum().sort_values().index
types = all_df.groupby('Genre_type')['Genre_type_encode'].sum().sort_values().index
days = all_df.groupby('Publication_Day')['Publication_Day_encode'].sum().sort_values().index
times = all_df.groupby('Publication_Time')['Publication_Time_encode'].sum().sort_values().index
episode_sentis = all_df.groupby('Episode_Sentiment')['Episode_Sentiment_encode'].sum().sort_values().index

all_df = all_df.drop(['Podcast_Name_encode', 'Episode_Title_encode', 
                      'Publication_Day_encode', 'Publication_Time_encode',
                      'Genre_encode', 'Episode_Sentiment_encode', 'Genre_type_encode'], axis=1)

# print('Number of unique Podcast_Name: 48\n', podcast_names, '\n')
# print('Number of unique Genre: 10\n', genres)
# print('Number of unique Genre_type: 20\n', types)
# print('Number of unique Publication_Day: 7\n', days)
# print('Number of unique Publication_Time: 4\n', times)
# print('Number of unique Episode_Sentiment: 3\n',episode_sentis)


# Label encoding of categorical features
# All data

all_df['Episode_Title'] = all_df['Episode_Title'].str.replace('Episode ', '').astype('category')

cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Genre_type']

lists = [podcast_names, genres, days, times, episode_sentis, types]

for i, cat in enumerate(cat_cols):
    class_le = LabelEncoder()
    class_le.classes_ = lists[i]
    all_df[cat] = class_le.transform(all_df[cat])
#    all_df[cat] = all_df[cat].astype('int8')
    all_df[cat] = all_df[cat].astype('category')


# World Watch        Episode 50     Sunday    Evening    3.400000  12 World Watch 'Sunday_Evening': 2,
# Sports Central     Episode 89     Wednesday Morning    3.000000  21 Sports Central 'Wednesday_Morning': 12,
# Study Sessions     Episode 50     Sunday    Morning    2.666667  11 Study Sessions 'Sunday_Morning': 0,
# Finance Focus      Episode 50     Sunday    Morning    2.600000  30 Finance Focus  'Sunday_Morning': 0,
# Sport Spot         Episode 35     Tuesday   Afternoon  2.000000  20 Sport Spot     'Tuesday_Afternoon': 9, 
# Business Insights  Episode 50     Friday    Evening    2.000000  40 Business Insights 'Friday_Evening': 22,
# Learning Lab       Episode 50     Sunday    Afternoon  2.000000  5 Learning Lab    'Sunday_Afternoon': 1,
# Finance Focus      Episode 42     Wednesday Morning    2.000000  30 Finance Focus  'Wednesday_Morning': 12,
# Innovators         Episode 28     Saturday  Morning    2.000000  8 Innovators      'Saturday_Morning': 24,
# Finance Focus      Episode 50     Sunday    Evening    1.846154  30 Finance Focus  'Sunday_Evening': 2,
# Finance Focus      Episode 50     Friday    Morning    1.785714  30 Finance Focus  'Friday_Morning': 20, 
# Digital Digest     Episode 50     Sunday    Evening    1.625000  26 Digital Digest  'Sunday_Evening': 2,

zero_list = [[12,'50', 2],
             [21,'89',12],
             [11,'50',0],
             [30,'50',0],
             [20,'35',9],
             [40,'50',22],
             [5,'50',1],
             [30,'42',12],
             [8,'28',24],
             [30,'50',2],
             [30,'50',20],
             [26,'50',2]]

all_df['zero_set'] = 0

for i, list in enumerate(zero_list):
    name = list[0]
    title = list[1]
    dt = list[2]
#    display(all_df.loc[all_df.Podcast_Name.eq(name) & all_df.Episode_Title.eq(title) & all_df.day_time.eq(dt), 
#               ['Podcast_Name', 'Episode_Length_minutes', 'Guest_Popularity_percentage',
#                'Host_Popularity_percentage', 'Listening_Time_minutes', 'Episode_Sentiment']])
    
    all_df.loc[(all_df.Podcast_Name.eq(name)) & 
               (all_df.Episode_Title.eq(title)) &
               (all_df.day_time.eq(dt)) & 
               (all_df.Guest_Popularity_percentage.isnull()) & 
               (all_df.Episode_Sentiment.eq(0)), 'zero_set'] = 1
    
    if name == 8:
        all_df.loc[(all_df.Podcast_Name.eq(name)) & 
                   (all_df.Episode_Title.eq(title)) &
                   (all_df.day_time.eq(dt)), 'zero_set'] = 1

    if (name == 11) | (name == 12) | (name == 40):
        all_df.loc[(all_df.Podcast_Name.eq(name)) & 
                   (all_df.Episode_Title.eq(title)) &
                   (all_df.day_time.eq(dt)) & 
                   (all_df.Guest_Popularity_percentage.isnull()) & 
                   (all_df.Episode_Sentiment.eq(1)), 'zero_set'] = 1


# World Watch        Episode 48     Sunday    Evening    1.600000  12 World Watch     'Sunday_Evening': 2,
# Sports Central     Episode 66     Sunday    Morning    1.588235  21 Sports Central  'Sunday_Morning': 0,
# World Watch        Episode 66     Sunday    Evening    1.538462  12 World Watch     'Sunday_Evening': 2,
# Game Day           Episode 48     Sunday    Morning    1.500000  39 Game Day        'Sunday_Morning': 0,
####### Study Sessions     Episode 42     Sunday    Morning    1.500000  11 Study Sessions   'Sunday_Morning': 0,
####### Digital Digest     Episode 50     Friday    Evening    1.500000  26 Digital Digest   'Friday_Evening': 22,
####### Finance Focus      Episode 38     Saturday  Morning    1.500000  30 Finance Focus    'Saturday_Morning': 24,
# Sports Central     Episode 50     Friday    Morning    1.500000  21 Sports Central   'Friday_Morning': 20, 
# Sports Central     Episode 66     Sunday    Evening    1.473684  21 Sports Central   'Sunday_Evening': 2,

zero_list_2 = [[12,'48', 2],
               [21,'66', 0],
               [12,'66', 2],
               [39,'48', 0],
#               [11,'42', 0],
#               [26,'50',22],
#               [30,'38',24],
               [21,'50',20],
               [21,'66',2]]

for i, list in enumerate(zero_list_2):
    name = list[0]
    title = list[1]
    dt = list[2]
#    display(all_df.loc[all_df.Podcast_Name.eq(name) & 
#            all_df.Episode_Title.eq(title) &
#            all_df.day_time.eq(dt) & all_df.Episode_Length_minutes.ge(8), 
#               ['Podcast_Name', 'Episode_Length_minutes', 'Guest_Popularity_percentage',
#                'Host_Popularity_percentage', 'Listening_Time_minutes', 'Episode_Sentiment']])
    
    all_df.loc[(all_df.Podcast_Name.eq(name)) & 
               (all_df.Episode_Title.eq(title)) &
               (all_df.day_time.eq(dt)) & 
               (all_df.Guest_Popularity_percentage.isnull()) & 
               (all_df.Episode_Sentiment.eq(0)) & 
               (all_df.Episode_Length_minutes.le(8)), 'zero_set'] = 1 
    
    if (name == 12) | (name == 21):
        all_df.loc[(all_df.Podcast_Name.eq(name)) & 
                   (all_df.Episode_Title.eq(title)) &
                   (all_df.day_time.eq(dt)) & 
                   (all_df.Guest_Popularity_percentage.isnull()) & 
                   (all_df.Episode_Sentiment.eq(1)) &
                   (all_df.Episode_Length_minutes.le(8)), 'zero_set'] = 1
    

all_df['zero_set'] = all_df['zero_set'].astype('int8')



# Finance Focus      Episode 42     Sunday    Morning    1.461538  30 Finance Focus   'Sunday_Morning': 0,   
# Sports Central     Episode 66     Wednesday Morning    1.400000  21 Sports Central  'Wednesday_Morning': 12,
# Finance Focus      Episode 66     Friday    Evening    1.400000  30 Finance Focus   'Friday_Evening': 22,
# Finance Focus      Episode 50     Friday    Evening    1.375000  30 Finance Focus   'Friday_Evening': 22,
# Finance Focus      Episode 48     Wednesday Morning    1.375000  30 Finance Focus   'Wednesday_Morning': 12,
# Finance Focus      Episode 48     Friday    Morning    1.333333  30 Finance Focus   'Friday_Morning': 20, 
# Study Sessions     Episode 50     Sunday    Afternoon  1.333333  11 Study Sessions  'Sunday_Afternoon': 1,
# Sports Central     Episode 66     Sunday    Afternoon  1.333333  21 Sports Central  'Sunday_Afternoon': 1,
# Sports Central     Episode 42     Sunday    Morning    1.176471  21 Sports Central   'Sunday_Morning': 0,
# Sport Spot         Episode 66     Sunday    Evening    1.166667  20 Sport Spot        'Sunday_Evening': 2,
# Business Insights  Episode 50     Sunday    Evening    1.111111  40 Business Insights  'Sunday_Evening': 2,
# Finance Focus      Episode 48     Friday    Evening    1.111111  30 Finance Focus    'Friday_Evening': 22,
# Finance Focus      Episode 48     Sunday    Morning    1.111111  30 Finance Focus    'Sunday_Morning': 0,
# Finance Focus      Episode 69     Friday    Morning    1.105263  30 Finance Focus    'Friday_Morning': 20, 
# Finance Focus      Episode 48     Sunday    Evening    1.090909  30 Finance Focus   'Sunday_Evening': 2,

zero_list_3 = [[30,'42',0],
               [21,'66',12],
               [30,'66',22],
               [30,'50',22],
               [30,'48',12],
               [30,'48',20],
               [11,'50',1],
               [21,'66',1],
               [21,'42',0],
               [20,'66',2],
               [40,'50',2],
               [30,'48',22],
               [30,'48',0],
               [30,'69',20],
               [30,'48',2]]

for i, list in enumerate(zero_list_3):
    name = list[0]
    title = list[1]
    dt = list[2]

#    display(all_df.loc[all_df.Podcast_Name.eq(name) & 
#            all_df.Episode_Title.eq(title) &
#            all_df.day_time.eq(dt), 
#                  ['Podcast_Name', 'Episode_Length_minutes', 'Guest_Popularity_percentage',
#                   'Host_Popularity_percentage', 'Listening_Time_minutes', 'Episode_Sentiment']])

    all_df.loc[(all_df.Podcast_Name.eq(name)) & 
               (all_df.Episode_Title.eq(title)) &
               (all_df.day_time.eq(dt)) & 
               (all_df.Guest_Popularity_percentage.isnull()) & 
               (all_df.Episode_Sentiment.eq(0)) & 
               (all_df.Episode_Length_minutes.le(8)), 'zero_set'] = 1 
    
    if (i != 2) & (i != 7):
        all_df.loc[(all_df.Podcast_Name.eq(name)) & 
                   (all_df.Episode_Title.eq(title)) &
                   (all_df.day_time.eq(dt)) & 
                   (all_df.Guest_Popularity_percentage.isnull()) & 
                   (all_df.Episode_Sentiment.eq(1)) & 
                   (all_df.Episode_Length_minutes.le(8)), 'zero_set'] = 1 
    
    


# Outlier: Numerical features of training data and test data
#         'Episode_Length_minutes','Host_Popularity_percentage', 
#         'Guest_Popularity_percentage', 'Number_of_Ads'

num_cols = ['Episode_Length_minutes',
            'Host_Popularity_percentage', 
            'Guest_Popularity_percentage']

all_p99 = all_df[num_cols].quantile(0.99)
all_p01 = all_df[num_cols].quantile(0.01)

all_df[num_cols] = all_df[num_cols].clip(all_p01, all_p99, axis=1)

all_df['Number_of_Ads'] = all_df['Number_of_Ads'].apply(
    lambda x: x if x < 3.0 else (4.0 if x < 50 else (5.0 if x < 103 else 1.0)))

all_df['Number_of_Ads'] = all_df['Number_of_Ads'].astype('int8')


all_df[['Episode_Length_minutes','Host_Popularity_percentage', 
        'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']].describe().T


# Fill in missing values using the k-Nearest Neighbors
# All data

all_groups_df = all_df.groupby(['Podcast_Name', 'Episode_Title'])['id'].agg('count').reset_index()
all_groups = all_groups_df.loc[all_groups_df.id > 10, ['Podcast_Name', 'Episode_Title']].reset_index()

all_df['epi_len_knn'] = all_df['Episode_Length_minutes'].values
all_df['guest_knn'] = all_df['Guest_Popularity_percentage'].values

for i in range(len(all_groups)):
    name = all_groups.loc[i, 'Podcast_Name']
    title = all_groups.loc[i, 'Episode_Title']
    
    X = all_df.loc[(all_df.Podcast_Name == name) &
                   (all_df.Episode_Title == title), ['Episode_Length_minutes',
                                                     'Host_Popularity_percentage',
                                                     'Guest_Popularity_percentage',
                                                     'Number_of_Ads',
                                                    ]]
    
    if (X['Episode_Length_minutes'].isnull().sum() > 0) | (X['Guest_Popularity_percentage'].isnull().sum() > 0):
    # (X['Number_of_Ads'].isnull().sum() > 0)
        array_X = np.array(X.values)
        
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
        imputer_X = imputer.fit_transform(array_X)
        data = pd.DataFrame(imputer_X, 
                            columns=['Episode_Length_minutes',
                                     'Host_Popularity_percentage',
                                     'Guest_Popularity_percentage',
                                     'Number_of_Ads',
                                     ])
            
        all_df.loc[(all_df.Podcast_Name == name) & 
                   (all_df.Episode_Title == title), 'epi_len_knn'] = data['Episode_Length_minutes'].values.astype('float32')
        all_df.loc[(all_df.Podcast_Name == name) & 
                   (all_df.Episode_Title == title), 'guest_knn'] = data['Guest_Popularity_percentage'].values.astype('float32')
      



# Correct the value

all_df.loc[all_df.Episode_Length_minutes.isnull(), 
           'epi_len_knn'] = all_df.loc[all_df.Episode_Length_minutes.isnull(), 'epi_len_knn'] + 10


all_df[['Episode_Length_minutes','Host_Popularity_percentage', 
        'Guest_Popularity_percentage', 'Number_of_Ads', 
        'Listening_Time_minutes', 
        'epi_len_knn', 'guest_knn']].describe().T


# Add features: 'group1_epi_len_max', 'group1_guest_max', 'group1_host_max'

all_df['group1_epi_len_max'] = all_df.groupby(['Podcast_Name', 'Episode_Title', 
                                               'Publication_Day', 'Publication_Time',
                                               'Episode_Sentiment'])['epi_len_knn'].transform('max').astype('float32')
all_df['group1_guest_max'] = all_df.groupby(['Podcast_Name', 'Episode_Title',
                                             'Publication_Day', 'Publication_Time',
                                             'Episode_Sentiment'])['guest_knn'].transform('max').astype('float32')
all_df['group1_host_max'] = all_df.groupby(['Podcast_Name', 'Episode_Title', 
                                            'Publication_Day', 'Publication_Time',
                                            'Episode_Sentiment'])['Host_Popularity_percentage'].transform('max').astype('float32')

# all_df.head(3)


# Add features: 'group3_epi_len_mean', 'group3_guest_mean', 'group3_host_mean'

all_df['group3_epi_len_mean'] = all_df.groupby(['Podcast_Name', 'Episode_Title', 
                                                'Publication_Day', 'Publication_Time',
                                                'Episode_Sentiment'])['epi_len_knn'].transform('mean').astype('float32')
all_df['group3_guest_mean'] = all_df.groupby(['Podcast_Name', 'Episode_Title', 
                                              'Publication_Day', 'Publication_Time',
                                              'Episode_Sentiment'])['guest_knn'].transform('mean').astype('float32')
all_df['group3_host_mean'] = all_df.groupby(['Podcast_Name', 'Episode_Title', 
                                             'Publication_Day', 'Publication_Time',
                                             'Episode_Sentiment'])['Host_Popularity_percentage'].transform('mean').astype('float32')

# all_df.head(3)


# Add features: 'group6_epi_len_mean', 'group7_epi_len_mean'

all_df['group6_epi_len_mean'] = all_df.groupby(['Episode_Title'])['epi_len_knn'].transform('mean').astype('float32')
all_df['group7_epi_len_mean'] = all_df.groupby(['Publication_Day', 'Publication_Time',])['epi_len_knn'].transform('mean').astype('float32')



# Add features: 'day_time_sin', 'day_time_cos'

# Encode cyclical Features
def encode(data, col, max_val):
    sin = f'{col}_sin'
    cos = f'{col}_cos'
    data[sin] = np.sin(2 * np.pi * (data[col] + 1) / max_val)
    data[cos] = np.cos(2 * np.pi * (data[col] + 1) / max_val)
    return data


# all_df = encode(all_df, 'Publication_Day', all_df.Publication_Day.max()+1)
# all_df = encode(all_df, 'Publication_Time', all_df.Publication_Time.max()+1)

all_df = encode(all_df, 'day_time', all_df.day_time.max()+1)
# all_df['day_time'] = all_df['day_time'].astype('category')
all_df['day_time'] = all_df['day_time']

# sns.scatterplot(data=all_df, x='Publication_Day_sin', y='Publication_Day_cos').set_aspect('equal')
sns.scatterplot(data=all_df, x='day_time_sin', y='day_time_cos').set_aspect('equal')


# Popularity_percentage

target_zero = all_df.loc[all_df.Listening_Time_minutes.eq(0.0)]
target_non_zero = all_df.loc[all_df.Listening_Time_minutes.ne(0.0)]

# Case:'Listening_Time_minutes' = 0
print("Target equal zero: Host_Popularity_percentage's ratio:      ", target_zero['Host_Popularity_percentage'].quantile(0.5), '%')
print("Target equal zero: Guest_Popularity_percentage's ratio:     ", target_zero['guest_knn'].quantile(0.5), '%')

# Case: 'Listening_Time_minutes' != 0
print("Target not equal zero: Host_Popularity_percentage's ratio:  ", target_non_zero['Host_Popularity_percentage'].quantile(0.5), '%')
print("Target not equal zero: Guest_Popularity_percentage's ratio: ", target_non_zero['guest_knn'].quantile(0.5), '%')

# Host_Popularity - Guest_Popularity
print("Target equal zero:　    host - guest", 
      (target_zero['Host_Popularity_percentage'].quantile(0.5) - target_zero['guest_knn'].quantile(0.5)).round(), '%')
print("Target not equal zero:　host - guest", 
      (target_non_zero['Host_Popularity_percentage'].quantile(0.5) - target_non_zero['guest_knn'].quantile(0.5)).round(), '%')

# Host_Popularity + Guest_Popularity
print("Target equal zero:　    host + guest", 
      (target_zero['Host_Popularity_percentage'].quantile(0.5) + target_zero['guest_knn'].quantile(0.5)).round(), '%')
print("Target not equal zero:　host + guest", 
      (target_non_zero['Host_Popularity_percentage'].quantile(0.5) + target_non_zero['guest_knn'].quantile(0.5)).round(), '%')

# Add features  
# Host_Popularity - Guest_Popularity

all_df['Host_guest_diff'] = np.abs(all_df['Host_Popularity_percentage'] - all_df['guest_knn']).astype('float32')

del target_zero, target_non_zero


# Add features: 'epi_len_epi_senti'

all_df['epi_len_epi_senti'] = np.round(all_df['epi_len_knn'] * (all_df['Episode_Sentiment']).astype('int8'), 5).astype('float32')


train_df = all_df.loc[~all_df.Listening_Time_minutes.isnull()]
test_df = all_df.loc[all_df.Listening_Time_minutes.isnull()]
test_df = test_df.drop('Listening_Time_minutes', axis=1)

print(train_df.shape, test_df.shape)
del all_df


train_df.info()


# XGBoost 

base_train_df = train_df.drop(['Listening_Time_minutes', 
                               'id', 
#                               'Episode_Length_minutes',        # use Nan 
#                               'Guest_Popularity_percentage',   # use Nan
#                               'epi_len_knn',                   # use KNN data
                               'guest_knn', 
                              ], axis=1)
base_test_df = test_df.drop([
                             'id', 
#                             'Episode_Length_minutes',        # use Nan 
#                             'Guest_Popularity_percentage',   # use Nan 
#                             'epi_len_knn',                   # use KNN data
                             'guest_knn',                   
                            ], axis=1)

sc = MinMaxScaler()

sc.fit(base_train_df)
X_std = sc.transform(base_train_df)
test_X_std = sc.transform(base_test_df)

y = train_df['Listening_Time_minutes']

X_train, X_valid, y_train, y_valid = train_test_split(
    X_std, y, test_size=0.2,
    random_state=42, shuffle=True
)

print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

# 250430 use optuna
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist', 
    'enable_categorical': True,
    'num_round': 10000,
    'random_state': 42,
    'learning_rate': 0.00704530083465739,
    'num_round': 10000,
    'random_state': 42,
    'max_depth': 9,
    'min_child_weight': 7.80375183440352,
    'colsample_bytree': 0.9045042518863866,
    'subsample': 0.9425164197814674,
    'gamma': 0.7991585662251379,
    'alpha': 46.1479362306784,
    'lambda': 78.05291763084026,
    'verbosity': 0 ,
}

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_eval = xgb.DMatrix(X_valid, label=y_valid)
evals = [(xgb_train, 'train'), (xgb_eval, 'eval')]

model_xgb = xgb.train(
    xgb_params, xgb_train,
    evals=evals,
    num_boost_round=10000,
    early_stopping_rounds=50,
    verbose_eval=200,
)
    
y_pred_xgb = model_xgb.predict(xgb_eval)
score = np.sqrt(mean_squared_error(y_valid, y_pred_xgb))
print('rmse:', score)



x = sc.inverse_transform(X_valid)
df = pd.DataFrame(x, columns=base_train_df.columns)
df['target'] = y_valid.values
df['y_pred'] = y_pred_xgb

# Correct the values: less than zero
df.loc[df.y_pred < 0, 'y_pred'] = 0
y_zero_pred = df.y_pred.values
print('rmse:', np.sqrt(mean_squared_error(y_valid, y_zero_pred)))

# Correct the values: one condition 'zero_set' = 1
df.loc[df.zero_set == 1, 'y_pred'] = 0
y_zero_pred = df.y_pred.values
print('rmse:', np.sqrt(mean_squared_error(y_valid, y_zero_pred)))


xgb_test = xgb.DMatrix(pd.DataFrame(test_X_std))

pred_xgb = model_xgb.predict(xgb_test)

test_X = sc.inverse_transform(test_X_std)
df = pd.DataFrame(test_X, columns=base_test_df.columns)
df['y_pred'] = pred_xgb

df.loc[df.y_pred < 0, 'y_pred'] = 0
df.loc[df.zero_set == 1, 'y_pred'] = 0

y_pred = df.y_pred.values

preds_array_xgb = np.array(y_pred)
submission_df['Listening_Time_minutes'] = preds_array_xgb

display(submission_df.head())
print(submission_df.shape)


submission_df.to_csv('/kaggle/working/predict_listening_time_xgb_16.csv', index=False) 




