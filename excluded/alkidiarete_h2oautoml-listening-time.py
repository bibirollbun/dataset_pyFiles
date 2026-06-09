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
import matplotlib.pyplot as plt
import seaborn as sns
import h2o
from h2o.frame import H2OFrame
from h2o.automl import H2OAutoML
from h2o.automl import get_leaderboard


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train = train[train['Number_of_Ads']<10]


day_map = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
           'Thursday': 4, 'Friday': 5, 'Saturday': 6}
time_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

# Preprocessing function
def preprocess(df):
    # Convert Episode_Title to integer
    df['Episode_Title'] = df['Episode_Title'].str.replace('Episode', '', regex=False).astype(int)
    
    # Create Is_weekend flag
    df['Is_weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Encode categorical features
    df['Publication_Day'] = df['Publication_Day'].map(day_map)
    df['Publication_Time'] = df['Publication_Time'].map(time_map)
    
    # Encode sentiment if available
    if 'Episode_Sentiment' in df.columns:
        df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)
    
    return df



train = preprocess(train)
test = preprocess(test)


train = train.drop(columns = ['id'])
test = test.drop(columns = ['id'])


train.head()


test.head()


h2o.init()


train_h2o = h2o.H2OFrame(train)


test_h2o = h2o.H2OFrame(test)


x = train_h2o .columns
y = 'Listening_Time_minutes'
x.remove(y)



%%time

aml = H2OAutoML(
    max_runtime_secs=10800, 
    seed=15,
    sort_metric="RMSE",
    distribution="AUTO",
    nfolds=5  
)

aml.train(
    x=x,
    y=y,
    training_frame=train_h2o
)


lb = aml.leaderboard
lb.head(rows = lb.nrows)


best_model = aml.get_best_model()
best_model


predictions = best_model.predict(test_h2o)


predictions_h2o = predictions.as_data_frame()


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

sub['Listening_Time_minutes'] = predictions_h2o
sub.to_csv('submission.csv', index=False)
sub.head()


h2o.cluster().shutdown()

