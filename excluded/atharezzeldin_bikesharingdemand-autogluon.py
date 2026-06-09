!pip install -U pip
!pip install -U setuptools wheel
!pip install -U "mxnet<2.0.0" bokeh==2.0.1
!pip install autogluon --no-cache-dir
# Without --no-cache-dir, smaller aws instances may have trouble installing


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from autogluon.tabular import TabularPredictor

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


train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv",parse_dates=['datetime'])
train.head()


train.info()


original = train.copy()
original.set_index('datetime', inplace=True)



plt.figure(figsize=(14, 6))
plt.plot(original.index, original['count'], label='Bike Rentals', alpha=0.7)
plt.title("Bike Rental Count Over Time")
plt.xlabel("Date and Time")
plt.ylabel("Count")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv",parse_dates=['datetime'])
test.head()


test.info()


submission = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv", parse_dates=['datetime'])
submission.head()


for df in [train, test]:
    
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

# Drop original datetime
train = train.drop(columns=['datetime'])
test = test.drop(columns=['datetime'])


train = train.drop(columns=['casual', 'registered'])


test.info()


train.info()


train["season"] = train["season"].astype('category')
train["weather"] = train['weather'].astype('category')
train["holiday"]= train["holiday"].astype('category')
train["workingday"]= train["workingday"].astype('category')


test["season"] = test['season'].astype('category')
test["weather"] = test['weather'].astype('category')
test["holiday"]= test["holiday"].astype('category')
test['workingday']= test['workingday'].astype('category')


train.hist(bins=20, figsize=(20, 10));


train['count'] = np.log1p(train['count'])  # Log transform



# plt.figure(figsize=(8, 6))
# sns.boxplot(x=train['count'])
# plt.title("Box Plot of Bike Rental Counts")
# plt.xlabel("Count")
# plt.show()


predictor = TabularPredictor(label='count', eval_metric='rmse').fit(
    train,
    presets='best_quality',
    time_limit=600
   
)


predictor.fit_summary()


lb = predictor.leaderboard(extra_info=True)
print(lb[['model', 'score_val', 'can_infer', 'fit_order', 'stack_level']])


predictor.leaderboard(silent= True).plot(kind='bar', x='model' , y='score_val' )


# feature_importance = predictor.feature_importance(data=train)
# feature_importance


# predictions = predictor.predict(test)
# predictions.head()
predictions_log = predictor.predict(test)
predictions_log.head()


predictions_log.describe()


predictions = np.expm1(predictions_log)
predictions.head()


predictions.describe()


negative_counts= predictions [predictions <0 ].count()
negative_counts


predictions = predictions.clip(lower=0)


submission["count"] = predictions
submission.to_csv("submission.csv", index=False)




