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


!pip install -U pip
!pip install -U setuptools wheel
!pip install -U "mxnet<2.0.0" bokeh==2.0.1
!pip install autogluon --no-cache-dir
!pip install -U --quiet "xgboost>=2.1.4"
# Without --no-cache-dir, smaller aws instances may have trouble installing


import pandas as pd
from autogluon.tabular import TabularPredictor


train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv',parse_dates=['datetime'])
train.head()


test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv',parse_dates=['datetime'])
test.head()


submission = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv',parse_dates=['datetime'])
submission.head()


train_data = train.drop(columns=['casual', 'registered'])
train_data[['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp','atemp', 'humidity', 'windspeed']]


predictor = TabularPredictor(
    label='count',
    eval_metric='root_mean_squared_error'
).fit(
    train_data=train_data,
    time_limit=600,              # 10 minutes
    presets='best_quality'       # focus on best model
)


predictor.fit_summary()


predictor.leaderboard(silent=True).plot(kind="bar", x="model", y="score_val")


test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv',
                   parse_dates=['datetime'])

predictions = predictor.predict(test)        # <-- use **predict**, not evaluate



predictions.head()


# Describe the `predictions` series to see if there are any negative values
predictions.describe()

# How many negative values do we have?
(predictions < 0).sum()

# Set them to zero
predictions = predictions.clip(lower=0)



# 1) Build the submission DataFrame
submission = pd.DataFrame({
    "datetime": test["datetime"],   # from the original test.csv
    "count":    predictions         # the Series you just generated
})

# 2) Save it to disk
submission.to_csv("submission.csv", index=False)

# 3) Quick peek
submission.head()






corr_series = (
    train_data
    .corr(numeric_only=True)['count']
    .drop('count')                       # remove self‑correlation
    .sort_values(ascending=False)
)

# Display correlations in an interactive table
import ace_tools as tools; tools.display_dataframe_to_user(
    "Feature ↔ count Pearson correlations",
    corr_series.to_frame(name="Pearson r")
)

# Plot the correlations
plt.figure()
corr_series.plot(kind='barh')            # horizontal bar chart
plt.gca().invert_yaxis()                 # highest correlations at the top
plt.xlabel("Pearson correlation with 'count'")
plt.title("Feature importance via correlation")
plt.tight_layout()
plt.show()




