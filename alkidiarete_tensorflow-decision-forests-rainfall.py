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


import tensorflow_decision_forests as tfdf

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from IPython.core.magic import register_line_magic
from IPython.display import Javascript

try:
  from wurlitzer import sys_pipes
except:
  from colabtools.googlelog import CaptureLog as sys_pipes

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

test_ids = test['id']

train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


test['winddirection'].fillna(train['winddirection'].median(), inplace=True)

train['temp_spread'] = train['maxtemp'] - train['mintemp']
test['temp_spread'] = test['maxtemp'] - test['mintemp']

train['dew_depression'] = train['temparature'] - train['dewpoint']
test['dew_depression'] = test['temparature'] - test['dewpoint']

for df in [train, test]:
    radians = np.radians(df['winddirection'])
    df['wind_u'] = df['windspeed'] * np.sin(radians)  # Komponen Timur-Barat
    df['wind_v'] = df['windspeed'] * np.cos(radians)  # Komponen Utara-Selatan

train['cloud_sun_ratio'] = train['cloud'] / (train['sunshine'] + 1e-6)
test['cloud_sun_ratio'] = test['cloud'] / (test['sunshine'] + 1e-6)

threshold = 2.0  
train['saturated_air'] = (train['dew_depression'] < threshold).astype(int)
test['saturated_air'] = (test['dew_depression'] < threshold).astype(int)

train['pressure_humidity'] = train['pressure'] * train['humidity']
test['pressure_humidity'] = test['pressure'] * test['humidity']

train['thermal_comfort'] = 0.5 * (train['temparature'] + train['maxtemp']) * train['humidity'] / 100
test['thermal_comfort'] = 0.5 * (test['temparature'] + test['maxtemp']) * test['humidity'] / 100

train['wind_chill'] = 13.12 + 0.6215*train['temparature'] - 11.37*(train['windspeed']**0.16) + 0.3965*train['temparature']*(train['windspeed']**0.16)
test['wind_chill'] = 13.12 + 0.6215*test['temparature'] - 11.37*(test['windspeed']**0.16) + 0.3965*test['temparature']*(test['windspeed']**0.16)

for df in [train, test]:
    df['wind_dir_sin'] = np.sin(2 * np.pi * df['winddirection'] / 360)
    df['wind_dir_cos'] = np.cos(2 * np.pi * df['winddirection'] / 360)


train = train.drop(['day'], axis=1)
test = test.drop(['day'], axis=1)


train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train, label="rainfall")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test)


model = tfdf.keras.RandomForestModel()

model.compile(metrics=["accuracy"])

with sys_pipes():
  model.fit(x=train_ds)


model.make_inspector().evaluation()


model.summary()


model.make_inspector().variable_importances()


logs = model.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")

plt.show()


predictions = model.predict(test_ds)

predictions = predictions.flatten()  


submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': predictions
})

submission.to_csv('submission.csv', index=False)

submission.head()

