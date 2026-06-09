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


X_full = pd.read_csv("../input/playground-series-s5e9/train.csv")     # the training dataset
X_test = pd.read_csv("../input/playground-series-s5e9/test.csv")      # the test dataset

sample = pd.read_csv("../input/playground-series-s5e9/sample_submission.csv") # sample submission, just for reference


# taking a glance over the data
X_full.head()


# a basic info, it's best to have a look here before going ahead
X_full.describe()


# looking over the features, and if missing values exist
X_full.info()


dump = X_full.pop('id') # not needed for now
#X_full = pd.concat([X_full, X_full_og])


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# separating features and labels

y = X_full.pop('BeatsPerMinute')
ids = X_test.pop('id')


# the log of AcousticQuality gives better result. (Just a hit and trial)
X_full['new'] = np.log(X_full['AcousticQuality'])
X_test['new'] = np.log(X_test['AcousticQuality'])


# splitting the data
X_train, X_val, y_train, y_val = train_test_split(X_full, y, test_size = 0.2, random_state = 0)


# As the title of notebook suggest, we have a linear regression model
model = LinearRegression()
model.fit(X_train, y_train) # fitting over the train data


# making prediction and validating
pred = model.predict(X_val)
mean_squared_error(pred, y_val, squared = False) # validation


# checking the impact of individual columns 

for i in X_full:
    score = -1 * cross_val_score(model, X_full[[i]], y,
                            cv = 5, scoring = 'neg_mean_squared_error')
    print(i , ":", (score ** 0.5).mean())


# special_cols = ['MoodScore', 'TrackDurationMs',
#                'RhythmScore', 'VocalContent',
#                'LivePerformanceLikelihood', 'AudioLoudness',
#                'new', 'AcousticQuality', 'Energy']

# these columns are special, because model performs well when others are excluded


"""Let's make use of all the columns"""


# Cross Validation over whole data

score = -1 * cross_val_score(model, X_full, y,
                            cv = 5, scoring = 'neg_mean_squared_error')
(score ** 0.5).mean()



model.fit(X_full, y)  # here comes the final model


final = model.predict(X_test)


output = pd.DataFrame({'id' : ids, 
                       'BeatsPerMinute' : final})
output.head()


output.to_csv('submission.csv', index = False)

