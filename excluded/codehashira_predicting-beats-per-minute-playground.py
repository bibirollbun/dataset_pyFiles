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


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
y = train.pop('BeatsPerMinute')
X = train
X.head()
X.columns


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def add_features(df):
    df.copy()
    df["Energy_per_sec"] = df["Energy"] / (df["TrackDurationMs"] / 1000)
    df["Rhythm_per_sec"] = df["RhythmScore"] / (df["TrackDurationMs"] / 1000)
    
    df["TemporalDensity"] = (
        df["InstrumentalScore"] + df["VocalContent"] + df["RhythmScore"]
    ) / (df["TrackDurationMs"] / 1000)
    
    df["Rhythm_x_Energy"] = df["RhythmScore"] * df["Energy"]
    df["log_loudness"] = np.log1p(df["AudioLoudness"])
    df["sqrt_energy"] = np.sqrt(df["Energy"])
    df["Rhythm_to_Energy"] = df["RhythmScore"] / (df["Energy"] + 1e-6)
    
    # Latent genre cluster
    kmeans = KMeans(n_clusters=6, random_state=42)
    df["ClusterGenre"] = kmeans.fit_predict(
        df[["RhythmScore","MoodScore","Energy","VocalContent","InstrumentalScore"]]
    )
    return df
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
X_train, X_test = add_features(X_train), add_features(X_test)


from lightgbm import LGBMRegressor
param = {'n_estimators':90,'device':'gpu','learning_rate': 0.023608029933362674, 'num_leaves': 40, 'max_depth': 10, 'min_child_samples': 73, 'subsample': 0.8451064985255994, 'colsample_bytree': 0.7660067859149915, 'reg_alpha': 8.179934181989463, 'reg_lambda': 4.692202186153051e-07}
lgbm_model = LGBMRegressor(**param)
lgbm_model.fit(X_train, y_train)
lgbm_preds =  lgbm_model.predict(X_test)
lgbm_score = mean_squared_error(y_test, lgbm_preds, squared = False)
lgbm_score


X, test = add_features(X), add_features(test)
lgbm_model.fit(X,y)
predictions = lgbm_model.predict(test)


submission_pd = pd.DataFrame({'id': test['id'],
                             'BeatsPerMinute': predictions})
submission_pd.to_csv('submission.csv', index = False)

