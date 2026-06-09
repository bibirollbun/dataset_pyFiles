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


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
import optuna
import xgboost as xgb


sub_ = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
train_ = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_ = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
#DATASETS SUMMARY
class get_summary:
    def __init__(self, x):
        self.x = x if isinstance(x, pd.DataFrame) else pd.DataFrame()
    def data_set(self):
        #checks for duplicate
        duplicate = self.x.duplicated().any()
        #drop duplicates 
        if duplicate == True:
            self.x.drop_duplicates(inplace=True)
            self.x.reset_index(drop=True)
        #checks for empty values
        null = self.x.isna().sum().any()
        #missing values
        total_missing = self.x.isnull().sum().sum()
        #data types
        data_type = self.x.dtypes
        #shape
        shapes = self.x.shape
        return f"Duplicate: {duplicate}\nNull: {null}\nMissing_value: {total_missing}\nTypes:\n{data_type}\nShape: {shapes}"
     #missing values
    def total_missing(self):
        missing_vals = self.x.isnull().sum()
        cols_with_missing = missing_vals[missing_vals > 0]
        return cols_with_missing.to_dict()
print(f"Training dataset:\n{get_summary(train_).data_set()}\nTest dataset:\n{get_summary(test_).data_set()}")
print(f"columns with missing values train\n{get_summary(train_).total_missing()}\ncolumns with missing values test\n{get_summary(test_).total_missing()}")


train_.describe().T


for val in train_[['Publication_Day', 'Publication_Time']]:
    plt.figure(figsize=(10, 5))
    sns.countplot(train_, x=val)
    plt.title(f"Listening time by {val}")
    plt.show()

for col in ['Genre', 'Episode_Sentiment']:
    vals = train_[col].value_counts()
    labl = vals.index
    plt.pie(x=vals,
            labels=labl,
            autopct='%.2f',
            explode=[0.1 if val == vals.max() else 0 for val in vals])
    plt.title(f"Listening distribution by {col}")
    plt.show()


train_.head(5)


#train
columns_to_drop = ['id',
                    'Podcast_Name',
                    'Episode_Title', 
                    'Episode_Length_minutes',
                    'Guest_Popularity_percentage',
                    'Listening_Time_minutes']
train_features = train_.drop(train_[columns_to_drop], axis=1)
train_features['Number_of_Ads'] = train_features['Number_of_Ads'].fillna(train_features['Number_of_Ads'].mean())
train_features.head(3)


#test
columns_to_drop = ['id',
                    'Podcast_Name',
                    'Episode_Title', 
                    'Episode_Length_minutes',
                    'Guest_Popularity_percentage']
test_features = test_.drop(test_[columns_to_drop], axis=1)
test_features.head(2)


#train
def mapping_values(train_features):
    train_features['Publication_Day'] = train_features['Publication_Day'].map({'Monday': 1.1,
                                                                               'Tuesday': 1.2,
                                                                               'Wednesday': 1.3,
                                                                               'Thursday': 1.4,
                                                                               'Friday': 1.5,
                                                                               'Saturday': 1.6,
                                                                               'Sunday': 1.7})
    train_features['Publication_Time'] = train_features['Publication_Time'].map({'Night': 0.1,
                                                                                 'Evening': 0.2,
                                                                                 'Afternoon': 0.3,
                                                                                 'Morning': 0.4})
    train_features['Episode_Sentiment'] = train_features['Episode_Sentiment'].map({'Neutral': 1.5,
                                                                                   'Negative': 1.6,
                                                                                   'Positive': 1.7})
    train_features['Genre'] = train_features['Genre'].map({'Sports': 0.0,
                                                           'Technology': 0.01,
                                                           'True Crime': 0.02,
                                                           'Lifestyle': 0.03,
                                                           'Comedy': 0.04,
                                                           'Business': 0.05,
                                                           'Health': 0.06,
                                                           'News': 0.07,
                                                           'Music': 0.08,
                                                           'Education': 0.09})
    return train_features.head(2)

mapping_values(train_features)


#test
mapping_values(test_features)


#scaling train
X = pd.DataFrame(StandardScaler().fit_transform(train_features))
X.head(2)


#scaling test
X_test = pd.DataFrame(StandardScaler().fit_transform(test_features))
X_test.head(2)


#target
y = train_.Listening_Time_minutes
y.head(2)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, stratify=y, random_state=31)


def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 1.0)
        }

    model = xgb.XGBRegressor(**params, n_jobs=6, random_state=31)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val, pred))
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)


best_params = study.best_params
xgb_model = xgb.XGBRegressor(**best_params, n_jobs=6, random_state=31)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_val)
score = np.sqrt(mean_squared_error(y_val, xgb_pred))
print(f"RMSE: {score:.4f}")


predict = xgb_model.predict(X_test)
predict[:5]


submission = sub_
submission = pd.DataFrame({'id': test_['id'], 'Listening_Time_minutes': predict})
submission.to_csv("submission.csv", index=False)
submission.head(3)

