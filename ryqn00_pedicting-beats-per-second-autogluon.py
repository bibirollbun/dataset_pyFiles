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
train.head()


test.head()


train.describe()


train.info()


test.describe()


test.info()


import matplotlib.pyplot as plt
import seaborn as sns

for col in train.columns:
    # Check if column is numeric
    if train[col].dtype in ['int64', 'float64']:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=train, x=col)
        plt.title(f'Boxplot of {col}')
        plt.show()


train.columns




train.corr()
sns.heatmap(train.corr())


!pip install autogluon.tabular


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autogluon.core import TabularDataset

X_train, X_val = train_test_split(train, test_size=0.2, random_state=42)

X_train['ScaledTrackDurationMs'] = (X_train['TrackDurationMs'] - X_train['TrackDurationMs'].mean()) / X_train['TrackDurationMs'].std()
X_train.drop('TrackDurationMs', axis=1)
X_val['ScaledTrackDurationMs'] = (X_val['TrackDurationMs'] - X_val['TrackDurationMs'].mean()) / X_val['TrackDurationMs'].std()
X_val.drop('TrackDurationMs', axis=1)


def bin_column(df, column, bins, bin_names=None):
    if bin_names is None:
        bin_names = [f'{b:.1f}_to_{b_next:.1f}' for b, b_next in zip(bins[:-1], bins[1:])]
    df[column + '_binned'] = pd.cut(df[column], bins=bins, labels=bin_names, include_lowest=True)
    return df

bins = [0.025, 0.1, 0.15, 0.2]
train = bin_column(train, 'VocalContent', bins)
test = bin_column(test, 'VocalContent', bins)

bins = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
train = bin_column(train, 'AcousticQuality', bins)
test = bin_column(test, 'AcousticQuality', bins)

bins = [0.001, 0.2, 0.4, 0.6, 0.8, 1.0]
train = bin_column(train, 'InstrumentalScore', bins)
test = bin_column(test, 'InstrumentalScore', bins)

bins = [0.05, 0.2, 0.4]
train = bin_column(train, 'LivePerformanceLikelihood', bins)
test = bin_column(test, 'LivePerformanceLikelihood', bins)


bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
X_train = bin_column(X_train, 'MoodScore', bins)
X_val = bin_column(X_val, 'MoodScore', bins)


numerical_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'ScaledTrackDurationMs', 'Energy']

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    # df['TrackDurationMin'] = df['TrackDurationMs'] / 60000 
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    
    df_new['acoustic_instrumental_ratio'] = df_new['AcousticQuality'] / (df_new['InstrumentalScore'] + 1e-6)
    df_new['RhythmEnergyRatio'] = df_new['RhythmScore'] / (df_new['Energy'] + 1e-8)
    df_new['VocalInstrumentalRatio'] = df_new['VocalContent'] / (df_new['InstrumentalScore'] + 1e-8)
    df['EnergyBin'] = pd.cut(df['Energy'], bins=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    df['RhythmBin'] = pd.cut(df['RhythmScore'], bins=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    
    return df_new

X_train = add_feature_cross_terms(X_train, numerical_features)
X_val = add_feature_cross_terms(X_val, numerical_features)


X_train = TabularDataset(X_train)
X_val = TabularDataset(X_val)


def add_feature_sq_terms(df, numerical_features):
    for feature in numerical_features:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))
    return df
    
# train = add_feature_sq_terms(train, numerical_features)
# test = add_feature_sq_terms(test, numerical_features)


from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='BeatsPerMinute', problem_type='regression', eval_metric='root_mean_squared_error', path='/kaggle/working/predictor').fit(X_train, presets='best_quality')


predictor.leaderboard(X_val)


predictor.feature_importance(X_val)


test['ScaledTrackDurationMs'] = (test['TrackDurationMs'] - test['TrackDurationMs'].mean()) / test['TrackDurationMs'].std()
test.drop('TrackDurationMs', axis=1)


bins = [0.025, 0.1, 0.15, 0.2]
test = bin_column(test, 'VocalContent', bins)

bins = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
test = bin_column(test, 'AcousticQuality', bins)

bins = [0.001, 0.2, 0.4, 0.6, 0.8, 1.0]
test = bin_column(test, 'InstrumentalScore', bins)

bins = [0.05, 0.2, 0.4]
test = bin_column(test, 'LivePerformanceLikelihood', bins)


bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
test = bin_column(test, 'MoodScore', bins)

test = add_feature_cross_terms(test, numerical_features)

test = TabularDataset(test)


y_pred = predictor.predict(test)


#!pip install autogluon.tabular[mitra]


submission = pd.DataFrame({'id' : test['id'], 'BeatsPerMinute' : y_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)

