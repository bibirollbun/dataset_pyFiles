import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss

import seaborn as sns
import matplotlib.pyplot as plt


win_features = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
lose_features = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO','LStl', 'LBlk', 'LPF']

len(win_features), len(lose_features)


mncaa_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')
mncaa_detailed['mncaa'] = 1

print(mncaa_detailed.shape)

mncaa_detailed.head()


regular_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')
regular_detailed['mncaa'] = 0

print(regular_detailed.shape)

regular_detailed.head()


detailed = pd.concat([mncaa_detailed, regular_detailed], axis=0).reset_index(drop=True)

print(detailed.shape)
print(detailed[['Season', 'DayNum', 'WTeamID', 'LTeamID']].duplicated().sum())

detailed.head()


def get_team_features(detailed, feature_start, feature_end):

    features_df = detailed[detailed.Season.between(feature_start, feature_end)]
    
    win_df = features_df[['WTeamID'] + win_features]
    win_df.columns = [col[1:] for col in win_df.columns]
    
    lose_df = features_df[['LTeamID'] + lose_features]
    lose_df.columns = [col[1:] for col in lose_df.columns]
    
    features_df = pd.concat([win_df, lose_df], axis=0).reset_index(drop=True)
    features_df = features_df.groupby('TeamID').agg('mean')

    return features_df


features_df = get_team_features(detailed, 2013, 2015)

print(features_df.shape)

features_df.head()


def get_target(detailed, target_start, target_end):

    target_df = detailed[detailed.Season.between(target_start, target_end)]

    first_win = np.random.rand((target_df.shape[0])) < 0.5
    first_win = pd.Series(first_win, index=target_df.index)

    first_team = np.where(first_win, target_df['WTeamID'], target_df['LTeamID'])
    second_team = np.where(first_win, target_df['LTeamID'], target_df['WTeamID'])
    target = first_win.astype(int)
    
    target_df = pd.DataFrame([target_df.Season.values, first_team, second_team, target]).T
    target_df.columns = ['Season', 'first_team', 'second_team', 'target']

    return target_df


target_df = get_target(detailed, 2016, 2018)

print(target_df.shape)

target_df.head()


def join_features(target_df, features_df):

    df = (
        target_df
        .merge(features_df.add_suffix('_1'), how='left', left_on='first_team', right_on='TeamID')
        .merge(features_df.add_suffix('_2'), how='left', left_on='second_team', right_on='TeamID')
    )

    return df


df = join_features(target_df, features_df)

print(df.shape)

df.head()


df.columns


def get_cv_split(detailed, season, feature_period, target_period, valid_period):

    feature_start, feature_end = season, season + feature_period - 1
    target_start, target_end = feature_end + 1, feature_end + target_period
    valid_start, valid_end = target_end + 1, target_end + valid_period

    features_df = get_team_features(detailed, feature_start, feature_end)
    target_df = get_target(detailed, target_start, valid_end)

    df = join_features(target_df, features_df)
    to_drop = ['Season', 'first_team', 'second_team', 'target']
    
    train_df = df[df.Season.between(target_start, target_end)]
    X_train, y_train = train_df.drop(columns=to_drop), train_df.target

    valid_df = df[df.Season.between(valid_start, valid_end)]
    X_valid, y_valid = valid_df.drop(columns=to_drop), valid_df.target

    return X_train, y_train, X_valid, y_valid


scores = []
train_period, target_period, valid_period = 3, 2, 1

for season in range(2013, 2018 + 1):
    
    X_train, y_train, X_valid, y_valid = get_cv_split(
        detailed, 
        season,
        train_period, 
        target_period, 
        valid_period
    )

    model = LGBMClassifier(verbose=0)

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_valid)[:, 1]

    scores.append(brier_score_loss(y_valid, y_pred))


sns.lineplot(scores)


np.mean(scores), np.std(scores)

