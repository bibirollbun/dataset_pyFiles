import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import mean_squared_error # Equivalent to Brier Score in this scenario

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
os.chdir('/kaggle/input/march-machine-learning-mania-2025')
print(os.listdir())


# Load Data
tourney_results = pd.concat([
    pd.read_csv("MNCAATourneyCompactResults.csv"),
    pd.read_csv("WNCAATourneyCompactResults.csv"),
], ignore_index=True) \
    .assign(League = lambda x: np.where(x['WTeamID'] < 3000, 1, 0))

regular_results = pd.concat([
    pd.read_csv("MRegularSeasonCompactResults.csv"),
    pd.read_csv("WRegularSeasonCompactResults.csv"),
], ignore_index=True) \
    .assign(League = lambda x: np.where(x['WTeamID'] < 3000, 1, 0))

teams_df = pd.concat([
    pd.read_csv('MTeams.csv'),
    pd.read_csv('WTeams.csv') \
        .assign(TeamName = lambda x: x['TeamName'] + " - Womens"),
], ignore_index=True) \
    .assign(League = lambda x: np.where(x['TeamID'] < 3000, 1, 0))

display(tourney_results.head())
display(regular_results.head())
display(teams_df)


# These utility functions are used to create a similar format to the final submission format.
def add_id(df):
    df['LowTeamID'] = df[['WTeamID', 'LTeamID']].min(axis=1) 
    df['HighTeamID'] = df[['WTeamID', 'LTeamID']].max(axis=1)
    df['ID'] = df['Season'].astype('str') + '_' + df['LowTeamID'].astype('str') + "_" + df['HighTeamID'].astype('str')
    df.drop(['LowTeamID', 'HighTeamID'], axis=1, inplace=True)
    return df

def id_to_teams(df):
    df['LowTeamID'] = df['ID'].str.split('_', expand=True)[1].astype('int')
    df['HighTeamID'] = df['ID'].str.split('_', expand=True)[2].astype('int')
    return df

def prep_tourney(df):
    df = df.copy()
    df = add_id(df)
    df = id_to_teams(df)
    df['Actual'] = np.where(df['WTeamID'] == df['LowTeamID'], 1, 0)
    return df[['ID', 'Season', 'LowTeamID', 'HighTeamID', 'Actual']]

prep_tourney(tourney_results).sample(20) # sampling to get a better look at the data


# Basic Elo
class Elo:
    def __init__(self, default_rating=1000, k=50, scale_factor=400): # these are my starting ratings
        self.default_rating = default_rating # this value
        self.k = k
        self.scale_factor = scale_factor
    
    def calc_prob(self, rating_a, rating_b):
        prob_a = 1 / (1 + 10 ** ((rating_b - rating_a) / self.scale_factor))
        prob_b = 1 - prob_a
        return prob_a, prob_b
    
    def update_ratings(self, rating_a, rating_b, outcome):
        prob_a, prob_b = self.calc_prob(rating_a, rating_b)
        rating_a_new = rating_a + self.k * (outcome - prob_a)
        rating_b_new = rating_b + self.k * ((1 - outcome) - prob_b)
        return rating_a_new, rating_b_new
    
    def calc_rating_history(self, df, teams_df):
        df = df.copy()
        # 'Date' is a psuedo date used to show progression within a season and year
        # without the long gaps. This is just one way of simplifying that.
        df['Date'] = df['DayNum'] + 154 * (df['Season'] - df['Season'].min()) 
        
        ids = pd.concat([df['WTeamID'], df['LTeamID']]).unique()
        team_ratings = dict.fromkeys(ids, self.default_rating)
        ratings_history = []
        
        for _, row in df.iterrows():
            team_a = row.WTeamID
            team_b = row.LTeamID
            date = row.Date
            
            a_new, b_new = self.update_ratings(team_ratings[team_a], team_ratings[team_b], outcome=1)
            team_ratings[team_a] = a_new
            team_ratings[team_b] = b_new
        
            ratings_history.append({'Season': row['Season'], 'DayNum': row['DayNum'], 'Date': date, 'TeamID': team_a, 'Rating': a_new})
            ratings_history.append({'Season': row['Season'], 'DayNum': row['DayNum'], 'Date': date, 'TeamID': team_b, 'Rating': b_new})
        
        r_history = pd.DataFrame(ratings_history)
        r_history = r_history.merge(teams_df[['TeamID', 'TeamName']], on='TeamID', how='left')
        r_history['GameType'] = np.where(r_history['DayNum'] <= 132, 'Regular', 'Tourney')
        return r_history
    
    def pre_tourney_ratings(self, r_history):
        return (r_history.loc[r_history['GameType'] == 'Regular']
                .sort_values(['Season', 'TeamID', 'Date'])
                .groupby(['Season', 'TeamID'], as_index=False)['Rating']
                .last())
    
    def add_elo(self, df, elo_df):
        df = df.merge(elo_df, how='left', left_on=['Season', 'LowTeamID'], right_on=['Season', 'TeamID'])
        df = df.merge(elo_df, how='left', left_on=['Season', 'HighTeamID'], right_on=['Season', 'TeamID'])
        df = df.drop(['TeamID_x', 'TeamID_y'], axis=1)
        df = df.rename(columns={'Rating_x': 'EloRating_x', 'Rating_y': 'EloRating_y'})
        df['EloPred'] = df.apply(lambda x: self.calc_prob(x['EloRating_x'], x['EloRating_y'])[0], axis=1)
        return df
    
    def eval_elo(self, games_df, teams_df, tourney_df):
        r_history = self.calc_rating_history(games_df, teams_df=teams_df)
        pre_tourney_df = self.pre_tourney_ratings(r_history)
        tourney_df = self.add_elo(tourney_df, pre_tourney_df)
        tourney_df['Brier'] = (tourney_df['Actual'] - tourney_df['EloPred']) ** 2
        return tourney_df
    
tourney_df = (
    prep_tourney(tourney_results)
)

e = Elo(k=32, scale_factor=400)
season_scores = e.eval_elo(games_df=regular_results, teams_df=teams_df, tourney_df=tourney_df).groupby('Season', as_index=False)['Brier'].mean()
print('Mean Brier: ', season_scores['Brier'].mean())
season_scores.query('Season > 2003')


