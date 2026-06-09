import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error # Equivalent to Brier Score in this scenario

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
os.chdir('/kaggle/input/march-machine-learning-mania-2025')
print(os.listdir())


# Load Data
tourney_results = pd.concat([
    pd.read_csv("MNCAATourneyCompactResults.csv"),
    pd.read_csv("WNCAATourneyCompactResults.csv"),
], ignore_index=True)

seeds = pd.concat([
    pd.read_csv("MNCAATourneySeeds.csv"),
    pd.read_csv("WNCAATourneySeeds.csv"),
], ignore_index=True) 
seeds['SeedNum'] = seeds['Seed'].str[1:3].astype(int)

display(tourney_results.head())
display(seeds.head())


# I use these functions here to create a format that is equivalent to our final submission
def add_id(df):
    df['LowTeamID'] = df[['WTeamID', 'LTeamID']].min(axis=1) 
    df['HighTeamID'] = df[['WTeamID', 'LTeamID']].max(axis=1)
    df['ID'] = df['Season'].astype('str') + '_' + df['LowTeamID'].astype('str') + "_" + df['HighTeamID'].astype('str') # This notebook is sponsored by Raid Shadow Lengeds. Please use code... jk ;)
    df.drop(['LowTeamID', 'HighTeamID'], axis=1, inplace=True)
    return df

def id_to_teams(df):
    df['LowTeamID'] = df['ID'].str.split('_', expand=True)[1].astype('int')
    df['HighTeamID'] = df['ID'].str.split('_', expand=True)[2].astype('int')
    return df

def prep_tourney(df):
    df = add_id(df)
    df = id_to_teams(df)
    df['Actual'] = np.where(df['WTeamID'] == df['LowTeamID'], 1, 0)
    return df[['ID', 'Season', 'LowTeamID', 'HighTeamID', 'Actual']]

prep_tourney(tourney_results).sample(20) # sampling to get an idea of full dataframe


# Here we add the seeds and calculate the difference between the two teams
tourney_df = (
    prep_tourney(tourney_results)
    .merge(seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'LowTeamID'], right_on=['Season', 'TeamID'])
    .merge(seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'HighTeamID'], right_on=['Season', 'TeamID'])
    .drop(['TeamID_x', 'TeamID_y'], axis=1)
    .assign(SeedDiff = lambda x: x['SeedNum_x'] - x['SeedNum_y'],
            League = lambda x: np.where(x['LowTeamID'] < 3000, 1, 0)) # League Refers to Men=1, Women=0
)
tourney_df


split_year = 2020
train = tourney_df[tourney_df['Season'] < split_year].copy()
test = tourney_df[tourney_df['Season'] >= split_year].copy()

X_train = train[["SeedDiff"]]
y_train = train["Actual"]

X_test = test[["SeedDiff"]]
y_test = test["Actual"]

X_train


# historical probability
seed_prob = train.groupby('SeedDiff', as_index=False)['Actual'].mean()
seed_prob.plot(x='SeedDiff', y='Actual')

seed_dict = dict(zip(seed_prob['SeedDiff'], seed_prob['Actual']))
seed_dict

y_pred = test['SeedDiff'].map(seed_dict)
print('Brier Score (MSE):', mean_squared_error(y_test, y_pred))


# Logistic Regression of Seed Difference
logit_model = LogisticRegression()
logit_model.fit(X_train, y_train)
y_pred = logit_model.predict_proba(X_test)[:, 1]

# mse/brier
mse = mean_squared_error(y_test, y_pred)
print(f"Brier Score (MSE): {mse}")

# Plot logistic curve
seed_df = pd.DataFrame({'SeedDiff': pd.Series(range(-15, 16))})
# seed_df

seed_prob.plot(x='SeedDiff', y='Actual')

logistic_probs = logit_model.predict_proba(seed_df)[:, 1]
plt.plot(seed_df['SeedDiff'], logistic_probs)
plt.xlabel("Seed Difference")
plt.ylabel("Win Probability")
plt.title("Logistic Regression Win Probability by Seed Difference")
plt.show()


# I'm just using the last four years but it would be wise to expand this range
years = [2021, 2022, 2023, 2024]

mse_scores = []
for split_year in years: 
    train = tourney_df[tourney_df['Season'] < split_year].copy()
    test = tourney_df[tourney_df['Season'] == split_year].copy()
    
    X_train = train[["SeedDiff"]]
    y_train = train["Actual"]
    
    X_test = test[["SeedDiff"]]
    y_test = test["Actual"]

    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict_proba(X_test)[:, 1]
    
    # mse/brier
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

print('Logistic Regression Scores:')
print(pd.DataFrame({'Year': years, 'MSE': mse_scores}))


# Lets try with mapping historical probabilities only - for science!
years = [2021, 2022, 2023, 2024]

mse_scores = []
for split_year in years: 
    train = tourney_df[tourney_df['Season'] < split_year].copy()
    test = tourney_df[tourney_df['Season'] == split_year].copy()
    
    X_train = train[["SeedDiff"]]
    y_train = train["Actual"]
    
    X_test = test[["SeedDiff"]]
    y_test = test["Actual"]

    # historical probability
    seed_prob = train.groupby('SeedDiff', as_index=False)['Actual'].mean()
    seed_dict = dict(zip(seed_prob['SeedDiff'], seed_prob['Actual']))
    seed_dict

    y_pred = test['SeedDiff'].map(seed_dict)
    
    # mse/brier
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

print('Historical Probabily Scores:')
print(pd.DataFrame({'Year': years, 'MSE': mse_scores}))

