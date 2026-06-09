import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm


class config:

    def __init__(self):
        self.initial_rating = 800
        self.alpha = 0.0
        self.reg_weight = 1.0
        self.tourney_weight = 2.0
        self.ncaa_weight = 1.3
        self.K = 120

CFG = config()


teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
games = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
games_secondary = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSecondaryTourneyCompactResults.csv')
games_ncaa = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
games_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')


def add_weight(games, weight_val=1.0):

    games['weight'] = weight_val
    return games

games = add_weight(games, weight_val=CFG.reg_weight)
games_secondary = add_weight(games_secondary, weight_val=CFG.tourney_weight)
games_ncaa = add_weight(games_ncaa, weight_val=CFG.ncaa_weight)


ordered_games = pd.concat([games, games_secondary, games_ncaa])
ordered_games = ordered_games.sort_values(['Season', 'DayNum'])
ordered_games


teams.head()


def elo_prob(rating1, rating2, initial_rating=CFG.initial_rating):
    return 1.0 / (1.0 + 10 ** ((rating1 - rating2) / initial_rating))

def update_elo(ratingW, ratingL, initial_rating, K=CFG.K, margin=0.0, weight=1.0):

    # Assume that the first team is the winning team,
    # Second team is the losing team
    # No ties in BBall

    K *= weight
    
    probW = elo_prob(ratingL, ratingW, initial_rating)
    probL = elo_prob(ratingW, ratingL, initial_rating)

    ratingW = ratingW + (K + margin) * (1 - probW)
    ratingL = ratingL + (K - margin) * (-probL)

    ratingW = ratingW if ratingW > 400 else 400
    ratingL = ratingL if ratingL > 400 else 400

    return ratingW, ratingL


def compute_elo(teams, games, start_elo=CFG.initial_rating, alpha = CFG.alpha):

    elo_history = {}
    loss = []
    
    # Loop through games and compute ELO for each of the teams
    for idx, game in tqdm(games.iterrows(), total=len(games)):
        w_team = game['WTeamID']
        l_team = game['LTeamID']

        weight = game['weight']

        # If teams have elo history, select the most recent rating
        # else, give them the starting value
        if w_team not in elo_history.keys():
            w_team_elo = start_elo
        else:
            w_team_elo = elo_history[w_team]

        if l_team not in elo_history.keys():
            l_team_elo = start_elo
        else:
            l_team_elo = elo_history[l_team]


        # Approximate formula for point spread to win probability
        def implied_spread(w_prob):

            # Inverse implied odds in units of 1/100 American betting units
            ml_odds = 1 / ( (1/w_prob) - 1)
            spread = np.sqrt(abs(ml_odds - 1) / 0.045)

            if ml_odds >= 1:
                return spread
            else:
                return -spread

        w_prob = elo_prob(l_team_elo, w_team_elo, start_elo)
        
        implied_margin = implied_spread(w_prob)
        #print(w_team_elo, l_team_elo, w_prob, implied_margin, game['WScore'] - game['LScore'])

        margin = alpha * (game['LScore'] - game['WScore'] + implied_margin)

        #print(margin)

        if game.Season == 2025: loss.append((1-w_prob)**2)

        if game.Season > 2020:
            w_team_elo, l_team_elo = update_elo(w_team_elo, l_team_elo, start_elo, margin=margin, weight=weight)
        else:
            w_team_elo, l_team_elo = update_elo(w_team_elo, l_team_elo, start_elo, margin=margin, weight=weight)
        
        elo_history[w_team] = w_team_elo
        elo_history[l_team] = l_team_elo

        #if idx == 100: break

    return elo_history, loss
        
elo_history, loss = compute_elo(teams, ordered_games)
print('Brier Score (2025): {}'.format(np.mean(loss)))


from collections import OrderedDict

def get_current_elo(elo_history):

    return OrderedDict({k: v for k, v in sorted(elo_history.items(), key=lambda item: item[1], reverse=True)})

current_elo = get_current_elo(elo_history)


rankings = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv')
rankings = rankings[(rankings['Season'] == 2025)].merge(teams, on='TeamID').sort_values(['RankingDayNum'], ascending=False)
current_day = rankings[rankings['RankingDayNum'] == rankings.iloc[0]['RankingDayNum']]


def plot_top_n(current_elo, teams, rankings, n=25):

    top_n_elo = []
    top_n_team = []
    
    for i, (team, elo) in enumerate(current_elo.items()):

        if i == n: break
        
        top_n_elo.append(elo)
        top_n_team.append(teams[teams.TeamID == team]['TeamName'].values[0])
        curr_rank = rankings[rankings['TeamID'] == team]['OrdinalRank'].median()

        top_n_team[i] += " ({:.00f})".format(curr_rank)

    plt.barh(top_n_team[::-1], top_n_elo[::-1])

    plt.show()

plot_top_n(current_elo, teams, rankings)


teams.TeamID


def generate_prob(teams, current_elo, initial_elo=CFG.initial_rating, mf='M'):

    print(mf)
    keys = []
    probs = []

    sample = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')
    
    for i, key in enumerate(tqdm(sample.ID, total=len(sample))):

        team1 = int(key.split('_')[1])
        team2 = int(key.split('_')[2])

        if mf == 'M' and (team1 > 2000 or team2 > 2000): continue
        elif mf == 'W' and (team1 < 2000 or team2 < 2000): continue

        key = f'2025_{team1}_{team2}'
        
        if team1 not in current_elo.keys():
            elo_team1 = initial_elo
        else:
            elo_team1 = current_elo[team1]

        if team2 not in current_elo.keys():
            elo_team2 = initial_elo
        else:
            elo_team2 = current_elo[team2]
        
        prob = elo_prob(elo_team2, elo_team1)

        keys.append(key)
        probs.append(prob)

    return pd.DataFrame({'ID': keys, 'Pred': probs})  


def submission(mf='M'):

    if mf == 'M':
        teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
        games = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
        games_secondary = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSecondaryTourneyCompactResults.csv')
        games_ncaa = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
        games_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')
    elif mf == 'W':
        print('Womens')
        teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeams.csv')
        games = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv')
        games_secondary = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WSecondaryTourneyCompactResults.csv')
        games_ncaa = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
        games_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv')

    games = add_weight(games, weight_val=CFG.reg_weight)
    games_secondary = add_weight(games_secondary, weight_val=CFG.tourney_weight)
    games_ncaa = add_weight(games_ncaa, weight_val=CFG.ncaa_weight)

    ordered_games = pd.concat([games, games_secondary, games_ncaa])
    ordered_games = ordered_games.sort_values(['Season', 'DayNum'])

    elo_history, loss = compute_elo(teams, ordered_games)
    print('Brier Score (2025): {}'.format(np.mean(loss)))

    current_elo = get_current_elo(elo_history)

    return generate_prob(teams, current_elo, mf=mf)

mens = submission(mf='M')
womens = submission(mf='W')


submission = pd.concat([mens, womens])

print(len(mens) + len(womens))
print(len(submission), len(pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')))
print(len(mens), len(womens))

submission.to_csv('submission.csv', index=False)


submission.tail()










