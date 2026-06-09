









import pandas as pd

# Load game results
df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2024/MNCAATourneyCompactResults.csv')
print(df.columns.tolist())

# Aggregate win-side stats
wins = df.groupby('WTeamID').agg(
    games_played=('WTeamID', 'count'),
    points_scored=('WScore', 'sum'),
    points_allowed=('LScore', 'sum'),
).rename_axis('TeamID')

# Aggregate loss-side stats
losses = df.groupby('LTeamID').agg(
    games_played_l=('LTeamID', 'count'),
    points_scored_l=('LScore', 'sum'),
    points_allowed_l=('WScore', 'sum'),
).rename_axis('TeamID')

# Combine win/loss stats
team_stats = wins.add(losses, fill_value=0)
team_stats['avg_score'] = team_stats['points_scored'] / team_stats['games_played']
team_stats['avg_allowed'] = team_stats['points_allowed'] / team_stats['games_played']
team_stats['win_ratio'] = wins['games_played'] / team_stats['games_played']



def make_pairwise_samples(df, team_stats):
    samples = []
    labels = []
    group_sizes = []

    for _, row in df.iterrows():
        w, l = row['WTeamID'], row['LTeamID']

        w_feat = team_stats.loc[w][['avg_score', 'avg_allowed', 'win_ratio']].values
        l_feat = team_stats.loc[l][['avg_score', 'avg_allowed', 'win_ratio']].values

        samples.append(w_feat)
        samples.append(l_feat)
        labels.append(0)  # Winner ranks higher
        labels.append(1)  # Loser ranks lower
        group_sizes.append(2)  # One group of two rows

    return pd.DataFrame(samples), labels, group_sizes



from xgboost import XGBRanker

X_train, y_train, group = make_pairwise_samples(df, team_stats)

model = XGBRanker(
    objective='rank:pairwise',
    learning_rate=0.1,
    max_depth=4,
    n_estimators=100
)

model.fit(X_train, y_train, group=group)


def predict_match(teamA, teamB):
    a_stats = team_stats.loc[teamA][['avg_score', 'avg_allowed', 'win_ratio']]
    b_stats = team_stats.loc[teamB][['avg_score', 'avg_allowed', 'win_ratio']]
    feat = (a_stats - b_stats).values

    preds = model.predict(pd.DataFrame([feat]))
    return preds[0]  # Higher score means teamA is favored

# Example
teamA, teamB = 1101, 1208
score = predict_match(teamA, teamB)
print(f"{teamA} vs {teamB}: Score = {score} â†’ {'Team A is predicted to win' if score > 0 else 'Team B is predicted to win'}")




