import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class MarchMadnessPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.team_stats = {}

    def load_data(self, base_path):
        """
        Load all necessary data files
        """
        # Load basic team data
        self.men_teams = pd.read_csv(f"{base_path}/MTeams.csv")
        self.women_teams = pd.read_csv(f"{base_path}/WTeams.csv")

        # Load regular season and tournament results
        self.men_regular = pd.read_csv(f"{base_path}/MRegularSeasonDetailedResults.csv")
        self.women_regular = pd.read_csv(f"{base_path}/WRegularSeasonDetailedResults.csv")
        self.men_tourney = pd.read_csv(f"{base_path}/MNCAATourneyDetailedResults.csv")
        self.women_tourney = pd.read_csv(f"{base_path}/WNCAATourneyDetailedResults.csv")

        # Load team rankings for men's only (no rankings for women's teams)
        self.massey = pd.read_csv(f"{base_path}/MMasseyOrdinals.csv")

    def calculate_team_stats(self, games_df, season):
        """
        Calculate per-game average statistics for each team in a given season.
        """
        stats = {}

        for _, game in games_df[games_df['Season'] == season].iterrows():
            for team_id in [game['WTeamID'], game['LTeamID']]:
                if team_id not in stats:
                    stats[team_id] = {'games': 0, 'wins': 0, 'points_scored': 0, 'points_allowed': 0,
                                      'fg_pct': 0, 'fg3_pct': 0, 'ft_pct': 0, 'rebounds': 0, 'assists': 0,
                                      'steals': 0, 'blocks': 0, 'turnovers': 0}
            
            # Update winner stats
            w_stats = stats[game['WTeamID']]
            w_stats['games'] += 1
            w_stats['wins'] += 1
            w_stats['points_scored'] += game['WScore']
            w_stats['points_allowed'] += game['LScore']
            w_stats['fg_pct'] += game['WFGM'] / game['WFGA'] if game['WFGA'] > 0 else 0
            w_stats['fg3_pct'] += game['WFGM3'] / game['WFGA3'] if game['WFGA3'] > 0 else 0
            w_stats['ft_pct'] += game['WFTM'] / game['WFTA'] if game['WFTA'] > 0 else 0
            w_stats['rebounds'] += game['WOR'] + game['WDR']
            w_stats['assists'] += game['WAst']
            w_stats['steals'] += game['WStl']
            w_stats['blocks'] += game['WBlk']
            w_stats['turnovers'] += game['WTO']

            # Update loser stats
            l_stats = stats[game['LTeamID']]
            l_stats['games'] += 1
            l_stats['points_scored'] += game['LScore']
            l_stats['points_allowed'] += game['WScore']
            l_stats['fg_pct'] += game['LFGM'] / game['LFGA'] if game['LFGA'] > 0 else 0
            l_stats['fg3_pct'] += game['LFGM3'] / game['LFGA3'] if game['LFGA3'] > 0 else 0
            l_stats['ft_pct'] += game['LFTM'] / game['LFTA'] if game['LFTA'] > 0 else 0
            l_stats['rebounds'] += game['LOR'] + game['LDR']
            l_stats['assists'] += game['LAst']
            l_stats['steals'] += game['LStl']
            l_stats['blocks'] += game['LBlk']
            l_stats['turnovers'] += game['LTO']

        # Compute per-game averages
        for team_id, team_stats in stats.items():
            games = team_stats['games']
            if games > 0:
                for key in ['points_scored', 'points_allowed', 'fg_pct', 'fg3_pct', 'ft_pct',
                            'rebounds', 'assists', 'steals', 'blocks', 'turnovers']:
                    team_stats[key] /= games
                team_stats['win_pct'] = team_stats['wins'] / games

        return stats

    def prepare_training_data(self, start_season, end_season):
        """
        Prepare training data from historical games.
        """
        X, y = [], []

        for season in range(start_season, end_season + 1):
            # Compute team statistics
            men_stats = self.calculate_team_stats(self.men_regular, season)
            women_stats = self.calculate_team_stats(self.women_regular, season)
            self.team_stats[season] = {**men_stats, **women_stats}

            # Process historical tournament games
            for tourney_df in [self.men_tourney, self.women_tourney]:
                season_games = tourney_df[tourney_df['Season'] == season]
                for _, game in season_games.iterrows():
                    teamA, teamB = game['WTeamID'], game['LTeamID']

                    if teamA in self.team_stats[season] and teamB in self.team_stats[season]:
                        features = self.get_matchup_features(teamA, teamB, season)
                        X.append(features)
                        y.append(1)  # Winner is first team

                        # Reverse matchup
                        features_reversed = self.get_matchup_features(teamB, teamA, season)
                        X.append(features_reversed)
                        y.append(0)  # Winner is second team

        return np.array(X), np.array(y)

    def get_matchup_features(self, teamA, teamB, season):
        """
        Create feature vector for a matchup.
        """
        if season not in self.team_stats:
            return None
        
        stats = self.team_stats[season]
        if teamA not in stats or teamB not in stats:
            return None

        tA, tB = stats[teamA], stats[teamB]

        return [
            tA['win_pct'] - tB['win_pct'],
            tA['points_scored'] - tB['points_scored'],
            tA['points_allowed'] - tB['points_allowed'],
            tA['fg_pct'] - tB['fg_pct'],
            tA['fg3_pct'] - tB['fg3_pct'],
            tA['ft_pct'] - tB['ft_pct'],
            tA['rebounds'] - tB['rebounds'],
            tA['assists'] - tB['assists'],
            tA['steals'] - tB['steals'],
            tA['blocks'] - tB['blocks'],
            tA['turnovers'] - tB['turnovers']
        ]

    def train_model(self, start_season=2003, end_season=2024):
        """
        Train model on historical data.
        """
        X, y = self.prepare_training_data(start_season, end_season)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_2025(self):
        """
        Generate 2025 predictions for all possible matchups.
        """
        predictions = []
        all_teams = pd.concat([self.men_teams['TeamID'], self.women_teams['TeamID']]).unique()

        for teamA, teamB in combinations(sorted(all_teams), 2):
            features = self.get_matchup_features(teamA, teamB, 2024)  # Use most recent stats
            if features:
                pred = self.model.predict_proba(self.scaler.transform([features]))[0][1]
                predictions.append({'ID': f"2025_{teamA}_{teamB}", 'Pred': pred})

        return pd.DataFrame(predictions)

# Run everything
predictor = MarchMadnessPredictor()
predictor.load_data("/kaggle/input/march-machine-learning-mania-2025")
predictor.train_model()
submission = predictor.predict_2025()
print(submission.head())



from IPython.display import FileLink


# Save to CSV
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"✅ Submission file saved as {submission_path}, containing only 2025 matchups.")
FileLink("submission.csv")

