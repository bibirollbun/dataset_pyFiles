import numpy as np

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataLoader:
    def __init__(self,
                 data_dir: str = "/kaggle/input/march-machine-learning-mania-2025",
                 target_season: int = 2024):
        self.data_dir = data_dir
        self.target_season = target_season

        # Paths
        self.path_men_rs_detailed = f"{self.data_dir}/MRegularSeasonDetailedResults.csv"
        self.path_men_tourney_compact = f"{self.data_dir}/MNCAATourneyCompactResults.csv"
        self.path_men_teams = f"{self.data_dir}/MTeams.csv"
        self.path_men_massey = f"{self.data_dir}/MMasseyOrdinals.csv"

        self.path_women_rs_detailed = f"{self.data_dir}/WRegularSeasonDetailedResults.csv"
        self.path_women_teams = f"{self.data_dir}/WTeams.csv"
        # (Add WNCAATourneyCompactResults.csv or WNCAATourneySeeds.csv if needed)

        # DataFrames
        self.df_men_rs = None
        self.df_men_tourney = None
        self.df_men_teams = None
        self.df_men_massey = None

        self.df_women_rs = None
        self.df_women_teams = None

    def load_data(self):
        # Men
        self.df_men_rs = pd.read_csv(self.path_men_rs_detailed)
        self.df_men_tourney = pd.read_csv(self.path_men_tourney_compact)
        self.df_men_teams = pd.read_csv(self.path_men_teams)
        self.df_men_massey = pd.read_csv(self.path_men_massey)

        # Women
        self.df_women_rs = pd.read_csv(self.path_women_rs_detailed)
        self.df_women_teams = pd.read_csv(self.path_women_teams)

    def filter_season(self, df: pd.DataFrame, season_col: str = "Season") -> pd.DataFrame:
        return df[df[season_col] == self.target_season].copy()


###############################################################################
#                  2. FEATURE ENGINEERING (Margin, SoS, etc.)
###############################################################################
class MarginFeatureEngineer:
    """
    Computes margin-based features, such as average scoring margin and
    an adjusted margin that accounts for Strength of Schedule (SoS).
    """

    def __init__(self,
                 df_rs_detailed_filtered: pd.DataFrame,
                 df_teams: pd.DataFrame):
        """
        :param df_rs_detailed_filtered: Regular season *filtered* data
               for the target season only (MEN or WOMEN).
        :param df_teams: The MTeams or WTeams dataset (for merging team names).
        """
        self.df_rs_detailed = df_rs_detailed_filtered
        self.df_teams = df_teams

    def build_teamgames_df(self) -> pd.DataFrame:
        """
        Builds a row for each (Team, Game) from the perspective of that team.
        """
        # Winner perspective
        win_cols = {
            "WTeamID": "TeamID",
            "WScore": "PointsFor",
            "LTeamID": "OppTeamID",
            "LScore": "PointsAgainst"
        }
        df_win = self.df_rs_detailed.rename(columns=win_cols).copy()
        df_win["Result"] = 1

        # Loser perspective
        loss_cols = {
            "LTeamID": "TeamID",
            "LScore": "PointsFor",
            "WTeamID": "OppTeamID",
            "WScore": "PointsAgainst"
        }
        df_loss = self.df_rs_detailed.rename(columns=loss_cols).copy()
        df_loss["Result"] = 0

        # Merge common columns
        common_cols = [
            "Season", "DayNum", "NumOT", "WLoc",
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR",
            "WAst", "WTO", "WStl", "WBlk", "WPF",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR",
            "LAst", "LTO", "LStl", "LBlk", "LPF"
        ]
        for c in common_cols:
            df_win[c] = self.df_rs_detailed[c]
            df_loss[c] = self.df_rs_detailed[c]

        df_teamgames = pd.concat([df_win, df_loss], ignore_index=True)
        df_teamgames["ScoreMargin"] = df_teamgames["PointsFor"] - df_teamgames["PointsAgainst"]
        return df_teamgames

    def compute_margin(self, df_teamgames: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the average margin (RawMargin) for each team, merges with team info.
        """
        df_margin = (
            df_teamgames
            .groupby("TeamID")["ScoreMargin"]
            .mean()
            .reset_index()
            .rename(columns={"ScoreMargin": "RawMargin"})
        )

        # Merge on team info (TeamID, TeamName, etc.)
        df_margin = df_margin.merge(
            self.df_teams[["TeamID", "TeamName"]],
            on="TeamID",
            how="left"
        )
        return df_margin

    def compute_strength_of_schedule(self,
                                     df_teamgames: pd.DataFrame,
                                     df_margin: pd.DataFrame) -> pd.DataFrame:
        """
        Computes a simple SoS: average of opponents' RawMargin.
        """
        raw_margin_map = dict(zip(df_margin["TeamID"], df_margin["RawMargin"]))

        # Table of (TeamID, OppTeamID)
        df_opponents = df_teamgames[["TeamID", "OppTeamID"]].drop_duplicates()

        def get_opponent_average_margin(team_id):
            opps = df_opponents[df_opponents["TeamID"] == team_id]["OppTeamID"].values
            if len(opps) == 0:
                return 0.0
            margins = [raw_margin_map.get(o, 0.0) for o in opps]
            return np.mean(margins)

        team_ids = df_margin["TeamID"].unique()
        sos_list = []
        for t in team_ids:
            sos_val = get_opponent_average_margin(t)
            sos_list.append({"TeamID": t, "SoS": sos_val})

        df_sos = pd.DataFrame(sos_list)
        return df_sos

    def compute_adjusted_margin(self,
                                df_margin: pd.DataFrame,
                                df_sos: pd.DataFrame,
                                alpha: float = 0.5) -> pd.DataFrame:
        """
        Combines RawMargin and SoS into an 'AdjMargin' metric.
        """
        df_combined = df_margin.merge(df_sos, on="TeamID", how="left")
        df_combined["AdjMargin"] = df_combined["RawMargin"] + alpha * df_combined["SoS"]

        # Sort & rank
        df_combined["RankByAdjMargin"] = df_combined["AdjMargin"].rank(method="dense", ascending=False)
        df_combined = df_combined.sort_values("AdjMargin", ascending=False)
        return df_combined


##############################################################################
#                             3. PREDICTION
###############################################################################
class AdvancedFeatureEngineer:
    """
    Builds advanced basketball metrics: Offensive/Defensive Efficiency,
    Four Factors, etc.
    """

    def __init__(self, df_rs_detailed_filtered, df_teams):
        self.df_rs = df_rs_detailed_filtered
        self.df_teams = df_teams

    def build_teamgames_df(self) -> pd.DataFrame:
        """
        Similar to build_teamgames_df from MarginFeatureEngineer,
        but we'll compute advanced stats as well.
        """
        # 1) Winner perspective
        win_cols = {
            "WTeamID": "TeamID",
            "WScore": "PointsFor",
            "LTeamID": "OppTeamID",
            "LScore": "PointsAgainst",
            "WFGM": "FGM",
            "WFGA": "FGA",
            "WFGM3": "FGM3",
            "WFGA3": "FGA3",
            "WFTM": "FTM",
            "WFTA": "FTA",
            "WOR": "OR",
            "WDR": "DR",
            "WAst": "Ast",
            "WTO": "TO",
            "WStl": "Stl",
            "WBlk": "Blk",
            "WPF": "PF"
        }
        df_win = self.df_rs.rename(columns=win_cols).copy()
        df_win["Result"] = 1

        # 2) Loser perspective
        loss_cols = {
            "LTeamID": "TeamID",
            "LScore": "PointsFor",
            "WTeamID": "OppTeamID",
            "WScore": "PointsAgainst",
            "LFGM": "FGM",
            "LFGA": "FGA",
            "LFGM3": "FGM3",
            "LFGA3": "FGA3",
            "LFTM": "FTM",
            "LFTA": "FTA",
            "LOR": "OR",
            "LDR": "DR",
            "LAst": "Ast",
            "LTO": "TO",
            "LStl": "Stl",
            "LBlk": "Blk",
            "LPF": "PF"
        }
        df_loss = self.df_rs.rename(columns=loss_cols).copy()
        df_loss["Result"] = 0

        # Common columns
        common_cols = ["Season", "DayNum", "NumOT", "WLoc"]
        for c in common_cols:
            df_win[c] = self.df_rs[c]
            df_loss[c] = self.df_rs[c]

        # Combine
        df_teamgames = pd.concat([df_win, df_loss], ignore_index=True)

        # Basic margin
        df_teamgames["ScoreMargin"] = df_teamgames["PointsFor"] - df_teamgames["PointsAgainst"]

        # 3) Compute possessions
        # possessions = FGA - OR + TO + 0.475 * FTA
        df_teamgames["Possessions"] = (
            df_teamgames["FGA"]
            - df_teamgames["OR"]
            + df_teamgames["TO"]
            + 0.475 * df_teamgames["FTA"]
        )

        # 4) Offensive Efficiency
        # OffEff = 100 * PointsFor / possessions
        df_teamgames["OffEff"] = 100.0 * df_teamgames["PointsFor"] / df_teamgames["Possessions"].clip(lower=1e-9)

        # 5) Defensive Efficiency
        # DefEff = 100 * PointsAgainst / possessions
        df_teamgames["DefEff"] = 100.0 * df_teamgames["PointsAgainst"] / df_teamgames["Possessions"].clip(lower=1e-9)

        # 6) eFG% = (FGM + 0.5 * FGM3) / FGA
        df_teamgames["eFG"] = (df_teamgames["FGM"] + 0.5 * df_teamgames["FGM3"]) / df_teamgames["FGA"].clip(lower=1e-9)

        # 7) TOV% = TO / possessions
        df_teamgames["TOV%"] = df_teamgames["TO"] / df_teamgames["Possessions"].clip(lower=1e-9)

        # 8) OR% = OR / (OR + OppDR) -> we need OppDR.
        #    This is tricky at the single-team perspective.
        #    One approach is to also store "OppDR" in the same row.
        #    But let's just do a "team average" approach:
        #    We'll omit exact OR% for now or do an approximation.
        # For a more accurate approach, weâ€™d merge WDR / LDR from the â€œopposite perspective.â€�

        # 9) FTRate = FTA / FGA
        df_teamgames["FTRate"] = df_teamgames["FTA"] / df_teamgames["FGA"].clip(lower=1e-9)

        return df_teamgames

    def aggregate_team_stats(self, df_teamgames: pd.DataFrame) -> pd.DataFrame:
        """
        Group by TeamID, compute season averages (or totals) for advanced stats.
        """
        agg_dict = {
            "ScoreMargin": "mean",
            "OffEff": "mean",
            "DefEff": "mean",
            "eFG": "mean",
            "TOV%": "mean",
            "FTRate": "mean",
            "Possessions": "mean"  # or sum, if you prefer
        }

        df_agg = df_teamgames.groupby("TeamID").agg(agg_dict).reset_index()

        # Merge in TeamName
        df_agg = df_agg.merge(
            self.df_teams[["TeamID", "TeamName"]],
            on="TeamID",
            how="left"
        )
        return df_agg


class MarginBasedPredictor:
    """
    A simple predictor that uses the difference in average margin (or adjusted margin)
    to compute the probability of TeamA beating TeamB:

    P(A wins) = 0.5 + scale * (MarginA - MarginB), clamped to [0,1].
    """

    def __init__(self, margin_map: dict, scale: float = 0.02):
        """
        :param margin_map: dict of {TeamID: margin_value}
        :param scale: scale factor in the linear formula.
        """
        self.margin_map = margin_map
        self.scale = scale

    def predict_probability(self, teamA: int, teamB: int) -> float:
        """Return P(teamA beats teamB)."""
        marginA = self.margin_map.get(teamA, 0.0)
        marginB = self.margin_map.get(teamB, 0.0)
        diff = marginA - marginB
        raw_prob = 0.5 + self.scale * diff
        # clamp
        prob = max(0, min(1, raw_prob))
        return prob

    def predict_tournament(self, df_tourney: pd.DataFrame) -> pd.DataFrame:
        """
        For each real 2024 tournament game in df_tourney, compute probability
        that the lower-ID team beats the higher-ID team (to measure Brier score).
        """
        predictions = []
        for _, row in df_tourney.iterrows():
            wteam = row["WTeamID"]
            lteam = row["LTeamID"]

            # Identify lower/higher for standard "ID" format
            lower_id = min(wteam, lteam)
            higher_id = max(wteam, lteam)

            # actual outcome
            actual_lower_wins = 1.0 if (lower_id == wteam) else 0.0

            # predicted
            pred_prob_lower = self.predict_probability(lower_id, higher_id)

            predictions.append({
                "Season": row["Season"],
                "DayNum": row["DayNum"],
                "LowerTeamID": lower_id,
                "HigherTeamID": higher_id,
                "PredProbLowerWins": pred_prob_lower,
                "ActualLowerWins": actual_lower_wins
            })

        df_preds = pd.DataFrame(predictions)
        return df_preds




class EloRater:
    """
    A simple NCAA Elo rating system with improvements:
      1. Margin-of-victory multiplier,
      2. Refined home-court advantage handling,
      3. Dynamic (decaying) K-factor.
    """

    def __init__(self,
                 k_factor: float = 20.0,
                 base_rating: float = 1500.0,
                 home_advantage: float = 50.0,
                 margin_of_victory_mult: bool = False):
        """
        :param k_factor: Base K-factor.
        :param base_rating: Starting rating for each team.
        :param home_advantage: Advantage in rating points for the home team.
        :param margin_of_victory_mult: If True, incorporate margin-of-victory factor.
        """
        self.k_factor = k_factor
        self.base_rating = base_rating
        self.home_advantage = home_advantage
        self.margin_of_victory_mult = margin_of_victory_mult

        # Dictionary to store current rating for each team
        self.ratings = {}
        # Dictionary to track number of games played per team for dynamic K
        self.games_played = {}

    def get_rating(self, team_id: int) -> float:
        if team_id not in self.ratings:
            self.ratings[team_id] = self.base_rating
            self.games_played[team_id] = 0
        return self.ratings[team_id]

    def set_rating(self, team_id: int, new_rating: float):
        self.ratings[team_id] = new_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Compute expected score for team A using the logistic formula."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_elo(self, team_a: int, team_b: int, score_a: float, score_b: float, wloc: str = "N",
                   point_diff: float = 0.0):
        """
        Update Elo ratings with improvements.

        :param team_a: ID for team A.
        :param team_b: ID for team B.
        :param score_a: 1 if team A wins, 0 if loses.
        :param score_b: 1 if team B wins, 0 if loses.
        :param wloc: 'H', 'A', or 'N'. If 'H', team A is at home; if 'A', team B is home.
        :param point_diff: Absolute margin of victory.
        """
        # 1) Get current ratings
        Ra = self.get_rating(team_a)
        Rb = self.get_rating(team_b)

        # 2) Apply refined home-court advantage: adjust both teams.
        if wloc == "H":
            # team_a is home
            adj_a = Ra + self.home_advantage
            adj_b = Rb - self.home_advantage
        elif wloc == "A":
            # team_b is home
            adj_a = Ra - self.home_advantage
            adj_b = Rb + self.home_advantage
        else:
            adj_a = Ra
            adj_b = Rb

        # 3) Compute expected scores using adjusted ratings
        Ea = self.expected_score(adj_a, adj_b)
        Eb = 1.0 - Ea

        # 4) Compute dynamic K-factor based on average games played
        avg_games = (self.games_played.get(team_a, 0) + self.games_played.get(team_b, 0)) / 2.0
        dynamic_k = self.k_factor / (1.0 + 0.03 * avg_games)

        # 5) Compute margin-of-victory factor if enabled
        factor = 1.0
        if self.margin_of_victory_mult:
            if point_diff == 0:
                factor = 1.0
            else:
                # Scale margin factor: larger wins yield higher rating changes
                diff = abs(Ra - Rb)
                factor = math.log(abs(point_diff) + 1) * (2.2 / ((diff * 0.001) + 2.2))

        # 6) Update ratings using dynamic K and margin factor
        new_Ra = Ra + dynamic_k * factor * (score_a - Ea)
        new_Rb = Rb + dynamic_k * factor * (score_b - Eb)

        self.set_rating(team_a, new_Ra)
        self.set_rating(team_b, new_Rb)

        # 7) Increment games played count
        self.games_played[team_a] = self.games_played.get(team_a, 0) + 1
        self.games_played[team_b] = self.games_played.get(team_b, 0) + 1

    def rate_all_games(self, df_games: pd.DataFrame):
        """
        Process a DataFrame of games (with columns "DayNum", "WTeamID", "LTeamID",
        "WLoc", and optionally "point_diff") in chronological order.
        """
        df_sorted = df_games.sort_values("DayNum")
        for _, row in df_sorted.iterrows():
            wteam = row["WTeamID"]
            lteam = row["LTeamID"]
            wloc = row.get("WLoc", "N")  # "H", "A", or "N"
            # Assume point difference is stored under "point_diff"
            point_diff = row.get("point_diff", 0.0)
            self.update_elo(
                team_a=wteam,
                team_b=lteam,
                score_a=1.0,
                score_b=0.0,
                wloc=wloc,
                point_diff=point_diff
            )

def compute_elo_feature(df_rs_detailed: pd.DataFrame,
                        teams_df: pd.DataFrame,
                        season: int) -> pd.DataFrame:
    """
    Compute final Elo ratings for all teams in 'season' from the detailed results.
    Return a DataFrame with columns: [TeamID, TeamName, Elo].
    """

    # Filter to that season
    df_season = df_rs_detailed[df_rs_detailed["Season"] == season].copy()

    # Initialize Elo
    elo_rater = EloRater(k_factor=20.0, base_rating=1500.0, home_advantage=50.0)
    # Rate the regular season
    elo_rater.rate_all_games(df_season)

    # Now we have final Elo for each team that played
    results = []
    for team_id in df_season["WTeamID"].unique():
        results.append((team_id, elo_rater.get_rating(team_id)))
    for team_id in df_season["LTeamID"].unique():
        results.append((team_id, elo_rater.get_rating(team_id)))

    df_elo = pd.DataFrame(list(set(results)), columns=["TeamID", "Elo"])

    # Merge on team name
    df_elo = df_elo.merge(teams_df[["TeamID", "TeamName"]], on="TeamID", how="left")

    return df_elo



class Evaluator:
    """
    Evaluates predictions and compares with external ranking systems (e.g., KenPom).
    """

    @staticmethod
    def brier_score(df_preds: pd.DataFrame,
                    pred_col: str = "PredProbLowerWins",
                    actual_col: str = "ActualLowerWins") -> float:
        """Compute the Brier score for a set of predictions."""
        df_preds["squared_error"] = (df_preds[pred_col] - df_preds[actual_col]) ** 2
        return df_preds["squared_error"].mean()

    @staticmethod
    def correlation_with_massey(df_our: pd.DataFrame,
                                df_massey: pd.DataFrame,
                                metric_col: str = "AdjMargin",
                                ord_col: str = "OrdinalRank") -> dict:
        """
        Merge your feature data (e.g. AdjMargin) with a Massey Ordinals DataFrame,
        then compute Pearson & Spearman correlation.

        Assumes:
          df_our has ['TeamID', metric_col]
          df_massey has ['TeamID', ord_col]
        """
        df_compare = df_our.merge(
            df_massey[["TeamID", ord_col]],
            on="TeamID",
            how="inner"
        )

        pearson = df_compare[metric_col].corr(df_compare[ord_col], method="pearson")
        spearman = df_compare[metric_col].corr(df_compare[ord_col], method="spearman")

        return {
            "pearson": pearson,
            "spearman": spearman
        }

    @staticmethod
    def correlation_pearson_spearman(df, feature_col, rank_col):
        pearson = df[feature_col].corr(df[rank_col], method="pearson")
        spearman = df[feature_col].corr(df[rank_col], method="spearman")
        return pearson, spearman

class Visualizer:
    """
    Provides various plots to help compare features and model outputs.
    """

    @staticmethod
    def scatter_raw_sos(df: pd.DataFrame, x_col: str = "RawMargin", y_col: str = "SoS", hue_col: str = "AdjMargin"):
        plt.figure(figsize=(8, 6))
        scatter_plot = sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=hue_col,
            palette="vlag"
        )
        scatter_plot.set_title("Raw Margin vs. Strength of Schedule")
        plt.show()

    @staticmethod
    def distribution_comparison(df: pd.DataFrame, col1: str = "RawMargin", col2: str = "AdjMargin"):
        plt.figure(figsize=(10, 4))
        # Subplot 1
        plt.subplot(1, 2, 1)
        sns.histplot(df[col1], kde=True, color="steelblue")
        plt.title(f"Distribution of {col1}")

        # Subplot 2
        plt.subplot(1, 2, 2)
        sns.histplot(df[col2], kde=True, color="darkorange")
        plt.title(f"Distribution of {col2}")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def bar_top_n(df: pd.DataFrame, value_col: str = "AdjMargin", label_col: str = "TeamName", n: int = 15):
        df_top = df.sort_values(value_col, ascending=False).head(n)
        plt.figure(figsize=(8, 6))
        bar_chart = sns.barplot(
            data=df_top,
            y=label_col,
            x=value_col,
            palette="vlag"
        )
        bar_chart.set_title(f"Top {n} Teams by {value_col}")
        bar_chart.set_xlabel(value_col)
        bar_chart.set_ylabel(label_col)
        plt.gca().invert_yaxis()  # best team at top
        plt.show()

    @staticmethod
    def scatter_adjmargin_vs_ordinal(df_compare: pd.DataFrame,
                                     adj_col: str = "AdjMargin",
                                     ord_col: str = "OrdinalRank"):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_compare,
            x=adj_col,
            y=ord_col
        )
        plt.title("AdjMargin vs. Public OrdinalRank (lower = better)")
        plt.xlabel("Our Adjusted Margin")
        plt.ylabel("Ordinal Rank")
        plt.grid(True)
        plt.show()


def calculate_all_matchup_win_probs(df_features: pd.DataFrame,
                                    elo_col: str = "Elo",
                                    offeff_col: str = "OffEff",
                                    season: int = 2024) -> pd.DataFrame:
    """
    Enumerate every possible matchup among the teams in df_features,
    then calculate the probability that Team A beats Team B.

    This example uses a very simple logistic formula combining Elo difference
    and OffEff difference:

        p(A beats B) = 1 / (1 + exp( - [ beta_elo*(EloA - EloB) + beta_off*(OffEffA - OffEffB) ] ))

    You can customize the formula (e.g., add DefEff, TOV%, etc., or
    just do the standard Elo approach: p=1/(1+10^((EloB-EloA)/400)) ).

    :param df_features: DataFrame with at least columns:
                          ["TeamID", "Elo", "OffEff", ...]
    :param elo_col: Name of the Elo column to use
    :param offeff_col: Name of the Offensive Efficiency column to use
    :param season: Used for constructing an ID if you want a "Season_TeamA_TeamB" format
    :return: DataFrame with columns: [ID, TeamA, TeamB, ProbAWin]
             where ID = f"{season}_{TeamA}_{TeamB}"
    """

    # 1) Verify columns exist
    for col in [elo_col, offeff_col, "TeamID"]:
        if col not in df_features.columns:
            raise ValueError(f"Required column '{col}' not found in df_features.")

    # 2) Extract unique teams
    team_ids = sorted(df_features["TeamID"].unique())
    n_teams = len(team_ids)
    print(f"Calculating matchup probabilities for {n_teams} teams => ~{n_teams*(n_teams-1)/2:.0f} pairs...")

    # 3) Build a quick lookup dict for each feature
    #    e.g. elo_map[team_id], offeff_map[team_id]
    elo_map = dict(zip(df_features["TeamID"], df_features[elo_col]))
    off_map = dict(zip(df_features["TeamID"], df_features[offeff_col]))

    # We'll define some coefficients for a naive logistic formula
    beta_elo = 0.01   # example coefficient
    beta_off = 0.05   # example coefficient

    def matchup_probability(tA, tB):
        # Option A: standard Elo logistic
        #   p = 1 / (1 + 10^((EloB - EloA)/400))
        #
        # Option B: a custom formula with OffEff difference
        # For demonstration:
        elo_diff = elo_map[tA] - elo_map[tB]
        off_diff = off_map[tA] - off_map[tB]
        linear_score = beta_elo * elo_diff + beta_off * off_diff
        p = 1.0 / (1.0 + np.exp(-linear_score))
        return p

    # 4) Generate every pair (TeamA, TeamB), TeamA < TeamB
    rows = []
    for i in range(n_teams):
        for j in range(i+1, n_teams):
            tA = team_ids[i]
            tB = team_ids[j]

            pA = matchup_probability(tA, tB)
            # p(A beats B)
            # If you want also p(B beats A), you'd store that, or just do a single row.

            rows.append({
                "ID": f"{season}_{tA}_{tB}",
                "TeamA": tA,
                "TeamB": tB,
                "ProbAWin": pA
            })

    df_out = pd.DataFrame(rows, columns=["ID", "TeamA", "TeamB", "ProbAWin"])
    return df_out


class TournamentResultsCalculator:
    """
    Calculates how far each team advanced in the NCAA tournament
    (men's or women's). Provides a round # for each team, e.g.:
      0 = play-in,
      1 = Round of 64,
      2 = Round of 32,
      3 = Sweet 16,
      4 = Elite 8,
      5 = Final 4,
      6 = Championship game,
      7 = Champion (optional).
    """

    # Approximate day ranges for the men's tournament:
    # - 134,135: play-in -> round = 0
    # - 136,137: Round of 64 -> round = 1
    # - 138,139: Round of 32 -> round = 2
    # - 143,144: Sweet 16 -> round = 3
    # - 145,146: Elite 8 -> round = 4
    # - 152: Final 4 -> round = 5
    # - 154: Championship -> round = 6
    # If you want to mark the champion as round 7, we can do that, too.

    @staticmethod
    def daynum_to_round(daynum: int) -> int:
        """Return an integer round index based on daynum (men's approximate)."""
        if daynum in [134, 135]:
            return 0
        elif daynum in [136, 137]:
            return 1
        elif daynum in [138, 139]:
            return 2
        elif daynum in [143, 144]:
            return 3
        elif daynum in [145, 146]:
            return 4
        elif daynum == 152:
            return 5
        elif daynum == 154:
            return 6
        else:
            # If it doesn't match any known day range, assume 0 or skip
            return 0

    def compute_tournament_outcomes(self, df_tourney: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame of tournament games (Compact Results),
        compute how far each team advanced.

        Returns a DataFrame with columns: [TeamID, RoundReached].
        """
        # We'll store the maximum round each team achieved as 'RoundReached'.
        team_rounds = {}  # team_id -> max round reached

        # Sort games by DayNum ascending
        df_sorted = df_tourney.sort_values("DayNum")

        for _, row in df_sorted.iterrows():
            day = row["DayNum"]
            round_now = self.daynum_to_round(day)

            # Winner/Loser
            wteam = row["WTeamID"]
            lteam = row["LTeamID"]

            # The losing team is eliminated in the current round (round_now).
            # The winning team *advances* to the next round (round_now+1).
            # We'll track the maximum they'd have reached so far.

            loser_prev = team_rounds.get(lteam, 0)
            winner_prev = team_rounds.get(wteam, 0)

            # The loser is set to at least round_now if it's higher than before
            team_rounds[lteam] = max(loser_prev, round_now)

            # The winner's round should be at least round_now+1
            # (since they advance to the next).
            team_rounds[wteam] = max(winner_prev, round_now + 1)

        # Convert to DataFrame
        df_rounds = pd.DataFrame([
            {"TeamID": t, "RoundReached": r}
            for t, r in team_rounds.items()
        ])

        return df_rounds

    @staticmethod
    def visualize_feature_vs_round(
        df_features: pd.DataFrame,   # e.g. [TeamID, SomeFeature, AnotherFeature, ...]
        df_rounds: pd.DataFrame,     # [TeamID, RoundReached]
        feature_col: str = "OffEff"
    ):
        """
        Merge features with RoundReached, then plot.
        We can do a boxplot or swarmplot to see how the feature
        changes by the round a team advanced to.
        """
        df_plot = df_features.merge(df_rounds, on="TeamID", how="inner")

        # We'll do a boxplot. RoundReached is discrete:
        sns.boxplot(data=df_plot, x="RoundReached", y=feature_col)
        plt.title(f"{feature_col} vs. Round Reached")
        plt.show()

        # Optionally a swarmplot or stripplot for more granular distribution
        sns.stripplot(data=df_plot, x="RoundReached", y=feature_col, color="red", alpha=0.5)
        plt.title(f"{feature_col} vs. Round Reached (stripplot overlay)")
        plt.show()


def evaluate_all_matchups_brier(
    df_probs: pd.DataFrame,
    df_actual_compact: pd.DataFrame
) -> float:
    """
    Calculate the Brier score for the subset of matchups that actually occurred,
    merging the predictions from df_probs with real outcomes from df_actual_compact.

    :param df_probs: output from `calculate_all_matchup_win_probs`, e.g.:
        [ID, TeamA, TeamB, ProbAWin]
    :param df_actual_compact: actual game results for the same season, e.g.:
        MRegularSeasonCompactResults or MNCAATourneyCompactResults, with columns:
        [WTeamID, LTeamID, ...].
    :return: brier_score (float)
    """

    # 1) Build a historical "lower/higher" ID outcome data
    #    For each game in df_actual_compact, identify lower/higher ID
    #    and define actual_lower_wins = 1 if lower==winner else 0
    rows = []
    for _, row in df_actual_compact.iterrows():
        w = row["WTeamID"]
        l = row["LTeamID"]
        lower = min(w, l)
        higher = max(w, l)
        # lower team won if lower == w
        actual_lower_wins = 1.0 if lower == w else 0.0
        rows.append({
            "TeamA": lower,  # consistent with df_probs
            "TeamB": higher,
            "ActualLowerWins": actual_lower_wins
        })

    df_actual = pd.DataFrame(rows)

    # 2) Merge predictions with these actual outcomes
    #    You only get rows for matchups that actually happened
    df_merged = df_probs.merge(
        df_actual, on=["TeamA", "TeamB"], how="inner"
    )
    # df_merged columns are now:
    #   [ID, TeamA, TeamB, ProbAWin, ActualLowerWins]

    # 3) For the Evaluator, we rename columns to match the defaults:
    #    pred_col="PredProbLowerWins", actual_col="ActualLowerWins"
    df_merged = df_merged.rename(columns={
        "ProbAWin": "PredProbLowerWins"
    })

    # 4) Now compute the Brier score
    score = Evaluator.brier_score(
        df_preds=df_merged,
        pred_col="PredProbLowerWins",
        actual_col="ActualLowerWins"
    )

    return score




# ### Step 0: Load Data
# Load detailed regular-season data:
# - Historical data (2024) for validation.
# - Future season data (2025) for predictions.




data_dir="/kaggle/input/march-machine-learning-mania-2025"
past_year=2024
target_year=2025


print(f"\n=== Loading data from {data_dir} ===")
# We'll do one loader for the 'past_year' (men only) to demonstrate
dl_past = DataLoader(data_dir=data_dir, target_season=past_year)
dl_past.load_data()

# We'll do another for the 'target_year' (men + women)
dl_future = DataLoader(data_dir=data_dir, target_season=target_year)
dl_future.load_data()


print(f"=== STEP 1: Validate Features on {past_year} Data ===")

# 1A) Build advanced features for menâ€™s {past_year}
df_men_past = dl_past.filter_season(dl_past.df_men_rs)
fe_past = AdvancedFeatureEngineer(df_men_past, dl_past.df_men_teams)
df_teamgames_past = fe_past.build_teamgames_df()
df_adv_past = fe_past.aggregate_team_stats(df_teamgames_past)
print("\nSample advanced features (men, past year):")
print(df_adv_past.head(5))

# 1B) Compute Elo for past_year
df_elo_past = compute_elo_feature(dl_past.df_men_rs, dl_past.df_men_teams, past_year)
df_adv_past = df_adv_past.merge(df_elo_past[["TeamID", "Elo"]], on="TeamID", how="left")

# 1C) Compare to KenPom (POM, RankingDayNum=133)
df_massey_past = dl_past.filter_season(dl_past.df_men_massey)
df_pom_past = df_massey_past[
    (df_massey_past["SystemName"] == "POM") &
    (df_massey_past["RankingDayNum"] == 133)
]
df_compare_past = df_adv_past.merge(df_pom_past[["TeamID", "OrdinalRank"]], on="TeamID", how="inner")

# Correlation
p_off, s_off = Evaluator.correlation_pearson_spearman(df_compare_past, "OffEff", "OrdinalRank")
p_elo, s_elo = Evaluator.correlation_pearson_spearman(df_compare_past, "Elo", "OrdinalRank")
print(f"\nOffEff vs. KenPom => pearson={p_off:.3f}, spearman={s_off:.3f}")
print(f"Elo vs. KenPom => pearson={p_elo:.3f}, spearman={s_elo:.3f}")

# Quick scatter
sns.scatterplot(data=df_compare_past, x="OffEff", y="OrdinalRank")
plt.title(f"OffEff vs KenPom ({past_year})")
plt.show()


df_tourney_past = dl_past.filter_season(dl_past.df_men_tourney)
calc = TournamentResultsCalculator()
df_rounds_past = calc.compute_tournament_outcomes(df_tourney_past)
calc.visualize_feature_vs_round(df_adv_past, df_rounds_past, "OffEff")
calc.visualize_feature_vs_round(df_adv_past, df_rounds_past, "Elo")


print(f"\n=== Brier Score for {past_year} results ===")
df_probs_past = calculate_all_matchup_win_probs(
    df_features=df_adv_past,
    elo_col="Elo",
    offeff_col="OffEff",
    season=past_year
)

df_men_rs_past_compact = dl_past.filter_season(dl_past.df_men_rs)[
    ["Season", "WTeamID", "LTeamID", "DayNum", "WLoc", "WScore", "LScore"]
]
score_past = evaluate_all_matchups_brier(df_probs_past, df_men_rs_past_compact)
print(f"Brier Score on {past_year} menâ€™s regular-season matchups = {score_past:.4f}")


print(f"\n=== STEP 2: Rank Teams in {target_year} (Men only) ===")
df_men_tfuture = dl_future.filter_season(dl_future.df_men_rs)
fe_future_men = AdvancedFeatureEngineer(df_men_tfuture, dl_future.df_men_teams)
df_tg_future_men = fe_future_men.build_teamgames_df()
df_adv_future_men = fe_future_men.aggregate_team_stats(df_tg_future_men)
df_elo_future_men = compute_elo_feature(dl_future.df_men_rs, dl_future.df_men_teams, target_year)
df_adv_future_men = df_adv_future_men.merge(df_elo_future_men[["TeamID","Elo"]], on="TeamID", how="left")

df_ranked_future_men = df_adv_future_men.sort_values("Elo", ascending=False)
print(df_ranked_future_men.head(10)[["TeamName","Elo"]])


print(f"\n=== STEP 3: Predict Win Probability for {target_year} (Men + Women) ===")
# Women
df_women_tfuture = dl_future.filter_season(dl_future.df_women_rs)
fe_future_women = AdvancedFeatureEngineer(df_women_tfuture, dl_future.df_women_teams)
df_tg_future_women = fe_future_women.build_teamgames_df()
df_adv_future_women = fe_future_women.aggregate_team_stats(df_tg_future_women)
df_elo_future_women = compute_elo_feature(dl_future.df_women_rs, dl_future.df_women_teams, target_year)
df_adv_future_women = df_adv_future_women.merge(df_elo_future_women[["TeamID","Elo"]], on="TeamID", how="left")

# Combine men + women
df_all_future = pd.concat([df_adv_future_men, df_adv_future_women], ignore_index=True).drop_duplicates("TeamID")

df_probs_future = calculate_all_matchup_win_probs(
    df_features=df_all_future,
    elo_col="Elo",
    offeff_col="OffEff",
    season=target_year
)
print(df_probs_future.head(10))
print(f"Created {df_probs_future.shape[0]} possible matchups for {target_year} (men+women).")


print(f"\n=== STEP 4: Generate Submission for {target_year} ===")

def is_men(tid):
    return 1000 <= tid < 2000
def is_women(tid):
    return 3000 <= tid < 4000

mask_same = []
for _, row in df_probs_future.iterrows():
    A, B = row["TeamA"], row["TeamB"]
    same_gender = ((is_men(A) and is_men(B)) or (is_women(A) and is_women(B)))
    mask_same.append(same_gender)

df_probs_future_same = df_probs_future[mask_same].copy()
df_probs_future_same.rename(columns={"ProbAWin":"Pred"}, inplace=True)
df_submit = df_probs_future_same[["ID","Pred"]].copy()

df_submit.to_csv(f"submission.csv", index=False)
print(f"Saved final {target_year} submission -> {df_submit.shape[0]} same-gender matchups.")

