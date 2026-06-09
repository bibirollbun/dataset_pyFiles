import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')



# Define data path
DATA_PATH = "/kaggle/input/march-machine-learning-mania-2025"


# --------------------------------------------------
# 1. Load the Sample Submission & Parse Matchup IDs
# --------------------------------------------------
sample_sub = pd.read_csv(f"{DATA_PATH}/SampleSubmissionStage2.csv")
print("Sample submission shape:", sample_sub.shape)

def parse_matchup_id(matchup_id):
    # Expected format: "2025_XXXX_YYYY"
    parts = matchup_id.split('_')
    return int(parts[1]), int(parts[2])

# Create new columns for the two team IDs (TeamID_A and TeamID_B)
sample_sub[['TeamID_A', 'TeamID_B']] = sample_sub['ID'].apply(lambda x: pd.Series(parse_matchup_id(x)))


# Load tournament seeds for men's and women's tournaments for season 2025
seeds_m = pd.read_csv(f"{DATA_PATH}/MNCAATourneySeeds.csv")
seeds_w = pd.read_csv(f"{DATA_PATH}/WNCAATourneySeeds.csv")
seeds_m = seeds_m[seeds_m['Season'] == 2024].copy()
seeds_w = seeds_w[seeds_w['Season'] == 2024].copy()
seeds_m['Tournament'] = "M"
seeds_w['Tournament'] = "W"
seeds_all = pd.concat([seeds_m, seeds_w], ignore_index=True)

# Extract numeric seed (e.g., "W01" -> 1)
seeds_all["SeedNum"] = seeds_all["Seed"].apply(lambda x: int(x[1:3]))

# Load team data for both men's and women's teams
m_teams = pd.read_csv(f"{DATA_PATH}/MTeams.csv")
w_teams = pd.read_csv(f"{DATA_PATH}/WTeams.csv")
teams_all = pd.concat([m_teams, w_teams], ignore_index=True)
# Create a lower-case version of team names for merging
teams_all["TeamName_lower"] = teams_all["TeamName"].str.lower()

# Merge seeds with team data (so we have TeamName_lower with each TeamID)
seeds_all = seeds_all.merge(teams_all[["TeamID", "TeamName_lower"]], on="TeamID", how="left")

# Load rating data
# For men's teams, use KenPom ratings (assumed file has columns: TeamName, AdjEM)
kenpom = pd.read_csv("/kaggle/input/kenpom-ratings/Kenpom_Ratings.csv")
kenpom["TeamName_lower"] = kenpom["Team"].str.lower()

# For women's teams, use Massey ratings (assumed file has columns: TeamName, Power)
massey = pd.read_csv("/kaggle/input/massey-ratings/export.csv")
massey["TeamName_lower"] = massey["Team"].str.lower()


kenpom.columns


massey.columns


# Merge ratings with seeds
# For men's teams (Tournament == "M"), merge with kenpom; for women's, merge with massey.
seeds_m_final = seeds_all[seeds_all["Tournament"]=="M"].merge(
    kenpom.iloc[:, [21, 4]],
    on="TeamName_lower",
    how="left"
)
seeds_w_final = seeds_all[seeds_all["Tournament"]=="W"].merge(
    massey[['TeamName_lower', 'Pwr']],
    on="TeamName_lower",
    how="left"
)
seeds_final = pd.concat([seeds_m_final, seeds_w_final], ignore_index=True)


# ----- 2. Define Win Probability Functions -----
def kenpom_wpct(adj1, adj2):
    """Men's win probability using AdjEM.
       tscore = 0.7*(AdjEM1 - AdjEM2) / 9."""
    tscore = 0.7 * (adj1 - adj2) / 9.0
    return norm.cdf(tscore)

def massey_wpct(pwr1, pwr2, home=0):
    """Women's win probability using Massey's Power.
       tscore = 0.7*(Power1 - Power2 + hfa) / 12, with hfa=2.73*home."""
    hfa = 2.73 * home
    tscore = 0.7 * (pwr1 - pwr2 + hfa) / 12.0
    return norm.cdf(tscore)

# Wrapper: choose based on tournament letter ("M" or "W")
def wpct_function(tournament, val1, val2, home=0):
    if tournament == "M":
        return kenpom_wpct(val1, val2)
    else:
        return massey_wpct(val1, val2, home)


# ----- 3. Simulate a Bracket (Detailed) -----
def simulate_bracket_detailed(teams_df, tournament):
    """
    Simulate one bracket.
    teams_df: DataFrame for one tournament (men's or women's), sorted by SeedNum.
    Returns a DataFrame with one row per game, containing:
      - Tournament, Round, TeamID_A, TeamID_B, Winner.
    """
    rounds = []
    current_round = teams_df.sort_values("SeedNum").reset_index(drop=True).copy()
    round_num = 1
    while len(current_round) > 1:
        n = len(current_round)
        # Pair first half vs. reversed second half (assumes even number of teams)
        teamA = current_round.iloc[:n//2].reset_index(drop=True)
        teamB = current_round.iloc[n//2:][::-1].reset_index(drop=True)
        round_matchups = []
        winners = []
        for i in range(len(teamA)):
            if tournament == "M":
                # Use AdjEM
                adjA = teamA.loc[i, "NetRtg"]
                adjB = teamB.loc[i, "NetRtg"]
                prob = kenpom_wpct(adjA, adjB) if pd.notnull(adjA) and pd.notnull(adjB) else 0.5
            else:
                # Use Power
                pwrA = teamA.loc[i, "Pwr"]
                pwrB = teamB.loc[i, "Pwr"]
                prob = massey_wpct(pwrA, pwrB, home=0) if pd.notnull(pwrA) and pd.notnull(pwrB) else 0.5
            # Simulate game outcome
            winner = teamA.loc[i, "TeamID"] if np.random.rand() < prob else teamB.loc[i, "TeamID"]
            winners.append(winner)
            # Record matchup (ensure lower TeamID is TeamID_A)
            t1, t2 = sorted([teamA.loc[i, "TeamID"], teamB.loc[i, "TeamID"]])
            round_matchups.append({
                "Tournament": tournament,
                "Round": round_num,
                "TeamID_A": t1,
                "TeamID_B": t2,
                "Winner": winner
            })
        rounds.extend(round_matchups)
        # Prepare teams for next round by filtering current teams by winners
        current_round = teams_df[teams_df["TeamID"].isin(winners)].copy()
        current_round = current_round.sort_values("SeedNum").reset_index(drop=True)
        round_num += 1
    return pd.DataFrame(rounds)

def simulate_brackets(teams_df, tournament, n_sim=50):
    sim_list = []
    for sim in range(n_sim):
        bracket = simulate_bracket_detailed(teams_df, tournament)
        bracket["Bracket"] = sim + 1
        sim_list.append(bracket)
    return pd.concat(sim_list, ignore_index=True)

# Separate men's and women's seeds
men_teams = seeds_final[seeds_final["Tournament"]=="M"].copy()
women_teams = seeds_final[seeds_final["Tournament"]=="W"].copy()

# Simulate 50 brackets for each tournament
men_sim = simulate_brackets(men_teams, "M", n_sim=1000)
women_sim = simulate_brackets(women_teams, "W", n_sim=1000)
simulations = pd.concat([men_sim, women_sim], ignore_index=True)


# ----- 4. Aggregate Simulation Results into Matchup Probabilities -----
# Create all possible matchups from tournament teams.
tournament_team_ids = seeds_final["TeamID"].unique()
all_pairs = list(combinations(sorted(tournament_team_ids), 2))
matchup_df = pd.DataFrame(all_pairs, columns=["TeamID_A", "TeamID_B"])
matchup_df["ID"] = matchup_df.apply(lambda row: f"2025_{row['TeamID_A']}_{row['TeamID_B']}", axis=1)
matchup_df["wins"] = 0
matchup_df["total"] = 0

# Update counts based on simulation outcomes.
for idx, game in simulations.iterrows():
    t1 = game["TeamID_A"]
    t2 = game["TeamID_B"]
    matchup_id = f"2025_{t1}_{t2}"
    # Find row in matchup_df with this ID.
    m_idx = matchup_df.index[matchup_df["ID"] == matchup_id].tolist()
    if m_idx:
        m_idx = m_idx[0]
        matchup_df.at[m_idx, "total"] += 1
        if game["Winner"] == t1:
            matchup_df.at[m_idx, "wins"] += 1

# Calculate predicted win probability for the lower TeamID in each matchup.
alpha = 1  # Smoothing parameter
matchup_df["Pred"] = matchup_df.apply(lambda row: (row["wins"] + alpha) / (row["total"] + 2*alpha)
                                      if row["total"] > 0 else 0.5, axis=1)



print(matchup_df.columns)
print(matchup_df.head())


print(sample_sub["ID"].head())


sample_sub["ID"] = sample_sub["ID"].str.strip()
matchup_df["ID"] = matchup_df["ID"].str.strip()


submission = sample_sub.merge(matchup_df[["ID", "Pred"]], on="ID", how="left")
print(submission.columns)
print(submission.head())


# ----- 5. Merge with Sample Submission and Save -----
submission = sample_sub.merge(matchup_df[["ID", "Pred"]], on="ID", how="left")
submission["Pred"] = submission["Pred_y"].fillna(submission["Pred_x"])
submission = submission[["ID", "Pred"]].sort_values("ID")
print("Final submission row count:", submission.shape[0])
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")

