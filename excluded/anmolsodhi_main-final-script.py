import pandas as pd
import os
import numpy as np
from scipy.stats import norm
import re
from tqdm import tqdm

# ---------------------------
# 1. Data Loading
# ---------------------------
print("Loading data...")
base_path = "../input/march-machine-learning-mania-2025/"

# Load full seeds files for 2025
MSeeds_full = pd.read_csv(os.path.join(base_path, "MNCAATourneySeeds.csv"))
WSeeds_full = pd.read_csv(os.path.join(base_path, "WNCAATourneySeeds.csv"))

# Filter to Season 2025 and remove play-in teams (seeds ending in 'a' or 'b')
MSeeds = MSeeds_full[(MSeeds_full['Season'] == 2025) & (~MSeeds_full['Seed'].str.endswith(('a','b')))]
WSeeds = WSeeds_full[(WSeeds_full['Season'] == 2025) & (~WSeeds_full['Seed'].str.endswith(('a','b')))]

# Load team spellings files
MTeamSpellings = pd.read_csv(os.path.join(base_path, "MTeamSpellings.csv"))
WTeamSpellings = pd.read_csv(os.path.join(base_path, "WTeamSpellings.csv"))

# Load Nate Silver's ratings files
print("\nLoading external ratings...")
silver_path = "../input/nate-silver-march-madness-data-set/"
msilver = pd.read_excel(os.path.join(silver_path, "Sbcb_Mens_Odds_March_16_2025.xlsx"))
wsilver = pd.read_excel(os.path.join(silver_path, "Sbcb_Womens_Odds_March_17_2025.xlsx"))

# Verify critical super team IDs are present in seeds
assert 1163 in MSeeds['TeamID'].values, "UConn (1163) missing from men's seeds"
assert 3376 in WSeeds['TeamID'].values, "South Carolina (3376) missing from women's seeds"

# ---------------------------
# 2. Data Processing
# ---------------------------
def normalize_team_name(name):
    """
    Normalize a team name:
      - Lowercase
      - Strip whitespace
      - Remove any parenthetical content (e.g., "(ny)")
      - Remove extra spaces
    """
    if pd.isna(name):
        return name
    name = name.lower().strip()
    name = re.sub(r"\(.*?\)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def process_ratings(ratings_df, spellings_df, seeds_df, tournament):
    """
    1. Normalize team names in ratings and spellings.
    2. Merge ratings with spellings (left join) to get TeamID.
    3. Merge with seeds using a RIGHT join so that all teams in the seeds file are included.
    4. Fill missing win_odds with the median win_odds from the ratings.
    5. Print warnings for missing seed or win_odds data.
    6. Return DataFrame with columns: [TeamID, win_odds, Seed, Tournament].
    """
    # Normalize names
    ratings_df['team_name'] = ratings_df['team_name'].apply(normalize_team_name)
    spellings_df['normalized_spelling'] = spellings_df['TeamNameSpelling'].apply(normalize_team_name)
    
    # Merge ratings with spellings (left join)
    merged = pd.merge(
        ratings_df,
        spellings_df,
        left_on='team_name',
        right_on='normalized_spelling',
        how='left'
    )
    if merged['normalized_spelling'].isna().any():
        missing_spellings = merged[merged['normalized_spelling'].isna()]['team_name'].unique()
        print(f"Warning: Missing team spellings for: {missing_spellings}.")
    
    # Merge with seeds using RIGHT join to ensure all seed teams are included
    tournament_teams = pd.merge(
        merged,
        seeds_df[['TeamID', 'Seed']],
        on='TeamID',
        how='right'
    )
    
    # Fill missing win_odds with the median from ratings_df.
    median_win_odds = ratings_df['win_odds'].median()
    tournament_teams['win_odds'] = tournament_teams['win_odds'].fillna(median_win_odds)
    
    if tournament_teams[['Seed', 'win_odds']].isna().any().any():
        missing_seed = tournament_teams[tournament_teams['Seed'].isna()]['TeamID'].unique()
        missing_odds = tournament_teams[tournament_teams['win_odds'].isna()]['TeamID'].unique()
        print(f"Warning: Missing seed data for TeamIDs: {missing_seed}.")
        print(f"Warning: Missing win_odds data for TeamIDs: {missing_odds}.")
    
    return tournament_teams[['TeamID', 'win_odds', 'Seed']].assign(Tournament=tournament)

print("\nProcessing men's ratings...")
m_ratings = process_ratings(msilver, MTeamSpellings, MSeeds, 'M')
print("\nProcessing women's ratings...")
w_ratings = process_ratings(wsilver, WTeamSpellings, WSeeds, 'W')

# Combine ratings for both genders and compute Power = 1 / win_odds
all_ratings = pd.concat([m_ratings, w_ratings])
all_ratings['Power'] = 1 / all_ratings['win_odds']

# Apply super team overrides: UConn (1163) and South Carolina (3376) get Power = 200.
all_ratings.loc[(all_ratings['Tournament'] == 'M') & (all_ratings['TeamID'] == 1163), 'Power'] = 200
all_ratings.loc[(all_ratings['Tournament'] == 'W') & (all_ratings['TeamID'] == 3376), 'Power'] = 200

# ---------------------------
# 3. Ensure 64 Teams in Bracket
# ---------------------------
def ensure_64_teams(teams_df, tournament):
    """
    Ensure that the main bracket has exactly 64 teams.
    If there are fewer teams, add placeholder teams with default win_odds and set Power=0 (guaranteed loss).
    """
    expected_count = 64
    current_count = len(teams_df)
    if current_count < expected_count:
        print(f"Warning: {tournament} bracket has only {current_count} teams. Adding {expected_count - current_count} placeholder teams.")
        median_odds = teams_df['win_odds'].median()
        placeholders = []
        for i in range(expected_count - current_count):
            placeholders.append({
                'TeamID': 9999 + i,  # Unique placeholder IDs (avoid conflict with real TeamIDs)
                'win_odds': median_odds,
                'Seed': f"X{current_count + i + 1:02d}",
                'Tournament': tournament,
                'Power': 0.0  # Set to 0 to guarantee loss
            })
        teams_df = pd.concat([teams_df, pd.DataFrame(placeholders)], ignore_index=True)
    return teams_df

# Build separate complete DataFrames for men's and women's teams.
m_ratings_complete = all_ratings[all_ratings['Tournament'] == 'M'].dropna(subset=['Seed'])
w_ratings_complete = all_ratings[all_ratings['Tournament'] == 'W'].dropna(subset=['Seed'])

men_count = len(m_ratings_complete)
women_count = len(w_ratings_complete)
print(f"\nMen's teams available: {men_count}; Women's teams available: {women_count}.")

m_ratings_final = m_ratings_complete.copy()
w_ratings_final = w_ratings_complete.copy()  # FIX: Use women's complete data here!

m_ratings_final = ensure_64_teams(m_ratings_final, 'M')
w_ratings_final = ensure_64_teams(w_ratings_final, 'W')

print(f"Men's unique teams: {m_ratings_final['TeamID'].nunique()}")  # Should be 64
print(f"Women's unique teams: {w_ratings_final['TeamID'].nunique()}")  # Should be 64

assert len(m_ratings_final) == 64, f"Men's bracket has {len(m_ratings_final)} teams; expected 64."
assert len(w_ratings_final) == 64, f"Women's bracket has {len(w_ratings_final)} teams; expected 64."

# ---------------------------
# 4. Bracket Simulation Functions
# ---------------------------
def extract_numeric_seed(seed_str):
    """Extract the numeric part of the seed (e.g., 'W01' -> 1)."""
    try:
        return int(seed_str[1:])
    except:
        return np.nan

def simulate_bracket(teams_df, gender, n_rounds=6):
    """
    Simulate a single tournament bracket round-by-round.
    Expects the bracket to have exactly 64 teams.
    Returns a list of game records with keys: 'Round', 'Team1', 'Team2', 'winner'.
    For women's rounds 1-2, prints HFA values for debugging.
    """
    expected = {1: 64, 2: 32, 3: 16, 4: 8, 5: 4, 6: 2}
    teams_df = teams_df.copy()
    teams_df['seed_num'] = teams_df['Seed'].apply(extract_numeric_seed)
    current = teams_df.sort_values('seed_num').reset_index(drop=True)
    
    if len(current) != 64:
        raise ValueError(f"Unexpected team count: {len(current)}. Expected 64 teams in main bracket.")
    
    games = []
    rnd = 1
    while rnd <= n_rounds and len(current) > 1:
        if len(current) != expected[rnd]:
            raise ValueError(f"Round {rnd}: expected {expected[rnd]} teams, got {len(current)}")
        winners = []
        for i in range(0, len(current), 2):
            t1 = current.iloc[i]
            t2 = current.iloc[i+1]
            if gender == 'M':
                p = norm.cdf((t1['Power'] - t2['Power']) / 11)
            else:
                if rnd <= 2:
                    seed1 = t1['seed_num']
                    seed2 = t2['seed_num']
                    hfa = 2.73 * (seed1 <= 4) - 2.73 * (seed2 <= 4)
                    # Debug print for HFA in women's rounds 1-2:
                    print(f"Round {rnd}: {t1['TeamID']} (Seed {seed1}) vs {t2['TeamID']} (Seed {seed2}) | HFA = {hfa}")
                    p = norm.cdf((t1['Power'] - t2['Power'] + hfa) / 11.5)
                else:
                    p = norm.cdf((t1['Power'] - t2['Power']) / 11.5)
            winner = t1 if np.random.rand() < p else t2
            games.append({
                'Round': rnd,
                'Team1': t1['TeamID'],
                'Team2': t2['TeamID'],
                'winner': winner['TeamID']
            })
            winners.append(winner)
        current = pd.DataFrame(winners).reset_index(drop=True)
        rnd += 1
    return games

# ---------------------------
# 5. Run Simulations and Aggregate Results
# ---------------------------
N_SIM = 5000
results_dict = {}  # key: "2025_<lowerTeamID>_<higherTeamID>", value: [lower_team_wins, total_occurrences]

def record_game(game):
    t1, t2 = game['Team1'], game['Team2']
    matchup_id = f"2025_{min(t1, t2)}_{max(t1, t2)}"
    lower = min(t1, t2)
    win = 1 if game['winner'] == lower else 0
    if matchup_id not in results_dict:
        results_dict[matchup_id] = [0, 0]
    results_dict[matchup_id][0] += win
    results_dict[matchup_id][1] += 1

print("\nSimulating Men's Brackets...")
for _ in tqdm(range(N_SIM), desc="Simulating Men's Brackets"):
    sim_games = simulate_bracket(m_ratings_final, 'M')
    for game in sim_games:
        record_game(game)

print("\nSimulating Women's Brackets...")
for _ in tqdm(range(N_SIM), desc="Simulating Women's Brackets"):
    sim_games = simulate_bracket(w_ratings_final, 'W')
    for game in sim_games:
        record_game(game)

simulated_results = []
for matchup_id, (lwins, total) in results_dict.items():
    prob = lwins / total if total > 0 else 0.5
    simulated_results.append({'ID': matchup_id, 'Pred': prob})
sim_df = pd.DataFrame(simulated_results)

# ---------------------------
# 6. Create Final Submission
# ---------------------------
sample_sub = pd.read_csv(os.path.join(base_path, "SampleSubmissionStage2.csv"))
submission = sample_sub[['ID']].merge(sim_df, on='ID', how='left')
submission['Pred'] = submission['Pred'].fillna(0.5)
submission = submission.sort_values('ID').round({'Pred': 5})

def validate_submission(sub_df):
    assert sub_df.shape == sample_sub.shape, f"Shape mismatch: {sub_df.shape} vs {sample_sub.shape}"
    assert list(sub_df.columns) == list(sample_sub.columns), "Column names mismatch"
    assert sub_df['ID'].str.startswith('2025_').all(), "Invalid ID format"
    assert sub_df['Pred'].between(0,1).all(), "Invalid predictions"
    print(f"Validation passed! Pred range: {sub_df['Pred'].min():.3f}-{sub_df['Pred'].max():.3f}")

validate_submission(submission)
submission.to_csv("submission.csv", index=False)
print("\nSubmission created successfully!")


