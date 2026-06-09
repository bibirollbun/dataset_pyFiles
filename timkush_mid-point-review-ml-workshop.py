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


# #!/usr/bin/env python3
# """
# main.py
# Author: YOU
# Date: 2025-01-01

# A single script that:
#  1) Loads all competition CSV data from the '"/kaggle/input/march-machine-learning-mania-2024"/' folder.
#  2) Trains a logistic regression model for Men's (M) data & another for Women's (W).
#  3) Produces a portfolio of bracket predictions (M and W) in the EXACT required submission format:
#     RowId,Tournament,Bracket,Slot,Team
#  4) Writes submission.csv with up to ~4000 total brackets (2000 M + 2000 W by default).

# Goal: Comply with the Kaggle 'March Machine Learning Mania 2024' instructions
#       by outputting a bracket portfolio with valid paths, scoring well via
#       the Brier Bracket Score, and finishing in a normal laptop's run-time.

# Usage:
#   python main.py
# Outputs:
#   submission.csv
# """

# import os
# import sys
# import numpy as np
# import pandas as pd

# # We use scikit-learn's LogisticRegression to predict win probabilities
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import log_loss

# ###############################################################################
# # 1) LOAD THE CSV FILES
# ###############################################################################

# def load_csv_safely(path):
#     """
#     Tries reading CSV with UTF-8 first, then fallback to 'latin1'/cp1252 if needed.
#     This helps fix "UnicodeDecodeError: 'utf-8' codec can't decode byte" issues.
#     """
#     try:
#         return pd.read_csv(path)
#     except UnicodeDecodeError:
#         print(f"Retrying {path} with latin1 encoding...")
#         return pd.read_csv(path, encoding="latin1")

# def load_all_data(data_dir="data"):
#     """
#     Loads all the CSV files we need, returns them as a dict of DataFrames.
#     If a file is missing or has an encoding issue, we print a warning but continue.
#     """
#     csv_files = [
#         # Men’s data
#         "MSeasons.csv",
#         "MTeams.csv",
#         "MRegularSeasonCompactResults.csv",
#         "MRegularSeasonDetailedResults.csv",
#         "MNCAATourneyCompactResults.csv",
#         "MNCAATourneyDetailedResults.csv",
#         "MNCAATourneySeeds.csv",
#         "MNCAATourneySlots.csv",
#         # Women’s data
#         "WSeasons.csv",
#         "WTeams.csv",
#         "WRegularSeasonCompactResults.csv",
#         "WRegularSeasonDetailedResults.csv",
#         "WNCAATourneyCompactResults.csv",
#         "WNCAATourneyDetailedResults.csv",
#         "WNCAATourneySeeds.csv",
#         "WNCAATourneySlots.csv",
#         # Shared
#         "Cities.csv",  # men’s
#         "Cities.csv",  # women's doesn't exist, so might fail. We'll try
#         "MGameCities.csv",
#         "WGameCities.csv",
#         # Possibly more merges needed:
#         "MTeamCoaches.csv",
#         "MTeamConferences.csv",
#         "Conferences.csv",
#         # Next-year seeds file used for 2024 bracket
#         # (During the competition, it is actually 2023 seeds, then replaced after selection Sunday.)
#         "2024_tourney_seeds.csv"
#     ]

#     dfs = {}
#     for f in csv_files:
#         path = os.path.join(data_dir, f)
#         if not os.path.exists(path):
#             print(f"WARNING: {f} not found in {data_dir}!")
#             continue
#         try:
#             df = load_csv_safely(path)
#             dfs[f] = df
#         except Exception as e:
#             print(f"WARNING: error reading {f}: {e}")
#             # We'll skip it
#     return dfs


# ###############################################################################
# # 1B) OPTIONAL EVALUATION: SPLIT & LOG LOSS
# ###############################################################################

# def evaluate_model_time_split_men(df_all, year_split=2022):
#     """
#     Time-based split for men’s dataset:
#      - Train on seasons <= year_split
#      - Validate on seasons > year_split
#      We'll measure log loss on the validation subset.
#     """
#     df_train = df_all[df_all["Season"] <= year_split]
#     df_val   = df_all[df_all["Season"] >  year_split]

#     if len(df_train)==0 or len(df_val)==0:
#         print("WARNING: Not enough men’s data for train/val split; skipping evaluation.")
#         return

#     # Features for the baseline: ScoreDiff, Loc
#     feats = ["ScoreDiff", "Loc"]

#     X_train = df_train[feats].values
#     y_train = df_train["Result"].values

#     scaler = StandardScaler()
#     Xs_train = scaler.fit_transform(X_train)

#     model = LogisticRegression(random_state=42, max_iter=500)
#     model.fit(Xs_train, y_train)

#     X_val = df_val[feats].values
#     y_val = df_val["Result"].values
#     Xs_val= scaler.transform(X_val)
#     preds = model.predict_proba(Xs_val)[:,1]
#     ll = log_loss(y_val, preds)
#     print(f"Men’s validation log loss (<= {year_split} vs. > {year_split}): {ll:.5f}")


# def evaluate_model_time_split_women(df_all, year_split=2022):
#     """
#     Time-based split for women’s dataset:
#      - Train on seasons <= year_split
#      - Validate on seasons > year_split
#      We'll measure log loss on the validation subset.
#     """
#     df_train = df_all[df_all["Season"] <= year_split]
#     df_val   = df_all[df_all["Season"] >  year_split]

#     if len(df_train)==0 or len(df_val)==0:
#         print("WARNING: Not enough women’s data for train/val split; skipping evaluation.")
#         return

#     # Features for the baseline: ScoreDiff, Loc
#     feats = ["ScoreDiff", "Loc"]

#     X_train = df_train[feats].values
#     y_train = df_train["Result"].values

#     scaler = StandardScaler()
#     Xs_train = scaler.fit_transform(X_train)

#     model = LogisticRegression(random_state=43, max_iter=500)
#     model.fit(Xs_train, y_train)

#     X_val = df_val[feats].values
#     y_val = df_val["Result"].values
#     Xs_val= scaler.transform(X_val)
#     preds = model.predict_proba(Xs_val)[:,1]
#     ll = log_loss(y_val, preds)
#     print(f"Women’s validation log loss (<= {year_split} vs. > {year_split}): {ll:.5f}")


# ###############################################################################
# # 2) SIMPLE MODEL TRAINING
# ###############################################################################

# def prep_training_data_men(dfs, season_start=2003, season_end=2023):
#     """
#     Prepare men’s training data from MRegularSeasonDetailedResults + MNCAATourneyDetailedResults.
#     We'll do a minimal approach: for each game, we extract a handful of features, then
#     produce two rows (Team A vs Team B, Team B vs Team A).
#     We'll do a logistic regression to estimate p(Team A beats Team B).
#     """

#     needed = [
#         "MRegularSeasonDetailedResults.csv",
#         "MNCAATourneyDetailedResults.csv",
#     ]
#     for nf in needed:
#         if nf not in dfs:
#             raise ValueError(f"Missing {nf} in dfs!")

#     reg = dfs["MRegularSeasonDetailedResults.csv"]
#     tour = dfs["MNCAATourneyDetailedResults.csv"]

#     # Combine reg + tour
#     allgames = pd.concat([reg, tour], ignore_index=True)
#     # Filter by season
#     allgames = allgames[(allgames["Season"]>=season_start) & (allgames["Season"]<=season_end)].copy()
    
#     # Minimal features: Score margin, home/away (WLoc can be 'H' = 1, 'A' = -1, 'N' = 0)
#     def loc_value(loc):
#         if loc=='H':
#             return 1
#         elif loc=='A':
#             return -1
#         else:
#             return 0

#     rows = []
#     for idx, row in allgames.iterrows():
#         season = row["Season"]
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         margin = row["WScore"] - row["LScore"]
#         # We do a fallback of 0 if WLoc not present
#         wloc = loc_value(row.get("WLoc", "N"))

#         # Win side
#         rows.append({
#             "Season": season,
#             "TeamA": wtid,
#             "TeamB": ltid,
#             "ScoreDiff": margin,
#             "Loc": wloc,
#             "Result": 1
#         })
#         # Loss side (mirror)
#         rows.append({
#             "Season": season,
#             "TeamA": ltid,
#             "TeamB": wtid,
#             "ScoreDiff": -margin,
#             "Loc": -wloc,
#             "Result": 0
#         })

#     df_model = pd.DataFrame(rows)
#     return df_model

# def train_model_men(df_model):
#     """
#     Train a logistic regression on df_model.
#     We'll use [ScoreDiff, Loc] as features. Very simple.
#     """
#     X = df_model[["ScoreDiff","Loc"]].values
#     y = df_model["Result"].values
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     model = LogisticRegression(
#         random_state=42,
#         max_iter=500,
#     )
#     model.fit(Xs, y)
#     return model, scaler


# def prep_training_data_women(dfs, season_start=2010, season_end=2023):
#     """
#     Similar approach, but for women's data (WRegularSeasonDetailedResults, WNCAATourneyDetailedResults).
#     """
#     needed = [
#         "WRegularSeasonDetailedResults.csv",
#         "WNCAATourneyDetailedResults.csv",
#     ]
#     for nf in needed:
#         if nf not in dfs:
#             raise ValueError(f"Missing {nf} in dfs!")
#     reg = dfs["WRegularSeasonDetailedResults.csv"]
#     tour = dfs["WNCAATourneyDetailedResults.csv"]

#     allgames = pd.concat([reg, tour], ignore_index=True)
#     allgames = allgames[(allgames["Season"]>=season_start) & (allgames["Season"]<=season_end)].copy()

#     def loc_value(loc):
#         if loc=='H':
#             return 1
#         elif loc=='A':
#             return -1
#         else:
#             return 0

#     rows = []
#     for idx, row in allgames.iterrows():
#         season = row["Season"]
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         margin = row["WScore"] - row["LScore"]
#         wloc = loc_value(row.get("WLoc", "N"))

#         rows.append({
#             "Season": season,
#             "TeamA": wtid,
#             "TeamB": ltid,
#             "ScoreDiff": margin,
#             "Loc": wloc,
#             "Result": 1
#         })
#         rows.append({
#             "Season": season,
#             "TeamA": ltid,
#             "TeamB": wtid,
#             "ScoreDiff": -margin,
#             "Loc": -wloc,
#             "Result": 0
#         })
#     df_model = pd.DataFrame(rows)
#     return df_model

# def train_model_women(df_model):
#     """
#     Train logistic regression on women's data.
#     """
#     X = df_model[["ScoreDiff","Loc"]].values
#     y = df_model["Result"].values
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)
#     model = LogisticRegression(random_state=43, max_iter=500)
#     model.fit(Xs, y)
#     return model, scaler


# ###############################################################################
# # 3) BUILD THE 2024 TOURNAMENT BRACKET STRUCTURE FOR MEN & WOMEN, THEN FILL IT
# ###############################################################################

# def load_slots_and_seeds_2024(dfs, men=True):
#     """
#     For men, read from:
#       - "2024_tourney_seeds.csv" with Tournament='M'
#       - "MNCAATourneySlots.csv" Season=2024
#     For women, similarly with 'W'.
#     """
#     if "2024_tourney_seeds.csv" not in dfs:
#         raise ValueError("Missing 2024_tourney_seeds.csv in DFS!")
#     df_seeds = dfs["2024_tourney_seeds.csv"]
#     if men:
#         df_seeds = df_seeds[df_seeds["Tournament"]=='M'].copy()
#         slot_file = "MNCAATourneySlots.csv"
#         if slot_file not in dfs:
#             raise ValueError("No MNCAATourneySlots.csv for men!")
#         df_slots = dfs[slot_file]
#         df_slots = df_slots[df_slots["Season"]==2024].copy()
#     else:
#         df_seeds = df_seeds[df_seeds["Tournament"]=='W'].copy()
#         slot_file = "WNCAATourneySlots.csv"
#         if slot_file not in dfs:
#             raise ValueError("No WNCAATourneySlots.csv for women!")
#         df_slots = dfs[slot_file]
#         df_slots = df_slots[df_slots["Season"]==2024].copy()

#     seeds_dict = {}
#     for idx, row in df_seeds.iterrows():
#         seed = row["Seed"]
#         tid = row["TeamID"]
#         seeds_dict[seed] = tid

#     bracket_slots = {}
#     for idx, row in df_slots.iterrows():
#         slot = row["Slot"]
#         strong = row["StrongSeed"]
#         weak = row["WeakSeed"]
#         bracket_slots[slot] = (strong, weak)
#     return seeds_dict, bracket_slots


# def predict_game_probability(teamA, teamB, model, scaler, men, dfs):
#     """
#     Predict p(TeamA beats TeamB) using the logistic regression from the trained model.
#     We'll do a simplest approach: no partial 2024 data, just 0 ScoreDiff => 0.0, Loc=0 => neutral => ~0.5.
#     """
#     diff = 0.0  # fallback for ScoreDiff
#     loc = 0.0   # fallback for neutral
#     X = np.array([[diff, loc]])
#     Xs = scaler.transform(X)
#     p = model.predict_proba(Xs)[0,1]
#     return p


# def simulate_bracket_once(seeds_dict, bracket_slots, model, scaler, men, dfs):
#     """
#     Simulate the bracket by topological order from Round1..Round6,
#     picking winners randomly based on predicted probabilities from predict_game_probability.
#     """
#     slot_keys = list(bracket_slots.keys())

#     def slot_sort_key(s):
#         if s=="R6CH": 
#             return (6,"CH")
#         if s in ["R5WX","R5YZ"]:
#             return (5, s[2:])
#         if s.startswith("R"):
#             rnd = int(s[1])
#             tail= s[2:]
#             return (rnd, tail)
#         return (99, s)
#     slot_keys.sort(key=slot_sort_key)

#     bracket_winners = {}

#     def occupant_teamid(label):
#         if label in seeds_dict:
#             return seeds_dict[label]
#         # else assume it's a slot name => bracket_winners => occupant
#         if label in bracket_winners:
#             return occupant_teamid(bracket_winners[label])
#         return None

#     for slot in slot_keys:
#         strong, weak = bracket_slots[slot]
#         # occupant label might be seeds or previous slot names
#         occA = bracket_winners.get(strong, strong)
#         occB = bracket_winners.get(weak, weak)

#         tidA = occupant_teamid(occA)
#         tidB = occupant_teamid(occB)
#         if (tidA is None) or (tidB is None):
#             # fallback
#             bracket_winners[slot] = occA
#         else:
#             pA = predict_game_probability(tidA, tidB, model, scaler, men, dfs)
#             if np.random.rand() < pA:
#                 bracket_winners[slot] = occA
#             else:
#                 bracket_winners[slot] = occB

#     return bracket_winners


# def build_bracket_rows(bracket_winners, men=True):
#     """
#     Convert bracket_winners to lines: (slot, team, 'M'/'W')
#     """
#     slots = list(bracket_winners.keys())
#     def slot_sort_key(s):
#         if s=="R6CH": 
#             return (6,"CH")
#         if s in ["R5WX","R5YZ"]:
#             return (5, s[2:])
#         if s.startswith("R"):
#             rnd = int(s[1])
#             tail = s[2:]
#             return (rnd, tail)
#         return (99, s)
#     slots.sort(key=slot_sort_key)

#     tchar = 'M' if men else 'W'
#     bracket_rows = []
#     for slot in slots:
#         pick = bracket_winners[slot]
#         bracket_rows.append((slot, pick, tchar))
#     return bracket_rows


# ###############################################################################
# # 4) MAIN
# ###############################################################################

# def main():
#     print("Loading data...")
#     dfs = load_all_data("/kaggle/input/march-machine-learning-mania-2024") 
#     print("Data loaded.")

#     # 4.1) Prepare men’s training data, evaluate, then train
#     print("Preparing men’s training data (2003–2023)...")
#     dfm = prep_training_data_men(dfs, 2003, 2023)
#     print(f"Men’s training set: {len(dfm)} rows")

#     # OPTIONAL EVALUATION: time-based split
#     evaluate_model_time_split_men(dfm, year_split=2022)

#     men_model, men_scaler = train_model_men(dfm)
#     print("Men’s model trained.")

#     # 4.2) Prepare women’s training data, evaluate, then train
#     print("Preparing women’s training data (2010–2023)...")
#     dfw = prep_training_data_women(dfs, 2010, 2023)
#     print(f"Women’s training set: {len(dfw)} rows")

#     # OPTIONAL EVALUATION: time-based split
#     evaluate_model_time_split_women(dfw, year_split=2022)

#     women_model, women_scaler = train_model_women(dfw)
#     print("Women’s model trained.")

#     # 4.3) Load 2024 bracket structure for men + seeds
#     print("Loading Men’s 2024 bracket structure + seeds...")
#     men_seeds_dict, men_bracket_slots = load_slots_and_seeds_2024(dfs, men=True)

#     # 4.4) Load 2024 bracket structure for women + seeds
#     print("Loading Women’s 2024 bracket structure + seeds...")
#     women_seeds_dict, women_bracket_slots = load_slots_and_seeds_2024(dfs, men=False)

#     # 4.5) Generate multiple bracket predictions for men + women
#     NUM_BRACKETS_MEN = 100
#     NUM_BRACKETS_WOMEN = 100

#     big_rows = []  # will hold (RowId, Tournament, Bracket, Slot, Team)
#     rowId = 1

#     # MEN
#     for b_idx in range(1, NUM_BRACKETS_MEN+1):
#         winners = simulate_bracket_once(men_seeds_dict, men_bracket_slots, men_model, men_scaler, True, dfs)
#         bracket_rows = build_bracket_rows(winners, men=True)
#         for (slot, pick, tchar) in bracket_rows:
#             big_rows.append((rowId, tchar, b_idx, slot, pick))
#             rowId += 1

#     # WOMEN
#     for b_idx in range(1, NUM_BRACKETS_WOMEN+1):
#         winners = simulate_bracket_once(women_seeds_dict, women_bracket_slots, women_model, women_scaler, False, dfs)
#         bracket_rows = build_bracket_rows(winners, men=False)
#         for (slot, pick, tchar) in bracket_rows:
#             big_rows.append((rowId, tchar, b_idx, slot, pick))
#             rowId += 1

#     # 4.6) Output to submission.csv in EXACT required format:
#     df_sub = pd.DataFrame(big_rows, columns=["RowId","Tournament","Bracket","Slot","Team"])
#     df_sub.to_csv("submission.csv", index=False)
#     print(f"Successfully wrote {len(df_sub)} rows to submission.csv!")
#     print("Done. This bracket portfolio can be submitted to Kaggle.")


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# """
# main.py
# Author: YOU
# Date: 2025-01-01

# Improved baseline logistic regression for March Madness 2024, with:
#  1) More features (margin of victory, offensive rebounds, personal fouls, blocks)
#  2) Exponential decay weighting for older seasons
#  3) Precomputed aggregator for 2024 partial data to avoid repeated scanning
#  4) Correct final bracket output format EXACTLY as required:
#     "RowId,Tournament,Bracket,Slot,Team"

# Usage:
#   python main.py

# Outputs:
#   submission.csv
# """

# import os
# import sys
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler


# ###############################################################################
# # 0) LOAD CSV FILES
# ###############################################################################

# def load_csv_safely(path):
#     """
#     Tries reading CSV with UTF-8 first, then fallback to 'latin1'/cp1252 if needed.
#     Helps fix "UnicodeDecodeError: 'utf-8' codec can't decode byte" issues.
#     """
#     try:
#         return pd.read_csv(path)
#     except UnicodeDecodeError:
#         print(f"Retrying {path} with latin1 encoding...")
#         return pd.read_csv(path, encoding="latin1")

# def load_all_data(data_dir="data"):
#     """
#     Loads all the CSV files we need, returns them as a dict of DataFrames.
#     We skip any missing files but warn.
#     """
#     csv_files = [
#         # Men’s data
#         "MSeasons.csv",
#         "MTeams.csv",
#         "MRegularSeasonCompactResults.csv",
#         "MRegularSeasonDetailedResults.csv",
#         "MNCAATourneyCompactResults.csv",
#         "MNCAATourneyDetailedResults.csv",
#         "MNCAATourneySeeds.csv",
#         "MNCAATourneySlots.csv",
#         # Women’s data
#         "WSeasons.csv",
#         "WTeams.csv",
#         "WRegularSeasonCompactResults.csv",
#         "WRegularSeasonDetailedResults.csv",
#         "WNCAATourneyCompactResults.csv",
#         "WNCAATourneyDetailedResults.csv",
#         "WNCAATourneySeeds.csv",
#         "WNCAATourneySlots.csv",
#         # Shared
#         "Cities.csv",  # men’s
#         "MGameCities.csv",
#         "WGameCities.csv",
#         # Possibly more merges needed:
#         "MTeamCoaches.csv",
#         "MTeamConferences.csv",
#         "Conferences.csv",
#         # Next-year seeds (2024)
#         "2024_tourney_seeds.csv"
#     ]

#     dfs = {}
#     for f in csv_files:
#         path = os.path.join(data_dir, f)
#         if not os.path.exists(path):
#             print(f"WARNING: {f} not found in {data_dir}!")
#             continue
#         try:
#             df = load_csv_safely(path)
#             dfs[f] = df
#         except Exception as e:
#             print(f"WARNING: error reading {f}: {e}")
#     return dfs


# ###############################################################################
# # 1) PREPARE TRAINING DATA WITH EXTRA FEATURES + DECAY WEIGHT
# ###############################################################################

# def prep_training_data_men(dfs, season_start=2003, season_end=2023, decay_base=0.94):
#     """
#     Prepare men’s training data from MRegularSeasonDetailedResults + MNCAATourneyDetailedResults,
#     but incorporate extra features:
#      - margin of victory
#      - offensive rebounds
#      - personal fouls
#      - blocks
#     We'll do a logistic regression to estimate p(Team A beats Team B).
#     We also apply exponential decay weighting so that older seasons weigh less.

#     If you don't have these columns in your data, adapt accordingly.
#     """
#     needed = [
#         "MRegularSeasonDetailedResults.csv",
#         "MNCAATourneyDetailedResults.csv",
#     ]
#     for nf in needed:
#         if nf not in dfs:
#             raise ValueError(f"Missing {nf} in dfs!")

#     reg = dfs["MRegularSeasonDetailedResults.csv"]
#     tour = dfs["MNCAATourneyDetailedResults.csv"]
#     allgames = pd.concat([reg, tour], ignore_index=True)
#     allgames = allgames[(allgames["Season"] >= season_start) & (allgames["Season"] <= season_end)].copy()

#     def loc_value(loc):
#         if loc=='H':
#             return 1
#         elif loc=='A':
#             return -1
#         else:
#             return 0

#     rows = []
#     for idx, row in allgames.iterrows():
#         season = row["Season"]
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         wscore = row["WScore"]
#         lscore = row["LScore"]
#         margin = wscore - lscore
#         wloc = loc_value(row.get("WLoc","N"))  # default N if missing
#         # Additional stats
#         wor = row.get("WOR", 0)
#         lor = row.get("LOR", 0)
#         wpf = row.get("WPF", 0)
#         lpf = row.get("LPF", 0)
#         wblk= row.get("WBlk", 0)
#         lblk= row.get("LBlk", 0)

#         # Decay weight factor
#         # e.g. if season=2023, diff from end=0 => factor=decay_base^0=1
#         # if season=2022 => factor=decay_base^(2023-2022)=decay_base^1
#         # ...
#         years_ago = season_end - season
#         weight = (decay_base**years_ago)

#         # Winning side
#         rows.append({
#             "Season": season,
#             "TeamA": wtid,
#             "TeamB": ltid,
#             "Margin": margin,
#             "Loc": wloc,
#             "OR_diff": (wor - lor),
#             "PF_diff": (wpf - lpf),
#             "Blk_diff": (wblk - lblk),
#             "Result": 1,
#             "Weight": weight
#         })
#         # Losing side (mirror game)
#         rows.append({
#             "Season": season,
#             "TeamA": ltid,
#             "TeamB": wtid,
#             "Margin": -margin,
#             "Loc": -wloc,
#             "OR_diff": (lor - wor),
#             "PF_diff": (lpf - wpf),
#             "Blk_diff": (lblk - wblk),
#             "Result": 0,
#             "Weight": weight
#         })

#     df_model = pd.DataFrame(rows)
#     return df_model


# def train_model_men(df_model):
#     """
#     Train logistic regression on men’s data with these features:
#       [Margin, Loc, OR_diff, PF_diff, Blk_diff]
#     Weighted by 'df_model["Weight"]'.
#     """
#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff"]
#     X = df_model[feats].values
#     y = df_model["Result"].values
#     w = df_model["Weight"].values  # exponential decay weighting

#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     model = LogisticRegression(
#         random_state=42,
#         max_iter=600,
#     )
#     model.fit(Xs, y, sample_weight=w)
#     return model, scaler


# def prep_training_data_women(dfs, season_start=2010, season_end=2023, decay_base=0.94):
#     """
#     Similar approach for women's data. 
#     """
#     needed = [
#         "WRegularSeasonDetailedResults.csv",
#         "WNCAATourneyDetailedResults.csv",
#     ]
#     for nf in needed:
#         if nf not in dfs:
#             raise ValueError(f"Missing {nf} in dfs!")
#     reg = dfs["WRegularSeasonDetailedResults.csv"]
#     tour = dfs["WNCAATourneyDetailedResults.csv"]
#     allgames = pd.concat([reg, tour], ignore_index=True)
#     allgames = allgames[(allgames["Season"] >= season_start) & (allgames["Season"] <= season_end)].copy()

#     def loc_value(loc):
#         if loc=='H':
#             return 1
#         elif loc=='A':
#             return -1
#         else:
#             return 0

#     rows = []
#     for idx, row in allgames.iterrows():
#         season = row["Season"]
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         wscore = row["WScore"]
#         lscore = row["LScore"]
#         margin = wscore - lscore
#         wloc = loc_value(row.get("WLoc","N"))
#         wor = row.get("WOR", 0)
#         lor = row.get("LOR", 0)
#         wpf = row.get("WPF", 0)
#         lpf = row.get("LPF", 0)
#         wblk= row.get("WBlk", 0)
#         lblk= row.get("LBlk", 0)

#         years_ago = season_end - season
#         weight = (decay_base**years_ago)

#         rows.append({
#             "Season": season,
#             "TeamA": wtid,
#             "TeamB": ltid,
#             "Margin": margin,
#             "Loc": wloc,
#             "OR_diff": (wor - lor),
#             "PF_diff": (wpf - lpf),
#             "Blk_diff": (wblk - lblk),
#             "Result": 1,
#             "Weight": weight
#         })
#         rows.append({
#             "Season": season,
#             "TeamA": ltid,
#             "TeamB": wtid,
#             "Margin": -margin,
#             "Loc": -wloc,
#             "OR_diff": (lor - wor),
#             "PF_diff": (lpf - wpf),
#             "Blk_diff": (lblk - wblk),
#             "Result": 0,
#             "Weight": weight
#         })

#     df_model = pd.DataFrame(rows)
#     return df_model


# def train_model_women(df_model):
#     """
#     Train logistic regression on women’s data with same features:
#     """
#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff"]
#     X = df_model[feats].values
#     y = df_model["Result"].values
#     w = df_model["Weight"].values

#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)
#     model = LogisticRegression(random_state=43, max_iter=600)
#     model.fit(Xs, y, sample_weight=w)
#     return model, scaler


# ###############################################################################
# # 2) LOAD 2024 SLOTS/SEEDS AND PRECOMPUTE 2024 TEAM STATS
# ###############################################################################

# def load_slots_and_seeds_2024(dfs, men=True):
#     """
#     For men:  we read only rows where Tournament='M' from 2024_tourney_seeds.csv,
#               bracket from MNCAATourneySlots.csv Season=2024
#     For women: we read rows where Tournament='W',
#                bracket from WNCAATourneySlots.csv Season=2024
#     """
#     if "2024_tourney_seeds.csv" not in dfs:
#         raise ValueError("Missing 2024_tourney_seeds.csv in DFS!")
#     df_seeds = dfs["2024_tourney_seeds.csv"]

#     if men:
#         df_seeds = df_seeds[df_seeds["Tournament"] == "M"].copy()
#         slot_file = "MNCAATourneySlots.csv"
#         if slot_file not in dfs:
#             raise ValueError("No MNCAATourneySlots.csv for men!")
#         df_slots = dfs[slot_file]
#         df_slots = df_slots[df_slots["Season"] == 2024].copy()
#     else:
#         df_seeds = df_seeds[df_seeds["Tournament"] == "W"].copy()
#         slot_file = "WNCAATourneySlots.csv"
#         if slot_file not in dfs:
#             raise ValueError("No WNCAATourneySlots.csv for women!")
#         df_slots = dfs[slot_file]
#         df_slots = df_slots[df_slots["Season"] == 2024].copy()

#     seeds_dict = {}
#     for idx, row in df_seeds.iterrows():
#         seed = row["Seed"]  # e.g. "W01"
#         tid  = row["TeamID"]
#         seeds_dict[seed] = tid

#     bracket_slots = {}
#     for idx, row in df_slots.iterrows():
#         slot = row["Slot"]
#         strong = row["StrongSeed"]
#         weak = row["WeakSeed"]
#         bracket_slots[slot] = (strong, weak)

#     return seeds_dict, bracket_slots


# def build_2024_aggregator(dfs, men=True):
#     """
#     Read the 2024 M/WRegularSeasonDetailedResults once, build aggregator: 
#     aggregator[teamId] = { 'margin':..., 'or':..., 'pf':..., 'blk':..., 'games':... }
#     So we can quickly compute an average feature for 2024 partial data.
#     """
#     if men:
#         df_reg2024 = dfs.get("MRegularSeasonDetailedResults.csv", None)
#     else:
#         df_reg2024 = dfs.get("WRegularSeasonDetailedResults.csv", None)

#     if df_reg2024 is None:
#         return {}
#     df_reg2024 = df_reg2024[df_reg2024["Season"] == 2024].copy()
#     if len(df_reg2024) == 0:
#         return {}

#     stats = {}
#     for idx, row in df_reg2024.iterrows():
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         margin = row["WScore"] - row["LScore"]
#         wor = row.get("WOR", 0)
#         lor = row.get("LOR", 0)
#         wpf = row.get("WPF", 0)
#         lpf = row.get("LPF", 0)
#         wblk= row.get("WBlk",0)
#         lblk= row.get("LBlk",0)

#         if wtid not in stats:
#             stats[wtid] = {"margin":0,"or":0,"pf":0,"blk":0,"games":0}
#         stats[wtid]["margin"] += margin
#         stats[wtid]["or"] += (wor)
#         stats[wtid]["pf"] += (wpf)
#         stats[wtid]["blk"]+= (wblk)
#         stats[wtid]["games"] += 1

#         if ltid not in stats:
#             stats[ltid] = {"margin":0,"or":0,"pf":0,"blk":0,"games":0}
#         stats[ltid]["margin"] += (-margin)
#         stats[ltid]["or"] += (lor)
#         stats[ltid]["pf"] += (lpf)
#         stats[ltid]["blk"]+= (lblk)
#         stats[ltid]["games"] += 1

#     return stats


# def get_2024_team_averages(teamId, aggregator):
#     """
#     aggregator => aggregator[teamId] = dict(margin=..., or=..., pf=..., blk=..., games=...)
#     returns tuple: (avg_margin, avg_or, avg_pf, avg_blk)
#     fallback 0 if no data
#     """
#     if (teamId not in aggregator) or (aggregator[teamId]["games"] == 0):
#         return (0.0, 0.0, 0.0, 0.0)
#     st = aggregator[teamId]
#     g = st["games"]
#     return (
#         st["margin"] / g,
#         st["or"]     / g,
#         st["pf"]     / g,
#         st["blk"]    / g,
#     )


# ###############################################################################
# # 3) GAME PREDICTION & BRACKET SIMULATION
# ###############################################################################

# def predict_game_probability(teamA, teamB, model, scaler, men, aggregator_2024):
#     """
#     We do a simple approach: incorporate the difference in average margin, OR, PF, BLK from aggregator_2024.
#     The other features are set to neutral (Loc=0).
#     If aggregator_2024 is empty, fallback to zero differences => 0.5 for all matches.
#     """
#     # If aggregator is empty => fallback 0
#     if not aggregator_2024:
#         # Then all features = 0 => model => ~0.5
#         feats = np.array([[0,0,0,0,0]], dtype=float)  # [Margin,Loc,OR_diff,PF_diff,Blk_diff]
#         feats_s = scaler.transform(feats)
#         return model.predict_proba(feats_s)[0,1]

#     a_margin, a_or, a_pf, a_blk = get_2024_team_averages(teamA, aggregator_2024)
#     b_margin, b_or, b_pf, b_blk = get_2024_team_averages(teamB, aggregator_2024)

#     margin_diff = a_margin - b_margin
#     or_diff = a_or - b_or
#     pf_diff = a_pf - b_pf
#     blk_diff= a_blk- b_blk
#     loc=0.0  # neutral
#     row_feat = [[margin_diff, loc, or_diff, pf_diff, blk_diff]]
#     row_s = scaler.transform(row_feat)
#     pA = model.predict_proba(row_s)[0,1]
#     return pA


# def simulate_bracket_once(seeds_dict, bracket_slots, model, scaler, men, aggregator_2024):
#     """
#     Fill out bracket topologically from Round1 to Round6, picking winners.
#     Return bracket_winners dict: bracket_winners[slot] = 'W01' or 'W05', etc. 
#     """
#     slot_keys = list(bracket_slots.keys())

#     def slot_sort_key(s):
#         # "R5WX", "R6CH", "R1W1"
#         # we'll parse the round number after 'R'
#         # fallback to very large if not found
#         if s=="R6CH": 
#             return (6,"CH")
#         if s in ["R5WX","R5YZ"]:
#             return (5,s[2:])
#         if s.startswith("R"):
#             # e.g. "R2W1" => round=2
#             rnd = int(s[1])
#             tail= s[2:]  # "W1"
#             return (rnd, tail)
#         # fallback
#         return (99,s)
#     slot_keys.sort(key=slot_sort_key)
#     bracket_winners = {}

#     def occupant_teamid(label):
#         # If label is a base seed like "W01", we fetch seeds_dict. 
#         if label in seeds_dict:
#             return seeds_dict[label]
#         # If label is a slot name, we must see who won that slot
#         if label in bracket_winners:
#             return occupant_teamid(bracket_winners[label])
#         return None

#     for slot in slot_keys:
#         strong, weak = bracket_slots[slot]
#         # occupant labels might be seeds or slots themselves
#         occA = bracket_winners.get(strong, strong)
#         occB = bracket_winners.get(weak, weak)

#         tidA = occupant_teamid(occA)
#         tidB = occupant_teamid(occB)
#         if (tidA is None) or (tidB is None):
#             # fallback
#             bracket_winners[slot] = occA
#         else:
#             pA = predict_game_probability(tidA, tidB, model, scaler, men, aggregator_2024)
#             if np.random.rand() < pA:
#                 bracket_winners[slot] = occA
#             else:
#                 bracket_winners[slot] = occB

#     return bracket_winners


# def build_bracket_rows(bracket_winners, men=True):
#     """
#     Convert the bracket_winners dictionary into lines of the final submission format:
#       RowId, Tournament, Bracket, Slot, Team
#     BUT omit any lines referencing play-in seeds or slots.
#     """
#     # 1) Sort the bracket slots we want to keep
#     #    We only want those that start with 'R' (e.g. R1W1, R1W2, ... R6CH, R5WX, etc.)
#     #    We'll also special-case the R5WX / R5YZ / R6CH naming.
#     def slot_sort_key(s):
#         # This helps us order from Round 1 to Round 6, final
#         if s == "R6CH":
#             return (6, "CH")
#         elif s in ("R5WX", "R5YZ"):
#             return (5, s[2:])  # "WX" or "YZ"
#         elif s.startswith("R"):  # e.g. "R1W1"
#             round_num = int(s[1])  # the digit after R
#             return (round_num, s[2:])
#         else:
#             # For anything that doesn't start with R, we'll put it at the end or skip it
#             return (999, s)

#     # 2) Filter out any bracket slots that do not start with 'R' (like play-in seeds X16, X16a, W16a, etc.)
#     #    or seeds that contain 'a'/'b'.
#     valid_slots = []
#     for slot in bracket_winners.keys():
#         # We'll skip seeds or play-in references if they do NOT start with "R"
#         if slot.startswith("R"):
#             valid_slots.append(slot)
#         # If you want to be even stricter, skip also any reference if '16a' or '16b' is in it:
#         # e.g. if 'a' in slot or 'b' in slot: skip
#         # But typically just skipping non-R is enough if your main bracket is labeled R1..R6.

#     # Sort those valid slots in ascending bracket order
#     valid_slots.sort(key=slot_sort_key)

#     # 3) Prepare the final bracket lines
#     bracket_rows = []
#     tchar = 'M' if men else 'W'
#     row_list = []
#     for slot in valid_slots:
#         pick = bracket_winners[slot]
#         # Also skip if the 'pick' is a play-in seed label (like 'X16', 'W16a', etc.)
#         # We'll define a quick check:
#         # if pick.startswith("W") or pick.startswith("X") or pick.startswith("Y") or pick.startswith("Z"):
#             # example seed might be 'W16' or 'X16a'
#             # if 'a' in pick or 'b' in pick, skip
#             # if 'a' in pick or 'b' in pick:
#             #     if 'a' in pick:
#             #         row_list.append((slot, pick[:-1], tchar))
#             #     continue
#         if pick.endswith('a') or pick.endswith('b'):
#             pick = pick[:-1]
#         bracket_rows.append((slot, pick, tchar))
        
#         # If we reach here, we keep the line
#         row_list.append((slot, pick, tchar))

#     return row_list

# ###############################################################################
# # 4) MAIN
# ###############################################################################

# def main():
#     print("Loading data...")
#     # Adjust data_dir to your location:
#     data_dir = "/kaggle/input/march-machine-learning-mania-2024"  
#     dfs = load_all_data(data_dir)
#     print("Data loaded.")

#     # 4.1) Train men’s model
#     print("Preparing men’s training data (2003–2023) with exponential decay weighting...")
#     dfm = prep_training_data_men(dfs, 2003, 2023, decay_base=0.94)
#     print(f"Men’s training set: {len(dfm)} rows")
#     men_model, men_scaler = train_model_men(dfm)
#     print("Men’s model trained.")

#     # 4.2) Train women’s model
#     print("Preparing women’s training data (2010–2023) with exponential decay weighting...")
#     dfw = prep_training_data_women(dfs, 2010, 2023, decay_base=0.94)
#     print(f"Women’s training set: {len(dfw)} rows")
#     women_model, women_scaler = train_model_women(dfw)
#     print("Women’s model trained.")

#     # 4.3) Load 2024 bracket structure for men + seeds
#     print("Loading Men’s 2024 bracket structure + seeds...")
#     men_seeds_dict, men_bracket_slots = load_slots_and_seeds_2024(dfs, men=True)

#     # 4.4) Load 2024 bracket structure for women + seeds
#     print("Loading Women’s 2024 bracket structure + seeds...")
#     women_seeds_dict, women_bracket_slots = load_slots_and_seeds_2024(dfs, men=False)

#     # 4.5) Build aggregator for partial 2024 data (men + women)
#     men_2024_agg = build_2024_aggregator(dfs, men=True)
#     women_2024_agg = build_2024_aggregator(dfs, men=False)

#     # 4.6) Generate multiple bracket predictions for men + women
#     # For demonstration, let's do 10 each to keep it short.
#     NUM_BRACKETS_MEN = 500
#     NUM_BRACKETS_WOMEN = 500

#     big_rows = []  # list of (RowId, Tournament, Bracket, Slot, Team)
#     rowId = 0

#     # MEN
#     for b_idx in range(1, NUM_BRACKETS_MEN + 1):
#         winners = simulate_bracket_once(men_seeds_dict, men_bracket_slots,
#                                         men_model, men_scaler, True, men_2024_agg)
#         bracket_rows = build_bracket_rows(winners, men=True)
#         for (slot, pick, tchar) in bracket_rows:
#             rowId += 1
#             big_rows.append((rowId, tchar, b_idx, slot, pick))

#     # WOMEN
#     for b_idx in range(NUM_BRACKETS_MEN + 1, NUM_BRACKETS_MEN + NUM_BRACKETS_WOMEN + 1):
#         winners = simulate_bracket_once(women_seeds_dict, women_bracket_slots,
#                                         women_model, women_scaler, False, women_2024_agg)
#         bracket_rows = build_bracket_rows(winners, men=False)
#         for (slot, pick, tchar) in bracket_rows:
#             rowId += 1
#             big_rows.append((rowId, tchar, b_idx, slot, pick))

#     # 4.7) Save EXACT required columns: RowId,Tournament,Bracket,Slot,Team
#     df_sub = pd.DataFrame(big_rows, columns=["RowId","Tournament","Bracket","Slot","Team"])
#     df_sub.to_csv("submission_2.csv", index=False)
#     print(f"Saved {len(df_sub)} rows to submission.csv!")
#     print("Done. You can now submit submission.csv to Kaggle.")


# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python3
# """
# main.py
# Author: YOU
# Date: 2025-01-01

# Improved baseline logistic regression for March Madness 2024, with:
#  1) More features (margin of victory, offensive rebounds, personal fouls, blocks)
#  2) Exponential decay weighting for older seasons
#  3) Precomputed aggregator for 2024 partial data to avoid repeated scanning
#  4) **Now also includes a feature from MMasseyOrdinals_thruSeason2024_day128.csv** 
#     (a simple average ranking from the last available day for each season).
#  5) Correct final bracket output format EXACTLY as required:
#     "RowId,Tournament,Bracket,Slot,Team"

# Usage:
#   python main.py

# Outputs:
#   submission.csv
# """

# import os
# import sys
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler


# ###############################################################################
# # 0) LOAD CSV FILES
# ###############################################################################

# def load_csv_safely(path):
#     """
#     Tries reading CSV with UTF-8 first, then fallback to 'latin1'/cp1252 if needed.
#     Helps fix "UnicodeDecodeError: 'utf-8' codec can't decode byte" issues.
#     """
#     try:
#         return pd.read_csv(path)
#     except UnicodeDecodeError:
#         print(f"Retrying {path} with latin1 encoding...")
#         return pd.read_csv(path, encoding="latin1")

# def load_all_data(data_dir="data"):
#     """
#     Loads all the CSV files we need, returns them as a dict of DataFrames.
#     We skip any missing files but warn.
#     """
#     csv_files = [
#         # Men’s data
#         "MSeasons.csv",
#         "MTeams.csv",
#         "MRegularSeasonCompactResults.csv",
#         "MRegularSeasonDetailedResults.csv",
#         "MNCAATourneyCompactResults.csv",
#         "MNCAATourneyDetailedResults.csv",
#         "MNCAATourneySeeds.csv",
#         "MNCAATourneySlots.csv",
#         # Women’s data
#         "WSeasons.csv",
#         "WTeams.csv",
#         "WRegularSeasonCompactResults.csv",
#         "WRegularSeasonDetailedResults.csv",
#         "WNCAATourneyCompactResults.csv",
#         "WNCAATourneyDetailedResults.csv",
#         "WNCAATourneySeeds.csv",
#         "WNCAATourneySlots.csv",
#         # Shared
#         "Cities.csv",  # men’s
#         "MGameCities.csv",
#         "WGameCities.csv",
#         # Possibly more merges needed:
#         "MTeamCoaches.csv",
#         "MTeamConferences.csv",
#         "Conferences.csv",
#         # Next-year seeds (2024)
#         "2024_tourney_seeds.csv",
#         # The new dataset for men’s Massey Ordinals (ranking)
#         "MMasseyOrdinals_thruSeason2024_day128.csv"
#     ]

#     dfs = {}
#     for f in csv_files:
#         path = os.path.join(data_dir, f)
#         if not os.path.exists(path):
#             print(f"WARNING: {f} not found in {data_dir}!")
#             continue
#         try:
#             df = load_csv_safely(path)
#             dfs[f] = df
#         except Exception as e:
#             print(f"WARNING: error reading {f}: {e}")
#     return dfs


# ###############################################################################
# # 1) MASSEY ORDINALS: BUILD A TEAM-SEASON FEATURE
# ###############################################################################
# def build_massey_feature(dfs, from_season=2003, to_season=2024):
#     """
#     We'll parse the men’s file 'MMasseyOrdinals_thruSeason2024_day128.csv'.
#     The approach:
#       - For each (Season, TeamID), we find all SystemName’s final ranking near day=128 or 133, 
#         then average them into a single 'AvgRank' (the lower => the better).
#       - We'll store that in a dict so we can incorporate it into our training 
#         data. For older seasons, we do an exponential decay if desired, or just keep it straightforward.
#     We'll keep it simple: for each Season/TeamID, we store 
#         'MasseyRank' = average rank among all systems from last day available (RankingDayNum=128 or 133).
#       If missing => fallback to 200. 
#     """
#     filename = "MMasseyOrdinals_thruSeason2024_day128.csv"
#     if filename not in dfs:
#         print("WARNING: MMasseyOrdinals CSV not found, skipping that feature!")
#         return {}
    
#     massey = dfs[filename].copy()
#     # filter to from_season..to_season
#     massey = massey[(massey["Season"]>=from_season) & (massey["Season"]<=to_season)].copy()

#     # We only want final or near-final ranking day => typically day=128 or day=133
#     # We'll pick day=128 or the maximum day for that season
#     # A simpler approach is to just pick rows where RankingDayNum=128 or 133:
#     # but let's just pick day=128 to ensure it's the consistent "pre-tourney" 
#     # as the problem statement suggests day=128 is a fallback if final systems not out by day=133
#     massey = massey[massey["RankingDayNum"]==128].copy()

#     # group by (Season, TeamID), average OrdinalRank across all systems
#     gp = massey.groupby(["Season","TeamID"])["OrdinalRank"].mean().reset_index()
#     gp.rename(columns={"OrdinalRank":"AvgRank"}, inplace=True)

#     # store in a dict: massey_dict[(Season,TeamID)] = rank
#     massey_dict = {}
#     for idx, row in gp.iterrows():
#         season = row["Season"]
#         tid = row["TeamID"]
#         rk = row["AvgRank"]
#         massey_dict[(season,tid)] = rk

#     return massey_dict


# ###############################################################################
# # 2) PREPARE TRAINING DATA WITH EXTRA FEATURES + MASSEY RANK
# ###############################################################################

# def prep_training_data_men(dfs, season_start=2003, season_end=2023, decay_base=0.94):
#     """
#     Prepare men’s training data from MRegularSeasonDetailedResults + MNCAATourneyDetailedResults,
#     incorporate extra features:
#      - margin of victory
#      - offensive rebounds
#      - personal fouls
#      - blocks
#      - **Massey average rank** for each team/season
#     We'll do a logistic regression to estimate p(Team A beats Team B).
#     """
#     needed = [
#         "MRegularSeasonDetailedResults.csv",
#         "MNCAATourneyDetailedResults.csv",
#     ]
#     for nf in needed:
#         if nf not in dfs:
#             raise ValueError(f"Missing {nf} in dfs!")

#     reg = dfs["MRegularSeasonDetailedResults.csv"]
#     tour = dfs["MNCAATourneyDetailedResults.csv"]
#     allgames = pd.concat([reg, tour], ignore_index=True)
#     allgames = allgames[(allgames["Season"] >= season_start) & (allgames["Season"] <= season_end)].copy()

#     # Build massey dictionary
#     massey_dict = build_massey_feature(dfs, from_season=season_start, to_season=2024)

#     def loc_value(loc):
#         if loc=='H':
#             return 1
#         elif loc=='A':
#             return -1
#         else:
#             return 0

#     rows = []
#     for idx, row in allgames.iterrows():
#         season = row["Season"]
#         wtid   = row["WTeamID"]
#         ltid   = row["LTeamID"]
#         wscore = row["WScore"]
#         lscore = row["LScore"]
#         margin = wscore - lscore
#         wloc   = loc_value(row.get("WLoc","N"))
#         wor    = row.get("WOR", 0)
#         lor    = row.get("LOR", 0)
#         wpf    = row.get("WPF", 0)
#         lpf    = row.get("LPF", 0)
#         wblk   = row.get("WBlk",0)
#         lblk   = row.get("LBlk",0)

#         # Decay weight factor
#         years_ago = (season_end - season)
#         weight = (decay_base**years_ago)

#         # massey ranks
#         w_massey = massey_dict.get((season,wtid), 200.0)
#         l_massey = massey_dict.get((season,ltid), 200.0)

#         # winning side
#         rows.append({
#             "Season": season,
#             "TeamA": wtid,
#             "TeamB": ltid,
#             "Margin": margin,
#             "Loc": wloc,
#             "OR_diff": (wor - lor),
#             "PF_diff": (wpf - lpf),
#             "Blk_diff":(wblk - lblk),
#             "A_massey": w_massey,
#             "B_massey": l_massey,
#             "Result": 1,
#             "Weight": weight
#         })
#         # losing side
#         rows.append({
#             "Season": season,
#             "TeamA": ltid,
#             "TeamB": wtid,
#             "Margin": -margin,
#             "Loc": -wloc,
#             "OR_diff": (lor - wor),
#             "PF_diff": (lpf - wpf),
#             "Blk_diff": (lblk - wblk),
#             "A_massey": l_massey,
#             "B_massey": w_massey,
#             "Result": 0,
#             "Weight": weight
#         })

#     df_model = pd.DataFrame(rows)
#     return df_model


# def train_model_men(df_model):
#     """
#     Train logistic regression on men’s data with features:
#       [Margin, Loc, OR_diff, PF_diff, Blk_diff, (A_massey - B_massey)]
#     Weighted by 'df_model["Weight"]'.
#     """
#     # We'll define a new column: masseyDiff = A_massey - B_massey
#     df_model["masseyDiff"] = df_model["A_massey"] - df_model["B_massey"]
#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","masseyDiff"]
#     X = df_model[feats].values
#     y = df_model["Result"].values
#     w = df_model["Weight"].values

#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     model = LogisticRegression(
#         random_state=42,
#         max_iter=600,
#     )
#     model.fit(Xs, y, sample_weight=w)
#     return model, scaler


# def prep_training_data_women(dfs, season_start=2010, season_end=2023, decay_base=0.94):
#     """
#     Similar approach for women's data, but we skip the Massey feature, 
#     because the provided 'MMasseyOrdinals_thruSeason2024_day128.csv' is for men’s teams only.
#     (If you had women's ordinal data, you'd do similarly. But we skip it here.)
#     """
#     needed = [
#         "WRegularSeasonDetailedResults.csv",
#         "WNCAATourneyDetailedResults.csv",
#     ]
#     for nf in needed:
#         if nf not in dfs:
#             raise ValueError(f"Missing {nf} in dfs!")
#     reg = dfs["WRegularSeasonDetailedResults.csv"]
#     tour= dfs["WNCAATourneyDetailedResults.csv"]
#     allgames = pd.concat([reg, tour], ignore_index=True)
#     allgames= allgames[(allgames["Season"]>=season_start)&(allgames["Season"]<=season_end)].copy()

#     def loc_value(loc):
#         if loc=='H':
#             return 1
#         elif loc=='A':
#             return -1
#         else:
#             return 0

#     rows=[]
#     for idx, row in allgames.iterrows():
#         season= row["Season"]
#         wtid  = row["WTeamID"]
#         ltid  = row["LTeamID"]
#         wscore= row["WScore"]
#         lscore= row["LScore"]
#         margin= wscore-lscore
#         wloc  = loc_value(row.get("WLoc","N"))
#         wor   = row.get("WOR",0)
#         lor   = row.get("LOR",0)
#         wpf   = row.get("WPF",0)
#         lpf   = row.get("LPF",0)
#         wblk  = row.get("WBlk",0)
#         lblk  = row.get("LBlk",0)

#         years_ago= (season_end - season)
#         weight   = (decay_base**years_ago)

#         rows.append({
#             "Season": season,
#             "TeamA": wtid,
#             "TeamB": ltid,
#             "Margin": margin,
#             "Loc": wloc,
#             "OR_diff":(wor-lor),
#             "PF_diff":(wpf-lpf),
#             "Blk_diff":(wblk-lblk),
#             "Result":1,
#             "Weight":weight
#         })
#         rows.append({
#             "Season": season,
#             "TeamA": ltid,
#             "TeamB": wtid,
#             "Margin": -margin,
#             "Loc": -wloc,
#             "OR_diff":(lor-wor),
#             "PF_diff":(lpf-wpf),
#             "Blk_diff":(lblk-wblk),
#             "Result":0,
#             "Weight":weight
#         })

#     df_model=pd.DataFrame(rows)
#     return df_model


# def train_model_women(df_model):
#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff"]
#     X= df_model[feats].values
#     y= df_model["Result"].values
#     w= df_model["Weight"].values

#     scaler= StandardScaler()
#     Xs= scaler.fit_transform(X)
#     model= LogisticRegression(random_state=43, max_iter=600)
#     model.fit(Xs, y, sample_weight=w)
#     return model, scaler


# ###############################################################################
# # 2) LOAD 2024 SLOTS/SEEDS AND PRECOMPUTE 2024 TEAM STATS
# ###############################################################################

# def load_slots_and_seeds_2024(dfs, men=True):
#     """
#     For men:  we read only rows where Tournament='M' from 2024_tourney_seeds.csv,
#               bracket from MNCAATourneySlots.csv Season=2024
#     For women: we read rows where Tournament='W',
#                bracket from WNCAATourneySlots.csv Season=2024
#     """
#     if "2024_tourney_seeds.csv" not in dfs:
#         raise ValueError("Missing 2024_tourney_seeds.csv in DFS!")
#     df_seeds = dfs["2024_tourney_seeds.csv"]

#     if men:
#         df_seeds= df_seeds[df_seeds["Tournament"]=="M"].copy()
#         slot_file= "MNCAATourneySlots.csv"
#         if slot_file not in dfs:
#             raise ValueError("No MNCAATourneySlots.csv for men!")
#         df_slots= dfs[slot_file]
#         df_slots= df_slots[df_slots["Season"]==2024].copy()
#     else:
#         df_seeds= df_seeds[df_seeds["Tournament"]=="W"].copy()
#         slot_file= "WNCAATourneySlots.csv"
#         if slot_file not in dfs:
#             raise ValueError("No WNCAATourneySlots.csv for women!")
#         df_slots= dfs[slot_file]
#         df_slots= df_slots[df_slots["Season"]==2024].copy()

#     seeds_dict={}
#     for idx,row in df_seeds.iterrows():
#         seed=row["Seed"] # e.g. "W01"
#         tid= row["TeamID"]
#         seeds_dict[seed]=tid

#     bracket_slots={}
#     for idx,row in df_slots.iterrows():
#         slot= row["Slot"]
#         strong=row["StrongSeed"]
#         weak=  row["WeakSeed"]
#         bracket_slots[slot]=(strong, weak)

#     return seeds_dict, bracket_slots


# def build_2024_aggregator(dfs, men=True):
#     """
#     Read the 2024 M/WRegularSeasonDetailedResults once, build aggregator: 
#     aggregator[teamId] = { 'margin':..., 'or':..., 'pf':..., 'blk':..., 'games':... }
#     So we can quickly compute an average feature for 2024 partial data.
#     """
#     if men:
#         df_reg2024 = dfs.get("MRegularSeasonDetailedResults.csv", None)
#     else:
#         df_reg2024 = dfs.get("WRegularSeasonDetailedResults.csv", None)

#     if df_reg2024 is None:
#         return {}
#     df_reg2024 = df_reg2024[df_reg2024["Season"] == 2024].copy()
#     if len(df_reg2024) == 0:
#         return {}

#     stats={}
#     for idx, row in df_reg2024.iterrows():
#         wtid= row["WTeamID"]
#         ltid= row["LTeamID"]
#         margin= row["WScore"]-row["LScore"]
#         wor= row.get("WOR",0)
#         lor= row.get("LOR",0)
#         wpf= row.get("WPF",0)
#         lpf= row.get("LPF",0)
#         wblk= row.get("WBlk",0)
#         lblk= row.get("LBlk",0)

#         if wtid not in stats:
#             stats[wtid]={"margin":0,"or":0,"pf":0,"blk":0,"games":0}
#         stats[wtid]["margin"]+= margin
#         stats[wtid]["or"]    += wor
#         stats[wtid]["pf"]    += wpf
#         stats[wtid]["blk"]   += wblk
#         stats[wtid]["games"] += 1

#         if ltid not in stats:
#             stats[ltid]={"margin":0,"or":0,"pf":0,"blk":0,"games":0}
#         stats[ltid]["margin"]+= -margin
#         stats[ltid]["or"]    += lor
#         stats[ltid]["pf"]    += lpf
#         stats[ltid]["blk"]   += lblk
#         stats[ltid]["games"] += 1

#     return stats

# def get_2024_team_averages(teamId, aggregator):
#     """
#     aggregator => aggregator[teamId] = dict(margin=..., or=..., pf=..., blk=..., games=...)
#     returns tuple: (avg_margin, avg_or, avg_pf, avg_blk)
#     fallback 0 if no data
#     """
#     if (teamId not in aggregator) or (aggregator[teamId]["games"]==0):
#         return (0.0,0.0,0.0,0.0)
#     st= aggregator[teamId]
#     g= st["games"]
#     return (
#         st["margin"]/g,
#         st["or"]/g,
#         st["pf"]/g,
#         st["blk"]/g
#     )


# ###############################################################################
# # 3) GAME PREDICTION & BRACKET SIMULATION
# ###############################################################################


# def predict_game_probability(teamA, teamB, model, scaler, men, aggregator_2024):
#     """
#     Predict game probability for teamA beating teamB using logistic regression.
#     Fixes inconsistent feature length issue between training and prediction.
#     """
#     if not aggregator_2024:
#         # Ensure feature consistency during both training and prediction
#         if men:
#             feats = np.array([[0, 0, 0, 0, 0, 0]], dtype=float)  # 6 features for men
#         else:
#             feats = np.array([[0, 0, 0, 0, 0]], dtype=float)  # 5 features for women
#         feats_s = scaler.transform(feats)
#         return model.predict_proba(feats_s)[0, 1]

#     # Compute average stats for both teams using the partial 2024 data
#     a_margin, a_or, a_pf, a_blk = get_2024_team_averages(teamA, aggregator_2024)
#     b_margin, b_or, b_pf, b_blk = get_2024_team_averages(teamB, aggregator_2024)

#     # Compute feature differences for prediction
#     margin_diff = a_margin - b_margin
#     or_diff = a_or - b_or
#     pf_diff = a_pf - b_pf
#     blk_diff = a_blk - b_blk
#     loc = 0.0  # Neutral venue assumed

#     # **Feature Count Consistency Fix**
#     if men:
#         masseyDiff = 0.0
#         row_feat = [[margin_diff, loc, or_diff, pf_diff, blk_diff, masseyDiff]]  # 6 features for men
#     else:
#         row_feat = [[margin_diff, loc, or_diff, pf_diff, blk_diff]]  # 5 features for women

#     # Transform features with scaler and predict
#     row_s = scaler.transform(row_feat)
#     pA = model.predict_proba(row_s)[0, 1]
#     return pA



# def simulate_bracket_once(seeds_dict, bracket_slots, model, scaler, men, aggregator_2024):
#     """
#     Fill out bracket from Round1..Round6. Return bracket_winners dict.
#     """
#     slot_keys= list(bracket_slots.keys())
#     def slot_sort_key(s):
#         if s=="R6CH": return (6,"CH")
#         if s in ("R5WX","R5YZ"): return (5,s[2:])
#         if s.startswith("R"):
#             rnd= int(s[1])
#             tail= s[2:]
#             return (rnd,tail)
#         return (99,s)
#     slot_keys.sort(key=slot_sort_key)
#     bracket_winners={}

#     def occupant_teamid(label):
#         # seeds or previous slot winner
#         if label in seeds_dict:
#             return seeds_dict[label]
#         if label in bracket_winners:
#             return occupant_teamid(bracket_winners[label])
#         return None

#     for slot in slot_keys:
#         strong, weak= bracket_slots[slot]
#         occA= bracket_winners.get(strong, strong)
#         occB= bracket_winners.get(weak, weak)
#         tidA= occupant_teamid(occA)
#         tidB= occupant_teamid(occB)
#         if (tidA is None) or (tidB is None):
#             bracket_winners[slot]= occA
#         else:
#             pA= predict_game_probability(tidA, tidB, model, scaler, men, aggregator_2024)
#             if np.random.rand()< pA:
#                 bracket_winners[slot]= occA
#             else:
#                 bracket_winners[slot]= occB

#     return bracket_winners

# def build_bracket_rows(bracket_winners, men=True):
#     """
#     Convert bracket_winners to lines in final submission format, skipping any "play-in" seeds.
#     Sort by round => R1..R6, skipping seeds not starting with 'R'.
#     """
#     def slot_sort_key(s):
#         if s=="R6CH": return (6,"CH")
#         if s in ("R5WX","R5YZ"): return (5,s[2:])
#         if s.startswith("R"):
#             r= int(s[1])
#             tail= s[2:]
#             return (r,tail)
#         return (999,s)

#     # filter
#     valid_slots= []
#     for slot in bracket_winners.keys():
#         if slot.startswith("R"): 
#             valid_slots.append(slot)
#     valid_slots.sort(key=slot_sort_key)

#     bracket_rows=[]
#     tchar= 'M' if men else 'W'
#     for slot in valid_slots:
#         pick= bracket_winners[slot]
#         # skip if pick has 'a' or 'b'
#         # if 'a' in pick or 'b' in pick:
#         #     continue
#         # bracket_rows.append((slot,pick,tchar))
#         if pick.endswith('a') or pick.endswith('b'):
#             pick = pick[:-1]
#         bracket_rows.append((slot, pick, tchar))
#     return bracket_rows

# ###############################################################################
# # 4) MAIN
# ###############################################################################

# def main():
#     print("Loading data...")
#     data_dir= "/kaggle/input/march-machine-learning-mania-2024"
#     dfs= load_all_data(data_dir)
#     print("Data loaded.")

#     # 4.1) Train men’s model with Massey
#     print("Preparing men’s data (2003–2023) + massey ranks + exponential weighting...")
#     dfm= prep_training_data_men(dfs, season_start=2003, season_end=2023, decay_base=0.94)
#     print(f"Men’s training set: {len(dfm)} rows")
#     men_model, men_scaler= train_model_men(dfm)
#     print("Men’s model trained (with Massey rank).")

#     # 4.2) Train women’s model
#     print("Preparing women’s data (2010–2023) with exponential weighting (no massey data).")
#     dfw= prep_training_data_women(dfs, 2010, 2023, 0.94)
#     print(f"Women’s training set: {len(dfw)} rows")
#     women_model, women_scaler= train_model_women(dfw)
#     print("Women’s model trained.")

#     # 4.3) Load 2024 bracket structure for men + seeds
#     men_seeds_dict, men_bracket_slots= load_slots_and_seeds_2024(dfs, men=True)

#     # 4.4) Load 2024 bracket structure for women + seeds
#     women_seeds_dict, women_bracket_slots= load_slots_and_seeds_2024(dfs, men=False)

#     # 4.5) Build aggregator for partial 2024 data
#     men_2024_agg  = build_2024_aggregator(dfs, men=True)
#     women_2024_agg= build_2024_aggregator(dfs, men=False)

#     # We'll produce, for example, 300 men’s brackets & 300 women’s
#     N_M= 500
#     N_W= 500
#     big_rows=[]
#     rowId= 0

#     # Men
#     for b_idx in range(1, N_M+1):
#         winners= simulate_bracket_once(men_seeds_dict, men_bracket_slots, men_model, men_scaler, True, men_2024_agg)
#         bracket_rows= build_bracket_rows(winners, men=True)
#         for (slot, pick, tch) in bracket_rows:
#             rowId+=1
#             big_rows.append((rowId, tch, b_idx, slot, pick))

#     # Women
#     for b_idx in range(N_M+1, N_M+N_W+1):
#         winners= simulate_bracket_once(women_seeds_dict, women_bracket_slots, women_model, women_scaler, False, women_2024_agg)
#         bracket_rows= build_bracket_rows(winners, men=False)
#         for (slot, pick, tch) in bracket_rows:
#             rowId+=1
#             big_rows.append((rowId, tch, b_idx, slot, pick))

#     # final output
#     df_sub= pd.DataFrame(big_rows, columns=["RowId","Tournament","Bracket","Slot","Team"])
#     df_sub.to_csv("submission.csv", index=False)
#     print(f"Wrote {len(df_sub)} lines to submission.csv in the corrected format!")
#     print("Done.")

# if __name__=="__main__":
#     main()






# #!/usr/bin/env python3
# """
# main.py
# Author: YOU
# Date: 2025-01-01

# Improved baseline logistic regression for March Madness 2024, with:
#  1) More features (margin of victory, offensive rebounds, personal fouls, blocks)
#  2) Exponential decay weighting for older seasons
#  3) Precomputed aggregator for 2024 partial data to avoid repeated scanning
#  4) Correct final bracket output format EXACTLY as required: "RowId,Tournament,Bracket,Slot,Team"
#  5) Additional optional evaluation step: time-based split + log loss
#  6) [ADDED FOR MASSEY] Incorporation of MMasseyOrdinals_thruSeason2024_day128.csv (men's rank data)
#  7) [CHANGED FOR FIX] Women's pipeline also has "Rank_diff"=0 so both models have 6 features.
# """

# import os
# import sys
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import log_loss

# ###############################################################################
# # 0) LOAD CSV FILES
# ###############################################################################

# def load_csv_safely(path):
#     """
#     Tries reading CSV with UTF-8 first, then fallback to 'latin1'/cp1252 if needed.
#     Helps fix "UnicodeDecodeError: 'utf-8' codec can't decode byte" issues.
#     """
#     try:
#         return pd.read_csv(path)
#     except UnicodeDecodeError:
#         print(f"Retrying {path} with latin1 encoding...")
#         return pd.read_csv(path, encoding="latin1")


# def load_all_data(data_dir="data"):
#     """
#     Loads all the CSV files we need, returns them as a dict of DataFrames.
#     We skip any missing files but warn.
#     """
#     csv_files = [
#         # Men’s data
#         "MSeasons.csv",
#         "MTeams.csv",
#         "MRegularSeasonCompactResults.csv",
#         "MRegularSeasonDetailedResults.csv",
#         "MNCAATourneyCompactResults.csv",
#         "MNCAATourneyDetailedResults.csv",
#         "MNCAATourneySeeds.csv",
#         "MNCAATourneySlots.csv",
#         # Women’s data
#         "WSeasons.csv",
#         "WTeams.csv",
#         "WRegularSeasonCompactResults.csv",
#         "WRegularSeasonDetailedResults.csv",
#         "WNCAATourneyCompactResults.csv",
#         "WNCAATourneyDetailedResults.csv",
#         "WNCAATourneySeeds.csv",
#         "WNCAATourneySlots.csv",
#         # Shared
#         "Cities.csv",
#         "MGameCities.csv",
#         "WGameCities.csv",
#         # Possibly more merges needed:
#         "MTeamCoaches.csv",
#         "MTeamConferences.csv",
#         "Conferences.csv",
#         # Next-year seeds (2024)
#         "2024_tourney_seeds.csv",
#         # For Massey (men's only)
#         "MMasseyOrdinals_thruSeason2024_day128.csv"
#     ]

#     dfs = {}
#     for f in csv_files:
#         path = os.path.join(data_dir, f)
#         if not os.path.exists(path):
#             print(f"WARNING: {f} not found in {data_dir}!")
#             continue
#         try:
#             df = load_csv_safely(path)
#             dfs[f] = df
#         except Exception as e:
#             print(f"WARNING: error reading {f}: {e}")
#     return dfs


# ###############################################################################
# # PARSE MEN'S MASSEY ORDINALS
# ###############################################################################

# def parse_mens_massey_ranks(dfs):
#     """
#     From 'MMasseyOrdinals_thruSeason2024_day128.csv', create:
#         mens_massey[(season, teamID)] = avg_ordinal_rank
#     filtering to RankingDayNum <=128, then averaging OrdinalRank.
#     """
#     file_key = "MMasseyOrdinals_thruSeason2024_day128.csv"
#     if file_key not in dfs:
#         print("WARNING: No Massey Ordinals CSV found. We'll skip it.")
#         return {}

#     df_m = dfs[file_key].copy()
#     df_m = df_m[df_m["RankingDayNum"] <= 128]

#     grouped = df_m.groupby(["Season","TeamID"])["OrdinalRank"].mean().reset_index()

#     massey_dict = {}
#     for _, row in grouped.iterrows():
#         season = row["Season"]
#         tid    = row["TeamID"]
#         avgRank= row["OrdinalRank"]
#         massey_dict[(season, tid)] = avgRank
#     return massey_dict

# ###############################################################################
# # 1) PREPARE TRAINING DATA + DECAY WEIGHT
# ###############################################################################

# def prep_training_data_men(dfs, mens_massey, season_start=2003, season_end=2023, decay_base=0.94):
#     """
#     Men’s training with (Margin, Loc, OR_diff, PF_diff, Blk_diff, Rank_diff).
#     """
#     needed = [
#         "MRegularSeasonDetailedResults.csv",
#         "MNCAATourneyDetailedResults.csv",
#     ]
#     for nf in needed:
#         if nf not in dfs:
#             raise ValueError(f"Missing {nf} in dfs!")

#     reg = dfs["MRegularSeasonDetailedResults.csv"]
#     tour = dfs["MNCAATourneyDetailedResults.csv"]
#     allgames = pd.concat([reg, tour], ignore_index=True)
#     allgames = allgames[(allgames["Season"] >= season_start) & (allgames["Season"] <= season_end)].copy()

#     def loc_value(loc):
#         if loc == 'H':
#             return 1
#         elif loc == 'A':
#             return -1
#         else:
#             return 0

#     rows = []
#     for _, row in allgames.iterrows():
#         season = row["Season"]
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         margin = row["WScore"] - row["LScore"]
#         wloc = loc_value(row.get("WLoc", "N"))
#         wor = row.get("WOR", 0)
#         lor = row.get("LOR", 0)
#         wpf = row.get("WPF", 0)
#         lpf = row.get("LPF", 0)
#         wblk= row.get("WBlk",0)
#         lblk= row.get("LBlk",0)

#         # Exponential decay weighting
#         years_ago = season_end - season
#         weight = (decay_base ** years_ago)

#         # Rank from men’s_massey
#         rankW = mens_massey.get((season, wtid), 0.0)
#         rankL = mens_massey.get((season, ltid), 0.0)

#         # Winner side
#         rows.append({
#             "Season": season,
#             "TeamA": wtid,
#             "TeamB": ltid,
#             "Margin": margin,
#             "Loc": wloc,
#             "OR_diff": (wor - lor),
#             "PF_diff": (wpf - lpf),
#             "Blk_diff": (wblk - lblk),
#             "Rank_diff": (rankW - rankL),
#             "Result": 1,
#             "Weight": weight
#         })
#         # Loser side
#         rows.append({
#             "Season": season,
#             "TeamA": ltid,
#             "TeamB": wtid,
#             "Margin": -margin,
#             "Loc": -wloc,
#             "OR_diff": (lor - wor),
#             "PF_diff": (lpf - wpf),
#             "Blk_diff": (lblk - wblk),
#             "Rank_diff": (rankL - rankW),
#             "Result": 0,
#             "Weight": weight
#         })

#     return pd.DataFrame(rows)


# def train_model_men(df_model):
#     """
#     6 features: [Margin, Loc, OR_diff, PF_diff, Blk_diff, Rank_diff].
#     """
#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
#     X = df_model[feats].values
#     y = df_model["Result"].values
#     w = df_model["Weight"].values

#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     model = LogisticRegression(random_state=42, max_iter=600)
#     model.fit(Xs, y, sample_weight=w)
#     return model, scaler


# # (CHANGED FOR FIX) - Make women's code also produce "Rank_diff"=0
# def prep_training_data_women(dfs, season_start=2010, season_end=2023, decay_base=0.94):
#     """
#     Women’s training data with same 6 columns: 
#     (Margin, Loc, OR_diff, PF_diff, Blk_diff, Rank_diff=0).
#     """
#     needed = [
#         "WRegularSeasonDetailedResults.csv",
#         "WNCAATourneyDetailedResults.csv",
#     ]
#     for nf in needed:
#         if nf not in dfs:
#             raise ValueError(f"Missing {nf} in dfs!")

#     reg = dfs["WRegularSeasonDetailedResults.csv"]
#     tour = dfs["WNCAATourneyDetailedResults.csv"]
#     allgames = pd.concat([reg, tour], ignore_index=True)
#     allgames = allgames[(allgames["Season"]>=season_start) & (allgames["Season"]<=season_end)].copy()

#     def loc_value(loc):
#         if loc == 'H':
#             return 1
#         elif loc == 'A':
#             return -1
#         else:
#             return 0

#     rows = []
#     for _, row in allgames.iterrows():
#         season = row["Season"]
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         margin = row["WScore"] - row["LScore"]
#         wloc = loc_value(row.get("WLoc","N"))
#         wor = row.get("WOR",0)
#         lor = row.get("LOR",0)
#         wpf = row.get("WPF",0)
#         lpf = row.get("LPF",0)
#         wblk= row.get("WBlk",0)
#         lblk= row.get("LBlk",0)

#         years_ago = season_end - season
#         weight = (decay_base**years_ago)

#         # (CHANGED FOR FIX) "Rank_diff"=0 for women's data
#         rows.append({
#             "Season": season,
#             "TeamA": wtid,
#             "TeamB": ltid,
#             "Margin": margin,
#             "Loc": wloc,
#             "OR_diff": (wor - lor),
#             "PF_diff": (wpf - lpf),
#             "Blk_diff": (wblk - lblk),
#             "Rank_diff": 0.0,  # always zero for women's data
#             "Result": 1,
#             "Weight": weight
#         })
#         rows.append({
#             "Season": season,
#             "TeamA": ltid,
#             "TeamB": wtid,
#             "Margin": -margin,
#             "Loc": -wloc,
#             "OR_diff": (lor - wor),
#             "PF_diff": (lpf - wpf),
#             "Blk_diff": (lblk - wblk),
#             "Rank_diff": 0.0,  # always zero
#             "Result": 0,
#             "Weight": weight
#         })

#     return pd.DataFrame(rows)


# def train_model_women(df_model):
#     """
#     Women’s also use 6 features, but 'Rank_diff' is always 0 in training.
#     """
#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
#     X = df_model[feats].values
#     y = df_model["Result"].values
#     w = df_model["Weight"].values

#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)
#     model = LogisticRegression(random_state=43, max_iter=600)
#     model.fit(Xs, y, sample_weight=w)
#     return model, scaler


# ###############################################################################
# # 1b) OPTIONAL EVALUATION: SPLIT & LOG LOSS
# ###############################################################################

# def evaluate_model_time_split_men(df_all, year_split=2022):
#     """
#     Time-based split for men’s dataset:
#      - train on seasons <= year_split
#      - validate on seasons > year_split
#      measure log loss on validation
#     """
#     df_train = df_all[df_all["Season"] <= year_split]
#     df_val   = df_all[df_all["Season"] >  year_split]

#     if len(df_train)==0 or len(df_val)==0:
#         print("WARNING: Not enough data for train/val split; skipping evaluation.")
#         return

#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
#     X_train = df_train[feats].values
#     y_train = df_train["Result"].values
#     w_train = df_train["Weight"].values

#     scaler = StandardScaler()
#     Xs_train = scaler.fit_transform(X_train)

#     model = LogisticRegression(random_state=42, max_iter=600)
#     model.fit(Xs_train, y_train, sample_weight=w_train)

#     X_val  = df_val[feats].values
#     y_val  = df_val["Result"].values
#     Xs_val = scaler.transform(X_val)
#     preds  = model.predict_proba(Xs_val)[:,1]
#     ll = log_loss(y_val, preds)
#     print(f"Men’s validation log loss (<= {year_split} vs. > {year_split}): {ll:.5f}")


# ###############################################################################
# # 2) LOAD 2024 SLOTS/SEEDS + PARTIAL 2024 AGGREGATOR
# ###############################################################################

# def load_slots_and_seeds_2024(dfs, men=True):
#     """
#     For men: read '2024_tourney_seeds.csv' => seeds with 'M'
#              'MNCAATourneySlots.csv' => bracket
#     For women: read '2024_tourney_seeds.csv' => seeds with 'W'
#                'WNCAATourneySlots.csv' => bracket
#     """
#     if "2024_tourney_seeds.csv" not in dfs:
#         raise ValueError("Missing 2024_tourney_seeds.csv!")
#     df_seeds = dfs["2024_tourney_seeds.csv"]

#     if men:
#         df_seeds = df_seeds[df_seeds["Tournament"] == "M"].copy()
#         slot_file = "MNCAATourneySlots.csv"
#         if slot_file not in dfs:
#             raise ValueError("No MNCAATourneySlots.csv for men!")
#         df_slots = dfs[slot_file]
#         df_slots = df_slots[df_slots["Season"]==2024].copy()
#     else:
#         df_seeds = df_seeds[df_seeds["Tournament"] == "W"].copy()
#         slot_file = "WNCAATourneySlots.csv"
#         if slot_file not in dfs:
#             raise ValueError("No WNCAATourneySlots.csv for women!")
#         df_slots = dfs[slot_file]
#         df_slots = df_slots[df_slots["Season"]==2024].copy()

#     seeds_dict = {}
#     for _, row in df_seeds.iterrows():
#         seed = row["Seed"]
#         tid  = row["TeamID"]
#         seeds_dict[seed] = tid

#     bracket_slots = {}
#     for _, row in df_slots.iterrows():
#         slot = row["Slot"]
#         bracket_slots[slot] = (row["StrongSeed"], row["WeakSeed"])

#     return seeds_dict, bracket_slots


# def build_2024_aggregator(dfs, men=True):
#     """
#     aggregator[teamId] = { margin, or, pf, blk, games }
#     from the 2024 partial results
#     """
#     if men:
#         df_reg2024 = dfs.get("MRegularSeasonDetailedResults.csv", None)
#     else:
#         df_reg2024 = dfs.get("WRegularSeasonDetailedResults.csv", None)

#     if df_reg2024 is None:
#         return {}
#     df_reg2024 = df_reg2024[df_reg2024["Season"]==2024].copy()
#     if len(df_reg2024)==0:
#         return {}

#     stats={}
#     for _, row in df_reg2024.iterrows():
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         margin = row["WScore"] - row["LScore"]
#         wor = row.get("WOR",0)
#         lor = row.get("LOR",0)
#         wpf = row.get("WPF",0)
#         lpf = row.get("LPF",0)
#         wblk= row.get("WBlk",0)
#         lblk= row.get("LBlk",0)

#         if wtid not in stats:
#             stats[wtid] = {"margin":0,"or":0,"pf":0,"blk":0,"games":0}
#         if ltid not in stats:
#             stats[ltid] = {"margin":0,"or":0,"pf":0,"blk":0,"games":0}

#         stats[wtid]["margin"] += margin
#         stats[wtid]["or"]     += wor
#         stats[wtid]["pf"]     += wpf
#         stats[wtid]["blk"]    += wblk
#         stats[wtid]["games"]  += 1

#         stats[ltid]["margin"] -= margin
#         stats[ltid]["or"]     += lor
#         stats[ltid]["pf"]     += lpf
#         stats[ltid]["blk"]    += lblk
#         stats[ltid]["games"]  += 1

#     return stats


# def get_2024_team_averages(teamId, aggregator):
#     """
#     aggregator[teamId] => dict(margin=..., or=..., pf=..., blk=..., games=...)
#     returns (avg_margin, avg_or, avg_pf, avg_blk) or (0,0,0,0)
#     """
#     if (teamId not in aggregator) or (aggregator[teamId]["games"]==0):
#         return (0,0,0,0)
#     st = aggregator[teamId]
#     g  = st["games"]
#     return (
#         st["margin"]/g,
#         st["or"]/g,
#         st["pf"]/g,
#         st["blk"]/g
#     )


# ###############################################################################
# # 3) PREDICT GAME PROBABILITY + BRACKET SIMULATION
# ###############################################################################

# def predict_game_probability(teamA, teamB, model, scaler, men, aggregator_2024):
#     """
#     Both men/women are now trained on 6 features. For 2024 aggregator, rank_diff=0
#     """
#     # aggregator => only margin, or, pf, blk. We also have 'loc=0' => neutral, rank_diff=0
#     a_margin, a_or, a_pf, a_blk = get_2024_team_averages(teamA, aggregator_2024)
#     b_margin, b_or, b_pf, b_blk = get_2024_team_averages(teamB, aggregator_2024)

#     margin_diff = a_margin - b_margin
#     or_diff = a_or - b_or
#     pf_diff = a_pf - b_pf
#     blk_diff = a_blk - b_blk
#     loc = 0.0
#     rank_diff = 0.0  # no rank for partial 2024 aggregator

#     # Now we have 6 features
#     row_feat = np.array([[margin_diff, loc, or_diff, pf_diff, blk_diff, rank_diff]], dtype=float)
#     row_s    = scaler.transform(row_feat)
#     pA       = model.predict_proba(row_s)[0,1]
#     return pA


# def simulate_bracket_once(seeds_dict, bracket_slots, model, scaler, men, aggregator_2024):
#     """
#     Fill out bracket from Round1..Round6, picking winners by predicted probability.
#     Return bracket_winners: slot->winning seed label
#     """
#     slot_keys = list(bracket_slots.keys())

#     def slot_sort_key(s):
#         if s=="R6CH":
#             return (6,"CH")
#         elif s in ["R5WX","R5YZ"]:
#             return (5,s[2:])
#         elif s.startswith("R"):
#             rnd = int(s[1])
#             tail= s[2:]
#             return (rnd, tail)
#         else:
#             return (99, s)

#     slot_keys.sort(key=slot_sort_key)
#     bracket_winners = {}

#     def occupant_teamid(label):
#         if label in seeds_dict:
#             return seeds_dict[label]
#         if label in bracket_winners:
#             return occupant_teamid(bracket_winners[label])
#         return None

#     for slot in slot_keys:
#         strong, weak = bracket_slots[slot]
#         occA = bracket_winners.get(strong, strong)
#         occB = bracket_winners.get(weak, weak)

#         tidA = occupant_teamid(occA)
#         tidB = occupant_teamid(occB)
#         if (tidA is None) or (tidB is None):
#             bracket_winners[slot] = occA
#         else:
#             pA = predict_game_probability(tidA, tidB, model, scaler, men, aggregator_2024)
#             if np.random.rand() < pA:
#                 bracket_winners[slot] = occA
#             else:
#                 bracket_winners[slot] = occB

#     return bracket_winners


# def build_bracket_rows(bracket_winners, men=True):
#     """
#     Convert bracket_winners => (RowId, Tournament, Bracket, Slot, Team)
#     skipping non-"R" slots (like play-ins).
#     """
#     def slot_sort_key(s):
#         if s=="R6CH":
#             return (6,"CH")
#         if s in ["R5WX","R5YZ"]:
#             return (5,s[2:])
#         if s.startswith("R"):
#             rnd=int(s[1])
#             tail=s[2:]
#             return (rnd, tail)
#         return (999,s)

#     valid_slots = []
#     for slot in bracket_winners.keys():
#         if slot.startswith("R"):
#             valid_slots.append(slot)
#     valid_slots.sort(key=slot_sort_key)

#     tchar = 'M' if men else 'W'
#     bracket_rows = []
#     for slot in valid_slots:
#         pick = bracket_winners[slot]
#         # strip a/b suffix if exist
#         if pick.endswith('a') or pick.endswith('b'):
#             pick = pick[:-1]
#         bracket_rows.append((slot, pick, tchar))
#     return bracket_rows


# ###############################################################################
# # 4) MAIN
# ###############################################################################

# def main():
#     print("Loading data...")
#     data_dir = "/kaggle/input/march-machine-learning-mania-2024"  # or your local path
#     dfs = load_all_data(data_dir)
#     print("Data loaded.")

#     # Parse men’s massey
#     mens_massey = parse_mens_massey_ranks(dfs)

#     # 1) Men’s data
#     print("Preparing men’s training data (2003–2023) w/ exponent. decay + Rank_diff...")
#     dfm = prep_training_data_men(dfs, mens_massey, season_start=2003, season_end=2023, decay_base=0.94)
#     print(f"Men’s training set: {len(dfm)} rows")
#     evaluate_model_time_split_men(dfm, year_split=2022)
#     men_model, men_scaler = train_model_men(dfm)
#     print("Men’s final model trained (6 features).")

#     # 2) Women’s data
#     print("Preparing women’s training data (2010–2023) w/ exponent. decay + Rank_diff=0...")
#     dfw = prep_training_data_women(dfs, 2010, 2023, decay_base=0.94)
#     print(f"Women’s training set: {len(dfw)} rows")
#     # (We could do an evaluation too, omitted here)
#     women_model, women_scaler = train_model_women(dfw)
#     print("Women’s final model trained (6 features, Rank_diff always 0).")

#     # 3) 2024 bracket seeds/slots
#     print("Loading Men’s 2024 bracket structure + seeds...")
#     men_seeds_dict, men_bracket_slots = load_slots_and_seeds_2024(dfs, men=True)
#     print("Loading Women’s 2024 bracket structure + seeds...")
#     women_seeds_dict, women_bracket_slots = load_slots_and_seeds_2024(dfs, men=False)

#     # 4) 2024 partial data aggregator
#     men_2024_agg   = build_2024_aggregator(dfs, men=True)
#     women_2024_agg = build_2024_aggregator(dfs, men=False)

#     # 5) Simulate multiple bracket predictions
#     NUM_BRACKETS_MEN   = 100
#     NUM_BRACKETS_WOMEN = 100

#     big_rows = []
#     rowId = 0

#     # MEN
#     for b_idx in range(1, NUM_BRACKETS_MEN + 1):
#         winners = simulate_bracket_once(
#             men_seeds_dict, men_bracket_slots,
#             men_model, men_scaler, True, men_2024_agg
#         )
#         bracket_rows = build_bracket_rows(winners, men=True)
#         for (slot, pick, tchar) in bracket_rows:
#             rowId += 1
#             big_rows.append((rowId, tchar, b_idx, slot, pick))

#     # WOMEN
#     for b_idx in range(1, NUM_BRACKETS_WOMEN + 1):
#         winners = simulate_bracket_once(
#             women_seeds_dict, women_bracket_slots,
#             women_model, women_scaler, False, women_2024_agg
#         )
#         bracket_rows = build_bracket_rows(winners, men=False)
#         for (slot, pick, tchar) in bracket_rows:
#             rowId += 1
#             big_rows.append((rowId, tchar, b_idx, slot, pick))

#     # 6) Output submission
#     df_sub = pd.DataFrame(big_rows, columns=["RowId","Tournament","Bracket","Slot","Team"])
#     df_sub.to_csv("submission.csv", index=False)
#     print(f"Saved {len(df_sub)} rows to submission.csv!  Done.")


# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python3
# """
# main.py
# Author: YOU
# Date: 2025-01-01

# Improved baseline logistic regression for March Madness 2024, with:
#  1) More features (margin of victory, offensive rebounds, personal fouls, blocks)
#  2) Exponential decay weighting for older seasons
#  3) Precomputed aggregator for 2024 partial data to avoid repeated scanning
#  4) Correct final bracket output format EXACTLY as required: "RowId,Tournament,Bracket,Slot,Team"
#  5) Additional optional evaluation step: time-based split + log loss for both men and women
#  6) Incorporation of MMasseyOrdinals_thruSeason2024_day128.csv for men, with "Rank_diff"=0 for women
# """

# import os
# import sys
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import log_loss

# ###############################################################################
# # 0) LOAD CSV FILES
# ###############################################################################

# def load_csv_safely(path):
#     """
#     Tries reading CSV with UTF-8 first, then fallback to 'latin1'/cp1252 if needed.
#     Helps fix "UnicodeDecodeError: 'utf-8' codec can't decode byte" issues.
#     """
#     try:
#         return pd.read_csv(path)
#     except UnicodeDecodeError:
#         print(f"Retrying {path} with latin1 encoding...")
#         return pd.read_csv(path, encoding="latin1")


# def load_all_data(data_dir="data"):
#     """
#     Loads all the CSV files we need, returns them as a dict of DataFrames.
#     We skip any missing files but warn.
#     """
#     csv_files = [
#         # Men’s data
#         "MSeasons.csv",
#         "MTeams.csv",
#         "MRegularSeasonCompactResults.csv",
#         "MRegularSeasonDetailedResults.csv",
#         "MNCAATourneyCompactResults.csv",
#         "MNCAATourneyDetailedResults.csv",
#         "MNCAATourneySeeds.csv",
#         "MNCAATourneySlots.csv",
#         # Women’s data
#         "WSeasons.csv",
#         "WTeams.csv",
#         "WRegularSeasonCompactResults.csv",
#         "WRegularSeasonDetailedResults.csv",
#         "WNCAATourneyCompactResults.csv",
#         "WNCAATourneyDetailedResults.csv",
#         "WNCAATourneySeeds.csv",
#         "WNCAATourneySlots.csv",
#         # Shared
#         "Cities.csv",
#         "MGameCities.csv",
#         "WGameCities.csv",
#         # Possibly more merges needed:
#         "MTeamCoaches.csv",
#         "MTeamConferences.csv",
#         "Conferences.csv",
#         # Next-year seeds (2024)
#         "2024_tourney_seeds.csv",
#         # For Massey (men's only)
#         "MMasseyOrdinals_thruSeason2024_day128.csv"
#     ]

#     dfs = {}
#     for f in csv_files:
#         path = os.path.join(data_dir, f)
#         if not os.path.exists(path):
#             print(f"WARNING: {f} not found in {data_dir}!")
#             continue
#         try:
#             df = load_csv_safely(path)
#             dfs[f] = df
#         except Exception as e:
#             print(f"WARNING: error reading {f}: {e}")
#     return dfs

# ###############################################################################
# # PARSE MEN'S MASSEY RANKS
# ###############################################################################

# def parse_mens_massey_ranks(dfs):
#     """
#     From 'MMasseyOrdinals_thruSeason2024_day128.csv', create:
#         mens_massey[(season, teamID)] = avg_ordinal_rank
#     filtering to RankingDayNum <=128, then averaging OrdinalRank.
#     """
#     file_key = "MMasseyOrdinals_thruSeason2024_day128.csv"
#     if file_key not in dfs:
#         print("WARNING: No Massey Ordinals CSV found. We'll skip it.")
#         return {}

#     df_m = dfs[file_key].copy()
#     df_m = df_m[df_m["RankingDayNum"] <= 128]

#     grouped = df_m.groupby(["Season","TeamID"])["OrdinalRank"].mean().reset_index()

#     massey_dict = {}
#     for _, row in grouped.iterrows():
#         season = row["Season"]
#         tid    = row["TeamID"]
#         avgRank= row["OrdinalRank"]
#         massey_dict[(season, tid)] = avgRank
#     return massey_dict

# ###############################################################################
# # 1) PREPARE TRAINING DATA + DECAY WEIGHT
# ###############################################################################

# def prep_training_data_men(dfs, mens_massey, season_start=2003, season_end=2023, decay_base=0.94):
#     """
#     Men’s training with (Margin, Loc, OR_diff, PF_diff, Blk_diff, Rank_diff).
#     """
#     needed = [
#         "MRegularSeasonDetailedResults.csv",
#         "MNCAATourneyDetailedResults.csv",
#     ]
#     for nf in needed:
#         if nf not in dfs:
#             raise ValueError(f"Missing {nf} in dfs!")

#     reg = dfs["MRegularSeasonDetailedResults.csv"]
#     tour = dfs["MNCAATourneyDetailedResults.csv"]
#     allgames = pd.concat([reg, tour], ignore_index=True)
#     allgames = allgames[(allgames["Season"] >= season_start) & (allgames["Season"] <= season_end)].copy()

#     def loc_value(loc):
#         if loc == 'H':
#             return 1
#         elif loc == 'A':
#             return -1
#         else:
#             return 0

#     rows = []
#     for _, row in allgames.iterrows():
#         season = row["Season"]
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         margin = row["WScore"] - row["LScore"]
#         wloc = loc_value(row.get("WLoc", "N"))
#         wor = row.get("WOR", 0)
#         lor = row.get("LOR", 0)
#         wpf = row.get("WPF", 0)
#         lpf = row.get("LPF", 0)
#         wblk= row.get("WBlk",0)
#         lblk= row.get("LBlk",0)

#         # Exponential decay weighting
#         years_ago = season_end - season
#         weight = (decay_base ** years_ago)

#         # Rank from men’s_massey
#         rankW = mens_massey.get((season, wtid), 0.0)
#         rankL = mens_massey.get((season, ltid), 0.0)

#         # Winner side
#         rows.append({
#             "Season": season,
#             "TeamA": wtid,
#             "TeamB": ltid,
#             "Margin": margin,
#             "Loc": wloc,
#             "OR_diff": (wor - lor),
#             "PF_diff": (wpf - lpf),
#             "Blk_diff": (wblk - lblk),
#             "Rank_diff": (rankW - rankL),
#             "Result": 1,
#             "Weight": weight
#         })
#         # Loser side
#         rows.append({
#             "Season": season,
#             "TeamA": ltid,
#             "TeamB": wtid,
#             "Margin": -margin,
#             "Loc": -wloc,
#             "OR_diff": (lor - wor),
#             "PF_diff": (lpf - wpf),
#             "Blk_diff": (lblk - wblk),
#             "Rank_diff": (rankL - rankW),
#             "Result": 0,
#             "Weight": weight
#         })

#     return pd.DataFrame(rows)

# def train_model_men(df_model):
#     """
#     6 features: [Margin, Loc, OR_diff, PF_diff, Blk_diff, Rank_diff].
#     """
#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
#     X = df_model[feats].values
#     y = df_model["Result"].values
#     w = df_model["Weight"].values

#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     model = LogisticRegression(random_state=42, max_iter=600)
#     model.fit(Xs, y, sample_weight=w)
#     return model, scaler

# def prep_training_data_women(dfs, season_start=2010, season_end=2023, decay_base=0.94):
#     """
#     Women’s training data with same 6 columns: 
#     (Margin, Loc, OR_diff, PF_diff, Blk_diff, Rank_diff=0).
#     """
#     needed = [
#         "WRegularSeasonDetailedResults.csv",
#         "WNCAATourneyDetailedResults.csv",
#     ]
#     for nf in needed:
#         if nf not in dfs:
#             raise ValueError(f"Missing {nf} in dfs!")

#     reg = dfs["WRegularSeasonDetailedResults.csv"]
#     tour = dfs["WNCAATourneyDetailedResults.csv"]
#     allgames = pd.concat([reg, tour], ignore_index=True)
#     allgames = allgames[(allgames["Season"]>=season_start) & (allgames["Season"]<=season_end)].copy()

#     def loc_value(loc):
#         if loc == 'H':
#             return 1
#         elif loc == 'A':
#             return -1
#         else:
#             return 0

#     rows = []
#     for _, row in allgames.iterrows():
#         season = row["Season"]
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         margin = row["WScore"] - row["LScore"]
#         wloc = loc_value(row.get("WLoc","N"))
#         wor = row.get("WOR",0)
#         lor = row.get("LOR",0)
#         wpf = row.get("WPF",0)
#         lpf = row.get("LPF",0)
#         wblk= row.get("WBlk",0)
#         lblk= row.get("LBlk",0)

#         years_ago = season_end - season
#         weight = (decay_base**years_ago)

#         # "Rank_diff"=0 for women's data
#         rows.append({
#             "Season": season,
#             "TeamA": wtid,
#             "TeamB": ltid,
#             "Margin": margin,
#             "Loc": wloc,
#             "OR_diff": (wor - lor),
#             "PF_diff": (wpf - lpf),
#             "Blk_diff": (wblk - lblk),
#             "Rank_diff": 0.0,
#             "Result": 1,
#             "Weight": weight
#         })
#         rows.append({
#             "Season": season,
#             "TeamA": ltid,
#             "TeamB": wtid,
#             "Margin": -margin,
#             "Loc": -wloc,
#             "OR_diff": (lor - wor),
#             "PF_diff": (lpf - wpf),
#             "Blk_diff": (lblk - wblk),
#             "Rank_diff": 0.0,
#             "Result": 0,
#             "Weight": weight
#         })

#     return pd.DataFrame(rows)

# def train_model_women(df_model):
#     """
#     Women’s also use 6 features (Rank_diff always 0).
#     """
#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
#     X = df_model[feats].values
#     y = df_model["Result"].values
#     w = df_model["Weight"].values

#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)
#     model = LogisticRegression(random_state=43, max_iter=600)
#     model.fit(Xs, y, sample_weight=w)
#     return model, scaler

# ###############################################################################
# # 1b) OPTIONAL EVALUATION: SPLIT & LOG LOSS
# ###############################################################################

# def evaluate_model_time_split_men(df_all, year_split=2022):
#     """
#     Time-based split for men’s dataset:
#      - train on seasons <= year_split
#      - validate on seasons > year_split
#      measure log loss on validation
#     """
#     df_train = df_all[df_all["Season"] <= year_split]
#     df_val   = df_all[df_all["Season"] >  year_split]

#     if len(df_train)==0 or len(df_val)==0:
#         print("WARNING: Not enough men’s data for train/val split; skipping evaluation.")
#         return

#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
#     X_train = df_train[feats].values
#     y_train = df_train["Result"].values
#     w_train = df_train["Weight"].values

#     scaler = StandardScaler()
#     Xs_train = scaler.fit_transform(X_train)

#     model = LogisticRegression(random_state=42, max_iter=600)
#     model.fit(Xs_train, y_train, sample_weight=w_train)

#     X_val  = df_val[feats].values
#     y_val  = df_val["Result"].values
#     Xs_val = scaler.transform(X_val)
#     preds  = model.predict_proba(Xs_val)[:,1]
#     ll = log_loss(y_val, preds)
#     print(f"Men’s validation log loss (<= {year_split} vs. > {year_split}): {ll:.5f}")


# # ---- ADDED FOR WOMEN’S DATA ----
# def evaluate_model_time_split_women(df_all, year_split=2022):
#     """
#     Time-based split for women’s dataset:
#      - train on seasons <= year_split
#      - validate on seasons > year_split
#      measure log loss on validation
#     """
#     df_train = df_all[df_all["Season"] <= year_split]
#     df_val   = df_all[df_all["Season"] >  year_split]

#     if len(df_train)==0 or len(df_val)==0:
#         print("WARNING: Not enough women’s data for train/val split; skipping evaluation.")
#         return

#     feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
#     X_train = df_train[feats].values
#     y_train = df_train["Result"].values
#     w_train = df_train["Weight"].values

#     scaler = StandardScaler()
#     Xs_train = scaler.fit_transform(X_train)

#     model = LogisticRegression(random_state=43, max_iter=600)
#     model.fit(Xs_train, y_train, sample_weight=w_train)

#     X_val  = df_val[feats].values
#     y_val  = df_val["Result"].values
#     Xs_val = scaler.transform(X_val)
#     preds  = model.predict_proba(Xs_val)[:,1]
#     ll = log_loss(y_val, preds)
#     print(f"Women’s validation log loss (<= {year_split} vs. > {year_split}): {ll:.5f}")
# # --------------------------------

# ###############################################################################
# # 2) LOAD 2024 SLOTS/SEEDS + PARTIAL 2024 AGGREGATOR
# ###############################################################################

# def load_slots_and_seeds_2024(dfs, men=True):
#     """
#     For men: read '2024_tourney_seeds.csv' => seeds with 'M'
#              'MNCAATourneySlots.csv' => bracket
#     For women: read '2024_tourney_seeds.csv' => seeds with 'W'
#                'WNCAATourneySlots.csv' => bracket
#     """
#     if "2024_tourney_seeds.csv" not in dfs:
#         raise ValueError("Missing 2024_tourney_seeds.csv!")
#     df_seeds = dfs["2024_tourney_seeds.csv"]

#     if men:
#         df_seeds = df_seeds[df_seeds["Tournament"] == "M"].copy()
#         slot_file = "MNCAATourneySlots.csv"
#         if slot_file not in dfs:
#             raise ValueError("No MNCAATourneySlots.csv for men!")
#         df_slots = dfs[slot_file]
#         df_slots = df_slots[df_slots["Season"]==2024].copy()
#     else:
#         df_seeds = df_seeds[df_seeds["Tournament"] == "W"].copy()
#         slot_file = "WNCAATourneySlots.csv"
#         if slot_file not in dfs:
#             raise ValueError("No WNCAATourneySlots.csv for women!")
#         df_slots = dfs[slot_file]
#         df_slots = df_slots[df_slots["Season"]==2024].copy()

#     seeds_dict = {}
#     for _, row in df_seeds.iterrows():
#         seed = row["Seed"]
#         tid  = row["TeamID"]
#         seeds_dict[seed] = tid

#     bracket_slots = {}
#     for _, row in df_slots.iterrows():
#         slot = row["Slot"]
#         bracket_slots[slot] = (row["StrongSeed"], row["WeakSeed"])

#     return seeds_dict, bracket_slots


# def build_2024_aggregator(dfs, men=True):
#     """
#     aggregator[teamId] = { margin, or, pf, blk, games }
#     from the 2024 partial results
#     """
#     if men:
#         df_reg2024 = dfs.get("MRegularSeasonDetailedResults.csv", None)
#     else:
#         df_reg2024 = dfs.get("WRegularSeasonDetailedResults.csv", None)

#     if df_reg2024 is None:
#         return {}
#     df_reg2024 = df_reg2024[df_reg2024["Season"]==2024].copy()
#     if len(df_reg2024)==0:
#         return {}

#     stats={}
#     for _, row in df_reg2024.iterrows():
#         wtid = row["WTeamID"]
#         ltid = row["LTeamID"]
#         margin = row["WScore"] - row["LScore"]
#         wor = row.get("WOR",0)
#         lor = row.get("LOR",0)
#         wpf = row.get("WPF",0)
#         lpf = row.get("LPF",0)
#         wblk= row.get("WBlk",0)
#         lblk= row.get("LBlk",0)

#         if wtid not in stats:
#             stats[wtid] = {"margin":0,"or":0,"pf":0,"blk":0,"games":0}
#         if ltid not in stats:
#             stats[ltid] = {"margin":0,"or":0,"pf":0,"blk":0,"games":0}

#         stats[wtid]["margin"] += margin
#         stats[wtid]["or"]     += wor
#         stats[wtid]["pf"]     += wpf
#         stats[wtid]["blk"]    += wblk
#         stats[wtid]["games"]  += 1

#         stats[ltid]["margin"] -= margin
#         stats[ltid]["or"]     += lor
#         stats[ltid]["pf"]     += lpf
#         stats[ltid]["blk"]    += lblk
#         stats[ltid]["games"]  += 1

#     return stats


# def get_2024_team_averages(teamId, aggregator):
#     """
#     aggregator[teamId] => dict(margin=..., or=..., pf=..., blk=..., games=...)
#     returns (avg_margin, avg_or, avg_pf, avg_blk) or (0,0,0,0)
#     """
#     if (teamId not in aggregator) or (aggregator[teamId]["games"]==0):
#         return (0,0,0,0)
#     st = aggregator[teamId]
#     g  = st["games"]
#     return (
#         st["margin"]/g,
#         st["or"]/g,
#         st["pf"]/g,
#         st["blk"]/g
#     )

# ###############################################################################
# # 3) PREDICT GAME PROBABILITY + BRACKET SIMULATION
# ###############################################################################

# def predict_game_probability(teamA, teamB, model, scaler, men, aggregator_2024):
#     """
#     Both men/women are now trained on 6 features. For 2024 aggregator, rank_diff=0
#     """
#     a_margin, a_or, a_pf, a_blk = get_2024_team_averages(teamA, aggregator_2024)
#     b_margin, b_or, b_pf, b_blk = get_2024_team_averages(teamB, aggregator_2024)

#     margin_diff = a_margin - b_margin
#     or_diff = a_or - b_or
#     pf_diff = a_pf - b_pf
#     blk_diff= a_blk - b_blk
#     loc=0.0
#     rank_diff=0.0  # for 2024 partial aggregator
    
#     row_feat = np.array([[margin_diff, loc, or_diff, pf_diff, blk_diff, rank_diff]], dtype=float)
#     row_s    = scaler.transform(row_feat)
#     pA       = model.predict_proba(row_s)[0,1]
#     return pA

# def simulate_bracket_once(seeds_dict, bracket_slots, model, scaler, men, aggregator_2024):
#     """
#     Fill out bracket from Round1..Round6, picking winners by predicted probability.
#     Return bracket_winners: slot->winning seed label
#     """
#     slot_keys = list(bracket_slots.keys())

#     def slot_sort_key(s):
#         if s=="R6CH":
#             return (6,"CH")
#         elif s in ["R5WX","R5YZ"]:
#             return (5,s[2:])
#         elif s.startswith("R"):
#             rnd = int(s[1])
#             tail= s[2:]
#             return (rnd, tail)
#         else:
#             return (99, s)

#     slot_keys.sort(key=slot_sort_key)
#     bracket_winners = {}

#     def occupant_teamid(label):
#         if label in seeds_dict:
#             return seeds_dict[label]
#         if label in bracket_winners:
#             return occupant_teamid(bracket_winners[label])
#         return None

#     for slot in slot_keys:
#         strong, weak = bracket_slots[slot]
#         occA = bracket_winners.get(strong, strong)
#         occB = bracket_winners.get(weak, weak)

#         tidA = occupant_teamid(occA)
#         tidB = occupant_teamid(occB)
#         if (tidA is None) or (tidB is None):
#             bracket_winners[slot] = occA
#         else:
#             pA = predict_game_probability(tidA, tidB, model, scaler, men, aggregator_2024)
#             if np.random.rand() < pA:
#                 bracket_winners[slot] = occA
#             else:
#                 bracket_winners[slot] = occB

#     return bracket_winners

# def build_bracket_rows(bracket_winners, men=True):
#     """
#     Convert bracket_winners => (RowId, Tournament, Bracket, Slot, Team)
#     skipping non-"R" slots (like play-ins).
#     """
#     def slot_sort_key(s):
#         if s=="R6CH":
#             return (6,"CH")
#         if s in ["R5WX","R5YZ"]:
#             return (5,s[2:])
#         if s.startswith("R"):
#             rnd=int(s[1])
#             tail=s[2:]
#             return (rnd, tail)
#         return (999,s)

#     valid_slots = []
#     for slot in bracket_winners.keys():
#         if slot.startswith("R"):
#             valid_slots.append(slot)
#     valid_slots.sort(key=slot_sort_key)

#     tchar = 'M' if men else 'W'
#     bracket_rows = []
#     for slot in valid_slots:
#         pick = bracket_winners[slot]
#         # strip a/b suffix if exist
#         if pick.endswith('a') or pick.endswith('b'):
#             pick = pick[:-1]
#         bracket_rows.append((slot, pick, tchar))
#     return bracket_rows

# ###############################################################################
# # 4) MAIN
# ###############################################################################

# def main():
#     print("Loading data...")
#     data_dir = "/kaggle/input/march-machine-learning-mania-2024"  # or your local path
#     dfs = load_all_data(data_dir)
#     print("Data loaded.")

#     # Parse men’s massey
#     mens_massey = parse_mens_massey_ranks(dfs)

#     # 1) Men’s data
#     print("Preparing men’s training data (2003–2023) w/ exponent. decay + Rank_diff...")
#     dfm = prep_training_data_men(dfs, mens_massey, season_start=2003, season_end=2023, decay_base=0.94)
#     print(f"Men’s training set: {len(dfm)} rows")

#     # Evaluate men’s split (optional)
#     evaluate_model_time_split_men(dfm, year_split=2022)

#     men_model, men_scaler = train_model_men(dfm)
#     print("Men’s final model trained (6 features).")

#     # 2) Women’s data
#     print("Preparing women’s training data (2010–2023) w/ exponent. decay + Rank_diff=0...")
#     dfw = prep_training_data_women(dfs, 2010, 2023, decay_base=0.94)
#     print(f"Women’s training set: {len(dfw)} rows")

#     # (ADDED) Evaluate women’s split
#     evaluate_model_time_split_women(dfw, year_split=2022)

#     women_model, women_scaler = train_model_women(dfw)
#     print("Women’s final model trained (6 features, Rank_diff always 0).")

#     # 3) 2024 bracket seeds/slots
#     print("Loading Men’s 2024 bracket structure + seeds...")
#     men_seeds_dict, men_bracket_slots = load_slots_and_seeds_2024(dfs, men=True)
#     print("Loading Women’s 2024 bracket structure + seeds...")
#     women_seeds_dict, women_bracket_slots = load_slots_and_seeds_2024(dfs, men=False)

#     # 4) 2024 partial data aggregator
#     men_2024_agg   = build_2024_aggregator(dfs, men=True)
#     women_2024_agg = build_2024_aggregator(dfs, men=False)

#     # 5) Simulate multiple bracket predictions
#     NUM_BRACKETS_MEN   = 1000
#     NUM_BRACKETS_WOMEN = 1000

#     big_rows = []
#     rowId = 0

#     # MEN
#     for b_idx in range(1, NUM_BRACKETS_MEN + 1):
#         winners = simulate_bracket_once(
#             men_seeds_dict, men_bracket_slots,
#             men_model, men_scaler, True, men_2024_agg
#         )
#         bracket_rows = build_bracket_rows(winners, men=True)
#         for (slot, pick, tchar) in bracket_rows:
#             rowId += 1
#             big_rows.append((rowId, tchar, b_idx, slot, pick))

#     # WOMEN
#     for b_idx in range(1, NUM_BRACKETS_WOMEN + 1):
#         winners = simulate_bracket_once(
#             women_seeds_dict, women_bracket_slots,
#             women_model, women_scaler, False, women_2024_agg
#         )
#         bracket_rows = build_bracket_rows(winners, men=False)
#         for (slot, pick, tchar) in bracket_rows:
#             rowId += 1
#             big_rows.append((rowId, tchar, b_idx, slot, pick))

#     # 6) Output submission
#     df_sub = pd.DataFrame(big_rows, columns=["RowId","Tournament","Bracket","Slot","Team"])
#     df_sub.to_csv("submission_3.csv", index=False)
#     print(f"Saved {len(df_sub)} rows to submission.csv!  Done.")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
main.py
Author: Ben, Tim and Omri
Date: 2025-01-27
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from sklearn.metrics import mean_squared_error  # for Brier score

###############################################################################
# 0) LOAD CSV FILES
###############################################################################

def load_csv_safely(path):
    """
    Tries reading CSV with UTF-8 first, then fallback to 'latin1'/cp1252 if needed.
    Helps fix "UnicodeDecodeError: 'utf-8' codec can't decode byte" issues.
    """
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        print(f"Retrying {path} with latin1 encoding...")
        return pd.read_csv(path, encoding="latin1")


def load_all_data(data_dir="data"):
    """
    Loads all the CSV files we need, returns them as a dict of DataFrames.
    We skip any missing files but warn.
    """
    csv_files = [
        # Men’s data
        "MSeasons.csv",
        "MTeams.csv",
        "MRegularSeasonCompactResults.csv",
        "MRegularSeasonDetailedResults.csv",
        "MNCAATourneyCompactResults.csv",
        "MNCAATourneyDetailedResults.csv",
        "MNCAATourneySeeds.csv",
        "MNCAATourneySlots.csv",
        # Women’s data
        "WSeasons.csv",
        "WTeams.csv",
        "WRegularSeasonCompactResults.csv",
        "WRegularSeasonDetailedResults.csv",
        "WNCAATourneyCompactResults.csv",
        "WNCAATourneyDetailedResults.csv",
        "WNCAATourneySeeds.csv",
        "WNCAATourneySlots.csv",
        # Shared
        "Cities.csv",
        "MGameCities.csv",
        "WGameCities.csv",
        # Possibly more merges needed:
        "MTeamCoaches.csv",
        "MTeamConferences.csv",
        "Conferences.csv",
        # Next-year seeds (2024)
        "2024_tourney_seeds.csv",
        # For Massey (men's only)
        "MMasseyOrdinals_thruSeason2024_day128.csv"
    ]

    dfs = {}
    for f in csv_files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"WARNING: {f} not found in {data_dir}!")
            continue
        try:
            df = load_csv_safely(path)
            dfs[f] = df
        except Exception as e:
            print(f"WARNING: error reading {f}: {e}")
    return dfs

###############################################################################
# PARSE MEN'S MASSEY RANKS
###############################################################################

def parse_mens_massey_ranks(dfs):
    """
    From 'MMasseyOrdinals_thruSeason2024_day128.csv', create:
        mens_massey[(season, teamID)] = avg_ordinal_rank
    filtering to RankingDayNum <=128, then averaging OrdinalRank.
    """
    file_key = "MMasseyOrdinals_thruSeason2024_day128.csv"
    if file_key not in dfs:
        print("WARNING: No Massey Ordinals CSV found. We'll skip it.")
        return {}

    df_m = dfs[file_key].copy()
    df_m = df_m[df_m["RankingDayNum"] <= 128]

    grouped = df_m.groupby(["Season","TeamID"])["OrdinalRank"].mean().reset_index()

    massey_dict = {}
    for _, row in grouped.iterrows():
        season = row["Season"]
        tid    = row["TeamID"]
        avgRank= row["OrdinalRank"]
        massey_dict[(season, tid)] = avgRank
    return massey_dict

###############################################################################
# 1) PREPARE TRAINING DATA + DECAY WEIGHT
###############################################################################

def prep_training_data_men(dfs, mens_massey, season_start=2003, season_end=2023, decay_base=0.94):
    """
    Men’s training with (Margin, Loc, OR_diff, PF_diff, Blk_diff, Rank_diff).
    """
    needed = [
        "MRegularSeasonDetailedResults.csv",
        "MNCAATourneyDetailedResults.csv",
    ]
    for nf in needed:
        if nf not in dfs:
            raise ValueError(f"Missing {nf} in dfs!")

    reg = dfs["MRegularSeasonDetailedResults.csv"]
    tour = dfs["MNCAATourneyDetailedResults.csv"]
    allgames = pd.concat([reg, tour], ignore_index=True)
    allgames = allgames[(allgames["Season"] >= season_start) & (allgames["Season"] <= season_end)].copy()

    def loc_value(loc):
        if loc == 'H':
            return 1
        elif loc == 'A':
            return -1
        else:
            return 0

    rows = []
    for _, row in allgames.iterrows():
        season = row["Season"]
        wtid = row["WTeamID"]
        ltid = row["LTeamID"]
        margin = row["WScore"] - row["LScore"]
        wloc = loc_value(row.get("WLoc", "N"))
        wor = row.get("WOR", 0)
        lor = row.get("LOR", 0)
        wpf = row.get("WPF", 0)
        lpf = row.get("LPF", 0)
        wblk= row.get("WBlk",0)
        lblk= row.get("LBlk",0)

        # Exponential decay weighting
        years_ago = season_end - season
        weight = (decay_base ** years_ago)

        # Rank from men’s_massey
        rankW = mens_massey.get((season, wtid), 0.0)
        rankL = mens_massey.get((season, ltid), 0.0)

        # Winner side
        rows.append({
            "Season": season,
            "TeamA": wtid,
            "TeamB": ltid,
            "Margin": margin,
            "Loc": wloc,
            "OR_diff": (wor - lor),
            "PF_diff": (wpf - lpf),
            "Blk_diff": (wblk - lblk),
            "Rank_diff": (rankW - rankL),
            "Result": 1,
            "Weight": weight
        })
        # Loser side
        rows.append({
            "Season": season,
            "TeamA": ltid,
            "TeamB": wtid,
            "Margin": -margin,
            "Loc": -wloc,
            "OR_diff": (lor - wor),
            "PF_diff": (lpf - wpf),
            "Blk_diff": (lblk - wblk),
            "Rank_diff": (rankL - rankW),
            "Result": 0,
            "Weight": weight
        })

    return pd.DataFrame(rows)

def train_model_men(df_model):
    """
    6 features: [Margin, Loc, OR_diff, PF_diff, Blk_diff, Rank_diff].
    """
    feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
    X = df_model[feats].values
    y = df_model["Result"].values
    w = df_model["Weight"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=600)
    model.fit(Xs, y, sample_weight=w)
    return model, scaler

def prep_training_data_women(dfs, season_start=2010, season_end=2023, decay_base=0.94):
    """
    Women’s training data with same 6 columns:
    (Margin, Loc, OR_diff, PF_diff, Blk_diff, Rank_diff=0).
    """
    needed = [
        "WRegularSeasonDetailedResults.csv",
        "WNCAATourneyDetailedResults.csv",
    ]
    for nf in needed:
        if nf not in dfs:
            raise ValueError(f"Missing {nf} in dfs!")

    reg = dfs["WRegularSeasonDetailedResults.csv"]
    tour = dfs["WNCAATourneyDetailedResults.csv"]
    allgames = pd.concat([reg, tour], ignore_index=True)
    allgames = allgames[(allgames["Season"]>=season_start) & (allgames["Season"]<=season_end)].copy()

    def loc_value(loc):
        if loc == 'H':
            return 1
        elif loc == 'A':
            return -1
        else:
            return 0

    rows = []
    for _, row in allgames.iterrows():
        season = row["Season"]
        wtid = row["WTeamID"]
        ltid = row["LTeamID"]
        margin = row["WScore"] - row["LScore"]
        wloc = loc_value(row.get("WLoc","N"))
        wor = row.get("WOR",0)
        lor = row.get("LOR",0)
        wpf = row.get("WPF",0)
        lpf = row.get("LPF",0)
        wblk= row.get("WBlk",0)
        lblk= row.get("LBlk",0)

        years_ago = season_end - season
        weight = (decay_base**years_ago)

        # "Rank_diff"=0 for women's data
        rows.append({
            "Season": season,
            "TeamA": wtid,
            "TeamB": ltid,
            "Margin": margin,
            "Loc": wloc,
            "OR_diff": (wor - lor),
            "PF_diff": (wpf - lpf),
            "Blk_diff": (wblk - lblk),
            "Rank_diff": 0.0,
            "Result": 1,
            "Weight": weight
        })
        rows.append({
            "Season": season,
            "TeamA": ltid,
            "TeamB": wtid,
            "Margin": -margin,
            "Loc": -wloc,
            "OR_diff": (lor - wor),
            "PF_diff": (lpf - wpf),
            "Blk_diff": (lblk - wblk),
            "Rank_diff": 0.0,
            "Result": 0,
            "Weight": weight
        })

    return pd.DataFrame(rows)

def train_model_women(df_model):
    """
    Women’s also use 6 features (Rank_diff always 0).
    """
    feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
    X = df_model[feats].values
    y = df_model["Result"].values
    w = df_model["Weight"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(random_state=43, max_iter=600)
    model.fit(Xs, y, sample_weight=w)
    return model, scaler

###############################################################################
# 1b) OPTIONAL EVALUATION: SPLIT & LOG LOSS + BRIER
###############################################################################

def evaluate_model_time_split_men(df_all, year_split=2022):
    """
    Time-based split for men’s dataset:
     - train on seasons <= year_split
     - validate on seasons > year_split
     measure log loss on validation
     also measure Brier score on validation
    """
    df_train = df_all[df_all["Season"] <= year_split]
    df_val   = df_all[df_all["Season"] >  year_split]

    if len(df_train)==0 or len(df_val)==0:
        print("WARNING: Not enough men’s data for train/val split; skipping evaluation.")
        return

    feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
    X_train = df_train[feats].values
    y_train = df_train["Result"].values
    w_train = df_train["Weight"].values

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)

    model = LogisticRegression(random_state=42, max_iter=600)
    model.fit(Xs_train, y_train, sample_weight=w_train)

    X_val  = df_val[feats].values
    y_val  = df_val["Result"].values
    Xs_val = scaler.transform(X_val)
    preds  = model.predict_proba(Xs_val)[:,1]
    ll = log_loss(y_val, preds)
    brier = mean_squared_error(y_val, preds)  # Brier is MSE of prob vs. outcome

    print(f"Men’s validation log loss (<= {year_split} vs. > {year_split}): {ll:.5f}")
    print(f"Men’s validation brier score (<= {year_split} vs. > {year_split}): {brier:.5f}")

def evaluate_model_time_split_women(df_all, year_split=2022):
    """
    Time-based split for women’s dataset:
     - train on seasons <= year_split
     - validate on seasons > year_split
     measure log loss on validation
     also measure Brier score
    """
    df_train = df_all[df_all["Season"] <= year_split]
    df_val   = df_all[df_all["Season"] >  year_split]

    if len(df_train)==0 or len(df_val)==0:
        print("WARNING: Not enough women’s data for train/val split; skipping evaluation.")
        return

    feats = ["Margin","Loc","OR_diff","PF_diff","Blk_diff","Rank_diff"]
    X_train = df_train[feats].values
    y_train = df_train["Result"].values
    w_train = df_train["Weight"].values

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)

    model = LogisticRegression(random_state=43, max_iter=600)
    model.fit(Xs_train, y_train, sample_weight=w_train)

    X_val  = df_val[feats].values
    y_val  = df_val["Result"].values
    Xs_val = scaler.transform(X_val)
    preds  = model.predict_proba(Xs_val)[:,1]
    ll = log_loss(y_val, preds)
    brier = mean_squared_error(y_val, preds)

    print(f"Women’s validation log loss (<= {year_split} vs. > {year_split}): {ll:.5f}")
    print(f"Women’s validation brier score (<= {year_split} vs. > {year_split}): {brier:.5f}")

###############################################################################
# 2) LOAD 2024 SLOTS/SEEDS + PARTIAL 2024 AGGREGATOR
###############################################################################

def load_slots_and_seeds_2024(dfs, men=True):
    """
    For men: read '2024_tourney_seeds.csv' => seeds with 'M'
             'MNCAATourneySlots.csv' => bracket
    For women: read '2024_tourney_seeds.csv' => seeds with 'W'
               'WNCAATourneySlots.csv' => bracket
    """
    if "2024_tourney_seeds.csv" not in dfs:
        raise ValueError("Missing 2024_tourney_seeds.csv!")
    df_seeds = dfs["2024_tourney_seeds.csv"]

    if men:
        df_seeds = df_seeds[df_seeds["Tournament"] == "M"].copy()
        slot_file = "MNCAATourneySlots.csv"
        if slot_file not in dfs:
            raise ValueError("No MNCAATourneySlots.csv for men!")
        df_slots = dfs[slot_file]
        df_slots = df_slots[df_slots["Season"]==2024].copy()
    else:
        df_seeds = df_seeds[df_seeds["Tournament"] == "W"].copy()
        slot_file = "WNCAATourneySlots.csv"
        if slot_file not in dfs:
            raise ValueError("No WNCAATourneySlots.csv for women!")
        df_slots = dfs[slot_file]
        df_slots = df_slots[df_slots["Season"]==2024].copy()

    seeds_dict = {}
    for _, row in df_seeds.iterrows():
        seed = row["Seed"]
        tid  = row["TeamID"]
        seeds_dict[seed] = tid

    bracket_slots = {}
    for _, row in df_slots.iterrows():
        slot = row["Slot"]
        bracket_slots[slot] = (row["StrongSeed"], row["WeakSeed"])

    return seeds_dict, bracket_slots


def build_2024_aggregator(dfs, men=True):
    """
    aggregator[teamId] = { margin, or, pf, blk, games }
    from the 2024 partial results
    """
    if men:
        df_reg2024 = dfs.get("MRegularSeasonDetailedResults.csv", None)
    else:
        df_reg2024 = dfs.get("WRegularSeasonDetailedResults.csv", None)

    if df_reg2024 is None:
        return {}
    df_reg2024 = df_reg2024[df_reg2024["Season"]==2024].copy()
    if len(df_reg2024)==0:
        return {}

    stats={}
    for _, row in df_reg2024.iterrows():
        wtid = row["WTeamID"]
        ltid = row["LTeamID"]
        margin = row["WScore"] - row["LScore"]
        wor = row.get("WOR",0)
        lor = row.get("LOR",0)
        wpf = row.get("WPF",0)
        lpf = row.get("LPF",0)
        wblk= row.get("WBlk",0)
        lblk= row.get("LBlk",0)

        if wtid not in stats:
            stats[wtid] = {"margin":0,"or":0,"pf":0,"blk":0,"games":0}
        if ltid not in stats:
            stats[ltid] = {"margin":0,"or":0,"pf":0,"blk":0,"games":0}

        stats[wtid]["margin"] += margin
        stats[wtid]["or"]     += wor
        stats[wtid]["pf"]     += wpf
        stats[wtid]["blk"]    += wblk
        stats[wtid]["games"]  += 1

        stats[ltid]["margin"] -= margin
        stats[ltid]["or"]     += lor
        stats[ltid]["pf"]     += lpf
        stats[ltid]["blk"]    += lblk
        stats[ltid]["games"]  += 1

    return stats

def get_2024_team_averages(teamId, aggregator):
    """
    aggregator[teamId] => dict(margin=..., or=..., pf=..., blk=..., games=...)
    returns (avg_margin, avg_or, avg_pf, avg_blk) or (0,0,0,0)
    """
    if (teamId not in aggregator) or (aggregator[teamId]["games"]==0):
        return (0,0,0,0)
    st = aggregator[teamId]
    g  = st["games"]
    return (
        st["margin"]/g,
        st["or"]/g,
        st["pf"]/g,
        st["blk"]/g
    )

###############################################################################
# 3) PREDICT GAME PROBABILITY + BRACKET SIMULATION
###############################################################################

def parse_seed_number(seed_label):
    """
    Extract integer from seed_label (e.g. "W01" -> 1, "X15" -> 15).
    We only look for the numeric portion.
    If not found, return 99.
    """
    import re
    match = re.search(r'(\d+)', seed_label)
    if match:
        return int(match.group(1))
    return 99

def predict_game_probability(teamA, teamB, model, scaler, men, aggregator_2024,
                             mens_massey=None,
                             seedA="X01", seedB="X16"):
    """
    Both men/women are now trained on 6 features. For 2024 aggregator:
     - margin/or/pf/blk from aggregator_2024
     - rank_diff from men’s massey if men, else 0
     - loc = +1 if A seed < B seed, -1 if A seed > B seed, 0 if tie
       (i.e. treat better seed as "home")
    """
    a_margin, a_or, a_pf, a_blk = get_2024_team_averages(teamA, aggregator_2024)
    b_margin, b_or, b_pf, b_blk = get_2024_team_averages(teamB, aggregator_2024)

    margin_diff = a_margin - b_margin
    or_diff = a_or - b_or
    pf_diff = a_pf - b_pf
    blk_diff= a_blk - b_blk

    # For men, we might have 2024 rank in mens_massey:
    # if not present, fallback to 0
    if men and mens_massey is not None:
        rankA = mens_massey.get((2024, teamA), 0.0)
        rankB = mens_massey.get((2024, teamB), 0.0)
        rank_diff = rankA - rankB
    else:
        rank_diff = 0.0

    # Determine loc based on seeds: better seed = "home" => loc=+1
    sA = parse_seed_number(seedA)
    sB = parse_seed_number(seedB)
    if sA < sB:
        loc = 1.0
    elif sA > sB:
        loc = -1.0
    else:
        loc = 0.0

    row_feat = np.array([[margin_diff, loc, or_diff, pf_diff, blk_diff, rank_diff]], dtype=float)
    row_s    = scaler.transform(row_feat)
    pA       = model.predict_proba(row_s)[0,1]
    return pA

def simulate_bracket_once(seeds_dict, bracket_slots, model, scaler, men, aggregator_2024,
                          mens_massey=None):
    """
    Fill out bracket from Round1..Round6, picking winners by predicted probability.
    Return bracket_winners: slot->winning seed label
    """
    slot_keys = list(bracket_slots.keys())

    def slot_sort_key(s):
        if s=="R6CH":
            return (6,"CH")
        elif s in ["R5WX","R5YZ"]:
            return (5,s[2:])
        elif s.startswith("R"):
            rnd = int(s[1])
            tail= s[2:]
            return (rnd, tail)
        else:
            return (99, s)

    slot_keys.sort(key=slot_sort_key)
    bracket_winners = {}

    def occupant_teamid(label):
        if label in seeds_dict:
            return seeds_dict[label]
        if label in bracket_winners:
            return occupant_teamid(bracket_winners[label])
        return None

    # We'll also pass the original seeds to the predict fn, so we can get loc
    # strongSeed, weakSeed are the seeds if they exist in seeds_dict
    # occupant is the 'label' which might be a seed or a prior slot
    def get_seed_label(lab):
        # If it's a direct seed (like "W01","X11","Z16"), return it
        if lab in seeds_dict:
            return lab
        # else see if bracket_winners[lab] is a seed
        if lab in bracket_winners and bracket_winners[lab] in seeds_dict:
            return bracket_winners[lab]
        return lab  # fallback

    for slot in slot_keys:
        strong, weak = bracket_slots[slot]
        occA = bracket_winners.get(strong, strong)
        occB = bracket_winners.get(weak, weak)

        tidA = occupant_teamid(occA)
        tidB = occupant_teamid(occB)
        if (tidA is None) or (tidB is None):
            bracket_winners[slot] = occA
        else:
            seedA = get_seed_label(occA)
            seedB = get_seed_label(occB)
            pA = predict_game_probability(tidA, tidB,
                                          model, scaler, men, aggregator_2024,
                                          mens_massey=mens_massey,
                                          seedA=seedA, seedB=seedB)
            if np.random.rand() < pA:
                bracket_winners[slot] = occA
            else:
                bracket_winners[slot] = occB

    return bracket_winners

def build_bracket_rows(bracket_winners, men=True):
    """
    Convert bracket_winners => (RowId, Tournament, Bracket, Slot, Team)
    skipping non-"R" slots (like play-ins).
    """
    def slot_sort_key(s):
        if s=="R6CH":
            return (6,"CH")
        if s in ["R5WX","R5YZ"]:
            return (5,s[2:])
        if s.startswith("R"):
            rnd=int(s[1])
            tail=s[2:]
            return (rnd, tail)
        return (999,s)

    valid_slots = []
    for slot in bracket_winners.keys():
        if slot.startswith("R"):
            valid_slots.append(slot)
    valid_slots.sort(key=slot_sort_key)

    tchar = 'M' if men else 'W'
    bracket_rows = []
    for slot in valid_slots:
        pick = bracket_winners[slot]
        # strip a/b suffix if exist
        if pick.endswith('a') or pick.endswith('b'):
            pick = pick[:-1]
        bracket_rows.append((slot, pick, tchar))
    return bracket_rows

###############################################################################
# 4) MAIN
###############################################################################

def main():
    print("Loading data...")
    data_dir = "/kaggle/input/march-machine-learning-mania-2024"  # or your local path
    dfs = load_all_data(data_dir)
    print("Data loaded.")

    # Parse men’s massey
    mens_massey = parse_mens_massey_ranks(dfs)

    # 1) Men’s data
    print("Preparing men’s training data (2003–2023) w/ exponent. decay + Rank_diff...")
    dfm = prep_training_data_men(dfs, mens_massey, season_start=2003, season_end=2023, decay_base=0.94)
    print(f"Men’s training set: {len(dfm)} rows")

    # Evaluate men’s split (optional)
    evaluate_model_time_split_men(dfm, year_split=2022)

    men_model, men_scaler = train_model_men(dfm)
    print("Men’s final model trained (6 features).")

    # 2) Women’s data
    print("Preparing women’s training data (2010–2023) w/ exponent. decay + Rank_diff=0...")
    dfw = prep_training_data_women(dfs, 2010, 2023, decay_base=0.94)
    print(f"Women’s training set: {len(dfw)} rows")

    # (ADDED) Evaluate women’s split
    evaluate_model_time_split_women(dfw, year_split=2022)

    women_model, women_scaler = train_model_women(dfw)
    print("Women’s final model trained (6 features, Rank_diff always 0).")

    # 3) 2024 bracket seeds/slots
    print("Loading Men’s 2024 bracket structure + seeds...")
    men_seeds_dict, men_bracket_slots = load_slots_and_seeds_2024(dfs, men=True)
    print("Loading Women’s 2024 bracket structure + seeds...")
    women_seeds_dict, women_bracket_slots = load_slots_and_seeds_2024(dfs, men=False)

    # 4) 2024 partial data aggregator
    men_2024_agg   = build_2024_aggregator(dfs, men=True)
    women_2024_agg = build_2024_aggregator(dfs, men=False)

    # 5) Simulate multiple bracket predictions
    NUM_BRACKETS_MEN   = 10
    NUM_BRACKETS_WOMEN = 10

    big_rows = []
    rowId = 0

    # MEN
    for b_idx in range(1, NUM_BRACKETS_MEN + 1):
        winners = simulate_bracket_once(
            men_seeds_dict, men_bracket_slots,
            men_model, men_scaler, True, men_2024_agg,
            mens_massey=mens_massey  # pass in men’s massey for rank_diff
        )
        bracket_rows = build_bracket_rows(winners, men=True)
        for (slot, pick, tchar) in bracket_rows:
            rowId += 1
            big_rows.append((rowId, tchar, b_idx, slot, pick))

    # WOMEN
    for b_idx in range(1, NUM_BRACKETS_WOMEN + 1):
        winners = simulate_bracket_once(
            women_seeds_dict, women_bracket_slots,
            women_model, women_scaler, False, women_2024_agg,
            mens_massey=None  # women => rank_diff=0 always
        )
        bracket_rows = build_bracket_rows(winners, men=False)
        for (slot, pick, tchar) in bracket_rows:
            rowId += 1
            big_rows.append((rowId, tchar, b_idx, slot, pick))

    # 6) Output submission
    df_sub = pd.DataFrame(big_rows, columns=["RowId","Tournament","Bracket","Slot","Team"])
    df_sub.to_csv("submission.csv", index=False)
    print(f"Saved {len(df_sub)} rows to submission.csv!  Done.")


if __name__ == "__main__":
    main()


