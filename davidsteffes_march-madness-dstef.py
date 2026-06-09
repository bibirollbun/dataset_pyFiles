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
        x = os.path.join(dirname, filename)
        print(x)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


dataframes = {}  # dictionary to hold all dataframes

# Walk through the input directory
for dirname, _, filenames in os.walk('/kaggle/input/march-machine-learning-mania-2025'):
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(dirname, filename)
            # Use the filename (minus extension) as the dictionary key
            df_name = filename.replace('.csv', '')
            dataframes[df_name] = pd.read_csv(file_path)

# Now you can access each DataFrame by its filename key
print(dataframes.keys())


for key, df in dataframes.items():
    globals()[key] = df


lowest_idx_2025 = MMasseyOrdinals.loc[MMasseyOrdinals['Season'] == 2025, 'OrdinalRank'].idxmin()
lowest_row_2025 = MMasseyOrdinals.loc[lowest_idx_2025]
print(lowest_row_2025)


MRegularSeasonDetailedResults


MNCAATourneyDetailedResults


MRegularSeasonDetailedResults['istourney'] = 0
MNCAATourneyDetailedResults['istourney'] = 1


MDetailedResults = pd.concat([MRegularSeasonDetailedResults, MNCAATourneyDetailedResults])


MDetailedResults.head()


MDetailedResults.columns


MNCAATourneySeeds


MNCAATourneySeeds = MNCAATourneySeeds[MNCAATourneySeeds['Season'] > 2002]


MNCAATourneySeeds


df_men = MDetailedResults.copy()  # has WTeamID, LTeamID, etc.
df_teams = MTeams[['TeamID', 'TeamName']]          # just the columns you need

# Merge on WTeamID first
df_men = (
    df_men.merge(df_teams, how='left',
                     left_on='WTeamID', right_on='TeamID')
    .rename(columns={'TeamName': 'WTeamName'})
    .drop('TeamID', axis=1)
)

# Merge on LTeamID next
df_men = (
    df_men.merge(df_teams, how='left',
                     left_on='LTeamID', right_on='TeamID')
    .rename(columns={'TeamName': 'LTeamName'})
    .drop('TeamID', axis=1)
)

df_men.head()


import pandas as pd

# --- Example setup ---
# MDetailedResults has columns like:
#   Season, DayNum, WTeamID, WScore, LTeamID, LScore, ...
# MNCAATourneySeeds has columns like:
#   Season, Seed, TeamID

# 1) Merge WSeed
tmp_w = MNCAATourneySeeds[['Season','TeamID','Seed']] \
            .rename(columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'})
df_merged = MDetailedResults.merge(
    tmp_w,
    on=['Season','WTeamID'],
    how='left'
)

# 2) Merge LSeed
tmp_l = MNCAATourneySeeds[['Season','TeamID','Seed']] \
            .rename(columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'})
df_merged = df_merged.merge(
    tmp_l,
    on=['Season','LTeamID'],
    how='left'
)

# Suppose you already have is_tourney=1 for tourney games, 0 otherwise
df_merged.loc[df_merged['istourney'] == 0, ['WSeed','LSeed']] = pd.NA


# df_merged now has MDetailedResults plus two new columns: WSeed and LSeed.
# Regular-season rows (or teams that never got a seed) will have NaN in WSeed/LSeed.



df_merged.head()


df = df_merged.copy()  # e.g., your merged Detailed Results with Team Names


import pandas as pd
import numpy as np

def add_elo_features_with_rolling_momentum(df, base_elo=1500, k_factor=20, window=5):
    """
    Computes pre-game Elo and shifted Elo change features.
    
    For each game in df (assumed sorted chronologically by Season and DayNum),
    compute:
      - Pre-game Elo for each team ("WTeamEloBefore", "LTeamEloBefore")
      - The previous game's Elo change (delta) for each team,
        stored as "WTeamEloDelta_lastGame" and "LTeamEloDelta_lastGame".
      - Similarly, the second derivative (delta2) from the previous game,
        as "WTeamEloDelta2_lastGame" and "LTeamEloDelta2_lastGame".
      - Update Elo using the current game's result (but do not store post-game Elo).
      - Compute a rolling momentum measure for each team based on the previous game’s delta,
        as a 5-game rolling average (shifted so that for the current game, only past games are used).
      
    Notes:
      - This function does not retain any current-game "EloAfter" values.
      - Final regular-season Elo and elo_minus_seedAvgElo are not computed.
      - It is critical that the computation is done in strict chronological order.
    """
    
    # Work on a copy
    df = df.copy()
    
    # Sort by Season and DayNum
    df = df.sort_values(by=["Season", "DayNum"]).reset_index(drop=True)
    
    # Initialize dictionaries for each team (keyed by (Season, team))
    current_elo = {}       # current Elo rating (pre-game)
    last_delta = {}        # Elo delta from the previous game (first derivative)
    last_delta2 = {}       # Elo change-of-change from the previous game (second derivative)
    
    # Create new columns to store pre-game Elo and shifted delta features.
    df["WTeamEloBefore"] = np.nan
    df["LTeamEloBefore"] = np.nan
    df["WTeamEloDelta_lastGame"] = np.nan
    df["LTeamEloDelta_lastGame"] = np.nan
    df["WTeamEloDelta2_lastGame"] = np.nan
    df["LTeamEloDelta2_lastGame"] = np.nan
    
    # We'll compute Elo for each game. For the game outcome,
    # assume that the winner gets score=1 and the loser 0.
    # (This is used to update Elo, but note: we never store the current game's delta.)
    
    for i, row in df.iterrows():
        season = row["Season"]
        wteam = row["WTeamID"]
        lteam = row["LTeamID"]
        
        # Initialize team data if not seen before in this season:
        if (season, wteam) not in current_elo:
            current_elo[(season, wteam)] = base_elo
            last_delta[(season, wteam)] = np.nan  # no prior game, so set to NaN
            last_delta2[(season, wteam)] = np.nan
        if (season, lteam) not in current_elo:
            current_elo[(season, lteam)] = base_elo
            last_delta[(season, lteam)] = np.nan
            last_delta2[(season, lteam)] = np.nan
        
        # Retrieve pre-game Elo values:
        w_elo_before = current_elo[(season, wteam)]
        l_elo_before = current_elo[(season, lteam)]
        df.at[i, "WTeamEloBefore"] = w_elo_before
        df.at[i, "LTeamEloBefore"] = l_elo_before
        
        # *** BEFORE updating Elo for this game, store the previous game's deltas ***
        # For the winning team, store its last known delta and delta2:
        df.at[i, "WTeamEloDelta_lastGame"] = last_delta.get((season, wteam), np.nan)
        df.at[i, "WTeamEloDelta2_lastGame"] = last_delta2.get((season, wteam), np.nan)
        # For the losing team:
        df.at[i, "LTeamEloDelta_lastGame"] = last_delta.get((season, lteam), np.nan)
        df.at[i, "LTeamEloDelta2_lastGame"] = last_delta2.get((season, lteam), np.nan)
        
        # Compute expected outcome using Elo ratings:
        exp_w = 1.0 / (1.0 + 10.0 ** ((l_elo_before - w_elo_before) / 400.0))
        # For consistency, exp_l = 1 - exp_w, but not used separately
        
        # Actual outcomes: winner gets 1, loser gets 0.
        score_w, score_l = 1.0, 0.0
        
        # Update Elo: (these are the new Elo ratings after the game)
        new_w_elo = w_elo_before + k_factor * (score_w - exp_w)
        new_l_elo = l_elo_before + k_factor * (score_l - (1 - exp_w))
        
        # Compute current game’s delta (first derivative)
        w_delta = new_w_elo - w_elo_before
        l_delta = new_l_elo - l_elo_before
        
        # Compute current game’s delta2: change in delta relative to previous delta.
        # If previous delta is NaN, then delta2 remains NaN.
        w_delta2 = w_delta - (last_delta.get((season, wteam)) if pd.notnull(last_delta.get((season, wteam)) ) else np.nan)
        l_delta2 = l_delta - (last_delta.get((season, lteam)) if pd.notnull(last_delta.get((season, lteam)) ) else np.nan)
        
        # Now update the dictionaries for next game:
        current_elo[(season, wteam)] = new_w_elo
        current_elo[(season, lteam)] = new_l_elo
        
        last_delta[(season, wteam)] = w_delta
        last_delta[(season, lteam)] = l_delta
        
        last_delta2[(season, wteam)] = w_delta2
        last_delta2[(season, lteam)] = l_delta2
    
    # 5) Compute rolling momentum based on the shifted EloDelta.
    #    We use the shifted delta columns (i.e. the previous game's delta) to compute a rolling average.
    # First, create a long-format DataFrame for team games:
    df_wgames = df[["Season","DayNum","WTeamID","WTeamEloDelta_lastGame"]].copy()
    df_wgames.rename(columns={"WTeamID": "TeamID", "WTeamEloDelta_lastGame": "EloDelta_lastGame"}, inplace=True)
    
    df_lgames = df[["Season","DayNum","LTeamID","LTeamEloDelta_lastGame"]].copy()
    df_lgames.rename(columns={"LTeamID": "TeamID", "LTeamEloDelta_lastGame": "EloDelta_lastGame"}, inplace=True)
    
    df_long = pd.concat([df_wgames, df_lgames], ignore_index=True)
    df_long = df_long.sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)
    
    # Now compute a 5-game rolling average of the previous delta (momentum),
    # ensuring that we only consider games before the current one.
    df_long["Momentum_5"] = df_long.groupby(["Season", "TeamID"])["EloDelta_lastGame"] \
                                   .transform(lambda x: x.shift(0).rolling(window=window, min_periods=1).mean())
    # Note: Here, because EloDelta_lastGame is already the previous game's delta,
    # we do not need an additional shift.
    
    # 6) Merge rolling momentum back into the original df for winners and losers.
    df = df.merge(
        df_long[["Season","TeamID","DayNum","Momentum_5"]],
        how="left",
        left_on=["Season","WTeamID","DayNum"],
        right_on=["Season","TeamID","DayNum"]
    )
    df.rename(columns={"Momentum_5": "WTeamMomentum_5"}, inplace=True)
    df.drop(columns="TeamID", inplace=True)
    
    df = df.merge(
        df_long[["Season","TeamID","DayNum","Momentum_5"]],
        how="left",
        left_on=["Season","LTeamID","DayNum"],
        right_on=["Season","TeamID","DayNum"]
    )
    df.rename(columns={"Momentum_5": "LTeamMomentum_5"}, inplace=True)
    df.drop(columns="TeamID", inplace=True)
    
    # 7) Finally, drop columns we do not want in the final output.
    #    Remove any "EloAfter" columns, and any features that could leak the current game's info.
    drop_cols_final = ["WTeamEloAfter", "LTeamEloAfter"]
    df.drop(columns=drop_cols_final, errors="ignore", inplace=True)
    
    # Also, we no longer want to include any final regular-season Elo or elo_minus_seedAvgElo features.
    drop_more = ["WTeam_regSeasonEnd", "LTeam_regSeasonEnd", "Lowerelo_minus_seedAvgElo", "Higherelo_minus_seedAvgElo"]
    df.drop(columns=drop_more, errors="ignore", inplace=True)
    
    return df

# Example usage:
# df_with_elo = add_elo_features_with_rolling_momentum(df, base_elo=1500, k_factor=20)
# df_with_elo.head()



df_with_elo = add_elo_features_with_rolling_momentum(df, base_elo=1500, k_factor = 28)
df_with_elo.head()


df = df_with_elo.copy()
df = df.sort_values(by=["Season", "DayNum"]).reset_index(drop=True)


df.columns


import pandas as pd

def merge_sel_ordinals_onto_games(df_games, df_massey):
    """
    Merge only the 'SEL' system's OrdinalRank from df_massey onto df_games.
    
    df_games : DataFrame with columns [Season, DayNum, WTeamID, LTeamID, ...]
    df_massey: DataFrame with columns [Season, RankingDayNum, SystemName, TeamID, OrdinalRank]
               containing multiple systems.
               
    Returns
    -------
    A merged DataFrame with two extra columns: WOrdinalRank and LOrdinalRank (for SEL).
    """
    # 1) Filter for SEL only
    df_sel = df_massey[df_massey["SystemName"] == "SEL"].copy()
    
    # 2) Create a long version of df_games
    w_part = df_games[["Season","DayNum","WTeamID"]].rename(
        columns={"WTeamID":"TeamID"}
    )
    w_part["is_winner"] = True
    
    l_part = df_games[["Season","DayNum","LTeamID"]].rename(
        columns={"LTeamID":"TeamID"}
    )
    l_part["is_winner"] = False
    
    df_long = pd.concat([w_part, l_part], ignore_index=True)
    
    # Sort for asof merges
    df_long = df_long.sort_values(["Season","TeamID","DayNum"])
    df_sel = df_sel.sort_values(["Season","TeamID","RankingDayNum"])
    
    merged_list = []
    
    # 3) Groupwise merge_asof for each (Season, TeamID)
    for (season, teamid), group_games in df_long.groupby(["Season","TeamID"]):
        # Slice SEL data for that (season, team)
        mask = (df_sel["Season"] == season) & (df_sel["TeamID"] == teamid)
        group_sel = df_sel.loc[mask, ["RankingDayNum","OrdinalRank"]].copy()
        group_sel = group_sel.rename(columns={"RankingDayNum":"DayNum"})
        
        # Sort
        group_games = group_games.sort_values("DayNum")
        group_sel = group_sel.sort_values("DayNum")
        
        # Merge asof
        merged = pd.merge_asof(
            group_games,
            group_sel,
            on="DayNum",
            direction="backward"
        )
        merged_list.append(merged)
    
    df_long_merged = pd.concat(merged_list, ignore_index=True)
    
    # Now df_long_merged has [Season, TeamID, DayNum, is_winner, OrdinalRank]
    # 4) Pivot back to w/l columns
    w_sel = df_long_merged[df_long_merged["is_winner"]==True][
        ["Season","DayNum","TeamID","OrdinalRank"]
    ].rename(columns={
        "TeamID":"WTeamID",
        "OrdinalRank":"WOrdinalRank_SEL"
    })
    
    l_sel = df_long_merged[df_long_merged["is_winner"]==False][
        ["Season","DayNum","TeamID","OrdinalRank"]
    ].rename(columns={
        "TeamID":"LTeamID",
        "OrdinalRank":"LOrdinalRank_SEL"
    })
    
    # Merge back with df_games
    df_games = df_games.merge(w_sel, how="left", on=["Season","DayNum","WTeamID"])
    df_games = df_games.merge(l_sel, how="left", on=["Season","DayNum","LTeamID"])
    
    return df_games


# Example usage:
# df_with_sel = merge_sel_ordinals_onto_games(df_games, MMasseyOrdinals)
# df_with_sel.head()



df_with_sel = merge_sel_ordinals_onto_games(df, MMasseyOrdinals)


df = df_with_sel.copy()


import pandas as pd
import numpy as np

def create_deep_exponential_decay_features(df, alpha=0.2):
    """
    Takes in a DataFrame 'df' with columns (at least):
      Season, DayNum,
      WTeamID, WScore, WFGM, WFGA, WFGM3, WFGA3, WFTM, WFTA, WOR, WDR, WAst, WTO, WStl, WBlk, WPF,
      LTeamID, LScore, LFGM, LFGA, LFGM3, LFGA3, LFTM, LFTA, LOR, LDR, LAst, LTO, LStl, LBlk, LPF, ...
    and returns a DataFrame with new columns for each team’s exponentially weighted stats.

    We go "deep" by computing many stats from the columns, such as:
      - Basic raw totals (Score, FGM, FGA, etc.)
      - Shooting percentages (FGpct, 2Ppct, 3Ppct, eFGpct, FTpct)
      - Distribution metrics (Shots2Dist, Shots3Dist, FTrate)
      - Rebounding & sum: OR, DR, total Rebounds (Reb)
      - Assists/TO ratio, steals, blocks, fouls
      - Possessions & offensive rating (OffRtg)
    Then we create an EWM version of each of these with alpha=0.2, shifted by 1 game to prevent leakage.
    """

    # ---- 1) LONG-FORM CONVERSION (WINNER -> single row, LOSER -> single row) ----
    df_w = df[[
        "Season","DayNum","WTeamID","WScore","WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA",
        "WOR","WDR","WAst","WTO","WStl","WBlk","WPF"
    ]].copy()
    df_w.rename(columns={
        "WTeamID":"TeamID",
        "WScore":"Score",
        "WFGM":"FGM",
        "WFGA":"FGA",
        "WFGM3":"FGM3",
        "WFGA3":"FGA3",
        "WFTM":"FTM",
        "WFTA":"FTA",
        "WOR":"OR",
        "WDR":"DR",
        "WAst":"Ast",
        "WTO":"TO",
        "WStl":"Stl",
        "WBlk":"Blk",
        "WPF":"PF"
    }, inplace=True)
    df_w["is_winner"] = 1

    df_l = df[[
        "Season","DayNum","LTeamID","LScore","LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA",
        "LOR","LDR","LAst","LTO","LStl","LBlk","LPF"
    ]].copy()
    df_l.rename(columns={
        "LTeamID":"TeamID",
        "LScore":"Score",
        "LFGM":"FGM",
        "LFGA":"FGA",
        "LFGM3":"FGM3",
        "LFGA3":"FGA3",
        "LFTM":"FTM",
        "LFTA":"FTA",
        "LOR":"OR",
        "LDR":"DR",
        "LAst":"Ast",
        "LTO":"TO",
        "LStl":"Stl",
        "LBlk":"Blk",
        "LPF":"PF"
    }, inplace=True)
    df_l["is_winner"] = 0

    df_long = pd.concat([df_w, df_l], ignore_index=True)
    df_long.sort_values(["Season","TeamID","DayNum"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    # ---- 2) COMPUTE PER-GAME STATS ----
    # We'll create a bunch of columns. We must handle division-by-zero carefully.

    # Basic raw columns are already there: Score, FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF.

    # 2-pt Shots:
    df_long["FGM2"] = df_long["FGM"] - df_long["FGM3"]
    df_long["FGA2"] = df_long["FGA"] - df_long["FGA3"]

    # Shooting percentages:
    df_long["FGpct"] = np.where(df_long["FGA"] != 0, df_long["FGM"] / df_long["FGA"], np.nan)
    df_long["FG2pct"] = np.where(df_long["FGA2"] != 0, df_long["FGM2"] / df_long["FGA2"], np.nan)
    df_long["FG3pct"] = np.where(df_long["FGA3"] != 0, df_long["FGM3"] / df_long["FGA3"], np.nan)
    df_long["eFGpct"] = np.where(df_long["FGA"] != 0,
                                 (df_long["FGM"] + 0.5 * df_long["FGM3"]) / df_long["FGA"],
                                 np.nan)
    df_long["FTpct"] = np.where(df_long["FTA"] != 0, df_long["FTM"] / df_long["FTA"], np.nan)

    # Distributions (what fraction of attempts are 2 vs. 3 vs. FT?):
    df_long["Shots2Dist"] = np.where(df_long["FGA"] != 0, df_long["FGA2"] / df_long["FGA"], np.nan)
    df_long["Shots3Dist"] = np.where(df_long["FGA"] != 0, df_long["FGA3"] / df_long["FGA"], np.nan)
    df_long["FTrate"]     = np.where(df_long["FGA"] != 0, df_long["FTA"] / df_long["FGA"], np.nan)

    # Rebounding:
    df_long["Reb"] = df_long["OR"] + df_long["DR"]

    # Assist/TO ratio:
    df_long["AstTO"] = np.where(df_long["TO"] != 0, df_long["Ast"] / df_long["TO"], np.nan)

    # Possessions (an approximation):
    # Standard formula: poss = FGA + 0.475*FTA + TO - OR
    # or sometimes FGA - OR + TO + 0.475*FTA
    # We'll pick one approach:
    df_long["Poss"] = df_long["FGA"] + 0.475*df_long["FTA"] + df_long["TO"] - df_long["OR"]
    # Offensive Rating (points scored per 100 possessions)
    df_long["OffRtg"] = np.where(df_long["Poss"] > 0, (df_long["Score"] / df_long["Poss"]) * 100, np.nan)

    # Potentially we can do raw per-game stats for Stl, Blk, PF, OR, DR, etc. 
    # We'll do EWM on them too.

    # ---- 3) DETERMINE WHICH STATS TO APPLY EWM ON ----
    # We'll go fairly broad to "touch every purposeful column" from the dataset:
    stats_to_ewm = [
        # raw
        "Score", "FGM", "FGA", "FGM2", "FGA2", "FGM3", "FGA3", "FTM", "FTA",
        "OR", "DR", "Reb", "Ast", "TO", "Stl", "Blk", "PF", 
        # rates & percentages
        "FGpct", "FG2pct", "FG3pct", "eFGpct", "FTpct",
        "Shots2Dist", "Shots3Dist", "FTrate", "AstTO",
        # advanced
        "Poss", "OffRtg"
    ]

    # ---- 4) EXPONENTIAL WEIGHTED MEAN (EWM) WITH 1-GAME SHIFT PER (Season, TeamID) ----
    def ewm_transform(group):
        group = group.copy()
        for col in stats_to_ewm:
            # SHIFT by 1 so we don't leak current-game info
            shifted = group[col].shift(1)
            # EWM with alpha=alpha
            ewm_series = shifted.ewm(alpha=alpha, adjust=False).mean()
            group[col + "_ewm"] = ewm_series
        return group

    df_long = df_long.groupby(["Season","TeamID"], group_keys=False).apply(ewm_transform)

    # ---- 5) MERGE BACK INTO THE ORIGINAL WIDE DF ----
    # We'll keep just the final EWM columns + (Season, DayNum, TeamID, is_winner)
    ewm_cols = [c + "_ewm" for c in stats_to_ewm]
    merge_cols = ["Season","TeamID","DayNum","is_winner"] + ewm_cols

    # Winner side
    df_long_w = df_long[df_long["is_winner"]==1][merge_cols].rename(columns={"TeamID":"WTeamID"})
    # rename "Score_ewm" -> "WScore_ewm", "FGpct_ewm" -> "WFGpct_ewm", etc.
    for c in ewm_cols:
        new_name = "W" + c.replace("_ewm","") + "_ewm"  # e.g. WScore_ewm
        df_long_w.rename(columns={c: new_name}, inplace=True)

    # Loser side
    df_long_l = df_long[df_long["is_winner"]==0][merge_cols].rename(columns={"TeamID":"LTeamID"})
    for c in ewm_cols:
        new_name = "L" + c.replace("_ewm","") + "_ewm"
        df_long_l.rename(columns={c: new_name}, inplace=True)

    # Merge them back
    df_merged = df.merge(
        df_long_w.drop(columns=["is_winner"]),
        how="left",
        on=["Season","DayNum","WTeamID"]
    )
    df_merged = df_merged.merge(
        df_long_l.drop(columns=["is_winner"]),
        how="left",
        on=["Season","DayNum","LTeamID"]
    )

    return df_merged

# EXAMPLE USAGE:
# df_deep = create_deep_exponential_decay_features(df, alpha=0.2)
# 
# - This will create a large number of new columns:
#   WScore_ewm, WFGpct_ewm, WFG2pct_ewm, WFG3pct_ewm, WShots2Dist_ewm, WShots3Dist_ewm, ...
#   plus the same for LTeam (LScore_ewm, LFGpct_ewm, etc.).
# - They are all based on exponential smoothing of the team's prior games,
#   with alpha=0.2 weighting recent games more heavily.
# 
# You can then feed these columns into your predictive model, 
# alongside Elo, seeds, Massey Ordinals, etc.



df_deep = create_deep_exponential_decay_features(df, alpha=0.2)


df = df_deep.copy()


import numpy as np
import pandas as pd

def label_lower_id_team(df):
    df = df.copy()
    # Identify which is lower/higher ID
    df["LowerTeamID"] = df[["WTeamID","LTeamID"]].min(axis=1)
    df["HigherTeamID"] = df[["WTeamID","LTeamID"]].max(axis=1)
    
    # LowerTeamWon = 1 if the lower ID was the winner
    # If WTeamID == LowerTeamID, that means the lower ID is the winner
    df["LowerTeamWon"] = np.where(df["WTeamID"] == df["LowerTeamID"], 1, 0)
    
    return df

df_final = label_lower_id_team(df)



df_final.columns


import numpy as np
import pandas as pd

def map_wl_columns_to_lower_higher(df, skip_cols=None):
    """
    For each column that starts with 'W' or 'L', 
    create corresponding 'Lower___' and 'Higher___' columns,
    depending on whether WTeamID == LowerTeamID or not.

    Params
    ------
    df : pd.DataFrame
        Must include columns:
          - WTeamID, LTeamID (the team IDs)
          - LowerTeamID, HigherTeamID (from a prior step)
          - 'Wxxx' and 'Lxxx' columns that you want mapped
    skip_cols : set or list of columns to ignore
        e.g. {'WTeamID','LTeamID','WLoc','WSeed',...} 
        (defaults to None, which means skip just the IDs automatically)

    Returns
    -------
    df_out : pd.DataFrame
        A copy of df with new columns:
        Lower___, Higher___ for each matching pair (Wxxx, Lxxx).
    """
    df = df_final.copy()
    if skip_cols is None:
        # By default, skip columns that are obviously ID or labeling
        skip_cols = set(["WTeamID","LTeamID","LowerTeamID","HigherTeamID"])
    else:
        skip_cols = set(skip_cols).union(["WTeamID","LTeamID","LowerTeamID","HigherTeamID"])

    # We'll collect new columns in a dict, then assign at the end
    new_cols = {}

    # 1) Identify all columns that start with 'W'
    w_cols = [c for c in df.columns if c.startswith("W") and c not in skip_cols]
    
    for w_col in w_cols:
        # Derive the corresponding L-column name by replacing leading 'W' with 'L'
        base_name = w_col[1:]  # e.g. if w_col = 'WFGpct_ewm', base_name = 'FGpct_ewm'
        l_col = "L" + base_name
        
        # Only proceed if the L-column also exists
        if l_col not in df.columns:
            continue  # skip if there's no matching L-col

        # We'll create "Lower{base_name}" and "Higher{base_name}"
        lower_col = "Lower" + base_name
        higher_col = "Higher" + base_name

        # Construct the new columns by checking if WTeamID == LowerTeamID
        # For each row:
        #   if WTeam is the lower ID, then Lower{base_name} = df[w_col], Higher{base_name} = df[l_col]
        #   else Lower{base_name} = df[l_col], Higher{base_name} = df[w_col]
        
        lower_vals = np.where(
            df["WTeamID"] == df["LowerTeamID"],
            df[w_col],
            df[l_col]
        )
        higher_vals = np.where(
            df["WTeamID"] == df["LowerTeamID"],
            df[l_col],
            df[w_col]
        )

        new_cols[lower_col] = lower_vals
        new_cols[higher_col] = higher_vals

    # 2) Assign these new columns to the DataFrame
    for col_name, vals in new_cols.items():
        df[col_name] = vals

    return df

# Example usage:
# df_final = label_lower_id_team(df_final)  # Must have LowerTeamID, HigherTeamID
# df_final = map_wl_columns_to_lower_higher(df_final, skip_cols=["WLoc","LLoc","WScore","LScore"])
# 
# This will produce new columns like LowerEloBefore, HigherEloBefore, LowerFGpct_ewm, HigherFGpct_ewm, etc.



df_last = map_wl_columns_to_lower_higher(df_final, skip_cols=None)


df_2025 = df_last[df_last['Season'] == 2025]


print(df_last.columns.tolist())



import re
import pandas as pd

def drop_non_safe_columns(df):
    """
    Aggressively drops columns that might leak outcome information, while ensuring
    that the following critical columns are always kept:
      - 'Season', 'DayNum', 'WTeamMomentum_5', 'LTeamMomentum_5', 
        'LowerTeamID', 'HigherTeamID', 'LowerTeamWon'
    
    Additionally, we allow columns that match safe patterns:
      - Seed columns and missing indicators (e.g. WSeed, LSeed, LowerSeed, HigherSeed, WSeed_missing, LSeed_missing)
      - Pre-game Elo ratings (WTeamEloBefore, LTeamEloBefore)
      - Shifted Elo deltas and delta2 (columns ending with '_lastGame')
      - Rolling momentum features (columns ending with '_Momentum_5')
      - Ordinal ranks (WOrdinalRank_SEL, LOrdinalRank_SEL, LowerOrdinalRank_SEL, HigherOrdinalRank_SEL)
      - Exponential weighted metrics (any column ending with '_ewm')
    
    Any column not matching these allowed patterns will be dropped.
    """
    # Explicit columns we want to always keep
    always_keep = {"Season", "DayNum", "WTeamMomentum_5", "LTeamMomentum_5", "LowerTeamID", "HigherTeamID", "LowerTeamWon",'WTeamID', "LTeamID"}
    
    # Regex pattern for columns we consider safe.
    allowed_pattern = re.compile(
        r'^(WSeed|LSeed|LowerSeed|HigherSeed|WSeed_missing|LSeed_missing|'
        r'WTeamEloBefore|LTeamEloBefore|'
        r'.*_lastGame$|'           # shifted deltas, etc.
        r'.*_Momentum_5$|'         # rolling momentum features
        r'(WOrdinalRank_SEL|LOrdinalRank_SEL|LowerOrdinalRank_SEL|HigherOrdinalRank_SEL)|'
        r'.*_ewm$'                # exponential weighted metrics
        r')'
    )
    
    # Build the list of allowed columns.
    keep_cols = []
    for col in df.columns:
        if col in always_keep:
            keep_cols.append(col)
        elif allowed_pattern.match(col):
            keep_cols.append(col)
    
    # For debugging, show which columns will be dropped.
    dropped_cols = [col for col in df.columns if col not in keep_cols]
    print("Allowed columns:", keep_cols)
    print("Dropped columns:", dropped_cols)
    
    return df[keep_cols].copy()

# Example usage:
# Assuming df is your DataFrame after feature engineering.
df_safe = drop_non_safe_columns(df_last)
print("Final columns after aggressive drop:")
print(df_safe.columns.tolist())



df = df_safe.copy()


# For the winning team seed:
df["WSeed_missing"] = df["WSeed"].isna().astype(int)
df["WSeed"] = df["WSeed"].fillna(17)

# For the losing team seed:
df["LSeed_missing"] = df["LSeed"].isna().astype(int)
df["LSeed"] = df["LSeed"].fillna(17)



import pandas as pd
import numpy as np

def parse_seed(seed_val):
    """
    Converts a seed string like 'W01', 'X16b', 'Z28', or '17' into (region, seedNum).
    region is 1..4 if W/X/Y/Z, else NaN.
    seedNum is 1..16 if valid, else NaN.
    'a'/'b' suffix is ignored, and if the seed >16 or format is off, seedNum=NaN.
    '17' => (NaN, NaN). 'Z28' => (4, NaN). 'X16b' => (2, 16).
    """
    if pd.isna(seed_val):
        return (np.nan, np.nan)

    seed_str = str(seed_val).strip().upper()
    # Region map
    region_map = {'W': 1, 'X': 2, 'Y': 3, 'Z': 4}

    # 1) Check first character for region
    first_char = seed_str[0]  # e.g. 'W' in 'W01'
    if first_char in region_map:
        region_code = region_map[first_char]
        # Next two chars are the digits
        seed_digits = seed_str[1:3]  # e.g. '01' in 'W01'
    else:
        # No valid region letter => region=NaN
        region_code = np.nan
        # Possibly the entire string is numeric, e.g. '17'
        # We'll take first 2 chars for seed digits
        seed_digits = seed_str[:2]

    # 2) Convert seed_digits to int
    try:
        seed_num = int(seed_digits)
        # If seed_num > 16 => treat as invalid
        if seed_num > 16:
            seed_num = 17
    except ValueError:
        seed_num = np.nan

    return (region_code, seed_num)



def parse_seeds_into_numeric(df):
    df = df.copy()
    
    # Make new columns for winning seed
    df["WRegion"], df["WSeedNum"] = zip(*df["WSeed"].apply(parse_seed))
    
    # Make new columns for losing seed
    df["LRegion"], df["LSeedNum"] = zip(*df["LSeed"].apply(parse_seed))
    
    return df

df_parsed = parse_seeds_into_numeric(df)



df = df_parsed.drop(columns = ['WSeed', 'LSeed', 'WRegion', 'LRegion', 'LowerSeed', 'HigherSeed'])



print(df.columns.to_list())


import pandas as pd
import numpy as np
import itertools
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

##############################################################################
# 1) PREPARE THE FULL DATAFRAME "df" (Assumed already created)
#    df contains seasons from 2003 to 2025 with feature-engineered columns.
#    It includes, for example, columns:
#       'Season', 'DayNum', 'WTeamEloBefore', 'LTeamEloBefore', 
#       'WTeamEloDelta_lastGame', 'LTeamEloDelta_lastGame', ...,
#       and (if still present) the original team IDs.
##############################################################################

# ---------------------------------------------------------------------
# TRAINING: Split historical data and train the model
# We'll train on seasons <= 2021 and test on 2022–2024.
# ---------------------------------------------------------------------
train_df = df[df["Season"] <= 2021].copy()
test_df  = df[(df["Season"] > 2021) & (df["Season"] < 2025)].copy()

# Define columns to drop from features (we want only numeric stats)
drop_cols = ["LowerTeamWon", "Season", "DayNum", "WLoc",
             "WTeamID", "LTeamID", "TeamID", "LowerTeamID", "HigherTeamID",
             "WSeed_missing", "LSeed_missing", "WSeedNum", "LSeedNum"]

X_train = train_df.drop(columns=drop_cols, errors="ignore")
y_train = train_df["LowerTeamWon"]

X_test = test_df.drop(columns=drop_cols, errors="ignore")
y_test = test_df["LowerTeamWon"]

# Replace pd.NA with np.nan and impute missing values with a sentinel (-999)
X_train = X_train.replace({pd.NA: np.nan})
X_test  = X_test.replace({pd.NA: np.nan})
imputer = SimpleImputer(strategy="constant", fill_value=-999)
X_train_filled = pd.DataFrame(imputer.fit_transform(X_train),
                              columns=X_train.columns,
                              index=X_train.index)
X_test_filled = pd.DataFrame(imputer.transform(X_test),
                             columns=X_test.columns,
                             index=X_test.index)

# Align columns between train and test sets
X_train_encoded, X_test_encoded = X_train_filled.align(X_test_filled, join='outer', axis=1, fill_value=0)

# Check dtypes (they should all be numeric)
print("Dtypes in X_train_encoded:\n", X_train_encoded.dtypes)

# Train the RandomForest model
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train_encoded, y_train)

# Evaluate on historical test set
y_pred = rf.predict(X_test_encoded)
y_prob = rf.predict_proba(X_test_encoded)[:, 1]
print("Historical Test Accuracy:", accuracy_score(y_test, y_pred))
print("Historical Test Brier Score:", brier_score_loss(y_test, y_prob))

##############################################################################
# 2) EXTRACT THE 2025 DATA
##############################################################################
df_2025 = df[df["Season"] == 2025].copy()
# Do NOT drop the original team IDs from df_2025— we need them to assign team features.
# (We assume df_2025 still contains columns like 'WTeamEloBefore', 'LTeamEloBefore', etc.)

##############################################################################
# 3) CREATE A FUNCTION TO EXTRACT THE TEAM'S PRE-TOURNAMENT STATS
##############################################################################
def get_team_pre_tournament_stats(team_id, df_2025, max_day=132):
    """
    For a given team in 2025, find its most recent game (with DayNum <= max_day)
    and return a unified feature vector (dictionary) for that team.
    
    We use the fact that in df_2025, a team appears in either WTeam* columns (if it was the winner)
    or in LTeam* columns (if it was the loser). This function checks which role the team played in its
    last game and returns that row’s stats.
    """
    df_team = df_2025[
        (df_2025["DayNum"] <= max_day) &
        (((df_2025["WTeamID"] == team_id)) | ((df_2025["LTeamID"] == team_id)))
    ]
    if df_team.empty:
        raise ValueError(f"No 2025 games found for team {team_id} up to DayNum {max_day}.")
    df_team = df_team.sort_values("DayNum")
    last_game = df_team.iloc[-1]
    
    # If the team appears as the winner, return the "W..." stats; if as the loser, return the "L..." stats.
    if last_game["WTeamID"] == team_id:
        # Build a dict for all training feature columns starting with "W"
        features = {col: last_game[col] for col in last_game.index if col.startswith("W")}
    elif last_game["LTeamID"] == team_id:
        features = {col: last_game[col] for col in last_game.index if col.startswith("L")}
    else:
        raise ValueError("Team not found in last game row.")
    return features

##############################################################################
# 4) BUILD A DICTIONARY OF TEAM FEATURES FOR 2025
##############################################################################
# Use the original team IDs from df_2025. If your final df_2025 has both WTeamID and LTeamID
# removed by mapping, then you must use the LowerTeamID/HigherTeamID columns.
# Here, we assume df_2025 still contains WTeamID and LTeamID.
teams_2025 = set(df_2025["WTeamID"]).union(df_2025["LTeamID"])
teams_2025 = sorted(list(teams_2025))
print("Number of teams in 2025:", len(teams_2025))

team_stats = {}
for team in teams_2025:
    try:
        team_stats[team] = get_team_pre_tournament_stats(team, df_2025, max_day=132)
    except ValueError as e:
        print(f"Warning: {e}")

##############################################################################
# 5) GENERATE ALL POSSIBLE 2025 MATCHUPS
##############################################################################
# We need to predict for every possible pairing (round-robin) among the 2025 teams.
matchups = []
for t1, t2 in itertools.combinations(teams_2025, 2):
    matchup_id = f"2025_{t1:04d}_{t2:04d}"
    matchups.append({"ID": matchup_id, "LowerTeamID": t1, "HigherTeamID": t2})
df_matchups = pd.DataFrame(matchups)
print("Number of matchups:", len(df_matchups))

##############################################################################
# 6) BUILD MATCHUP FEATURE VECTORS & MAKE PREDICTIONS
##############################################################################
# IMPORTANT: The model was trained on features with names as in X_train_encoded.
# We assume that those training columns include, for example, 'WTeamEloBefore', 'LTeamEloBefore', etc.
# For each matchup, we want to create a row with the same columns:
#   - For the lower team (by team id), we use its pre-tournament stats from get_team_pre_tournament_stats
#     and assign them to the "W..." columns.
#   - For the higher team, we use its pre-tournament stats and assign them to the "L..." columns.
# This creates a synthetic "133rd game" row that mimics the training data format.

feature_order = X_train_encoded.columns.tolist()

submission_rows = []
for _, row in df_matchups.iterrows():
    t1 = row["LowerTeamID"]   # lower team id
    t2 = row["HigherTeamID"]  # higher team id
    match_id = row["ID"]
    
    if t1 not in team_stats or t2 not in team_stats:
        continue
    
    # For the matchup row, we assign:
    # - All training columns starting with "W" come from team t1's stats.
    # - All training columns starting with "L" come from team t2's stats.
    matchup_features = {}
    for col in feature_order:
        if col.startswith("W"):
            # Use lower team (t1)
            matchup_features[col] = team_stats[t1].get(col, np.nan)
        elif col.startswith("L"):
            matchup_features[col] = team_stats[t2].get(col, np.nan)
        else:
            # For any column not following the convention, set to a default value
            matchup_features[col] = np.nan
            
    # Convert to DataFrame row and reindex to ensure same order as training features
    df_matchup_feats = pd.DataFrame([matchup_features])
    df_matchup_feats = df_matchup_feats.reindex(columns=feature_order, fill_value=np.nan)
    df_matchup_feats = df_matchup_feats.fillna(-999)
    
    # Predict probability that the lower team wins (class 1)
    prob_lower_wins = rf.predict_proba(df_matchup_feats)[:, 1][0]
    submission_rows.append({"ID": match_id, "Pred": prob_lower_wins})

df_submission = pd.DataFrame(submission_rows, columns=["ID", "Pred"])

##############################################################################
# 7) OPTIONAL: CLIP EXTREME PROBABILITIES
##############################################################################
df_submission.loc[df_submission["Pred"] > 0.9, "Pred"] = 1.0
df_submission.loc[df_submission["Pred"] < 0.1, "Pred"] = 0.0

##############################################################################
# 8) WRITE THE SUBMISSION FILE
##############################################################################
df_submission.to_csv("submission.csv", index=False)
print("submission.csv created with", len(df_submission), "rows.")



df_submission.describe()




