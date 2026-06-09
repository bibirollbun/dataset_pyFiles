# This Python 3 environment comes with many analytics libraries installed

import numpy as np 
import pandas as pd 

# Input data files are available in the read-only "../input/" directory



# 1. Find all (game_id, play_id) combos that have at least one "Defensive Coverage" player
coverage_plays = (
    input_df[input_df['player_role'] == 'Defensive Coverage']
    [['game_id', 'play_id']]
    .drop_duplicates()
)

print("Number of plays with at least one coverage defender:", len(coverage_plays))

# 2. Filter supplementary plays to C/I passes with known route
candidate_plays = supp_df[
    (supp_df['pass_result'].isin(['C', 'I'])) &
    (~supp_df['route_of_targeted_receiver'].isna())
]

# 3. Keep only plays that ALSO have a coverage defender in tracking
candidate_plays = candidate_plays.merge(
    coverage_plays,
    on=['game_id', 'play_id'],
    how='inner'
)

print("Candidate plays meeting all conditions:", len(candidate_plays))

# 4. Pick the first such play as our new example
example_play = candidate_plays.iloc[0]
example_play



# Use the new example_play
gid = example_play['game_id']
pid = example_play['play_id']

print("Game ID:", gid)
print("Play ID:", pid)

# Filter tracking data for just this play
input_play = input_df[(input_df['game_id'] == gid) & (input_df['play_id'] == pid)]
output_play = output_df[(output_df['game_id'] == gid) & (output_df['play_id'] == pid)]

print("\nInput tracking rows for this play:", len(input_play))
print("Output tracking rows for this play:", len(output_play))

display(input_play.head())
display(output_play.head())



coverage_defenders = (
    input_play[input_play['player_role'] == 'Defensive Coverage']
    [['nfl_id', 'player_name']]
    .drop_duplicates()
)

coverage_defenders



import numpy as np

db_track = db_track.copy()
db_track['dx'] = db_track['x'].diff()
db_track['dy'] = db_track['y'].diff()

# Direction of movement in degrees (NaN for first frame)
db_track['dir_deg'] = np.degrees(np.arctan2(db_track['dy'], db_track['dx']))

display(db_track)



import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot(db_track['frame_id'], db_track['dir_deg'], marker='o')
plt.xlabel('Frame ID (ball in air)')
plt.ylabel('Movement direction (degrees)')
plt.title(f'Movement direction over time: {db_name}')
plt.grid(True)
plt.show()



# All player IDs available in output (ball-in-air) for this play
output_ids = output_play['nfl_id'].unique()

# Coverage defenders who ALSO appear in the output data
valid_defenders = (
    coverage_defenders[coverage_defenders['nfl_id'].isin(output_ids)]
    .reset_index(drop=True)
)

valid_defenders



# Pick the 3rd defender (index 2) from valid_defenders
chosen_defender = valid_defenders.iloc[2]

db_id = chosen_defender['nfl_id']
db_name = chosen_defender['player_name']

print("Chosen defender ID:", db_id)
print("Chosen defender name:", db_name)



# Tracking data for the chosen cornerback while the ball is in the air
db_track = (
    output_play[output_play['nfl_id'] == db_id]
    .sort_values('frame_id')
    [['frame_id', 'x', 'y']]
)

print("Total frames for this defender:", len(db_track))
display(db_track)



import numpy as np

db_track = db_track.copy()
db_track['dx'] = db_track['x'].diff()
db_track['dy'] = db_track['y'].diff()
db_track['dir_deg'] = np.degrees(np.arctan2(db_track['dy'], db_track['dx']))

display(db_track)



import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot(db_track['frame_id'], db_track['dir_deg'], marker='o')
plt.xlabel('Frame ID (ball in air)')
plt.ylabel('Movement direction (degrees)')
plt.title(f'Movement direction over time: {db_name}')
plt.grid(True)
plt.show()



db_track['dir_change'] = db_track['dir_deg'].diff()

display(db_track[['frame_id','dir_deg','dir_change']])



# Find the frame with the largest change in direction (hip flip candidate)
flip_idx = db_track['dir_change'].abs().idxmax()
flip_row = db_track.loc[flip_idx]

hip_flip_frame = int(flip_row['frame_id'])
hip_flip_change = float(flip_row['dir_change'])

print("Hip flip frame:", hip_flip_frame)
print("Change in direction at flip (degrees):", hip_flip_change)
display(flip_row)



first_frame = int(db_track['frame_id'].min())
latency_frames = hip_flip_frame - first_frame

print("First output frame:", first_frame)
print("Hip-flip latency (frames):", latency_frames)



import numpy as np

def compute_hip_flip_latency(output_df, game_id, play_id, defender_id, fps=10):
    """
    Compute hip-flip latency for one defender on one play.
    Uses only output (ball-in-air) tracking: frame_id, x, y.
    
    Returns a dict or None if not enough data.
    """
    # 1) Filter tracking for this defender on this play
    track = (
        output_df[
            (output_df['game_id'] == game_id) &
            (output_df['play_id'] == play_id) &
            (output_df['nfl_id'] == defender_id)
        ]
        .sort_values('frame_id')
        [['frame_id', 'x', 'y']]
    )
    
    # Need at least 3 frames to see a flip
    if len(track) < 3:
        return None
    
    # 2) Compute movement deltas and direction (like you did manually)
    track = track.copy()
    track['dx'] = track['x'].diff()
    track['dy'] = track['y'].diff()
    track['dir_deg'] = np.degrees(np.arctan2(track['dy'], track['dx']))
    track['dir_change'] = track['dir_deg'].diff()
    
    # If all dir_change are NaN (no movement), bail out
    if track['dir_change'].abs().max() == 0 or track['dir_change'].abs().isna().all():
        return None
    
    # 3) Hip flip = frame with largest absolute change in direction
    flip_idx = track['dir_change'].abs().idxmax()
    flip_frame = int(track.loc[flip_idx, 'frame_id'])
    first_frame = int(track['frame_id'].min())
    latency_frames = flip_frame - first_frame
    
    return {
        'game_id': int(game_id),
        'play_id': int(play_id),
        'defender_id': int(defender_id),
        'hip_flip_frame': flip_frame,
        'latency_frames': latency_frames,
        'latency_seconds': latency_frames / fps
    }



result = compute_hip_flip_latency(output_df, gid, pid, db_id)
result



results = []

for _, row in valid_defenders.iterrows():
    did = row['nfl_id']
    name = row['player_name']
    
    res = compute_hip_flip_latency(output_df, gid, pid, did)
    if res is not None:
        res['player_name'] = name
        results.append(res)

hfl_play_df = pd.DataFrame(results)
hfl_play_df



# Normalize latency within this play so lower = better, range 0–1
max_lat = hfl_play_df['latency_frames'].max()

hfl_play_df['HLS_play_norm'] = 1 - (hfl_play_df['latency_frames'] / max_lat)

# Sort best to worst
hfl_play_df_sorted = hfl_play_df.sort_values('HLS_play_norm', ascending=False)
hfl_play_df_sorted



all_output_dfs = []

# There are 18 weeks: w01, w02, ..., w18
for week in range(1, 19):
    filename = f'output_2023_w{week:02d}.csv'
    path = f'{base_path}/train/{filename}'
    print("Loading:", path)
    
    df_week = pd.read_csv(path)
    df_week['week'] = week  # optional: tag which week it came from
    all_output_dfs.append(df_week)

# Combine all weeks into one big DataFrame
output_all = pd.concat(all_output_dfs, ignore_index=True)

print("Total rows in all output tracking:", len(output_all))
output_all.head()



# We only need a few columns from the input tracking
input_cols = ['game_id', 'play_id', 'nfl_id', 'player_role', 'player_name', 'player_side']

all_input_dfs = []

for week in range(1, 19):
    filename = f'input_2023_w{week:02d}.csv'
    path = f'{base_path}/train/{filename}'
    print("Loading:", path)
    
    df_week = pd.read_csv(path, usecols=input_cols)
    df_week['week'] = week  # tag week for later use
    all_input_dfs.append(df_week)

# Combine all weeks into one big DataFrame
input_all = pd.concat(all_input_dfs, ignore_index=True)

print("Total rows in all input tracking (reduced cols):", len(input_all))
input_all.head()



# Filter only defensive coverage players from the input data
coverage_players = input_all[input_all['player_role'] == 'Defensive Coverage']

# Select only the important ID columns
coverage_players = coverage_players[['game_id', 'play_id', 'nfl_id', 'player_name', 'week']]

print("Total defensive coverage tracking rows:", len(coverage_players))

# Now find which of these appear in the output tracking
merged = coverage_players.merge(
    output_all[['game_id', 'play_id', 'nfl_id']].drop_duplicates(),
    on=['game_id', 'play_id', 'nfl_id'],
    how='inner'
)

print("Coverage defenders who have ball-in-air tracking:", len(merged))

merged.head()



print("len(output_all):", len(output_all))
output_all[['game_id','play_id']].drop_duplicates().head(10)



coverage_players = input_all[input_all['player_role'] == 'Defensive Coverage']
coverage_players = coverage_players[['game_id', 'play_id', 'nfl_id', 'player_name', 'week']]

print("Total defensive coverage rows:", len(coverage_players))

merged = coverage_players.merge(
    output_all[['game_id','play_id','nfl_id']].drop_duplicates(),
    on=['game_id','play_id','nfl_id'],
    how='inner'
)

print("Coverage defenders with ball-in-air tracking:", len(merged))
display(merged.head(10))



print("Number of unique defenders:", merged['nfl_id'].nunique())
print("Number of unique names:", merged['player_name'].nunique())

# Show a few random players from all over, not just the top
merged.sample(10)[['game_id','play_id','nfl_id','player_name','week']]



# One row per defender per play
pairs = merged[['game_id', 'play_id', 'nfl_id', 'player_name', 'week']].drop_duplicates()

print("Unique defender–play pairs:", len(pairs))
pairs.head()



# We'll test on the first N defender–play pairs
N = 500   # you can change this number later (e.g., 2000, 10000, etc.)

subset = pairs.head(N)

results = []

for idx, row in subset.iterrows():
    gid = row['game_id']
    pid = row['play_id']
    did = row['nfl_id']
    name = row['player_name']
    wk = row['week']
    
    res = compute_hip_flip_latency(output_all, gid, pid, did)
    
    if res is not None:
        res['player_name'] = name
        res['week'] = wk
        results.append(res)

hfl_results_sample = pd.DataFrame(results)

print("Rows with valid HFL:", len(hfl_results_sample))
hfl_results_sample.head()



# Add a simple per-play normalized score: lower latency = better (0–1)
hfl_results_sample['HLS_play_norm'] = 1 - (
    hfl_results_sample['latency_frames'] /
    hfl_results_sample['latency_frames'].max()
)

hfl_results_sample.sort_values('HLS_play_norm', ascending=False).head(10)



player_summary = (
    hfl_results_sample
    .groupby(['defender_id', 'player_name'], as_index=False)
    .agg(
        plays_with_hfl = ('latency_frames', 'count'),
        avg_latency_frames = ('latency_frames', 'mean'),
        avg_latency_seconds = ('latency_seconds', 'mean')
    )
)

# Lower latency = better. Let's convert to a 0–100 score within this sample.
max_lat = player_summary['avg_latency_frames'].max()
player_summary['HLS_0_100'] = 100 * (1 - player_summary['avg_latency_frames'] / max_lat)

# Sort best to worst (fastest hips at the top)
player_summary_sorted = player_summary.sort_values('HLS_0_100', ascending=False)

print("Players in this sample:", len(player_summary_sorted))
player_summary_sorted.head(10)



# Pick any player by name (must appear in player_summary_sorted)
target_name = "Trent McDuffie"   # change this to whoever you want

player_row = player_summary_sorted[player_summary_sorted['player_name'] == target_name]

player_row



if len(player_row) == 1:
    r = player_row.iloc[0]
    
    name = r['player_name']
    plays = int(r['plays_with_hfl'])
    avg_frames = r['avg_latency_frames']
    avg_seconds = r['avg_latency_seconds']
    score = r['HLS_0_100']
    
    print(f"---- HIP-FLIP LATENCY SCOUTING SUMMARY ----")
    print(f"Player: {name}")
    print(f"Plays measured: {plays}")
    print(f"Average hip-flip latency: {avg_seconds:.3f} seconds ({avg_frames:.2f} frames)")
    print(f"Hip-Flip Latency Score (HLS): {score:.1f} / 100")
    
    # Simple qualitative band
    if score > 85:
        tier = "Elite transition efficiency"
    elif score > 70:
        tier = "Above-average hip mobility"
    elif score > 50:
        tier = "Average transition speed"
    else:
        tier = "Below-average transition speed"
    
    print(f"Qualitative evaluation: {tier}")
else:
    print("Player not found or not unique.")



# One row per defender–play pair across the season
pairs = merged[['game_id', 'play_id', 'nfl_id', 'player_name', 'week']].drop_duplicates()

print("Unique defender–play pairs:", len(pairs))
pairs.head()



N = 2000   # you can change this later if you want
subset = pairs.head(N)

results = []

for idx, row in subset.iterrows():
    gid = row['game_id']
    pid = row['play_id']
    did = row['nfl_id']
    name = row['player_name']
    wk = row['week']
    
    res = compute_hip_flip_latency(output_all, gid, pid, did)
    
    if res is not None:
        res['player_name'] = name
        res['week'] = wk
        results.append(res)

hfl_results_sample = pd.DataFrame(results)

print("Rows with valid HFL:", len(hfl_results_sample))
hfl_results_sample.head()



player_summary = (
    hfl_results_sample
    .groupby(['defender_id', 'player_name'], as_index=False)
    .agg(
        plays_with_hfl = ('latency_frames', 'count'),
        avg_latency_frames = ('latency_frames', 'mean'),
        avg_latency_seconds = ('latency_seconds', 'mean')
    )
)

max_lat = player_summary['avg_latency_frames'].max()
player_summary['HLS_0_100'] = 100 * (1 - player_summary['avg_latency_frames'] / max_lat)

player_summary_sorted = player_summary.sort_values('HLS_0_100', ascending=False)

print("Players in this sample:", len(player_summary_sorted))
player_summary_sorted.head(10)





min_plays = 5  # change this later if you want (e.g., 10)

player_summary_filtered = (
    player_summary_sorted[player_summary_sorted['plays_with_hfl'] >= min_plays]
    .reset_index(drop=True)
)

print("Players with at least", min_plays, "HFL plays:", len(player_summary_filtered))
player_summary_filtered.head(10)



if len(player_summary_filtered) == 0:
    print("No players meet the min_plays threshold. Try lowering min_plays.")
else:
    # Take the top player after filtering
    r = player_summary_filtered.iloc[0]
    
    name = r['player_name']
    plays = int(r['plays_with_hfl'])
    avg_frames = r['avg_latency_frames']
    avg_seconds = r['avg_latency_seconds']
    score = r['HLS_0_100']
    
    print(f"---- HIP-FLIP LATENCY SCOUTING SUMMARY ----")
    print(f"Player: {name}")
    print(f"Plays measured: {plays}")
    print(f"Average hip-flip latency: {avg_seconds:.3f} seconds ({avg_frames:.2f} frames)")
    print(f"Hip-Flip Latency Score (HLS): {score:.1f} / 100")
    
    if score > 85:
        tier = "Elite transition efficiency"
    elif score > 70:
        tier = "Above-average hip mobility"
    elif score > 50:
        tier = "Average transition speed"
    else:
        tier = "Below-average transition speed"
    
    print(f"Qualitative evaluation: {tier}")



from pathlib import Path
import pandas as pd

base_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final"
train_dir = Path(base_path) / "train"

# Check that we actually see the 36 input files
input_files = sorted(train_dir.glob("input_2023_w*.csv"))
print("Number of input files found:", len(input_files))
input_files[:3]



input_all = pd.concat([
    pd.read_csv(f, usecols=[
        "game_id", "play_id", "nfl_id", "frame_id",
        "player_position", "player_name", "player_role",
        "x", "y", "dir"
    ])
    for f in input_files
], ignore_index=True)

print("Rows in input_all:", len(input_all))
input_all.head()



hfl_merged = hfl_results_sample.merge(
    input_all[["game_id", "play_id", "nfl_id", "player_position"]],
    left_on=["game_id", "play_id", "defender_id"],
    right_on=["game_id", "play_id", "nfl_id"],
    how="left"
)

print("Rows in merged HFL table:", len(hfl_merged))
hfl_merged.head()



coverage_positions = ["CB", "FS", "SS", "DB", "NB", "S"]

hfl_db_only = hfl_merged[hfl_merged["player_position"].isin(coverage_positions)].copy()

print("Rows for DB-only:", len(hfl_db_only))
hfl_db_only["player_position"].value_counts().head(10)



db_summary = (
    hfl_db_only
    .groupby(["defender_id", "player_name", "player_position"], as_index=False)
    .agg(
        plays_with_hfl = ("latency_seconds", "count"),
        avg_latency_seconds = ("latency_seconds", "mean")
    )
)

# require at least 5 reps (you can change this number)
min_plays = 5
db_summary = db_summary[db_summary["plays_with_hfl"] >= min_plays].reset_index(drop=True)

# convert to HLS 0–100 (lower latency = better)
max_lat = db_summary["avg_latency_seconds"].max()
db_summary["HLS_0_100"] = 100 * (1 - db_summary["avg_latency_seconds"] / max_lat)

db_summary_sorted = db_summary.sort_values("HLS_0_100", ascending=False)

print("DBs with at least", min_plays, "HFL plays:", len(db_summary_sorted))
db_summary_sorted.head(10)



import pandas as pd

# 1️⃣ Attach positions to coverage defenders with ball-in-air tracking
db_source = merged.merge(
    input_all[["game_id", "play_id", "nfl_id", "player_position"]],
    on=["game_id", "play_id", "nfl_id"],
    how="left"
)

coverage_positions = ["CB", "FS", "SS", "DB", "NB", "S"]

db_pairs = (
    db_source[db_source["player_position"].isin(coverage_positions)]
    [["game_id", "play_id", "nfl_id", "player_name", "week", "player_position"]]
    .drop_duplicates()
)

print("DB defender–play pairs available:", len(db_pairs))
print(db_pairs.head())

# 2️⃣ Run Hip-Flip Latency on a bigger DB-only sample
#    (you can change N to 5000, 10000, etc.)

subset = db_pairs.head(N)

results = []

for _, row in subset.iterrows():
    gid = row["game_id"]
    pid = row["play_id"]
    did = row["nfl_id"]
    name = row["player_name"]
    wk = row["week"]
    pos = row["player_position"]
    
    res = compute_hip_flip_latency(output_all, gid, pid, did)
    
    if res is not None:
        res["player_name"] = name
        res["week"] = wk
        res["player_position"] = pos
        results.append(res)

hfl_results_db_sample = pd.DataFrame(results)

print("\nRows with valid DB Hip-Flip Latency:", len(hfl_results_db_sample))
display(hfl_results_db_sample.head())

# 3️⃣ Build DB-only leaderboard (season-style summary for this sample)
db_summary = (
    hfl_results_db_sample
    .groupby(["defender_id", "player_name", "player_position"], as_index=False)
    .agg(
        plays_with_hfl = ("latency_seconds", "count"),
        avg_latency_seconds = ("latency_seconds", "mean")
    )
)

min_plays = 5   # require at least this many HFL reps
db_summary = db_summary[db_summary["plays_with_hfl"] >= min_plays].reset_index(drop=True)

max_lat = db_summary["avg_latency_seconds"].max()
db_summary["HLS_0_100"] = 100 * (1 - db_summary["avg_latency_seconds"] / max_lat)

db_summary_sorted = db_summary.sort_values("HLS_0_100", ascending=False)

print("\nDBs with at least", min_plays, "HFL reps:", len(db_summary_sorted))
display(db_summary_sorted.head(10))



pairs = db_pairs.copy()



N = len(db_pairs)   # use all DB defender–play pairs
N



# Season-style summary for DBs
db_summary = (
    hfl_results_db_sample
    .groupby(["defender_id", "player_name", "player_position"], as_index=False)
    .agg(
        plays_with_hfl = ("latency_seconds", "count"),
        avg_latency_seconds = ("latency_seconds", "mean")
    )
)

print("Total DBs with at least 1 HFL rep:", len(db_summary))
db_summary.head()



# Require a minimum number of reps to trust the metric
min_plays = 10   # you can change this later (e.g., 5, 15, 20)

db_summary_filtered = db_summary[db_summary["plays_with_hfl"] >= min_plays].reset_index(drop=True)

print("DBs with at least", min_plays, "HFL reps:", len(db_summary_filtered))

# Convert avg latency to a 0–100 Hip-Flip Latency Score (lower latency = better)
max_lat = db_summary_filtered["avg_latency_seconds"].max()
db_summary_filtered["HLS_0_100"] = 100 * (
    1 - db_summary_filtered["avg_latency_seconds"] / max_lat
)

db_summary_sorted = db_summary_filtered.sort_values("HLS_0_100", ascending=False)

db_summary_sorted.head(10)



import numpy as np
import pandas as pd

# --- Rebuild 'pairs' if needed (game, play, defender) ---

try:
    pairs  # see if it already exists
except NameError:
    # Defensive coverage players (from input_all)
    coverage_players = input_all[
        (input_all["player_side"] == "Defense") &
        (input_all["player_role"] == "Defensive Coverage")
    ][[
        "game_id", "play_id", "nfl_id",
        "player_name", "player_position"
    ]]

    # Make sure each defender/play appears once
    pairs = coverage_players.drop_duplicates(
        subset=["game_id", "play_id", "nfl_id"]
    ).reset_index(drop=True)

print("Number of defender–play pairs:", len(pairs))

# --- Compute GDCL latency for each pair ---

gdcl_results_all = []

gdcl_threshold_deg = 2.0   # smaller than HFL (captures more subtle changes)

for idx, row in pairs.iterrows():
    g  = row["game_id"]
    p  = row["play_id"]
    nid = row["nfl_id"]

    # Ball-in-air tracking for this defender
    df_track = (
        output_all[
            (output_all["game_id"] == g) &
            (output_all["play_id"] == p) &
            (output_all["nfl_id"] == nid)
        ]
        .sort_values("frame_id")
        .copy()
    )

    if len(df_track) < 3:
        continue

    # Position differences between frames
    dx = df_track["x"].diff()
    dy = df_track["y"].diff()

    # Direction of movement in radians & degrees
    dir_rad = np.arctan2(dy, dx)
    dir_deg = np.degrees(dir_rad)

    df_track["dir_deg"] = dir_deg

    # Find first frame where direction changes more than the small threshold
    first_change_frame = None
    for i in range(1, len(df_track)):
        d_angle = abs(df_track["dir_deg"].iloc[i] - df_track["dir_deg"].iloc[i-1])
        if d_angle > gdcl_threshold_deg:
            first_change_frame = int(df_track["frame_id"].iloc[i])
            break

    # If no meaningful change, skip this play
    if first_change_frame is None:
        continue

    first_frame = int(df_track["frame_id"].min())
    latency_frames  = first_change_frame - first_frame
    latency_seconds = latency_frames / 10.0   # 10 Hz tracking

    gdcl_results_all.append([
        g, p, nid,
        row["player_name"],
        row["player_position"],
        latency_seconds
    ])

gdcl_results_all = pd.DataFrame(
    gdcl_results_all,
    columns=["game_id", "play_id", "nfl_id",
             "player_name", "player_position",
             "gdcl_latency_seconds"]
)

print("Rows in gdcl_results_all:", len(gdcl_results_all))
gdcl_results_all.head()



# Aggregate GDCL per defender for the season
gdcl_summary = (
    gdcl_results_all
        .groupby(["nfl_id", "player_name", "player_position"], as_index=False)
        .agg(
            plays_with_gdcl=("gdcl_latency_seconds", "count"),
            avg_gdcl_seconds=("gdcl_latency_seconds", "mean"),
        )
)

print("Number of DBs with at least 1 GDCL rep:", len(gdcl_summary))
gdcl_summary.head(10)



# Make sure the HLS table uses nfl_id (not defender_id)
hls = db_summary_sorted.rename(columns={"defender_id": "nfl_id"})

print("HLS rows:", len(hls))
print("GDCL rows:", len(gdcl_summary))



# Merge HLS + GDCL per defender
combined = hls.merge(
    gdcl_summary,
    on=["nfl_id", "player_name", "player_position"],
    how="outer"
)

print("Combined rows:", len(combined))
combined.head(10)



# Scale GDCL (lower = better)
max_gdcl = combined["avg_gdcl_seconds"].max()

combined["GDCL_0_100"] = 100 * (1 - combined["avg_gdcl_seconds"] / max_gdcl)

combined.head(10)



combined["TAI_0_100"] = (
    0.5 * combined["HLS_0_100"].fillna(0) +
    0.5 * combined["GDCL_0_100"].fillna(0)
)

combined = combined.sort_values("TAI_0_100", ascending=False)

combined.head(15)



combined.to_csv("/kaggle/working/transition_agility_index.csv", index=False)



final = combined[[
    'nfl_id', 'player_name', 'player_position',
    'plays_with_hfl', 'HLS_0_100',
    'plays_with_gdcl', 'GDCL_0_100',
    'TAI_0_100'
]].sort_values('TAI_0_100', ascending=False)

final.head(20)



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
sns.scatterplot(
    data=combined,
    x='HLS_0_100',
    y='GDCL_0_100',
    hue='TAI_0_100',
    palette='viridis'
)
plt.title("Hip-Flip Latency vs Good Direction Change Latency")
plt.show()



import os

# 1) Hip-Flip Latency leaderboard (HLS only)
db_summary_sorted.to_csv("/kaggle/working/hfl_leaderboard.csv", index=False)

# 2) Good Direction Change Latency leaderboard (GDCL only)
gdcl_summary.to_csv("/kaggle/working/gdcl_leaderboard.csv", index=False)

# 3) Combined Transition Agility Index (TAI)
combined.to_csv("/kaggle/working/tai_leaderboard.csv", index=False)

print("Files in /kaggle/working now:")
print(os.listdir("/kaggle/working"))


