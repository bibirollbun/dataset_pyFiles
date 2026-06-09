# === Updated metrics function with Route Mirroring + grouping block ===
import numpy as np
import pandas as pd

def calculate_play_metrics_with_route_mirroring(play_df):
    """
    Returns a pandas.Series with:
      - primary_defender_id
      - receiver_id
      - coverage_tightness (avg separation)
      - ball_hawk_score (avg closing speed toward ball landing point)
      - route_mirroring (avg frame-level mirror score combining dir alignment and speed ratio)
    Works with estimated vx/vy (prefers vx/vy columns), otherwise computes simple diffs.
    """
    keys = ['primary_defender_id','receiver_id','coverage_tightness','ball_hawk_score','route_mirroring']
    res = pd.Series({k: np.nan for k in keys})
    try:
        pdf = play_df.copy()
        # basic guards
        if 'player_role' not in pdf.columns or {'x','y','frame_id'}.difference(pdf.columns):
            return res

        # normalize roles
        pdf['role_lower'] = pdf['player_role'].astype(str).str.lower()
        receiver_mask = pdf['role_lower'].str.contains('target|receiver', na=False)
        defender_mask = pdf['role_lower'].str.contains('defens|coverage|defend|db', na=False)
        if not receiver_mask.any() or not defender_mask.any():
            return res

        receiver = pdf[receiver_mask].copy()
        defenders = pdf[defender_mask].copy()

        # determine start frame
        start_frame = int(pdf['frame_id'].min())

        rec_start = receiver[receiver['frame_id'] == start_frame]
        def_start = defenders[defenders['frame_id'] == start_frame]
        if rec_start.empty:
            rec_start = receiver.groupby('nfl_id', dropna=True).first().reset_index()
        if def_start.empty:
            def_start = defenders.groupby('nfl_id', dropna=True).first().reset_index()
        if rec_start.empty or def_start.empty:
            return res

        # ball landing coords
        if 'ball_land_x' not in pdf.columns or pdf['ball_land_x'].dropna().empty:
            return res
        ball_land_x = float(pdf['ball_land_x'].dropna().iloc[0])
        ball_land_y = float(pdf['ball_land_y'].dropna().iloc[0])

        # receiver basic
        rec_row = rec_start.iloc[0]
        rec_x, rec_y = rec_row.get('x', np.nan), rec_row.get('y', np.nan)
        res['receiver_id'] = rec_row.get('nfl_id', np.nan)

        # primary defender selection: closest at start frame
        def_start = def_start.copy()
        def_start['dist_to_rec'] = np.sqrt((def_start['x'] - rec_x)**2 + (def_start['y'] - rec_y)**2)
        if def_start['dist_to_rec'].isnull().all():
            return res
        primary_idx = def_start['dist_to_rec'].idxmin()
        primary_defender_id = def_start.loc[primary_idx, 'nfl_id']
        res['primary_defender_id'] = primary_defender_id

        # Build receiver & defender paths; prefer existing vx/vy if present
        recv_cols = ['frame_id','x','y']
        if 'vx' in pdf.columns and 'vy' in pdf.columns:
            recv_cols += ['vx','vy']
        receiver_path = receiver[recv_cols].rename(columns={'x':'rec_x','y':'rec_y','vx':'rec_vx','vy':'rec_vy'}).copy()

        def_cols = ['frame_id','x','y']
        if 'vx' in pdf.columns and 'vy' in pdf.columns:
            def_cols += ['vx','vy']
        defender_path = defenders[def_cols].rename(columns={'x':'def_x','y':'def_y','vx':'def_vx','vy':'def_vy'}).copy()
        defender_path = defender_path[defender_path['nfl_id'] == primary_defender_id] if 'nfl_id' in defender_path.columns else defender_path
        # note: if 'nfl_id' column was kept we already filtered; otherwise we filter above using defenders selection
        if 'nfl_id' not in defender_path.columns and primary_defender_id is not None:
            # defender_path may already be only primary defender due to earlier filtering; if not, filter on id in play_df
            defender_path = defenders[defenders['nfl_id'] == primary_defender_id][def_cols].rename(columns={'x':'def_x','y':'def_y','vx':'def_vx','vy':'def_vy'}).copy()

        if receiver_path.empty or defender_path.empty:
            return res

        merged = pd.merge(receiver_path, defender_path, on='frame_id', how='inner', suffixes=('_r','_d')).sort_values('frame_id').reset_index(drop=True)
        if merged.empty:
            return res

        # ---- Coverage Tightness ----
        merged['separation'] = np.sqrt((merged['rec_x'] - merged['def_x'])**2 + (merged['rec_y'] - merged['def_y'])**2)
        res['coverage_tightness'] = float(merged['separation'].mean())

        # ---- Ball Hawk Score ----
        merged['ball_vec_x'] = ball_land_x - merged['def_x']
        merged['ball_vec_y'] = ball_land_y - merged['def_y']
        norm_ball = np.sqrt(merged['ball_vec_x']**2 + merged['ball_vec_y']**2).replace(0,1.0)
        merged['ball_dir_x'] = merged['ball_vec_x'] / norm_ball
        merged['ball_dir_y'] = merged['ball_vec_y'] / norm_ball

        # defender velocity: prefer def_vx/def_vy (already in merged if vx present), otherwise try s/dir fallback (rare)
        if {'def_vx','def_vy'}.issubset(merged.columns):
            closing_speed = merged['def_vx'] * merged['ball_dir_x'] + merged['def_vy'] * merged['ball_dir_y']
            res['ball_hawk_score'] = float(closing_speed.mean())
        elif {'s','dir'}.issubset(merged.columns):
            math_dir = np.deg2rad(90.0 - merged['dir'].astype(float))
            def_vx = merged['s'].astype(float) * np.cos(math_dir)
            def_vy = merged['s'].astype(float) * np.sin(math_dir)
            closing_speed = def_vx * merged['ball_dir_x'] + def_vy * merged['ball_dir_y']
            res['ball_hawk_score'] = float(closing_speed.mean())
        else:
            res['ball_hawk_score'] = np.nan

        # ---- Route Mirroring ----
        # Get receiver velocities: prefer rec_vx/rec_vy, otherwise compute from rec_x diffs across merged frames
        if {'rec_vx','rec_vy'}.issubset(merged.columns):
            merged['r_vx'] = merged['rec_vx'].astype(float)
            merged['r_vy'] = merged['rec_vy'].astype(float)
        else:
            # compute per-frame diff (units per frame). Use forward/backward diff to estimate velocity centered on frame.
            merged['r_vx'] = merged['rec_x'].diff().fillna(0)
            merged['r_vy'] = merged['rec_y'].diff().fillna(0)

        # Defender velocities: prefer def_vx/def_vy, otherwise estimate from def_x diffs
        if {'def_vx','def_vy'}.issubset(merged.columns):
            merged['d_vx'] = merged['def_vx'].astype(float)
            merged['d_vy'] = merged['def_vy'].astype(float)
        else:
            merged['d_vx'] = merged['def_x'].diff().fillna(0)
            merged['d_vy'] = merged['def_y'].diff().fillna(0)

        # speeds
        merged['r_speed'] = np.sqrt(merged['r_vx']**2 + merged['r_vy']**2)
        merged['d_speed'] = np.sqrt(merged['d_vx']**2 + merged['d_vy']**2)

        # avoid divide by zero: where speed==0 set direction to NaN (these frames are not informative)
        r_nonzero = merged['r_speed'] > 1e-6
        d_nonzero = merged['d_speed'] > 1e-6
        valid_both = r_nonzero & d_nonzero

        if valid_both.any():
            # normalized directions
            merged.loc[valid_both, 'r_dir_x'] = merged.loc[valid_both, 'r_vx'] / merged.loc[valid_both, 'r_speed']
            merged.loc[valid_both, 'r_dir_y'] = merged.loc[valid_both, 'r_vy'] / merged.loc[valid_both, 'r_speed']
            merged.loc[valid_both, 'd_dir_x'] = merged.loc[valid_both, 'd_vx'] / merged.loc[valid_both, 'd_speed']
            merged.loc[valid_both, 'd_dir_y'] = merged.loc[valid_both, 'd_vy'] / merged.loc[valid_both, 'd_speed']

            # directional alignment (cosine similarity)
            merged.loc[valid_both, 'dir_score'] = (
                merged.loc[valid_both, 'r_dir_x'] * merged.loc[valid_both, 'd_dir_x'] +
                merged.loc[valid_both, 'r_dir_y'] * merged.loc[valid_both, 'd_dir_y']
            ).clip(-1.0, 1.0)

            # speed ratio (0..1)
            merged.loc[valid_both, 'speed_ratio'] = (
                np.minimum(merged.loc[valid_both, 'r_speed'], merged.loc[valid_both, 'd_speed']) /
                np.maximum(merged.loc[valid_both, 'r_speed'], merged.loc[valid_both, 'd_speed'])
            )

            # frame-level mirror score: product of direction alignment and speed ratio
            merged.loc[valid_both, 'mirror_frame_score'] = merged.loc[valid_both, 'dir_score'] * merged.loc[valid_both, 'speed_ratio']

            # aggregate to play-level: mean of frame scores
            route_mirroring = float(merged.loc[valid_both, 'mirror_frame_score'].mean())
            res['route_mirroring'] = route_mirroring
        else:
            res['route_mirroring'] = np.nan

        return res

    except Exception as e:
        # in case of any unexpected error return NaNs (optionally log e)
        # print("Exception in route mirroring calc:", e)
        return pd.Series({k: np.nan for k in keys})


# ---- Group by plays and compute metrics (use include_groups=False to silence deprecation warning) ----
if 'df' not in globals():
    raise RuntimeError("Dataframe 'df' not found. Please ensure your merged/enriched tracking DataFrame is stored in variable `df`.")

if {'game_id','play_id'}.issubset(df.columns):
    play_metrics = df.groupby(['game_id','play_id'], group_keys=False, sort=False).apply(
        lambda g: calculate_play_metrics_with_route_mirroring(g),
        include_groups=False
    ).reset_index()

    # tidy types for convenience
    if 'primary_defender_id' in play_metrics.columns:
        play_metrics['primary_defender_id'] = pd.to_numeric(play_metrics['primary_defender_id'], errors='coerce').astype('Int64')
    if 'receiver_id' in play_metrics.columns:
        play_metrics['receiver_id'] = pd.to_numeric(play_metrics['receiver_id'], errors='coerce').astype('Int64')

    # save results
    play_metrics.to_csv('play_metrics_with_route_mirroring.csv', index=False)
    print("Saved play_metrics_with_route_mirroring.csv — rows:", len(play_metrics))

    # short summary
    total_plays = df[['game_id','play_id']].drop_duplicates().shape[0]
    print(f"Total plays found: {total_plays}")
    print("Plays with coverage_tightness computed:", play_metrics['coverage_tightness'].notna().sum())
    print("Plays with ball_hawk_score computed:", play_metrics['ball_hawk_score'].notna().sum())
    print("Plays with route_mirroring computed:", play_metrics['route_mirroring'].notna().sum())

    # show first 10 rows
    display(play_metrics.head(10))
else:
    raise RuntimeError("Dataframe must contain 'game_id' and 'play_id' columns for grouping.")



import matplotlib.pyplot as plt
import numpy as np

# load results & pick a sample play
pm = pd.read_csv('play_metrics.csv')
sample_game, sample_play = pm.loc[0, ['game_id','play_id']].astype(int).tolist()

# filter frames for that play
sample = df[(df['game_id']==sample_game) & (df['play_id']==sample_play)]

rid = int(pm.loc[0,'receiver_id'])
pdid = int(pm.loc[0,'primary_defender_id'])

recv = sample[sample['nfl_id']==rid][['frame_id','x','y']].rename(columns={'x':'rec_x','y':'rec_y'})
defn = sample[sample['nfl_id']==pdid][['frame_id','x','y','vx','vy']].rename(columns={'x':'def_x','y':'def_y'})

merged = pd.merge(recv, defn, on='frame_id', how='inner').sort_values('frame_id').reset_index(drop=True)
if merged.empty:
    print("No overlapping frames for this sample play.")
else:
    # separation
    merged['separation'] = np.sqrt((merged['rec_x']-merged['def_x'])**2 + (merged['rec_y']-merged['def_y'])**2)
    # closing speed toward ball_land (if ball_land in sample)
    if 'ball_land_x' in sample.columns and not sample['ball_land_x'].dropna().empty:
        bx = float(sample['ball_land_x'].dropna().iloc[0])
        by = float(sample['ball_land_y'].dropna().iloc[0])
        merged['ball_vec_x'] = bx - merged['def_x']
        merged['ball_vec_y'] = by - merged['def_y']
        norm = np.sqrt(merged['ball_vec_x']**2 + merged['ball_vec_y']**2).replace(0,1)
        merged['ball_dir_x'] = merged['ball_vec_x']/norm
        merged['ball_dir_y'] = merged['ball_vec_y']/norm
        # use per-frame vx/vy; multiply by frame_rate if you want per-second
        merged['closing_speed'] = merged['vx'] * merged['ball_dir_x'] + merged['vy'] * merged['ball_dir_y']
    else:
        merged['closing_speed'] = np.nan

    # plot
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(merged['frame_id'], merged['separation'], marker='o')
    plt.title('Separation over frames')
    plt.xlabel('frame_id')
    plt.ylabel('separation (field units)')

    plt.subplot(1,2,2)
    plt.plot(merged['frame_id'], merged['closing_speed'], marker='o')
    plt.title('Closing speed (toward ball)')
    plt.xlabel('frame_id')
    plt.ylabel('speed (units/frame)')
    plt.tight_layout()
    plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Load play-level metrics ----------------
try:
    pm = pd.read_csv('play_metrics_with_route_mirroring.csv')
    print("Loaded play_metrics_with_route_mirroring.csv")
except FileNotFoundError:
    try:
        pm  # use in-memory variable if it exists
        print("Using in-memory play_metrics variable")
    except NameError:
        raise RuntimeError("play_metrics_with_route_mirroring.csv not found and no in-memory 'pm' variable. Run metrics step first.")

# ensure we have the dataframe variable in pm
if 'pm' not in globals():
    pm = pd.read_csv('play_metrics_with_route_mirroring.csv')

# Clean types
pm['primary_defender_id'] = pd.to_numeric(pm['primary_defender_id'], errors='coerce').astype('Int64')
pm['receiver_id'] = pd.to_numeric(pm['receiver_id'], errors='coerce').astype('Int64')

# Drop rows where primary_defender_id is missing
pm = pm.dropna(subset=['primary_defender_id']).copy()

# ---------------- Aggregate per defender ----------------
agg = pm.groupby('primary_defender_id').agg(
    plays_count = ('coverage_tightness', 'count'),
    avg_coverage = ('coverage_tightness', 'mean'),
    median_coverage = ('coverage_tightness', 'median'),
    std_coverage = ('coverage_tightness', 'std'),
    avg_ballhawk = ('ball_hawk_score', 'mean'),
    median_ballhawk = ('ball_hawk_score', 'median'),
    std_ballhawk = ('ball_hawk_score', 'std'),
    avg_mirroring = ('route_mirroring', 'mean'),
    median_mirroring = ('route_mirroring', 'median'),
    std_mirroring = ('route_mirroring', 'std'),
).reset_index()

# ---------------- Filter by minimum plays ----------------
min_plays_threshold = 5   # tune this as needed
agg = agg[agg['plays_count'] >= min_plays_threshold].copy()
print(f"Defenders with >= {min_plays_threshold} plays: {len(agg)}")

# ---------------- Normalization (min-max) ----------------
def minmax_scale(series):
    if series.isnull().all():
        return series
    mn, mx = series.min(), series.max()
    if mn == mx:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

# coverage: lower is better → invert after scaling
agg['coverage_norm'] = minmax_scale(agg['avg_coverage'])
agg['coverage_inv'] = 1.0 - agg['coverage_norm']

# ballhawk: higher is better
agg['ballhawk_norm'] = minmax_scale(agg['avg_ballhawk'])

# mirroring: higher is better
agg['mirror_norm'] = minmax_scale(agg['avg_mirroring'])

# ---------------- Combined score ----------------
# Tune these weights to emphasize different skills
weight_coverage = 0.45
weight_ballhawk = 0.25
weight_mirroring = 0.30

# make sure weights sum to 1 (normalize if not)
w_sum = weight_coverage + weight_ballhawk + weight_mirroring
weight_coverage /= w_sum
weight_ballhawk /= w_sum
weight_mirroring /= w_sum

agg['combined_score'] = (
    agg['coverage_inv'] * weight_coverage +
    agg['ballhawk_norm'] * weight_ballhawk +
    agg['mirror_norm'] * weight_mirroring
)

# ---------------- Ranking and Output ----------------
top_n = 30
top_by_coverage = agg.sort_values('avg_coverage').head(top_n)
top_by_ballhawk = agg.sort_values('avg_ballhawk', ascending=False).head(top_n)
top_by_mirroring = agg.sort_values('avg_mirroring', ascending=False).head(top_n)
top_by_combined = agg.sort_values('combined_score', ascending=False).head(top_n)

print(f"\nDefenders with >= {min_plays_threshold} plays: {len(agg)}")

print("\n=== Top by avg coverage (lower better) ===")
display(top_by_coverage[['primary_defender_id','plays_count','avg_coverage']].head(10))

print("\n=== Top by avg ball_hawk_score (higher better) ===")
display(top_by_ballhawk[['primary_defender_id','plays_count','avg_ballhawk']].head(10))

print("\n=== Top by avg route_mirroring (higher better) ===")
display(top_by_mirroring[['primary_defender_id','plays_count','avg_mirroring']].head(10))

print(f"\n=== Top by combined score "
      f"(weights: coverage {weight_coverage:.2f}, ballhawk {weight_ballhawk:.2f}, mirroring {weight_mirroring:.2f}) ===")
display(top_by_combined[['primary_defender_id','plays_count','combined_score',
                         'avg_coverage','avg_ballhawk','avg_mirroring']].head(10))

# Save leaderboard
agg_sorted = agg.sort_values('combined_score', ascending=False).reset_index(drop=True)
agg_sorted.to_csv('defender_leaderboard_3metrics.csv', index=False)
print("\n✅ Saved defender_leaderboard_3metrics.csv")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Load metrics ----------
pm = pd.read_csv("play_metrics_with_route_mirroring.csv")

# Clean types
pm['primary_defender_id'] = pd.to_numeric(pm['primary_defender_id'], errors='coerce').astype('Int64')
pm = pm.dropna(subset=['primary_defender_id'])

# ---------- Aggregate per defender ----------
agg = pm.groupby('primary_defender_id').agg(
    plays_count=('coverage_tightness','count'),
    avg_coverage=('coverage_tightness','mean'),
    avg_ballhawk=('ball_hawk_score','mean'),
    avg_mirroring=('route_mirroring','mean')
).reset_index()

# ---------- Filter defenders ----------
min_plays_threshold = 5
agg = agg[agg['plays_count'] >= min_plays_threshold].copy()

# ---------- Normalize metrics ----------
def minmax(series):
    mn, mx = series.min(), series.max()
    if mn == mx: return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

agg['coverage_norm'] = minmax(agg['avg_coverage'])
agg['coverage_inv'] = 1 - agg['coverage_norm']
agg['ballhawk_norm'] = minmax(agg['avg_ballhawk'])
agg['mirror_norm']   = minmax(agg['avg_mirroring'])

# ---------- Overall Defender Score ----------
w_cov, w_bh, w_mir = 0.40, 0.25, 0.35
agg['ODS'] = (agg['coverage_inv']*w_cov +
              agg['ballhawk_norm']*w_bh +
              agg['mirror_norm']*w_mir)

# ---------- Rankings ----------
agg_sorted = agg.sort_values('ODS', ascending=False).reset_index(drop=True)
agg_sorted.to_csv("defender_overall_score.csv", index=False)

print(f"✅ Saved defender_overall_score.csv — {len(agg_sorted)} defenders ranked")
print("\n=== Top 10 Overall Defenders ===")
display(agg_sorted[['primary_defender_id','plays_count','ODS',
                    'avg_coverage','avg_ballhawk','avg_mirroring']].head(10))

# ---------- Visualizations ----------
top_n = 15
top_overall = agg_sorted.head(top_n)

plt.figure(figsize=(10,6))
plt.bar(top_overall['primary_defender_id'].astype(str), top_overall['ODS'])
plt.xticks(rotation=90)
plt.title(f"Top {top_n} Defenders by Overall Defender Score")
plt.ylabel("Overall Defender Score (ODS)")
plt.tight_layout()
plt.show()

# Individual metric comparison
fig, axes = plt.subplots(1,3, figsize=(15,5), sharey=True)
axes[0].bar(top_overall['primary_defender_id'].astype(str), top_overall['coverage_inv'])
axes[0].set_title("Coverage (inverted, higher=better)")
axes[1].bar(top_overall['primary_defender_id'].astype(str), top_overall['ballhawk_norm'])
axes[1].set_title("Ball Hawk (normalized)")
axes[2].bar(top_overall['primary_defender_id'].astype(str), top_overall['mirror_norm'])
axes[2].set_title("Route Mirroring (normalized)")
for ax in axes: ax.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()



# ROSTER MERGE: try players.csv first, otherwise build roster from input_df
import os
import pandas as pd

# 1) Adjust this path to the top-level folder of your dataset if different.
kaggle_input_root = '/kaggle/input'

# Helper: try to find likely roster files under /kaggle/input
possible_rosters = []
for root, dirs, files in os.walk(kaggle_input_root):
    for f in files:
        name = f.lower()
        if 'player' in name and name.endswith('.csv'):
            possible_rosters.append(os.path.join(root, f))

print("Found candidate roster files (first 10):", possible_rosters[:10])

# Try to load players.csv if present
roster = None
if possible_rosters:
    # prefer a file called players.csv if it exists
    players_paths = [p for p in possible_rosters if os.path.basename(p).lower() == 'players.csv']
    candidate = players_paths[0] if players_paths else possible_rosters[0]
    try:
        roster = pd.read_csv(candidate, low_memory=False)
        print("Loaded roster from:", candidate)
    except Exception as e:
        print("Could not read candidate roster:", candidate, " — error:", e)
        roster = None

# If roster still None, try to build from input_df (you mentioned you have input_df loaded earlier)
if roster is None:
    try:
        # input_df should already exist in your notebook from earlier steps
        input_df  # just referencing to trigger NameError if not present
        print("Building roster from input_df (using nfl_id, player_name, player_position if present).")
        cols = []
        for c in ['nfl_id','player_name','player_position','position','display_name']:
            if c in input_df.columns:
                cols.append(c)
        # Prefer column names mapping: nfl_id, display_name, position
        if 'nfl_id' not in input_df.columns:
            raise KeyError("input_df does not contain 'nfl_id' — cannot build roster automatically.")
        # find name column
        name_col = None
        for nc in ['display_name','player_name','name']:
            if nc in input_df.columns:
                name_col = nc
                break
        pos_col = None
        for pc in ['player_position','position','pos']:
            if pc in input_df.columns:
                pos_col = pc
                break

        roster = input_df[['nfl_id'] + ([name_col] if name_col else []) + ([pos_col] if pos_col else [])].drop_duplicates().rename(
            columns={name_col: 'display_name', pos_col: 'position'} if name_col or pos_col else {}
        )
        print("Built roster from input_df; sample rows:")
        display(roster.head(10))
    except NameError:
        roster = None
        print("input_df not found in memory — cannot build roster. If you have a roster CSV, upload it to the Kaggle input or run the earlier load steps to create input_df.")

if roster is None:
    raise RuntimeError("Roster not found and could not be built. Place a roster CSV in the dataset or ensure input_df is loaded.")

# Normalize roster columns for merging
# Expect roster has 'nfl_id', 'display_name', 'position' (if not present, create placeholders)
if 'nfl_id' not in roster.columns:
    raise RuntimeError("Roster doesn't contain nfl_id column; cannot proceed.")
for col in ['display_name','position']:
    if col not in roster.columns:
        roster[col] = None

# Make sure nfl_id numeric
roster['nfl_id'] = pd.to_numeric(roster['nfl_id'], errors='coerce').astype('Int64')

# --- Now create agg if not created yet (aggregate play_metrics into defender-level agg) ---
# If you already have `agg` or `agg_named` from earlier leaderboard code, this will reuse it.
try:
    agg  # if exists, keep it
except NameError:
    # build agg from play_metrics_with_route_mirroring.csv
    pm_path = 'play_metrics_with_route_mirroring.csv'
    if not os.path.exists(pm_path):
        raise RuntimeError(f"{pm_path} not found. Run the metric computation step first.")
    pm = pd.read_csv(pm_path)
    pm['primary_defender_id'] = pd.to_numeric(pm['primary_defender_id'], errors='coerce').astype('Int64')
    pm = pm.dropna(subset=['primary_defender_id'])
    agg = pm.groupby('primary_defender_id').agg(
        plays_count=('coverage_tightness','count'),
        avg_coverage=('coverage_tightness','mean'),
        avg_ballhawk=('ball_hawk_score','mean'),
        avg_mirroring=('route_mirroring','mean')
    ).reset_index()
    # filter (same threshold you used before)
    min_plays_threshold = 5
    agg = agg[agg['plays_count'] >= min_plays_threshold].copy()

# Normalize & compute ODS (same as earlier)
def minmax(series):
    mn, mx = series.min(), series.max()
    if mn == mx: return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

agg['coverage_norm'] = minmax(agg['avg_coverage'])
agg['coverage_inv'] = 1 - agg['coverage_norm']
agg['ballhawk_norm'] = minmax(agg['avg_ballhawk'])
agg['mirror_norm'] = minmax(agg['avg_mirroring'])

w_cov, w_bh, w_mir = 0.40, 0.25, 0.35
agg['ODS'] = (agg['coverage_inv']*w_cov +
              agg['ballhawk_norm']*w_bh +
              agg['mirror_norm']*w_mir)

# Merge roster to build agg_named
agg_named = agg.merge(roster[['nfl_id','display_name','position']], left_on='primary_defender_id', right_on='nfl_id', how='left')
agg_named = agg_named.drop(columns=['nfl_id'])
agg_named['display_name'] = agg_named['display_name'].fillna(agg_named['primary_defender_id'].astype(str))
agg_named['position'] = agg_named['position'].fillna('UNK')

# Save and show
agg_named = agg_named.sort_values('ODS', ascending=False).reset_index(drop=True)
agg_named.to_csv('defender_leaderboard_named.csv', index=False)
print("Created defender_leaderboard_named.csv — sample:")
display(agg_named.head(15))



import pandas as pd
import numpy as np

# --- Load play-level metrics ---
pm = pd.read_csv("play_metrics_with_route_mirroring.csv")

# Clean defender IDs
pm['primary_defender_id'] = pd.to_numeric(pm['primary_defender_id'], errors='coerce').astype('Int64')
pm = pm.dropna(subset=['primary_defender_id'])

# --- Aggregate defender averages ---
agg = pm.groupby('primary_defender_id').agg(
    plays_count=('coverage_tightness','count'),
    avg_coverage=('coverage_tightness','mean'),
    avg_ballhawk=('ball_hawk_score','mean'),
    avg_mirroring=('route_mirroring','mean')
).reset_index()

# --- Filter defenders with minimum plays ---
min_plays_threshold = 5
agg = agg[agg['plays_count'] >= min_plays_threshold].copy()

# --- Normalize metrics (all scaled to 0–1, higher=better) ---
def minmax(series):
    mn, mx = series.min(), series.max()
    if mn == mx: return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

agg['coverage_norm'] = minmax(agg['avg_coverage'])
agg['coverage_inv'] = 1 - agg['coverage_norm']   # invert coverage (lower is better)
agg['ballhawk_norm'] = minmax(agg['avg_ballhawk'])
agg['mirror_norm']   = minmax(agg['avg_mirroring'])

# --- Overall Defender Score (ODS) ---
w_cov, w_bh, w_mir = 0.40, 0.25, 0.35
agg['ODS'] = (agg['coverage_inv']*w_cov +
              agg['ballhawk_norm']*w_bh +
              agg['mirror_norm']*w_mir)

# --- Merge roster for names/positions ---
try:
    roster = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2026-analytics/players.csv")
    agg_named = agg.merge(roster[['nfl_id','display_name','position']],
                          left_on='primary_defender_id', right_on='nfl_id', how='left')
    agg_named = agg_named.drop(columns=['nfl_id'])
except FileNotFoundError:
    print("⚠️ Roster file not found, continuing without names")
    agg_named = agg.copy()
    agg_named['display_name'] = agg_named['primary_defender_id']
    agg_named['position'] = "UNK"

# --- Save leaderboard ---
agg_named = agg_named.sort_values('ODS', ascending=False).reset_index(drop=True)
agg_named.to_csv("defender_leaderboard_named.csv", index=False)

print(f"✅ Created defender_leaderboard_named.csv with {len(agg_named)} defenders")
display(agg_named.head(15))



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_radar(player_row, title=None, metrics=None, labels=None):
    """
    Robust radar (spider) chart function.
    - player_row: a Series-like row containing metric values.
    - metrics: list of column names to read from player_row (defaults to
               ['coverage_inv','ballhawk_norm','mirror_norm','ODS']).
    - labels: display labels for the metrics (defaults accordingly).
    """
    if metrics is None:
        metrics = ['coverage_inv','ballhawk_norm','mirror_norm','ODS']
    if labels is None:
        labels = ['Coverage','Ball Hawk','Mirroring','Overall']
    assert len(metrics) == len(labels), "metrics and labels must match length"

    # Extract values; replace NaN with 0.0 (or 0.5 if you prefer neutral)
    values = []
    for m in metrics:
        v = player_row.get(m, np.nan)
        if pd.isna(v):
            v = 0.0
        values.append(float(v))

    # Ensure values are in [0,1] (they should be if normalized earlier)
    values = [max(0.0, min(1.0, v)) for v in values]

    # number of variables
    n = len(labels)

    # compute angles for each axis (n angles)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    # close the loop by appending first value & angle
    values += values[:1]
    angles += angles[:1]

    # Plot
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, marker='o', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight='bold')
    plt.show()


# Example: draw radar charts for top 5 defenders in agg_named
# Make sure agg_named exists and contains the normalized columns used below.
required_cols = ['coverage_inv','ballhawk_norm','mirror_norm','ODS','display_name','position']
missing = [c for c in required_cols if c not in agg_named.columns]
if missing:
    raise RuntimeError(f"agg_named is missing required columns: {missing}. Run the leaderboard creation block first.")

# Plot top N
top_n = 5
for i in range(min(top_n, len(agg_named))):
    row = agg_named.iloc[i]
    title = f"{row['display_name']} ({row.get('position','UNK')}) — ODS {row['ODS']:.2f}"
    plot_radar(row, title=title)



from sklearn.cluster import KMeans

# Use normalized metrics only
X = agg_named[['coverage_inv','ballhawk_norm','mirror_norm']].fillna(0)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
agg_named['cluster'] = kmeans.fit_predict(X)

print("Cluster counts:")
print(agg_named['cluster'].value_counts())

# Inspect cluster centroids
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['coverage','ballhawk','mirroring'])
print("\nCluster archetypes (centroids):")
display(centroids)

# Visualize clusters (2D PCA for easy plotting)
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X)
agg_named['pca1'], agg_named['pca2'] = X_pca[:,0], X_pca[:,1]

plt.figure(figsize=(8,6))
for c in agg_named['cluster'].unique():
    subset = agg_named[agg_named['cluster']==c]
    plt.scatter(subset['pca1'], subset['pca2'], label=f"Cluster {c}", alpha=0.7)
plt.legend()
plt.title("Defender Archetypes (PCA view)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()



# ===== Clean, single-cell final report =====
# Paste & run this in your Kaggle notebook (assumes previous files exist)
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from IPython.display import display, Markdown, HTML

# --- Settings (tweak if needed) ---
PLAY_METRICS_CSV = "play_metrics_with_route_mirroring.csv"
LEADERBOARD_CSV = "defender_leaderboard_named.csv"
MIN_PLAYS = 5
TOP_N = 10   # how many to show in leaderboard and bar chart
RADAR_N = 3  # top K to show radar charts
W_COV, W_BH, W_MIR = 0.40, 0.25, 0.35  # ODS weights

# --- Helper functions ---
def minmax(s):
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

def find_roster():
    root = "/kaggle/input"
    for r, dirs, files in os.walk(root):
        for f in files:
            if f.lower() == "players.csv" or ("player" in f.lower() and f.lower().endswith(".csv")):
                return os.path.join(r, f)
    return None

def plot_radar_axes(ax, vals, labels, title):
    n = len(vals)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    vals = vals + vals[:1]
    angles = angles + angles[:1]
    ax.plot(angles, vals, marker='o', linewidth=1.8)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_title(title, fontsize=10, weight='bold')

# --- 1) Load play-level metrics and show small sample (clean) ---
if not os.path.exists(PLAY_METRICS_CSV):
    raise RuntimeError(f"{PLAY_METRICS_CSV} not found — run metrics generation first.")

plays = pd.read_csv(PLAY_METRICS_CSV)
display(Markdown("## 1) Sample play-level metrics (first 10 rows)"))
cols_sample = ['game_id','play_id','primary_defender_id','receiver_id','coverage_tightness','ball_hawk_score','route_mirroring']
cols_sample = [c for c in cols_sample if c in plays.columns]
display(plays[cols_sample].head(10).style.set_table_attributes("style='display:inline'").set_caption("Sample plays"))

# --- 2) Build defender leaderboard with ODS and merge roster if possible ---
plays['primary_defender_id'] = pd.to_numeric(plays['primary_defender_id'], errors='coerce').astype('Int64')
plays = plays.dropna(subset=['primary_defender_id'])

agg = plays.groupby('primary_defender_id').agg(
    plays_count=('coverage_tightness','count'),
    avg_coverage=('coverage_tightness','mean'),
    avg_ballhawk=('ball_hawk_score','mean'),
    avg_mirroring=('route_mirroring','mean')
).reset_index()

agg = agg[agg['plays_count'] >= MIN_PLAYS].copy()
agg['coverage_norm'] = minmax(agg['avg_coverage'])
agg['coverage_inv'] = 1.0 - agg['coverage_norm']
agg['ballhawk_norm'] = minmax(agg['avg_ballhawk'])
agg['mirror_norm'] = minmax(agg['avg_mirroring'])
agg['ODS'] = agg['coverage_inv']*W_COV + agg['ballhawk_norm']*W_BH + agg['mirror_norm']*W_MIR

# attempt to merge roster for nicer labels
roster_path = find_roster()
if roster_path:
    roster = pd.read_csv(roster_path, low_memory=False)
    # find reasonable column names
    id_col = next((c for c in ['nfl_id','nflId','player_id','id'] if c in roster.columns), None)
    name_col = next((c for c in ['display_name','full_name','player_name','name'] if c in roster.columns), None)
    pos_col = next((c for c in ['position','player_position','pos'] if c in roster.columns), None)
    roster['nfl_id'] = pd.to_numeric(roster[id_col], errors='coerce').astype('Int64') if id_col else None
    roster['display_name'] = roster[name_col] if name_col in roster.columns else roster.get('display_name', None)
    roster['position'] = roster[pos_col] if pos_col in roster.columns else roster.get('position', None)
    roster = roster[['nfl_id','display_name','position']].drop_duplicates(subset=['nfl_id'])
    agg_named = agg.merge(roster, left_on='primary_defender_id', right_on='nfl_id', how='left').drop(columns=['nfl_id'])
else:
    # fallback: show ID as name
    agg_named = agg.copy()
    agg_named['display_name'] = agg_named['primary_defender_id'].astype(str)
    agg_named['position'] = 'UNK'

# reorder columns for display
display_cols = ['primary_defender_id','display_name','position','plays_count','ODS','avg_coverage','avg_ballhawk','avg_mirroring']
display_cols = [c for c in display_cols if c in agg_named.columns]

# Save leaderboard
agg_named = agg_named.sort_values('ODS', ascending=False).reset_index(drop=True)
agg_named.to_csv(LEADERBOARD_CSV, index=False)

display(Markdown("## 2) Top defenders (leaderboard)"))
display(agg_named[display_cols].head(TOP_N).style.set_table_attributes("style='display:inline'").set_caption("Top defenders by ODS"))

# --- 3) Clean bar chart for Top N by ODS ---
display(Markdown("## 3) Top defenders by Overall Defender Score (ODS)"))
top = agg_named.head(TOP_N).copy()
plt.figure(figsize=(10,4))
plt.bar(top['display_name'].astype(str), top['ODS'])
plt.xticks(rotation=45, ha='right')
plt.ylabel("ODS (0-1)")
plt.title(f"Top {TOP_N} Defenders by ODS")
plt.tight_layout()
plt.show()

# --- 4) Compact radar charts for top RADAR_N defenders (single row) ---
display(Markdown(f"## 4) Skill profiles (top {RADAR_N} defenders)"))
labels = ['Coverage','Ball Hawk','Mirroring','Overall']
metrics = ['coverage_inv','ballhawk_norm','mirror_norm','ODS']

fig, axes = plt.subplots(1, min(RADAR_N, len(agg_named)), figsize=(4*min(RADAR_N, len(agg_named)),4), subplot_kw=dict(polar=True))
if min(RADAR_N, len(agg_named)) == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    row = agg_named.iloc[i]
    vals = [float(row.get(m,0.0) if not pd.isna(row.get(m)) else 0.0) for m in metrics]
    vals = [max(0.0, min(1.0, v)) for v in vals]
    plot_radar_axes(ax, vals, labels, f"{row.get('display_name','ID:'+str(int(row['primary_defender_id'])))}\n{row.get('position','')}\nODS {row['ODS']:.2f}")
plt.tight_layout()
plt.show()

# --- 5) Clustering & PCA scatter (compact) ---
display(Markdown("## 5) Defender archetypes (clustering & PCA)"))
X = agg_named[['coverage_inv','ballhawk_norm','mirror_norm']].fillna(0)
if len(X) >= 3:
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    agg_named['cluster'] = kmeans.labels_
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['coverage_inv','ballhawk_norm','mirror_norm'])
    display(Markdown("**Cluster centroids (archetypes)**"))
    display(centroids.style.set_table_attributes("style='display:inline'"))
    # PCA for 2D scatter
    X_pca = PCA(n_components=2).fit_transform(X)
    agg_named['pca1'], agg_named['pca2'] = X_pca[:,0], X_pca[:,1]
    plt.figure(figsize=(8,5))
    for c in sorted(agg_named['cluster'].unique()):
        s = agg_named[agg_named['cluster']==c]
        plt.scatter(s['pca1'], s['pca2'], label=f"Cluster {c}", alpha=0.8)
    plt.legend(); plt.title("Defender archetypes (PCA projection)"); plt.xlabel("PCA1"); plt.ylabel("PCA2")
    plt.tight_layout(); plt.show()
else:
    display(Markdown("_Not enough defenders to perform clustering (need >=3)_"))

# --- 6) Auto-generate an executive summary file (optional) ---
summary_md = f"""
# Executive Summary — Defender Analytics

**What**: Three defender metrics were created from tracking: 
- **Coverage Tightness** (lower is better),
- **Ball Hawk Score** (higher is better),
- **Route Mirroring** (higher is better).

These were combined into an **Overall Defender Score (ODS)** with weights:
- Coverage {W_COV}, Ball Hawk {W_BH}, Mirroring {W_MIR}.

**Top results**: saved to `{LEADERBOARD_CSV}` (top {TOP_N} shown above).  
**Clustering**: simple K-Means found archetypes — centroids printed above.

_Notes_: metrics are normalized (min-max). Coverage was inverted so higher=better for ODS.
"""
with open("EXECUTIVE_SUMMARY.md","w") as f:
    f.write(summary_md)

display(Markdown("✅ **Report generated.** Files saved:"))
display(HTML(f"<ul><li><b>{PLAY_METRICS_CSV}</b></li><li><b>{LEADERBOARD_CSV}</b></li><li><b>EXECUTIVE_SUMMARY.md</b></li></ul>"))

# ===== done =====



# show top-level sizes under /kaggle/input and /kaggle/working
!du -sh /kaggle/input/* | sort -h
!du -sh /kaggle/working/* | sort -h




import os
from pathlib import Path

root = Path('/kaggle/input')
if root.exists():
    csvs = list(root.rglob('*.csv'))
    if not csvs:
        print("No CSVs found under /kaggle/input")
    else:
        sizes = []
        for p in csvs:
            sizes.append((p, p.stat().st_size))
        sizes.sort(key=lambda x: x[1], reverse=True)
        for p, s in sizes[:50]:   # show top 50 largest csvs
            print(f"{p} \t {s/1024/1024:.2f} MB")
else:
    print("/kaggle/input not found")



import psutil
mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total/1024**3:.1f} GB, Available: {mem.available/1024**3:.1f} GB")




# ===== Full-dataset run (pandas, memory-optimized, parallel per-play) =====
# Paste into one Kaggle cell and run. Adjust small variables below if desired.

import os, glob, gc, math
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from tqdm.auto import tqdm

# --------- Settings (tweak if needed) ----------
INPUT_ROOT = "/kaggle/input"
OUT_PLAY_METRICS = "play_metrics_full.csv"
OUT_LEADERBOARD = "defender_leaderboard_full.csv"
MIN_PLAYS_FOR_LEADERBOARD = 5
N_JOBS = max(1, multiprocessing.cpu_count() - 1)  # parallel workers for play processing

# --------- Helper: classify CSV files by sample columns ----------
def classify_csv(path, nrows=200):
    try:
        sample = pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        return "unknown"
    cols = set(sample.columns.str.lower())
    # tracking heuristics
    if {'frame_id','x','y'}.issubset(cols) or {'x','y','frame_id'} & cols:
        return "tracking"
    # input file heuristics: player_role, ball_land_x, nfl_id hints
    if {'player_role','ball_land_x'}.intersection(cols) or 'player_name' in cols:
        return "input"
    # supplementary heuristics
    if 'pass_result' in cols or 'play_description' in cols:
        return "supplementary"
    # fallback: unknown
    return "unknown"

# --------- find all csv files under /kaggle/input (common Kaggle dataset layout) ----------
all_csvs = [str(p) for p in Path(INPUT_ROOT).rglob("*.csv")]
print("Found CSV files:", len(all_csvs))

tracking_files, input_files, supp_files, other_files = [], [], [], []
for f in all_csvs:
    kind = classify_csv(f, nrows=150)
    if kind == "tracking":
        tracking_files.append(f)
    elif kind == "input":
        input_files.append(f)
    elif kind == "supplementary":
        supp_files.append(f)
    else:
        other_files.append(f)

print(f"tracking: {len(tracking_files)}, input: {len(input_files)}, supplementary: {len(supp_files)}, other: {len(other_files)}")

if len(tracking_files) == 0:
    raise RuntimeError("No tracking CSVs found under /kaggle/input. Check dataset organization.")

# --------- read and concat input and supplementary (small) ----------
# load ALL input files (player roles, ball landing coords)
usecols_input_guess = ['game_id','play_id','nfl_id','player_role','ball_land_x','ball_land_y','player_name','player_position']
input_dfs = []
for p in input_files:
    try:
        df = pd.read_csv(p, usecols=[c for c in usecols_input_guess if c in pd.read_csv(p, nrows=0).columns], low_memory=False)
        input_dfs.append(df)
    except Exception:
        # fallback: read whole file and select columns present
        df = pd.read_csv(p, low_memory=False)
        cols = [c for c in usecols_input_guess if c in df.columns]
        input_dfs.append(df[cols])
if input_dfs:
    input_df = pd.concat(input_dfs, ignore_index=True).drop_duplicates()
else:
    input_df = pd.DataFrame(columns=['game_id','play_id','nfl_id','player_role','ball_land_x','ball_land_y'])
print("Input rows:", len(input_df))

# try to load a single supplementary file (if present)
supplementary_df = pd.DataFrame()
if supp_files:
    # pick first supplementary file
    try:
        supplementary_df = pd.read_csv(supp_files[0], low_memory=False)
        print("Loaded supplementary:", supp_files[0], "rows:", len(supplementary_df))
    except Exception:
        supplementary_df = pd.DataFrame()

# --------- read and concat tracking files in memory-stepwise (only needed cols) ----------
# tracking columns we'd ideally like:
tracking_needed = ['game_id','play_id','nfl_id','frame_id','x','y','s','dir','o','a']
# We'll detect columns for each file and read only present ones
tracking_parts = []
for p in tqdm(tracking_files, desc="Reading tracking files"):
    try:
        cols = pd.read_csv(p, nrows=0, low_memory=False).columns.tolist()
    except Exception:
        cols = None
    if cols:
        usecols = [c for c in tracking_needed if c in cols]
        # always include game_id/play_id/nfl_id/frame_id/x/y (if available)
        fallback_cols = ['game_id','play_id','nfl_id','frame_id','x','y']
        for c in fallback_cols:
            if c in (pd.read_csv(p, nrows=0, low_memory=False).columns.tolist()) and c not in usecols:
                usecols.append(c)
        try:
            df = pd.read_csv(p, usecols=usecols, low_memory=False)
            tracking_parts.append(df)
        except Exception:
            # fallback: read entire file (rare)
            df = pd.read_csv(p, low_memory=False)
            tracking_parts.append(df[[c for c in df.columns if c in tracking_needed or c in ['game_id','play_id','nfl_id','frame_id','x','y']]])
    else:
        print("Could not detect columns for", p)

# concat all tracking parts
tracking = pd.concat(tracking_parts, ignore_index=True)
print("Total tracking rows:", len(tracking))
del tracking_parts
gc.collect()

# --------- Merge input_df info (player_role, ball_land_x/y) into tracking by (game_id,play_id,nfl_id) ----------
if not input_df.empty:
    # make sure keys have matching dtype
    for c in ['game_id','play_id','nfl_id']:
        if c in input_df.columns and c in tracking.columns:
            try:
                input_df[c] = pd.to_numeric(input_df[c], errors='coerce').astype('Int64')
                tracking[c] = pd.to_numeric(tracking[c], errors='coerce').astype('Int64')
            except Exception:
                pass
    player_info = input_df[['game_id','play_id','nfl_id','player_role','ball_land_x','ball_land_y']].drop_duplicates()
    tracking = tracking.merge(player_info, on=['game_id','play_id','nfl_id'], how='left')

# --------- Convert dtypes to save memory where possible ----------
for c in ['game_id','play_id','nfl_id','frame_id']:
    if c in tracking.columns:
        tracking[c] = pd.to_numeric(tracking[c], errors='coerce').astype('Int64')
for c in ['x','y','s','dir','a','o']:
    if c in tracking.columns:
        tracking[c] = pd.to_numeric(tracking[c], errors='coerce').astype('float32')

# --------- Estimate velocities (vx, vy, s_est) per player over frames (vectorized) ----------
tracking = tracking.sort_values(['game_id','play_id','nfl_id','frame_id']).reset_index(drop=True)
grp = tracking.groupby(['game_id','play_id','nfl_id'], sort=False)
# compute framewise diffs
tracking[['dx','dy','dframe']] = grp[['x','y','frame_id']].diff().fillna(0)
# guard dframe==0
tracking['dframe'] = tracking['dframe'].replace(0, 1.0)
tracking['vx'] = tracking['dx'] / tracking['dframe']
tracking['vy'] = tracking['dy'] / tracking['dframe']
tracking['s_est'] = np.sqrt(tracking['vx'].astype(float)**2 + tracking['vy'].astype(float)**2)
# cleanup temporary cols
tracking.drop(columns=[c for c in ['dx','dy','dframe'] if c in tracking.columns], inplace=True)
gc.collect()

# --------- Define metric function (same as earlier; compact) ----------
def calculate_play_metrics_with_route_mirroring(play_df):
    # Keep compact and robust — return a dict for this play
    keys = ['game_id','play_id','primary_defender_id','receiver_id','coverage_tightness','ball_hawk_score','route_mirroring']
    out = {k: np.nan for k in keys}
    try:
        pdf = play_df.copy()
        # basic guards
        if 'player_role' not in pdf.columns or {'x','y','frame_id'}.difference(pdf.columns):
            return out
        pdf['role_lower'] = pdf['player_role'].astype(str).str.lower()
        rec_mask = pdf['role_lower'].str.contains('target|receiver', na=False)
        def_mask = pdf['role_lower'].str.contains('defens|coverage|defend|db', na=False)
        if not rec_mask.any() or not def_mask.any():
            return out
        receiver = pdf[rec_mask].copy()
        defenders = pdf[def_mask].copy()
        start_frame = int(pdf['frame_id'].min())
        rec_start = receiver[receiver['frame_id']==start_frame]
        def_start = defenders[defenders['frame_id']==start_frame]
        if rec_start.empty:
            rec_start = receiver.groupby('nfl_id', dropna=True).first().reset_index()
        if def_start.empty:
            def_start = defenders.groupby('nfl_id', dropna=True).first().reset_index()
        if rec_start.empty or def_start.empty:
            return out
        # ball landing
        if 'ball_land_x' not in pdf.columns or pdf['ball_land_x'].dropna().empty:
            return out
        ball_x = float(pdf['ball_land_x'].dropna().iloc[0])
        ball_y = float(pdf['ball_land_y'].dropna().iloc[0])
        rec_row = rec_start.iloc[0]
        rec_x = rec_row.get('x', np.nan); rec_y = rec_row.get('y', np.nan)
        out['receiver_id'] = rec_row.get('nfl_id', np.nan)
        # primary defender at start
        def_start = def_start.copy()
        def_start['dist_to_rec'] = np.sqrt((def_start.get('x',0)-rec_x)**2 + (def_start.get('y',0)-rec_y)**2)
        if def_start['dist_to_rec'].isnull().all():
            return out
        primary_idx = def_start['dist_to_rec'].idxmin()
        primary_defender_id = def_start.loc[primary_idx, 'nfl_id']
        out['primary_defender_id'] = primary_defender_id
        # build paths
        recv_cols = ['frame_id','x','y','vx','vy'] if {'vx','vy'}.issubset(receiver.columns) else ['frame_id','x','y']
        defender_cols = ['frame_id','x','y','vx','vy'] if {'vx','vy'}.issubset(defenders.columns) else ['frame_id','x','y']
        recv_path = receiver[[c for c in recv_cols if c in receiver.columns]].rename(columns={'x':'rec_x','y':'rec_y','vx':'rec_vx','vy':'rec_vy'})
        defender_path = defenders[defenders['nfl_id']==primary_defender_id][[c for c in defender_cols if c in defenders.columns]].rename(columns={'x':'def_x','y':'def_y','vx':'def_vx','vy':'def_vy'})
        if recv_path.empty or defender_path.empty:
            return out
        merged = pd.merge(recv_path, defender_path, on='frame_id', how='inner').sort_values('frame_id')
        if merged.empty:
            return out
        # coverage
        merged['separation'] = np.sqrt((merged['rec_x']-merged['def_x'])**2 + (merged['rec_y']-merged['def_y'])**2)
        out['coverage_tightness'] = float(merged['separation'].mean())
        # ball hawk
        merged['ball_vec_x'] = ball_x - merged['def_x']; merged['ball_vec_y'] = ball_y - merged['def_y']
        norm_ball = np.sqrt(merged['ball_vec_x']**2 + merged['ball_vec_y']**2).replace(0,1.0)
        merged['ball_dir_x'] = merged['ball_vec_x'] / norm_ball; merged['ball_dir_y'] = merged['ball_vec_y'] / norm_ball
        if {'def_vx','def_vy'}.issubset(merged.columns):
            closing_speed = merged['def_vx'] * merged['ball_dir_x'] + merged['def_vy'] * merged['ball_dir_y']
            out['ball_hawk_score'] = float(closing_speed.mean())
        else:
            out['ball_hawk_score'] = np.nan
        # route mirroring: get rec_vx/rec_vy
        if 'rec_vx' in merged.columns and 'rec_vy' in merged.columns:
            merged['r_vx'] = merged['rec_vx']; merged['r_vy'] = merged['rec_vy']
        else:
            merged['r_vx'] = merged['rec_x'].diff().fillna(0); merged['r_vy'] = merged['rec_y'].diff().fillna(0)
        if 'def_vx' not in merged.columns or 'def_vy' not in merged.columns:
            merged['d_vx'] = merged['def_x'].diff().fillna(0); merged['d_vy'] = merged['def_y'].diff().fillna(0)
        else:
            merged['d_vx'] = merged['def_vx']; merged['d_vy'] = merged['def_vy']
        merged['r_speed'] = np.sqrt(merged['r_vx']**2 + merged['r_vy']**2)
        merged['d_speed'] = np.sqrt(merged['d_vx']**2 + merged['d_vy']**2)
        valid = (merged['r_speed'] > 1e-6) & (merged['d_speed'] > 1e-6)
        if valid.any():
            merged.loc[valid, 'r_dir_x'] = merged.loc[valid,'r_vx']/merged.loc[valid,'r_speed']
            merged.loc[valid, 'r_dir_y'] = merged.loc[valid,'r_vy']/merged.loc[valid,'r_speed']
            merged.loc[valid, 'd_dir_x'] = merged.loc[valid,'d_vx']/merged.loc[valid,'d_speed']
            merged.loc[valid, 'd_dir_y'] = merged.loc[valid,'d_vy']/merged.loc[valid,'d_speed']
            merged.loc[valid, 'dir_score'] = (merged.loc[valid,'r_dir_x']*merged.loc[valid,'d_dir_x'] + merged.loc[valid,'r_dir_y']*merged.loc[valid,'d_dir_y']).clip(-1.0,1.0)
            merged.loc[valid, 'speed_ratio'] = np.minimum(merged.loc[valid,'r_speed'], merged.loc[valid,'d_speed']) / np.maximum(merged.loc[valid,'r_speed'], merged.loc[valid,'d_speed'])
            merged.loc[valid, 'mirror_frame'] = merged.loc[valid,'dir_score'] * merged.loc[valid,'speed_ratio']
            out['route_mirroring'] = float(merged.loc[valid,'mirror_frame'].mean())
        else:
            out['route_mirroring'] = np.nan
        out['game_id'] = int(pdf['game_id'].iloc[0]) if 'game_id' in pdf.columns else np.nan
        out['play_id'] = int(pdf['play_id'].iloc[0]) if 'play_id' in pdf.columns else np.nan
        return out
    except Exception as e:
        # optional: print("error", e)
        return out

# --------- Prepare per-play keys to process in parallel ----------
plays_keys = tracking[['game_id','play_id']].drop_duplicates().dropna().astype(int)
play_list = list(plays_keys.itertuples(index=False, name=None))
print("Total plays to process:", len(play_list))

# helper to extract play df and compute metrics
def process_play_pair(gid_pid):
    gid, pid = gid_pid
    play_df = tracking[(tracking['game_id']==gid) & (tracking['play_id']==pid)]
    if play_df.empty:
        return None
    return calculate_play_metrics_with_route_mirroring(play_df)

# run in parallel (joblib)
results = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(process_play_pair)(kp) for kp in tqdm(play_list, desc="Processing plays"))

# collect results
metrics = [r for r in results if r is not None]
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT_PLAY_METRICS, index=False)
print("Saved per-play metrics to:", OUT_PLAY_METRICS, "rows:", len(metrics_df))

# --------- Defender-level leaderboard (aggregate) ----------
metrics_df['primary_defender_id'] = pd.to_numeric(metrics_df['primary_defender_id'], errors='coerce').astype('Int64')
metrics_df = metrics_df.dropna(subset=['primary_defender_id'])
agg = metrics_df.groupby('primary_defender_id').agg(
    plays_count=('coverage_tightness','count'),
    avg_coverage=('coverage_tightness','mean'),
    avg_ballhawk=('ball_hawk_score','mean'),
    avg_mirroring=('route_mirroring','mean')
).reset_index()
agg = agg[agg['plays_count'] >= MIN_PLAYS_FOR_LEADERBOARD].copy()

# normalize metrics, compute ODS
def minmax(series):
    mn, mx = series.min(), series.max()
    if mn == mx or pd.isna(mn) or pd.isna(mx):
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)
agg['coverage_norm'] = minmax(agg['avg_coverage'])
agg['coverage_inv'] = 1.0 - agg['coverage_norm']
agg['ballhawk_norm'] = minmax(agg['avg_ballhawk'])
agg['mirror_norm'] = minmax(agg['avg_mirroring'])
agg['ODS'] = agg['coverage_inv']*W_COV + agg['ballhawk_norm']*W_BH + agg['mirror_norm']*W_MIR

# try merge roster if a players.csv exists
roster_path = None
for p in all_csvs:
    if os.path.basename(p).lower() == 'players.csv':
        roster_path = p; break
if roster_path:
    roster = pd.read_csv(roster_path, low_memory=False)
    id_col = next((c for c in ['nfl_id','nflId','player_id','id'] if c in roster.columns), None)
    name_col = next((c for c in ['display_name','full_name','player_name','name'] if c in roster.columns), None)
    pos_col = next((c for c in ['position','player_position','pos'] if c in roster.columns), None)
    roster['nfl_id'] = pd.to_numeric(roster[id_col], errors='coerce').astype('Int64') if id_col else None
    roster['display_name'] = roster[name_col] if name_col in roster.columns else roster.get('display_name', None)
    roster['position'] = roster[pos_col] if pos_col in roster.columns else roster.get('position', None)
    roster = roster[['nfl_id','display_name','position']].drop_duplicates(subset=['nfl_id'])
    agg_named = agg.merge(roster, left_on='primary_defender_id', right_on='nfl_id', how='left').drop(columns=['nfl_id'])
else:
    agg_named = agg.copy()
    agg_named['display_name'] = agg_named['primary_defender_id'].astype(str)
    agg_named['position'] = 'UNK'

agg_named = agg_named.sort_values('ODS', ascending=False).reset_index(drop=True)
agg_named.to_csv(OUT_LEADERBOARD, index=False)
print("Saved defender leaderboard to:", OUT_LEADERBOARD, "rows:", len(agg_named))

# -- Done --
print("Full run complete. Play-metrics rows:", len(metrics_df), "Defender rows:", len(agg_named))



# Diagnostic step (run this cell)
import os, pandas as pd
from pathlib import Path

print("== Diagnostic check: play_metrics_full.csv and input files ==")
pm_path = "play_metrics_full.csv"
if os.path.exists(pm_path):
    pm = pd.read_csv(pm_path)
    print(f"\nplay_metrics file found: {pm_path}  rows = {len(pm)}")
    print("Columns:", pm.columns.tolist())
    if 'primary_defender_id' in pm.columns:
        nonnull = pm['primary_defender_id'].notna().sum()
        pct = 100.0 * nonnull / len(pm) if len(pm)>0 else 0.0
        print(f"primary_defender_id non-null: {nonnull} / {len(pm)}  ({pct:.1f}%)")
    else:
        print("primary_defender_id column: NOT PRESENT")
    print("\nFirst 8 rows of play_metrics (showing key cols):")
    cols_to_show = [c for c in ['game_id','play_id','primary_defender_id','receiver_id','coverage_tightness','ball_hawk_score','route_mirroring'] if c in pm.columns]
    display(pm[cols_to_show].head(8))
else:
    print(f"\nplay_metrics file NOT found at: {pm_path}")
    pm = None

# Search for input-like files under /kaggle/input
print("\nSearching /kaggle/input for candidate input files (player_role / ball_land columns)...")
candidates = []
for p in Path('/kaggle/input').rglob('*.csv'):
    name = p.name.lower()
    if 'input' in name or 'player' in name or 'train' in name:
        candidates.append(str(p))
# Deduplicate and print a few
candidates = list(dict.fromkeys(candidates))
print("Candidate CSVs found:", len(candidates))
for c in candidates[:10]:
    print(" -", c)

# Show sample columns for promising candidates (first one or two)
checked = 0
for c in candidates[:6]:
    try:
        sample = pd.read_csv(c, nrows=3)
    except Exception as e:
        print("  cannot read", c, ":", e)
        continue
    cols = sample.columns.tolist()
    # check for required columns
    has_role = any(x.lower()=='player_role' or 'role' in x.lower() for x in cols)
    has_ball = any('ball_land' in x.lower() for x in cols)
    has_nfl = any(x.lower()=='nfl_id' or 'nfl' in x.lower() for x in cols)
    print(f"\nSample columns for {c}:")
    print(cols)
    print("Contains player_role?", has_role, "| contains ball_land_x/ball_land_y?", has_ball, "| contains nfl_id?", has_nfl)
    checked += 1
    if checked>=3:
        break

# Final guidance message
print("\n=== Guidance ===")
if pm is None:
    print("1) play_metrics_full.csv missing. Re-run the per-play metric pipeline (full-run cell).")
else:
    if 'primary_defender_id' not in pm.columns or pm['primary_defender_id'].notna().sum() == 0:
        print("2) primary_defender_id is missing or all-null in play_metrics_full.csv.")
        print("   Likely cause: the pipeline did not merge the `input` files containing player_role/ball_land_x before computing metrics.")
        print("   Next action: locate the correct input CSV (one of the candidate files above should contain 'player_role' and 'ball_land_x'), then re-run the full metrics cell so each play is computed with player_role & ball landing info.")
    else:
        print("3) primary_defender_id is present. Good — you can regenerate the defender leaderboard from play_metrics_full.csv now.")
        print("   If you want, run: ")
        print("      # rebuild leaderboard from saved play metrics")
        print("      import pandas as pd")
        print("      pm = pd.read_csv('play_metrics_full.csv')")
        print("      pm['primary_defender_id']=pd.to_numeric(pm['primary_defender_id'],errors='coerce').astype('Int64')")
        print("      pm = pm.dropna(subset=['primary_defender_id'])")
        print("      # then aggregate & compute ODS as earlier")
        
print("\nAfter running this cell, paste the printed output here and I'll give the exact one-line or one-cell fix to run next.")



# Robust locator + test-merger for tracking/input files
import os
from pathlib import Path
import pandas as pd
import re

ROOT = Path("/kaggle/input")
print("Searching under:", ROOT)

# 1) list top CSV files to inspect names (first 60)
all_csvs = list(ROOT.rglob("*.csv"))
print("Total CSVs found:", len(all_csvs))
print("\nExample CSVs (first 60):")
for p in all_csvs[:60]:
    print(" ", p)

# 2) Heuristic functions to detect tracking vs input by peeking columns
def is_tracking(path):
    try:
        cols = pd.read_csv(path, nrows=2).columns.str.lower().tolist()
    except Exception:
        return False
    # tracking usually has frame_id, x, y and nfl_id
    return ('frame_id' in cols and 'x' in cols and 'y' in cols)

def is_input(path):
    try:
        cols = pd.read_csv(path, nrows=2).columns.str.lower().tolist()
    except Exception:
        return False
    # input files often have player_role and ball_land_x
    return ('player_role' in cols or 'ball_land_x' in cols or 'player_name' in cols)

# 3) find candidates
tracking_candidates = []
input_candidates = []
for p in all_csvs:
    pstr = str(p)
    if is_tracking(pstr):
        tracking_candidates.append(pstr)
    if is_input(pstr):
        input_candidates.append(pstr)

print(f"\nHeuristic results -> tracking candidates: {len(tracking_candidates)}, input candidates: {len(input_candidates)}")

# show examples
print("\nFirst 12 tracking candidates:")
for t in tracking_candidates[:12]:
    print(" ", t)
print("\nFirst 12 input candidates:")
for t in input_candidates[:12]:
    print(" ", t)

# 4) try pairing by week/order using filename patterns
def extract_week_identifier(path_str):
    # typical filenames: input_2023_w01.csv or output_2023_w01.csv etc.
    m = re.search(r'(?:w|week|_w)(\d{1,2})', path_str, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # try year-week pattern like 20230907 or _w01 alternative patterns
    m2 = re.search(r'2023[_-]?w?(\d{1,2})', path_str, flags=re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return None

# build dicts
track_by_week = {}
for p in tracking_candidates:
    w = extract_week_identifier(p)
    track_by_week.setdefault(w, []).append(p)

input_by_week = {}
for p in input_candidates:
    w = extract_week_identifier(p)
    input_by_week.setdefault(w, []).append(p)

print("\nMatched weeks (example keys):")
print(" tracking weeks:", sorted(k for k in track_by_week.keys() if k is not None)[:10])
print(" input weeks:   ", sorted(k for k in input_by_week.keys() if k is not None)[:10])

# 5) create list of matched pairs (week-based first), then fallback to best-effort by reading columns
pairs = []
# match by week id when possible
for w in sorted(set(list(track_by_week.keys()) + list(input_by_week.keys()))):
    tlist = track_by_week.get(w, [])
    ilist = input_by_week.get(w, [])
    if tlist and ilist:
        # pair the first track file with first input file for this week
        pairs.append((tlist[0], ilist[0]))
# fallback: if no week-based pairs, try to pair first available tracking with first input
if not pairs and tracking_candidates and input_candidates:
    pairs.append((tracking_candidates[0], input_candidates[0]))

print(f"\nNumber of initial pairs found: {len(pairs)}")
if len(pairs) > 0:
    print("\nFirst pair:")
    print(" TRACKING:", pairs[0][0])
    print(" INPUT:   ", pairs[0][1])
else:
    print("No pairs found by week; trying column-based best-effort pairing...")

# 6) If no pairs yet, attempt best-effort by checking which tracking file's game_id/play_id match input file game_id/play_id sample
if not pairs:
    for t in tracking_candidates[:10]:
        for i in input_candidates[:10]:
            try:
                tdf = pd.read_csv(t, nrows=20)
                idf = pd.read_csv(i, nrows=20)
                # if share many game_id/play_id pairs in sample, treat as match
                common = set(tdf.columns).intersection(set(idf.columns))
                if {'game_id','play_id','nfl_id'}.issubset({c.lower() for c in tdf.columns}) and {'game_id','play_id'}.issubset({c.lower() for c in idf.columns}):
                    pairs.append((t,i)); break
            except Exception:
                continue
        if pairs: break
    if pairs:
        print("Found a pair via column-based heuristic:")
        print(" TRACKING:", pairs[0][0])
        print(" INPUT:   ", pairs[0][1])

# 7) Try a test merge of first matched pair to confirm necessary columns exist
if pairs:
    tpath, ipath = pairs[0]
    print("\nAttempting a test merge of the first pair (reading small number of rows)...")
    try:
        tdf = pd.read_csv(tpath, nrows=200)
        idf = pd.read_csv(ipath, nrows=200)
        # lower-case columns map
        print("Tracking columns:", tdf.columns.tolist())
        print("Input columns:   ", idf.columns.tolist())
        # try merge
        if 'game_id' in tdf.columns and 'play_id' in tdf.columns:
            merged = tdf.merge(idf, on=['game_id','play_id'], how='left', suffixes=('_trk','_in'))
            print("\nMerged columns sample:", merged.columns.tolist()[:40])
            # show presence of key columns
            key_present = { 'player_role': 'player_role' in merged.columns or 'player_role' in [c.lower() for c in merged.columns],
                           'ball_land_x': any('ball_land_x'==c.lower() for c in merged.columns) or any('ball_land' in c.lower() for c in merged.columns)}
            print("Key columns present after merge:", key_present)
            display(merged.head(6))
        else:
            print("TRACKING file lacks 'game_id'/'play_id' columns in sample; cannot merge reliably.")
    except Exception as e:
        print("Test merge failed:", e)
else:
    print("No candidate pair found. Please paste a few CSV names from /kaggle/input that look like tracking files (or attach a screenshot).")

# Final instructions printed:
print("\n--- NEXT STEP ---")
if pairs:
    print("If the test merge above included player_role and ball_land columns, we can proceed to re-run the full pipeline pairing each matching tracking file with its input file.")
    print("Say 'yes' and I will produce the exact full-run code that loops through all pairs, merges them and recomputes metrics.")
else:
    print("No suitable tracking/input pairs found automatically. If you see tracking files in the 'Example CSVs' list above, tell me one full path for a tracking file and one full path for its corresponding input file and I'll write the merge + metric rerun cell for you.")


# ====== Full re-run over all matched tracking+input files ======
import os, re, gc
from pathlib import Path
import pandas as pd, numpy as np
from joblib import Parallel, delayed
import multiprocessing
from tqdm.auto import tqdm

ROOT = Path("/kaggle/input")
OUT_PLAY = "play_metrics_full.csv"
OUT_LEADER = "defender_leaderboard_full.csv"
MIN_PLAYS = 5
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# ------- helper: find CSVs and classify -------
all_csvs = list(ROOT.rglob("*.csv"))
def peek_cols(p):
    try:
        return pd.read_csv(p, nrows=2).columns.tolist()
    except Exception:
        return []

def is_tracking(p):
    cols = [c.lower() for c in peek_cols(p)]
    return ('frame_id' in cols and 'x' in cols and 'y' in cols)

def is_input(p):
    cols = [c.lower() for c in peek_cols(p)]
    return ('player_role' in cols or 'ball_land_x' in cols or 'player_to_predict' in cols or 'player_name' in cols)

# build lists
tracking_files = [str(p) for p in all_csvs if is_tracking(p)]
input_files    = [str(p) for p in all_csvs if is_input(p)]

print("Found tracking files:", len(tracking_files), "input files:", len(input_files))
if len(tracking_files)==0 or len(input_files)==0:
    raise RuntimeError("Could not locate tracking or input files automatically. Inspect /kaggle/input and rerun.")

# ------- pair by week identifier when possible -------
def extract_week(path):
    s = str(path)
    m = re.search(r'(?:_w|_week|w)(\d{1,2})', s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.search(r'2023(\d{2})', s)   # sometimes date-like chunk
    if m2:
        # fallback: return two-digit chunk
        try:
            return int(m2.group(1))
        except:
            return None
    return None

track_by_week = {}
for p in tracking_files:
    w = extract_week(p)
    track_by_week.setdefault(w, []).append(p)
input_by_week = {}
for p in input_files:
    w = extract_week(p)
    input_by_week.setdefault(w, []).append(p)

pairs = []
# pair same-week first
for w in sorted(set(list(track_by_week.keys()) + list(input_by_week.keys()))):
    if w in track_by_week and w in input_by_week:
        pairs.append((track_by_week[w][0], input_by_week[w][0]))
# fallback: zip-by-order if not enough pairs
if not pairs:
    m = min(len(tracking_files), len(input_files))
    pairs = list(zip(tracking_files[:m], input_files[:m]))

print("Prepared", len(pairs), "tracking+input pairs to process (sample):")
for a,b in pairs[:6]:
    print("  ", Path(a).name, "<-->", Path(b).name)

# ------- robust metric function (same logic, returns dict) -------
def calculate_play_metrics_with_route_mirroring(play_df):
    out = {
        'game_id': np.nan, 'play_id': np.nan,
        'primary_defender_id': np.nan, 'receiver_id': np.nan,
        'coverage_tightness': np.nan, 'ball_hawk_score': np.nan, 'route_mirroring': np.nan
    }
    try:
        pdf = play_df.copy()
        # ensure necessary columns
        if 'player_role' not in pdf.columns or not {'x','y','frame_id'}.issubset(set(pdf.columns)):
            return out
        pdf['role_lower'] = pdf['player_role'].astype(str).str.lower()
        recv = pdf[pdf['role_lower'].str.contains('target|targeted|receiver', na=False)]
        defs = pdf[pdf['role_lower'].str.contains('defens|coverage|defend|db|cb|safety|lb|olb|mlb', na=False)]
        if recv.empty or defs.empty:
            return out
        start_frame = int(pdf['frame_id'].min())
        rec_start = recv[recv['frame_id']==start_frame] if not recv[recv['frame_id']==start_frame].empty else recv.groupby('nfl_id', dropna=True).first().reset_index()
        def_start = defs[defs['frame_id']==start_frame] if not defs[defs['frame_id']==start_frame].empty else defs.groupby('nfl_id', dropna=True).first().reset_index()
        if rec_start.empty or def_start.empty:
            return out
        # ball landing coords
        if 'ball_land_x' not in pdf.columns or pdf['ball_land_x'].dropna().empty:
            return out
        ball_x = float(pdf['ball_land_x'].dropna().iloc[0])
        ball_y = float(pdf['ball_land_y'].dropna().iloc[0])
        rec_row = rec_start.iloc[0]
        rec_x, rec_y = rec_row.get('x',np.nan), rec_row.get('y',np.nan)
        out['receiver_id'] = rec_row.get('nfl_id', np.nan)
        # choose primary defender closest at start
        def_start = def_start.copy()
        def_start['dist_to_rec'] = np.sqrt((def_start.get('x',0)-rec_x)**2 + (def_start.get('y',0)-rec_y)**2)
        if def_start['dist_to_rec'].isnull().all():
            return out
        primary_idx = def_start['dist_to_rec'].idxmin()
        primary_defender_id = def_start.loc[primary_idx,'nfl_id']
        out['primary_defender_id'] = primary_defender_id
        # build time series
        recv_path = recv[['frame_id','x','y']].rename(columns={'x':'rec_x','y':'rec_y'})
        defender_path = defs[defs['nfl_id']==primary_defender_id][['frame_id','x','y']].rename(columns={'x':'def_x','y':'def_y'})
        merged = pd.merge(recv_path, defender_path, on='frame_id', how='inner').sort_values('frame_id')
        if merged.empty:
            return out
        merged['separation'] = np.sqrt((merged['rec_x']-merged['def_x'])**2 + (merged['rec_y']-merged['def_y'])**2)
        out['coverage_tightness'] = float(merged['separation'].mean())
        # ball-hawk (need def velocity); estimate defender velocity if not provided
        if {'def_vx','def_vy'}.issubset(merged.columns):
            vx = merged['def_vx']; vy = merged['def_vy']
        else:
            # approximate via diff of def_x/def_y
            merged['def_vx'] = merged['def_x'].diff().fillna(0)
            merged['def_vy'] = merged['def_y'].diff().fillna(0)
            vx = merged['def_vx']; vy = merged['def_vy']
        merged['ball_vec_x'] = ball_x - merged['def_x']; merged['ball_vec_y'] = ball_y - merged['def_y']
        normb = np.sqrt(merged['ball_vec_x']**2 + merged['ball_vec_y']**2).replace(0,1.0)
        merged['ball_dir_x'] = merged['ball_vec_x'] / normb; merged['ball_dir_y'] = merged['ball_vec_y'] / normb
        closing = vx * merged['ball_dir_x'] + vy * merged['ball_dir_y']
        out['ball_hawk_score'] = float(closing.mean())
        # route mirroring: use receiver diff for direction & defender diff
        merged['r_vx'] = merged['rec_x'].diff().fillna(0); merged['r_vy'] = merged['rec_y'].diff().fillna(0)
        merged['d_vx'] = vx; merged['d_vy'] = vy
        merged['r_speed'] = np.sqrt(merged['r_vx']**2 + merged['r_vy']**2)
        merged['d_speed'] = np.sqrt(merged['d_vx']**2 + merged['d_vy']**2)
        valid = (merged['r_speed'] > 1e-6) & (merged['d_speed'] > 1e-6)
        if valid.any():
            merged.loc[valid, 'r_dir_x'] = merged.loc[valid,'r_vx']/merged.loc[valid,'r_speed']
            merged.loc[valid, 'r_dir_y'] = merged.loc[valid,'r_vy']/merged.loc[valid,'r_speed']
            merged.loc[valid, 'd_dir_x'] = merged.loc[valid,'d_vx']/merged.loc[valid,'d_speed']
            merged.loc[valid, 'd_dir_y'] = merged.loc[valid,'d_vy']/merged.loc[valid,'d_speed']
            merged.loc[valid, 'dir_score'] = (merged.loc[valid,'r_dir_x']*merged.loc[valid,'d_dir_x'] + merged.loc[valid,'r_dir_y']*merged.loc[valid,'d_dir_y']).clip(-1,1)
            merged.loc[valid, 'speed_ratio'] = np.minimum(merged.loc[valid,'r_speed'], merged.loc[valid,'d_speed']) / np.maximum(merged.loc[valid,'r_speed'], merged.loc[valid,'d_speed'])
            merged.loc[valid, 'mirror_frame'] = merged.loc[valid,'dir_score'] * merged.loc[valid,'speed_ratio']
            out['route_mirroring'] = float(merged.loc[valid,'mirror_frame'].mean())
        else:
            out['route_mirroring'] = np.nan
        out['game_id'] = int(pdf['game_id'].iloc[0]) if 'game_id' in pdf.columns else np.nan
        out['play_id'] = int(pdf['play_id'].iloc[0]) if 'play_id' in pdf.columns else np.nan
        return out
    except Exception:
        return out

# ------- core loop: for each matched pair, merge & process plays -------
all_play_results = []
for tpath, ipath in tqdm(pairs, desc="Pairs"):
    # load sensible columns to reduce memory load
    tcols = peek_cols(tpath)
    icol = peek_cols(ipath)
    use_tcols = [c for c in ['game_id','play_id','nfl_id','frame_id','x','y','s','dir'] if c in tcols]
    use_icols = [c for c in ['game_id','play_id','nfl_id','player_role','ball_land_x','ball_land_y','player_to_predict','player_position','player_name'] if c in icol]
    try:
        trk = pd.read_csv(tpath, usecols=use_tcols, low_memory=False)
        inp = pd.read_csv(ipath, usecols=use_icols, low_memory=False)
    except Exception:
        trk = pd.read_csv(tpath, low_memory=False)
        inp = pd.read_csv(ipath, low_memory=False)
    # ensure key dtypes
    for c in ['game_id','play_id','nfl_id','frame_id']:
        if c in trk.columns:
            trk[c] = pd.to_numeric(trk[c], errors='coerce')
        if c in inp.columns:
            inp[c] = pd.to_numeric(inp[c], errors='coerce')
    # merge on game/play/nfl_id
    merged = trk.merge(inp, on=['game_id','play_id','nfl_id'], how='left', suffixes=('_trk','_in'))
    # Standardize role: create player_role if missing
    if 'player_role' not in merged.columns:
        if 'player_to_predict' in merged.columns:
            # mark targeted receiver where True
            merged['player_role'] = merged['player_to_predict'].astype(bool).map({True:'Targeted Receiver', False:'Other Route Runner'})
        else:
            # use position heuristics: defensive positions -> Defensive Coverage
            def is_def_pos(p):
                if pd.isna(p): return False
                s = str(p).upper()
                for tok in ['CB','DB','S','FS','SS','LB','MLB','OLB','SAFETY']:
                    if tok in s: return True
                return False
            merged['player_role'] = merged.get('player_position', '').apply(lambda v: 'Defensive Coverage' if is_def_pos(v) else 'Other Route Runner')
    # ensure ball_land_x/y exist (if input uses different names, try variants)
    if 'ball_land_x' not in merged.columns and 'ball_land_x_in' in merged.columns:
        merged['ball_land_x'] = merged['ball_land_x_in']
    if 'ball_land_y' not in merged.columns and 'ball_land_y_in' in merged.columns:
        merged['ball_land_y'] = merged['ball_land_y_in']
    # compute simple velocities if not present
    if not {'vx','vy'}.issubset(set(merged.columns)):
        merged = merged.sort_values(['game_id','play_id','nfl_id','frame_id']).reset_index(drop=True)
        grp = merged.groupby(['game_id','play_id','nfl_id'], sort=False)
        merged[['dx','dy','dframe']] = grp[['x','y','frame_id']].diff().fillna(0)
        merged['dframe'] = merged['dframe'].replace(0,1.0)
        merged['vx'] = merged['dx'] / merged['dframe']
        merged['vy'] = merged['dy'] / merged['dframe']
        # drop temp
        merged.drop(columns=[c for c in ['dx','dy','dframe'] if c in merged.columns], inplace=True)
    # now iterate plays in this merged chunk
    play_keys = merged[['game_id','play_id']].drop_duplicates().dropna()
    play_list = list(play_keys.itertuples(index=False, name=None))
    # process each play (serially or in small parallel batch to keep memory stable)
    # here use local parallelization per pair for speed
    def process_one_play(k):
        gid,pid = k
        pdf = merged[(merged['game_id']==gid) & (merged['play_id']==pid)]
        return calculate_play_metrics_with_route_mirroring(pdf)
    results = Parallel(n_jobs=max(1, min(N_JOBS, 4)), backend="loky")(delayed(process_one_play)(k) for k in play_list)
    all_play_results.extend(results)
    # memory cleanup
    del merged, trk, inp, play_keys, play_list, results
    gc.collect()

# ------- collect results and save -------
play_df = pd.DataFrame(all_play_results)
play_df.to_csv(OUT_PLAY, index=False)
print("Saved per-play metrics:", OUT_PLAY, "rows:", len(play_df))

# ------- defender-level aggregation -------
if 'primary_defender_id' in play_df.columns:
    play_df['primary_defender_id'] = pd.to_numeric(play_df['primary_defender_id'], errors='coerce').astype('Int64')
    valid = play_df.dropna(subset=['primary_defender_id']).copy()
    print("Valid plays with defender id:", len(valid))
    agg = valid.groupby('primary_defender_id').agg(
        plays_count=('coverage_tightness','count'),
        avg_coverage=('coverage_tightness','mean'),
        avg_ballhawk=('ball_hawk_score','mean'),
        avg_mirroring=('route_mirroring','mean')
    ).reset_index()
    agg = agg[agg['plays_count'] >= MIN_PLAYS].copy()
    # normalize + ODS
    def minmax(s):
        if s.isna().all(): return s
        mn,mx = s.min(), s.max()
        if mn==mx: return pd.Series(0.5,index=s.index)
        return (s-mn)/(mx-mn)
    agg['coverage_inv'] = 1.0 - minmax(agg['avg_coverage'])
    agg['ballhawk_norm'] = minmax(agg['avg_ballhawk'])
    agg['mirror_norm'] = minmax(agg['avg_mirroring'])
    agg['ODS'] = agg['coverage_inv']*0.40 + agg['ballhawk_norm']*0.25 + agg['mirror_norm']*0.35
    agg = agg.sort_values('ODS', ascending=False).reset_index(drop=True)
    agg.to_csv(OUT_LEADER, index=False)
    print("Saved defender leaderboard:", OUT_LEADER, "rows:", len(agg))
else:
    print("primary_defender_id not in play_df columns; leaderboard cannot be built.")

# final summary
print("Done. play rows:", len(play_df))


