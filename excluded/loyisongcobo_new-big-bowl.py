











import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


# Tracking assumptions
FRAME_DT = 0.1                  # seconds per frame (10Hz)


# Commitment hyperparameters 
COS_THETA_THRESHOLD = 0.7       # directional commitment
SUSTAIN_FRAMES = 2              # sustained frames required
DEFENDER_RADIUS = 15.0          # yards around receiver


# Data Setup
ROOT = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final"

TRAIN_DIR = f"{ROOT}/train"
SUPP_DIR  = ROOT  

# --- Check directory structure ---
print("Directory structure:")
for path, subdirs, files in os.walk(ROOT):
    print(path)
    for f in files[:5]:
        print("    ", f)
    print()

# Input files (pre-throw)
input_files = sorted(glob.glob(f"{TRAIN_DIR}/input_2023_w*.csv"))
print(f"Found {len(input_files)} input files")
df_input = pd.concat([pd.read_csv(f) for f in input_files], ignore_index=True)

# Output files (post-throw)
output_files = sorted(glob.glob(f"{TRAIN_DIR}/output_2023_w*.csv"))
print(f"Found {len(output_files)} output files")
df_output = pd.concat([pd.read_csv(f) for f in output_files], ignore_index=True)

# Supplementary (metadata)
supp_files = glob.glob(f"{ROOT}/supplementary*.csv")
print(f"Found {len(supp_files)} supplementary files")
df_supp = pd.concat([pd.read_csv(f, low_memory=False) for f in supp_files], ignore_index=True)

print(f"\nData shapes:")
print(f"df_input:  {df_input.shape}")
print(f"df_output: {df_output.shape}")
print(f"df_supp:   {df_supp.shape}")

# Show columns
print(f"\ndf_input columns: {list(df_input.columns)}")
print(f"\ndf_supp columns (first 20): {list(df_supp.columns)[:20]}")




# our valid_play_tuples available:
valid_play_tuples = df_input[df_input.player_role == 'Targeted Receiver'] \
    .groupby(['game_id', 'play_id'])['nfl_id'] \
    .first() \
    .reset_index() \
    .apply(lambda row: (row['game_id'], row['play_id'], row['nfl_id']), axis=1) \
    .tolist()

print(f"Found {len(valid_play_tuples)} valid plays")


def angle_to_unit_vector(deg):
    rad = np.deg2rad(deg)
    return np.array([np.cos(rad), np.sin(rad)])


def unit_vector_to(A, B):
    v = B - A
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.array([0.0, 0.0])
    return v / norm


def get_release_frame(df_input_play):
    """
    Ball release frame = last frame in input tracking for this play
    """
    return df_input_play["frame_id"].max()


def get_targeted_receiver(df_play):
    """
    Returns nfl_id of targeted receiver
    """
    recs = df_play[df_play["player_role"] == "Targeted Receiver"]
    if len(recs) != 1:
        raise ValueError("Play does not have exactly one targeted receiver.")
    return recs.iloc[0]["nfl_id"]


def get_defenders_near_receiver(df_frame, receiver_pos, radius=DEFENDER_RADIUS):
    defenders = df_frame[df_frame["player_side"] == "Defense"].copy()
    if defenders.empty:
        return defenders

    dx = defenders["x"].values - receiver_pos[0]
    dy = defenders["y"].values - receiver_pos[1]
    dist = np.sqrt(dx**2 + dy**2)

    defenders["dist_to_receiver"] = dist
    return defenders[defenders["dist_to_receiver"] <= radius]


def compute_cos_theta(defender_row, receiver_pos):
    """
    cos(theta) between defender velocity and vector to receiver
    """
    # Velocity direction unit vector
    v_hat = angle_to_unit_vector(defender_row["dir"])

    # Direction toward receiver
    d_hat = unit_vector_to(
        np.array([defender_row["x"], defender_row["y"]]),
        receiver_pos
    )

    return np.dot(v_hat, d_hat)


def find_plays_with_valid_receiver(df_input, df_supp=None, max_plays=None):
    """
    Find all plays that have a valid targeted receiver.
    
    Returns:
        valid_plays: List of (game_id, play_id, target_nfl_id) tuples
        error_log: List of (game_id, play_id, error_message) for debugging
    """
    print("Scanning for plays with valid targeted receiver...")
    
    valid_plays = []
    error_log = []
    
    # Get unique plays
    unique_plays = df_input[['game_id', 'play_id']].drop_duplicates()
    
    if max_plays is not None:
        unique_plays = unique_plays.head(max_plays)
    
    total_plays = len(unique_plays)
    print(f"Checking {total_plays:,} unique plays...")
    
    # Count plays with each role type (for diagnostics)
    role_counts = {}
    
    for idx, row in tqdm(unique_plays.iterrows(), total=len(unique_plays)):
        gid = row['game_id']
        pid = row['play_id']
        
        try:
            df_play = df_input[(df_input.game_id == gid) & (df_input.play_id == pid)]
            
            if df_play.empty:
                error_log.append((gid, pid, "No data in df_input"))
                continue
            
            targeted = df_play[df_play['player_role'] == 'Targeted Receiver']
            
            if targeted.empty:
                error_log.append((gid, pid, "No 'Targeted Receiver' role"))
                continue
            
            # Get unique targeted receiver IDs (should be 1, but handle edge cases)
            target_ids = targeted['nfl_id'].unique()
            
            if len(target_ids) > 1:
                # Multiple players marked as targeted? We'll take the most common
                target_counts = targeted['nfl_id'].value_counts()
                target_nfl = target_counts.index[0]
                error_log.append((gid, pid, f"Multiple target IDs: {target_ids}, using {target_nfl}"))
            else:
                target_nfl = target_ids[0]
            
            # Verify we have enough frames for analysis (at least 3 frames)
            target_frames = targeted['frame_id'].nunique()
            if target_frames < 3:
                error_log.append((gid, pid, f"Only {target_frames} frames for target"))
                continue
            
            # Track role distribution for this play
            roles = df_play['player_role'].unique()
            for role in roles:
                role_counts[role] = role_counts.get(role, 0) + 1
            

            valid_plays.append((gid, pid, int(target_nfl)))
            
        except Exception as e:
            error_log.append((gid, pid, f"Error: {str(e)}"))
    
    
    print(f"\nâœ… Found {len(valid_plays):,} valid plays out of {total_plays:,} total")
    
    if error_log:
        print(f"\nâ�Œ {len(error_log):,} plays had issues:")
        error_types = {}
        for _, _, error in error_log:
            error_types[error] = error_types.get(error, 0) + 1
        
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {error}: {count:,} plays")
    
    return valid_plays, error_log

# Running the function (we limit to first 1000 plays for quick test)
from tqdm import tqdm

valid_plays, errors = find_plays_with_valid_receiver(df_input, max_plays=1000)

# Now we can actually use valid_plays
if valid_plays:
    print(f"\nFirst 5 valid plays:")
    for i, (gid, pid, target_id) in enumerate(valid_plays[:5]):
        print(f"  {i+1}. game_id={gid}, play_id={pid}, target_nfl={target_id}")
    
    # Test with the first valid play
    g, p, target_nfl = valid_plays[0]
    
    print(f"\nTesting ASA computation on first valid play:")
    print(f"game_id={g}, play_id={p}, target_nfl={target_nfl}")
    
    # Get the data for this play
    df_in_play = df_input[(df_input.game_id == g) & (df_input.play_id == p)]
    df_out_play = df_output[(df_output.game_id == g) & (df_output.play_id == p)]
    
    print(f"Input frames: {df_in_play['frame_id'].nunique()}")
    print(f"Output frames: {df_out_play['frame_id'].nunique() if not df_out_play.empty else 0}")
    
    # Show the targeted receiver details
    target_rows = df_in_play[df_in_play['nfl_id'] == target_nfl]
    print(f"\nTargeted receiver: {target_rows.iloc[0]['player_name']} ({target_rows.iloc[0]['player_position']})")
    print(f"Frames available: {target_rows['frame_id'].nunique()}")
    
    # Show ball landing coordinates
    ball_land_x = target_rows.iloc[0]['ball_land_x']
    ball_land_y = target_rows.iloc[0]['ball_land_y']
    print(f"Ball landing: ({ball_land_x:.2f}, {ball_land_y:.2f})")
    
    # Check for defenders near receiver
    last_frame = df_in_play['frame_id'].max()
    last_frame_data = df_in_play[df_in_play['frame_id'] == last_frame]
    receiver_pos = target_rows[target_rows['frame_id'] == last_frame][['x', 'y']].iloc[0]
    
    defenders_nearby = last_frame_data[last_frame_data['player_side'] == 'Defense'].copy()
    if not defenders_nearby.empty:
        defenders_nearby['distance'] = np.sqrt(
            (defenders_nearby['x'] - receiver_pos['x'])**2 + 
            (defenders_nearby['y'] - receiver_pos['y'])**2
        )
        print(f"\nDefenders nearby at throw (within 15 yards):")
        print(defenders_nearby[['player_name', 'player_position', 'distance']].sort_values('distance').head(3))
    
else:
    print("â�Œ No valid plays found!")


def detect_commitment_frame(df_defender_track, receiver_track):
    """
    df_defender_track: defender rows AFTER release, sorted by frame_id
    receiver_track: receiver rows AFTER release, sorted by frame_id

    Returns: commitment frame_id or None
    """

    frames = df_defender_track["frame_id"].values

    cos_vals = []
    for i in range(len(df_defender_track)):
        defender = df_defender_track.iloc[i]
        receiver = receiver_track.iloc[i]

        receiver_pos = np.array([receiver["x"], receiver["y"]])
        cos_theta = compute_cos_theta(defender, receiver_pos)
        cos_vals.append(cos_theta)

    cos_vals = np.array(cos_vals)

    # Sustained threshold logic
    for i in range(len(cos_vals) - SUSTAIN_FRAMES + 1):
        window = cos_vals[i:i + SUSTAIN_FRAMES]
        if np.all(window >= COS_THETA_THRESHOLD):
            return frames[i]

    return None


def compute_ASA_for_play(df_input_play, df_output_play):
    """
    Returns dict with ASA metrics for the play
    """

    # --- release frame ---
    t_rel = get_release_frame(df_input_play)

    # --- targeted receiver ---
    receiver_id = get_targeted_receiver(df_input_play)

    # Receiver track after release
    receiver_track = df_output_play[df_output_play["nfl_id"] == receiver_id].sort_values("frame_id")

    if receiver_track.empty:
        return None

    ASA_values = []

    # Iterate over frames to find defender tracks
    for defender_id in df_output_play["nfl_id"].unique():
        if defender_id == receiver_id:
            continue

        df_def = df_output_play[df_output_play["nfl_id"] == defender_id].sort_values("frame_id")

        if df_def.empty:
            continue

        # Commitment detection
        t_commit = detect_commitment_frame(df_def, receiver_track)

        if t_commit is not None:
            ASA = (t_commit - 1) * FRAME_DT   # output frame_id starts at 1
            ASA_values.append(ASA)

    if not ASA_values:
        return {
            "ASA_min": np.nan,
            "ASA_mean": np.nan,
            "ASA_count": 0
        }

    return {
        "ASA_min": np.min(ASA_values),
        "ASA_mean": np.mean(ASA_values),
        "ASA_count": len(ASA_values)
    }


def compute_ACP_at_catch(df_output_play):
    """
    Computes ACP using final frame geometry
    """
    final_frame = df_output_play["frame_id"].max()
    df_final = df_output_play[df_output_play["frame_id"] == final_frame]

    receiver = df_final[df_final["player_role"] == "Targeted Receiver"]
    if receiver.empty:
        return np.nan

    receiver_pos = receiver[["x", "y"]].values[0]

    defenders = df_final[df_final["player_side"] == "Defense"]

    if defenders.empty:
        return 0.0

    dists = np.sqrt(
        (defenders["x"] - receiver_pos[0])**2 +
        (defenders["y"] - receiver_pos[1])**2
    )

    # Simple inverse-distance containment
    ACP = np.sum(1 / (dists + 1e-6))
    return ACP


def build_ASA_ACP_dataset(df_input, df_output):
    results = []

    plays = df_input[["game_id", "play_id"]].drop_duplicates()

    for _, row in tqdm(plays.iterrows(), total=len(plays)):
        g = row["game_id"]
        p = row["play_id"]

        df_in_play = df_input[(df_input.game_id == g) & (df_input.play_id == p)]
        df_out_play = df_output[(df_output.game_id == g) & (df_output.play_id == p)]

        if df_out_play.empty:
            continue

        try:
            asa_metrics = compute_ASA_for_play(df_in_play, df_out_play)
            acp = compute_ACP_at_catch(df_out_play)

            if asa_metrics is None:
                continue

            results.append({
                "game_id": g,
                "play_id": p,
                "ASA_min": asa_metrics["ASA_min"],
                "ASA_mean": asa_metrics["ASA_mean"],
                "ASA_count": asa_metrics["ASA_count"],
                "ACP": acp
            })

        except Exception:
            continue

    return pd.DataFrame(results)


def compute_ASA_final(df_input_play, df_output_play, fps=10.0):
    """
    FINAL, PHYSICALLY CORRECT ASA computation.
    Velocity = Î”position/Î”t (mathematical definition, not estimation)
    
    Parameters:
    -----------
    df_input_play : DataFrame
        Input tracking data for ONE play (pre-throw)
        Must contain: game_id, play_id, nfl_id, frame_id, x, y, 
                     player_role, player_name, player_position, player_side
    
    df_output_play : DataFrame
        Output tracking data for ONE play (post-throw)
        Must contain: game_id, play_id, nfl_id, frame_id, x, y
    
    fps : float, default=10.0
        Frames per second of tracking data
    
    Returns:
    --------
    dict or None
        ASA results dictionary if successful, None otherwise
    """
    import numpy as np
    
    # ======================
    # CONSTANTS
    # ======================
    COS_THETA_THRESHOLD = 0.7    # Directional commitment threshold
    SUSTAIN_FRAMES = 2           # Frames needed for sustained commitment
    FRAME_DT = 1.0 / fps         # Seconds per frame
    
   
    if df_input_play.empty or df_output_play.empty:
        return None
    
    
    required_input = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y', 
                     'player_role', 'player_name']
    for col in required_input:
        if col not in df_input_play.columns:
            return None
    
    
    # What is our THROW FRAME?
    
    t_rel = df_input_play['frame_id'].max()  # Last input frame = throw
    

    
    target_rows = df_input_play[df_input_play.player_role == 'Targeted Receiver']
    if target_rows.empty:
        return None
    
    receiver_id = target_rows.iloc[0]['nfl_id']
    receiver_name = target_rows.iloc[0]['player_name']
    
    # ======================
    # GET RECEIVER POST-THROW TRACK
    # ======================
    receiver_track = df_output_play[df_output_play.nfl_id == receiver_id].sort_values('frame_id')
    if receiver_track.empty:
        return None
    
    # ======================
    # PROCESS EACH DEFENDER
    # ======================
    defender_results = []
    
    for defender_id in df_output_play['nfl_id'].unique():
        # Skip the receiver
        if defender_id == receiver_id:
            continue
        
        # Get defender's post-throw track
        defender_track = df_output_play[df_output_play.nfl_id == defender_id].sort_values('frame_id')
        
        # Need at least SUSTAIN_FRAMES + 1 frames (need previous for velocity)
        if len(defender_track) < SUSTAIN_FRAMES + 1:
            continue
        
        # Get defender info from input data
        def_info = df_input_play[df_input_play.nfl_id == defender_id]
        if def_info.empty:
            continue             # Defender not in input data
        
        defender_name = def_info.iloc[0]['player_name']
        defender_position = def_info.iloc[0].get('player_position', 'Unknown')
        
        # ======================
        # COMPUTE cosÎ¸ FOR EACH FRAME
        # ======================
        cos_vals = []   # cosÎ¸ values
        frames = []     # Corresponding frame IDs
        
        # Start from frame 2 (we need frame 1 to actually compute velocity)
        for i in range(1, len(defender_track)):
            current = defender_track.iloc[i]
            previous = defender_track.iloc[i-1]
            frame_id = current['frame_id']
            
            # Get receiver position at this frame
            receiver_frame = receiver_track[receiver_track.frame_id == frame_id]
            if receiver_frame.empty:
                continue
            
            receiver_row = receiver_frame.iloc[0]
            
            # ======================
            # PHYSICS: VELOCITY FROM POSITION
            # ======================
            # ğ�‘£âƒ—(ğ�‘¡) = [ğ�‘�ğ�‘œğ�‘ (ğ�‘¡) - ğ�‘�ğ�‘œğ�‘ (ğ�‘¡-1)] / Î”ğ�‘¡
            dx = current['x'] - previous['x']
            dy = current['y'] - previous['y']
            velocity = np.array([dx, dy]) / FRAME_DT  # yds/sec
            
            # ======================
            # POSITIONS
            # ======================
            defender_pos = np.array([current['x'], current['y']])
            receiver_pos = np.array([receiver_row['x'], receiver_row['y']])
            
            # ======================
            # VECTOR TO RECEIVER
            # ======================
            defender_to_receiver = receiver_pos - defender_pos
            
            # ======================
            # COMPUTE cosÎ¸
            # ======================
            vel_norm = np.linalg.norm(velocity)
            to_norm = np.linalg.norm(defender_to_receiver)
            
            # Handle edge cases (stationary or very close)
            if vel_norm < 0.1 or to_norm < 0.1:
                cos_theta = 0.0
            else:
                # Our beloved dot prod lol  --  cosÎ¸ = (ğ�‘£âƒ— Â· ğ�‘¢âƒ—) / (â€–ğ�‘£âƒ—â€– â€–ğ�‘¢âƒ—â€–)
                cos_theta = np.dot(velocity, defender_to_receiver) / (vel_norm * to_norm)
                # Clip to handle numerical errors
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
            cos_vals.append(cos_theta)
            frames.append(frame_id)
        
        # We need enough valid frames to analyze
        if len(cos_vals) < SUSTAIN_FRAMES:
            continue
        
        # ======================
        # FIND COMMITMENT FRAME
        # ======================
        t_commit = None
        
        # Look for SUSTAIN_FRAMES consecutive frames where cosÎ¸ â‰¥ threshold
        for i in range(len(cos_vals) - SUSTAIN_FRAMES + 1):
            window = cos_vals[i:i + SUSTAIN_FRAMES]
            if all(c >= COS_THETA_THRESHOLD for c in window):
                t_commit = frames[i]
                break
        
        # If no commitment found, use last frame (conservative)
        if t_commit is None:
            t_commit = frames[-1]
            commitment_status = "no_commitment"
        else:
            commitment_status = "committed"
        
        # ======================
        # CALCULATE ASA
        # ======================
        
        # Remember, Frame 1 = first post-throw frame, so ASA = (t_commit - 1) Ã— Î”t
        ASA = (t_commit - 1) * FRAME_DT
        
        # ======================
        # STORE RESULTS
        # ======================
        defender_results.append({
            'nfl_id': defender_id,
            'player_name': defender_name,
            'player_position': defender_position,
            't_commit': t_commit,
            'ASA': ASA,
            'commitment_status': commitment_status,
            'avg_cos_theta': float(np.mean(cos_vals)),
            'min_cos_theta': float(np.min(cos_vals)),
            'max_cos_theta': float(np.max(cos_vals)),
            'frames_analyzed': len(cos_vals)
        })
    
    # ======================
    # FINAL RESULT
    # ======================
    if not defender_results:
        return None
    
    # Sort defenders by ASA (earliest commitment first)
    defender_results.sort(key=lambda x: x['ASA'])
    
    return {
        'game_id': int(df_input_play['game_id'].iloc[0]),
        'play_id': int(df_input_play['play_id'].iloc[0]),
        'receiver_id': receiver_id,
        'receiver_name': receiver_name,
        't_rel': t_rel,
        'defender_results': defender_results
    }



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



def compute_velocity_based_acp(df_out_play, df_input_play, fps=10.0, r_max=15.0, sigma=5.0):
    """
    Simplified physics-based ACP with only velocity-based computation.
    Returns ACP_catch and ACP_mean.
    """
    FRAME_DT = 1.0 / fps
    
    # 1. Identify targeted receiver
    target_rows = df_input_play[df_input_play["player_role"] == "Targeted Receiver"]
    if target_rows.empty:
        return None
    
    receiver_id = target_rows.iloc[0]["nfl_id"]
    
    # 2. Sort frames
    df_out_sorted = df_out_play.sort_values("frame_id").reset_index(drop=True)
    frames = df_out_sorted["frame_id"].unique()
    
    if len(frames) < 2:
        return None
    
    acp_values = []
    
    # 3. Process each frame
    for i in range(1, len(frames)):
        frame_now = frames[i]
        frame_prev = frames[i-1]
        
        df_now = df_out_sorted[df_out_sorted.frame_id == frame_now]
        df_prev = df_out_sorted[df_out_sorted.frame_id == frame_prev]
        
        rec_row = df_now[df_now.nfl_id == receiver_id]
        if rec_row.empty:
            continue
            
        receiver_pos = rec_row[["x", "y"]].values[0]
        acp_frame = 0.0
        
        # Process each non-receiver player
        for _, player_row in df_now.iterrows():
            if player_row["nfl_id"] == receiver_id:
                continue
            
            defender_pos = np.array([player_row["x"], player_row["y"]])
            dist = np.linalg.norm(receiver_pos - defender_pos)
            
            if dist > r_max:
                continue
            
            prev_row = df_prev[df_prev.nfl_id == player_row["nfl_id"]]
            if prev_row.empty:
                continue
                
            prev_pos = prev_row[["x", "y"]].values[0]
            
            # Velocity-based computation
            velocity = (defender_pos - prev_pos) / FRAME_DT
            speed = np.linalg.norm(velocity)
            
            if speed == 0:
                continue
            
            # Unit vectors
            v_hat = velocity / speed
            u_hat = (receiver_pos - defender_pos) / dist
            
            # Closing effectiveness (only positive contributes)
            gamma = np.dot(v_hat, u_hat)
            
            if gamma > 0:
                # Distance weighting
                weight = np.exp(-dist / sigma)
                acp_frame += weight * gamma
        
        acp_values.append(acp_frame)
    
    if not acp_values:
        return None
    
    return {
        "ACP_catch": float(acp_values[-1]),
        "ACP_mean": float(np.mean(acp_values))
    }


def build_asa_acp_dataset(df_input, df_output):
    """
    Build dataset with both ASA and ACP metrics.
    """
    results = []
    
    plays = df_input[["game_id", "play_id"]].drop_duplicates()
    
    for _, row in tqdm(plays.iterrows(), total=len(plays)):
        g = row["game_id"]
        p = row["play_id"]
        
        df_in_play = df_input[(df_input.game_id == g) & (df_input.play_id == p)]
        df_out_play = df_output[(df_output.game_id == g) & (df_output.play_id == p)]
        
        if df_out_play.empty:
            continue
        
        try:
            # Compute ASA (assume this function exists)
            asa_metrics = compute_ASA_for_play(df_in_play, df_out_play)
            acp_metrics = compute_velocity_based_acp(df_out_play, df_in_play)
            
            if asa_metrics is None or acp_metrics is None:
                continue
            
            results.append({
                "game_id": g,
                "play_id": p,
                "ASA_min": asa_metrics["ASA_min"],
                "ASA_mean": asa_metrics["ASA_mean"],
                "ASA_count": asa_metrics["ASA_count"],
                "ACP_catch": acp_metrics["ACP_catch"],
                "ACP_mean": acp_metrics["ACP_mean"]
            })
            
        except Exception:
            continue
    
    return pd.DataFrame(results)


###############################################################################
# RESULTS ANALYSIS AND VISUALIZATION
###############################################################################

def analyze_and_visualize_acp(results_df):
    """
    Analyze ACP results and create visualizations.
    """
    print("=" * 60)
    print("ACP RESULTS ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print(f"\nDataset Statistics:")
    print(f"Total plays analyzed: {len(results_df)}")
    
    print(f"\nACP_catch Statistics:")
    print(f"Mean: {results_df['ACP_catch'].mean():.4f}")
    print(f"Std:  {results_df['ACP_catch'].std():.4f}")
    print(f"Min:  {results_df['ACP_catch'].min():.4f}")
    print(f"25%:  {results_df['ACP_catch'].quantile(0.25):.4f}")
    print(f"50%:  {results_df['ACP_catch'].quantile(0.50):.4f}")
    print(f"75%:  {results_df['ACP_catch'].quantile(0.75):.4f}")
    print(f"Max:  {results_df['ACP_catch'].max():.4f}")
    
    print(f"\nACP_mean Statistics:")
    print(f"Mean: {results_df['ACP_mean'].mean():.4f}")
    print(f"Std:  {results_df['ACP_mean'].std():.4f}")
    print(f"Min:  {results_df['ACP_mean'].min():.4f}")
    print(f"Max:  {results_df['ACP_mean'].max():.4f}")
    
    print(f"\nASA Statistics:")
    print(f"ASA_min mean: {results_df['ASA_min'].mean():.4f}")
    print(f"ASA_mean mean: {results_df['ASA_mean'].mean():.4f}")
    
    # Correlations
    print(f"\nCorrelations:")
    print(f"ASA_mean vs ACP_catch: {results_df['ASA_mean'].corr(results_df['ACP_catch']):.4f}")
    print(f"ASA_min vs ACP_catch:  {results_df['ASA_min'].corr(results_df['ACP_catch']):.4f}")
    print(f"ASA_mean vs ACP_mean:  {results_df['ASA_mean'].corr(results_df['ACP_mean']):.4f}")
    
    # Distribution analysis
    print(f"\nACP Distribution Analysis:")
    low_acp = results_df[results_df['ACP_catch'] < 0.5]
    med_acp = results_df[(results_df['ACP_catch'] >= 0.5) & (results_df['ACP_catch'] < 1.5)]
    high_acp = results_df[results_df['ACP_catch'] >= 1.5]
    
    print(f"Low ACP (< 0.5): {len(low_acp)} plays ({len(low_acp)/len(results_df)*100:.1f}%)")
    print(f"Medium ACP (0.5-1.5): {len(med_acp)} plays ({len(med_acp)/len(results_df)*100:.1f}%)")
    print(f"High ACP (â‰¥ 1.5): {len(high_acp)} plays ({len(high_acp)/len(results_df)*100:.1f}%)")
    
    # Create visualizations
    create_acp_visualizations(results_df)
    
    return results_df


def create_acp_visualizations(results_df):
    """
    Create and save visualizations for ACP analysis.
    """
    plt.style.use('default')
    
    # 1. Distribution of ACP_catch
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ACP Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: ACP_catch histogram
    axes[0, 0].hist(results_df['ACP_catch'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('ACP_catch')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of ACP at Catch')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: ACP_mean histogram
    axes[0, 1].hist(results_df['ACP_mean'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0, 1].set_xlabel('ACP_mean')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Mean ACP')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter - ASA_mean vs ACP_catch
    scatter1 = axes[0, 2].scatter(results_df['ASA_mean'], results_df['ACP_catch'], 
                                 alpha=0.6, c=results_df['ASA_min'], cmap='viridis', s=30)
    axes[0, 2].set_xlabel('ASA_mean (degrees)')
    axes[0, 2].set_ylabel('ACP_catch')
    axes[0, 2].set_title('ASA_mean vs ACP_catch')
    axes[0, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 2], label='ASA_min')
    
    # Plot 4: Scatter - ASA_min vs ACP_mean
    scatter2 = axes[1, 0].scatter(results_df['ASA_min'], results_df['ACP_mean'], 
                                 alpha=0.6, c=results_df['ASA_mean'], cmap='plasma', s=30)
    axes[1, 0].set_xlabel('ASA_min (degrees)')
    axes[1, 0].set_ylabel('ACP_mean')
    axes[1, 0].set_title('ASA_min vs ACP_mean')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1, 0], label='ASA_mean')
    
    # Plot 5: Box plot of ACP_catch
    axes[1, 1].boxplot(results_df['ACP_catch'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
    axes[1, 1].set_ylabel('ACP_catch')
    axes[1, 1].set_title('Box Plot of ACP at Catch')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks([1], [''])
    
    # Plot 6: ACP_catch vs ACP_mean
    axes[1, 2].scatter(results_df['ACP_catch'], results_df['ACP_mean'], 
                      alpha=0.6, s=30, color='purple')
    axes[1, 2].set_xlabel('ACP_catch')
    axes[1, 2].set_ylabel('ACP_mean')
    axes[1, 2].set_title('Catch ACP vs Mean ACP')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(results_df['ACP_catch'], results_df['ACP_mean'], 1)
    p = np.poly1d(z)
    axes[1, 2].plot(results_df['ACP_catch'].sort_values(), 
                   p(results_df['ACP_catch'].sort_values()), 
                   "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/acp_analysis_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Create correlation heatmap
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Select numeric columns for correlation
    numeric_cols = ['ASA_min', 'ASA_mean', 'ASA_count', 'ACP_catch', 'ACP_mean']
    corr_matrix = results_df[numeric_cols].corr()
    
    im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax2.set_xticks(range(len(corr_matrix.columns)))
    ax2.set_yticks(range(len(corr_matrix.columns)))
    ax2.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax2.set_yticklabels(corr_matrix.columns)
    ax2.set_title('Correlation Matrix: ASA vs ACP Metrics')
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig('/kaggle/working/acp_correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualizations saved to Kaggle working directory:")
    print(f"1. /kaggle/working/acp_analysis_grid.png")
    print(f"2. /kaggle/working/acp_correlation_matrix.png")


def save_detailed_results(results_df):
    """
    Save detailed results to CSV files.
    """
    # Save full results
    results_df.to_csv('/kaggle/working/asa_acp_full_results.csv', index=False)
    
    # Save summary statistics
    summary_stats = results_df.describe().round(4)
    summary_stats.to_csv('/kaggle/working/asa_acp_summary_statistics.csv')
    
    # Save top/bottom plays
    top_acp = results_df.nlargest(20, 'ACP_catch')[['game_id', 'play_id', 'ACP_catch', 'ACP_mean', 'ASA_mean']]
    low_acp = results_df.nsmallest(20, 'ACP_catch')[['game_id', 'play_id', 'ACP_catch', 'ACP_mean', 'ASA_mean']]
    
    top_acp.to_csv('/kaggle/working/top_20_highest_acp_plays.csv', index=False)
    low_acp.to_csv('/kaggle/working/top_20_lowest_acp_plays.csv', index=False)
    
    print(f"\nResults saved to Kaggle working directory:")
    print(f"1. /kaggle/working/asa_acp_full_results.csv")
    print(f"2. /kaggle/working/asa_acp_summary_statistics.csv")
    print(f"3. /kaggle/working/top_20_highest_acp_plays.csv")
    print(f"4. /kaggle/working/top_20_lowest_acp_plays.csv")




# RUN FINAL VERSION
print("="*80)
print("OUR FINAL PHYSICS-BASED ASA COMPUTATION")
print("Velocity = Î”position/Î”t (MATHEMATICAL DEFINITION)")
print("="*80)

asa_final = compute_ASA_final(df_in_play, df_out_play)

if asa_final:
    print(f"\nâœ… FINAL ASA COMPUTATION SUCCESSFUL!")
    print(f"Play: {asa_final['game_id']}/{asa_final['play_id']}")
    print(f"Receiver: {asa_final['receiver_name']}")
    print(f"Throw frame: {asa_final['t_rel']}")
    print(f"Found {len(asa_final['defender_results'])} defenders")
    
    print(f"\nDEFENDER ASA RESULTS:")
    print("-" * 70)
    
    for defender in asa_final['defender_results']:
        print(f"{defender['player_name']} ({defender['player_position']}):")
        print(f"  ASA: {defender['ASA']:.3f}s after throw")
        print(f"  Commitment frame: {defender['t_commit']}")
        print(f"  Status: {defender['commitment_status']}")
        print(f"  cosÎ¸: avg={defender['avg_cos_theta']:.3f}, range=[{defender['min_cos_theta']:.3f}, {defender['max_cos_theta']:.3f}]")
        print(f"  Frames: {defender['frames_analyzed']}")
        print()
    
    #  VALIDATION: This is our internal system check, checking if our results make sense (this is just to flag an 'unsual' outcomes, not alter with it)
    print("\n" + "="*80)
    print("VALIDATION CHECK")
    print("="*80)
    
    
    print("1. ASA values check:")
    for defender in asa_final['defender_results']:
        asa_val = defender['ASA']
        if asa_val < 0:
            print(f"  â�Œ {defender['player_name']}: ASA={asa_val:.3f}s (NEGATIVE!)")
        elif asa_val > 3.0:
            print(f"  âš ï¸�  {defender['player_name']}: ASA={asa_val:.3f}s (>3s, possibly too long)")
        else:
            print(f"  âœ… {defender['player_name']}: ASA={asa_val:.3f}s (reasonable)")
    
    # 2. Check cosÎ¸ ranges
    print("\n2. cosÎ¸ ranges check:")
    for defender in asa_final['defender_results']:
        min_cos = defender['min_cos_theta']
        max_cos = defender['max_cos_theta']
        
        if min_cos < -0.5 or max_cos > 1.1:  # Allow slight >1 for numerical error
            print(f"  âš ï¸�  {defender['player_name']}: cosÎ¸ range [{min_cos:.3f}, {max_cos:.3f}] (unusual)")
        else:
            print(f"  âœ… {defender['player_name']}: cosÎ¸ range [{min_cos:.3f}, {max_cos:.3f}] (valid)")
    
    # 3. Visualize for sanity check
    try:
        import matplotlib.pyplot as plt
        
        # Constants
        COS_THETA_THRESHOLD = 0.7
        
        # Plot 1: cosÎ¸ over time for each defender
        print("\nGenerating visualization...")
        
        # Create figure with subplots for each defender
        n_defenders = len(asa_final['defender_results'])
        fig, axes = plt.subplots(n_defenders, 1, figsize=(12, 4*n_defenders))
        
        # Handle single defender case (axes not as list)
        if n_defenders == 1:
            axes = [axes]
        
        for idx, defender in enumerate(asa_final['defender_results']):
            ax = axes[idx]
            
            # Get defender data
            defender_id = defender['nfl_id']
            defender_track = df_out_play[df_out_play.nfl_id == defender_id].sort_values('frame_id')
            receiver_track = df_out_play[df_out_play.nfl_id == asa_final['receiver_id']].sort_values('frame_id')
            
            # Recompute cosÎ¸ for plotting
            cos_vals_plot = []
            frames_plot = []
            
            for i in range(1, len(defender_track)):
                current = defender_track.iloc[i]
                previous = defender_track.iloc[i-1]
                frame_id = current['frame_id']
                
                # Get receiver at this frame
                receiver_frame = receiver_track[receiver_track.frame_id == frame_id]
                if receiver_frame.empty:
                    continue
                
                receiver_row = receiver_frame.iloc[0]
                
                # Velocity
                dx = current['x'] - previous['x']
                dy = current['y'] - previous['y']
                velocity = np.array([dx, dy]) / 0.1  # 10 Hz = 0.1s per frame
                
                # Positions
                defender_pos = np.array([current['x'], current['y']])
                receiver_pos = np.array([receiver_row['x'], receiver_row['y']])
                
                # Compute cosÎ¸
                to_receiver = receiver_pos - defender_pos
                vel_norm = np.linalg.norm(velocity)
                to_norm = np.linalg.norm(to_receiver)
                
                if vel_norm > 0.1 and to_norm > 0.1:
                    cos_theta = np.dot(velocity, to_receiver) / (vel_norm * to_norm)
                    cos_vals_plot.append(cos_theta)
                    frames_plot.append(frame_id)
            
            # Plot cosÎ¸ over time
            if cos_vals_plot:
                ax.plot(frames_plot, cos_vals_plot, 'b.-', linewidth=2, markersize=6)
                ax.axhline(y=COS_THETA_THRESHOLD, color='r', linestyle='--', alpha=0.7, 
                          label=f'Threshold ({COS_THETA_THRESHOLD})')
                
                # Mark commitment point
                if defender['commitment_status'] == 'committed':
                    ax.axvline(x=defender['t_commit'], color='g', linestyle='--', alpha=0.7,
                              label=f'Commitment (ASA={defender["ASA"]:.2f}s)')
                
                ax.set_ylabel('cosÎ¸')
                ax.set_ylim([-1.1, 1.1])
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{defender["player_name"]} ({defender["player_position"]}) - ASA: {defender["ASA"]:.2f}s')
                ax.legend(loc='upper right')
            
            # Set xlabel for bottom subplot
            if idx == n_defenders - 1:
                ax.set_xlabel('Frame ID (post-throw)')
        
        plt.suptitle(f'ASA Analysis - Play {asa_final["game_id"]}/{asa_final["play_id"]}\nReceiver: {asa_final["receiver_name"]}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Plot 2: Field positions with velocity vectors
        print("\nGenerating field visualization...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_aspect('equal')
        
        # Plot receiver trajectory
        receiver_track = df_out_play[df_out_play.nfl_id == asa_final['receiver_id']].sort_values('frame_id')
        ax.plot(receiver_track['x'], receiver_track['y'], 'b-', linewidth=3, alpha=0.7, label='Receiver path')
        ax.plot(receiver_track.iloc[0]['x'], receiver_track.iloc[0]['y'], 'bo', markersize=12, label='Receiver start')
        ax.plot(receiver_track.iloc[-1]['x'], receiver_track.iloc[-1]['y'], 'b*', markersize=20, label='Receiver end')
        
        # Plot defender trajectories with velocity vectors
        colors = ['red', 'green', 'purple', 'orange', 'brown']
        
        for idx, defender in enumerate(asa_final['defender_results']):
            defender_id = defender['nfl_id']
            defender_track = df_out_play[df_out_play.nfl_id == defender_id].sort_values('frame_id')
            
            # Choose color
            color = colors[idx % len(colors)]
            
            # Plot defender path
            ax.plot(defender_track['x'], defender_track['y'], color=color, linestyle='-', 
                    linewidth=2, alpha=0.7, label=f"{defender['player_name'][:10]}")
            
            # Plot velocity vectors (every 2nd frame for clarity)
            STEP = 2
            for i in range(1, len(defender_track), STEP):
                if i < len(defender_track):
                    dx = defender_track.iloc[i]['x'] - defender_track.iloc[i-1]['x']
                    dy = defender_track.iloc[i]['y'] - defender_track.iloc[i-1]['y']
                    
                    # Scale for visibility
                    scale = 5.0
                    ax.arrow(
                        defender_track.iloc[i-1]['x'],
                        defender_track.iloc[i-1]['y'],
                        dx * scale,
                        dy * scale,
                        head_width=0.3,
                        head_length=0.4,
                        fc=color,
                        ec=color,
                        alpha=0.5,
                        length_includes_head=True
                    )
            
            # Mark commitment point on field
            if defender['commitment_status'] == 'committed':
                commit_frame = defender['t_commit']
                commit_pos = defender_track[defender_track.frame_id == commit_frame]
                if not commit_pos.empty:
                    ax.scatter(
                        commit_pos.iloc[0]['x'],
                        commit_pos.iloc[0]['y'],
                        s=200,
                        marker='s',
                        edgecolor=color,
                        facecolor='none',
                        linewidth=3,
                        label=f"{defender['player_name'][:5]} commitment"
                    )
        
        ax.set_xlabel('X (yards)')
        ax.set_ylabel('Y (yards)')
        ax.set_title(f'Field Positions with Velocity Vectors\nPlay {asa_final["game_id"]}/{asa_final["play_id"]}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.show()
        
        # Plot 3: Combined ASA timeline
        print("\nGenerating ASA timeline...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart of ASA values
        defender_names = [f"{d['player_name'][:10]}\n({d['player_position']})" 
                         for d in asa_final['defender_results']]
        asa_values = [d['ASA'] for d in asa_final['defender_results']]
        colors = ['green' if d['commitment_status'] == 'committed' else 'orange' 
                 for d in asa_final['defender_results']]
        
        bars = ax1.bar(defender_names, asa_values, color=colors, edgecolor='black')
        ax1.axhline(y=0, color='red', linestyle='-', alpha=0.5, linewidth=2)
        ax1.set_ylabel('ASA (seconds after throw)')
        ax1.set_title('Angle of Sustained Ambiguity (ASA)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, asa_val in zip(bars, asa_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{asa_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # ASA timeline plot
        max_time = max(asa_values) + 0.5 if asa_values else 2.0
        
        for idx, defender in enumerate(asa_final['defender_results']):
            color = colors[idx]
            name = defender['player_name'][:10]
            
            # Draw timeline bar
            ax2.axhline(y=name, xmin=0, xmax=defender['ASA']/max_time,
                       color=color, linewidth=10, alpha=0.5)
            
            # Mark commitment point
            ax2.plot(defender['ASA'], name, 'o', color=color, 
                    markersize=10, label=f"{name}: {defender['ASA']:.2f}s")
        
        ax2.set_xlabel('Time after throw (seconds)')
        ax2.set_title('ASA Timeline: When Defenders Committed')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.legend(loc='lower right', fontsize=8)
        
        plt.suptitle(f'ASA Summary - Play {asa_final["game_id"]}/{asa_final["play_id"]}\nReceiver: {asa_final["receiver_name"]}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\nâš ï¸�  Visualization error: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print("â�Œ Final ASA computation failed")


# SIMPLE TABLE REPRESENTATION
table = """| Defender       | Position | ASA (seconds) | Commitment Frame | avg cosÎ¸ | cosÎ¸ Range      |
| -------------- | -------- | ------------- | ---------------- | -------- | --------------- |
| Justin Reid    | SS       |   1.10s       | Frame 12         | 0.598    | [0.062, 0.892]  |
| L'Jarius Sneed | CB       |   1.30s       | Frame 14         | 0.357    | [-0.509, 0.954] |"""

print(table)

# Save to file
with open("asa_table.md", "w") as f:
    f.write(table)
print("âœ… Table saved to 'asa_table.md'")


def save_plots_locally(asa_final, df_out_play, save_dir='asa_plots'):
    """
    Save all ASA plots locally with high quality.
    """
    import os
    import matplotlib.pyplot as plt
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Constants
    COS_THETA_THRESHOLD = 0.7
    
    # Plot 1: Individual defender alignment plots
    print(f"\nSaving individual defender plots to '{save_dir}/'...")
    
    fig, axes = plt.subplots(len(asa_final['defender_results']), 1, 
                            figsize=(10, 4*len(asa_final['defender_results'])))
    if len(asa_final['defender_results']) == 1:
        axes = [axes]
    
    for idx, defender in enumerate(asa_final['defender_results']):
        ax = axes[idx]
        
        # Get defender data
        defender_id = defender['nfl_id']
        defender_track = df_out_play[df_out_play.nfl_id == defender_id].sort_values('frame_id')
        receiver_track = df_out_play[df_out_play.nfl_id == asa_final['receiver_id']].sort_values('frame_id')
        
        # Recompute cosÎ¸ for plotting
        cos_vals_plot = []
        frames_plot = []
        
        for i in range(1, len(defender_track)):
            current = defender_track.iloc[i]
            previous = defender_track.iloc[i-1]
            frame_id = current['frame_id']
            
            receiver_frame = receiver_track[receiver_track.frame_id == frame_id]
            if receiver_frame.empty:
                continue
            
            receiver_row = receiver_frame.iloc[0]
            
            # Velocity
            dx = current['x'] - previous['x']
            dy = current['y'] - previous['y']
            velocity = np.array([dx, dy]) / 0.1
            
            # Positions
            defender_pos = np.array([current['x'], current['y']])
            receiver_pos = np.array([receiver_row['x'], receiver_row['y']])
            
            # cosÎ¸
            to_receiver = receiver_pos - defender_pos
            vel_norm = np.linalg.norm(velocity)
            to_norm = np.linalg.norm(to_receiver)
            
            if vel_norm > 0.1 and to_norm > 0.1:
                cos_theta = np.dot(velocity, to_receiver) / (vel_norm * to_norm)
                cos_vals_plot.append(cos_theta)
                frames_plot.append(frame_id)
        
        # Plot
        if cos_vals_plot:
            ax.plot(frames_plot, cos_vals_plot, 'b.-', linewidth=2, markersize=6)
            ax.axhline(y=COS_THETA_THRESHOLD, color='r', linestyle='--', alpha=0.7, 
                      label=f'Threshold ({COS_THETA_THRESHOLD})')
            
            # Mark commitment point
            if defender['commitment_status'] == 'committed':
                ax.axvline(x=defender['t_commit'], color='g', linestyle='--', alpha=0.7,
                          label=f'Commitment (ASA={defender["ASA"]:.2f}s)')
            
            ax.set_ylabel('cosÎ¸')
            ax.set_ylim([-1.1, 1.1])
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{defender["player_name"]} ({defender["player_position"]}) - ASA: {defender["ASA"]:.2f}s')
            ax.legend(loc='upper right')
        
        if idx == len(asa_final['defender_results']) - 1:
            ax.set_xlabel('Frame ID (post-throw)')
    
    plt.suptitle(f'ASA Analysis - Play {asa_final["game_id"]}/{asa_final["play_id"]}\nReceiver: {asa_final["receiver_name"]}', fontsize=14)
    plt.tight_layout()
    
    # Save individual defender plot
    individual_filename = f"{save_dir}/play_{asa_final['game_id']}_{asa_final['play_id']}_defenders.png"
    plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
    print(f"âœ… Saved: {individual_filename}")
    plt.close()
    
    # Plot 2: Comprehensive analysis plot (4 subplots)
    print(f"Saving comprehensive analysis plot...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: ASA Comparison
    ax1 = plt.subplot(2, 2, 1)
    defenders = [d['player_name'] for d in asa_final['defender_results']]
    asa_values = [d['ASA'] for d in asa_final['defender_results']]
    colors = ['green' if d['commitment_status'] == 'committed' else 'orange' 
             for d in asa_final['defender_results']]
    
    bars = ax1.bar(defenders, asa_values, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='red', linestyle='-', alpha=0.5, linewidth=2)
    ax1.set_ylabel('ASA (seconds after throw)')
    ax1.set_title('Angle of Sustained Ambiguity (ASA)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, asa_val in zip(bars, asa_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{asa_val:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Field View
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Plot receiver trajectory
    receiver_track = df_out_play[df_out_play.nfl_id == asa_final['receiver_id']].sort_values('frame_id')
    ax2.plot(receiver_track['x'], receiver_track['y'], 'b-', linewidth=3, alpha=0.7, label='Receiver path')
    ax2.plot(receiver_track['x'].iloc[0], receiver_track['y'].iloc[0], 'bo', markersize=12, label='Start')
    ax2.plot(receiver_track['x'].iloc[-1], receiver_track['y'].iloc[-1], 'b*', markersize=20, label='End')
    
    # Plot defender trajectories with commitment points
    for defender in asa_final['defender_results']:
        defender_id = defender['nfl_id']
        defender_track = df_out_play[df_out_play.nfl_id == defender_id].sort_values('frame_id')
        
        # Choose color based on position
        color = 'red' if 'CB' in defender['player_position'] else 'green'
        
        ax2.plot(defender_track['x'], defender_track['y'], color=color, linestyle='-', 
                linewidth=2, alpha=0.7, label=f"{defender['player_name'][:10]}")
        
        # Mark commitment point
        if defender['commitment_status'] == 'committed':
            commit_frame = defender['t_commit']
            commit_pos = defender_track[defender_track.frame_id == commit_frame]
            if not commit_pos.empty:
                ax2.plot(commit_pos.iloc[0]['x'], commit_pos.iloc[0]['y'], 
                        color=color, marker='s', markersize=10, 
                        markeredgewidth=2, markerfacecolor='none',
                        label=f"{defender['player_name'][:5]} commit")
    
    ax2.set_xlabel('X (yards)')
    ax2.set_ylabel('Y (yards)')
    ax2.set_title('Player Trajectories Post-Throw')
    ax2.legend(loc='upper left', fontsize=8)
    
    # Plot 3: Alignment over time
    ax3 = plt.subplot(2, 2, 3)
    
    for defender in asa_final['defender_results']:
        defender_id = defender['nfl_id']
        defender_track = df_out_play[df_out_play.nfl_id == defender_id].sort_values('frame_id')
        
        # Recompute cosÎ¸ for plotting
        cos_vals = []
        frames = []
        
        for i in range(1, len(defender_track)):
            current = defender_track.iloc[i]
            previous = defender_track.iloc[i-1]
            frame_id = current['frame_id']
            
            receiver_frame = receiver_track[receiver_track.frame_id == frame_id]
            if receiver_frame.empty:
                continue
            
            receiver_row = receiver_frame.iloc[0]
            
            # Velocity
            dx = current['x'] - previous['x']
            dy = current['y'] - previous['y']
            velocity = np.array([dx, dy]) / 0.1
            
            # Positions
            defender_pos = np.array([current['x'], current['y']])
            receiver_pos = np.array([receiver_row['x'], receiver_row['y']])
            
            # cosÎ¸
            to_receiver = receiver_pos - defender_pos
            vel_norm = np.linalg.norm(velocity)
            to_norm = np.linalg.norm(to_receiver)
            
            if vel_norm > 0.1 and to_norm > 0.1:
                cos_theta = np.dot(velocity, to_receiver) / (vel_norm * to_norm)
                cos_vals.append(cos_theta)
                frames.append(frame_id)
        
        if cos_vals:
            color = 'red' if 'CB' in defender['player_position'] else 'green'
            label = f"{defender['player_name']} (ASA={defender['ASA']:.2f}s)"
            ax3.plot(frames, cos_vals, color=color, linewidth=2, label=label)
            
            # Mark commitment point
            if defender['commitment_status'] == 'committed':
                ax3.axvline(x=defender['t_commit'], color=color, 
                           linestyle='--', alpha=0.5)
    
    ax3.axhline(y=0.7, color='black', linestyle='--', alpha=0.5, label='Commitment threshold (0.7)')
    ax3.set_xlabel('Frame ID (post-throw)')
    ax3.set_ylabel('cosÎ¸ (alignment)')
    ax3.set_title('Defender Alignment Over Time')
    ax3.set_ylim([-1.1, 1.1])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)
    
    # Plot 4: ASA Timeline (FIXED VERSION)
    ax4 = plt.subplot(2, 2, 4)
    
    # Create timeline - FIX: Use proper plotting syntax
    max_time = max([d['ASA'] for d in asa_final['defender_results']]) + 0.5
    
    # Prepare data for timeline
    timeline_data = []
    for defender in asa_final['defender_results']:
        color = 'red' if 'CB' in defender['player_position'] else 'green'
        
        # Draw timeline
        ax4.axhline(y=defender['player_name'], xmin=0, xmax=defender['ASA']/max_time,
                   color=color, linewidth=8, alpha=0.3)
        
        # Collect points for plotting
        timeline_data.append({
            'x': defender['ASA'],
            'y': defender['player_name'],
            'color': color,
            'label': f"{defender['player_name']}: {defender['ASA']:.2f}s"
        })
    
    # Plot points with proper syntax
    for data in timeline_data:
        ax4.plot(data['x'], data['y'], 'o', color=data['color'], 
                markersize=10, label=data['label'])
    
    ax4.set_xlabel('Time after throw (seconds)')
    ax4.set_title('ASA Timeline: When Defenders Committed')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.legend(fontsize=8)
    
    plt.suptitle(f'ASA Analysis: {asa_final["receiver_name"]} vs Coverage\nPlay {asa_final["game_id"]}/{asa_final["play_id"]}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comprehensive plot
    comprehensive_filename = f"{save_dir}/play_{asa_final['game_id']}_{asa_final['play_id']}_comprehensive.png"
    plt.savefig(comprehensive_filename, dpi=300, bbox_inches='tight')
    print(f"âœ… Saved: {comprehensive_filename}")
    plt.close()
    
    # Plot 3: Simple ASA comparison (for presentations)
    print(f"Saving simple ASA comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Simple bar chart
    bars = ax1.bar(defenders, asa_values, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='red', linestyle='-', alpha=0.5, linewidth=2)
    ax1.set_ylabel('ASA (seconds after throw)', fontsize=12)
    ax1.set_title('Angle of Sustained Ambiguity (ASA)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, asa_val in zip(bars, asa_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{asa_val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Simple field plot
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Receiver
    ax2.plot(receiver_track['x'], receiver_track['y'], 'b-', linewidth=3, alpha=0.7, label='Receiver')
    ax2.plot(receiver_track['x'].iloc[-1], receiver_track['y'].iloc[-1], 'b*', markersize=20)
    
    # Defenders with commitment points
    for defender in asa_final['defender_results']:
        defender_id = defender['nfl_id']
        defender_track = df_out_play[df_out_play.nfl_id == defender_id].sort_values('frame_id')
        
        color = 'red' if 'CB' in defender['player_position'] else 'green'
        
        ax2.plot(defender_track['x'], defender_track['y'], color=color, linestyle='-', 
                linewidth=2, alpha=0.7, label=f"{defender['player_name'][:10]} (ASA={defender['ASA']:.2f}s)")
        
        # Mark commitment point
        if defender['commitment_status'] == 'committed':
            commit_frame = defender['t_commit']
            commit_pos = defender_track[defender_track.frame_id == commit_frame]
            if not commit_pos.empty:
                ax2.plot(commit_pos.iloc[0]['x'], commit_pos.iloc[0]['y'], 
                        color=color, marker='s', markersize=10, 
                        markeredgewidth=2, markerfacecolor='none')
    
    ax2.set_xlabel('X (yards)', fontsize=12)
    ax2.set_ylabel('Y (yards)', fontsize=12)
    ax2.set_title('Player Trajectories with Commitment Points', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    
    plt.suptitle(f'Play {asa_final["game_id"]}/{asa_final["play_id"]}: {asa_final["receiver_name"]} vs Coverage', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    simple_filename = f"{save_dir}/play_{asa_final['game_id']}_{asa_final['play_id']}_simple.png"
    plt.savefig(simple_filename, dpi=300, bbox_inches='tight')
    print(f"âœ… Saved: {simple_filename}")
    plt.close()
    
    # Also save results as CSV
    print(f"Saving ASA results as CSV...")
    
    results_data = []
    for defender in asa_final['defender_results']:
        results_data.append({
            'game_id': asa_final['game_id'],
            'play_id': asa_final['play_id'],
            'receiver_id': asa_final['receiver_id'],
            'receiver_name': asa_final['receiver_name'],
            'defender_id': defender['nfl_id'],
            'defender_name': defender['player_name'],
            'defender_position': defender['player_position'],
            'ASA': defender['ASA'],
            't_commit': defender['t_commit'],
            'commitment_status': defender['commitment_status'],
            'avg_cos_theta': defender['avg_cos_theta'],
            'min_cos_theta': defender['min_cos_theta'],
            'max_cos_theta': defender['max_cos_theta'],
            'frames_analyzed': defender['frames_analyzed']
        })
    
    import pandas as pd
    results_df = pd.DataFrame(results_data)
    csv_filename = f"{save_dir}/play_{asa_final['game_id']}_{asa_final['play_id']}_results.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"âœ… Saved: {csv_filename}")
    
    print(f"\nğŸ“� All plots and data saved to '{save_dir}/' directory!")
    print(f"   - {individual_filename}")
    print(f"   - {comprehensive_filename}")
    print(f"   - {simple_filename}")
    print(f"   - {csv_filename}")
    
    return save_dir




if asa_final:
    # ... [existing visualization code] ...
    
    print("\n" + "="*80)
    print("AUTOMATICALLY SAVING PLOTS...")
    print("="*80)
    
    save_dir = 'asa_plots'
    saved_dir = save_plots_locally(asa_final, df_out_play, save_dir)
    
    print(f"\nğŸ�‰ All plots saved to '{saved_dir}/'")
    
    # Create zip file automatically
    try:
        import zipfile
        import os
        
        zip_filename = f"{save_dir}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(save_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(save_dir))
                    zipf.write(file_path, arcname)
        
        print(f"ğŸ“¦ Created zip archive: {zip_filename}")
        print(f"   File size: {os.path.getsize(zip_filename) / 1024:.1f} KB")
        print(f"   Download with: files.download('{zip_filename}')")
        
        # In Kaggle/Colab, you can download with:
        from google.colab import files
        files.download(zip_filename)
        
    except Exception as e:
        print(f"Note: Could not create zip file: {e}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

###############################################################################
# APPENDIX: VARIABLES AND FORMULAS
###############################################################################

def generate_acp_asa_appendix():
    """
    Generate comprehensive appendix with all variables, formulas, and code.
    Organized in order of appearance from the theoretical framework.
    """
    
    appendix = {
        "metadata": {
            "generated_date": datetime.now().isoformat(),
            "framework": "Aerial Containment Pressure (ACP) & Angle of Sustained Ambiguity (ASA)",
            "author": "Defensive Coverage Analysis Team"
        },
        
        "section_0_definitions": {
            "title": "0. Core Definitions",
            "variables": {
                "ACP": "Aerial Containment Pressure - observable measure of how constrained a receiver is at the catch point",
                "ASA": "Angle of Sustained Ambiguity - latent decision-timing metric explaining when pressure forms",
                "t_rel": "Frame when ball is released (last frame in input tracking)",
                "t": "Post-release frame index (t â‰¥ 1, ball airborne)"
            }
        },
        
        "section_1_tracking": {
            "title": "1. Ball Release and Tracking Window",
            "variables": {
                "t_rel": "Ball release frame = max(frame_id) in input tracking",
                "t": "Frame index in output tracking (post-release)",
                "frames_post_throw": "Set of frames where ball is airborne"
            }
        },
        
        "section_2_player_state": {
            "title": "2. Player State",
            "variables": {
                "R(t)": "Position vector at frame t: [x(t), y(t)] (yards)",
                "x(t)": "x-coordinate (yards across field width)",
                "y(t)": "y-coordinate (yards along field length)",
                "V(t)": "Velocity vector at frame t (yards/second)",
                "s(t)": "Speed = ||V(t)|| (yards/second)",
                "Vx(t)": "x-component of velocity",
                "Vy(t)": "y-component of velocity"
            },
            "formulas": {
                "velocity_computation": "V(t) = (R(t) - R(t-1)) / Î”t",
                "speed": "s(t) = sqrt(Vx(t)Â² + Vy(t)Â²)",
                "frame_duration": "Î”t = 1/fps (default: 0.1 seconds for fps=10)"
            }
        },
        
        "section_3_catch_location": {
            "title": "3. Catch Location",
            "variables": {
                "C": "Ball landing location: [x_land, y_land]",
                "x_land": "x-coordinate of ball landing",
                "y_land": "y-coordinate of ball landing"
            },
            "notes": "C is used only for outcome evaluation, not player intent"
        },
        
        "section_4_reachable_region": {
            "title": "4. Reachable Region",
            "variables": {
                "F(t)": "Reachable region at frame t",
                "p": "Any point in RÂ²",
                "Î”Ï„": "Future horizon for reachability (seconds)"
            },
            "formulas": {
                "reachable_region": "F(t) = {p âˆˆ RÂ² : ||p - R(t)|| â‰¤ s(t)Î”Ï„}",
                "radius": "r_F(t) = s(t)Î”Ï„"
            }
        },
        
        "section_5_aerial_interaction": {
            "title": "5. Aerial Interaction Space",
            "variables": {
                "AIS(t)": "Aerial Interaction Space at frame t",
                "center": "Centered at R_rec(t)",
                "evolves": "Dynamically as defenders close, redirect, or maintain leverage"
            }
        },
        
        "section_6_acp_overview": {
            "title": "6. Aerial Containment Pressure (ACP) - Overview",
            "variables": {
                "ACP(t)": "Aerial Containment Pressure at frame t",
                "factors": "Function of: defender count, distances, closing directions"
            },
            "interpretation": {
                "low_ACP": "Open catch environment",
                "high_ACP": "Tightly contested or disrupted catch"
            }
        },
        
        "section_7_acp_detailed": {
            "title": "7. ACP Detailed Formulation",
            "subsections": {
                "7.1_targeted_receiver": {
                    "variables": {
                        "R_rec(t)": "Position of targeted receiver at frame t",
                        "nfl_id_rec": "Unique identifier for targeted receiver"
                    }
                },
                "7.2_defender_set": {
                    "variables": {
                        "D(t)": "Set of defenders within interaction radius",
                        "d": "Individual defender",
                        "r_max": "Maximum interaction radius (default: 15.0 yards)"
                    },
                    "formulas": {
                        "defender_set": "D(t) = {d : ||R_d(t) - R_rec(t)|| â‰¤ r_max}"
                    }
                },
                "7.3_relative_position": {
                    "variables": {
                        "u_d(t)": "Relative position vector from defender to receiver",
                        "Ã»_d(t)": "Unit vector toward receiver",
                        "vÌ‚_d(t)": "Unit velocity vector of defender"
                    },
                    "formulas": {
                        "relative_position": "u_d(t) = R_rec(t) - R_d(t)",
                        "unit_to_receiver": "Ã»_d(t) = u_d(t) / ||u_d(t)||",
                        "unit_velocity": "vÌ‚_d(t) = V_d(t) / ||V_d(t)||"
                    }
                },
                "7.4_closing_effectiveness": {
                    "variables": {
                        "Î³_d(t)": "Closing effectiveness term",
                        "Î¸_d(t)": "Angle between vÌ‚_d(t) and Ã»_d(t)"
                    },
                    "formulas": {
                        "closing_effectiveness": "Î³_d(t) = max(0, vÌ‚_d(t) Â· Ã»_d(t))",
                        "interpretation": {
                            "Î³=1": "Closing directly toward receiver",
                            "Î³=0": "Lateral or retreating movement"
                        }
                    }
                },
                "7.5_distance_weighting": {
                    "variables": {
                        "w_d(t)": "Distance weighting function",
                        "Ïƒ": "Spatial decay parameter (default: 5.0 yards)"
                    },
                    "formulas": {
                        "weighting": "w_d(t) = exp(-||u_d(t)|| / Ïƒ)"
                    }
                },
                "7.6_frame_level_acp": {
                    "variables": {
                        "ACP(t)": "Frame-level Aerial Containment Pressure"
                    },
                    "formulas": {
                        "acp_frame": "ACP(t) = Î£_{dâˆˆD(t)} [w_d(t) Â· Î³_d(t)]"
                    }
                },
                "7.7_catch_point_acp": {
                    "variables": {
                        "ACP_catch": "ACP at final frame of play",
                        "t_end": "Final frame when play concludes"
                    },
                    "formulas": {
                        "catch_acp": "ACP_catch = ACP(t_end)"
                    },
                    "interpretation": {
                        "low": "Receiver catches in space, minimal contest",
                        "moderate": "One or more defenders closing, contested catch likely",
                        "high": "Multiple defenders converging, interception/breakup likely"
                    }
                }
            }
        },
        
        "section_8_commitment": {
            "title": "8. Commitment and ASA",
            "subsections": {
                "8.1_commitment_definition": {
                    "variables": {
                        "t_commit_d": "Commitment frame for defender d",
                        "irreversible": "Movement eliminates class of feasible coverage outcomes"
                    }
                },
                "8.2_observable_signals": {
                    "variables": {
                        "cos(Î¸)_d(t)": "Directional alignment = vÌ‚_d(t) Â· Ã»_d(t)",
                        "Ï„": "Alignment threshold (default: 0.5)"
                    }
                },
                "8.3_sustained_commitment": {
                    "variables": {
                        "k": "Sustained frames required (default: 3 frames)",
                        "sustained_period": "[t, t+k] frames"
                    },
                    "formulas": {
                        "commitment_condition": "cos(Î¸)_d(t') â‰¥ Ï„ âˆ€ t' âˆˆ [t, t+k]"
                    }
                },
                "8.4_commitment_frame": {
                    "variables": {
                        "t_commit_d": "First irreversible directional decision toward receiver"
                    },
                    "formulas": {
                        "commitment_frame": "t_commit_d = min {t: cos(Î¸)_d(t') â‰¥ Ï„ âˆ€ t' âˆˆ [t, t+k]}"
                    }
                },
                "8.5_asa_definition": {
                    "variables": {
                        "ASA_d": "Angle of Sustained Ambiguity for defender d",
                        "Î”t": "Frame duration (0.1 seconds for fps=10)"
                    },
                    "formulas": {
                        "asa": "ASA_d = (t_commit_d - t_rel) Ã— Î”t"
                    }
                }
            }
        },
        
        "section_9_acp_growth": {
            "title": "9. ACP Growth Mechanism",
            "variables": {
                "active_defenders": "Defenders with Î³_d(t) > 0",
                "passive_defenders": "Defenders preserving ambiguity (Î³_d(t) â‰ˆ 0)"
            },
            "formulas": {
                "acp_growth": "ACP(t) â†‘ as |{d: t â‰¥ t_commit_d}| â†‘",
                "causal_chain": "ASA â†’ ACP Dynamics â†’ Catch Environment"
            }
        },
        
        "section_10_parameters": {
            "title": "10. Operational Parameters",
            "parameters": {
                "fps": "Frames per second = 10.0",
                "Î”t": "Frame duration = 0.1 seconds",
                "r_max": "Interaction radius = 15.0 yards",
                "Ïƒ": "Spatial decay = 5.0 yards",
                "Ï„": "Alignment threshold = 0.5",
                "k": "Sustained frames = 3",
                "Î”Ï„": "Reachability horizon = 0.5 seconds"
            }
        },
        
        "section_11_code_functions": {
            "title": "11. Core Python Functions",
            "functions": {
                "compute_velocity_based_acp": {
                    "purpose": "Calculate ACP_catch and ACP_mean for a play",
                    "inputs": ["df_out_play", "df_input_play", "fps", "r_max", "sigma"],
                    "outputs": ["ACP_catch", "ACP_mean"],
                    "algorithm": [
                        "1. Identify targeted receiver",
                        "2. Sort frames chronologically",
                        "3. For each post-throw frame:",
                        "   a. Get receiver position",
                        "   b. For each defender within r_max:",
                        "      i. Compute velocity from position difference",
                        "      ii. Calculate Î³ = max(0, vÌ‚ Â· Ã»)",
                        "      iii. Calculate w = exp(-dist/Ïƒ)",
                        "      iv. Add wÂ·Î³ to frame ACP",
                        "4. Return ACP at final frame and mean ACP"
                    ]
                },
                "compute_ASA_for_play": {
                    "purpose": "Calculate ASA metrics for defenders",
                    "outputs": ["ASA_min", "ASA_mean", "ASA_count"],
                    "algorithm": [
                        "1. Identify defenders",
                        "2. Track directional alignment cos(Î¸)",
                        "3. Detect sustained commitment (cos(Î¸) â‰¥ Ï„ for k frames)",
                        "4. Compute ASA = (t_commit - t_rel) Ã— Î”t",
                        "5. Aggregate across defenders"
                    ]
                }
            }
        },
        
        "section_12_formula_summary": {
            "title": "12. Formula Summary",
            "acp_formulas": [
                "1. Relative position: u_d(t) = R_rec(t) - R_d(t)",
                "2. Unit vectors: Ã»_d(t) = u_d(t)/||u_d(t)||, vÌ‚_d(t) = V_d(t)/||V_d(t)||",
                "3. Closing effectiveness: Î³_d(t) = max(0, vÌ‚_d(t) Â· Ã»_d(t))",
                "4. Distance weighting: w_d(t) = exp(-||u_d(t)||/Ïƒ)",
                "5. Frame ACP: ACP(t) = Î£_{dâˆˆD(t)} w_d(t)Â·Î³_d(t)",
                "6. Catch ACP: ACP_catch = ACP(t_end)",
                "7. Mean ACP: ACP_mean = mean(ACP(t) for t > t_rel)"
            ],
            "asa_formulas": [
                "1. Directional alignment: cos(Î¸)_d(t) = vÌ‚_d(t) Â· Ã»_d(t)",
                "2. Commitment condition: cos(Î¸)_d(t') â‰¥ Ï„ âˆ€ t' âˆˆ [t, t+k]",
                "3. Commitment frame: t_commit_d = min{t satisfying commitment condition}",
                "4. ASA: ASA_d = (t_commit_d - t_rel) Ã— Î”t"
            ]
        }
    }
    
    return appendix






