# ==========================================
# CELL 1: Imports & Setup
# ==========================================
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define paths (Kaggle specific paths)
# Define paths (Kaggle specific paths)
def find_config_paths():
    # 1. Determine search root
    search_root = "/kaggle/input"
    if not os.path.exists(search_root):
        # Local fallback
        search_root = "c:/analytics"
    
    print(f"Searching for data in: {search_root} ...")
    
    # 2. Find supplementary_data.csv
    supp_path = None
    for root, dirs, files in os.walk(search_root):
        if "supplementary_data.csv" in files:
            supp_path = os.path.join(root, "supplementary_data.csv")
            break
            
    if supp_path is None:
        raise FileNotFoundError("Could not find 'supplementary_data.csv'. Please ensure the dataset is added.")
        
    print(f"Found supplementary data at: {supp_path}")
    
    # 3. Derive Base Directory and Train Directory
    base_dir = os.path.dirname(supp_path)
    train_dir = os.path.join(base_dir, "train")
    
    # Verify train directory exists
    if not os.path.exists(train_dir):
        # Sometimes train is not in the same dir, let's search for it or assume standard structure
        # Let's try to find input_2023_w01.csv to be sure
        for root, dirs, files in os.walk(search_root):
            if "input_2023_w01.csv" in files:
                train_dir = root
                break
    
    print(f"Train directory set to: {train_dir}")
    
    return supp_path, train_dir

SUPPLEMENTARY_PATH, TRAIN_DIR = find_config_paths()

# We will use Week 1 for demonstration
INPUT_FILE = os.path.join(TRAIN_DIR, "input_2023_w01.csv")
OUTPUT_FILE = os.path.join(TRAIN_DIR, "output_2023_w01.csv")

print("Setup Complete. Data paths defined.")

# ==========================================
# CELL 2: Data Processing (Full Season DPE)
# ==========================================
def process_full_season(train_dir, supp_path):
    print("Starting Full Season Processing (Weeks 1-18)...")
    supp_df = pd.read_csv(supp_path)
    
    all_results = []
    
    # Iterate through all 18 weeks
    for week in range(1, 19):
        week_str = f"w{week:02d}"
        input_file = os.path.join(train_dir, f"input_2023_{week_str}.csv")
        output_file = os.path.join(train_dir, f"output_2023_{week_str}.csv")
        
        if not os.path.exists(input_file) or not os.path.exists(output_file):
            print(f"Skipping Week {week} (Files not found)")
            continue
            
        print(f"Processing Week {week}...")
        
        try:
            # Load Input (Metadata + Ball Landing)
            input_cols = ['game_id', 'play_id', 'nfl_id', 'player_side', 'ball_land_x', 'ball_land_y']
            input_df = pd.read_csv(input_file, usecols=input_cols)
            play_meta = input_df.drop_duplicates(subset=['game_id', 'play_id', 'nfl_id'])
            
            # Load Output (Trajectories)
            output_df = pd.read_csv(output_file)
            
            # Merge
            merged_df = pd.merge(output_df, play_meta, on=['game_id', 'play_id', 'nfl_id'], how='inner')
            
            # Filter for Defense
            defense_df = merged_df[merged_df['player_side'] == 'Defense'].copy()
            
            if defense_df.empty:
                continue

            # Calculate Vectors
            defense_df['vec_to_ball_x'] = defense_df['ball_land_x'] - defense_df['x']
            defense_df['vec_to_ball_y'] = defense_df['ball_land_y'] - defense_df['y']
            
            # Calculate Velocity
            defense_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'], inplace=True)
            defense_df['vx'] = defense_df.groupby(['game_id', 'play_id', 'nfl_id'])['x'].diff().fillna(0)
            defense_df['vy'] = defense_df.groupby(['game_id', 'play_id', 'nfl_id'])['y'].diff().fillna(0)
            
            # Calculate Efficiency
            defense_df['speed'] = np.sqrt(defense_df['vx']**2 + defense_df['vy']**2)
            defense_df['dist_to_ball'] = np.sqrt(defense_df['vec_to_ball_x']**2 + defense_df['vec_to_ball_y']**2)
            
            dot_prod = defense_df['vx'] * defense_df['vec_to_ball_x'] + defense_df['vy'] * defense_df['vec_to_ball_y']
            denom = defense_df['speed'] * defense_df['dist_to_ball']
            
            defense_df['efficiency'] = np.nan
            mask = (denom > 0)
            defense_df.loc[mask, 'efficiency'] = dot_prod.loc[mask] / denom.loc[mask]
            
            # --- NEW: Reaction Time Analysis ---
            # Define "Reaction" as the first frame where efficiency > 0.5 (moving significantly towards ball)
            # We calculate the time from ball snap (frame 1) to this reaction frame.
            # Note: This is a simplified proxy.
            reaction_df = defense_df[defense_df['efficiency'] > 0.5].groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].min().reset_index()
            reaction_df.rename(columns={'frame_id': 'reaction_frame'}, inplace=True)
            
            # Aggregate per player-play
            week_results = defense_df.groupby(['game_id', 'play_id', 'nfl_id']).agg({
                'efficiency': 'mean',
                'speed': 'mean'
            }).reset_index()
            
            # Merge Reaction Time
            week_results = pd.merge(week_results, reaction_df, on=['game_id', 'play_id', 'nfl_id'], how='left')
            
            all_results.append(week_results)
            
        except Exception as e:
            print(f"Error processing Week {week}: {e}")
            continue

    if not all_results:
        return pd.DataFrame()
        
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.columns = ['game_id', 'play_id', 'nfl_id', 'avg_efficiency', 'avg_speed', 'reaction_frame']
    
    # Add Context
    final_df = pd.merge(final_df, supp_df[['game_id', 'play_id', 'pass_result', 'possession_team']], on=['game_id', 'play_id'], how='left')
    
    # Add Player Names (Robust Recursive Search)
    print("Searching for players.csv...")
    players_path = None
    
    # Search in the same directory as supplementary data
    search_dirs = [
        os.path.dirname(supp_path),
        os.path.dirname(os.path.dirname(supp_path)), # One level up
        "/kaggle/input" # Root input
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            if "players.csv" in files:
                players_path = os.path.join(root, "players.csv")
                print(f"Found players.csv at: {players_path}")
                break
        if players_path:
            break
            
    if players_path and os.path.exists(players_path):
        players_df = pd.read_csv(players_path)
        # Ensure nflId is int for merging
        if 'nflId' in players_df.columns:
            final_df = pd.merge(final_df, players_df[['nflId', 'displayName', 'position']], left_on='nfl_id', right_on='nflId', how='left')
    else:
        print("WARNING: players.csv not found in any search path. Leaderboards will use NFL IDs.")
        final_df['displayName'] = final_df['nfl_id'].astype(str)
        final_df['position'] = 'UNK'
    
    return final_df

print("Processing Full Season Data...")
dpe_results = process_full_season(TRAIN_DIR, SUPPLEMENTARY_PATH)

# ==========================================
# CELL 3: Verification & Leaderboards
# ==========================================
def verify_and_rank(df):
    print("\n--- 1. Data Verification ---")
    # Check 1: Value Bounds
    out_of_bounds = df[(df['avg_efficiency'] < -1.0001) | (df['avg_efficiency'] > 1.0001)]
    if not out_of_bounds.empty:
        print(f"WARNING: Found {len(out_of_bounds)} records with invalid DPE scores!")
    else:
        print("PASSED: All DPE scores are within valid range [-1, 1].")
        
    # Check 2: Missing Data
    missing_reaction = df['reaction_frame'].isna().sum()
    print(f"Info: {missing_reaction} plays have no clear 'reaction' moment (efficiency never > 0.5).")
    
    print("\n--- 2. DPE Leaderboards (Minimum 50 Plays) ---")
    
    # Player Leaderboard
    player_stats = df.groupby(['nfl_id', 'displayName', 'position']).agg({
        'avg_efficiency': 'mean',
        'reaction_frame': 'mean',
        'game_id': 'count'
    }).reset_index()
    
    player_stats.rename(columns={'game_id': 'play_count'}, inplace=True)
    qualified_players = player_stats[player_stats['play_count'] >= 50].sort_values('avg_efficiency', ascending=False)
    
    print("\nTop 10 Defenders by Pursuit Efficiency:")
    print(qualified_players.head(10)[['displayName', 'position', 'avg_efficiency', 'reaction_frame', 'play_count']].to_string(index=False))
    
    print("\nTop 10 Fastest Reactors (Lowest Reaction Frame):")
    print(qualified_players.sort_values('reaction_frame').head(10)[['displayName', 'position', 'reaction_frame', 'avg_efficiency']].to_string(index=False))

    # Statistical Significance (Full Season)
    print("\n--- 3. Statistical Validation (Full Season) ---")
    relevant = ['C', 'I', 'IN']
    df_filtered = df[df['pass_result'].isin(relevant)].copy()
    
    complete = df_filtered[df_filtered['pass_result'] == 'C']['avg_efficiency'].dropna()
    intercepted = df_filtered[df_filtered['pass_result'] == 'IN']['avg_efficiency'].dropna()
    
    t_stat, p_val = stats.ttest_ind(complete, intercepted)
    print(f"T-Test (Complete vs Interception): P-Value = {p_val:.5e}")
    if p_val < 0.05:
        print("CONCLUSION: The DPE metric SIGNIFICANTLY differentiates between interceptions and completions.")

verify_and_rank(dpe_results)

# ==========================================
# CELL 4: Visualization (HTML Generation)
# ==========================================
def generate_visualization():
    # Hardcoded sample play for demonstration
    TARGET_GAME_ID = 2023090700
    TARGET_PLAY_ID = 3461
    HTML_OUTPUT = "dpe_visualization.html"
    
    print(f"\nGenerating Visualization for Game {TARGET_GAME_ID} Play {TARGET_PLAY_ID}...")

    def get_play_data(file_path, game_id, play_id):
        chunksize = 100000
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            play_data = chunk[(chunk['game_id'] == game_id) & (chunk['play_id'] == play_id)]
            if not play_data.empty:
                return play_data
        return pd.DataFrame()

    play_input = get_play_data(INPUT_FILE, TARGET_GAME_ID, TARGET_PLAY_ID)
    play_output = get_play_data(OUTPUT_FILE, TARGET_GAME_ID, TARGET_PLAY_ID)

    if play_input.empty:
        print("No input data found for visualization.")
        return

    ball_land_x = play_input['ball_land_x'].iloc[0]
    ball_land_y = play_input['ball_land_y'].iloc[0]

    # Calculate DPE for this play locally
    defenders = play_input[play_input['player_side'] == 'Defense']['nfl_id'].unique()
    dpe_scores = {}
    
    for nfl_id in defenders:
        p_out = play_output[play_output['nfl_id'] == nfl_id].copy().sort_values('frame_id')
        if p_out.empty:
            continue
            
        p_out['vec_to_ball_x'] = ball_land_x - p_out['x']
        p_out['vec_to_ball_y'] = ball_land_y - p_out['y']
        p_out['vx'] = p_out['x'].diff().fillna(0)
        p_out['vy'] = p_out['y'].diff().fillna(0)
        p_out['speed'] = np.sqrt(p_out['vx']**2 + p_out['vy']**2)
        p_out['dist_to_ball'] = np.sqrt(p_out['vec_to_ball_x']**2 + p_out['vec_to_ball_y']**2)
        
        dot_prod = p_out['vx'] * p_out['vec_to_ball_x'] + p_out['vy'] * p_out['vec_to_ball_y']
        denom = p_out['speed'] * p_out['dist_to_ball']
        
        mask = (denom > 0)
        p_out.loc[mask, 'efficiency'] = dot_prod.loc[mask] / denom.loc[mask]
        
        avg_eff = p_out['efficiency'].mean()
        if not np.isnan(avg_eff):
            dpe_scores[int(nfl_id)] = round(avg_eff, 2)

    # Prepare JSON
    players = {}
    for _, row in play_input.iterrows():
        nid = int(row['nfl_id'])
        if nid not in players:
            players[nid] = {
                'id': nid,
                'side': row['player_side'],
                'dpe': dpe_scores.get(nid, None),
                'frames': []
            }
        players[nid]['frames'].append({
            'x': row['x'],
            'y': row['y'],
            'type': 'input'
        })
        
    if not play_output.empty:
        for _, row in play_output.iterrows():
            nid = int(row['nfl_id'])
            if nid in players:
                players[nid]['frames'].append({
                    'x': row['x'],
                    'y': row['y'],
                    'type': 'output'
                })

    data_json = json.dumps({
        'gameId': int(TARGET_GAME_ID),
        'playId': int(TARGET_PLAY_ID),
        'players': list(players.values()),
        'ballLand': {'x': ball_land_x, 'y': ball_land_y}
    })

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DPE Visualization</title>
    <style>
        body {{ font-family: sans-serif; background: #333; color: white; }}
        #container {{ position: relative; width: 1200px; margin: 20px auto; }}
        canvas {{ background-color: #2a4d2a; border: 2px solid white; }}
        .legend {{ margin-top: 10px; text-align: center; }}
        .legend span {{ margin: 0 10px; }}
    </style>
</head>
<body>
    <div id="container">
        <h2>Defensive Pursuit Efficiency (DPE) - Game {TARGET_GAME_ID} Play {TARGET_PLAY_ID}</h2>
        <canvas id="field" width="1200" height="533"></canvas>
        <div class="legend">
            <span style="color: cyan">Offense</span>
            <span style="color: #00ff00">High DPE (>0.7)</span>
            <span style="color: yellow">Med DPE (0.4-0.7)</span>
            <span style="color: red">Low DPE (<0.4)</span>
            <span style="color: white">Ball Landing (X)</span>
        </div>
    </div>
    <script>
        const data = {data_json};
        const canvas = document.getElementById('field');
        const ctx = canvas.getContext('2d');
        const SCALE = 10;

        function getColor(p) {{
            if (p.side === 'Offense') return 'cyan';
            if (p.dpe === null) return 'gray';
            if (p.dpe > 0.7) return '#00ff00';
            if (p.dpe > 0.4) return 'yellow';
            return 'red';
        }}

        function drawField() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            for(let i=10; i<=110; i+=10) {{
                ctx.beginPath();
                ctx.moveTo(i*SCALE, 0);
                ctx.lineTo(i*SCALE, canvas.height);
                ctx.stroke();
            }}
        }}

        function drawPlayers() {{
            data.players.forEach(p => {{
                ctx.fillStyle = getColor(p);
                
                // Draw Trajectory
                ctx.beginPath();
                p.frames.forEach((f, i) => {{
                    const x = f.x * SCALE;
                    const y = (53.3 - f.y) * SCALE;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }});
                ctx.strokeStyle = getColor(p);
                ctx.lineWidth = 2;
                ctx.stroke();

                // Draw End Position & Label
                const last = p.frames[p.frames.length - 1];
                const lx = last.x * SCALE;
                const ly = (53.3 - last.y) * SCALE;
                
                ctx.beginPath();
                ctx.arc(lx, ly, 4, 0, 2 * Math.PI);
                ctx.fill();
                
                if (p.dpe !== null) {{
                    ctx.fillStyle = 'white';
                    ctx.font = '10px Arial';
                    ctx.fillText(p.dpe, lx + 5, ly - 5);
                }}
            }});
            
            // Ball Land
            const bx = data.ballLand.x * SCALE;
            const by = (53.3 - data.ballLand.y) * SCALE;
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(bx-5, by-5);
            ctx.lineTo(bx+5, by+5);
            ctx.moveTo(bx+5, by-5);
            ctx.lineTo(bx-5, by+5);
            ctx.stroke();
        }}

        drawField();
        drawPlayers();
    </script>
</body>
</html>
    """
    
    with open(HTML_OUTPUT, "w") as f:
        f.write(html_content)
    
    print(f"DPE Visualization saved to {HTML_OUTPUT}")

# Run Visualization
generate_visualization()


