import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import glob
import warnings


# warnings preventions
warnings.filterwarnings('ignore')
plt.style.use('dark_background')


BASE_DIR = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final'


def find_best_play():
    print("Scanning for the perfect play...")
    #here i check the first few weeks for a play that is long enough 60-120 frames
    # this ensures the video is not too short or too boring.
    output_files = sorted(glob.glob(f'{BASE_DIR}/train/output_2023_w0[1-5].csv'))
    
    best_meta = None
    max_frames = 0
    
    for f in output_files:
        try:
            df_check = pd.read_csv(f, usecols=['game_id', 'play_id', 'frame_id'])
            counts = df_check.groupby(['game_id', 'play_id'])['frame_id'].nunique()
            valid_plays = counts[(counts > 60) & (counts < 120)]
            
            if not valid_plays.empty:
                curr_max = valid_plays.max()
                if curr_max > max_frames:
                    max_frames = curr_max
                    gid, pid = valid_plays.idxmax()
                    week_num = f.split('_w')[-1][:2]
                    best_meta = (gid, pid, week_num)
        except: continue
                
    if best_meta:
        print(f" -> Selected: Game {best_meta[0]} Play {best_meta[1]} (Week {best_meta[2]})")
        return best_meta
    else:
        # fallingback to a known play if scan fails
        return (2023090700, 877, '01')


def load_and_merge_play(game_id, play_id, week_num):
    input_path = f'{BASE_DIR}/train/input_2023_w{week_num}.csv'
    output_path = f'{BASE_DIR}/train/output_2023_w{week_num}.csv'
    
    df_in = pd.read_csv(input_path)
    df_out = pd.read_csv(output_path)
    
    # filter out for specific play
    play_in = df_in[(df_in['game_id'] == game_id) & (df_in['play_id'] == play_id)].copy()
    play_out = df_out[(df_out['game_id'] == game_id) & (df_out['play_id'] == play_id)].copy()
    
    if play_out.empty: raise ValueError("Output data is empty.")

    # merge metadata Who is Who?
    if 'player_side' in play_in.columns:
        meta_map = play_in[['nfl_id', 'player_side']].drop_duplicates().set_index('nfl_id')
        play_out = play_out.join(meta_map, on='nfl_id')
        play_out['TeamType'] = play_out['player_side'].map({'Offense': 'Home', 'Defense': 'Away'})
    else:
        play_out['TeamType'] = 'Unknown'

    play_out.dropna(subset=['TeamType'], inplace=True)

    #and calculate velocity v = dX/dt
    play_out = play_out.sort_values(['nfl_id', 'frame_id'])
    play_out['vx'] = play_out.groupby('nfl_id')['x'].diff().fillna(0) / 0.1
    play_out['vy'] = play_out.groupby('nfl_id')['y'].diff().fillna(0) / 0.1
    
    # pivot to wide format for Physics engine
    play_out['player_id'] = play_out['TeamType'] + '_' + play_out['nfl_id'].astype(str)
    wide_df = play_out.pivot_table(index='frame_id', columns='player_id', values=['x', 'y', 'vx', 'vy'])
    wide_df.columns = [f"{col[1]}_{col[0]}" for col in wide_df.columns]
    
    # add target Ball Landing Spot
    land_x = play_in['ball_land_x'].iloc[0] if 'ball_land_x' in play_in.columns else 60
    land_y = play_in['ball_land_y'].iloc[0] if 'ball_land_y' in play_in.columns else 26.65
    wide_df['ball_x'] = land_x
    wide_df['ball_y'] = land_y
    
    return wide_df


def calculate_control_surface(row, field_dimen=(120, 53.3), n_grid_x=30):
    n_grid_y = int(n_grid_x * field_dimen[1] / field_dimen[0])
    xgrid = np.linspace(0, 120, n_grid_x)
    ygrid = np.linspace(0, 53.3, n_grid_y)
    PPCFa = np.zeros((n_grid_y, n_grid_x))
    
    att_players = []
    def_players = []
    
    # Parse players from row
    for col in row.index:
        if '_x' in col and 'ball' not in col:
            pid = col.replace('_x', '')
            team = pid.split('_')[0]
            
            p_obj = type('', (), {})() 
            p_obj.pos = np.array([row[pid+'_x'], row[pid+'_y']])
            p_obj.vmax = 11.0 # NFL Elite Speed (approx 24mph)
            p_obj.reaction = 0.4 # Elite Reaction Time
            
            if team == 'Home': att_players.append(p_obj)
            else: def_players.append(p_obj)
            
    # Calculate Control
    if not att_players or not def_players: return PPCFa

    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            target = np.array([xgrid[j], ygrid[i]])
            tau_att = np.min([p.reaction + np.linalg.norm(target - p.pos)/p.vmax for p in att_players])
            tau_def = np.min([p.reaction + np.linalg.norm(target - p.pos)/p.vmax for p in def_players])
            
            # Sigmoid Function
            PPCFa[i, j] = 1 / (1 + np.exp(-1.5 * (tau_def - tau_att)))
            
    return PPCFa


try:
    # 1. Get Data
    best_meta = find_best_play()
    game_id, play_id, week_num = best_meta
    wide_df = load_and_merge_play(game_id, play_id, week_num)
    
    # 2. Broadcast Loop (Forward -> Reverse -> Forward)
    # This makes the video 3x longer and looks like a TV replay
    seq = wide_df.index.tolist()
    frames_to_plot = seq + seq[::-1] + seq 
    
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('#2E8B57')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    for x in range(10, 111, 10): ax.axvline(x, c='white', alpha=0.5)

    # Init Objects
    heatmap = ax.imshow(np.zeros((13, 30)), extent=(0, 120, 0, 53.3), origin='lower', cmap='bwr', alpha=0.6, vmin=0, vmax=1)
    scat_off = ax.scatter([], [], c='blue', s=80, edgecolors='white', label='Offense')
    scat_def = ax.scatter([], [], c='red', s=80, edgecolors='white', label='Defense')
    scat_target = ax.scatter([], [], c='gold', marker='x', s=200, linewidth=3, label='Target')
    title = ax.text(60, 50, "", color='white', ha='center', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    # 4. Animate
    def update(frame_id):
        row = wide_df.loc[frame_id]
        
        # Update Heatmap
        heatmap.set_data(calculate_control_surface(row))
        
        # Update Players
        off_cols = [c.replace('_x','') for c in wide_df.columns if 'Home' in c and '_x' in c]
        scat_off.set_offsets(np.c_[[row[c+'_x'] for c in off_cols], [row[c+'_y'] for c in off_cols]])
        
        def_cols = [c.replace('_x','') for c in wide_df.columns if 'Away' in c and '_x' in c]
        scat_def.set_offsets(np.c_[[row[c+'_x'] for c in def_cols], [row[c+'_y'] for c in def_cols]])
        
        # Update Target
        scat_target.set_offsets(np.c_[row['ball_x'], row['ball_y']])
        title.set_text(f"PCC Model | Game {game_id} | Frame {frame_id}")
        return heatmap, scat_off, scat_def, scat_target, title

    print(f"Rendering {len(frames_to_plot)} frames... (Approx 2-3 mins)")
    
    # FIXED: Calling animation.FuncAnimation explicitly to avoid 'module not callable' error
    anim = animation.FuncAnimation(fig, update, frames=frames_to_plot, interval=100, blit=True)
    anim.save('PCC_Analysis_Final.mp4', writer='ffmpeg', fps=10)

    print("SUCCESS: Video saved as 'PCC_Analysis_Final.mp4'. Download it from the Output tab!")
    
except Exception as e:
    print(f"Error: {e}")


from IPython.display import Video, display
import os

# Define the filename we just saved
video_path = 'PCC_Analysis_Final.mp4'

if os.path.exists(video_path):
    print(f"Displaying {video_path}...")
    # Embed the video with specific width for better viewing
    display(Video(video_path, embed=True, width=800, height=450))
else:
    print(f"Error: {video_path} not found. Checking current directory...")
    print(os.listdir('.'))




