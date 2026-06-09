import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import glob
import random
import numpy as np
from math import sqrt
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
from IPython.display import HTML
from IPython.display import Image as IPImage, display
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', 'Columns .* have mixed types', category=pd.errors.DtypeWarning)


supplementary_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')
supplementary_data.head(20)


supplementary_data.info()


input_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv')
input_data.head(20)


input_data.info()


output_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w01.csv')
output_data.head(20)


output_data.info()


train_folder = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
input_files = sorted(glob.glob(os.path.join(train_folder, "input_2023_w*.csv")))
output_files = sorted(glob.glob(os.path.join(train_folder, "output_2023_w*.csv")))

stats = []
for in_f, out_f in zip(input_files, output_files):
    week = os.path.basename(in_f).split("_w")[-1].split(".")[0]
    df_in = pd.read_csv(in_f, low_memory=False)
    df_out = pd.read_csv(out_f, low_memory=False)
    merged = pd.merge(df_in, df_out, on=['game_id','play_id'], how='left')
    stats.append({
        "week": week,
        "input_rows": len(df_in),
        "output_rows": len(df_out),
        "merged_rows": len(merged),
        "input_cols": len(df_in.columns),
        "output_cols": len(df_out.columns),
        "merged_cols": len(merged.columns),
        "null_values": merged.isnull().sum().sum(),
    })
    del df_in, df_out, merged

summary = pd.DataFrame(stats)
print(summary)


df_in_list, df_out_list = [], []

for in_f, out_f in zip(input_files, output_files):
    df_in = pd.read_csv(in_f, usecols=['game_id','play_id','frame_id','player_name','player_position','x','y','s','a','dir','o'], low_memory=False)
    df_out = pd.read_csv(out_f, usecols=['game_id','play_id','nfl_id','x','y'], low_memory=False)
    df_in_list.append(df_in)
    df_out_list.append(df_out)

df_in_all = pd.concat(df_in_list, ignore_index=True)
df_out_all = pd.concat(df_out_list, ignore_index=True)

desc_stats = df_in_all[['s','a','dir','x','y']].describe().T
corr_matrix = df_in_all[['s','a','dir','x','y','o']].corr()
pos_counts = df_in_all['player_position'].value_counts().head(15)
play_counts = df_in_all.groupby('play_id')['player_name'].count().describe()
game_counts = df_in_all.groupby('game_id')['play_id'].nunique().describe()

print("Descriptive Stats:\n", desc_stats)
print("\nCorrelation Matrix:\n", corr_matrix)
print("\nTop Player Positions:\n", pos_counts)
print("\nPlayers per Play:\n", play_counts)
print("\nPlays per Game:\n", game_counts)


plt.figure(figsize=(12,6))
sns.histplot(df_in_all['s'], bins=50, kde=True)
plt.title('Distribution of Player Speed')
plt.show()


plt.figure(figsize=(12,6))
sns.histplot(df_in_all['a'], bins=50, kde=True)
plt.title('Distribution of Player Acceleration')
plt.show()


plt.figure(figsize=(14,6))
sns.boxplot(data=df_in_all, x='player_position', y='s', order=pos_counts.index)
plt.xticks(rotation=45)
plt.title('Speed by Player Position')
plt.show()


sample_df = df_in_all.sample(10000, random_state=42)
plt.figure(figsize=(8,5))
plt.scatter(sample_df['x'], sample_df['y'], alpha=0.4, s=10)
plt.title('Player Field Positions (x vs y)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


train_folder = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/'
input_sample = pd.read_csv(f'{train_folder}input_2023_w01.csv', low_memory=False)
output_sample = pd.read_csv(f'{train_folder}output_2023_w01.csv', low_memory=False)
df = pd.merge(input_sample, output_sample, on=['game_id','play_id'], how='left')

num_cols = df.select_dtypes(include=['float64','int64']).columns
desc = df[num_cols].describe()

Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).sum()


plt.figure(figsize=(10,5))
sns.barplot(x=outliers.index, y=outliers.values)
plt.xticks(rotation=90)
plt.title("Outlier Count per Numeric Feature")
plt.tight_layout()
plt.show()


clean_num = df[num_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
corr = clean_num.corr().fillna(0)

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap")
plt.show()


sample_df = df.sample(n=5000, random_state=42)
sns.pairplot(sample_df[num_cols[:5]])
plt.show()


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[num_cols].fillna(0))
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
dummy_target = np.random.randint(0,2,len(df))
xgb.fit(scaled_data, dummy_target)
importance = pd.Series(xgb.feature_importances_, index=num_cols).sort_values(ascending=False)


plt.figure(figsize=(10,5))
sns.barplot(x=importance.values[:15], y=importance.index[:15])
plt.title("Top Feature Importances (XGBoost Proxy)")
plt.tight_layout()
plt.show()


base = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final'
train_folder = os.path.join(base, 'train')
supp_path = os.path.join(base, 'supplementary_data.csv')
input_files = sorted(glob.glob(os.path.join(train_folder, "input_2023_w*.csv")))
output_files = sorted(glob.glob(os.path.join(train_folder, "output_2023_w*.csv")))
supp = pd.read_csv(supp_path, low_memory=False)
out_path = '/kaggle/working/merged_postthrow_all_weeks.csv'
if os.path.exists(out_path):
    os.remove(out_path)
first_write = True
total_rows = 0
for in_f, out_f in zip(input_files, output_files):
    df_in = pd.read_csv(in_f, low_memory=False)
    df_out = pd.read_csv(out_f, low_memory=False)
    grp = df_in.groupby(['game_id','play_id'])['frame_id'].max().reset_index().rename(columns={'frame_id':'throw_frame'})
    df_out = df_out.merge(grp, on=['game_id','play_id'], how='left')
    df_out = df_out[df_out['throw_frame'].notnull()].copy()
    df_out['throw_frame'] = df_out['throw_frame'].astype(int)
    df_out['abs_frame_id'] = df_out['throw_frame'] + df_out['frame_id'].astype(int)
    player_info = df_in.drop_duplicates(subset=['game_id','play_id','nfl_id'])[[
        'game_id','play_id','nfl_id','player_name','player_position','player_side',
        'player_role','player_weight','player_height','play_direction','absolute_yardline_number'
    ]]
    merged = df_out.merge(player_info, on=['game_id','play_id','nfl_id'], how='left')
    merged = merged.merge(supp[['game_id','play_id','pass_result','pass_length','offense_formation','receiver_alignment','route_of_targeted_receiver','defenders_in_the_box']], on=['game_id','play_id'], how='left')
    merged['week'] = int(os.path.basename(in_f).split('_w')[-1].split('.')[0])
    cols_order = ['week','game_id','play_id','nfl_id','frame_id','abs_frame_id','x','y',
                  'player_name','player_position','player_side','player_role','player_weight','player_height','play_direction','absolute_yardline_number',
                  'pass_result','pass_length','offense_formation','receiver_alignment','route_of_targeted_receiver','defenders_in_the_box']
    cols_present = [c for c in cols_order if c in merged.columns]
    to_write = merged[cols_present]
    if first_write:
        to_write.to_csv(out_path, index=False, mode='w')
        first_write = False
    else:
        to_write.to_csv(out_path, index=False, mode='a', header=False)
    total_rows += len(to_write)
    print(os.path.basename(in_f), "-> rows written so far:", total_rows)
print("TOTAL ROWS WRITTEN:", total_rows)
print("MERGED FILE:", out_path)


base = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final'
train = os.path.join(base, 'train')
supp_path = os.path.join(base, 'supplementary_data.csv')
input_files = sorted(glob.glob(os.path.join(train, "input_2023_w*.csv")))
output_files = sorted(glob.glob(os.path.join(train, "output_2023_w*.csv")))
if not os.path.exists(base) or len(input_files)==0 or len(output_files)==0 or not os.path.exists(supp_path):
    raise FileNotFoundError("Data files not found. Check that paths exist and files are uploaded.")
supp = pd.read_csv(supp_path, low_memory=False)[['game_id','play_id','pass_result','pass_length','offense_formation','receiver_alignment','route_of_targeted_receiver','defenders_in_the_box']]


out_players = '/kaggle/working/player_movement_metrics.csv'
out_plays = '/kaggle/working/play_movement_metrics.csv'
if os.path.exists(out_players): os.remove(out_players)
if os.path.exists(out_plays): os.remove(out_plays)


def monotone_chain(points):

    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def polygon_area(points):

    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


frame_rate = 10.0
player_rows = []
play_rows = []

for in_f, out_f in zip(input_files, output_files):
    df_in = pd.read_csv(in_f, low_memory=False)
    df_out = pd.read_csv(out_f, low_memory=False)
    df_in['frame_id'] = df_in['frame_id'].astype(int)
    max_map = df_in.groupby(['game_id','play_id'])['frame_id'].max().reset_index().rename(columns={'frame_id':'throw_frame'})
    df_out = df_out.merge(max_map, on=['game_id','play_id'], how='left')
    df_out = df_out[df_out['throw_frame'].notnull()].copy()
    df_out['throw_frame'] = df_out['throw_frame'].astype(int)
    df_out['abs_frame_id'] = df_out['throw_frame'] + df_out['frame_id'].astype(int)
    df_in_players = df_in.drop_duplicates(subset=['game_id','play_id','nfl_id'])[['game_id','play_id','nfl_id','player_name','player_position','player_side','player_role','player_weight','player_height','play_direction','absolute_yardline_number']]
    merged_week = df_out.merge(df_in_players, on=['game_id','play_id','nfl_id'], how='left')
    merged_week = merged_week.merge(supp, on=['game_id','play_id'], how='left')
    merged_week['week'] = int(os.path.basename(in_f).split('_w')[-1].split('.')[0])
    plays = merged_week[['game_id','play_id']].drop_duplicates().values.tolist()
    for game_id, play_id in plays:
        play_df = merged_week[(merged_week['game_id']==game_id)&(merged_week['play_id']==play_id)].copy()
        if play_df.empty: continue
        play_df = play_df.sort_values('abs_frame_id')
        frames = sorted(play_df['abs_frame_id'].unique())
        groups = play_df.groupby('nfl_id')
        play_off_speed_sum=0.0; play_off_count=0; play_def_speed_sum=0.0; play_def_count=0
        separation_acc = {}
        for nfl_id, g in groups:
            g = g.sort_values('abs_frame_id')
            xs = g['x'].astype(float).values; ys = g['y'].astype(float).values
            frs = g['abs_frame_id'].values
            if len(xs)<2:
                total_distance=0.0; avg_speed=0.0; max_speed=0.0; max_acc=0.0; displacement=0.0
            else:
                dx = np.diff(xs); dy = np.diff(ys)
                dists = np.sqrt(dx*dx + dy*dy)
                dt = np.diff(frs)/frame_rate
                dt[dt==0]=1.0/frame_rate
                speeds = dists/dt
                total_distance = float(dists.sum())
                avg_speed = float(speeds.mean()) if len(speeds)>0 else 0.0
                max_speed = float(speeds.max()) if len(speeds)>0 else 0.0
                acc = np.diff(speeds)/(1.0/frame_rate) if len(speeds)>1 else np.array([0.0])
                max_acc = float(acc.max()) if len(acc)>0 else 0.0
                displacement = float(sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2))
            role = g['player_role'].iloc[0] if 'player_role' in g.columns else ''
            side = g['player_side'].iloc[0] if 'player_side' in g.columns else ''
            player_rows.append([game_id,play_id,nfl_id,role,side,total_distance,displacement,avg_speed,max_speed,max_acc,merged_week['week'].iloc[0]])
            if str(side).lower().startswith('off'):
                play_off_speed_sum += avg_speed; play_off_count += 1
            else:
                play_def_speed_sum += avg_speed; play_def_count += 1
        for f in frames:
            frame_df = play_df[play_df['abs_frame_id']==f]
            offs = frame_df[frame_df['player_side'].str.lower().str.contains('off', na=False)]
            defs = frame_df[frame_df['player_side'].str.lower().str.contains('def', na=False)]
            def_coords = defs[['x','y']].astype(float).values
            if len(def_coords)==0: continue
            for _, row in offs.iterrows():
                ox, oy, nid = float(row['x']), float(row['y']), row['nfl_id']
                dists = np.sqrt((def_coords[:,0]-ox)**2 + (def_coords[:,1]-oy)**2)
                min_dist = float(dists.min()) if len(dists)>0 else np.nan
                separation_acc.setdefault((game_id,play_id,nid),[]).append(min_dist)
        for key, arr in separation_acc.items():
            mean_sep = float(np.nanmean(arr)) if len(arr)>0 else np.nan
            game_id_k, play_id_k, nid_k = key
            player_rows.append([game_id_k,play_id_k,nid_k,'','',np.nan,np.nan,np.nan,np.nan,np.nan,merged_week['week'].iloc[0]])
        off_pts = play_df[play_df['player_side'].str.lower().str.contains('off', na=False)][['x','y']].astype(float).values.tolist()
        def_pts = play_df[play_df['player_side'].str.lower().str.contains('def', na=False)][['x','y']].astype(float).values.tolist()
        def hull_area(pts):
            if len(pts)<3: return 0.0
            hull = monotone_chain(pts); return polygon_area(hull)
        off_area = hull_area(off_pts); def_area = hull_area(def_pts)
        play_rows.append([game_id,play_id,(play_off_speed_sum/play_off_count) if play_off_count>0 else 0.0,(play_def_speed_sum/play_def_count) if play_def_count>0 else 0.0,off_area,def_area,merged_week['week'].iloc[0]])
    df_in=None; df_out=None; merged_week=None


pl_df = pd.DataFrame(play_rows, columns=['game_id','play_id','avg_off_speed','avg_def_speed','off_area','def_area','week'])
pl_df.to_csv(out_plays, index=False)
cols = ['game_id','play_id','nfl_id','player_role','player_side','total_distance','displacement','avg_speed','max_speed','max_acc','week']
p_rows = pd.DataFrame(player_rows, columns=cols)
agg = p_rows.groupby(['game_id','play_id','nfl_id']).agg({
    'player_role':'first','player_side':'first','total_distance':'first','displacement':'first','avg_speed':'first','max_speed':'first','max_acc':'first','week':'first'
}).reset_index()
agg.to_csv(out_players, index=False)
print("written", out_players, out_plays)


train_folder = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
supp_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"
supp = pd.read_csv(supp_path)

input_files = sorted([os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.startswith("input_")])
output_files = sorted([os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.startswith("output_")])

in_f = random.choice(input_files)
out_f = in_f.replace("input_", "output_")

df_in = pd.read_csv(in_f, low_memory=False)
df_out = pd.read_csv(out_f, low_memory=False)

df_in['frame_id'] = df_in['frame_id'].astype(int)
max_map = df_in.groupby(['game_id','play_id'])['frame_id'].max().reset_index().rename(columns={'frame_id':'throw_frame'})

df_out = df_out.merge(max_map, on=['game_id','play_id'], how='left')
df_out = df_out[df_out['throw_frame'].notnull()].copy()
df_out['throw_frame'] = df_out['throw_frame'].astype(int)

df_out['abs_frame_id'] = df_out['throw_frame'] + df_out['frame_id'].astype(int)

df_in_players = df_in.drop_duplicates(subset=['game_id','play_id','nfl_id'])[['game_id','play_id','nfl_id','player_name','player_side','player_role','player_position']]
merged = df_out.merge(df_in_players, on=['game_id','play_id','nfl_id'], how='left')

play_ids = merged['play_id'].drop_duplicates().sample(3, random_state=42).tolist()
field_width, field_length = 53.3, 120

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, field_length)
ax.set_ylim(0, field_width)
off_scatter, = ax.plot([], [], 'ro', label='Offense', markersize=8)
def_scatter, = ax.plot([], [], 'bo', label='Defense', markersize=8)
ax.legend()
ax.set_title("NFL Player Movement After Throw")
plt.xlabel("Field Length (Yards)")
plt.ylabel("Field Width (Yards)")

frames = []
for pid in play_ids:
    play_df = merged[merged['play_id'] == pid]
    play_df = play_df.sort_values('abs_frame_id')
    for fid in sorted(play_df['abs_frame_id'].unique()):
        frame_df = play_df[play_df['abs_frame_id'] == fid]
        frames.append((pid, frame_df))

def init():
    off_scatter.set_data([], [])
    def_scatter.set_data([], [])
    return off_scatter, def_scatter

def update(frame):
    pid, frame_df = frame
    ax.set_title(f"Game Play ID: {pid} (Frame: {frame_df['abs_frame_id'].iloc[0]})")
    
    off_df = frame_df[frame_df['player_side'].str.lower().str.contains('off', na=False)]
    def_df = frame_df[frame_df['player_side'].str.lower().str.contains('def', na=False)]
    
    off_scatter.set_data(off_df['x'], off_df['y'])
    def_scatter.set_data(def_df['x'], def_df['y'])
    
    return off_scatter, def_scatter

anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=100)

HTML(anim.to_jshtml())


base_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
week_file = "input_2023_w01.csv"
save_dir = "/kaggle/working/frames_enhanced"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(f"{base_path}/{week_file}", low_memory=False)
unique_plays = df.groupby(['game_id', 'play_id']).size().index.tolist()
sample_plays = random.sample(unique_plays, 3)

def draw_field(ax):
    ax.set_facecolor('#1E5128')
    ax.plot([0,120],[0,0],color='white')
    ax.plot([0,120],[53.3,53.3],color='white')
    for x in range(10,120,10):
        ax.plot([x,x],[0,53.3],color='white',linestyle='--',linewidth=0.8)
    ax.axvspan(0,10,facecolor='blue',alpha=0.15)
    ax.axvspan(110,120,facecolor='red',alpha=0.15)
    ax.set_xlim(0,120)
    ax.set_ylim(0,53.3)
    ax.axis('off')

def role_color(role):
    if pd.isna(role): return '#D3D3D3'
    r = str(role).lower()
    if 'target' in r: return '#00FF7F'
    if 'passer' in r: return '#FF8C00'
    if 'defense' in r: return '#1E90FF'
    if 'offense' in r: return '#FF3333'
    if 'ball' in r: return '#FFD700'
    return '#D3D3D3'

def team_marker(side):
    return 'o' if side == 'home' else 's'

for play_idx, (game_id, play_id) in enumerate(sample_plays, 1):
    data = df[(df['game_id']==game_id) & (df['play_id']==play_id)].copy()
    frame_ids = sorted(data['frame_id'].unique())
    fig, ax = plt.subplots(figsize=(12,6))
    draw_field(ax)
    trail_length = 5
    player_trails = {}

    def init():
        draw_field(ax)
        return []

    def update(frame_id):
        ax.clear()
        draw_field(ax)
        frame = data[data['frame_id']==frame_id]
        for _, row in frame.iterrows():
            pid = row['nfl_id']
            player_trails.setdefault(pid, []).append((row['x'], row['y']))
            if len(player_trails[pid]) > trail_length:
                player_trails[pid].pop(0)
            trail = np.array(player_trails[pid])
            ax.plot(trail[:,0], trail[:,1], color=role_color(row['player_role']), alpha=0.5, linewidth=2)
            ax.scatter(row['x'], row['y'], c=role_color(row['player_role']), s=120, edgecolor='black', marker=team_marker(row['player_side']), zorder=3)
            dx = np.cos(np.deg2rad(row['dir'])) * (row['s'] / 3)
            dy = np.sin(np.deg2rad(row['dir'])) * (row['s'] / 3)
            ax.arrow(row['x'], row['y'], dx, dy, color='white', head_width=0.5, alpha=0.6, zorder=2)
            if "target" in str(row['player_role']).lower() or "passer" in str(row['player_role']).lower():
                ax.text(row['x'], row['y']+1.2, row['player_name'].split()[0], color='white', fontsize=7, ha='center', weight='bold')
        if not pd.isna(frame.iloc[0]['ball_land_x']):
            ax.scatter(frame.iloc[0]['ball_land_x'], frame.iloc[0]['ball_land_y'], color='#FFD700', s=300, marker='*', edgecolor='black', linewidth=1.2, zorder=4)
        ax.set_title(f"Game {game_id} | Play {play_id} | Frame {frame_id}", color='white', fontsize=12)
        return []

    anim = FuncAnimation(fig, update, frames=frame_ids, init_func=init, blit=False, interval=120)
    gif_path = f"{save_dir}/game{game_id}_play{play_id}_enhanced.gif"
    anim.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    display(HTML(anim.to_jshtml()))


base_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
save_dir = "animated_plays"
os.makedirs(save_dir, exist_ok=True)

week_file = random.choice(sorted([f for f in os.listdir(base_path) if f.startswith("input_")]))
df = pd.read_csv(f"{base_path}/{week_file}", low_memory=False)

plays = df.groupby(['game_id','play_id']).size().sample(3, random_state=42).index.tolist()
field_color = '#123524'

def draw_field(ax):
    ax.set_facecolor(field_color)
    ax.plot([0,120],[0,0],color='white')
    ax.plot([0,120],[53.3,53.3],color='white')
    for x in range(10,120,10):
        ax.plot([x,x],[0,53.3],color='white',linestyle='--',linewidth=0.8)
    ax.axvspan(0,10,facecolor='#1B4F72',alpha=0.15)
    ax.axvspan(110,120,facecolor='#78281F',alpha=0.15)
    ax.set_xlim(0,120)
    ax.set_ylim(0,53.3)
    ax.axis('off')

def role_color(role):
    if pd.isna(role): return '#95A5A6'
    if 'target' in str(role).lower(): return '#00FF00'
    if 'passer' in str(role).lower(): return '#FFA500'
    if 'defense' in str(role).lower(): return '#3498DB'
    if 'offense' in str(role).lower(): return '#E74C3C'
    if 'ball' in str(role).lower(): return '#F1C40F'
    return 'white'

def team_marker(side):
    return 'o' if side == 'home' else 's'

trail_length = 5
frames_all = []

for play in plays:
    data = df[(df['game_id']==play[0]) & (df['play_id']==play[1])].copy()
    frame_ids = sorted(data['frame_id'].unique())
    player_trails = {}
    for i, frame_id in enumerate(frame_ids):
        frame = data[data['frame_id']==frame_id]
        fig, ax = plt.subplots(figsize=(16,8))
        draw_field(ax)
        for _, row in frame.iterrows():
            pid = row['nfl_id']
            player_trails.setdefault(pid, []).append((row['x'], row['y']))
            if len(player_trails[pid]) > trail_length:
                player_trails[pid].pop(0)
            trail = np.array(player_trails[pid])
            ax.plot(trail[:,0], trail[:,1], color=role_color(row['player_role']), alpha=0.4, linewidth=2)
            ax.scatter(row['x'], row['y'], c=role_color(row['player_role']), s=150, edgecolor='black', marker=team_marker(row['player_side']), zorder=3)
            dx = np.cos(np.deg2rad(row['dir'])) * (row['s'] / 3)
            dy = np.sin(np.deg2rad(row['dir'])) * (row['s'] / 3)
            ax.arrow(row['x'], row['y'], dx, dy, color='white', head_width=0.6, alpha=0.6, zorder=2)
            if "target" in str(row['player_role']).lower() or "passer" in str(row['player_role']).lower():
                ax.text(row['x'], row['y']+1.5, row['player_name'].split()[0], color='white', fontsize=7, ha='center', weight='bold')
        if not pd.isna(frame.iloc[0]['ball_land_x']):
            ax.scatter(frame.iloc[0]['ball_land_x'], frame.iloc[0]['ball_land_y'], color='yellow', s=300, marker='*', edgecolor='black', linewidth=1.5, zorder=4)
        title = f"Game {play[0]} | Play {play[1]} | Frame {frame_id}"
        ax.set_title(title, fontsize=14, color='white', pad=10)
        plt.tight_layout()
        frame_path = f"{save_dir}/frame_{play[0]}_{play[1]}_{i:04d}.png"
        plt.savefig(frame_path, dpi=120, bbox_inches='tight', facecolor=field_color)
        plt.close()
        frames_all.append(frame_path)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(frame_ids)} frames for play {play[1]}")

frames = [Image.open(fp) for fp in frames_all if os.path.exists(fp)]
frames[0].save('nfl_multi_play_animation.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)
print("Animated visualization created: nfl_multi_play_animation.gif")


frames = [Image.open(fp) for fp in frames_all if os.path.exists(fp)]
frames[0].save('nfl_multi_play_animation.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)

print("Animated visualization created: nfl_multi_play_animation.gif")
display(IPImage(filename='nfl_multi_play_animation.gif'))


merged_path = '/kaggle/working/merged_postthrow_all_weeks.csv'
if not os.path.exists(merged_path):
    raise FileNotFoundError(merged_path + " not found. Please create merged_postthrow_all_weeks.csv before running this cell.")

df = pd.read_csv(merged_path, low_memory=False)
df = df.dropna(subset=['x','y'])
df['x'] = df['x'].astype(float)
df['y'] = df['y'].astype(float)

bins_x = 60
bins_y = 30
x_edges = np.linspace(0,120,bins_x+1)
y_edges = np.linspace(0,53.3,bins_y+1)

def density_counts(subdf):
    H, _, _ = np.histogram2d(subdf['y'], subdf['x'], bins=[y_edges, x_edges])
    H = H[::-1]
    return H

layers = {}
layers['all'] = density_counts(df)
off = df[df['player_side'].str.lower().str.contains('off', na=False)]
layers['offense'] = density_counts(off)
defn = df[df['player_side'].str.lower().str.contains('def', na=False)]
layers['defense'] = density_counts(defn)
targets = df[df['player_role'].str.lower().str.contains('target', na=False)]
layers['target'] = density_counts(targets)
passers = df[df['player_role'].str.lower().str.contains('passer', na=False)]
layers['passer'] = density_counts(passers)

os.makedirs('/kaggle/working/heatmaps', exist_ok=True)
for name, H in layers.items():
    plt.figure(figsize=(10,5))
    sns.heatmap(H, cmap='magma', cbar_kws={'label':'counts'}, xticklabels=False, yticklabels=False)
    plt.title(f'Player Density Heatmap — {name}')
    out_png = f'/kaggle/working/heatmaps/heatmap_{name}.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

z_all = layers['all']
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
y_centers_plot = y_centers[::-1]

fig = go.Figure()
for layer_name in ['all','offense','defense','target','passer']:
    z = layers[layer_name]
    fig.add_trace(go.Heatmap(
        z=z,
        x=x_centers,
        y=y_centers_plot,
        colorscale='Magma',
        colorbar=dict(title='counts'),
        visible=(layer_name=='all'),
        hovertemplate='x:%{x:.1f}<br>y:%{y:.1f}<br>count:%{z}<extra></extra>'
    ))

updatemenus = [
    dict(
        buttons=[
            dict(method='update', args=[{'visible':[name=='all' for name in ['all','offense','defense','target','passer']]},
                                        {'title':'Density: All Players'}], label='All'),
            dict(method='update', args=[{'visible':[name=='offense' for name in ['all','offense','defense','target','passer']]},
                                        {'title':'Density: Offense'}], label='Offense'),
            dict(method='update', args=[{'visible':[name=='defense' for name in ['all','offense','defense','target','passer']]},
                                        {'title':'Density: Defense'}], label='Defense'),
            dict(method='update', args=[{'visible':[name=='target' for name in ['all','offense','defense','target','passer']]},
                                        {'title':'Density: Targets'}], label='Targets'),
            dict(method='update', args=[{'visible':[name=='passer' for name in ['all','offense','defense','target','passer']]},
                                        {'title':'Density: Passers'}], label='Passers')
        ],
        direction='down',
        pad={'r':10,'t':10},
        showactive=True,
        x=0.02,
        xanchor='left',
        y=1.12,
        yanchor='top'
    )
]

fig.update_layout(
    title='Player Density Heatmap — All',
    updatemenus=updatemenus,
    autosize=True,
    xaxis=dict(title='Field Length (yards)'),
    yaxis=dict(title='Field Width (yards)'),
    margin=dict(l=20, r=20, t=80, b=20)
)

display(HTML(fig.to_html(full_html=False)))
print("Static PNG heatmaps saved to /kaggle/working/heatmaps/")


merged_path = '/kaggle/working/merged_postthrow_all_weeks.csv'
if not os.path.exists(merged_path):
    raise FileNotFoundError(merged_path + " not found. Please create merged_postthrow_all_weeks.csv before running this cell.")

df = pd.read_csv(merged_path, low_memory=False)
df = df.dropna(subset=['x','y'])
df['x'] = df['x'].astype(float)
df['y'] = df['y'].astype(float)

bins_x, bins_y = 60, 30
x_edges = np.linspace(0,120,bins_x+1)
y_edges = np.linspace(0,53.3,bins_y+1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
y_centers_plot = y_centers[::-1]

def density(subdf):
    H, _, _ = np.histogram2d(subdf['y'], subdf['x'], bins=[y_edges, x_edges])
    H = gaussian_filter(H[::-1], sigma=1)
    return H

layers = {
    'All Players': density(df),
    'Offense': density(df[df['player_side'].str.lower().str.contains('off', na=False)]),
    'Defense': density(df[df['player_side'].str.lower().str.contains('def', na=False)]),
    'Passers': density(df[df['player_role'].str.lower().str.contains('passer', na=False)]),
    'Targets': density(df[df['player_role'].str.lower().str.contains('target', na=False)])
}

save_dir = '/kaggle/working/heatmaps_enhanced'
os.makedirs(save_dir, exist_ok=True)

for name, H in layers.items():
    plt.figure(figsize=(10,5))
    sns.heatmap(H, cmap='viridis', cbar_kws={'label':'Density'}, xticklabels=False, yticklabels=False)
    plt.title(f'{name} Spatial Density')
    plt.xlabel('Field Length (yards)')
    plt.ylabel('Field Width (yards)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{name.lower().replace(" ","_")}_heatmap.png', dpi=150)
    plt.close()

traces = []
for idx, (name, H) in enumerate(layers.items()):
    visible = True if idx == 0 else False
    traces.append(go.Heatmap(
        z=H,
        x=x_centers,
        y=y_centers_plot,
        colorscale='Viridis',
        colorbar=dict(title='Density'),
        hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Density: %{z:.1f}<extra></extra>',
        visible=visible
    ))

buttons = []
for i, name in enumerate(layers.keys()):
    vis = [False] * len(layers)
    vis[i] = True
    buttons.append(dict(label=name,
                        method='update',
                        args=[{'visible': vis},
                              {'title': f'{name} Spatial Density'}]))

fig = go.Figure(data=traces)
fig.update_layout(
    title='Player Spatial Density Heatmap (Interactive)',
    xaxis_title='Field Length (yards)',
    yaxis_title='Field Width (yards)',
    updatemenus=[dict(buttons=buttons, direction='down', x=0.02, y=1.12)],
    margin=dict(l=20, r=20, t=70, b=20)
)

centroids = (
    df.groupby(['frame_id','player_side'])
      .agg({'x':'mean','y':'mean'})
      .reset_index()
      .pivot(index='frame_id', columns='player_side', values=['x','y'])
)
if 'offense' in str(centroids.columns):
    fig.add_trace(go.Scatter(
        x=centroids['x'].iloc[:,0],
        y=centroids['y'].iloc[:,0],
        mode='lines',
        line=dict(color='orange', width=3),
        name='Offense centroid trail'
    ))
if 'defense' in str(centroids.columns):
    fig.add_trace(go.Scatter(
        x=centroids['x'].iloc[:,1],
        y=centroids['y'].iloc[:,1],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Defense centroid trail'
    ))

display(HTML(fig.to_html(full_html=False)))
print(f"Interactive visualization displayed.\n Static heatmaps saved in: {save_dir}")


merged_path = '/kaggle/working/merged_postthrow_all_weeks.csv'
if not os.path.exists(merged_path):
    raise FileNotFoundError(merged_path + " not found. Please create merged_postthrow_all_weeks.csv before running this cell.")

df = pd.read_csv(merged_path, low_memory=False)
df = df.dropna(subset=['x','y'])
df['x'] = df['x'].astype(float)
df['y'] = df['y'].astype(float)

bins_x, bins_y = 60, 30
x_edges = np.linspace(0,120,bins_x+1)
y_edges = np.linspace(0,53.3,bins_y+1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
y_centers_plot = y_centers[::-1]

def density(subdf):
    H, _, _ = np.histogram2d(subdf['y'], subdf['x'], bins=[y_edges, x_edges])
    H = gaussian_filter(H[::-1], sigma=1)
    return H

layers = {
    'All Players': density(df),
    'Offense': density(df[df['player_side'].str.lower().str.contains('off', na=False)]),
    'Defense': density(df[df['player_side'].str.lower().str.contains('def', na=False)]),
    'Passers': density(df[df['player_role'].str.lower().str.contains('passer', na=False)]),
    'Targets': density(df[df['player_role'].str.lower().str.contains('target', na=False)])
}

save_dir = '/kaggle/working/heatmaps_enhanced'
os.makedirs(save_dir, exist_ok=True)

for name, H in layers.items():
    plt.figure(figsize=(10,5))
    sns.heatmap(H, cmap='viridis', cbar_kws={'label':'Density'}, xticklabels=False, yticklabels=False)
    plt.title(f'{name} Spatial Density')
    plt.xlabel('Field Length (yards)')
    plt.ylabel('Field Width (yards)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{name.lower().replace(" ","_")}_heatmap.png', dpi=150)
    plt.close()

traces = []
for idx, (name, H) in enumerate(layers.items()):
    visible = True if idx == 0 else False
    traces.append(go.Heatmap(
        z=H,
        x=x_centers,
        y=y_centers_plot,
        colorscale='Viridis',
        colorbar=dict(title='Density'),
        hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Density: %{z:.1f}<extra></extra>',
        visible=visible
    ))

buttons = []
for i, name in enumerate(layers.keys()):
    vis = [False] * len(layers)
    vis[i] = True
    buttons.append(dict(label=name,
                        method='update',
                        args=[{'visible': vis},
                              {'title': f'{name} Spatial Density'}]))

fig = go.Figure(data=traces)
fig.update_layout(
    title='Player Spatial Density Heatmap (Interactive)',
    xaxis_title='Field Length (yards)',
    yaxis_title='Field Width (yards)',
    updatemenus=[dict(buttons=buttons, direction='down', x=0.02, y=1.12)],
    margin=dict(l=20, r=20, t=70, b=20)
)

centroids = (
    df.groupby(['frame_id','player_side'])
      .agg({'x':'mean','y':'mean'})
      .reset_index()
      .pivot(index='frame_id', columns='player_side', values=['x','y'])
)
if 'offense' in str(centroids.columns):
    fig.add_trace(go.Scatter(
        x=centroids['x'].iloc[:,0],
        y=centroids['y'].iloc[:,0],
        mode='lines',
        line=dict(color='orange', width=3),
        name='Offense centroid trail'
    ))
if 'defense' in str(centroids.columns):
    fig.add_trace(go.Scatter(
        x=centroids['x'].iloc[:,1],
        y=centroids['y'].iloc[:,1],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Defense centroid trail'
    ))

display(HTML(fig.to_html(full_html=False)))
print(f"Interactive visualization displayed.\n Static heatmaps saved in: {save_dir}")

