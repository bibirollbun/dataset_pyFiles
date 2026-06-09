# Imports & global settings (no seaborn per competition UI constraints)
import os, glob, math, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 120)

# Sampling/compute knobs (tweak for full run on Kaggle)
DT = 0.10   # assumed 10 Hz frame step; only scales integrals
MAX_PLAYS = None  # e.g., 500 for a faster demo; None for all
USE_TTR = False   # use distance proxies by default (faster); set True for kinematic TTR

# Output folder for figures/tables (Kaggle Working)
OUT_DIR = Path('./artifacts')
OUT_DIR.mkdir(exist_ok=True, parents=True)
print('Artifacts will be saved to:', OUT_DIR.resolve())


from pathlib import Path
import glob

def discover_paths():
    base = Path('/kaggle/input/nfl-big-data-bowl-2026-analytics')
    supp_candidates = list(base.rglob('supplementary_data.csv'))
    if not supp_candidates:
        raise FileNotFoundError('supplementary_data.csv introuvable. Attache le dataset de la compétition à ce notebook.')
    supp = supp_candidates[0]
    root = supp.parent
    train_dir = root / 'train'
    inputs  = sorted(glob.glob(str(train_dir / 'input_2023_w*.csv')))
    outputs = sorted(glob.glob(str(train_dir / 'output_2023_w*.csv')))
    if not inputs or not outputs:
        raise FileNotFoundError(f'CSV input/output introuvables sous {train_dir}')
    print(f'Root: {root}')
    print(f'{len(inputs)} input files, {len(outputs)} output files')
    print('ex:', inputs[0], '|', outputs[0])
    return {'base': root, 'inputs': inputs, 'outputs': outputs, 'supp': supp}

paths = discover_paths()
paths



# Light-weight schema to reduce memory usage when reading
usecols_in = [
    'game_id','play_id','nfl_id','frame_id','player_to_predict',
    'player_role','player_side','player_name','player_position',
    'x','y','s','a','o','dir','num_frames_output','ball_land_x','ball_land_y',
    'play_direction'
]
usecols_out = ['game_id','play_id','nfl_id','frame_id','x','y']

def read_inputs(input_paths, usecols=usecols_in, max_rows=None):
    dfs = []
    for p in input_paths:
        df = pd.read_csv(p, usecols=usecols, nrows=max_rows)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def read_outputs(output_paths, usecols=usecols_out, max_rows=None):
    dfs = []
    for p in output_paths:
        df = pd.read_csv(p, usecols=usecols, nrows=max_rows)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df_in  = read_inputs(paths['inputs'])
df_out = read_outputs(paths['outputs'])
supp   = pd.read_csv(paths['supp'])

print(df_in.shape, df_out.shape, supp.shape)
df_in.head()


# === Section: timeline & rôles (drop-in fix) ===

# 1) Throw frame par play (dernier frame côté input)
throw_idx = (
    df_in
    .groupby(['game_id', 'play_id'])['frame_id']
    .max()
    .rename('throw_frame')
    .reset_index()
)

# 2) Rôles au moment du lancer (avec noms & positions)
roles_at_throw = (
    df_in
    .merge(throw_idx, on=['game_id', 'play_id'])
    .query('frame_id == throw_frame')
    [['game_id', 'play_id', 'nfl_id', 'player_role', 'player_side',
      'player_name', 'player_position']]
)

# 3) Point d’atterrissage de la balle (par play)
land = (
    df_in
    .dropna(subset=['ball_land_x', 'ball_land_y'])
    .groupby(['game_id', 'play_id'], as_index=False)[['ball_land_x', 'ball_land_y']]
    .first()
)

# 4) Plays valides (pas annulés) + résultats de passe
supp_valid = supp[
    (supp['play_nullified_by_penalty'] == 'N') &
    (supp['pass_result'].isin(['C', 'I', 'IN']))
].copy()

# 5) Assemblage côté output (frames post-lancer)
out = (
    df_out
    .merge(
        supp_valid[['game_id', 'play_id', 'pass_result', 'pass_length',
                    'team_coverage_man_zone', 'team_coverage_type']],
        on=['game_id', 'play_id'], how='inner'
    )
    .merge(roles_at_throw, on=['game_id', 'play_id', 'nfl_id'], how='left')
    .merge(land, on=['game_id', 'play_id'], how='left')
)

# (optionnel) Sous-échantillonnage pour itérer plus vite
if MAX_PLAYS:
    keep = (
        out[['game_id', 'play_id']]
        .drop_duplicates()
        .head(int(MAX_PLAYS))
    )
    out = out.merge(keep, on=['game_id', 'play_id'], how='inner')

print(out.shape)
out.head()



# --- TTR (optional): simple turn+go model in yards ---
V_MAX = 10.5  # yd/s ~9.6 m/s
A_PLUS = 4.4  # yd/s^2 ~4.0 m/s^2
W_MAX = math.radians(240)  # rad/s

def _wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def ttr_turn_go(x, y, s, dir_deg, target_xy, dt=0.05):
    theta = math.radians(dir_deg if not np.isnan(dir_deg) else 0.0)
    tx, ty = target_xy
    vecx, vecy = tx - x, ty - y
    bearing = math.atan2(vecy, vecx)
    t_turn = abs(_wrap(bearing - theta))/W_MAX
    # integrate straight run with accel limit
    p = np.array([x, y], dtype=float)
    v = float(s if not np.isnan(s) else 0.0)
    t_run = 0.0
    while np.linalg.norm([tx - p[0], ty - p[1]]) > 0.15:
        v = min(v + A_PLUS*dt, V_MAX)
        p = p + v*np.array([math.cos(bearing), math.sin(bearing)])*dt
        t_run += dt
        if t_run > 6.0:
            break
    return t_turn + t_run

# Vectorized helpers for per-frame computations
def frame_metrics_distance(frame_df):
    Lx = frame_df['ball_land_x'].iloc[0]
    Ly = frame_df['ball_land_y'].iloc[0]
    wr = frame_df[frame_df['player_role']=='Targeted Receiver']
    if wr.empty:
        return None
    wrx, wry = wr['x'].iloc[0], wr['y'].iloc[0]
    wr_dL = math.hypot(wrx - Lx, wry - Ly)
    defenders = frame_df[frame_df['player_side']=='Defense']
    if defenders.empty:
        return {'AA': np.nan, 'wr_dL': wr_dL}
    dmin = (defenders[['x','y']].to_numpy() - np.array([[Lx, Ly]])).astype(float)
    dmin = np.sqrt((dmin**2).sum(axis=1)).min()
    return {'AA': dmin - wr_dL, 'wr_dL': wr_dL}

def frame_metrics_ttr(frame_df):
    Lx = frame_df['ball_land_x'].iloc[0]
    Ly = frame_df['ball_land_y'].iloc[0]
    wr = frame_df[frame_df['player_role']=='Targeted Receiver']
    if wr.empty:
        return None
    wr_row = wr.iloc[0]
    t_wr = ttr_turn_go(wr_row['x'], wr_row['y'], wr_row.get('s', 0.0), wr_row.get('dir', 0.0), (Lx, Ly))
    defenders = frame_df[frame_df['player_side']=='Defense']
    if defenders.empty:
        return {'AA': np.nan, 'wr_dL': np.nan}
    t_def_min = float('inf')
    for _, d in defenders.iterrows():
        t_def = ttr_turn_go(d['x'], d['y'], d.get('s', 0.0), d.get('dir', 0.0), (Lx, Ly))
        if t_def < t_def_min:
            t_def_min = t_def
    return {'AA': t_def_min - t_wr, 'wr_dL': np.nan}

def summarize_play(play_df, use_ttr=USE_TTR):
    frames = sorted(play_df['frame_id'].unique())
    if len(frames) == 0:
        return None
    metrics = []
    for fr in frames:
        s = play_df[play_df['frame_id']==fr]
        m = frame_metrics_ttr(s) if use_ttr else frame_metrics_distance(s)
        if m is None:
            continue
        metrics.append((fr, m['AA'], m.get('wr_dL', np.nan)))
    if not metrics:
        return None
    arr = np.array(metrics)
    AA_series = arr[:,1]
    wr_dL_series = arr[:,2]
    lead_rate = float(np.nanmean(AA_series > 0))
    AA_arrival = float(AA_series[-1])
    AA_integrated = float(np.nansum(AA_series) * DT)
    # CWC proxy using WR distance decay (for distance mode only)
    if not use_ttr:
        d0 = float(wr_dL_series[0]) if not np.isnan(wr_dL_series[0]) else np.nan
        dend = float(wr_dL_series[-1]) if not np.isnan(wr_dL_series[-1]) else np.nan
        if d0 and d0 > 1e-6 and not np.isnan(dend):
            CWC_norm_dist = float(1.0 - (dend / d0))
            CWC_rate = float((d0 - dend) / (len(frames) * DT))
        else:
            CWC_norm_dist, CWC_rate = np.nan, np.nan
    else:
        CWC_norm_dist, CWC_rate = np.nan, np.nan
    return {
        'lead_rate': lead_rate,
        'AA_arrival': AA_arrival,
        'AA_integrated': AA_integrated,
        'CWC_norm_dist': CWC_norm_dist,
        'CWC_rate': CWC_rate
    }


rows = []
for (g,p), gp in out.groupby(['game_id','play_id']):
    s = summarize_play(gp.sort_values('frame_id'), use_ttr=USE_TTR)
    if s is None:
        continue
    s.update({'game_id': g, 'play_id': p, 'catch': 1 if gp['pass_result'].iloc[0]=='C' else 0})
    s.update({'pass_length': gp['pass_length'].iloc[0],
              'team_coverage_man_zone': gp['team_coverage_man_zone'].iloc[0],
              'team_coverage_type': gp['team_coverage_type'].iloc[0]})
    rows.append(s)

feat = pd.DataFrame(rows)
print(feat.shape)
feat.head()


def cv_metrics(X, y, C=1.0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, briers, logs = [], [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=500, C=C, solver='lbfgs')
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], p))
        briers.append(brier_score_loss(y[te], p))
        logs.append(log_loss(y[te], p, eps=1e-6))
    return {'AUC': float(np.mean(aucs)), 'Brier': float(np.mean(briers)), 'LogLoss': float(np.mean(logs))}

y = feat['catch'].astype(int).to_numpy()
X_base = pd.DataFrame({
    'intercept': 1.0,
    'pass_length': feat['pass_length'].fillna(0.0)
}).to_numpy()

X_full = pd.DataFrame({
    'intercept': 1.0,
    'pass_length': feat['pass_length'].fillna(0.0),
    'lead_rate': feat['lead_rate'].fillna(0.0),
    'AA_arrival': feat['AA_arrival'].fillna(0.0),
    'AA_integrated': feat['AA_integrated'].fillna(0.0),
    'CWC_norm_dist': feat['CWC_norm_dist'].fillna(0.0),
    'CWC_rate': feat['CWC_rate'].fillna(0.0)
}).to_numpy()

m_base = cv_metrics(X_base, y)
m_full = cv_metrics(X_full, y)
print('=== SCORECARD (5-fold OOF) ===')
print(f"Baseline (air yards):     AUC={m_base['AUC']:.3f}  Brier={m_base['Brier']:.3f}  LogLoss={m_base['LogLoss']:.3f}")
print(f"+ AA/CWC features:        AUC={m_full['AUC']:.3f}  Brier={m_full['Brier']:.3f}  LogLoss={m_full['LogLoss']:.3f}")
print(f"ΔAUC = {m_full['AUC']-m_base['AUC']:.3f}   ΔBrier = {m_base['Brier']-m_full['Brier']:.3f}")

# Save to artifacts
with open(OUT_DIR/'scorecard.json','w') as f:
    json.dump({'baseline': m_base, 'full': m_full}, f, indent=2)
print('Saved:', OUT_DIR/'scorecard.json')


# Build a per-frame AA timeline for one example play
example_key = feat[['game_id','play_id']].iloc[0].to_dict() if len(feat)>0 else None
timeline = None
if example_key:
    g,p = example_key['game_id'], example_key['play_id']
    gp = out[(out.game_id==g)&(out.play_id==p)].sort_values('frame_id')
    frames, AAs, WRd = [], [], []
    for fr in sorted(gp['frame_id'].unique()):
        s = gp[gp['frame_id']==fr]
        m = frame_metrics_ttr(s) if USE_TTR else frame_metrics_distance(s)
        if m is None:
            continue
        frames.append(fr)
        AAs.append(m['AA'])
        WRd.append(m.get('wr_dL', np.nan))
    timeline = pd.DataFrame({'frame': frames, 'AA': AAs, 'wr_dL': WRd})
    # Plot AA timeline
    plt.figure(figsize=(8,4))
    plt.plot(timeline['frame'], timeline['AA'])
    plt.axhline(0, linestyle='--')
    plt.xlabel('Frame since throw (post-throw index)')
    plt.ylabel('Arrival Advantage (proxy)')
    plt.title(f'AA Timeline — game {g}, play {p}')
    fig_path = OUT_DIR/'aa_timeline_example.png'
    plt.tight_layout(); plt.savefig(fig_path, dpi=180); plt.close()
    print('Saved figure:', fig_path)

# Coverage table
tab = (feat
       .assign(bucket=lambda d: pd.cut(d['pass_length'], bins=[-50,-1,9,19,99], labels=['Behind LOS','0-9','10-19','20+']))
       .groupby(['team_coverage_man_zone','bucket'])
       .agg(lead_rate=('lead_rate','mean'),
            AA_arrival=('AA_arrival','mean'),
            CWC_norm_dist=('CWC_norm_dist','mean'),
            n=('game_id','count'))
       .reset_index()
      )
tab_path = OUT_DIR/'coverage_split_table.csv'
tab.to_csv(tab_path, index=False)
tab.head()


# Map targeted receiver & defenders per play for summary (uses roles_at_throw)
wr_ids = (roles_at_throw[roles_at_throw['player_role']=='Targeted Receiver']
          [['game_id','play_id','nfl_id','player_name','player_position']]
          .rename(columns={'nfl_id':'wr_id'}))
defense_team = supp_valid[['game_id','play_id','defensive_team']]
feat_keyed = (feat.merge(wr_ids, on=['game_id','play_id'], how='left')
                   .merge(defense_team, on=['game_id','play_id'], how='left'))

# WR leaderboard (min plays filter)
wr_lb = (feat_keyed.dropna(subset=['wr_id'])
         .groupby(['wr_id','player_name','player_position'], dropna=False)
         .agg(n=('game_id','count'),
              lead_rate=('lead_rate','mean'),
              AA_arrival=('AA_arrival','mean'),
              CWC_norm_dist=('CWC_norm_dist','mean'))
         .query('n >= 10')
         .sort_values('lead_rate', ascending=False)
         .reset_index())
wr_lb_path = OUT_DIR/'leaderboard_wr_named.csv'
wr_lb.to_csv(wr_lb_path, index=False)

# Team (defense) leaderboard
team_lb = (feat_keyed.dropna(subset=['defensive_team'])
           .groupby('defensive_team')
           .agg(n=('game_id','count'),
                CWC_norm_dist=('CWC_norm_dist','mean'),
                AA_arrival=('AA_arrival','mean'))
           .query('n >= 50')
           .sort_values('CWC_norm_dist', ascending=False)
           .reset_index())
team_lb_path = OUT_DIR/'leaderboard_defense.csv'
team_lb.to_csv(team_lb_path, index=False)

wr_lb.head(), team_lb.head()



from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
X_full = pd.DataFrame({
    'intercept': 1.0,
    'pass_length': feat['pass_length'].fillna(0.0),
    'lead_rate': feat['lead_rate'].fillna(0.0),
    'AA_arrival': feat['AA_arrival'].fillna(0.0),
    'AA_integrated': feat['AA_integrated'].fillna(0.0),
    'CWC_norm_dist': feat['CWC_norm_dist'].fillna(0.0),
    'CWC_rate': feat['CWC_rate'].fillna(0.0)
}).to_numpy()
y = feat['catch'].astype(int).to_numpy()
clf = LogisticRegression(max_iter=500).fit(X_full, y)
p = clf.predict_proba(X_full)[:,1]
frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'--')
plt.plot(mean_pred, frac_pos, marker='o')
plt.xlabel('Predicted catch prob')
plt.ylabel('Observed frequency')
plt.title('Calibration (AA/CWC model)')
plt.tight_layout(); plt.savefig(OUT_DIR/'calibration_curve.png', dpi=180); plt.close()
print('Saved:', OUT_DIR/'calibration_curve.png')


