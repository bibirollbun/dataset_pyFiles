import json # Accquire data
import os # Manipulate files
import pandas as pd # Data
# Plot stuff
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import networkx as nx # Graph analysis
import numpy as np # Numeric manipulation
import scipy as sp # Numeric manipulation
import time # Time stuff
import warnings # Goodbye warnings

# Additional Plotting stuff
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from PIL import Image


# Pandas and warning stuff
pd.set_option('display.max_columns', 500)
warnings.filterwarnings('ignore')

# Plot configuration
font_path = "/kaggle/input/font-for-report/Helotypo.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Helotypo'


# Colormap configuration
original_cmap = cm.get_cmap('cool', 256)
colors = original_cmap(np.linspace(0,1, 256))
neg_colors = colors[:128]
pos_colors = colors[128:]

neg_alpha = np.linspace(1, 0, 128)
pos_alpha = np.linspace(0, 1 , 128)
new_colors = np.vstack([neg_colors, pos_colors])

new_colors[:128, 3] = neg_alpha
new_colors[128:, 3] = pos_alpha

custom_cmap = LinearSegmentedColormap.from_list('tr_cool', new_colors)


# Makes the selection of the game or play random.
RANDOM = False


raw_data_path = '/kaggle/input/'

# We are just using one path for specific simulation.
data_path = 'nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w18.csv'
supplementary_data_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"

complete_data_path = os.path.join(raw_data_path, data_path)
ref_datapaths = os.path.join(raw_data_path, 'nfl-big-data-bowl-2023/')
ref_relevant_key = 'week'
ref_players_file = 'players.csv'

icons = {
    "Offense": "/kaggle/input/nfl-figures-and-icons/attacking_player.png",
    "Defense": "/kaggle/input/nfl-figures-and-icons/defensive_player.png",
    'ball': '/kaggle/input/nfl-figures-and-icons/nfl_ball.png',
}

nfl_image = '/kaggle/input/nfl-logos/NFL-logo.png'


def transform_height(height):
    """Get height as feet"""
    feet, inches = height.split('-')
    numerical_feet = int(feet) + int(inches) * 0.0833333
    return numerical_feet


def _height_to_inches(h):
    """Helper function to convert height into a float value"""
    if pd.isna(h): return np.nan
    if isinstance(h, (int, float)): return float(h)
    s = str(h)
    if "-" in s:
        ft, inch = s.split("-", 1)
        try: return 12*int(ft) + int(inch)
        except: pass
    try: return float(s)
    except: return np.nan


def add_team_cohesion(frame_df, radius=12.0):
    """Adds column 'team_cohesion' per frame: proximity * alignment (dir & orientation)."""
    angle_const = 0
    df = frame_df.copy()
    X = df['x'].to_numpy()
    Y = df['y'].to_numpy()
    side = df['player_side_numerical'].to_numpy()
    th = np.deg2rad((df['dir'] + angle_const).to_numpy())
    ori = np.deg2rad((df['o'] + angle_const).to_numpy())
    n = len(df)
    cohesion = np.zeros(n)
    for i in range(n):
        same = np.where((side == side[i]) & (np.arange(n) != i))[0]
        if same.size == 0:
            continue
        dx = X[same] - X[i]; dy = Y[same] - Y[i]
        dist = np.hypot(dx, dy)
        m = dist < radius
        if not np.any(m): continue
        dx = dx[m]; dy = dy[m]; dist = dist[m]
        jidx = same[m]
        alpha = np.arctan2(dy, dx)  # bearing i->j
        look_dir = np.cos(th[i] - alpha) * np.cos(th[jidx] - alpha)
        look_ori = np.cos(ori[i] - alpha) * np.cos(ori[jidx] - alpha)
        look = 0.5*(look_dir + look_ori)
        w = 1.0 / (dist + 1e-6)
        cohesion[i] = (w * np.maximum(0.0, look)).sum() / (w.sum() + 1e-6)
    df['team_cohesion'] = cohesion
    return df


def normalize_two_sided(D, eps=1e-12):
    """
    Piecewise normalization:
      negatives -> [-1, 0], positives -> [0, 1], zeros stay 0.
      D is a 2D numpy array (the aggregated dominance field).
    """
    # split into positive and negative parts
    pos = np.maximum(D, 0.0)
    neg = np.minimum(D, 0.0)

    pmax = float(pos.max()) if pos.size else 0.0
    nmin = float(neg.min()) if neg.size else 0.0 # <= 0

    # scale each side independently
    if pmax > eps:
        pos = pos / pmax
    else:
        pos = np.zeros_like(pos)

    if nmin < -eps: # nmin is negative
        neg = neg / (-nmin) # divide by abs(min) -> [-1, 0]
    else:
        neg = np.zeros_like(neg)

    return pos + neg


def circle_mask(X, Y, cx, cy, R):
    """Boolean mask for points within radius R of (cx, cy)."""
    mask = (X - cx)**2 + (Y - cy)**2 <= R**2
    return mask
    
    
def _front_back_masks(X, Y, cx, cy, theta_rad, sector_deg=120):
    """
    Wedge in front/back of player's facing direction.
    sector_deg is the full apex angle (e.g., 120° -> ±60° half-angle).
    """
    ux, uy = np.cos(theta_rad), np.sin(theta_rad)
    dx, dy = X - cx, Y - cy
    dist = np.hypot(dx, dy)
    dot = dx*ux + dy*uy
    cosang = np.zeros_like(dist)
    m = dist > 1e-6
    cosang[m] = dot[m] / dist[m]
    cthr = np.cos(np.deg2rad(sector_deg/2))
    front = cosang >= cthr
    back  = cosang <= -cthr
    return front, back

    
def player_local_stats(row, D, X, Y, Gx=None, Gy=None, R=5.0, sector_deg=120):
    """
    row: Series with x, y, dir, player_side_numerical
    D: 2D total dominance field (ideally already normalized to [-1,1] two-sided)
    X,Y: meshgrid
    Gx,Gy: gradients of D along x,y (optional; pass to speed up)
    """
    angle_const = 0
    cx, cy = float(row['x']), float(row['y'])
    mask_c = circle_mask(X, Y, cx, cy, R)
    vals = D[mask_c]
    if vals.size == 0:
        return dict(mean_val=np.nan, std_val=np.nan, median=np.nan,
                    p10=np.nan, p90=np.nan, share_team=np.nan,
                    front_mean=np.nan, back_mean=np.nan, grad_outward=np.nan)

    sign_team = 1.0 if row['player_side_numerical'] > 0 else -1.0

    # basic distribution
    mean_val = float(vals.mean())
    std_val = float(vals.std())
    median = float(np.median(vals))
    p10, p90 = np.quantile(vals, [0.1, 0.9])

    # territorial share (relative to player side)
    share_team  = float((sign_team * vals > 0).mean())
    share_team_scaled = share_team * abs(mean_val)

    # front vs back
    th = np.deg2rad(row['dir'] + angle_const)
    front_m, back_m = _front_back_masks(X, Y, cx, cy, th, sector_deg=sector_deg)
    fm = mask_c & front_m
    bm = mask_c & back_m
    front_mean = float(D[fm].mean()) if np.any(fm) else np.nan
    back_mean  = float(D[bm].mean()) if np.any(bm) else np.nan

    # outward radial gradient (positive = dominance growing as we move away)
    if Gx is None or Gy is None:
        # approximate grid spacings from X,Y if uniform
        dx = float(X[0,1] - X[0,0])
        dy = float(Y[1,0] - Y[0,0])
        Gx, Gy = np.gradient(D, dx, dy, edge_order=1)
    dx = X - cx
    dy = Y - cy
    r = np.hypot(dx, dy)
    rh = np.zeros_like(r)
    m = (mask_c) & (r > 1e-6)
    rh[m] = (Gx[m]*dx[m] + Gy[m]*dy[m]) / r[m]
    grad_outward = float(rh[m].mean()) if np.any(m) else np.nan

    dict_result = dict(
        mean_val=mean_val, 
        std_val=std_val,
        median=median,
        p10=p10, 
        p90=p90, 
        share_team=share_team, 
        share_team_scaled=share_team_scaled,
        front_mean=front_mean, 
        back_mean=back_mean,
        grad_outward=grad_outward
    )

    return dict_result


def collect_frame_player_stats(frame_df, D, X, Y, R=5.0, sector_deg=120):
    """
    frame_df: DataFrame of the frame (one row per player)
    D,X,Y: dominance field and its grid
    Returns a DataFrame with per-player stats for this frame.
    """
    # precompute gradient once per frame for speed
    dx = float(X[0,1] - X[0,0])
    dy = float(Y[1,0] - Y[0,0])
    Gx, Gy = np.gradient(D, dx, dy, edge_order=1)

    out = []
    for _, row in frame_df.iterrows():
        stats = player_local_stats(row, D, X, Y, Gx=Gx, Gy=Gy, R=R, sector_deg=sector_deg)
        stats.update({
            'game_id': row.get('game_id'),
            'play_id': row.get('play_id'),
            'frame_id': row.get('frame_id'),
            'nfl_id': row.get('nfl_id'),
            'player_name': row.get('player_name'),
            'team_sign': row.get('player_side_numerical'),
            'player_position': row.get('player_position'),
            'player_role': row.get('player_role'),
        })
        out.append(stats)
    return pd.DataFrame(out)

def get_closest_players(frame_dataset):
    """Given a dataset representing a frame, get the closest player for each player in the df."""
    closest_player_distances = pd.DataFrame(sp.spatial.distance_matrix(
          frame_dataset[['x', 'y']],
          frame_dataset[['x', 'y']]
    ), index=frame_dataset.nfl_id, columns=frame_dataset.nfl_id).replace(0, 100)
    
    closest_player = closest_player_distances.apply(
        lambda x: x.idxmin()
    ).to_dict()

    closest_distances = closest_player_distances.apply(lambda x: x.min()).to_dict()
    
    frame_dataset['closest_player'] = frame_dataset['nfl_id'].map(closest_player)
    frame_dataset['closest_distances'] = frame_dataset['nfl_id'].map(closest_distances)
    return frame_dataset

def community_generator(game_play):
    """Function to generate a community from a specific game and play"""
    closest_player_distances = pd.DataFrame(sp.spatial.distance_matrix(
            game_play[['x', 'y']],
            game_play[['x', 'y']]
        ), index=game_play.nfl_id, columns=game_play.nfl_id).replace(0, 100)
    reciprocal_closest_player_distances = 1/closest_player_distances
    reciprocal_threshold = pd.Series(reciprocal_closest_player_distances.to_numpy().flatten()).quantile(0.8)
    reciprocal_mask = reciprocal_closest_player_distances > reciprocal_threshold
    adj_closest_player_matrix = (reciprocal_closest_player_distances)*(reciprocal_mask)
    closest_player_g = nx.from_pandas_adjacency(adj_closest_player_matrix)
    player_communities = nx.community.louvain_communities(closest_player_g, weight='weight', seed=1)
    return player_communities


def dominance_function(row,
                       t_h=0.5,
                       r0=3.0,
                       s_ref=10.93,   # yd/s  (~22.36 mph)
                       a_ref=7.0,     # yd/s^2 (tunable)
                       h_ref=74.0,    # inches (~6'2")
                       w_ref=220.0,   # lb
                       betas=(0.8, 0.4, 0.3, 0.6),  # beta_s, beta_a, beta_h, gamma_s
                       alphas=(0.4, 0.3, 0.2, 0.1), # ke, momentum, force, action
                       lambda_cohesion=0.5):
    """
    Returns f(X, Y) -> dominance for this player over the whole grid (vectorized).
    Expects row to have: x,y,s,a,dir,o,player_weight,player_height,player_side_numerical[,team_cohesion]
    """
    beta_s, beta_a, beta_h, gamma_s = betas
    a_ke, a_p, a_f, a_act = alphas

    # Angles
    angle_const = 0
    theta = np.deg2rad(row['dir'] + angle_const)
    c, s = np.sin(theta), np.cos(theta)

    # Forward-looking mean (assume accel collinear with dir)
    x0 = row['x'] + t_h*row['s']*c + 0.5*(t_h**2)*row['a']*c
    y0 = row['y'] + t_h*row['s']*s + 0.5*(t_h**2)*row['a']*s

    # Height -> reach base
    h_in = _height_to_inches(row['player_height'])
    H = 0.0 if np.isnan(h_in) else (h_in - h_ref) / h_ref
    r_reach = r0 * (1.0 + 0.5*H)

    # Anisotropic spreads
    s_long = r_reach * (1.0 + beta_s*(row['s']/s_ref) + beta_a*(abs(row['a'])/a_ref))
    s_lat  = r_reach * (1.0 + beta_h*H) / (1.0 + gamma_s*(row['s']/s_ref))
    s_lat  = max(s_lat, 0.6*r_reach)

    a2, b2 = s_long**2, s_lat**2   # along, across (in rotated frame)

    # Mechanical amplitude (ref-normalized, unit-safe by consistency)
    w = row['player_weight']
    spd = row['s']
    acc = abs(row['a'])
    d = spd * t_h
    KE = 0.5*w*spd**2
    MOM  = w*spd
    FORC = w*acc
    ACT  = w*spd**2*t_h

    KE_ref   = 0.5*w_ref*s_ref**2
    MOM_ref  = w_ref*s_ref
    FORC_ref = w_ref*a_ref
    ACT_ref  = w_ref*s_ref**2*t_h

    M = (a_ke*(KE/KE_ref) +
         a_p *(MOM/MOM_ref) +
         a_f *(FORC/FORC_ref) +
         a_act*(ACT/ACT_ref))
    M = max(M, 1e-6)

    # Orientation-to-motion stability
    S = 0.5*(1.0 + np.cos(np.deg2rad(row['o'] - row['dir'])))

    # Teamwork (optional column; default 0)
    C = float(row['team_cohesion']) if 'team_cohesion' in row.index else 0.0
    A = M * S * (1.0 + lambda_cohesion*C)

    sign = row['player_side_numerical']

    # Return vectorized callable over grid X, Y
    def f(X, Y):
        dx = X - x0
        dy = Y - y0
        # rotate residuals into (along, across) frame — no explicit matrix inverse needed
        u = c*dx + s*dy  # along dir
        v = -s*dx + c*dy  # across dir
        quad = (u*u)/a2 + (v*v)/b2
        G = np.exp(-0.5*quad) / (2*np.pi*np.sqrt(a2*b2))
        return sign * A * G

    return f




def plot_field(subplots=1, figsize=(12*2, 5.33333*2)):
  '''Function to plot the football field. '''
  stadium_limit_dimension_y =  (0, 53.3333333)
  stadium_limit_dimension_x = (0, 120)
  color__green = "#586d3d"
  color__white = "#DADADA"
  color__brown = "#663831"
  color__yellow = "#cbb67c"
  text_position = 5
  portery_yards = 6.166667
  numbers = {20:'1 0', 30:'2 0', 40:'3 0', 50:'4 0', 60:'5 0', 70:'4 0', 80:'3 0', 90:'2 0', 100:'1 0'}

  fig, ax = plt.subplots(subplots, figsize=figsize)
  ax.set_xlim(stadium_limit_dimension_x)
  ax.set_ylim(stadium_limit_dimension_y)
  ax.vlines(
      [i for i in range(0, max(stadium_limit_dimension_x), 10)],
      min(stadium_limit_dimension_y),
      max(stadium_limit_dimension_y),
      color=color__white,
      alpha=0.5
  )

  ax.vlines(
      [i for i in range(10, max(stadium_limit_dimension_x)-10, 5)],
      min(stadium_limit_dimension_y),
      max(stadium_limit_dimension_y),
      color=color__white,
      alpha=0.3
  )

  vlines2 = [i for i in range(10, max(stadium_limit_dimension_x)-10+1, 1)]
  vlines2_size=1
  ax.vlines(
      vlines2,
      min(stadium_limit_dimension_y),
      min(stadium_limit_dimension_y) + vlines2_size,
      color=color__white,
      alpha=0.3
  )
  ax.vlines(
      vlines2,
      max(stadium_limit_dimension_y),
      max(stadium_limit_dimension_y) - vlines2_size,
      color=color__white,
      alpha=0.5
  )

  vlines3_size = 0.5
  medium_vlines_pos = (min(stadium_limit_dimension_y) + max(stadium_limit_dimension_y))/2 + portery_yards/2
  medium_vlines_neg = (min(stadium_limit_dimension_y) + max(stadium_limit_dimension_y))/2 - portery_yards/2
  ax.vlines(
      vlines2,
      medium_vlines_pos + vlines3_size,
      medium_vlines_pos - vlines3_size,
      color=color__white,
      alpha=0.3
  )
  ax.vlines(
      vlines2,
      medium_vlines_neg + vlines3_size,
      medium_vlines_neg - vlines3_size,
      color=color__white,
      alpha=0.3
  )
  for p, n in numbers.items():
    ax.text(p, min(stadium_limit_dimension_y) + text_position, n, horizontalalignment='center', verticalalignment='center', color=color__white, rotation=0)
    ax.text(p, max(stadium_limit_dimension_y) - text_position, n, horizontalalignment='center', verticalalignment='center', color=color__white, rotation=180)

  triangle_pos = [(i + (i-5))/2 for i in range(20, max(stadium_limit_dimension_x)//2-10+1, 10)]
  ax.scatter(triangle_pos, [min(stadium_limit_dimension_y) + text_position]*len(triangle_pos), marker='<', s=10, color=color__white)
  ax.scatter(triangle_pos, [max(stadium_limit_dimension_y) - text_position]*len(triangle_pos), marker='<', s=10, color=color__white)

  triangle_pos2 = [(i + (i+5))/2 for i in range(70, max(stadium_limit_dimension_x)-10, 10)]
  ax.scatter(triangle_pos2, [min(stadium_limit_dimension_y) + text_position]*len(triangle_pos2), marker='>', s=10, color=color__white)
  ax.scatter(triangle_pos2, [max(stadium_limit_dimension_y) - text_position]*len(triangle_pos2), marker='>', s=10, color=color__white)
  ax.set_facecolor(color__green)
  ax.set_xticks([])
  ax.set_yticks([])
  logosize = 7
  extent = (
      stadium_limit_dimension_x[1]//2-logosize,
      stadium_limit_dimension_x[1]//2+logosize,
      stadium_limit_dimension_y[1]//2-logosize,
      stadium_limit_dimension_y[1]//2+logosize
  )
  ax.imshow(plt.imread(nfl_image), aspect='equal', extent=extent, alpha=0.5)
  return fig, ax

def plot_field(ax=None, figsize=(24, 10.6667), nfl_image=None):
    """
    Plot an NFL field on the given axis (or create one if ax is None).

    Parameters
    ----------
    ax : matplotlib.axes.Axes or iterable[Axes], optional
        Axis (or collection of axes) to draw on. If None, a new figure/axis is created.
    figsize : tuple, optional
        Figure size used only if ax is None.
    nfl_image : str or array-like, optional
        Path to an image or already-loaded array to place at midfield (faint logo).

    Returns
    -------
    If a new axis is created: (fig, ax)
    If ax is provided: ax (same object), for convenience
    """
    # ---- constants ----
    stadium_limit_dimension_y = (0, 53.3333333)
    stadium_limit_dimension_x = (0, 120)
    
    color__green  = "#586d3d"
    color__white  = "#DADADA"
    
    text_position = 5
    portery_yards = 6.166667
    numbers = {20:'1 0', 30:'2 0', 40:'3 0', 50:'4 0', 60:'5 0', 70:'4 0', 80:'3 0', 90:'2 0', 100:'1 0'}

    def _draw(ax_):
        # limits
        ax_.set_xlim(stadium_limit_dimension_x)
        ax_.set_ylim(stadium_limit_dimension_y)

        # 10-yd major lines
        ax_.vlines(
            [i for i in range(0, int(max(stadium_limit_dimension_x))+1, 10)],
            min(stadium_limit_dimension_y),
            max(stadium_limit_dimension_y),
            color=color__white, alpha=0.5
        )

        # 5-yd minor lines (no endzones)
        ax_.vlines(
            [i for i in range(10, int(max(stadium_limit_dimension_x))-10, 5)],
            min(stadium_limit_dimension_y),
            max(stadium_limit_dimension_y),
            color=color__white, alpha=0.3
        )

        # small ticks on sidelines every yard
        vlines2 = [i for i in range(10, int(max(stadium_limit_dimension_x))-10+1, 1)]
        v2 = 1.0
        ax_.vlines(vlines2, min(stadium_limit_dimension_y),
                   min(stadium_limit_dimension_y)+v2, color=color__white, alpha=0.3)
        ax_.vlines(vlines2, max(stadium_limit_dimension_y),
                   max(stadium_limit_dimension_y)-v2, color=color__white, alpha=0.5)

        # hash marks (inside numbers)
        v3 = 0.5
        mid_y = (min(stadium_limit_dimension_y) + max(stadium_limit_dimension_y)) / 2
        hash_pos = mid_y + portery_yards/2
        hash_neg = mid_y - portery_yards/2
        ax_.vlines(vlines2, hash_pos + v3, hash_pos - v3, color=color__white, alpha=0.3)
        ax_.vlines(vlines2, hash_neg + v3, hash_neg - v3, color=color__white, alpha=0.3)

        # yard numbers
        for p, n in numbers.items():
            ax_.text(p, min(stadium_limit_dimension_y) + text_position, n,
                     ha='center', va='center', color=color__white, rotation=0)
            ax_.text(p, max(stadium_limit_dimension_y) - text_position, n,
                     ha='center', va='center', color=color__white, rotation=180)

        # little triangles next to numbers (directional)
        triangle_pos  = [(i + (i-5))/2 for i in range(20, int(max(stadium_limit_dimension_x)//2)-10+1, 10)]
        triangle_pos2 = [(i + (i+5))/2 for i in range(70, int(max(stadium_limit_dimension_x))-10, 10)]
        ax_.scatter(triangle_pos,  [min(stadium_limit_dimension_y) + text_position]*len(triangle_pos),
                    marker='<', s=10, color=color__white)
        ax_.scatter(triangle_pos,  [max(stadium_limit_dimension_y) - text_position]*len(triangle_pos),
                    marker='<', s=10, color=color__white)
        ax_.scatter(triangle_pos2, [min(stadium_limit_dimension_y) + text_position]*len(triangle_pos2),
                    marker='>', s=10, color=color__white)
        ax_.scatter(triangle_pos2, [max(stadium_limit_dimension_y) - text_position]*len(triangle_pos2),
                    marker='>', s=10, color=color__white)

        # aesthetics
        ax_.set_facecolor(color__green)
        ax_.patch.set_alpha(0.8)
        ax_.set_xticks([])
        ax_.set_yticks([])
        ax_.set_aspect('equal')

        # midfield logo (optional)
        if nfl_image is not None:
            if isinstance(nfl_image, str):
                img = plt.imread(nfl_image)
            else:
                img = nfl_image  # assume array
            logosize = 7
            extent = (
                stadium_limit_dimension_x[1]//2 - logosize,
                stadium_limit_dimension_x[1]//2 + logosize,
                stadium_limit_dimension_y[1]//2 - logosize,
                stadium_limit_dimension_y[1]//2 + logosize
            )
            ax_.imshow(img, aspect='equal', extent=extent, alpha=0.5)

        return ax_

    # ---- handle single / multiple axes or create new ----
    if ax is None:
        fig, ax_new = plt.subplots(1, 1, figsize=figsize)
        _draw(ax_new)
        return fig, ax_new

    # ax provided
    try:
        # If it's array-like of axes
        _ = iter(ax)
        for a in ax:
            _draw(a)
        return ax
    except TypeError:
        # Single axis
        return _draw(ax)

def create_gif(image_paths, output_gif_path, fps=5):
  images = [Image.open(image_path) for image_path in image_paths]
  images[0].save(
    output_gif_path,
    save_all=True,
    append_images=images[1:],
    fps=fps,
    loop=0 # 0 means infinite loop
  )

def save_gif(filename, fps=5):
    image_paths = [os.path.join(f'../experiments/figures/{filename}', p) for p in sorted(os.listdir(f'../experiments/figures/{filename}'))]
    #filename_split = filename.split('_')[-1]
    output_gif_path = f"../experiments/outputs/output__{filename}.gif"
    create_gif(image_paths, output_gif_path, fps=fps)

def create_folder(filename, path="../experiments/figures/"):
    complete_path = os.path.join(path, filename)
    if os.path.exists(complete_path):
        return None
    os.mkdir(complete_path)
    return complete_path

def save_figs(information, filename, show=False, save=False, stop_counter=None):
    complete_path = create_folder(filename=filename)
    counter=0
    for idx, row in information.iterrows():
        increasing_information = information.loc[:idx]
        time_component = pd.to_datetime(increasing_information.time, format='ISO8601')
        time_component = (time_component - time_component.min()).dt.total_seconds()

        fig, ax = plot_field(subplots=2, figsize=(15,10))
        
        draw_graph(row.graphs, ax=ax[0])
    
        ax[1].plot(
            increasing_information.time_component, 
            increasing_information.model_pred,
            color='blue', 
            marker='o',
            markerfacecolor='k',
            markersize=3,
            label='Model score',
        )
        increasing_information_events = increasing_information[
            increasing_information.event.notna()
        ]
    
        ax[1].vlines(
            increasing_information_events.time_component, 
            [0 - 0.0] * increasing_information_events.shape[0], 
            [1 - 0.05] * increasing_information_events.shape[0],
            color='k',
            alpha=0.8,
        )
    
        counter_text = 0
        for _, row_event in increasing_information_events.iterrows():
            counter_text +=1
            ax[1].text(
                row_event.time_component, 
                counter_text%2 - (-1)**counter_text*0.05,
                row_event.event, 
                size=5,  
                horizontalalignment='center', 
                verticalalignment='center',
            )
        ax[1].set_ylim(-0.15, 1.1)
    
        plt.suptitle('Play development over time.')
        ax[1].set_title('Overall play behavior')
        ax[1].set_title(f'Model prediction [game:{game_id}, play:{play_id}]')
        ax[1].set_xlabel('Seconds from ball snap')
        ax[1].set_ylabel('Probability of Pass Forward prediction \n(caught) at  the next 0.5 second.')
        ax[1].grid(linestyle='--')
    
        ax[1].legend()
        if show:
            plt.show()
        if save:
            plt.savefig(f'{complete_path}/test_{str(counter).zfill(3)}.png')
        plt.close(fig)
        counter += 1
        if stop_counter:
            if counter==stop_counter:
                break

def scatter_png(ax, df, x='x', y='y', side='player_side',
                icons=None, size=0.02, zorder=3):
    """
    Draw PNG icons at (x,y) positions.
    size: fraction of x-axis span used as icon width (keeps PNG aspect).
    """
    # preload images once
    imgs = {k: mpimg.imread(v) for k, v in icons.items()}

    # set nice limits if none exist
    if not ax.has_data():
        ax.set_xlim(df[x].min()-2, df[x].max()+2)
        ax.set_ylim(df[y].min()-2, df[y].max()+2)

    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xspan = x1 - x0
    yspan = y1 - y0

    for _, r in df.iterrows():
        img = imgs[r[side]]
        h, w = img.shape[:2]
        w_data = xspan * size
        # keep image aspect, correct for axis scaling
        h_data = w_data * (h / w) * (yspan / xspan)

        ax.imshow(
            img,
            extent=(r[x] - w_data/2, r[x] + w_data/2, r[y] - h_data/2, r[y] + h_data/2),
            origin="upper",
            zorder=zorder,
            interpolation="none"
        )

    return ax

def player_ids(ax, df, x='x', y='y', ids='nfl_id', id_func=None):        
    for _, r in df.iterrows():
        if id_func is None:
            text = r[ids]
        else:
            text = id_func(r[ids])
        ax.text(
            s= text,
            x=r[x] + r['player_side_numerical'],
            y=r[y] + r['player_side_numerical']*2,
            ha='center',
            va='center',
        )
    return ax

def plot_ball_land(ax, df, icons, size=0.02, zorder=3):
    """
    Draw PNG ball icons at the landing (x,y) position.
    size: fraction of x-axis span used as icon width (keeps PNG aspect).
    """
    # preload images once
    imgs = {k: mpimg.imread(v) for k, v in icons.items()}

    # set nice limits if none exist
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xspan = x1 - x0
    yspan = y1 - y0

    img = imgs['ball']
    h, w = img.shape[:2]
    w_data = xspan * size
    # keep image aspect, correct for axis scaling
    h_data = w_data * (h / w) * (yspan / xspan)
    ball_land_x = df.ball_land_x.unique()[0]
    ball_land_y = df.ball_land_x.unique()[0]

    ax.imshow(
        img,
        extent=(ball_land_x - w_data/4, ball_land_x + w_data/4, ball_land_y - h_data/2, ball_land_y + h_data/2),
            origin="upper",
            zorder=zorder,
            interpolation="none"
        )

    return ax

def plot_player_arrows(ax, test, dir_col, scale=1, color='k', rotate=False):
    angle_const = 0
    if rotate:
        angle_const = 180
    theta = np.deg2rad((test[dir_col] + angle_const).to_numpy())
    ax.quiver(
        test.x,
        test.y, 
        2*np.sin(theta), 
        2*np.cos(theta),
        color=color,
        scale_units='xy',
        scale=scale,
        headwidth=5,
        headlength=4,
        width=0.001,
        edgecolors='k',
        linewidths=0.5,
    )
    return ax

def auto_zoom(ax, X, Y, Z, frac_of_max=0.01, pad=0.10):
    """
    Zoom to where |Z| >= frac_of_max * max(|Z|).
    pad is a % of the detected width/height.
    """
    Zabs = np.abs(Z)
    m = Zabs >= (Zabs.max() * frac_of_max)
    if not np.any(m):
        return  # nothing above threshold

    xmin, xmax = X[m].min(), X[m].max()
    ymin, ymax = Y[m].min(), Y[m].max()
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - pad*dx, xmax + pad*dx)
    ax.set_ylim(ymin - pad*dy, ymax + pad*dy)
    


# Historical data for reference
ref_paths = [path for path in os.listdir(ref_datapaths) if ref_relevant_key in path]
ref_datasets = [pd.read_csv(os.path.join(ref_datapaths,path)) for path in ref_paths]
ref_tracking_data = pd.concat(ref_datasets)

# Players data contain additional information for data
ref_players_data = pd.read_csv(os.path.join(ref_datapaths, ref_players_file))

# Load data
data_test = pd.read_csv(complete_data_path)
supplementary_data = pd.read_csv(supplementary_data_path)


# Historical references
tracking_data_cols = ['nflId', 's', 'a']
players_data_cols = ['nflId', 'height', 'weight', 'officialPosition']
relevant_data = ref_tracking_data[
    ref_tracking_data.nflId.notnull()
][tracking_data_cols].merge(
    ref_players_data[players_data_cols],
    on='nflId',
    how='left'
)

relevant_data['numerical_height'] = relevant_data.height.apply(transform_height)

# This are the historical references
ref_metrics = relevant_data.drop('nflId', axis=1).groupby('officialPosition').max().T.to_dict()
#del relevant_data, ref_players_data, ref_tracking_data


# Random selection
if RANDOM:
    random_sample = data_test.groupby(['game_id', 'play_id']).frame_id.max()[
        data_test.groupby(['game_id', 'play_id']).frame_id.median() == data_test.groupby(['game_id', 'play_id']).frame_id.median()
    ].sample(1)
    game_id = random_sample.index[0][0]
    play_id = random_sample.index[0][1]

else:
    # Hardcoded selection
    game_id = 2024010711
    play_id = 55

# Game and play selection
specific_game = data_test[data_test.game_id == game_id]
specific_play = specific_game[specific_game.play_id == play_id]

# Get the supplementary data of the selection.
specific_gameplay_supplementary_data = supplementary_data[
    (supplementary_data.game_id==game_id) & 
    (supplementary_data.play_id==play_id)
]


closest_players_communities = specific_play.groupby('frame_id').apply(community_generator)
player_communities = list(closest_players_communities.value_counts().index)[0]


n = 2
xs = np.linspace(0, 120, 240 * n)       
ys = np.linspace(0, 53.333, 106 * n)
X, Y = np.meshgrid(xs, ys, indexing="xy")


# Dominance and stats computation:
all_dominance = []
total_all_stats_frame = []
total_frames = specific_play.frame_id.max()

for frame_id in specific_play.frame_id.unique():
    test = specific_play[specific_play.frame_id==frame_id]
    test['player_side_numerical']= test.player_side.map({'Defense': -1, 'Offense': 1})
    total_dominance = np.zeros_like(X, dtype=float)
    for _, player in test.iterrows():
        f = dominance_function(player) # returns a callable
        total_dominance += f(X, Y) # works on the whole grid at once
        
    total_dominance_norm = normalize_two_sided(total_dominance)
    stats_frame = collect_frame_player_stats(
        frame_df=test, # per-frame player DF
        D=total_dominance_norm, # Normalized to [-1,1] piecewise
        X=X, Y=Y,
        R=5.0,
        sector_deg=120
    )

    total_all_stats_frame.append(stats_frame)
    all_dominance.append(total_dominance_norm)

all_dominance_np = np.array(all_dominance)
dominance_gradient = np.gradient(all_dominance_np, axis=0)
total_all_stats_frame_df = pd.concat(total_all_stats_frame)


team_color = {1: 'magenta', -1: 'cyan'}
markers = ['s', '*', 'd', 'o', 'v', '^', 'p', 'X', 'D']

window = 4 # Rolling window for visualization
q = 'share_team' # This is the DOMINANCE INDEX


def_team = specific_gameplay_supplementary_data.defensive_team.values[0]
pos_team = specific_gameplay_supplementary_data.possession_team.values[0]

def_team_img_path = f'/kaggle/input/nfl-logos/{def_team}.png'
pos_team_img_path = f'/kaggle/input/nfl-logos/{pos_team}.png'

# Game status:
game_status_dict = specific_gameplay_supplementary_data.iloc[0].to_dict()
gsd_vta = game_status_dict['visitor_team_abbr']
gsd_hta = game_status_dict['home_team_abbr']
gsd_psvs = game_status_dict['pre_snap_visitor_score']
gsd_pshs = game_status_dict['pre_snap_home_score']
txt_game_status = f"""Game date: {game_status_dict['game_date']} | Season: {game_status_dict['season']} | Week: {game_status_dict['week']}
{game_status_dict['quarter']}° Quarter, {game_status_dict['game_clock']} on the clock - {game_status_dict['down']}° Down, {game_status_dict['yards_to_go']} Yards to go. 
Score: ({gsd_hta}) {gsd_pshs} - {gsd_psvs} ({gsd_vta})"""

max_pos_id = total_all_stats_frame_df[total_all_stats_frame_df.team_sign==1].groupby(['nfl_id']).share_team.mean().idxmax()
max_def_id = total_all_stats_frame_df[total_all_stats_frame_df.team_sign==-1].groupby(['nfl_id']).share_team.mean().idxmax()

max_pos_name = (total_all_stats_frame_df[total_all_stats_frame_df.nfl_id==max_pos_id].player_name.unique()[0]).replace(' ', '\n')
max_def_name = (total_all_stats_frame_df[total_all_stats_frame_df.nfl_id==max_def_id].player_name.unique()[0]).replace(' ', '\n')


def plot_report(test, total_dominance_norm, frame_id, savepath=None):
    """Function that plots the whole report with main field and interaction graphs"""
    if os.path.exists(savepath):
        print(f'File already exists at {savepath}.')
        return 

    # Main plot configuration
    fig = plt.figure(figsize=(24*2,80*2))
    gs = gridspec.GridSpec(80, 24, wspace=0, hspace=0)
    
    # Upper Corners
    up_corner_l = plt.subplot(gs[0,0])
    up_corner_l.axis('off')
    up_corner_r = plt.subplot(gs[0, -1])
    up_corner_r.axis('off')
    
    # Dominance Index
    title = plt.subplot(gs[2:3+1, 1:14+1])
    title.text(0.0, 0.5, 'Dominance index', ha='left', va='center', fontsize=180, color='k', fontweight='bold', **{'fontname':'Helotypo'})
    title.axis('off')
    
    # NFL logo:
    logo = plt.subplot(gs[0:4+1, 18:22+1])
    img = Image.open(nfl_image)
    img_array = np.array(img)
    logo.imshow(img_array)
    logo.axis('off')
    
    # Game status text:
    game_status_text = plt.subplot(gs[5, 2:6+1])
    game_status_text.text(0.0, 0.5, 'Game status:', ha='left', va='center', fontsize=50, color='k', fontweight='bold')
    game_status_text.axis('off')
    
    # Game status:
    game_status = plt.subplot(gs[5:6+1, 2:9+1])
    game_status.text(
        0.01, 0.5, txt_game_status, fontsize=35, va='top', ha='left',   
        bbox=dict(boxstyle='round,pad=0.5', fc='gainsboro', ec='k', lw=1)
    )
    game_status.axis('off')
    
    # Posesion team
    team_pos_plot = plt.subplot(gs[5:7+1, 12:14+1])
    for spine in team_pos_plot.spines.values():
        spine.set_visible(False)
    round_box = FancyBboxPatch(
        (-0.02, -0.02), # a bit outside the axes
        1.04, 1.04, # slightly bigger than 1x1
        boxstyle="round,pad=0.02,rounding_size=0.12",
        transform=team_pos_plot.transAxes,
        linewidth=3,
        edgecolor='magenta',
        facecolor="gainsboro",
        clip_on=False, # <-- important: don't clip to rectangular axes
        zorder=0,
    )
    team_pos_plot.add_patch(round_box)
    team_pos_plot.tick_params(axis='x', length=0, labelbottom=False)
    team_pos_plot.tick_params(axis='y', length=0, labelleft=False)
    img = Image.open(pos_team_img_path)
    img_array = np.array(img)
    team_pos_plot.imshow(img_array, zorder=1)
    team_pos_plot.set_xlabel(pos_team, fontsize=30, labelpad=40)
    
    # Defensive team
    team_def_plot = plt.subplot(gs[5:7+1, 16:18+1])
    for spine in team_def_plot.spines.values():
        spine.set_visible(False)
    round_box = FancyBboxPatch(
        (-0.02, -0.02), # a bit outside the axes
        1.04, 1.04, # slightly bigger than 1x1
        boxstyle="round,pad=0.02,rounding_size=0.12",
        transform=team_def_plot.transAxes,
        linewidth=3,
        edgecolor='cyan',
        facecolor="gainsboro",
        clip_on=False, # <-- important: don't clip to rectangular axes
        zorder=0,
    )
    team_def_plot.add_patch(round_box)
    
    team_def_plot.tick_params(axis='x', length=0, labelbottom=False)
    team_def_plot.tick_params(axis='y', length=0, labelleft=False)
    img = Image.open(def_team_img_path)
    img_array = np.array(img)
    team_def_plot.imshow(img_array)
    team_def_plot.set_xlabel(def_team, fontsize=30, labelpad=40)
    
    # Vs
    vs = plt.subplot(gs[6, 15])
    vs.text(0.5, 0.5, 'VS', ha='center', va='center', fontsize=50, color='k')
    vs.axis('off')
    
    # Gif
    ax_main = plt.subplot(gs[11:11+8, 3:3+18])
    if True:
        ax_main = plot_field(ax_main, nfl_image=None) # Plot the field
        ax_main.set_aspect('equal')
        plot_ball_land(ax_main, test, icons, size=0.02) # Plot the ball
        player_ids(ax_main, test, x='x', y='y', ids='player_name', id_func=lambda x: x.replace(' ', '\n'))
        
        im = ax_main.imshow(
            total_dominance_norm,
            origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            aspect="equal",
            cmap=custom_cmap
        )
        ax_main.invert_xaxis() # This can be commented if no inversion is needed
        ax_main.invert_yaxis() # This too.
        scatter_png(ax_main, test, x='x', y='y', side='player_side', icons=icons, size=0.02)
        plot_player_arrows(ax_main, test, dir_col='dir', scale=0.7, rotate=True)
        plot_player_arrows(ax_main, test, dir_col='o', scale=1.1, color='purple', rotate=True)

    ax_x = plt.subplot(gs[10, 3:3+18], sharex=ax_main)
    if True:
        x = X[0]
        extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
        ax_x.imshow((total_dominance_norm.sum(axis=0)[np.newaxis,:])**1, cmap=custom_cmap, aspect="auto", extent=extent,)# vmin=-1,vmax=1)
        ax_x.set_yticks([])
        ax_x.axis('off')
    ax_y = plt.subplot(gs[11:11+8, 2], sharey=ax_main)
    ax_y2 = plt.subplot(gs[11:11+8, 21], sharey=ax_main)
    if True:
        x = Y[:,1]
        extent = [0,1, x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.]
        ax_y.imshow((total_dominance_norm.sum(axis=1)[:,np.newaxis][::-1]), cmap=custom_cmap, aspect="auto", extent=extent,)# vmin=-1,vmax=1)
        ax_y.set_yticks([])
        ax_y.axis('off')
        ax_y2.imshow((total_dominance_norm.sum(axis=1)[:,np.newaxis][::-1])**1, cmap=custom_cmap, aspect="auto", extent=extent,)# vmin=-1,vmax=1)
        ax_y2.set_yticks([])
        ax_y2.axis('off')    

    # Max avg.
    max_avg_dom = plt.subplot(gs[20, 7:16+1])
    #max_avg_dom.text(0.5, 0.5, 'Highest Average Dominance\n(during the play)', ha='center', va='center', fontsize=80, color='k', fontweight='bold')
    max_avg_dom.text(0.5, 0.5, 'Highest Average Dominance', ha='center', va='center', fontsize=80, color='k', fontweight='bold')
    max_avg_dom.axis('off')
    
    # Defensive team Domination
    def_team_dom = plt.subplot(gs[22:25+1, 13:16+1])
    for spine in def_team_dom.spines.values():
        spine.set_visible(False)
    round_box = FancyBboxPatch(
        (-0.02, -0.02),   # a bit outside the axes
        1.04, 1.04,       # slightly bigger than 1x1
        boxstyle="round,pad=0.02,rounding_size=0.12",
        transform=def_team_dom.transAxes,
        linewidth=3,
        edgecolor='cyan',
        facecolor="paleturquoise",
        clip_on=False,    # <-- important: don't clip to rectangular axes
        zorder=0,
    )
    def_team_dom.add_patch(round_box)
    def_team_dom.tick_params(axis='x', length=0, labelbottom=False)
    def_team_dom.tick_params(axis='y', length=0, labelleft=False)
    def_team_dom.text(0.5, 0.5, max_def_name, va='center', ha='center', fontsize=60)
    def_team_dom.set_title('Def. Dominance', fontsize=40, pad=40)

    # Attk team Domination
    pos_team_dom = plt.subplot(gs[22:25+1, 7:10+1])
    for spine in pos_team_dom.spines.values():
        spine.set_visible(False)
    round_box = FancyBboxPatch(
        (-0.02, -0.02),   # a bit outside the axes
        1.04, 1.04,       # slightly bigger than 1x1
        boxstyle="round,pad=0.02,rounding_size=0.12",
        transform=pos_team_dom.transAxes,
        linewidth=3,
        edgecolor='magenta',
        facecolor="plum",
        clip_on=False,    # <-- important: don't clip to rectangular axes
        zorder=0,
    )
    pos_team_dom.add_patch(round_box)
    
    pos_team_dom.tick_params(axis='x', length=0, labelbottom=False)
    pos_team_dom.tick_params(axis='y', length=0, labelleft=False)
    pos_team_dom.text(0.5, 0.5, max_pos_name, va='center', ha='center', fontsize=60)
    pos_team_dom.set_title('Pos. Dominance', fontsize=40, pad=40)
    
    # Interacciones
    subtitle = plt.subplot(gs[27, 2:5+1])
    subtitle.text(0.0, 0.5, 'Player Interactions:', ha='left', va='center', fontsize=100, color='k', fontweight='bold')
    subtitle.axis('off')
    
    # Graficas de interaccion
    community_subplot_width = 7
    community_subplot_skip_spaces = 2
    
    for c, community in enumerate(player_communities):
        community_subplot_initial_pos = 29 + c * community_subplot_width
        community_subplot_final_plot = 29 + (c + 1) * community_subplot_width-community_subplot_skip_spaces
        community_subplot = plt.subplot(gs[community_subplot_initial_pos: community_subplot_final_plot , 2:19])
        participants = list(community)
        total_participants_all_stats_frame_df = total_all_stats_frame_df[
            total_all_stats_frame_df.nfl_id.isin(participants)
        ]
        # Plot each player individually
        for marker_idx, _nfl_id in enumerate(total_participants_all_stats_frame_df.nfl_id.unique()):
            s = total_participants_all_stats_frame_df[total_participants_all_stats_frame_df.nfl_id==_nfl_id]
            s2 = s[['frame_id', q, 'team_sign']].rolling(window=window, on='frame_id').mean()
            s2.plot(x='frame_id', y=q, ax=community_subplot, label='_nolegend_', 
                    color=s2.team_sign.map(team_color).mode()[0], linewidth=5, alpha=0.3, linestyle='--')
            player_name = f"{s['player_name'].unique()[0]} ({s['player_position'].unique()[0]})"

            # Plot strong line (marked over time)
            s2[s2.frame_id == frame_id].plot(
                x='frame_id', y=q, ax=community_subplot, label=player_name, linewidth=6,
                color=s2.team_sign.map(team_color).mode()[0], marker=markers[marker_idx], markeredgecolor='k', markersize=15,
            )
            s2[s2.frame_id <= frame_id].plot(
                x='frame_id', y=q, ax=community_subplot, label='_nolegend_', linewidth=6,
                color=s2.team_sign.map(team_color).mode()[0], 
            )
                    
        community_subplot.set_ylim(-0.1,1.1)
        community_subplot.set_ylabel('Dominance index.', fontsize=30)
        community_subplot.set_xlabel('Time passing (by frame [1 frame is 0.1 seconds]).', fontsize=30)
        community_subplot.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0, fontsize=25)

    # Lower Corners
    lo_corner_l = plt.subplot(gs[community_subplot_final_plot + 2, 0])
    lo_corner_l.axis('off')
    lo_corner_r = plt.subplot(gs[community_subplot_final_plot + 2, -1])
    lo_corner_r.axis('off')

    # Authors

    authors = plt.subplot(gs[community_subplot_final_plot + 2, -6:-1])
    authors_text = 'Created by: Daniel Hernández Mota, and Diana Myriam Barboza Belmudez.'
    authors.text(0.5, 0.5, authors_text, va='center', ha='right', fontsize=10, alpha=0.5)
    authors.axis('off')
    
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        print(f'Figure saved at {savepath}')
    else:
        plt.show('all')
    time.sleep(1)


def get_info_for_specific_frame(frame_id):
    """It is possible to get the information of a specific frame, without the need to compute previous."""
    specific_play_frame = specific_play[specific_play.frame_id==frame_id]
    individual_stats = total_all_stats_frame_df[total_all_stats_frame_df.frame_id==frame_id]
    specific_play_frame['player_side_numerical'] = specific_play_frame.player_side.map({'Defense': -1, 'Offense': 1})
    specific_play_frame = add_team_cohesion(specific_play_frame, radius=12.0)
    
    total_dominance = np.zeros_like(X, dtype=float)
    
    for _, player in specific_play_frame.iterrows():
        f = dominance_function(player)
        total_dominance += f(X, Y)
    total_dominance_norm = normalize_two_sided(total_dominance)
    
    return specific_play_frame, total_dominance_norm, frame_id, individual_stats

def save_info_for_ml_model(test, individual_stats, filepath):
    if os.path.exists(filepath):
        print(f'File already exists at {filepath}.')
        return None
    merged_test_stats = test.merge(
        individual_stats[['nfl_id', 'share_team']],
        on='nfl_id',
        how='left'
    )
    helper_ids = ['game_id', 'play_id','nfl_id', 'frame_id']
    helper_context = ['play_direction', 'absolute_yardline_number']
    helper_features = [
        'player_height', 'player_weight', 'player_position',
        'x', 'y','s', 'a', 'dir', 'o',
        'player_side_numerical', 'team_cohesion'
    ]
    helper_predictor = ['share_team']
    helper_variables = helper_ids + helper_context + helper_features + helper_predictor
    data = merged_test_stats[helper_variables]
    data.to_parquet(filepath)
    print(f'File saved at {filepath}.')
    return 


# Iterate for all the frames in the play
# for frame_id in specific_play.frame_id.unique():
# Or just some frames
for frame_id in [10, 30, 40]:
    # folder_name = f'report__{game_id}_{play_id}'
    # create_folder(folder_name)
    # figure_path = f'../experiments/figures/{folder_name}/frame_{str(frame_id).zfill(3)}.png'
    # data_path = f'../data/interim/{folder_name}_{str(frame_id).zfill(3)}.parquet'
    specific_play_frame, total_dominance_norm, frame_id, individual_stats = get_info_for_specific_frame(frame_id)
    plot_report(specific_play_frame, total_dominance_norm, frame_id, savepath='')
    # save_info_for_ml_model(specific_play_frame, individual_stats, data_path)
    del specific_play_frame, total_dominance_norm, frame_id, individual_stats
    


# We are not computing in this demo, but the gif has already been computed.

# print(folder_name)
# save_gif(folder_name, fps=10)

