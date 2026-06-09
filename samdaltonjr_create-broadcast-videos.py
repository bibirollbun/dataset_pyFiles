import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyBboxPatch, Wedge, Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patheffects as pe
from matplotlib import rc
import matplotlib.gridspec as gridspec
import os
from pathlib import Path as FilePath
from IPython.display import Video
import warnings
warnings.filterwarnings('ignore')
rc('animation', html='jshtml')
print('Libraries loaded!')


# Configuration
DATA_DIR = FilePath('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train')
SUPP_FILE = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv'

COLORS = {
    'offense': '#1565C0', 'defense': '#B71C1C', 'receiver': '#FFD700',
    'highlight_def': '#FF5722', 'ball_land': '#FF9800', 'field': '#2E7D32',
    'field_dark': '#1B5E20', 'gauge_low': '#F44336', 'gauge_mid': '#FF9800',
    'gauge_high': '#4CAF50', 'sidebar_bg': '#1a1a1a',
}
TRAIL_STYLE = {'pre_alpha': 0.25, 'pre_linewidth': 2, 'post_alpha': 0.7, 'post_linewidth': 4}
GHOST_ALPHA = 0.35
print('Config loaded!')


def load_full_play_data(game_id, play_id, week, data_dir=DATA_DIR, supp_file=SUPP_FILE):
    """Load pre-throw and post-throw data, identify ghost players."""
    week_str = str(week).zfill(2)
    input_df = pd.read_csv(data_dir / f'input_2023_w{week_str}.csv')
    input_df = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)].copy()
    output_df = pd.read_csv(data_dir / f'output_2023_w{week_str}.csv')
    output_df = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)].copy()
    
    # Identify ghost players
    input_players = set(input_df['nfl_id'].unique())
    output_players = set(output_df['nfl_id'].unique())
    ghost_ids = input_players - output_players
    
    # Get last known positions for ghosts
    last_frame = input_df['frame_id'].max()
    last_data = input_df[input_df['frame_id'] == last_frame]
    ghost_positions = {}
    for pid in ghost_ids:
        pdata = last_data[last_data['nfl_id'] == pid]
        if len(pdata) > 0:
            ghost_positions[pid] = {
                'x': pdata['x'].iloc[0], 'y': pdata['y'].iloc[0],
                'position': pdata['player_position'].iloc[0] if 'player_position' in pdata.columns else '?',
                'side': pdata['player_side'].iloc[0] if 'player_side' in pdata.columns else 'Unknown',
            }
    
    input_df['phase'] = 'pre_throw'
    output_df['phase'] = 'post_throw'
    max_input_frame = input_df['frame_id'].max()
    output_df['frame_id'] = output_df['frame_id'] + max_input_frame
    
    input_track = input_df[['game_id','play_id','nfl_id','frame_id','x','y','phase']].copy()
    output_track = output_df[['game_id','play_id','nfl_id','frame_id','x','y','phase']].copy()
    combined_df = pd.concat([input_track, output_track], ignore_index=True)
    
    supp = None
    if os.path.exists(supp_file):
        supp = pd.read_csv(supp_file)
        supp = supp[(supp['game_id']==game_id) & (supp['play_id']==play_id)]
    
    print(f'  Pre-throw players: {len(input_players)}, Post-throw: {len(output_players)}, Ghosts: {len(ghost_ids)}')
    return input_df, output_df, combined_df, supp, ghost_positions

def get_player_info(input_df):
    player_info = input_df.groupby('nfl_id').first()[['player_name','player_position','player_side','player_role']].reset_index()
    targeted_rec = player_info[player_info['player_role'] == 'Targeted Receiver']
    return player_info, targeted_rec

def get_ball_landing(input_df):
    return input_df['ball_land_x'].iloc[0], input_df['ball_land_y'].iloc[0]

def get_qb_position(input_df):
    passer = input_df[input_df['player_role'] == 'Passer']
    if len(passer) > 0:
        last = passer[passer['frame_id'] == passer['frame_id'].max()]
        if len(last) > 0: return last['x'].iloc[0], last['y'].iloc[0]
    return None, None

def calc_separation(output_df, rec_id, def_id):
    seps = []
    for frame in sorted(output_df['frame_id'].unique()):
        fd = output_df[output_df['frame_id'] == frame]
        rec = fd[fd['nfl_id'] == rec_id]
        defender = fd[fd['nfl_id'] == def_id]
        if len(rec) > 0 and len(defender) > 0:
            sep = np.sqrt((rec['x'].iloc[0]-defender['x'].iloc[0])**2 + (rec['y'].iloc[0]-defender['y'].iloc[0])**2)
            seps.append({'frame': frame, 'separation': sep})
    return pd.DataFrame(seps)

def find_closest_defender(input_df, rec_id):
    last_frame = input_df['frame_id'].max()
    fd = input_df[input_df['frame_id'] == last_frame]
    rec = fd[fd['nfl_id'] == rec_id]
    if len(rec) == 0: return None, None
    rx, ry = rec['x'].iloc[0], rec['y'].iloc[0]
    defenders = fd[fd['player_side'] == 'Defense']
    min_d, closest = float('inf'), None
    for _, row in defenders.iterrows():
        d = np.sqrt((rx-row['x'])**2 + (ry-row['y'])**2)
        if d < min_d: min_d, closest = d, row['nfl_id']
    return closest, min_d

print('Data functions loaded!')


def draw_field(ax, x_min, x_max, y_min, y_max):
    """Draw football field with yard lines and yard numbers."""
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.add_patch(Rectangle((x_min,y_min), x_max-x_min, y_max-y_min, facecolor=COLORS['field'], zorder=0))
    
    # End zones
    if x_min < 10:
        ax.add_patch(Rectangle((x_min,y_min), 10-x_min, y_max-y_min, facecolor=COLORS['field_dark'], zorder=1))
    if x_max > 110:
        ax.add_patch(Rectangle((110,y_min), x_max-110, y_max-y_min, facecolor=COLORS['field_dark'], zorder=1))
    
    # Yard lines
    for yard in range(0, 121, 5):
        if x_min <= yard <= x_max:
            lw = 0.5 if yard % 10 == 0 else 0.25
            ax.add_patch(Rectangle((yard-lw/2, y_min), lw, y_max-y_min, facecolor='white', alpha=0.9 if yard%10==0 else 0.6, zorder=2))
    
    # Yard numbers (every 10 yards from 10 to 50 and back down)
    # NFL field: 0-10 is end zone, 10-50-10 is the field
    # x coordinate 10 = goal line, x=60 = 50 yard line, x=110 = other goal line
    for yard_x in range(20, 110, 10):  # Draw at 20, 30, 40... 100
        if x_min <= yard_x <= x_max:
            # Convert to yard number (10, 20, 30, 40, 50, 40, 30, 20, 10)
            if yard_x <= 60:
                yard_num = yard_x - 10  # 20->10, 30->20, ..., 60->50
            else:
                yard_num = 110 - yard_x  # 70->40, 80->30, ..., 100->10
            
            # Draw numbers near top and bottom of visible field
            # Top numbers
            if y_max > 45:
                ax.text(yard_x, min(y_max - 3, 50), str(yard_num), ha='center', va='center',
                       fontsize=18, fontweight='bold', color='white', alpha=0.7,
                       path_effects=[pe.withStroke(linewidth=2, foreground=COLORS['field_dark'])], zorder=3)
            # Bottom numbers
            if y_min < 8:
                ax.text(yard_x, max(y_min + 3, 3), str(yard_num), ha='center', va='center',
                       fontsize=18, fontweight='bold', color='white', alpha=0.7,
                       path_effects=[pe.withStroke(linewidth=2, foreground=COLORS['field_dark'])], zorder=3)
    
    # Hash marks (short lines at each yard)
    for yard in range(10, 111):
        if x_min <= yard <= x_max:
            # Left hash (around y=23.5)
            if y_min < 23.5 < y_max:
                ax.plot([yard, yard], [23.2, 23.8], color='white', lw=0.5, alpha=0.5, zorder=2)
            # Right hash (around y=29.8)
            if y_min < 29.8 < y_max:
                ax.plot([yard, yard], [29.5, 30.1], color='white', lw=0.5, alpha=0.5, zorder=2)
    
    ax.set_aspect('equal'); ax.axis('off')

def draw_football(ax, x, y, px, py, size=1.0):
    dx, dy = x-px, y-py
    angle = np.degrees(np.arctan2(dy, dx)) + 90 if abs(dx) > 0.001 or abs(dy) > 0.001 else 90
    ax.add_patch(Ellipse((x,y), 0.8*size, 1.5*size, angle=angle, facecolor='#8B4513', edgecolor='white', lw=1.5, zorder=15))

def draw_player(ax, x, y, pos, color, size=400, highlighted=False, name=None, alpha=1.0):
    sz = size * 1.3 if highlighted else size
    if highlighted and alpha == 1.0: ax.scatter(x, y, c=color, s=sz*1.5, zorder=9, alpha=0.3)
    ax.scatter(x, y, c=color, s=sz, zorder=10, alpha=alpha, edgecolor='white' if alpha==1.0 else '#888', lw=2.5 if highlighted else 2)
    abbrev = {'WR':'WR','TE':'TE','RB':'RB','QB':'QB','CB':'CB','SS':'SS','FS':'FS','MLB':'LB','ILB':'LB','OLB':'LB','DE':'DE','DT':'DT','T':'OL','G':'OL','C':'OL','S':'S','DB':'DB','LB':'LB'}
    txt = abbrev.get(pos, pos[:2] if pos else '?')
    ax.text(x, y, txt, ha='center', va='center', fontsize=9 if highlighted else 7, fontweight='bold', color='white', alpha=alpha, zorder=11, path_effects=[pe.withStroke(linewidth=2, foreground='black')])
    if highlighted and name and alpha == 1.0:
        ax.annotate(name.split()[-1], (x, y+3), ha='center', fontsize=11, color='white', fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground='black')], zorder=12)

def draw_ghosts(ax, ghost_positions):
    """Draw ghost players at their last known positions."""
    for pid, pdata in ghost_positions.items():
        color = COLORS['defense'] if pdata.get('side') == 'Defense' else COLORS['offense']
        draw_player(ax, pdata['x'], pdata['y'], pdata.get('position','?'), color, size=180, alpha=GHOST_ALPHA)

print('Drawing functions loaded!')


def get_trail(combined_df, player_id, current_frame, throw_frame):
    """Get pre/post throw trails."""
    pdata = combined_df[(combined_df['nfl_id']==player_id) & (combined_df['frame_id']<=current_frame)].sort_values('frame_id')
    pre, post = [], []
    for _, row in pdata.iterrows():
        pt = (row['x'], row['y'])
        if row['frame_id'] <= throw_frame: pre.append(pt)
        else:
            if len(post) == 0 and len(pre) > 0: post.append(pre[-1])
            post.append(pt)
    return pre, post

def draw_trail(ax, pre, post, color):
    """Draw differentiated pre/post trails."""
    if len(pre) >= 2:
        pts = np.array(pre)
        ax.plot(pts[:,0], pts[:,1], color=color, alpha=TRAIL_STYLE['pre_alpha'], lw=TRAIL_STYLE['pre_linewidth'], solid_capstyle='round', zorder=4)
    if len(post) >= 2:
        pts = np.array(post)
        ax.plot(pts[:,0], pts[:,1], color=color, alpha=TRAIL_STYLE['post_alpha']*0.3, lw=TRAIL_STYLE['post_linewidth']+4, solid_capstyle='round', zorder=4)
        ax.plot(pts[:,0], pts[:,1], color=color, alpha=TRAIL_STYLE['post_alpha'], lw=TRAIL_STYLE['post_linewidth'], solid_capstyle='round', zorder=5)

print('Trail functions loaded!')


def draw_gauge(ax, pct, phase='pre_throw'):
    ax.clear(); ax.set_xlim(-1.5,1.5); ax.set_ylim(-0.6,1.6)
    ax.set_facecolor(COLORS['sidebar_bg']); ax.axis('off')
    ax.text(0, 1.45, 'CLOSURE', ha='center', fontsize=14, fontweight='bold', color='#AAA')
    ax.add_patch(Wedge((0,0), 1.0, 0, 180, width=0.25, facecolor='#333', edgecolor='#555', lw=2, zorder=1))
    if phase == 'pre_throw':
        ax.text(0, -0.35, '--', ha='center', fontsize=24, fontweight='bold', color='#666')
        ax.plot([0,-0.72], [0,0], color='#666', lw=3, zorder=4)
    else:
        fc = COLORS['gauge_low'] if pct <= 30 else COLORS['gauge_mid'] if pct <= 60 else COLORS['gauge_high']
        fill_angle = 180 * pct / 100
        if fill_angle > 0: ax.add_patch(Wedge((0,0), 1.0, 180-fill_angle, 180, width=0.25, facecolor=fc, zorder=2))
        na = np.radians(180-fill_angle)
        ax.plot([0, 0.72*np.cos(na)], [0, 0.72*np.sin(na)], color='white', lw=3, zorder=4)
        ax.text(0, -0.35, f'{pct:.0f}%', ha='center', fontsize=24, fontweight='bold', color='white')
    ax.add_patch(Circle((0,0), 0.12, facecolor='#222', edgecolor='white', lw=2, zorder=5))

def draw_stats(ax, def_name, def_pos, rec_name, start_sep, curr_sep, pct, result, route, phase):
    ax.clear(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_facecolor(COLORS['sidebar_bg']); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.05,0.88), 0.9, 0.10, boxstyle='round,pad=0.02', facecolor='#FFD700'))
    ax.text(0.5, 0.93, 'CLOSURE RATIO', ha='center', fontsize=12, fontweight='bold', color='black')
    ax.text(0.5, 0.78, def_name, ha='center', fontsize=14, fontweight='bold', color='white')
    ax.text(0.5, 0.70, f'{def_pos} vs {rec_name}', ha='center', fontsize=10, color='#AAA')
    ax.text(0.5, 0.62, route if route else '', ha='center', fontsize=10, color='#888', style='italic')
    pc = {'pre_throw': ('#2196F3','PRE-THROW'), 'frozen': ('#FFD700','FINAL')}.get(phase, ('#4CAF50','BALL IN AIR'))
    ax.add_patch(FancyBboxPatch((0.25,0.48), 0.5, 0.08, boxstyle='round,pad=0.02', facecolor=pc[0], alpha=0.9))
    ax.text(0.5, 0.52, pc[1], ha='center', fontsize=10, fontweight='bold', color='white')
    if phase != 'pre_throw':
        y = 0.38
        for lbl, val in [('Start Sep', f'{start_sep:.1f} yds'), ('Current', f'{curr_sep:.1f} yds'), ('Closed', f'{start_sep-curr_sep:.1f} yds')]:
            ax.text(0.12, y, lbl, ha='left', fontsize=10, color='#AAA')
            ax.text(0.88, y, val, ha='right', fontsize=11, fontweight='bold', color='white')
            y -= 0.09
    else:
        ax.text(0.5, 0.30, 'Tracking route...', ha='center', fontsize=11, color='#666', style='italic')
    rc = '#4CAF50' if result in ['INT','I','IN'] else '#F44336'
    rt = {'C':'COMPLETE','I':'INCOMPLETE','IN':'INTERCEPTION'}.get(result, result)
    ax.add_patch(FancyBboxPatch((0.15,0.03), 0.7, 0.08, boxstyle='round,pad=0.02', facecolor=rc))
    ax.text(0.5, 0.07, rt, ha='center', fontsize=9, fontweight='bold', color='white')

print('Sidebar functions loaded!')


def create_full_play_video(game_id, play_id, week, output_file='full_play.mp4', fps=10, dpi=120, title_frames=20, freeze_frames=30, defender_id=None):
    """
    Create full play video with ghost players and yard line numbers.
    """
    print(f'Loading game {game_id}, play {play_id}, week {week}...')
    input_df, output_df, combined_df, supp, ghost_positions = load_full_play_data(game_id, play_id, week)
    player_info, targeted_rec = get_player_info(input_df)
    
    rec_id = targeted_rec['nfl_id'].iloc[0]
    rec_name = targeted_rec['player_name'].iloc[0]
    
    route = supp['route_of_targeted_receiver'].iloc[0] if supp is not None and len(supp)>0 and 'route_of_targeted_receiver' in supp.columns else 'Unknown'
    result = supp['pass_result'].iloc[0] if supp is not None and len(supp)>0 and 'pass_result' in supp.columns else 'C'
    
    land_x, land_y = get_ball_landing(input_df)
    qb_x, qb_y = get_qb_position(input_df)
    throw_frame = input_df['frame_id'].max()

    if defender_id is not None:
        def_id = defender_id
    else:
        def_id, _ = find_closest_defender(input_df, rec_id)
    def_info = player_info[player_info['nfl_id'] == def_id]
    def_name = def_info['player_name'].iloc[0] if len(def_info)>0 else 'Unknown'
    def_pos = def_info['player_position'].iloc[0] if len(def_info)>0 else 'DEF'
    
    sep_df = calc_separation(output_df, rec_id, def_id)
    start_sep = sep_df['separation'].iloc[0] if len(sep_df)>0 else 0
    min_sep = sep_df['separation'].min() if len(sep_df)>0 else 0
    final_closure = (start_sep - min_sep) / start_sep if start_sep > 0 else 0
    
    print(f'  Defender: {def_name} ({def_pos})')
    print(f'  Start: {start_sep:.1f} yds -> Min: {min_sep:.1f} yds')
    print(f'  Final Closure: {final_closure*100:.0f}%')
    
    # Setup figure
    fig = plt.figure(figsize=(18,9)); fig.set_facecolor('#0a0a0a')
    gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,1], left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.03, hspace=0.05)
    ax_field = fig.add_subplot(gs[:,0])
    ax_gauge = fig.add_subplot(gs[0,1])
    ax_stats = fig.add_subplot(gs[1,1])
    
    # Field bounds
    all_x = list(combined_df['x']) + [land_x] + [g['x'] for g in ghost_positions.values()]
    all_y = list(combined_df['y']) + [land_y] + [g['y'] for g in ghost_positions.values()]
    if qb_x: all_x.append(qb_x); all_y.append(qb_y)
    x_min, x_max = max(0, min(all_x)-10), min(120, max(all_x)+10)
    y_min, y_max = max(0, min(all_y)-6), min(53.3, max(all_y)+6)
    
    all_frames = sorted(combined_df['frame_id'].unique())
    n_frames = len(all_frames)
    n_post = len(output_df['frame_id'].unique())
    total_frames = title_frames + n_frames + freeze_frames
    
    print(f'  Total frames: {total_frames} (title:{title_frames}, play:{n_frames}, freeze:{freeze_frames})')
    
    def animate(frame_num):
        ax_field.clear(); ax_field.set_facecolor('#0a0a0a')
        
        # Title card
        if frame_num < title_frames:
            ax_field.set_xlim(0,1); ax_field.set_ylim(0,1); ax_field.axis('off')
            alpha = min(1.0, frame_num/8)
            ax_field.text(0.5, 0.6, 'CLOSURE RATIO', ha='center', fontsize=42, fontweight='bold', color='white', alpha=alpha, path_effects=[pe.withStroke(linewidth=3, foreground='#FFD700')])
            ax_field.text(0.5, 0.48, 'Full Play Analysis', ha='center', fontsize=18, color='#AAA', alpha=alpha, style='italic')
            ax_field.text(0.5, 0.32, def_name, ha='center', fontsize=24, fontweight='bold', color='white', alpha=alpha*0.9)
            ax_field.text(0.5, 0.24, f'vs {rec_name}', ha='center', fontsize=18, color='#AAA', alpha=alpha*0.9)
            ax_gauge.clear(); ax_gauge.set_facecolor(COLORS['sidebar_bg']); ax_gauge.axis('off')
            ax_stats.clear(); ax_stats.set_facecolor(COLORS['sidebar_bg']); ax_stats.axis('off')
            return []
        
        idx = min(frame_num - title_frames, n_frames - 1)
        is_frozen = (frame_num - title_frames) >= n_frames
        current_frame = all_frames[idx]
        frame_data = combined_df[combined_df['frame_id'] == current_frame]
        is_post = current_frame > throw_frame
        phase = 'frozen' if is_frozen else 'post_throw' if is_post else 'pre_throw'
        
        # Calc closure
        curr_sep, closure = start_sep, 0
        if is_post and len(sep_df) > 0:
            post_idx = max(0, min(idx - (n_frames - n_post), len(sep_df)-1))
            curr_sep = sep_df.iloc[post_idx]['separation']
            min_so_far = sep_df.iloc[:post_idx+1]['separation'].min()
            closure = (start_sep - min_so_far) / start_sep if start_sep > 0 else 0
        
        draw_field(ax_field, x_min, x_max, y_min, y_max)
        
        if is_post and qb_x:
            ax_field.plot([qb_x, land_x], [qb_y, land_y], color='yellow', lw=2, ls='--', alpha=0.4, zorder=3)
            ax_field.scatter(land_x, land_y, c=COLORS['ball_land'], s=350, marker='X', zorder=5, edgecolor='black', lw=2.5)
        
        # Trails
        pre_t, post_t = get_trail(combined_df, rec_id, current_frame, throw_frame)
        draw_trail(ax_field, pre_t, post_t, COLORS['receiver'])
        if def_id:
            pre_t, post_t = get_trail(combined_df, def_id, current_frame, throw_frame)
            draw_trail(ax_field, pre_t, post_t, COLORS['highlight_def'])
        
        # Ghost players (only post-throw)
        if is_post:
            draw_ghosts(ax_field, ghost_positions)
        
        # Active players
        for _, row in frame_data.iterrows():
            pid = row['nfl_id']
            pinfo = player_info[player_info['nfl_id'] == pid]
            if len(pinfo) == 0: continue
            side, name, pos = pinfo['player_side'].iloc[0], pinfo['player_name'].iloc[0], pinfo['player_position'].iloc[0]
            if pid == rec_id: draw_player(ax_field, row['x'], row['y'], pos, COLORS['receiver'], 500, True, name)
            elif pid == def_id: draw_player(ax_field, row['x'], row['y'], pos, COLORS['highlight_def'], 500, True, name)
            elif side == 'Defense': draw_player(ax_field, row['x'], row['y'], pos, COLORS['defense'], 220)
            else: draw_player(ax_field, row['x'], row['y'], pos, COLORS['offense'], 220)
        
        # Separation line
        if is_post and def_id:
            rec_d = frame_data[frame_data['nfl_id'] == rec_id]
            def_d = frame_data[frame_data['nfl_id'] == def_id]
            if len(rec_d)>0 and len(def_d)>0:
                rx, ry, dx, dy = rec_d['x'].iloc[0], rec_d['y'].iloc[0], def_d['x'].iloc[0], def_d['y'].iloc[0]
                ax_field.plot([rx,dx], [ry,dy], 'white', lw=2.5, ls='--', alpha=0.9, zorder=8)
                ax_field.annotate(f'{curr_sep:.1f} yds', ((rx+dx)/2, (ry+dy)/2+2.5), ha='center', fontsize=12, color='white', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='black', edgecolor='white', alpha=0.9), zorder=13)
        
        # Ball
        if is_post and qb_x:
            post_idx = idx - (n_frames - n_post)
            prog = post_idx / max(n_post-1, 1)
            bx = qb_x + (land_x - qb_x) * prog
            by = qb_y + (land_y - qb_y) * prog
            px = qb_x + (land_x - qb_x) * max(0, (post_idx-1)/max(n_post-1,1)) if post_idx > 0 else qb_x
            py = qb_y + (land_y - qb_y) * max(0, (post_idx-1)/max(n_post-1,1)) if post_idx > 0 else qb_y
            h = 4 * prog * (1 - prog)
            draw_football(ax_field, bx, by, px, py, 1.0 + 0.8*h)
        
        # Flash at throw
        if current_frame == throw_frame + 1 and qb_x:
            ax_field.scatter(qb_x, qb_y, s=2000, c='yellow', alpha=0.5, zorder=14)
        
        # Phase indicator
        pc = {'frozen': ('#FFD700','FINAL'), 'post_throw': ('#4CAF50','BALL IN AIR')}.get(phase, ('#2196F3','PRE-THROW'))
        ax_field.text(0.98, 0.98, f'{pc[1]} | {idx+1}/{n_frames}', transform=ax_field.transAxes, fontsize=10, ha='right', va='top', color=pc[0], fontweight='bold')
        ax_field.text(0.02, 0.02, f'{def_name} ({def_pos}) vs {rec_name}', transform=ax_field.transAxes, fontsize=10, color='white', fontweight='bold', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8, edgecolor='#FFD700'))
        
        draw_gauge(ax_gauge, closure*100, phase)
        draw_stats(ax_stats, def_name, def_pos, rec_name, start_sep, curr_sep, closure*100, result, route, phase)
        return []
    
    print('Creating animation...')
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000/fps, blit=True)
    print(f'Saving to {output_file}...')
    anim.save(output_file, writer=FFMpegWriter(fps=fps, bitrate=5000), dpi=dpi)
    plt.close()
    print(f'Done! Final closure: {final_closure*100:.0f}%')
    return output_file

print('Video function loaded!')


#Kyle Hamilton breakup
create_full_play_video(
    game_id = 2023110502,
    play_id= 3010,
    week=9,
    output_file='kyle_hamilton_breakup_v2.mp4',
    fps=10,
    dpi=120,
    freeze_frames=30  # 3 seconds at 10fps
)


create_full_play_video(
    game_id=2023102904,
    play_id= 1747,
    week=8,
    output_file='jalen_ramsey_INT_v3.mp4',
    fps=10,
    dpi=120,
    freeze_frames=30  # 3 seconds at 10fps
)


create_full_play_video(
    game_id = 2023121801,
    play_id = 4106,
    week = 15,
    output_file='Julian_Love_INT.mp4',
    fps=10,
    dpi=120,
    freeze_frames=30,
    defender_id=47891# 3 seconds at 10fps
)


create_full_play_video(
    game_id = 2023121801,
    play_id = 3397,
    week = 15,
    output_file='Julian_Love_DeVonta_comp.mp4',
    fps=10,
    dpi=120,
    freeze_frames=30  # 3 seconds at 10fps
)




