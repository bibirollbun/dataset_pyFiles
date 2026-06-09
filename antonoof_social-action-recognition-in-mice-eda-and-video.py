%matplotlib inline
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

from PIL import Image
from IPython.display import HTML
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
submission = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/sample_submission.csv')


train.head()


test


train.info()


all(train.columns == test.columns)


submission.head()


# train annotation
pd.read_parquet('/kaggle/input/MABe-mouse-behavior-detection/train_annotation/AdaptableSnail/1212811043.parquet').head()


# train tracking
tracking_data = pd.read_parquet('/kaggle/input/MABe-mouse-behavior-detection/train_tracking/AdaptableSnail/1212811043.parquet')
tracking_data.head()


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')

# mouse image positioning function
def add_mouse_image(fig, ax, img_path='/kaggle/input/mousenotebook/mouse.png', position=(0.5, 0.5), zoom=0.15):
    img = mpimg.imread(img_path)
    x, y = position
    
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), xycoords='figure fraction', frameon=False, boxcoords="figure fraction")
    fig.add_artist(ab)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle('Laboratory Analysis & Tracking Methods', fontsize=18, fontweight='bold', y=0.95)

# Laboratory Distribution (Top 8 labs)
lab_counts = train['lab_id'].value_counts().head(8)
colors = plt.cm.Set3(np.linspace(0.2, 0.8, len(lab_counts)))

bars = ax1.bar(range(len(lab_counts)), lab_counts.values, color=colors, 
               edgecolor='black', alpha=0.85, linewidth=1.5)
ax1.set_title('Video Distribution by Laboratory', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Laboratory ID', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Videos', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(lab_counts)))
ax1.set_xticklabels([lab[:12] + '...' if len(lab) > 12 else lab for lab in lab_counts.index], 
                   rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 3,
            f'{int(height)}', ha='center', va='bottom', 
            fontweight='bold', fontsize=10, color='darkblue')

# Tracking Methods (Pie chart)
tracking_counts = train['tracking_method'].value_counts()
explode = [0.15] * len(tracking_counts)

wedges, texts, autotexts = ax2.pie(tracking_counts.values, 
                                  labels=tracking_counts.index, 
                                  autopct='%1.1f%%', 
                                  startangle=90, 
                                  explode=explode,
                                  colors=plt.cm.Pastel2(range(len(tracking_counts))),
                                  textprops={'fontsize': 11})

ax2.set_title('Mouse Tracking Methods', fontsize=14, fontweight='bold', pad=15)

add_mouse_image(fig, ax1, img_path='/kaggle/input/mousenotebook/mouse3.png', position=(0.56, 0.17), zoom=0.1)

plt.tight_layout()
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle('Experimental Setup & Mouse Information', fontsize=18, fontweight='bold', y=0.95)

# Arena Shapes Distribution
arena_shapes = train['arena_shape'].value_counts()
colors_shape = ['#FFD700', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

wedges, texts, autotexts = ax1.pie(arena_shapes.values, 
                                  labels=arena_shapes.index, 
                                  autopct='%1.1f%%', 
                                  startangle=45,
                                  explode=[0.15] * len(arena_shapes),
                                  colors=colors_shape[:len(arena_shapes)],
                                  textprops={'fontsize': 11})

ax1.set_title('Arena Shapes Distribution', fontsize=14, fontweight='bold', pad=15)

for autotext in autotexts:
    autotext.set_color('darkblue')
    autotext.set_fontweight('bold')

# Mouse Strain Analysis (Top 6 strains)
mouse_columns = [col for col in train.columns if 'strain' in col and 'mouse' in col]
all_strains = pd.concat([train[col] for col in mouse_columns]).dropna()
top_strains = all_strains.value_counts().head(6)

colors_strain = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_strains)))

bars = ax2.bar(range(len(top_strains)), top_strains.values, color=colors_strain,
               edgecolor='black', alpha=0.85)
ax2.set_title('Most Common Mouse Strains', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Mouse Strain', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Occurrences', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(top_strains)))
ax2.set_xticklabels(top_strains.index, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height)}', ha='center', va='bottom', 
            fontweight='bold', fontsize=10)

add_mouse_image(fig, ax1, img_path='/kaggle/input/mousenotebook/mouse1.webp', position=(0.436, 0.2), zoom=0.53)

plt.tight_layout()
plt.show()


def plot_trajectories_and_speed(tracking_data, video_meta, frame_range=(0, 1000)):
    # Filter data
    df = tracking_data[
        (tracking_data['video_frame'] >= frame_range[0]) & 
        (tracking_data['video_frame'] <= frame_range[1])
    ].copy()
    df = df[df['bodypart'] == 'body_center']

    # Colors for mice
    colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle('Mouse Movement Analysis', fontsize=18, fontweight='bold', y=0.96)

    # Trajectories
    for mouse_id in sorted(df['mouse_id'].unique()):
        mouse = df[df['mouse_id'] == mouse_id].sort_values('video_frame')
        c = colors.get(mouse_id, 'gray')
        ax1.plot(mouse['x'], mouse['y'], color=c, alpha=0.8, linewidth=2.5, label=f'Mouse {mouse_id}')
        ax1.scatter(mouse['x'].iloc[::20], mouse['y'].iloc[::20], c=c, s=40, edgecolor='white', linewidth=0.5, zorder=5)

    ax1.set(xlabel='X (pixels)', ylabel='Y (pixels)', title='Movement Trajectories')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.set_facecolor('#f9f9f9')
    
    for mouse_id in sorted(df['mouse_id'].unique()):
        mouse = df[df['mouse_id'] == mouse_id].sort_values('video_frame').copy()
        mouse['speed'] = np.sqrt(mouse['x'].diff()**2 + mouse['y'].diff()**2).fillna(0)
        mouse['speed_smooth'] = mouse['speed'].rolling(15, min_periods=1).mean()
        ax2.plot(mouse['video_frame'], mouse['speed_smooth'], 
                color=colors.get(mouse_id, 'gray'), label=f'Mouse {mouse_id}', linewidth=3)

    ax2.set(xlabel='Frame', ylabel='Speed (pixels/frame)', title='Smoothed Speed Over Time')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f9f9f9')
    
    add_mouse_image(fig, ax1, img_path='/kaggle/input/mousenotebook/mouse6.png', position=(0.65, 0.82), zoom=0.08)

    plt.tight_layout()
    plt.show()

sample_video = train.iloc[0]
lab_id = sample_video['lab_id']
video_id = sample_video['video_id']

tracking_path = f'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{lab_id}/{video_id}.parquet'
tracking_data = pd.read_parquet(tracking_path)

plot_trajectories_and_speed(tracking_data, sample_video, frame_range=(0, 1000))


def plot_density_and_distance(tracking_data, video_meta, frame_range=(0, 1000), img_path='/kaggle/input/mousenotebook/mouse1.png'):
    # Filter data
    df = tracking_data[
        (tracking_data['video_frame'] >= frame_range[0]) & 
        (tracking_data['video_frame'] <= frame_range[1])
    ].copy()
    df = df[df['bodypart'] == 'body_center']

    # Colors
    colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4'}

    # Create side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1.1]})
    fig.suptitle(' Movement Density & Social Distance', fontsize=18, fontweight='bold', y=0.96)

    # Movement Density
    x = df['x'].values
    y = df['y'].values

    hb = ax1.hist2d(x, y, bins=40, cmap='hot', alpha=0.85)
    cbar = plt.colorbar(hb[3], ax=ax1)
    cbar.set_label('Visit Density', fontsize=11, fontweight='bold')

    ax1.set(xlabel='X (pixels)', ylabel='Y (pixels)', title='Movement Density Heatmap')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#000000')
    cbar.ax.tick_params(labelsize=10)

    # Inter-Mouse Distances
    mice = sorted(df['mouse_id'].unique())

    if len(mice) > 1:
        distance_stats = []

        for i in range(len(mice)):
            for j in range(i + 1, len(mice)):
                m1 = df[df['mouse_id'] == mice[i]].set_index('video_frame')[['x', 'y']]
                m2 = df[df['mouse_id'] == mice[j]].set_index('video_frame')[['x', 'y']]
                merged = m1.join(m2, lsuffix='_1', rsuffix='_2', how='inner')

                if len(merged) == 0:
                    continue

                dist = np.sqrt((merged['x_1'] - merged['x_2'])**2 + (merged['y_1'] - merged['y_2'])**2)
                dist_smooth = dist.rolling(20, min_periods=1).mean()
                distance_stats.append(dist.mean())

                ax2.plot(merged.index, dist_smooth,
                        label=f'Mouse {mice[i]} â†” {mice[j]}',
                        linewidth=2.5, alpha=0.85)

        ax2.set(xlabel='Frame', ylabel='Distance (pixels)', title='Inter-Mouse Distance')
        ax2.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f9f9f9')

    else:
        ax2.text(0.5, 0.5, 'Only one mouse detected', ha='center', va='center',
                transform=ax2.transAxes, fontsize=13, fontweight='bold', color='#555')
        ax2.set_title('ğŸ“� Inter-Mouse Distance', fontsize=14, fontweight='bold')
        ax2.axis('off')
    
    add_mouse_image(fig, ax1, img_path='/kaggle/input/mousenotebook/mouse4.png', position=(0.64, 0.2), zoom=0.35)

    plt.tight_layout()
    plt.show()


sample_video = train.iloc[0]
lab_id = sample_video['lab_id']
video_id = sample_video['video_id']

tracking_path = f'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{lab_id}/{video_id}.parquet'
tracking_data = pd.read_parquet(tracking_path)

plot_density_and_distance(tracking_data, sample_video, frame_range=(0, 1000))


def plot_behavior_analysis(tracking_data, annotation_data, video_meta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1.2, 1]})
    fig.suptitle('Behavior Analysis', fontsize=18, fontweight='bold', y=0.96)

    # Filter body center data
    body_center_data = tracking_data[tracking_data['bodypart'] == 'body_center']
    colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4'}

    # Trajectories with behavior highlights
    for mouse_id in body_center_data['mouse_id'].unique():
        mouse = body_center_data[body_center_data['mouse_id'] == mouse_id]
        ax1.plot(mouse['x'], mouse['y'], color=colors.get(mouse_id, 'gray'), alpha=0.3, linewidth=1, label=f'Mouse {mouse_id}')

    # Highlight behaviors if annotations exist
    if not annotation_data.empty:
        unique_actions = annotation_data['action'].unique()
        behavior_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_actions)))
        color_map = dict(zip(unique_actions, behavior_colors))

        plotted_labels = set()
        for _, ann in annotation_data.iterrows():
            agent_id = int(ann['agent_id'].replace('mouse', '')) if isinstance(ann['agent_id'], str) else int(ann['agent_id'])
            behavior_frames = tracking_data[
                (tracking_data['video_frame'] >= ann['start_frame']) &
                (tracking_data['video_frame'] <= ann['stop_frame']) &
                (tracking_data['mouse_id'] == agent_id)
            ]
            if behavior_frames.empty:
                continue

            behavior_center = behavior_frames[behavior_frames['bodypart'] == 'body_center']
            label = ann['action'] if ann['action'] not in plotted_labels else ""
            plotted_labels.add(ann['action'])

            ax1.scatter(behavior_center['x'], behavior_center['y'],
                       c=[color_map[ann['action']]],
                       s=60, alpha=0.8, edgecolors='black', linewidth=0.5,
                       label=label)

    ax1.set(xlabel='X (pixels)', ylabel='Y (pixels)', title='Behavior Annotations on Trajectories')
    ax1.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.set_facecolor('#f9f9f9')

    # Behavior statistics
    if not annotation_data.empty:
        behavior_stats = annotation_data.groupby('action').agg(
            count=('start_frame', 'count'),
            avg_duration=('start_frame', lambda x: (annotation_data.loc[x.index, 'stop_frame'] - x).mean())
        ).round(1).sort_values('count', ascending=False)

        x_pos = np.arange(len(behavior_stats))
        bars = ax2.bar(x_pos, behavior_stats['count'],
                      color=plt.cm.Pastel1(range(len(behavior_stats))),
                      edgecolor='black', alpha=0.85, linewidth=0.8)

        ax2.set(xlabel='Behavior Type', ylabel='Occurrences', title='Behavior Frequency')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(behavior_stats.index, rotation=45, ha='right', fontsize=10)
        ax2.grid(axis='y', alpha=0.4)
        ax2.set_facecolor('#f9f9f9')

        # Add bar labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No behavior annotations\navailable', ha='center', va='center',
                transform=ax2.transAxes, fontsize=13, fontweight='bold', color='#666', linespacing=1.5)
        ax2.set_title('Behavior Frequency', fontsize=14, fontweight='bold')
        ax2.axis('off')
    
    add_mouse_image(fig, ax1, img_path='/kaggle/input/mousenotebook/mouse1.png', position=(0.85, 0.63), zoom=0.9)

    plt.tight_layout()
    plt.show()

sample_video = train.iloc[0]
lab_id = sample_video['lab_id']
video_id = sample_video['video_id']

tracking_path = f'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{lab_id}/{video_id}.parquet'
annotation_path = f'/kaggle/input/MABe-mouse-behavior-detection/train_annotation/{lab_id}/{video_id}.parquet'

tracking_data = pd.read_parquet(tracking_path)
annotation_data = pd.read_parquet(annotation_path)

plot_behavior_analysis(tracking_data, annotation_data, sample_video)


sample_video = train.iloc[0]
lab_id = sample_video['lab_id']
video_id = sample_video['video_id']

tracking_path = f'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{lab_id}/{video_id}.parquet'
tracking_data = pd.read_parquet(tracking_path)

body_center_data = tracking_data[tracking_data['bodypart'] == 'body_center']

colors = {
    1: '#FF006E',
    2: '#3A86FF',
    3: '#FFA500',
    4: '#38B000',
}

fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1.2]})
fig.suptitle('Mouse Speed & Position Distribution', fontsize=18, fontweight='bold', y=0.96)

# Speed Distribution (Boxplot)
speed_data = []

for mouse_id in sorted(body_center_data['mouse_id'].unique()):
    mouse = body_center_data[body_center_data['mouse_id'] == mouse_id].sort_values('video_frame').copy()
    mouse['speed'] = np.sqrt(mouse['x'].diff()**2 + mouse['y'].diff()**2).fillna(0)
    mouse['mouse_id'] = mouse_id
    speed_data.append(mouse[['mouse_id', 'speed']])

if speed_data:
    speed_df = pd.concat(speed_data, ignore_index=True)
    palette = [colors.get(i, 'gray') for i in sorted(speed_df['mouse_id'].unique())]
    sns.boxplot(data=speed_df, x='mouse_id', y='speed', ax=ax2,
                palette=palette,
                width=0.6, linewidth=1.5, fliersize=4, boxprops=dict(edgecolor="black"))

    ax2.set(xlabel='Mouse ID', ylabel='Speed (pixels/frame)', title='Speed Distribution by Mouse')
    ax2.grid(axis='y', alpha=0.4, linestyle='--')
    ax2.set_facecolor('#fafafa')
    ax2.spines[['top', 'right']].set_visible(False)

    # add mean value
    means = speed_df.groupby('mouse_id')['speed'].mean()
    for i, mean_val in enumerate(means):
        ax2.text(i, mean_val + 0.3, f'Î¼={mean_val:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11, color='#000000')

else:
    ax2.text(0.5, 0.5, 'No speed data available', ha='center', va='center',
             transform=ax2.transAxes, fontsize=13, fontweight='bold', color='#666')
    ax2.set_title('Speed Distribution by Mouse')
    ax2.axis('off')

# Position Allocation (KDE)
for mouse_id in sorted(body_center_data['mouse_id'].unique()):
    mouse = body_center_data[body_center_data['mouse_id'] == mouse_id]
    color = colors.get(mouse_id, 'gray')
    sns.kdeplot(data=mouse, x='x', y='y', ax=ax1, color=color, linewidths=3, alpha=0.65, fill=False, levels=10)

ax1.set(xlabel='X (pixels)', ylabel='Y (pixels)', title='Position Density (KDE)')
legend_elements = [Line2D([0], [0], color=colors.get(i, 'gray'), lw=4, label=f'Mouse {i}') for i in sorted(body_center_data['mouse_id'].unique())]

ax1.legend(handles=legend_elements,
           loc='upper right',
           fontsize=12,
           frameon=True,
           fancybox=True,
           shadow=True,
           ncol=1)

ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal', adjustable='datalim')
ax1.set_facecolor('#f0f4f8')
ax1.spines[['top', 'right']].set_visible(False)

add_mouse_image(fig, ax1, img_path='/kaggle/input/mousenotebook/mouse2.png', position=(0.5, 0.2), zoom=0.1)

plt.tight_layout()
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')

train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
sample_video = train.iloc[0]
lab_id = sample_video['lab_id']
video_id = sample_video['video_id']

tracking_path = f'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{lab_id}/{video_id}.parquet'
df_tracking = pd.read_parquet(tracking_path)

# Filter body center only
df = df_tracking[df_tracking['bodypart'] == 'body_center'].copy()
df['mouse_id'] = df['mouse_id'].astype(str)
mouse_ids = df['mouse_id'].unique()

# Assign colors
colors = plt.cm.Set1(np.linspace(0, 1, len(mouse_ids)))
color_map = dict(zip(mouse_ids, colors))

# Video dimensions
video_width = sample_video['video_width_pix']
video_height = sample_video['video_height_pix']

fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, video_width)
ax.set_ylim(0, video_height)
ax.invert_yaxis()
ax.set_title(f'Mouse Movement Animation â€” Video {video_id}', fontsize=16, fontweight='bold')
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
ax.grid(True, alpha=0.3)

# Arena border
arena = Rectangle((0, 0), video_width, video_height, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
ax.add_patch(arena)

# Initialize squares and trails
squares = {}
trail_lines = {}
trail_data = {mouse: {'x': [], 'y': []} for mouse in mouse_ids}

for mouse in mouse_ids:
    sq = Rectangle((0, 0), 20, 20, facecolor=color_map[mouse], edgecolor='black', alpha=0.9, label=f'Mouse {mouse}')
    ax.add_patch(sq)
    squares[mouse] = sq
    trail_lines[mouse], = ax.plot([], [], color=color_map[mouse], alpha=0.6, linewidth=2, linestyle='-')

ax.legend(loc='upper right', fontsize=11)

# Animation
df_sorted = df.sort_values('video_frame').reset_index(drop=True)
frames = sorted(df_sorted['video_frame'].unique())[:270]

def animate(frame):
    updated = []
    for mouse in mouse_ids:
        mouse_data = df_sorted[(df_sorted['mouse_id'] == mouse) & (df_sorted['video_frame'] <= frame)]
        if mouse_data.empty:
            continue
        last_pos = mouse_data.iloc[-1]
        x, y = last_pos['x'], last_pos['y']
        
        # Update square position
        squares[mouse].set_xy((x - 10, y - 10))
        updated.append(squares[mouse])
        
        # Update trail
        trail_data[mouse]['x'].append(x)
        trail_data[mouse]['y'].append(y)
        trail_lines[mouse].set_data(trail_data[mouse]['x'], trail_data[mouse]['y'])
        updated.append(trail_lines[mouse])
    
    ax.set_title(f'Video {video_id} â€” Frame {int(frame)}', fontsize=14)
    
    return updated

ani = animation.FuncAnimation(fig, animate, frames=frames, interval=80, blit=True, repeat=False)

plt.close(fig)
HTML(ani.to_jshtml())

