# Core libraries
import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects

# Statistical & ML utilities
from scipy import stats
from scipy.spatial.distance import cdist, euclidean
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 10

# Custom color palettes
BEHAVIOR_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))
LAB_COLORS = plt.cm.Set3(np.linspace(0, 1, 12))

print("âœ… Libraries loaded successfully!")
print(f"ğŸ“Š Pandas: {pd.__version__} | NumPy: {np.__version__}")


# Load metadata
train_df = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
test_df = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')

print("="*70)
print("ğŸ“¦ DATASET OVERVIEW")
print("="*70)
print(f"Training videos: {len(train_df):,}")
print(f"Test videos: {len(test_df):,}")
print(f"\nğŸ�¢ Unique laboratories: {train_df['lab_id'].nunique()}")
print(f"ğŸ�¬ Total video hours: {train_df['video_duration_sec'].sum() / 3600:.1f}h")

# Identify annotated videos
annotated_mask = train_df['behaviors_labeled'].notna()
annotated_df = train_df[annotated_mask].copy()

print(f"\nâœ… Annotated videos: {len(annotated_df):,} ({len(annotated_df)/len(train_df)*100:.1f}%)")
print(f"â�Œ Unannotated videos: {(~annotated_mask).sum():,}")

# Quick lab breakdown
print("\n" + "="*70)
print("ğŸ”¬ TOP CONTRIBUTING LABS")
print("="*70)
top_labs = annotated_df['lab_id'].value_counts().head(10)
for i, (lab, count) in enumerate(top_labs.items(), 1):
    bar = "â–ˆ" * int(count / top_labs.max() * 30)
    print(f"{i:2d}. {lab:25s} â”‚ {bar:30s} â”‚ {count:3d} videos")


# Collect all annotations from parquet files
all_annotations = []

print("ğŸ”„ Loading annotation files...")
for idx, row in annotated_df.iterrows():
    lab_id = row['lab_id']
    video_id = row['video_id']
    
    annotation_path = f'/kaggle/input/MABe-mouse-behavior-detection/train_annotation/{lab_id}/{video_id}.parquet'
    
    try:
        annot = pd.read_parquet(annotation_path)
        annot['video_id'] = video_id
        annot['lab_id'] = lab_id
        annot['fps'] = row['frames_per_second']
        annot['duration_frames'] = annot['stop_frame'] - annot['start_frame']
        annot['duration_seconds'] = annot['duration_frames'] / annot['fps']
        all_annotations.append(annot)
    except FileNotFoundError:
        continue

annotations_full = pd.concat(all_annotations, ignore_index=True)

print(f"âœ… Loaded {len(annotations_full):,} behavioral annotations")
print(f"ğŸ“Š Unique behaviors: {annotations_full['action'].nunique()}")
print(f"ğŸ�¬ Videos with annotations: {annotations_full['video_id'].nunique()}")


# Compute predictability metrics for each behavior
behavior_stats = []

for behavior in annotations_full['action'].unique():
    behavior_data = annotations_full[annotations_full['action'] == behavior]
    
    # Frequency metrics
    total_occurrences = len(behavior_data)
    num_videos = behavior_data['video_id'].nunique()
    num_labs = behavior_data['lab_id'].nunique()
    
    # Duration metrics
    mean_duration = behavior_data['duration_seconds'].mean()
    std_duration = behavior_data['duration_seconds'].std()
    cv_duration = std_duration / mean_duration if mean_duration > 0 else np.inf
    
    # Temporal clustering (using video-level variance)
    video_counts = behavior_data.groupby('video_id').size()
    clustering_score = video_counts.std() / video_counts.mean() if len(video_counts) > 1 else 0
    
    # Predictability score (0-100)
    # Higher = easier to predict
    frequency_score = min(np.log10(total_occurrences + 1) / 4 * 100, 100)
    stability_score = max(0, 100 - cv_duration * 20)
    generalization_score = (num_labs / annotated_df['lab_id'].nunique()) * 100
    
    predictability = (frequency_score * 0.4 + stability_score * 0.3 + generalization_score * 0.3)
    
    behavior_stats.append({
        'behavior': behavior,
        'count': total_occurrences,
        'videos': num_videos,
        'labs': num_labs,
        'mean_duration_sec': mean_duration,
        'std_duration_sec': std_duration,
        'cv_duration': cv_duration,
        'clustering': clustering_score,
        'predictability_score': predictability
    })

behavior_df = pd.DataFrame(behavior_stats).sort_values('predictability_score', ascending=False)

print("ğŸ�¯ Behavior Predictability Ranking (Top 10 Most Predictable)")
print("="*80)
for i, row in behavior_df.head(10).iterrows():
    print(f"{row['behavior']:20s} â”‚ Score: {row['predictability_score']:5.1f} â”‚ "
          f"Count: {row['count']:6,d} â”‚ Labs: {row['labs']:2d} â”‚ "
          f"Duration: {row['mean_duration_sec']:5.2f}s Â± {row['std_duration_sec']:5.2f}s")

print("\nâš ï¸� Challenging Behaviors (Bottom 5)")
print("="*80)
for i, row in behavior_df.tail(5).iterrows():
    print(f"{row['behavior']:20s} â”‚ Score: {row['predictability_score']:5.1f} â”‚ "
          f"Count: {row['count']:6,d} â”‚ Labs: {row['labs']:2d}")


fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Predictability Score Distribution
ax1 = fig.add_subplot(gs[0, 0])
top_15 = behavior_df.nlargest(15, 'predictability_score')

colors = plt.cm.RdYlGn(top_15['predictability_score'] / 100)
bars = ax1.barh(range(len(top_15)), top_15['predictability_score'], color=colors, edgecolor='black', linewidth=1.2)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15['behavior'], fontsize=10)
ax1.set_xlabel('Predictability Score', fontsize=12, fontweight='bold')
ax1.set_title('ğŸ�¯ Top 15 Most Predictable Behaviors', fontsize=13, fontweight='bold', pad=15)
ax1.axvline(50, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Threshold')
ax1.grid(axis='x', alpha=0.3)
ax1.legend()

for i, (score, count) in enumerate(zip(top_15['predictability_score'], top_15['count'])):
    ax1.text(score + 1, i, f"{count:,}", va='center', fontsize=9, fontweight='bold')

# Plot 2: Frequency vs Duration Stability
ax2 = fig.add_subplot(gs[0, 1])
scatter_data = behavior_df[behavior_df['count'] >= 10]  # Filter rare behaviors for clarity

scatter = ax2.scatter(scatter_data['count'], 
                     scatter_data['cv_duration'],
                     s=scatter_data['labs'] * 30,
                     c=scatter_data['predictability_score'],
                     cmap='viridis',
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=1)

ax2.set_xscale('log')
ax2.set_xlabel('Frequency (log scale)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Duration Variability (CV)', fontsize=12, fontweight='bold')
ax2.set_title('ğŸ“Š Behavior Characteristics Map', fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)

# Annotate interesting behaviors
for _, row in scatter_data.nlargest(5, 'predictability_score').iterrows():
    ax2.annotate(row['behavior'][:8], 
                (row['count'], row['cv_duration']),
                fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Predictability Score', fontsize=10, fontweight='bold')

# Plot 3: Duration Distribution for Top Behaviors
ax3 = fig.add_subplot(gs[1, :])
top_behaviors = behavior_df.nlargest(8, 'count')['behavior'].tolist()

duration_data = []
labels = []
for beh in top_behaviors:
    durations = annotations_full[annotations_full['action'] == beh]['duration_seconds'].values
    # Cap at 99th percentile to avoid outliers
    durations_capped = durations[durations <= np.percentile(durations, 99)]
    duration_data.append(durations_capped)
    labels.append(f"{beh}\n(n={len(durations)})")

bp = ax3.boxplot(duration_data, 
                 labels=labels,
                 patch_artist=True,
                 showfliers=False,
                 medianprops=dict(color='red', linewidth=2),
                 boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

ax3.set_ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
ax3.set_title('â�±ï¸� Duration Distributions for Common Behaviors', fontsize=13, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, ax3.get_ylim()[1])

plt.suptitle('Behavior Complexity & Predictability Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.show()


# Sample diverse videos for spatial analysis
def load_tracking_sample(num_videos=15):
    """Load tracking data from diverse sources"""
    tracking_samples = []
    
    # Sample from different labs
    sampled_videos = annotated_df.groupby('lab_id').apply(
        lambda x: x.sample(min(2, len(x)), random_state=42)
    ).reset_index(drop=True)[:num_videos]
    
    for idx, row in sampled_videos.iterrows():
        lab_id = row['lab_id']
        video_id = row['video_id']
        
        tracking_path = f'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{lab_id}/{video_id}.parquet'
        annotation_path = f'/kaggle/input/MABe-mouse-behavior-detection/train_annotation/{lab_id}/{video_id}.parquet'
        
        try:
            tracking = pd.read_parquet(tracking_path)
            annotations = pd.read_parquet(annotation_path)
            
            # Normalize coordinates
            tracking['x_norm'] = tracking['x'] / row['pix_per_cm_approx']
            tracking['y_norm'] = tracking['y'] / row['pix_per_cm_approx']
            
            tracking['video_id'] = video_id
            tracking['lab_id'] = lab_id
            tracking['arena_width'] = row['arena_width_cm']
            tracking['arena_height'] = row['arena_height_cm']
            
            tracking_samples.append({
                'tracking': tracking,
                'annotations': annotations,
                'metadata': row
            })
            
            print(f"âœ“ Loaded: {lab_id[:20]:20s} | Video {video_id} | {len(tracking):,} frames")
            
        except Exception as e:
            continue
    
    return tracking_samples

print("ğŸ”„ Loading tracking data samples...")
tracking_data_samples = load_tracking_sample(num_videos=15)
print(f"\nâœ… Successfully loaded {len(tracking_data_samples)} videos")


# Analyze spatial patterns for behaviors
def compute_spatial_features(tracking_samples):
    """Compute where behaviors occur in the arena"""
    behavior_locations = defaultdict(list)
    
    for sample in tracking_samples:
        tracking = sample['tracking']
        annotations = sample['annotations']
        metadata = sample['metadata']
        
        arena_w = metadata['arena_width_cm']
        arena_h = metadata['arena_height_cm']
        
        # Get body center positions
        body_center = tracking[tracking['bodypart'] == 'body_center'].copy()
        
        if body_center.empty:
            # Fallback to any available bodypart
            available_parts = tracking['bodypart'].unique()
            if len(available_parts) > 0:
                body_center = tracking[tracking['bodypart'] == available_parts[0]].copy()
            else:
                continue
        
        # Process each annotation
        for _, ann in annotations.iterrows():
            action = ann['action']
            agent_id = ann['agent_id']
            start_frame = ann['start_frame']
            stop_frame = ann['stop_frame']
            
            # Get positions during this behavior
            behavior_positions = body_center[
                (body_center['mouse_id'] == agent_id) &
                (body_center['video_frame'] >= start_frame) &
                (body_center['video_frame'] <= stop_frame)
            ]
            
            if len(behavior_positions) > 0:
                # Compute spatial metrics (normalized to arena)
                x_norm = behavior_positions['x_norm'].mean() / arena_w
                y_norm = behavior_positions['y_norm'].mean() / arena_h
                
                # Distance to center
                center_dist = np.sqrt((x_norm - 0.5)**2 + (y_norm - 0.5)**2)
                
                # Distance to nearest wall
                wall_dist = min(x_norm, 1-x_norm, y_norm, 1-y_norm)
                
                behavior_locations[action].append({
                    'x': x_norm,
                    'y': y_norm,
                    'center_dist': center_dist,
                    'wall_dist': wall_dist
                })
    
    return behavior_locations

print("ğŸ§® Computing spatial patterns...")
spatial_patterns = compute_spatial_features(tracking_data_samples)

# Summarize
print("\nğŸ“Š Spatial Pattern Summary:")
print("="*70)
for behavior in sorted(spatial_patterns.keys())[:10]:
    locs = spatial_patterns[behavior]
    if len(locs) >= 5:
        avg_center_dist = np.mean([l['center_dist'] for l in locs])
        avg_wall_dist = np.mean([l['wall_dist'] for l in locs])
        print(f"{behavior:20s} â”‚ Samples: {len(locs):4d} â”‚ "
              f"Center dist: {avg_center_dist:.3f} â”‚ Wall dist: {avg_wall_dist:.3f}")


# Visualize where behaviors occur
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

# Select top behaviors by sample count
top_spatial_behaviors = sorted(spatial_patterns.items(), 
                               key=lambda x: len(x[1]), 
                               reverse=True)[:9]

for idx, (behavior, locations) in enumerate(top_spatial_behaviors):
    ax = axes[idx]
    
    if len(locations) == 0:
        continue
    
    # Extract coordinates
    x_coords = [loc['x'] for loc in locations]
    y_coords = [loc['y'] for loc in locations]
    
    # Create 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=20, range=[[0, 1], [0, 1]])
    
    # Smooth heatmap
    heatmap_smooth = gaussian_filter1d(gaussian_filter1d(heatmap, sigma=1, axis=0), sigma=1, axis=1)
    
    # Plot
    im = ax.imshow(heatmap_smooth.T, origin='lower', extent=[0, 1, 0, 1], 
                   cmap='hot', aspect='auto', alpha=0.8)
    
    # Draw arena boundary
    arena_rect = Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='cyan', facecolor='none')
    ax.add_patch(arena_rect)
    
    # Mark center
    ax.plot(0.5, 0.5, 'g*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f'{behavior}\n(n={len(locations)})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Normalized X')
    ax.set_ylabel('Normalized Y')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('ğŸ—ºï¸� Spatial Distribution of Behaviors in Arena', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()


# Compute behavior transitions
def extract_behavior_sequences(annotations_df, min_videos=3):
    """Extract behavior transition patterns"""
    
    # Group by video and agent
    sequences = []
    
    for (video_id, agent_id), group in annotations_df.groupby(['video_id', 'agent_id']):
        # Sort by time
        group_sorted = group.sort_values('start_frame')
        
        behaviors = group_sorted['action'].tolist()
        timestamps = group_sorted['start_frame'].tolist()
        
        if len(behaviors) >= 2:
            sequences.append({
                'video_id': video_id,
                'agent_id': agent_id,
                'behaviors': behaviors,
                'timestamps': timestamps
            })
    
    # Compute transitions
    transitions = defaultdict(int)
    
    for seq in sequences:
        behaviors = seq['behaviors']
        for i in range(len(behaviors) - 1):
            transition = (behaviors[i], behaviors[i+1])
            transitions[transition] += 1
    
    return sequences, transitions

print("ğŸ”„ Analyzing behavior sequences...")
sequences, transitions = extract_behavior_sequences(annotations_full)

print(f"âœ… Extracted {len(sequences):,} behavior sequences")
print(f"ğŸ“Š Unique transitions: {len(transitions):,}")

# Top transitions
print("\nğŸ”� Most Common Behavior Transitions:")
print("="*70)
sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
for i, ((from_beh, to_beh), count) in enumerate(sorted_transitions[:15], 1):
    print(f"{i:2d}. {from_beh:15s} â†’ {to_beh:15s} â”‚ {count:4d} times")


# Create transition matrix for top behaviors
top_behaviors_for_matrix = behavior_df.nlargest(12, 'count')['behavior'].tolist()

# Build transition matrix
transition_matrix = np.zeros((len(top_behaviors_for_matrix), len(top_behaviors_for_matrix)))

for (from_beh, to_beh), count in transitions.items():
    if from_beh in top_behaviors_for_matrix and to_beh in top_behaviors_for_matrix:
        i = top_behaviors_for_matrix.index(from_beh)
        j = top_behaviors_for_matrix.index(to_beh)
        transition_matrix[i, j] = count

# Normalize by row (probability of transitioning to next behavior)
row_sums = transition_matrix.sum(axis=1, keepdims=True)
transition_probs = np.divide(transition_matrix, row_sums, 
                             where=row_sums!=0, 
                             out=np.zeros_like(transition_matrix))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Raw counts
im1 = ax1.imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
ax1.set_xticks(range(len(top_behaviors_for_matrix)))
ax1.set_yticks(range(len(top_behaviors_for_matrix)))
ax1.set_xticklabels(top_behaviors_for_matrix, rotation=45, ha='right', fontsize=9)
ax1.set_yticklabels(top_behaviors_for_matrix, fontsize=9)
ax1.set_xlabel('To Behavior', fontsize=12, fontweight='bold')
ax1.set_ylabel('From Behavior', fontsize=12, fontweight='bold')
ax1.set_title('Behavior Transition Counts', fontsize=13, fontweight='bold', pad=15)
plt.colorbar(im1, ax=ax1, label='Count')

# Add text annotations for high values
for i in range(len(top_behaviors_for_matrix)):
    for j in range(len(top_behaviors_for_matrix)):
        if transition_matrix[i, j] >= 10:
            text = ax1.text(j, i, int(transition_matrix[i, j]),
                          ha="center", va="center", color="white", fontsize=8, fontweight='bold')

# Probabilities
im2 = ax2.imshow(transition_probs, cmap='Blues', aspect='auto', vmin=0, vmax=0.5)
ax2.set_xticks(range(len(top_behaviors_for_matrix)))
ax2.set_yticks(range(len(top_behaviors_for_matrix)))
ax2.set_xticklabels(top_behaviors_for_matrix, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(top_behaviors_for_matrix, fontsize=9)
ax2.set_xlabel('To Behavior', fontsize=12, fontweight='bold')
ax2.set_ylabel('From Behavior', fontsize=12, fontweight='bold')
ax2.set_title('Transition Probabilities', fontsize=13, fontweight='bold', pad=15)
plt.colorbar(im2, ax=ax2, label='Probability')

# Add text for high probabilities
for i in range(len(top_behaviors_for_matrix)):
    for j in range(len(top_behaviors_for_matrix)):
        if transition_probs[i, j] >= 0.1:
            text = ax2.text(j, i, f'{transition_probs[i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=8, fontweight='bold')

plt.suptitle('ğŸ”„ Behavior Transition Analysis', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()


def engineer_features_from_tracking(tracking_df, window_sizes=[5, 10, 30]):
    """
    Comprehensive feature engineering from pose tracking data
    """
    features_list = []
    
    # Process each mouse separately
    for mouse_id in tracking_df['mouse_id'].unique():
        mouse_data = tracking_df[tracking_df['mouse_id'] == mouse_id].copy()
        
        # Get body center (or fallback)
        body_parts = ['body_center', 'nose', 'head']
        center_data = None
        for part in body_parts:
            temp = mouse_data[mouse_data['bodypart'] == part].copy()
            if len(temp) > 0:
                center_data = temp.sort_values('video_frame')
                break
        
        if center_data is None or len(center_data) < 10:
            continue
        
        # Basic position
        x = center_data['x_norm'].values
        y = center_data['y_norm'].values
        frames = center_data['video_frame'].values
        
        # Velocity (first derivative)
        vx = np.diff(x, prepend=x[0])
        vy = np.diff(y, prepend=y[0])
        speed = np.sqrt(vx**2 + vy**2)
        
        # Acceleration (second derivative)
        ax = np.diff(vx, prepend=vx[0])
        ay = np.diff(vy, prepend=vy[0])
        acceleration = np.sqrt(ax**2 + ay**2)
        
        # Angular features
        angle = np.arctan2(vy, vx)
        angle_change = np.abs(np.diff(angle, prepend=angle[0]))
        angle_change = np.where(angle_change > np.pi, 2*np.pi - angle_change, angle_change)
        
        # Distance to center
        arena_w = center_data['arena_width'].iloc[0]
        arena_h = center_data['arena_height'].iloc[0]
        center_x, center_y = arena_w / 2, arena_h / 2
        dist_to_center = np.sqrt((x - center_x/arena_w)**2 + (y - center_y/arena_h)**2)
        
        # Distance to walls (normalized)
        dist_to_wall = np.minimum.reduce([
            x, 
            1 - x, 
            y, 
            1 - y
        ])
        
        # Rolling statistics for multiple windows
        feature_dict = {
            'video_id': center_data['video_id'].iloc[0],
            'mouse_id': mouse_id,
            'frame': frames,
            'x': x,
            'y': y,
            'speed': speed,
            'acceleration': acceleration,
            'angle_change': angle_change,
            'dist_to_center': dist_to_center,
            'dist_to_wall': dist_to_wall
        }
        
        # Add rolling features
        for window in window_sizes:
            feature_dict[f'speed_mean_{window}'] = pd.Series(speed).rolling(window, min_periods=1, center=True).mean().values
            feature_dict[f'speed_std_{window}'] = pd.Series(speed).rolling(window, min_periods=1, center=True).std().fillna(0).values
            feature_dict[f'acceleration_mean_{window}'] = pd.Series(acceleration).rolling(window, min_periods=1, center=True).mean().values
            feature_dict[f'angle_change_sum_{window}'] = pd.Series(angle_change).rolling(window, min_periods=1, center=True).sum().values
        
        features_df = pd.DataFrame(feature_dict)
        features_list.append(features_df)
    
    if len(features_list) > 0:
        return pd.concat(features_list, ignore_index=True)
    return pd.DataFrame()

print("ğŸ”§ Engineering features from tracking data...")
print("This may take a minute...")

# Engineer features for sample videos
all_features = []
for i, sample in enumerate(tracking_data_samples[:10]):
    features = engineer_features_from_tracking(sample['tracking'])
    if len(features) > 0:
        all_features.append(features)
    print(f"âœ“ Processed video {i+1}/10")

if len(all_features) > 0:
    combined_features = pd.concat(all_features, ignore_index=True)
    print(f"\nâœ… Generated {len(combined_features):,} feature vectors")
    print(f"ğŸ“Š Feature columns: {len(combined_features.columns)}")
else:
    print("âš ï¸� No features generated")
    combined_features = pd.DataFrame()


# Merge features with annotations to see feature distributions per behavior
if len(combined_features) > 0 and len(tracking_data_samples) > 0:
    
    # Collect annotated features
    behavior_features = []
    
    for sample in tracking_data_samples[:10]:
        annotations = sample['annotations']
        video_id = sample['metadata']['video_id']
        
        video_features = combined_features[combined_features['video_id'] == video_id]
        
        for _, ann in annotations.iterrows():
            agent_id = ann['agent_id']
            action = ann['action']
            start = ann['start_frame']
            stop = ann['stop_frame']
            
            # Get features for this behavior window
            mask = (
                (video_features['mouse_id'] == agent_id) &
                (video_features['frame'] >= start) &
                (video_features['frame'] <= stop)
            )
            
            behavior_window = video_features[mask].copy()
            
            if len(behavior_window) > 0:
                behavior_window['behavior'] = action
                behavior_features.append(behavior_window)
    
    if len(behavior_features) > 0:
        behavior_features_df = pd.concat(behavior_features, ignore_index=True)
        
        print(f"âœ… Extracted features for {len(behavior_features_df):,} annotated frames")
        print(f"ğŸ�­ Behaviors represented: {behavior_features_df['behavior'].nunique()}")
        
        # Select top behaviors for visualization
        top_behaviors_for_features = behavior_features_df['behavior'].value_counts().head(6).index.tolist()
        
        # Plot feature distributions
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        feature_cols = ['speed', 'acceleration', 'angle_change', 'dist_to_center', 'speed_mean_10', 'speed_std_30']
        
        for idx, feat in enumerate(feature_cols):
            ax = axes[idx]
            
            for behavior in top_behaviors_for_features:
                data = behavior_features_df[behavior_features_df['behavior'] == behavior][feat].dropna()
                
                # Remove extreme outliers for visualization
                data = data[data <= data.quantile(0.99)]
                
                if len(data) > 10:
                    ax.hist(data, bins=30, alpha=0.5, label=behavior, density=True)
            
            ax.set_xlabel(feat.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(alpha=0.3)
            ax.set_title(f'Distribution: {feat}', fontsize=12, fontweight='bold')
        
        plt.suptitle('ğŸ”¬ Feature Distributions Across Behaviors', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Feature separability analysis
        print("\n" + "="*70)
        print("ğŸ“Š FEATURE SEPARABILITY ANALYSIS")
        print("="*70)
        
        for feat in feature_cols:
            # Compute between-behavior variance
            behavior_means = []
            for behavior in top_behaviors_for_features:
                data = behavior_features_df[behavior_features_df['behavior'] == behavior][feat].dropna()
                if len(data) > 0:
                    behavior_means.append(data.mean())
            
            if len(behavior_means) > 1:
                separability = np.std(behavior_means) / (np.mean(behavior_means) + 1e-6)
                print(f"{feat:25s} â”‚ Separability: {separability:.4f}")
    else:
        print("âš ï¸� No behavior-feature mapping created")
else:
    print("âš ï¸� Skipping feature analysis - insufficient data")


# Comprehensive lab profiling
lab_profiles = []

for lab_id in annotated_df['lab_id'].unique():
    lab_data = annotated_df[annotated_df['lab_id'] == lab_id]
    
    # Basic stats
    num_videos = len(lab_data)
    total_duration = lab_data['video_duration_sec'].sum()
    
    # Tracking info
    tracking_methods = lab_data['tracking_method'].unique()
    body_parts_sets = lab_data['body_parts_tracked'].unique()
    
    # Parse body parts
    all_body_parts = set()
    for bp_str in body_parts_sets:
        try:
            parts = json.loads(bp_str)
            all_body_parts.update(parts)
        except:
            pass
    
    # Arena info
    arena_shapes = lab_data['arena_shape'].unique()
    avg_arena_size = lab_data['arena_width_cm'].mean()
    
    # Behavior info
    all_behaviors = set()
    for behaviors_str in lab_data['behaviors_labeled'].dropna():
        try:
            behaviors = json.loads(behaviors_str)
            for b in behaviors:
                # Parse "mouse1,mouse2,action" format
                parts = b.split(',')
                if len(parts) >= 3:
                    all_behaviors.add(parts[2])
        except:
            pass
    
    # FPS
    fps_values = lab_data['frames_per_second'].unique()
    avg_fps = lab_data['frames_per_second'].mean()
    
    lab_profiles.append({
        'lab_id': lab_id,
        'num_videos': num_videos,
        'total_hours': total_duration / 3600,
        'tracking_methods': ', '.join(tracking_methods),
        'num_body_parts': len(all_body_parts),
        'body_parts': ', '.join(sorted(list(all_body_parts))[:5]) + '...' if len(all_body_parts) > 5 else ', '.join(sorted(list(all_body_parts))),
        'arena_shapes': ', '.join(arena_shapes),
        'avg_arena_cm': avg_arena_size,
        'num_behaviors': len(all_behaviors),
        'fps_range': f"{fps_values.min():.0f}-{fps_values.max():.0f}",
        'avg_fps': avg_fps
    })

lab_profiles_df = pd.DataFrame(lab_profiles).sort_values('num_videos', ascending=False)

print("ğŸ�¢ LABORATORY PROFILES")
print("="*100)
print(lab_profiles_df.to_string(index=False))


# Analyze body part tracking across labs
bodypart_by_lab = defaultdict(lambda: defaultdict(int))

for idx, row in annotated_df.iterrows():
    lab_id = row['lab_id']
    try:
        body_parts = json.loads(row['body_parts_tracked'])
        for part in body_parts:
            bodypart_by_lab[lab_id][part] += 1
    except:
        pass

# Create body part matrix
all_body_parts_list = set()
for lab_parts in bodypart_by_lab.values():
    all_body_parts_list.update(lab_parts.keys())
all_body_parts_list = sorted(list(all_body_parts_list))

top_labs = lab_profiles_df.head(10)['lab_id'].tolist()

# Build matrix
bp_matrix = np.zeros((len(top_labs), len(all_body_parts_list)))

for i, lab in enumerate(top_labs):
    for j, part in enumerate(all_body_parts_list):
        bp_matrix[i, j] = bodypart_by_lab[lab].get(part, 0)

# Normalize by number of videos
for i, lab in enumerate(top_labs):
    num_videos = lab_profiles_df[lab_profiles_df['lab_id'] == lab]['num_videos'].iloc[0]
    if num_videos > 0:
        bp_matrix[i, :] = (bp_matrix[i, :] > 0).astype(float)  # Binary: available or not

# Plot
fig, ax = plt.subplots(figsize=(16, 8))

im = ax.imshow(bp_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(all_body_parts_list)))
ax.set_yticks(range(len(top_labs)))
ax.set_xticklabels(all_body_parts_list, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels([lab[:25] for lab in top_labs], fontsize=10)

ax.set_xlabel('Body Part', fontsize=12, fontweight='bold')
ax.set_ylabel('Laboratory', fontsize=12, fontweight='bold')
ax.set_title('ğŸ�¯ Body Part Tracking Availability Across Labs', fontsize=14, fontweight='bold', pad=20)

# Add grid
for i in range(len(top_labs) + 1):
    ax.axhline(i - 0.5, color='gray', linewidth=0.5)
for j in range(len(all_body_parts_list) + 1):
    ax.axvline(j - 0.5, color='gray', linewidth=0.5)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Available', fontsize=11, fontweight='bold')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['No', 'Yes'])

plt.tight_layout()
plt.show()

# Summary
print("\nğŸ“Š Body Part Tracking Summary:")
print("="*70)
for part in all_body_parts_list:
    num_labs = sum(1 for lab in top_labs if bodypart_by_lab[lab].get(part, 0) > 0)
    print(f"{part:25s} â”‚ Available in {num_labs}/{len(top_labs)} labs")


# Analyze which behaviors appear in which labs
behavior_by_lab = defaultdict(set)

for idx, row in annotated_df.iterrows():
    lab_id = row['lab_id']
    try:
        behaviors_str = row['behaviors_labeled']
        if pd.notna(behaviors_str):
            behaviors = json.loads(behaviors_str)
            for b in behaviors:
                parts = b.split(',')
                if len(parts) >= 3:
                    action = parts[2]
                    behavior_by_lab[lab_id].add(action)
    except:
        pass

# Get top behaviors and labs
top_behaviors_list = behavior_df.head(20)['behavior'].tolist()
top_labs_list = lab_profiles_df.head(10)['lab_id'].tolist()

# Build matrix
behavior_lab_matrix = np.zeros((len(top_behaviors_list), len(top_labs_list)))

for i, behavior in enumerate(top_behaviors_list):
    for j, lab in enumerate(top_labs_list):
        if behavior in behavior_by_lab[lab]:
            behavior_lab_matrix[i, j] = 1

# Plot
fig, ax = plt.subplots(figsize=(14, 10))

im = ax.imshow(behavior_lab_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(top_labs_list)))
ax.set_yticks(range(len(top_behaviors_list)))
ax.set_xticklabels([lab[:20] for lab in top_labs_list], rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(top_behaviors_list, fontsize=10)

ax.set_xlabel('Laboratory', fontsize=12, fontweight='bold')
ax.set_ylabel('Behavior', fontsize=12, fontweight='bold')
ax.set_title('ğŸ�­ Behavior Annotation Coverage Across Labs', fontsize=14, fontweight='bold', pad=20)

# Add grid
for i in range(len(top_behaviors_list) + 1):
    ax.axhline(i - 0.5, color='white', linewidth=1)
for j in range(len(top_labs_list) + 1):
    ax.axvline(j - 0.5, color='white', linewidth=1)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Annotated', fontsize=11, fontweight='bold')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['No', 'Yes'])

plt.tight_layout()
plt.show()

# Identify universal vs lab-specific behaviors
print("\nğŸŒ� Behavior Generalization Analysis:")
print("="*70)

universal_behaviors = []
lab_specific_behaviors = []

for behavior in top_behaviors_list:
    num_labs = sum(1 for lab in top_labs_list if behavior in behavior_by_lab[lab])
    coverage = num_labs / len(top_labs_list)
    
    if coverage >= 0.5:
        universal_behaviors.append((behavior, num_labs))
    else:
        lab_specific_behaviors.append((behavior, num_labs))

print("\nâœ… Universal Behaviors (>50% lab coverage):")
for beh, count in universal_behaviors:
    print(f"  â€¢ {beh:20s} â†’ {count}/{len(top_labs_list)} labs")

print("\nâš ï¸� Lab-Specific Behaviors (<50% lab coverage):")
for beh, count in lab_specific_behaviors[:10]:
    print(f"  â€¢ {beh:20s} â†’ {count}/{len(top_labs_list)} labs")

