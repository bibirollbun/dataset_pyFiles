# Imports with error handling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


#splliting the importing
import seaborn as sns
import os
import gc
from datetime import datetime


#other part
import matplotlib.patches as patches
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML, display
from warnings import filterwarnings
filterwarnings('ignore')


#classification
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available")


#adiitional import
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


#Other 
from scipy import stats
from scipy.spatial.distance import euclidean, cdist


#until scaler
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#PCA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression


#extra metrics
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import euclidean
from scipy.stats import zscore #for z analysis


#linking chart
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import IncrementalPCA #for minimizing strain
from sklearn.cluster import MiniBatchKMeans


# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
if not os.path.exists('/kaggle/working/EDA'):
    os.makedirs('/kaggle/working/EDA')
eda_path = '/kaggle/working/EDA/'

print("="*120)
print(" "*25 + "NFL BIG DATA BOWL 2026 - FULL ANALYSIS P1")
print("="*120)


#emergency tqdm
from tqdm import tqdm


#making the base path
base_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final"
supplement = f"{base_path}/supplementary_data.csv"
main_t_path = f"{base_path}/train" #main training path


#result check
print(supplement)
print(main_t_path)


#initiating data loading in supplementary data
try:
    supplementary_df = pd.read_csv(supplement)
    print(f"âœ“ Supplementary data: {supplementary_df.shape[0]:,} plays loaded")
except Exception as e:
    print(f"Error loading supplementary data: {e}")
    supplementary_df = pd.DataFrame()


# Load tracking data with comprehensive error handling
all_input = []
all_output = []
weeks_loaded = []
#week range
weeks = range(1,19)


#loading inputs and outputs
for week in weeks:
    input_file = os.path.join(f"{main_t_path}/input_2023_w{week:02d}.csv")
    output_file = os.path.join(f"{main_t_path}/output_2023_w{week:02d}.csv")

    if os.path.exists(input_file) and os.path.exists(output_file):
        try:
            # --- Read input in chunks to minimize RAM ---
            chunk_list = []
            for chunk in pd.read_csv(input_file, chunksize= 15000, low_memory=False):
                # Fill missing player positions inside each chunk
                if 'player_position' in chunk.columns:
                    chunk['player_position'] = chunk['player_position'].fillna('LS')
                chunk_list.append(chunk)

            input_df = pd.concat(chunk_list, ignore_index=True, sort=True, copy= True)

            # --- Read output in chunks (if needed) ---
            output_chunks = pd.read_csv(output_file, chunksize= 15000, low_memory=False)
            output_df = pd.concat(output_chunks, ignore_index=True, sort=True, copy= True)

            # Add week info
            input_df['week'] = week
            output_df['week'] = week

            # Store results
            all_input.append(input_df)
            all_output.append(output_df)
            weeks_loaded.append(week)

            print(f"âœ“ Week {week}: {input_df.shape[0]:,} input records loaded")

            # Optional: cleanup to free RAM
            del chunk_list, input_df, output_df

        except Exception as e:
            print(f"âš ï¸� Error loading week {week}: {e}")
    else:
        print(f"âš ï¸� Missing files for week {week}")


# Combine efficiently
if all_input:
    input_combined = pd.concat(all_input, ignore_index=True, copy = True, sort = True)
    output_combined = pd.concat(all_output, ignore_index=True, copy = True, sort = True)
    print(f"\nâœ… Loaded {len(weeks_loaded)} weeks: {input_combined.shape[0]:,} total records")
else:
    input_combined = pd.DataFrame()
    output_combined = pd.DataFrame()
    print("Error: No data loaded")


# Check missing player positions
missing_count = input_combined['player_position'].isna().sum()
print(f"Missing player positions after fill: {missing_count}")


# Sanity check the LS replacements
print(input_combined['player_position'].value_counts().head(10))


#initiaintg gc
gc.collect()


# Print columns for each
display(HTML("<div class='insight-box'>ğŸ•µï¸� Supplementary Columns: " + ', '.join(supplementary_df.columns) + "</div>"))


#in input
display(HTML("<div class='insight-box'>ğŸ•µï¸� Pre-Throw Columns: " + ', '.join(input_combined.columns) + "</div>"))


# in output
display(HTML("<div class='insight-box'>ğŸ•µï¸� In-Air Columns: " + ', '.join(output_combined.columns) + "</div>"))


#in all input Sample heads
display(supplementary_df.head(6))


#display in input
display(input_combined.head(6))


#poitraying input in data
display(output_combined.head(6))


#merging data for easier analysis
position_df = pd.concat([input_combined,output_combined], ignore_index=True, sort = True, copy = True).sort_values(['game_id', 'play_id', 'frame_id'])


# Number of unique plays
unique_plays = position_df[['game_id', 'play_id']].copy().drop_duplicates()
display(HTML(f"<div class='insight-box'>ğŸ”¥ Number of Pass Plays: {len(unique_plays)}</div>"))


#initiaitng full position
full_df = position_df.merge(supplementary_df, on=['game_id', 'play_id'], how='left', copy=True)


#another gc
gc.collect()


#Function to decalre safe data
def safe_describe(df, name):
    """Safely describe a dataframe"""
    if df.empty:
        print(f"{name}: Empty dataframe")
        return
    
    print(f"\n{name} Overview:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  Missing values: {missing.sum()} total")
        print(f"  Columns with missing: {(missing > 0).sum()}")
    
    # Check data types
    dtypes = df.dtypes.value_counts()
    print(f"  Data types: {dict(dtypes)}")
    
    return df.describe()


# Analyze each dataset
safe_describe(input_combined, "Input Data") #in input


#Analisis in Output
safe_describe(output_combined, "Output Data")


#Analysis in uniqueness
safe_describe(unique_plays, "Output Data")


#id exclusion
exclude_cols = ['play_id', 'nfl_id', 'frame_id', 'game_id']
numeric_cols = [c for c in full_df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]


# Quick stats table (fixed: use go.Table)
stats_table = np.round(full_df[numeric_cols].describe().T.reset_index().rename(columns={'index': 'metric'}),4)
fig = go.Figure(data=[go.Table(
    header=dict(values=['metric', 'mean', 'std', 'min', 'max'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[stats_table['metric'], stats_table['mean'], stats_table['std'], stats_table['min'], stats_table['max']],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(title='Tracking Data Stats Snapshot')
fig.show(renderer='iframe')


#another gc
gc.collect()


# position listing
position_list = full_df['player_position'].copy().unique()


# Numerical features for analysis
numerical_features = ['s', 'a', 'o', 'dir', 'x', 'y']


#guide in listing
print(position_list)


# Create output directory
if not os.path.exists('/kaggle/working/EDA'):
    os.makedirs('/kaggle/working/EDA')
#declaring save path
eda_path = '/kaggle/working/EDA/'


#Add physics validation
print("âš–ï¸� PHYSICS VALIDATION ANALYSIS")
print("-" * 30)
#maximal human limit analysis
avg_human_speed = 8.8  # approximation is ~25 mph
avg_human_accel = 8.0   # realistic limits
#minimal limit
minimal_speed = 6.59


#inspecting speed demons and tortoises
speed_ex = (input_combined['s'] > avg_human_speed).sum()
speed_tr = (input_combined['s'] < minimal_speed).sum()
accel_bs = (input_combined['a'] > avg_human_accel).sum() #booster


#data length
total_records = len(input_combined)
#percentage
print(f"ğŸš€ Speed > {avg_human_speed} yds/s: {speed_ex:,}/{total_records:,} ({100*speed_ex/total_records:.2f}%)")
print(f"ğŸ�¢ Speed < {minimal_speed} yds/s: {speed_tr:,}/{total_records:,} ({100*speed_tr/total_records:.2f}%)")
print(f"ğŸ’¨ Accel > {avg_human_accel} yds/sÂ²: {accel_bs:,}/{total_records:,} ({100*accel_bs/total_records:.2f}%)")


#position list
positions = [position_list[0],position_list[1],position_list[2],position_list[3],position_list[4],position_list[6],position_list[9],position_list[17]]
#inspections
print(positions)


#defensive analysis
defensive = [positions[4],positions[5],position_list[10]]
#inspection
print(defensive)


#another gc
gc.collect()


#data filtering
position_data = input_combined[input_combined['player_position'].isin(positions)].copy()
position_data['turtoise'] = position_data['s'] < minimal_speed


#summary
position_stats = position_data.groupby('player_position')['s'].agg(['mean', 'std', 'max', 'min']).round(3).copy()
slow_ratio = (position_data['turtoise'].sum() / speed_tr)


#statistics
print("\nğŸ�ƒ Position Speed Statistics (Including Slow Players):")
for pos in position_stats.index:
    stats = position_stats.loc[pos]
    print(f"   â€¢ {pos}: Î¼={stats['mean']}, Ïƒ={stats['std']}, max={stats['max']}, min={stats['min']} yds/s")

print(f"\nğŸ�¢ Slow Player Ratio (<{minimal_speed} yds/s): {slow_ratio:.3f}")


# Create violin plot for speed distribution
fig = px.violin(
    position_data,
    x='player_position',
    y='s',
    color='player_position',
    box=True,
    points='outliers',
    title='Speed Distribution by Position (Including Slow Players) ğŸ�»ğŸ�¢',
)

# Add horizontal lines for physics limits
fig.add_hline(
    y=avg_human_speed,
    line_dash="dash",
    line_color="red",
    annotation_text="Maximal Average Human Speed (~18 mph)",
)
fig.add_hline(
    y=avg_human_accel,
    line_dash="dot",
    line_color="green",
    annotation_text="Average Human Acceleration ",)
fig.add_hline(
    y=minimal_speed,
    line_dash="dot",
    line_color="blue",
    annotation_text="Minimal Observed Speed Limit"
)

fig.show(renderer='iframe')


#emergency gc
gc.collect()


#table comparison
summary_table = (
    position_data
    .assign(speed_category=lambda df: pd.cut(
        df['s'],
        bins=[0, minimal_speed, avg_human_speed, df['s'].max()],
        labels=['Below Min', 'Normal', 'Above Max']
    ))
    .groupby(['player_position', 'speed_category'])
    .size()
    .unstack(fill_value=0)
)

print("\nğŸ“Š Speed Category Summary by Position:")
display(summary_table)


#initiaitng function
print("\nğŸ�ˆ SECTION 3: FIELD POSITION ANALYSIS (25+ VISUALIZATIONS)")
print("-"*100)
# initiating condition for 'huge' visual grid
if not input_combined.empty:
    # Create massive field visualization grid
    fig = plt.figure(figsize=(30, 35))
    gs = GridSpec(7, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Sample data for efficiency
    viz_sample = input_combined.sample(min(100000, len(input_combined)))
    
    chart_count = 0
    
    # Helper function for safe plotting
    def safe_hist2d(ax, x_col, y_col, data, title, cmap='YlOrRd'):
        try:
            if x_col in data.columns and y_col in data.columns:
                valid_data = data[[x_col, y_col]].dropna()
                if len(valid_data) > 0:
                    h = ax.hist2d(valid_data[x_col], valid_data[y_col], 
                                 bins=[40, 20], cmap=cmap, cmin=1)
                    ax.set_title(title, fontsize=10, fontweight='bold')
                    ax.set_xlabel('X (yards)', fontsize=8)
                    ax.set_ylabel('Y (yards)', fontsize=8)
                    plt.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
                    return True
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
        return False
    # 1. Overall density
    ax1 = fig.add_subplot(gs[0, 0])
    if safe_hist2d(ax1, 'x', 'y', viz_sample, 'Overall Player Density'):
        chart_count += 1
    
    # 2. Speed zones by quantile
    for i, quantile in enumerate([0.5, 0.75, 0.9, 0.95]):
        ax = fig.add_subplot(gs[0, i+1] if i < 3 else gs[1, i-3])
        try:
            speed_threshold = viz_sample['s'].quantile(quantile)
            high_speed = viz_sample[viz_sample['s'] > speed_threshold]
            if safe_hist2d(ax, 'x', 'y', high_speed, f'Speed > {quantile*100:.0f}th %ile', 'Reds'):
                chart_count += 1
        except:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    
    # 3. Acceleration zones by quantile
    for i, quantile in enumerate([0.5, 0.75, 0.9]):
        ax = fig.add_subplot(gs[1, i+1])
        try:
            acc_threshold = viz_sample['a'].quantile(quantile)
            high_acc = viz_sample[viz_sample['a'] > acc_threshold]
            if safe_hist2d(ax, 'x', 'y', high_acc, f'Accel > {quantile*100:.0f}th %ile', 'Blues'):
                chart_count += 1
        except:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    # 4. Player roles
    roles = ['Targeted Receiver', 'Passer', 'Defensive Coverage', 'Other Route Runner']
    for i, role in enumerate(roles):
        ax = fig.add_subplot(gs[2, i])
        try:
            role_data = viz_sample[viz_sample['player_role'] == role]
            if safe_hist2d(ax, 'x', 'y', role_data, f'{role} Positions', 'Greens'):
                chart_count += 1
        except:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    
    # 5. Top positions
    try:
        top_positions = viz_sample['player_position'].value_counts().head(8).index
        for i, pos in enumerate(top_positions):
            ax = fig.add_subplot(gs[3 + i//4, i%4])
            pos_data = viz_sample[viz_sample['player_position'] == pos]
            if safe_hist2d(ax, 'x', 'y', pos_data, f'{pos} Heat Map', 'viridis'):
                chart_count += 1
    except:
        pass
    # 6. Ball landing zones
    ax = fig.add_subplot(gs[5, 0])
    if safe_hist2d(ax, 'ball_land_x', 'ball_land_y', viz_sample, 'Ball Landing Zones', 'Oranges'):
        chart_count += 1
    
    # 7. Direction-based movement
    for i, dir_range in enumerate([(0, 90), (90, 180), (180, 270), (270, 360)]):
        ax = fig.add_subplot(gs[5, i+1] if i < 3 else gs[6, i-3])
        try:
            dir_data = viz_sample[(viz_sample['dir'] >= dir_range[0]) & 
                                  (viz_sample['dir'] < dir_range[1])]
            if safe_hist2d(ax, 'x', 'y', dir_data, f'Direction {dir_range[0]}Â°-{dir_range[1]}Â°', 'plasma'):
                chart_count += 1
        except:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    # 8. Player side comparison
    for i, side in enumerate(['Offense', 'Defense']):
        ax = fig.add_subplot(gs[6, i+1])
        try:
            side_data = viz_sample[viz_sample['player_side'] == side]
            if safe_hist2d(ax, 'x', 'y', side_data, f'{side} Positions', 'coolwarm'):
                chart_count += 1
        except:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    
    # 9. Frame-based analysis (early vs late frames)
    ax = fig.add_subplot(gs[6, 3])
    try:
        early_frames = viz_sample[viz_sample['frame_id'] <= 5]
        if safe_hist2d(ax, 'x', 'y', early_frames, 'Early Frames (1-5)', 'spring'):
            chart_count += 1
    except:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    
    plt.suptitle(f'Comprehensive Field Analysis ({chart_count} Visualizations)', 
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{eda_path}field_analysis_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"âœ“ Created {chart_count} field visualizations")


#space for gc
gc.collect()


#movement analysis
print("n\ predicting movement")
if not input_combined.empty:
    fig, axes = plt.subplots(5, 4, figsize=(24, 25))
    axes = axes.flatten()
    chart_idx = 0
    
    # Numerical features for analysis
    numerical_features = ['s', 'a', 'o', 'dir', 'x', 'y']
    
    # 1-6. Distribution plots for each feature
    for i, feature in enumerate(numerical_features):
        try:
            data = input_combined[feature].dropna().sample(min(10000, len(input_combined)))
            axes[chart_idx].hist(data, bins=50, color=plt.cm.Set3(i), edgecolor='black', alpha=0.7)
            axes[chart_idx].set_xlabel(feature)
            axes[chart_idx].set_ylabel('Frequency')
            axes[chart_idx].set_title(f'{feature} Distribution', fontweight='bold')
            axes[chart_idx].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
            axes[chart_idx].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.2f}')
            axes[chart_idx].legend(fontsize=8)
            axes[chart_idx].grid(True, alpha=0.3)
            chart_idx += 1
        except Exception as e:
            axes[chart_idx].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
            chart_idx += 1
    # 7-12. Scatter plots for feature relationships
    feature_pairs = [('s', 'a'), ('x', 'y'), ('o', 'dir'), ('s', 'x'), ('a', 'y'), ('dir', 'o')]
    for feat1, feat2 in feature_pairs:
        try:
            sample = input_combined[[feat1, feat2]].dropna().sample(min(5000, len(input_combined)))
            axes[chart_idx].scatter(sample[feat1], sample[feat2], alpha=0.3, s=1)
            axes[chart_idx].set_xlabel(feat1)
            axes[chart_idx].set_ylabel(feat2)
            axes[chart_idx].set_title(f'{feat1} vs {feat2}', fontweight='bold')
            axes[chart_idx].grid(True, alpha=0.3)
            chart_idx += 1
        except:
            axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            chart_idx += 1
     # 13. Speed by player role
    try:
        role_speeds = input_combined.groupby('player_role')['s'].agg(['mean', 'std']).sort_values('mean')
        axes[chart_idx].barh(range(len(role_speeds)), role_speeds['mean'], 
                            xerr=role_speeds['std'], color='#3498db')
        axes[chart_idx].set_yticks(range(len(role_speeds)))
        axes[chart_idx].set_yticklabels(role_speeds.index, fontsize=8)
        axes[chart_idx].set_xlabel('Speed (y/s)')
        axes[chart_idx].set_title('Speed by Role', fontweight='bold')
        axes[chart_idx].grid(True, alpha=0.3)
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
    # 14. Acceleration by position
    try:
        top_pos = input_combined['player_position'].value_counts().head(10).index
        pos_acc = input_combined[input_combined['player_position'].isin(top_pos)].groupby('player_position')['a'].mean()
        axes[chart_idx].bar(range(len(pos_acc)), pos_acc.values, color='#e74c3c')
        axes[chart_idx].set_xticks(range(len(pos_acc)))
        axes[chart_idx].set_xticklabels(pos_acc.index, rotation=45, ha='right', fontsize=8)
        axes[chart_idx].set_ylabel('Acceleration (y/sÂ²)')
        axes[chart_idx].set_title('Acceleration by Position', fontweight='bold')
        axes[chart_idx].grid(True, alpha=0.3)
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
     # 15. Speed distribution comparison (Offense vs Defense)
    try:
        for side in ['Offense', 'Defense']:
            side_speeds = input_combined[input_combined['player_side'] == side]['s'].dropna()
            axes[chart_idx].hist(side_speeds, bins=30, alpha=0.5, label=side, density=True)
        axes[chart_idx].set_xlabel('Speed (y/s)')
        axes[chart_idx].set_ylabel('Density')
        axes[chart_idx].set_title('Speed: Offense vs Defense', fontweight='bold')
        axes[chart_idx].legend()
        axes[chart_idx].grid(True, alpha=0.3)
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
    # 16. Direction polar plot
    try:
        dir_sample = input_combined['dir'].dropna().sample(min(5000, len(input_combined)))
        dir_hist, dir_bins = np.histogram(dir_sample, bins=36, range=(0, 360))
        theta = np.linspace(0, 2*np.pi, 36, endpoint=False)
        axes[chart_idx].remove()
        ax_polar = fig.add_subplot(5, 4, chart_idx+1, projection='polar')
        ax_polar.bar(theta, dir_hist, width=2*np.pi/36, bottom=0)
        ax_polar.set_title('Direction Distribution (Polar)', fontweight='bold', pad=20)
        chart_idx += 1
    except:
        chart_idx += 1
    # 17. Orientation polar plot
    try:
        o_sample = input_combined['o'].dropna().sample(min(5000, len(input_combined)))
        o_hist, o_bins = np.histogram(o_sample, bins=36, range=(0, 360))
        theta = np.linspace(0, 2*np.pi, 36, endpoint=False)
        axes[chart_idx].remove()
        ax_polar2 = fig.add_subplot(5, 4, chart_idx+1, projection='polar')
        ax_polar2.bar(theta, o_hist, width=2*np.pi/36, bottom=0, color='orange')
        ax_polar2.set_title('Orientation Distribution (Polar)', fontweight='bold', pad=20)
        chart_idx += 1
    except:
        chart_idx += 1
    # 18-20. Box plots for remaining positions
    for i in range(chart_idx, min(chart_idx + 3, 20)):
        try:
            feature = numerical_features[i % len(numerical_features)]
            top_pos = input_combined['player_position'].value_counts().head(5).index
            box_data = [input_combined[input_combined['player_position'] == pos][feature].dropna() 
                       for pos in top_pos]
            bp = axes[i].boxplot(box_data, labels=top_pos, patch_artist=True)
            for patch, color in zip(bp['boxes'], plt.cm.Set2(range(len(top_pos)))):
                patch.set_facecolor(color)
            axes[i].set_ylabel(feature)
            axes[i].set_title(f'{feature} by Top Positions', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        except:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
    
    plt.suptitle('Comprehensive Movement Analysis (20 Charts)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{eda_path}movement_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"âœ“ Created movement analysis visualizations")


#section analysisis : output
if not full_df.empty:
    fig, axes = plt.subplots(5, 4, figsize=(24, 25))
    axes = axes.flatten()
    chart_idx = 0
    # 1. Pass result distribution
    try:
        if 'pass_result' in full_df.columns:
            pass_counts = full_df['pass_result'].value_counts()
            axes[chart_idx].pie(pass_counts.values, labels=pass_counts.index, 
                               autopct='%1.1f%%', startangle=45)
            axes[chart_idx].set_title('Pass Result Distribution', fontweight='bold')
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
    
    # 2. EPA distribution
    try:
        if 'expected_points_added' in full_df.columns:
            epa_data = full_df['expected_points_added'].dropna()
            axes[chart_idx].hist(epa_data, bins=50, color='#3498db', edgecolor='black')
            axes[chart_idx].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[chart_idx].set_xlabel('EPA')
            axes[chart_idx].set_ylabel('Frequency')
            axes[chart_idx].set_title('EPA Distribution', fontweight='bold')
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
     # 3. Pass length distribution
    try:
        if 'pass_length' in full_df.columns:
            pass_len = full_df['pass_length'].dropna()
            axes[chart_idx].hist(pass_len, bins=40, color='#2ecc71', edgecolor='black')
            axes[chart_idx].set_xlabel('Pass Length (yards)')
            axes[chart_idx].set_ylabel('Frequency')
            axes[chart_idx].set_title('Pass Length Distribution', fontweight='bold')
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
    
    # 4. Yards gained distribution
    try:
        if 'yards_gained' in full_df.columns:
            yards = full_df['yards_gained'].dropna()
            axes[chart_idx].hist(yards, bins=50, color='#e74c3c', edgecolor='black')
            axes[chart_idx].set_xlabel('Yards Gained')
            axes[chart_idx].set_ylabel('Frequency')
            axes[chart_idx].set_title('Yards Gained Distribution', fontweight='bold')
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
    # 5. Down analysis
    try:
        if 'down' in full_df.columns and 'pass_result' in plays_data.columns:
            down_success = full_df.groupby('down')['pass_result'].apply(
                lambda x: (x == 'C').mean() * 100 if len(x) > 0 else 0
            )
            axes[chart_idx].bar(down_success.index, down_success.values, color='#9b59b6')
            axes[chart_idx].set_xlabel('Down')
            axes[chart_idx].set_ylabel('Completion %')
            axes[chart_idx].set_title('Completion Rate by Down', fontweight='bold')
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
    
    # 6. Quarter analysis
    try:
        if 'quarter' in full_df.columns:
            quarter_counts = full_df['quarter'].value_counts().sort_index()
            axes[chart_idx].bar(quarter_counts.index, quarter_counts.values, color='#f39c12')
            axes[chart_idx].set_xlabel('Quarter')
            axes[chart_idx].set_ylabel('Number of Plays')
            axes[chart_idx].set_title('Plays by Quarter', fontweight='bold')
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
    # 7. Play action analysis
    try:
        if 'play_action' in full_df.columns:
            pa_stats = full_df.groupby('play_action').agg({
                'pass_result': lambda x: (x == 'C').mean() * 100 if len(x) > 0 else 0
            })
            # Handle variable number of play_action values
            pa_values = pa_stats['pass_result'].values
            pa_labels = [f"PA={i}" for i in pa_stats.index]
            axes[chart_idx].bar(range(len(pa_values)), pa_values, color='#16a085')
            axes[chart_idx].set_xticks(range(len(pa_values)))
            axes[chart_idx].set_xticklabels(pa_labels)
            axes[chart_idx].set_ylabel('Completion %')
            axes[chart_idx].set_title('Play Action Impact', fontweight='bold')
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
    # 8. Coverage type
    try:
        if 'team_coverage_man_zone' in full_df.columns:
            coverage_stats = full_df['team_coverage_man_zone'].value_counts()
            axes[chart_idx].bar(range(len(coverage_stats)), coverage_stats.values, 
                               color=['#FF6B6B', '#4ECDC4'][:len(coverage_stats)])
            axes[chart_idx].set_xticks(range(len(coverage_stats)))
            axes[chart_idx].set_xticklabels(coverage_stats.index)
            axes[chart_idx].set_ylabel('Count')
            axes[chart_idx].set_title('Coverage Type Distribution', fontweight='bold')
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
     # 9. Formation analysis
    try:
        if 'offense_formation' in full_df.columns:
            formation_counts = full_df['offense_formation'].value_counts().head(10)
            axes[chart_idx].barh(range(len(formation_counts)), formation_counts.values, color='#8e44ad')
            axes[chart_idx].set_yticks(range(len(formation_counts)))
            axes[chart_idx].set_yticklabels(formation_counts.index, fontsize=8)
            axes[chart_idx].set_xlabel('Count')
            axes[chart_idx].set_title('Top 10 Formations', fontweight='bold')
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
    # 10. Route analysis
    try:
        if 'route_of_targeted_receiver' in full_df.columns:
            route_counts = full_df['route_of_targeted_receiver'].value_counts().head(10)
            axes[chart_idx].barh(range(len(route_counts)), route_counts.values, color='#27ae60')
            axes[chart_idx].set_yticks(range(len(route_counts)))
            axes[chart_idx].set_yticklabels(route_counts.index, fontsize=8)
            axes[chart_idx].set_xlabel('Count')
            axes[chart_idx].set_title('Top 10 Routes', fontweight='bold')
        chart_idx += 1
    except:
        axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
        chart_idx += 1
    # 11-20. Additional strategic metrics
    strategic_columns = ['dropback_type', 'pass_location_type', 'receiver_alignment', 
                        'defenders_in_the_box', 'dropback_distance', 'penalty_yards',
                        'pre_penalty_yards_gained', 'home_final_score', 'visitor_final_score',
                        'pre_snap_home_score']
    
    for col in strategic_columns:
        if chart_idx >= 20:
            break
        try:
            if col in full_df.columns:
                data = full_df[col].dropna()
                if data.dtype in ['int64', 'float64']:
                    axes[chart_idx].hist(data, bins=30, edgecolor='black')
                    axes[chart_idx].set_xlabel(col.replace('_', ' ').title())
                    axes[chart_idx].set_ylabel('Frequency')
                else:
                    value_counts = data.value_counts().head(10)
                    axes[chart_idx].bar(range(len(value_counts)), value_counts.values)
                    axes[chart_idx].set_xticks(range(len(value_counts)))
                    axes[chart_idx].set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=8)
                    axes[chart_idx].set_ylabel('Count')
                axes[chart_idx].set_title(col.replace('_', ' ').title(), fontweight='bold', fontsize=10)
                axes[chart_idx].grid(True, alpha=0.3)
            chart_idx += 1
        except:
            axes[chart_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            chart_idx += 1
     # Hide unused subplots
    for i in range(chart_idx, 20):
        axes[i].axis('off')
    
    plt.suptitle('Play Outcome Analysis (20 Charts)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{eda_path}play_outcome_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("âœ“ Created play outcome visualizations")

    


#another gc again
gc.collect()


# --- Player Position Distribution ---
plt.figure(figsize=(10,6))
plt.style.use("seaborn-v0_8-darkgrid")

# Data Inspection
pos_counts = input_combined['player_position'].value_counts() / 1e5

sns.barplot(
    y=pos_counts.index,
    x=pos_counts.values,
    palette="mako",
    edgecolor="black"
)

plt.title("ğŸ�ˆ Player Position Distribution (All Weeks)", fontsize=16, fontweight='bold')
plt.xlabel("Cases (in Hundred Thousands)", fontsize=12)
plt.ylabel("Player Position", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# --- Speed & Acceleration Distribution Chart---
fig, ax = plt.subplots(1, 2, figsize=(14,6))
plt.style.use("seaborn-v0_8-muted")

# Speed Distribution
sns.histplot(input_combined['s'], bins=30, kde=True, color="#1f77b4", ax=ax[0], alpha=0.8)
ax[0].set_title("Speed Distribution (yards/sec)", fontsize=14, fontweight='bold')
ax[0].set_xlabel("Speed (yards/s)", fontsize=12)
ax[0].set_ylabel("Frequency", fontsize=12)
ax[0].grid(True, linestyle='--', alpha=0.5)

# Acceleration Distribution
sns.histplot(input_combined['a'], bins=30, kde=True, color="#ff7f0e", ax=ax[1], alpha=0.8)
ax[1].set_title("Acceleration Distribution (yards/secÂ²)", fontsize=14, fontweight='bold')
ax[1].set_xlabel("Acceleration (yards/secÂ²)", fontsize=12)
ax[1].set_ylabel("Frequency", fontsize=12)
ax[1].grid(True, linestyle='--', alpha=0.5)

plt.suptitle("âš¡ Speed and Acceleration Profiles", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# --- Field Position Plot ---
sample_play = input_combined[
    (input_combined['game_id'] == input_combined['game_id'].iloc[0]) &
    (input_combined['play_id'] == input_combined['play_id'].iloc[0])
]

plt.figure(figsize=(10,6))
plt.style.use("seaborn-v0_8-poster")

sns.scatterplot(
    data=sample_play,
    x='x', y='y',
    hue='player_side', style='player_role',
    s=90, palette='coolwarm', edgecolor='black', alpha=0.9
)

plt.xlim(0, 120)
plt.ylim(0, 53.3)
plt.xlabel("Long Axis (yards)", fontsize=12)
plt.ylabel("Short Axis (yards)", fontsize=12)
plt.title("ğŸ�Ÿï¸� Player Positions â€” Sample Play Snapshot", fontsize=16, fontweight='bold')

# Add field-like background
plt.gca().set_facecolor("#f7f7f7")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Player Side / Role", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#additoinal gc
gc.collect()


#data preparation
numerical_features = ['s', 'a', 'o', 'dir', 'x', 'y']


# Clean up NaNs in the numerical columns
input_combined = input_combined.dropna(subset=numerical_features).reset_index(drop=True)
print("Rows after dropna:", len(input_combined))


# convenience variable used by subsequent cells
_processed_input = input_combined[numerical_features].copy()


# Regulation/Controls
batch_size_tsne = 700         # unused in this simplified per-subsample approach, kept for parity
max_samples_for_tsne = 1500   # cap samples used for t-SNE to control runtime
random_state = 40
n_iter = 700             # t-SNE iterations (reduce if slow)


# Prepare scaled data and subsample for t-SNE
scaler = StandardScaler()
scaled_all = scaler.fit_transform(_processed_input.values)
n_use = min(len(scaled_all), max_samples_for_tsne)
scaled = scaled_all[:n_use]


# Compute t-SNE on the subsample (single run for simplicity / reproducibility)
if len(scaled) >= 2:
    perplexity = max(5, min(50, max(5, len(scaled) // 10)))
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, init='pca', n_iter=n_iter)
    tsne_result = tsne.fit_transform(scaled)
else:
    tsne_result = np.zeros((len(scaled), 2))


# Plot
plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=6, alpha=0.6)
plt.title('t-SNE (subsampled)', fontweight='bold')
plt.xlabel('TSNE-1'); plt.ylabel('TSNE-2')
plt.tight_layout()
plt.savefig('cluster_tsne.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"t-SNE saved to cluster_tsne.png (used {n_use} samples).")


# Controls
batch_size_kmeans = 900    # batch used in MiniBatchKMeans
optimal_k = 4              # number of clusters you want
pca_batch = 1000           # number of rows for IPCA partial_fit chunks (if needed)


# Scale input
scaler = StandardScaler()
scaled = scaler.fit_transform(_processed_input.values)


# Fit MiniBatchKMeans
mbk = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, batch_size=batch_size_kmeans)
mbk.fit(scaled)
kmeans_labels = mbk.predict(scaled)
kmeans_centers = mbk.cluster_centers_


# For visualization, project the scaled data to 2D using IncrementalPCA (memory-friendly)
ipca = IncrementalPCA(n_components=2)
n_batches = int(np.ceil(len(scaled) / pca_batch))
for i in range(n_batches):
    s = i * pca_batch
    e = min((i + 1) * pca_batch, len(scaled))
    ipca.partial_fit(scaled[s:e])

pca_result = np.zeros((len(scaled), 2))
for i in range(n_batches):
    s = i * pca_batch
    e = min((i + 1) * pca_batch, len(scaled))
    pca_result[s:e] = ipca.transform(scaled[s:e])

# Project cluster centers to PCA space for plotting
centers_pca = ipca.transform(kmeans_centers)

# Plot clusters (projected)
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, s=6, alpha=0.6)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='x', s=80, linewidths=2)  # cluster centers
plt.title(f'MiniBatchKMeans (k={optimal_k}) projected via IPCA', fontweight='bold')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('cluster_kmeans.png', dpi=150, bbox_inches='tight')
plt.show()
print("MiniBatchKMeans saved to cluster_kmeans.png")


# Controls
batch_size_dbscan = 700
eps = 0.5
min_samples = 5
pca_batch = 1000  # for projection


#batches in DBSCAN : Scale input
scaler = StandardScaler()
scaled = scaler.fit_transform(_processed_input.values)


#batch setting in Batch-based DBSCAN
n_batches = int(np.ceil(len(scaled) / batch_size_dbscan))
db_labels = np.full(len(scaled), -1, dtype=int)
cluster_offset = 0

for i in range(n_batches):
    s = i * batch_size_dbscan
    e = min((i + 1) * batch_size_dbscan, len(scaled))
    batch = scaled[s:e]
    if len(batch) == 0:
        continue
    db = DBSCAN(eps=eps, min_samples=min_samples)
    local_labels = db.fit_predict(batch)
    # shift local cluster ids (except noise = -1)
    if local_labels.max() != -1:
        local_positive = local_labels != -1
        local_labels[local_positive] = local_labels[local_positive] + cluster_offset
        cluster_offset += local_labels.max() + 1
    db_labels[s:e] = local_labels



# For plotting, project to 2D with IPCA
ipca = IncrementalPCA(n_components=2)
n_batches_pca = int(np.ceil(len(scaled) / pca_batch))
for i in range(n_batches_pca):
    s = i * pca_batch
    e = min((i + 1) * pca_batch, len(scaled))
    ipca.partial_fit(scaled[s:e])

pca_result = np.zeros((len(scaled), 2))
for i in range(n_batches_pca):
    s = i * pca_batch
    e = min((i + 1) * pca_batch, len(scaled))
    pca_result[s:e] = ipca.transform(scaled[s:e])

# Visualize DBSCAN labeling (noise is -1)
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=db_labels, s=6, alpha=0.6)
plt.title(f'Batch-based DBSCAN (eps={eps}, min_samples={min_samples})', fontweight='bold')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('cluster_dbscan.png', dpi=150, bbox_inches='tight')
plt.show()
print("Batch-based DBSCAN saved to cluster_dbscan.png")


# Controls
pca_batch = 1000
n_components = min(6, len(numerical_features))  # compute several PCs for scree


# Scale input
scaler = StandardScaler()
scaled = scaler.fit_transform(_processed_input.values)


# Fit Incremental PCA
ipca = IncrementalPCA(n_components=n_components)
n_batches = int(np.ceil(len(scaled) / pca_batch))
for i in range(n_batches):
    s = i * pca_batch
    e = min((i + 1) * pca_batch, len(scaled))
    ipca.partial_fit(scaled[s:e])


# Screening the plot
explained = ipca.explained_variance_ratio_
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(explained) + 1), explained, marker='o')
plt.title('Incremental PCA Scree Plot', fontweight='bold')
plt.xlabel('PC #'); plt.ylabel('Explained Variance Ratio')
plt.tight_layout()
plt.savefig('pca_scree.png', dpi=150, bbox_inches='tight')
plt.show()
print("PCA scree saved to pca_scree.png")

# PC1 feature importance (absolute loadings)
pc1_loadings = np.abs(ipca.components_[0])
plt.figure(figsize=(8, 4))
plt.bar(range(len(numerical_features)), pc1_loadings)
plt.xticks(range(len(numerical_features)), numerical_features, rotation=45)
plt.title('PC1 Absolute Loadings', fontweight='bold')
plt.tight_layout()
plt.savefig('pca_pc1_loadings.png', dpi=150, bbox_inches='tight')
plt.show()
print("PC1 loadings saved to pca_pc1_loadings.png")

