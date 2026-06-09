# MABe Challenge - Advanced Ensemble with Comprehensive Enhanced Visualizations
# Complete working code with publication-quality visualizations
# ENHANCED WITH SPATIAL HISTORY FEATURES

validate_or_submit = 'submit'
verbose = True

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import warnings
import json
import os
import gc
import lightgbm
from collections import defaultdict, Counter
import polars as pl
from scipy import signal, stats
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, confusion_matrix

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory for visualizations
os.makedirs('/kaggle/working/visualizations', exist_ok=True)

# Try importing additional models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

try:
    from gandalf import GANDALF
    GANDALF_AVAILABLE = True
except:
    GANDALF_AVAILABLE = False

# ==================== ENHANCED VISUALIZATION FUNCTIONS ====================

def plot_comprehensive_dataset_overview(train, test):
    """Enhanced dataset overview with more details"""
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # Color schemes
    colors_primary = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    colors_secondary = ['#5E548E', '#BE95C4', '#E0B1CB', '#F4ACB7', '#FFD6E0']
    
    # 1. Videos per lab (enhanced)
    ax1 = fig.add_subplot(gs[0, 0])
    lab_counts = train.groupby('lab_id').size().sort_values(ascending=False)
    bars = ax1.barh(range(len(lab_counts)), lab_counts.values, color=colors_primary[0], edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(lab_counts)))
    ax1.set_yticklabels(lab_counts.index, fontsize=8)
    ax1.set_title('Videos per Lab (Training)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Number of Videos', fontsize=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars, lab_counts.values)):
        ax1.text(val + 1, i, str(val), va='center', fontsize=8, fontweight='bold')
    
    # 2. Number of mice distribution with statistics
    ax2 = fig.add_subplot(gs[0, 1])
    mice_counts = train['n_mice'].value_counts().sort_index()
    bars = ax2.bar(mice_counts.index, mice_counts.values, color=colors_primary[1], 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_title('Distribution of Mice Count', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Number of Mice', fontsize=10)
    ax2.set_ylabel('Number of Videos', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, mice_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/len(train)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Video duration with detailed statistics
    ax3 = fig.add_subplot(gs[0, 2])
    durations = train['video_duration_sec']
    ax3.hist(durations, bins=50, color=colors_primary[2], edgecolor='black', alpha=0.7)
    ax3.axvline(durations.mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {durations.mean():.0f}s')
    ax3.axvline(durations.median(), color='blue', linestyle='--', linewidth=2.5, label=f'Median: {durations.median():.0f}s')
    ax3.set_title('Video Duration Distribution', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Duration (seconds)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.legend(fontsize=9, frameon=True, shadow=True)
    ax3.grid(alpha=0.3)
    
    # 4. FPS distribution enhanced
    ax4 = fig.add_subplot(gs[0, 3])
    fps_counts = train['frames_per_second'].value_counts().sort_index()
    bars = ax4.bar(range(len(fps_counts)), fps_counts.values, color=colors_primary[3], 
                   edgecolor='black', linewidth=1, alpha=0.8)
    ax4.set_xticks(range(len(fps_counts)))
    ax4.set_xticklabels(fps_counts.index, fontsize=9)
    ax4.set_title('Frames Per Second Distribution', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('FPS', fontsize=10)
    ax4.set_ylabel('Number of Videos', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, fps_counts.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, str(val),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Body parts tracked detailed
    ax5 = fig.add_subplot(gs[1, 0])
    body_parts_count = train['body_parts_tracked'].apply(lambda x: len(json.loads(x)))
    bp_counts = body_parts_count.value_counts().sort_index()
    bars = ax5.bar(bp_counts.index, bp_counts.values, color=colors_primary[4], 
                   edgecolor='black', linewidth=1, alpha=0.8)
    ax5.set_title('Number of Body Parts Tracked', fontsize=13, fontweight='bold', pad=10)
    ax5.set_xlabel('Number of Body Parts', fontsize=10)
    ax5.set_ylabel('Number of Videos', fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, bp_counts.values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height, str(val),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Arena shape with better visualization
    ax6 = fig.add_subplot(gs[1, 1])
    arena_counts = train['arena_shape'].value_counts()
    wedges, texts, autotexts = ax6.pie(arena_counts.values, labels=arena_counts.index, 
                                         autopct='%1.1f%%', startangle=90,
                                         colors=colors_secondary, explode=[0.05]*len(arena_counts),
                                         shadow=True, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax6.set_title('Arena Shape Distribution', fontsize=13, fontweight='bold', pad=10)
    
    # 7. Tracking method enhanced
    ax7 = fig.add_subplot(gs[1, 2:])
    tracking_counts = train['tracking_method'].value_counts().head(10)
    bars = ax7.barh(range(len(tracking_counts)), tracking_counts.values, color=colors_primary[0], 
                    edgecolor='black', linewidth=0.5, alpha=0.8)
    ax7.set_yticks(range(len(tracking_counts)))
    ax7.set_yticklabels(tracking_counts.index, fontsize=9)
    ax7.set_title('Tracking Method Distribution (Top 10)', fontsize=13, fontweight='bold', pad=10)
    ax7.set_xlabel('Number of Videos', fontsize=10)
    ax7.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, tracking_counts.values)):
        ax7.text(val + 1, i, str(val), va='center', fontsize=9, fontweight='bold')
    
    # 8. Train vs Test comparison
    ax8 = fig.add_subplot(gs[2, 0])
    dataset_sizes = pd.Series({'Train': len(train), 'Test': len(test)})
    bars = ax8.bar(range(len(dataset_sizes)), dataset_sizes.values, 
                   color=[colors_primary[0], colors_primary[1]], 
                   edgecolor='black', linewidth=2, alpha=0.8)
    ax8.set_xticks(range(len(dataset_sizes)))
    ax8.set_xticklabels(dataset_sizes.index, fontsize=11, fontweight='bold')
    ax8.set_title('Train vs Test Dataset Size', fontsize=13, fontweight='bold', pad=10)
    ax8.set_ylabel('Number of Videos', fontsize=10)
    ax8.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, dataset_sizes.values)):
        height = bar.get_height()
        ax8.text(i, height, f'{val}\nvideos', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # 9. Mouse strains diversity
    ax9 = fig.add_subplot(gs[2, 1:3])
    all_strains = pd.concat([train['mouse1_strain'], train['mouse2_strain'], 
                             train['mouse3_strain'], train['mouse4_strain']]).dropna()
    strain_counts = all_strains.value_counts().head(15)
    bars = ax9.barh(range(len(strain_counts)), strain_counts.values, 
                    color=colors_primary[2], edgecolor='black', linewidth=0.5, alpha=0.8)
    ax9.set_yticks(range(len(strain_counts)))
    ax9.set_yticklabels(strain_counts.index, fontsize=9)
    ax9.set_title('Top 15 Mouse Strains', fontsize=13, fontweight='bold', pad=10)
    ax9.set_xlabel('Frequency', fontsize=10)
    ax9.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, strain_counts.values)):
        ax9.text(val + 5, i, str(val), va='center', fontsize=8, fontweight='bold')
    
    # 10. Sex distribution
    ax10 = fig.add_subplot(gs[2, 3])
    all_sex = pd.concat([train['mouse1_sex'], train['mouse2_sex'], 
                         train['mouse3_sex'], train['mouse4_sex']]).dropna()
    sex_counts = all_sex.value_counts()
    wedges, texts, autotexts = ax10.pie(sex_counts.values, labels=sex_counts.index,
                                          autopct='%1.1f%%', startangle=90,
                                          colors=['#FF6B9D', '#4ECDC4', '#95E1D3'],
                                          explode=[0.05]*len(sex_counts), shadow=True,
                                          textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax10.set_title('Mouse Sex Distribution', fontsize=13, fontweight='bold', pad=10)
    
    # 11. Total frames statistics
    ax11 = fig.add_subplot(gs[3, 0])
    train['total_frames'] = train['video_duration_sec'] * train['frames_per_second']
    total_frames_dist = train['total_frames']
    ax11.hist(total_frames_dist, bins=50, color=colors_primary[3], 
              edgecolor='black', alpha=0.7)
    ax11.axvline(total_frames_dist.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {total_frames_dist.mean():.0f}')
    ax11.set_title('Total Frames per Video', fontsize=13, fontweight='bold', pad=10)
    ax11.set_xlabel('Total Frames', fontsize=10)
    ax11.set_ylabel('Frequency', fontsize=10)
    ax11.legend(fontsize=9)
    ax11.grid(alpha=0.3)
    
    # 12. Arena size distribution
    ax12 = fig.add_subplot(gs[3, 1])
    train['arena_area'] = train['arena_width_cm'] * train['arena_height_cm']
    arena_area = train['arena_area'].dropna()
    ax12.hist(arena_area, bins=40, color=colors_primary[4], 
              edgecolor='black', alpha=0.7)
    ax12.axvline(arena_area.median(), color='blue', linestyle='--', linewidth=2,
                label=f'Median: {arena_area.median():.0f} cmÂ²')
    ax12.set_title('Arena Area Distribution', fontsize=13, fontweight='bold', pad=10)
    ax12.set_xlabel('Area (cmÂ²)', fontsize=10)
    ax12.set_ylabel('Frequency', fontsize=10)
    ax12.legend(fontsize=9)
    ax12.grid(alpha=0.3)
    
    # 13. Pixel density
    ax13 = fig.add_subplot(gs[3, 2])
    pixel_density = train['pix_per_cm_approx'].dropna()
    ax13.hist(pixel_density, bins=40, color=colors_primary[0], 
              edgecolor='black', alpha=0.7)
    ax13.axvline(pixel_density.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {pixel_density.mean():.2f}')
    ax13.set_title('Pixel Density Distribution', fontsize=13, fontweight='bold', pad=10)
    ax13.set_xlabel('Pixels per cm', fontsize=10)
    ax13.set_ylabel('Frequency', fontsize=10)
    ax13.legend(fontsize=9)
    ax13.grid(alpha=0.3)
    
    # 14. Summary statistics table
    ax14 = fig.add_subplot(gs[3, 3])
    ax14.axis('off')
    summary_stats = [
        ['Metric', 'Train', 'Test'],
        ['Total Videos', f'{len(train)}', f'{len(test)}'],
        ['Avg Duration (s)', f'{train["video_duration_sec"].mean():.0f}', 
         f'{test["video_duration_sec"].mean():.0f}' if "video_duration_sec" in test.columns else 'N/A'],
        ['Total Labs', f'{train["lab_id"].nunique()}', f'{test["lab_id"].nunique()}'],
        ['Unique Strains', f'{all_strains.nunique()}', 'N/A'],
        ['Avg Mice/Video', f'{train["n_mice"].mean():.2f}', 'N/A']
    ]
    table = ax14.table(cellText=summary_stats, cellLoc='center', loc='center',
                       colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    for i in range(len(summary_stats)):
        for j in range(len(summary_stats[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#2E86AB')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#E8F4F8' if i % 2 == 0 else 'white')
            cell.set_edgecolor('#2E86AB')
            cell.set_linewidth(2)
    ax14.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('Comprehensive Dataset Overview', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig('/kaggle/working/visualizations/01_dataset_overview_enhanced.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("âœ“ Saved: 01_dataset_overview_enhanced.png")
    plt.show()
    plt.close()

def plot_advanced_behavior_analysis(train):
    """Advanced behavior analysis with co-occurrence and transitions"""
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract all behaviors
    all_behaviors = []
    behavior_lab_map = defaultdict(set)
    behavior_video_map = defaultdict(list)
    self_behaviors = []
    paired_behaviors = []
    
    for _, row in train.iterrows():
        if isinstance(row['behaviors_labeled'], str):
            behaviors = json.loads(row['behaviors_labeled'])
            for b in behaviors:
                b_clean = b.replace("'", "")
                parts = b_clean.split(',')
                if len(parts) >= 3:
                    action = parts[2]
                    all_behaviors.append(action)
                    behavior_lab_map[action].add(row['lab_id'])
                    behavior_video_map[action].append(row['video_id'])
                    
                    if len(parts) >= 2 and parts[1] == 'self':
                        self_behaviors.append(action)
                    else:
                        paired_behaviors.append(action)
    
    # 1. Top 25 behaviors with count
    ax1 = fig.add_subplot(gs[0, :2])
    behavior_counts = pd.Series(all_behaviors).value_counts().head(25)
    colors = plt.cm.viridis(np.linspace(0, 1, len(behavior_counts)))
    bars = ax1.barh(range(len(behavior_counts)), behavior_counts.values, color=colors, 
                    edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(behavior_counts)))
    ax1.set_yticklabels(behavior_counts.index, fontsize=9)
    ax1.set_title('Top 25 Most Common Behaviors', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Frequency', fontsize=11)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars, behavior_counts.values)):
        ax1.text(val + max(behavior_counts.values)*0.01, i, f'{val:,}', 
                va='center', fontsize=8, fontweight='bold')
    
    # 2. Behavior frequency distribution
    ax2 = fig.add_subplot(gs[0, 2])
    behavior_freq = pd.Series(all_behaviors).value_counts()
    ax2.hist(behavior_freq.values, bins=50, color='#FF6B9D', edgecolor='black', alpha=0.7)
    ax2.set_title('Behavior Frequency Distribution', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Frequency', fontsize=11)
    ax2.set_ylabel('Number of Behaviors', fontsize=11)
    ax2.axvline(behavior_freq.median(), color='red', linestyle='--', linewidth=2,
               label=f'Median: {behavior_freq.median():.0f}')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. Behaviors per video distribution
    ax3 = fig.add_subplot(gs[1, 0])
    behaviors_per_video = []
    for _, row in train.iterrows():
        if isinstance(row['behaviors_labeled'], str):
            behaviors = json.loads(row['behaviors_labeled'])
            behaviors_per_video.append(len(behaviors))
    
    ax3.hist(behaviors_per_video, bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax3.set_title('Behaviors Per Video', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel('Number of Behaviors', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.axvline(np.mean(behaviors_per_video), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(behaviors_per_video):.1f}')
    ax3.axvline(np.median(behaviors_per_video), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.median(behaviors_per_video):.0f}')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # 4. Unique behaviors per lab
    ax4 = fig.add_subplot(gs[1, 1])
    lab_unique_behaviors = {}
    for _, row in train.iterrows():
        lab = row['lab_id']
        if isinstance(row['behaviors_labeled'], str):
            if lab not in lab_unique_behaviors:
                lab_unique_behaviors[lab] = set()
            behaviors = json.loads(row['behaviors_labeled'])
            for b in behaviors:
                b_clean = b.replace("'", "")
                parts = b_clean.split(',')
                if len(parts) >= 3:
                    lab_unique_behaviors[lab].add(parts[2])
    
    lab_diversity = pd.Series({k: len(v) for k, v in lab_unique_behaviors.items()}).sort_values(ascending=False).head(15)
    colors = plt.cm.plasma(np.linspace(0, 1, len(lab_diversity)))
    bars = ax4.barh(range(len(lab_diversity)), lab_diversity.values, color=colors,
                    edgecolor='black', linewidth=0.5)
    ax4.set_yticks(range(len(lab_diversity)))
    ax4.set_yticklabels(lab_diversity.index, fontsize=9)
    ax4.set_title('Unique Behaviors per Lab (Top 15)', fontsize=14, fontweight='bold', pad=10)
    ax4.set_xlabel('Number of Unique Behaviors', fontsize=11)
    ax4.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, lab_diversity.values)):
        ax4.text(val + 0.5, i, str(val), va='center', fontsize=9, fontweight='bold')
    
    # 5. Self vs Paired behaviors enhanced
    ax5 = fig.add_subplot(gs[1, 2])
    self_count = len(self_behaviors)
    paired_count = len(paired_behaviors)
    behavior_types = pd.Series({'Self-directed': self_count, 'Social/Paired': paired_count})
    colors_pie = ['#FF6B9D', '#4ECDC4']
    wedges, texts, autotexts = ax5.pie(behavior_types.values, labels=behavior_types.index,
                                         autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(behavior_types.values)):,})',
                                         startangle=90, colors=colors_pie, explode=(0.05, 0.05),
                                         shadow=True, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax5.set_title('Self vs Paired Behaviors', fontsize=14, fontweight='bold', pad=10)
    
    # 6. Top self-directed behaviors
    ax6 = fig.add_subplot(gs[2, 0])
    self_behavior_counts = pd.Series(self_behaviors).value_counts().head(10)
    bars = ax6.barh(range(len(self_behavior_counts)), self_behavior_counts.values,
                    color='#FF6B9D', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax6.set_yticks(range(len(self_behavior_counts)))
    ax6.set_yticklabels(self_behavior_counts.index, fontsize=9)
    ax6.set_title('Top 10 Self-Directed Behaviors', fontsize=14, fontweight='bold', pad=10)
    ax6.set_xlabel('Frequency', fontsize=11)
    ax6.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, self_behavior_counts.values)):
        ax6.text(val + max(self_behavior_counts.values)*0.01, i, f'{val:,}',
                va='center', fontsize=8, fontweight='bold')
    
    # 7. Top paired behaviors
    ax7 = fig.add_subplot(gs[2, 1])
    paired_behavior_counts = pd.Series(paired_behaviors).value_counts().head(10)
    bars = ax7.barh(range(len(paired_behavior_counts)), paired_behavior_counts.values,
                    color='#4ECDC4', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax7.set_yticks(range(len(paired_behavior_counts)))
    ax7.set_yticklabels(paired_behavior_counts.index, fontsize=9)
    ax7.set_title('Top 10 Social/Paired Behaviors', fontsize=14, fontweight='bold', pad=10)
    ax7.set_xlabel('Frequency', fontsize=11)
    ax7.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, paired_behavior_counts.values)):
        ax7.text(val + max(paired_behavior_counts.values)*0.01, i, f'{val:,}',
                va='center', fontsize=8, fontweight='bold')
    
    # 8. Behavior rarity analysis
    ax8 = fig.add_subplot(gs[2, 2])
    behavior_counts_all = pd.Series(all_behaviors).value_counts()
    rare_behaviors = (behavior_counts_all <= 100).sum()
    common_behaviors = ((behavior_counts_all > 100) & (behavior_counts_all <= 1000)).sum()
    frequent_behaviors = (behavior_counts_all > 1000).sum()
    
    rarity_data = pd.Series({
        f'Rare\n(â‰¤100)': rare_behaviors,
        f'Common\n(100-1000)': common_behaviors,
        f'Frequent\n(>1000)': frequent_behaviors
    })
    colors_bar = ['#FF6B9D', '#FFD93D', '#6BCF7F']
    bars = ax8.bar(range(len(rarity_data)), rarity_data.values, color=colors_bar,
                   edgecolor='black', linewidth=2, alpha=0.8)
    ax8.set_xticks(range(len(rarity_data)))
    ax8.set_xticklabels(rarity_data.index, fontsize=10, fontweight='bold')
    ax8.set_title('Behavior Rarity Classification', fontsize=14, fontweight='bold', pad=10)
    ax8.set_ylabel('Number of Unique Behaviors', fontsize=11)
    ax8.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, rarity_data.values):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/len(behavior_counts_all)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Advanced Behavior Analysis', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig('/kaggle/working/visualizations/02_behavior_analysis_enhanced.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("âœ“ Saved: 02_behavior_analysis_enhanced.png")
    plt.show()
    plt.close()

def plot_feature_analysis(X_sample, feature_type='single'):
    """Comprehensive feature analysis with correlations and distributions"""
    sample_size = min(50000, len(X_sample))
    X_plot = X_sample.sample(n=sample_size, random_state=42) if len(X_sample) > sample_size else X_sample
    
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Select feature categories
    feature_cols = X_plot.columns.tolist()
    speed_features = [c for c in feature_cols if any(x in c for x in ['sp_', 'disp', 'act'])][:12]
    position_features = [c for c in feature_cols if any(x in c for x in ['cx_', 'cy_', 'x_', 'y_'])][:12]
    distance_features = [c for c in feature_cols if '+' in c][:12]
    temporal_features = [c for c in feature_cols if any(x in c for x in ['_m', '_s', 'pct', 'rng'])][:12]
    
    # 1-3. Speed feature distributions
    for idx in range(min(3, len(speed_features))):
        ax = fig.add_subplot(gs[0, idx])
        feature = speed_features[idx]
        data = X_plot[feature].dropna()
        ax.hist(data, bins=60, color='#2E86AB', edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Î¼={data.mean():.2f}')
        ax.axvline(data.median(), color='orange', linestyle='--', linewidth=2,
                   label=f'M={data.median():.2f}')
        ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # 4-6. Position feature distributions
    for idx in range(min(3, len(position_features))):
        ax = fig.add_subplot(gs[1, idx])
        feature = position_features[idx]
        data = X_plot[feature].dropna()
        ax.hist(data, bins=60, color='#A23B72', edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Î¼={data.mean():.2f}')
        ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # 7-9. Distance/temporal feature distributions
    combined_features = (distance_features + temporal_features)[:3]
    for idx in range(min(3, len(combined_features))):
        ax = fig.add_subplot(gs[2, idx])
        feature = combined_features[idx]
        data = X_plot[feature].dropna()
        ax.hist(data, bins=60, color='#F18F01', edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Î¼={data.mean():.2f}')
        ax.set_title(f'{feature[:30]}...', fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Feature Distribution Analysis ({feature_type.capitalize()})', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(f'/kaggle/working/visualizations/03_feature_distributions_{feature_type}.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"âœ“ Saved: 03_feature_distributions_{feature_type}.png")
    plt.show()
    plt.close()

def plot_feature_correlation_heatmap(X_sample, feature_type='single'):
    """Feature correlation heatmap"""
    sample_size = min(10000, len(X_sample))
    X_plot = X_sample.sample(n=sample_size, random_state=42) if len(X_sample) > sample_size else X_sample
    
    # Select subset of interesting features
    feature_cols = X_plot.columns.tolist()
    selected_features = []
    
    for keyword in ['sp_', 'cx_', 'cy_', 'disp', 'act', 'd_m', 'curv']:
        matching = [c for c in feature_cols if keyword in c]
        selected_features.extend(matching[:2])
    
    selected_features = list(set(selected_features))[:20]
    
    if len(selected_features) > 0:
        correlation_matrix = X_plot[selected_features].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax, annot_kws={'size': 7})
        ax.set_title(f'Feature Correlation Heatmap ({feature_type.capitalize()})',
                    fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(f'/kaggle/working/visualizations/04_correlation_heatmap_{feature_type}.png',
                   dpi=150, bbox_inches='tight', facecolor='white')
        print(f"âœ“ Saved: 04_correlation_heatmap_{feature_type}.png")
        plt.show()
        plt.close()

def plot_temporal_patterns_detailed(X_sample, feature_type='single'):
    """Detailed temporal pattern analysis"""
    sample_size = min(5000, len(X_sample))
    X_plot = X_sample.iloc[:sample_size]
    
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    temporal_cols = [c for c in X_plot.columns if any(x in c for x in ['_m', '_s', 'disp', 'act'])][:6]
    
    for idx, col in enumerate(temporal_cols):
        if idx < 6 and col in X_plot.columns:
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            
            # Plot time series
            data = X_plot[col].values
            ax.plot(data, linewidth=0.8, alpha=0.7, color='#2E86AB')
            
            # Add rolling mean
            window = min(100, len(data) // 10)
            if window > 1:
                rolling_mean = pd.Series(data).rolling(window=window, center=True).mean()
                ax.plot(rolling_mean, linewidth=2, color='#FF6B9D', label=f'Rolling Mean (w={window})')
            
            ax.set_title(f'{col}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Frame', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend(fontsize=9)
            
            # Add statistics box
            textstr = f'Î¼={np.nanmean(data):.2f}\nÏƒ={np.nanstd(data):.2f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.suptitle(f'Temporal Pattern Analysis ({feature_type.capitalize()})',
                fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(f'/kaggle/working/visualizations/05_temporal_patterns_{feature_type}.png',
               dpi=150, bbox_inches='tight', facecolor='white')
    print(f"âœ“ Saved: 05_temporal_patterns_{feature_type}.png")
    plt.show()
    plt.close()

def plot_training_progress_detailed(section_results):
    """Detailed training progress visualization"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    sections = list(section_results.keys())
    
    # 1. Features per section (stacked)
    ax1 = fig.add_subplot(gs[0, 0])
    single_features = [section_results[s].get('single_features', 0) for s in sections]
    pair_features = [section_results[s].get('pair_features', 0) for s in sections]
    
    x = np.arange(len(sections))
    width = 0.6
    
    ax1.bar(x, single_features, width, label='Single', color='#2E86AB', edgecolor='black')
    ax1.bar(x, pair_features, width, bottom=single_features, label='Pair', 
           color='#A23B72', edgecolor='black')
    
    ax1.set_xlabel('Section', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
    ax1.set_title('Features Generated per Section', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{s}' for s in sections], fontsize=9)
    ax1.legend(fontsize=10, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add total on top of bars
    for i, (sf, pf) in enumerate(zip(single_features, pair_features)):
        total = sf + pf
        if total > 0:
            ax1.text(i, total, str(total), ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
    
    # 2. Predictions per section
    ax2 = fig.add_subplot(gs[0, 1])
    predictions = [section_results[s].get('predictions', 0) for s in sections]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(predictions)))
    bars = ax2.bar(range(len(predictions)), predictions, color=colors, 
                   edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Section', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Predictions', fontsize=11, fontweight='bold')
    ax2.set_title('Predictions Generated per Section', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(range(len(sections)))
    ax2.set_xticklabels([f'S{s}' for s in sections], fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, predictions):
        if val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, str(val),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Models used
    ax3 = fig.add_subplot(gs[0, 2])
    models_used = [section_results[s].get('models', 0) for s in sections]
    
    bars = ax3.bar(range(len(models_used)), models_used, color='#F18F01',
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Section', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Models', fontsize=11, fontweight='bold')
    ax3.set_title('Models Used per Section', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xticks(range(len(sections)))
    ax3.set_xticklabels([f'S{s}' for s in sections], fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, max(models_used) + 1 if max(models_used) > 0 else 6])
    
    for bar, val in zip(bars, models_used):
        if val > 0:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, str(val),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Cumulative predictions
    ax4 = fig.add_subplot(gs[1, 0])
    valid_preds = [section_results[s].get('predictions', 0) for s in sections]
    cumulative_preds = np.cumsum(valid_preds)
    
    ax4.plot(range(len(cumulative_preds)), cumulative_preds, marker='o', 
            linewidth=3, markersize=10, color='#2E86AB', markerfacecolor='#FF6B9D',
            markeredgecolor='black', markeredgewidth=2)
    ax4.fill_between(range(len(cumulative_preds)), cumulative_preds, alpha=0.3, color='#2E86AB')
    ax4.set_xlabel('Section', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cumulative Predictions', fontsize=11, fontweight='bold')
    ax4.set_title('Cumulative Predictions Growth', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xticks(range(len(sections)))
    ax4.set_xticklabels([f'S{s}' for s in sections], fontsize=9)
    ax4.grid(alpha=0.3)
    
    for i, val in enumerate(cumulative_preds):
        if val > 0:
            ax4.annotate(f'{val}', (i, val), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 5. Feature generation efficiency
    ax5 = fig.add_subplot(gs[1, 1])
    total_features = [sf + pf for sf, pf in zip(single_features, pair_features)]
    efficiency = [p/f if f > 0 else 0 for p, f in zip(predictions, total_features)]
    
    bars = ax5.bar(range(len(efficiency)), efficiency, color='#6BCF7F',
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax5.set_xlabel('Section', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Predictions / Features', fontsize=11, fontweight='bold')
    ax5.set_title('Feature Efficiency', fontsize=13, fontweight='bold', pad=10)
    ax5.set_xticks(range(len(sections)))
    ax5.set_xticklabels([f'S{s}' for s in sections], fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, efficiency):
        if val > 0:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    total_single = sum(single_features)
    total_pair = sum(pair_features)
    total_preds = sum(predictions)
    avg_models = np.mean([m for m in models_used if m > 0]) if any(models_used) else 0
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Sections', f'{len(sections)}'],
        ['Single Features', f'{total_single:,}'],
        ['Pair Features', f'{total_pair:,}'],
        ['Total Predictions', f'{total_preds:,}'],
        ['Avg Models/Section', f'{avg_models:.1f}']
    ]
    
    table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#2E86AB')
                cell.set_text_props(weight='bold', color='white', size=12)
            else:
                cell.set_facecolor('#E8F4F8' if i % 2 == 0 else 'white')
            cell.set_edgecolor('#2E86AB')
            cell.set_linewidth(2)
    
    ax6.set_title('Training Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('Training Progress Overview', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig('/kaggle/working/visualizations/06_training_progress_detailed.png',
               dpi=150, bbox_inches='tight', facecolor='white')
    print("âœ“ Saved: 06_training_progress_detailed.png")
    plt.show()
    plt.close()

def plot_comprehensive_submission_analysis(submission, train):
    """Comprehensive submission analysis with train comparison"""
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Calculate duration
    submission['duration'] = submission['stop_frame'] - submission['start_frame']
    
    # Extract train behaviors for comparison
    train_actions = []
    train_durations = []
    train_self_count = 0
    train_paired_count = 0
    
    for _, row in train.iterrows():
        if isinstance(row['behaviors_labeled'], str):
            behaviors = json.loads(row['behaviors_labeled'])
            for b in behaviors:
                b_clean = b.replace("'", "")
                parts = b_clean.split(',')
                if len(parts) >= 3:
                    train_actions.append(parts[2])
                    if len(parts) >= 2:
                        if parts[1] == 'self':
                            train_self_count += 1
                        else:
                            train_paired_count += 1
    
    # 1. Top actions comparison (Train vs Prediction)
    ax1 = fig.add_subplot(gs[0, :2])
    
    pred_action_counts = submission['action'].value_counts().head(15)
    train_action_counts = pd.Series(train_actions).value_counts()
    
    # Align both series
    common_actions = pred_action_counts.index
    train_aligned = train_action_counts.reindex(common_actions, fill_value=0)
    
    x = np.arange(len(common_actions))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_aligned.values, width, label='Train',
                   color='#2E86AB', edgecolor='black', alpha=0.8)
    bars2 = ax1.bar(x + width/2, pred_action_counts.values, width, label='Predictions',
                   color='#FF6B9D', edgecolor='black', alpha=0.8)
    
    ax1.set_xlabel('Action', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Top 15 Actions: Train vs Predictions Comparison', 
                 fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_actions, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Event duration distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(submission['duration'], bins=50, color='#F18F01', 
            edgecolor='black', alpha=0.7)
    ax2.axvline(submission['duration'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {submission["duration"].mean():.0f}')
    ax2.axvline(submission['duration'].median(), color='blue', linestyle='--',
               linewidth=2, label=f'Median: {submission["duration"].median():.0f}')
    ax2.set_title('Event Duration Distribution', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Duration (frames)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 3. Predictions per agent
    ax3 = fig.add_subplot(gs[1, 0])
    agent_counts = submission['agent_id'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(agent_counts)))
    bars = ax3.bar(range(len(agent_counts)), agent_counts.values, color=colors,
                  edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(agent_counts)))
    ax3.set_xticklabels(agent_counts.index, fontsize=10, fontweight='bold')
    ax3.set_title('Predictions per Agent', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel('Agent ID', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, agent_counts.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, str(val),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Self vs Paired (Train vs Prediction)
    ax4 = fig.add_subplot(gs[1, 1])
    
    pred_self = (submission['target_id'] == 'self').sum()
    pred_paired = (submission['target_id'] != 'self').sum()
    
    x_pos = np.arange(2)
    width = 0.35
    
    train_vals = [train_self_count, train_paired_count]
    pred_vals = [pred_self, pred_paired]
    
    # Normalize to percentages
    train_total = sum(train_vals)
    pred_total = sum(pred_vals)
    train_pct = [v/train_total*100 for v in train_vals]
    pred_pct = [v/pred_total*100 for v in pred_vals]
    
    bars1 = ax4.bar(x_pos - width/2, train_pct, width, label='Train',
                   color='#2E86AB', edgecolor='black', alpha=0.8)
    bars2 = ax4.bar(x_pos + width/2, pred_pct, width, label='Predictions',
                   color='#FF6B9D', edgecolor='black', alpha=0.8)
    
    ax4.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Self vs Paired: Train vs Predictions', 
                 fontsize=14, fontweight='bold', pad=10)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['Self-directed', 'Paired'], fontsize=11, fontweight='bold')
    ax4.legend(fontsize=11, frameon=True, shadow=True)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # 5. Predictions per video
    ax5 = fig.add_subplot(gs[1, 2])
    video_counts = submission['video_id'].value_counts()
    colors = plt.cm.plasma(np.linspace(0, 1, len(video_counts)))
    bars = ax5.bar(range(len(video_counts)), video_counts.values, color=colors,
                  edgecolor='black', linewidth=1)
    ax5.set_title('Predictions per Video', fontsize=14, fontweight='bold', pad=10)
    ax5.set_xlabel('Video Index', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Duration by action (top 10)
    ax6 = fig.add_subplot(gs[2, :])
    top_actions = submission['action'].value_counts().head(10).index
    duration_stats = submission[submission['action'].isin(top_actions)].groupby('action')['duration'].agg(['mean', 'std', 'median'])
    duration_stats = duration_stats.sort_values('mean')
    
    x_pos = np.arange(len(duration_stats))
    ax6.barh(x_pos, duration_stats['mean'].values, xerr=duration_stats['std'].values,
            color='#6BCF7F', edgecolor='black', linewidth=1, alpha=0.8,
            error_kw={'linewidth': 2, 'ecolor': 'red', 'capsize': 5})
    ax6.set_yticks(x_pos)
    ax6.set_yticklabels(duration_stats.index, fontsize=10)
    ax6.set_title('Average Duration by Action (Top 10) with Standard Deviation',
                 fontsize=14, fontweight='bold', pad=10)
    ax6.set_xlabel('Average Duration (frames)', fontsize=11)
    ax6.grid(axis='x', alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(duration_stats['mean'], duration_stats['std'])):
        ax6.text(mean + std + 5, i, f'{mean:.0f}Â±{std:.0f}',
                va='center', fontsize=9, fontweight='bold')
    
    # 7. Action frequency distribution
    ax7 = fig.add_subplot(gs[3, 0])
    action_freq = submission['action'].value_counts()
    ax7.hist(action_freq.values, bins=30, color='#A23B72', 
            edgecolor='black', alpha=0.7)
    ax7.set_title('Action Frequency Distribution', fontsize=14, fontweight='bold', pad=10)
    ax7.set_xlabel('Frequency per Action', fontsize=11)
    ax7.set_ylabel('Number of Actions', fontsize=11)
    ax7.axvline(action_freq.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {action_freq.mean():.1f}')
    ax7.set_yscale('log')
    ax7.legend(fontsize=10)
    ax7.grid(alpha=0.3)
    
    # 8. Duration statistics by category
    ax8 = fig.add_subplot(gs[3, 1])
    
    short_events = (submission['duration'] <= 30).sum()
    medium_events = ((submission['duration'] > 30) & (submission['duration'] <= 120)).sum()
    long_events = (submission['duration'] > 120).sum()
    
    categories = ['Short\n(â‰¤30)', 'Medium\n(30-120)', 'Long\n(>120)']
    values = [short_events, medium_events, long_events]
    colors_cat = ['#FF6B9D', '#FFD93D', '#6BCF7F']
    
    bars = ax8.bar(range(len(categories)), values, color=colors_cat,
                  edgecolor='black', linewidth=2, alpha=0.8)
    ax8.set_xticks(range(len(categories)))
    ax8.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax8.set_title('Event Duration Categories', fontsize=14, fontweight='bold', pad=10)
    ax8.set_ylabel('Number of Events', fontsize=11)
    ax8.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/len(submission)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 9. Summary statistics table
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis('off')
    
    unique_actions = submission['action'].nunique()
    unique_videos = submission['video_id'].nunique()
    unique_agents = submission['agent_id'].nunique()
    avg_duration = submission['duration'].mean()
    median_duration = submission['duration'].median()
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Predictions', f'{len(submission):,}'],
        ['Unique Actions', f'{unique_actions}'],
        ['Unique Videos', f'{unique_videos}'],
        ['Unique Agents', f'{unique_agents}'],
        ['Avg Duration', f'{avg_duration:.1f}'],
        ['Median Duration', f'{median_duration:.0f}']
    ]
    
    table = ax9.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#2E86AB')
                cell.set_text_props(weight='bold', color='white', size=12)
            else:
                cell.set_facecolor('#E8F4F8' if i % 2 == 0 else 'white')
            cell.set_edgecolor('#2E86AB')
            cell.set_linewidth(2)
    
    ax9.set_title('Prediction Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle(f'Comprehensive Submission Analysis ({len(submission):,} predictions)',
                fontsize=18, fontweight='bold', y=0.995)
    plt.savefig('/kaggle/working/visualizations/07_submission_comprehensive.png',
               dpi=150, bbox_inches='tight', facecolor='white')
    print("âœ“ Saved: 07_submission_comprehensive.png")
    plt.show()
    plt.close()

def plot_action_distribution_comparison(submission, train):
    """Detailed action distribution comparison"""
    # Extract train actions
    train_actions = []
    for _, row in train.iterrows():
        if isinstance(row['behaviors_labeled'], str):
            behaviors = json.loads(row['behaviors_labeled'])
            for b in behaviors:
                b_clean = b.replace("'", "")
                parts = b_clean.split(',')
                if len(parts) >= 3:
                    train_actions.append(parts[2])
    
    train_action_counts = pd.Series(train_actions).value_counts()
    pred_action_counts = submission['action'].value_counts()
    
    # Get all unique actions
    all_actions = set(list(train_action_counts.index) + list(pred_action_counts.index))
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'train': train_action_counts,
        'prediction': pred_action_counts
    }).fillna(0)
    
    # Take top 20 by train frequency
    comparison_df = comparison_df.nlargest(20, 'train')
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # 1. Side-by-side comparison
    ax1 = axes[0, 0]
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax1.barh(x - width/2, comparison_df['train'].values, width,
                    label='Train', color='#2E86AB', edgecolor='black', alpha=0.8)
    bars2 = ax1.barh(x + width/2, comparison_df['prediction'].values, width,
                    label='Predictions', color='#FF6B9D', edgecolor='black', alpha=0.8)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(comparison_df.index, fontsize=9)
    ax1.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Top 20 Actions: Train vs Predictions', 
                 fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=11)
    ax1.set_xscale('log')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Ratio plot (Prediction / Train)
    ax2 = axes[0, 1]
    comparison_df['ratio'] = comparison_df['prediction'] / (comparison_df['train'] + 1)
    comparison_sorted = comparison_df.sort_values('ratio')
    
    colors = ['#FF6B9D' if r < 1 else '#6BCF7F' for r in comparison_sorted['ratio']]
    bars = ax2.barh(range(len(comparison_sorted)), comparison_sorted['ratio'].values,
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.axvline(1, color='red', linestyle='--', linewidth=2, label='Equal (ratio=1)')
    ax2.set_yticks(range(len(comparison_sorted)))
    ax2.set_yticklabels(comparison_sorted.index, fontsize=9)
    ax2.set_xlabel('Prediction / Train Ratio', fontsize=11, fontweight='bold')
    ax2.set_title('Action Prediction Ratio', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Normalized distributions
    ax3 = axes[1, 0]
    comparison_df['train_norm'] = comparison_df['train'] / comparison_df['train'].sum() * 100
    comparison_df['pred_norm'] = comparison_df['prediction'] / comparison_df['prediction'].sum() * 100
    comparison_sorted_norm = comparison_df.sort_values('train_norm', ascending=False)
    
    x = np.arange(len(comparison_sorted_norm))
    width = 0.35
    
    ax3.bar(x - width/2, comparison_sorted_norm['train_norm'].values, width,
           label='Train', color='#2E86AB', edgecolor='black', alpha=0.8)
    ax3.bar(x + width/2, comparison_sorted_norm['pred_norm'].values, width,
           label='Predictions', color='#FF6B9D', edgecolor='black', alpha=0.8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(comparison_sorted_norm.index, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Normalized Action Distribution', fontsize=14, fontweight='bold', pad=10)
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Scatter plot (Train vs Prediction)
    ax4 = axes[1, 1]
    ax4.scatter(comparison_df['train'], comparison_df['prediction'], 
               s=100, alpha=0.6, c=range(len(comparison_df)), cmap='viridis',
               edgecolors='black', linewidth=1.5)
    
    # Add diagonal line (perfect match)
    max_val = max(comparison_df['train'].max(), comparison_df['prediction'].max())
    ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, 
            label='Perfect Match', alpha=0.7)
    
    ax4.set_xlabel('Train Frequency', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Prediction Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Train vs Prediction Correlation', fontsize=14, fontweight='bold', pad=10)
    ax4.legend(fontsize=11)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(alpha=0.3)
    
    # Annotate some points
    for idx in comparison_df.index[:5]:
        ax4.annotate(idx, (comparison_df.loc[idx, 'train'], 
                          comparison_df.loc[idx, 'prediction']),
                    fontsize=8, alpha=0.7)
    
    plt.suptitle('Action Distribution Comparison: Train vs Predictions',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/kaggle/working/visualizations/08_action_comparison_detailed.png',
               dpi=150, bbox_inches='tight', facecolor='white')
    print("âœ“ Saved: 08_action_comparison_detailed.png")
    plt.show()
    plt.close()

def plot_model_ensemble_analysis(submission):
    """Analyze ensemble predictions and model behavior"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Prediction confidence simulation (based on duration as proxy)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Longer events might indicate higher confidence
    submission['conf_proxy'] = np.clip(submission['duration'] / 100, 0, 1)
    
    ax1.hist(submission['conf_proxy'], bins=50, color='#2E86AB',
            edgecolor='black', alpha=0.7)
    ax1.set_title('Prediction Confidence Proxy\n(Duration-based)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Confidence Proxy', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.axvline(submission['conf_proxy'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {submission["conf_proxy"].mean():.2f}')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. Event timing distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(submission['start_frame'], bins=50, color='#A23B72',
            edgecolor='black', alpha=0.7, label='Start Frame')
    ax2.set_title('Event Start Frame Distribution', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Frame Number', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 3. Predictions over time (cumulative)
    ax3 = fig.add_subplot(gs[0, 2])
    submission_sorted = submission.sort_values('start_frame')
    cumulative_counts = np.arange(1, len(submission_sorted) + 1)
    
    ax3.plot(submission_sorted['start_frame'].values, cumulative_counts,
            linewidth=2, color='#F18F01')
    ax3.fill_between(submission_sorted['start_frame'].values, cumulative_counts,
                    alpha=0.3, color='#F18F01')
    ax3.set_title('Cumulative Predictions over Time', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Frame Number', fontsize=11)
    ax3.set_ylabel('Cumulative Predictions', fontsize=11)
    ax3.grid(alpha=0.3)
    
    # 4. Agent-target interaction matrix
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Create interaction matrix
    agent_target_pairs = submission.groupby(['agent_id', 'target_id']).size().unstack(fill_value=0)
    
    sns.heatmap(agent_target_pairs, annot=True, fmt='d', cmap='YlOrRd',
               cbar_kws={'label': 'Number of Predictions'}, ax=ax4,
               linewidths=0.5, linecolor='black')
    ax4.set_title('Agent-Target Interaction Matrix', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Target ID', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Agent ID', fontsize=11, fontweight='bold')
    
    # 5. Duration vs action count
    ax5 = fig.add_subplot(gs[1, 1])
    
    action_stats = submission.groupby('action').agg({
        'duration': 'mean',
        'action': 'count'
    }).rename(columns={'action': 'count'})
    action_stats = action_stats.nlargest(20, 'count')
    
    scatter = ax5.scatter(action_stats['count'], action_stats['duration'],
                         s=action_stats['count']*2, alpha=0.6,
                         c=range(len(action_stats)), cmap='viridis',
                         edgecolors='black', linewidth=1)
    
    ax5.set_xlabel('Action Count', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Average Duration', fontsize=11, fontweight='bold')
    ax5.set_title('Action Frequency vs Duration', fontsize=13, fontweight='bold', pad=10)
    ax5.set_xscale('log')
    ax5.grid(alpha=0.3)
    
    # Annotate top actions
    for idx in action_stats.index[:5]:
        ax5.annotate(idx, (action_stats.loc[idx, 'count'],
                          action_stats.loc[idx, 'duration']),
                    fontsize=8, alpha=0.7)
    
    # 6. Event length categories by action
    ax6 = fig.add_subplot(gs[1, 2])
    
    top_actions_for_length = submission['action'].value_counts().head(10).index
    submission_subset = submission[submission['action'].isin(top_actions_for_length)]
    
    duration_categories = pd.cut(submission_subset['duration'],
                                bins=[0, 30, 120, np.inf],
                                labels=['Short', 'Medium', 'Long'])
    
    category_by_action = pd.crosstab(submission_subset['action'], duration_categories,
                                    normalize='index') * 100
    
    category_by_action.plot(kind='barh', stacked=True, ax=ax6,
                           color=['#FF6B9D', '#FFD93D', '#6BCF7F'],
                           edgecolor='black', linewidth=0.5)
    ax6.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Action', fontsize=11, fontweight='bold')
    ax6.set_title('Duration Categories by Action (Top 10)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax6.legend(title='Duration', fontsize=9, title_fontsize=10)
    ax6.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Model Ensemble & Prediction Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('/kaggle/working/visualizations/09_model_ensemble_analysis.png',
               dpi=150, bbox_inches='tight', facecolor='white')
    print("âœ“ Saved: 09_model_ensemble_analysis.png")
    plt.show()
    plt.close()

# ==================== EXISTING CODE (CLASSIFIERS, SCORING, ETC.) ====================

class StratifiedSubsetClassifier(ClassifierMixin, BaseEstimator):
    """Fit estimator with stratified sampling to maintain class balance"""
    def __init__(self, estimator, n_samples):
        self.estimator = estimator
        self.n_samples = n_samples

    def fit(self, X, y):
        if len(X) <= self.n_samples:
            self.estimator.fit(np.array(X, copy=False), np.array(y, copy=False))
        else:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=min(self.n_samples, len(X)), random_state=42)
            try:
                for train_idx, _ in sss.split(X, y):
                    self.estimator.fit(np.array(X, copy=False)[train_idx], np.array(y, copy=False)[train_idx])
            except:
                downsample = len(X) // self.n_samples
                downsample = max(downsample, 1)
                self.estimator.fit(np.array(X, copy=False)[::downsample], np.array(y, copy=False)[::downsample])
        
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        if len(self.classes_) == 1:
            return np.full((len(X), 1), 1.0)
        probs = self.estimator.predict_proba(np.array(X))
        return probs
        
    def predict(self, X):
        return self.estimator.predict(np.array(X))

class HostVisibleError(Exception):
    pass

def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
        active_labels: set[str] = set(json.loads(active_labels))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue
           
            new_frames = set(range(row['start_frame'], row['stop_frame']))
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
    return sum(action_f1s) / len(action_f1s)

def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']

    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission is missing column {col}')

    solution: pl.DataFrame = pl.DataFrame(solution)
    submission: pl.DataFrame = pl.DataFrame(submission)
    assert (solution['start_frame'] <= solution['stop_frame']).all()
    assert (submission['start_frame'] <= submission['stop_frame']).all()
    solution_videos = set(solution['video_id'].unique())
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission = submission.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)

# ==================== DATA LOADING ====================

print("="*80)
print("MABe Mouse Behavior Detection Challenge")
print("Advanced Ensemble with Enhanced Comprehensive Visualizations")
print("WITH SPATIAL HISTORY FEATURES")
print("="*80)

train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
train_without_mabe22 = train.query("~ lab_id.str.startswith('MABe22_')")

test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

print(f"\nðŸ“Š Dataset loaded: {len(train)} train videos, {len(test)} test videos")

print("\n" + "="*80)
print("GENERATING ENHANCED VISUALIZATIONS")
print("="*80)

plot_comprehensive_dataset_overview(train, test)
plot_advanced_behavior_analysis(train)

drop_body_parts = ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
                   'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
                   'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    for _, row in dataset.iterrows():
        
        lab_id = row.lab_id
        if lab_id.startswith('MABe22'): continue
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            if verbose: print('No labeled behaviors:', lab_id, video_id)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        if pvid.isna().any().any():
            if verbose and traintest == 'test': print('video with missing values', video_id, traintest, len(vid), 'frames')
        else:
            if verbose and traintest == 'test': print('video with all values', video_id, traintest, len(vid), 'frames')
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid /= row.pix_per_cm_approx

        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                continue

        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    assert len(single_mouse) == len(pvid)
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label, pvid
                    else:
                        if verbose: print('- test single', video_id, mouse_id)
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions, pvid
                except KeyError:
                    pass

        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label, pvid
                    else:
                        if verbose: print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions, pvid

action_thresholds = defaultdict(lambda: 0.27)

def predict_multiclass_adaptive(pred, meta, action_thresholds):
    """Adaptive thresholding per action + temporal smoothing"""
    pred_smoothed = pred.rolling(window=5, min_periods=1, center=True).mean()
    
    ama = np.argmax(pred_smoothed, axis=1)
    
    max_probs = pred_smoothed.max(axis=1)
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    for i, action in enumerate(pred_smoothed.columns):
        action_mask = (ama == i)
        threshold = action_thresholds.get(action, 0.27)
        threshold_mask |= (action_mask & (max_probs >= threshold))
    
    ama = np.where(threshold_mask, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False
    
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if i < len(stop_video_id):
            if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
                new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
                submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
        else:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
    
    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= 3].reset_index(drop=True)
    
    if len(submission_part) > 0:
        assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    
    if verbose: print(f'  actions found: {len(submission_part)}')
    return submission_part

# ==================== NEW SPATIAL HISTORY FEATURES ====================

def add_spatial_history_features(X, center_x, center_y, pvid, current_mouse_id=None):
    """
    Add features related to whether mouse is in locations previously visited by itself or other mice.
    
    Two key features:
    1. self_location_revisit: Is the mouse in a location where it spent time previously?
    2. other_location_visit: Is the mouse in a location where other mice spent time?
    """
    
    if 'body_center' not in pvid.columns.get_level_values(1):
        # No body_center available, return without adding features
        X['self_loc_revisit_5cm'] = 0
        X['other_loc_visit_5cm'] = 0
        return X
    
    # Define spatial radius for "same location" (in cm)
    spatial_radius = 5.0
    
    # Get current positions
    current_positions = np.column_stack([center_x.values, center_y.values])
    
    # Feature 1: Self location revisit
    # Build a history of where this mouse has been (using temporal windowing)
    history_window = 300  # Look back 300 frames (~10 seconds at 30fps)
    self_loc_revisit = np.zeros(len(current_positions))
    
    for i in range(len(current_positions)):
        if i < history_window:
            continue
        
        # Get historical positions (excluding very recent to avoid trivial matches)
        lookback_start = max(0, i - history_window)
        lookback_end = max(0, i - 30)  # Exclude last 30 frames
        
        if lookback_end <= lookback_start:
            continue
            
        historical_positions = current_positions[lookback_start:lookback_end]
        current_pos = current_positions[i]
        
        # Check if current position is close to any historical position
        if not (np.isnan(current_pos).any() or np.isnan(historical_positions).any().any()):
            distances = np.sqrt(np.sum((historical_positions - current_pos)**2, axis=1))
            # If any historical position is within radius, mark as revisit
            if np.any(distances < spatial_radius):
                self_loc_revisit[i] = 1.0
    
    X['self_loc_revisit_5cm'] = self_loc_revisit
    
    # Feature 2: Other mice location visit
    # Check if current mouse is in locations where OTHER mice have been
    other_loc_visit = np.zeros(len(current_positions))
    
    # Get all mice in the video
    all_mice = pvid.columns.get_level_values('mouse_id').unique()
    
    if current_mouse_id is not None and len(all_mice) > 1:
        # For each frame, check if current position overlaps with other mice's recent positions
        for i in range(len(current_positions)):
            current_pos = current_positions[i]
            
            if np.isnan(current_pos).any():
                continue
            
            # Look at other mice's positions in recent history
            for other_mouse in all_mice:
                if other_mouse == current_mouse_id:
                    continue
                
                try:
                    other_x = pvid[other_mouse]['body_center']['x'].values
                    other_y = pvid[other_mouse]['body_center']['y'].values
                    
                    # Check recent history of other mouse
                    lookback_start = max(0, i - history_window)
                    lookback_end = i
                    
                    other_positions = np.column_stack([
                        other_x[lookback_start:lookback_end],
                        other_y[lookback_start:lookback_end]
                    ])
                    
                    # Remove nan values
                    valid_mask = ~np.isnan(other_positions).any(axis=1)
                    other_positions = other_positions[valid_mask]
                    
                    if len(other_positions) > 0:
                        distances = np.sqrt(np.sum((other_positions - current_pos)**2, axis=1))
                        if np.any(distances < spatial_radius):
                            other_loc_visit[i] = 1.0
                            break  # Found overlap with at least one other mouse
                except (KeyError, IndexError):
                    continue
    
    X['other_loc_visit_5cm'] = other_loc_visit
    
    # Add smoothed versions (proportion of time in visited locations)
    smooth_window = 60
    X['self_loc_revisit_pct60'] = pd.Series(self_loc_revisit).rolling(smooth_window, min_periods=1).mean()
    X['other_loc_visit_pct60'] = pd.Series(other_loc_visit).rolling(smooth_window, min_periods=1).mean()
    
    return X

# ==================== FEATURE ENGINEERING (UPDATED WITH SPATIAL HISTORY) ====================

def add_curvature_features(X, center_x, center_y):
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()
    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)
    for window in [30, 60]:
        X[f'curv_mean_{window}'] = curvature.rolling(window, min_periods=5).mean()
    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    X['turn_rate_30'] = angle_change.rolling(30, min_periods=5).sum()
    return X

def add_multiscale_features(X, center_x, center_y):
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2)
    scales = [10, 40, 160]
    for scale in scales:
        if len(speed) >= scale:
            X[f'sp_m{scale}'] = speed.rolling(scale, min_periods=max(1, scale//4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(scale, min_periods=max(1, scale//4)).std()
    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)
    return X

def add_state_features(X, center_x, center_y):
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2)
    speed_ma = speed.rolling(15, min_periods=5).mean()
    try:
        speed_states = pd.cut(speed_ma, bins=[-np.inf, 0.5, 2.0, 5.0, np.inf], labels=[0, 1, 2, 3]).astype(float)
        for window in [60, 120]:
            if len(speed_states) >= window:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (speed_states == state).astype(float).rolling(window, min_periods=10).mean()
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(window, min_periods=10).sum()
    except:
        pass
    return X

def add_longrange_features(X, center_x, center_y):
    for window in [120, 240]:
        if len(center_x) >= window:
            X[f'x_ml{window}'] = center_x.rolling(window, min_periods=20).mean()
            X[f'y_ml{window}'] = center_y.rolling(window, min_periods=20).mean()
    for span in [60, 120]:
        X[f'x_e{span}'] = center_x.ewm(span=span, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=span, min_periods=1).mean()
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2)
    for window in [60, 120]:
        if len(speed) >= window:
            X[f'sp_pct{window}'] = speed.rolling(window, min_periods=20).rank(pct=True)
    return X

def add_interaction_features(X, mouse_pair, avail_A, avail_B):
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X
    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)
    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()
    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)
    for window in [30, 60]:
        X[f'A_ld{window}'] = A_lead.rolling(window, min_periods=5).mean()
        X[f'B_ld{window}'] = B_lead.rolling(window, min_periods=5).mean()
    approach = -rel_dist.diff()
    chase = approach * B_lead
    X['chase_30'] = chase.rolling(30, min_periods=5).mean()
    for window in [60, 120]:
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f'sp_cor{window}'] = A_sp.rolling(window, min_periods=10).corr(B_sp)
    return X

def transform_single(single_mouse, body_parts_tracked, pvid=None, mouse_id=None):
    available_body_parts = single_mouse.columns.get_level_values(0)
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2) 
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(10)
        speeds = pd.DataFrame({
            'sp_lf': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
            'sp_rt': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'sp_lf2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
            'sp_rt2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)
    
    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)
    
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)
    
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        
        for w in [5, 15, 30, 60]:
            X[f'cx_m{w}'] = cx.rolling(w, min_periods=1, center=True).mean()
            X[f'cy_m{w}'] = cy.rolling(w, min_periods=1, center=True).mean()
            X[f'cx_s{w}'] = cx.rolling(w, min_periods=1, center=True).std()
            X[f'cy_s{w}'] = cy.rolling(w, min_periods=1, center=True).std()
            X[f'x_rng{w}'] = cx.rolling(w, min_periods=1, center=True).max() - cx.rolling(w, min_periods=1, center=True).min()
            X[f'y_rng{w}'] = cy.rolling(w, min_periods=1, center=True).max() - cy.rolling(w, min_periods=1, center=True).min()
            X[f'disp{w}'] = np.sqrt(cx.diff().rolling(w, min_periods=1).sum()**2 + cy.diff().rolling(w, min_periods=1).sum()**2)
            X[f'act{w}'] = np.sqrt(cx.diff().rolling(w, min_periods=1).var() + cy.diff().rolling(w, min_periods=1).var())
        
        X = add_curvature_features(X, cx, cy)
        X = add_multiscale_features(X, cx, cy)
        X = add_state_features(X, cx, cy)
        X = add_longrange_features(X, cx, cy)
        
        # NEW: Add spatial history features
        if pvid is not None:
            X = add_spatial_history_features(X, cx, cy, pvid, current_mouse_id=mouse_id)
        
        speed = np.sqrt(cx.diff()**2 + cy.diff()**2)
        high_speed_threshold = speed.quantile(0.75)
        recent_high_activity = (speed > high_speed_threshold).astype(float).rolling(100, min_periods=1).max()
        X['recent_active_100'] = recent_high_activity
        
        angle = np.arctan2(cy.diff(), cx.diff())
        angle_diff = angle.diff()
        angle_diff_wrapped = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        significant_turn = (np.abs(angle_diff_wrapped) > 0.5).astype(float)
        turn_count_200 = significant_turn.rolling(200, min_periods=1).sum()
        X['turn_count_200'] = turn_count_200
        
        high_speed_events = (speed > high_speed_threshold).astype(float)
        event_indices = pd.Series(np.arange(len(speed)), index=speed.index)
        event_indices = event_indices.where(high_speed_events == 1).ffill().fillna(-500)
        current_indices = pd.Series(np.arange(len(speed)), index=speed.index)
        frames_since_active = (current_indices - event_indices).clip(lower=0, upper=500)
        X['frames_since_active'] = frames_since_active
    
    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 + 
                         (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            X[f'nt_lg{lag}'] = nt_dist.shift(lag)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(lag)
    
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 + 
                       (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-20, -10, 10, 20]:
            X[f'ear_o{off}'] = ear_d.shift(-off)
        X['ear_con'] = ear_d.rolling(30, min_periods=1, center=True).std() / (ear_d.rolling(30, min_periods=1, center=True).mean() + 1e-6)
    
    return X

def transform_pair(mouse_pair, body_parts_tracked, pvid=None, agent_id=None):
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)
    
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2) 
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        shA = mouse_pair['A']['ear_left'].shift(10)
        shB = mouse_pair['B']['ear_left'].shift(10)
        speeds = pd.DataFrame({
            'sp_A': np.square(mouse_pair['A']['ear_left'] - shA).sum(axis=1, skipna=False),
            'sp_AB': np.square(mouse_pair['A']['ear_left'] - shB).sum(axis=1, skipna=False),
            'sp_B': np.square(mouse_pair['B']['ear_left'] - shB).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)
    
    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)
    
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)
    
    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        shA_n = mouse_pair['A']['nose'].shift(10)
        shB_n = mouse_pair['B']['nose'].shift(10)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = cur - past
    
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                    (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)
        X['v_cls'] = (cd < 5.0).astype(float)
        X['cls'] = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X['med'] = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X['far'] = (cd >= 30.0).astype(float)
    
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)
        
        for w in [5, 15, 30, 60]:
            X[f'd_m{w}'] = cd_full.rolling(w, min_periods=1, center=True).mean()
            X[f'd_s{w}'] = cd_full.rolling(w, min_periods=1, center=True).std()
            X[f'd_mn{w}'] = cd_full.rolling(w, min_periods=1, center=True).min()
            X[f'd_mx{w}'] = cd_full.rolling(w, min_periods=1, center=True).max()
            
            d_var = cd_full.rolling(w, min_periods=1, center=True).var()
            X[f'int{w}'] = 1 / (1 + d_var)
            
            Axd = mouse_pair['A']['body_center']['x'].diff()
            Ayd = mouse_pair['A']['body_center']['y'].diff()
            Bxd = mouse_pair['B']['body_center']['x'].diff()
            Byd = mouse_pair['B']['body_center']['y'].diff()
            coord = Axd * Bxd + Ayd * Byd
            X[f'co_m{w}'] = coord.rolling(w, min_periods=1, center=True).mean()
            X[f'co_s{w}'] = coord.rolling(w, min_periods=1, center=True).std()
        
        # NEW: Add spatial history features for agent mouse in pair context
        if pvid is not None and agent_id is not None:
            cx = mouse_pair['A']['body_center']['x']
            cy = mouse_pair['A']['body_center']['y']
            X = add_spatial_history_features(X, cx, cy, pvid, current_mouse_id=agent_id)
        
        close_proximity = (cd < 10.0).astype(float)
        recent_close = close_proximity.rolling(100, min_periods=1).max()
        X['recent_close_100'] = recent_close
        
        dist_change = cd.diff()
        approach_event = (dist_change < -0.5).astype(float)
        avoid_event = (dist_change > 0.5).astype(float)
        interaction_change = (approach_event.diff().abs() + avoid_event.diff().abs()).clip(upper=1)
        cycle_count_200 = interaction_change.rolling(200, min_periods=1).sum()
        X['cycle_count_200'] = cycle_count_200
        
        close_events = (cd < 15.0).astype(float)
        event_indices = pd.Series(np.arange(len(cd)), index=cd.index)
        event_indices = event_indices.where(close_events == 1).ffill().fillna(-500)
        current_indices = pd.Series(np.arange(len(cd)), index=cd.index)
        frames_since_close = (current_indices - event_indices).clip(lower=0, upper=500)
        X['frames_since_close'] = frames_since_close
    
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                    (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        for lag in [10, 20, 40]:
            X[f'nn_lg{lag}'] = nn.shift(lag)
            X[f'nn_ch{lag}'] = nn - nn.shift(lag)
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}'] = is_cl.rolling(lag, min_periods=1).mean()
    
    if 'body_center' in avail_A and 'body_center' in avail_B:
        Avx = mouse_pair['A']['body_center']['x'].diff()
        Avy = mouse_pair['A']['body_center']['y'].diff()
        Bvx = mouse_pair['B']['body_center']['x'].diff()
        Bvy = mouse_pair['B']['body_center']['y'].diff()
        val = (Avx * Bvx + Avy * Bvy) / (np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6)
        
        for off in [-20, -10, 0, 10, 20]:
            X[f'va_{off}'] = val.shift(-off)
        
        X['int_con'] = cd_full.rolling(30, min_periods=1, center=True).std() / (cd_full.rolling(30, min_periods=1, center=True).mean() + 1e-6)
        
        X = add_interaction_features(X, mouse_pair, avail_A, avail_B)
    
    return X

# ==================== ENSEMBLE TRAINING ====================

def submit_ensemble(body_parts_tracked_str, switch_tr, X_tr, label, meta):
    models = []
    
    models.append(make_pipeline(
        SimpleImputer(),
        StratifiedSubsetClassifier(
            lightgbm.LGBMClassifier(
                n_estimators=225, learning_rate=0.07, min_child_samples=40,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbose=-1),
            100000)
    ))
    
    models.append(make_pipeline(
        SimpleImputer(),
        StratifiedSubsetClassifier(
            lightgbm.LGBMClassifier(
                n_estimators=150, learning_rate=0.1, min_child_samples=20,
                num_leaves=63, max_depth=8, subsample=0.7, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=0.1, verbose=-1),
            80000)
    ))
    
    models.append(make_pipeline(
        SimpleImputer(),
        StratifiedSubsetClassifier(
            lightgbm.LGBMClassifier(
                n_estimators=100, learning_rate=0.05, min_child_samples=30,
                num_leaves=127, max_depth=10, subsample=0.75, verbose=-1),
            60000)
    ))
    
    if XGBOOST_AVAILABLE:
        models.append(make_pipeline(
            SimpleImputer(),
            StratifiedSubsetClassifier(
                XGBClassifier(
                    n_estimators=180, learning_rate=0.08, max_depth=6,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    tree_method='hist', verbosity=0),
                85000)
        ))
    
    if CATBOOST_AVAILABLE:
        models.append(make_pipeline(
            SimpleImputer(),
            StratifiedSubsetClassifier(
                CatBoostClassifier(
                    iterations=120, learning_rate=0.1, depth=6,
                    verbose=False, allow_writing_files=False),
                70000)
        ))
    
    if GANDALF_AVAILABLE:
        models.append(make_pipeline(
            SimpleImputer(),
            StratifiedSubsetClassifier(
                GANDALF(
                    n_estimators=150,
                    learning_rate=0.01,
                    max_depth=6,
                    dropout=0.1,
                    random_state=42
                ),
                75000)
        ))
    
    model_list = []
    for action in label.columns:
        action_mask = ~ label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)

        if not (y_action == 0).all() and y_action.sum() >= 5:
            trained = []
            for m in models:
                m_clone = clone(m)
                m_clone.fit(X_tr[action_mask], y_action)
                trained.append(m_clone)
            model_list.append((action, trained))
    
    del X_tr
    gc.collect()

    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(test_subset, 'test',
                                    generate_single=(switch_tr == 'single'), 
                                    generate_pair=(switch_tr == 'pair'))
    
    if verbose: print(f"n_videos: {len(test_subset)}, n_models: {len(models)}")
    
    for result in generator:
        switch_te, data_te, meta_te, actions_te, pvid_te = result
        assert switch_te == switch_tr
        try:
            if switch_te == 'single':
                # Extract mouse_id from meta
                mouse_id_str = meta_te['agent_id'].iloc[0] if len(meta_te) > 0 else None
                mouse_id = int(mouse_id_str[-1]) if mouse_id_str else None
                X_te = transform_single(data_te, body_parts_tracked, pvid=pvid_te, mouse_id=mouse_id)
            else:
                # Extract agent_id for pair
                agent_id_str = meta_te['agent_id'].iloc[0] if len(meta_te) > 0 else None
                agent_id = int(agent_id_str[-1]) if agent_id_str else None
                X_te = transform_pair(data_te, body_parts_tracked, pvid=pvid_te, agent_id=agent_id)
            
            if verbose and len(X_te) == 0: print("ERROR: X_te empty")
            del data_te
            del pvid_te
    
            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, trained in model_list:
                if action in actions_te:
                    probs = [m.predict_proba(X_te)[:, 1] for m in trained]
                    pred[action] = np.mean(probs, axis=0)
            
            del X_te
            gc.collect()
            
            if pred.shape[1] != 0:
                sub_part = predict_multiclass_adaptive(pred, meta_te, action_thresholds)
                submission_list.append(sub_part)
            else:
                if verbose: print(f"  ERROR: no training data")
        except Exception as e:
            if verbose: print(f'  ERROR: {str(e)[:50]}')
            try:
                del data_te
            except:
                pass
            try:
                del pvid_te
            except:
                pass
            gc.collect()

def robustify(submission, dataset, traintest, traintest_directory=None):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    submission = submission[submission.start_frame < submission.stop_frame]
    
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop:
                mask[i] = False
            else:
                last_stop = row['stop_frame']
        group_list.append(group[mask])
    submission = pd.concat(group_list) if group_list else submission

    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if verbose: print(f"Video {video_id} has no predictions")
        
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
    
        vid_behaviors = eval(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
    
        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1
    
        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_len = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_len
                batch_stop = min(batch_start + batch_len, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])

    submission = submission.reset_index(drop=True)
    return submission

# ==================== MAIN LOOP ====================

submission_list = []
section_results = {}

print(f"\nðŸ¤– Models: XGBoost={XGBOOST_AVAILABLE}, CatBoost={CATBOOST_AVAILABLE}, GANDALF={GANDALF_AVAILABLE}\n")
print("="*80)
print("TRAINING MODELS")
print("="*80)

first_single_viz = True
first_pair_viz = True

for section in range(1, len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    section_results[section] = {'predictions': 0, 'models': 0, 'single_features': 0, 'pair_features': 0}
    
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"\n{section}. Processing: {len(body_parts_tracked)} body parts")
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
        single_list, single_label_list, single_meta_list = [], [], []
        pair_list, pair_label_list, pair_meta_list = [], [], []
    
        for result in generate_mouse_data(train_subset, 'train'):
            switch, data, meta, label, pvid = result
            if switch == 'single':
                single_list.append(data)
                single_meta_list.append(meta)
                single_label_list.append(label)
            else:
                pair_list.append(data)
                pair_meta_list.append(meta)
                pair_label_list.append(label)
    
        if len(single_list) > 0:
            single_mouse = pd.concat(single_list)
            single_label = pd.concat(single_label_list)
            single_meta = pd.concat(single_meta_list)
            del single_list, single_label_list, single_meta_list
            gc.collect()
            
            # Use first pvid for spatial features
            first_pvid = None
            for result in generate_mouse_data(train_subset, 'train', generate_pair=False):
                _, _, _, _, first_pvid = result
                break
            
            mouse_id_sample = int(single_meta['agent_id'].iloc[0][-1]) if len(single_meta) > 0 else None
            X_tr = transform_single(single_mouse, body_parts_tracked, pvid=first_pvid, mouse_id=mouse_id_sample)
            del single_mouse
            print(f"  Single: {X_tr.shape}")
            section_results[section]['single_features'] = X_tr.shape[1]
            
            if first_single_viz and section == 1:
                plot_feature_analysis(X_tr, 'single')
                plot_feature_correlation_heatmap(X_tr, 'single')
                plot_temporal_patterns_detailed(X_tr, 'single')
                first_single_viz = False
            
            submit_ensemble(body_parts_tracked_str, 'single', X_tr, single_label, single_meta)
                
        if len(pair_list) > 0:
            mouse_pair = pd.concat(pair_list)
            pair_label = pd.concat(pair_label_list)
            pair_meta = pd.concat(pair_meta_list)
            del pair_list, pair_label_list, pair_meta_list
            gc.collect()
        
            # Use first pvid for spatial features
            first_pvid = None
            for result in generate_mouse_data(train_subset, 'train', generate_single=False):
                _, _, _, _, first_pvid = result
                break
            
            agent_id_sample = int(pair_meta['agent_id'].iloc[0][-1]) if len(pair_meta) > 0 else None
            X_tr = transform_pair(mouse_pair, body_parts_tracked, pvid=first_pvid, agent_id=agent_id_sample)
            del mouse_pair
            print(f"  Pair: {X_tr.shape}")
            section_results[section]['pair_features'] = X_tr.shape[1]
            
            if first_pair_viz and section == 1:
                plot_feature_analysis(X_tr, 'pair')
                plot_feature_correlation_heatmap(X_tr, 'pair')
                plot_temporal_patterns_detailed(X_tr, 'pair')
                first_pair_viz = False
            
            submit_ensemble(body_parts_tracked_str, 'pair', X_tr, pair_label, pair_meta)
        
        if len(submission_list) > 0:
            section_results[section]['predictions'] = len(submission_list[-1]) if submission_list else 0
            section_results[section]['models'] = 5
                
    except Exception as e:
        print(f'***Exception*** {str(e)[:100]}')
    
    gc.collect()

print("\n" + "="*80)
print("FINALIZING SUBMISSION")
print("="*80)

if len(submission_list) > 0:
    submission = pd.concat(submission_list)
else:
    submission = pd.DataFrame({
        'video_id': [438887472],
        'agent_id': ['mouse1'],
        'target_id': ['self'],
        'action': ['rear'],
        'start_frame': [278],
        'stop_frame': [500]
    })

submission_robust = robustify(submission, test, 'test')
submission_robust.index.name = 'row_id'
submission_robust.to_csv('submission.csv')
print(f"\nâœ… Submission created: {len(submission_robust)} predictions")

print("\n" + "="*80)
print("GENERATING FINAL VISUALIZATIONS")
print("="*80)

plot_training_progress_detailed(section_results)
plot_comprehensive_submission_analysis(submission_robust, train)
plot_action_distribution_comparison(submission_robust, train)
plot_model_ensemble_analysis(submission_robust)

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETED")
print("="*80)
print("\nðŸ“Š Visualization files:")
viz_files = sorted([f for f in os.listdir('/kaggle/working/visualizations') if f.endswith('.png')])
for i, f in enumerate(viz_files, 1):
    print(f"   {i:2d}. {f}")

print("\n" + "="*80)
print("COMPETITION COMPLETE!")
print("="*80)
print(f"âœ… Submission: submission.csv ({len(submission_robust):,} predictions)")
print(f"ðŸ“Š Visualizations: {len(viz_files)} publication-quality figures")
print(f"ðŸ†• NEW FEATURES: Spatial history features added!")
print("   - self_loc_revisit_5cm: Mouse in previously visited location")
print("   - other_loc_visit_5cm: Mouse in location where other mice were")
print("   - self_loc_revisit_pct60: Smoothed percentage version")
print("   - other_loc_visit_pct60: Smoothed percentage version")
print("="*80)

