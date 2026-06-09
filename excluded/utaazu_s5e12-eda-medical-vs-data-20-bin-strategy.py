import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import warnings

# --- Configuration ---
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Color Palette (Consistent with Visualization Key)
COLORS = {
    'neg': '#3498db',   # Blue
    'pos': '#e74c3c',   # Red
    'med': '#2ecc71',   # Green
    'ai': '#f39c12',    # Orange
    'test': '#9b59b6',  # Purple
    'bad': '#95a5a6'    # Grey (for overfitting examples)
}

print("âœ… Configuration Complete.")


# ==============================================================================
# ğŸ›  Helper Functions for Statistical Analysis & Visualization
# ==============================================================================

def get_ai_thresholds(X, y, max_depth=3):
    """Learns data-driven thresholds using a shallow Decision Tree."""
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    X_filled = X.fillna(X.median()).values.reshape(-1, 1)
    dt.fit(X_filled, y)
    return sorted(list(set([t for t in dt.tree_.threshold if t != -2])))

def calculate_ks_statistic(data, feature, target):
    """Calculates KS statistic and finding the max separation point."""
    neg = data[data[target] == 0][feature].dropna()
    pos = data[data[target] == 1][feature].dropna()
    ks_stat, _ = stats.ks_2samp(neg, pos)
    
    grid = np.linspace(data[feature].min(), data[feature].max(), 500)
    cdf_neg = np.array([stats.percentileofscore(neg, x) for x in grid]) / 100
    cdf_pos = np.array([stats.percentileofscore(pos, x) for x in grid]) / 100
    max_idx = np.argmax(np.abs(cdf_neg - cdf_pos))
    
    return ks_stat, grid[max_idx], grid, cdf_neg, cdf_pos, max_idx

def calculate_psi(expected, actual, buckets=10):
    """
    Calculates Population Stability Index (PSI).
    PSI < 0.1: Stable | 0.1-0.25: Minor Shift | > 0.25: Major Shift
    """
    def sub_psi(e_perc, a_perc):
        if a_perc == 0: a_perc = 0.0001
        if e_perc == 0: e_perc = 0.0001
        return (e_perc - a_perc) * np.log(e_perc / a_perc)

    breakpoints = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected, breakpoints)
    # Handle duplicate bin edges
    breakpoints = np.unique(breakpoints)
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents))])
    return psi_value

# ==============================================================================
# ğŸ“Š Plotting Functions
# ==============================================================================

def plot_part1_medical_vs_data(df, col, target, medical_refs, display_name=None):
    if display_name is None: display_name = col
    ai_thresholds = get_ai_thresholds(df[col], df[target])
    ks_stat, ks_point, grid, cdf_neg, cdf_pos, max_idx = calculate_ks_statistic(df, col, target)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Distribution & Thresholds
    sns.kdeplot(data=df, x=col, hue=target, palette={0: COLORS['neg'], 1: COLORS['pos']}, 
                fill=True, common_norm=False, alpha=0.2, linewidth=2, ax=axes[0])
    for ref in medical_refs:
        axes[0].axvline(ref, color=COLORS['med'], linestyle='--', linewidth=2.5, label='Medical Guideline' if ref==medical_refs[0] else "")
    for t in ai_thresholds:
        axes[0].axvline(t, color=COLORS['ai'], linestyle='-', linewidth=2, alpha=0.9, label='AI Split' if t==ai_thresholds[0] else "")
    
    axes[0].set_title(f'Medical vs. Data Reality: {display_name}', fontweight='bold')
    axes[0].legend(loc='upper right')
    
    # Right: CDF & KS
    lower, upper = df[col].quantile(0.005), df[col].quantile(0.995)
    axes[1].plot(grid, cdf_neg, color=COLORS['neg'], label='No Diabetes', linewidth=2)
    axes[1].plot(grid, cdf_pos, color=COLORS['pos'], label='Diabetes', linewidth=2)
    axes[1].plot([ks_point, ks_point], [cdf_neg[max_idx], cdf_pos[max_idx]], 
                 color='black', linestyle='-', linewidth=3, label=f'Max KS: {ks_stat:.3f}')
    
    axes[1].annotate(f'Optimal Split: {ks_point:.1f}', 
                     xy=(ks_point, (cdf_neg[max_idx] + cdf_pos[max_idx])/2),
                     xytext=(ks_point + (upper-lower)*0.1, 0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    axes[1].set_title(f'Statistical Separation (CDF)', fontweight='bold')
    axes[1].set_xlim(lower, upper)
    axes[0].set_xlim(lower, upper)
    axes[1].legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_part2_distribution_shift(train, test, col, q=20):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Raw Drift
    sns.kdeplot(train[col].dropna(), label='Train Set', fill=True, alpha=0.3, color=COLORS['neg'], ax=axes[0])
    sns.kdeplot(test[col].dropna(), label='Test Set', fill=True, alpha=0.3, color=COLORS['test'], ax=axes[0])
    axes[0].set_title(f"Raw Distribution Drift: {col}", fontweight='bold')
    axes[0].legend()
    
    # Right: Binned Alignment
    combined = pd.concat([train[col], test[col]])
    binned = pd.qcut(combined.rank(method='first'), q=q, labels=False, duplicates='drop')
    t_counts = binned.iloc[:len(train)].value_counts(normalize=True).sort_index()
    te_counts = binned.iloc[len(train):].value_counts(normalize=True).sort_index()
    psi = calculate_psi(train[col], test[col], buckets=q)
    
    x = np.arange(len(t_counts))
    axes[1].bar(x - 0.2, t_counts.values, width=0.4, label='Train (Binned)', color=COLORS['neg'], alpha=0.7)
    axes[1].bar(x + 0.2, te_counts.values, width=0.4, label='Test (Binned)', color=COLORS['test'], alpha=0.7)
    axes[1].set_title(f"Rank-Based Alignment (q={q}) | PSI: {psi:.4f}\n(Low PSI = Stable)", fontweight='bold')
    axes[1].set_xlabel('Bin ID')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def plot_part3_resolution_tradeoff(df, col, target, q_list=[10, 20, 50]):
    fig, axes = plt.subplots(1, len(q_list), figsize=(20, 6), sharey=True)
    
    for i, q in enumerate(q_list):
        df_temp = df.copy()
        df_temp['bin'] = pd.qcut(df_temp[col].rank(method='first'), q=q, labels=False, duplicates='drop')
        bin_stats = df_temp.groupby('bin')[target].mean()
        
        # Color styling
        color = COLORS['bad'] if q == 50 else (COLORS['pos'] if q == 20 else COLORS['neg'])
        title_suffix = " (Coarse)" if q==10 else (" (Optimal)" if q==20 else " (Noisy/Overfit)")
        
        sns.barplot(x=bin_stats.index, y=bin_stats.values, ax=axes[i], color=color, alpha=0.6)
        axes[i].plot(bin_stats.index, bin_stats.values, color='black', marker='o', linewidth=2, alpha=0.7, label='Risk Trend')
        
        axes[i].set_title(f'{q} Bins{title_suffix}', fontweight='bold')
        axes[i].set_xlabel('Bin ID')
        if i == 0: axes[i].set_ylabel('Probability of Diabetes')
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.5)
        
    plt.suptitle(f"Granularity Trade-off Analysis: {col.upper()}", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()


# --- Load Data ---
print("ğŸ“‚ Loading Data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
target_col = 'diagnosed_diabetes'

print(f"Train Shape: {train.shape}")
print(f"Test Shape:  {test.shape}")

# Selected features for deep-dive analysis
features_to_analyze = [
    ("physical_activity_minutes_per_week", [150], "Physical Activity (min/week)"),
    ("triglycerides", [150], "Triglycerides (mg/dL)"),
    ("bmi", [25.0, 30.0], "BMI"),
    ("age", [40, 65], "Age"),
]


print("\n--- Phase 1: Medical Thresholds vs. Data Reality ---")
for col, refs, disp in features_to_analyze:
    plot_part1_medical_vs_data(train, col, target_col, refs, display_name=disp)


print("\n--- Phase 2: Distribution Shift & PSI Verification ---")
for col, _, disp in features_to_analyze:
    plot_part2_distribution_shift(train, test, col, q=20)


print("\n--- Phase 3: Granularity Trade-off (10 vs 20 vs 50 Bins) ---")
for col, _, disp in features_to_analyze:
    plot_part3_resolution_tradeoff(train, col, target_col, q_list=[10, 20, 50])

