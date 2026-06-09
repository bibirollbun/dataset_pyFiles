import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import networkx as nx
from collections import Counter

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.4f}'.format)

print("Libraries loaded successfully!")


# Load data
DATA_PATH = '/kaggle/input/mercor-cheating-detection/'

train = pd.read_csv(f'{DATA_PATH}train.csv')
test = pd.read_csv(f'{DATA_PATH}test.csv')
graph = pd.read_csv(f'{DATA_PATH}social_graph.csv', names=['source', 'target'])

feature_cols = [c for c in train.columns if c.startswith('feature_')]

print("="*80)
print("ğŸ“Š DATASET OVERVIEW")
print("="*80)
print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Graph edges: {len(graph):,}")
print(f"Feature columns: {len(feature_cols)}")


# Labeled vs Unlabeled breakdown
labeled_mask = train['is_cheating'].notna()
labeled_train = train[labeled_mask].copy()
unlabeled_train = train[~labeled_mask].copy()

print("="*80)
print("ğŸ�¯ TARGET VARIABLE ANALYSIS")
print("="*80)

print(f"\nğŸ“Œ Labeled samples: {labeled_mask.sum():,} ({100*labeled_mask.mean():.1f}%)")
print(f"ğŸ“Œ Unlabeled samples: {(~labeled_mask).sum():,} ({100*(~labeled_mask).mean():.1f}%)")

# High confidence clean breakdown
hcc_mask = train['high_conf_clean'] == 1.0
print(f"\nâœ… High confidence clean: {hcc_mask.sum():,} ({100*hcc_mask.mean():.1f}%)")
print(f"â�“ Not high_conf_clean: {(~hcc_mask).sum():,} ({100*(~hcc_mask).mean():.1f}%)")

# Class distribution
cheaters = (labeled_train['is_cheating'] == 1).sum()
non_cheaters = (labeled_train['is_cheating'] == 0).sum()

print(f"\nğŸ�¯ CLASS DISTRIBUTION (Labeled Only):")
print(f"   Non-cheaters (0): {non_cheaters:,} ({100*non_cheaters/len(labeled_train):.2f}%)")
print(f"   Cheaters (1): {cheaters:,} ({100*cheaters/len(labeled_train):.2f}%)")
print(f"   Imbalance ratio: {non_cheaters/cheaters:.1f}:1")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Labeled vs Unlabeled
axes[0].pie([labeled_mask.sum(), (~labeled_mask).sum()], 
            labels=['Labeled', 'Unlabeled'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'])
axes[0].set_title('Labeled vs Unlabeled')

# Plot 2: Class distribution
axes[1].bar(['Non-Cheaters', 'Cheaters'], [non_cheaters, cheaters], 
            color=['#3498db', '#e74c3c'])
axes[1].set_title('Class Distribution (Labeled)')
axes[1].set_ylabel('Count')
for i, v in enumerate([non_cheaters, cheaters]):
    axes[1].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

# Plot 3: High conf clean
axes[2].pie([hcc_mask.sum(), (~hcc_mask).sum()], 
            labels=['High Conf Clean', 'Other'], autopct='%1.1f%%',
            colors=['#9b59b6', '#f39c12'])
axes[2].set_title('High Confidence Clean')

plt.tight_layout()
plt.show()


# Cross-tabulation: high_conf_clean vs is_cheating
print("\nğŸ“Š CROSS-TABULATION: high_conf_clean vs is_cheating")
print("="*60)

crosstab = pd.crosstab(train['high_conf_clean'], train['is_cheating'], 
                       margins=True, dropna=False)
print(crosstab)

print("\nğŸ’¡ KEY INSIGHT:")
print("   - high_conf_clean=1 users are NEVER labeled as cheaters")
print("   - This confirms they can be used as pseudo-negatives (with caution)")


print("="*80)
print("ğŸ“ˆ FEATURE STATISTICS")
print("="*80)

# Basic stats for features
stats_df = train[feature_cols].describe().T
stats_df['missing_%'] = train[feature_cols].isna().mean() * 100
stats_df['unique'] = train[feature_cols].nunique()
stats_df['zeros_%'] = (train[feature_cols] == 0).mean() * 100

print(stats_df.round(2))


# Missing value analysis
print("\nğŸ“Š MISSING VALUE ANALYSIS")
print("="*60)

missing_train = train[feature_cols].isna().mean() * 100
missing_test = test[feature_cols].isna().mean() * 100

missing_df = pd.DataFrame({
    'Train Missing %': missing_train,
    'Test Missing %': missing_test,
    'Diff': abs(missing_train - missing_test)
}).sort_values('Train Missing %', ascending=False)

print(missing_df.round(2))

# Visualize missing values
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(feature_cols))
width = 0.35

ax.bar(x - width/2, missing_train.values, width, label='Train', color='#3498db')
ax.bar(x + width/2, missing_test.values, width, label='Test', color='#e74c3c')
ax.set_xlabel('Features')
ax.set_ylabel('Missing %')
ax.set_title('Missing Values by Feature')
ax.set_xticks(x)
ax.set_xticklabels([f.replace('feature_', 'f') for f in feature_cols], rotation=45)
ax.legend()
plt.tight_layout()
plt.show()


# Missing value patterns - do they correlate with cheating?
print("\nğŸ”� MISSING VALUES vs CHEATING STATUS")
print("="*60)

labeled_train['total_missing'] = labeled_train[feature_cols].isna().sum(axis=1)

missing_by_class = labeled_train.groupby('is_cheating')['total_missing'].agg(['mean', 'std', 'median'])
print("\nAverage missing values per class:")
print(missing_by_class.round(2))

# Statistical test
cheater_missing = labeled_train[labeled_train['is_cheating']==1]['total_missing']
non_cheater_missing = labeled_train[labeled_train['is_cheating']==0]['total_missing']
t_stat, p_val = stats.ttest_ind(cheater_missing, non_cheater_missing)
print(f"\nT-test: t={t_stat:.3f}, p-value={p_val:.2e}")
print(f"ğŸ’¡ {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'} difference in missingness!")

# Visualize
fig, ax = plt.subplots(figsize=(10, 4))
labeled_train.boxplot(column='total_missing', by='is_cheating', ax=ax)
ax.set_title('Missing Values Distribution by Cheating Status')
ax.set_xlabel('Is Cheating')
ax.set_ylabel('Total Missing Features')
plt.suptitle('')
plt.show()


# Per-feature missing analysis vs cheating
print("\nğŸ“Š PER-FEATURE MISSING RATE BY CHEATING STATUS")
print("="*60)

cheaters_df = labeled_train[labeled_train['is_cheating'] == 1]
non_cheaters_df = labeled_train[labeled_train['is_cheating'] == 0]

feature_missing_analysis = []
for col in feature_cols:
    cheater_miss = cheaters_df[col].isna().mean() * 100
    non_cheater_miss = non_cheaters_df[col].isna().mean() * 100
    diff = cheater_miss - non_cheater_miss
    feature_missing_analysis.append({
        'feature': col,
        'cheater_missing_%': cheater_miss,
        'non_cheater_missing_%': non_cheater_miss,
        'diff': diff
    })

missing_analysis_df = pd.DataFrame(feature_missing_analysis).sort_values('diff', ascending=False)
print(missing_analysis_df.round(2).to_string(index=False))

print("\nğŸ’¡ INSIGHT: Features where cheaters have MORE missing values might indicate evasion!")


# Distribution comparison for each feature
print("="*80)
print("ğŸ“Š FEATURE DISTRIBUTIONS BY CLASS")
print("="*80)

fig, axes = plt.subplots(6, 3, figsize=(15, 20))
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    ax = axes[idx]
    
    # Get data
    cheater_vals = cheaters_df[col].dropna()
    non_cheater_vals = non_cheaters_df[col].dropna()
    
    # KDE plot
    if len(cheater_vals) > 10 and len(non_cheater_vals) > 10:
        try:
            sns.kdeplot(non_cheater_vals, ax=ax, label='Non-Cheater', color='blue', alpha=0.6)
            sns.kdeplot(cheater_vals, ax=ax, label='Cheater', color='red', alpha=0.6)
        except:
            ax.hist(non_cheater_vals, bins=30, alpha=0.5, label='Non-Cheater', color='blue', density=True)
            ax.hist(cheater_vals, bins=30, alpha=0.5, label='Cheater', color='red', density=True)
    
    ax.set_title(col.replace('feature_', 'f'), fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()


# Statistical tests for each feature
print("\nğŸ“Š STATISTICAL SIGNIFICANCE TESTS (Mann-Whitney U)")
print("="*60)

stat_tests = []
for col in feature_cols:
    cheater_vals = cheaters_df[col].dropna()
    non_cheater_vals = non_cheaters_df[col].dropna()
    
    if len(cheater_vals) > 10 and len(non_cheater_vals) > 10:
        try:
            stat, p_val = stats.mannwhitneyu(cheater_vals, non_cheater_vals, alternative='two-sided')
            effect_size = abs(cheater_vals.mean() - non_cheater_vals.mean()) / (non_cheater_vals.std() + 1e-8)
        except:
            stat, p_val, effect_size = np.nan, np.nan, np.nan
    else:
        stat, p_val, effect_size = np.nan, np.nan, np.nan
    
    stat_tests.append({
        'feature': col,
        'cheater_mean': cheater_vals.mean() if len(cheater_vals) > 0 else np.nan,
        'non_cheater_mean': non_cheater_vals.mean() if len(non_cheater_vals) > 0 else np.nan,
        'p_value': p_val,
        'effect_size': effect_size,
        'significant': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    })

stat_tests_df = pd.DataFrame(stat_tests).sort_values('effect_size', ascending=False)
print(stat_tests_df.round(4).to_string(index=False))

print("\nğŸ’¡ Features with HIGH effect size are most discriminative!")


# Correlation analysis
print("="*80)
print("ğŸ“Š FEATURE CORRELATIONS")
print("="*80)

# Correlation with target
target_corr = labeled_train[feature_cols + ['is_cheating']].corr()['is_cheating'].drop('is_cheating').sort_values()
print("\nCorrelation with is_cheating:")
print(target_corr.round(4))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Correlation with target
colors = ['green' if x > 0 else 'red' for x in target_corr.values]
axes[0].barh(target_corr.index.str.replace('feature_', 'f'), target_corr.values, color=colors)
axes[0].set_xlabel('Correlation with is_cheating')
axes[0].set_title('Feature Correlation with Target')
axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Feature-feature correlation heatmap
corr_matrix = train[feature_cols].corr()
sns.heatmap(corr_matrix, ax=axes[1], cmap='RdBu_r', center=0, 
            xticklabels=[f.replace('feature_', 'f') for f in feature_cols],
            yticklabels=[f.replace('feature_', 'f') for f in feature_cols],
            annot=False)
axes[1].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()

# Highly correlated feature pairs
print("\nğŸ”— HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.5):")
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.5:
            print(f"   {feature_cols[i]} <-> {feature_cols[j]}: {corr:.3f}")


print("="*80)
print("ğŸ•¸ï¸� SOCIAL GRAPH ANALYSIS")
print("="*80)

# Build graph
G = nx.from_pandas_edgelist(graph, 'source', 'target', create_using=nx.Graph())

print(f"\nğŸ“Š Graph Statistics:")
print(f"   Nodes: {G.number_of_nodes():,}")
print(f"   Edges: {G.number_of_edges():,}")
print(f"   Density: {nx.density(G):.6f}")

# Connected components
components = list(nx.connected_components(G))
comp_sizes = [len(c) for c in components]
print(f"\n   Connected components: {len(components):,}")
print(f"   Largest component: {max(comp_sizes):,} nodes")
print(f"   Smallest component: {min(comp_sizes)} nodes")
print(f"   Median component size: {np.median(comp_sizes):.0f}")

# Degree statistics
degrees = [d for n, d in G.degree()]
print(f"\nğŸ“Š Degree Statistics:")
print(f"   Mean degree: {np.mean(degrees):.2f}")
print(f"   Median degree: {np.median(degrees):.0f}")
print(f"   Max degree: {max(degrees)}")
print(f"   Min degree: {min(degrees)}")
print(f"   Std degree: {np.std(degrees):.2f}")


# Degree distribution visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Degree histogram
axes[0].hist(degrees, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('Count')
axes[0].set_title('Degree Distribution')
axes[0].set_xlim(0, np.percentile(degrees, 99))

# Log-log degree distribution
degree_counts = Counter(degrees)
x = list(degree_counts.keys())
y = list(degree_counts.values())
axes[1].scatter(x, y, alpha=0.5, s=10)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel('Degree (log)')
axes[1].set_ylabel('Count (log)')
axes[1].set_title('Log-Log Degree Distribution')

# Component size distribution
axes[2].hist(comp_sizes, bins=50, edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Component Size')
axes[2].set_ylabel('Count')
axes[2].set_title('Component Size Distribution')
axes[2].set_xlim(0, np.percentile(comp_sizes, 99))

plt.tight_layout()
plt.show()


# Graph features vs cheating
print("\nğŸ“Š GRAPH FEATURES VS CHEATING STATUS")
print("="*60)

# Add graph features to labeled data
degree_map = dict(G.degree())
comp_size_map = {}
for comp in components:
    size = len(comp)
    for node in comp:
        comp_size_map[node] = size

labeled_train['degree'] = labeled_train['user_hash'].map(degree_map).fillna(0)
labeled_train['component_size'] = labeled_train['user_hash'].map(comp_size_map).fillna(1)
labeled_train['in_graph'] = labeled_train['user_hash'].isin(G.nodes()).astype(int)

# Stats by class
print("\nDegree by class:")
print(labeled_train.groupby('is_cheating')['degree'].describe().round(2))

print("\nComponent size by class:")
print(labeled_train.groupby('is_cheating')['component_size'].describe().round(2))

print("\nIn graph by class:")
print(labeled_train.groupby('is_cheating')['in_graph'].mean())


# Visualize graph features by class
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Degree by class
for label, color in [(0, 'blue'), (1, 'red')]:
    data = labeled_train[labeled_train['is_cheating'] == label]['degree']
    data = data[data < data.quantile(0.99)]  # Remove outliers
    axes[0].hist(data, bins=30, alpha=0.5, label=f'Class {label}', color=color, density=True)
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('Density')
axes[0].set_title('Degree Distribution by Class')
axes[0].legend()

# Component size by class
for label, color in [(0, 'blue'), (1, 'red')]:
    data = labeled_train[labeled_train['is_cheating'] == label]['component_size']
    data = data[data < data.quantile(0.99)]
    axes[1].hist(data, bins=30, alpha=0.5, label=f'Class {label}', color=color, density=True)
axes[1].set_xlabel('Component Size')
axes[1].set_ylabel('Density')
axes[1].set_title('Component Size Distribution by Class')
axes[1].legend()

# In graph proportion
in_graph_by_class = labeled_train.groupby('is_cheating')['in_graph'].mean()
axes[2].bar(['Non-Cheater', 'Cheater'], in_graph_by_class.values, color=['blue', 'red'])
axes[2].set_ylabel('Proportion in Graph')
axes[2].set_title('Graph Membership by Class')

plt.tight_layout()
plt.show()


# Neighbor cheating analysis
print("\nğŸ“Š NEIGHBOR CHEATING ANALYSIS")
print("="*60)

user_to_label = labeled_train.set_index('user_hash')['is_cheating'].to_dict()

neighbor_cheat_ratios = {}
num_cheater_neighbors = {}

for node in G.nodes():
    neighbors = list(G.neighbors(node))
    labeled_neighbors = [n for n in neighbors if n in user_to_label]
    
    if labeled_neighbors:
        cheat_ratio = np.mean([user_to_label[n] for n in labeled_neighbors])
        n_cheaters = sum([user_to_label[n] for n in labeled_neighbors])
    else:
        cheat_ratio = 0
        n_cheaters = 0
    
    neighbor_cheat_ratios[node] = cheat_ratio
    num_cheater_neighbors[node] = n_cheaters

labeled_train['neighbor_cheat_ratio'] = labeled_train['user_hash'].map(neighbor_cheat_ratios).fillna(0)
labeled_train['num_cheater_neighbors'] = labeled_train['user_hash'].map(num_cheater_neighbors).fillna(0)

print("\nNeighbor cheat ratio by class:")
print(labeled_train.groupby('is_cheating')['neighbor_cheat_ratio'].describe().round(4))

print("\nNumber of cheater neighbors by class:")
print(labeled_train.groupby('is_cheating')['num_cheater_neighbors'].describe().round(2))


# KEY INSIGHT: Cheating is clustered!
print("\n" + "="*80)
print("ğŸ”¥ KEY INSIGHT: CHEATING IS HIGHLY CLUSTERED IN THE GRAPH!")
print("="*80)

cheater_nodes = set(labeled_train[labeled_train['is_cheating'] == 1]['user_hash'])
non_cheater_nodes = set(labeled_train[labeled_train['is_cheating'] == 0]['user_hash'])

# Check if cheaters tend to connect to other cheaters
cheater_cheater_edges = 0
cheater_non_cheater_edges = 0
non_cheater_non_cheater_edges = 0

for u, v in G.edges():
    u_cheat = u in cheater_nodes
    v_cheat = v in cheater_nodes
    u_non = u in non_cheater_nodes
    v_non = v in non_cheater_nodes
    
    if u_cheat and v_cheat:
        cheater_cheater_edges += 1
    elif (u_cheat and v_non) or (u_non and v_cheat):
        cheater_non_cheater_edges += 1
    elif u_non and v_non:
        non_cheater_non_cheater_edges += 1

print(f"\n   Cheater-Cheater edges: {cheater_cheater_edges:,}")
print(f"   Cheater-NonCheater edges: {cheater_non_cheater_edges:,}")
print(f"   NonCheater-NonCheater edges: {non_cheater_non_cheater_edges:,}")

# Calculate homophily ratio
n_cheaters = len(cheater_nodes)
n_non_cheaters = len(non_cheater_nodes)
expected_cc = (n_cheaters / (n_cheaters + n_non_cheaters)) ** 2
actual_cc = cheater_cheater_edges / (cheater_cheater_edges + cheater_non_cheater_edges + non_cheater_non_cheater_edges + 1)

print(f"\n   Expected cheater-cheater edge ratio (random): {expected_cc:.4f}")
print(f"   Actual cheater-cheater edge ratio: {actual_cc:.4f}")
print(f"   Homophily factor: {actual_cc / (expected_cc + 1e-8):.1f}x")

print("\nğŸ’¡ IMPLICATION: neighbor_cheat_ratio is a VERY STRONG signal!")


# Feature 012 (time-related?) analysis
print("="*80)
print("ğŸ”� FEATURE_012 DEEP DIVE (Suspected: Completion Time)")
print("="*80)

f012_cheater = labeled_train[labeled_train['is_cheating']==1]['feature_012']
f012_non_cheater = labeled_train[labeled_train['is_cheating']==0]['feature_012']

print(f"\nCheaters - feature_012:")
print(f"   Mean: {f012_cheater.mean():.4f}")
print(f"   Median: {f012_cheater.median():.4f}")
print(f"   % with value > 0: {(f012_cheater > 0).mean()*100:.1f}%")

print(f"\nNon-cheaters - feature_012:")
print(f"   Mean: {f012_non_cheater.mean():.4f}")
print(f"   Median: {f012_non_cheater.median():.4f}")
print(f"   % with value > 0: {(f012_non_cheater > 0).mean()*100:.1f}%")

# Thresholds
p95 = labeled_train['feature_012'].quantile(0.95)
print(f"\n95th percentile threshold: {p95:.4f}")
print(f"Cheaters above p95: {(f012_cheater > p95).mean()*100:.1f}%")
print(f"Non-cheaters above p95: {(f012_non_cheater > p95).mean()*100:.1f}%")


# Feature 015 (activity score?) analysis
print("\n" + "="*80)
print("ğŸ”� FEATURE_015 DEEP DIVE")
print("="*80)

f015_cheater = labeled_train[labeled_train['is_cheating']==1]['feature_015']
f015_non_cheater = labeled_train[labeled_train['is_cheating']==0]['feature_015']

print(f"\nCheaters - feature_015:")
print(f"   Mean: {f015_cheater.mean():.4f}")
print(f"   Median: {f015_cheater.median():.4f}")
print(f"   Std: {f015_cheater.std():.4f}")

print(f"\nNon-cheaters - feature_015:")
print(f"   Mean: {f015_non_cheater.mean():.4f}")
print(f"   Median: {f015_non_cheater.median():.4f}")
print(f"   Std: {f015_non_cheater.std():.4f}")

# Binned analysis
print("\nCheating rate by feature_015 bins:")
labeled_train['f015_bin_temp'] = pd.qcut(labeled_train['feature_015'].fillna(-1), q=10, duplicates='drop')
print(labeled_train.groupby('f015_bin_temp')['is_cheating'].agg(['mean', 'count']).round(4))


# Feature 018 analysis
print("\n" + "="*80)
print("FEATURE_018 ANALYSIS")
print("="*80)

f018_cheater = labeled_train[labeled_train['is_cheating']==1]['feature_018']
f018_non_cheater = labeled_train[labeled_train['is_cheating']==0]['feature_018']

print(f"\nCheaters - feature_018:")
print(f"   Mean: {f018_cheater.mean():.4f}")
print(f"   Median: {f018_cheater.median():.4f}")
print(f"   % missing: {f018_cheater.isna().mean()*100:.1f}%")

print(f"\nNon-cheaters - feature_018:")
print(f"   Mean: {f018_non_cheater.mean():.4f}")
print(f"   Median: {f018_non_cheater.median():.4f}")
print(f"   % missing: {f018_non_cheater.isna().mean()*100:.1f}%")

# Cheating rate by f018 value
print("\nCheating rate by feature_018 ranges:")
labeled_train['f018_range'] = pd.cut(labeled_train['feature_018'], 
                                      bins=[0, 0.25, 0.5, 0.75, 1.0], 
                                      labels=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0'])
print(labeled_train.groupby('f018_range')['is_cheating'].agg(['mean', 'count']).round(4))


# Cost structure
COST_FN = 600      # Missed cheater
COST_FP_BLOCK = 300  # Wrongly blocked
COST_FP_REVIEW = 150 # Wrongly sent to review
COST_TP_REVIEW = 5   # Correctly sent to review

print("="*80)
print("COST ANALYSIS")
print("="*80)

n_cheaters = cheaters
n_non_cheaters = non_cheaters
total = n_cheaters + n_non_cheaters

print(f"\nBaseline costs (single-action baselines):")
print(f"   All pass (predict 0): {n_cheaters * COST_FN:,}")
print(f"   All block (predict 1): {n_non_cheaters * COST_FP_BLOCK:,}")
print(f"   All review: ${n_non_cheaters * COST_FP_REVIEW + n_cheaters * COST_TP_REVIEW:,}")

print(f"\nCost ratios:")
print(f"   FN/FP_block = {COST_FN/COST_FP_BLOCK:.1f}x")
print(f"   FN/FP_review = {COST_FN/COST_FP_REVIEW:.1f}x")
print(f"   FP_block/FP_review = {COST_FP_BLOCK/COST_FP_REVIEW:.1f}x")

print("\nInsight:")
print("   - Missing a cheater costs 2x more than wrongly blocking")
print("   - Missing a cheater costs 4x more than sending to review")
print("   - Prioritize recall for cheaters")


# Optimal strategy simulation
print("\nOPTIMAL THRESHOLD STRATEGY")
print("="*60)

# Calculate break-even points
# When should we block vs review vs pass?
# E[cost|block] = P(innocent) * 300
# E[cost|review] = P(innocent) * 150 + P(cheater) * 5
# E[cost|pass] = P(cheater) * 600

# Block vs Review: 300 * P(innocent) = 150 * P(innocent) + 5 * P(cheater)
# 150 * P(innocent) = 5 * P(cheater)
# P(cheater) / P(innocent) = 30 => P(cheater) â‰ˆ 0.968

# Review vs Pass: 150 * P(innocent) + 5 * P(cheater) = 600 * P(cheater)
# 150 * P(innocent) = 595 * P(cheater)
# P(cheater) / P(innocent) = 150/595 â‰ˆ 0.252 => P(cheater) â‰ˆ 0.201

print("\nTheoretical thresholds:")
print(f"   t_low (pass vs review): ~0.201")
print(f"   t_high (review vs block): ~0.968")

print("\nNote: actual thresholds depend on calibration")


print("="*80)
print("TRAIN VS TEST DISTRIBUTION COMPARISON")
print("="*80)

# Check for distribution shift
ks_tests = []
for col in feature_cols:
    train_vals = train[col].dropna()
    test_vals = test[col].dropna()
    
    if len(train_vals) > 0 and len(test_vals) > 0:
        stat, p_val = stats.ks_2samp(train_vals, test_vals)
        ks_tests.append({
            'feature': col,
            'train_mean': train_vals.mean(),
            'test_mean': test_vals.mean(),
            'ks_stat': stat,
            'p_value': p_val,
            'shift': 'YES' if p_val < 0.01 else 'no'
        })

ks_df = pd.DataFrame(ks_tests).sort_values('ks_stat', ascending=False)
print("\nKolmogorov-Smirnov test for distribution shift:")
print(ks_df.round(4).to_string(index=False))

shifts = (ks_df['shift'] == 'YES').sum()
print(f"\nFeatures with significant distribution shift: {shifts}/{len(feature_cols)}")


# Visual comparison
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

# Pick top features with most shift
top_shift = ks_df.head(9)['feature'].tolist()

for idx, col in enumerate(top_shift):
    ax = axes[idx]
    train_vals = train[col].dropna()
    test_vals = test[col].dropna()
    
    ax.hist(train_vals, bins=30, alpha=0.5, label='Train', density=True, color='blue')
    ax.hist(test_vals, bins=30, alpha=0.5, label='Test', density=True, color='red')
    ax.set_title(f"{col}\nKS={ks_df[ks_df['feature']==col]['ks_stat'].values[0]:.3f}")
    ax.legend()

plt.suptitle('Train vs Test Distribution (Features with Most Shift)', fontsize=14)
plt.tight_layout()
plt.show()


# Per-feature missingness difference visualization (Cheaters vs Non-cheaters)
print("\n" + "="*80)
print("PER-FEATURE MISSINGNESS: CHEATERS MINUS NON-CHEATERS")
print("="*80)

try:
    viz_df = missing_analysis_df.copy()
except NameError:
    cheaters_df = labeled_train[labeled_train['is_cheating'] == 1]
    non_cheaters_df = labeled_train[labeled_train['is_cheating'] == 0]
    rows = []
    for col in feature_cols:
        rows.append({
            'feature': col,
            'cheater_missing_%': cheaters_df[col].isna().mean() * 100,
            'non_cheater_missing_%': non_cheaters_df[col].isna().mean() * 100,
        })
    viz_df = pd.DataFrame(rows)
    viz_df['diff'] = viz_df['cheater_missing_%'] - viz_df['non_cheater_missing_%']

viz_df = viz_df.sort_values('diff', ascending=False)
print(viz_df[['feature','cheater_missing_%','non_cheater_missing_%','diff']].round(2).to_string(index=False))

# Bar chart of top differences
top_n = 20
fig, ax = plt.subplots(figsize=(12, 6))
plot_df = viz_df.head(top_n)
ax.barh(plot_df['feature'].str.replace('feature_', 'f'), plot_df['diff'], color=['#e74c3c' if d>0 else '#3498db' for d in plot_df['diff']])
ax.set_xlabel('Missingness difference (cheaters - non-cheaters), %')
ax.set_title(f'Top {top_n} Features: Missingness Difference by Class')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()


# Violin plots for top discriminative features by class
print("\n" + "="*80)
print("VIOLIN PLOTS: TOP FEATURES BY CLASS")
print("="*80)

# Determine top features by effect size
try:
    top_feats = stat_tests_df.dropna(subset=['effect_size']).sort_values('effect_size', ascending=False)['feature'].head(6).tolist()
except NameError:
    # Fallback: use correlation with target
    corr_series = labeled_train[feature_cols + ['is_cheating']].corr()['is_cheating'].drop('is_cheating').abs()
    top_feats = corr_series.sort_values(ascending=False).head(6).index.tolist()

print("Selected features:", ', '.join([f.replace('feature_', 'f') for f in top_feats]))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(top_feats):
    ax = axes[i]
    df_plot = labeled_train[['is_cheating', col]].dropna()
    # Convert to category for clean hue
    df_plot['is_cheating'] = df_plot['is_cheating'].astype(int)
    try:
        sns.violinplot(data=df_plot, x='is_cheating', y=col, hue='is_cheating', split=True, ax=ax, palette=['#3498db', '#e74c3c'], legend=False)
    except Exception:
        sns.boxplot(data=df_plot, x='is_cheating', y=col, ax=ax)
    ax.set_title(col.replace('feature_', 'f'))
    ax.set_xlabel('is_cheating')

plt.tight_layout()
plt.show()


# Pairplot for top features (sampled)
print("\n" + "="*80)
print("PAIRPLOT: TOP FEATURES (SAMPLED)")
print("="*80)

# Reuse top_feats if available, otherwise compute
try:
    top_feats
except NameError:
    try:
        top_feats = stat_tests_df.dropna(subset=['effect_size']).sort_values('effect_size', ascending=False)['feature'].head(4).tolist()
    except Exception:
        corr_series = labeled_train[feature_cols + ['is_cheating']].corr()['is_cheating'].drop('is_cheating').abs()
        top_feats = corr_series.sort_values(ascending=False).head(4).index.tolist()

# Limit to 4 features for pairplot
top_feats = top_feats[:4]
print("Selected features:", ', '.join([f.replace('feature_', 'f') for f in top_feats]))

# Sample labeled data for speed
sample_n = min(5000, len(labeled_train))
sample_df = labeled_train[['is_cheating'] + top_feats].dropna().sample(n=sample_n, random_state=42)
sample_df['is_cheating'] = sample_df['is_cheating'].astype(int)

try:
    g = sns.pairplot(sample_df, vars=top_feats, hue='is_cheating', corner=True, plot_kws={'alpha':0.5, 's':20})
    g.fig.suptitle('Pairplot of Top Features by Class (sampled)', y=1.02)
    plt.show()
except Exception as e:
    print("Pairplot failed:", e)
    # Fallback: scatter matrix via pandas
    pd.plotting.scatter_matrix(sample_df[top_feats], figsize=(10, 10), diagonal='hist')
    plt.suptitle('Scatter Matrix Fallback')
    plt.show()


# Neighbor cheat ratio vs degree (scatter, sampled)
print("\n" + "="*80)
print("NEIGHBOR CHEAT RATIO VS DEGREE")
print("="*80)

# Ensure required columns exist; compute if missing
need_cols = ['neighbor_cheat_ratio', 'degree']
if not all(col in labeled_train.columns for col in need_cols):
    # Build degree map and neighbor cheat ratio
    degree_map = dict(G.degree())
    user_to_label = labeled_train.set_index('user_hash')['is_cheating'].to_dict()
    ratios = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        labeled_neighbors = [n for n in neighbors if n in user_to_label]
        if labeled_neighbors:
            ratios[node] = np.mean([user_to_label[n] for n in labeled_neighbors])
        else:
            ratios[node] = 0.0
    labeled_train['degree'] = labeled_train['user_hash'].map(degree_map).fillna(0)
    labeled_train['neighbor_cheat_ratio'] = labeled_train['user_hash'].map(ratios).fillna(0.0)

# Sample for plotting
plot_df = labeled_train[['is_cheating','degree','neighbor_cheat_ratio']].dropna().copy()
plot_df['is_cheating'] = plot_df['is_cheating'].astype(int)
plot_df = plot_df.sample(n=min(10000, len(plot_df)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plot_df['is_cheating'].map({0:'#3498db', 1:'#e74c3c'})
ax.scatter(plot_df['degree'], plot_df['neighbor_cheat_ratio'], c=colors, alpha=0.4, s=12)
ax.set_xlabel('Degree')
ax.set_ylabel('Neighbor cheat ratio')
ax.set_title('Neighbor cheat ratio vs Degree (sampled)')
ax.set_xlim(left=0)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# Binned cheating rate by degree
plot_df['degree_bin'] = pd.qcut(plot_df['degree'], q=10, duplicates='drop')
rate_df = plot_df.groupby('degree_bin')['is_cheating'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(rate_df)), rate_df['is_cheating'], marker='o')
ax.set_xticks(range(len(rate_df)))
ax.set_xticklabels(rate_df['degree_bin'].astype(str), rotation=45)
ax.set_ylabel('Cheating rate')
ax.set_title('Cheating rate by degree decile (sampled)')
plt.tight_layout()
plt.show()


# Ego-graph visualization around a sampled cheater
print("\n" + "="*80)
print("EGO-GRAPH: CHEATER'S 1-HOP NEIGHBORHOOD")
print("="*80)

cheater_nodes = set(labeled_train[labeled_train['is_cheating'] == 1]['user_hash'])
cheater_nodes_in_graph = list(cheater_nodes & set(G.nodes()))

if len(cheater_nodes_in_graph) == 0:
    print("No labeled cheaters found in graph.")
else:
    node = np.random.choice(cheater_nodes_in_graph)
    ego = nx.ego_graph(G, node, radius=1)

    # Color by label
    label_map = labeled_train.set_index('user_hash')['is_cheating'].to_dict()
    node_colors = []
    for n in ego.nodes():
        if label_map.get(n, np.nan) == 1:
            node_colors.append('#e74c3c')  # Cheater
        elif label_map.get(n, np.nan) == 0:
            node_colors.append('#3498db')  # Non-cheater
        else:
            node_colors.append('#95a5a6')  # Unlabeled

    pos = nx.spring_layout(ego, seed=42)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(ego, pos, node_color=node_colors, node_size=120, alpha=0.9)
    nx.draw_networkx_edges(ego, pos, alpha=0.3)
    plt.title("Cheater's 1-Hop Ego Network (red=cheater, blue=non-cheater, gray=unlabeled)")
    plt.axis('off')
    plt.show()


# Train vs Test KDE for top shift features
print("\n" + "="*80)
print("TRAIN VS TEST: KDE FOR TOP SHIFT FEATURES")
print("="*80)

try:
    top_shift = ks_df.sort_values('ks_stat', ascending=False)['feature'].head(6).tolist()
except NameError:
    # Fallback: compute KS quickly
    rows = []
    for col in feature_cols:
        tv = train[col].dropna()
        sv = test[col].dropna()
        if len(tv) > 0 and len(sv) > 0:
            stat, p_val = stats.ks_2samp(tv, sv)
            rows.append({'feature': col, 'ks_stat': stat})
    top_shift = pd.DataFrame(rows).sort_values('ks_stat', ascending=False)['feature'].head(6).tolist()

print("Selected features:", ', '.join([f.replace('feature_', 'f') for f in top_shift]))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(top_shift):
    ax = axes[i]
    tv = train[col].dropna()
    sv = test[col].dropna()
    try:
        sns.kdeplot(tv, ax=ax, label='Train', color='#3498db')
        sns.kdeplot(sv, ax=ax, label='Test', color='#e74c3c')
    except Exception:
        ax.hist(tv, bins=30, alpha=0.5, label='Train', density=True, color='#3498db')
        ax.hist(sv, bins=30, alpha=0.5, label='Test', density=True, color='#e74c3c')
    ax.set_title(col.replace('feature_', 'f'))
    ax.legend()

plt.tight_layout()
plt.show()

