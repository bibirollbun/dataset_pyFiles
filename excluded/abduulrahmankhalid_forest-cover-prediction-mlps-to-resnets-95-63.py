# Essential imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 100)
plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')


import random, numpy as np, torch
# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)


df_train = pd.read_csv('/kaggle/input/beyond-nti-r-1-c-2/train.csv').drop(columns=['Id'],axis=1)

df_train


df_train.info()


LABEL = "Cover_Type"

NUM_FEATS = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                  'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

CAT_FEATS = [x for x in df_train.columns if x not in NUM_FEATS and x != LABEL]
ORG_FEATS = df_train.drop(LABEL, axis=1).columns.tolist()


def reduceMemoryUsage(df):

  df_reduced = df.copy()

  for col in NUM_FEATS:
    df_reduced[col] = df_reduced[col].astype('int16')

  df_reduced['Slope'] = df_reduced['Slope'].astype('int8')

  for col in CAT_FEATS:
    df_reduced[col] = df_reduced[col].astype('category')

  return df_reduced


df_train_reduced = reduceMemoryUsage(df_train)


df_train_reduced.info()


dtypes = pd.DataFrame(df_train.dtypes,columns=["Data Type"])
dtypes["Unique Values"]=df_train.nunique()
dtypes["Null Values"]=df_train.isnull().sum()
dtypes["% null Values"]=df_train.isnull().sum()/len(df_train)
dtypes.sort_values(by="Null Values" , ascending=False).style.background_gradient(cmap='YlOrRd',axis=0)


# Get original numerical features (before feature engineering)
original_num_feats = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                     'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                     'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

org_train_df = df_train.copy()

print("=== NUMERICAL FEATURES ANALYSIS ===")
print(f"Number of original numerical features: {len(original_num_feats)}")
print(f"Original numerical features: {original_num_feats}")

# Check for missing values
missing_values = df_train[original_num_feats].isnull().sum()
print(f"\n=== MISSING VALUES ===")
if missing_values.sum() == 0:
    print("âœ… No missing values in numerical features!")
else:
    print("â›” Missing values found:")
    print(missing_values[missing_values > 0])

# Check for duplicate values
duplicate_values = df_train.duplicated().sum()
print(f"\n=== DUPLICATE VALUES ===")
if duplicate_values.sum() == 0:
    print("âœ… No duplicate values in numerical features!")
else:
    print(f"â›” Duplicate values found:{len(duplicate_values)}")

# Basic statistics
num_stats = df_train[original_num_feats].describe().T
print("\n=== BASIC STATISTICS ===")
display(num_stats.style.background_gradient(cmap='viridis').format(precision=2))


# Distribution analysis
print("\n=== DISTRIBUTION ANALYSIS ===")
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.ravel()

for i, feature in enumerate(original_num_feats):
    if i < len(axes):
        # Histogram
        axes[i].hist(df_train[feature], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

        # Add statistics text
        mean_val = df_train[feature].mean()
        std_val = df_train[feature].std()
        skew_val = df_train[feature].skew()
        axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
        axes[i].text(0.02, 0.98, f'Skewness: {skew_val:.2f}', transform=axes[i].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[i].legend()

# Remove extra subplots
for i in range(len(original_num_feats), len(axes)):
    axes[i].remove()

plt.tight_layout()
plt.show()

# Skewness analysis
skewness = df_train[original_num_feats].skew().sort_values(ascending=False)
print("\n=== SKEWNESS ANALYSIS ===")
print("Features sorted by skewness (higher = more right-skewed):")
for feature, skew_val in skewness.items():
    skew_level = "Highly skewed" if abs(skew_val) > 1 else "Moderately skewed" if abs(skew_val) > 0.5 else "Approximately normal"
    print(f"{feature}: {skew_val:.3f} ({skew_level})")


def plot_distributions_iqr(data, features, iqr_multiplier=1.5, bins=50,
                           num_plots_per_line=4, figsize_per_row=(16,6),
                           return_outlier_df=False):
    """
    Plot distributions with IQR-based outlier marking (Tukey method).
    """
    features = list(features)
    num_features = len(features)
    num_rows = (num_features + num_plots_per_line - 1) // num_plots_per_line
    fig_w, fig_h_per_row = figsize_per_row
    fig, axs = plt.subplots(num_rows, num_plots_per_line, figsize=(fig_w, fig_h_per_row * num_rows))
    axs = np.array(axs).reshape(-1)  # flatten for indexing

    outlier_records = []

    for i, feature in enumerate(features):
        ax = axs[i]
        if feature not in data.columns:
            ax.set_title(f"{feature} (missing)")
            ax.axis('off')
            continue

        s = data[feature].dropna()
        if not np.issubdtype(s.dtype, np.number):
            ax.set_title(f"{feature} (non-numeric)")
            ax.axis('off')
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        # degenerate / constant column
        if iqr == 0 or np.isnan(iqr):
            ax.hist(s, bins=bins, alpha=0.7, color='blue')
            ax.axvline(s.mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
            ax.set_title(f"{feature} (constant or zero IQR)")
            ax.legend(loc='upper right', fontsize=9)
            continue

        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)

        # mask outliers
        outlier_mask = (s < lower_bound) | (s > upper_bound)
        n_outliers = int(outlier_mask.sum())
        outlier_vals = s[outlier_mask].values

        # histogram
        ax.hist(s, bins=bins, alpha=0.7, color='blue')
        # vertical lines
        ax.axvline(q1, color='purple', linestyle='dashed', linewidth=1.8, label='Q1')
        ax.axvline(q3, color='purple', linestyle='dashed', linewidth=1.8, label='Q3')
        ax.axvline(lower_bound, color='orange', linestyle='dashed', linewidth=2, label=f'Lower fence ({iqr_multiplier}Ã—IQR)')
        ax.axvline(upper_bound, color='orange', linestyle='dashed', linewidth=2, label=f'Upper fence ({iqr_multiplier}Ã—IQR)')
        ax.set_title(feature)
        ax.set_xlabel(feature)

        # overlay outliers as red x with slight y jitter
        if outlier_vals.size > 0:
            y_jitter = np.random.uniform(0.02, 0.06, size=outlier_vals.shape)
            ax.scatter(outlier_vals, y_jitter, marker='x', color='red', label=f'Outliers (n={n_outliers})')
            for idx, val in zip(s[outlier_mask].index, outlier_vals):
                outlier_records.append({
                    'feature': feature,
                    'index': idx,
                    'value': val,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
        else:
            # Ensure a legend entry still appears for consistency
            ax.plot([], [], color='red', marker='x', linestyle='None', label=f'Outliers (n={n_outliers})')

        # place legend in upper right
        ax.legend(loc='upper right', fontsize=9)

    # turn off any empty axes
    for j in range(num_features, num_rows * num_plots_per_line):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

    if return_outlier_df:
        if outlier_records:
            return pd.DataFrame(outlier_records).sort_values(['feature','index']).reset_index(drop=True)
        else:
            return pd.DataFrame(columns=['feature','index','value','q1','q3','iqr','lower_bound','upper_bound'])

    plt.style.use('ggplot')

iqr_multiplier = 1.5

outliers_iqr = plot_distributions_iqr(df_train, NUM_FEATS, iqr_multiplier=iqr_multiplier, return_outlier_df=True)


# Outlier detection using IQR
print("\n=== OUTLIER ANALYSIS ===\n")
outlier_summary = {}
for feature in original_num_feats:
    Q1 = df_train[feature].quantile(0.25)
    Q3 = df_train[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_train[(df_train[feature] < lower_bound) | (df_train[feature] > upper_bound)][feature]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df_train)) * 100

    outlier_summary[feature] = {
        'count': outlier_count,
        'percentage': outlier_percentage,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

    print(f"{feature}: {outlier_count} outliers ({outlier_percentage:.2f}%)")

print("\nFeatures with high outlier percentage (>5%):")
high_outlier_features = [f for f, stats in outlier_summary.items() if stats['percentage'] > 5]
if high_outlier_features:
    for feature in high_outlier_features:
        print(f"- {feature}: {outlier_summary[feature]['percentage']:.2f}%")
else:
    print("No features with high outlier percentage found.")

print(f"\nâš ï¸� Detected {len(outliers_iqr)} samples that are more than {iqr_multiplier}Ã—IQR away from Q1 and Q3 = {round((len(outliers_iqr) / df_train.shape[0]) *100, 2)}% of the data")


# Categorical features analysis
print("=== CATEGORICAL FEATURES ANALYSIS ===\n")

# Separate wilderness and soil features
wilderness_features = [col for col in df_train.columns if 'Wilderness_Area' in col]
soil_features = [col for col in df_train.columns if 'Soil_Type' in col]

# Combine all categorical features
all_categorical_features = wilderness_features + soil_features

print(f"Wilderness Area features: {len(wilderness_features)}")
print(f"Soil Type features: {len(soil_features)}")
print(f"Total categorical features: {len(wilderness_features) + len(soil_features)}")


# Quick visualization of all categorical features distribution
print("=== ALL CATEGORICAL FEATURES DISTRIBUTION ===")

# Calculate percentage of ones for each categorical feature
cat_percentages = []
for feature in all_categorical_features:
    feature_data = org_train_df[feature].astype(int)
    pct_ones = (feature_data.sum() / len(feature_data)) * 100
    cat_percentages.append({
        'Feature': feature.replace('_', ' '),
        'Percentage_Ones': pct_ones
    })

cat_pct_df = pd.DataFrame(cat_percentages).sort_values('Percentage_Ones', ascending=True)

# Create a horizontal bar chart showing the sparsity
plt.figure(figsize=(12, 16))
y_pos = range(len(cat_pct_df))
colors = ['darkred' if pct < 1 else 'red' if pct < 5 else 'orange' if pct < 20 else 'green'
          for pct in cat_pct_df['Percentage_Ones']]

bars = plt.barh(y_pos, cat_pct_df['Percentage_Ones'], color=colors, alpha=0.7, edgecolor='black')
plt.yticks(y_pos, cat_pct_df['Feature'], fontsize=10)
plt.xlabel('Percentage of Ones (%)', fontsize=12, fontweight='bold')
plt.title('All Categorical Features Distribution\n(Percentage of Non-Zero Values)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, cat_pct_df['Percentage_Ones'])):
    if pct > 1:  # Only show labels for bars that are visible
        plt.text(pct + max(cat_pct_df['Percentage_Ones']) * 0.01, i, f'{pct:.1f}%',
                va='center', fontsize=8, fontweight='bold')
    else:
        plt.text(0.1, i, f'{pct:.2f}%', va='center', fontsize=8, fontweight='bold', color='gray')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='darkred', label='<1% (Very Sparse)'),
    Patch(facecolor='red', label='1-5% (Sparse)'),
    Patch(facecolor='orange', label='5-20% (Moderate)'),
    Patch(facecolor='green', label='>20% (Dense)')
]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.show()

# Summary stats
sparse_features = len(cat_pct_df[cat_pct_df['Percentage_Ones'] < 5])
very_sparse_features = len(cat_pct_df[cat_pct_df['Percentage_Ones'] < 1])
print(f"\nğŸ“Š Categorical Features Summary:")
print(f"â€¢ Total features: {len(cat_pct_df)}")
print(f"â€¢ Very sparse (<1%): {very_sparse_features} features")
print(f"â€¢ Sparse (1-5%): {sparse_features - very_sparse_features} features")
print(f"â€¢ Moderate-Dense (>5%): {len(cat_pct_df) - sparse_features} features")
print(f"â€¢ Average sparsity: {100 - cat_pct_df['Percentage_Ones'].mean():.1f}% zeros")
print(f"\nâœ… Most categorical features are highly sparse (mostly zeros)")


# Wilderness Areas Analysis
print("\n=== WILDERNESS AREAS ANALYSIS ===\n")

# Convert to numeric before summing
wilderness_counts = df_train[wilderness_features].astype(int).sum().sort_values(ascending=False)
print("Wilderness Area distribution:")

wilderness_names = {
    'Wilderness_Area1': 'Rawah Wilderness Area',
    'Wilderness_Area2': 'Neota Wilderness Area',
    'Wilderness_Area3': 'Comanche Peak Wilderness Area',
    'Wilderness_Area4': 'Cache la Poudre Wilderness Area'
}

for area, count in wilderness_counts.items():
    percentage = (count / len(df_train)) * 100
    area_name = wilderness_names.get(area, area)
    print(f"{area_name}: {count:,} samples ({percentage:.2f}%)")

# Visualize wilderness areas
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Wilderness area distribution
wilderness_counts.plot(kind='bar', ax=axes[0], color='lightgreen', edgecolor='black')
axes[0].set_title('Wilderness Area Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels([f'Area {i+1}' for i in range(len(wilderness_counts))], rotation=0)
for i, v in enumerate(wilderness_counts.values):
    axes[0].text(i, v + max(wilderness_counts.values) * 0.01, f'{v:,}', ha='center', va='bottom')

# Wilderness area vs Cover Type
wilderness_covertype = pd.crosstab(
    df_train[wilderness_features].astype(str).idxmax(axis=1), # Convert to string
    df_train[LABEL],
    normalize='index'
) * 100

sns.heatmap(wilderness_covertype, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Cover Type Distribution by Wilderness Area (%)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Cover Type')
axes[1].set_ylabel('Wilderness Area')

plt.tight_layout()
plt.show()


# Soil Types Analysis
print("\n=== SOIL TYPES ANALYSIS ===")

# Convert to numeric before summing
soil_counts = df_train[soil_features].astype(int).sum().sort_values(ascending=False)
print(f"Active soil types: {(soil_counts > 0).sum()}/{len(soil_features)}")
print(f"Unused soil types: {(soil_counts == 0).sum()}")

# Show top 15 most common soil types
print("\nTop 15 most common soil types:")
for i, (soil, count) in enumerate(soil_counts.head(15).items()):
    percentage = (count / len(df_train)) * 100
    soil_num = soil.replace('Soil_Type', '')
    print(f"{soil} (Type {soil_num}): {count:,} samples ({percentage:.2f}%)")

# Unused soil types
unused_soils = soil_counts[soil_counts == 0]
if len(unused_soils) > 0:
    print(f"\nUnused soil types: {list(unused_soils.index)}")
else:
    print("\nAll soil types are used in the dataset.")

# Visualize top soil types and cover type distribution
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Top 15 soil types distribution
top_soils = soil_counts.head(15)
bars = axes[0].bar(range(len(top_soils)), top_soils.values, color='brown', alpha=0.7, edgecolor='black')
axes[0].set_title('Top 15 Soil Types Distribution', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Soil Type')
axes[0].set_ylabel('Count')
axes[0].set_xticks(range(len(top_soils)))
axes[0].set_xticklabels([s.replace('Soil_Type', 'Type ') for s in top_soils.index], rotation=45)

# Add count labels on bars
for i, (bar, val) in enumerate(zip(bars, top_soils.values)):
    axes[0].text(i, val + max(top_soils.values) * 0.01, f'{val:,}', ha='center', va='bottom', fontsize=9)

top_15_soils = soil_counts.head(15).index.tolist()

# Create a DataFrame for easier analysis
soil_covertype_data = []
for soil_type in top_15_soils:
    soil_mask = df_train[soil_type].astype(int) == 1
    if soil_mask.sum() > 0:  # Only analyze if soil type is present
        cover_dist = df_train[soil_mask][LABEL].value_counts(normalize=True).sort_index() * 100
        for cover_type, percentage in cover_dist.items():
            soil_covertype_data.append({
                'Soil_Type': soil_type,
                'Cover_Type': cover_type,
                'Percentage': percentage
            })

soil_covertype_df = pd.DataFrame(soil_covertype_data)
soil_covertype_pivot = soil_covertype_df.pivot(index='Soil_Type', columns='Cover_Type', values='Percentage').fillna(0)

# Create heatmap
sns.heatmap(soil_covertype_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Percentage'})
axes[1].set_title('Cover Type Distribution by Top 15 Soil Types (%)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Cover Type')
axes[1].set_ylabel('Soil Type')
axes[1].set_yticklabels([s.replace('Soil_Type', 'Type ') for s in soil_covertype_pivot.index], rotation=0)

plt.tight_layout()
plt.show()


# Feature importance analysis for categorical features
print("\n=== CATEGORICAL FEATURE IMPORTANCE ===")
from sklearn.feature_selection import chi2

cat_feature_importance = []

for feature in all_categorical_features:
    if df_train[feature].astype(int).sum() > 0:  # Only consider features that are actually used
        chi2_stat, p_value = chi2(df_train[[feature]].astype(int), df_train[LABEL])
        cat_feature_importance.append({
            'Feature': feature,
            'Chi2_Statistic': chi2_stat[0],
            'P_Value': p_value[0],
            'Count': df_train[feature].astype(int).sum()
        })

cat_importance_df = pd.DataFrame(cat_feature_importance)
cat_importance_df = cat_importance_df.sort_values('Chi2_Statistic', ascending=False)

print("Top 15 categorical features by Chi-square statistic:")
display(cat_importance_df.head(15).style.background_gradient(subset=['Chi2_Statistic'], cmap='viridis').format({
    'Chi2_Statistic': '{:.2f}',
    'P_Value': '{:.2e}',
    'Count': '{:,}'
}))

# Check for features with very low representation
low_representation = cat_importance_df[cat_importance_df['Count'] < len(df_train) * 0.01]  # Less than 1%
if len(low_representation) > 0:
    print(f"\nâš ï¸�  Features with very low representation (<1% of data): {len(low_representation)}")
    print("Consider removing these features:")
    for _, row in low_representation.iterrows():
        print(f"- {row['Feature']}: {row['Count']} samples ({row['Count']/len(df_train)*100:.3f}%)")
else:
    print("\nâœ… All categorical features have reasonable representation.")

# Summary statistics
print("\n=== CATEGORICAL FEATURES SUMMARY ===\n")
print(f"Total categorical features: {len(all_categorical_features)}")
print(f"Active features (count > 0): {len(cat_importance_df)}")
print(f"Unused features: {len(all_categorical_features) - len(cat_importance_df)}")
print(f"Features with >5% representation: {len(cat_importance_df[cat_importance_df['Count'] > len(df_train) * 0.05])}")
print(f"Features with 1-5% representation: {len(cat_importance_df[(cat_importance_df['Count'] > len(df_train) * 0.01) & (cat_importance_df['Count'] <= len(df_train) * 0.05)])}")
print(f"Features with <1% representation: {len(low_representation)}")


# Target variable analysis
print("=== TARGET VARIABLE ANALYSIS ===")
print(f"Dataset shape: {df_train.shape}")
print(f"Target variable: {LABEL}")
print(f"Number of classes: {df_train[LABEL].nunique()}")
print(f"Class range: {df_train[LABEL].min()} to {df_train[LABEL].max()}")

# Class distribution
class_counts = df_train[LABEL].value_counts().sort_index()
class_percentages = df_train[LABEL].value_counts(normalize=True).sort_index() * 100

print("\n=== CLASS DISTRIBUTION ===")
for class_id, count in class_counts.items():
    percentage = class_percentages[class_id]
    cover_types = {
        1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
        4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"
    }
    print(f"Class {class_id} ({cover_types.get(class_id, 'Unknown')}): {count:,} samples ({percentage:.2f}%)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
class_counts.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Cover Type Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Cover Type')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + max(class_counts.values) * 0.01, f'{v:,}', ha='center', va='bottom')

# Remove extra subplots
axes[1].remove()


plt.tight_layout()
plt.show()

# Check for class imbalance
max_count = class_counts.max()
min_count = class_counts.min()
imbalance_ratio = max_count / min_count
print(f"\n=== CLASS IMBALANCE ANALYSIS ===")
print(f"Most frequent class: {class_counts.idxmax()} with {max_count:,} samples")
print(f"Least frequent class: {class_counts.idxmin()} with {min_count:,} samples")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("âš ï¸�  Significant class imbalance detected!")
else:
    print("âœ… Class distribution is relatively balanced.")


# Correlation analysis
print("=== CORRELATION ANALYSIS ===")

# Feature-feature correlations
correlation_matrix = df_train[original_num_feats].corr()

# High correlation pairs
print("\n=== HIGH CORRELATION PAIRS (|r| > 0.5) ===")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            high_corr_pairs.append({
                'Feature1': correlation_matrix.columns[i],
                'Feature2': correlation_matrix.columns[j],
                'Correlation': corr_val
            })

if high_corr_pairs:
    for pair in sorted(high_corr_pairs, key=lambda x: abs(x['Correlation']), reverse=True):
        print(f"{pair['Feature1']} â†” {pair['Feature2']}: {pair['Correlation']:.3f}")
else:
    print("No highly correlated feature pairs found.")

# Create correlation heatmap
plt.figure(figsize=(16, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# Feature-target correlations
print("\n=== FEATURE-TARGET CORRELATIONS ===")
target_correlations = []
for feature in original_num_feats:
    corr_val = df_train[feature].corr(df_train[LABEL])
    target_correlations.append({'Feature': feature, 'Correlation': corr_val})

target_correlations = sorted(target_correlations, key=lambda x: abs(x['Correlation']), reverse=True)

print("Features ranked by correlation with target (Cover_Type):")
for item in target_correlations:
    corr_strength = "Strong" if abs(item['Correlation']) > 0.5 else "Moderate" if abs(item['Correlation']) > 0.3 else "Weak"
    print(f"{item['Feature']}: {item['Correlation']:.3f} ({corr_strength})")

# Visualize feature-target correlations
feature_names = [item['Feature'] for item in target_correlations]
corr_values = [item['Correlation'] for item in target_correlations]

plt.figure(figsize=(12, 6))
colors = ['red' if x < 0 else 'blue' for x in corr_values]
bars = plt.barh(range(len(feature_names)), corr_values, color=colors, alpha=0.7)
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Correlation with Cover_Type')
plt.title('Feature-Target Correlations', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.grid(axis='x', alpha=0.3)

# Add correlation values on bars
for i, (bar, val) in enumerate(zip(bars, corr_values)):
    plt.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}',
             va='center', ha='left' if val >= 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.show()


# Mutual Information analysis (for non-linear relationships)
print("\n=== MUTUAL INFORMATION ANALYSIS ===")
from sklearn.feature_selection import mutual_info_classif

sample_df = df_train.sample(frac=0.1, random_state=42)

mi_scores = mutual_info_classif(sample_df[original_num_feats], sample_df[LABEL], random_state=42)
mi_results = list(zip(original_num_feats, mi_scores))
mi_results = sorted(mi_results, key=lambda x: x[1], reverse=True)

# Create a DataFrame from the results
mi_df = pd.DataFrame({
    'Feature': original_num_feats,
    'MI_Score': mi_scores,
    'Feature_Type': 'Numerical'
}).sort_values('MI_Score', ascending=False).reset_index(drop=True)

print("\nTop 20 features by Mutual Information Score:")
display(mi_df.head(20).style.background_gradient(subset=['MI_Score'], cmap='viridis').format({'MI_Score': '{:.4f}'}))

# Visualize top features
plt.figure(figsize=(12, 6))
top_features = mi_df.head(20)

# Since all analyzed features are numerical, the color will be consistent.
colors = ['lightcoral' if ft == 'Numerical' else 'lightblue' for ft in top_features['Feature_Type']]
bars = plt.bar(top_features['Feature'], top_features['MI_Score'], color=colors, edgecolor='black', alpha=0.8)

plt.title('Top 20 Features by Mutual Information Score', fontsize=16, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.xticks(rotation=45, ha='right')

# Add value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.001, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

from matplotlib.patches import Patch
# Add legend
legend_elements = [Patch(facecolor='lightcoral', edgecolor='black', label='Numerical')]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()


# Compare correlation vs mutual information
comparison_data = []

for feature in original_num_feats:
    corr_val = next(item['Correlation'] for item in target_correlations if item['Feature'] == feature)
    mi_val = next(score for feat, score in mi_results if feat == feature)
    comparison_data.append({'Feature': feature, 'Correlation': abs(corr_val), 'MI': mi_val})

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('MI', ascending=False).reset_index(drop=True)

# Visualize correlation

print("\n=== CORRELATION vs MUTUAL INFORMATION COMPARISON ===")
display(comparison_df.style.background_gradient(subset=['Correlation', 'MI'], cmap='viridis').format(precision=4))


# ===== STEP 1: DATA BACKUP & VALIDATION =====

print("=== DATA PREPARATION PIPELINE ===")
print("Step 1: Data Backup & Validation")

# Create backup of original data
org_train_df = df_train.copy()
print(f"âœ… Original data backed up: {org_train_df.shape}")

# Data validation
print(f"\nğŸ“Š Initial Data Summary:")
print(f"  â€¢ Dataset shape: {df_train.shape}")
print(f"  â€¢ Memory usage: {df_train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  â€¢ Features: {len(NUM_FEATS)} numerical, {len(CAT_FEATS)} categorical")
print(f"  â€¢ Target variable: {LABEL} with {df_train[LABEL].nunique()} classes")


print("âœ… Step 1 Complete: Data validated and backed up")


# ===== STEP 2: DATA QUALITY ASSESSMENT =====
print("\nStep 2: Data Quality Assessment")

# 2.1 Missing Values Analysis
print(f"\nğŸ”� Missing Values Analysis:")
missing_values = df_train.isnull().sum()
total_missing = missing_values.sum()
print(f"  â€¢ Total missing values: {total_missing}")

if total_missing > 0:
    missing_percentage = (missing_values / len(df_train)) * 100
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_values[missing_values > 0],
        'Missing_Percentage': missing_percentage[missing_values > 0]
    }).sort_values('Missing_Percentage', ascending=False)
    print(missing_summary)
else:
    print("  âœ… No missing values detected")

# 2.2 Duplicate Records Analysis
print(f"\nğŸ”� Duplicate Records Analysis:")
duplicate_count = df_train.duplicated().sum()
duplicate_percentage = (duplicate_count / len(df_train)) * 100
print(f"  â€¢ Duplicate records: {duplicate_count} ({duplicate_percentage:.3f}%)")

if duplicate_count > 0:
    print("  âš ï¸� Duplicates detected - will be removed")
    df_train.drop_duplicates(inplace=True)
    print(f"  âœ… Removed {duplicate_count} duplicates. New shape: {df_train.shape}")
else:
    print("  âœ… No duplicate records found")

# 2.3 Categorical Features Validation
print(f"\nğŸ”� Categorical Features Validation:")
cat_issues = []
for col in CAT_FEATS:
    unique_vals = df_train[col].unique()
    # Check for unexpected values in binary categorical features
    if len(unique_vals) > 2:
        cat_issues.append(f"{col}: {len(unique_vals)} values {list(unique_vals)}")
    elif not all(val in [0, 1] for val in unique_vals if pd.notna(val)):
        cat_issues.append(f"{col}: Non-binary values {list(unique_vals)}")

if cat_issues:
    print("  âš ï¸� Categorical feature issues detected:")
    for issue in cat_issues:
        print(f"    - {issue}")
else:
    print("  âœ… All categorical features validated")

# 2.4 Target Variable Validation
print(f"\nğŸ”� Target Variable Validation:")
target_values = df_train[LABEL].unique()
expected_classes = [1, 2, 3, 4, 5, 6, 7]
missing_classes = set(expected_classes) - set(target_values)
unexpected_classes = set(target_values) - set(expected_classes)

print(f"  â€¢ Found classes: {target_values}")
if missing_classes:
    print(f"  âš ï¸� Missing expected classes: {sorted(missing_classes)}")
if unexpected_classes:
    print(f"  âš ï¸� Unexpected classes found: {sorted(unexpected_classes)}")

print("âœ… Step 2 Complete: Data quality assessed")


# ===== STEP 3: DATA CLEANING =====
print("\nStep 3: Data Cleaning")

# 3.1 Analyze problematic classes and features
print(f"\nğŸ”� Identifying Data Issues:")

# Analyze class distribution for anomalies
class_counts = df_train[LABEL].value_counts().sort_index()
print(f"  â€¢ Class distribution analysis:")
for class_id, count in class_counts.items():
    percentage = (count / len(df_train)) * 100
    print(f"    Class {class_id}: {count:,} samples ({percentage:.3f}%)")

# Identify extremely rare classes (< 0.001% of data)
rare_threshold = len(df_train) * 0.00001  # 0.001%
rare_classes = class_counts[class_counts < rare_threshold]
print(f"\n  â€¢ Extremely rare classes (< {rare_threshold:.0f} samples): {list(rare_classes.index)}")

# 3.2 Analyze zero-variance categorical features
print(f"\nğŸ”� Zero-Variance Feature Analysis:")
zero_variance_features = []
for col in CAT_FEATS:
    if df_train[col].nunique() <= 1:
        zero_variance_features.append(col)

print(f"  â€¢ Zero-variance features: {zero_variance_features}")

# 3.3 Smart cleaning decisions
print(f"\nğŸ§¹ Cleaning Decisions:")

# Remove extremely rare classes (if any)
if len(rare_classes) > 0:
    for rare_class in rare_classes.index:
        rare_indices = df_train[df_train[LABEL] == rare_class].index
        print(f"  â€¢ Removing {len(rare_indices)} samples of extremely rare class {rare_class}")
        df_train.drop(index=rare_indices, inplace=True)

# Remove zero-variance features
if zero_variance_features:
    print(f"  â€¢ Removing zero-variance features: {zero_variance_features}")
    df_train.drop(columns=zero_variance_features, inplace=True)
    # Update feature lists
    for feat in zero_variance_features:
        if feat in CAT_FEATS:
            CAT_FEATS.remove(feat)

# Reset index after dropping rows
df_train.reset_index(drop=True, inplace=True)

print(f"\nğŸ“Š Post-Cleaning Summary:")
print(f"  â€¢ Final dataset shape: {df_train.shape}")
print(f"  â€¢ Remaining classes: {sorted(df_train[LABEL].unique())}")
print(f"  â€¢ Features: {len(NUM_FEATS)} numerical, {len(CAT_FEATS)} categorical")

print("âœ… Step 3 Complete: Data cleaned")


# ===== STEP 4: TRAIN/TEST DATA SPLITTING =====

print("\nStep 4:  Train/Test Data Splitting")

from sklearn.model_selection import train_test_split

def test_data_split(df, target_column, test_size=0.05, random_state=42):
    """
    Data splitting with train and test sets
    """
    print(f"\nğŸ”„ Performing stratified train/test split...")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"  â€¢ Total samples: {len(df):,}")
    print(f"  â€¢ Features: {X.shape[1]}")
    print(f"  â€¢ Classes: {y.nunique()}")

    # Check class distribution before split
    class_counts = y.value_counts().sort_index()
    min_class_count = class_counts.min()

    # Single split: train vs test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Print split summary
    print(f"\nğŸ“Š Split Summary:")
    print(f"  â€¢ Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"  â€¢ Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

    # Validate class distribution preservation
    print(f"\nğŸ”� Class Distribution Validation:")
    original_dist = y.value_counts(normalize=True).sort_index() * 100
    train_dist = y_train.value_counts(normalize=True).sort_index() * 100
    test_dist = y_test.value_counts(normalize=True).sort_index() * 100

    # Create DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Original': original_dist,
        'Training': train_dist,
        'Test': test_dist
    })

    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    comparison_df.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_title('Class Distribution Comparison Across Splits', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cover Type Class')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Dataset Split')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.show()

    print(f"  â€¢ Original dataset class distribution:")
    for class_id, percentage in original_dist.items():
        print(f"    Class {class_id}: {percentage:.2f}%")

    print(f"  â€¢ Training set class distribution:")
    for class_id, percentage in train_dist.items():
        print(f"    Class {class_id}: {percentage:.2f}%")

    print(f"  â€¢ Test set class distribution:")
    for class_id, percentage in test_dist.items():
        print(f"    Class {class_id}: {percentage:.2f}%")

    return X_train, X_test, y_train, y_test

# Perform the split
X_train, X_test, y_train, y_test = test_data_split(
    df_train, LABEL, test_size=0.05, random_state=0
)

print("âœ… Step 4 Complete: Data split into train and test sets.")


# ===== STEP 5: DATA PREPARATION FOR Preprocessing =====
print("\nStep 5: Data Preparation for Preprocessing")

# Reset indices to ensure clean data structure
print(f"ğŸ”§ Resetting indices and preparing data...")
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print(f"  âœ… Indices reset for all splits")

# Validate feature lists are current
current_numerical = [col for col in X_train.columns if col in NUM_FEATS]
current_categorical = [col for col in X_train.columns if col in CAT_FEATS]

print(f"\nğŸ“Š Feature Validation:")
print(f"  â€¢ Expected numerical features: {len(NUM_FEATS)}")
print(f"  â€¢ Found numerical features: {len(current_numerical)}")
print(f"  â€¢ Expected categorical features: {len(CAT_FEATS)}")
print(f"  â€¢ Found categorical features: {len(current_categorical)}")

# Update feature lists if necessary
NUM_FEATS = current_numerical
CAT_FEATS = current_categorical

print(f"  âœ… Feature lists updated and validated")

print("âœ… Step 5 Complete: Data prepared for Preprocessing")


# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print("âœ… Label encoding completed")
print("Original classes:", sorted(df_train[LABEL].unique()))
print(f"Encoded classes: {sorted(set(y_train))}")
print(f"Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")


# Check baseline skewness of numerical features
skewnessDF = pd.DataFrame(X_train[NUM_FEATS].skew(), columns=["Skewness"]).sort_values(by='Skewness', ascending=False)
print("ğŸ“Š Baseline Skewness Analysis:")
print(f"Features with high skewness (>0.5): {len(skewnessDF[abs(skewnessDF['Skewness']) > 0.5])}")
print(f"Most skewed features:")
display(skewnessDF.head(10).style.background_gradient(cmap='Reds'))


# Check baseline kurtosis
kurtDF = pd.DataFrame(X_train[NUM_FEATS].kurtosis(), columns=["Kurtosis"]).sort_values(by='Kurtosis', ascending=False)
print("ğŸ“Š Baseline Kurtosis Analysis:")
print(f"Features with high kurtosis (>1.0): {len(kurtDF[abs(kurtDF['Kurtosis']) > 1.0])}")
print(f"Most high kurtosis features:")
display(kurtDF.head(10).style.background_gradient(cmap='Oranges'))


# Apply QuantileTransformer to numerical features (converts to normal distribution)
from sklearn.preprocessing import QuantileTransformer

print("ğŸ”§ Applying QuantileTransformer to numerical features...")
transformer = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)

# Transform numerical features
X_train_transformed = X_train.copy()
X_test_transformed = X_test.copy()

X_train_transformed[NUM_FEATS] = transformer.fit_transform(X_train[NUM_FEATS])
X_test_transformed[NUM_FEATS] = transformer.transform(X_test[NUM_FEATS])

print("âœ… QuantileTransformer applied to numerical features")


# Check skewness after transformation
skewnessDF_after = pd.DataFrame(X_train_transformed[NUM_FEATS].skew(), columns=["Skewness_After"]).sort_values(by='Skewness_After', ascending=False)

# Compare before and after
comparison = pd.DataFrame({
    'Before': skewnessDF['Skewness'],
    'After': skewnessDF_after['Skewness_After']
})
comparison['Improvement'] = abs(comparison['Before']) - abs(comparison['After'])

print("ğŸ”§ Transformation Results:")
print(f"Mean absolute skewness before: {abs(skewnessDF['Skewness']).mean():.3f}")
print(f"Mean absolute skewness after: {abs(skewnessDF_after['Skewness_After']).mean():.3f}")
print(f"Features with |skew| < 0.5: {len(skewnessDF_after[abs(skewnessDF_after['Skewness_After']) < 0.5])}/{len(NUM_FEATS)}")

display(comparison.head(10).style.background_gradient(subset=['Improvement'], cmap='Greens'))


# Check kurtosis after transformation
kurtDF_after = pd.DataFrame(X_train_transformed[NUM_FEATS].kurtosis(), columns=["Kurtosis_After"]).sort_values(by='Kurtosis_After', ascending=False)

# Compare before and after
comparison = pd.DataFrame({
    'Before': kurtDF['Kurtosis'],
    'After': kurtDF_after['Kurtosis_After']
})
comparison['Improvement'] = abs(comparison['Before']) - abs(comparison['After'])

print("ğŸ”§ Transformation Results:")
print(f"Mean absolute kurtosis before: {abs(kurtDF['Kurtosis']).mean():.3f}")
print(f"Mean absolute kurtosis after: {abs(kurtDF_after['Kurtosis_After']).mean():.3f}")
print(f"Features with |kurt| < 0.5: {len(kurtDF_after[abs(kurtDF_after['Kurtosis_After']) < 0.5])}/{len(NUM_FEATS)}")

display(comparison.head(10).style.background_gradient(subset=['Improvement'], cmap='Greens'))


# Apply RobustScaler for final scaling (handles outliers better)
from sklearn.preprocessing import RobustScaler

print("âš–ï¸� Applying RobustScaler to numerical features...")
scaler = RobustScaler()

# Create final datasets
X_train_final = X_train_transformed.copy()
X_test_final = X_test_transformed.copy()

# Scale only numerical features
X_train_final[NUM_FEATS] = scaler.fit_transform(X_train_transformed[NUM_FEATS])
X_test_final[NUM_FEATS] = scaler.transform(X_test_transformed[NUM_FEATS])

print("âœ… RobustScaler applied to numerical features")
print(f"Final training shape: {X_train_final.shape}")
print(f"Final test shape: {X_test_final.shape}")


# Final quality validation
print("ğŸ�¯ Final Data Quality Check:")

# Check final skewness
final_skewness = X_train_final[NUM_FEATS].skew()
excellent_features = len(final_skewness[abs(final_skewness) < 0.5])
good_features = len(final_skewness[(abs(final_skewness) >= 0.5) & (abs(final_skewness) < 1.0)])

print(f"âœ… No missing values: {X_train_final.isnull().sum().sum() == 0}")
print(f"âœ… No infinite values: {not np.isinf(X_train_final.select_dtypes(include=[np.number])).any().any()}")
print(f"âœ… Excellent features (|skew| < 0.5): {excellent_features}/{len(NUM_FEATS)}")
print(f"âœ… Good features (|skew| < 1.0): {good_features}/{len(NUM_FEATS)}")


print("\nğŸš€ Data is ready for neural network training!")


# Optional: Create sklearn pipeline for future use
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer


# Create preprocessing pipeline
transformer = ColumnTransformer([
    ("quantile_transformer", QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42), NUM_FEATS),
], remainder='passthrough', verbose_feature_names_out=False)

scaler = ColumnTransformer([
    ("robust_scaler", RobustScaler(), NUM_FEATS),
], remainder='passthrough', verbose_feature_names_out=False)

preprocessing_pipeline = Pipeline([
    ('transformer', transformer.set_output(transform="pandas")),
    ('scaler', scaler)
])

print("ğŸ“¦ Preprocessing pipeline created for future use:")

preprocessing_pipeline


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.init as init


# ========================
# Dataset Class
# ========================

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Data Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=False)

X_train_scaled = scaler.fit_transform(X_train[NUM_FEATS])
X_test_scaled = scaler.transform(X_test[NUM_FEATS])

print("âœ… Data scaling completed")


# memory usage
initial_train_memory = X_train.memory_usage(deep=True).sum() / 1024**2
initial_test_memory = X_test.memory_usage(deep=True).sum() / 1024**2

print(f"Memory usage of X_train: {initial_train_memory:.2f} MB")
print(f"Memory usage of X_test: {initial_test_memory:.2f} MB")


def optimizeScaledFeatures(arr, array_name="Array"):
    """
    Optimize memory usage for scaled numpy arrays.
    Converts float64 to float32 for scaled data while preserving precision.
    """
    initial_memory = arr.nbytes / 1024**2
    print(f"ğŸ”§ Optimizing {array_name}...")
    print(f"Initial: {arr.shape} {arr.dtype}, {initial_memory:.2f} MB")

    # Check if data appears to be scaled (most scaled data will be within reasonable bounds)
    data_min = arr.min()
    data_max = arr.max()
    data_range = data_max - data_min

    # For scaled data, convert float64 -> float32 (huge memory savings)
    if arr.dtype == np.float64:
        # Check if values are within float32 range and likely scaled
        if np.abs(data_min) < 1e6 and np.abs(data_max) < 1e6:
            arr_optimized = arr.astype(np.float32)
            final_memory = arr_optimized.nbytes / 1024**2
            memory_reduction = (initial_memory - final_memory) / initial_memory * 100

            print(f"Final: {arr_optimized.shape} {arr_optimized.dtype}, {final_memory:.2f} MB")
            print(f"Memory reduction: {memory_reduction:.1f}%")
            print(f"Data range: [{data_min:.3f}, {data_max:.3f}]")
            print("âœ… Optimization completed")

            return arr_optimized
        else:
            print(f"âš ï¸� Data range too large for safe conversion: [{data_min:.3f}, {data_max:.3f}]")
            return arr

    # Already optimized or not suitable for optimization
    elif arr.dtype == np.float32:
        print("âœ… Already optimized (float32)")
        return arr
    else:
        print(f"â„¹ï¸� Data type {arr.dtype} - no optimization needed")
        return arr

def optimizeCatFeatures(df, dataframe_name="DataFrame"):
    """
    Optimize memory usage for categorical features.
    Converts object dtype to category for categorical features.
    """
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"ğŸ”§ Optimizing categorical features in {dataframe_name}...")
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    df_optimized = df.copy()
    memory_savings = 0

    for col in CAT_FEATS:
        if col in df_optimized.columns:
            initial_col_memory = df_optimized[col].memory_usage(deep=True) / 1024**2
            df_optimized[col] = df_optimized[col].astype('category')
            final_col_memory = df_optimized[col].memory_usage(deep=True) / 1024**2
            col_savings = initial_col_memory - final_col_memory
            memory_savings += col_savings

    final_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
    total_reduction = (initial_memory - final_memory) / initial_memory * 100

    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Total memory reduction: {total_reduction:.1f}%")
    print(f"Categorical features optimized: {len(CAT_FEATS)}")
    print("âœ… Categorical optimization completed")

    return df_optimized


X_train[NUM_FEATS] = optimizeScaledFeatures(X_train_scaled, "X_train_scaled")
print()
X_test[NUM_FEATS] = optimizeScaledFeatures(X_test_scaled, "X_test_scaled")


X_train[CAT_FEATS] = optimizeCatFeatures(X_train[CAT_FEATS], "X_train")
print()
X_test[CAT_FEATS] = optimizeCatFeatures(X_test[CAT_FEATS], "X_test")


# final memory usage
final_train_memory = X_train.memory_usage(deep=True).sum() / 1024**2
final_test_memory = X_test.memory_usage(deep=True).sum() / 1024**2

print(f"\nFinal memory usage of X_train: {final_train_memory:.2f} MB (reduced by {(initial_train_memory - final_train_memory) / initial_train_memory * 100:.1f}%)")
print(f"Final memory usage of X_test: {final_test_memory:.2f} MB (reduced by {(initial_test_memory - final_test_memory) / initial_test_memory * 100:.1f}%)")


# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print("âœ… Label encoding completed")
print("Original classes:", sorted(df_train[LABEL].unique()))
print(f"Encoded classes: {sorted(set(y_train))}")
print(f"Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")


from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(zip(np.unique(y_train), class_weights))
class_weight_dict


from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, f1_score, balanced_accuracy_score, log_loss
from sklearn.metrics import ConfusionMatrixDisplay

def make_classification_plots(model_preds, model_preds_proba, title):
  plt.style.use('default')
  fig, axes = plt.subplots(1, 2, figsize=(15, 5))

  cm_matrix = confusion_matrix(y_test, model_preds)

  # 1 - Summary Statistics
  accuracy = accuracy_score(y_test, model_preds) # % positive out of all predicted positives
  balanced_accuracy = balanced_accuracy_score(y_test, model_preds)
  logLoss = log_loss(y_test, model_preds_proba)
  precision = precision_score(y_test, model_preds, average='weighted') # % positive out of all predicted positives
  recall =  recall_score(y_test, model_preds, average='weighted') # % positive out of all supposed to be positives
  f1 = f1_score(y_test, model_preds, average='weighted')
  stats_summary = '[Summary Statistics]\nAccuracy = {:.2%} | Balanced Accuracy = {:.2%} | Log Loss = {:.3} | Precision = {:.2%} | Recall = {:.2%} | F1-Score = {:.2%}'.format(accuracy, balanced_accuracy, logLoss, precision, recall, f1)
  print(stats_summary)

  # 2 : Confusion Matrix
  ConfusionMatrixDisplay.from_predictions(y_test, model_preds, ax=axes[0])
  axes[0].set_title("Confusion Matrix")

  # 3 : Normalized Confusion Matrix
  ConfusionMatrixDisplay.from_predictions(y_test, model_preds, ax=axes[1], normalize="true", values_format=".0%")
  axes[1].set_title("Normalized Confusion Matrix")

  fig.suptitle(f"{title} Test Metrics", fontsize=20, fontweight='bold')
  plt.tight_layout()
  plt.show()

  # 4 : Classification Report
  class_report = classification_report(y_test, model_preds)
  print('\n', f"{title} Classification Report: \n {class_report}")

  plt.style.use('ggplot')


def plot_loss_and_accuracy(history):
    """
    history: dict with keys 'train_loss','val_loss','train_acc','val_acc'
    """
    tr_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])
    tr_acc = np.array(history["train_acc"])
    val_acc = np.array(history["val_acc"])
    epochs = np.arange(1, len(tr_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14,4))
    axes[0].plot(epochs, tr_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.4)

    axes[1].plot(epochs, tr_acc, label="Train Acc")
    axes[1].plot(epochs, val_acc, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()


class ForestNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ForestNet, self).__init__()

        self.net = nn.Sequential(
            self._block(input_dim, 128),
            self._block(128, 64),
            self._block(64, 32),
            nn.Linear(32, num_classes)
        )

    def _block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========================
# Training Loop
# ========================

BATCH_SIZE = 512
LR = 0.001
MAX_EPOCHS = 5
PATIENCE = 5

def train_torch_model(X_train, y_train, X_test, y_test, num_classes,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=4, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Styled Logging for Every Epoch ---
        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


from torchsummary import summary

model = ForestNet(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6,
                          batch_size=512, lr=0.001, max_epochs=50, patience=5)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


plot_loss_and_accuracy(torch_history)


class ForestNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ForestNet, self).__init__()

        self.net = nn.Sequential(
            self._block(input_dim, 128),
            self._block(128, 64),
            self._block(64, 32),
            nn.Linear(32, num_classes)
        )

    def _block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print(f"Class weights: {class_weight_dict}")


# ========================
# Training Loop
# ========================

BATCH_SIZE = 1024
LR = 0.001
MAX_EPOCHS = 50
PATIENCE = 5

def train_torch_model(X_train, y_train, X_test, y_test, num_classes,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=4, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classWeights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=classWeights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Styled Logging for Every Epoch ---
        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


from torchsummary import summary

model = ForestNet(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6,
                          batch_size=1024, lr=0.001, max_epochs=50, patience=5)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


plot_loss_and_accuracy(torch_history)


class ForestNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ForestNet, self).__init__()

        self.net = nn.Sequential(
            self._block(input_dim, 512),
            self._block(512, 256),
            self._block(256, 128),
            self._block(128, 64),
            self._block(64, 32),
            nn.Linear(32, num_classes)
        )

    def _block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========================
# Training Loop
# ========================

BATCH_SIZE = 2048
LR = 0.001
MAX_EPOCHS = 100
PATIENCE = 10

def train_torch_model(X_train, y_train, X_test, y_test, num_classes,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=4, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classWeights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=5)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Scheduler & Logging ---
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


from torchsummary import summary

model = ForestNet(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6,
                          batch_size=2048, lr=0.001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


plot_loss_and_accuracy(torch_history)


class ForestNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ForestNet, self).__init__()

        self.hidden_layers = nn.Sequential(
            self._block(input_dim, 512),
            self._block(512, 256),
            self._block(256, 128),
            self._block(128, 64),
            self._block(64, 32),
        )

        self.fc_out = nn.Linear(32, num_classes)

        # Weight init
        self._initialize_weights()

    def _block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.fc_out:
                    # Apply Xavier initialization ONLY to the final layer
                    init.xavier_uniform_(m.weight)
                else:
                    # Apply he Normal to all other (hidden) layers
                    init.kaiming_normal_(m.weight, nonlinearity="relu")

                # Initialize biases to zero for all linear layers
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    init.ones_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.fc_out(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Exclude non-trainable parameters from weight decay (like biases and batch norm parameters)

def make_optimizer(model, base_lr=1e-3, weight_decay=1e-5):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    optim = optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                   {"params": no_decay, "weight_decay": 0.0}],
                  lr=base_lr)
    return optim



# ========================
# Training Loop
# ========================

BATCH_SIZE = 2048
LR = 0.0001
MAX_EPOCHS = 100
PATIENCE = 10

def train_torch_model(X_train, y_train, X_test, y_test, num_classes, weight_decay=1e-5,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=2, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classWeights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, base_lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=5)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Scheduler & Logging ---
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


from torchsummary import summary

model = ForestNet(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6, weight_decay=1e-5,
                          batch_size=2048, lr=0.0001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


plot_loss_and_accuracy(torch_history)


from sklearn.preprocessing import QuantileTransformer

transformer = QuantileTransformer(output_distribution='normal', copy=False)

X_train[NUM_FEATS] = transformer.fit_transform(X_train[NUM_FEATS])
X_test[NUM_FEATS] = transformer.transform(X_test[NUM_FEATS])

print("âœ… Data Transformation completed")


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler(copy=False)

X_train_scaled = scaler.fit_transform(X_train[NUM_FEATS])
X_test_scaled = scaler.transform(X_test[NUM_FEATS])


print("âœ… Data scaling completed")


X_train[NUM_FEATS] = optimizeScaledFeatures(X_train_scaled, "X_train_scaled")
print()
X_test[NUM_FEATS] = optimizeScaledFeatures(X_test_scaled, "X_test_scaled")

X_train[CAT_FEATS] = optimizeCatFeatures(X_train[CAT_FEATS], "X_train")
print()
X_test[CAT_FEATS] = optimizeCatFeatures(X_test[CAT_FEATS], "X_test")


class ForestNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ForestNet, self).__init__()

        self.net = nn.Sequential(
            self._block(input_dim, 512),
            self._block(512, 256),
            self._block(256, 128),
            self._block(128, 64),
            self._block(64, 32),
            nn.Linear(32, num_classes)
        )

    def _block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========================
# Training Loop
# ========================

BATCH_SIZE = 2048
LR = 0.001
MAX_EPOCHS = 100
PATIENCE = 10

def train_torch_model(X_train, y_train, X_test, y_test, num_classes,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=4, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classWeights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=5)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Scheduler & Logging ---
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


from torchsummary import summary

model = ForestNet(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6, 
                          batch_size=2048, lr=0.001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


plot_loss_and_accuracy(torch_history)


def featureEngineering(df):

  # Manhhattan distance to Hydrology
  df["mnhttn_dist_hydrlgy"] = np.abs(df["Horizontal_Distance_To_Hydrology"]) + np.abs(df["Vertical_Distance_To_Hydrology"])
  # Euclidean distance to Hydrology
  df["ecldn_dist_hydrlgy"] = (np.abs(df["Horizontal_Distance_To_Hydrology"])**2 + np.abs(df["Vertical_Distance_To_Hydrology"])**2)**0.5

  # Combining Soil Type
  soil_features = [x for x in df.columns if x.startswith("Soil_Type")]
  df["soil_type_count"] = df[soil_features].astype(int).sum(axis=1)
  # Combining Wilderness Areas
  wilderness_features = [x for x in df.columns if x.startswith("Wilderness_Area")]
  df["wilderness_area_count"] = df[wilderness_features].astype(int).sum(axis=1)

  # Fixing Aspect Values
  df["Aspect"][df["Aspect"] < 0] += 360
  df["Aspect"][df["Aspect"] > 359] -= 360

  # Fixing Hillshade Values
  df.loc[df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
  df.loc[df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
  df.loc[df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
  df.loc[df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
  df.loc[df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
  df.loc[df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255

  return df

NUM_FEATS.append("mnhttn_dist_hydrlgy")
NUM_FEATS.append("ecldn_dist_hydrlgy")
NUM_FEATS.append("soil_type_count")
NUM_FEATS.append("wilderness_area_count")


X_train = featureEngineering(X_train)
X_test = featureEngineering(X_test)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=False)

X_train_scaled = scaler.fit_transform(X_train[NUM_FEATS])
X_test_scaled = scaler.transform(X_test[NUM_FEATS])

print("âœ… Data scaling completed")


X_train[NUM_FEATS] = optimizeScaledFeatures(X_train_scaled, "X_train_scaled")
print()
X_test[NUM_FEATS] = optimizeScaledFeatures(X_test_scaled, "X_test_scaled")

X_train[CAT_FEATS] = optimizeCatFeatures(X_train[CAT_FEATS], "X_train")
print()
X_test[CAT_FEATS] = optimizeCatFeatures(X_test[CAT_FEATS], "X_test")


class ForestNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ForestNet, self).__init__()

        self.net = nn.Sequential(
            self._block(input_dim, 512),
            self._block(512, 256),
            self._block(256, 128),
            self._block(128, 64),
            self._block(64, 32),
            nn.Linear(32, num_classes)
        )

    def _block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========================
# Training Loop
# ========================

BATCH_SIZE = 2048
LR = 0.001
MAX_EPOCHS = 100
PATIENCE = 10

def train_torch_model(X_train, y_train, X_test, y_test, num_classes,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=4, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classWeights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=5)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Scheduler & Logging ---
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


from torchsummary import summary

model = ForestNet(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6,
                          batch_size=2048, lr=0.001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


plot_loss_and_accuracy(torch_history)


X_train = featureEngineering(X_train)
X_test = featureEngineering(X_test)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=False)

X_train_scaled = scaler.fit_transform(X_train[NUM_FEATS])
X_test_scaled = scaler.transform(X_test[NUM_FEATS])

print("âœ… Data scaling completed")


X_train[NUM_FEATS] = optimizeScaledFeatures(X_train_scaled, "X_train_scaled")
print()
X_test[NUM_FEATS] = optimizeScaledFeatures(X_test_scaled, "X_test_scaled")

X_train[CAT_FEATS] = optimizeCatFeatures(X_train[CAT_FEATS], "X_train")
print()
X_test[CAT_FEATS] = optimizeCatFeatures(X_test[CAT_FEATS], "X_test")


class ForestNetDeeper(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ForestNetDeeper, self).__init__()

        self.net = nn.Sequential(
            self._block(input_dim, 512, 0.1),
            self._block(512, 512, 0.1),
            self._block(512, 256, 0.1),
            self._block(256, 256, 0.1),
            self._block(256, 128, 0.1),
            self._block(128, 128, 0.1),
            self._block(128, 64, 0.1),
            self._block(64, 32, 0.1),
            self._block(32, 16, 0.1),
            nn.Linear(16, num_classes)
        )

    def _block(self, in_dim, out_dim, p=0.1):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.net(x)


class ForestNetWider(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ForestNetWider, self).__init__()

        self.net = nn.Sequential(
            self._block(input_dim, 1024, 0.1),
            self._block(1024, 512, 0.1),
            self._block(512, 256, 0.1),
            self._block(256, 128, 0.1),
            self._block(128, 64, 0.1),
            self._block(64, 32, 0.1),
            self._block(32, 16, 0.1),
            nn.Linear(16, num_classes)
        )

    def _block(self, in_dim, out_dim, p=0.1):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========================
# Training Loop
# ========================

BATCH_SIZE = 2048
LR = 0.001
MAX_EPOCHS = 100
PATIENCE = 10

def train_torch_model(X_train, y_train, X_test, y_test, num_classes,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=4, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=5)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Scheduler & Logging ---
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


from torchsummary import summary

model = ForestNetDeeper(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6,
                          batch_size=2048, lr=0.001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


from torchsummary import summary

model = ForestNetWider(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6, 
                          batch_size=2048, lr=0.001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


plot_loss_and_accuracy(torch_history)


class ForestNet(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()

        self.hidden_layers = nn.Sequential(
            self._block(input_dim, 1024, 0.1),
            self._block(1024, 512, 0.1),
            self._block(512, 256, 0.1),
            self._block(256, 128, 0.1),
            self._block(128, 64, 0.1),
            self._block(64, 32, 0.1),
            self._block(32, 16, 0.1),
        )

        self.fc_out = nn.Linear(16, num_classes)

        self._initialize_weights()

    def _block(self, in_dim, out_dim, p):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SELU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(p=p),
            # nn.AlphaDropout(p=p)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.fc_out:
                    # Apply Xavier initialization ONLY to the final layer
                    init.xavier_uniform_(m.weight)
                else:
                    # Apply LeCun Normal to all other (hidden) layers
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')

                # Initialize biases to zero for all linear layers
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.fc_out(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========================
# Training Loop
# ========================

BATCH_SIZE = 2048
LR = 0.001
MAX_EPOCHS = 100
PATIENCE = 10

def train_torch_model(X_train, y_train, X_test, y_test, num_classes,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=4, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=5)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Scheduler & Logging ---
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


from torchsummary import summary

model = ForestNet(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6,
                          batch_size=2048, lr=0.001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


plot_loss_and_accuracy(torch_history)


class ForestNet(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()

        self.hidden_layers = nn.Sequential(
            self._block(input_dim, 1024, 0.1),
            self._block(1024, 512, 0.1),
            self._block(512, 256, 0.1),
            self._block(256, 128, 0.1),
            self._block(128, 64, 0.1),
            self._block(64, 32, 0.1),
            self._block(32, 16, 0.1),
        )

        self.fc_out = nn.Linear(16, num_classes)

        self._initialize_weights()

    def _block(self, in_dim, out_dim, p):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(p=p),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.fc_out:
                    # Apply Xavier initialization ONLY to the final layer
                    init.xavier_uniform_(m.weight)
                else:
                    # Apply Kaiming Normal to all other (hidden) layers
                    init.kaiming_normal_(m.weight)

                # Initialize biases to zero for all linear layers
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.fc_out(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Exclude non-trainable parameters from weight decay (like biases and batch norm parameters)
from torch.optim import AdamW

def make_optimizer(model, base_lr=1e-3, weight_decay=1e-5):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    optim = AdamW([{"params": decay, "weight_decay": weight_decay},
                   {"params": no_decay, "weight_decay": 0.0}],
                  lr=base_lr)
    return optim



# ========================
# Training Loop
# ========================

from torch.optim.lr_scheduler import ExponentialLR

BATCH_SIZE = 2048
LR = 0.0001
MAX_EPOCHS = 100
PATIENCE = 10
gamma = 0.97  

def train_torch_model(X_train, y_train, X_test, y_test, num_classes, weight_decay=1e-5,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=4, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classWeights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, base_lr=lr, weight_decay=weight_decay)

    # ExponentialLR scheduler: lr <- lr * gamma each epoch
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Scheduler & Logging ---
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


from torchsummary import summary

model = ForestNet(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6, weight_decay=1e-5,
                          batch_size=2048, lr=0.001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


# ========================
# Training Loop
# ========================

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

BATCH_SIZE = 2048
LR = 0.0001
MAX_EPOCHS = 100
PATIENCE = 10

def train_torch_model(X_train, y_train, X_test, y_test, num_classes, weight_decay=1e-5,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=4, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classWeights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, base_lr=lr, weight_decay=1e-5)

    # T_0 = initial cycle length (in epochs), T_mult multiplies cycle length after each restart
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Scheduler & Logging ---
        # step scheduler with fractional epoch to make warm restarts smooth:
        scheduler.step(epoch + batch_idx / len(train_loader))
        
        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6, weight_decay=1e-5,
                          batch_size=2048, lr=0.001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


plot_loss_and_accuracy(torch_history)


class ForestNetSkip(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()

        # Main blocks
        self.block1 = self._block(input_dim, 512, dropout)
        self.block2 = self._block(512, 256, dropout)
        self.block3 = self._block(256, 128, dropout)
        self.block4 = self._block(128, 64, dropout)
        self.block5 = self._block(64, 32,  dropout)
        self.block6 = self._block(32, 16,  dropout)

        # Projection shortcuts: Linear(no_bias) -> BatchNorm1d
        # BN stabilizes/aligns projections with main path
        self.shortcut12 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                        nn.BatchNorm1d(256))
        self.shortcut23 = nn.Sequential(nn.Linear(256, 128, bias=False),
                                        nn.BatchNorm1d(128))
        self.shortcut34 = nn.Sequential(nn.Linear(128, 64, bias=False),
                                        nn.BatchNorm1d(64))
        self.shortcut45 = nn.Sequential(nn.Linear(64, 32,  bias=False),
                                        nn.BatchNorm1d(32))
        self.shortcut56 = nn.Sequential(nn.Linear(32, 16,  bias=False),
                                        nn.BatchNorm1d(16))

        # Final head
        self.fc_out = nn.Linear(16, num_classes)

        # Weight init
        self._initialize_weights()

    def _block(self, in_dim, out_dim, p):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p)
        )

    def forward(self, x):

        x1 = self.block1(x)
        x2 = self.block2(x1) + self.shortcut12(x1)
        x3 = self.block3(x2) + self.shortcut23(x2)
        x4 = self.block4(x3) + self.shortcut34(x3)
        x5 = self.block5(x4) + self.shortcut45(x4)
        x6 = self.block6(x5) + self.shortcut56(x5)
        out = self.fc_out(x6)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.fc_out:
                    init.xavier_uniform_(m.weight)
                else:
                    init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    init.ones_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


class ForestNetGatedSkip(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()

        # Main blocks
        self.block1 = self._block(input_dim, 512, dropout)
        self.block2 = self._block(512, 256, dropout)
        self.block3 = self._block(256, 128, dropout)
        self.block4 = self._block(128, 64, dropout)
        self.block5 = self._block(64, 32, dropout)
        self.block6 = self._block(32, 16, dropout)

        # Projection shortcuts: Linear(no_bias) -> BatchNorm1d
        # BN stabilizes/aligns projections with main path
        self.shortcut12 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                        nn.BatchNorm1d(256))
        self.shortcut23 = nn.Sequential(nn.Linear(256, 128, bias=False),
                                        nn.BatchNorm1d(128))
        self.shortcut34 = nn.Sequential(nn.Linear(128, 64, bias=False),
                                        nn.BatchNorm1d(64))
        self.shortcut45 = nn.Sequential(nn.Linear(64, 32, bias=False),
                                        nn.BatchNorm1d(32))
        self.shortcut56 = nn.Sequential(nn.Linear(32, 16, bias=False),
                                        nn.BatchNorm1d(16))

        # Learnable scalar gates for each shortcut (start at 0 -> shortcut initially off)
        # Scalar gating is cheap and effective
        self.gate12 = nn.Parameter(torch.zeros(1))
        self.gate23 = nn.Parameter(torch.zeros(1))
        self.gate34 = nn.Parameter(torch.zeros(1))
        self.gate45 = nn.Parameter(torch.zeros(1))
        self.gate56 = nn.Parameter(torch.zeros(1))

        # Final head
        self.fc_out = nn.Linear(16, num_classes)

        # Weight init
        self._initialize_weights()

    def _block(self, in_dim, out_dim, p):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p)
        )

    def forward(self, x):
        # forward with gated projected shortcuts
        x1 = self.block1(x)

        proj12 = self.shortcut12(x1)
        x2 = self.block2(x1) + (self.gate12 * proj12)

        proj23 = self.shortcut23(x2)
        x3 = self.block3(x2) + (self.gate23 * proj23)

        proj34 = self.shortcut34(x3)
        x4 = self.block4(x3) + (self.gate34 * proj34)

        proj45 = self.shortcut45(x4)
        x5 = self.block5(x4) + (self.gate45 * proj45)

        proj56 = self.shortcut56(x5)
        x6 = self.block6(x5) + (self.gate56 * proj56)

        out = self.fc_out(x6)
        return out

    def _initialize_weights(self):
        # He for hidden linear layers and projection linears; Xavier for final head
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.fc_out:
                    init.xavier_uniform_(m.weight)
                else:
                    init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    init.ones_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========================
# Training Loop
# ========================

BATCH_SIZE = 2048
LR = 0.001
MAX_EPOCHS = 100
PATIENCE = 10

def train_torch_model(X_train, y_train, X_test, y_test, num_classes,
                      batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      num_workers=4, pin_memory=True):

    # --- Boilerplate ---
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classWeights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = ForestNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=5)

    best_test_loss = float("inf")
    trigger_times = 0
    best_path = "best_model.pth"

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    print("ğŸš€ " + "="*80)
    print(f"ğŸ”¥ STARTING NEURAL NETWORK TRAINING FOR {max_epochs} EPOCHS")
    print("ğŸš€ " + "="*80)

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            train_correct += correct
            train_total += y_batch.size(0)

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.to(device, non_blocking=pin_memory), y_test_batch.to(device, non_blocking=pin_memory)

                outputs = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test_batch).sum().item()
                test_correct += correct
                test_total += y_test_batch.size(0)

        # --- Metric Calculation ---
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_accuracy = 100 * (train_correct / train_total)
        test_accuracy = 100 * (test_correct / test_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(test_loss)
        history["val_acc"].append(test_accuracy)

        # --- Scheduler & Logging ---
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create dynamic progress bar visualization based on number of epochs
        progress = (epoch + 1) / max_epochs
        # Adjust bar length based on number of epochs for better visualization
        if max_epochs <= 20:
            bar_length = max_epochs  # One character per epoch for small epoch counts
        elif max_epochs <= 50:
            bar_length = 40
        elif max_epochs <= 100:
            bar_length = 50
        else:
            bar_length = 60

        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        # Color coding for accuracy
        if test_accuracy >= 90:
            acc_emoji = "ğŸŸ¢"
        elif test_accuracy >= 80:
            acc_emoji = "ğŸŸ¡"
        elif test_accuracy >= 70:
            acc_emoji = "ğŸŸ "
        else:
            acc_emoji = "ğŸ”´"

        # Styled epoch output with dynamic progress bar
        print(f"ğŸ“Š Epoch [{epoch+1:3d}/{max_epochs}] [{bar}] {progress*100:6.1f}%")
        print(f"   ğŸ�‹ï¸�  Train â†’ Loss: {train_loss:7.4f} | Acc: {train_accuracy:6.2f}% | âš™ï¸� LR: {current_lr:.10f}")
        print(f"   ğŸ�¯  Valid â†’ Loss: {test_loss:7.4f} | Acc: {test_accuracy:6.2f}% {acc_emoji}")

        # Special formatting for milestone epochs
        if (epoch + 1) % 10 == 0:
            print("   " + "â”€" * 50)
        else:
            print()

        # --- Early stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_path)

        else:
            trigger_times += 1
            print(f"  -->â�³ No improvement. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("\n" + "ğŸ›‘ " + "="*60)
                print(f"ğŸ›‘ EARLY STOPPING at Epoch {epoch+1}")
                print(f"ğŸ�† Best Validation Loss: {best_test_loss:.4f}")
                print(f"ğŸ’¾ Best model restored and ready!")
                print("ğŸ›‘ " + "="*60)
                break

    print("\n" + "ğŸ�‰ " + "="*60)
    print("ğŸ�‰ TRAINING COMPLETED SUCCESSFULLY!")
    print("ğŸ�‰ " + "="*60)

    # --- Load Best Model ---
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)

    return model, history

def predict_torch_model(model, X_test, device=device, return_probs=True, batch_size=BATCH_SIZE):

    model.eval()
    all_preds = []

    # Convert input
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    elif hasattr(X_test, "values"):  # pandas dataframe
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

    X_test = X_test.to(device)

    with torch.no_grad():
        # Loop through X_test in batches
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]

            outputs = model(X_batch)

            if return_probs:
                preds = torch.softmax(outputs, dim=1)  # probabilities
                preds = preds.cpu().numpy()
                preds = preds / preds.sum(axis=1, keepdims=True)
            else:
                preds = torch.argmax(outputs, dim=1)   # class indices
                preds = preds.cpu().numpy()

            all_preds.extend(preds)


    return np.array(all_preds)


from torchsummary import summary

model = ForestNetSkip(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6, 
                          batch_size=2048, lr=0.001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


from torchsummary import summary

model = ForestNetGatedSkip(input_dim=X_train.shape[1], num_classes=np.unique(y_train).shape[0])
model.to(device)
summary(model, input_size=(X_train.shape[1],))


torch_model, torch_history = train_torch_model(X_train, y_train, X_test, y_test, num_classes=6, 
                          batch_size=2048, lr=0.001, max_epochs=100, patience=10)


torchPreds = predict_torch_model(torch_model, X_test, return_probs=False)
torchPredsProba = predict_torch_model(torch_model, X_test)

make_classification_plots(torchPreds, torchPredsProba, 'Torch NN')


plot_loss_and_accuracy(torch_history)


df_test = pd.read_csv('/kaggle/input/beyond-nti-r-1-c-2/test.csv').drop(columns=['Id'],axis=1)
df_test


df_test.info()


df_test = featureEngineering(df_test)


for col in ORG_FEATS:
  df_test[col] = df_test[col].astype('int16')

df_test['Slope'] = df_test['Slope'].astype('int8')

for col in CAT_FEATS:
  df_test[col] = df_test[col].astype('category')


df_test.drop(columns = ["Soil_Type7" , "Soil_Type15"] , inplace =True)


df_test.info()


df_test[NUM_FEATS] = scaler.transform(df_test[NUM_FEATS])


SubPreds = predict_torch_model(torch_model, df_test, return_probs=False)


sub_df = pd.read_csv("/kaggle/input/beyond-nti-r-1-c-2/sample_submission.csv")
sub_df


sub_df = pd.read_csv("/kaggle/input/beyond-nti-r-1-c-2/sample_submission.csv")
submission = sub_df.copy()
submission['Cover_Type'] = le.inverse_transform(SubPreds).astype('int8')
submission.to_csv("submission.csv",index=None)
submission.head()


submission.Cover_Type.value_counts()

