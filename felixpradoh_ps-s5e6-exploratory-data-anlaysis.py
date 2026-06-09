import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.ticker as mticker
from matplotlib.ticker import EngFormatter

import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

# --- GLOBAL PATCH FOR ALL HEATMAPS ---
# Function to format values in engineering notation
def eng_format(x, pos=None):
    if x >= 1e3:
        return f"{x/1e3:.1f}k"
    elif x >= 1e6:
        return f"{x/1e6:.1f}M"
    return f"{int(x)}" if x == int(x) else f"{x:.1f}"

# Patch heatmap function to apply engineering format automatically
_original_heatmap = sns.heatmap

def heatmap_eng(*args, annot=True, fmt='', cbar_kws=None, **kwargs):
    if cbar_kws is None:
        cbar_kws = {}
    cbar_kws = dict(cbar_kws)
    if 'format' not in cbar_kws:
        cbar_kws['format'] = EngFormatter()
    
    ax = kwargs.get('ax', None)
    hm = _original_heatmap(*args, annot=annot, fmt=fmt, cbar_kws=cbar_kws, **kwargs)
    
    # Get current axis if not specified
    if ax is None:
        ax = plt.gca()
    
    # Format cell annotations
    for t in ax.texts:
        try:
            val = float(t.get_text())
            t.set_text(eng_format(val))
        except:
            pass
    
    # Format colorbar
    if hasattr(ax, 'collections') and ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.formatter = EngFormatter()
            cbar.update_ticks()
    
    return hm

# Replace the original function.
sns.heatmap = heatmap_eng


input_dir = '/kaggle/input/playground-series-s5e6'
train_df = pd.read_csv(f'{input_dir}/train.csv')
test_df = pd.read_csv(f'{input_dir}/test.csv')
sample_submission = pd.read_csv(f'{input_dir}/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


print("=== DATASET INFO ===\n")
train_df.info()
print("\n=== FIRST 5 ROWS ===\n")
display(train_df.head())
print("\n=== DESCRIPTIVE STATISTICS ===\n")
display(train_df.describe().round(2))


plt.figure(figsize=(12, 6))
fertilizer_counts = train_df['Fertilizer Name'].value_counts()

# Define consistent colors for both plots
colors = sns.color_palette('Set2', n_colors=len(fertilizer_counts))

plt.subplot(1, 2, 1)
fertilizer_counts.plot(kind='bar', color=colors)
plt.title('Fertilizer Type Distribution')
plt.xlabel('Fertilizer Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.pie(fertilizer_counts.values, labels=fertilizer_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('Fertilizer Percentage Distribution')
plt.tight_layout()
plt.show()
print(f"Number of unique fertilizer types: {train_df['Fertilizer Name'].nunique()}")
print(f"\nTop 10 most common fertilizers:")
print(fertilizer_counts.head(10))


numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i, col in enumerate(numeric_cols):
    axes[i].hist(train_df[col], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()


categorical_cols = ['Soil Type', 'Crop Type']

# Define colors for each categorical variable
soil_colors = sns.color_palette('Set2', n_colors=train_df['Soil Type'].nunique())
crop_colors = sns.color_palette('Set3', n_colors=train_df['Crop Type'].nunique())
colors_dict = {'Soil Type': soil_colors, 'Crop Type': crop_colors}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for i, col in enumerate(categorical_cols):
    train_df[col].value_counts().plot(kind='bar', ax=axes[i], color=colors_dict[col])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

for col in categorical_cols:
    print(f"\n{col}:")
    print(train_df[col].value_counts())
    print(f"Unique values: {train_df[col].nunique()}")
    print(f"Data type: {train_df[col].dtype}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, col in enumerate(categorical_cols):
    train_df[col].value_counts().plot(kind='pie', ax=axes[i], autopct='%1.1f%%', colors=colors_dict[col])
    axes[i].set_title(f'Percentage Distribution of {col}')
    axes[i].set_ylabel('')
plt.tight_layout()
plt.show()

# Summary for Categorical Feature Analysis
summary_df = train_df[categorical_cols].copy()
summary_df['count'] = 1
summary_df = summary_df.groupby(categorical_cols).count().reset_index()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    summary_df.pivot_table(index='Soil Type', columns='Crop Type', values='count'),
    cmap='YlGnBu',
    annot=True,
    annot_kws={"fontsize":10},
    ax=ax,
    linewidths=0.5,
    linecolor='gray'
)

ax.set_title('Heatmap of Soil Type and Crop Type Counts')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


print("\n=== MISSING VALUES IN DATASET ===\n")
display(train_df.isnull().sum())
print("\n=== PERCENTAGE OF MISSING VALUES ===\n")
display(train_df.isnull().mean() * 100)
print("\n=== DESCRIPTION with CATEGORICAL VARIABLES ===\n ")
display(train_df.describe(include='all'))


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    if column != 'Temparature' and lower_bound < 0:
        lower_bound = 0
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound
numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
outlier_stats = []
for col in numeric_cols:
    outliers, lower, upper = detect_outliers_iqr(train_df, col)
    stats = {
        'Variable': col,
        'Mean': train_df[col].mean(),
        'Median': train_df[col].median(),
        'Std': train_df[col].std(),
        'Min': train_df[col].min(),
        'Max': train_df[col].max(),
        'Q1': train_df[col].quantile(0.25),
        'Q3': train_df[col].quantile(0.75),
        'IQR': train_df[col].quantile(0.75) - train_df[col].quantile(0.25),
        'Lower_Bound': lower,
        'Upper_Bound': upper,
        'N_Outliers': len(outliers),
        'Pct_Outliers': (len(outliers) / len(train_df)) * 100,
        'Range': train_df[col].max() - train_df[col].min(),
        'Coef_Variation': (train_df[col].std() / train_df[col].mean()) * 100
    }
    outlier_stats.append(stats)
outlier_df = pd.DataFrame(outlier_stats)
print("ğŸ“Š DESCRIPTIVE STATISTICS AND OUTLIER ANALYSIS:")
print("=" * 100)
display(outlier_df.round(2))


def create_engineered_features(df):
    """
    Create engineered features based on domain knowledge
    """
    df_eng = df.copy()
    
    # NPK Ratios (avoid division by zero)
    df_eng['N_P_ratio'] = df_eng['Nitrogen'] / (df_eng['Phosphorous'] + 0.001)
    df_eng['N_K_ratio'] = df_eng['Nitrogen'] / (df_eng['Potassium'] + 0.001)
    df_eng['P_K_ratio'] = df_eng['Phosphorous'] / (df_eng['Potassium'] + 0.001)
    
    # Total NPK
    df_eng['Total_NPK'] = df_eng['Nitrogen'] + df_eng['Phosphorous'] + df_eng['Potassium']
    
    # Environmental indices
    df_eng['Temp_Hum_index'] = df_eng['Temparature'] * df_eng['Humidity'] / 100
    df_eng['Moist_Balance'] = df_eng['Moisture'] - df_eng['Humidity']
    df_eng['Environ_Stress'] = np.sqrt((df_eng['Temparature'] - 25)**2 + (df_eng['Humidity'] - 65)**2)
    
    # Categorical binning
    df_eng['Temp_Cat'] = pd.cut(df_eng['Temparature'], bins=3, labels=['Low', 'Medium', 'High'])
    df_eng['Hum_Cat'] = pd.cut(df_eng['Humidity'], bins=3, labels=['Low', 'Medium', 'High'])
    df_eng['N_Level'] = pd.cut(df_eng['Nitrogen'], bins=3, labels=['Low', 'Medium', 'High'])
    df_eng['K_Level'] = pd.cut(df_eng['Potassium'], bins=3, labels=['Low', 'Medium', 'High'])
    df_eng['P_Level'] = pd.cut(df_eng['Phosphorous'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Combinations
    df_eng['Soil_Crop_Combo'] = df_eng['Soil Type'].astype(str) + '_' + df_eng['Crop Type'].astype(str)
    
    # NPK Balance and dominant nutrient
    npk_mean = df_eng[['Nitrogen', 'Phosphorous', 'Potassium']].mean(axis=1)
    df_eng['NPK_Balance'] = df_eng[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1) / npk_mean
    
    # Dominant NPK
    npk_cols = ['Nitrogen', 'Phosphorous', 'Potassium']
    df_eng['Dominant_NPK'] = df_eng[npk_cols].idxmax(axis=1)
    df_eng['Dominant_NPK_Level'] = df_eng['Dominant_NPK'].map({'Nitrogen': 'N', 'Phosphorous': 'P', 'Potassium': 'K'})
    
    # Temperature-Moisture interaction
    df_eng['Temp_Moist_inter'] = df_eng['Temparature'] * df_eng['Moisture'] / 100
    
    return df_eng

# Apply feature engineering to both train and test
print("ğŸ”§ Applying feature engineering...")
train_engineered = create_engineered_features(train_df)
test_engineered = create_engineered_features(test_df)

print(f"Original features: {train_df.shape[1]}")
print(f"After feature engineering: {train_engineered.shape[1]}")
print(f"New features added: {train_engineered.shape[1] - train_df.shape[1]}")


def apply_encoding(train_df, test_df, target_col='Fertilizer Name'):
    categorical_cols = [
        'Soil Type', 'Crop Type', 'Temp_Cat', 'Hum_Cat', 
        'N_Level', 'K_Level', 'P_Level', 'Soil_Crop_Combo'
    ]
    existing_categorical = [col for col in categorical_cols if col in train_df.columns and col in test_df.columns]
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    label_encoders = {}
    for col in existing_categorical:
        le = LabelEncoder()
        combined_values = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(combined_values)
        train_encoded[f'{col}_encoded'] = le.transform(train_df[col].astype(str))
        test_encoded[f'{col}_encoded'] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
    if target_col in train_encoded.columns:
        target_encoder = LabelEncoder()
        train_encoded[f'{target_col}_encoded'] = target_encoder.fit_transform(train_encoded[target_col])
        label_encoders[target_col] = target_encoder
    return train_encoded, test_encoded, label_encoders

# Now apply encoding to the engineered datasets
train_encoded, test_encoded, encoders_dict = apply_encoding(train_engineered, test_engineered)

print(f"âœ… Encoding completed")
print(f"Train encoded shape: {train_encoded.shape}")
print(f"Test encoded shape: {test_encoded.shape}")
print(f"Available encoders: {list(encoders_dict.keys())}")


from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Prepare data for MI (encode categorical variables temporarily)
df_mi = train_df.copy()
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_target = LabelEncoder()
df_mi['Soil_encoded'] = le_soil.fit_transform(df_mi['Soil Type'])
df_mi['Crop_encoded'] = le_crop.fit_transform(df_mi['Crop Type'])
df_mi['Fertilizer_encoded'] = le_target.fit_transform(df_mi['Fertilizer Name'])

features_original = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil_encoded', 'Crop_encoded']
X_mi = df_mi[features_original]
y_mi = df_mi['Fertilizer_encoded']
mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features=[6, 7], random_state=513)
mi_df = pd.DataFrame({'Feature': features_original, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values('MI_Score', ascending=False).head(15)

# Use Set2 palette and assign colors to features
palette = sns.color_palette('Set2', n_colors=len(mi_df))
feature_colors = dict(zip(mi_df['Feature'], palette))

fig, ax1 = plt.subplots(figsize=(9, 5))
bar = sns.barplot(data=mi_df, y='Feature', x='MI_Score', palette=palette, edgecolor='black', ax=ax1)

plt.title('Top 15 Mutual Information Scores (Before Encoding)')
ax1.set_xlabel('Mutual Information Score')
ax1.set_ylabel('Feature')
ax1.legend(loc='lower right')
plt.tight_layout()
plt.show()
mi_df


# Select all numeric and encoded columns except the target
target_col = 'Fertilizer Name_encoded'
feature_cols = [col for col in train_encoded.columns if col != target_col and col != 'Fertilizer Name']

# Remove any remaining string columns
numeric_cols = []
for col in feature_cols:
    if train_encoded[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        numeric_cols.append(col)
    else:
        print(f"âš ï¸� Skipping non-numeric column: {col} (dtype: {train_encoded[col].dtype})")

print(f"ğŸ“Š Numeric features for MI: {len(numeric_cols)}")

X_mi_enc = train_encoded[numeric_cols]
y_mi_enc = train_encoded[target_col]

# Identify discrete features (encoded categorical variables)
discrete_features = [i for i, col in enumerate(numeric_cols) if col.endswith('_encoded')]
print(f"ğŸ“Š Discrete features indices: {discrete_features}")

# Calculate mutual information
mi_scores_enc = mutual_info_classif(X_mi_enc, y_mi_enc, discrete_features=discrete_features, random_state=513)
mi_df_enc = pd.DataFrame({'Feature': numeric_cols, 'MI_Score': mi_scores_enc})
mi_df_enc = mi_df_enc.sort_values('MI_Score', ascending=False)

# Plot with consistent palette
palette_enc = sns.color_palette('Set2', n_colors=len(mi_df_enc))

# Single comprehensive plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=mi_df_enc, y='Feature', x='MI_Score', palette=palette_enc, edgecolor='black', ax=ax)
ax.set_title('Mutual Information Scores (After Encoding & Feature Engineering)', fontsize=14, pad=20)
ax.set_xlabel('Mutual Information Score', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# Display the results
print("ğŸ“Š Features by Mutual Information:")
display(mi_df_enc)


# Let's compare specific features to understand the MI changes
print("ğŸ”� COMPARING MUTUAL INFORMATION CHANGES")
print("=" * 60)

# Extract original feature MI scores for comparison
original_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
original_mi = mi_df[mi_df['Feature'].isin(original_features)].set_index('Feature')

# Extract the same features from the new MI analysis
after_mi = mi_df_enc[mi_df_enc['Feature'].isin(original_features)].set_index('Feature')

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Before_MI': original_mi['MI_Score'],
    'After_MI': after_mi['MI_Score']
}).fillna(0)

comparison_df['MI_Change'] = comparison_df['After_MI'] - comparison_df['Before_MI']
comparison_df['Pct_Change'] = (comparison_df['MI_Change'] / comparison_df['Before_MI'] * 100).round(2)

print("\nğŸ“Š Original Features MI Comparison:")
display(comparison_df.round(4))

# Show top engineered features
engineered_features = mi_df_enc[~mi_df_enc['Feature'].isin(original_features + ['Soil_encoded', 'Crop_encoded'])].head(10)
print("\nğŸ”§ Top 10 Engineered Features by MI:")
display(engineered_features)

# Calculate some statistics
print("\nğŸ“ˆ Key Insights:")
print(f"â€¢ Highest MI original feature: {original_mi['MI_Score'].idxmax()} ({original_mi['MI_Score'].max():.4f})")
print(f"â€¢ Highest MI engineered feature: {engineered_features.iloc[0]['Feature']} ({engineered_features.iloc[0]['MI_Score']:.4f})")
print(f"â€¢ Average MI original features: {original_mi['MI_Score'].mean():.4f}")
print(f"â€¢ Average MI engineered features: {engineered_features['MI_Score'].mean():.4f}")

# Show correlation between related features
print("\nğŸ”— Example: Why N_P_ratio might have higher MI than individual N or P:")
if 'N_P_ratio' in train_engineered.columns:
    correlation_with_target = train_engineered[['Nitrogen', 'Phosphorous', 'N_P_ratio']].corrwith(
        train_encoded['Fertilizer Name_encoded']
    )
    print("Correlation with target (Fertilizer):")
    for feature, corr in correlation_with_target.items():
        print(f"  â€¢ {feature}: {corr:.4f}")


# Visualize the comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Original vs After MI for same features
if not comparison_df.empty:
    x_pos = np.arange(len(comparison_df))
    width = 0.35
    
    ax1.bar(x_pos - width/2, comparison_df['Before_MI'], width, label='Before Feature Engineering', alpha=0.8, color='lightblue')
    ax1.bar(x_pos + width/2, comparison_df['After_MI'], width, label='After Feature Engineering', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Mutual Information Score')
    ax1.set_title('MI Comparison: Before vs After Feature Engineering\n(Same Features)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(comparison_df.index, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Plot 2: Top features overall (mix of original and engineered)
top_features = mi_df_enc.head(12)
colors = ['lightcoral' if any(eng in feat for eng in ['_ratio', '_index', '_Balance', '_inter', '_Level', 'Combo']) 
          else 'lightblue' for feat in top_features['Feature']]

ax2.barh(range(len(top_features)), top_features['MI_Score'], color=colors, alpha=0.8)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['Feature'])
ax2.set_xlabel('Mutual Information Score')
ax2.set_title('Top 12 Features by MI\n(Red=Engineered, Blue=Original)')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("\nğŸ�¯ Summary of MI Changes:")
print("1. ğŸ”§ Engineered features often have HIGHER MI than original features")
print("2. ğŸ“Š Ratios capture relationships better than absolute values")
print("3. ğŸ§® Feature interactions reveal hidden patterns")
print("4. ğŸ�² Different feature spaces lead to different MI rankings")
print("5. ğŸŒ± Domain knowledge (agronomy) guides effective feature engineering")


# 1. Soil-Crop Combination Analysis
print("ğŸŒ± SOIL-CROP COMBINATION ANALYSIS")
print("=" * 50)

# Create a pivot table for soil-crop-fertilizer relationships
soil_crop_fert = train_engineered.groupby(['Soil Type', 'Crop Type', 'Fertilizer Name']).size().reset_index(name='Count')
soil_crop_pivot = soil_crop_fert.pivot_table(index=['Soil Type', 'Crop Type'], columns='Fertilizer Name', values='Count', fill_value=0)

# Top 10 soil-crop combinations
top_combinations = train_engineered['Soil_Crop_Combo'].value_counts().head(10)
print(f"\nğŸ“ˆ Top 10 Soil-Crop Combinations:")
for combo, count in top_combinations.items():
    print(f"  â€¢ {combo}: {count} samples")

# Visualize soil-crop combinations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Soil-Crop combination frequency
top_combinations.plot(kind='bar', ax=ax1, color='lightblue', edgecolor='black')
ax1.set_title('Top 10 Soil-Crop Combinations')
ax1.set_xlabel('Soil-Crop Combination')
ax1.set_ylabel('Frequency')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Plot 2: Heatmap of Soil Type vs Crop Type frequency
soil_crop_counts = train_engineered.groupby(['Soil Type', 'Crop Type']).size().unstack(fill_value=0)
sns.heatmap(
    soil_crop_counts,
    annot=True,
    cmap='YlOrRd',
    ax=ax2,
    linewidths=0.5,
    linecolor='gray'
)
ax2.set_title('Soil Type vs Crop Type Heatmap')
ax2.set_xlabel('Crop Type')
ax2.set_ylabel('Soil Type')

# Plot 3: Fertilizer distribution for top soil-crop combo
top_combo = top_combinations.index[0]
top_combo_data = train_engineered[train_engineered['Soil_Crop_Combo'] == top_combo]
fert_dist = top_combo_data['Fertilizer Name'].value_counts()
fert_dist.plot(kind='pie', ax=ax3, autopct='%1.1f%%')
ax3.set_title(f'Fertilizer Distribution\nfor {top_combo}')
ax3.set_ylabel('')

# Plot 4: Average NPK for different soil types
npk_by_soil = train_engineered.groupby('Soil Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean()
npk_by_soil.plot(kind='bar', ax=ax4, width=0.8)
ax4.set_title('Average NPK Levels by Soil Type')
ax4.set_xlabel('Soil Type')
ax4.set_ylabel('NPK Levels')
ax4.legend(title='Nutrients')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical insight
print(f"\nğŸ”� Insights:")
print(f"â€¢ Total unique soil-crop combinations: {train_engineered['Soil_Crop_Combo'].nunique()}")
print(f"â€¢ Most common combination: {top_combinations.index[0]} ({top_combinations.iloc[0]} samples)")
print(f"â€¢ Average samples per combination: {len(train_engineered) / train_engineered['Soil_Crop_Combo'].nunique():.1f}")


# 2. NPK Ratios Analysis
print("\n\nğŸ§ª NPK RATIOS ANALYSIS")
print("=" * 50)

# Calculate ratio statistics by fertilizer
ratio_cols = ['N_P_ratio', 'N_K_ratio', 'P_K_ratio']
ratio_stats = train_engineered.groupby('Fertilizer Name')[ratio_cols].agg(['mean', 'std']).round(3)

print("\nğŸ“Š NPK Ratio Statistics by Fertilizer (Top 5):")
top_fertilizers = train_engineered['Fertilizer Name'].value_counts().head(5).index
display(ratio_stats.loc[top_fertilizers])

# Visualize NPK ratios
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: N_P_ratio distribution by top fertilizers
for i, fert in enumerate(top_fertilizers[:3]):
    data = train_engineered[train_engineered['Fertilizer Name'] == fert]['N_P_ratio']
    ax1.hist(data, bins=20, alpha=0.6, label=fert, edgecolor='black')
ax1.set_title('N/P Ratio Distribution (Top 3 Fertilizers)')
ax1.set_xlabel('N/P Ratio')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: N_K_ratio vs P_K_ratio scatter
scatter_colors = sns.color_palette('Set1', n_colors=len(top_fertilizers))
for i, fert in enumerate(top_fertilizers):
    data = train_engineered[train_engineered['Fertilizer Name'] == fert]
    ax2.scatter(data['N_K_ratio'], data['P_K_ratio'], alpha=0.6, 
               label=fert, color=scatter_colors[i], s=30)
ax2.set_title('N/K vs P/K Ratios by Fertilizer Type')
ax2.set_xlabel('N/K Ratio')
ax2.set_ylabel('P/K Ratio')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

# Plot 3: Average ratios by fertilizer (bar plot)
ratio_means = train_engineered.groupby('Fertilizer Name')[ratio_cols].mean()
top_ratio_means = ratio_means.loc[top_fertilizers]
top_ratio_means.plot(kind='bar', ax=ax3, width=0.8)
ax3.set_title('Average NPK Ratios by Fertilizer Type')
ax3.set_xlabel('Fertilizer Type')
ax3.set_ylabel('Ratio Value')
ax3.legend(title='Ratios')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# Plot 4: Total NPK distribution
ax4.hist(train_engineered['Total_NPK'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
ax4.set_title('Total NPK Distribution')
ax4.set_xlabel('Total NPK (N + P + K)')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find fertilizers with extreme ratios
print("\nğŸ�¯ Extreme Ratio Analysis:")
print(f"â€¢ Highest N/P ratio: {train_engineered['N_P_ratio'].max():.2f}")
print(f"â€¢ Lowest N/P ratio: {train_engineered['N_P_ratio'].min():.2f}")
print(f"â€¢ Most balanced NPK (lowest std): {train_engineered.groupby('Fertilizer Name')[['Nitrogen', 'Phosphorous', 'Potassium']].std().mean(axis=1).idxmin()}")


# 3. Environmental Stress and Categorical Features
print("\n\nğŸŒ¡ï¸� ENVIRONMENTAL ANALYSIS")
print("=" * 50)

# Environmental stress analysis
stress_quartiles = train_engineered['Environ_Stress'].quantile([0.25, 0.5, 0.75])
print(f"\nğŸ“ˆ Environmental Stress Quartiles:")
print(f"â€¢ Q1 (Low stress): {stress_quartiles[0.25]:.2f}")
print(f"â€¢ Q2 (Medium stress): {stress_quartiles[0.5]:.2f}")
print(f"â€¢ Q3 (High stress): {stress_quartiles[0.75]:.2f}")

# Create stress categories
train_engineered['Stress_Category'] = pd.cut(train_engineered['Environ_Stress'], 
                                           bins=3, labels=['Low', 'Medium', 'High'])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Environmental stress distribution
ax1.hist(train_engineered['Environ_Stress'], bins=30, alpha=0.7, color='orange', edgecolor='black')
ax1.axvline(stress_quartiles[0.25], color='red', linestyle='--', label='Q1')
ax1.axvline(stress_quartiles[0.5], color='red', linestyle='--', label='Q2')
ax1.axvline(stress_quartiles[0.75], color='red', linestyle='--', label='Q3')
ax1.set_title('Environmental Stress Distribution')
ax1.set_xlabel('Environmental Stress Index')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Stress category vs Fertilizer
stress_fert = pd.crosstab(train_engineered['Stress_Category'], 
                         train_engineered['Fertilizer Name'], normalize='index') * 100
# Show only top 5 fertilizers
top_ferts = train_engineered['Fertilizer Name'].value_counts().head(5).index
stress_fert[top_ferts].plot(kind='bar', stacked=True, ax=ax2, width=0.8)
ax2.set_title('Fertilizer Distribution by Stress Category (%)')
ax2.set_xlabel('Stress Category')
ax2.set_ylabel('Percentage')
ax2.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.tick_params(axis='x', rotation=0)

# Plot 3: Temperature-Humidity index vs Fertilizer
top_3_ferts = top_ferts[:3]
for i, fert in enumerate(top_3_ferts):
    data = train_engineered[train_engineered['Fertilizer Name'] == fert]['Temp_Hum_index']
    ax3.hist(data, bins=20, alpha=0.6, label=fert, edgecolor='black')
ax3.set_title('Temperature-Humidity Index (Top 3 Fertilizers)')
ax3.set_xlabel('Temp-Humidity Index')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Moisture Balance distribution
ax4.hist(train_engineered['Moist_Balance'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', label='Balance Point')
ax4.set_title('Moisture Balance Distribution')
ax4.set_xlabel('Moisture Balance (Soil Moisture - Humidity)')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Environmental insights
print(f"\nğŸŒ� Environmental Insights:")
print(f"â€¢ Average environmental stress: {train_engineered['Environ_Stress'].mean():.2f}")
print(f"â€¢ Samples with positive moisture balance: {(train_engineered['Moist_Balance'] > 0).sum()} ({(train_engineered['Moist_Balance'] > 0).mean()*100:.1f}%)")
print(f"â€¢ High stress samples: {(train_engineered['Stress_Category'] == 'High').sum()} ({(train_engineered['Stress_Category'] == 'High').mean()*100:.1f}%)")


# 4. Categorical Level Analysis
print("\n\nğŸ“Š CATEGORICAL LEVEL ANALYSIS")
print("=" * 50)

# Analyze the categorical binning we created
categorical_features = ['Temp_Cat', 'Hum_Cat', 'N_Level', 'K_Level', 'P_Level']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, cat_feature in enumerate(categorical_features):
    # Count distribution
    counts = train_engineered[cat_feature].value_counts()
    counts.plot(kind='bar', ax=axes[i], color=['lightcoral', 'lightblue', 'lightgreen'], 
               edgecolor='black', width=0.8)
    axes[i].set_title(f'{cat_feature} Distribution')
    axes[i].set_xlabel('Category')
    axes[i].set_ylabel('Frequency')
    axes[i].tick_params(axis='x', rotation=0)
    axes[i].grid(True, alpha=0.3)
    
    # Add percentage labels
    total = len(train_engineered)
    for j, (cat, count) in enumerate(counts.items()):
        pct = count / total * 100
        axes[i].text(j, count + total*0.01, f'{pct:.1f}%', ha='center', va='bottom')

# Plot 6: Dominant nutrient analysis
dominant_counts = train_engineered['Dominant_NPK_Level'].value_counts()
dominant_counts.plot(kind='pie', ax=axes[5], autopct='%1.1f%%', 
                    colors=['lightcoral', 'lightblue', 'lightgreen'])
axes[5].set_title('Dominant Nutrient Distribution')
axes[5].set_ylabel('')

plt.tight_layout()
plt.show()

# Cross-tabulation analysis
print("\nğŸ”� Cross-Feature Analysis:")
print("\nDominant Nutrient vs Top 3 Fertilizers:")
top_3_fertilizers = train_engineered['Fertilizer Name'].value_counts().head(3).index
crosstab = pd.crosstab(train_engineered[train_engineered['Fertilizer Name'].isin(top_3_fertilizers)]['Dominant_NPK_Level'],
                      train_engineered[train_engineered['Fertilizer Name'].isin(top_3_fertilizers)]['Fertilizer Name'],
                      normalize='columns') * 100
display(crosstab.round(1))

# NPK Balance insights
print(f"\nâš–ï¸� NPK Balance Analysis:")
print(f"â€¢ Average NPK balance: {train_engineered['NPK_Balance'].mean():.3f}")
print(f"â€¢ Most balanced samples (NPK_Balance < 0.1): {(train_engineered['NPK_Balance'] < 0.1).sum()} ({(train_engineered['NPK_Balance'] < 0.1).mean()*100:.1f}%)")
print(f"â€¢ Highly imbalanced samples (NPK_Balance > 1.0): {(train_engineered['NPK_Balance'] > 1.0).sum()} ({(train_engineered['NPK_Balance'] > 1.0).mean()*100:.1f}%)")

