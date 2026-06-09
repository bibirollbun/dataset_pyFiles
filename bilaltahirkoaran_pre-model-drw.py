import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
plt.style.use('default')
sns.set_palette("husl")


df=pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")


df.head(5)


print(f"Dataset Shape: {df.shape[0]:,} rows Ã— {df.shape[1]:,} columns")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(df.dtypes.value_counts())



df.columns


missing_summary = df.isnull().sum()
missing_percentage = (missing_summary / len(df)) * 100
print(missing_summary.sum())
print(missing_percentage.sum())


print(f"Time Range: {df.index.min()} to {df.index.max()}")
time_diff = df.index.max() - df.index.min()
print(f"Total Duration: {time_diff}")


df[['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']].describe()


df_visual=df.copy()



# 1. TARGET VARIABLE ANALYSIS

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distribution of target variable
axes[0,0].hist(df_visual['label'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Distribution of Target Variable (Label)')
axes[0,0].set_xlabel('Label Value')
axes[0,0].set_ylabel('Frequency')

# Box plot for target variable
axes[0,1].boxplot(df_visual['label'])
axes[0,1].set_title('Box Plot of Target Variable')
axes[0,1].set_ylabel('Label Value')

# Q-Q plot for normality check
stats.probplot(df_visual['label'], dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot: Label vs Normal Distribution')

# Time series plot of target variable (sample)
sample_data = df_visual.sample(n=min(10000, len(df_visual))).sort_index()
axes[1,1].plot(sample_data.index, sample_data['label'], alpha=0.7, linewidth=0.5, color='red')
axes[1,1].set_title('Time Series Plot of Target Variable (Sample)')
axes[1,1].set_xlabel('Timestamp')
axes[1,1].set_ylabel('Label Value')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Target variable statistics
print(f"\nTarget Variable Statistics:")
print(f"Mean: {df_visual['label'].mean():.6f}")
print(f"Std: {df_visual['label'].std():.6f}")
print(f"Skewness: {df_visual['label'].skew():.6f}")
print(f"Kurtosis: {df_visual['label'].kurtosis():.6f}")
print(f"Range: [{df_visual['label'].min():.3f}, {df_visual['label'].max():.3f}]")



# 2. MARKET FEATURES CORRELATION ANALYSIS

# Correlation matrix for market features
market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
correlation_matrix = df_visual[market_features + ['label']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix: Market Features vs Target', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# Multicollinearity detection
print("\nMulticollinearity Analysis:")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:  # High correlation threshold
            high_corr_pairs.append((correlation_matrix.columns[i], 
                                  correlation_matrix.columns[j], corr_val))

for pair in high_corr_pairs:
    print(f"High correlation: {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")



# 3. VOLUME AND QUANTITY DISTRIBUTION ANALYSIS

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Distribution plots for each market feature (log-transformed)
for i, feature in enumerate(market_features):
    row = i // 3
    col = i % 3
    
    # Log scale for better visualization
    data_log = np.log1p(df_visual[feature])
    
    axes[row, col].hist(data_log, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
    axes[row, col].set_title(f'Log Distribution of {feature}')
    axes[row, col].set_xlabel(f'Log({feature})')
    axes[row, col].set_ylabel('Frequency')
    
    # Add statistics text
    axes[row, col].text(0.02, 0.98, f'Skew: {data_log.skew():.2f}', 
                       transform=axes[row, col].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Remove empty subplot
if len(market_features) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# Statistical summary
print("\nSkewness Analysis (Original Scale):")
for feature in market_features:
    skew_val = df_visual[feature].skew()
    print(f"{feature}: {skew_val:.3f} {'(Highly Skewed)' if abs(skew_val) > 2 else '(Moderately Skewed)' if abs(skew_val) > 1 else '(Normal)'}")



# 4. MARKET MICROSTRUCTURE ANALYSIS

# Create market microstructure features
df_visual['bid_ask_ratio'] = df_visual['bid_qty'] / (df_visual['ask_qty'] + 1e-8)
df_visual['volume_imbalance'] = (df_visual['buy_qty'] - df_visual['sell_qty']) / (df_visual['buy_qty'] + df_visual['sell_qty'] + 1e-8)
df_visual['total_order_book'] = df_visual['bid_qty'] + df_visual['ask_qty']
df_visual['liquidity_ratio'] = df_visual['volume'] / (df_visual['total_order_book'] + 1e-8)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Bid-Ask Ratio Distribution
axes[0,0].hist(np.log1p(df_visual['bid_ask_ratio']), bins=50, alpha=0.7, color='lightcoral')
axes[0,0].set_title('Distribution of Bid-Ask Ratio (Log Scale)')
axes[0,0].set_xlabel('Log(Bid Qty / Ask Qty)')
axes[0,0].set_ylabel('Frequency')

# Volume Imbalance Distribution
axes[0,1].hist(df_visual['volume_imbalance'], bins=50, alpha=0.7, color='lightgreen')
axes[0,1].set_title('Distribution of Volume Imbalance')
axes[0,1].set_xlabel('(Buy Qty - Sell Qty) / (Buy Qty + Sell Qty)')
axes[0,1].set_ylabel('Frequency')

# Liquidity Ratio vs Label
sample_indices = np.random.choice(len(df_visual), size=min(5000, len(df_visual)), replace=False)
axes[1,0].scatter(df_visual['liquidity_ratio'].iloc[sample_indices], 
                 df_visual['label'].iloc[sample_indices], alpha=0.5, s=1, color='purple')
axes[1,0].set_title('Liquidity Ratio vs Target Label')
axes[1,0].set_xlabel('Liquidity Ratio')
axes[1,0].set_ylabel('Label')

# Volume vs Label (enhanced)
axes[1,1].scatter(df_visual['volume'].iloc[sample_indices], 
                 df_visual['label'].iloc[sample_indices], alpha=0.5, s=1, color='orange')
axes[1,1].set_title('Volume vs Target Label (Sample)')
axes[1,1].set_xlabel('Volume')
axes[1,1].set_ylabel('Label')

plt.tight_layout()
plt.show()

# Market microstructure correlations
microstructure_features = ['bid_ask_ratio', 'volume_imbalance', 'liquidity_ratio']
micro_corr = df_visual[microstructure_features + ['label']].corr()['label']
print("\nMicrostructure Features Correlation with Target:")
for feature in microstructure_features:
    print(f"{feature}: {micro_corr[feature]:.6f}")



# 5. TEMPORAL PATTERNS ANALYSIS

# Extract time components
df_visual['hour'] = df_visual.index.hour
df_visual['minute'] = df_visual.index.minute
df_visual['day_of_week'] = df_visual.index.dayofweek
df_visual['is_weekend'] = (df_visual.index.dayofweek >= 5).astype(int)

# Advanced hourly analysis
hourly_stats = df_visual.groupby('hour').agg({
    'label': ['mean', 'std', 'count'],
    'volume': ['mean', 'std'],
    'volume_imbalance': 'mean'
}).round(6)

# Weekly analysis
weekly_stats = df_visual.groupby('day_of_week').agg({
    'label': ['mean', 'std'],
    'volume': 'mean'
}).round(6)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Hourly label mean
axes[0,0].plot(hourly_stats.index, hourly_stats[('label', 'mean')], 
               marker='o', linewidth=2, markersize=6, color='red')
axes[0,0].set_title('Average Target Label by Hour')
axes[0,0].set_xlabel('Hour of Day')
axes[0,0].set_ylabel('Average Label')
axes[0,0].grid(True, alpha=0.3)

# Hourly volume mean
axes[0,1].plot(hourly_stats.index, hourly_stats[('volume', 'mean')], 
               marker='o', linewidth=2, markersize=6, color='blue')
axes[0,1].set_title('Average Volume by Hour')
axes[0,1].set_xlabel('Hour of Day')
axes[0,1].set_ylabel('Average Volume')
axes[0,1].grid(True, alpha=0.3)

# Hourly volume imbalance
axes[0,2].plot(hourly_stats.index, hourly_stats[('volume_imbalance', 'mean')], 
               marker='o', linewidth=2, markersize=6, color='green')
axes[0,2].set_title('Average Volume Imbalance by Hour')
axes[0,2].set_xlabel('Hour of Day')
axes[0,2].set_ylabel('Volume Imbalance')
axes[0,2].grid(True, alpha=0.3)

# Day of week patterns
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1,0].bar(range(7), weekly_stats[('label', 'mean')].values, color='lightblue', alpha=0.7)
axes[1,0].set_title('Average Target Label by Day of Week')
axes[1,0].set_xlabel('Day of Week')
axes[1,0].set_ylabel('Average Label')
axes[1,0].set_xticks(range(7))
axes[1,0].set_xticklabels(day_names)

# Weekend vs Weekday analysis
weekend_analysis = df_visual.groupby('is_weekend').agg({
    'label': ['mean', 'std'],
    'volume': 'mean'
}).round(6)

axes[1,1].bar(['Weekday', 'Weekend'], weekend_analysis[('label', 'mean')].values, 
              color=['lightcoral', 'lightgreen'], alpha=0.7)
axes[1,1].set_title('Weekday vs Weekend Label Distribution')
axes[1,1].set_ylabel('Average Label')

# Hourly volatility
axes[1,2].plot(hourly_stats.index, hourly_stats[('label', 'std')], 
               marker='s', linewidth=2, markersize=6, color='purple')
axes[1,2].set_title('Label Volatility by Hour')
axes[1,2].set_xlabel('Hour of Day')
axes[1,2].set_ylabel('Label Standard Deviation')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical significance test for temporal patterns
from scipy.stats import f_oneway
hourly_groups = [df_visual[df_visual['hour'] == h]['label'].values for h in range(24)]
f_stat, p_value = f_oneway(*hourly_groups)
print(f"\nHourly Pattern ANOVA Test:")
print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.2e}")
print(f"Hourly patterns are {'statistically significant' if p_value < 0.05 else 'not significant'}")


# 6. ADVANCED CORRELATION ANALYSIS

# Identify X features with meaningful correlation
exclude_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 
                'bid_ask_ratio', 'volume_imbalance', 'total_order_book', 'liquidity_ratio',
                'hour', 'minute', 'day_of_week', 'is_weekend', 'label']
x_features_subset = [col for col in df_visual.columns if col not in exclude_cols]

# Calculate correlations with target
correlations = df_visual[x_features_subset + ['label']].corr()['label']
x_correlations = correlations[:-1]  # Exclude label self-correlation

# Find significant correlations
abs_corr = x_correlations.abs()
significant_features = abs_corr[abs_corr > 0.05].sort_values(ascending=False)

print(f"Features with |correlation| > 0.05:")
print(significant_features)

# Top correlated X features analysis
if len(significant_features) >= 6:
    top_x_features = significant_features.head(6).index.tolist()
else:
    top_x_features = significant_features.index.tolist()

if len(top_x_features) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, feature in enumerate(top_x_features):
        if i >= 6:  # Limit to 6 features
            break
            
        row = i // 3
        col = i % 3
        
        # Distribution plot
        axes[row, col].hist(df_visual[feature], bins=50, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'Distribution of {feature}\n(Corr: {x_correlations[feature]:.4f})')
        axes[row, col].set_xlabel(f'{feature}')
        axes[row, col].set_ylabel('Frequency')
    
    # Remove empty subplots if necessary
    total_plots = len(top_x_features)
    if total_plots < 6:
        for i in range(total_plots, 6):
            row = i // 3
            col = i % 3
            fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.show()
    
    # Correlation matrix for top X features
    if len(top_x_features) > 1:
        plt.figure(figsize=(10, 8))
        top_x_corr = df_visual[top_x_features + ['label']].corr()
        sns.heatmap(top_x_corr, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5)
        plt.title('Correlation Matrix: Top X Features vs Target')
        plt.tight_layout()
        plt.show()

else:
    print("No X features found with correlation > 0.05")



# Step 4: Data Cleaning & Feature Engineering
print("DATA CLEANING & FEATURE ENGINEERING")

df_processed = df.copy()



# 4.1 MEMORY OPTIMIZATION

def reduce_mem_usage(dataframe, dataset_name):
    """Optimize memory usage by downcasting numeric types"""
    print(f'Reducing memory usage for: {dataset_name}')
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype
        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)
    
    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print(f'--- Memory usage before: {initial_mem_usage:.2f} MB')
    print(f'--- Memory usage after: {final_mem_usage:.2f} MB')
    print(f'--- Decreased by: {100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}%\n')
    return dataframe

# Apply memory optimization
df_processed = reduce_mem_usage(df_processed, "main_dataset")



# 4.2 DATA QUALITY ASSESSMENT

# Check for infinite values
print("Checking for infinite values...")
inf_summary = np.isinf(df_processed.select_dtypes(include=[np.number])).sum()
inf_cols = inf_summary[inf_summary > 0]

if len(inf_cols) > 0:
    print(f"Found infinite values in {len(inf_cols)} columns:")
    for col, count in inf_cols.items():
        print(f"  {col}: {count} infinite values")
    
    # Drop columns with infinite values
    df_processed = df_processed.drop(columns=inf_cols.index.tolist())
    print(f"Dropped {len(inf_cols)} columns with infinite values")
else:
    print("âœ“ No infinite values found")

# Check for duplicate timestamps
duplicate_indices = df_processed.index.duplicated().sum()
if duplicate_indices > 0:
    print(f"Found {duplicate_indices} duplicate timestamps - removing...")
    df_processed = df_processed[~df_processed.index.duplicated(keep='first')]
else:
    print("âœ“ No duplicate timestamps")

# Detect low-variance features (< 10 unique values)
print("\nDetecting low-variance features...")
low_variance_cols = []
for col in df_processed.columns:
    if df_processed[col].nunique() < 10:
        low_variance_cols.append(col)

if len(low_variance_cols) > 0:
    df_processed = df_processed.drop(columns=low_variance_cols)
    print(f"Removed {len(low_variance_cols)} low-variance features")
else:
    print("âœ“ No low-variance features found")

print(f"Dataset shape after cleaning: {df_processed.shape}")



# 4.3 MULTICOLLINEARITY RESOLUTION

# Create derived features before dropping
market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

# Ensure we have these columns
available_market_features = [col for col in market_features if col in df_processed.columns]
print(f"Available market features: {available_market_features}")

if 'buy_qty' in df_processed.columns and 'sell_qty' in df_processed.columns:
    # Buy-Sell dynamics
    df_processed['buy_sell_ratio'] = df_processed['buy_qty'] / (df_processed['sell_qty'] + 1e-8)
    df_processed['volume_imbalance'] = (df_processed['buy_qty'] - df_processed['sell_qty']) / (df_processed['buy_qty'] + df_processed['sell_qty'] + 1e-8)
    df_processed['net_flow'] = df_processed['buy_qty'] - df_processed['sell_qty']
    
    # Now drop the highly correlated originals
    df_processed = df_processed.drop(columns=['buy_qty', 'sell_qty'])
    print("âœ“ Created buy_sell_ratio, volume_imbalance, net_flow")
    print("âœ“ Dropped buy_qty, sell_qty (high correlation with volume)")

if 'bid_qty' in df_processed.columns and 'ask_qty' in df_processed.columns:
    # Order book features
    df_processed['bid_ask_ratio'] = df_processed['bid_qty'] / (df_processed['ask_qty'] + 1e-8)
    df_processed['bid_ask_spread'] = df_processed['bid_qty'] - df_processed['ask_qty']
    df_processed['total_order_book'] = df_processed['bid_qty'] + df_processed['ask_qty']
    print("âœ“ Created order book features: bid_ask_ratio, bid_ask_spread, total_order_book")

# Liquidity measures
if 'volume' in df_processed.columns and 'total_order_book' in df_processed.columns:
    df_processed['liquidity_ratio'] = df_processed['volume'] / (df_processed['total_order_book'] + 1e-8)
    print("âœ“ Created liquidity_ratio")



# 4.4 OUTLIER DETECTION AND TREATMENT

def detect_outliers_iqr(data, column, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return lower_bound, upper_bound

# Define features to check for outliers
outlier_features = ['volume'] + [col for col in df_processed.columns 
                                if col.startswith('bid_') or col.startswith('ask_') or 
                                   col.endswith('_ratio') or col.endswith('_imbalance')]

print("Applying outlier treatment to key features...")
outlier_summary = {}

for feature in outlier_features:
    if feature in df_processed.columns:
        before_count = len(df_processed)
        lower, upper = detect_outliers_iqr(df_processed, feature, multiplier=2.0)  # Less aggressive
        
        # Cap outliers instead of removing (preserve data)
        df_processed[feature] = df_processed[feature].clip(lower=lower, upper=upper)
        
        outlier_summary[feature] = {
            'lower_bound': lower,
            'upper_bound': upper,
            'treatment': 'capped'
        }

print(f"âœ“ Applied outlier capping to {len(outlier_summary)} features")



# 4.5 TEMPORAL FEATURE ENGINEERING (Based on EDA insights)

# Basic time components
df_processed['hour'] = df_processed.index.hour
df_processed['minute'] = df_processed.index.minute  
df_processed['day_of_week'] = df_processed.index.dayofweek
df_processed['day_of_month'] = df_processed.index.day
df_processed['month'] = df_processed.index.month

# EDA-based temporal patterns
# Peak hour effect (21:00 showed highest label values)
df_processed['is_peak_hour'] = (df_processed['hour'] == 21).astype(int)

# High volume hours (14:00-16:00 from EDA)
df_processed['is_high_volume_period'] = ((df_processed['hour'] >= 14) & 
                                        (df_processed['hour'] <= 16)).astype(int)

# High volatility period (13:00-15:00 from EDA)
df_processed['is_high_volatility_period'] = ((df_processed['hour'] >= 13) & 
                                            (df_processed['hour'] <= 15)).astype(int)

# Weekend effect
df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)

# Cyclical encoding for time features (preserves circular nature)
df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
df_processed['minute_sin'] = np.sin(2 * np.pi * df_processed['minute'] / 60)
df_processed['minute_cos'] = np.cos(2 * np.pi * df_processed['minute'] / 60)
df_processed['day_of_week_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
df_processed['day_of_week_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)

print("âœ“ Created temporal features:")
print("  - Basic: hour, minute, day_of_week, day_of_month, month")
print("  - EDA-based: is_peak_hour, is_high_volume_period, is_high_volatility_period")
print("  - Cyclical: hour_sin/cos, minute_sin/cos, day_of_week_sin/cos")
print("  - Weekend: is_weekend")



# 4.6 ROLLING STATISTICS AND LAG FEATURES

# Define rolling windows (in minutes)
windows = [5, 15, 30, 60]  
key_features = ['volume', 'bid_ask_ratio', 'volume_imbalance', 'liquidity_ratio']

# Only create rolling features for available columns
available_key_features = [col for col in key_features if col in df_processed.columns]

for window in windows:
    print(f"  Creating {window}-minute rolling features...")
    
    for feature in available_key_features:
        # Rolling mean
        df_processed[f'{feature}_ma_{window}'] = df_processed[feature].rolling(window=window, min_periods=1).mean()
        
        # Rolling standard deviation (volatility)
        df_processed[f'{feature}_std_{window}'] = df_processed[feature].rolling(window=window, min_periods=1).std()
        
        # Rolling min/max for range analysis
        df_processed[f'{feature}_min_{window}'] = df_processed[feature].rolling(window=window, min_periods=1).min()
        df_processed[f'{feature}_max_{window}'] = df_processed[feature].rolling(window=window, min_periods=1).max()

# Create lag features (previous values)
print("Creating lag features...")
lag_periods = [1, 3, 5, 10, 30]  # minutes

for feature in available_key_features:
    for lag in lag_periods:
        df_processed[f'{feature}_lag_{lag}'] = df_processed[feature].shift(lag)

print(f"âœ“ Created rolling features for windows: {windows}")
print(f"âœ“ Created lag features for periods: {lag_periods}")



# 4.7 X FEATURES ENGINEERING (Based on EDA correlation analysis)

# Find all X features
all_x_features = [col for col in df_processed.columns if col.startswith('X') and col != 'X']

# Significant X features from EDA (update this list based on your actual EDA results)
significant_x_features = ['X21', 'X20', 'X28', 'X19', 'X29', 'X863']

# Keep only significant X features that exist in the dataset
available_significant_x = [col for col in significant_x_features if col in df_processed.columns]

# Drop non-significant X features
x_features_to_drop = [col for col in all_x_features if col not in available_significant_x]

if len(x_features_to_drop) > 0:
    df_processed = df_processed.drop(columns=x_features_to_drop)
    print(f"âœ“ Dropped {len(x_features_to_drop)} non-significant X features")
    print(f"âœ“ Kept {len(available_significant_x)} significant X features: {available_significant_x}")
else:
    print("âœ“ All X features are significant or already filtered")

# Create X feature combinations (based on EDA showing high inter-correlation)
if len(available_significant_x) >= 2:
    print("Creating X feature interactions...")
    
    # Create PCA-like combinations for highly correlated X features
    # Top 3 X features from EDA
    top_x_features = available_significant_x[:3]
    
    if len(top_x_features) >= 2:
        df_processed['x_feature_sum'] = df_processed[top_x_features].sum(axis=1)
        df_processed['x_feature_mean'] = df_processed[top_x_features].mean(axis=1)
        df_processed['x_feature_std'] = df_processed[top_x_features].std(axis=1)
        
        print(f"âœ“ Created X feature combinations from: {top_x_features}")



# 4.8 ADVANCED FEATURE ENGINEERING

# Volume-based features
if 'volume' in df_processed.columns:
    # Volume momentum
    df_processed['volume_momentum_5'] = df_processed['volume'] - df_processed['volume_ma_5']
    df_processed['volume_momentum_30'] = df_processed['volume'] - df_processed['volume_ma_30']
    
    # Volume volatility ratio
    df_processed['volume_volatility_ratio'] = df_processed['volume_std_15'] / (df_processed['volume_ma_15'] + 1e-8)

# Market microstructure interactions
if 'volume_imbalance' in df_processed.columns and 'volume' in df_processed.columns:
    df_processed['volume_x_imbalance'] = df_processed['volume'] * df_processed['volume_imbalance']

if 'bid_ask_ratio' in df_processed.columns and 'volume' in df_processed.columns:
    df_processed['bid_ask_x_volume'] = df_processed['bid_ask_ratio'] * df_processed['volume']

# Temporal interactions
if 'hour' in df_processed.columns and 'volume' in df_processed.columns:
    df_processed['hour_x_volume'] = df_processed['hour'] * df_processed['volume']

# Market regime features
if 'volume_std_30' in df_processed.columns and 'volume_ma_30' in df_processed.columns:
    # Market regime based on volume volatility
    df_processed['market_regime'] = pd.qcut(df_processed['volume_std_30'] / (df_processed['volume_ma_30'] + 1e-8), 
                                           q=3, labels=[0, 1, 2], duplicates='drop').astype(float)

print("âœ“ Created advanced features:")
print("  - Volume momentum and volatility features")
print("  - Microstructure interaction features")
print("  - Temporal interaction features")
print("  - Market regime classification")



# 4.9 FEATURE SCALING AND NORMALIZATION

# Identify features that need log transformation (highly skewed from EDA)
skewed_features = []
skewness_threshold = 2.0

numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col != 'label']  # Exclude target

print("Analyzing skewness for feature transformation...")

for feature in numeric_cols:
    if df_processed[feature].min() >= 0:  # Only for non-negative features
        skewness = df_processed[feature].skew()
        if abs(skewness) > skewness_threshold:
            skewed_features.append(feature)
            # Create log-transformed version
            df_processed[f'{feature}_log'] = np.log1p(df_processed[feature])

if len(skewed_features) > 0:
    print(f"âœ“ Created log-transformed versions for {len(skewed_features)} highly skewed features")
else:
    print("âœ“ No highly skewed features requiring log transformation")



# 4.10 FINAL DATA PREPARATION

# Handle missing values created by rolling/lag features
print("Handling missing values...")
missing_before = df_processed.isnull().sum().sum()

if missing_before > 0:
    print(f"Found {missing_before} missing values to handle...")
    
    # Get all numeric columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'label']  # Exclude target
    
    # Handle missing values column by column
    for col in numeric_cols:
        missing_count = df_processed[col].isnull().sum()
        if missing_count > 0:
            try:
                # For lag features (1-5 periods), try forward fill first
                if any(lag_pattern in col for lag_pattern in ['lag_1', 'lag_3', 'lag_5']):
                    # Simple forward fill using shift
                    mask = df_processed[col].isnull()
                    df_processed.loc[mask, col] = df_processed[col].shift(1)[mask]
                
                # Fill any remaining missing with median
                if df_processed[col].isnull().sum() > 0:
                    median_val = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(median_val)
                    
            except Exception as e:
                # If there's still an issue, just use median fill
                print(f"  Warning: Using median fill for {col} due to: {str(e)[:50]}...")
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
    
    missing_after = df_processed.isnull().sum().sum()
    print(f"âœ“ Reduced missing values from {missing_before} to {missing_after}")
else:
    print("âœ“ No missing values to handle")

# Feature categorization for analysis
original_market_features = [col for col in ['bid_qty', 'ask_qty', 'volume'] if col in df_processed.columns]
derived_market_features = [col for col in df_processed.columns if any(x in col for x in ['ratio', 'imbalance', 'spread', 'liquidity'])]
temporal_features = [col for col in df_processed.columns if any(x in col for x in ['hour', 'minute', 'day', 'weekend', 'peak', 'volatility_period'])]
rolling_features = [col for col in df_processed.columns if '_ma_' in col or '_std_' in col or '_min_' in col or '_max_' in col]
lag_features = [col for col in df_processed.columns if '_lag_' in col]
x_features = [col for col in df_processed.columns if col.startswith('X')]
interaction_features = [col for col in df_processed.columns if '_x_' in col or col in ['x_feature_sum', 'x_feature_mean', 'x_feature_std']]

print(f"\nğŸ“Š FEATURE ENGINEERING SUMMARY:")
print(f"{'='*50}")
print(f"Original Market Features: {len(original_market_features)}")
print(f"Derived Market Features: {len(derived_market_features)}")
print(f"Temporal Features: {len(temporal_features)}")
print(f"Rolling Statistics: {len(rolling_features)}")
print(f"Lag Features: {len(lag_features)}")
print(f"X Features (selected): {len(x_features)}")
print(f"Interaction Features: {len(interaction_features)}")
print(f"{'='*50}")
print(f"Total Features: {df_processed.shape[1]-1} (excluding target)")
print(f"Dataset Shape: {df_processed.shape}")

# Memory optimization for final dataset
df_processed = reduce_mem_usage(df_processed, "final_processed_dataset")

print(f"\nâœ… FEATURE ENGINEERING COMPLETED!")
print(f"Ready for model training with {df_processed.shape[1]-1} features")



# 4.11 FEATURE IMPORTANCE PREVIEW (Quick Assessment)

# Quick Random Forest to assess feature importance
print("Running quick feature importance assessment...")

# Prepare data for quick assessment
feature_cols = [col for col in df_processed.columns if col != 'label']
X_preview = df_processed[feature_cols].copy()
y_preview = df_processed['label'].copy()

# Handle any remaining missing values
X_preview = X_preview.fillna(X_preview.median())

# Sample for quick assessment (use 10k rows for speed)
sample_size = min(10000, len(X_preview))
sample_idx = np.random.choice(len(X_preview), sample_size, replace=False)

X_sample = X_preview.iloc[sample_idx]
y_sample = y_preview.iloc[sample_idx]

# Quick Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_quick = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_quick.fit(X_sample, y_sample)

# Get top 15 features
feature_importance = pd.DataFrame({
    'feature': X_sample.columns,
    'importance': rf_quick.feature_importances_
}).sort_values('importance', ascending=False)

print("\nğŸ�† TOP 15 MOST IMPORTANT FEATURES:")
print("-" * 45)
for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
    print(f"{i:2d}. {row['feature']:<25} {row['importance']:.6f}")

# Analyze feature types in top 15
top_15_features = feature_importance.head(15)['feature'].tolist()
feature_type_analysis = {
    'Temporal': len([f for f in top_15_features if any(x in f for x in ['hour', 'minute', 'day', 'weekend', 'peak'])]),
    'Rolling': len([f for f in top_15_features if '_ma_' in f or '_std_' in f]),
    'Lag': len([f for f in top_15_features if '_lag_' in f]),
    'X Features': len([f for f in top_15_features if f.startswith('X')]),
    'Market': len([f for f in top_15_features if any(x in f for x in ['volume', 'ratio', 'imbalance'])]),
    'Interaction': len([f for f in top_15_features if '_x_' in f or 'momentum' in f])
}

print(f"\nğŸ“Š TOP 15 FEATURES BY TYPE:")
print("-" * 30)
for feat_type, count in feature_type_analysis.items():
    if count > 0:
        print(f"{feat_type}: {count} features")

print(f"\nğŸ�¯ KEY INSIGHTS:")
print(f"â€¢ Temporal features are {'highly' if feature_type_analysis['Temporal'] >= 3 else 'moderately' if feature_type_analysis['Temporal'] >= 1 else 'not'} important")
print(f"â€¢ Rolling statistics show {'strong' if feature_type_analysis['Rolling'] >= 3 else 'moderate' if feature_type_analysis['Rolling'] >= 1 else 'weak'} predictive power")
print(f"â€¢ X features contribute {'significantly' if feature_type_analysis['X Features'] >= 2 else 'moderately' if feature_type_analysis['X Features'] >= 1 else 'minimally'}")

print(f"\nâœ… Feature engineering complete! Ready for model development.")
print("="*70)


