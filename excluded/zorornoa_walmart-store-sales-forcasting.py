import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)


train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip', parse_dates=['Date'])
test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip', parse_dates=['Date'])
stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip', parse_dates=['Date'])


print(f"Train: {train.shape}")
print(f"Test: {test.shape}")
print(f"Stores: {stores.shape}")
print(f"Features: {features.shape}")


train.info()
print(f"\nDate range: {train['Date'].min()} to {train['Date'].max()}")
print(f"Total weeks: {train['Date'].nunique()}")
print(f"Unique stores: {train['Store'].nunique()}")
print(f"Unique departments: {train['Dept'].nunique()}")


print(f"Date range: {test['Date'].min()} to {test['Date'].max()}")
print(f"Holiday weeks in test: {test['IsHoliday'].sum()}")


print(stores['Type'].value_counts())
print(f"\nStore size range: {stores['Size'].min():,} to {stores['Size'].max():,} sq ft")


print(f"Markdown columns: {[col for col in features.columns if 'MarkDown' in col]}")
print(f"Missing values in features:")
print(features.isnull().sum())


# Merge train data with stores and features
train_full = pd.merge(train, stores, on='Store', how='left')
train_full = pd.merge(train_full, features, on=['Store', 'Date'], how='left')

# Merge test data with stores and features
test_full = pd.merge(test, stores, on='Store', how='left')
test_full = pd.merge(test_full, features, on=['Store', 'Date'], how='left')

print(f"Train shape after merging: {train_full.shape}")
print(f"Test shape after merging: {test_full.shape}")


# Handle missing values
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

# Fill markdown missing values with 0
train_full[markdown_cols] = train_full[markdown_cols].fillna(0)
test_full[markdown_cols] = test_full[markdown_cols].fillna(0)

# Fill other missing values
for df in [train_full, test_full]:
    df['CPI'] = df['CPI'].fillna(df['CPI'].mean())
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].mean())
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())
    df['Fuel_Price'] = df['Fuel_Price'].fillna(df['Fuel_Price'].mean())


def create_features(df):
    """Create comprehensive features for modeling"""
    df = df.copy()
    
    # Date-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    
    # Holiday features
    df['IsSuperBowl'] = ((df['Month'] == 2) & (df['Week'].isin([6, 7, 8]))).astype(int)
    df['IsLaborDay'] = ((df['Month'] == 9) & (df['Week'].isin([36, 37]))).astype(int)
    df['IsThanksgiving'] = ((df['Month'] == 11) & (df['Week'].isin([47, 48]))).astype(int)
    df['IsChristmas'] = ((df['Month'] == 12) & (df['Week'].isin([51, 52]))).astype(int)
    
    # Days to next holiday
    def days_to_next_holiday(date):
        holidays = [
            pd.Timestamp(f'{date.year}-02-12'),  # Super Bowl
            pd.Timestamp(f'{date.year}-09-10'),  # Labor Day
            pd.Timestamp(f'{date.year}-11-26'),  # Thanksgiving
            pd.Timestamp(f'{date.year}-12-31')   # Christmas
        ]
        days_diff = [(holiday - date).days for holiday in holidays]
        positive_diffs = [d for d in days_diff if d >= 0]
        return min(positive_diffs) if positive_diffs else min([abs(d) for d in days_diff])
    
    df['Days_To_Holiday'] = df['Date'].apply(days_to_next_holiday)
    df['Holiday_Proximity'] = 1 / (df['Days_To_Holiday'] + 1)
    
    # Markdown features
    df['Total_Markdown'] = df[markdown_cols].sum(axis=1)
    df['Markdown_Count'] = (df[markdown_cols] > 0).sum(axis=1)
    df['Avg_Markdown'] = df[markdown_cols].mean(axis=1)
    
    # Store features
    df['Store_Type_Encoded'] = LabelEncoder().fit_transform(df['Type'])
    
    # Economic interaction features
    df['CPI_Unemployment_Ratio'] = df['CPI'] / (df['Unemployment'] + 1)
    df['Fuel_Temperature_Index'] = df['Fuel_Price'] * df['Temperature'] / 100
    
    # Seasonality features
    df['Is_Summer'] = ((df['Month'] >= 6) & (df['Month'] <= 8)).astype(int)
    df['Is_Winter'] = ((df['Month'] == 12) | (df['Month'] <= 2)).astype(int)
    df['Is_Spring'] = ((df['Month'] >= 3) & (df['Month'] <= 5)).astype(int)
    df['Is_Fall'] = ((df['Month'] >= 9) & (df['Month'] <= 11)).astype(int)
    
    # Week of month
    df['WeekOfMonth'] = df['Date'].apply(lambda d: (d.day-1)//7 + 1)
    
    return df


train_full = create_features(train_full)
test_full = create_features(test_full)


print(f"âœ“ Total features created: {len(train_full.columns)}")
print(f"Sample features: {list(train_full.columns[-15:])}")


# Sort data for time-series operations
train_full = train_full.sort_values(['Store', 'Dept', 'Date'])


for lag in [1, 2, 3, 4, 5, 12, 52]:
    train_full[f'Lag_{lag}'] = train_full.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)


for window in [4, 8, 13, 26]:
    train_full[f'Rolling_Mean_{window}'] = train_full.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
    )
    train_full[f'Rolling_Std_{window}'] = train_full.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
    )


# Calculate expanding statistics
train_full['Expanding_Mean'] = train_full.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
    lambda x: x.expanding().mean().shift(1)
)
train_full['Expanding_Std'] = train_full.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
    lambda x: x.expanding().std().shift(1)
)


# Fill NaN values for historical features
historical_features = [col for col in train_full.columns if 'Lag_' in col or 'Rolling_' in col or 'Expanding_' in col]
train_full[historical_features] = train_full[historical_features].fillna(0)

print(f"âœ“ Created {len(historical_features)} historical features")


from sklearn.cluster import KMeans

# First, let's check what columns we have
print("Available columns in train_full:")
print(train_full.columns.tolist())

# Check if IsHoliday column exists and its source
print("\nChecking IsHoliday column...")
if 'IsHoliday' in train_full.columns:
    print(f"IsHoliday column exists, dtype: {train_full['IsHoliday'].dtype}")
    print(f"Unique values: {train_full['IsHoliday'].unique()}")
else:
    # Check for IsHoliday_x or IsHoliday_y (common after merging)
    holiday_cols = [col for col in train_full.columns if 'Holiday' in col or 'holiday' in col]
    print(f"Holiday-related columns: {holiday_cols}")
    
    # Rename if necessary
    if 'IsHoliday_x' in train_full.columns:
        train_full['IsHoliday'] = train_full['IsHoliday_x']
        test_full['IsHoliday'] = test_full['IsHoliday_x']
        print("âœ“ Using IsHoliday_x as IsHoliday")
    elif 'IsHoliday_y' in train_full.columns:
        train_full['IsHoliday'] = train_full['IsHoliday_y']
        test_full['IsHoliday'] = test_full['IsHoliday_y']
        print("âœ“ Using IsHoliday_y as IsHoliday")

# Calculate department statistics
# Use the correct holiday column name
dept_stats = train_full.groupby('Dept').agg({
    'Weekly_Sales': ['mean', 'std', 'count']
}).round(2)

# Add holiday ratio if column exists
if 'IsHoliday' in train_full.columns:
    holiday_ratio = train_full.groupby('Dept')['IsHoliday'].mean()
    dept_stats['Holiday_Ratio'] = holiday_ratio
else:
    # Create a placeholder if holiday column doesn't exist
    dept_stats['Holiday_Ratio'] = 0.15  # Approximate holiday frequency

dept_stats.columns = ['Dept_Mean', 'Dept_Std', 'Dept_Count', 'Holiday_Ratio']
dept_stats['Dept_CV'] = dept_stats['Dept_Std'] / dept_stats['Dept_Mean']

# Filter departments with sufficient data
dept_stats_filtered = dept_stats[dept_stats['Dept_Count'] >= 50].copy()

print(f"\nDepartments for clustering: {len(dept_stats_filtered)} out of {len(dept_stats)}")

# Cluster departments based on sales patterns
print("\nClustering departments...")
X_cluster = dept_stats_filtered[['Dept_Mean', 'Dept_CV', 'Holiday_Ratio']].values
X_scaled = StandardScaler().fit_transform(X_cluster)

# Use K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
dept_stats_filtered['Dept_Cluster'] = kmeans.fit_predict(X_scaled)

# Map clusters back to all departments
dept_cluster_map = dept_stats_filtered['Dept_Cluster'].to_dict()

# Apply clustering to train and test data
train_full['Dept_Cluster'] = train_full['Dept'].map(dept_cluster_map)
test_full['Dept_Cluster'] = test_full['Dept'].map(dept_cluster_map)

# Fill missing clusters with mode
cluster_mode = train_full['Dept_Cluster'].mode()[0] if not train_full['Dept_Cluster'].mode().empty else 1
train_full['Dept_Cluster'] = train_full['Dept_Cluster'].fillna(cluster_mode)
test_full['Dept_Cluster'] = test_full['Dept_Cluster'].fillna(cluster_mode)

print("âœ“ Department clustering completed")
print(f"Cluster distribution:")
print(train_full['Dept_Cluster'].value_counts().sort_index())

# Analyze cluster characteristics
print("\n CLUSTER CHARACTERISTICS:")
print("-" * 30)
for cluster in sorted(train_full['Dept_Cluster'].unique()):
    cluster_data = train_full[train_full['Dept_Cluster'] == cluster]
    depts = cluster_data['Dept'].unique()
    avg_sales = cluster_data['Weekly_Sales'].mean()
    std_sales = cluster_data['Weekly_Sales'].std()
    
    print(f"\nCluster {cluster}:")
    print(f"  Departments: {len(depts)}")
    print(f"  Sample depts: {sorted(depts)[:5]}{'...' if len(depts) > 5 else ''}")
    print(f"  Avg sales: ${avg_sales:,.0f}")
    print(f"  Std sales: ${std_sales:,.0f}")
    print(f"  CV: {std_sales/avg_sales:.2f}")


# Calculate store-department historical statistics
store_dept_stats = train_full.groupby(['Store', 'Dept']).agg({
    'Weekly_Sales': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
store_dept_stats.columns = ['Store_Dept_Mean', 'Store_Dept_Std', 'Store_Dept_Min', 
                           'Store_Dept_Max', 'Store_Dept_Count']

# Merge with train data
train_full = pd.merge(train_full, store_dept_stats, on=['Store', 'Dept'], how='left')

# For store-dept combinations not in training, calculate fallback values
store_stats = train_full.groupby('Store').agg({
    'Weekly_Sales': 'mean',
    'Size': 'first'
}).rename(columns={'Weekly_Sales': 'Store_Mean'})

dept_stats_overall = train_full.groupby('Dept')['Weekly_Sales'].mean().rename('Dept_Mean_Overall')

# Fill missing store-dept stats
train_full['Store_Dept_Mean'] = train_full['Store_Dept_Mean'].fillna(
    train_full['Store'].map(store_stats['Store_Mean']) * 0.7 + 
    train_full['Dept'].map(dept_stats_overall) * 0.3
)
train_full['Store_Dept_Std'] = train_full['Store_Dept_Std'].fillna(train_full['Store_Dept_Mean'] * 0.3)

print("âœ“ Store-department features created")


# Calculate basic statistics
print("ğŸ“ˆ SALES STATISTICS:")
print("-" * 30)
print(f"Total sales: ${train_full['Weekly_Sales'].sum():,.0f}")
print(f"Average weekly sales: ${train_full['Weekly_Sales'].mean():,.0f}")
print(f"Median weekly sales: ${train_full['Weekly_Sales'].median():,.0f}")
print(f"Max weekly sales: ${train_full['Weekly_Sales'].max():,.0f}")
print(f"Min weekly sales: ${train_full['Weekly_Sales'].min():,.0f}")
print(f"Std dev: ${train_full['Weekly_Sales'].std():,.0f}")

# Holiday impact analysis
print("\nğŸ�¯ HOLIDAY IMPACT:")
print("-" * 30)
holiday_sales = train_full.groupby('IsHoliday')['Weekly_Sales'].mean()
holiday_lift = (holiday_sales[True] - holiday_sales[False]) / holiday_sales[False] * 100
print(f"Non-holiday average: ${holiday_sales[False]:,.0f}")
print(f"Holiday average: ${holiday_sales[True]:,.0f}")
print(f"Holiday lift: {holiday_lift:.1f}%")

# Store type analysis
print("\nğŸ�ª STORE TYPE ANALYSIS:")
print("-" * 30)
store_type_sales = train_full.groupby('Type')['Weekly_Sales'].mean()
for store_type, avg_sales in store_type_sales.items():
    print(f"Type {store_type}: ${avg_sales:,.0f}")

# Department analysis
print("\nğŸ“Š DEPARTMENT ANALYSIS:")
print("-" * 30)
top_depts = train_full.groupby('Dept')['Weekly_Sales'].sum().nlargest(5)
print("Top 5 departments by total sales:")
for dept, sales in top_depts.items():
    print(f"  Dept {dept}: ${sales:,.0f}")


fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Sales distribution
axes[0, 0].hist(train_full['Weekly_Sales'].clip(upper=50000), bins=100, edgecolor='black')
axes[0, 0].set_title('Weekly Sales Distribution (Clipped at $50k)')
axes[0, 0].set_xlabel('Weekly Sales ($)')
axes[0, 0].set_ylabel('Frequency')

# 2. Holiday vs Non-Holiday
holiday_comparison = train_full.groupby('IsHoliday')['Weekly_Sales'].mean()
axes[0, 1].bar(['Non-Holiday', 'Holiday'], holiday_comparison.values)
axes[0, 1].set_title('Average Sales: Holiday vs Non-Holiday')
axes[0, 1].set_ylabel('Average Weekly Sales ($)')
for i, v in enumerate(holiday_comparison.values):
    axes[0, 1].text(i, v, f'${v:,.0f}', ha='center', va='bottom')

# 3. Monthly sales pattern
monthly_sales = train_full.groupby('Month')['Weekly_Sales'].mean()
axes[0, 2].plot(monthly_sales.index, monthly_sales.values, marker='o')
axes[0, 2].set_title('Average Monthly Sales Pattern')
axes[0, 2].set_xlabel('Month')
axes[0, 2].set_ylabel('Average Weekly Sales ($)')
axes[0, 2].set_xticks(range(1, 13))

# 4. Store type comparison
store_type_avg = train_full.groupby('Type')['Weekly_Sales'].mean()
axes[1, 0].bar(store_type_avg.index, store_type_avg.values)
axes[1, 0].set_title('Average Sales by Store Type')
axes[1, 0].set_xlabel('Store Type')
axes[1, 0].set_ylabel('Average Weekly Sales ($)')

# 5. Sales trend over time
weekly_trend = train_full.groupby('Date')['Weekly_Sales'].sum() / 1e6
axes[1, 1].plot(weekly_trend.index, weekly_trend.values)
axes[1, 1].set_title('Total Weekly Sales Trend')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Sales ($ Millions)')

# 6. Department cluster sales
cluster_sales = train_full.groupby('Dept_Cluster')['Weekly_Sales'].mean()
axes[1, 2].bar([f'Cluster {i}' for i in cluster_sales.index], cluster_sales.values)
axes[1, 2].set_title('Average Sales by Department Cluster')
axes[1, 2].set_ylabel('Average Weekly Sales ($)')

plt.tight_layout()
plt.show()


# Define features for modeling
feature_columns = [
    # Store features
    'Store', 'Dept', 'Size', 'Store_Type_Encoded',
    
    # Date features
    'Year', 'Month', 'Week', 'DayOfWeek', 'Quarter',
    'Days_To_Holiday', 'Holiday_Proximity',
    
    # Holiday flags
    'IsHoliday', 'IsSuperBowl', 'IsLaborDay', 'IsThanksgiving', 'IsChristmas',
    
    # Seasonality
    'Is_Summer', 'Is_Winter', 'Is_Spring', 'Is_Fall',
    
    # Historical features
    'Lag_1', 'Lag_4', 'Lag_52',
    'Rolling_Mean_4', 'Rolling_Std_4',
    'Rolling_Mean_13', 'Rolling_Std_13',
    'Expanding_Mean', 'Expanding_Std',
    
    # Store-department features
    'Store_Dept_Mean', 'Store_Dept_Std',
    
    # Department cluster
    'Dept_Cluster',
    
    # Markdown features
    'Total_Markdown', 'Markdown_Count', 'Avg_Markdown',
    
    # Economic features
    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'CPI_Unemployment_Ratio', 'Fuel_Temperature_Index'
]

# Target variable
target_column = 'Weekly_Sales'

# Filter out rows with missing target or features
train_model = train_full.dropna(subset=[target_column]).copy()

# Fill any remaining NaN values
train_model[feature_columns] = train_model[feature_columns].fillna(0)

print(f"Training data shape: {train_model.shape}")
print(f"Number of features: {len(feature_columns)}")
print(f"Target variable: {target_column}")


def calculate_wmae(y_true, y_pred, is_holiday):
    """
    Calculate Weighted Mean Absolute Error (WMAE)
    w = 5 if week is holiday week, 1 otherwise
    """
    weights = np.where(is_holiday, 5, 1)
    absolute_errors = np.abs(y_true - y_pred)
    weighted_errors = absolute_errors * weights
    return np.sum(weighted_errors) / np.sum(weights)

def wmae_score(model, X, y, is_holiday):
    """Wrapper for WMAE calculation"""
    y_pred = model.predict(X)
    return calculate_wmae(y, y_pred, is_holiday)

print("âœ“ WMAE function defined")
print("Note: Holiday weeks are weighted 5x in the competition evaluation")


# Prepare data for baseline models
X_baseline = train_model[feature_columns].copy()
y_baseline = train_model[target_column].copy()
is_holiday_baseline = train_model['IsHoliday'].copy()


# Time-based split
split_date = pd.Timestamp('2012-06-01')
train_mask = train_model['Date'] < split_date
val_mask = train_model['Date'] >= split_date

X_train_base, X_val_base = X_baseline[train_mask], X_baseline[val_mask]
y_train_base, y_val_base = y_baseline[train_mask], y_baseline[val_mask]
holiday_train, holiday_val = is_holiday_baseline[train_mask], is_holiday_baseline[val_mask]

print(f"Training samples: {len(X_train_base):,}")
print(f"Validation samples: {len(X_val_base):,}")


# Baseline 1: Mean prediction
mean_pred = np.full_like(y_val_base, y_train_base.mean())
wmae_mean = calculate_wmae(y_val_base, mean_pred, holiday_val)
print(f"\n1. Mean Prediction WMAE: ${wmae_mean:,.0f}")

# Baseline 2: Store-Dept mean
store_dept_means = train_model[train_mask].groupby(['Store', 'Dept'])['Weekly_Sales'].mean()
store_dept_pred = X_val_base[['Store', 'Dept']].apply(
    lambda x: store_dept_means.get((x['Store'], x['Dept']), y_train_base.mean()), axis=1
)
wmae_store_dept = calculate_wmae(y_val_base, store_dept_pred, holiday_val)
print(f"2. Store-Dept Mean WMAE: ${wmae_store_dept:,.0f}")

# Baseline 3: Random Forest
print("\nTraining Random Forest baseline...")
rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


rf_model.fit(X_train_base, y_train_base)
rf_pred = rf_model.predict(X_val_base)
wmae_rf = calculate_wmae(y_val_base, rf_pred, holiday_val)
print(f"3. Random Forest WMAE: ${wmae_rf:,.0f}")


print(f"BEST BASELINE: Store-Dept Mean (WMAE: ${wmae_store_dept:,.0f})")


# Prepare weights for training (5x for holiday weeks)
train_weights = np.where(holiday_train, 5, 1)

print("Training XGBoost model with holiday weighting...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

# Train with validation set
xgb_model.fit(
    X_train_base, y_train_base,
    sample_weight=train_weights,
    eval_set=[(X_val_base, y_val_base)],
    verbose=50
)


# Predictions
xgb_train_pred = xgb_model.predict(X_train_base)
xgb_val_pred = xgb_model.predict(X_val_base)

# Calculate WMAE
xgb_train_wmae = calculate_wmae(y_train_base, xgb_train_pred, holiday_train)
xgb_val_wmae = calculate_wmae(y_val_base, xgb_val_pred, holiday_val)

print(f"\nXGBoost Model Performance:")
print(f"- Training WMAE: ${xgb_train_wmae:,.0f}")
print(f"- Validation WMAE: ${xgb_val_wmae:,.0f}")


# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nğŸ“Š TOP 10 FEATURES:")
print(feature_importance.head(10).to_string(index=False))


print("Training LightGBM model...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(
    X_train_base, y_train_base,
    sample_weight=train_weights,
    eval_set=[(X_val_base, y_val_base)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
)


# Predictions
lgb_train_pred = lgb_model.predict(X_train_base)
lgb_val_pred = lgb_model.predict(X_val_base)

# Calculate WMAE
lgb_train_wmae = calculate_wmae(y_train_base, lgb_train_pred, holiday_train)
lgb_val_wmae = calculate_wmae(y_val_base, lgb_val_pred, holiday_val)

print(f"\nLightGBM Model Performance:")
print(f"- Training WMAE: ${lgb_train_wmae:,.0f}")
print(f"- Validation WMAE: ${lgb_val_wmae:,.0f}")


# Create ensemble predictions
print("Creating ensemble of XGBoost and LightGBM...")
ensemble_val_pred = (xgb_val_pred * 0.6) + (lgb_val_pred * 0.4)

# Calculate ensemble WMAE
ensemble_val_wmae = calculate_wmae(y_val_base, ensemble_val_pred, holiday_val)

print(f"\nEnsemble Model Performance:")
print(f"- XGBoost WMAE: ${xgb_val_wmae:,.0f}")
print(f"- LightGBM WMAE: ${lgb_val_wmae:,.0f}")
print(f"- Ensemble (60/40) WMAE: ${ensemble_val_wmae:,.0f}")


models_summary = pd.DataFrame({
    'Model': ['Store-Dept Mean', 'Random Forest', 'XGBoost', 'LightGBM', 'Ensemble'],
    'Validation WMAE': [wmae_store_dept, wmae_rf, xgb_val_wmae, lgb_val_wmae, ensemble_val_wmae]
})
print(models_summary.to_string(index=False))


# Calculate store-dept statistics from training data
store_dept_test_stats = train_full.groupby(['Store', 'Dept']).agg({
    'Weekly_Sales': ['mean', 'std']
}).round(2)
store_dept_test_stats.columns = ['Store_Dept_Mean_Test', 'Store_Dept_Std_Test']

# Merge with test data
test_full = pd.merge(test_full, store_dept_test_stats, on=['Store', 'Dept'], how='left')

# Fill missing values for store-dept combinations not in training
store_means = train_full.groupby('Store')['Weekly_Sales'].mean()
dept_means = train_full.groupby('Dept')['Weekly_Sales'].mean()

test_full['Store_Dept_Mean_Test'] = test_full['Store_Dept_Mean_Test'].fillna(
    test_full['Store'].map(store_means) * 0.7 + 
    test_full['Dept'].map(dept_means) * 0.3
)
test_full['Store_Dept_Std_Test'] = test_full['Store_Dept_Std_Test'].fillna(
    test_full['Store_Dept_Mean_Test'] * 0.3
)

# Use store-dept mean as lag features for test data
test_full['Lag_1'] = test_full['Store_Dept_Mean_Test']
test_full['Lag_4'] = test_full['Store_Dept_Mean_Test']
test_full['Lag_52'] = test_full['Store_Dept_Mean_Test']
test_full['Rolling_Mean_4'] = test_full['Store_Dept_Mean_Test']
test_full['Rolling_Std_4'] = test_full['Store_Dept_Std_Test']
test_full['Rolling_Mean_13'] = test_full['Store_Dept_Mean_Test']
test_full['Rolling_Std_13'] = test_full['Store_Dept_Std_Test']
test_full['Expanding_Mean'] = test_full['Store_Dept_Mean_Test']
test_full['Expanding_Std'] = test_full['Store_Dept_Std_Test']
test_full['Store_Dept_Mean'] = test_full['Store_Dept_Mean_Test']
test_full['Store_Dept_Std'] = test_full['Store_Dept_Std_Test']

# Ensure all feature columns exist in test data
for col in feature_columns:
    if col not in test_full.columns:
        test_full[col] = 0

# Prepare test features
X_test = test_full[feature_columns].fillna(0)

print(f"Test data prepared: {X_test.shape}")
print(f"Missing values: {X_test.isnull().sum().sum()}")


# Prepare full training data
X_full = train_model[feature_columns].fillna(0)
y_full = train_model[target_column]
full_weights = np.where(train_model['IsHoliday'], 5, 1)

# Train XGBoost on full data
print("Training final XGBoost model...")
xgb_final = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
xgb_final.fit(X_full, y_full, sample_weight=full_weights)


# Train LightGBM on full data
print("Training final LightGBM model...")
lgb_final = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
lgb_final.fit(X_full, y_full, sample_weight=full_weights)


print("\nGenerating predictions...")
xgb_test_pred = xgb_final.predict(X_test)
lgb_test_pred = lgb_final.predict(X_test)


# Ensemble predictions
ensemble_test_pred = (xgb_test_pred * 0.6) + (lgb_test_pred * 0.4)

print("âœ“ Predictions generated successfully!")
print(f"\nPrediction statistics:")
print(f"- XGBoost: Mean=${xgb_test_pred.mean():,.0f}, Min=${xgb_test_pred.min():,.0f}, Max=${xgb_test_pred.max():,.0f}")
print(f"- LightGBM: Mean=${lgb_test_pred.mean():,.0f}, Min=${lgb_test_pred.min():,.0f}, Max=${lgb_test_pred.max():,.0f}")
print(f"- Ensemble: Mean=${ensemble_test_pred.mean():,.0f}, Min=${ensemble_test_pred.min():,.0f}, Max=${ensemble_test_pred.max():,.0f}")


# Create submission DataFrame
submission = pd.DataFrame({
    'Id': test_full.apply(lambda x: f"{x['Store']}_{x['Dept']}_{x['Date'].strftime('%Y-%m-%d')}", axis=1),
    'Weekly_Sales': ensemble_test_pred
})

# Ensure no negative predictions
submission['Weekly_Sales'] = submission['Weekly_Sales'].clip(lower=0)

# Save submission file
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)

print(f"âœ“ Submission file created: {submission_file}")
print(f"Submission shape: {submission.shape}")
print(f"\nSubmission preview:")
print(submission.head(10))

print(f"\nğŸ“Š SUBMISSION STATISTICS:")
print("-" * 30)
print(f"Total predicted sales: ${submission['Weekly_Sales'].sum():,.0f}")
print(f"Average prediction: ${submission['Weekly_Sales'].mean():,.0f}")
print(f"Median prediction: ${submission['Weekly_Sales'].median():,.0f}")
print(f"Min prediction: ${submission['Weekly_Sales'].min():,.0f}")
print(f"Max prediction: ${submission['Weekly_Sales'].max():,.0f}")
print(f"Rows with 0 prediction: {(submission['Weekly_Sales'] == 0).sum()}")


# Analyze holiday week predictions
test_full['Predicted_Sales'] = ensemble_test_pred
holiday_predictions = test_full.groupby('IsHoliday')['Predicted_Sales'].agg(['mean', 'count'])

print("\nğŸ�¯ HOLIDAY WEEK PREDICTIONS:")
print("-" * 30)
print(f"Holiday weeks: {holiday_predictions.loc[True, 'count']:,} rows")
print(f"Non-holiday weeks: {holiday_predictions.loc[False, 'count']:,} rows")
print(f"\nAverage predicted sales:")
print(f"- Holiday weeks: ${holiday_predictions.loc[True, 'mean']:,.0f}")
print(f"- Non-holiday weeks: ${holiday_predictions.loc[False, 'mean']:,.0f}")
print(f"- Holiday lift: {(holiday_predictions.loc[True, 'mean'] / holiday_predictions.loc[False, 'mean'] - 1) * 100:.1f}%")


# Visualize predictions by store type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predictions by store type
store_type_pred = test_full.groupby('Type')['Predicted_Sales'].mean()
axes[0].bar(store_type_pred.index, store_type_pred.values)
axes[0].set_title('Average Predicted Sales by Store Type')
axes[0].set_xlabel('Store Type')
axes[0].set_ylabel('Predicted Weekly Sales ($)')

# Predictions by month
month_pred = test_full.groupby('Month')['Predicted_Sales'].mean()
axes[1].plot(month_pred.index, month_pred.values, marker='o')
axes[1].set_title('Predicted Sales by Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Predicted Weekly Sales ($)')
axes[1].set_xticks(range(1, 13))

plt.tight_layout()
plt.show()




