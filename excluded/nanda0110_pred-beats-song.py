import numpy as np
import pandas as pd
SEED = 42
np.random.seed(SEED)

df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


def inspect_data(df):
    print("ğŸ”¹ Dataset Shape:", df.shape)
    print("\nğŸ”¹ Data Types:")
    print(df.dtypes)
    print("\nğŸ”¹ Duplicate IDs:", df['id'].duplicated().sum())
    print("\nğŸ”¹ Summary Statistics:")
    print(df.describe())

# Run inspection
inspect_data(df)


def analyze_missing_values(df):
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing %': missing_percent
    }).sort_values(by='Missing %', ascending=False)
    
    print("\nğŸ”¹ Missing Values per Column:")
    print(missing_df)
    return missing_df

missing_df = analyze_missing_values(df)


def handle_missing_values(df, threshold=5):
    """
    Impute columns with < threshold% missing using median,
    and drop columns with >= threshold% missing.
    """
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct == 0:
            continue
        elif missing_pct < threshold:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"âœ… Filled missing values in '{col}' with median ({median_value:.4f})")
        else:
            df.drop(columns=[col], inplace=True)
            print(f"âš ï¸� Dropped column '{col}' with {missing_pct:.2f}% missing values")
    return df

df_clean = handle_missing_values(df)


print("\nâœ… After Cleaning:")
print(df_clean.isnull().sum().sum(), "missing values remaining")
print("Final shape:", df_clean.shape)


import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def descriptive_statistics(df):
    print("ğŸ”¹ Descriptive Statistics (Mean, Median, Min, Max, Std):\n")
    desc = df.describe().T
    desc['median'] = df.median()
    print(desc[['mean', 'median', 'min', 'max', 'std']])

    # Compare feature scales visually
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df.drop(columns=['id']), orient='h')
    plt.title("Feature Scale Comparison (Boxplot)")
    plt.show()

descriptive_statistics(df_clean)


def feature_distribution(df, target_col='BeatsPerMinute'):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(['id'])
    
    # Plot histograms / density plots
    df[numeric_cols].hist(figsize=(16, 10), bins=40, edgecolor='black')
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.show()

    # Focus on target feature
    plt.figure(figsize=(6, 4))
    sns.histplot(df[target_col], kde=True, bins=40, color='blue')
    plt.title(f"Distribution of {target_col}")
    plt.show()
    
    # Detect extreme outliers
    bpm_outliers = df[(df[target_col] > 250) | (df[target_col] < 40)]
    print(f"âš ï¸� Potential extreme outliers in '{target_col}': {len(bpm_outliers)} rows")

feature_distribution(df_clean)


def correlation_heatmap(df, target_col='BeatsPerMinute'):
    plt.figure(figsize=(10, 8))
    corr = df.drop(columns=['id']).corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False, center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Print top correlated features with target
    print(f"\nğŸ”¹ Top features correlated with {target_col}:\n")
    target_corr = corr[target_col].sort_values(ascending=False)
    print(target_corr.head(6))

correlation_heatmap(df_clean)


def multicollinearity_analysis(df):
    print("\nğŸ”¹ Variance Inflation Factor (VIF) Analysis:\n")
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['id'], errors='ignore')
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_df.columns
    vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i)
                       for i in range(len(numeric_df.columns))]
    
    print(vif_data.sort_values(by='VIF', ascending=False))
    high_vif = vif_data[vif_data["VIF"] > 10]
    if not high_vif.empty:
        print("\nâš ï¸� High multicollinearity detected in:")
        print(high_vif)
    else:
        print("\nâœ… No significant multicollinearity detected.")

multicollinearity_analysis(df_clean)


from sklearn.preprocessing import StandardScaler, MinMaxScaler

target_col = 'BeatsPerMinute'

X = df.drop(columns=[target_col])
y = df[target_col]

redundant_features = ['RhythmScore'] 
X = X.drop(columns=redundant_features)


# Identify feature groups
standard_scale_features = ['TrackDurationMs', 'AudioLoudness']
minmax_scale_features = [
    'Energy', 'MoodScore', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'VocalContent'
]

# Create copies for transformation
X_scaled = X.copy()

# Apply StandardScaler
standard_scaler = StandardScaler()
X_scaled[standard_scale_features] = standard_scaler.fit_transform(X[standard_scale_features])

# Apply MinMaxScaler
minmax_scaler = MinMaxScaler()
X_scaled[minmax_scale_features] = minmax_scaler.fit_transform(X[minmax_scale_features])


print("\nâœ… Data Preparation Completed:")
print(f"Features after preprocessing: {X_scaled.shape[1]}")
print(f"Feature columns:\n{list(X_scaled.columns)}")

print("\nSample preview of scaled data:")
display(X_scaled.head())


# Drop redundant features (based on same selection as training)
df_test_prep = df_test.drop(columns=redundant_features, errors='ignore')

# Ensure the test set has the same columns as X (training predictors)
df_test_prep = df_test_prep[X.columns]


# Use the fitted scalers from the training set (DO NOT refit)
df_test_scaled = df_test_prep.copy()
df_test_scaled[standard_scale_features] = standard_scaler.transform(df_test_prep[standard_scale_features])
df_test_scaled[minmax_scale_features] = minmax_scaler.transform(df_test_prep[minmax_scale_features])


print("\nâœ… Test Data Prepared Successfully:")
print(f"Test data shape: {df_test_scaled.shape}")
print(f"Feature columns:\n{list(df_test_scaled.columns)}")

print("\nSample preview of scaled test data:")
display(df_test_scaled.head())


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def feature_engineering(df):
    """
    Create new musical interaction features and non-linear transformations
    (excluding RhythmScore due to high multicollinearity).
    """
    df = df.copy()
    
    # --- Interaction Features ---
    df['LoudnessEnergy'] = df['AudioLoudness'] * df['Energy']                 # overall intensity
    df['AcousticInstrumental'] = df['AcousticQuality'] * df['InstrumentalScore']  # instrumental clarity
    df['MoodVocal'] = df['MoodScore'] * df['VocalContent']                   # emotional expression
    
    # --- Non-linear Transformations ---
    df['LogTrackDuration'] = np.log1p(df['TrackDurationMs'])                 # stabilize duration variance
    df['SqrtEnergy'] = np.sqrt(df['Energy'])                                 # add energy diversity
    df['SquareEnergy'] = np.square(df['Energy'])                             # emphasize high energy

    return df

# Apply feature engineering
X_train = feature_engineering(X_scaled)
X_test  = feature_engineering(df_test_scaled)

# Drop RhythmScore and target handling
cols_to_drop = ['id']
X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
X_test  = X_test.drop(columns=cols_to_drop, errors='ignore')

print("Feature Engineering Completed (RhythmScore removed).")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


standard_features = ['AudioLoudness', 'TrackDurationMs', 'LogTrackDuration']
minmax_features = [col for col in X_train.columns if col not in standard_features]

scaler_std = StandardScaler()
scaler_mm  = MinMaxScaler()

# Apply scaling
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[standard_features] = scaler_std.fit_transform(X_train[standard_features])
X_test_scaled[standard_features]  = scaler_std.transform(X_test[standard_features])

X_train_scaled[minmax_features] = scaler_mm.fit_transform(X_train[minmax_features])
X_test_scaled[minmax_features]  = scaler_mm.transform(X_test[minmax_features])

print("Feature Scaling Completed!")
print(f"Scaled Train shape: {X_train_scaled.shape}, Scaled Test shape: {X_test_scaled.shape}")


xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Train quick model
xgb_model.fit(X_train_scaled, y)


importance = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance.head(15), x='Importance', y='Feature', palette='viridis')
plt.title("Top 15 Most Important Features (Without RhythmScore)")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
    X_train_scaled, y, test_size=0.2, random_state=42
)



xgb = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    random_state=42,
    tree_method='hist',        # Faster training
    reg_lambda=1.0,            # L2 regularization
    reg_alpha=0.1,             # L1 regularization
    early_stopping_rounds=20,  # Stop early if no improvement
    n_jobs=-1
)

# Reduced grid for faster tuning
param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.03, 0.05],
    'max_depth': [4, 5],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)



grid_search.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)

best_model = grid_search.best_estimator_

y_pred_val = best_model.predict(X_val_scaled)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

print("âœ… Best Parameters:", grid_search.best_params_)
print("ğŸ“‰ Validation RMSE:", round(rmse_val, 4))


y_test_pred = best_model.predict(X_test_scaled)
df_test['BeatsPerMinute'] = y_test_pred


df_submission = df_test[['id', 'BeatsPerMinute']]
df_submission.to_csv('submission.csv', index=False, sep=',')

