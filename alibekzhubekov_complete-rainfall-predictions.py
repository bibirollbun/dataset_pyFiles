# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)



# ================================================================
# Time Series Feature Engineering Class
# ================================================================

class TimeSeriesFeatureEngineering:
    def __init__(self, 
                 lag_features=[1, 7, 14, 30],
                 rolling_windows=[7, 14, 30],
                 trend_periods=[7, 14, 30]):
        """Initialize the TimeSeriesFeatureEngineering class"""
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.trend_periods = trend_periods
        self.scaler = StandardScaler()
        
    def create_lag_features(self, df, target_col):
        """Create lag features for the target variable"""
        df_copy = df.copy()
        
        for lag in self.lag_features:
            df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
            
        return df_copy
    
    def create_rolling_features(self, df, target_col):
        """Create rolling window statistics"""
        df_copy = df.copy()
        
        for window in self.rolling_windows:
            # Calculate rolling mean
            df_copy[f'{target_col}_rolling_mean_{window}'] = (
                df_copy[target_col].rolling(window=window).mean()
            )
            
            # Calculate rolling standard deviation
            df_copy[f'{target_col}_rolling_std_{window}'] = (
                df_copy[target_col].rolling(window=window).std()
            )
            
            # Calculate rolling min and max
            df_copy[f'{target_col}_rolling_min_{window}'] = (
                df_copy[target_col].rolling(window=window).min()
            )
            df_copy[f'{target_col}_rolling_max_{window}'] = (
                df_copy[target_col].rolling(window=window).max()
            )
            
        return df_copy
    
    def create_trend_indicators(self, df, target_col):
        """Create trend indicators including momentum and rate of change"""
        df_copy = df.copy()
        
        for period in self.trend_periods:
            # Momentum (absolute change)
            df_copy[f'{target_col}_momentum_{period}'] = (
                df_copy[target_col] - df_copy[target_col].shift(period)
            )
            
            # Rate of Change (ROC)
            df_copy[f'{target_col}_roc_{period}'] = (
                (df_copy[target_col] - df_copy[target_col].shift(period)) / 
                df_copy[target_col].shift(period) * 100
            )
            
            # Exponential Moving Average (EMA)
            df_copy[f'{target_col}_ema_{period}'] = (
                df_copy[target_col].ewm(span=period, adjust=False).mean()
            )
            
        return df_copy
    
    def prepare_features(self, df, target_col, scale_features=True):
        """Prepare all time series features"""
        df_features = df.copy()
        
        # Create all features
        df_features = self.create_lag_features(df_features, target_col)
        df_features = self.create_rolling_features(df_features, target_col)
        df_features = self.create_trend_indicators(df_features, target_col)
        
        # Drop rows with NaN values (caused by lag/rolling operations)
        df_features = df_features.dropna()
        
        if scale_features:
            # Get feature columns (excluding target)
            feature_cols = [col for col in df_features.columns if col != target_col]
            
            # Scale features
            df_features[feature_cols] = self.scaler.fit_transform(df_features[feature_cols])
        
        return df_features

def time_based_cv_split(X, y, n_splits=5, test_size=0.2):
    """Perform time-based cross-validation split"""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X) * test_size))
    return [(train_idx, test_idx) for train_idx, test_idx in tscv.split(X)]


# ================================================================
# Feature Engineering Functions
# ================================================================

def create_cyclical_features(df, col, period):
    """Create sin and cos features to capture cyclical nature of time variables"""
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/period)
    return df

def add_season_features(df, day_col):
    """Add season indicators based on day of year"""
    # Simple season (4 seasons)
    day = df[day_col]
    
    # Meteorological seasons in Northern Hemisphere
    conditions = [
        (day >= 1) & (day <= 59),     # Winter: Jan 1 - Feb 28
        (day >= 60) & (day <= 151),   # Spring: Mar 1 - May 31
        (day >= 152) & (day <= 243),  # Summer: Jun 1 - Aug 31
        (day >= 244) & (day <= 334),  # Fall: Sep 1 - Nov 30
        (day >= 335) & (day <= 365)   # Winter: Dec 1 - Dec 31
    ]
    seasons = [0, 1, 2, 3, 0]  # 0:Winter, 1:Spring, 2:Summer, 3:Fall
    df['season'] = np.select(conditions, seasons, default=0)
    
    # Create dummy variables for seasons
    for season in range(4):
        df[f'season_{season}'] = (df['season'] == season).astype(int)
    
    # Quarter of the year (1, 2, 3, 4)
    df['quarter'] = pd.cut(df[day_col], bins=[0, 90, 181, 273, 365], labels=[1, 2, 3, 4]).astype(int)
    
    # Create dummy variables for quarters
    for quarter in range(1, 5):
        df[f'quarter_{quarter}'] = (df['quarter'] == quarter).astype(int)
    
    return df

def create_weather_features(df):
    """Create advanced weather-related features"""
    df_copy = df.copy()
    
    # Temperature-related
    if 'temparature' in df_copy.columns and 'maxtemp' in df_copy.columns and 'mintemp' in df_copy.columns:
        # Temperature range
        df_copy['temp_range'] = df_copy['maxtemp'] - df_copy['mintemp']
        
        # Temperature relative to min/max
        df_copy['temp_relative_min'] = df_copy['temparature'] - df_copy['mintemp']
        df_copy['temp_relative_max'] = df_copy['maxtemp'] - df_copy['temparature']
        
        # Temperature variability
        df_copy['temp_variability'] = df_copy['temp_range'] / df_copy['temparature']
    
    # Humidity and temperature interaction
    if 'humidity' in df_copy.columns:
        # Different variants of humidity interactions
        if 'temparature' in df_copy.columns:
            df_copy['humidity_temp'] = df_copy['humidity'] * df_copy['temparature']
            df_copy['humidity_temp_ratio'] = df_copy['humidity'] / df_copy['temparature']
        
        # Humidity squared (to capture nonlinear effects)
        df_copy['humidity_squared'] = df_copy['humidity'] ** 2
    
    # Cloud and sunshine interactions
    if 'cloud' in df_copy.columns and 'sunshine' in df_copy.columns:
        df_copy['cloud_sunshine_ratio'] = df_copy['cloud'] / (df_copy['sunshine'] + 0.1)  # Adding 0.1 to avoid division by zero
        df_copy['cloud_sunshine_product'] = df_copy['cloud'] * df_copy['sunshine']
    
    # Wind features
    if 'windspeed' in df_copy.columns:
        df_copy['windspeed_squared'] = df_copy['windspeed'] ** 2
        
        if 'winddirection' in df_copy.columns:
            # Create wind component features (x and y direction)
            df_copy['wind_x'] = df_copy['windspeed'] * np.cos(np.radians(df_copy['winddirection']))
            df_copy['wind_y'] = df_copy['windspeed'] * np.sin(np.radians(df_copy['winddirection']))
    
    # Pressure interactions
    if 'pressure' in df_copy.columns:
        if 'temparature' in df_copy.columns:
            df_copy['pressure_temp'] = df_copy['pressure'] * df_copy['temparature']
        if 'humidity' in df_copy.columns:
            df_copy['pressure_humidity'] = df_copy['pressure'] * df_copy['humidity']
    
    return df_copy

def select_features_by_importance(X, y, threshold=0.01):
    """Select features based on mutual information with target"""
    # Ensure no missing values for mutual_info_classif
    X_temp = SimpleImputer(strategy='median').fit_transform(X)
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X_temp, y, random_state=42)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    
    # Sort features by importance
    mi_scores = mi_scores.sort_values(ascending=False)
    
    # Select features with importance above threshold
    selected_features = mi_scores[mi_scores > threshold].index.tolist()
    
    print(f"Top 10 features by mutual information:")
    for idx, (feature, score) in enumerate(mi_scores.head(10).items()):
        print(f"  {idx+1}. {feature}: {score:.4f}")
    
    return selected_features


# ================================================================
# Main Workflow
# ================================================================

print("====================================")
print("Advanced Rainfall Prediction Model")
print("====================================")

# Load the data
print("\nLoading and preprocessing data...")
base_dir = '/kaggle/input/playground-series-s5e3/'
train_data = pd.read_csv(base_dir + 'train.csv')
test_data = pd.read_csv(base_dir + 'test.csv')

# Check for missing values
print("Checking for missing values:")
print("  Train data missing values:", train_data.isna().sum().sum())
print("  Test data missing values:", test_data.isna().sum().sum())

# Separate features and target
X = train_data.drop(['id', 'rainfall'], axis=1)
y = train_data['rainfall']
test_features = test_data.drop(['id'], axis=1)

# Add day column if not exists (important for time series features)
if 'day' not in X.columns:
    print("WARNING: 'day' column not found in data, some time-based features may not work.")

# Print class distribution
print("\nClass distribution in training data:")
print(y.value_counts(normalize=True) * 100)


# ================================================================
# Feature Engineering
# ================================================================
print("\nPerforming feature engineering...")

# 1. Apply time series features if there's time information
# For time series features, we need to ensure data is in chronological order
if 'day' in X.columns:
    # Sort data by day to ensure correct time sequence
    train_data_sorted = train_data.sort_values('day').reset_index(drop=True)
    
    # Apply time series feature engineering
    # Choose a numeric column for time-based features
    numeric_cols = train_data_sorted.select_dtypes(include=['float64', 'int64']).columns
    target_col = numeric_cols[0]  # Use first numeric column
    
    print(f"Applying time series features to column: {target_col}")
    ts_features = TimeSeriesFeatureEngineering(
        lag_features=[1, 3, 7], 
        rolling_windows=[3, 7, 14], 
        trend_periods=[3, 7, 14]
    )
    train_data_with_ts = ts_features.prepare_features(
        train_data_sorted, target_col=target_col, scale_features=False
    )
    
    # Extract the additional features and add them to the original dataset
    ts_feature_cols = [col for col in train_data_with_ts.columns 
                     if col not in train_data.columns and col != target_col]
    
    # Join the time series features with the original X (only keeping rows without NaN)
    valid_indices = train_data_with_ts.index
    X_with_ts = X.loc[valid_indices].copy()
    
    for col in ts_feature_cols:
        X_with_ts[col] = train_data_with_ts[col]
    
    # Update X with time series features
    X = X_with_ts.copy()
    
    # Since we've removed some rows due to lagging, adjust y accordingly
    y = y.loc[valid_indices]
    
    print(f"Added {len(ts_feature_cols)} time series features")
    print(f"Data shape after time series feature engineering: {X.shape}")


# 2. Apply cyclical features for day
if 'day' in X.columns:
    X = create_cyclical_features(X, 'day', 365)
    test_features = create_cyclical_features(test_features, 'day', 365)
    
    # 3. Add seasonal features
    X = add_season_features(X, 'day')
    test_features = add_season_features(test_features, 'day')
    
# 4. Create weather features
X = create_weather_features(X)
test_features = create_weather_features(test_features)

# 5. Create polynomial features for important numerical variables
important_weather_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()[:4]
print(f"Creating polynomial features for: {important_weather_features}")

X_poly_subset = X[important_weather_features]
test_poly_subset = test_features[important_weather_features]

# Create polynomial features (degree 2) for selected features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X_poly_subset)
test_poly = poly.transform(test_poly_subset)

# Convert to DataFrame with appropriate column names
poly_feature_names = poly.get_feature_names_out(input_features=important_weather_features)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
test_poly_df = pd.DataFrame(test_poly, columns=poly_feature_names)

# Add polynomial features to original features
X = pd.concat([X.reset_index(drop=True), X_poly_df.iloc[:, len(important_weather_features):]], axis=1)
test_features = pd.concat([test_features.reset_index(drop=True), test_poly_df.iloc[:, len(important_weather_features):]], axis=1)

print(f"After feature engineering: {X.shape[1]} features created")

# ================================================================
# Clean data - handle NaNs and infinities
# ================================================================
print("\nCleaning data - handling NaNs and infinities...")

# Replace infinities with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)
test_features.replace([np.inf, -np.inf], np.nan, inplace=True)

# Count NaNs in dataset
nan_count_before = X.isna().sum().sum()
print(f"NaN values in training data before imputation: {nan_count_before}")

# Ensure we only impute columns that exist in both datasets
# Find common numerical columns
X_num_cols = X.select_dtypes(include=['float64', 'int64']).columns
test_num_cols = test_features.select_dtypes(include=['float64', 'int64']).columns
common_num_cols = [col for col in X_num_cols if col in test_num_cols]

# Print information about missing columns
missing_cols = [col for col in X_num_cols if col not in test_num_cols]
if missing_cols:
    print(f"Warning: {len(missing_cols)} columns in training not found in test data.")
    print(f"These include time series features that need special handling.")

# First, impute the common columns in both datasets
if common_num_cols:
    print(f"Imputing {len(common_num_cols)} common numerical columns...")
    common_imputer = SimpleImputer(strategy='median')
    X[common_num_cols] = common_imputer.fit_transform(X[common_num_cols])
    test_features[common_num_cols] = common_imputer.transform(test_features[common_num_cols])

# Then handle any remaining numerical columns in X separately
remaining_num_cols = [col for col in X_num_cols if col not in common_num_cols]
if remaining_num_cols:
    print(f"Imputing {len(remaining_num_cols)} columns only in training data...")
    X_imputer = SimpleImputer(strategy='median')
    X[remaining_num_cols] = X_imputer.fit_transform(X[remaining_num_cols])

# Verify that NaNs are handled
nan_count_after = X.isna().sum().sum()
print(f"NaN values in training data after imputation: {nan_count_after}")


# ================================================================
# Feature Selection
# ================================================================
print("\nPerforming feature selection...")

# 1. Check correlation to drop highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print(f"Dropping {len(to_drop)} highly correlated features")

# Drop highly correlated features
X = X.drop(to_drop, axis=1)
test_features = test_features.drop([col for col in to_drop if col in test_features.columns], axis=1)

# 2. Select features by importance
important_features = select_features_by_importance(X, y, threshold=0.005)
print(f"Selected {len(important_features)} features based on importance")

# Keep only important features that exist in both datasets
common_important_features = [col for col in important_features if col in test_features.columns]
if len(common_important_features) < len(important_features):
    print(f"Warning: {len(important_features) - len(common_important_features)} important features not found in test data.")
    print(f"Using {len(common_important_features)} common important features.")

X = X[common_important_features]
test_features = test_features[common_important_features]
print(f"Final feature count: {X.shape[1]}")


# ================================================================
# Train-Test Split
# ================================================================
print("\nSplitting data for training and validation...")

# Reset indices to make sure X and y are aligned
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Choose splitting strategy based on data characteristics
if 'day' in X.columns and len(set(X['day'])) > len(X) * 0.1:
    print("Using time-based cross-validation")
    # For time-based data, use time-based split
    
    # Create a DataFrame with both X and y to maintain alignment
    combined_df = pd.concat([X, pd.Series(y, name='target')], axis=1)
    combined_df_sorted = combined_df.sort_values('day')
    
    # Use 80% for training, 20% for validation
    split_idx = int(len(combined_df) * 0.8)
    
    # Split the combined DataFrame
    train_df = combined_df_sorted.iloc[:split_idx]
    val_df = combined_df_sorted.iloc[split_idx:]
    
    # Separate features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_val = val_df.drop('target', axis=1)
    y_val = val_df['target']
    
    print(f"Time-based split: {len(X_train)} training samples, {len(X_val)} validation samples")
else:
    print("Using stratified split")
    # For non-time data, use stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Stratified split: {len(X_train)} training samples, {len(X_val)} validation samples")


# ================================================================
# Model Training and Evaluation
# ================================================================
print("\nTraining and evaluating models...")

# Create a list of models
models = {
    'Random Forest': RandomForestClassifier(
        class_weight='balanced',
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    ),
    'SVM': SVC(
        probability=True, 
        class_weight='balanced',
        random_state=42
    )
}

best_model = None
best_score = 0
results = {}

# Create preprocessor
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train with cross-validation
    if 'day' in X.columns and len(set(X['day'])) > len(X) * 0.1:
        # Time-based cross-validation
        
        # Adjust parameters based on dataset size to avoid "too many splits" error
        n_samples = len(X_train)
        # Use smaller test size and fewer splits for smaller datasets
        if n_samples < 2000:  # Small dataset
            n_splits = 3
            test_size_fraction = 0.15
        else:  # Larger dataset
            n_splits = 5
            test_size_fraction = 0.2
        
        test_size_absolute = int(n_samples * test_size_fraction)
        
        print(f"Using time-based CV with {n_splits} splits and test size of {test_size_absolute} samples")
        cv_splits = time_based_cv_split(X_train, y_train, n_splits=n_splits, test_size=test_size_fraction)
        
        cv_scores = []
        for train_idx, test_idx in cv_splits:
            X_train_cv, X_test_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]
            
            pipeline.fit(X_train_cv, y_train_cv)
            y_pred_proba = pipeline.predict_proba(X_test_cv)[:, 1]
            
            cv_scores.append(roc_auc_score(y_test_cv, y_pred_proba))
        
        cv_score_mean = np.mean(cv_scores)
        cv_score_std = np.std(cv_scores)
    else:
        # Traditional stratified cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        cv_score_mean = cv_scores.mean()
        cv_score_std = cv_scores.std()
    
    print(f"Cross-validation ROC-AUC: {cv_score_mean:.4f} (+/- {cv_score_std:.4f})")
    
    # Fit on the whole training set
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_val)
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    # Calculate PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Store results
    results[name] = {
        'cv_score': cv_score_mean,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'model': pipeline
    }
    
    # Print results
    print(f"Validation Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  PR AUC:    {pr_auc:.4f}")
    
    # Check if this is the best model
    if pr_auc > best_score:
        best_score = pr_auc
        best_model = name


# ================================================================
# Create Ensemble Model
# ================================================================
print("\nCreating ensemble model...")

# Get the best 3 models
sorted_models = sorted(results.items(), key=lambda x: x[1]['pr_auc'], reverse=True)
top_models = [model_name for model_name, _ in sorted_models[:3]]

print(f"Top 3 models: {top_models}")

# Create a voting ensemble of the top 3 models
voting_ensemble = VotingClassifier(
    estimators=[
        (name.lower().replace(' ', '_'), results[name]['model']) 
        for name in top_models
    ],
    voting='soft'  # Use probability predictions for voting
)

# Train the voting ensemble
voting_ensemble.fit(X_train, y_train)

# Make predictions with the ensemble
y_pred_ensemble = voting_ensemble.predict(X_val)
y_pred_proba_ensemble = voting_ensemble.predict_proba(X_val)[:, 1]

# Calculate metrics for the ensemble
accuracy_ensemble = accuracy_score(y_val, y_pred_ensemble)
precision_ensemble = precision_score(y_val, y_pred_ensemble)
recall_ensemble = recall_score(y_val, y_pred_ensemble)
f1_ensemble = f1_score(y_val, y_pred_ensemble)
roc_auc_ensemble = roc_auc_score(y_val, y_pred_proba_ensemble)

# Calculate PR AUC for the ensemble
precision_curve_ensemble, recall_curve_ensemble, _ = precision_recall_curve(y_val, y_pred_proba_ensemble)
pr_auc_ensemble = auc(recall_curve_ensemble, precision_curve_ensemble)

# Print ensemble results
print("\nEnsemble Model Results:")
print(f"  Accuracy:  {accuracy_ensemble:.4f}")
print(f"  Precision: {precision_ensemble:.4f}")
print(f"  Recall:    {recall_ensemble:.4f}")
print(f"  F1 Score:  {f1_ensemble:.4f}")
print(f"  ROC AUC:   {roc_auc_ensemble:.4f}")
print(f"  PR AUC:    {pr_auc_ensemble:.4f}")

# Check if ensemble is better than the best individual model
if pr_auc_ensemble > best_score:
    best_score = pr_auc_ensemble
    best_model = "Ensemble"
    print("Ensemble model is the best model!")
    # Add ensemble to results
    results["Ensemble"] = {
        'accuracy': accuracy_ensemble,
        'precision': precision_ensemble,
        'recall': recall_ensemble,
        'f1_score': f1_ensemble,
        'roc_auc': roc_auc_ensemble,
        'pr_auc': pr_auc_ensemble,
        'model': voting_ensemble
    }
else:
    print(f"Best individual model ({best_model}) outperforms the ensemble.")

# Print summary of all models
print("\n===== Model Performance Summary =====")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  PR AUC:    {metrics['pr_auc']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

print(f"\nBest model: {best_model} with PR-AUC score {best_score:.4f}")


# ================================================================
# Generate Predictions on Test Set
# ================================================================
print("\nGenerating predictions for test set...")

# Use the best model to predict on test data
if best_model == "Ensemble":
    final_model = voting_ensemble
else:
    final_model = results[best_model]['model']

# Generate predictions
test_predictions = final_model.predict_proba(test_features)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file created successfully: submission.csv")

# Save feature importances if possible
if best_model in ["Random Forest", "Gradient Boosting", "AdaBoost"]:
    try:
        # Get the feature importances
        if best_model == "Ensemble":
            # For ensemble, use the first model's feature importances if available
            for estimator_name, estimator in voting_ensemble.named_estimators_.items():
                if hasattr(estimator.named_steps['model'], 'feature_importances_'):
                    model = estimator.named_steps['model']
                    break
        else:
            model = results[best_model]['model'].named_steps['model']
        
        importances = model.feature_importances_
        feature_names = X.columns
        
        # Create dataframe of feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Save to CSV
        feature_importance_df.to_csv('feature_importances.csv', index=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title(f'Top 20 Feature Importances - {best_model}')
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        plt.close()
        
        print("Feature importances saved to feature_importances.csv and feature_importances.png")
    except:
        print("Could not extract feature importances")

print("\nModel training and prediction completed successfully!") 




