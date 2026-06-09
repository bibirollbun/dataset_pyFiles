# Step 1: Data Loading and Initial Exploration for Road Accident Risk Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')

print("="*60)
print("ROAD ACCIDENT RISK PREDICTION - INITIAL EXPLORATION")
print("="*60)

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print("\n" + "="*60)
print("DATASET SHAPES")
print("="*60)
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

print("\n" + "="*60)
print("FIRST 5 ROWS OF TRAINING DATA")
print("="*60)
print(train_df.head())

print("\n" + "="*60)
print("COLUMN NAMES AND TYPES")
print("="*60)
print(train_df.dtypes)

print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print(train_df.describe())

print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print("Training set missing values:")
if missing_train.sum() == 0:
    print("No missing values! ✓")
else:
    print(missing_train[missing_train > 0])

print("\nTest set missing values:")
if missing_test.sum() == 0:
    print("No missing values! ✓")
else:
    print(missing_test[missing_test > 0])

print("\n" + "="*60)
print("TARGET VARIABLE ANALYSIS (accident_risk)")
print("="*60)
target = 'accident_risk'
print(f"Mean: {train_df[target].mean():.4f}")
print(f"Median: {train_df[target].median():.4f}")
print(f"Std: {train_df[target].std():.4f}")
print(f"Min: {train_df[target].min():.4f}")
print(f"Max: {train_df[target].max():.4f}")
print(f"Skewness: {train_df[target].skew():.4f}")

# Visualize target distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
axes[0].hist(train_df[target], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Accident Risk')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Accident Risk')
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(train_df[target], vert=True)
axes[1].set_ylabel('Accident Risk')
axes[1].set_title('Boxplot of Accident Risk')
axes[1].grid(True, alpha=0.3)

# QQ plot to check normality
from scipy import stats
stats.probplot(train_df[target], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("CATEGORICAL VS NUMERICAL FEATURES")
print("="*60)

# Identify feature types
feature_cols = [col for col in train_df.columns if col not in ['id', target]]
numerical_cols = []
categorical_cols = []

for col in feature_cols:
    if train_df[col].dtype in ['int64', 'float64']:
        n_unique = train_df[col].nunique()
        if n_unique < 20:  # Likely categorical if few unique values
            categorical_cols.append(col)
            print(f"{col}: {n_unique} unique values (treated as categorical)")
        else:
            numerical_cols.append(col)
    else:
        categorical_cols.append(col)

print(f"\nNumerical features: {len(numerical_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

if len(numerical_cols) > 0:
    print("\nNumerical columns:", numerical_cols[:10], "..." if len(numerical_cols) > 10 else "")
if len(categorical_cols) > 0:
    print("Categorical columns:", categorical_cols[:10], "..." if len(categorical_cols) > 10 else "")

print("\n" + "="*60)
print("CHECK FOR DUPLICATE ROWS")
print("="*60)
duplicates_train = train_df.duplicated().sum()
duplicates_test = test_df.duplicated().sum()
print(f"Duplicate rows in training set: {duplicates_train}")
print(f"Duplicate rows in test set: {duplicates_test}")

print("\n" + "="*60)
print("UNIQUE VALUES IN EACH COLUMN")
print("="*60)
for col in train_df.columns[:10]:  # Show first 10 columns
    n_unique = train_df[col].nunique()
    print(f"{col}: {n_unique} unique values")
    if n_unique < 10:  # If few values, show them
        print(f"  Values: {sorted(train_df[col].unique())}")

print("\n" + "="*60)
print("SAMPLE SUBMISSION CHECK")
print("="*60)
print("Sample submission head:")
print(sample_submission.head())
print(f"\nExpected columns: {list(sample_submission.columns)}")
print(f"Number of predictions needed: {len(sample_submission)}")


# Step 2: Feature Engineering and Encoding for Accident Risk Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# Store original shapes
print("="*60)
print("FEATURE ENGINEERING & ENCODING")
print("="*60)
print(f"Original train shape: {train_df.shape}")
print(f"Original test shape: {test_df.shape}")

# Combine for consistent processing
train_df['is_train'] = 1
test_df['is_train'] = 0
combined_df = pd.concat([train_df, test_df], ignore_index=True)

print("\n" + "="*60)
print("1. CREATING INTERACTION FEATURES")
print("="*60)

# High-risk weather conditions
combined_df['bad_weather'] = (combined_df['weather'].isin(['rainy', 'foggy'])).astype(int)
combined_df['night_driving'] = (combined_df['lighting'] == 'night').astype(int)
combined_df['poor_visibility'] = ((combined_df['lighting'].isin(['night', 'dim'])) | 
                                   (combined_df['weather'] == 'foggy')).astype(int)

# Dangerous combinations
combined_df['night_rain'] = ((combined_df['lighting'] == 'night') & 
                              (combined_df['weather'] == 'rainy')).astype(int)
combined_df['highway_night'] = ((combined_df['road_type'] == 'highway') & 
                                 (combined_df['lighting'] == 'night')).astype(int)
combined_df['rural_night'] = ((combined_df['road_type'] == 'rural') & 
                               (combined_df['lighting'] == 'night')).astype(int)
combined_df['foggy_curved'] = ((combined_df['weather'] == 'foggy') & 
                                (combined_df['curvature'] > 0.5)).astype(int)

# Speed risk factors
combined_df['high_speed'] = (combined_df['speed_limit'] >= 60).astype(int)
combined_df['high_speed_rain'] = ((combined_df['speed_limit'] >= 60) & 
                                   (combined_df['weather'] == 'rainy')).astype(int)
combined_df['high_speed_night'] = ((combined_df['speed_limit'] >= 60) & 
                                    (combined_df['lighting'] == 'night')).astype(int)

# School/holiday risk
combined_df['school_morning'] = ((combined_df['school_season'] == True) & 
                                  (combined_df['time_of_day'] == 'morning')).astype(int)
combined_df['school_afternoon'] = ((combined_df['school_season'] == True) & 
                                    (combined_df['time_of_day'] == 'afternoon')).astype(int)
combined_df['holiday_night'] = ((combined_df['holiday'] == True) & 
                                 (combined_df['lighting'] == 'night')).astype(int)

# Road complexity
combined_df['complex_road'] = ((combined_df['num_lanes'] >= 3) & 
                                (combined_df['curvature'] > 0.5)).astype(int)
combined_df['single_lane_curved'] = ((combined_df['num_lanes'] == 1) & 
                                      (combined_df['curvature'] > 0.5)).astype(int)

# Safety features
combined_df['no_signs_night'] = ((combined_df['road_signs_present'] == False) & 
                                  (combined_df['lighting'] == 'night')).astype(int)
combined_df['private_road_night'] = ((combined_df['public_road'] == False) & 
                                      (combined_df['lighting'] == 'night')).astype(int)

print("Created 17 interaction features")

print("\n" + "="*60)
print("2. CREATING AGGREGATE RISK SCORES")
print("="*60)

# Weather risk score
weather_risk = {'clear': 0, 'foggy': 2, 'rainy': 1}
combined_df['weather_risk'] = combined_df['weather'].map(weather_risk)

# Lighting risk score
lighting_risk = {'daylight': 0, 'dim': 1, 'night': 2}
combined_df['lighting_risk'] = combined_df['lighting'].map(lighting_risk)

# Time risk score
time_risk = {'morning': 1, 'afternoon': 0, 'evening': 2}
combined_df['time_risk'] = combined_df['time_of_day'].map(time_risk)

# Road type risk
road_risk = {'urban': 1, 'rural': 2, 'highway': 3}
combined_df['road_risk'] = combined_df['road_type'].map(road_risk)

# Combined risk scores
combined_df['visibility_score'] = combined_df['weather_risk'] + combined_df['lighting_risk']
combined_df['total_risk_score'] = (combined_df['weather_risk'] + 
                                    combined_df['lighting_risk'] + 
                                    combined_df['time_risk'] +
                                    combined_df['road_risk'])

print("Created 6 risk score features")

print("\n" + "="*60)
print("3. NUMERICAL TRANSFORMATIONS")
print("="*60)

# Curvature features
combined_df['curvature_squared'] = combined_df['curvature'] ** 2
# Handle potential NaN values in binning
combined_df['curvature_bins'] = pd.cut(combined_df['curvature'], 
                                        bins=[0, 0.25, 0.5, 0.75, 1.0], 
                                        labels=[0, 1, 2, 3])
# Convert to int, filling NaN with 0
combined_df['curvature_bins'] = combined_df['curvature_bins'].cat.codes
combined_df['high_curvature'] = (combined_df['curvature'] > 0.7).astype(int)

# Speed features
combined_df['speed_lanes_ratio'] = combined_df['speed_limit'] / combined_df['num_lanes']
combined_df['speed_curvature_risk'] = combined_df['speed_limit'] * combined_df['curvature']

print("Created 5 numerical features")

print("\n" + "="*60)
print("4. TARGET ENCODING FOR HIGH-CARDINALITY FEATURES")
print("="*60)

# Only do target encoding on training data
train_mask = combined_df['is_train'] == 1

# Function for target encoding with smoothing
def target_encode(df, col, target='accident_risk', smoothing=10):
    train_data = df[df['is_train'] == 1].copy()
    
    # Calculate global mean
    global_mean = train_data[target].mean()
    
    # Calculate category statistics
    agg = train_data.groupby(col)[target].agg(['count', 'mean'])
    
    # Smoothing
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
    
    # Create mapping
    encoding_map = smooth.to_dict()
    
    # Apply to full dataset
    df[f'{col}_target_enc'] = df[col].map(encoding_map).fillna(global_mean)
    
    return df

# Apply target encoding to categorical features
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']

for col in categorical_features:
    combined_df = target_encode(combined_df, col)
    print(f"Target encoded: {col}")

print("\n" + "="*60)
print("5. ONE-HOT ENCODING FOR LOW-CARDINALITY FEATURES")
print("="*60)

# One-hot encode main categorical features
onehot_features = ['road_type', 'weather', 'lighting', 'time_of_day']

for col in onehot_features:
    dummies = pd.get_dummies(combined_df[col], prefix=col, drop_first=False)
    combined_df = pd.concat([combined_df, dummies], axis=1)
    print(f"One-hot encoded {col}: {dummies.shape[1]} new columns")

print("\n" + "="*60)
print("6. BINARY ENCODING OPTIMIZATION")
print("="*60)

# Convert boolean to int
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
    combined_df[col] = combined_df[col].astype(int)

print(f"Converted {len(bool_cols)} boolean columns to integers")

print("\n" + "="*60)
print("7. FEATURE STATISTICS")
print("="*60)

# Split back to train and test
train_fe = combined_df[combined_df['is_train'] == 1].drop('is_train', axis=1)
test_fe = combined_df[combined_df['is_train'] == 0].drop('is_train', axis=1)

# Remove the 'is_train' column from test (it doesn't have accident_risk)
if 'accident_risk' in test_fe.columns:
    test_fe = test_fe.drop('accident_risk', axis=1)

print(f"Final train shape: {train_fe.shape}")
print(f"Final test shape: {test_fe.shape}")
print(f"Total features created: {train_fe.shape[1] - train_df.shape[1] + 1}")

# Show feature groups
print("\n" + "="*60)
print("FEATURE GROUPS SUMMARY")
print("="*60)
print(f"Original features: 13")
print(f"Interaction features: 17")
print(f"Risk scores: 6")
print(f"Numerical transformations: 5")
print(f"Target encoded features: {len(categorical_features)}")
print(f"One-hot encoded features: {sum([col.startswith(prefix) for col in combined_df.columns for prefix in onehot_features])}")
print(f"Total features: {train_fe.shape[1] - 2}")  # Excluding id and target

# Save engineered datasets
train_fe.to_csv('train_engineered.csv', index=False)
test_fe.to_csv('test_engineered.csv', index=False)
print("\n✓ Saved engineered datasets")


# Step 3: Fast Feature Selection (Optimized version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Load engineered data
train_fe = pd.read_csv('train_engineered.csv')
test_fe = pd.read_csv('test_engineered.csv')

# Prepare features and target
target = 'accident_risk'
feature_cols = [col for col in train_fe.columns if col not in ['id', target]]

# Drop original categorical columns that have been encoded
original_cats = ['road_type', 'lighting', 'weather', 'time_of_day']
feature_cols = [col for col in feature_cols if col not in original_cats]

X = train_fe[feature_cols]
y = train_fe[target]

print("="*60)
print("FAST FEATURE SELECTION")
print("="*60)
print(f"Analyzing {len(feature_cols)} features")

# We already have correlations from your output
print("\n" + "="*60)
print("1. TOP FEATURES BY CORRELATION")
print("="*60)

correlations = train_fe[feature_cols + [target]].corr()[target].drop(target).sort_values(ascending=False)

# Based on your output, these are the top correlation features
top_corr_features = [
    'speed_curvature_risk', 'curvature', 'curvature_bins', 'curvature_squared',
    'high_speed_night', 'high_speed', 'lighting_target_enc', 'lighting_night',
    'night_driving', 'speed_limit', 'visibility_score', 'high_curvature',
    'lighting_risk', 'foggy_curved', 'holiday_night', 'no_signs_night',
    'complex_road', 'total_risk_score', 'private_road_night', 'high_speed_rain'
]

print("Top 20 correlation-based features identified")

# 2. Skip MI, go straight to Random Forest (much faster)
print("\n" + "="*60)
print("2. RANDOM FOREST FEATURE IMPORTANCE (FAST)")
print("="*60)

# Use smaller sample for faster RF
sample_size = min(50000, len(X))
sample_indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X.iloc[sample_indices]
y_sample = y.iloc[sample_indices]

# Quick RF with fewer trees
rf = RandomForestRegressor(
    n_estimators=50,  # Reduced for speed
    max_depth=8,      # Reduced for speed
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

print(f"Training RF on {sample_size} samples...")
rf.fit(X_sample, y_sample)

rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features by Random Forest:")
print(rf_importance.head(20))

# 3. COMBINE RANKINGS (without MI)
print("\n" + "="*60)
print("3. COMBINED FEATURE RANKING")
print("="*60)

# Create combined score using only correlation and RF
feature_scores = pd.DataFrame({'feature': feature_cols})
feature_scores['corr_score'] = correlations.abs()[feature_cols].values
feature_scores['rf_score'] = rf_importance.set_index('feature')['importance'][feature_cols].values

# Normalize scores
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_scores['corr_norm'] = scaler.fit_transform(feature_scores[['corr_score']])
feature_scores['rf_norm'] = scaler.fit_transform(feature_scores[['rf_score']])

# Combined score (weighted average)
feature_scores['combined_score'] = (
    0.5 * feature_scores['corr_norm'] + 
    0.5 * feature_scores['rf_norm']
)

feature_scores = feature_scores.sort_values('combined_score', ascending=False)

print("\nTop 30 features by combined score:")
top_features_list = feature_scores.head(30)['feature'].tolist()
for i, (idx, row) in enumerate(feature_scores.head(30).iterrows(), 1):
    print(f"{i:2}. {row['feature']:30} Score: {row['combined_score']:.3f}")

# 4. SELECT FINAL FEATURES
print("\n" + "="*60)
print("4. FINAL FEATURE SELECTION")
print("="*60)

# Select features with good scores
threshold = 0.15  # Adjust this threshold as needed
selected_features = feature_scores[feature_scores['combined_score'] > threshold]['feature'].tolist()

# Ensure we have at least 25 features
if len(selected_features) < 25:
    selected_features = feature_scores.head(25)['feature'].tolist()

# Cap at 40 features to avoid overfitting
if len(selected_features) > 40:
    selected_features = selected_features[:40]

print(f"\nSelected {len(selected_features)} features")
print("Selected features:", selected_features[:10], "...")

# 5. CHECK FOR REDUNDANCY
print("\n" + "="*60)
print("5. REMOVING REDUNDANT FEATURES")
print("="*60)

# Remove highly correlated features
feature_corr = X[selected_features].corr()
to_remove = set()

for i in range(len(feature_corr.columns)):
    for j in range(i):
        if abs(feature_corr.iloc[i, j]) > 0.95:  # Very high correlation
            # Remove the one with lower importance
            feat1 = feature_corr.columns[i]
            feat2 = feature_corr.columns[j]
            score1 = feature_scores[feature_scores['feature'] == feat1]['combined_score'].values[0]
            score2 = feature_scores[feature_scores['feature'] == feat2]['combined_score'].values[0]
            
            if score1 < score2:
                to_remove.add(feat1)
                print(f"Removing {feat1} (corr with {feat2}: {feature_corr.iloc[i, j]:.3f})")
            else:
                to_remove.add(feat2)
                print(f"Removing {feat2} (corr with {feat1}: {feature_corr.iloc[i, j]:.3f})")

# Remove redundant features
selected_features = [f for f in selected_features if f not in to_remove]
print(f"\nFinal features after removing redundancy: {len(selected_features)}")

# Save selected features
selected_features_df = pd.DataFrame({'feature': selected_features})
selected_features_df.to_csv('selected_features.csv', index=False)

# Create datasets with selected features
X_train_selected = train_fe[selected_features + ['id', target]]
X_test_selected = test_fe[selected_features + ['id']]

print(f"\nTrain shape with selected features: {X_train_selected.shape}")
print(f"Test shape with selected features: {X_test_selected.shape}")

# Save selected datasets
X_train_selected.to_csv('train_selected.csv', index=False)
X_test_selected.to_csv('test_selected.csv', index=False)

print("\n✓ Saved selected feature datasets")


# Step 4: Advanced Gradient Boosting Models

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# Load selected feature datasets
train_df = pd.read_csv('train_selected.csv')
test_df = pd.read_csv('test_selected.csv')

# Prepare data
target = 'accident_risk'
feature_cols = [col for col in train_df.columns if col not in ['id', target]]

X = train_df[feature_cols].values
y = train_df[target].values
X_test = test_df[feature_cols].values

print("="*60)
print("ADVANCED GRADIENT BOOSTING MODELS")
print("="*60)
print(f"Training shape: {X.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Target mean: {y.mean():.4f}, std: {y.std():.4f}")

# Setup cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y, cv, model_name):
    """Evaluate model using cross-validation"""
    scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if model_name == 'CatBoost':
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        elif model_name == 'XGBoost':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif model_name == 'LightGBM':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.log_evaluation(0)])
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        scores.append(rmse)
        print(f"  Fold {fold}: RMSE = {rmse:.6f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"  Mean RMSE: {mean_score:.6f} (+/- {std_score:.6f})")
    return mean_score, std_score

print("\n" + "="*60)
print("MODEL TRAINING & EVALUATION")
print("="*60)

results = {}

# 1. XGBoost
print("\n1. XGBoost")
print("-" * 40)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_score, xgb_std = evaluate_model(xgb_model, X, y, kf, 'XGBoost')
results['XGBoost'] = {'score': xgb_score, 'std': xgb_std}

# 2. LightGBM
print("\n2. LightGBM")
print("-" * 40)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 500,
    'num_leaves': 50,
    'max_depth': -1,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 20,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_score, lgb_std = evaluate_model(lgb_model, X, y, kf, 'LightGBM')
results['LightGBM'] = {'score': lgb_score, 'std': lgb_std}

# 3. CatBoost
print("\n3. CatBoost")
print("-" * 40)

cb_params = {
    'iterations': 500,
    'depth': 8,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'random_state': 42,
    'verbose': False,
    'thread_count': -1
}

cb_model = cb.CatBoostRegressor(**cb_params)
cb_score, cb_std = evaluate_model(cb_model, X, y, kf, 'CatBoost')
results['CatBoost'] = {'score': cb_score, 'std': cb_std}

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('score')
print("\nModels ranked by RMSE (lower is better):")
print(results_df)

best_model_name = results_df.index[0]
best_score = results_df.iloc[0]['score']

print(f"\nBest model: {best_model_name} with RMSE: {best_score:.6f}")

print("\n" + "="*60)
print("TRAINING FINAL MODELS ON FULL DATA")
print("="*60)

# Store predictions
predictions = {}

# Train each model on full data
print("\n1. Training XGBoost on full data...")
xgb_model.fit(X, y)
predictions['xgboost'] = xgb_model.predict(X_test)
print(f"   Mean prediction: {predictions['xgboost'].mean():.4f}")

print("\n2. Training LightGBM on full data...")
lgb_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
predictions['lightgbm'] = lgb_model.predict(X_test)
print(f"   Mean prediction: {predictions['lightgbm'].mean():.4f}")

print("\n3. Training CatBoost on full data...")
cb_model.fit(X, y, verbose=False)
predictions['catboost'] = cb_model.predict(X_test)
print(f"   Mean prediction: {predictions['catboost'].mean():.4f}")

print("\n" + "="*60)
print("CREATING ENSEMBLE PREDICTIONS")
print("="*60)

# Simple average ensemble
ensemble_avg = (predictions['xgboost'] + predictions['lightgbm'] + predictions['catboost']) / 3

# Weighted ensemble based on CV scores
total_score = sum([1/results[m]['score'] for m in results.keys()])
weights = {m: (1/results[m]['score'])/total_score for m in results.keys()}

ensemble_weighted = (
    weights['XGBoost'] * predictions['xgboost'] +
    weights['LightGBM'] * predictions['lightgbm'] +
    weights['CatBoost'] * predictions['catboost']
)

print(f"Ensemble weights based on performance:")
for model, weight in weights.items():
    print(f"  {model}: {weight:.3f}")

# Clip predictions to [0, 1] range
predictions['ensemble_avg'] = np.clip(ensemble_avg, 0, 1)
predictions['ensemble_weighted'] = np.clip(ensemble_weighted, 0, 1)

print(f"\nEnsemble average mean: {predictions['ensemble_avg'].mean():.4f}")
print(f"Ensemble weighted mean: {predictions['ensemble_weighted'].mean():.4f}")

print("\n" + "="*60)
print("CREATING SUBMISSIONS")
print("="*60)

# Create submissions for each model
submission_files = {
    'xgboost': predictions['xgboost'],
    'lightgbm': predictions['lightgbm'],
    'catboost': predictions['catboost'],
    'ensemble_avg': predictions['ensemble_avg'],
    'ensemble_weighted': predictions['ensemble_weighted']
}

for name, preds in submission_files.items():
    submission = pd.DataFrame({
        'id': test_df['id'],
        'accident_risk': np.clip(preds, 0, 1)  # Ensure [0,1] range
    })
    
    filename = f'submission_{name}.csv'
    submission.to_csv(filename, index=False)
    print(f"Created: {filename}")
    print(f"  Mean: {preds.mean():.4f}, Std: {preds.std():.4f}")
    print(f"  Min: {preds.min():.4f}, Max: {preds.max():.4f}")

# Create final submission with best approach
final_submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': predictions['ensemble_weighted']
})
final_submission.to_csv('submission.csv', index=False)

print("\n✅ Main submission saved as: submission.csv")

