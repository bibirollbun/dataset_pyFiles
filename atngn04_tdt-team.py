import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score  # Use TimeSeriesSplit for time series CV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------
# Custom Transformer for Feature Engineering (with Lag & Rolling Statistics)
# -------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Create a copy to avoid modifying original data
        X = X.copy()
        # Convert 'month' column to datetime format
        X['month_dt'] = pd.to_datetime(X['month'], format='%Y-%m')
        # Extract year and month number
        X['year'] = X['month_dt'].dt.year
        X['month_num'] = X['month_dt'].dt.month
        
        # -------------------------
        # Existing Features (including Method 1)
        # -------------------------
        # Cyclical Encoding of the month:
        X['month_sin'] = np.sin(2 * np.pi * X['month_num'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month_num'] / 12)
        
        # Create a continuous time index feature (months since January 1992)
        base_date = pd.Timestamp("1992-01-01")
        X['time_index'] = ((X['month_dt'].dt.year - base_date.year) * 12 +
                           (X['month_dt'].dt.month - base_date.month))
        
        # Create a season feature:
        # Define: Winter = Dec, Jan, Feb; Spring = Mar, Apr, May; Summer = Jun, Jul, Aug; Fall = Sep, Oct, Nov.
        X['season'] = X['month_num'].apply(lambda x: 
                                           'winter' if x in [12, 1, 2] 
                                           else ('spring' if x in [3, 4, 5] 
                                                 else ('summer' if x in [6, 7, 8] else 'fall')))
        # Encode season numerically
        season_mapping = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        X['season_encoded'] = X['season'].map(season_mapping)
        
        # Create temperature range feature
        X['temp_range'] = X['TMAX'] - X['TMIN']
        
        # Create water area ratio: Water Area / Total Area (avoid division by zero)
        X['water_area_ratio'] = X['Water Area (sq mi)'] / (X['Total Area (sq mi)'] + 1e-8)
        
        # Create land area ratio: Land Area / Total Area
        X['land_area_ratio'] = X['Land Area (sq mi)'] / (X['Total Area (sq mi)'] + 1e-8)
        
        # Create an interaction feature between urbanization rate and land area
        X['urban_land_interaction'] = X['Urbanization Rate (%)'] * X['Land Area (sq mi)']
        
        # -------------------------
        # Existing Rolling Mean Features for Weather
        # -------------------------
        # Sort by STATE and month_dt for correct rolling computations
        X.sort_values(by=['STATE', 'month_dt'], inplace=True)
        X['PRCP_roll_mean'] = X.groupby('STATE')['PRCP'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        X['EVAP_roll_mean'] = X.groupby('STATE')['EVAP'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        X['TMIN_roll_mean'] = X.groupby('STATE')['TMIN'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        X['TMAX_roll_mean'] = X.groupby('STATE')['TMAX'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        
        # -------------------------
        # Method 2: Lag & Rolling Statistics for Weather Variables
        # -------------------------
        for var in ['PRCP', 'EVAP', 'TMIN', 'TMAX']:
            # Lag feature (previous month)
            X[f'{var}_lag1'] = X.groupby('STATE')[var].shift(1)
            # Rolling standard deviation over a 3-month window
            X[f'{var}_roll_std'] = X.groupby('STATE')[var].transform(lambda x: x.rolling(window=3, min_periods=1).std())
            # Difference between current value and previous month (lag difference)
            X[f'{var}_diff'] = X[var] - X[f'{var}_lag1']
        
        # Drop temporary columns if desired
        X.drop(columns=['month_dt', 'season'], inplace=True)
        
        return X

# -------------------------
# Load datasets
# -------------------------
merged_state_data = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/merged_state_data.csv")
weather_data = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/weather_monthly_state_aggregates.csv")
wildfire_data = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/wildfire_sizes_before_2010.csv")
submission_template = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/zero_submission.csv")

# Rename columns for consistency
weather_data.rename(columns={'State': 'STATE', 'year_month': 'month'}, inplace=True)
merged_state_data.rename(columns={'State': 'STATE'}, inplace=True)

# Merge wildfire data with weather and state data (for training)
train_data = pd.merge(wildfire_data, weather_data, on=['STATE', 'month'], how='left')
train_data = pd.merge(train_data, merged_state_data, on='STATE', how='left')

# Prepare test data (2011-2015)
test_data = pd.merge(submission_template.drop(columns=['total_fire_size']), weather_data, on=['STATE', 'month'], how='left')
test_data = pd.merge(test_data, merged_state_data, on='STATE', how='left')

# -------------------------
# Apply Feature Engineering
# -------------------------
fe = FeatureEngineer()
train_data = fe.transform(train_data)
test_data = fe.transform(test_data)

# -------------------------
# Compute Historical Wildfire Activity Feature
# -------------------------
# Calculate average wildfire size per STATE and month (using historical wildfire data)
wildfire_data_copy = wildfire_data.copy()
wildfire_data_copy['month_dt'] = pd.to_datetime(wildfire_data_copy['month'], format='%Y-%m')
wildfire_data_copy['month_num'] = wildfire_data_copy['month_dt'].dt.month
fire_avg = wildfire_data_copy.groupby(['STATE', 'month_num'])['total_fire_size'].mean().reset_index()
fire_avg.rename(columns={'total_fire_size': 'historical_fire_avg'}, inplace=True)

# Merge this feature into both train and test data using STATE and month_num
train_data = train_data.merge(fire_avg, on=['STATE', 'month_num'], how='left')
test_data = test_data.merge(fire_avg, on=['STATE', 'month_num'], how='left')

# -------------------------
# Ensure correct data types
# -------------------------
train_data['Urbanization Rate (%)'] = pd.to_numeric(train_data['Urbanization Rate (%)'], errors='coerce')
test_data['Urbanization Rate (%)'] = pd.to_numeric(test_data['Urbanization Rate (%)'], errors='coerce')

# -------------------------
# Define features (including new lag, rolling std, diff, and historical wildfire average features)
# -------------------------
features = [
    'PRCP', 'EVAP', 'TMIN', 'TMAX', 'mean_elevation', 
    'Land Area (sq mi)', 'Water Area (sq mi)', 'Total Area (sq mi)', 
    'Urbanization Rate (%)', 'year', 'month_num', 'month_sin', 'month_cos', 'time_index',
    'season_encoded', 'temp_range', 'water_area_ratio', 'land_area_ratio', 'urban_land_interaction',
    'PRCP_roll_mean', 'EVAP_roll_mean', 'TMIN_roll_mean', 'TMAX_roll_mean',
    'PRCP_lag1', 'EVAP_lag1', 'TMIN_lag1', 'TMAX_lag1',
    'PRCP_roll_std', 'EVAP_roll_std', 'TMIN_roll_std', 'TMAX_roll_std',
    'PRCP_diff', 'EVAP_diff', 'TMIN_diff', 'TMAX_diff',
    'historical_fire_avg'
]

# -------------------------
# Sort training data by time (important for time series cross validation)
# -------------------------
train_data = train_data.sort_values('month')

# -------------------------
# Prepare Target and Feature Matrices
# -------------------------
y = np.log1p(train_data['total_fire_size'])
X = train_data[features]

# -------------------------
# Define base estimators for stacking including a more sophisticated model (LGBM)
# -------------------------
estimators = [
    ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, gamma=1,
                         min_child_weight=3, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('ridge', Ridge(alpha=1.0)),
    ('lgbm', LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42))
]

stack_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

# -------------------------
# Build a pipeline that imputes missing values then applies stacking
# -------------------------
stack_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('stack', stack_regressor)
])

# -------------------------
# Define Custom Scoring Metric
# -------------------------
def wildfire_metric(y_true, y_pred):
    # Convert predictions back to the original scale
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    eps = 1e-15
    ratio_log = np.abs(np.log((y_pred_orig + eps) / (y_true_orig + eps)))
    return np.mean(np.minimum(ratio_log, 10))

custom_scorer = make_scorer(wildfire_metric, greater_is_better=False)

# -------------------------
# Evaluate using Time Series Cross-Validation with the Custom Scorer
# -------------------------
tscv = TimeSeriesSplit(n_splits=5)
custom_scores = cross_val_score(stack_pipeline, X, y, scoring=custom_scorer, cv=tscv, n_jobs=-1)
print("Stacking model cross-validated custom scores:", -custom_scores)
print("Average custom score:", -custom_scores.mean())

# -------------------------
# Fit the Stacking Pipeline on the Entire Training Data
# -------------------------
stack_pipeline.fit(X, y)

# -------------------------
# Prepare and Predict on Test Data
# -------------------------
X_test = test_data[features]
test_data['total_fire_size'] = np.expm1(stack_pipeline.predict(X_test))

# -------------------------
# Prepare the Submission File
# -------------------------
test_data['ID'] = range(len(test_data))
submission = test_data[['ID', 'STATE', 'month', 'total_fire_size']]
submission_file_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_file_path, index=False)
print(f'Submission file saved as: {submission_file_path}')


