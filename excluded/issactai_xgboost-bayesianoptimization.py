import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

SEED = 1161


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print(f"train.shape = {train.shape}, test.shape = {test.shape}")

print(train.isnull().sum()) # check for null data
print(test.isnull().sum()) # check for null data

train.drop_duplicates(inplace=True) # no duplicated rows
test.drop_duplicates(inplace=True) # no duplicated rows
print(f"After drop_duplicates: train.shape = {train.shape}, test.shape = {test.shape}")

# no need to fill missing data
# train.fillna(train.mean(), inplace=True)
# test.fillna(test.mean(), inplace=True)


# Feature engineering: Add new features based on existing features
# Idea from https://www.kaggle.com/code/swandipsingha/best-eda-xgb-lgbm-cnn
import numpy as np
import pandas as pd
import scipy.stats as stats  # Importing for Box-Cox and Yeo-Johnson transformations

# Define function to categorize wind direction into sectors 
def wind_sector(direction):
    if pd.isna(direction):
        return np.nan  # Preserve missing values for later handling
    direction = float(direction)
    if direction >= 315 or direction < 45:
        return 'North'
    elif direction >= 45 and direction < 135:
        return 'East'
    elif direction >= 135 and direction < 225:
        return 'South'
    else:
        return 'West'

def perform_feature_engineering(df):
    """
    Applies feature engineering to the dataframe, creating new features for weather prediction.
    """
    
    # 1. Seasonal Features using 'day' (cyclical representation of the year)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    # 2. Lagged Features (previous day's values for key predictors)
    #    Shift by 1, then fill any remaining NaNs with 0 (or a median if desired)
    df['cloud_lag1'] = df['cloud'].shift(1).fillna(0)
    df['sunshine_lag1'] = df['sunshine'].shift(1).fillna(0)
    df['humidity_lag1'] = df['humidity'].shift(1).fillna(0)

    # 3. Rolling Statistics (3-day trends for key predictors)
    #    Use rolling(window=3, min_periods=1) so the first 1-2 rows won't be NaN. Backfill if needed.
    df['cloud_roll3_mean'] = df['cloud'].rolling(window=3, min_periods=1).mean().bfill()
    df['sunshine_roll3_mean'] = df['sunshine'].rolling(window=3, min_periods=1).mean().bfill()
    df['humidity_roll3_mean'] = df['humidity'].rolling(window=3, min_periods=1).mean().bfill()

    # 4. Interaction Features (combinations of highly correlated features)
    df['cloud_humidity'] = (df['cloud'] * df['humidity']).fillna(0)  # Replace missing with 0
    df['sunshine_cloud_ratio'] = (df['sunshine'] / (df['cloud'] + 1e-5)).fillna(0)

    # 5. Meteorological Features
    #    Compute temperature range and pressure difference
    df['temp_range'] = (df['maxtemp'] - df['mintemp']).fillna(df['maxtemp'].median())
    df['pressure_diff'] = df['pressure'].diff().fillna(0)

    # 6. Additional Time-Based Interactions with 'day'
    df['cloud_day_sin'] = (df['cloud'] * df['day_sin']).fillna(0)
    df['sunshine_day_cos'] = (df['sunshine'] * df['day_cos']).fillna(0)
    df['humidity_roll3_day_sin'] = (df['humidity_roll3_mean'] * df['day_sin']).fillna(0)

    # 7. Categorical Feature: Wind Direction
    #    Map wind direction to bins and replace missing with 'Unknown'
    df['wind_sector'] = df['winddirection'].apply(wind_sector).fillna('Unknown')
    
    # 7.1. Wind and Cloud Interaction Features (NEW)
    #    Captures how changes in wind and cloud metrics interact.
    df['change_in_direction'] = abs(df['winddirection'] - df['winddirection'].shift(1)).fillna(0)
    df['cloud_wind_interaction'] = df['cloud'] * np.log1p(df['windspeed'])
    df['wind_cloud_interaction'] = np.log1p(df['cloud']) * df['windspeed']

    # 8. Logarithmic and Transform Features for 'cloud' variable (NEW)
    df['cloud_log'] = np.log1p(df['cloud'])  # Log transformation to handle skewness
    df['cloud_sqrt'] = np.sqrt(df['cloud'])    # Square root transformation
    # Box-Cox transformation (requires strictly positive values; add 1 to avoid zero)
    df['cloud_boxcox'], lambda_bc = stats.boxcox(df['cloud'] + 1)
    # Yeo-Johnson transformation (handles negative values as well)
    df['cloud_yeojohnson'], lambda_yj = stats.yeojohnson(df['cloud'])

    # 9. Additional Meteorological Features (NEW)
    #    Combining logarithmic transformations for pressure and dewpoint, and cloud & sunshine
    df['log_pressure_dewpoint'] = np.log1p(df['pressure']) + np.log1p(df['dewpoint'])
    df['log_cloud_sunshine'] = np.log1p(df['cloud']) + np.log1p(df['sunshine'])
    df['cloudtest'] = (df['cloud'] == 88).astype(int)  # Binary flag if cloud equals 88
    df['sin_day2'] = np.sin(2 * np.pi * df['day'] / (365 * 2))  # Alternative cyclical feature (half frequency)
    df['cos_day2'] = np.cos(2 * np.pi * df['day'] / (365 * 2))
    df['wet_bulb'] = (2/3 * df['temparature'] + 1/3 * df['dewpoint'])  # Weighted average for wet bulb temperature

    # 10. Added for this notebook
    df['d_temp'] = df['temparature'].diff().fillna(0)
    df['d_humidity'] = df['humidity'].diff().fillna(0)
    df['d_pressure'] = df['pressure'].diff().fillna(0)
    df['d_cloud'] = df['cloud'].diff().fillna(0)
    
    return df

# ----------------------
# Apply Feature Engineering to Combined Train & Test Data
# ----------------------
for df in [train, test]:
    df = perform_feature_engineering(df)

# ----------------------
# List of Newly Created Features
# ----------------------
newly_created_vars = [
    # 1. Cyclical Seasonal Features
    'day_sin', 'day_cos',
    
    # 2. Lagged Features
    'cloud_lag1', 'sunshine_lag1', 'humidity_lag1',
    
    # 3. Rolling Statistics
    'cloud_roll3_mean', 'sunshine_roll3_mean', 'humidity_roll3_mean',
    
    # 4. Interaction Features
    'cloud_humidity', 'sunshine_cloud_ratio',
    
    # 5. Meteorological Features
    'temp_range', 'pressure_diff',
    
    # 6. Time-Based Interactions
    'cloud_day_sin', 'sunshine_day_cos', 'humidity_roll3_day_sin',
    
    # 7.1. Wind and Cloud Interaction Features (NEW)
    'change_in_direction', 'cloud_wind_interaction', 'wind_cloud_interaction',
    
    # 8. Logarithmic and Transform Features for 'cloud'
    'cloud_log', 'cloud_sqrt', 'cloud_boxcox', 'cloud_yeojohnson',
    
    # 9. Additional Meteorological Features
    'log_pressure_dewpoint', 'log_cloud_sunshine', 'cloudtest', 
    'sin_day2', 'cos_day2', 'wet_bulb'
]

# Columns to encode
columns_to_encode = ['wind_sector']

# Perform one-hot encoding with prefix
train_encoded_data = pd.get_dummies(train[columns_to_encode], prefix=columns_to_encode)
test_encoded_data = pd.get_dummies(test[columns_to_encode], prefix=columns_to_encode)

# Apply standard scalar to numercial columns
num_cols = train.drop(columns=['wind_sector', 'rainfall', 'id']).select_dtypes(include=['float64', 'int64']).columns # Identify numerical columns: all columns

X_train_num = train[num_cols].copy()
X_test_num = test[num_cols].copy()

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)  # Fit and transform train data
X_test_num_scaled = scaler.transform(X_test_num) # Transform test (no fitting)
X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=num_cols, index=train.index)
X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=num_cols, index=test.index)

train_scaled = pd.concat([X_train_num_scaled, train_encoded_data, train["rainfall"]], axis=1)
test_scaled = pd.concat([X_test_num_scaled, test_encoded_data], axis=1)

# Train a Random Forest model
X = train_scaled.drop(columns=["rainfall"]).copy()
y = train_scaled["rainfall"].copy()
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced")
rf.fit(X, y)

# Get feature importances 
feature_importances = rf.feature_importances_
important_features = np.argsort(feature_importances)[::-1][:20]

# Get selected feature names and importance scores
selected_features = X.columns[important_features]
print(f"Top {len(selected_features)} important features from Random Forest:")
print(selected_features)

# Prepare feature matrix X and target variable y
X = train_scaled.drop(columns=["rainfall"]).copy()
y = train_scaled["rainfall"].copy()

# Compute correlation matrix
corr_matrix = X[selected_features].corr().abs()

# Create a mask to filter highly correlated features 
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_correlation = [column for column in upper.columns if any(upper[column].fillna(0) > 0.80)]

# Remove highly correlated features
final_features = [f for f in selected_features if f not in high_correlation]

# Display final selected features
print(f"Final Selected Features After Correlation Filtering: {final_features}")


# utility functions 
def get_train_val_test_data(train_idx, val_idx, feature_cols):
    x_train = train_scaled.loc[train_idx, feature_cols]
    y_train = train_scaled.loc[train_idx, 'rainfall']
    x_val = train_scaled.loc[val_idx, feature_cols]
    y_val = train_scaled.loc[val_idx, 'rainfall']
    return x_train, y_train, x_val, y_val, test_scaled[feature_cols]

def train_and_evaluate_XGBClassifier(x_train, y_train, x_val, y_val, params, x_test=None, verbose=0):
    model = XGBClassifier(**params)
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=verbose
    )
    
    train_pred = model.predict_proba(x_train)[:, 1]
    val_pred = model.predict_proba(x_val)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)

    test_pred = 0.0
    if x_test is not None:
        test_pred = model.predict_proba(x_test)[:, 1]

    return train_pred, train_auc, val_pred, val_auc, test_pred


# finding optimal parameters of xgboost by BayesianOptimization
def xgb_cv(max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda, seed=42):
    
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'n_estimators': 10000,
        'eval_metric': 'auc',
        'early_stopping_rounds': 80
    }
    
    cv_scores = []
    train_auc_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for train_idx, val_idx in kf.split(train_scaled):
        x_train, y_train, x_val, y_val, _ = get_train_val_test_data(train_idx, val_idx, final_features)
        _, train_auc, _, val_auc, _ = train_and_evaluate_XGBClassifier(x_train, y_train, x_val, y_val, params)
        
        cv_scores.append(val_auc)
        train_auc_scores.append(train_auc)
    
    mean_train_auc = np.mean(train_auc_scores)
    mean_val_auc = np.mean(cv_scores)
    overfit_gap = mean_train_auc - mean_val_auc
    print(f"Train AUC = {mean_train_auc:.3f}, Val AUC = {mean_val_auc:.3f}, Overfitting = {overfit_gap:.3f}")
    
    return mean_val_auc

pbounds = {
    'max_depth': (3, 10),           # Converted to int in xgb_cv
    'learning_rate': (0.01, 0.3),   # Continuous range
    'subsample': (0.5, 1.0),        # Continuous range
    'colsample_bytree': (0.5, 1.0), # Continuous range
    'gamma': (0, 5),                # Continuous range
    'reg_alpha': (0, 1),            # Continuous range
    'reg_lambda': (0, 2)            # Continuous range
}

# run BayesianOptimization
optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=pbounds,
    random_state=SEED,
    verbose=2
)

optimizer.maximize(init_points=100, n_iter=300)

# get optimal parameters
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
print('Best parameters:', best_params)


# execute k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
oof_xgb = np.zeros(len(train_scaled))
pred_xgb = np.zeros(len(test_scaled))

fold_train_auc = []
fold_val_auc = []

for i, (train_idx, val_idx) in enumerate(kf.split(train_scaled)):
    print("#"*25)
    print(f"### Fold {i+1}")

    x_train, y_train, x_val, y_val, x_test = get_train_val_test_data(train_idx, val_idx, final_features)
    _, train_auc, val_pred, val_auc, test_pred = train_and_evaluate_XGBClassifier(x_train, y_train, x_val, y_val, best_params, x_test, verbose=2)

    oof_xgb[val_idx] = val_pred
    fold_train_auc.append(train_auc)
    fold_val_auc.append(val_auc)
    pred_xgb += test_pred

pred_xgb /= k
sample = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sample.rainfall = pred_xgb
sample.to_csv("submission.csv", index=False)
print(sample.head())

