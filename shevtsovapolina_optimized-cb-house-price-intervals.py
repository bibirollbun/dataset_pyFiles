import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# Load data using Kaggle's input paths
train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')

# Prepare features and target
y = train['sale_price']
X = train.drop(columns=['sale_price', 'id'])
X_test = test.drop(columns=['id'])
test_ids = test['id']

# Process date features
def process_dates(df):
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df['sale_year'] = df['sale_date'].dt.year.astype('Int16')
    df['sale_month'] = df['sale_date'].dt.month.astype('Int8')
    df['sale_dayofweek'] = df['sale_date'].dt.dayofweek.astype('Int8')
    df['is_weekend'] = df['sale_dayofweek'].isin([5, 6]).astype('int8')
    df.drop(columns=['sale_date'], inplace=True)
    return df

X = process_dates(X)
X_test = process_dates(X_test)

# Categorical columns
cat_cols = [
    'sale_nbr', 'sale_warning', 'join_status', 'city', 'zoning', 'subdivision',
    'present_use', 'grade', 'fbsmt_grade', 'condition', 'wfnt', 'golf',
    'greenbelt', 'noise_traffic', 'view_rainier', 'view_olympics', 'view_cascades',
    'view_territorial', 'view_skyline', 'view_sound', 'view_lakewash',
    'view_lakesamm', 'view_otherwater', 'view_other', 'submarket'
]

# Optimize data types
def optimize_df(df):
    for col in df.columns:
        if col in cat_cols:
            df[col] = df[col].astype(str).fillna('Unknown')
        elif df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].astype('float32').fillna(df[col].median())
        elif df[col].dtype in ['int64', 'int32']:
            df[col] = df[col].astype('int32').fillna(df[col].median())
    return df

X = optimize_df(X)
X_test = optimize_df(X_test)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Log-transform target
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Create data pools
train_pool = Pool(X_train, y_train_log, cat_features=cat_cols)
val_pool = Pool(X_val, y_val_log, cat_features=cat_cols)
test_pool = Pool(X_test, cat_features=cat_cols)

# Train main model
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=4,
    random_seed=42,
    loss_function='RMSE',
    early_stopping_rounds=50,
    verbose=100
)
model.fit(train_pool, eval_set=val_pool)

# Predict on validation and test
y_val_pred_log = model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)
y_test_pred_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)

# Train error model for residuals
residuals_train = np.abs(y_train - np.expm1(model.predict(X_train)))
error_model = CatBoostRegressor(
    iterations=800,
    learning_rate=0.05,
    depth=5,
    random_seed=42,
    loss_function='MAE',
    verbose=0
)
error_model.fit(X_train, residuals_train, cat_features=cat_cols)
pred_std_test = error_model.predict(X_test)

# Prediction intervals using residuals
z = 1.64
pi_lower_ind = np.maximum(0, y_test_pred - z * pred_std_test)
pi_upper_ind = y_test_pred + z * pred_std_test

# Train quantile models
quantile_params = {
    'iterations': 2000,
    'depth': 6,
    'learning_rate': 0.03,
    'random_seed': 42,
    'cat_features': cat_cols,
    'early_stopping_rounds': 50,
    'verbose': 100
}

lower_model = CatBoostRegressor(loss_function='Quantile:alpha=0.05', **quantile_params)
upper_model = CatBoostRegressor(loss_function='Quantile:alpha=0.95', **quantile_params)

lower_model.fit(train_pool, eval_set=val_pool)
upper_model.fit(train_pool, eval_set=val_pool)

# Predict quantiles
pi_lower_q_log = lower_model.predict(X_test)
pi_upper_q_log = upper_model.predict(X_test)
pi_lower_q = np.expm1(pi_lower_q_log)
pi_upper_q = np.expm1(pi_upper_q_log)

# Winkler Interval Score
def winkler_score(y_true, l, u, alpha=0.1):
    score = []
    for yt, low, up in zip(y_true, l, u):
        if yt < low:
            w = (up - low) + (2 / alpha) * (low - yt)
        elif yt > up:
            w = (up - low) + (2 / alpha) * (yt - up)
        else:
            w = up - low
        score.append(w)
    return np.mean(score)

pi_lower_val = np.expm1(lower_model.predict(X_val))
pi_upper_val = np.expm1(upper_model.predict(X_val))
winkler = winkler_score(y_val.values, pi_lower_val, pi_upper_val, alpha=0.1)
print(f"Winkler Interval Score (quantile): {winkler:.2f}")

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'pi_lower': pi_lower_q,
    'pi_upper': pi_upper_q
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()

