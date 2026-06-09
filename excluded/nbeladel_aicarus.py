# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# BYE


# load the training data
wildfire_df = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/wildfire_sizes_before_2010.csv")
weather_df = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/weather_monthly_state_aggregates.csv")
state_attr_df = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/merged_state_data.csv")
# print a few datapoints
# df.head()
weather_df.keys # Unit TMIN & TMAX is degrees Celsius * 100
# state_df.head(100)
wildfire_df.rename(columns={"STATE": "State"}, inplace=True)
wildfire_df.rename(columns={"month": "year_month"}, inplace=True)


state_attr_df["State"]


df


import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


# 2. Merge data
df = weather_df.merge(wildfire_df, on=["State", "year_month"], how="inner")
df = df.merge(state_attr_df, on="State", how="left")
df["State"] = label_encoder.fit_transform(df["State"])
#df = df.dropna()
# 3. Feature engineering
df["temp_range"] = df["TMAX"] - df["TMIN"]
df["temp_avg"] = (df["TMAX"] + df["TMIN"]) / 2
df["Percentage of Federal Land"] = df["Percentage of Federal Land"].str.replace("%", "").astype(float)

# Optionally, create lag features or rolling means if you have enough data
#df = create_lag_features(df, cols=["PRCP", "temp_avg"], lags=[1,2,3])
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the data
df= pd.DataFrame(scaler.fit_transform(df), columns=['PRCP', 'EVAP', 'TMIN', 'TMAX', 'mean_elavation', 'Land Area (sq mi)', 'Water Area (sq mi)', 
                                                   'temp_range'])
# 4. Prepare target (consider log transform)
df["burned_area_log"] = np.log1p(df["total_fire_size"])
# 5. Split into train/test by time
# Example: assume df has a "year_month" that can be turned into a datetime
df["date"] = pd.to_datetime(df["year_month"], format="%Y-%m")
df = df.sort_values("date")



print(df_normalized)

train_df = df[df["date"] < "2006-01-01"]
test_df  = df[df["date"] >= "2006-01-01"]

X_train = train_df.drop(columns=["total_fire_size", "burned_area_log", "date", "year_month"])
y_train = train_df["burned_area_log"]

X_test  = test_df.drop(columns=["total_fire_size", "burned_area_log", "date", "year_month"])
y_test  = test_df["burned_area_log"]


def calculate_rolling_average(df, state, year, month):
    # Filter data for the given state and month
    state_data = df[(df['State'] == state) & (df['month'] == month)]
    
    # Define the earliest year to consider (up to 5 years back)
    earliest_year = max(year - 1, df['year'].min())  # Ensure we don't go before the earliest year in the data
    
    # Filter data for the past 5 years (or as far back as possible)
    past_data = state_data[(state_data['year'] >= earliest_year) & (state_data['year'] < year)]
    
    # Calculate the rolling average
    if not past_data.empty:
        rolling_avg = past_data['burned_area_log'].mean()
    else:
        # If no past data is available, use the current year's data (if available)
        current_year_data = state_data[state_data['year'] == year]
        if not current_year_data.empty:
            rolling_avg = 0  # Use the current year's value
        else:
            rolling_avg = 0  # Default value if no data is available at all
    
    return rolling_avg


df[['year', 'month']] = df['year_month'].str.split('-', expand=True)
# Convert 'year' and 'month' to integers
df['year'] = df['year'].astype(int)
df['month'] = df['month'].astype(int)

df['rolling_avg_wildfire_coverage'] = df.apply(
    lambda row: calculate_rolling_average(df, row['State'], row['year'], row['month']), axis=1)


df.head(5000)


# Define the split point
val_start = pd.Timestamp("2007-01-01")

# Split the data
train_df = df[df["date"] < val_start]
val_df = df[df["date"] >= val_start]

# Print the sizes of each split
print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")


# Make the train/val/test dataframes
X_train = train_df.drop(columns=["burned_area_log", 'date', 'total_fire_size', 'year_month', ])
y_train = train_df["burned_area_log"]

X_test = val_df.drop(columns=["burned_area_log", 'date', 'total_fire_size', 'year_month'])
y_test = val_df["burned_area_log"]


import catboost
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# Example imports for different models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
# Example imports for different models
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


catboost_model = CatBoostRegressor(verbose=0)
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [3, 5, 7, 9],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.7, 0.8, 1.0],
    'border_count': [32, 64, 128]
}

grid_search = GridSearchCV(estimator=catboost_model,
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',  # Use appropriate scoring for your problem
                           cv=3,  # 3-fold cross-validation
                           verbose=1,
                           n_jobs=-1)  # Use all cores

random_search = RandomizedSearchCV(estimator=catboost_model,
                                   param_distributions=param_grid,
                                   scoring='neg_mean_squared_error',
                                   n_iter=100,  # Number of random combinations to try
                                   cv=3,
                                   verbose=1,
                                   n_jobs=-1)


grid_search.fit(X_train, y_train)


model = grid_search.best_estimator_


# # 6. Train a gradient boosting model
# model = xgb.XGBRegressor(
#     n_estimators=5000,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     random_state=42
# )

# model.fit(X_train, y_train)

# # 7. Evaluate
# # y_pred_log = model.predict(X_test)
# # y_pred = np.expm1(y_pred_log)  # convert back from log space
# # y_true = np.expm1(y_test)
# # y_pred_series = pd.Series(y_pred, index=y_test.index)
# # y_true_array = y_true.to_numpy()


# --- Define a List/Dictionary of Models with Hyperparameters ---
models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    "CatBoost": CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        random_state=42,
        verbose=0  # silent mode
    ),
    "LinearRegression": LinearRegression(),
    "SVR": SVR(C=1.0, epsilon=0.2),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}



# results = {}
# for name, model in models.items():
#     print(f"Training {name}...")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     results[name] = rmse
#     print(f"{name} RMSE: {rmse:.4f}\n")

# # --- Display Summary of Results ---
# print("Summary of Model Performances:")
# for name, rmse in results.items():
#     print(f"{name}: RMSE = {rmse:.4f}")


# # 6. Train a gradient boosting model
# model = xgb.XGBRegressor(
#     n_estimators=5000,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     random_state=42
# )

# model.fit(X_train, y_train)


weather_df["date"] = pd.to_datetime(weather_df["year_month"], format="%Y-%m")
test_df = weather_df.merge(state_attr_df, on="State", how="left")
test_df["Percentage of Federal Land"] = test_df["Percentage of Federal Land"].str.replace("%", "").astype(float)
test_df[['year', 'month']] = test_df['year_month'].str.split('-', expand=True)
# Convert 'year' and 'month' to integers
test_df = test_df.drop(columns=["year_month"])

# Extract 2010 data from df_train
baseline_2010 = df[df['year'] == 2009][['State', 'month', 'burned_area_log']]

# Rename the column to 'baseline_coverage'
baseline_2010


baseline_2010 = baseline_2010.rename(columns={'burned_area_log': 'rolling_avg_wildfire_coverage'})

test_df['year'] = test_df['year'].astype(int)
test_df['month'] = test_df['month'].astype(int)

test_df


test_df["temp_range"] = test_df["TMAX"] - test_df["TMIN"]
test_df["temp_avg"] = (test_df["TMAX"] + test_df["TMIN"]) / 2


test_df = test_df[test_df['date'] >= '2010-01-01']


state_df = test_df.copy()


test_df["State"] = label_encoder.fit_transform(test_df["State"])


test_df = test_df.merge(baseline_2010, on=['State', 'month'], how='left')


column_order = X_train.columns
test_df = test_df[column_order]


test_df = test_df.fillna(0)


pred = np.expm1(model.predict(test_df))


X_train


df = pd.DataFrame()


df['month'] = state_df['date']
df['total_fire_size'] = pred
df["STATE"] = state_df['State']
df['ID'] = range(len(df))

submission = df[['ID', 'STATE', 'month', 'total_fire_size']]


submission = submission[submission['month'] <= '2015-12-01']


submission['month'] = submission['month'].astype(str).str[:-3]  # Keeps only 'YYYY-MM' part


# get the predictions for 2010
wildfire_df.rename(columns={"year_month": "month"}, inplace=True)
wildfire_df.rename(columns={"State": "STATE"}, inplace=True)

last_year = wildfire_df[wildfire_df['month'].str[:4] == '2010']

last_year.head()


# duplicate the 2010 predictions for 2011 through 2015
dfs = []
for year in range(2011, 2016):
    new_df = last_year.copy()
    new_df['month'] = new_df['month'].str.replace('2010', str(year))
    dfs.append(new_df)
t = pd.concat(dfs)


# add ID column that kaggle wants (order does not matter though, items are match by (STATE, month) pair)
t['ID'] = range(len(t))

# order columns
t = t[['ID', 'STATE', 'month', 'total_fire_size']]
t.to_csv('submission.csv', index=False)

t.head()


df2_updated = t.copy()


df2_updated.set_index(['STATE', 'month'], inplace=True)
df1_indexed = submission.set_index(['STATE', 'month'])


df2_updated.update(df1_indexed)

# Reset the index to return to the original DataFrame structure.
result_df = df2_updated.reset_index()


sub = result_df[['ID', 'STATE', 'month', 'total_fire_size']]


sub['ID'] = range(len(sub))


sub


sub.to_csv('submission.csv')

