import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


import zipfile

zip_path = "/kaggle/input/nyc-taxi-trip-duration/train.zip"
extract_to = "/kaggle/working/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Done!")


import zipfile

zip_path = "/kaggle/input/nyc-taxi-trip-duration/test.zip"
extract_to = "/kaggle/working/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Done!")


TRAIN_DATA_PATH = "/kaggle/working/train.csv"
TEST_DATA_PATH = "/kaggle/working/test.csv"


train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)


train_df.head()


test_df.head()


train_df.drop(columns = ['dropoff_datetime'], inplace = True)


train_df


def plot_column_distribution_outlier_clipped(dataframe, column_name, transform=None, clipping = False, filter = None):
    df = dataframe.copy()
    if filter is not None:
        df = filter(df)
    if clipping:
        df[column_name] = np.clip(df[column_name], df[column_name].quantile(0.01), df[column_name].quantile(0.99))
    if transform is not None:
        df[column_name] = transform(df[column_name])
    plt.figure(figsize=(6, 4))
    plt.hist(df[column_name], bins=30, density=True)
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()



train_df.info()


train_df['distance'] = ((train_df['pickup_latitude'] - train_df['dropoff_latitude'])**2 + (train_df['pickup_longitude'] - train_df['dropoff_longitude'])**2)**0.5
test_df['distance'] = ((test_df['pickup_latitude'] - test_df['dropoff_latitude'])**2 + (test_df['pickup_longitude'] - test_df['dropoff_longitude'])**2)**0.5

train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])
train_df['pickup_hour'] = (train_df['pickup_datetime']).dt.hour
test_df['pickup_hour'] = (test_df['pickup_datetime']).dt.hour
train_df['pickup_day'] = (train_df['pickup_datetime']).dt.day
test_df['pickup_day'] = (test_df['pickup_datetime']).dt.day
train_df['pickup_month'] = (train_df['pickup_datetime']).dt.month
test_df['pickup_month'] = (test_df['pickup_datetime']).dt.month
train_df['dayofweek'] = train_df['pickup_datetime'].dt.dayofweek
test_df['dayofweek'] = test_df['pickup_datetime'].dt.dayofweek
train_df['is_weekend'] = train_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
test_df['is_weekend'] = test_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

train_df['lattitude_diff'] = np.abs(train_df['pickup_latitude'] - train_df['dropoff_latitude'])
test_df['lattitude_diff'] = np.abs(test_df['pickup_latitude'] - test_df['dropoff_latitude'])
train_df['longitude_diff'] = np.abs(train_df['pickup_longitude'] - train_df['dropoff_longitude'])
test_df['longitude_diff'] = np.abs(test_df['pickup_longitude'] - test_df['dropoff_longitude'])

train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].map({'Y': 1, 'N': 0})
test_df['store_and_fwd_flag'] = test_df['store_and_fwd_flag'].map({'Y': 1, 'N': 0})


def direction_to_quadrant(df):
    # Compute angle in radians
    delta_lat = df['dropoff_latitude'] - df['pickup_latitude']
    delta_lon = df['dropoff_longitude'] - df['pickup_longitude']
    angle = np.arctan2(delta_lat, delta_lon)  # angle in radians (-pi, pi)

    # Convert angle to 0-360 degrees
    angle_deg = np.degrees(angle) % 360

    # Define 8 quadrants (0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW)
    # Each quadrant is 45 degrees
    quadrant = (np.floor((angle_deg + 22.5) / 45) % 8).astype(int)
    
    # One-hot encode
    one_hot = pd.get_dummies(quadrant, prefix='dir')
    
    return one_hot

# Apply to train and test
train_df = pd.concat([train_df, direction_to_quadrant(train_df)], axis=1)
test_df = pd.concat([test_df, direction_to_quadrant(test_df)], axis=1)

train_df[[f"dir_{i}" for i in range(8)]] = train_df[[f"dir_{i}" for i in range(8)]].astype(int)
test_df[[f"dir_{i}" for i in range(8)]] = test_df[[f"dir_{i}" for i in range(8)]].astype(int)



train_df['distance_log1p'] = np.log1p(train_df['distance'])
test_df['distance_log1p'] = np.log1p(test_df['distance'])

train_df['longitude_diff_log1p'] = np.log1p(train_df['longitude_diff'])
test_df['longitude_diff_log1p'] = np.log1p(test_df['longitude_diff'])

train_df['lattitude_diff_log1p'] = np.log1p(train_df['lattitude_diff'])
test_df['lattitude_diff_log1p'] = np.log1p(test_df['lattitude_diff'])


for col in train_df.select_dtypes(include=['float64', 'int64']).columns:
    print("Plotting for:", col)
    plot_column_distribution_outlier_clipped(train_df, col, transform=None, clipping=True, filter=None)


train_df.info()


train_df.describe()


train_df.columns.tolist()


# plot correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = train_df.drop(columns=['id', 'pickup_datetime', 'dir_0', 'dir_1', 'dir_2', 'dir_3', 'dir_4', 'dir_5', 'dir_6', 'dir_7']).corr().abs()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Basic Regressors
def check_model_performance(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    score = mean_squared_error(y_test, y_pred, squared = False)
    print(f"Root Mean Squared Error for {model_name}:", score)
    print(f"R-squared for {model_name}:", r2_score(y_test, y_pred))


from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    SGDRegressor,
    PassiveAggressiveRegressor,
    HuberRegressor,
    TheilSenRegressor,
    RANSACRegressor
)

# KNN
from sklearn.neighbors import KNeighborsRegressor

# Trees / Ensembles
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    VotingRegressor,
    StackingRegressor
)

# Support Vector Machine
from sklearn.svm import SVR

# Third-party Gradient Boosting Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


train_df.shape


# ----------------- Downsample fractions per model type -----------------
downsample_fractions = {
    'linear': 0.05,      # Linear / GLM models
    'tree': 0.002,       # Decision Tree, Theil-Sen, RANSAC
    'ensemble': 0.002,   # Random Forest, Extra Trees, Bagging
    'boosting': 0.002,   # Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost
    'svm': 0.001,        # SVR
    'knn': 0.001,        # KNN
    'voting_stacking': 0.002  # Voting / Stacking
}

# ----------------- Models list with dynamic downsample fraction -----------------
models = [
    # KNN
    ('KNN', GridSearchCV(KNeighborsRegressor(),
                          {'n_neighbors': [3, 5, 7, 11],
                           'weights': ['uniform', 'distance']}, cv=5), downsample_fractions['knn']),

    # Tree / Ensemble Models
    ('Random Forest', RandomizedSearchCV(RandomForestRegressor(),
                                         {'n_estimators': [100, 300, 500],
                                          'max_depth': [None, 10, 20],
                                          'min_samples_split': [2, 5, 10]},
                                         n_iter=10, cv=5, random_state=42), downsample_fractions['ensemble']),
    ('Gradient Boosting', GridSearchCV(GradientBoostingRegressor(),
                                       {'learning_rate': [0.01, 0.05, 0.1],
                                        'n_estimators': [100, 200, 300],
                                        'max_depth': [2, 3, 4]}, cv=5), downsample_fractions['boosting']),

    # Gradient Boosting Libraries
    ('XGBoost', RandomizedSearchCV(xgb.XGBRegressor(tree_method='hist'),
                                   {'n_estimators': [200, 400, 600],
                                    'learning_rate': [0.01, 0.05, 0.1],
                                    'max_depth': [4, 6, 8]}, n_iter=10, cv=5, random_state=42), downsample_fractions['boosting']),
]





results = []

for name, model,downsample_fraction in models:
    
    downsampled_train_df = train_df.sample(frac=downsample_fraction, random_state=42)
    
    
    X_train, X_test, y_train, y_test = train_test_split(downsampled_train_df.drop(columns=['trip_duration', 'id']), downsampled_train_df['trip_duration'], test_size=0.2, random_state=42)
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    s = StandardScaler()
    X_train_scaled = s.fit_transform(X_train.drop(columns=['pickup_datetime']))
    X_test_scaled = s.transform(X_test.drop(columns=['pickup_datetime']))
    

    # --- Fit on all training data ---
    model.fit(X_train_scaled, y_train)
    
    if name != 'Stacking':
        print(f"Best parameters for {name}: {model.best_params_}")

    # --- Evaluate on test data ---
    y_pred = model.predict(X_test_scaled)
    
    test_rmse = np.sqrt(np.mean((y_pred -(y_test))**2))
    print(f"{name} | Test RMSE: {test_rmse:.4f}")
    print("--------------------------------------------------")
    
    results.append((name, test_rmse))



estimators = [
    ('knn', KNeighborsRegressor(n_neighbors=11, weights='distance')),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=2)),
    ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, tree_method='hist'))
]

# Define stacking regressor with LinearRegression as final estimator
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5
)


X_train, X_test, y_train, y_test = train_test_split(train_df.drop(columns=['trip_duration', 'id']), train_df['trip_duration'], test_size=0.2, random_state=42)
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

s = StandardScaler()
X_train_scaled = s.fit_transform(X_train.drop(columns=['pickup_datetime']))
X_test_scaled = s.transform(X_test.drop(columns=['pickup_datetime']))


# --- Fit on all training data ---
model.fit(X_train_scaled, y_train)

# --- Evaluate on test data ---
y_pred = model.predict(X_test_scaled)

test_rmse = np.sqrt(np.mean((y_pred -(y_test))**2))
print(f"Stacking Regressor | Test RMSE: {test_rmse:.4f}")
print("--------------------------------------------------")



X_train = train_df.drop(columns=['trip_duration', 'id', 'pickup_datetime'])
y_train = np.log1p(train_df['trip_duration'])
stacking_model.fit(X_train, y_train)
X_test = test_df.drop(columns=['id', 'pickup_datetime'])
test_predictions = stacking_model.predict(X_test)
test_predictions = np.expm1(test_predictions)  # Inverse of log1p to get original scale
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'trip_duration': test_predictions
})
submission_df.to_csv('/kaggle/working/submission_stacking_model.csv', index=False)




