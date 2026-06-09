import pandas as pd

# Load training data
train_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")

# Display basic info
print(train_df.info())
print(train_df.describe())
print(train_df.head())


import re

def split_plate(data):
    data = data.copy()
    data['unique_plate'] = data['plate'].str.extract(r'^([A-Z0-9]+[A-Z])')[0]
    data['region_code'] = data['plate'].str.extract(r'(\d+)$')[0]
    return data

train_df = split_plate(train_df)
train_df.head()


train_df['date'] = pd.to_datetime(train_df['date'])
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df.head()


train_df['plate_length'] = train_df['unique_plate'].str.len()
train_df['num_vowels'] = train_df['unique_plate'].str.count(r'[AEIOUY]')
train_df['num_digits'] = train_df['unique_plate'].str.count(r'\d')
train_df.head()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_df.unique_plate = le.fit_transform(train_df.unique_plate)


train_df_final = train_df.drop(['id', 'date', 'plate', 'month', 'day', 'plate_length'], axis=1)
train_df_final.head()


import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)  # First quartile
    Q3 = data[column].quantile(0.75)  # Third quartile
    IQR = Q3 - Q1  # Interquartile Range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Remove outliers from the 'price' column
df_cleaned = remove_outliers_iqr(train_df_final, 'price')

# Verify the effect
print("Original data size:", len(train_df_final))
print("Cleaned data size:", len(df_cleaned))


# Convert 'region_code' from object to int64
df_cleaned['region_code'] = df_cleaned['region_code'].astype('int64')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df_cleaned.drop('price', axis=1)
y = df_cleaned['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Top-performing models to evaluate
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
}

# Define parameter grids for grid search
param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },
}

# Use a parallel backend for GridSearchCV
results = {}
with joblib.parallel_backend("loky"):
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        param_grid = param_grids.get(model_name, {})
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        # Evaluate the best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[model_name] = {
            "Best Parameters": grid_search.best_params_,
            "RMSE": rmse,
            "R² Score": r2,
        }

        print(f"{model_name} Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"RMSE: {rmse}")
        print(f"R² Score: {r2}")
        print()

# Compare results for the top-performing models
print("Summary of Results:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print()


# Load test data
test_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")

# Preprocess test data
test_df = split_plate(test_df)
test_df['date'] = pd.to_datetime(test_df['date'])
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day

# Engineer features
test_df['plate_length'] = test_df['unique_plate'].str.len()
test_df['num_vowels'] = test_df['unique_plate'].str.count(r'[AEIOUY]')
test_df['num_digits'] = test_df['unique_plate'].str.count(r'\d')

test_df['region_code'] = test_df['region_code'].astype('int64')

test_df['unique_plate'] = le.fit_transform(test_df['unique_plate'])

test_df1 = test_df.drop(['id', 'date', 'plate', 'price', 'month', 'day', 'plate_length'], axis=1)


# Predict prices for test data
test_predictions = best_model.predict(test_df1)

# Save to submission file
submission = pd.DataFrame({
    "id": test_df["id"],
    "price": test_predictions
})
submission.to_csv("submission.csv", index=False)

