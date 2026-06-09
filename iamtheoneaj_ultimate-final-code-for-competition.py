import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv  # Explicit import for experimental feature
from sklearn.model_selection import HalvingRandomSearchCV


# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")



# Convert date to numerical features
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["dayofweek"] = train["date"].dt.dayofweek
test["year"] = test["date"].dt.year
test["month"] = test["date"].dt.month
test["day"] = test["date"].dt.day
test["dayofweek"] = test["date"].dt.dayofweek


# Encode categorical variables
label_encoders = {}
categorical_features = ["country", "store", "product"]
for col in categorical_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le



# Fill missing target values
train["num_sold"] = train["num_sold"].fillna(train["num_sold"].median())


# Select features and target
features = ["country", "store", "product", "year", "month", "day", "dayofweek"]
X = train[features]
y = train["num_sold"]
X_test = test[features]


# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define parameter grid for HalvingRandomSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [5, 10]
}


# Halving search for hyperparameter tuning
rf = RandomForestRegressor(random_state=42, n_jobs=4)
rf_search = HalvingRandomSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    factor=2,
    cv=3,
    scoring='neg_mean_absolute_percentage_error',
    random_state=42,
    n_jobs=4
)
rf_search.fit(X_train, y_train)


# Best model from HalvingRandomSearchCV
best_rf_model = rf_search.best_estimator_


# Validate model
y_pred_val = best_rf_model.predict(X_val)
mape_score = mean_absolute_percentage_error(y_val, y_pred_val)
print("Validation MAPE (Random Forest):", mape_score)



# Train XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.05, objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)



# Validate XGBoost model
y_pred_val_xgb = xgb_model.predict(X_val)
mape_score_xgb = mean_absolute_percentage_error(y_val, y_pred_val_xgb)
print("Validation MAPE (XGBoost):", mape_score_xgb)


# Predict on test set using best model (choose better one)
if mape_score_xgb < mape_score:
    test_predictions = xgb_model.predict(X_test)
    print("Using XGBoost for final predictions")
else:
    test_predictions = best_rf_model.predict(X_test)
    print("Using Random Forest for final predictions")

test_predictions = np.maximum(test_predictions, 0)  # Ensure no negative values



# Create submission file
submission = pd.DataFrame({"id": test["id"], "num_sold": test_predictions.astype(int)})
submission.to_csv("submission.csv", index=False)

print("Submission file saved successfully!")



print(submission.head())


