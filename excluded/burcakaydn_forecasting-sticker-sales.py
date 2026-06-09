import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import lightgbm as lgb

# Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")

# Remove any rows without num_sold
train.dropna(subset=["num_sold"], inplace=True)


# 2. Feature Engineering

# Convert date to datetime
train["date"] = pd.to_datetime(train["date"], format="%Y-%m-%d")
test["date"] = pd.to_datetime(test["date"], format="%Y-%m-%d")

# Extract basic features from date
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["dayofweek"] = train["date"].dt.dayofweek

test["year"] = test["date"].dt.year
test["month"] = test["date"].dt.month
test["day"] = test["date"].dt.day
test["dayofweek"] = test["date"].dt.dayofweek

# Encode categorical features
cat_cols = ["country", "store", "product"]
encoders = {}
for c in cat_cols:
    encoders[c] = LabelEncoder()
    train[c] = encoders[c].fit_transform(train[c].astype(str))

for c in cat_cols:
    # Use the trained encoder from train on the test
    test[c] = encoders[c].transform(test[c].astype(str))

# Our training features
features = ["country", "store", "product", "year", "month", "day", "dayofweek"]

X = train[features]
y = train["num_sold"].values


# 3. Train a Simple Model (LightGBM)
tscv = TimeSeriesSplit(n_splits=3)


param_grid = {
    "learning_rate": [0.01, 0.03],
    "n_estimators": [300, 500],
    "num_leaves": [31, 63],
    "random_state": [42],
    "n_jobs": [-1],
}

lgb_estimator = lgb.LGBMRegressor()

grid_search = GridSearchCV(
    estimator=lgb_estimator,
    param_grid=param_grid,
    scoring="neg_mean_absolute_percentage_error",
    cv=tscv,
    verbose=1
)

grid_search.fit(X, y)

print("\nBest CV Score (negative MAPE):", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)

# Refit a final model using the best parameters on the entire train set
best_model = lgb.LGBMRegressor(**grid_search.best_params_)
best_model.fit(X, y)


# 4. Predict on (Held-Out) Validation or Test

y_pred_train = best_model.predict(X)
train_mape = mean_absolute_percentage_error(y, y_pred_train)
print(f"Train MAPE (using best model): {train_mape:.4f}")


# 5. Predict on Test & Create Submission

X_test = test[features]
test_preds = best_model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "num_sold": test_preds
})

# Sort by id (just as an example)
submission = submission.sort_values("id", ascending=True)

# Print or save as CSV
submission.to_csv("submission.csv", index=False)
print(submission.head(10))

