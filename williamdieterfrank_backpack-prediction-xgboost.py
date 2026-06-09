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


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
training_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


train.info()


test.info()


training_extra.info()


training_extra["Price"].describe()


train.head(5)


print("Unique brands:",train["Brand"].unique())
print("Unique materials:",train["Material"].unique())
print("Unique sizes:",train["Size"].unique())
print("Unique style:",train["Style"].unique())
print("Unique color:",train["Color"].unique())


# Fill numerical columns with median
num_cols = ['Compartments', 'Weight Capacity (kg)']

for col in num_cols:
    median_value = train[col].median()
    train[col].fillna(median_value, inplace=True)
    test[col].fillna(median_value, inplace=True)
    training_extra[col].fillna(median_value, inplace=True)

# Fill categorical columns with "Unknown"
cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

for col in cat_cols:
    train[col].fillna("Unknown", inplace=True)
    test[col].fillna("Unknown", inplace=True)
    training_extra[col].fillna("Unknown", inplace=True)



from sklearn.preprocessing import LabelEncoder

label_encoders = {}  # Store encoders for later use

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    training_extra[col] = le.transform(training_extra[col])
    label_encoders[col] = le  # Store the encoder for future use




train.info()


X_train = train.drop(columns=["Price", "id"])
y_train = train["Price"]
X_train_extra = training_extra.drop(columns=["Price", "id"])
y_train_extra = training_extra["Price"]
X_test = test.drop(columns=["id"])


'''
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

param_grid = {
    "n_estimators": [500, 1000, 1500],  # Number of trees
    "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Step size shrinkage
    "max_depth": [4, 6, 8, 10],  # Tree depth
    "subsample": [0.6, 0.8, 1.0],  # Fraction of samples per tree
    "colsample_bytree": [0.6, 0.8, 1.0],  # Fraction of features per tree
    "gamma": [0, 0.1, 0.3, 0.5],  # Minimum loss reduction required to split a node
    "min_child_weight": [1, 3, 5]  # Minimum sum of instance weight needed in a child
}
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=30,  # Number of different combinations to try
    scoring="neg_mean_absolute_error",  # Metric to optimize
    cv=3,  # 3-fold cross-validation
    verbose=2,
    n_jobs=-1  # Use all available CPU cores
)

# Run hyperparameter tuning
random_search.fit(X_train_extra, y_train_extra, eval_set=[(X_train, y_train)], early_stopping_rounds=30, verbose=True)

print("Best Parameters:", random_search.best_params_)
best_xgb_model = random_search.best_estimator_
'''


import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the optimized model
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1200,  # Increase tree count
    learning_rate=0.04,  # Slightly higher learning rate
    max_depth=6,  # Allow more splits
    subsample=0.85,  # Increase sample diversity
    colsample_bytree=0.75,  # Increase feature sampling
    gamma=0.2,  # Reduce pruning
    min_child_weight=4,  # Balance complexity
    reg_alpha=0.05,  # L1 penalty
    reg_lambda=0.2,  # Reduce L2 regularization
    random_state=42,
    early_stopping_rounds=50,
    device="cuda"
)

# Train the improved model
xgb_model.fit(X_train_extra, y_train_extra, eval_set=[(X_train, y_train)], verbose=True)



y_pred_extra = xgb_model.predict(X_train_extra)

mae = mean_absolute_error(y_train_extra, y_pred_extra)
mse = mean_squared_error(y_train_extra, y_pred_extra)
rmse = mse ** 0.5
r2 = r2_score(y_train_extra, y_pred_extra)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")



y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Check distribution of predictions
print(f"Train Predictions - Min: {np.min(y_pred_train)}, Max: {np.max(y_pred_train)}, Std: {np.std(y_pred_train)}")
print(f"Test Predictions - Min: {np.min(y_pred_test)}, Max: {np.max(y_pred_test)}, Std: {np.std(y_pred_test)}")



import matplotlib.pyplot as plt

xgb.plot_importance(xgb_model)
plt.show()



y_test_pred = xgb_model.predict(X_test)

# Create a DataFrame with test predictions
test_results = test.copy()
test_results["Price"] = y_test_pred

print(test_results.head())  # Display the first few rows



submission = pd.DataFrame({
    "id": test["id"],  
    "Price": y_test_pred  # Predicted price
})

submission.to_csv("submission.csv", index=False)


submission.describe()

