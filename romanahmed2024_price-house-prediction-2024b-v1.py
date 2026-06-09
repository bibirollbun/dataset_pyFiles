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


train=pd.read_csv("/kaggle/input/price-house-prediction-2024-b-posgraduate-dsub/train_set.csv")
test=pd.read_csv("/kaggle/input/price-house-prediction-2024-b-posgraduate-dsub/test_set.csv")


import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error


train.head()


# Fill missing 'BuildingArea' values using median grouped by ['Suburb', 'Type']
train["BuildingArea"] = train.groupby(["Suburb", "Type"])["BuildingArea"].transform(lambda x: x.fillna(x.median()))
test["BuildingArea"] = test.groupby(["Suburb", "Type"])["BuildingArea"].transform(lambda x: x.fillna(x.median()))

# Fill missing 'YearBuilt' values using median grouped by ['Suburb', 'Type']
train["YearBuilt"] = train.groupby(["Suburb", "Type"])["YearBuilt"].transform(lambda x: x.fillna(x.median()))
test["YearBuilt"] = test.groupby(["Suburb", "Type"])["YearBuilt"].transform(lambda x: x.fillna(x.median()))

# Fill missing 'CouncilArea' values using mode grouped by 'Suburb'
train["CouncilArea"] = train.groupby("Suburb")["CouncilArea"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
test["CouncilArea"] = test.groupby("Suburb")["CouncilArea"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Fill any remaining missing 'BuildingArea' values using median grouped by 'Rooms'
train["BuildingArea"] = train.groupby("Rooms")["BuildingArea"].transform(lambda x: x.fillna(x.median()))
test["BuildingArea"] = test.groupby("Rooms")["BuildingArea"].transform(lambda x: x.fillna(x.median()))

# Fill any remaining missing 'YearBuilt' values using median grouped by 'Suburb'
train["YearBuilt"] = train.groupby("Suburb")["YearBuilt"].transform(lambda x: x.fillna(x.median()))
test["YearBuilt"] = test.groupby("Suburb")["YearBuilt"].transform(lambda x: x.fillna(x.median()))

# Fill any remaining missing 'YearBuilt' values using overall median
train["YearBuilt"] = train["YearBuilt"].fillna(train["YearBuilt"].median())
test["YearBuilt"] = test["YearBuilt"].fillna(test["YearBuilt"].median())

# Fill missing 'Car' values using median grouped by 'Rooms'
train["Car"] = train.groupby("Rooms")["Car"].transform(lambda x: x.fillna(x.median()))
test["Car"] = test.groupby("Rooms")["Car"].transform(lambda x: x.fillna(x.median()))


train.isnull().sum(),test.isnull().sum()


## start model buiding and calculate metrics
import warnings
warnings.filterwarnings("ignore")
# === Step 1: Separate numeric and categorical features ===
train_num = train.select_dtypes(include=["int64", "float64"]).copy()
train_cat = train.select_dtypes(include=["object"]).copy()

test_num = test.select_dtypes(include=["int64", "float64"]).copy()
test_cat = test.select_dtypes(include=["object"]).copy()

# === Step 2: Drop unwanted columns ===
drop_cols = ["index", "Price", "Date", "Address", "SellerG"]
train_num = train_num.drop(columns=drop_cols, errors="ignore")
test_num = test_num.drop(columns=["index", "Date", "Address", "SellerG"], errors="ignore")

# === Step 3: Log-transform highly skewed columns (optional) ===
skewed_cols = ['Landsize', 'BuildingArea']
for col in skewed_cols:
    if col in train_num.columns:
        train_num[col] = np.log1p(train_num[col])
    if col in test_num.columns:
        test_num[col] = np.log1p(test_num[col])

# === Step 4: Encode categorical features ===
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_cat_encoded = encoder.fit_transform(train_cat)
test_cat_encoded = encoder.transform(test_cat)

train_cat = pd.DataFrame(train_cat_encoded, columns=train_cat.columns)
test_cat = pd.DataFrame(test_cat_encoded, columns=test_cat.columns)

# === Step 5: Handle missing values with median imputation ===
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
train_num_imputed = imputer.fit_transform(train_num)
test_num_imputed = imputer.transform(test_num)

train_num = pd.DataFrame(train_num_imputed, columns=train_num.columns)
test_num = pd.DataFrame(test_num_imputed, columns=test_num.columns)

# === Step 6: Combine numeric and categorical features ===
X_train_final = pd.concat([train_num, train_cat], axis=1)
X_test_final = pd.concat([test_num, test_cat], axis=1)
y_train = train["Price"]

# === Step 7: Define base models for stacking ===
base_models = [
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('knn', KNeighborsRegressor())
]

# Final estimator
final_estimator = LinearRegression()

# === Step 8: Define hyperparameter grid ===
param_grid = {
    'ridge__alpha': [0.1, 1, 10, 100],
    'lasso__alpha': [0.01, 0.1, 1, 10],
    'knn__n_neighbors': [3, 5, 7, 10],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

# === Step 9: Set up stacking regressor ===
stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=final_estimator,
    n_jobs=-1
)

# === Step 10: Perform GridSearchCV for best hyperparameters ===
grid_search = GridSearchCV(
    estimator=stack_model,
    param_grid=param_grid,
    cv=9,
    n_jobs=-1,
    scoring='neg_root_mean_squared_error'
)

grid_search.fit(X_train_final, y_train)

# === Step 11: Evaluate best model ===
best_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test_final)

# If actual y_test is available:
# rmse_test = mean_squared_error(y_test, y_pred, squared=False)
# print(f"Test RMSE: {rmse_test:.4f}")

# === Print best params and CV RMSE ===
best_params = grid_search.best_params_
best_rmse_cv = -grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best CV RMSE: {best_rmse_cv:.4f}")



# === Step 8: Train the best model on full training data ===
best_model = grid_search.best_estimator_
best_model.fit(X_train_final, y_train)

# === Step 9: Predict on the test set ===
y_test_pred = best_model.predict(X_test_final)

# === Step 10: Prepare the submission file ===
submission = pd.DataFrame({
    'index': test['index'],   # make sure 'index' exists in your test DataFrame
    'Price': y_test_pred
})
# Save to CSV (include .csv extension)
submission.to_csv("submission.csv", index=False)


submission.head()




