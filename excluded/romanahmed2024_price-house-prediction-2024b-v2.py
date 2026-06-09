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



import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


# === Step 1: Separate target and drop unnecessary features ===
X = train.drop(columns=["Price", "index", "Date", "Address", "SellerG"])
y = train["Price"]
X_test = test.drop(columns=["index", "Date", "Address", "SellerG"])

# === Step 2: Identify numeric and categorical columns ===
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# === Step 3: Define numeric preprocessing pipeline ===
log_transform_cols = ['Landsize', 'BuildingArea']
def log_transform(df):
    df = df.copy()
    for col in log_transform_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df

numeric_pipeline = Pipeline([
    ('log', FunctionTransformer(log_transform, validate=False)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# === Step 4: Define categorical preprocessing pipeline ===
categorical_pipeline = Pipeline([
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# === Step 5: Combine preprocessor ===
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# === Step 6: Define base models and final model ===
base_models = [
    ('ridge', Ridge()),
    ('lasso', Lasso(max_iter=10000)),
    ('knn', KNeighborsRegressor())
]

final_estimator = LinearRegression()

# === Step 7: Create full pipeline ===
model = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', StackingRegressor(
        estimators=base_models,
        final_estimator=final_estimator,
        n_jobs=-1
    ))
])

# === Step 8: Define hyperparameter grid ===
param_grid = {
    'stacking__ridge__alpha': [1, 5, 10],
    'stacking__lasso__alpha': [0.1, 1, 5],
    'stacking__knn__n_neighbors': [3, 5, 7],
    'stacking__knn__weights': ['uniform', 'distance'],
    'stacking__knn__p': [1, 2]
}

# === Step 9: Apply GridSearchCV ===
grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid.fit(X, y)

# === Step 11: Print best score and parameters ===
print("Best Parameters:", grid.best_params_)
print("Best CV RMSE:", -grid.best_score_)


# === Step 10: Prediction and submission ===
y_test_pred = grid.predict(X_test)
submission = pd.DataFrame({'index': test['index'], 'Price': y_test_pred})
submission.to_csv("submission.csv", index=False)




