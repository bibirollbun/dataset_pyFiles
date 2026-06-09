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


from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

#Load data
traindf = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
testdf = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

#Split features & target
X = traindf.drop("accident_risk", axis=1)
y = traindf["accident_risk"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove 'id' from numerical columns if it's present
if 'id' in numerical_cols:
    numerical_cols.remove('id')

#Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

#Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

#Models to evaluate
models = {
    "LightGBM": lgb.LGBMRegressor(verbose=0),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

#Custom scorers
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

#Store results
results = {}

#Cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)

#Evaluate each model using cross-validation
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    cv_scores = cross_validate(
        pipeline,
        X, y,
        cv=cv,
        scoring={'RMSE': rmse_scorer, 'R2': 'r2'},
        return_train_score=False
    )

    rmse_scores = np.sqrt(-cv_scores['test_RMSE'])  # Convert negative MSE to RMSE
    r2_scores = cv_scores['test_R2']

    results[name] = {
        "RMSE Mean": rmse_scores.mean(),
        "RMSE Std": rmse_scores.std(),
        "R2 Mean": r2_scores.mean()
    }

    print(f"\n{name}")
    print(f" Avg RMSE: {rmse_scores.mean():.4f}")
    print(f" Std Dev: {rmse_scores.std():.4f}")
    print(f" Avg RÂ²: {r2_scores.mean():.4f}")

    if name == "LightGBM":
        best_pipeline = pipeline

#Fit best pipeline on full data
best_pipeline.fit(X, y)

#Final predictions on test data
final_preds = best_pipeline.predict(testdf)

#Submission file
submission = pd.DataFrame({
    'id': testdf['id'],
    'y': final_preds
})
submission.to_csv('submission.csv', index=False)

print("\n Submission file 'submission.csv' created using LightGBM pipeline with cross-validation!")



submission





