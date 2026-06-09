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


x=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')

x


test=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')
test


test_new=test.drop(columns=['id'])
test_new


null_counts = x.isnull().sum()
null_counts


y=x['target']
y



X=x.drop(columns=['target'])
X


# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)









# from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor
# from xgboost import XGBRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error, r2_score


# base_models = [
#     ('extra_trees', ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_split=5, random_state=42)),
#     ('xgboost', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42)),
#     ('random_forest', RandomForestRegressor(n_estimators=250, max_depth=15, min_samples_split=4, random_state=42))
# ]

# meta_model = Ridge(alpha=0.5)


# stack_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model, n_jobs=-1)

# stack_regressor.fit(X, y)

# y_pred = stack_regressor.predict(X)

# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y, y_pred)

# print("\nOptimized Stacking Regressor Evaluation:")
# print(f"Mean Squared Error: {mse}")
# print(f"R² Score: {r2}")

# print("Sample Predictions:", y_pred[:10])



from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# y and X are already prepared from your dataset
# y = x['target']
# X = x.drop(columns=['target'])

# Define the base models
base_models = [
    ('extra_trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
    ('xgboost', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
]

# Define the meta-model
meta_model = LinearRegression()

# Create the stacking regressor
stack_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Fit the stacking regressor
stack_regressor.fit(X, y)

# Make predictions on the dataset (assuming a test set is available)
y_pred_train = stack_regressor.predict(X)

# Calculate evaluation metrics
mse = mean_squared_error(y, y_pred_train)
r2 = r2_score(y, y_pred_train)

# Print results
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Optional: View a sample of predictions
print("Sample Predictions:", y_pred_train[:10])



# from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor
# from xgboost import XGBRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import GridSearchCV

# # Base models with hyperparameter tuning
# extra_trees = ExtraTreesRegressor(
#     n_estimators=500, max_depth=20, min_samples_split=3, min_samples_leaf=2, random_state=42
# )

# xgboost = XGBRegressor(
#     n_estimators=500, 
#     learning_rate=0.03, 
#     max_depth=10, 
#     subsample=0.9, 
#     colsample_bytree=0.9, 
#     gamma=0.1, 
#     reg_alpha=1, 
#     reg_lambda=1, 
#     random_state=42
# )

# random_forest = RandomForestRegressor(
#     n_estimators=400, max_depth=18, min_samples_split=3, min_samples_leaf=2, random_state=42
# )

# ridge_meta_model = Ridge(alpha=0.1)

# stacking_regressor = StackingRegressor(
#     estimators=[
#         ('extra_trees', extra_trees),
#         ('xgboost', xgboost),
#         ('random_forest', random_forest)
#     ],
#     final_estimator=ridge_meta_model,
#     n_jobs=-1
# )

# stacking_regressor.fit(X, y)

# # Make predictions
# y_pred = stacking_regressor.predict(X)

# # Evaluate the model
# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y, y_pred)

# print("\nFurther Optimized Stacking Regressor Evaluation:")
# print(f"Mean Squared Error: {mse}")
# print(f"R² Score: {r2}")


# print("Sample Predictions:", y_pred[:10])




y_pred = stack_regressor.predict(test_new)


submission = pd.DataFrame({"id": test["id"], "target": y_pred})

# Save file
submission.to_csv("submission_model.csv", index=False)
print("Submission file saved as submission_model.csv")

