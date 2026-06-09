# import the basics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


df_train.head()


# split training data in a training and validation set

from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

train_set, valid_set = train_test_split(df_train, test_size=0.2, random_state=RANDOM_STATE)



from sklearn.base import BaseEstimator, TransformerMixin


class BooleanConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.array(X).astype(str)
        X_bool = np.array([[val.strip().lower() == "yes" for val in col] for col in X.T]).T
        return X_bool


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

num_cols = ["Compartments", "Weight Capacity (kg)"]
cat_cols = ["Brand", "Material","Style", "Color"]
bool_cols = ["Laptop Compartment", "Waterproof"]
ord_cols = ["Size"]

pipeline_num = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

pipeline_cat = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

pipeline_bool = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("bool_maker", BooleanConverter())
])

size_order = [['Small', 'Medium', 'Large']]
pipeline_ord = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=size_order))
])

transformer = ColumnTransformer(transformers=[
    ('t_num', pipeline_num, num_cols),
    ('t_cat', pipeline_cat, cat_cols),
    ('t_bool', pipeline_bool, bool_cols),
    ('t_ord', pipeline_ord, ord_cols)
])

preprocessor = Pipeline(steps=[
    ("transformer", transformer)
])


# 1. split X and y
X_train = train_set.drop("Price", axis=1).copy()
y_train = train_set["Price"].copy()

# 2. preprocess data
X_train_transformed = preprocessor.fit_transform(X_train)


# 1. split X and y
X_valid = valid_set.drop("Price", axis=1).copy()
y_valid = valid_set["Price"].copy()

# 2. preprocess data
X_valid_transformed = preprocessor.transform(X_valid)


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

param_grid = {"n_estimators": [100, 200],
             "max_depth": [2, 4, 8],
             "learning_rate": [0.1, 0.5],
             "colsample_bytree": [0.4, 0.8, 1],
             "random_state": [42]}

xgb_reg = xgb.XGBRegressor()

grid_search = GridSearchCV(xgb_reg,
                          param_grid,
                          cv=3,
                          verbose=3,
                          scoring="neg_mean_squared_error",
                          return_train_score=True)

grid_search.fit(X_train_transformed, 
                y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)


final_model = grid_search.best_estimator_

# prepare the full dataset for fitting
X = df_train.drop("Price", axis=1).copy()
y = df_train["Price"].copy()

# preprocess and fit the X data
X_transformed = preprocessor.fit_transform(X)
final_model.fit(X_transformed, y)

# prepare the test set for fitting
# prepare the validation set for fitting
X_test = df_test
# preprocess the X data
X_test_transformed = preprocessor.transform(X_test)
y_pred = final_model.predict(X_test_transformed)

result = pd.DataFrame({
    "id": df_test.id,
    "Price": y_pred
})

result.to_csv('submission.csv', index=False)

