import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, StackingRegressor)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original_df = pd.read_csv('/kaggle/input/extrovert-introvert-dataset/personality_datasert.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


num_col = [col for col in train_df.select_dtypes(['number']).columns if col!='id']
cat_col = [col for col in train_df.select_dtypes(['object']).columns if col!='Personality']
target_col = "Personality"
id_col = 'id'


combined_df = pd.concat([train_df.drop(columns=id_col, axis=1),original_df], ignore_index=True).drop_duplicates()


le = LabelEncoder()
tgt_le = LabelEncoder()


non_impute_columns = [target_col]
features = combined_df.drop(columns=non_impute_columns)


encoders = {}

for col in cat_col:
    temp_col = combined_df[col].fillna("NaN_Placeholder")
    combined_df[col] = le.fit_transform(temp_col)  # Encode the column
    combined_df[col] = combined_df[col].where(temp_col != "NaN_Placeholder", np.nan) # Restore NaN values
    encoders[col] = {cls: le.transform([cls])[0] for cls in le.classes_ if cls != "NaN_Placeholder"}


numerical_columns = list(combined_df.select_dtypes('number').columns)


# Impute numerical columns
imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), random_state=42)
combined_df[numerical_columns] = imputer.fit_transform(combined_df[numerical_columns])


X = combined_df.drop(target_col, axis=1)
y = combined_df[target_col]


y_encoded = tgt_le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


preprocessor = ColumnTransformer([
    ("num", StandardScaler(), X.select_dtypes('number').columns),
    ("cat", OneHotEncoder(drop='first'), X.select_dtypes('object').columns)
])


alphas = np.logspace(-3, 3, 20)

models = {
        "LinearRegression": LinearRegression(),
        "Ridge": RidgeCV(alphas=alphas),
        "Lasso": LassoCV(alphas=alphas, cv=5),
        "ElasticNet": ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.9], cv=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
        "AdaBoost": AdaBoostRegressor(n_estimators=100),
        "Bagging": BaggingRegressor(n_estimators=100, n_jobs=-1),
        "DecisionTree": DecisionTreeRegressor(max_depth=5),
        "KNeighbors": KNeighborsRegressor(),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05, n_jobs=-1),
        "CatBoost": CatBoostRegressor(verbose=0)
    }


results = []
for name, model in models.items():
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    results.append((name, rmse, r2, mse, mae))


results = sorted(results, key=lambda x: x[1])
print("\nModel Performance on Used Car Price Prediction:\n")
for name, rmse, r2, mse, mae in results:
    print(f"{name:20s} | RMSE: {rmse:4.2f} | R2: {r2:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}")

