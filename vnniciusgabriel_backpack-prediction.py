!mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
!sudo apt install nvidia-driver-460 nvidia-cuda-toolkit clinfo
!apt-get update --fix-missing
!pip install -q  lightgbm==4.1.0 \
  --config-settings=cmake.define.USE_GPU=ON \
  --config-settings=cmake.define.OpenCL_INCLUDE_DIR="/usr/local/cuda/include/" \
  --config-settings=cmake.define.OpenCL_LIBRARY="/usr/local/cuda/lib64/libOpenCL.so"


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train_df = pd.concat([train_df, train_extra_df], axis=0, ignore_index=True)


train_df.isna().sum()


import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.2f} seconds")
        return result
    return wrapper       


@timeit
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    X = df.copy()
  
    dict_empty_values = {
        'Material': 'NaN',
        'Style': 'NaN',
        'Brand': 'NaN',
        'Size': 'NaN',
        'Waterproof': 'NaN',
        'Color': 'NaN',
        'Laptop Compartment': 'NaN'
    }

    X = X.fillna(dict_empty_values)

    X['Brand_Material'] = X['Brand'] + '_' + X['Material']
    X['Weight_Capacity_Per_Compartments'] = X['Weight Capacity (kg)'] / X['Compartments']

    categorical_features = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    return X


X = preprocess_data(train_df.drop(['id', 'Price'], axis=1))
y = train_df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


@timeit
def train_stacking_model():

    N_ITERATIONS = 1000
    RANDOM_STATE = 42

    estimators = [
        ('xgb', XGBRegressor(
            random_state=RANDOM_STATE,
            n_estimators=N_ITERATIONS,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            device='cuda:0',
            verbosity=0 
         )),
        ('lgbm', LGBMRegressor(
            random_state=RANDOM_STATE,
            n_estimators=N_ITERATIONS,
            device='gpu',
            gpu_platform_id=0,
            gpu_device_id=1,
            verbose=-1
        ))
    ]

    meta_model = Ridge(random_state=RANDOM_STATE, alpha=0.1)

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        passthrough=False        
    )

    stacking_regressor.fit(X_train, y_train)

    y_pred = stacking_regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"test RMSE for Stacking Regressor: {rmse:.4f}")

    return stacking_regressor

model = train_stacking_model()


X_test = preprocess_data(test_df.drop(['id'], axis=1))
y_pred = model.predict(X_test)

test_df['Price'] = y_pred
test_df[['id', 'Price']].to_csv('/kaggle/working/submission.csv', index=False)

