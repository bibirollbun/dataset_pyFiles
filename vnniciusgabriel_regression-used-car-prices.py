!pip install --upgrade scikit-learn==1.3.1


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.preprocessing import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')


df_train.head()


def plot_column_value_distribution(name: str, distribution: pd.DataFrame):

    plt.figure(figsize=(12, 6))
    sns.barplot(x=distribution.index, y=distribution.values, palette='viridis')

    plt.xticks(rotation=45, ha='right')
    plt.xlabel(f"{name}")
    plt.ylabel("count")
    plt.title(f"{name} distribution")

    plt.show()


plot_column_value_distribution('brand', df_train['brand'].value_counts())


plot_column_value_distribution('fuel_type', df_train['fuel_type'].value_counts())


plot_column_value_distribution('transmission', df_train['transmission'].value_counts())


plot_column_value_distribution('accident', df_train['accident'].value_counts())


from time import perf_counter
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        print(f'{func.__name__} took {perf_counter() - start:.2f} seconds')
        return result
    return wrapper


@timeit
def preprocess_data(df: pd.DataFrame, encoder: TargetEncoder, *, test: bool = False) -> pd.DataFrame:
    """
    preprocess data

    1. imputing missing data for fuel_type, accident and clean_title;
    2. extract columns from engine and impute missing data for this new columns;
    3. simplify transmission column;
    4. transform accident column to binary;
    5. fix fuel type merging - values with not supported values;
    6. encode category features;

    Args:
        df (pd.DataFrame): raw dataframe.
        encoder (TargetEncoder): target encoder.
        test (bool): is testing dataset?.
    Returns:
        pd.DataFrame: preprocessed dataframe.
    """
    if test:
        X = df.drop(['id'], axis=1)
    else:
        X = df.drop(['id', 'price'], axis=1)
        y = df['price']

    # 1. impute missing data
    X['fuel_type'] = X['fuel_type'].fillna('not supported')
    X['accident'] = X['accident'].fillna('None reported')
    X['clean_title'] = X['clean_title'].fillna('NO')

    # 2. extract columns from engine and impute missing data for this new columns
    X['hp'] = pd.to_numeric(X['engine'].str.extract(r'(\d+(\.\d+)?)HP')[0], errors='coerce')
    X['liters'] = pd.to_numeric(X['engine'].str.extract(r'(\d+\.\d+)L')[0], errors='coerce')
    X['cylinders'] = pd.to_numeric(X['engine'].str.extract(r'(\d+)\s+Cylinder')[0], errors='coerce')

    X['hp'] = X['hp'].fillna(X['hp'].median())
    X['liters'] = X['liters'].fillna(X['liters'].median())
    X['cylinders'] = X['cylinders'].fillna(X['cylinders'].median())

    X = X.drop('engine', axis=1)

    # 3. simplify transmission column
    X['transmission'] = X['transmission'].replace({
        r'(?i).*Dual Shift.*': 'Dual Shift',
        r'(?i).*(automatic|A/T).*': 'Automatic',
        r'(?i).*(manual|M/T).*': 'Manual',
        r'(?i).*CVT.*': 'CVT'
    }, regex=True)

    X['transmission'] = X['transmission'].where(X['transmission'].isin(['Dual Shift', 'Automatic', 'Manual', 'CVT']),'Other')

    # 4. transform accident column to binary
    X['has_accident'] = (X['accident'] != 'None reported').astype(int)
    X = X.drop('accident', axis=1)

    # 5. fix fuel type merging - values with not supported values
    X['fuel_type'] = X['fuel_type'].replace('-', 'not supported')

    # 6. encode category features
    category_cols = X.select_dtypes(include=['object']).columns
    if test:
        X[category_cols] = encoder.transform(X[category_cols])
    else:
        X[category_cols] = encoder.fit_transform(X[category_cols], y)

    if test:
        return X
    
    return X, y


encoder = TargetEncoder(smooth="auto", target_type='binary', cv=5, random_state=42)

X, y = preprocess_data(df_train, encoder)
print(f"shape scalled and preprocessed: {X.shape}")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'shape train: {X_train.shape}, test: {X_val.shape}')


X_train.head()


@timeit
def train_and_evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid: dict,
    cv: int =5,
    scoring: str='neg_mean_squared_error',
    n_iter: int = 10
):
    """
    trains a give model using GridSearchCV and evaluates its performace.

    Args:
        model: a scikit-learn estimator.
        X_train (pd.DataFrame or np.array): training features.
        y_train (pd.Series or np.array): training target.
        X_val (pd.DataFrame or np.array): validation features.
        y_val (pd.Series or np.array): validation target.
        param_grid (dict): dictionary of hyperparamets for grid search.
        cv (int): number of cross-validation folds.
        scoring (str): scoring metric to use for grid search (default: neg MSE).
        n_iter (int): number of parameter settings sampled (default: 10).
        
    Returns:
        dict: dictionary containing the best estimator, best pararamets (if grid searh was used)
        and evaluation metrics (MSE, RMSE, R2) on the val data.
    """

    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=cv, 
        scoring=scoring,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    return {
        'best_model': best_model,
        'best_params': best_params,
        'mse': mse,
        'rmse': rmse,
    }


models =  {
    'LGBMRegressor': {
        'estimator': LGBMRegressor(random_state=42),
        'param_grid': {
            'num_leaves': [31, 50, 70],
            'max_depth': [10, 20, 30],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [50, 100, 200]
        }
    },
    'XGBRegressor': {
        'estimator': XGBRegressor(random_state=42, objective='reg:squarederror'),
        'param_grid': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    },
    'RandomForestRegressor': {
        'estimator': RandomForestRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        }
    }
}

results = {}
for model_name, model_info in models.items():
    print(f'training {model_name}...')
    res = train_and_evaluate_model(
        model_info['estimator'], 
        X_train, 
        y_train, 
        X_val, 
        y_val,
        param_grid=model_info['param_grid'], 
        cv=5
    )
    results[model_name] = res
    print(f"{model_name} -- MSE: {res['mse']:.2f}, RMSE: {res['rmse']:.2f}")
    print("best paramaters:", res['best_params'])
    print("=" * 40)


xgb = results['XGBRegressor']['best_model']
X = preprocess_data(df_test, encoder, test=True)

y_pred = xgb.predict(X)

df_test['price'] = y_pred
df_test[['id', 'price']].to_csv('/kaggle/working/submission.csv', index=False)

