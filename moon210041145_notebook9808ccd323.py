# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import lightgbm as lgb

!pip install category_encoders  
import category_encoders as ce

import optuna

from tqdm.auto import tqdm



train = pd.read_csv('/kaggle/input/this-comp/train.csv')
test = pd.read_csv('/kaggle/input/this-comp/test.csv')
train_extra = pd.read_csv('/kaggle/input/this-comp/training_extra.csv')
train.head()
train.info()
train.describe()




train.isnull().sum()



import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore')



train['Material'].fillna('Unknown', inplace=True)
train['Size'].fillna('Unknown', inplace=True)
train['Style'].fillna('Unknown', inplace=True)



train['Brand'].fillna('Unknown', inplace=True)
train['Laptop Compartment'].fillna('Unknown', inplace=True)
train['Waterproof'].fillna('Unknown', inplace=True)
train['Color'].fillna('Unknown', inplace=True)

train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median(), inplace=True)



# Categorical encoding
categorical_cols = ['Brand', 'Material', 'Size', 'Style', 'Laptop Compartment', 'Waterproof', 'Color']
target_enc = ce.TargetEncoder(cols=categorical_cols)
train[categorical_cols] = target_enc.fit_transform(train[categorical_cols], train['Price'])



num_cols = ['Weight Capacity (kg)', 'Compartments']
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])



print(train.columns)



X = train.drop(columns=['id', 'Price'])
y = train['Price']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return mean_squared_error(y_valid, pred)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best Parameters:", study.best_params)

# Train the best model
best_model = XGBRegressor(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_valid)
print(f'Best Model RMSE: {np.sqrt(mean_squared_error(y_valid, predictions)):.2f}')

# Stacking model
estimators = [
    ('xgb', XGBRegressor(n_estimators=50, random_state=42)),
    ('ridge', Ridge(alpha=1.0)),
]
stack_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stack_model.fit(X_train, y_train)
stack_preds = stack_model.predict(X_valid)
print(f'Stacking Model RMSE: {np.sqrt(mean_squared_error(y_valid, stack_preds)):.2f}')



from sklearn.ensemble import StackingRegressor

estimators = [
    ('xgb', XGBRegressor(n_estimators=50)),
    ('ridge', Ridge(alpha=1.0)),
]

stack_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stack_model.fit(X_train, y_train)

preds = stack_model.predict(X_valid)
print("Stacking Model RMSE:", mean_squared_error(y_valid, preds, squared=False))


