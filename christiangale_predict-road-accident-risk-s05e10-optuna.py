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

import warnings
warnings.filterwarnings('ignore')

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv').set_index('id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv').set_index('id')


train.head()


train.info()


train.describe()


num_cols = train.select_dtypes(include='number').columns
cat_cols = train.select_dtypes(include='object').columns
bool_cols = train.select_dtypes(include='bool').columns


sentiment = {True: 1, False: 0}
train[bool_cols] = train[bool_cols].replace(sentiment)


sns.set_theme(style="whitegrid")  # style clean

for col in bool_cols:
    plt.figure(figsize=(8,5))
    ax = sns.barplot(
        data=train,
        x=col,
        y="accident_risk",
        estimator="mean",
        palette="coolwarm",   # coba ganti "coolwarm", "viridis", dll
        edgecolor="black"
    )
plt.title(f"Mean risk Based On {col}", fontsize=14, weight="bold")
plt.xlabel(col, fontsize=12)
plt.ylabel("Mean risk", fontsize=12)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


sns.set_theme(style="whitegrid")  # style clean

for col in cat_cols:
    plt.figure(figsize=(8,5))
    ax = sns.barplot(
        data=train,
        x=col,
        y="accident_risk",
        estimator="mean",
        palette="coolwarm",   # coba ganti "coolwarm", "viridis", dll
        edgecolor="black"
    )
plt.title(f"Mean risk Based On {col}", fontsize=14, weight="bold")
plt.xlabel(col, fontsize=12)
plt.ylabel("Mean risk", fontsize=12)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


train[num_cols].corr()


sns.heatmap(train[num_cols].corr())


sns.displot(train['accident_risk'], bins=30, kde=True)
plt.show()


train['curvature_bin'] = pd.qcut(train['curvature'], q=3, labels=['Low', 'Medium', 'High'])
curvature_risk = train.groupby('curvature_bin')['accident_risk'].mean().sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(x=curvature_risk.values, y=curvature_risk.index)

plt.title('Accident Risk by Road Curvature', fontsize=16)
plt.xlabel('Accident Risk', fontsize=12)
plt.ylabel('Curvature Level', fontsize=12)

oe_curvature = {
    "Low": 1,
    'Medium': 2,
    "High": 3
}

train['curvature_bin'] = train['curvature_bin'].replace(oe_curvature)


train['night_lightning'] = ((train['weather'] == 'foggy') & (train['lighting'] == 'night')).astype(int)


night_lightning_risk = train.groupby('night_lightning')['accident_risk'].mean().sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(y=night_lightning_risk.values, x=night_lightning_risk.index)

plt.title('Accident Risk by night_lightning_risk', fontsize=16)
plt.xlabel('Accident Risk', fontsize=12)
plt.ylabel('night_lightning_risk', fontsize=12)


X = train.drop(columns='accident_risk', axis=1)
y = train['accident_risk']


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)


X_train.head()


num_transformer = Pipeline([
    ('num', MinMaxScaler())
])

cat_transformer = Pipeline([
    ('cat', OneHotEncoder())
])

preprocess = ColumnTransformer([
    ('num_transformer', num_transformer, X.select_dtypes(include='number').columns),
    ('cat_transformer', cat_transformer, X.select_dtypes(include='object').columns),
])





# import optuna
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
#         'random_state': 42,
#         'objective': 'reg:squarederror',
#         'tree_method': 'hist',     
#         'device': 'cuda'           
#     }

#     model = Pipeline([
#         ('preprocess', preprocess),
#         ('model', XGBRegressor(**params))
#     ])

#     scores = cross_val_score(
#         model, X_train, y_train,
#         cv=5,
#         scoring='neg_root_mean_squared_error',
#         n_jobs=-1
#     )
    
#     return -np.mean(scores)

# study = optuna.create_study(direction='minimize', study_name="XGB_Optimization")
# study.optimize(objective, n_trials=30, timeout=1800)

# print("Best RMSE:", study.best_value)
# print("Best Params:", study.best_params)



test['curvature_bin'] = pd.qcut(test['curvature'], q=3, labels=['Low', 'Medium', 'High'])
oe_curvature = {
    "Low": 1,
    'Medium': 2,
    "High": 3
}

test['curvature_bin'] = test['curvature_bin'].replace(oe_curvature)
test['night_lightning'] = ((test['weather'] == 'foggy') & (test['lighting'] == 'night')).astype(int)
test[bool_cols] = test[bool_cols].replace(sentiment)


best_params = {
    'n_estimators': 1276, 
    'learning_rate': 0.005268819730480533, 
    'max_depth': 9, 'subsample': 0.916192189316052, 
    'colsample_bytree': 0.9531224568060148, 
    'reg_alpha': 0.31654887832397505, 
    'reg_lambda': 0.022186345591767367,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda'
}


model = Pipeline([
    ('preprocess', preprocess),
    ('model', XGBRegressor(**best_params))
]).fit(X, y)


y_pred = model.predict(test)
my_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv').set_index('id')
my_sub['accident_risk'] = y_pred
my_sub.to_csv('submission.csv')




