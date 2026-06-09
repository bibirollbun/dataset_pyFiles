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


# data handling
import pandas as pd
import numpy as np

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing and model selection
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer

# regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.ensemble import VotingRegressor,StackingRegressor

# evaluation metric
from sklearn.metrics import mean_squared_error


# ignore unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=FutureWarning, 
                        message='use_inf_as_na option is deprecated')

warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='pandas.io.formats.format')

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# path for training dataset
train_path = '/kaggle/input/crop-yield-prediction-challenge/crop_yield_train.csv'
# path for testing dataset
test_path = '/kaggle/input/crop-yield-prediction-challenge/crop_yield_test.csv'

# removing the limit for viewing all rows
pd.set_option('display.max_rows', None)
# removing the limit for viewing all features
pd.set_option('display.max_columns', None)

SEED=42 # seed for reproducibility
SPLIT=0.3 # testing split
FOLDS = 10 # no. of folds for cross-validation


# loading the training dataset
df_train = pd.read_csv(train_path, parse_dates=['harvest_date'])
#loading the testing dataset
df_test = pd.read_csv(test_path, parse_dates=['harvest_date'])


df_train.head() # view first 5 rows of training dataset


# drop the id column
df_train = df_train.drop('id', axis=1)

test_ids = df_test['id']
df_test = df_test.drop('id', axis=1)


# vewing the shape of the training dataset
print('='*40)
print(f'Shape of training dataset: {df_train.shape}')
print('='*40)


# checking the training dataset for missing values
missing_values = df_train.isnull().sum()

print('='*50)
print('Features with missing values:')
print(missing_values[missing_values > 0].sort_values(ascending=False))
print('='*50)
print('Percentage (%) of missing values in each feature:')
print(missing_values[missing_values > 0].sort_values(ascending=False) / len(df_train) * 100)
print('='*50)


# checking the training dataset for duplicate records
print('='*40)
print(f'Duplicate records in training dataset: {df_train.duplicated().sum()}')
print('='*40)


# viewing the columns in the training dataset
print('='*100)
print(f'Columns in training dataset: {df_train.columns}') # viewing the columns in the training dataset
print('='*100)


# checking the number of datatypes in training dataset
print('='*25)
print(df_train.dtypes.value_counts())
print('='*25)


# separating the numeric and categorical features
train_numeric = df_train.select_dtypes(include='number').columns
train_categorical = df_train.select_dtypes(include='object').columns
train_date = df_train.select_dtypes(include='datetime64').columns


# checking summary statistics in training dataset
df_train[train_numeric].describe()


# checking the skewness of numeric features in training dataset
for feature in train_numeric:
    print('='*50)
    print(f'Skewness of {feature}: {df_train[feature].skew()}')
print('='*50)


# helper function for plotting feature distribution
def plot_distribution(df, feature):
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()


# checking distribution of numeric features in training dataset
for feature in train_numeric:
    plot_distribution(df_train, feature)


# checking for outliers using boxplot in numeric features in training dataset
fig, ax = plt.subplots(3, 5, figsize=(15, 12))
ax = ax.flatten()

for i, col in enumerate(train_numeric):
    sns.boxplot(data=df_train, y=col, ax=ax[i])
    ax[i].set_title(f'Boxplot of {col}')

for i in range(len(train_numeric), 15):
    ax[i].set_visible(False)
    
plt.tight_layout()
plt.show()


# checking correlation of numeric features in training dataset
corr = df_train[train_numeric].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()


# checking the categorical features
print('='*100)
print(f'Categorical Features: {train_categorical}')
print('='*100)


# for simplicity, dropping field_id feature
df_train = df_train.drop('field_id', axis=1)
df_test = df_test.drop('field_id', axis=1)
train_categorical = df_train.select_dtypes(include='object').columns


# checking count of values in categorical features
df_train['harvest_date'].value_counts()


# checking count of values in categorical features
for cat_feature in train_categorical:
    print('='*30)
    print(f'Value Counts in {df_train[cat_feature].value_counts()}')

print('='*30)


# making harvest_month based on harvest_date
df_train['harvest_month'] = df_train['harvest_date'].dt.month
df_test['harvest_month'] = df_test['harvest_date'].dt.month


# dropping the harvest_date feature
df_train = df_train.drop('harvest_date', axis=1)
df_test = df_test.drop('harvest_date', axis=1)


df_train.head()


# encoding categorical features
season_order = [['Spring', 'Summer', 'Autumn']]

preprocessor = make_column_transformer(
    (OneHotEncoder(drop='first', sparse_output=False), ['crop_type', 'region']),
    (OrdinalEncoder(categories=season_order), ['season']),
    remainder='passthrough'
)


# remove the target to properly encode the independent features
X = df_train.drop('yield_tpha', axis=1)
y = df_train['yield_tpha']


X_train_encoded = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()
df_train = pd.DataFrame(X_train_encoded, columns=feature_names, index=df_train.index)

X_test_encoded = preprocessor.transform(df_test)
df_test = pd.DataFrame(X_test_encoded, columns=feature_names, index=df_test.index)


X = df_train


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT, random_state=SEED)


# scaling the independed features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# making a dictionary for regression models
models = {
    'Linear_Regression': LinearRegression(),
    'Ridge_Regression': Ridge(),
    'Lasso_Regression': Lasso(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'XGBRegressor': XGBRegressor(),
    'CatBoostRegressor': CatBoostRegressor(verbose=False),
    'LGBMRegressor': LGBMRegressor(),
    'SupportVectorRegression': SVR()
}


# rmse for evaluation 
def root_mean_squared_error(mse):
    return np.sqrt(mse)


# training and testing multiple regressors to get a baseline estimate
for name, model in models.items():
    print('='*50)
    print(f'Training {name}...')
    model.fit(X_train_scaled, y_train)
    y_predictions = model.predict(X_test_scaled)
    print(f'RMSE: {root_mean_squared_error(mean_squared_error(y_test, y_predictions))}')
print('='*50)


# parameter grid for GradientBoostingRegressor
param_grid_gbm = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'loss': ['squared_error', 'absolute_error', 'huber']
}


# tuning GradientBoostingRegressor
gbr = GradientBoostingRegressor()

random_search_gbr = RandomizedSearchCV(
    gbr, 
    param_grid_gbm, 
    n_iter=50, 
    cv=FOLDS, 
    scoring='neg_mean_squared_error', 
    n_jobs=-1, 
    random_state=SEED
)


print('='*30)
print('starting tuning GBR...')
print('='*30)


random_search_gbr.fit(X_train_scaled, y_train)

# best results
print("Best Parameters:", random_search_gbr.best_params_)
print("Best RMSE:", np.sqrt(-random_search_gbr.best_score_))

# get the best model
tuned_GBR = random_search_gbr.best_estimator_


print('='*30)
print('GBR tuning completed.')
print('='*30)


# parameter grid for LinearRegression
param_grid_lr = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}


# tuning LinearRegression
lr = LinearRegression()

random_search_lr = RandomizedSearchCV(
    lr,
    param_grid_lr,
    n_iter=50,
    cv=FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=SEED
)


print('='*30)
print('starting tuning of LR...')
print('='*30)


random_search_lr.fit(X_train_scaled, y_train)

# best results
print("Best Parameters:", random_search_lr.best_params_)
print("Best RMSE:", np.sqrt(-random_search_lr.best_score_))

# get the best model
tuned_LR = random_search_lr.best_estimator_


print('='*30)
print('LR tuning completed.')
print('='*30)


# parameter grid for RandomForestRegressor
param_grid_rfr = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}


# tuning RandomForestRegressor
rfr = RandomForestRegressor()

random_search_rfr = RandomizedSearchCV(
    rfr,
    param_grid_rfr,
    n_iter=50,
    cv=FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=SEED
)


print('='*30)
print('starting tuning of RFR...')
print('='*30)


random_search_rfr.fit(X_train_scaled, y_train)

# best results
print("Best Parameters:", random_search_rfr.best_params_)
print("Best RMSE:", np.sqrt(-random_search_rfr.best_score_))

# get the best model
tuned_RFR = random_search_rfr.best_estimator_


print('='*30)
print('RFR tuning completed.')
print('='*30)


# scaling the entire training dataset
X_scaled = scaler.fit_transform(X)
df_test_scaled = scaler.transform(df_test)


# making stacking regressor from the top 3 performing models
ensemble = StackingRegressor(
    estimators=[
        ('gbr', tuned_GBR),
        ('lr', tuned_LR),
        ('rfr', tuned_RFR)
    ],
    final_estimator=LinearRegression(),
    cv=FOLDS
)


# training the ensemble on training dataset
ensemble.fit(X_scaled, y)

# make predictions
predictions = ensemble.predict(df_test_scaled)


# saving test predictions
submission = pd.DataFrame({
    'id': test_ids,
    'yield_tpha': predictions
})

submission.to_csv('submission.csv', index=False)


print('='*30)
print('submission file saved successfully.')
print('='*30)

