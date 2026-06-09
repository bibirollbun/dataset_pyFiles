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


!pip install catboost


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

train_df = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip').set_index('Id')
train_df


test_df = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip').set_index('Id')
test_df


sample_submission_df = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv')
sample_submission_df


for col in train_df.columns:
  print(train_df[col].value_counts())


train_df.isna().sum()


test_df.isna().sum()


train_df.info()


numerical_columns = train_df.select_dtypes(include=np.number).columns
numerical_columns


for col in numerical_columns :
  plt.figure(figsize=(8, 5))
  sns.scatterplot(x=col, y='revenue', data=train_df)
  plt.title(f'{col} vs Revenue')
  plt.xlabel(f'{col}')
  plt.ylabel('Revenue')
plt.show()


plt.figure(figsize=(8, 5))
sns.displot(data=train_df['revenue'], kind="hist", bins=30, kde=True, color='skyblue', height=6, aspect=1.5)
plt.show()


train_df = train_df[train_df['revenue'] < 1e7]
train_df


corr = train_df[['P' + str(i) for i in range(1,38)]+['revenue']].corr()['revenue'].iloc[:-1]
train_df['P_weighed_sum'] = train_df[numerical_columns].mul(corr).sum(axis=1)
train_df


corr_matrix = train_df[numerical_columns].corr()

plt.figure(figsize=(30, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


train_df['Open Date'] = pd.to_datetime(train_df['Open Date'])

date_revenue = pd.DataFrame(
    {
        'Open Date' : train_df['Open Date'],
        'day': train_df['Open Date'].dt.day,
        'month': train_df['Open Date'].dt.month,
        'year': train_df['Open Date'].dt.year,
        'day_sin' : np.sin(train_df['Open Date'].dt.day*2*np.pi / 31),
        'day_cos' : np.cos(train_df['Open Date'].dt.day*2*np.pi / 31),
        'month_sin' : np.sin(train_df['Open Date'].dt.month*2*np.pi / 12),
        'month_cos' : np.cos(train_df['Open Date'].dt.month*2*np.pi / 12),
        'revenue' : train_df['revenue'],
    }
)

date_revenue.sort_values(by='Open Date',inplace=True)
date_revenue



plt.figure(figsize=(15, 6))
plt.plot(date_revenue['Open Date'],date_revenue['revenue'])
plt.title('Restaurant Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.grid(True)
plt.show()


for col in ['day', 'day_sin', 'day_cos', 'month', 'month_sin', 'month_cos', 'year']:
  plt.figure(figsize=(15, 6))

  grouped_df = date_revenue.groupby(col)['revenue'].mean().reset_index()

  plt.plot(grouped_df[col],grouped_df['revenue'])
  plt.title(f'Restaurant Revenue Over {col}')
  plt.xlabel(f'{col}')
  plt.ylabel('Revenue')
  plt.grid(True)

  if col == 'year':
    plt.xticks(range(date_revenue['year'].min(),date_revenue['year'].max()+1))
  elif col == 'month':
    plt.xticks(range(1,13))
  elif col == 'day':
    plt.xticks(range(1,32))
plt.show()


corr_matrix = date_revenue.corr()

plt.figure(figsize=(30, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


categorical_columns = ['City', 'City Group', 'Type']
categorical_columns


for col in categorical_columns :
  plt.figure(figsize=(8, 5))
  sns.barplot(x=col, y='revenue', data=train_df)
  plt.title(f'{col} vs Revenue')
  plt.xticks(rotation=90)
plt.show()


from sklearn.base import BaseEstimator, TransformerMixin

class PWeighedSum_Transformer(BaseEstimator, TransformerMixin):

  def __init__(self, cols=['P'+str(i) for i in range(1,38)], target_col='revenue'):
    self.target_col = target_col
    self.cols = cols
    self.corr = None

  def fit(self, X, y=None):

    X_df = X.copy()

    if y is None:
      raise ValueError("Target column is required.")

    X_df[self.target_col] = y
    self.corr = X_df[self.cols + [self.target_col]].corr()[self.target_col].iloc[:-1]
    return self

  def transform(self, X, y=None):
    X_df = X.copy()
    if self.corr is None:
      raise ValueError("Transformer has not been fitted yet.")

    X_df['P_weighed_sum'] = X_df[self.cols].mul(self.corr).sum(axis=1)
    return X_df[['P_weighed_sum']]


class OpenDate_Transfomer(BaseEstimator, TransformerMixin):

  def __init__(self, date_col = 'Open Date'):
    self.date_col = date_col

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X_df = X.copy()
    X_df[self.date_col] = pd.to_datetime(X_df[self.date_col])

    X_df['day'] = X_df[self.date_col].dt.day
    X_df['month'] = X_df[self.date_col].dt.month
    X_df['year'] = X_df[self.date_col].dt.year

    X_df['day_sin'] = np.sin(X_df[self.date_col].dt.day*2*np.pi / 31)
    X_df['day_cos'] = np.cos(X_df[self.date_col].dt.day*2*np.pi / 31)
    X_df['month_sin'] = np.sin(X_df[self.date_col].dt.month*2*np.pi / 12)
    X_df['month_cos'] = np.cos(X_df[self.date_col].dt.month*2*np.pi / 12)

    return X_df[['month_cos','day_cos', 'year']]


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline



p_cols = ['P'+str(i) for i in range(1,38)]
categorical_features = ['City', 'City Group', 'Type']
date_col = ['Open Date']
year_col = ['year']
output_col = ['revenue']



# The full pipeline

features_pipeline = Pipeline(
    steps=[
        ('preprocessor', ColumnTransformer(
              transformers=[
                  ('p_weighted', PWeighedSum_Transformer(), p_cols),
                  ('categorical', OneHotEncoder(sparse_output=False, drop ='first',handle_unknown='ignore'), categorical_features),
                  ('open_date', OpenDate_Transfomer(), date_col),
              ],
              remainder='drop',
          )),
        ('year_scaled', ColumnTransformer(
              transformers=[
                  ('year_scaled', StandardScaler(), [-1])
              ],
              remainder='passthrough',
          )
        )
    ]
)

output_pipeline = Pipeline(
    steps=[
        ('scaler', StandardScaler())
    ]
)



def get_feature_names(pipeline, categorical_columns):
  cat_transformer = pipeline.named_steps['preprocessor'].named_transformers_['categorical']
  cat_columns = cat_transformer.get_feature_names_out(categorical_columns).tolist()
  return ['P_weighed_sum'] + cat_columns + ['month_cos', 'day_cos']


X = train_df.drop('revenue', axis=1)
y = train_df['revenue']

X_preprocessed = features_pipeline.fit_transform(X, y)
y_preprocessed = output_pipeline.fit_transform(y.values.reshape(-1,1))


feature_names = get_feature_names(features_pipeline, categorical_features)

# Move years in the front because StandardScaler move the scaled column in the beginning
features_names = ['year_scaled'] + feature_names

X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=features_names)
X_preprocessed_df


X_test_preprocessed = features_pipeline.transform(test_df)
X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=features_names)
X_test_preprocessed_df


y_preprocessed_df = pd.DataFrame(y_preprocessed, columns=['revenue_scaled'])
y_preprocessed_df


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV


# models = {
#     'Linear Regression': LinearRegression(),
#     'Ridge': Ridge(alpha=1.0),
#     'Lasso': Lasso(alpha=1.0),
#     'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
#     'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
#     'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
#     'SVR': SVR(kernel='rbf', C=1.0),
#     'KNN': KNeighborsRegressor(n_neighbors=5),
#     'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
#     'CatBoost': cb.CatBoostRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbose=0),
#     'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
# }


# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))

# rmse_scorer = make_scorer(rmse, greater_is_better=False)

# cv_folds = 5
# scoring = 'r2'


# results = {}
# for name, model in models.items():
#     # R² scores
#     r2_scores = cross_val_score(model, X_preprocessed_df, y_preprocessed_df.values.reshape(-1), cv=cv_folds, scoring=scoring)
#     # RMSE scores
#     rmse_scores = cross_val_score(model, X_preprocessed_df, y_preprocessed_df.values.reshape(-1), cv=cv_folds, scoring=rmse_scorer)

#     results[name] = {
#         'R2 Mean': np.mean(r2_scores),
#         'R2 Std': np.std(r2_scores),
#         'RMSE Mean': -np.mean(rmse_scores),  # Negate to get positive RMSE
#         'RMSE Std': np.std(rmse_scores)
#     }


# results_df = pd.DataFrame(results).T
# results_df.sort_values(by='RMSE Mean')


# lgb_model = lgb.LGBMRegressor(random_state=42)

# param_grid = {
#     'n_estimators': [50, 100, 200],           # Number of trees
#     'max_depth': [2, 3, 5, -1],               # Max depth (-1 means no limit, but careful with small data)
#     'learning_rate': [0.01, 0.05, 0.1, 0.3],  # Learning rate
#     'num_leaves': [7, 15],                    # Max leaves in one tree (small for small data)
#     'min_child_samples': [5, 10, 20],         # Min samples in a leaf (prevent overfitting)
#     'subsample': [0.6, 0.8, 1.0],             # Fraction of data per tree
#     'colsample_bytree': [0.6, 0.8, 1.0],      # Fraction of features per tree
#     'reg_alpha': [0, 0.1, 1.0],               # L1 regularization
#     'reg_lambda': [0, 0.1, 1.0]               # L2 regularization
# }

# grid_search = GridSearchCV(
#     estimator=lgb_model,
#     param_grid=param_grid,
#     cv=4,
#     scoring=rmse_scorer,
#     n_jobs=-1,
#     verbose=1
# )


# grid_search.fit(X_preprocessed_df, y_preprocessed_df.values.reshape(-1))


# print("Best Parameters:", grid_search.best_params_)
# print("Best RMSE (CV):", -grid_search.best_score_)


best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'max_depth': -1,
    'min_child_samples': 5,
    'n_estimators': 50,
    'num_leaves': 15,
    'reg_alpha': 0,
    'reg_lambda': 0.1,
    'subsample': 0.6
    }

best_model = lgb.LGBMRegressor(**best_params, random_state=42)
best_model.fit(X_preprocessed_df, y_preprocessed_df.values.reshape(-1))


y_pred_scaled = best_model.predict(X_test_preprocessed_df)
y_pred = output_pipeline.inverse_transform(y_pred_scaled.reshape(-1,1))

y_pred


submission = pd.DataFrame({'Id': test_df.index, 'Prediction': y_pred.flatten()})
submission.to_csv('submission2.csv', index=False)

