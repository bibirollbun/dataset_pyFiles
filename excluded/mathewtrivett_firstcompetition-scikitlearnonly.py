!pip install --upgrade scikit-learn


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from fastai.tabular.all import *

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])


train_df.head()


train_df.describe()


train_df.describe(include='object')


train_df.isna().sum()


plt.figure(figsize=(15, 7))

sns.lineplot(
    train_df,
    x='date',
    y='num_sold',
    estimator='sum',
    errorbar=None,
)
plt.show()


plt.figure(figsize=(15, 7))
sns.lineplot(
    train_df,
    x='date',
    y='num_sold',
    hue='product',
    estimator='sum',
    errorbar=None,
)
plt.title('Sales by product')
plt.show()


plt.figure(figsize=(15, 7))
sns.lineplot(
    train_df,
    x='date',
    y='num_sold',
    hue='store',
    estimator='sum',
    errorbar=None,
)
plt.title('Sales by store')
plt.show()


plt.figure(figsize=(15, 7))
sns.lineplot(
    train_df,
    x='date',
    y='num_sold',
    hue='country',
    estimator='sum',
    errorbar=None,
)
plt.title('Sales by country')
plt.show()


sns.histplot(
    train_df,
    x='country',
    stat='count',
)
plt.title('Count of rows by country')
plt.show()


sns.histplot(
    train_df,
    x='product',
    stat='count',
)
plt.title('Count of rows by product')
plt.show()


sns.histplot(
    train_df,
    x='store',
    stat='count',
)
plt.title('Count of rows by store')
plt.show()


from sklearn.base import BaseEstimator, TransformerMixin

class AddDateParts(BaseEstimator, TransformerMixin):
    """Adds dateparts by calling the `add_datepart`_ function from fast.ai

    .. _add_datepart:
       https://docs.fast.ai/tabular.core.html#add_datepart
    """
    def __init__(
        self, 
        date_col: str, 
        drop: bool = False, 
        time: bool = False,
    ):
        self.date_col = date_col
        self.drop = drop
        self.time = time

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return add_datepart(
            X, 
            field_name=self.date_col,
            drop=self.drop, 
            time=self.time,
        )


from sklearn.base import BaseEstimator, TransformerMixin

class DayInMonthTransfomer(BaseEstimator, TransformerMixin):
    """Calculates cyclic features for day in month 
    considering the number of days in the month.
    """
    def __init__(self, date_col: str):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['day_sin'] = np.sin(self.__radians(X))
        X['day_cos'] = np.cos(self.__radians(X))
        return X

    def __radians(self, X):
        days_in_month = X[self.date_col].dt.days_in_month
        day_of_month = X[self.date_col].dt.day
        return 2 * np.pi * day_of_month / days_in_month
    


from sklearn.preprocessing import FunctionTransformer

def sin_transformer(period: int):
    """Flexible function transformer to calculate sine cyclic features for a given period
    """
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period: int):
    """Flexible function transformer to calculate cosine cyclic features for a given period
    """
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def to_float():
    """Converts the values to a float
    """
    return FunctionTransformer(lambda x: np.array(x, dtype=float))



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

preprocessor = Pipeline(
    steps=[
        ('add_dateparts', AddDateParts(date_col='date')),
        ('column_tranforms', ColumnTransformer(
            transformers=[
                ('dow_sin', sin_transformer(period=7), ['Dayofweek']),
                ('dow_cos', cos_transformer(period=7), ['Dayofweek']),
                ('week_sin', sin_transformer(period=52), ['Week']),
                ('week_cos', cos_transformer(period=52), ['Week']),
                ('month_sin', sin_transformer(period=12), ['Month']),
                ('month_cos', cos_transformer(period=12), ['Month']),
                ('day_of_year_sin', sin_transformer(period=365), ['Dayofyear']),
                ('day_of_year_cos', cos_transformer(period=365), ['Dayofyear']),
                ('day_sin', DayInMonthTransfomer(date_col='date'), []),
                ('day_cos', DayInMonthTransfomer(date_col='date'), []),
                ('day_spline', SplineTransformer(n_knots=6, degree=3, include_bias=False), ['Day']),
                ('dow_spline', SplineTransformer(n_knots=3, degree=3, include_bias=False), ['Dayofweek']),
                ('week_spline', SplineTransformer(n_knots=6, degree=3, include_bias=False), ['Week']),
                ('month_spline', SplineTransformer(n_knots=6, degree=3, include_bias=False), ['Month']),
                ('doy_spline',  SplineTransformer(n_knots=12, degree=3, include_bias=False), ['Dayofyear']),
                ('numericise_date_parts', to_float(), ['Day','Month','Week','Year','Dayofweek','Dayofyear']),
                ('cat', OneHotEncoder(), ['country', 'store', 'product'])
            ],
            remainder='drop',
        )),
    ]
)


from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

def evaluate(preprocessor, model, X, y, n_splits=5):
    """Cross validation wrapper
    """
    # Constructs a rolling window timeseries split
    tscv = TimeSeriesSplit(
        n_splits=5,
        test_size=5000,
    )
    results = []

    # Constructs a new composite pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model),
        ]
    )
    
    # Train and evalute the model for each split
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = mean_absolute_percentage_error(y_test, y_pred)
        results.append(score)
        print(f"Score {score}")
    return np.array(results), pipeline


train_df = train_df.dropna()


X, y = train_df.drop('num_sold', axis=1), train_df['num_sold']


from sklearn.ensemble import HistGradientBoostingRegressor

hgbr = HistGradientBoostingRegressor()


score, pipeline = evaluate(preprocessor, hgbr, X, y)
print(score.mean())


from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

hgbr = HistGradientBoostingRegressor()

grid_search_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', hgbr)
    ]
)

tscv = TimeSeriesSplit(
    n_splits=3,
    test_size=10000,
)

param_grid = {
    'model__learning_rate': [0.1],
    'model__max_iter': [250, 500, 1000],
}

grid_search = GridSearchCV(
    grid_search_pipeline, 
    param_grid,
    verbose=2,
    cv=tscv, 
    scoring='neg_mean_absolute_percentage_error',
)

grid_search.fit(X, y)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
best_model = grid_search.best_estimator_


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()


scores, pipeline = evaluate(preprocessor, dtr, X, y)
print(scores.mean())


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()


scores, pipeline = evaluate(preprocessor, rfr, X, y)
print(scores.mean())


from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

grid_search_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', rfr)
    ]
)

tscv = TimeSeriesSplit(
    n_splits=3,
    test_size=10000,
)

param_grid = {
    'model__n_estimators': [100],
    'model__max_depth': [15],
}

grid_search = GridSearchCV(
    grid_search_pipeline, 
    param_grid,
    verbose=2,
    cv=tscv, 
    scoring='neg_mean_absolute_percentage_error',
)

grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
best_model = grid_search.best_estimator_


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

rfr = RandomForestRegressor(
    max_depth=25,
    n_estimators=100,
)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', rfr),
    ]
)

pipeline.fit(X,y)





y_pred = pipeline.predict(test_df)


test_df['num_sold'] = y_pred


test_df.head()


plt.figure(figsize=(15, 7))
sns.lineplot(
    test_df,
    x='date',
    y='num_sold',
    hue='product',
    estimator='sum',
    errorbar=None,
)
plt.title('Sales by country')
plt.show()


sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


sample_submission_df.head()


submission = pd.DataFrame({
    "id": test_df['id'],
    "num_sold": y_pred
})

submission.head()


submission.to_csv('submission.csv', index=False)

