import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

!pip install --upgrade scikit-learn==1.6.1 --quiet
import sklearn
sklearn.set_config(transform_output="pandas")

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from joblib import parallel_backend

!pip install lightgbm==4.6.0 --quiet
import lightgbm as lgb



import warnings
warnings.filterwarnings("ignore", message=".*invalid value encountered in.*", category=RuntimeWarning)


def load_data(**kwargs):
    return pd.read_csv('../input/playground-series-s5e4/train.csv', **kwargs), pd.read_csv('../input/playground-series-s5e4/test.csv', **kwargs)

train, test = load_data(index_col = 'id')


X_train = train.drop(columns = 'Listening_Time_minutes')
y_train = train['Listening_Time_minutes']


from sklearn.impute import SimpleImputer

preprocessor = make_column_transformer(
    (make_pipeline(FunctionTransformer(lambda X: X.clip(5, 120)),
                   SimpleImputer(strategy='median', add_indicator=True)), ['Episode_Length_minutes']),
verbose_feature_names_out=False)


from sklearn.linear_model import LinearRegression

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X_train, y_train)


model['regressor'].feature_names_in_


model['regressor'].intercept_, model['regressor'].coef_


# Training score
y_pred = model.predict(X_train)
root_mean_squared_error(y_train, y_pred), r2_score(y_train, y_pred)


# CV-scores
with parallel_backend('threading', n_jobs=-1):
    cv_scores = -cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)

pd.Series(cv_scores).describe()


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures

preprocessor = make_pipeline(
    make_column_transformer(
        (make_pipeline(FunctionTransformer(lambda X: X.clip(5, 120)),
                       SimpleImputer(strategy='median', add_indicator=True)), ['Episode_Length_minutes']),
        (make_pipeline(FunctionTransformer(lambda X: X.where(X<=3, np.nan)),
                       SimpleImputer(strategy='most_frequent'),
                       OneHotEncoder(sparse_output=False)), ['Number_of_Ads']),
        (make_pipeline(FunctionTransformer(lambda X: X.replace({sentiment: 'Negative/Neutral' for sentiment in ['Neutral', 'Negative']})),
                       OrdinalEncoder()), ['Episode_Sentiment']),
        (make_pipeline(FunctionTransformer(lambda X: X.assign(Genre=np.where(X.isin(['True Crime', 'Technology']), 'Group1', 'Group2'))),
                       OrdinalEncoder()), ['Genre']),
    verbose_feature_names_out=False),
    PolynomialFeatures(degree=3, interaction_only=True)
)


model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X_train, y_train)


model['regressor'].feature_names_in_.shape


# Training score
y_pred = model.predict(X_train)
root_mean_squared_error(y_train, y_pred), r2_score(y_train, y_pred)


# CV-scores
with parallel_backend('threading', n_jobs=-1):
    cv_scores = -cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)

pd.Series(cv_scores).describe()


!pip install linear-tree --quiet



warnings.filterwarnings("ignore", message=".*BaseEstimator._validate_data.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)


preprocessor = make_column_transformer(
    (make_pipeline(FunctionTransformer(lambda X: X.clip(5, 120)),
                   SimpleImputer(strategy='median', add_indicator=True)), ['Episode_Length_minutes']),
    (make_pipeline(FunctionTransformer(lambda X: X.where(X<=3, np.nan)),
                   SimpleImputer(strategy='most_frequent'),
                   make_column_transformer((OneHotEncoder(sparse_output=False), ['Number_of_Ads']),
                                           ('passthrough', ['Number_of_Ads']), verbose_feature_names_out=False)), ['Number_of_Ads']),
    (OneHotEncoder(sparse_output=False), ['Episode_Sentiment']),
    (OneHotEncoder(sparse_output=False), ['Genre']),
verbose_feature_names_out=False)


from lineartree import LinearTreeRegressor

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearTreeRegressor(LinearRegression()))
])
model.fit(X_train, y_train)


model['regressor'].feature_names_in_


# Training score
y_pred = model.predict(X_train)
root_mean_squared_error(y_train, y_pred), r2_score(y_train, y_pred)


# CV-scores
with parallel_backend('threading', n_jobs=-1):
    cv_scores = -cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)

pd.Series(cv_scores).describe()


preprocessor = make_column_transformer(
    (make_pipeline(FunctionTransformer(lambda X: X.clip(5, 120)),
                   SimpleImputer(strategy='median')), ['Episode_Length_minutes']),
    (FunctionTransformer(lambda X: X.clip(20, 100)), ['Host_Popularity_percentage']),
    (make_pipeline(FunctionTransformer(lambda X: X.clip(0, 100)),
                   SimpleImputer(strategy='median')), ['Guest_Popularity_percentage']),
    (make_pipeline(FunctionTransformer(lambda X: X.where(X<=3, np.nan)),
                   SimpleImputer(strategy='most_frequent')), ['Number_of_Ads']),
    (FunctionTransformer(lambda X: X['Episode_Title'].str.extract(r'(\d+)')[0].astype('category').to_frame('Episode_Title')), ['Episode_Title']),
    (make_pipeline(OrdinalEncoder(),
                   FunctionTransformer(lambda X: X.astype('category'))), ['Podcast_Name', 'Genre', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']),
verbose_feature_names_out=False)


model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=1024, subsample=0.7, colsample_bytree=0.7, max_bin=1024,
                                    objective='regression', random_state=42, n_jobs=-1, verbose=-1, force_col_wise=True))
])
model.fit(X_train, y_train)


model['regressor'].feature_names_in_


# Training score
y_pred = model.predict(X_train)
root_mean_squared_error(y_train, y_pred), r2_score(y_train, y_pred)


# CV-scores
with parallel_backend('threading', n_jobs=-1):
    cv_scores = -cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)

pd.Series(cv_scores).describe()


y_test_pred = test.reset_index()[['id']].copy()
y_test_pred['Listening_Time_minutes'] = model.predict(test)

y_test_pred.head()


y_test_pred.to_csv('submission.csv', index=False)

