from os import path

import numpy as np
import pandas as pd

from sklearn.model_selection import (StratifiedKFold, cross_validate, RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from scipy.stats import uniform, loguniform


BASE_PATH='/kaggle/input/playground-series-s5e8'
RNG_SEED = 42


train = pd.read_csv(path.join(BASE_PATH, 'train.csv'), index_col='id')
train.shape


y = train.loc[:, 'y']
X = train.drop('y', axis='columns')
y.shape, X.shape


all_features = X.columns.tolist()
cat_features = ['job', 'education', 'contact', 'month', 'poutcome','marital']
bin_features = ['default', 'housing', 'loan']
num_features = [x for x in all_features if x not in cat_features+bin_features]

cat_features, bin_features, num_features


plain_num_features = ['day', 'pdays', 'previous']
spline_features = [x for x in num_features if x not in plain_num_features]

preprocessor = ColumnTransformer(transformers=[
    ('spline', Pipeline([
        ('spl', SplineTransformer(include_bias=False, knots='quantile')),
        ('scl', StandardScaler()),
    ]), spline_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features+bin_features),
    ('num', StandardScaler(), plain_num_features),
])
preprocessor


# 5 folds
CV5 = 5

param_grid = {'C': loguniform(1e-4, 1e4)}

gs = RandomizedSearchCV(
    LogisticRegression(
        solver='saga',
        penalty='elasticnet',
        l1_ratio=1.0,
        tol=0.001, 
        max_iter=1000,
        random_state=RNG_SEED,
        n_jobs=-1,
    ),
    param_distributions=param_grid,
    cv=CV5,
    n_iter=50,
    scoring='roc_auc', 
    random_state=RNG_SEED,
    n_jobs=-1,
    verbose=1,
)


gsPipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gs)
])

gsPipeline.fit(X, y)


gs.best_params_, gs.best_score_


cvPipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gs.best_estimator_)
])

cvPipeline.fit(X, y)


scores = cross_validate(
    cvPipeline, X, y, cv=CV5, 
    scoring='roc_auc', 
    return_train_score=True,
    n_jobs=-1, verbose=1,
)
scores


scores['test_score'].mean(), scores['train_score'].mean()


test = pd.read_csv(path.join(BASE_PATH, 'test.csv'), index_col='id')
test.shape


test.head()


y_proba = cvPipeline.predict_proba(test)[:, 1]
y_proba


submission = pd.DataFrame({
    'id': test.index,
    'y': y_proba
})
submission


submission.to_csv('submission.csv', index=False)

