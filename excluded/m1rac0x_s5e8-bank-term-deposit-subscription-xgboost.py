import pandas as pd

# Read file
X = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col = 'id')
X_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col = 'id')
y = X.y
X.pop('y')
print(X.head())

# Get cardinality
for col in X.columns:
    if X[col].dtype == 'object':
        print(col, X[col].nunique())

# Classify columns
cate_cols = [col for col in X.columns if X[col].dtype == 'object']
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
print(X[cate_cols].head())
print(X[num_cols].head())


# Preprocessing
# Max cardinality = 12: One-hot Encoding

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

num_transformer = SimpleImputer(strategy = 'median')
cate_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(transformers = [
    ('num', num_transformer, num_cols),
    ('cate', cate_transformer, cate_cols)
])


# Modeling

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Fit XGBoost model
xgb_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.049,
                        n_jobs = 4, random_state = 1)
xgb_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])
xgb_pipeline.fit(X, y, model__verbose = False)


# Get scores

scores = cross_val_score(xgb_pipeline, X, y, cv = 5,
                        scoring = 'roc_auc')
print('Scores:', scores)
print('Mean:', scores.mean())


# Get predictions and output

preds_test = xgb_pipeline.predict(X_test)

output = pd.DataFrame({'id': X_test.index, 'y': preds_test})
output.to_csv('submission.csv', index = False)

