import pandas as pd
import warnings

# Ignore these warnings that are from the test set
msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)

train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col='id')


train.head()


print("---Train info---")
print(train.info())
print("\n---Test info---")
print(test.info())


train.isna().sum()





from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical


X = train.drop(columns=['Listening_Time_minutes'])
y = train[['Listening_Time_minutes']]
y_flat = y.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y_flat, test_size=0.2, random_state=42)


columns_to_drop = ['Podcast_Name', 'Episode_Title']

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
categorical_features_to_encode = [col for col in categorical_features if col not in columns_to_drop]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features_to_encode)
    ],
    remainder='drop'
)


# base_model = LinearRegression()
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', base_model)
# ], verbose=True)
# pipeline.fit(X_train, y_train)
# best_model = pipeline


# base_model = RandomForestRegressor(random_state=42)

# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', base_model)
# ], verbose=True)

# # ------------------------------
# #  
# # ------------------------------
# search_spaces = {
#     'regressor__n_estimators': Categorical([50, 100, 150]),
#     'regressor__max_depth': Categorical([None, 5, 10, 15]),
#     'regressor__min_samples_split': Categorical([2, 5, 10]),
#     'regressor__min_samples_leaf': Categorical([1, 2, 4])
# }

# bayes_search = BayesSearchCV(
#     estimator=pipeline,
#     search_spaces=search_spaces,
#     n_iter=8,
#     scoring='neg_root_mean_squared_error',
#     cv=3,
#     random_state=42,
#     n_jobs=-1,
#     verbose=2
# )


# # Fit the model
# print("Starting pipeline fitting")
# bayes_search.fit(X_train, y_train)
# best_model = bayes_search.best_estimator_
# print("\nBest parameters found by BayesSearchCV:")
# print(bayes_search.best_params_)


# from catboost import CatBoostRegressor

# catboost_model = CatBoostRegressor(
#     iterations=500,          # Number of boosting iterations (trees)
#     learning_rate=0.05,      # Step size shrinkage
#     depth=6,                 # Depth of the trees
#     l2_leaf_reg=3,           # L2 regularization coefficient
#     loss_function='RMSE',    # Objective function for regression (Root Mean Squared Error)
#     eval_metric='RMSE',      # Metric used for evaluation during training (if early stopping is used)
#     random_state=42,         # For reproducibility
#     verbose=100                # Set to 100 to see progress every 100 iterations, 0 for silent
# )

# pipeline_catboost = Pipeline(steps=[
#     ('preprocessing', preprocessor),
#     ('regressor', catboost_model)
# ])

# pipeline_catboost.fit(X_train, y_train)
# best_model = pipeline_catboost


from xgboost import XGBRegressor

xgboost_model = XGBRegressor(
    n_estimators=4000,        # Number of boosting rounds (trees)
    learning_rate=0.05,      # Step size shrinkage (eta)
    max_depth=6,             # Maximum depth of a tree
    reg_lambda=3,            # L2 regularization term on weights (equivalent to CatBoost's l2_leaf_reg)
    objective='reg:squarederror', # Specify regression task with squared error objective
    eval_metric='rmse',      # Evaluation metric
    random_state=42,         # For reproducibility
    verbosity=2              # Set verbosity level (0 = silent, 1 = warning, 2 = info, 3 = debug)
    # Note: XGBoost can benefit from other parameters like subsample, colsample_bytree, etc.
    # tree_method='hist' is often faster for larger datasets.
)

# Combine the *same* preprocessor with the new XGBoost model
pipeline_xgboost = Pipeline(steps=[
    ('preprocessing', preprocessor), # Use the SAME preprocessor instance
    ('regressor', xgboost_model)     # Use the XGBoost model
])

pipeline_xgboost.fit(X_train, y_train)
best_model = pipeline_xgboost


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import numpy as np


# 1. Evaluate the final_model on the test set
test_score = best_model.score(X_test, y_test)
print(f"Test Score (R2): {test_score:.4f}")

# 2. Calculate RMSE on the test set
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test RMSE: {rmse:.4f}")


# Submission before evaluation to avoid invalid submission when error on evalutaion
sub["Listening_Time_minutes"] = best_model.predict(test)
sub.to_csv("submission.csv")
# Check the head of submission if it loooks good enough
print(sub.head())

