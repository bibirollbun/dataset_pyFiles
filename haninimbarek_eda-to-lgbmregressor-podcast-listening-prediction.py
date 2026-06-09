import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

train.drop(columns='id',inplace=True)
test.drop(columns='id',inplace=True)


train = train.dropna()
train['Number_of_Ads'] = train['Number_of_Ads'].apply(lambda x: min(x, 3))
Q1, Q3 = train["Episode_Length_minutes"].quantile([0.25, 0.75])
IQR = (Q3 - Q1) + 1.5 * Q3
train.loc[train["Episode_Length_minutes"] >= IQR, "Episode_Length_minutes"] = Q3


test['Number_of_Ads'] = test['Number_of_Ads'].apply(lambda x: min(x, 3))
Q1,Q3 = test["Episode_Length_minutes"].quantile([0.25,0.75])
IQR = (Q3 - Q1) + 1.5* Q3
test.loc[test["Episode_Length_minutes"] >= IQR, "Episode_Length_minutes"] = Q3


train_subset = train.sample(n=50000, random_state=42)


# Split the data
X = train_subset.drop("Listening_Time_minutes", axis=1)
y = train_subset["Listening_Time_minutes"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_val.shape


# Define target encoders
class TargetEncodingWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = TargetEncoder(cols=cols)

    def fit(self, X, y):
        self.encoder.fit(X[self.cols], y)
        return self

    def transform(self, X):
        X_copy = X.copy()
        encoded = self.encoder.transform(X_copy[self.cols])
        X_copy[self.cols] = encoded
        return X_copy


# Selectors
num_selector = make_column_selector(dtype_include='number')
low_cardinality = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
high_cardinality = ['Podcast_Name', 'Episode_Title']


# create a preprocessor
preprocessor = make_column_transformer(
    (SimpleImputer(), num_selector),
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_cardinality),
    remainder="drop"
)

# create pipeline
pipeline = make_pipeline(
    TargetEncodingWrapper(high_cardinality),# target encoding
    preprocessor,
    LGBMRegressor(random_state=42)
)


# Hyperparameter grid for LightGBM
param_grid = {
    'lgbmregressor__n_estimators': [120,150],
    'lgbmregressor__learning_rate': [0.07,0.09],
    'lgbmregressor__max_depth': [8,11],
    'lgbmregressor__num_leaves': [15,20],
    'lgbmregressor__min_child_samples': [30,40],
    'lgbmregressor__subsample': [0.8, 1.0],
    'lgbmregressor__colsample_bytree': [0.8, 1.0]
}


# Grid Search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train,y_train)


# Evaluation
y_pred = grid_search.predict(X_val)

print("Best Parameters:", grid_search.best_params_)
print("Best Train Score:", grid_search.best_score_)
print("Val R2 Score:", r2_score(y_val, y_pred))
print("RMSE:", mean_squared_error(y_val, y_pred, squared=False))


# Extract the trained LGBMRegressor from the pipeline
best_LGBMRegressor = grid_search.best_estimator_
lgb_model = best_LGBMRegressor.named_steps['lgbmregressor']

# Get feature importances from the LGBMRegressor model
feature_importance = lgb_model.feature_importances_

# Get transformed feature names after preprocessing
feature_names = best_LGBMRegressor.named_steps['columntransformer'].get_feature_names_out()

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("LGBMRegressor Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Split the data
X = train.drop("Listening_Time_minutes", axis=1)
y = train["Listening_Time_minutes"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_val.shape


model = grid_search.best_estimator_
model.fit(X_train,y_train)


# Evaluation
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

print("Train R2 Score:", r2_score(y_train, y_pred_train))
print("Val R2 Score:", r2_score(y_val, y_pred_val))
print("Train RMSE:", mean_squared_error(y_train, y_pred_train, squared=False))
print("Val RMSE:", mean_squared_error(y_val, y_pred_val, squared=False))


y_pred = model.predict(test)
submission['Listening_Time_minutes'] = y_pred
submission.to_csv("submission.csv", index=False)
submission.head()

