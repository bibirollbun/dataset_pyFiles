import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor


# import dataset
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


# Explore Data
train_df.head()


# Get an idea of how target (Calories) relates to columns
train_df.groupby('Sex')['Calories'].describe()


# Calories relation to Age
plt.scatter(x='Age', y='Calories', data=train_df)


# Calories relation to Height
plt.scatter(x='Height', y='Calories', data=train_df)


# Marking outliers to remove
train_df.query('Height < 140')
#remove id 222696, 712091


train_df.query('Height > 215')
# remove id 52405, 84554, 426190


# Calories relation to Weight
plt.scatter(x='Weight', y='Calories', data=train_df)


pd.set_option('display.max_rows', None)
train_df.query('Weight > 125')
# remove id 439957, 28136, 52405, 76593


# # Calories relation to Duration
fig = px.scatter(train_df, x='Duration', y='Calories', hover_data=['id', 'Duration', 'Calories'])
fig.show()


# Filtering outliers by id for each duration in the graph above
id_list = train_df.query('Duration < 9 and Calories > 74')['id'].tolist()
print(id_list)


id_list2 = train_df.query('Duration == 9 and Calories > 87')['id'].tolist()
print(id_list2)


id_list3 = train_df.query('Duration == 10 and Calories > 108')['id'].tolist()
print(id_list3)


id_list4 = train_df.query('Duration == 11 and Calories > 121')['id'].tolist()
print(id_list4)


id_list5 = train_df.query('Duration == 12 and Calories > 124')['id'].tolist()
print(id_list5)


id_list6 = train_df.query('Duration == 13 and Calories > 133')['id'].tolist()
print(id_list6)


id_list7 = train_df.query('Duration == 14 and Calories > 147')['id'].tolist()
print(id_list7)


id_list8 = train_df.query('Duration == 15 and Calories > 164')['id'].tolist()
print(id_list8)


id_list9 = train_df.query('Duration == 16 and (Calories > 176 or Calories < 23)')['id'].tolist()
print(id_list9)


id_list10 = train_df.query('Duration == 17 and (Calories > 166 or Calories < 28)')['id'].tolist()
print(id_list10)


id_list11 = train_df.query('Duration == 18 and Calories > 168')['id'].tolist()
print(id_list11)


id_list12 = train_df.query('Duration == 19 and Calories > 210')['id'].tolist()
print(id_list12)


id_list13 = train_df.query('Duration == 20 and Calories < 25')['id'].tolist()
print(id_list13)


id_list14 = train_df.query('Duration == 21 and Calories < 31')['id'].tolist()
print(id_list14)


id_list15 = train_df.query('Duration == 23 and Calories < 42')['id'].tolist()
print(id_list15)


id_list16 = train_df.query('Duration == 24 and (Calories > 252 or Calories < 32)')['id'].tolist()
print(id_list16)


id_list17 = train_df.query('Duration == 25 and Calories < 41')['id'].tolist()
print(id_list17)


id_list18 = train_df.query('Duration == 26 and (Calories > 271 or Calories < 62)')['id'].tolist()
print(id_list18)


id_list19 = train_df.query('Duration == 27 and (Calories > 289 or Calories < 68)')['id'].tolist()
print(id_list19)


id_list20 = train_df.query('Duration == 28 and (Calories > 295 or Calories < 65)')['id'].tolist()
print(id_list20)


id_list21 = train_df.query('Duration == 29 and (Calories > 300 or Calories < 92)')['id'].tolist()
print(id_list21)


id_list22 = train_df.query('Duration == 30 and (Calories > 300 or Calories < 89)')['id'].tolist()
print(id_list22)


# Calories relation to Heart Rate
plt.scatter(x='Heart_Rate', y='Calories', data=train_df)


# Filtering outliers by id for Heart Rate in the graph above
id_list23 = train_df.query('(Heart_Rate > 70 and Heart_Rate < 90) and Calories > 180')['id'].tolist()
print(id_list23)


# Calories relation to Body Temperature
fig = px.scatter(train_df, x='Body_Temp', y='Calories', hover_data=['id', 'Body_Temp', 'Calories'])
fig.show()


# Filtering outliers by id for each Body Temperature in the graph above
id_list24 = train_df.query('(Body_Temp > 37 and Body_Temp < 38.7) and Calories > 43')['id'].tolist()
print(id_list24)


id_list25 = train_df.query('Body_Temp == 38.7 and Calories > 71')['id'].tolist()
print(id_list25)


id_list26 = train_df.query('Body_Temp == 38.8 and Calories > 80')['id'].tolist()
print(id_list26)


id_list27 = train_df.query('Body_Temp == 38.9 and Calories > 84')['id'].tolist()
print(id_list27)


id_list28 = train_df.query('Body_Temp == 39 and Calories > 62')['id'].tolist()
print(id_list28)


id_list29 = train_df.query('Body_Temp == 39.1 and Calories > 68')['id'].tolist()
print(id_list29)


id_list30 = train_df.query('Body_Temp == 39.2 and Calories > 79')['id'].tolist()
print(id_list30)


id_list31 = train_df.query('Body_Temp == 39.3 and Calories > 87')['id'].tolist()
print(id_list31)


id_list32 = train_df.query('Body_Temp == 39.4 and Calories > 97')['id'].tolist()
print(id_list32)


id_list33 = train_df.query('Body_Temp == 39.5 and Calories > 106')['id'].tolist()
print(id_list33)


id_list34 = train_df.query('Body_Temp == 39.6 and Calories > 118')['id'].tolist()
print(id_list34)


id_list35 = train_df.query('Body_Temp == 39.7 and Calories > 133')['id'].tolist()
print(id_list35)


id_list36 = train_df.query('Body_Temp == 39.8 and Calories > 145')['id'].tolist()
print(id_list36)


id_list37 = train_df.query('Body_Temp == 40 and Calories > 202')['id'].tolist()
print(id_list37)


# Manually taken id's for Height and Weight into list : 439957, 28136, 52405, 76593, 84554, 426190, 222696, 712091
id_list38 = [439957, 28136, 52405, 76593, 84554, 426190, 222696, 712091]
print(id_list38)


# Add all lists together to be removed
lists_combined = id_list + id_list2 + id_list3 + id_list4 + id_list5 + id_list6 + id_list7 + id_list8 + id_list9 + id_list10 + id_list11 + id_list12 + id_list13 + id_list14 + id_list15 + id_list16 + id_list17 + id_list18 + id_list19 + id_list20 + id_list21 + id_list22 + id_list23 + id_list24 + id_list25 + id_list26 + id_list27 + id_list28 + id_list29 + id_list30 + id_list31 + id_list32 + id_list33 + id_list34 + id_list35 + id_list36 + id_list37 + id_list38


# Check all outliers to drop
unique_id_to_drop = set(lists_combined)
print(f'IDs to be dropped: {unique_id_to_drop}')


# Drop all outliers from combined list
train_df = train_df[train_df.id.isin(unique_id_to_drop) == False]


# Check train_df after dropping columns
train_df.info()


# Double check for missing values
train_df.isna().any()


# Drop target column from training sets for training models
X = train_df.drop('Calories', axis=1)
y = train_df['Calories']


# Split data for training model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 20)


# One hot encoding
ohe = OneHotEncoder(sparse_output=False)
SI = SimpleImputer(strategy = 'most_frequent')


# Categorical Columns
ohe_cols = ['Sex']


# Numerical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns


# One hot encoding pipeline and fill
ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


# Numerical pipeline, scale and fill
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


# Transformations to be applied for numerical and categorical
col_trans = ColumnTransformer(transformers=[
    ('ohe_pipeline', ohe_pipeline, ohe_cols),
    ('num_pipeline', num_pipeline, num_cols),
],
    remainder='passthrough')


# Preprocessing pipeline
pipeline = Pipeline(steps=[
    ('preprocessing', col_trans)
])


# Apply pipeline transformations
X_train_preprocessed = pipeline.fit_transform(X_train)
X_test_preprocessed = pipeline.transform(X_test)


# Linear Regression - Base Model
lr = LinearRegression()


# Fit model
lr.fit(X_train_preprocessed, y_train)


# Test on Validation set
y_pred_lr = lr.predict(X_test_preprocessed)


# Scoring with validation set
print("R² Score:", r2_score(y_test, y_pred_lr))
print("RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))


# Decision Tree Regressor - model 2
dtr = DecisionTreeRegressor(random_state=20)


# Best parameters after tuning
param_grid_dtr = {
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [8],
    'max_features': [None]
}


# GridsearchCV for best parameters scored on mean squared error
dtr_cv = GridSearchCV(dtr, param_grid_dtr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)


# Fit model
dtr_cv.fit(X_train_preprocessed, y_train)


# Best score
np.sqrt(-1*dtr_cv.best_score_)


# Best parameters from GridsearchCV
dtr_cv.best_params_


# Random Forest Regressor - model 3
rfr = RandomForestRegressor(random_state=20)


# Best parameters after tuning
param_grid_rfr = {
    'max_depth': [20],
    'n_estimators': [700],
    'min_samples_split': [20]
}


# GridsearchCV for best parameters scored on mean squared error
rfr_cv = GridSearchCV(rfr, param_grid_rfr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)


# Fit model
rfr_cv.fit(X_train_preprocessed, y_train)


# Best score
np.sqrt(-1*rfr_cv.best_score_)


# Best parameters
rfr_cv.best_params_


# XGB Regressor - model 4
xgb = XGBRegressor(random_state=20)


# Best parameters after tuning
param_grid_xgb = {
    'n_estimators': [500],
    'learning_rate': [0.1],
    'max_depth': [6],
    'min_child_weight': [10],
    'subsample': [1.0],
    'colsample_bytree': [0.7]
}


# GridsearchCV for best parameters scored on mean squared error
xgb_cv = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)


# Fit model
xgb_cv.fit(X_train_preprocessed, y_train)


# Best score
np.sqrt(-1*xgb_cv.best_score_)


# Best parameters
xgb_cv.best_params_


# Gradient boosting - model 5
gbr = GradientBoostingRegressor()


# Best parameters after tuning
param_grid_gbr = {
    'max_depth': [10],
    'n_estimators': [700],
    'min_samples_leaf': [25],
    'learning_rate': [0.1],
    'max_features': [0.3]
}


# GridsearchCV for best parameters scored on mean squared error
gbr_cv = GridSearchCV(gbr, param_grid_gbr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)


# Fit model
gbr_cv.fit(X_train_preprocessed, y_train)


# Best score
np.sqrt(-1*gbr_cv.best_score_)


# Best parameters
gbr_cv.best_params_


# lgbm - model 6
lgbmr = lgb.LGBMRegressor()


# Best parameters after tuning
param_grid_lgbmr = {
    'boosting': ['gbdt'],
    'data_sample_strategy': ['goss'],
    'num_leaves': [40],
    'learning_rate': [0.05],
    'n_estimators': [700]
}


# GridsearchCV for best parameters scored on mean squared error
lgbmr_cv = GridSearchCV(lgbmr, param_grid_lgbmr, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)


# Fit model
lgbmr_cv.fit(X_train_preprocessed, y_train)


# Best score
np.sqrt(-1 * lgbmr_cv.best_score_)


# Best parameters
lgbmr_cv.best_params_


# Cat boosting - model 7
cb = CatBoostRegressor(loss_function='RMSE', verbose=False)


# Best parameters after tuning
param_grid_cb ={
    'iterations': [1500],
    'depth': [10],
    'learning_rate': [0.03]
}


# GridsearchCV for best parameters scored on mean squared error
cb_cv = GridSearchCV(cb, param_grid_cb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)


# Fit model
cb_cv.fit(X_train_preprocessed, y_train)


# Best score
np.sqrt(-1 * cb_cv.best_score_)


# Best parameters
cb_cv.best_params_


# voting regressor - ensemble of best performing models
vr = VotingRegressor([('cb', cb_cv.best_estimator_),
                      ('lgbmr', lgbmr_cv.best_estimator_),
                      ('xgb', xgb_cv.best_estimator_)],
                    weights=[3,2,1])


# Fit model
vr.fit(X_train_preprocessed, y_train)


# Predict on Validation set
y_final_vr = vr.predict(X_test_preprocessed)


# Score from validation set
mean_squared_error(y_test, y_final_vr, squared=False)


# Stacking regressor - ensemble of best performing models
estimators = [
    ('cb', cb_cv.best_estimator_),
    ('lgbmr', lgbmr_cv.best_estimator_),
    ('xgb', xgb_cv.best_estimator_),
    ('gbr', gbr_cv.best_estimator_),
    ('rfr', rfr_cv.best_estimator_),
]


# Final ensemble model with stacking regressor
sr = StackingRegressor(
            estimators = estimators,
            final_estimator = vr
)


# Fit model
sr.fit(X_train_preprocessed, y_train)


# Predict on validation set
y_final_sr = sr.predict(X_test_preprocessed)


# Score on validation set
mean_squared_error(y_test, y_final_sr, squared=False)


# Preprocess final test set for submission results
final_df_preprocessing = pipeline.transform(test_df)


# Predict on preprocessed test set and create submission output as csv
y_sr = sr.predict(final_df_preprocessing)

df_output = pd.DataFrame({
    'id': test_df['id'],
    'Calories': y_sr
})

df_output.to_csv('submission.csv', index=False)

