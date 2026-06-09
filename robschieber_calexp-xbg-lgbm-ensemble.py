import os

import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgbm
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
try:
  from google.colab import userdata, data_table
  os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
  os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
except Exception:
  pass

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)



#BEST_XGB_PARAMS = {'learning_rate': 0.04, 'max_depth': 7, 'n_estimators': 280, 'reg_alpha': 0.5, 'reg_lambda': 0.5}
BEST_XGB_PARAMS = {'learning_rate': 0.04, 'max_depth': 7, 'n_estimators': 320, 'reg_alpha': 0.5, 'reg_lambda': 0.5}
BEST_LGBM_PARAMS = {'reg_lambda': 0, 'reg_alpha': 0.1, 'num_leaves': 20, 'n_estimators': 320, 'min_child_samples': 30, 'max_depth': 6, 'learning_rate': 0.07}
BEST_LGBM_PARAMS = {'reg_lambda': 0, 'reg_alpha': 0.1, 'num_leaves': 20, 'n_estimators': 320, 'min_child_samples': 30, 'max_depth': 6, 'learning_rate': 0.07}


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_log_error

def get_data(datafile):
  data = pd.read_csv(datafile)

  # feature engineering
  data = pd.read_csv(datafile)
  data['Sex'] = (data['Sex'] == 'female').astype('int8')
  data['Duration_HR'] = (data['Duration'] * data['Heart_Rate']).astype('float32')
  data['Body_Temp_Duration'] = (data['Body_Temp'] * data['Duration']).astype('float32')
  data['Age_Duration'] = (data['Age'] * data['Duration']).astype('float32')
  data['Sex_Duration'] = (data['Sex'] * data['Duration']).astype('float32')
  data['HR_Age'] = (data['Heart_Rate'] * data['Age']).astype('float32')  # New interaction
  data['Intensity'] = (data['Heart_Rate'] / data['Age']).astype('float32')  # Proxy for exercise intensity

  for col in ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']:
      if col in data:
          data[col] = data[col].astype('float32')

  return data

train_data = get_data('/kaggle/input/playground-series-s5e5/train.csv')
test_data = get_data('/kaggle/input/playground-series-s5e5/test.csv')



#numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Duration_HR', 'BMI', 'Body_Temp_Duration']
#categorical_features = ['Sex', 'Age_Group']

X = train_data.drop(['id', 'Calories' ], axis=1)
y = train_data['Calories']

# Log-transform target (add 1 to avoid log(0))
y_log = np.log1p(y)  # log(1 + y)
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Train XGBoost on log-transformed target
best_xgb_params = {'random_state': 42,
                   'learning_rate': 0.04,
                   'max_depth': 7, 'n_estimators': 280, 'reg_alpha': 0.5, 'reg_lambda': 0.5}

xgb_model = XGBRegressor(**BEST_XGB_PARAMS)
xgb_model.fit(X_train, y_train_log)

# Predict and inverse-transform
y_pred_log = xgb_model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # exp(y_pred_log) - 1
y_test = np.expm1(y_test_log)


# Evaluate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_test, np.clip(y_pred, 0, None)))
print(f"XGBoost RMSLE: {rmsle}")

# Feature importance
importance = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("Feature Importance:\n", importance)

# Submission
X_test_final = test_data.drop(['id'], axis=1)
y_pred_final_log = xgb_model.predict(X_test_final)
y_pred_final = np.expm1(y_pred_final_log)
y_pred_final = np.clip(y_pred_final, 0, None)
submission = pd.DataFrame({'id': test_data['id'], 'Calories': y_pred_final})
submission.to_csv('submission_xgb_rmsle.csv', index=False)



from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# RMSLE scorer
def rmsle_scorer(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    return np.sqrt(mean_squared_log_error(np.clip(y_true, 0, None), np.clip(y_pred, 0, None)))

rmsle = make_scorer(rmsle_scorer, greater_is_better=False)

# CV
scores = cross_val_score(xgb_model, X, y_log, cv=5, scoring=rmsle)
print("XGBoost CV RMSLE:", -scores.mean())


def finetune_xgb():
    # Subsample (10% = 70K rows)
    X_train_sub, _, y_train_sub, _ = train_test_split(X, y_log, train_size=0.1, random_state=42)
    
    # Tune
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [280, 300, 320],
        'max_depth': [7, 8, 9],
        'learning_rate': [0.04, 0.05, 0.06],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring=rmsle, n_jobs=-1)
    grid.fit(X_train_sub, y_train_sub)
    print("Best XGBoost parameters:", grid.best_params_)
    print("Best XGBoost CV RMSLE:", -grid.best_score_)


from sklearn.metrics import mean_squared_log_error, make_scorer

lgb_model = lgbm.LGBMRegressor(**BEST_LGBM_PARAMS)
lgb_model.fit(X_train, y_train_log)

# Evaluate on test set
y_pred_log = lgb_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test = np.expm1(y_test_log)
rmsle_test = np.sqrt(mean_squared_log_error(y_test, np.clip(y_pred, 0, None)))
print(f"LightGBM Test RMSLE: {rmsle_test}")

# Cross-validation
scores = cross_val_score(lgb_model, X, y_log, cv=5, scoring=rmsle)
print("LightGBM CV RMSLE:", -scores.mean())

# Submission
y_pred_final_log = lgb_model.predict(X_test_final)
y_pred_final = np.expm1(y_pred_final_log)
y_pred_final = np.clip(y_pred_final, 0, None)
submission = pd.DataFrame({'id': test_data['id'], 'Calories': y_pred_final})
submission.to_csv('submission_lgb_rmsle.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, train_test_split

def finetune_light_gbm():
    # Subsample (5% = 35K rows for faster tuning)
    X_train_sub, _, y_train_sub, _ = train_test_split(X, y_log, train_size=0.05, random_state=42)
    
    # Parameter distribution (RandomizedSearchCV for speed)
    param_dist = {
        'n_estimators': [280, 300, 320, 340],
        'max_depth': [6, 7, 8],
        'learning_rate': [0.05, 0.06, 0.07],
        'num_leaves': [20, 25, 30],
        'min_child_samples': [10, 20, 30],  # Lowered for more splits
        'reg_alpha': [0, 0.1],  # Lighter regularization
        'reg_lambda': [0, 0.1]
    }
    
    # Tune with RandomizedSearchCV
    lgb = lgbm.LGBMRegressor(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(lgb, param_dist, n_iter=20, cv=5, scoring=rmsle, n_jobs=-1, random_state=42)
    random_search.fit(X_train_sub, y_train_sub)
    print("Best LightGBM parameters:", random_search.best_params_)
    print("Best LightGBM CV RMSLE:", -random_search.best_score_)


# Full training with best params
lgb_model = lgbm.LGBMRegressor(**BEST_LGBM_PARAMS, random_state=42, n_jobs=-1)
lgb_model.fit(
    X_train, y_train_log,
    eval_set=[(X_test, y_test_log)],
    eval_metric='rmse',
    callbacks=[lgbm.early_stopping(stopping_rounds=10)]
)
y_pred_log = lgb_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test = np.expm1(y_test_log)
rmsle = np.sqrt(mean_squared_log_error(y_test, np.clip(y_pred, 0, None)))
print(f"Tuned LightGBM Test RMSLE: {rmsle}")

# Submission
y_pred_final_log = lgb_model.predict(X_test_final)
y_pred_final = np.expm1(y_pred_final_log)
y_pred_final = np.clip(y_pred_final, 0, None)
submission = pd.DataFrame({'id': test_data['id'], 'Calories': y_pred_final})
submission.to_csv('submission_lgb_tuned_rmsle.csv', index=False)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = train_data.select_dtypes(include=np.number).columns.to_list()
numeric_features.remove('id')
numeric_features.remove('Sex')
numeric_features.remove('Calories')
categorical_features = ['Sex']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# LGBM
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lgbm.LGBMRegressor(**BEST_LGBM_PARAMS, random_state=42, n_jobs=-1))
])

pipeline.fit(X_train, y_train_log)
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
rmsle = np.sqrt(mean_squared_log_error(y_test, np.clip(y_pred, 0, None)))

print("Pipeline RMSLE:", rmsle)

# Submission
y_pred_final_log = pipeline.predict(test_data)
y_pred_final = np.expm1(y_pred_final_log)
y_pred_final = np.clip(y_pred_final, 0, None)
submission = pd.DataFrame({'id': test_data['id'], 'Calories': y_pred_final})
submission.to_csv('submission_pipeline_rmsle.csv', index=False)



# Train XGBoost
#xgb_model = XGBRegressor(**BEST_XGB_PARAMS, n_jobs=-1)
#xgb_model.fit(X_train, y_train_log)
y_pred_xgb_log = xgb_model.predict(X_test)
y_pred_xgb = np.expm1(y_pred_xgb_log)

# Train LightGBM
#lgb_model = lgbm.LGBMRegressor(**BEST_LGB_PARAMS, random_state=42, n_jobs=-1)
#lgb_model.fit(X_train, y_train_log)
y_pred_lgb_log = lgb_model.predict(X_test)
y_pred_lgb = np.expm1(y_pred_lgb_log)

# Ensemble (weighted average)
y_pred_ensemble = 0.6 * y_pred_lgb + 0.4 * y_pred_xgb  # Weight LightGBM higher due to better leaderboard
rmsle = np.sqrt(mean_squared_log_error(y_test, np.clip(y_pred_ensemble, 0, None)))
print(f"Ensemble RMSLE: {rmsle}")

# Submission
y_pred_final_xgb_log = xgb_model.predict(X_test_final)
y_pred_final_xgb = np.expm1(y_pred_final_xgb_log)
y_pred_final_lgb_log = lgb_model.predict(X_test_final)
y_pred_final_lgb = np.expm1(y_pred_final_lgb_log)
y_pred_final_ensemble = 0.6 * y_pred_final_lgb + 0.4 * y_pred_final_xgb
y_pred_final_ensemble = np.clip(y_pred_final_ensemble, 0, None)
submission = pd.DataFrame({'id': test_data['id'], 'Calories': y_pred_final_ensemble})
submission.to_csv('submission_ensemble_rmsle.csv', index=False)

