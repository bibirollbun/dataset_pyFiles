import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import shap
import warnings

warnings.filterwarnings("ignore")


RANDOM_STATE = 42


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


for df in [train, test]:
    df['Height_m'] = df['Height'] / 100
    df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['Heart_Temp_Interaction'] = df['Heart_Rate'] * df['Body_Temp']
    df.drop('Height_m', axis=1, inplace=True)
    df['Sex'] = df['Sex'].astype('category')



le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


X = train.drop(columns=['Calories', 'id'])
y = train['Calories']
X_test = test.drop(columns=['id'])



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)



def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


lgb_model = lgb.LGBMRegressor(random_state=RANDOM_STATE)
lgb_model.fit(X_train, y_train)
lgb_rmsle = rmsle(y_val, lgb_model.predict(X_val))



xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE)
xgb_model.fit(X_train, y_train)
xgb_rmsle = rmsle(y_val, xgb_model.predict(X_val))


cat_model = CatBoostRegressor(
    iterations=300, learning_rate=0.1, depth=6,
    early_stopping_rounds=20, verbose=100, random_state=RANDOM_STATE
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
cat_rmsle = rmsle(y_val, cat_model.predict(X_val))



model_scores = {
    'LightGBM': lgb_rmsle,
    'XGBoost': xgb_rmsle,
    'CatBoost': cat_rmsle
}
best_model_name = min(model_scores, key=model_scores.get)
print("Model RMSLEs:", model_scores)
print("Best Model:", best_model_name)


if best_model_name == 'LightGBM':
    tuned_model = lgb.LGBMRegressor(random_state=RANDOM_STATE)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1]
    }
elif best_model_name == 'XGBoost':
    tuned_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1]
    }
else:  # CatBoost
    tuned_model = CatBoostRegressor(verbose=0, random_state=RANDOM_STATE)
    param_grid = {
        'iterations': [100, 200],
        'depth': [5, 10],
        'learning_rate': [0.05, 0.1]
    }

grid = GridSearchCV(tuned_model, param_grid, scoring=rmsle_scorer, cv=3, n_jobs=-1, verbose=1)
grid.fit(X, y)
final_model = grid.best_estimator_


if best_model_name in ['LightGBM', 'XGBoost']:
    explainer = shap.Explainer(final_model)
    shap_values = explainer(X)
elif best_model_name == 'CatBoost':
    explainer = shap.TreeExplainer(final_model, model_output='raw')
    shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, plot_type="bar", show=True)
shap.summary_plot(shap_values, X, show=True)


final_preds = np.maximum(0, final_model.predict(X_test))


submission = pd.DataFrame({'id': test['id'], 'Calories': final_preds})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved")

