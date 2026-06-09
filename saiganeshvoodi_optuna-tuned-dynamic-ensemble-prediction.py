import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor



def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


for df in [train, test]:
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['Effort_Index'] = (df['Heart_Rate'] * df['Duration']) / df['Weight']
    df['Temp_Stress_Index'] = df['Body_Temp'] * df['Heart_Rate']
    df['Adjusted_Intensity'] = df['Heart_Rate'] / df['BMI']
    df['Duration_squared'] = df['Duration'] ** 2


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


X = train.drop(columns=['id', 'Calories'])
y = train['Calories']
X_test = test.drop(columns=['id'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def lgb_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return rmsle(y_val, preds)


def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return rmsle(y_val, preds)


def cat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'random_seed': 42,
        'logging_level': 'Silent'
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    preds = model.predict(X_val)
    return rmsle(y_val, preds)



lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=30)


xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=30)


cat_study = optuna.create_study(direction='minimize')
cat_study.optimize(cat_objective, n_trials=30)


lgb_model = lgb.LGBMRegressor(**lgb_study.best_params)
xgb_model = xgb.XGBRegressor(**xgb_study.best_params)
cat_model = CatBoostRegressor(**cat_study.best_params, verbose=0)


lgb_model.fit(X, y)
xgb_model.fit(X, y)
cat_model.fit(X, y)


lgb_val = lgb_model.predict(X_val)
xgb_val = xgb_model.predict(X_val)
cat_val = cat_model.predict(X_val)

lgb_r = rmsle(y_val, lgb_val)
xgb_r = rmsle(y_val, xgb_val)
cat_r = rmsle(y_val, cat_val)


inv_rmsles = 1 / np.array([lgb_r, xgb_r, cat_r])
weights = inv_rmsles / inv_rmsles.sum()
print(f" Dynamic Weights: LGBM={weights[0]:.2f}, XGB={weights[1]:.2f}, CAT={weights[2]:.2f}")



lgb_test = lgb_model.predict(X_test)
xgb_test = xgb_model.predict(X_test)
cat_test = cat_model.predict(X_test)



final_preds = (
    weights[0] * lgb_test +
    weights[1] * xgb_test +
    weights[2] * cat_test
)


submission = pd.DataFrame({
    'id': test['id'],
    'Calories': np.maximum(0, final_preds)
})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved using Optuna-tuned blended ensemble")

