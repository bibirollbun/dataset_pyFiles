import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from optuna.samplers import TPESampler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from category_encoders import TargetEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost
import optuna


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.head()


train.info()


cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])


train['Compartments'] = train['Compartments'].astype(int)
test['Compartments'] = test['Compartments'].astype(int)


train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean())
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean())


TE = TargetEncoder(smoothing=20)  

for col in cat_cols:
    train[col] = TE.fit_transform(train[col], train['Price'])
    test[col] = TE.transform(test[col])


train.head()


X = train.drop(columns=['Price'])
y = train['Price']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),  # Increased limit
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),  # Lowered min for more regularization
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),  # L2 regularization
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'tree_method': 'gpu_hist',  # Enable GPU acceleration
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),  # Increased limit
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 31, 300),  # Increased upper bound
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'device': 'gpu',  # Enable GPU
        'random_state': 42
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),  # Increased limit
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
        'depth': trial.suggest_int('depth', 3, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_int('random_strength', 1, 20),  # Increased randomness for better exploration
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),  # Added bagging for better diversity
        'task_type': 'GPU',  # Enable GPU
        'random_seed': 42
    }

    model = catboost.CatBoostRegressor(**params, verbose=0)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)

    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))


#study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler())
#study_xgb.optimize(objective_xgb, n_trials=100)

#print("Best parameters for XGBoost:", study_xgb.best_params)
#print("Best RMSE for XGBoost:", study_xgb.best_value)


#study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler())
#study_lgb.optimize(objective_lgb, n_trials=10)

#print("Best parameters for LightGBM:", study_lgb.best_params)
#print("Best RMSE for LightGBM:", study_lgb.best_value)


#study_cat = optuna.create_study(direction='minimize', sampler=TPESampler())
#study_cat.optimize(objective_cat, n_trials=100)

#print("Best parameters for CatBoost:", study_cat.best_params)
#print("Best RMSE for CatBoost:", study_cat.best_value)


xgb_params = {
    'n_estimators': 819,
    'learning_rate': 0.18627202751580957,
    'max_depth': 3,
    'subsample': 0.6279814130215398,
    'colsample_bytree': 0.6063889723209597,
    'reg_alpha': 6.272824236480007,
    'reg_lambda': 6.607435556785976,
    'min_child_weight': 8,
    'tree_method': 'gpu_hist',
    'random_state': 42
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X, y)


lgb_params = {
    'n_estimators': 1557,
    'learning_rate': 0.08656127569637918,
    'max_depth': 3,
    'num_leaves': 228,
    'subsample': 0.3135660502065544,
    'colsample_bytree': 0.486020334219166,
    'reg_alpha': 3.470599982088023,
    'reg_lambda': 8.6477472607711,
    'min_child_samples': 31,
    'device': 'gpu',
    'random_state': 42
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X, y)


cat_params = {
    'iterations': 1681,
    'learning_rate': 0.1306078340791884,
    'depth': 3,
    'l2_leaf_reg': 5.07469987663452,
    'random_strength': 1,
    'border_count': 156,
    'bagging_temperature': 0.24470119655639067,
    'task_type': 'GPU',
    'random_seed': 42
}

cat_model = catboost.CatBoostRegressor(**cat_params, verbose=0)
cat_model.fit(X, y)


stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator=LinearRegression(),
    passthrough=True
)

stacking_model.fit(X, y)


test.head()


submission_ids = test['id']
pred = lgb_model.predict(test)


submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': pred
})


submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())

