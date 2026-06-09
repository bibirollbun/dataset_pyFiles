import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

print(f'Shape of Train Data: {df.shape}')
print(f'Shape of Test Data: {test_data.shape}')


cols = df.columns
cat_cols = [col for col in cols if df[col].dtype == "object"]
num_cols = [col for col in df.columns if col not in cat_cols and col != "id"]


df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
test_data['Sex'] = test_data['Sex'].map({'male':1, 'female': 0})


y = df['Calories']
Xx = df.drop(['Calories','id'],axis = 1)
test = test_data.drop(columns=['id'],axis=1)



from sklearn.preprocessing import StandardScaler

sc =  StandardScaler()
X = sc.fit_transform(Xx)
test = sc.transform(test)


# RMSLE scorer
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)  # clip negatives
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
    
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# import optuna

# def tune_model(X, y, model_type, n_trials=50):
#     def objective(trial):
#         if model_type == 'catboost':
#             params = {
#                 'iterations': trial.suggest_int('iterations', 250, 1000),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.009, 0.3),
#                 'depth': trial.suggest_int('depth', 3, 9),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 20),
#                 'random_strength': trial.suggest_float('random_strength', 0, 10),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5),
#                 'random_state': 42,
#                 'logging_level': 'Silent'
#             }
#             model = CatBoostRegressor(**params)
        
#         elif model_type == 'xgboost':
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#                 'max_depth': trial.suggest_int('max_depth', 3, 10),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#                 'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
#                 'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
#                 'random_state': 42
#             }
#             model = XGBRegressor(**params)

#         elif model_type == 'lgbm':
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#                 'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#                 'max_depth': trial.suggest_int('max_depth', 3, 15),
#                 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#                 'random_state': 42
#             }
#             model = LGBMRegressor(**params)
        
#         elif model_type == 'random_forest':
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#                 'max_depth': trial.suggest_int('max_depth', 5, 30),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
#                 'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
#                 'random_state': 42
#             }
#             model = RandomForestRegressor(**params)

#         elif model_type == 'extra_trees':
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#                 'max_depth': trial.suggest_int('max_depth', 5, 30),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
#                 'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
#                 'random_state': 42
#             }
#             model = ExtraTreesRegressor(**params)

#         else:
#             raise ValueError("Unsupported model type! Define parameters in the function..")

#         cv = KFold(n_splits=5, shuffle=True, random_state=42)
#         rmsle_scores = []

#         for train_idx, val_idx in cv.split(X, y):
#             X_train, X_val = X[train_idx], X[val_idx]
#             y_train, y_val = y[train_idx], y[val_idx]
            
#             model.fit(X_train, y_train)
#             preds = model.predict(X_val)
#             rmsle_scores.append(rmsle(y_val, preds))
        
#         return sum(rmsle_scores) / len(rmsle_scores)

#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=n_trials)

#     return study.best_params, study.best_value



# models = ['catboost', 'xgboost', 'lgbm']

# for model_name in models:
#     print(f"\nTuning model: {model_name.upper()}")
#     best_params, best_score = tune_model(X, y, model_type=model_name, n_trials=50)
#     print(f"Best RMSLE for {model_name}: {best_score:.5f}")
#     print(f"Best hyperparameters for {model_name}:\n{best_params}")




catboost_params = {
    'iterations': 893,
    'learning_rate': 0.0850161338239067,
    'depth': 9,
    'l2_leaf_reg': 1.0140499700812582,
    'random_strength': 0.037010325792570575,
    'bagging_temperature': 0.3126279893276654
}

lgbm_params = {'n_estimators': 782,
  'learning_rate': 0.05506011323286625,
  'num_leaves': 134,
  'max_depth': 10,
  'min_child_samples': 5,
  'subsample': 0.7611260022306007,
  'colsample_bytree': 0.6952827024784423}

xgboost_params = {'n_estimators': 958,
  'learning_rate': 0.04160746509917962,
  'max_depth': 10,
  'subsample': 0.8292096170231291,
  'colsample_bytree': 0.9700241648393241,
  'reg_alpha': 0.35915888619445435,
  'reg_lambda': 7.718789110221506}


import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Define base models
base_models = [
    ("catboost", CatBoostRegressor(**catboost_params, verbose=0)),
    ("xgboost", XGBRegressor(**xgboost_params, verbosity=0)),
    ("lightgbm", LGBMRegressor(**lgbm_params))
]

# Meta-model
meta_model = LinearRegression()

# Number of folds
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for OOF predictions and test predictions
oof_train = np.zeros((X.shape[0], len(base_models)))
oof_test = np.zeros((test.shape[0], len(base_models)))

# Loop over base models
for i, (name, model) in enumerate(base_models):
    oof_test_fold = np.zeros((test.shape[0], n_folds))
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid = X[valid_idx]
        
        model.fit(X_train, y_train)
        
        # OOF train predictions
        oof_train[valid_idx, i] = model.predict(X_valid).clip(0)
        
        # Test set predictions for current fold
        oof_test_fold[:, fold] = model.predict(test).clip(0)
    
    # Average the test set predictions across folds
    oof_test[:, i] = oof_test_fold.mean(axis=1)

# Train meta-model on OOF predictions
meta_model.fit(oof_train, y)

# Final stacked predictions on test set
final_predictions = meta_model.predict(oof_test).clip(0)



submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission['Calories'] = final_predictions
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv!")





