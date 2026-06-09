!pip install scikit-learn==1.3.2



import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_log_error, mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping

import keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers




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




import math

def rmse(y_true, y_pred):
    
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists must have the same length.")
    
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)



# catboost_params = {
#     'iterations': 893,
#     'learning_rate': 0.0850161338239067,
#     'depth': 9,
#     'l2_leaf_reg': 1.0140499700812582,
#     'random_strength': 0.037010325792570575,
#     'bagging_temperature': 0.3126279893276654
# }

# lgbm_params = {'n_estimators': 782,
#   'learning_rate': 0.05506011323286625,
#   'num_leaves': 134,
#   'max_depth': 10,
#   'min_child_samples': 5,
#   'subsample': 0.7611260022306007,
#   'colsample_bytree': 0.6952827024784423}

# xgboost_params = {'n_estimators': 958,
#   'learning_rate': 0.04160746509917962,
#   'max_depth': 10,
#   'subsample': 0.8292096170231291,
#   'colsample_bytree': 0.9700241648393241,
#   'reg_alpha': 0.35915888619445435,
#   'reg_lambda': 7.718789110221506}


xgboost_params = {
    'tree_method': 'hist',
    'n_estimators': 3000,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'verbosity': 0,
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'n_jobs': -1,
    'max_depth': 8,
    'min_child_weight': 10,
    'subsample': 0.8260966788901262,
    'reg_alpha': 0.27469472188551974,
    'reg_lambda': 0.5613776857654753,
    'colsample_bytree': 0.7965527339281658,
    'learning_rate': 0.01
}

catboost_params = {
    'verbose': 0,
    'random_state': 42,
    'eval_metric': 'RMSE',
    'n_estimators': 5000,
    'objective': 'RMSE',
    'learning_rate': 0.01,
}

lgbm_params = {
    'random_state': 42,
    'verbose': -1,
    'boosting_type': 'gbdt',
    'n_estimators': 5000,
    'eval_metric': 'rmse',
    'objective': 'regression_l2',
    'learning_rate': 0.01,
    'max_depth': 10,
    'num_leaves': 928,
    'min_child_samples': 8,
    'min_child_weight': 18,
    'colsample_bytree': 0.4009405711855729,
    'reg_alpha': 0.22713546532680443,
    'reg_lambda': 0.6266447966186705
}


class LinearRegression:
    def __init__(self):
        self.b0 = 0  # Intercept
        self.b1 = None  # For slope(s), can be float or list for multiple features
    
    def fit(self, X, y):
        """
        Fit the linear regression model.
        Supports simple (1D) or multiple (2D) features.
        
        X: list of lists or 2D array (samples x features) or 1D list for simple linear regression
        y: list or 1D array (target values)
        """
        # Convert X to 2D if 1D
        if all(not isinstance(i, (list, tuple)) for i in X):
            X = [[x] for x in X]
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # Calculate means
        mean_x = [sum(col[i] for col in X) / n_samples for i in range(n_features)]
        mean_y = sum(y) / n_samples
        
        # Calculate coefficients b1 (slopes)
        self.b1 = []
        for i in range(n_features):
            numerator = sum((X[j][i] - mean_x[i]) * (y[j] - mean_y) for j in range(n_samples))
            denominator = sum((X[j][i] - mean_x[i]) ** 2 for j in range(n_samples))
            if denominator == 0:
                coef = 0
            else:
                coef = numerator / denominator
            self.b1.append(coef)
        
        # Calculate intercept b0
        self.b0 = mean_y - sum(self.b1[i] * mean_x[i] for i in range(n_features))
        
    def predict(self, X):
        # Convert to 2D if needed
        if all(not isinstance(i, (list, tuple)) for i in X):
            X = [[x] for x in X]
        
        predictions = []
        for row in X:
            pred = self.b0 + sum(bi * xi for bi, xi in zip(self.b1, row))
            predictions.append(pred)
        return predictions



# Define base models
base_models = [
    ("catboost", CatBoostRegressor(**catboost_params)),
    ("xgboost", XGBRegressor(**xgboost_params)),
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
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        
        
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50)

        
        oof_train[valid_idx, i] = model.predict(X_valid).clip(0)
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





