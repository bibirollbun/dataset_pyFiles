import numpy as np 
import pandas as pd 

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor, DMatrix

import functools

import optuna

import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')


train_data


train_data.isnull().sum()


train_data.drop(['Podcast_Name', 'Episode_Title'], axis=1, inplace=True)
test_data.drop(['Podcast_Name', 'Episode_Title'], axis=1, inplace=True)


categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']


categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_pipeline, categorical_cols),
    ('num', numerical_pipeline, numerical_cols)
])


X = preprocessor.fit_transform(train_data.drop('Listening_Time_minutes', axis=1))
y = train_data.Listening_Time_minutes
XX = preprocessor.fit_transform(test_data)


def objective(trial, X, y, cv=5, metric='rmse'):

    # Define hyperparameter search space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42,
        'device': 'cuda'
    }
    
    # Create model with current hyperparameters
    model = XGBRegressor(**param)
    
    # Set up cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation based on selected metric
    if metric == 'rmse':
        scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
        score = scores.mean()  # Minimize RMSE
    
    return score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a partial function with fixed arguments
import functools
objective_func = functools.partial(objective, X=X_train, y=y_train, cv=5, metric='rmse')

# Create and run the study
study = optuna.create_study(direction='minimize')
study.optimize(objective_func, n_trials=15)

# Get the best parameters and train the final model
best_params = study.best_params
print("Best parameters:", best_params)
best_score = study.best_value
print(f"Best RMSE: {best_score}")


model = XGBRegressor(**best_params, device='cuda')


FOLDS = 7

kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_rmse = []
oof_test_preds = np.zeros(XX.shape[0])
oof_train_preds = np.zeros(len(y))

for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    model.fit(X_train, y_train,eval_set=[(X_valid, y_valid)], early_stopping_rounds=50, verbose=0) 

    booster = model.get_booster()

    if hasattr(model, 'best_ntree_limit'):
        best_ntree_limit = model.best_ntree_limit
    elif hasattr(model, 'best_iteration_'):
        best_ntree_limit = model.best_iteration_ + 1
    else:
        best_ntree_limit = model.n_estimators
            
    num_boosted_rounds = booster.num_boosted_rounds()

    best_ntree_limit = min(best_ntree_limit, num_boosted_rounds)

    y_pred = booster.predict(DMatrix(X_valid), iteration_range=(0, best_ntree_limit))
    test_pred = booster.predict(DMatrix(XX), iteration_range=(0, best_ntree_limit))
    oof_train_preds[train_idx] = booster.predict(DMatrix(X_train), iteration_range=(0, best_ntree_limit))
    
    oof_test_preds += test_pred
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    print(f"Fold {fold+1} --> RMSE: {rmse:.4f}")
    oof_rmse.append(rmse)


sub = pd.DataFrame({'id': test_data.index, 'Price': oof_test_preds/FOLDS})
sub.to_csv('submission.csv', index=False)

