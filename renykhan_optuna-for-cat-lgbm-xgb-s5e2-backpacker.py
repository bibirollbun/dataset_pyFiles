import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from tqdm import tqdm
from colorama import Fore, Style, init
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')


import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool



train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')



train_df.drop('id', axis = 1, inplace = True)
test_df.drop('id', axis = 1, inplace = True)


X = train_df.drop('Price', axis=1)
y = train_df['Price']


SEED = 42
target = 'Price'
n_splits = 10
n_trials = 50
e_stop = 200


# making a pipeline for imputation of numerical and categorical columns
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


'''
Threshold for skewness : 

-0.5 to 0.5 => approx. symmetric
-1 to -0.5 or 0.5 to 1 => Moderetly skewed
<-1 or >1 => Highly skewed

'''

# For numerical columns we use mean if skewness is b/w -1 to 1 else we use median
# For categorical columns we will use most frequent

def data_imputation_pipeline(df : pd.DataFrame):

    # seperate numerical and categorical columns
    numerical_cols = df.select_dtypes(include = ["number"]).columns
    categorical_cols = df.select_dtypes(include = ["object", "category"]).columns

    # define cols to use mean and those on which to use median
    mean_numerical_cols = [col for col in numerical_cols if abs(train_df[col].skew()) <= 1]
    median_numerical_cols = [col for col in numerical_cols if abs(train_df[col].skew()) > 1]

    # define transformers for numerical and categorical data
    mean_numerical_transformer = SimpleImputer(strategy = "mean")
    median_numerical_transformer = SimpleImputer(strategy = "median")
    categorical_transformer = SimpleImputer(strategy = "most_frequent")

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num1", mean_numerical_transformer, mean_numerical_cols),
            ("num2", median_numerical_transformer, median_numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # create a pipeline
    pipeline = Pipeline(steps = [("preprocessor", preprocessor)])

    return pipeline, mean_numerical_cols, median_numerical_cols, categorical_cols

def update_df(train_df, X, test_df):
    
    pipeline, mean_cols, median_cols, cat_cols = data_imputation_pipeline(X)

    # Fit-transform the training data
    transformed_X = pipeline.fit_transform(X)
    transformed_test_df = pipeline.fit_transform(test_df)
    
    # Convert back to DataFrame with proper column names
    column_order = mean_cols + median_cols + list(cat_cols)
    
    X = pd.DataFrame(transformed_X, columns=column_order)
    test_df = pd.DataFrame(transformed_test_df, columns=column_order)
    
    # Restore original data types
    for col in mean_cols + median_cols:
        X[col] = pd.to_numeric(X[col])
        test_df[col] = pd.to_numeric(test_df[col])
    
    for col in cat_cols:
        X[col] = X[col].astype(train_df[col].dtype)
        test_df[col] = test_df[col].astype(train_df[col].dtype)
        
    # Convert object to category type
    X = X.apply(lambda x: x.astype('category') if x.dtype == 'object' else x)
    test_df = test_df.apply(lambda x: x.astype('category') if x.dtype == 'object' else x)

    
    return X, test_df




X, test_data = update_df(train_df, X, test_df)


cat_c = list(X.select_dtypes(include = ["category"]).columns)


def objective_cat(trial):
    
    # Suggest hyperparameters for tuning
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.2)
    l2_leaf_reg = trial.suggest_loguniform('l2_leaf_reg', 1, 10)
    depth = trial.suggest_int('depth', 3, 8)
    iterations = trial.suggest_int('iterations', 1000, 3000)
    random_strength = trial.suggest_int('random_strength', 1, 20)
    bagging_temperature = trial.suggest_uniform('bagging_temperature', 0, 1)

    catboost_params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'learning_rate': learning_rate,
        'iterations': iterations,
        
        'depth': depth,
        'random_strength': random_strength,
        'l2_leaf_reg': l2_leaf_reg,
        'bagging_temperature': bagging_temperature,
        
        # Logging
        'verbose': 100,
        'random_seed': SEED,
        
        # Enable if using GPU
        'task_type': 'GPU',
    }

    
    folds = KFold(n_splits, shuffle = True, random_state=SEED)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(folds.split(X, y), desc = "Training Folds", total = n_splits)):
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]

        X_train_pool = Pool(X_train, y_train, cat_features = cat_c)
        X_valid_pool = Pool(X_val, y_val, cat_features=cat_c)

        model = CatBoostRegressor(**catboost_params)
        model.fit(X_train_pool, eval_set=X_valid_pool, early_stopping_rounds=e_stop, verbose=False)

        val_pred = model.predict(X_valid_pool)
        
        # Compute RMSE
        score = np.sqrt(mean_squared_error(y_val, val_pred))  
        scores.append(score)

    return np.mean(scores)  # Return average RMSE over folds


# # Run Optuna to find the best parameters
# study = optuna.create_study(direction='minimize')
# study.optimize(objective_cat, n_trials = n_trials) 


# # print the best hyperparameters
# best_cat_params = study.best_params
# print(f"Best parameters found: {best_cat_params}")


############### Best Parameters obtained
'''
Best parameters found: {'learning_rate': 0.13944051481200972, 'l2_leaf_reg': 7.554047383325137, 
                        'depth': 3, 'iterations': 1447, 'random_strength': 17, 
                        'bagging_temperature': 0.5838770203329602}
'''
############### Training of CAT is in V3 of the notebook


import lightgbm as lgb
from lightgbm import LGBMRegressor

def objective_lgbm(trial):
    
    # Suggest hyperparameters for tuning
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.2)
    max_depth = trial.suggest_int('max_depth', 3, 8)
    num_leaves = trial.suggest_int('num_leaves', 20, 50)
    n_estimators = trial.suggest_int('n_estimators', 1000, 3000)
    lambda_l1 = trial.suggest_loguniform('lambda_l1', 0.1, 10)
    lambda_l2 = trial.suggest_loguniform('lambda_l2', 0.1, 10)
    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.6, 1.0)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0.6, 1.0)
    
    lgbm_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'bagging_fraction': bagging_fraction,
        'feature_fraction': feature_fraction,

        # Logging
        'verbose': 100,
        'random_state': SEED,
        
        # Enable if using GPU
        'device': 'gpu', 
    }
    
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(folds.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]
        
        callbacks = [lgb.early_stopping(stopping_rounds = e_stop, verbose = False)]
        
        model = LGBMRegressor(**lgbm_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks = callbacks)

        val_pred = model.predict(X_val)
        
        # Compute RMSE
        score = np.sqrt(mean_squared_error(y_val, val_pred))
        scores.append(score)

    return np.mean(scores)  # Return average RMSE over folds



# Run Optuna to find the best parameters
study = optuna.create_study(direction='minimize')
study.optimize(objective_lgbm, n_trials=n_trials)


# print the best hyperparameters
best_lgbm_params = study.best_params
print(f"Best parameters found: {best_lgbm_params}")


from xgboost import XGBRegressor

def objective_xgb(trial):
    
    # Suggest hyperparameters for tuning
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.2)
    max_depth = trial.suggest_int('max_depth', 3, 8)
    n_estimators = trial.suggest_int('n_estimators', 1000, 3000)
    gamma = trial.suggest_loguniform('gamma', 0.1, 10)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
    subsample = trial.suggest_uniform('subsample', 0.6, 1.0)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 0.1, 10)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 0.1, 10)
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'gamma': gamma,
        'colsample_bytree': colsample_bytree,
        'subsample': subsample,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,

        'enable_categorical' : True,

        # Logging
        'verbose': 2,
        'random_state': SEED, 

        # Enable if using GPU
        'tree_method' : 'gpu_hist',
    }
    
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(folds.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]

        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=e_stop, verbose=False)

        val_pred = model.predict(X_val)
        
        # Compute RMSE
        score = np.sqrt(mean_squared_error(y_val, val_pred))
        scores.append(score)

    return np.mean(scores)  # Return average RMSE over folds


# Run Optuna to find the best parameters
study = optuna.create_study(direction='minimize')
study.optimize(objective_xgb, n_trials=n_trials)


# print the best hyperparameters
best_xgb_params = study.best_params
print(f"Best parameters found: {best_xgb_params}")




