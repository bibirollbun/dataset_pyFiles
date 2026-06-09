# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Install lifelines dependencies from provided input
!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np
import lifelines
from lifelines import KaplanMeierFitter, NelsonAalenFitter
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from sklearn.impute import SimpleImputer
import optuna
import logging
import warnings


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')


def load_data():
    train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv', index_col='ID')
    test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv', index_col='ID')
    sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv', index_col='ID')
    data_description = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
    return train, test, sub, data_description


def check_and_handle_missing_values(train, test):
    # Check for missing values in the training data
    missing_train = train.isnull().sum()
    total_train = train.shape[0]
    
    # Check for missing values in the test data
    missing_test = test.isnull().sum()
    total_test = test.shape[0]
    
    # Print out missing values summary
    print("Missing values in training data (with total counts):")
    print(pd.DataFrame({
        'missing_values': missing_train,
        'total_values': total_train,
        'missing_percentage': (missing_train / total_train) * 100
    }).loc[missing_train > 0])

    print("\nMissing values in test data (with total counts):")
    print(pd.DataFrame({
        'missing_values': missing_test,
        'total_values': total_test,
        'missing_percentage': (missing_test / total_test) * 100
    }).loc[missing_test > 0])
    
    # Identify target columns (exclude these from feature processing)
    target_cols = ['efs', 'efs_time']

    # Identify numeric and categorical columns (excluding target columns)
    num_cols = train.select_dtypes(include=[np.number]).columns.difference(target_cols)
    cat_cols = train.select_dtypes(include=[object, 'category']).columns.difference(target_cols)

    # Numeric Imputation (Median)
    num_imputer = SimpleImputer(strategy='median')
    train[num_cols] = num_imputer.fit_transform(train[num_cols])
    test[num_cols] = num_imputer.transform(test[num_cols])

    # Categorical Imputation (Most frequent)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])
    test[cat_cols] = cat_imputer.transform(test[cat_cols])

    return train, test


def preprocess_data(train, test, data_description):
    # Categorical & Numeric columns
    cat_cols = []
    num_cols = []
    for v, t in data_description[['variable', 'type']].values:
        if t == 'Categorical' and v != 'efs':
            cat_cols.append(v)
        elif not v in ['efs_time', 'efs']:
            num_cols.append(v)
    
    # Convert categorical columns to 'category' type
    train[cat_cols] = train[cat_cols].astype('category')
    test[cat_cols] = test[cat_cols].astype('category')
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    train[num_cols] = imputer.fit_transform(train[num_cols])
    test[num_cols] = imputer.transform(test[num_cols])
    
    # Feature engineering: Interaction terms and polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    train_poly = poly.fit_transform(train[num_cols])
    test_poly = poly.transform(test[num_cols])
    
    poly_cols = [f'poly_{i}' for i in range(train_poly.shape[1])]
    train_poly = pd.DataFrame(train_poly, columns=poly_cols, index=train.index)
    test_poly = pd.DataFrame(test_poly, columns=poly_cols, index=test.index)
    
    train = pd.concat([train, train_poly], axis=1)
    test = pd.concat([test, test_poly], axis=1)
    
    # Target encoding using NelsonAalenFitter and KaplanMeierFitter
    naf = NelsonAalenFitter()
    naf.fit(train['efs_time'], train['efs'])
    train['naf_label'] = -naf.cumulative_hazard_at_times(train['efs_time']).values
    train.loc[train['efs'] == 0, 'naf_label'] -= 0.1

    kmf = KaplanMeierFitter()
    kmf.fit(train['efs_time'], train['efs'])
    train['km_label'] = kmf.survival_function_at_times(train['efs_time']).values
    train.loc[train['efs'] == 0, 'km_label'] -= 0.1
    
    return train, test, cat_cols, num_cols + poly_cols


def tune_hyperparameters(train, cat_cols):
    """Tunes XGBoost hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2500),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'random_state': 42,
            'objective': 'reg:squarederror',
            'enable_categorical': True,
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 50),
            'eval_metric': 'rmse',
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }

        scores = []
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in cv.split(train):
            train_data = train.iloc[train_idx]
            val_data = train.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(
                train_data.drop(columns=['efs', 'efs_time', 'naf_label', 'km_label']),
                train_data['naf_label'],
                eval_set=[(val_data.drop(columns=['efs', 'efs_time', 'naf_label', 'km_label']), val_data['naf_label'])],
                verbose=False
            )

            preds = model.predict(val_data.drop(columns=['efs', 'efs_time', 'naf_label', 'km_label']))
            score = concordance_index(val_data['efs_time'], -preds, val_data['efs'])
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10, timeout=3600)
    return study.best_params


import logging
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train_and_predict(train, test, cat_cols, best_params):
    """Trains models and makes predictions."""
    target_cols = ['efs', 'efs_time', 'km_label', 'naf_label']

    # Clean best_params
    if 'gamma' in best_params:
        del best_params['gamma']
    if 'early_stopping_round' in best_params:
        best_params['early_stopping_rounds'] = best_params['early_stopping_round']
        del best_params['early_stopping_round']

    print(f"best_params after cleaning: {best_params}") #add this line.
    
    early_stopping_rounds = best_params.get('early_stopping_rounds', 20)
    xgb_params = {k: v for k, v in best_params.items() if k != 'early_stopping_rounds'}
    lgb_params = best_params.copy()

    models = {
        'xgb_naf': xgb.XGBRegressor(**xgb_params, enable_categorical=True, tree_method='gpu_hist', gpu_id=0),
        'lgb_naf': lgb.LGBMRegressor(**lgb_params, device='gpu', gpu_platform_id=0, gpu_device_id=0),
        'xgb_km': xgb.XGBRegressor(**xgb_params, enable_categorical=True, tree_method='gpu_hist', gpu_id=0),
        'lgb_km': lgb.LGBMRegressor(**lgb_params, device='gpu', gpu_platform_id=0, gpu_device_id=0)
    }

    X_train = train.drop(columns=target_cols)
    y_naf = train['naf_label']
    y_km = train['km_label']

    X_train_naf, X_val_naf, y_train_naf, y_val_naf = train_test_split(X_train, y_naf, test_size=0.2, random_state=42)
    X_train_km, X_val_km, y_train_km, y_val_km = train_test_split(X_train, y_km, test_size=0.2, random_state=42)

    for name, model in models.items():
        logging.info(f'Training {name}...')
        if 'naf' in name:
            if 'lgb' in name:
                model.fit(
                    X_train_naf, y_train_naf,
                    eval_set=[(X_val_naf, y_val_naf)],
                    callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                )
            else:
                model.fit(
                    X_train_naf, y_train_naf,
                    eval_set=[(X_val_naf, y_val_naf)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
        else:
            if 'lgb' in name:
                model.fit(
                    X_train_km, y_train_km,
                    eval_set=[(X_val_km, y_val_km)],
                    callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                )
            else:
                model.fit(
                    X_train_km, y_train_km,
                    eval_set=[(X_val_km, y_val_km)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )

    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(test)

    weights = {'xgb_naf': 0.3, 'lgb_naf': 0.3, 'xgb_km': 0.2, 'lgb_km': 0.2}
    final_preds = sum(preds[name] * weights[name] for name in preds)

    return final_preds


def save_submission(test, preds, sub):
    sub['prediction'] = preds
    sub.to_csv('submission.csv')
    logging.info('Submission file saved.')


def main():
    train, test, sub, data_description = load_data()
    
    train, test = check_and_handle_missing_values(train, test)
    
    train, test, cat_cols, num_cols = preprocess_data(train, test, data_description)
    best_params = tune_hyperparameters(train, cat_cols)
    final_preds = train_and_predict(train, test, cat_cols, best_params)
    save_submission(test, final_preds, sub)

if __name__ == '__main__':
    main()




