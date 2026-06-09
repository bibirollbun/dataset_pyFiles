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


import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin

from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('ggplot')
%matplotlib inline
pd.options.display.max_columns = 100
pd.options.display.float_format = '{:.4f}'.format

SEED = 42
N_FOLDS = 5   


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

def load_data():
    train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
    test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
    
    train = train.drop_duplicates()

    train = optimize_memory_usage(train)
    test = optimize_memory_usage(test)
    
    drop_list = [
        'X697', 'X698', 'X699', 'X700', 'X701', 'X702', 'X703', 'X704', 'X705', 'X706', 
        'X707', 'X708', 'X709', 'X710', 'X711', 'X712', 'X713', 'X714', 'X715', 'X716',
        'X717', 'X864', 'X867', 'X869', 'X870', 'X871', 'X872', 'X104', 'X110', 'X116',
        'X122', 'X128', 'X134', 'X140', 'X146', 'X152', 'X158', 'X164', 'X170', 'X176',
        'X182', 'X351', 'X357', 'X363', 'X369', 'X375', 'X381', 'X387', 'X393', 'X399',
        'X405', 'X411', 'X417', 'X423', 'X429'
    ]
    
    train = train.drop(columns=drop_list).reset_index(drop=True)
    test = test.drop(columns=["label"] + drop_list).reset_index(drop=True)
    
    X = train.drop(columns=["label"], axis=1)
    y = train["label"]
    
    X = variance_threshold(X, 0.04)
    test = test[X.columns]
    
    return X, y, test


def optimize_memory_usage(df, print_size=True):
    """
    Optimizes memory usage in a DataFrame by downcasting numeric columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to optimize.
        print_size (bool): If True, prints memory usage before and after optimization.

    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    # Types for optimization.
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Memory usage size before optimize (Mb).
    before_size = df.memory_usage().sum() / 1024**2
    
    for column in df.columns:
        column_type = df[column].dtype
        
        if column_type in numerics:
            try:
                if str(column_type).startswith('int'):
                    df[column] = pd.to_numeric(df[column], downcast='integer')
                else:
                    df[column] = pd.to_numeric(df[column], downcast='float')
                logger.info(f"Optimized column {column}: {column_type} -> {df[column].dtype}")
            except Exception as e:
                logger.error(f"Failed to optimize column {column}: {e}")
    
    # Memory usage size after optimize (Mb).
    after_size = df.memory_usage().sum() / 1024**2
    
    if print_size:
        print(
            'Memory usage size: before {:5.4f} Mb - after {:5.4f} Mb ({:.1f}%).'.format(
                before_size, after_size, 100 * (before_size - after_size) / before_size
            )
        )
    
    return df


def variance_threshold(df, threshold):
    var_thres = VarianceThreshold(threshold=threshold)
    var_thres.fit(df)
    new_cols = var_thres.get_support()
    return df.iloc[:, new_cols]

class FeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return add_features(X)

def add_features(df):
    features = pd.DataFrame(index=df.index)
    
    features['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    features['total_liquidity'] = df['bid_qty'] + df['ask_qty']
    features['trade_imbalance'] = df['buy_qty'] - df['sell_qty']
    features['total_trades'] = df['buy_qty'] + df['sell_qty']
    
    return pd.concat([df, features], axis=1)


def get_model_params():
    return {
        'catboost': [
            {'iterations': 150, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3,
             'border_count': 32, 'bagging_temperature': 1, 'random_strength': 1},
             {'iterations': 140, 'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 5,
              'border_count': 64, 'bagging_temperature': 0.5, 'random_strength': 2},
        ],
        'xgb': [
             {'n_estimators': 140, 'max_depth': 6, 'learning_rate': 0.01,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'min_child_weight': 1},
            {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.03,
             'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0.1, 'min_child_weight': 2},
        ],
        'lgbm': [
            {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.01,
             'num_leaves': 31, 'min_data_in_leaf': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8},
             {'n_estimators': 140, 'max_depth': 4, 'learning_rate': 0.05, 
              'num_leaves': 15, 'min_data_in_leaf': 10, 'feature_fraction': 0.7, 'bagging_fraction': 0.7}
        ],
        'hgbm': [
            {'max_iter': 100, 'max_depth': 8, 'learning_rate': 0.03,
             'min_samples_leaf': 15, 'l2_regularization': 0.2, 'max_bins': 255},
             {'max_iter': 100, 'max_depth': 6, 'learning_rate': 0.02,
              'min_samples_leaf': 25, 'l2_regularization': 0.1, 'max_bins': 255},
        ]
    }



def train_ensemble(X, y, test, model_params, n_folds=N_FOLDS):
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_predictions = {}
    test_predictions = {}
    
    models = []
    for i, params in enumerate(model_params['catboost'], 1):
        models.append((f'cat_{i}', CatBoostRegressor(**params, verbose=0, thread_count=1)))
    
    for i, params in enumerate(model_params['xgb'], 1):
        models.append((f'xgb_{i}', xgb.XGBRegressor(**params, n_jobs=1)))
    
    for i, params in enumerate(model_params['lgbm'], 1):
        models.append((f'lgb_{i}', LGBMRegressor(**params, verbose=-1, n_jobs=1)))

    for i, params in enumerate(model_params['hgbm'], 1):
        models.append((f'hgb_{i}', HistGradientBoostingRegressor(**params)))
    
    for name, model in models:
        print(f"\nTraining {name}...")
        oof = np.zeros(len(X))
        pred = np.zeros(len(test))
        
        for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
            X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            oof[val_idx] = model.predict(X_val)
            pred += model.predict(test) / folds.n_splits
            
            fold_score = pearsonr(y_val, oof[val_idx])[0]
            print(f'Fold {fold} Pearson: {fold_score:.4f}')
        
        full_score = pearsonr(y, oof)[0]
        print(f'{name} OOF Pearson: {full_score:.4f}')
        
        oof_predictions[name] = oof
        test_predictions[name] = pred
    
    return pd.DataFrame(oof_predictions), pd.DataFrame(test_predictions)


def optimize_weights(oof_df, y_true):
    model_columns = [col for col in oof_df.columns]
    
    def objective(weights):
        combined = sum(w * oof_df[model] for w, model in zip(weights, model_columns))
        return -pearsonr(y_true, combined)[0]  
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * len(model_columns)
    
    initial_weights = np.ones(len(model_columns)) / len(model_columns)
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result.success:
        print("Optimization warning:", result.message)
        return initial_weights
    
    return result.x


def create_submission(test_predictions, weights, model_names):
    sample = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
    sample['prediction'] = sum(w * test_predictions[name] for w, name in zip(weights, model_names))
    sample.to_csv('submission.csv', index=False)
    return sample


X, y, test = load_data()

feature_generator = FeatureGenerator()
X = feature_generator.fit_transform(X)
test = feature_generator.transform(test)

model_params = get_model_params()

oof_results, test_predictions = train_ensemble(X, y, test, model_params)

optimal_weights = optimize_weights(oof_results, y)

print("\nOptimized weights:")
for name, weight in zip(oof_results.columns, optimal_weights):
    score = pearsonr(y, oof_results[name])[0]
    print(f"{name}: {weight:.4f} (Pearson: {score:.4f})")


submission = create_submission(test_predictions, optimal_weights, oof_results.columns)
print("\nSubmission head:")
print(submission.head())


plt.figure(figsize=(10, 6))
weights_df = pd.DataFrame({
    'Model': oof_results.columns,
    'Weight': optimal_weights,
    'Pearson': [pearsonr(y, oof_results[col])[0] for col in oof_results.columns]
}).sort_values('Weight', ascending=False)

sns.barplot(x='Weight', y='Model', data=weights_df, palette='viridis')
plt.title('Model Weights in Ensemble')
plt.tight_layout()
plt.show()

