import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from itertools import combinations


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_train.head()


df_train['num_lanes'] = df_train['num_lanes'].astype(pd.Int8Dtype())
df_train['speed_limit'] = df_train['speed_limit'].astype(pd.Int8Dtype())
df_train['num_reported_accidents'] = df_train['num_reported_accidents'].astype(pd.Int8Dtype())

df_train['id'] = df_train['id'].astype(pd.Int32Dtype())

df_train['curvature'] = df_train['curvature'].astype(pd.Float32Dtype())
df_train['accident_risk'] = df_train['accident_risk'].astype(pd.Float32Dtype())

df_train.info()


df_train.shape


df_train.describe()


df_train.describe(include='O')


def preprocess(X):
    cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
    
    for cat in cat_cols:
        dummies = pd.get_dummies(X[cat])
        X = pd.concat([X, dummies], axis=1)
        X = X.drop([cat], axis=1)

    y = X.get('accident_risk')
    X = X.drop(['id', 'accident_risk'], axis=1, errors='ignore')
    
    num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
    
    for i in range(len(num_cols)):
        col1 = num_cols[i]
        X[f'{col1}_squared'] = X[col1] ** 2
        X[f'{col1}_log'] = np.log(X[col1] + 1)
        for j in range(i, len(num_cols)):
            col2 = num_cols[j]
            if col1 != col2:
                X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-6)

    return X, y

X_train, y_train = preprocess(df_train)


df_train = pd.concat([X_train, y_train], axis=1)

plt.figure(figsize=(20,16))
sns.heatmap(df_train.corr(), annot=True, fmt='.2f')


kf = KFold(n_splits=5, shuffle=True, random_state=42)


import xgboost as xgb

def train_xgb():
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'tree_method': 'hist',
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'eta': trial.suggest_float('eta', 1e-4, 1e-1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'device': 'cuda:0',
            'random_state': 42
        }

        rmse_scores = []

        X = X_train
        y = y_train
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

            model = xgb.train(
                params,
                dtrain,
                evals=[(dtrain, 'train'), (dval, 'validation')],
                num_boost_round=1000,
                early_stopping_rounds=50,
                verbose_eval=False
            )

            y_pred = model.predict(dval)
            rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Лучшие параметры XGBoost (K-Fold):", study.best_params)
    print("Лучшее среднее значение RMSE (K-Fold):", study.best_value)

    # train_test_split
    # {'lambda': 6.099844193540886e-07, 'alpha': 0.05726829923917858, 'max_depth': 8, 'min_child_weight': 5, 'eta': 0.007772245804721981, 'subsample': 0.7395565194948666, 'colsample_bytree': 0.6771420786620531}
    # 0.05620041932961015

    # KFold
    # {'lambda': 1.9839761587659618e-08, 'alpha': 7.876315235024374e-06, 'max_depth': 8, 'min_child_weight': 3, 'eta': 0.007312795508111595, 'subsample': 0.8850167151381121, 'colsample_bytree': 0.5443707900627289}
    # 0.05605218940229839

    return study.best_params, study.best_value

# best_params_xgb, best_value_xgb = train_xgb()


from catboost import CatBoostRegressor

def train_catboost():    
    def objective(trial):
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Poisson', 'Bayesian'])

        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'verbose': False,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'task_type': 'GPU',
            'random_seed': 42
        }

        if bootstrap_type == 'Poisson':
            params['bootstrap_type'] = 'Poisson'
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
        elif bootstrap_type == 'Bayesian':
            params['bootstrap_type'] = 'Bayesian'
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)

        model = CatBoostRegressor(**params)

        rmse_scores = []

        X = X_train
        y = y_train
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

            model = CatBoostRegressor(**params)

            model.fit(
                X_train_fold, y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                early_stopping_rounds=50,
                verbose=False
            )

            y_pred = model.predict(X_val_fold)
            rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    print("Лучшие параметры CatBoost:", study.best_params)
    print("Лучшее значение RMSE:", study.best_value)

    # train_test_split
    # {'bootstrap_type': 'Poisson', 'iterations': 942, 'learning_rate': 0.05174147762074759, 'depth': 8, 'l2_leaf_reg': 7.827229065342479, 'random_strength': 2.5785613130034084e-07, 'border_count': 171, 'subsample': 0.9986605429395348}
    # 0.056278168362776985

    # KFold
    # {'bootstrap_type': 'Poisson', 'iterations': 980, 'learning_rate': 0.042306076167191044, 'depth': 8, 'l2_leaf_reg': 0.08405555663301308, 'random_strength': 2.1147281717882366e-05, 'border_count': 179, 'subsample': 0.8209443623536437}
    # 0.05615104161868869
    
    return study.best_params, study.best_value

# best_params_catboost, best_value_catboost = train_catboost()


import lightgbm as lgb

def train_lightgbm():
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'verbosity': -1,
            'device': 'gpu',
            'random_state': 42,
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        rmse_scores = []

        X = X_train
        y = y_train

        for train_idx, val_idx in kf.split(X):
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

            model = lgb.LGBMRegressor(**params, n_estimators=1000)

            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
                eval_metric='rmse',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )

            y_pred = model.predict(X_val_fold)
            rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Лучшие параметры LightGBM (K-Fold):", study.best_params)
    print("Лучшее среднее значение RMSE (K-Fold):", study.best_value)

    # train_test_split
    # {'num_leaves': 201, 'max_depth': 12, 'learning_rate': 0.007659858075418448, 'min_child_weight': 0.1564566136923156, 'min_child_samples': 13, 'subsample': 0.8492346831952857, 'colsample_bytree': 0.654774558754155, 'reg_alpha': 6.04233270262445e-08, 'reg_lambda': 2.923753801415148e-07}
    # 0.056184367353557814

    # KFold
    # {'num_leaves': 192, 'max_depth': 11, 'learning_rate': 0.007772795134672776, 'min_child_weight': 9.849788275656909, 'min_child_samples': 5, 'subsample': 0.9991739271986981, 'colsample_bytree': 0.5782411912539667, 'reg_alpha': 4.492916953728361e-08, 'reg_lambda': 0.00020857283777526658}. 
    # 0.05602894628967696.

    return study.best_params, study.best_value

# best_params_lgbm, best_value_lgbm = train_lightgbm()


best_params = {'num_leaves': 192, 'max_depth': 11, 'learning_rate': 0.007772795134672776, 'min_child_weight': 9.849788275656909, 'min_child_samples': 5, 'subsample': 0.9991739271986981, 'colsample_bytree': 0.5782411912539667, 'reg_alpha': 4.492916953728361e-08, 'reg_lambda': 0.00020857283777526658}
model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
model.fit(X_train, y_train)


X_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
X_test, _ = preprocess(X_test) 

X_test.info()


y_test=model.predict(X_test)


X_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

submission = pd.DataFrame({
    'id': X_test['id'],
    'accident_risk': y_test
})

submission.to_csv('submission.csv', index=False)

