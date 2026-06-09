#Basic part
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#ML part
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from scipy.optimize import minimize


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test =  pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head()


train.info()


train.isna().sum()


train.describe(include='all').applymap('{:,.2f}'.format)


X = train.copy()
y = X.pop('rainfall')
X_test = test


# Cross-validation setup
FOLDS = 7
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

# XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'colsample_bytree': 0.9,
    'subsample': 0.9,
    'learning_rate': 0.05,
    'eval_metric': 'auc',
    'seed': 42,
    'verbosity': 0
}

# Training loop for XGBoost
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=500
    )
    
    # Accumulate test predictions
    if hasattr(model, 'best_ntree_limit'):
        best_iteration = model.best_ntree_limit
    else:
        best_iteration = None
    
    test_preds += (model.predict(dtest, ntree_limit=best_iteration) if best_iteration is not None else model.predict(dtest)) / FOLDS

    fold_auc = roc_auc_score(y_val, model.predict(dval, ntree_limit=best_iteration) if best_iteration is not None else model.predict(dval))
    print(f"Fold {fold} AUC: {fold_auc:.4f}")



import matplotlib.pyplot as plt

importance = model.get_score(importance_type='weight') 

importances_df = pd.DataFrame({
    'feature': list(importance.keys()),
    'importance': list(importance.values())
}).sort_values(by='importance', ascending=False)

print(importances_df)

xgb.plot_importance(model, max_num_features=20, importance_type='weight')
plt.show()


def create_features(df):

    df = df.copy()

    # --- Temperature features ---
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['relative_temp_dewpoint'] = df['temparature'] - df['dewpoint']

    # --- Humidity features ---
    df['relative_humidity'] = 100 * (
        np.exp((17.625 * df['dewpoint']) / (243.04 + df['dewpoint'])) /
        np.exp((17.625 * df['temparature']) / (243.04 + df['temparature']))
    )
    df['high_humidity_flag'] = (df['humidity'] > 80).astype(int)

    # --- Wind features ---
    df['strong_wind_flag'] = (df['windspeed'] > 25).astype(int)
    df['wind_vector_x'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
    df['wind_vector_y'] = df['windspeed'] * np.sin(np.radians(df['winddirection']))

    # --- Cloudiness features ---
    df['clear_sky_flag'] = (df['cloud'] < 20).astype(int)
    df['high_cloud_flag'] = (df['cloud'] > 80).astype(int)

    # --- Atmospheric pressure features ---
    df['pressure_change'] = df['pressure'].diff().fillna(0)
    df['pressure_trend'] = df['pressure'].diff(3).fillna(0)

    # --- Lags ---
    for lag in [1, 2, 3]:
        df[f'pressure_lag{lag}'] = df['pressure'].shift(lag).bfill()
        df[f'humidity_lag{lag}'] = df['humidity'].shift(lag).bfill()
        df[f'cloud_lag{lag}'] = df['cloud'].shift(lag).bfill()

    # --- Seasonality via sin/cos ---
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    return df


X = create_features(train)
y = X.pop('rainfall')
X_test = create_features(test)


# Cross-validation setup
FOLDS = 7
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

# XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'colsample_bytree': 0.9,
    'subsample': 0.9,
    'learning_rate': 0.05,
    'eval_metric': 'auc',
    'seed': 42,
    'verbosity': 0
}

# Training loop for XGBoost
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=500
    )
    
    # Accumulate test predictions
    if hasattr(model, 'best_ntree_limit'):
        best_iteration = model.best_ntree_limit
    else:
        best_iteration = None
    
    test_preds += (model.predict(dtest, ntree_limit=best_iteration) if best_iteration is not None else model.predict(dtest)) / FOLDS

    fold_auc = roc_auc_score(y_val, model.predict(dval, ntree_limit=best_iteration) if best_iteration is not None else model.predict(dval))
    print(f"Fold {fold} AUC: {fold_auc:.4f}")




import matplotlib.pyplot as plt

importance = model.get_score(importance_type='weight') 

importances_df = pd.DataFrame({
    'feature': list(importance.keys()),
    'importance': list(importance.values())
}).sort_values(by='importance', ascending=False)

print(importances_df)

xgb.plot_importance(model, max_num_features=20, importance_type='weight')
plt.show()


# submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
# submission['rainfall'] = test_preds
# submission.to_csv("submission.csv", index=False)
# print("Submission created!")


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import numpy as np
import optuna



def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
    }
    aucs = []
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
            end_iteration = model.best_iteration
        else:
            end_iteration = 1000
        preds = model.predict(dval, iteration_range=(0, end_iteration))
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
        mean_auc = np.mean(aucs)
    return mean_auc


X = train.copy()
y = X.pop('rainfall')

study_without_feature_engineering = optuna.create_study(direction='maximize')
study_without_feature_engineering.optimize(objective, n_trials=30)  

print("Best mean AUC:", study_without_feature_engineering.best_value)
print("Best params:", study_without_feature_engineering.best_params)


X = create_features(train)
y = X.pop('rainfall')

study_with_feature_engineering = optuna.create_study(direction='maximize')
study_with_feature_engineering.optimize(objective, n_trials=30) 

print("Best mean AUC:", study_with_feature_engineering.best_value)
print("Best params:", study_with_feature_engineering.best_params)


if study_with_feature_engineering.best_value > study_without_feature_engineering.best_value:
    best_params = study_with_feature_engineering.best_params
    X_full = create_features(train)
    X_test_full = create_features(test)
else:
    best_params = study_without_feature_engineering.best_params
    X_full = train.copy()
    X_test_full = test.copy()

y_full = X_full.pop('rainfall')
FOLDS = 7
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test_full))

# Обновляем параметры для совместимости
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'auc'
best_params['booster'] = 'gbtree'
best_params['seed'] = 42
best_params['verbosity'] = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), 1):
    X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test_full)

    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

  
    end_iteration = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration is not None else 1000
    fold_test_preds = model.predict(dtest, iteration_range=(0, end_iteration))

    test_preds += fold_test_preds / FOLDS

# Создание submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission['rainfall'] = test_preds
submission.to_csv("submission.csv", index=False)
print('submission created')

