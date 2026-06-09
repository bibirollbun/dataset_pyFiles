import pandas as pd
import numpy as np
import seaborn as sns
from catboost import CatBoostClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
original = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.head()


train.info()


train.isnull().sum()


train  = train.drop(columns=['id'])
test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mean())
test_ids = test["id"]
test  = test.drop(columns=['id'])


def engineer_features(df):
    df_engineered = df.copy()
    if 'temparature' in df_engineered.columns and 'temperature' not in df_engineered.columns:
        df_engineered = df_engineered.rename(columns={'temparature': 'temperature'})
    temp_col = 'temperature' if 'temperature' in df_engineered.columns else 'temparature'
    df_engineered['temp_range'] = df_engineered['maxtemp'] - df_engineered['mintemp']
    df_engineered['temp_change_rate'] = np.where(
        df_engineered['temp_range'] > 0,
        (df_engineered[temp_col] - df_engineered['mintemp']) / df_engineered['temp_range'],
        0.5
    )
    df_engineered['humidity_temp_ratio'] = df_engineered['humidity'] / (df_engineered[temp_col] + 0.1)
    df_engineered['dew_point_depression'] = df_engineered[temp_col] - df_engineered['dewpoint']
    df_engineered['dew_point_ratio'] = df_engineered['dewpoint'] / (df_engineered[temp_col] + 0.1)
    df_engineered['cloud_sunshine_ratio'] = df_engineered['cloud'] / (df_engineered['sunshine'] + 0.1)
    df_engineered['overcast_measure'] = df_engineered['cloud'] * (1 / (df_engineered['sunshine'] + 0.1))
    df_engineered['pressure_diff'] = df_engineered['pressure'].diff().fillna(0)
    df_engineered['pressure_change_rate'] = df_engineered['pressure_diff'] / df_engineered['pressure'] * 100
    df_engineered['wind_energy'] = df_engineered['windspeed'] ** 2
    df_engineered['effective_temp'] = df_engineered[temp_col] - (0.045 * df_engineered['windspeed'] * (df_engineered[temp_col] - 10))
    df_engineered['wind_sin'] = np.sin(np.radians(df_engineered['winddirection']))
    df_engineered['wind_cos'] = np.cos(np.radians(df_engineered['winddirection']))
    df_engineered['rain_indicator_strict'] = (
        (df_engineered['humidity'] > 90) & 
        (df_engineered['cloud'] > 85) & 
        (df_engineered['sunshine'] < 1)
    ).astype(int)
    df_engineered['rain_indicator_moderate'] = (
        (df_engineered['humidity'] > 80) & 
        (df_engineered['cloud'] > 70) & 
        (df_engineered['sunshine'] < 3)
    ).astype(int)
    df_engineered['pressure_wind_interaction'] = df_engineered['pressure'] * df_engineered['windspeed']
    df_engineered['humidity_squared'] = df_engineered['humidity'] ** 2
    df_engineered['cloud_squared'] = df_engineered['cloud'] ** 2
    df_engineered['humidity_cube'] = df_engineered['humidity'] ** 3
    df_engineered['log_sunshine'] = np.log1p(df_engineered['sunshine'])
    df_engineered['log_windspeed'] = np.log1p(df_engineered['windspeed'])
    if 'day' in df_engineered.columns:
        df_engineered['day_sin'] = np.sin(2 * np.pi * df_engineered['day'] / 365.25)
        df_engineered['day_cos'] = np.cos(2 * np.pi * df_engineered['day'] / 365.25)
    df_engineered['storm_indicator'] = (
        (df_engineered['windspeed'] > df_engineered['windspeed'].mean() + df_engineered['windspeed'].std()) &
        (df_engineered['pressure_diff'] < -df_engineered['pressure_diff'].std())
    ).astype(int)
    df_engineered['temp_dew_humidity_ratio'] = df_engineered['dew_point_depression'] / (df_engineered['humidity'] + 0.1)
    df_engineered = df_engineered.fillna(df_engineered.mean())
    return df_engineered


train = engineer_features(train)
test = engineer_features(test)


train.head()


X = train.drop(['rainfall'], axis=1)
y = train['rainfall']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 5000),  # Increased range for trees
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),  # Wider range for learning rate
        'depth': trial.suggest_int('depth', 4, 16),  # Increased range for more complex trees
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 20),  # Wider range for regularization
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),  # Allowing more randomness
        'random_strength': trial.suggest_float('random_strength', 0.0, 50),  # More randomness range
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Allowing more variation in training data
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),  # Increased feature sampling range
        'verbose': 0,
        'random_state': 42
    }

    model = CatBoostClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,  # Increased to allow better convergence
        verbose=0
    )
    
    y_pred = model.predict_proba(X_val)[:, 1]
    
    roc_auc = roc_auc_score(y_val, y_pred)
    
    return roc_auc


#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=10)

#best_params = study.best_params
#best_score = study.best_value


#print("Best ROC AUC:", study.best_value)
#print("Best Hyperparameters:", study.best_params)


best_params = {
    'iterations': 3646,
    'learning_rate': 0.1222,
    'depth': 6,
    'l2_leaf_reg': 3.1041,
    'border_count': 124,
    'bagging_temperature': 1.4084,
    'random_strength': 24.3560,
    'subsample': 0.7814,
    'colsample_bylevel': 0.9896,
    'eval_metric': 'AUC',
    'random_state': 42,
    'verbose': 100
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_true, y_pred_proba, preds = [], [], []

for train_index, test_index in skf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    catboost_model = CatBoostClassifier(**best_params)
    catboost_model.fit(X_train, y_train)

    y_val_pred_proba = catboost_model.predict_proba(X_val)[:, 1]
    y_true.extend(y_val)
    y_pred_proba.extend(y_val_pred_proba)

    test_preds = catboost_model.predict_proba(test[X.columns])[:, 1]
    preds.append(test_preds)

auc_roc = roc_auc_score(y_true, y_pred_proba)
print(f"ğŸ�† AUC-ROC Score: {auc_roc:.4f}")


test.head()


submission = pd.DataFrame({'id': test_ids, 'rainfall': np.mean(preds, axis=0)})
print(submission.head())
submission.to_csv('submission.csv', index=False)

