import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train_df.shape, test_df.shape, sample_sub.shape


train_df


train_df.info()


train_df.describe()


columns_to_drop = ['id', 'BeatsPerMinute']

X = train_df.drop(columns=columns_to_drop).copy()
y = train_df['BeatsPerMinute']

test_df = test_df.drop(columns=['id'])


target_column = 'BeatsPerMinute'
features = [col for col in train_df.columns if col != target_column]

correlations = train_df[features + [target_column]].corr()[target_column]
correlations = correlations.drop(target_column).sort_values(ascending=False)

print("REAL correlations with the target variable:")
print(correlations.head(10))


def advanced_music_features(df):
    df = df.copy()
    
    for col in ['MoodScore', 'RhythmScore', 'VocalContent', 'InstrumentalScore']:
        df[f'energy_{col.lower()}_interaction'] = df['Energy'] * df[col]
    
    df['energy_group'] = pd.cut(df['Energy'], bins=10, labels=False)
    df['mood_group'] = pd.cut(df['MoodScore'], bins=10, labels=False)
    
    df['energy_mood_ratio'] = df['Energy'] / (df['MoodScore'] + 0.001)
    df['rhythm_vocal_difference'] = df['RhythmScore'] - df['VocalContent']
    
    numeric_cols = ['Energy', 'MoodScore', 'RhythmScore', 'VocalContent']
    for col in numeric_cols:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_log'] = np.log1p(df[col] - df[col].min() + 1)
    
    return df


new_df = advanced_music_features(X)
new_test_df = advanced_music_features(test_df)


def scale_columns_advanced(data, columns, scaler_type='standard', return_scalers=False):

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data должен быть pandas DataFrame")
    
    if columns == 'all':
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    elif not isinstance(columns, (list, pd.Index)):
        raise ValueError("columns должен быть списком или 'all'")
    
    scalers = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }
    
    if scaler_type not in scalers:
        raise ValueError(f"scaler_type должен быть одним из: {list(scalers.keys())}")
    
    scaler_class = scalers[scaler_type]
    data_scaled = data.copy()
    scalers_dict = {}
    
    for column in columns:
        if column not in data_scaled.columns:
            continue
            
        if not pd.api.types.is_numeric_dtype(data_scaled[column]):
            continue
        
        scaler = scaler_class()
        data_scaled[column] = scaler.fit_transform(data_scaled[[column]]).flatten()
        scalers_dict[column] = scaler
    
    return (data_scaled, scalers_dict) if return_scalers else data_scaled


data_scaled = scale_columns_advanced(new_df, 'all', scaler_type='standard', return_scalers=False)
test_scaled = scale_columns_advanced(new_test_df, 'all', scaler_type='standard', return_scalers=False)


data_scaled


# hyperparameter from Optuna
lightgbm_best_params = {"n_estimators": 1000, 
                        'learning_rate': 0.010066166981703642, 
                        'num_leaves': 27, 
                        'max_depth': 11, 
                        'min_child_samples': 8, 
                        'subsample': 0.520912175912003, 
                        'colsample_bytree': 0.6804575153311426, 
                        'reg_alpha': 1.5616311520855303e-05, 
                        'reg_lambda': 2.420029847404031e-05,
                        "random_state": 42, 
                        "n_jobs": -1}

LGB_model = LGBMRegressor(**lightgbm_best_params)


# hyperparameter from Optuna
catboost_best_params = {"iterations": 1000, 
                        'learning_rate': 0.013363245137865713, 
                        'depth': 6, 
                        'l2_leaf_reg': 0.007033755461912123, 
                        'random_strength': 0.1726072121647917, 
                        'bootstrap_type': 'Bayesian', 
                        "random_state": 42, 
                        "task_type": "CPU", 
                        "verbose": 0}

CatB_model = CatBoostRegressor(**catboost_best_params)


def learning_func(X, y, test_df, models):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    
    final_pred_dict = {}
    for name, model in models.items():
        work_model = model
        work_model.fit(X, y)
        learn_pred = work_model.predict(X_test)
        print(f'{name}:',np.sqrt(mean_squared_error(y_test, learn_pred)).round(4))

        pred = work_model.predict(test_df)
        final_pred_dict[name] = pred

    return final_pred_dict


%%time

separate_ver = learning_func(data_scaled, y, test_scaled, {'LGB':LGB_model, 'CatB': CatB_model})


# sample_sub['BeatsPerMinute'] = separate_ver['LGB']

# sample_sub.to_csv("submission.csv", index=False)
# print("submission.csv is done!")


# sample_sub['BeatsPerMinute'] = separate_ver['CatB']

# sample_sub.to_csv("submission.csv", index=False)
# print("submission.csv is done!")


sample_sub['BeatsPerMinute'] = (separate_ver['CatB'] + separate_ver['LGB']) / 2

sample_sub.to_csv("submission.csv", index=False)
print("submission.csv is done!")

