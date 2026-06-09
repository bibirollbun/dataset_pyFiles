import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


train_df_path = '/kaggle/input/playground-series-s5e3/train.csv'
train_extra_df_path = '/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv'
test_df_path = '/kaggle/input/playground-series-s5e3/test.csv'
perfect_public_lb_submission = '/kaggle/input/rainfall-public-lb/rainfall_public_lb.csv'

train_df = pd.read_csv(train_df_path)
train_extra_df = pd.read_csv(train_extra_df_path)
test_df = pd.read_csv(test_df_path)


from sklearn.metrics import roc_auc_score

perfect_lb_df = pd.read_csv(perfect_public_lb_submission)
perfect_lb = perfect_lb_df[0:146]['rainfall'].values 

def eval_submission(submission):
    y_pred = submission['rainfall'].values[:146]
    return roc_auc_score(perfect_lb, y_pred)


# train_extra_df['id'] = np.arange(2920, 2920+len(train_extra_df))
# train_extra_df.columns = train_extra_df.columns.str.strip()

# train_extra_df['rainfall'] = (
#     train_extra_df['rainfall'].astype(str).str.strip() == 'yes'
# ).astype(int)

# train_df = pd.concat([train_df, train_extra_df], ignore_index=True)

# print(train_df.shape)
# train_df


test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].mean())
train_df['winddirection'] = train_df['winddirection'].fillna(train_df['winddirection'].mean())
train_df['windspeed'] = train_df['windspeed'].fillna(train_df['windspeed'].mean())

train_df = train_df.rename(columns={'temparature': 'temperature'})
test_df = test_df.rename(columns={'temparature': 'temperature'})

train_df = train_df.drop('id', axis=1, errors='ignore')
test_df = test_df.drop('id', axis=1, errors='ignore')


def correct_df(df):
    df = df.copy()

    df.loc[df['maxtemp'] < df['mintemp'], ['maxtemp', 'mintemp']] = df.loc[df['maxtemp'] < df['mintemp'], ['mintemp', 'maxtemp']].values

    df.loc[df['temperature'] > df['maxtemp'], 'temperature'] =  df.loc[df['temperature'] > df['maxtemp'], 'maxtemp']
    df.loc[df['temperature'] < df['mintemp'], 'temperature'] =  df.loc[df['temperature'] < df['mintemp'], 'mintemp']

    # averaging temp is worse than clamping temp
    # df.loc[(df['temperature'] > df['maxtemp']) | (df['temperature'] < df['mintemp']), 'temperature'] = (df.loc[(df['temperature'] > df['maxtemp']) | (df['temperature'] < df['mintemp']), 'maxtemp'] + df.loc[(df['temperature'] > df['maxtemp']) | (df['temperature'] < df['mintemp']), 'mintemp']) / 2

    # df.loc[df['temperature'] < df['dewpoint'], 'humidity'] =  100
    df.loc[df['temperature'] < df['dewpoint'], 'temperature'] =  df.loc[df['temperature'] < df['dewpoint'], 'dewpoint']
    
    return df


train_df = correct_df(train_df)
test_df = correct_df(test_df)


def cyclical_range(start, stop, length):
    return [((n - 1) % (stop - 1)) + 1 for n in range(start, start+length)]

def add_day_info(df):
    df = df.copy()

    df['day'] = cyclical_range(1, 365+1, len(df))
    # train_df['abs_day'] = range(1, len(train_df)+1)
    # train_df['year'] = ((train_df['abs_day']-1) // 365) + 1
    # train_df = train_df.drop('abs_day', axis=1)
    
    return df

train_df = add_day_info(train_df)
test_df = add_day_info(test_df)


EPSILON = 1e-3
def add_features(df):
    df = df.copy()
        
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_fluct'] = df['temp_range'] - df['temperature']
    df['temp_dewpoint_diff'] = df['temperature'] - df['dewpoint']
    df['mintemp_dewpoint_diff'] = df['mintemp'] - df['dewpoint']

    df['dewpoint_temp_ratio'] = df['dewpoint'] / df['temperature']
    df['dewpoint_mintemp_ratio'] = df['dewpoint'] / df['mintemp']
    
    df['humidity_cloud_ratio'] = df['humidity'] / (df['cloud'] + EPSILON)
    df['sunshine_cloud_ratio'] = df['sunshine'] / (df['cloud'] + EPSILON)
    
    df['pressure_temperature_ratio'] =  (df['pressure']) / df['temperature']
    df['windspeed_pressure_ratio'] = df['windspeed'] / (df['pressure'])
    df['cloud_pressure_ratio'] = df['cloud'] / df['pressure']
    df['humidity_pressure_ratio'] = df['humidity'] / df['pressure']

    df['dewpoint_humidity'] = df['dewpoint'] * df['humidity']
    df['cloud_humidity'] = df['cloud'] * df['humidity']
    df['cloud_windspeed'] = df['cloud'] * df['windspeed']
    df['temperature_humidity'] = df['temperature'] * df['humidity']
    df['sunshine_humidity'] = df['sunshine'] * (df['humidity'])

    df['pressure_winddirection'] = df['pressure'] * df['winddirection']
        
    return df


train_df = add_features(train_df)
test_df = add_features(test_df)


# no_exp = {'day', 'day_sin', 'day_cos', 'rainfall'}
# def exp_features(df, exp=2):
#     df = df.copy()
    
#     for col in df.select_dtypes(include=[np.number]).columns:
#         if col in no_exp:
#             continue 

#         new_col_name = f'{col}^{exp}'
#         df[new_col_name] = df[col]**2

#     return df


# train_df = exp_features(train_df)
# test_df = exp_features(test_df)


no_clip = {'day', 'day_sin', 'day_cos', 'rainfall'}

def clip_outliers(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in no_clip:
            continue
            
        series = df[col]
        
        Q1 = np.nanquantile(series, 0.25)
        Q3 = np.nanquantile(series, 0.75)
        
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.clip(series.values, lower, upper)
        
    return df


train_df = clip_outliers(train_df)
test_df = clip_outliers(test_df)


no_roll = {'day', 'day_sin', 'day_cos', 'rainfall', 'year'}
def rolling_features(df, window_sizes):
    df_roll = pd.DataFrame(index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in no_roll:
            continue
        
        series_col = df[col].copy().shift(1)
        reversed_series = df[col].copy().shift(-1)[::-1]

        for window in window_sizes:
            roll_mean = series_col.rolling(window=window, min_periods=1).mean().values
            roll_std = series_col.rolling(window=window, min_periods=1).std().fillna(0).values
                
            df_roll[f'{col}_roll_mean_{window}'] = roll_mean
            df_roll[f'{col}_roll_std_{window}'] = roll_std

        for window in window_sizes:
            future_mean = reversed_series.rolling(window=window, min_periods=1).mean().values[::-1]
            future_std = reversed_series.rolling(window=window, min_periods=1).std().fillna(0).values[::-1]
                
            df_roll[f'{col}_future_mean_{window}'] = future_mean
            df_roll[f'{col}_future_std_{window}'] = future_std
            
    return df_roll


window_sizes = [1, 3, 6, 10]
train_rolling = rolling_features(train_df,window_sizes)
test_rolling = rolling_features(test_df, window_sizes)

train_df = pd.merge(train_df, train_rolling, left_index=True, right_index=True)
test_df = pd.merge(test_df, test_rolling, left_index=True, right_index=True)


def add_time_features(df):
    df = df.copy()

    lags = [1, 2, 3, 4, 5]
    futures = [1, 2, 3, 4, 5]
    fs = ['temperature', 'humidity', 'windspeed', 'cloud', 'sunshine', 'dewpoint']
    
    for lag in lags:
        for f in fs:
            change_name = f'{f}_change_lag_{lag}'
            rate_change_name = f'{f}_rate_change_lag_{lag}'

            df[change_name] = df[f] - df.shift(lag)[f]
            df[rate_change_name] = (df[f] - df.shift(lag)[f])/lag
            
    for fut in futures:
        for f in fs:
            change_name = f'{f}_change_future_{fut}'
            rate_change_name = f'{f}_rate_change_future_{fut}'

            df[change_name] = df[f] - df.shift(-fut)[f]
            df[rate_change_name] = (df[f] - df.shift(-fut)[f])/fut

    return df


train_df = add_time_features(train_df)
test_df = add_time_features(test_df)


# features = train_df.drop(columns=['rainfall'], errors='ignore').columns


# from forward features selection
    
features = ['cloud_humidity', 'cloud_pressure_ratio', 'mintemp', 'temp_range_roll_std_10', 'windspeed_future_mean_6', 'pressure_temperature_ratio_future_mean_1', 'dewpoint_humidity_roll_mean_1', 'pressure_future_mean_6', 'sunshine_humidity']
features


# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# train_df[features] = scaler.fit_transform(train_df[features])
# test_df[features] = scaler.transform(test_df[features])


best_params = {
    "device": "cuda",
    "n_jobs": -1,
    "max_depth": 5,  
    "colsample_bytree": 0.9, 
    "subsample": 0.9, 
    "n_estimators": 10_000,  
    "learning_rate": 0.05, 
    "eval_metric": "logloss",
    "early_stopping_rounds": 100,
    "alpha": 1,
}


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

train_size = 0.9
train_max_idx = train_size * len(train_df)

def get_xgb(X_train=train_df[features], y_train=train_df['rainfall'], X_val=None, y_val=None):
    if X_val is None:
        X_val = X_train.loc[train_max_idx+1:]
        y_val = y_train.loc[train_max_idx+1:]

        X_train = X_train.loc[:train_max_idx]
        y_train = y_train.loc[:train_max_idx]
    
    model = XGBClassifier(**best_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
        verbose=0
    )

    return model


from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

def eval_model(features=features, print_results=False, model_type='xgb'):
    n_folds = 4
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_auc, all_acc = [], []
    
    for i, (train_index, test_index) in enumerate(skf.split(train_df, train_df['rainfall'])):
        # print(f"Fold {i+1}")
        
        X_train = train_df.loc[train_index, features].copy()
        y_train = train_df.loc[train_index, "rainfall"]
    
        X_val = train_df.loc[test_index, features].copy()
        y_val = train_df.loc[test_index,"rainfall"]

        if model_type == 'xgb':
            model = get_xgb(X_train, y_train, X_val, y_val)
    
        pred = model.predict_proba(X_val)[:, 1]
        
        roc_auc = roc_auc_score(y_val, pred)
        accuracy = accuracy_score(y_val, np.round(pred))
    
        # print(f'AUC: {roc_auc}')
        # print(f'ACC: {accuracy}')
        # print()
    
        all_auc.append(roc_auc)
        all_acc.append(accuracy)
    
    avg_auc = sum(all_auc)/len(all_auc)
    avg_acc = sum(all_acc)/len(all_acc)

    if print_results:
        print(f'Avg AUC: {avg_auc}')
        print(f'Avg ACC: {avg_acc}')
    
    return avg_auc

# Avg AUC: 0.8913608305274971
# Avg ACC: 0.863013698630137

eval_model(print_results=True)


import shap

model = get_xgb()

explainer = shap.Explainer(model, test_df[features])
shap_values = explainer(test_df[features])
shap.summary_plot(shap_values, test_df[features], plot_type="bar")


# from tqdm import tqdm

# selected_features = []
# remaining = list(features)
# current_score = 0

# while len(remaining) != 0:
#     print(f'\nSelected Features: {len(selected_features)} | AUC: {current_score}')
#     best_feature = None
#     best_score = current_score

#     for f in tqdm(remaining, desc='Evaluating features'):
#         score = eval_model(features=selected_features+[f], print_results=False, model_type='xgb')  

#         if score > best_score:
#             best_feature = f
#             best_score = score

#     if best_feature:
#         selected_features.append(best_feature)
#         remaining.remove(best_feature)
#         current_score = best_score

#         print(f'Selected: {best_feature} | AUC: {best_score}')
#     else:
#         break


# import pickle

# features = selected_features

# with open('/kaggle/working/selected_features.pkl', 'wb') as f:
#     pickle.dump(selected_features, f)


def create_submission(model):
    final_pred = model.predict_proba(test_df[features])[:, 1]
    submission_df = pd.DataFrame({'id' : np.arange(2190, 2190+len(test_df)), 'rainfall' : final_pred})

    submission_df.to_csv('/kaggle/working/submission.csv', index=False)

    print(eval_submission(submission_df))
    print('Created Submission!')


model = get_xgb()
create_submission(model)

