# Install sklearn (kagle's default version gives library conflicts with TabPFN)
!pip install --upgrade scikit-learn==1.5.2

# Install TabPFN
!pip install tabpfn

# TabPFN Extensions installs optional functionalities around the TabPFN model
# These include post-hoc ensembles, interpretability tools, and more
!git clone https://github.com/PriorLabs/tabpfn-extensions
!pip install -e tabpfn-extensions


import os

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, log_loss

import matplotlib.pyplot as plt
import seaborn as sns

from tabpfn import TabPFNClassifier

import torch

if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU. See section above for instructions.')


# This part is for import the hypertuning function
import sys

sys.path.append('./tabpfn-extensions/src/')

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


test.loc[test['winddirection'].isna(), 'winddirection'] = train['winddirection'].mode().values


test.isna().sum()


def relative_humidity(dewpoint, temparature):
    """
    Function from eda-relative-humidity nb to calculate relative humidity
    """
    HR = (np.exp((17.625 * dewpoint) / (243.04 + dewpoint)) / np.exp((17.625 * temparature) / (243.04 + temparature)))
    return HR


def generate_features(data):
    # Temperatures
    # Correction: there are some mintemp greater than regular temp, which is not possible (also happens with max temp)
    data[['maxtemp', 'temparature', 'mintemp']] = pd.DataFrame(data[['maxtemp', 'temparature', 'mintemp']].apply(lambda x: np.sort(x.values)[::-1], axis=1).to_list()).values
    data['temp_range'] = data['maxtemp'] - data['mintemp']
    data['temp_min_diff'] = data['temparature'] - data['mintemp']
    data['temp_max_diff'] = data['maxtemp'] - data['temparature']
    data['temp_relative_pct'] = (data['temparature'] - data['mintemp'])/(data['maxtemp'] - data['mintemp'])
    data['temp_dewpoint_diff'] = data['temparature'] - data['dewpoint']
    data['temp_min_dewpoint_diff'] = data['mintemp'] - data['dewpoint']
    data['dewpoint_reached'] = (data['temp_min_dewpoint_diff'] <= 0.1)*1

    # Day and pseudodate
    # Years and weeks
    data['day_is_1'] = (data['day'] == 1)*1
    data['day_is_1'] = data['day_is_1'].fillna(0)
    data['year'] = data['day_is_1'].cumsum()
    # Day correction. Some days are laveled grong, some january days have temperatures from august
    # As solution, the day is equal to id - 365*(year - 1) + 1
    # You can check it out generating temporal_id = day + 365*(data['year'] - 1) and plotting temperatures
    data['day'] = data['id'] - 365*(data['year'] - 1) + 1
    data = data.drop(columns='day_is_1')

    data['day_sin'] = data['day'].apply(lambda x: np.sin(x/365*2*np.pi))
    data['day_cos'] = data['day'].apply(lambda x: np.cos(x/365*2*np.pi))
    
    years = data['year'].unique()
    days = np.arange(1, 366)
    years_aux, days_aux = pd.core.reshape.util.cartesian_product([years, days])
    
    day_year_combinations = pd.DataFrame(dict(year=years_aux, day=days_aux))
    day_year_combinations['week_day'] = (list(np.arange(7))*(len(day_year_combinations)//7 + 1))[:len(day_year_combinations)]
    day_year_combinations['week_start'] = (day_year_combinations['week_day'] == 0)*1
    day_year_combinations.loc[day_year_combinations['day'] <= 6, 'week_start'] = 0
    day_year_combinations['week'] = day_year_combinations.groupby('year')['week_start'].apply(lambda x: x.cumsum()).reset_index(drop=True)
    day_year_combinations = day_year_combinations.drop(columns='week_start')
    data = pd.merge(data, day_year_combinations, on=['year', 'day'], how='left')

    data['week_sin'] = data['week'].apply(lambda x: np.sin(x/52*2*np.pi))
    data['week_cos'] = data['week'].apply(lambda x: np.cos(x/52*2*np.pi))

    # Sunshine
    data['sunshine_lvl'] = data['sunshine']/12
    data['sunshine_lvl_by_sin_day'] = data['sunshine_lvl']*data['day'].apply(lambda x: np.sin(x/365*np.pi))

    # Wind direction
    data['winddirection'] = data['winddirection'].round(-1)
    data['winddirection_sin'] = data['winddirection'].apply(lambda x: np.sin(x/300*2*np.pi))
    data['winddirection_cos'] = data['winddirection'].apply(lambda x: np.cos(x/300*2*np.pi))

    # Wind speed
    data['windspeed_log'] = data['windspeed'].apply(np.log)
    data['windspeed_by_winddirection_sin'] = data['windspeed']*data['winddirection_sin']
    data['windspeed_by_winddirection_cos'] = data['windspeed']*data['winddirection_cos']
    data['windspeed_log_by_winddirection_sin'] = data['windspeed_log']*data['winddirection_sin']
    data['windspeed_log_by_winddirection_cos'] = data['windspeed_log']*data['winddirection_cos']

    # Pressure
    train_data = data[data['source'] == 'train']
    train_medians = train_data.groupby('day')['pressure'].apply(np.median)
    train_medians = train_medians.reset_index()
    train_medians.columns = ['day', 'pressure_median']
    train_medians['rolling_mean'] = train_medians['pressure_median'].rolling(window=25, min_periods=1).mean()
    train_medians['pressure_rolling_mean'] = (train_medians['rolling_mean'] + train_medians['pressure_median'][::-1].rolling(window=25, min_periods=1).mean())/2
    
    data = pd.merge(data, train_medians[['day', 'pressure_rolling_mean']], on='day', how='left')
    data['pressure_no_trend'] = data['pressure'] - data['pressure_rolling_mean']

    # Humidity
    data['humidity_cat'] = pd.cut(data['humidity'], bins=[-1, 55, 80, 95, 101], labels=[0, 1, 2, 3]).astype(int)

    # Cloud
    data['cloud_by_sunshine'] = data['cloud'] / (data['sunshine'] + 1)
    data['cloud_cat'] = pd.cut(data['cloud'], bins=[-1, 40, 70, 90, 101], labels=[0, 1, 2, 3]).astype(int)  # 'no_cloud', 'partially_cloud', 'majority_cloud', 'totally_cloud'
    data['cloud_and_humidity_cat'] = data['humidity_cat'] + 4*data['cloud_cat']
    
    ##### Vars from eda-relative-humidity nb #####
    data['relative_humidity'] = relative_humidity(data['dewpoint'], data['temparature']) * 100
    data['relative_humidity'] = np.clip(0, 100, data['relative_humidity'])
    data['cloud_covered'] = data['cloud'] * data['sunshine']
    data['cloud_pressure'] = data['cloud'] / data['pressure']
    data['cloud_covered_by_pressure']= data['cloud_covered'] / data['pressure']
    data['cloud_direction'] = data['cloud_covered'] / (data['winddirection'] * data['windspeed'])
    data['cloud_sparsity'] = data['cloud_covered'] * data['cloud_direction']
    data['dewpoint_rupture'] = data['pressure'] / (data['dewpoint'] - data['temp_range'])
    data['dewpoint_rupture_over_cloud_covered'] =  data['cloud_covered_by_pressure'] / data['dewpoint_rupture']
    data['cloud_sparsity_by_max_temp_over_pressure'] = (data['cloud_sparsity'] * data['maxtemp']) / (1 + data['cloud_covered_by_pressure'])

    ##### Vars from ps5e3-rainfall-prediction-classification #####
    data['pressure_change'] = data.groupby('source')['pressure'].diff().fillna(0)
    data['pressure_acceleration'] = data.groupby('source')['pressure_change'].diff().fillna(0)
    data['humidity_dewpoint_ratio'] = data['humidity'] / data['dewpoint'].clip(lower=0.1)
    data['wind_humidity_factor'] = data['windspeed'] * (data['humidity'] / 100)
    data['temp_humidity_index'] = (0.8 * data['temparature']) + \
                                  ((data['humidity'] / 100) * \
                                  (data['temparature'] - 14.3)) + 46.4
    # Convert month to season (1-365 to 1-4)
    data['month'] = ((data['day'] - 1) // 30) + 1
    data['season'] = ((data['month'] - 1) // 3) + 1
    # Rolling averages for key meteorological variables
    for var in ['temparature', 'pressure', 'humidity', 'cloud', 'windspeed']:
        for window in [3, 7, 14]:
            data[f'{var}_rolling_{window}d'] = data.groupby('source')[var].rolling(window=window, min_periods=1).mean().reset_index().sort_values('level_1')[var].values
    
    # Weather pattern change features
    for var in ['temparature', 'pressure', 'humidity']:
        data[f'{var}_trend_3d'] = data.groupby('source')[var].diff(3).fillna(0)
    
    # Extreme weather indicators
    for var in ['temparature', 'humidity', 'pressure']:
        var_q05 = data.loc[data['source'] == 'train', var].quantile(0.05)
        var_q95 = data.loc[data['source'] == 'train', var].quantile(0.95)
        data[f'extreme_{var}'] = ((data[var] > var_q95) | (data[var] < var_q05))*1

    # Interaction terms between key variables
    data['temp_humidity_interaction'] = data['temparature'] * data['humidity']
    data['pressure_wind_interaction'] = data['pressure'] * data['windspeed']
    data['dewpoint_humidity_interaction'] = data['dewpoint'] * data['humidity']
    
    # Moving standard deviations for measuring variability
    for var in ['temparature', 'humidity', 'pressure']:
        for window in [7, 14]:
            data[f'{var}_std_{window}d'] = data.groupby('source')[var].rolling(window=window, min_periods=4).std().fillna(0).reset_index().sort_values('level_1')[var].values

    return data


train['source'] = 'train'
test['source'] = 'test'
train_and_test = pd.concat([train, test], ignore_index=True)

train_and_test = generate_features(train_and_test)

train_bis = train_and_test[train_and_test['source'] == 'train'].drop(columns='source').reset_index(drop=True)
test_bis = train_and_test[train_and_test['source'] == 'test'].drop(columns='source').reset_index(drop=True)


train_bis.columns


x_train = train_bis.drop(columns=['id', 'rainfall'])
y_train = train_bis['rainfall']

x_test = test_bis[x_train.columns]


metrics_dict = {}
for balance in [False]:  # True is ignored because it doesn't improve the auc, only changes the BCE value.
    for i in np.arange(2, 17, 2):
        roc_auc_score_list = []
        f1_score_list = []
        log_loss_list = []
        for val_year in [4, 5, 6]:
            x_train_fold = x_train[x_train['year'] < val_year]
            x_val_fold = x_train[x_train['year'] >= val_year]
            y_train_fold = y_train[x_train['year'] < val_year]
            y_val_fold = y_train[x_train['year'] >= val_year]
    
            model = TabPFNClassifier(balance_probabilities=balance, n_estimators=i, random_state=7)
            model.fit(x_train_fold, y_train_fold)
            y_pred = model.predict_proba(x_val_fold)[:,1]

            roc_auc_score_list.append(roc_auc_score(y_val_fold, y_pred))
            log_loss_list.append(log_loss(y_val_fold, y_pred))

        model = TabPFNClassifier(balance_probabilities=balance, n_estimators=i, random_state=7)
        model.fit(x_train, y_train)
        test['rainfall'] = model.predict_proba(x_test)[:, 1]
        final_submission = test[['id', 'rainfall']]
    
        name = f'TabPFN__n_est{i}__balance{balance}'
        final_submission.to_csv(f'rainfall_v2_{name}.csv', index=False)

        metrics_dict[name] = {'roc_auc_score': np.mean(roc_auc_score_list), 'log_loss': np.mean(log_loss_list), 'roc_auc_score_list': roc_auc_score_list, 'log_loss_list': log_loss_list}
        print(f'Model ({i}, {balance}):', metrics_dict[name])



auc_dict = {name.split('__')[1]: metrics_dict[name]['roc_auc_score'] for name in metrics_dict.keys()}
auc_df = pd.DataFrame(auc_dict, index=[0]).melt(var_name='Model', value_name='AUC')
auc_df.plot(x='Model', y='AUC', marker='o')

