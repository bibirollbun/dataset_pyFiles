import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import optuna 
import pandas as pd
import seaborn as sns
import sys
import warnings
from catboost import CatBoostClassifier
from itertools import permutations
from joblib import Parallel, delayed
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")


# training data
folder = "/kaggle/input/playground-series-s5e3/"
df_train = pd.read_csv(folder + "train.csv", sep=",")
print(df_train.shape)
df_train.head(1)


# test data
df_test = pd.read_csv(folder + "test.csv", sep=",")
print(df_test.shape)
df_test.head(1)


df_train.info()


# min / max day
df_train.day.min(), df_train.day.max(), 2190/365


# target check
df_train.rainfall.value_counts()


df_test.info()


# min / max day
df_test.day.min(), df_test.day.max(), 730/365


# weather related
def weather_engineering(data, features):
    for i in features:
        # log columns
        data[f"log_{i}"] = np.log(data[i] + 1)
        # polynomial terms
        data[f"{i}2"] = data[i]**2
    # two-way interaction terms
    for i, j in permutations(features, 2):
        data[f"{i}_{j}"] = data[i] * data[j]
        data[f"{i}_to_{j}"] = data[i] / (data[j] + 1)
    # three-way interaction terms
    for i, j, k in permutations(features, 3):
        data[f"{i}_{j}_over_{k}"] = data[i] * data[j] / (data[k] + 1)
    # temp columns
    data['temp_range'] = data['maxtemp'] - data['mintemp']
    data['temp_to_dew'] = data['temparature'] - data['dewpoint']
    data['mintemp_to_dew'] = data['mintemp'] - data['dewpoint']
    data['maxtemp_to_dew'] = data['maxtemp'] - data['dewpoint']
    # wind direction
    data['wind_dir_rad'] = np.deg2rad(data['winddirection'])
    data['wind_dir_sin'] = np.sin(data['wind_dir_rad'])
    data['wind_dir_cos'] = np.cos(data['wind_dir_rad'])
    data.drop(columns=['wind_dir_rad'], inplace=True)
    data['wind_chill'] = 13.12 + 0.6215 * data['temparature'] - 11.37 * (data['windspeed']**0.16) + 0.3965 * data['temparature'] * (data['windspeed']**0.16)
    return data


# season & date related
def date_engineering(data, features):
    # time of the year
    data['month'] = data['day'].apply(lambda x: math.floor(x / (365/12))) + 1
    data['month'] = np.where(data['month'] > 12, 12, data['month'])     
    # seasonal trends
    data['season'] = data['month'].apply(lambda x: 2 if 3 <= x <= 5     # Spring
                                              else 3 if 6 <= x <= 8     # Summer
                                              else 4 if 9 <= x <= 11    # Autumn
                                              else 1)                   # Winter    
    # seasonal interactions
    for feat in features: 
        data[f'season_{feat}'] = data['season'] * data[feat]
        data[f'season_{feat}_std'] = data[feat] - data.groupby('season')[feat].transform('mean')    
    # cyclic encoding s: capture the continuity over time
    data['day_sin']=np.sin(2 * np.pi * data['day'] / 365)
    data['day_cos']=np.cos(2 * np.pi * data['day'] / 365)
    return data


# time-relational
def timeseries_engineering(data, features):    
    for col in features: 
        # shifts and gaps
        for gap in [1, 3, 7]:
            data[col + f"_shift{gap}"] = data[col].shift(gap)
            data[col + f"_diff{gap}"] = data[col].diff(gap)
        # rolling window
        for offset in [3, 7, 14]:
            for stat in ['mean']:
                if ((offset == 1) & (stat == 'std')) == False:
                    data[f'{col}_rolling_{stat}_{offset}d'] = data[col].transform(lambda x: x.rolling(window=offset).agg(stat)) 
                    data[f'{col}_anomaly_{stat}_{offset}d'] = data[col] - data[col].rolling(offset, min_periods=1).agg(stat)
        # exponential weighted
        for alpha in [0.3, 0.5, 0.8]:
            data[f'{col}_ewm_alpha_{alpha}'] = data[col].transform(lambda x: x.ewm(alpha=alpha).mean()) 
    
    # fill missing columns
    data = data.transform(lambda x: x.interpolate(method='bfill').ffill())
    return data


# collect all together
def feature_engineering(data):
    #features = sorted(list(set(data.columns) - set(["id", "day", "rainfall"])))
    features = ['cloud', 'dewpoint', 'humidity', 'pressure', 'sunshine', 'temparature', 'windspeed']
    data = weather_engineering(data, features)
    data = date_engineering(data, features)
    data = timeseries_engineering(data, features)
    return data


# engineer data
# NOTE: this is gonna be a lot of features as you can imagine!! But we will do some good selection among them afterwards, so keep an eye on :)
df_train = feature_engineering(df_train) 
df_test = feature_engineering(df_test) 
print(df_train.shape)
df_train.tail(1)


df_train.info(verbose=True, show_counts=True)


# remove highly correlated features 
# set correlation matrix
features = sorted(list(set(df_train.columns) - set(["id", "day", "rainfall"])))
corr_matrix = df_train[features].corr().abs().round(2)
# select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# find features with correlation greater than threshold
threshold = 0.95 # NOTE: you can select as you wish
highly_corr = [column for column in upper.columns if any(upper[column] > threshold)]
print("Number of highly correlated columns:", len(highly_corr))
# drop features 
df_train.drop(highly_corr, axis=1, inplace=True)
df_test.drop(highly_corr, axis=1, inplace=True)
print("Shape becomes:", df_train.shape)


# check kolmogorov-smirnov stats
ks_results = {}
features = sorted(list(set(df_train.columns) - set(["id", "day", "rainfall"])))
for idx, col in enumerate(features):
    # split the column based on target values
    group_0 = df_train[df_train["rainfall"] == 0][col].dropna()
    group_1 = df_train[df_train["rainfall"] == 1][col].dropna()
    # compute Kolmogorov-Smirnov statistic
    ks_stat, _ = ks_2samp(group_0, group_1)
    ks_results[col] = ks_stat

# ks dataset
df_ks = pd.DataFrame.from_dict(ks_results, orient="index", columns=["ks_stat"]).sort_values(by=['ks_stat'], ascending=False).reset_index(names=["feature"])
df_ks


# remove meaningless features
ks_threshold = 0.3
meaningless_cols = df_ks[df_ks.ks_stat < ks_threshold].feature
print("Number of low KS-value columns:", len(meaningless_cols))
df_train.drop(meaningless_cols, axis=1, inplace=True)
df_test.drop(meaningless_cols, axis=1, inplace=True)
print("Shape becomes:", df_train.shape)


# set features
features = sorted(list(set(df_train.columns) - set(["id", "rainfall"])))
len(features)


#### plot numerical features: distribution + rolling window
nrows = int(len(features) / 2) + (len(features) % 2 > 0)
fig, axes = plt.subplots(nrows, 4, figsize=(20, nrows * 5))
axes = axes.ravel()
rolling_num = round(len(df_train) / 5)
for idx, col in enumerate(features):
    # 1.1. left plot: distribution
    ax_dist = axes[idx * 2]
    # KDE plot
    sns.kdeplot(data=df_train, x=col, hue="rainfall", ax=ax_dist, common_norm=False)
    ax_dist.set_title(f"Distribution of {col}")
    # 1.2. right plot: rolling window correlation
    ax_roll = axes[idx * 2 + 1]
    temp = df_train.sort_values(col)
    temp.reset_index(inplace=True)
    ax_roll.scatter(temp.index, temp["rainfall"].rolling(rolling_num).mean(), s=1, alpha=0.5, label="Rolling Target Mean")
    # null value analysis for rolling window
    null_mask = temp[col].isnull()
    if null_mask.sum() > 0:
        null_target_mean = temp.loc[null_mask, "rainfall"].mean()
        null_proportion = null_mask.mean()
        ax_roll.axhline(null_target_mean, color='red', linestyle='--', label=f"Null ({null_proportion:.1%}) Target Mean: {null_target_mean:.2f}")
    ax_roll.set_title(f"Rolling Correlation of {col}")
    ax_roll.legend(loc='best')
    del temp
# hide unused numerical subplots
for ax in axes[len(features) * 2:]:
    ax.set_visible(False)
fig.tight_layout()
plt.show()

