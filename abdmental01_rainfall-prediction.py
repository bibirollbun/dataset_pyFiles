%%time

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore
from IPython.display import clear_output
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import *
from sklearn.metrics import *
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
from tqdm import tqdm

def print_heading(title):
    print("*" * 50)
    print(f" {title} ")
    print("*" * 50)


%%time

SEED = 0
n_splits = 10

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv').drop('id', axis=1)
original = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')

original = original.rename(columns={'pressure ': 'pressure', 'humidity ': 'humidity', 'cloud ': 'cloud',
                                    '         winddirection':'winddirection'})
original = original[train.columns].replace({'yes': 1, 'no': 0})
train = pd.concat([train, original], axis=0, ignore_index=True)

test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

# Public Notebook Ref
def feature_engineering(df):
    df['cloud_humidity'] = df.cloud + df.humidity
    df['cloud_humidity_sunshine'] = df.cloud + df.humidity + df.sunshine
    df['cloud_sunshine'] = df.cloud * df.sunshine
    df['humidity_sunshine'] = df.humidity * df.sunshine
    df['day'] = pd.to_datetime(df['day'], errors='coerce')
    df['month'] = df['day'].dt.month
    df['day_of_week'] = df['day'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['avg_temp'] = (df['maxtemp'] + df['mintemp']) / 2
    df['temp_deviation'] = df['temparature'] - df['avg_temp']
    df['dew_point_depression'] = df['temparature'] - df['dewpoint']
    df['wind_dir_rad'] = np.deg2rad(df['winddirection'])
    df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
    df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
    df.drop(columns=['wind_dir_rad'], inplace=True)
    df['wind_chill'] = 13.12 + 0.6215 * df['temparature'] - 11.37 * (df['windspeed']**0.16) + 0.3965 * df['temparature'] * (df['windspeed']**0.16)
    df['humidity_temp'] = df['humidity'] * df['temparature']
    df['cloud_sunshine'] = df['cloud'] * df['sunshine']
    df['rolling_temp_mean'] = df['avg_temp'].rolling(window=7).mean()
    df['rolling_wind_mean'] = df['windspeed'].rolling(window=7).mean()
    df['rolling_humidity_mean'] = df['humidity'].rolling(window=7).mean()
    df['temp_lag_1'] = df['avg_temp'].shift(1)
    df['humidity_lag_1'] = df['humidity'].shift(1)
    df['windspeed_lag_1'] = df['windspeed'].shift(1)
    df['pressure_temp_interaction'] = df['pressure'] * df['avg_temp']
    df['windspeed_temp_interaction'] = df['windspeed'] * df['avg_temp']
    df['sunshine_cloud_interaction'] = df['sunshine'] * df['cloud']
    df['season'] = df['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else
                                      'Summer' if 6 <= x <= 8 else
                                      'Autumn' if 9 <= x <= 11 else 'Winter')

    for c in ['pressure', 'maxtemp', 'temparature', 'humidity']:
        for gap in [1]:
            df[c+f"_shift{gap}"] = df[c].shift(gap)
            df[c+f"_diff{gap}"] = df[c].diff(gap)

    df = pd.get_dummies(df, columns=['season'], drop_first=True)
    df.drop(columns=['day'], inplace=True)
    
    return df

train = feature_engineering(train)
test = feature_engineering(test)


%%time

y = train['rainfall']
X = train.drop('rainfall',axis=1)

def TRAIN_LGBM(p,X_test):
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"ðŸš€ Training Fold {fold + 1}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**p,
            random_seed=SEED,
            verbose=-1
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

        train_score = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        val_score = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"âœ… Fold {fold + 1} - Train AUC: {train_score:.5f}, Val AUC: {val_score:.5f}")

    final_auc = roc_auc_score(y, oof_preds)
    train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    clear_output()
    print(f"ðŸ”¥ Overall Train AUC: {train_auc:.5f}")
    print(f"ðŸŽ¯ Overall OOF AUC: {final_auc:.5f}")

    return oof_preds, test_preds


%%time

params = {'n_jobs': -1}

oof_preds, test_preds = TRAIN_LGBM(params, test) # CV : 0.88163


%%time

sample["rainfall"] = test_preds
sample.to_csv("submission.csv", index=False)
print_heading("Sub shape:")
print(sample.shape)
sample.head()

