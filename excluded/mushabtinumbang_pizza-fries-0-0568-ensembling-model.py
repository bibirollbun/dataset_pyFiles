# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install koolbox scikit-learn==1.5.2
!pip install lightautoml
!pip install ipywidgets
!pip install autogluon.tabular==1.3
!pip install ray==2.10.0
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


# Core Python
import warnings
from itertools import combinations

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model Training and Evaluation
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Encoding
from category_encoders.target_encoder import TargetEncoder

# Progress Bar
from tqdm import tqdm

# LightGBM

# XGBoost
import xgboost as xgb
from xgboost import XGBRegressor

# ensemble models
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from koolbox import Trainer

# Hyperparameter Optimization
import optuna

warnings.simplefilter('ignore')
pd.set_option('display.max_columns', 1000)

def apply_numerical(train, test):
    numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

    for curr, df in enumerate([train, test]):
        df_new = df.copy()
        for i in range(len(numerical_features)):
            for j in range(i + 1, len(numerical_features)):
                feature1 = numerical_features[i]
                feature2 = numerical_features[j]
                cross_term_name = f"{feature1}_x_{feature2}"
                df_new[cross_term_name] = df_new[feature1] * df_new[feature2]

        if curr == 0:
            train = df_new

        else:
            test = df_new
            
    return train, test

# TE Function
def target_encode(train, test, columns, pair_size, suffix=f'_mean', smooth=0.0):
    encoder = TargetEncoder(smoothing=smooth)

    for i, cols in enumerate(list(combinations(columns, pair_size))):
        
        new_col_name = '_'.join(cols) + f'{suffix}'
        train[new_col_name] = train[cols[0]].astype(str)
        for col in cols[1:]:
            train[new_col_name] = train[new_col_name] + '_' + train[col].astype(str)

        test[new_col_name] = test[cols[0]].astype(str)
        for col in cols[1:]:
            test[new_col_name] = test[new_col_name] + '_' + test[col].astype(str)

        train[new_col_name] = encoder.fit_transform(train[new_col_name], train['Calories'])
        train[new_col_name] = train[new_col_name].astype('float32')
        
        test[new_col_name] = encoder.transform(test[new_col_name])
        test[new_col_name] = test[new_col_name].astype('float32')

    return train, test

# remove unnecessarry columns
def remove_col(train, test, binned_cols):
    # determine columns to remove (unencoded)
    cols = binned_cols
    cols.remove('Sex')
    train.drop(columns=binned_cols, errors="ignore", inplace=True)
    test.drop(columns=binned_cols, errors="ignore", inplace=True)

    return train, test

### BINNING SECTION
def apply_bin_te(train, test):
    # Creating new column 'BMI'
    train['BMI']=train['Weight']/(train['Height'] ** 2).round(2)
    test['BMI']=test['Weight']/(test['Height'] ** 2).round(2)

    # age binning
    train['age_bin'] = pd.cut(train['Age'], bins=[0, 30, 50, 70, 100], labels=['young', 'millenial', 'boomer', 'grandpa'])
    test['age_bin'] = pd.cut(test['Age'], bins=[0, 30, 50, 70, 100], labels=['young', 'millenial', 'boomer', 'grandpa'])
    
    # bmi binning
    train['BMI_bin'] = pd.cut(train['BMI'], bins=[0, 20, 30, 40, 50], labels=['young', 'millenial', 'boomer', 'grandpa'])
    test['BMI_bin'] = pd.cut(test['BMI'], bins=[0, 20, 30, 40, 50], labels=['young', 'millenial', 'boomer', 'grandpa'])
    
    # Set 3 categorical columns for TE
    binned_cols = ['Sex', 'age_bin', 'BMI_bin']
    for r in [1, 2, 3]:
        print(f"Target Encoding r={r}..")
        train, test = target_encode(
            train=train, test=test, 
            columns=binned_cols, 
            pair_size=r, 
            smooth=0.0, 
        )
    
    # remove unprocessed TE cols (ONLY IF WE RUN THE PREVIOUS LINES / SET CATEGORICAL PAIRS FOR TE)
    train, test = remove_col(train, test, binned_cols)

    return train, test

def clean_duplicates(df):
    feature_cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    target_col = 'Calories'
    
    df['feature_key'] = df[feature_cols].apply(lambda row: '_'.join(map(str, row)), axis=1)
    target_means = df.groupby('feature_key')[target_col].transform('mean')
    duplicate_mask = df.duplicated(subset=feature_cols, keep=False)    
    
    df.loc[duplicate_mask, target_col] = target_means[duplicate_mask]
    df = df.drop(columns=['feature_key'])

    return df
    
def apply_ratio(train, test):
    # weight_per_height 
    train['weight_per_height'] = train['Weight'] / (train['Height'] + 1)
    test['weight_per_height'] = test['Weight'] / (test['Height'] + 1)
    
    # weight_per_age
    train['weight_per_age'] = train['Weight'] / (train['Age'] + 1)
    test['weight_per_age'] = test['Weight'] / (test['Age'] + 1)
    
    # hr_per_weight
    train['hr_per_weight'] = train['Heart_Rate'] / train['Weight']
    test['hr_per_weight'] = test['Heart_Rate'] / test['Weight']
    
    # duration_per_age
    train['duration_per_age'] = train['Duration'] / (train['Age'] + 1)
    test['duration_per_age'] = test['Duration'] / (test['Age'] + 1 )
    
    # duration_per_hr
    train['duration_per_hr']=train['Duration']*train['Heart_Rate']
    test['duration_per_hr']=test['Duration']*test['Heart_Rate']
    
    # hr_per_maximum_hr
    train['hr_per_maximum_hr'] = train['Heart_Rate'] / (220 - train['Age'] + 1)
    test['hr_per_maximum_hr'] = test['Heart_Rate'] / (220 - test['Age'] + 1)
    
    # hr_per_minute
    train['hr_per_minute'] = train['Heart_Rate'] / train['Duration']
    test['hr_per_minute'] = test['Heart_Rate'] / test['Duration']
    
    # body_temp_per_min
    train['body_temp_per_min'] = train['Body_Temp'] / train['Duration']
    test['body_temp_per_min'] = test['Body_Temp'] / test['Duration']
    
    # duration_per_weight
    train['duration_per_weight'] = train['Duration'] * train['Weight']
    test['duration_per_weight'] = test['Duration'] * test['Weight']
    
    # Intensity
    train['duration_per_weight']=train['Duration']/train['Weight']
    test['duration_per_weight']=test['Duration']/test['Weight']

    return train, test

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df[((df["Calories"] / df["Duration"] < 11) | (df["Duration"] > 22)) & 
              (df["Calories"] < 300)].copy()
    
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Add BMI column (Weight / Height)
    df['BMI'] = df['Weight'] / df['Height']
    
    return df

def add_feature_cross_terms(df, list1, list2):
    df_new = df.copy()
    # Ä°ki liste arasÄ±ndaki ikili Ã§arpÄ±mlar
    for feature1 in list1:
        for feature2 in list2:
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

def add_categorical_aggregations(df):
    # Kategorik sÃ¼tunlar
    categorical_cols = ['Sex']
    # SayÄ±sal sÃ¼tunlar
    numerical_cols = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']
    
    # TÃ¼m kategorik sÃ¼tun kombinasyonlarÄ± iÃ§in dÃ¶ngÃ¼
    for i in range(1, len(categorical_cols) + 1):
        if i == 1:  # Tek kategorik sÃ¼tun
            for cat_col in categorical_cols:
                # Her sayÄ±sal sÃ¼tun iÃ§in agregasyonlar
                aggs = df.groupby(cat_col).agg({
                    num_col: ['min', 'max'] for num_col in numerical_cols
                })
                
                # SÃ¼tun isimlerini dÃ¼zleÅŸtir
                aggs.columns = [f"{cat_col}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                
                # Ana veri Ã§erÃ§evesiyle birleÅŸtir
                df = df.merge(aggs, on=cat_col, how='left')
        
        elif i == 2:  # Ä°kili kategorik sÃ¼tun kombinasyonlarÄ±
            for j in range(len(categorical_cols)):
                for k in range(j+1, len(categorical_cols)):
                    cat_col1 = categorical_cols[j]
                    cat_col2 = categorical_cols[k]
                    
                    # Her sayÄ±sal sÃ¼tun iÃ§in agregasyonlar
                    aggs = df.groupby([cat_col1, cat_col2]).agg({
                        num_col: ['min', 'max'] for num_col in numerical_cols
                    })
                    
                    # SÃ¼tun isimlerini dÃ¼zleÅŸtir
                    aggs.columns = [f"{cat_col1}_{cat_col2}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                    
                    # Ana veri Ã§erÃ§evesiyle birleÅŸtir
                    df = df.merge(aggs, on=[cat_col1, cat_col2], how='left')
        
        elif i == 3:  # ÃœÃ§lÃ¼ kategorik sÃ¼tun kombinasyonu
            # TÃ¼m kategorik sÃ¼tunlar iÃ§in agregasyonlar
            aggs = df.groupby(categorical_cols).agg({
                num_col: ['min', 'max'] for num_col in numerical_cols
            })
            
            # SÃ¼tun isimlerini dÃ¼zleÅŸtir
            aggs.columns = [f"all_cat_{num_col}_{agg}" for num_col, agg in aggs.columns]
            
            # Ana veri Ã§erÃ§evesiyle birleÅŸtir
            df = df.merge(aggs, on=categorical_cols, how='left')
    
    return df

def apply_200_features(train, test):
    unique_durations_train = train['Duration'].unique()
    unique_durations_test = test['Duration'].unique()
    
    # Train iÃ§in her bir Duration deÄŸeri iÃ§in yeni feature'lar oluÅŸturalÄ±m
    for duration in unique_durations_train:
        # Yeni sÃ¼tun isimleri oluÅŸturalÄ±m
        heart_rate_col = f'Heart_Rate_Duration_{int(duration)}'
        body_temp_col = f'Body_Temp_Duration_{int(duration)}'
        
        # Yeni sÃ¼tunlarÄ± oluÅŸturalÄ±m
        # EÄŸer Duration deÄŸeri belirli bir deÄŸere eÅŸitse Heart_Rate ve Body_Temp deÄŸerlerini al, deÄŸilse 0 yap
        train[heart_rate_col] = np.where(train['Duration'] == duration, train['Heart_Rate'], 0)
        train[body_temp_col] = np.where(train['Duration'] == duration, train['Body_Temp'], 0)
    
    # Test iÃ§in her bir Duration deÄŸeri iÃ§in yeni feature'lar oluÅŸturalÄ±m
    for duration in unique_durations_test:
        # Yeni sÃ¼tun isimleri oluÅŸturalÄ±m
        heart_rate_col = f'Heart_Rate_Duration_{int(duration)}'
        body_temp_col = f'Body_Temp_Duration_{int(duration)}'
        
        # Yeni sÃ¼tunlarÄ± oluÅŸturalÄ±m
        # EÄŸer Duration deÄŸeri belirli bir deÄŸere eÅŸitse Heart_Rate ve Body_Temp deÄŸerlerini al, deÄŸilse 0 yap
        test[heart_rate_col] = np.where(test['Duration'] == duration, test['Heart_Rate'], 0)
        test[body_temp_col] = np.where(test['Duration'] == duration, test['Body_Temp'], 0)
    
    unique_ages_train = train['Age'].unique()
    unique_ages_test = test['Age'].unique()
    
    # Train iÃ§in her bir Age deÄŸeri iÃ§in yeni feature'lar oluÅŸturalÄ±m
    for age in unique_ages_train:
        # Yeni sÃ¼tun isimleri oluÅŸturalÄ±m
        heart_rate_col = f'Heart_Rate_Age_{int(age)}'
        body_temp_col = f'Body_Temp_Age_{int(age)}'
        
        # Yeni sÃ¼tunlarÄ± oluÅŸturalÄ±m
        # EÄŸer Age deÄŸeri belirli bir deÄŸere eÅŸitse Heart_Rate ve Body_Temp deÄŸerlerini al, deÄŸilse 0 yap
        train[heart_rate_col] = np.where(train['Age'] == age, train['Heart_Rate'], 0)
        train[body_temp_col] = np.where(train['Age'] == age, train['Body_Temp'], 0)
    
    # Test iÃ§in her bir Age deÄŸeri iÃ§in yeni feature'lar oluÅŸturalÄ±m
    for age in unique_ages_test:
        # Yeni sÃ¼tun isimleri oluÅŸturalÄ±m
        heart_rate_col = f'Heart_Rate_Age_{int(age)}'
        body_temp_col = f'Body_Temp_Age_{int(age)}'
        
        # Yeni sÃ¼tunlarÄ± oluÅŸturalÄ±m
        # EÄŸer Age deÄŸeri belirli bir deÄŸere eÅŸitse Heart_Rate ve Body_Temp deÄŸerlerini al, deÄŸilse 0 yap
        test[heart_rate_col] = np.where(test['Age'] == age, test['Heart_Rate'], 0)
        test[body_temp_col] = np.where(test['Age'] == age, test['Body_Temp'], 0)
    
    list1 = ['Duration', 'Heart_Rate', 'Body_Temp']
    list2 = ['Sex']
    
    train = add_feature_cross_terms(train, list1, list2)
    test = add_feature_cross_terms(test, list1, list2)
    
    # Sex deÄŸerlerini tersine Ã§evirme (0 -> 1, 1 -> 0)
    train['Sex_Reversed'] = 1 - train['Sex']
    test['Sex_Reversed'] = 1 - test['Sex']
    
    list1 = ['Duration', 'Heart_Rate', 'Body_Temp']
    list2 = ['Sex', 'Sex_Reversed']
    
    train = add_feature_cross_terms(train, list1, list2)
    test = add_feature_cross_terms(test, list1, list2)
    train.drop(columns=['Sex_Reversed'],inplace=True)
    test.drop(columns=['Sex_Reversed'],inplace=True)

    # KullanÄ±m Ã¶rneÄŸi:
    train = add_categorical_aggregations(train)
    test = add_categorical_aggregations(test)

    # Check if column order is the same
    columns_match = train.columns.equals(test.columns.append(pd.Index(['Calories'])))
    print(f"Is column order the same: {columns_match}")
    
    # If column order is not the same, fix it
    if not columns_match:
        # Drop the Calories column from train
        train_without_calories = train.drop(columns=['Calories'])
        
        # Get the order of columns from test that exist in train
        common_columns = [col for col in test.columns if col in train_without_calories.columns]
        
        # Reorder train and test DataFrames using the new order
        train_without_calories = train_without_calories[common_columns]
        test = test[common_columns]
        
        # Add the Calories column back to train
        train = pd.concat([train_without_calories, train['Calories']], axis=1)
        
        print("Column order has been fixed.")
    
    # Check again after dropping Calories
    train_without_calories = train.drop(columns=['Calories'])
    columns_match_after_drop = train_without_calories.columns.equals(test.columns)
    print(f"Is column order the same after dropping 'Calories': {columns_match_after_drop}")

    return (train, test)
    
class config_:
    # data paths
    TRAIN_DATA_PATH = '/kaggle/input/playground-series-s5e5/train.csv'
    TEST_DATA_PATH = '/kaggle/input/playground-series-s5e5/test.csv'
    SUBMISSION_DATA_PATH = '/kaggle/input/playground-series-s5e5/sample_submission.csv'
    MODEL_PARAMS_PATH = '/kaggle/input/model-params/model_params.json'
    OOF_PREDS_PATH = '/kaggle/input/oof-preds/'
    
    # dict mapper
    sex_dict = {
        'male' : 0,
        'female' : 1
    }

    # target
    target = "Calories"

    # learning params
    metric = root_mean_squared_error
    n_folds = 5
    seed = 42
    cv = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    
    n_optuna_trials = 250

    # pipeline params
    split_gender = False
    FE_apply_numerical = True
    FE_apply_numerical_ensemble = False
    FE_apply_200_features = True
    autogluon_best_model_only = True
    RUN_mutual_info = False
    RUN_singlemodel = False
    RUN_ensemble = True
    RUN_autogluon = True
    RUN_lightautoml = True
    RUN_train_lightautoml = False
    RUN_XGB = False
    RUN_CB = False
    RUN_LGBM = True
    RUN_optuna = True


# Set PARAMS
config_.split_gender = False # Set to False if you want to train the model with all genders at once, True if you want to split model training by genders
config_.FE_apply_numerical = True # Set to False if you dont want to use the apply_numerical function for Feature Engineering
config_.FE_apply_200_features = False # Set to True to apply 200 features.
config_.FE_apply_numerical_ensemble = False # Set to True to apply the numerical feature engineering for ensemble models
config_.RUN_ensemble = True # False to run single model only. True to run ensembling
config_.RUN_mutual_info = False # True to run mutual info regression for each features (might take a while)
config_.RUN_singlemodel = False # True to run single model functions
config_.RUN_optuna = True # True to run optuna hyperparameter optimization for Ridge
config_.RUN_autogluon = True # True to use AutoGluon as one of the ensemble models
config_.autogluon_best_model_only = False # True to use the the best model of autogluon only, False to use all
config_.RUN_lightautoml = True # True to use lightautoml as one of the ensemble models
config_.RUN_train_lightautoml = False # True to train lightautoml from scratch again (might take a while depending on how much features youre using)
config_.RUN_XGB = True # True to use XGB as one of the ensemble models
config_.RUN_CB = True # True to use CatBoost as one of the ensemble models
config_.RUN_LGBM = True # True to use LGBM as one of the ensemble models


# read
train = pd.read_csv(config_.TRAIN_DATA_PATH, index_col='id')
test = pd.read_csv(config_.TEST_DATA_PATH, index_col='id')

# map gender
train['Sex'] = train['Sex'].replace(config_.sex_dict)
test['Sex'] = test['Sex'].replace(config_.sex_dict)

# remove duplicates
cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
train = clean_duplicates(train)
train = train.drop_duplicates()
train = train.groupby(by=cols)['Calories'].min().reset_index()

# apply feature eng
if config_.FE_apply_numerical:
    train, test = apply_numerical(train, test)

if config_.FE_apply_200_features:
    train, test = apply_200_features(train, test)
    
# set x n y
X = train.drop(config_.target, axis=1)
y = np.log1p(train[config_.target])
X_test = test

print(train.shape)
train.head()


from IPython.display import display

if config_.RUN_mutual_info:
    mutual_info = mutual_info_regression(X, y, random_state=config_.seed)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X.columns
    mutual_info = pd.DataFrame(mutual_info.sort_values(ascending=False), columns=['Mutual Information'])
    display(mutual_info.style.bar(subset=['Mutual Information'], cmap='RdYlGn'))



import time 

if config_.RUN_singlemodel:
    FOLDS = 5
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    models = {
        'CatBoost': CatBoostRegressor(verbose=100, random_seed=42, cat_features=['Sex'], early_stopping_rounds=100),
        'XGBoost': XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9, n_estimators=2000, learning_rate=0.02,
                                gamma=0.01, max_delta_step=2, early_stopping_rounds=100, eval_metric='rmse',
                                enable_categorical=True, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=10, colsample_bytree=0.7,
                                  subsample=0.9, random_state=42, verbose=-1)
    }
    
    if config_.split_gender:
        # Dataset split
        train_male = train[train["Sex"] == 0]
        train_female = train[train["Sex"] == 1]
        test_male = test[test["Sex"] == 0]
        test_female = test[test["Sex"] == 1]
    
        # Split data by gender
        datasets = [
            ("male", train_male, test_male),
            ("female", train_female, test_female)
        ]
        results = {
            model_name: {
                gender: {
                    'oof': np.zeros(len(train_df)),
                    'pred': np.zeros(len(test_df)),
                    'rmsle': []
                }
                for gender, train_df, test_df in datasets
            }
            for model_name in models
        }
    
        for name, model in models.items():
            for gender, train_df, test_df in datasets:
                print(f"\n=== Training {name} ({gender}) ===")
                train_df = train_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)
    
                X = train_df.drop(columns=["Calories"])
                y = np.log1p(train_df["Calories"])
                X_test = test_df
    
                for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
                    print(f"\nFold {i + 1}")
                    x_train, y_train = X.iloc[train_idx], y[train_idx]
                    x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
                    x_test = X_test.copy()
    
                    start = time.time()
                    if name == 'XGBoost':
                        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
                    elif name == 'CatBoost':
                        model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
                    else:
                        model.fit(x_train, y_train)
    
                    oof_pred = model.predict(x_valid)
                    test_pred = model.predict(x_test)
    
                    results[name][gender]['oof'][valid_idx] = oof_pred
                    results[name][gender]['pred'] += test_pred / FOLDS
    
                    rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
                    results[name][gender]['rmsle'].append(rmsle)
    
                    print(f"Fold {i + 1} RMSLE: {rmsle:.4f}")
                    print(f"Training time: {time.time() - start:.1f} sec")
    
        print("\n=== Model Comparison ===")
        for name in models:
            for gender in ['male', 'female']:
                mean_rmsle = np.mean(results[name][gender]['rmsle'])
                std_rmsle = np.std(results[name][gender]['rmsle'])
                print(f"{name} ({gender}) - Mean RMSLE: {mean_rmsle:.4f} Â± {std_rmsle:.4f}")
    
    else:
        # Train using all data without gender split
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        X_test = test.reset_index(drop=True)
    
        results = {
            name: {'oof': np.zeros(len(train)), 'pred': np.zeros(len(test)), 'rmsle': []}
            for name in models
        }
    
        for name, model in models.items():
            print(f"\n=== Training {name} ===")
            for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
                print(f"\nFold {i + 1}")
                x_train, y_train = X.iloc[train_idx], y[train_idx]
                x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
                x_test = X_test.copy()
    
                start = time.time()
                if name == 'XGBoost':
                    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
                elif name == 'CatBoost':
                    model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
                else:
                    model.fit(x_train, y_train)
    
                oof_pred = model.predict(x_valid)
                test_pred = model.predict(x_test)
    
                results[name]['oof'][valid_idx] = oof_pred
                results[name]['pred'] += test_pred / FOLDS
    
                rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
                results[name]['rmsle'].append(rmsle)
    
                print(f"Fold {i + 1} RMSLE: {rmsle:.4f}")
                print(f"Training time: {time.time() - start:.1f} sec")
    
        print("\n=== Model Comparison ===")
        for name in models:
            mean_rmsle = np.mean(results[name]['rmsle'])
            std_rmsle = np.std(results[name]['rmsle'])
            print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} Â± {std_rmsle:.4f}")



if config_.RUN_singlemodel:
    if not config_.split_gender:
        # Get best model (overall)
        best_model_name, best_model_score = min(
            [(model, np.mean(results[model]['rmsle'])) for model in results],
            key=lambda x: x[1]
        )    
        best_model = models[best_model_name[0]]
    
        # Feature importance
        if best_model_name == 'CatBoost':
            feature_importance = best_model.get_feature_importance()
            feature_names = X.columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)
    
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title(f'{best_model_name} Feature Importance')
            plt.show()
    
        # Predict and post-process
        y_preds = np.expm1(results[best_model_name]['pred'])
        y_preds = np.clip(y_preds, 1, 314)
    
        submission['Calories'] = y_preds
        submission.to_csv(f'submission_{best_model_name}_{best_model_name[1]}.csv', index=False)
    
        print("\nSubmission Head:")
        print(submission.head())
        print(f"\nPredict Mean: {y_preds.mean():.2f}")
        print(f"Predict Median: {np.median(y_preds):.2f}")
    
    else:
        # Compute best model name globally (average of male and female)
        best_model_name, avg_rmsle = min(
            [(name, (np.mean(r['male']['rmsle']) + np.mean(r['female']['rmsle'])) / 2) for name, r in results.items()],
            key=lambda x: x[1]
        )
        best_model = models[best_model_name]
    
        # Define male/female best model name using same global best
        best_model_name_male = best_model_name
        best_model_name_female = best_model_name
    
        # Feature importance per gender (if applicable)
        for gender in ['male', 'female']:
            if best_model_name == 'CatBoost':
                feature_importance = best_model.get_feature_importance()
                feature_names = train.drop(columns=["Calories"]).columns
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
                importance_df = importance_df.sort_values('Importance', ascending=False)
    
                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.title(f'{best_model_name} Feature Importance ({gender})')
                plt.show()
    
        # Transform preds male
        y_preds_male = np.expm1(results[best_model_name]['male']['pred'])
        y_preds_male = np.clip(y_preds_male, 1, 314)
    
        # Transform preds female
        y_preds_female = np.expm1(results[best_model_name]['female']['pred'])
        y_preds_female = np.clip(y_preds_female, 1, 314)
    
        # Assign predictions
        test_male["Calories"] = y_preds_male
        test_female["Calories"] = y_preds_female
    
        # Combine & save submission
        test_all = pd.concat([test_male, test_female])
        submission = test_all.sort_values(by="id").reset_index()
        submission = submission[["id", "Calories"]]
        submission.to_csv(f'submission_{best_model_name}_gendersplit_{avg_rmsle:.4f}.csv', index=False)


# def model_optimize(trial, x, y, model_name):
#     # Split data
#     x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

#     if model_name == "xgb":
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#             'max_depth': trial.suggest_int('max_depth', 3, 10),
#             'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
#             'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
#             'gamma': trial.suggest_float('gamma', 0.0, 1.0),
#             'tree_method': 'gpu_hist',  # GPU enabled
#             'predictor': 'gpu_predictor',
#             'objective': 'reg:squarederror',
#             'random_state': 42,
#             'n_jobs': -1
#         }
#         model = XGBRegressor(**params)
#         model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=50, verbose=False)

#     elif model_name == "lightgbm":
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#             'num_leaves': trial.suggest_int('num_leaves', 20, 512),
#             'max_depth': trial.suggest_int('max_depth', 3, 12),
#             'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
#             'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
#             'device': 'gpu',  # GPU enabled
#             'random_state': 42,
#             'n_jobs': -1,
#             'verbose' : -1
#         }
#         model = LGBMRegressor(**params)
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")  # suppress warnings from LightGBM
#             model.fit(
#                 x_train, y_train,
#                 eval_set=[(x_val, y_val)],
#                 eval_metric='rmse',
#                 callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
#             )

#     elif model_name == "catboost":
#         params = {
#             'iterations': trial.suggest_int('iterations', 300, 1500),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#             'depth': trial.suggest_int('depth', 4, 10),
#             'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
#             'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
#             'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
#             'loss_function': 'RMSE',
#             'verbose': False,
#             'task_type': 'GPU',  # GPU enabled
#             'random_state': 42
#         }
#         model = CatBoostRegressor(**params)
#         model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50)

#     else:
#         raise ValueError("model_name must be one of: 'xgb', 'lightgbm', or 'catboost'")

#     # Predict and calculate RMSLE
#     y_pred = np.maximum(0, model.predict(x_val))
#     rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred)))
#     return rmsle


# model_list = {
#     0 : "xgb",
#     1 : "catboost",
#     2 : "lightgbm"
# } 

# params = {
#     "male": {},
#     "female": {}
# }

# for gender, df in [("male", train[train["Sex"] == 0]), ("female", train[train["Sex"] == 1])]:
#     for curr_opt in range(3):  # assuming you're using 3 models (0, 1, 2)
#         model = model_list[curr_opt]

#         print(f"\n=== Optimizing for {gender.upper()} with {model.upper()} ===")

#         x = df.drop(columns=['Sex', 'Calories'])
#         y = np.log1p(df['Calories'])  # log transform for RMSLE

#         study = optuna.create_study(direction="minimize")
#         study.optimize(lambda trial: model_optimize(trial, x, y, model_name=model), n_trials=50, show_progress_bar=True)

#         print(f"\nBest RMSLE for {gender} + {model}: {study.best_value}")
#         print(f"Best Params:\n{study.best_params}")

#         # Store best parameters
#         params[gender][model] = study.best_params

# import json

# with open("model_params.json", "w") as f:
#     json.dump(params, f, indent=4)


histgb_params = {
    "l2_regularization": 10.412017522533768,
    "learning_rate": 0.011702680619474444,
    "max_depth": 59,
    "max_features": 0.30616140080552673,
    "max_iter": 4454,
    "max_leaf_nodes": 385,
    "min_samples_leaf": 50,
    "random_state": 42
}

lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.8213924491907012,
    "learning_rate": 0.059976685297931195,
    "min_child_samples": 10,
    "min_child_weight": 0.5425237767880097,
    "n_estimators": 50000,
    "n_jobs": -1,
    "num_leaves": 89,
    "random_state": 42,
    "reg_alpha": 2.0325709613371545,
    "reg_lambda": 87.27971117911044,
    "subsample": 0.6452823633939004,
    "verbose": -1
}

lgbm_goss_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.9068724002629094,
    "learning_rate": 0.06459027654473874,
    "min_child_samples": 39,
    "min_child_weight": 0.5337673729810578,
    "n_estimators": 50000,
    "n_jobs": -1,
    "num_leaves": 13,
    "random_state": 42,
    "reg_alpha": 1.603969498256519,
    "reg_lambda": 10.806488455621444,
    "subsample": 0.5966412222358356,
    "verbose": -1
}

xgb_params = {
    "colsample_bylevel": 0.8606487417581108,
    "colsample_bynode": 0.9410596660335436,
    "colsample_bytree": 0.9407540036296737,
    "early_stopping_rounds": 100,
    "eval_metric": "rmse",
    "gamma": 0.023260595738991977,
    "learning_rate": 0.03669372905801298,
    "max_depth": 11,
    "max_leaves": 51,
    "min_child_weight": 96,
    "n_estimators": 50000,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 2.953205886504917,
    "reg_lambda": 67.64147033446291,
    "subsample": 0.6973241930754311,
    "verbosity": 0
}

cb_params = {
    "border_count": 88,
    "colsample_bylevel": 0.7903437608890396,
    "depth": 8,
    "eval_metric": "RMSE",
    "iterations": 50000,
    "l2_leaf_reg": 6.065104074215131,
    "learning_rate": 0.030946464122148992,
    "min_child_samples": 138,
    "random_state": 42,
    "random_strength": 0.035251008593976785,
    "verbose": False
}

import json
# Load params from JSON file
with open(config_.MODEL_PARAMS_PATH, "r") as f:
    params = json.load(f)

print(params)


# set x n y
X = train.drop(config_.target, axis=1)
y = np.log1p(train[config_.target])
X_test = test
X


if not config_.split_gender:
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X_test = test.reset_index(drop=True)

else:
    # set x n y
    X = train.drop(config_.target, axis=1)
    y = np.log1p(train[config_.target])
    X_test = test

    # Dataset split
    train_male = train[train["Sex"] == 0]
    train_female = train[train["Sex"] == 1]
    test_male = test[test["Sex"] == 0]
    test_female = test[test["Sex"] == 1]

    datasets = [
        ("male", train_male, test_male),
        ("female", train_female, test_female)
    ]

scores = {}
oof_preds = {}
test_preds = {}


import glob
import joblib

if config_.RUN_autogluon:
    oof_preds_files = glob.glob(f'/kaggle/input/autogluon-ml/*_oof_preds_*.pkl')
    test_preds_files = glob.glob(f'/kaggle/input/autogluon-ml/*_test_preds_*.pkl')

    if not config_.autogluon_best_model_only:
        for i in range(len(oof_preds_files)):
            ag_oof_preds = np.log1p(joblib.load(oof_preds_files[i]))
            ag_test_preds = np.log1p(joblib.load(test_preds_files[i]))
            
            ag_scores = []
            split = KFold(n_splits=config_.n_folds, random_state=config_.seed, shuffle=True).split(X, y)
            for _, val_idx in split:
                y_val = y[val_idx]
                y_preds = ag_oof_preds[val_idx]   
                score = root_mean_squared_error(y_preds, y_val)
                ag_scores.append(score)
                
            oof_preds[f"AutoGluon {i}"], test_preds[f"AutoGluon {i}"], scores[f"AutoGluon {i}"] = ag_oof_preds, ag_test_preds, ag_scores
    else:
        ag_oof_preds = np.log1p(joblib.load(oof_preds_files[1]))
        ag_test_preds = np.log1p(joblib.load(test_preds_files[1]))
        
        ag_scores = []
        split = KFold(n_splits=config_.n_folds, random_state=config_.seed, shuffle=True).split(X, y)
        for _, val_idx in split:
            y_val = y[val_idx]
            y_preds = ag_oof_preds[val_idx]   
            score = root_mean_squared_error(y_preds, y_val)
            ag_scores.append(score)
            
        oof_preds[f"AutoGluon"], test_preds[f"AutoGluon"], scores[f"AutoGluon"] = ag_oof_preds, ag_test_preds, ag_scores


# read
train_automl = pd.read_csv(config_.TRAIN_DATA_PATH, index_col='id')
test_automl = pd.read_csv(config_.TEST_DATA_PATH, index_col='id')

# map gender
train_automl['Sex'] = train_automl['Sex'].replace(config_.sex_dict)
test_automl['Sex'] = test_automl['Sex'].replace(config_.sex_dict)

# remove duplicates
train_automl = clean_duplicates(train_automl)

# apply feature eng
if config_.FE_apply_numerical_ensemble:
    train_automl, test_automl = apply_numerical(train_automl, test_automl)

train_automl = train_automl.drop_duplicates()
train_automl[config_.target] = np.log1p(train_automl[config_.target])
train_automl[config_.target].describe()

if config_.split_gender:
    train_automl_male = train_automl[train_automl['Sex'] == 0]
    train_automl_female = train_automl[train_automl['Sex'] == 1]
    test_automl_male = test_automl[test_automl['Sex'] == 0]
    test_automl_female = test_automl[test_automl['Sex'] == 1]
    print(train_automl_male.head())
    print(train_automl_male.shape, train_automl_female.shape)
    print(test_automl_male.shape, test_automl_female.shape)

else:
    print(train_automl.shape, test_automl.shape)

train_automl = train_automl.reset_index(drop=True)
print(train_automl.head())


from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import glob
import joblib

if config_.RUN_lightautoml:
    if config_.RUN_train_lightautoml:
        task = Task('reg', metric='mse')
        automl = TabularAutoML(
            task = task, 
            timeout = 3600*10,
            cpu_limit = os.cpu_count(),
            nn_params = {
            'stop_by_metric': True,
                'verbose_bar': True,
                'n_epochs': 50,
                'device': 'cuda'},
            reader_params = {'n_jobs': os.cpu_count(), 'cv': 5, 'random_state': 42, 'advanced_roles': True},
             general_params = {"use_algos": [['dense']]}
        )
        
        if not config_.split_gender:
            out_of_fold_predictions = automl.fit_predict(
                train_automl,
                roles = {
                    'target': 'Calories',
                }, 
                verbose = 3
            )
        
        else:
            automl_male = automl
            automl_female = automl
            out_of_fold_predictions_male = automl_male.fit_predict(
                train_automl_male.reset_index(drop=True),
                roles = {
                    'target': 'Calories',
                }, 
                verbose = 3
            )
        
            out_of_fold_predictions_female = automl_female.fit_predict(
                train_automl_female.reset_index(drop=True),
                roles = {
                    'target': 'Calories',
                }, 
                verbose = 3
            )
    
    else:
        oof_preds_files = glob.glob(f'/kaggle/input/light-automl/*_oof_preds_*.pkl')
        test_preds_files = glob.glob(f'/kaggle/input/light-automl/*_test_preds_*.pkl')
        ag_oof_preds = np.log1p(joblib.load(oof_preds_files[0]))
        ag_test_preds = np.log1p(joblib.load(test_preds_files[0]))
        ag_scores = []
        split = KFold(n_splits=config_.n_folds, random_state=config_.seed, shuffle=True).split(X, y)
        for _, val_idx in split:
            y_val = y[val_idx]
            y_preds = ag_oof_preds[val_idx]   
            score = root_mean_squared_error(y_preds, y_val)
            ag_scores.append(score)
            
        oof_preds[f"LightAutoML"], test_preds[f"LightAutoML"], scores[f"LightAutoML"] = ag_oof_preds, ag_test_preds, ag_scores



import joblib

def save_preds(preds, cv_score, name, type, is_ensemble):
    base_path = "oof_preds" if type == "oof" else "test_preds"
    base_path = "." if is_ensemble else base_path
    joblib.dump(np.expm1(preds), f"{base_path}/{name}_{type}_preds_{cv_score:.6f}.pkl")

def save_submission(test_preds, score, name):
    sub = pd.read_csv(config_.SUBMISSION_DATA_PATH)
    sub[config_.target] = np.expm1(test_preds)
    sub.to_csv(f"sub_{name}_{score:.6f}.csv", index=False)

def get_oof_scores(train_automl, out_of_fold_predictions):
    # Get true target values
    y_true = train_automl['Calories'].values
    y_oof = out_of_fold_predictions.data[:, 0]
    
    # Recreate the same CV split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold_idx, (_, val_idx) in enumerate(kf.split(train_automl)):
        y_true_fold = y_true[val_idx]
        y_pred_fold = y_oof[val_idx]
        
        rmse_fold = mean_squared_error(y_true_fold, y_pred_fold, squared=False)
        fold_scores.append(rmse_fold)
    
    
    print("Per-fold CV RMSEs:", fold_scores)
    print("Mean CV RMSE:", np.mean(fold_scores))

    return fold_scores

if config_.RUN_train_lightautoml:
    if not config_.split_gender:
        # Check Score
        CV_automl = mean_squared_error(train_automl.Calories, out_of_fold_predictions.data[:, 0], squared=False)
        print(f'CV Score: {CV_automl}')
        
        # Test Preds
        y_pred = automl.predict(test).data[:, 0]
        
        # save oof
        save_preds(out_of_fold_predictions.data.ravel(), CV_automl, "LightAutoML", "oof", True)
        save_preds(y_pred, CV_automl, "LightAutoML", "test_preds", True)
        save_submission(y_pred, CV_automl, 'lightautoml')
        
        # get score
        fold_scores = get_oof_scores(train_automl, out_of_fold_predictions)
        
        # Insert for Ridge
        scores["LightAutoML"] = fold_scores
        oof_preds["LightAutoML"] = out_of_fold_predictions.data.ravel()
        test_preds["LightAutoML"] = y_pred
    
    else:
        # Combine all oof values into one DF
        train_automl_male['oof'] = out_of_fold_predictions_male.data.ravel()
        train_automl_female['oof'] = out_of_fold_predictions_female.data.ravel()
        train_automl_all = pd.concat([train_automl_male, train_automl_female])
        oof_all = train_automl_all.sort_values(by="id").reset_index()
        oof_all = oof_all.oof.values
    
        # Check Score
        CV_automl_all = mean_squared_error(train_automl.Calories, oof_all, squared=False)
        CV_automl_male = mean_squared_error(train_automl_male.Calories, out_of_fold_predictions_male.data[:, 0], squared=False)
        CV_automl_female = mean_squared_error(train_automl_female.Calories, out_of_fold_predictions_female.data[:, 0], squared=False)
        print(f'CV Score All: {CV_automl_all} \nCV Score Male: {CV_automl_male} \nCV Score Female: {CV_automl_female}')
            
        # Test Preds
        y_pred_male = automl_male.predict(test_male).data[:, 0]
        y_pred_female = automl_female.predict(test_female).data[:, 0]
        
        # save oof
        save_preds(out_of_fold_predictions_male, CV_automl_male, "LightAutoML (Male)", "oof", True)
        save_preds(out_of_fold_predictions_female, CV_automl_female, "LightAutoML (Female)", "oof", True)
        save_preds(y_pred_male, CV_automl_male, "LightAutoML (Male)", "test_preds", True)
        save_preds(y_pred_female, CV_automl_male, "LightAutoML (Female)", "test_preds", True)
    
        # Combine all preds values into one DF
        test_automl_male['Calories'] = y_pred_male
        test_automl_female['Calories'] = y_pred_female
        test_automl_all = pd.concat([test_automl_male, test_automl_female])
        y_pred_all = test_automl_all.sort_values(by="id").reset_index()
        y_pred_all = y_pred_all.Calories.values
        save_submission(y_pred_all, CV_automl_all, 'lightautoml')
        
        # get score
        fold_scores_male = get_oof_scores(train_automl_male, out_of_fold_predictions_male)
        fold_scores_female = get_oof_scores(train_automl_female, out_of_fold_predictions_female)
        
        # Insert for Ridge
        scores["LightAutoML (Male)"] = fold_scores_male
        scores["LightAutoML (Female)"] = fold_scores_female
        oof_preds["LightAutoML (Male)"] = out_of_fold_predictions_male.data.ravel()
        oof_preds["LightAutoML (Female)"] = out_of_fold_predictions_female.data.ravel()
        test_preds["LightAutoML (Male)"] = y_pred_male
        test_preds["LightAutoML (Female)"] = y_pred_female


if config_.RUN_LGBM:
    fit_args = {
        "eval_metric": "rmse",
        "callbacks": [
            log_evaluation(period=1000), 
            early_stopping(stopping_rounds=100)
        ]
    }
    
    if config_.split_gender:
        for gender, train_df, test_df in datasets:
            print(f"\nTraining LightGBM for {gender.upper()}")
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
    
            X = train_df.drop(columns=["Sex", "Calories"])
            y = np.log1p(train_df["Calories"])
            X_test = test_df.drop(columns=["Sex"])
    
            lgbm_params = params[gender]["lightgbm"]
    
            lgbm_params.update({
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
                "verbosity": -1,
                "num_threads": 2
            })
    
            lgbm_trainer = Trainer(
                LGBMRegressor(**lgbm_params),
                cv=config_.cv,
                metric=config_.metric,
                use_early_stopping=True,
                task="regression"
            )
    
            lgbm_trainer.fit(X, y, fit_args=fit_args)
    
            model_key = f"LightGBM_{gender}"
            scores[model_key] = lgbm_trainer.fold_scores
            oof_preds[model_key] = lgbm_trainer.oof_preds
            test_preds[model_key] = lgbm_trainer.predict(X_test)
    
    else:
        print(f"Training LGBM with Shape {X.shape}")
        lgbm_params.update({
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "verbosity": -1,
            "num_threads": 2
        })
    
        lgbm_trainer = Trainer(
            LGBMRegressor(**lgbm_params),
            cv=config_.cv,
            metric=config_.metric,
            use_early_stopping=True,
            task="regression"
        )
    
        lgbm_trainer.fit(X, y, fit_args=fit_args)

        y_pred = lgbm_trainer.predict(X_test)
        scores["LightGBM (gbdt)"] = lgbm_trainer.fold_scores
        oof_preds["LightGBM (gbdt)"] = lgbm_trainer.oof_preds
        test_preds["LightGBM (gbdt)"] = y_pred

        cv = np.mean(lgbm_trainer.fold_scores)
        print(f'CV Score: {cv}')
        
        print('Saving...')
        save_preds(lgbm_trainer.oof_preds, cv, "LGBM", "oof", True)
        save_preds(y_pred, cv, "LGBM", "test_preds", True)
        save_submission(y_pred, cv, 'lgbm')


xgb_params = {    
    'max_depth': 9,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'gamma': 0.01,
    'max_delta_step': 2,
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'random_state': 42,
    'early_stopping_rounds': 100,
    'tree_method': 'gpu_hist'  # GPU hÄ±zlandÄ±rma iÃ§in gpu_hist kullanÄ±yoruz
}

if config_.RUN_XGB:
    if config_.split_gender:
        fit_args = {
            "eval_metric": "rmse",
            "early_stopping_rounds": 300,
            "verbose": 1000
        }
    
        for gender, train_df, test_df in datasets:
            print(f"\nTraining XGBoost for {gender.upper()}")
    
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
    
            X = train_df.drop(columns=["Sex", "Calories"])
            y = np.log1p(train_df["Calories"])
            X_test = test_df.drop(columns=["Sex"])
    
            xgb_params = params[gender]["xgb"]
            xgb_params["n_estimators"] = 3000
            xgb_params.update({
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "n_jobs": -1,
                "objective": "reg:squarederror"
            })
    
            xgb_trainer = Trainer(
                XGBRegressor(**xgb_params),
                cv=config_.cv,
                metric=config_.metric,
                use_early_stopping=True,
                task="regression"
            )
    
            xgb_trainer.fit(X, y, fit_args=fit_args)
    
            model_key = f"XGBoost_{gender}"
            scores[model_key] = xgb_trainer.fold_scores
            oof_preds[model_key] = xgb_trainer.oof_preds
            test_preds[model_key] = xgb_trainer.predict(X_test)
    
    else:
        print(f"Training XGB with Shape {X.shape}")
        fit_args = {
            "verbose": 1000
        }
        xgb_params.update({
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            "n_jobs": -1,
            "objective": "reg:squarederror"
        })
    
        xgb_trainer = Trainer(
            XGBRegressor(**xgb_params),
            cv=config_.cv,
            metric=config_.metric,
            use_early_stopping=True,
            task="regression"
        )
    
        xgb_trainer.fit(X, y, fit_args=fit_args)

        y_pred_xgb = xgb_trainer.predict(X_test)
        scores["XGBoost"] = xgb_trainer.fold_scores
        oof_preds["XGBoost"] = xgb_trainer.oof_preds
        test_preds["XGBoost"] = y_pred_xgb

        cv = np.mean(xgb_trainer.fold_scores)
        print(f'CV Score: {cv}')
        
        print('Saving...')
        save_preds(xgb_trainer.oof_preds, cv, "XGBoost", "oof", True)
        save_preds(y_pred_xgb, cv, "XGBoost", "test_preds", True)
        save_submission(y_pred_xgb, cv, 'xgb')


cat_params = {    
    'iterations': 3000,
    'learning_rate': 0.02,
    'depth': 12,
    #'bootstrap_type':'Bernoulli',
    #'grow_policy':'Lossguide',
    #'boosting_type':'Plain',
    'loss_function': 'RMSE',
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 200,
    'cat_features': ['Sex'],
    'verbose': 100,
    'task_type': 'GPU',  # GPU kullanÄ±mÄ±nÄ± etkinleÅŸtir
    #'devices': '0',      # KullanÄ±lacak GPU cihazÄ± (0, 1, vs.)

}

if config_.RUN_CB:
    if config_.split_gender:
        fit_args = {
            "early_stopping_rounds": 300,
            "verbose": 100
        }
    
        for gender, train_df, test_df in datasets:
            print(f"\nTraining CatBoost for {gender.upper()}")
    
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
    
            X = train_df.drop(columns=["Sex", "Calories"])
            y = np.log1p(train_df["Calories"])
            X_test = test_df.drop(columns=["Sex"])
    
            cat_params = params[gender]["catboost"]
            cat_params.update({
                "task_type": "GPU",
                "devices": "0",
                "loss_function": "RMSE",
                "verbose": False,
                "random_state": 42
            })
    
            cat_trainer = Trainer(
                CatBoostRegressor(**cat_params),
                cv=config_.cv,
                metric=config_.metric,
                use_early_stopping=True,
                task="regression"
            )
    
            cat_trainer.fit(X, y, fit_args=fit_args)
    
            model_key = f"CatBoost_{gender}"
            scores[model_key] = cat_trainer.fold_scores
            oof_preds[model_key] = cat_trainer.oof_preds
            test_preds[model_key] = cat_trainer.predict(X_test)
    
    else:
        print(f"Training CatBoost with Shape {X.shape}")
        fit_args = {
            "verbose": 1000,
            "early_stopping_rounds": 100,
            "use_best_model": True
        }
    
        cb_params.update({
            "task_type": "GPU",
            "devices": "0",
        })
    
        cb_params.pop("colsample_bylevel", None)
    
        cat_trainer = Trainer(
            CatBoostRegressor(**cb_params),
            cv=config_.cv,
            metric=config_.metric,
            use_early_stopping=True,
            task="regression"
        )
    
        cat_trainer.fit(X, y, fit_args=fit_args)

        y_pred_cb = cat_trainer.predict(X_test)
        scores["CatBoost"] = cat_trainer.fold_scores
        oof_preds["CatBoost"] = cat_trainer.oof_preds
        test_preds["CatBoost"] = y_pred_cb

        
        cv = np.mean(cat_trainer.fold_scores)
        print(f'CV Score: {cv}')
        
        print('Saving...')
        save_preds(cat_trainer.oof_preds, cv, "CatBoost", "oof", True)
        save_preds(y_pred_cb, cv, "CatBoost", "test_preds", True)
        save_submission(y_pred_cb, cv, 'catboost')



if config_.RUN_XGB & config_.RUN_CB:
    save_submission((y_pred_cb + y_pred_xgb)/2, (np.mean(scores['CatBoost']) + np.mean(scores['XGBoost'])) / 2, 'CB+XGB')


import glob
import joblib

if config_.RUN_autogluon:
    oof_preds_files = glob.glob(f'/kaggle/input/autogluon-ml/*_oof_preds_*.pkl')
    test_preds_files = glob.glob(f'/kaggle/input/autogluon-ml/*_test_preds_*.pkl')

    if not config_.autogluon_best_model_only:
        for i in range(len(oof_preds_files)):
            ag_oof_preds = np.log1p(joblib.load(oof_preds_files[i]))
            ag_test_preds = np.log1p(joblib.load(test_preds_files[i]))
            
            ag_scores = []
            split = KFold(n_splits=config_.n_folds, random_state=config_.seed, shuffle=True).split(X, y)
            for _, val_idx in split:
                y_val = y[val_idx]
                y_preds = ag_oof_preds[val_idx]   
                score = root_mean_squared_error(y_preds, y_val)
                ag_scores.append(score)
                
            oof_preds[f"AutoGluon {i}"], test_preds[f"AutoGluon {i}"], scores[f"AutoGluon {i}"] = ag_oof_preds, ag_test_preds, ag_scores
    else:
        ag_oof_preds = np.log1p(joblib.load(oof_preds_files[1]))
        ag_test_preds = np.log1p(joblib.load(test_preds_files[1]))
        
        ag_scores = []
        split = KFold(n_splits=config_.n_folds, random_state=config_.seed, shuffle=True).split(X, y)
        for _, val_idx in split:
            y_val = y[val_idx]
            y_preds = ag_oof_preds[val_idx]   
            score = root_mean_squared_error(y_preds, y_val)
            ag_scores.append(score)
            
        oof_preds[f"AutoGluon"], test_preds[f"AutoGluon"], scores[f"AutoGluon"] = ag_oof_preds, ag_test_preds, ag_scores


import joblib

if config_.split_gender:
    male_oof_preds = {}
    female_oof_preds = {}
    male_test_preds = {}
    female_test_preds = {}
    
    if RUN_lightautoml:
        male_oof_preds['LightAutoML_male'] = oof_preds['LightAutoML (Male)']
        female_oof_preds['LightAutoML_female'] = oof_preds['LightAutoML (Female)']
        male_test_preds['LightAutoML_male'] = test_preds['LightAutoML (Male)']
        female_test_preds['LightAutoML_female'] = test_preds['LightAutoML (Female)']
    
    if RUN_LGBM:
        male_oof_preds['LightGBM_male'] = oof_preds['LightGBM_male']
        female_oof_preds['LightGBM_female'] = oof_preds['LightGBM_female']
        male_test_preds['LightGBM_male'] = test_preds['LightGBM_male']
        female_test_preds['LightGBM_female'] = test_preds['LightGBM_female']
    
    if RUN_XGB:
        male_oof_preds['XGBoost_male'] = oof_preds['XGBoost_male']
        female_oof_preds['XGBoost_female'] = oof_preds['XGBoost_female']
        male_test_preds['XGBoost_male'] = test_preds['XGBoost_male']
        female_test_preds['XGBoost_female'] = test_preds['XGBoost_female']
    
    if RUN_CB:
        male_oof_preds['CatBoost_male'] = oof_preds['CatBoost_male']
        female_oof_preds['CatBoost_female'] = oof_preds['CatBoost_female']
        male_test_preds['CatBoost_male'] = test_preds['CatBoost_male']
        female_test_preds['CatBoost_female'] = test_preds['CatBoost_female']


    X_male = pd.DataFrame(male_oof_preds)
    X_test_male = pd.DataFrame(male_test_preds)
    
    X_female = pd.DataFrame(female_oof_preds)
    X_test_female = pd.DataFrame(female_test_preds)
    
    joblib.dump(X_male, "oof_preds_male.pkl")
    joblib.dump(X_test_male, "test_preds_male.pkl")
    
    joblib.dump(X_female, "oof_preds_female.pkl")
    joblib.dump(X_test_female, "test_preds_female.pkl")

    # Make sure to reset index if needed
    X_male = X_male.reset_index(drop=True)
    X_female = X_female.reset_index(drop=True)
    
    # y
    y_male = np.log1p(train_male["Calories"])
    y_female = np.log1p(train_female["Calories"])
    
    # Make sure to reset index if needed
    y_male = y_male.reset_index(drop=True)
    y_female = y_female.reset_index(drop=True)

    print(f"X_male shape: {X_male.shape}, y_male shape: {y_male.shape}")
    print(f"X_female shape: {X_female.shape}, y_female shape: {y_female.shape}")

else:
    X = pd.DataFrame(oof_preds)
    X_test = pd.DataFrame(test_preds)

    print(f"X shape: {X.shape},y shape: {y.shape}")


if config_.split_gender:
    def objective_male(trial):    
        params = {
            "random_state": config_.seed,
            "alpha": trial.suggest_float("alpha", 0, 10),
            "tol": trial.suggest_float("tol", 1e-6, 1e-2)
        }
        trainer = Trainer(
            Ridge(**params),
            cv=config_.cv,
            metric=config_.metric,
            task="regression",
            verbose=False
        )
        trainer.fit(X_male, y_male)
        return np.mean(trainer.fold_scores)

    def objective_female(trial):    
        params = {
            "random_state": config_.seed,
            "alpha": trial.suggest_float("alpha", 0, 10),
            "tol": trial.suggest_float("tol", 1e-6, 1e-2)
        }
        trainer = Trainer(
            Ridge(**params),
            cv=config_.cv,
            metric=config_.metric,
            task="regression",
            verbose=False
        )
        trainer.fit(X_female, y_female)
        return np.mean(trainer.fold_scores)

    if config_.RUN_optuna:
        sampler = optuna.samplers.TPESampler(seed=config_.seed, multivariate=True)

        study_male = optuna.create_study(direction="minimize", sampler=sampler)
        study_female = optuna.create_study(direction="minimize", sampler=sampler)

        study_male.optimize(objective_male, n_trials=config_.n_optuna_trials, n_jobs=4, catch=(ValueError,))
        study_female.optimize(objective_female, n_trials=config_.n_optuna_trials, n_jobs=4, catch=(ValueError,))

        best_params_male = study_male.best_params
        best_params_female = study_female.best_params

        ridge_params_male = {
            "random_state": config_.seed,
            "alpha": best_params_male["alpha"],
            "tol": best_params_male["tol"]
        }

        ridge_params_female = {
            "random_state": config_.seed,
            "alpha": best_params_female["alpha"],
            "tol": best_params_female["tol"]
        }

    else:
        ridge_params_male = {"random_state": config_.seed}
        ridge_params_female = {"random_state": config_.seed}

else:
    def objective(trial):    
        params = {
            "random_state": config_.seed,
            "alpha": trial.suggest_float("alpha", 0, 10),
            "tol": trial.suggest_float("tol", 1e-6, 1e-2)
        }
        trainer = Trainer(
            Ridge(**params),
            cv=config_.cv,
            metric=config_.metric,
            task="regression",
            verbose=True
        )
        trainer.fit(X, y)
        
        return trainer.overall_score
        
    if config_.RUN_optuna:
        sampler = optuna.samplers.TPESampler(seed=config_.seed, multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=config_.n_optuna_trials, n_jobs=-1, catch=(ValueError,))
        best_params = study.best_params

        ridge_params = {
            "random_state": config_.seed,
            "alpha": best_params["alpha"],
            "tol": best_params["tol"]
        }

    else:
        ridge_params = ridge_params = {
          "random_state": 42,
          "alpha": 1.3240892420349015,
          "tol": 0.006830172345091519
        }
                


if not config_.RUN_optuna:
    ridge_params = {
      "random_state": 42,
      "alpha": 1.3240892420349015,
      "tol": 0.006830172345091519
    }

if config_.split_gender:
    try:
        ridge_params_male
    except Exception as e:
        ridge_params_male = ridge_params
        ridge_params_female = ridge_params
    print(json.dumps(ridge_params_male, indent=2))
    print(json.dumps(ridge_params_female, indent=2))

else:
    print(json.dumps(ridge_params, indent=2))


if config_.split_gender:
    # Train Ridge model for MALE
    ridge_trainer_male = Trainer(
        Ridge(**ridge_params_male),
        cv=config_.cv,
        metric=config_.metric,
        task="regression"
    )
    ridge_trainer_male.fit(X_male, y_male)
    scores["Ridge (ensemble) Male"] = ridge_trainer_male.fold_scores
    ridge_test_preds_male = np.expm1(ridge_trainer_male.predict(X_test_male))

    # Train Ridge model for FEMALE
    ridge_trainer_female = Trainer(
        Ridge(**ridge_params_female),
        cv=config_.cv,
        metric=config_.metric,
        task="regression"
    )
    ridge_trainer_female.fit(X_female, y_female)
    scores["Ridge (ensemble) Female"] = ridge_trainer_female.fold_scores
    ridge_test_preds_female = np.expm1(ridge_trainer_female.predict(X_test_female))

else:
    # Train single Ridge model (no gender split)
    ridge_trainer = Trainer(
        Ridge(**ridge_params),
        cv=config_.cv,
        metric=config_.metric,
        task="regression"
    )
    ridge_trainer.fit(X, y)
    scores["Ridge (ensemble)"] = ridge_trainer.fold_scores
    ridge_test_preds = np.expm1(ridge_trainer.predict(X_test))



def plot_weights(weights, feature_names, title):
    sorted_indices = np.argsort(weights[0])[::-1]
    sorted_coeffs = np.array(weights[0])[sorted_indices]
    sorted_names = np.array(feature_names)[sorted_indices]

    plt.figure(figsize=(10, weights.shape[1] * 0.5))
    ax = sns.barplot(x=sorted_coeffs, y=sorted_names, palette="RdYlGn_r")

    for i, (value, name) in enumerate(zip(sorted_coeffs, sorted_names)):
        ha = "left" if value >= 0 else "right"
        ax.text(value, i, f"{value:.3f}", va="center", ha=ha, color="black")

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.1 * abs(xlim[0]), xlim[1] + 0.1 * abs(xlim[1]))

    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# --------------------------------------------
# Ridge Coefficient Plotting Based on Config
# --------------------------------------------

if config_.split_gender:
    # Male
    ridge_coeffs_male = np.zeros((1, X_male.shape[1]))
    for m in ridge_trainer_male.estimators:
        ridge_coeffs_male += m.coef_
    ridge_coeffs_male = ridge_coeffs_male / len(ridge_trainer_male.estimators)

    plot_weights(ridge_coeffs_male, X_male.columns, "Ridge Coefficients (Male)")

    # Female
    ridge_coeffs_female = np.zeros((1, X_female.shape[1]))
    for m in ridge_trainer_female.estimators:
        ridge_coeffs_female += m.coef_
    ridge_coeffs_female = ridge_coeffs_female / len(ridge_trainer_female.estimators)

    plot_weights(ridge_coeffs_female, X_female.columns, "Ridge Coefficients (Female)")

else:
    ridge_coeffs = np.zeros((1, X.shape[1]))
    for m in ridge_trainer.estimators:
        ridge_coeffs += m.coef_
    ridge_coeffs = ridge_coeffs / len(ridge_trainer.estimators)

    plot_weights(ridge_coeffs, X.columns, "Ridge Coefficients (All)")



if config_.split_gender:
    avg_rmsle_male = np.mean(ridge_trainer_male.fold_scores)
    avg_rmsle_female = np.mean(ridge_trainer_female.fold_scores)
    avg_rmsle = (avg_rmsle_male + avg_rmsle_female) / 2
    combined_preds = np.concatenate([ridge_test_preds_male, ridge_test_preds_female])
    print(f"Average RMSLE: {avg_rmsle:.4f}")
    print(f"Predict Mean (combined): {combined_preds.mean():.2f}")
    print(f"Predict Median (combined): {np.median(combined_preds):.2f}")
        
else:
    avg_rmsle = np.mean(ridge_trainer.fold_scores)
    print(f"RMSLE: {avg_rmsle:.4f}")
    print(f"Predict Mean: {ridge_test_preds.mean():.2f}")
    print(f"Predict Median: {np.median(ridge_test_preds):.2f}")


if config_.split_gender:
    # Gender Split: Assign prediction results
    test_male["Calories"] = ridge_test_preds_male
    test_female["Calories"] = ridge_test_preds_female

    # Concatenate the two DataFrames
    test_all = pd.concat([test_male, test_female])

    # Sort by 'id' column and reset index
    submission = test_all.sort_values(by="id").reset_index()

    # Set the format for submission
    submission = submission[["id", "Calories"]]


else:
    # Non-Gender Split: Assign prediction results
    sub = pd.read_csv(config_.TEST_DATA_PATH)
    sub[config_.target] = ridge_test_preds

    # Optionally clean duplicates
    sub = clean_duplicates(sub)

    # Set the format for submission
    submission = sub[["id", "Calories"]]


submission


# Save to the working directory (Kaggle format)
import os
submission_data_path = '/kaggle/working/'
submission.to_csv(os.path.join(submission_data_path, f"sub_ensemble{f'_gendersplit' if config_.split_gender else ''}_{avg_rmsle:.6f}.csv"), index=False)

