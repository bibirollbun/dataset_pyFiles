import pandas as pd
import re
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from functools import partial
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
# Hyperopt
import hyperopt
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from catboost import CatBoostRegressor
# Lightgbm
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error, make_scorer
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=UserWarning)
# display full dataframe
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


TRAIN_PATH = "/kaggle/input/russian-car-plates-prices-prediction/train.csv"
TEST_PATH = "/kaggle/input/russian-car-plates-prices-prediction/test.csv"
SUBMISSION_PATH = '/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv'

TARGET = 'log_price'

SCORING = "mape"

MAX_EVALUATIONS = 10

NUM_FOLDS = 5

HYPER_SPACE = {
    'objective': 'regression',
    'metric': "mape",
    # 'boosting': 'gbdt',  # Keep as gradient boosting
    'silent': True,
    'verbose': -1,
    # Fewer estimators
    'n_estimators': hp.choice('n_estimators', list(range(200, 15700, 200))),
    'max_depth': hp.randint('max_depth', 5, 25),  # Shallower trees
    # Fewer leaves
    'num_leaves': hp.choice('num_leaves', list(range(200, 8000, 200))),
    # Limit subsampling to speed up training
    'subsample': hp.uniform('subsample', 0.2, 0.9),
    # Less feature sampling
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.9),
    # Slightly higher learning rate for fewer estimators
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.05),
    # Regularization to reduce overfitting
    'min_child_samples': hp.choice('min_child_samples', list(range(5, 80, 10))),
    # Keep lightweight
    'min_child_weight': hp.uniform('min_child_weight', 0.01, 0.2),
    # Limit splits to reduce tree depth
    'min_split_gain': hp.uniform('min_split_gain', 0.0, 0.4),
    'reg_alpha': hp.uniform('reg_alpha', 0.1, 18),  # Increase regularization
    'reg_lambda': hp.uniform('reg_lambda', 0.1, 18),  # Increase regularization
}

DATE_COLUMN = "date"

VALIDATION_DAYS = 0

TEST_DAYS = 2

CATEGORY_COLUMNS = ['unique_plate', 'plate_series', 'region_code', 'first_4_chars','first_char','middle_letters',
 'char_2',
 'char_3',
 'char_4',
 'char_5',
 'char_6',
 'char_7',
 'char_8',
 'char_9']

SCORING_FEATURES = ['unique_plate', 'region_code', 'plate_series',
 'first_char', 'first_4_chars', 'first_number_block',
 'last_number_block', 'middle_letters', 'year', 'month', 'day_of_year',
 'week_of_year', 'quarter', 'day_of_week', 'is_weekend', 'day_sin',
 'day_cos', 'month_sin', 'month_cos', 'week_sin', 'week_cos',
 'day_of_week_sin', 'day_of_week_cos', 'is_month_start', 'is_month_end',
 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end',
 'plate_length', 'num_vowels', 'num_digits',
 'char_2',
 'char_3',
 'char_4',
 'char_5',
 'char_6',
 'char_7',
 'char_8',
 'char_9',
 'log_price_mean_day_rolling_mean_1',
 'log_price_mean_day_rolling_std_1',
 'log_price_std_day_rolling_mean_1',
 'log_price_std_day_rolling_std_1',
 'log_price_min_day_rolling_mean_1',
 'log_price_min_day_rolling_std_1',
 'log_price_max_day_rolling_mean_1',
 'log_price_max_day_rolling_std_1',
 'log_price_skew_day_rolling_mean_1',
 'log_price_skew_day_rolling_std_1',
 'log_price_median_day_rolling_mean_1',
 'log_price_median_day_rolling_std_1',
 'log_price_mean_day_rolling_mean_2',
 'log_price_mean_day_rolling_std_2',
 'log_price_std_day_rolling_mean_2',
 'log_price_std_day_rolling_std_2',
 'log_price_min_day_rolling_mean_2',
 'log_price_min_day_rolling_std_2',
 'log_price_max_day_rolling_mean_2',
 'log_price_max_day_rolling_std_2',
 'log_price_skew_day_rolling_mean_2',
 'log_price_skew_day_rolling_std_2',
 'log_price_median_day_rolling_mean_2',
 'log_price_median_day_rolling_std_2',
 'log_price_mean_day_rolling_mean_3',
 'log_price_mean_day_rolling_std_3',
 'log_price_std_day_rolling_mean_3',
 'log_price_std_day_rolling_std_3',
 'log_price_min_day_rolling_mean_3',
 'log_price_min_day_rolling_std_3',
 'log_price_max_day_rolling_mean_3',
 'log_price_max_day_rolling_std_3',
 'log_price_skew_day_rolling_mean_3',
 'log_price_skew_day_rolling_std_3',
 'log_price_median_day_rolling_mean_3',
 'log_price_median_day_rolling_std_3',
 'log_price_mean_day_rolling_mean_4',
 'log_price_mean_day_rolling_std_4',
 'log_price_std_day_rolling_mean_4',
 'log_price_std_day_rolling_std_4',
 'log_price_min_day_rolling_mean_4',
 'log_price_min_day_rolling_std_4',
 'log_price_max_day_rolling_mean_4',
 'log_price_max_day_rolling_std_4',
 'log_price_skew_day_rolling_mean_4',
 'log_price_skew_day_rolling_std_4',
 'log_price_median_day_rolling_mean_4',
 'log_price_median_day_rolling_std_4',
 'log_price_mean_day_rolling_mean_8',
 'log_price_mean_day_rolling_std_8',
 'log_price_std_day_rolling_mean_8',
 'log_price_std_day_rolling_std_8',
 'log_price_min_day_rolling_mean_8',
 'log_price_min_day_rolling_std_8',
 'log_price_max_day_rolling_mean_8',
 'log_price_max_day_rolling_std_8',
 'log_price_skew_day_rolling_mean_8',
 'log_price_skew_day_rolling_std_8',
 'log_price_median_day_rolling_mean_8',
 'log_price_median_day_rolling_std_8',
 'log_price_mean_day_rolling_mean_12',
 'log_price_mean_day_rolling_std_12',
 'log_price_std_day_rolling_mean_12',
 'log_price_std_day_rolling_std_12',
 'log_price_min_day_rolling_mean_12',
 'log_price_min_day_rolling_std_12',
 'log_price_max_day_rolling_mean_12',
 'log_price_max_day_rolling_std_12',
 'log_price_skew_day_rolling_mean_12',
 'log_price_skew_day_rolling_std_12',
 'log_price_median_day_rolling_mean_12',
 'log_price_median_day_rolling_std_12',
 'log_price_mean_day_rolling_mean_24',
 'log_price_mean_day_rolling_std_24',
 'log_price_std_day_rolling_mean_24',
 'log_price_std_day_rolling_std_24',
 'log_price_min_day_rolling_mean_24',
 'log_price_min_day_rolling_std_24',
 'log_price_max_day_rolling_mean_24',
 'log_price_max_day_rolling_std_24',
 'log_price_skew_day_rolling_mean_24',
 'log_price_skew_day_rolling_std_24',
 'log_price_median_day_rolling_mean_24',
 'log_price_median_day_rolling_std_24',
 'log_price_mean_day_rolling_mean_36',
 'log_price_mean_day_rolling_std_36',
 'log_price_std_day_rolling_mean_36',
 'log_price_std_day_rolling_std_36',
 'log_price_min_day_rolling_mean_36',
 'log_price_min_day_rolling_std_36',
 'log_price_max_day_rolling_mean_36',
 'log_price_max_day_rolling_std_36',
 'log_price_skew_day_rolling_mean_36',
 'log_price_skew_day_rolling_std_36',
 'log_price_median_day_rolling_mean_36',
 'log_price_median_day_rolling_std_36',
 'log_price_mean_region_code_rolling_mean_1_week',
 'log_price_mean_region_code_rolling_max_1_week',
 'log_price_mean_region_code_rolling_std_1_week',
 'log_price_mean_region_code_rolling_mean_2_week',
 'log_price_mean_region_code_rolling_max_2_week',
 'log_price_mean_region_code_rolling_std_2_week',
 'log_price_mean_region_code_rolling_mean_3_week',
 'log_price_mean_region_code_rolling_max_3_week',
 'log_price_mean_region_code_rolling_std_3_week',
 'log_price_mean_region_code_rolling_mean_4_week',
 'log_price_mean_region_code_rolling_max_4_week',
 'log_price_mean_region_code_rolling_std_4_week',
 'log_price_mean_region_code_rolling_mean_8_week',
 'log_price_mean_region_code_rolling_max_8_week',
 'log_price_mean_region_code_rolling_std_8_week',
 'log_price_mean_region_code_rolling_mean_12_week',
 'log_price_mean_region_code_rolling_max_12_week',
 'log_price_mean_region_code_rolling_std_12_week',
 'log_price_mean_region_code_rolling_mean_24_week',
 'log_price_mean_region_code_rolling_max_24_week',
 'log_price_mean_region_code_rolling_std_24_week',
 'log_price_mean_region_code_rolling_mean_36_week',
 'log_price_mean_region_code_rolling_max_36_week',
 'log_price_mean_region_code_rolling_std_36_week',
 'log_price_mean_first_char_rolling_mean_1_week',
 'log_price_mean_first_char_rolling_max_1_week',
 'log_price_mean_first_char_rolling_std_1_week',
 'log_price_mean_first_char_rolling_mean_2_week',
 'log_price_mean_first_char_rolling_max_2_week',
 'log_price_mean_first_char_rolling_std_2_week',
 'log_price_mean_first_char_rolling_mean_3_week',
 'log_price_mean_first_char_rolling_max_3_week',
 'log_price_mean_first_char_rolling_std_3_week',
 'log_price_mean_first_char_rolling_mean_4_week',
 'log_price_mean_first_char_rolling_max_4_week',
 'log_price_mean_first_char_rolling_std_4_week',
 'log_price_mean_first_char_rolling_mean_8_week',
 'log_price_mean_first_char_rolling_max_8_week',
 'log_price_mean_first_char_rolling_std_8_week',
 'log_price_mean_first_char_rolling_mean_12_week',
 'log_price_mean_first_char_rolling_max_12_week',
 'log_price_mean_first_char_rolling_std_12_week',
 'log_price_mean_first_char_rolling_mean_24_week',
 'log_price_mean_first_char_rolling_max_24_week',
 'log_price_mean_first_char_rolling_std_24_week',
 'log_price_mean_first_char_rolling_mean_36_week',
 'log_price_mean_first_char_rolling_max_36_week',
 'log_price_mean_first_char_rolling_std_36_week',
 'log_price_mean_first_number_block_rolling_mean_1_week',
 'log_price_mean_first_number_block_rolling_max_1_week',
 'log_price_mean_first_number_block_rolling_std_1_week',
 'log_price_mean_first_number_block_rolling_mean_2_week',
 'log_price_mean_first_number_block_rolling_max_2_week',
 'log_price_mean_first_number_block_rolling_std_2_week',
 'log_price_mean_first_number_block_rolling_mean_3_week',
 'log_price_mean_first_number_block_rolling_max_3_week',
 'log_price_mean_first_number_block_rolling_std_3_week',
 'log_price_mean_first_number_block_rolling_mean_4_week',
 'log_price_mean_first_number_block_rolling_max_4_week',
 'log_price_mean_first_number_block_rolling_std_4_week',
 'log_price_mean_first_number_block_rolling_mean_8_week',
 'log_price_mean_first_number_block_rolling_max_8_week',
 'log_price_mean_first_number_block_rolling_std_8_week',
 'log_price_mean_first_number_block_rolling_mean_12_week',
 'log_price_mean_first_number_block_rolling_max_12_week',
 'log_price_mean_first_number_block_rolling_std_12_week',
 'log_price_mean_first_number_block_rolling_mean_24_week',
 'log_price_mean_first_number_block_rolling_max_24_week',
 'log_price_mean_first_number_block_rolling_std_24_week',
 'log_price_mean_first_number_block_rolling_mean_36_week',
 'log_price_mean_first_number_block_rolling_max_36_week',
 'log_price_mean_first_number_block_rolling_std_36_week',
 'log_price_mean_middle_letters_rolling_mean_1_week',
 'log_price_mean_middle_letters_rolling_max_1_week',
 'log_price_mean_middle_letters_rolling_std_1_week',
 'log_price_mean_middle_letters_rolling_mean_2_week',
 'log_price_mean_middle_letters_rolling_max_2_week',
 'log_price_mean_middle_letters_rolling_std_2_week',
 'log_price_mean_middle_letters_rolling_mean_3_week',
 'log_price_mean_middle_letters_rolling_max_3_week',
 'log_price_mean_middle_letters_rolling_std_3_week',
 'log_price_mean_middle_letters_rolling_mean_4_week',
 'log_price_mean_middle_letters_rolling_max_4_week',
 'log_price_mean_middle_letters_rolling_std_4_week',
 'log_price_mean_middle_letters_rolling_mean_8_week',
 'log_price_mean_middle_letters_rolling_max_8_week',
 'log_price_mean_middle_letters_rolling_std_8_week',
 'log_price_mean_middle_letters_rolling_mean_12_week',
 'log_price_mean_middle_letters_rolling_max_12_week',
 'log_price_mean_middle_letters_rolling_std_12_week',
 'log_price_mean_middle_letters_rolling_mean_24_week',
 'log_price_mean_middle_letters_rolling_max_24_week',
 'log_price_mean_middle_letters_rolling_std_24_week',
 'log_price_mean_middle_letters_rolling_mean_36_week',
 'log_price_mean_middle_letters_rolling_max_36_week',
 'log_price_mean_middle_letters_rolling_std_36_week'
]



def summary_df(df):
    # Print the shape of the dataset
    print(f"Dataset Shape: {df.shape}")

    # Create a summary dataframe with dtypes and name of columns
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]

    # Calculate the number of missing values
    summary['Missing'] = df.isnull().sum().values

    # Calculate the number of unique values
    summary['Uniques'] = df.nunique().values

    # Initialize columns for summary statistics
    summary['Mean'] = None
    summary['Min'] = None
    summary['25%'] = None
    summary['50%'] = None
    summary['75%'] = None
    summary['Max'] = None
    summary['Std'] = None

    # Compute statistics for numerical columns
    num_cols = df.select_dtypes(include=['number']).columns
    summary.loc[summary['Name'].isin(
        num_cols), 'Mean'] = df[num_cols].mean().values
    summary.loc[summary['Name'].isin(
        num_cols), 'Min'] = df[num_cols].min().values
    summary.loc[summary['Name'].isin(
        num_cols), '25%'] = df[num_cols].quantile(0.25).values
    summary.loc[summary['Name'].isin(
        num_cols), '50%'] = df[num_cols].median().values
    summary.loc[summary['Name'].isin(
        num_cols), '75%'] = df[num_cols].quantile(0.75).values
    summary.loc[summary['Name'].isin(
        num_cols), 'Max'] = df[num_cols].max().values
    summary.loc[summary['Name'].isin(
        num_cols), 'Std'] = df[num_cols].std().values

    return summary


def plot_target_density(df ,log_feat, target = TARGET):
    df_plot = df.copy()
    # Set up the figure
    plt.figure(figsize=(12, 6))
    if log_feat == True:
        df_plot[target] = np.log(df_plot[target] + 1)
    # Plot the distributions for Fraud and NoFraud
    g = sns.kdeplot(df_plot[target], shade=True)

    # Add legend
    g.legend()

    # Set title and labels
    if log_feat == True:
        title_add = "Log "
    else: title_add = ""
    g.set_title(title_add + "Target Distribution", fontsize=20)
    g.set_xlabel(target, fontsize=18)
    g.set_ylabel("Density", fontsize=18)

    # Show plot
    plt.show()


def plot_avg_price(df, freq='W', log_feat = True, target = TARGET):
    """
    freq: Resample frequency
        'D' - Daily
        'W' - Weekly
        'M' - Monthly
        'Q' - Quarterly
        'Y' - Yearly
    """
    df = df.copy()
    if log_feat == True:
        df[target] = np.log(df[target] + 1)
    df['date'] = pd.to_datetime(df['date'])  # ensure datetime
    
    # Group by chosen frequency
    df_resampled = df.set_index('date').resample(freq)[target].mean().reset_index()

    # Plot
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df_resampled, x='date', y=target, marker='o')
    plt.title(f'Average Price ({freq} frequency)')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def split_plate(df):
    df = df.copy()
    df['plate'] = df['plate'].astype(str)
    df['plate_series'] = df["plate"].apply(lambda plate: plate[0]+plate[4:6]).astype(str)
    df['unique_plate'] = df['plate'].str.extract(r'^([A-Z0-9]+[A-Z])')
    df['region_code'] = df['plate'].str.extract(r'(\d+)$')
    df['first_char'] = df['plate'].str[0]
    df['first_4_chars'] = df['plate'].str[:4]
    df['first_number_block'] = df['plate'].str.extract(r'^[A-Z](\d+)').astype(int)
    df['last_number_block'] = df['plate'].str.extract(r'(\d+)[A-Z]*$').astype(int)
    df['middle_letters'] = df['plate'].str.extract(r'\d+([A-Z]+)\d+')
    for i in range(df['plate'].str.len().max()):
        df[f'char_{i+1}'] = df['plate'].apply(lambda x: x[i] if i < len(x) else np.nan)
        
    return df


def date_features(df):
    df["date"] = pd.to_datetime(df['date'])

    # Basic date parts
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.weekday  # Monday=0, Sunday=6
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # 1 if Sat/Sun, else 0

    # Cyclical encoding
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Special day indicators
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)

    return df


def rest_features(df):
    df['plate_length'] = df['unique_plate'].str.len()
    df['num_vowels'] = df['unique_plate'].str.count(r'[AEIOUY]')
    df['num_digits'] = df['unique_plate'].str.count(r'\d')

    return df

def aggregate_features(df, groupby_columns, agg_features, agg_funcs):
    """
    Dynamically aggregates multiple numerical features using specified aggregation functions.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - groupby_columns (list): The columns to group by.
    - agg_features (list): The numeric columns to aggregate.
    - agg_funcs (list): The aggregation functions to apply (e.g., ["mean", "std", "min", "max"]).

    Returns:
    - pd.DataFrame: The aggregated dataframe.
    """
    # Create the aggregation dictionary dynamically
    agg_dict = {feature: agg_funcs for feature in agg_features}
    
    # Perform aggregation
    df_agg = df.groupby(groupby_columns).agg(agg_dict).reset_index()
    
    # Flatten column names (from multi-index to single level)
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]
    # Ensure all aggregated feature names end with "_day"
    df_agg.columns = [col + "_day" if col not in groupby_columns else col for col in df_agg.columns]

    # Merge back with original dataframe
    df = df.merge(df_agg, on=groupby_columns, how="left")
    
    return df


def aggregate_features(df, groupby_columns, agg_features, agg_funcs, suffix='_day'):
    """
    Dynamically aggregates multiple numerical features using specified aggregation functions
    and allows dynamic suffix for aggregated features.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - groupby_columns (list): The columns to group by.
    - agg_features (list): The numeric columns to aggregate.
    - agg_funcs (list): The aggregation functions to apply (e.g., ["mean", "std", "min", "max"]).
    - suffix (str): The suffix to add to the aggregated feature names (default is "_day").

    Returns:
    - pd.DataFrame: The aggregated dataframe.
    """
    # Create the aggregation dictionary dynamically
    agg_dict = {feature: agg_funcs for feature in agg_features}
    
    # Perform aggregation
    df_agg = df.groupby(groupby_columns).agg(agg_dict).reset_index()
    
    # Flatten column names (from multi-index to single level)
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]
    
    # Apply dynamic suffix to aggregated columns (except groupby columns)
    df_agg.columns = [col + suffix if col not in groupby_columns else col for col in df_agg.columns]

    # Merge back with the original dataframe
    df = df.merge(df_agg, on=groupby_columns, how="left")
    
    return df


def create_rolling_features(df, year_col='year', week_col='week_of_year', rolling_features=None, windows=[3, 7, 14], agg_funcs=["mean", "std"]):
    """
    Creates rolling window features for specified numerical columns.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - rolling_features (list): List of feature names to create rolling features for (default: all numerical features).
    - windows (list): List of rolling window sizes (e.g., [3, 6, 12] for 3, 6, and 12 time-period rolling windows).
    - agg_funcs (list): List of aggregation functions to apply (e.g., ["mean", "std", "min", "max"]).
    
    Returns:
    - pd.DataFrame: Original dataframe with only new rolling features merged.
    """
    df = df.copy()
    df['date_combo'] = pd.to_datetime(df[year_col].astype(str) + df[week_col].astype(str) + '1', format='%G%V%u')
    # df = df.sort_values(by='date_combo')
    df = df.sort_values(by=['year', 'week_of_year']).reset_index(drop=True)

    # Identify all numerical features if not explicitly provided
    if rolling_features is None:
        rolling_features = [col for col in df.columns if col.endswith("_day")]
    
    # Create a dataframe for rolling calculations
    df_roll = df[['date_combo'] + rolling_features].copy().drop_duplicates()

    # Generate rolling features
    rolling_columns = ['date_combo']  # Keep grouping column for merging
    for window in windows:
        for feature in rolling_features:
            for agg_func in agg_funcs:
                roll_col = f"{feature}_rolling_{agg_func}_{window}"
                df_roll[roll_col] = df_roll[feature].transform(lambda x: x.shift(1).rolling(window).agg(agg_func))
                rolling_columns.append(roll_col)

    # Merge rolling features back to the original dataframe without including the original features
    df = df.merge(df_roll[rolling_columns], on='date_combo', how="left")

    return df

def compute_rolling_week_lags(df, groupby_columns, price_col='price', windows=[3, 7, 14], agg_funcs=["mean", "std", "min", "max"], suffix="_week"):
    """
    Computes rolling window features for specified numerical columns over different week lags with various aggregation functions.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe with 'year', 'week_of_year', 'code', and 'price'.
    - groupby_columns (list): The columns to group by (e.g., ['code']).
    - price_col (str): The column for price values (default: 'price').
    - windows (list): List of rolling window sizes in weeks (e.g., [3, 7, 14]).
    - agg_funcs (list): List of aggregation functions to apply (e.g., ["mean", "std", "min", "max"]).
    - suffix (str): The suffix to append to feature names (default is "_week").
    
    Returns:
    - pd.DataFrame: Original dataframe with new rolling week features added.
    """
    df = df.copy()
    
    # Convert year and week_of_year into a datetime object, pointing to the start of each week
    df['date'] = pd.to_datetime(df['year'].astype(str) + df['week_of_year'].astype(str).str.zfill(2) + '0', format='%Y%U%w')
    
    # Ensure correct sorting of the dataframe
    df = df.sort_values(by=groupby_columns + ['date'])
    
    # Create a dataframe for rolling calculations
    df_roll = df[[*groupby_columns, 'date', price_col]].copy()
    
    # Generate rolling features
    rolling_columns = [*groupby_columns, 'date']  # Keep grouping and date column for merging
    for window in windows:
        for agg_func in agg_funcs:
            roll_col = f"{price_col}_rolling_{agg_func}_{window}{suffix}"
            df_roll[roll_col] = (
                df_roll.groupby(groupby_columns)[price_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).agg(agg_func))
            )
            rolling_columns.append(roll_col)
    
    # Merge rolling features back to the original dataframe without including the original price column
    df = df.merge(df_roll[rolling_columns], on=[*groupby_columns, 'date'], how="left")
    
    return df

class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupTimeSeriesSplit
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                           'b', 'b', 'b', 'b', 'b',\
                           'c', 'c', 'c', 'c',\
                           'd', 'd', 'd'])
    >>> gtss = GroupTimeSeriesSplit(n_splits=3)
    >>> for train_idx, test_idx in gtss.split(groups, groups=groups):
    ...     print("TRAIN:", train_idx, "TEST:", test_idx)
    ...     print("TRAIN GROUP:", groups[train_idx],\
                  "TEST GROUP:", groups[test_idx])
    TRAIN: [0, 1, 2, 3, 4, 5] TEST: [6, 7, 8, 9, 10]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a']\
    TEST GROUP: ['b' 'b' 'b' 'b' 'b']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] TEST: [11, 12, 13, 14]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b']\
    TEST GROUP: ['c' 'c' 'c' 'c']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\
    TEST: [15, 16, 17]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c']\
    TEST GROUP: ['d' 'd' 'd']
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_size=None
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        group_test_size = n_groups // n_folds
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)
            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end -
                                          self.max_train_size:train_end]
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)
            yield [int(i) for i in train_array], [int(i) for i in test_array]

def train_val_test_split(
    df: pd.DataFrame, num_val_days: int, num_test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset to train/val/test based on column started_at."""

    max_date = df[DATE_COLUMN].max().ceil("d")
    split_date = max_date - pd.Timedelta(days=num_test_days + num_val_days)

    if not (split_date > df[DATE_COLUMN].min()):
        raise ValueError(
            f"Not enough data to split if {num_val_days=} and {num_test_days=}"
        )

    mask = df[DATE_COLUMN] < split_date
    training_data, val_data = df[mask], df[~mask]

    mask = val_data[DATE_COLUMN] < (
        max_date - pd.Timedelta(days=num_test_days))
    val_data, test_data = val_data[mask], val_data[~mask]

    print(
        f"Training data: {training_data.shape} \n{training_data[DATE_COLUMN].agg([min, max]).to_string()}\n"
        f"Val data: {val_data.shape} \n{val_data[DATE_COLUMN].agg([min, max]).to_string()}\n"
        f"Test data: {test_data.shape} \n{test_data[DATE_COLUMN].agg([min, max]).to_string()}"
    )

    return training_data, val_data, test_data


def group_experiment(df, date_column):
    df = df.copy()
    # sort values first
    df = df.sort_values([date_column])

    search_date_df = pd.DataFrame(df[date_column].drop_duplicates())
    search_date_df = search_date_df.sort_values([date_column]).reset_index()
    # drop column
    search_date_df = search_date_df.drop(['index'], axis=1)
    search_date_df = search_date_df.reset_index()
    search_date_df.rename(columns={search_date_df.columns[0]: "order",
                                   search_date_df.columns[1]: date_column, }, inplace=True)

    print(search_date_df.columns)
    print(search_date_df.head(3))

    search_date_df[date_column] = search_date_df[date_column].astype(int)
    df[date_column] = df[date_column].astype(int)

    # join with data
    # bring in the order of observations
    df = pd.merge(df, search_date_df[[date_column, 'order']].drop_duplicates(),
                  on=[date_column],
                  how='inner')

    return df

def smape(y_true, y_pred):
    y_pred = np.exp(y_pred)  
    y_true = np.exp(y_true) 
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

def lgb_smape(y_pred, dataset):
    y_true = dataset.get_label()
    
    # if your labels are in log-space (as in your smape func), use exp()
    y_pred = np.exp(y_pred)
    y_true = np.exp(y_true)

    smape_val = np.mean(
        2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)
    )
    return 'smape', smape_val, False

smape_scorer = make_scorer(name='smape1', score_func=smape, greater_is_better=False)

 

def time_series_to_minimize(hyperparameters,
                            scoring,
                            time_train,
                            x_train,
                            y_train):

    folds = NUM_FOLDS
    res_vec = np.zeros((folds, 1))
    prv = np.zeros((x_train.shape[0], folds))
    for (ii, (id0, id1)) in enumerate(GroupTimeSeriesSplit(n_splits=folds).split(x_train, groups=pd.DataFrame(time_train)['order'])):
        # print("fold: ", ii)
        x0, x1 = x_train.iloc[id0], x_train.iloc[id1]
        y0, y1 = y_train[id0], y_train[id1]

        fit_params = {
            # "early_stopping_rounds":50,
            "eval_metric": scoring,
            "eval_set": [(x0, y0), (x1, y1)],
            'eval_names': ['train', 'valid'],
        }

        model = lgb.LGBMRegressor(**hyperparameters)

        model.fit(x0, y0, **fit_params
                  )

        val_preds = model.predict(x1)
        # validation score
        score_rmse = np.sqrt(mean_squared_error( y1, val_preds))
        score_mae = mean_absolute_error(y1, val_preds)
        score_mape = mean_absolute_percentage_error(y1, val_preds)
        smape_score = smape(y1, val_preds)

        res_vec[ii] = smape_score

        del model, x0, x1, y0, y1
    return res_vec.mean()

def cv_tuning(hyperparameters,
              scoring,
              x_train,
              y_train):

    folds = NUM_FOLDS
    res_vec = np.zeros((folds, 1))
    prv = np.zeros((x_train.shape[0], folds))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    for (ii, (id0, id1)) in enumerate(cv.split(x_train)):
        print("fold: ", ii)
        x0, x1 = x_train.iloc[id0], x_train.iloc[id1]
        y0, y1 = y_train[id0], y_train[id1]

        fit_params = {
            # "early_stopping_rounds":100,
            "eval_metric": scoring,
            "eval_set": [(x0, y0), (x1, y1)],
            'eval_names': ['train', 'valid'],
        }

        model = lgb.LGBMRegressor(**hyperparameters)

        model.fit(x0, y0, **fit_params)

        val_preds = model.predict(x1)
        # validation score
        score_rmse = np.sqrt(mean_squared_error( y1, val_preds))
        score_mae = mean_absolute_error(y1, val_preds)
        score_mape = mean_absolute_percentage_error(y1, val_preds)
        smape_score = smape(y1, val_preds)
        res_vec[ii] = smape_score
        print("SMAPE: ", smape_score)
        del model, x0, x1, y0, y1
    return res_vec.mean()



def find_hyperparams(x_train_val, y_train_val, scoring=SCORING, hyper_space=HYPER_SPACE, max_evaluations=MAX_EVALUATIONS):
    optimization = fmin(fn=partial(cv_tuning,
                                   scoring=scoring,
                                   x_train=x_train_val,
                                   y_train=y_train_val
                                   ),
                        space=hyper_space,
                        algo=tpe.suggest,
                        trials=Trials(),
                        max_evals=max_evaluations)

    # fit model
    best_params = space_eval(HYPER_SPACE, optimization)
    print("hyperparams are: ", best_params)
    return best_params

def plot_models_importance(df, model, number_of_features):
    # Get feature importances
    feature_importance = model.feature_importances_
    # Get feature names (if available)
    df = df[SCORING_FEATURES]
    # You need to have a list of feature names in the same order as your data
    # Replace with your feature names if available
    feature_names = SCORING_FEATURES
    # Sort features by importance in ascending order
    sorted_idx = np.argsort(feature_importance)
    sorted_feature_names = [feature_names[i] for i in sorted_idx]
    sorted_feature_importance = feature_importance[sorted_idx]
    # # Create a bar plot of sorted feature importance in ascending order
    # Get the top x features
    top_n = number_of_features
    top_features = sorted_feature_names[-top_n:]
    top_importance = sorted_feature_importance[-top_n:]
    # Create a bar plot of the top x feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_importance)),
             top_importance, tick_label=top_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Top Feature Importance Plot')
    plt.savefig('lgbm_importance.png')
    plt.show()


train_df = pd.read_csv(TRAIN_PATH)
train_df.head()


train_df = split_plate(train_df)
train_df = date_features(train_df)
train_df = rest_features(train_df)
train_df[TARGET] = np.log1p(train_df["price"])
train_df[CATEGORY_COLUMNS] = train_df[CATEGORY_COLUMNS].fillna("missing")
train_df[CATEGORY_COLUMNS] = train_df[CATEGORY_COLUMNS].astype("category")


WINDOWS = [1,2,3,4,8,12,24,36]
AGGREGATE_FEATURES = [TARGET]
AGGREGATION_METHODS = ["mean", "std", "min", "max", "skew", "median"]
train_df['date_day_group'] = train_df['year'].astype(str) + "-" + train_df['week_of_year'].astype(str)
train_df = aggregate_features(train_df, groupby_columns=['year', 'week_of_year'], agg_features=AGGREGATE_FEATURES, agg_funcs=AGGREGATION_METHODS)
train_df = create_rolling_features(train_df, windows = WINDOWS)


train_df = aggregate_features(train_df, groupby_columns=['year', 'week_of_year', "first_char"], agg_features=AGGREGATE_FEATURES, agg_funcs=AGGREGATION_METHODS, suffix = '_first_char')
train_df = aggregate_features(train_df, groupby_columns=['year', 'week_of_year', "region_code"], agg_features=AGGREGATE_FEATURES, agg_funcs=AGGREGATION_METHODS, suffix = '_region_code')
train_df = aggregate_features(train_df, groupby_columns=['year', 'week_of_year', "first_number_block"], agg_features=AGGREGATE_FEATURES, agg_funcs=AGGREGATION_METHODS, suffix = '_first_number_block')
train_df = aggregate_features(train_df, groupby_columns=['year', 'week_of_year', "middle_letters"], agg_features=AGGREGATE_FEATURES, agg_funcs=AGGREGATION_METHODS, suffix = '_middle_letters')

char_columns = [col for col in train_df.columns if col.endswith('_first_char')]
train_df_first_char = train_df[['year', 'week_of_year', 'first_char'] + char_columns].drop_duplicates()

region_code_columns = [col for col in train_df.columns if col.endswith('_region_code')]
train_df_region_code = train_df[['year', 'week_of_year', 'region_code'] + region_code_columns].drop_duplicates()

first_number_block_columns = [col for col in train_df.columns if col.endswith('_first_number_block')]
train_df_first_number_block = train_df[[ 'year', 'week_of_year', 'first_number_block'] + first_number_block_columns].drop_duplicates()

middle_letters_columns = [col for col in train_df.columns if col.endswith('_middle_letters')]
train_df_middle_letters = train_df[['year', 'week_of_year', 'middle_letters'] + middle_letters_columns].drop_duplicates()

train_df_region_code = compute_rolling_week_lags(train_df_region_code, groupby_columns=['region_code'], price_col = [col for col in train_df.columns if col.endswith('_mean_region_code')][0], windows=WINDOWS, agg_funcs=["mean", "max",  "std"])
train_df_first_char = compute_rolling_week_lags(train_df_first_char, groupby_columns=['first_char'], price_col = [col for col in train_df.columns if col.endswith('_mean_first_char')][0], windows=WINDOWS, agg_funcs=["mean", "max", "std"])
train_df_first_number_block = compute_rolling_week_lags(train_df_first_number_block, groupby_columns=['first_number_block'], price_col = [col for col in train_df.columns if col.endswith('_mean_first_number_block')][0], windows=WINDOWS, agg_funcs=["mean", "max", "std"])
train_df_middle_letters = compute_rolling_week_lags(train_df_middle_letters, groupby_columns=['middle_letters'], price_col = [col for col in train_df.columns if col.endswith('_mean_middle_letters')][0], windows=WINDOWS, agg_funcs=["mean", "max", "std"])


rolling_code_columns = [col for col in train_df_region_code.columns if '_region_code_rolling_' in col]
rolling_first_char_columns = [col for col in train_df_first_char.columns if '_first_char_rolling_' in col]
rolling_first_number_block_columns = [col for col in train_df_first_number_block.columns if '_first_number_block_rolling_' in col]
rolling_middle_letters_columns = [col for col in train_df_middle_letters.columns if '_middle_letters_rolling_' in col]


train_df = pd.merge(train_df,
                    train_df_region_code[['year', 'week_of_year', 'region_code'] + rolling_code_columns].drop_duplicates(),
                    on = ['year', 'week_of_year', 'region_code'],
                    how = 'left')

train_df = pd.merge(train_df,
                    train_df_first_char[['year', 'week_of_year', 'first_char'] + rolling_first_char_columns].drop_duplicates(),
                    on = ['year', 'week_of_year', 'first_char'],
                    how = 'left')

train_df = pd.merge(train_df,
                    train_df_first_number_block[['year', 'week_of_year', 'first_number_block'] + rolling_first_number_block_columns].drop_duplicates(),
                    on = ['year', 'week_of_year', 'first_number_block'],
                    how = 'left')

train_df = pd.merge(train_df,
                    train_df_middle_letters[['year', 'week_of_year', 'middle_letters'] + rolling_middle_letters_columns].drop_duplicates(),
                    on = ['year', 'week_of_year', 'middle_letters'],
                    how = 'left')

train_df.head()


# list(train_df.columns)





summary_df(train_df)


plot_target_density(train_df, True)


training_data, val_data, test_data = train_val_test_split(train_df, VALIDATION_DAYS, TEST_DAYS)
training_data['started_at_week'] = training_data['date'].dt.to_period('W').dt.start_time
x_train, y_train = (training_data[SCORING_FEATURES], training_data[TARGET])
params_model = find_hyperparams(x_train, y_train)


plot_avg_price(train_df, freq='W')
plot_avg_price(train_df, freq='M')
plot_avg_price(train_df, freq='Q')
plot_avg_price(train_df, freq='Y')


# params_model = { 'colsample_bytree': 0.6242859400475513, 'learning_rate': 0.02513121164288046, 'max_depth': 17, 'metric': 'rmse', 'min_child_samples': 65, 'min_child_weight': 0.1562004337057197, 'min_split_gain': 0.055729125013148834, 'n_estimators': 10000, 'num_leaves': 7400, 'objective': 'regression', 'reg_alpha': 3.4032151149355405, 'reg_lambda': 4.534044097116439, 'silent': True, 'subsample': 0.2661174556022754, 'verbose': -1}


def train_lgb(df, best_params, time_train):
    
    X = df[SCORING_FEATURES]
    y = df[TARGET]
    model = LGBMRegressor(**best_params)
    models = []
    fold_scores = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)): 
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = LGBMRegressor(**best_params)
        model.fit(X_train, y_train)
        models.append(model)

        y_pred_probs = model.predict(X_val)           
        score = np.sqrt(mean_squared_error(y_val, y_pred_probs))
        smape_score = smape(y_val, y_pred_probs)
        fold_scores.append(score)
        print(f'Fold {fold_idx+1} RMSE: {score:.4f}')
        print(f'Fold {fold_idx+1} SMAPE: {smape_score:.4f}')
        del model, X_train, y_train, X_val, y_val, y_pred_probs
        gc.collect() 
    
    return models

def infer_lgb(data, models):
    return np.mean([model.predict(data) for model in models], axis=0)


train_df['started_at_week'] = train_df['date'].dt.to_period('W').dt.start_time
train_df = group_experiment(train_df, 'started_at_week')
time_train_1 = train_df['order'].copy()
lgbm_models = train_lgb(train_df, params_model, time_train_1)



test_preds = infer_lgb(test_data[SCORING_FEATURES], lgbm_models)
test_score_rmse = np.sqrt(mean_squared_error(test_data[TARGET], test_preds))
test_score_smape = smape(test_data[TARGET], test_preds)
print("rmse test: ", test_score_rmse)
print("smape test: ", test_score_smape)


trained_model = lgb.LGBMRegressor(**params_model)
trained_model.fit(x_train, y_train, eval_metric=lgb_smape)
test_score_rmse = np.sqrt(mean_squared_error(test_data[TARGET], trained_model.predict(test_data[SCORING_FEATURES])))
test_score_smape = smape(test_data[TARGET], trained_model.predict(test_data[SCORING_FEATURES]))
print("rmse test: ", test_score_rmse)
print("smape test: ", test_score_smape)
model = lgb.LGBMRegressor(**params_model)
model.fit(train_df[SCORING_FEATURES], train_df[TARGET])


cat_boost_params = {'iterations': 13792, 'depth': 8, 'learning_rate': 0.017802832240854636, 'l2_leaf_reg': 0.15303993386915848, 'random_strength': 0.39870369064882943, 'bagging_temperature': 0.03688739222255916, 'border_count': 245}
cat_model = CatBoostRegressor(**cat_boost_params, cat_features=CATEGORY_COLUMNS)
cat_model.fit(train_df[SCORING_FEATURES], train_df[TARGET])


test_score_rmse = np.sqrt(mean_squared_error(test_data[TARGET], cat_model.predict(test_data[SCORING_FEATURES])))
test_score_smape = smape(test_data[TARGET], cat_model.predict(test_data[SCORING_FEATURES]))
print("rmse test: ", test_score_rmse)
print("smape test: ", test_score_smape)


xgboost_best_params = {'max_depth': 8, 'learning_rate': 0.015729336196446484, 'min_child_weight': 13, 'subsample': 0.8551353235666302, 'colsample_bytree': 0.982989013184657, 'n_estimators': 1012, 'reg_alpha': 0.3865501186148324, 'reg_lambda': 0.6594097307698463}
xgb_boost = XGBRegressor(enable_categorical=True,  tree_method='gpu_hist', **xgboost_best_params)
xgb_boost.fit(train_df[SCORING_FEATURES], train_df[TARGET], 
              eval_set=[(train_df[SCORING_FEATURES], train_df[TARGET])],
              early_stopping_rounds=200,
              eval_metric="mape", verbose=100)
test_score_rmse = np.sqrt(mean_squared_error(test_data[TARGET], xgb_boost.predict(test_data[SCORING_FEATURES])))
test_score_smape = smape(test_data[TARGET], xgb_boost.predict(test_data[SCORING_FEATURES]))

print("rmse test: ", test_score_rmse)
print("smape test: ", test_score_smape)


plot_models_importance(test_data, trained_model, 25)


test_df = pd.read_csv(TEST_PATH, index_col=0)
test_df = split_plate(test_df)
test_df = date_features(test_df)
test_df = rest_features(test_df)
log_price_columns = [col for col in train_df.columns if col.startswith('log_price_')]
# Combine all the columns you want to exclude
excluded_columns = set(char_columns + first_number_block_columns + middle_letters_columns + region_code_columns + rolling_code_columns + rolling_first_char_columns + rolling_first_number_block_columns + rolling_middle_letters_columns)
# Subtract to get only the desired ones
filtered_log_price_columns = [col for col in log_price_columns if col not in excluded_columns]

test_df = pd.merge(test_df,
                   train_df[['year', 'week_of_year'] + filtered_log_price_columns].drop_duplicates(),
                    on = ['year', 'week_of_year'],
                    how = 'left')

test_df = pd.merge(test_df,
                    train_df_region_code[['year', 'week_of_year', 'region_code'] + rolling_code_columns].drop_duplicates(),
                    on = ['year', 'week_of_year', 'region_code'],
                    how = 'left')


test_df = pd.merge(test_df,
                    train_df_first_char[['year', 'week_of_year', 'first_char'] + rolling_first_char_columns].drop_duplicates(),
                    on = ['year', 'week_of_year', 'first_char'],
                    how = 'left')

test_df = pd.merge(test_df,
                    train_df_first_number_block[['year', 'week_of_year', 'first_number_block'] + rolling_first_number_block_columns].drop_duplicates(),
                    on = ['year', 'week_of_year', 'first_number_block'],
                    how = 'left')

test_df = pd.merge(test_df,
                    train_df_middle_letters[['year', 'week_of_year', 'middle_letters'] + rolling_middle_letters_columns].drop_duplicates(),
                    on = ['year', 'week_of_year', 'middle_letters'],
                    how = 'left')

test_df[CATEGORY_COLUMNS] = test_df[CATEGORY_COLUMNS].fillna("missing")
test_df[CATEGORY_COLUMNS] = test_df[CATEGORY_COLUMNS].astype("category")

test_df.head()


submission = pd.read_csv(SUBMISSION_PATH)
lgbm_preds =  np.expm1(infer_lgb(test_df[SCORING_FEATURES], lgbm_models))
cat_preds =  np.expm1(cat_model.predict(test_df[SCORING_FEATURES]))
xgb_preds =  np.expm1(xgb_boost.predict(test_df[SCORING_FEATURES]))
submission['price'] = (cat_preds + lgbm_preds) / 2
test_df['price_prediction'] =  (cat_preds + lgbm_preds) / 2
submission.to_csv('submission.csv', index=False)
submission.head()


submission_cat = pd.read_csv(SUBMISSION_PATH)
submission_cat['price'] = cat_preds
submission_cat.to_csv('submission2.csv', index=False)


submission_xgb = pd.read_csv(SUBMISSION_PATH)
submission_xgb['price'] = xgb_preds
submission_xgb.to_csv('submission3.csv', index=False)


submission_lgb = pd.read_csv(SUBMISSION_PATH)
submission_lgb['price'] = lgbm_preds
submission_lgb.to_csv('submission4.csv', index=False)


plot_target_density(submission, True, 'price')


plot_avg_price(test_df, freq='W', target = 'price_prediction')
plot_avg_price(test_df, freq='M', target = 'price_prediction')
plot_avg_price(test_df, freq='Q', target = 'price_prediction')
plot_avg_price(test_df, freq='Y', target = 'price_prediction')










