import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler
from typing import Tuple, List
from datetime import datetime, timedelta
import warnings
import shap
import gc
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
import functools
from functools import partial
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score, roc_auc_score, roc_curve, precision_recall_curve
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool, EFeaturesSelectionAlgorithm, EShapCalcType
from plotly.offline import init_notebook_mode, iplot
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
init_notebook_mode()
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=UserWarning)
# display full dataframe
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


TRAIN_PATH = "/kaggle/input/playground-series-s5e3/train.csv"
TRAIN_PATH_EXTRA = "/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e3/test.csv"

TARGET = "rainfall"
MAIN_FEATURES = ['pressure',
                 'maxtemp',
                 'temparature',
                 'mintemp',
                 'dewpoint',
                 'humidity',
                 'cloud',
                 'sunshine',
                 'winddirection',
                 'windspeed']


# VALIDATION_DAYS = 0
# TEST_DAYS = 30
LAGS = [1, 3, 5, 7, 14, 21, 30]
WINDOWS = [3, 6, 9, 12, 15]
AGGREGATE_FEATURES = ["pressure", "temparature", "humidity", "cloud", "sunshine", "dewpoint", "winddirection", "windspeed", "maxtemp", "mintemp"]
AGGREGATION_METHODS = ["mean", "std", "min", "max", "skew", "median"]
SCORING = 'roc_auc'
NUM_OF_EVALS = 50
NUM_MODELS = 5
LGBM_HYPER_SPACE = {'objective': 'binary',
                    'metric': 'auc',
                    'boosting': 'gbdt',
                    'silent': True,
                    # "device": "gpu", 
                    'verbose': -1,
                    'scale_pos_weight': hp.uniform('scale_pos_weight', 0.1, 0.3),
                    'extra_trees': hp.choice('extra_trees', [True, False]),
                    'n_estimators': hp.choice('n_estimators', list(range(50, 6000, 200))),
                    'max_depth': hp.choice('max_depth', list(range(6, 22, 1))),
                    'num_leaves': hp.choice('num_leaves', list(range(50, 6000, 150))),
                    'subsample': hp.choice('subsample', [.5, .6, .7, .8, .9, 1]),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                    'reg_alpha': hp.uniform('reg_alpha', 0, 20),
                    'reg_lambda': hp.uniform('reg_lambda', 0, 20),
                    'max_bin': hp.choice('max_bin', list(range(64, 512, 32))),
                    'min_child_samples': hp.choice('min_child_samples', list(range(5, 30, 5))),
                    'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),
                    'bagging_fraction': hp.uniform('bagging_fraction', 0.2, 1),
                    'feature_fraction': hp.uniform('feature_fraction', 0.2, 1),
                    'min_split_gain': hp.uniform('min_split_gain', 0, 0.9),
                    }


def import_data():
    train=pd.read_csv(TRAIN_PATH)
    train['is_synthetic'] = 1
    train_extra=pd.read_csv(TRAIN_PATH_EXTRA)
    train_extra['rainfall'] = train_extra['rainfall'].map({'no': 0, 'yes': 1})
    train_extra = train_extra.reset_index(names='id')
    train_extra.columns = train_extra.columns.str.strip()
    train_extra['is_synthetic'] = 0
    train_extra = train_extra[list(train.columns)]
    test=pd.read_csv(TEST_PATH)
    print("Train, Test Shapes are: ", train.shape, train_extra.shape, test.shape)

    return train, train_extra, test

'''Create function for providing summary statistics in a table'''

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


def plot_count_distribution(df, feature_name):
    total = len(df)
    plt.figure(figsize=(12, 7))

    g = sns.countplot(x=feature_name, data=df, color='blue')
    g.set_title(f"Distribution \nTotal Observations: {total}",
                fontsize=20)
    g.set_xlabel(feature_name, fontsize=18)
    g.set_ylabel('Count', fontsize=18)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
               height + 4,
               '{:1.2f}%'.format(height/total*100),
               ha="center", fontsize=15)
    g.set_ylim(0, total * 1.00)

    plt.show()


def aggcol(df: pd.DataFrame, group_by, col: str, agg: str) -> pd.Series:
    return df.groupby(group_by)[col].transform(agg)

def plot_distribution(df, column):
    plt.figure(figsize=(10, 5))
    
    # Histogram with KDE (Kernel Density Estimation)
    sns.histplot(df[column], bins=30, kde=True, color='b', edgecolor='black', alpha=0.7)
    
    # Labels & Title
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")
    
    # Show Plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_target_time(df, target_column="rainfall", day_column="day", aggregate="mean", time_freq="D"):
    """
    Plots the aggregated target variable over time.

    Parameters:
    - df: DataFrame containing the data.
    - target_column: The column to aggregate (e.g., "rainfall").
    - day_column: The column representing the day of the year (1-365).
    - aggregate: Aggregation function ("mean", "sum", "median", etc.).
    - time_freq: Time frequency ("D" = daily, "W" = weekly, "M" = monthly).
    """
    df = df.copy()
    
    # Convert day number to a date (assuming year 2024 for consistency)
    df["date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(df[day_column] - 1, unit="D")

    # Resample based on chosen frequency
    df_resampled = df.set_index("date").resample(time_freq).agg({target_column: aggregate}).reset_index()
    df_resampled.rename(columns={target_column: "agg_target"}, inplace=True)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df_resampled["date"], y=df_resampled["agg_target"], marker='o', linestyle='-', color='b')

    # Labels and formatting
    plt.xlabel("Date")
    plt.ylabel(f"{aggregate.capitalize()} {target_column.capitalize()}")
    plt.title(f"{time_freq}-level {target_column.capitalize()} ({aggregate.capitalize()})")

    # Dynamically setting y-axis ticks
    y_min, y_max = df_resampled["agg_target"].min(), df_resampled["agg_target"].max()
    y_ticks = np.linspace(y_min, y_max, num=6)
    plt.yticks(y_ticks, labels=[f"{ytick:.2f}" for ytick in y_ticks])

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Show plot
    plt.show()

def plot_correlation_matrix(df):
    # Compute correlation matrix
    corr_matrix = df.corr()

    # Create a mask to hide the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))

    # Plot heatmap with the mask
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    # Title
    plt.title("Feature Correlation Heatmap")
    plt.show()


def evaluate_feature_density_by_target(df, feature, target, log_feat):
    df_plot = df.copy()
    # Set up the figure
    plt.figure(figsize=(12, 6))
    if log_feat == True:
        df_plot[feature] = np.log(df_plot[feature] + 1)
    # Plot the distributions for Fraud and NoFraud
    g = sns.kdeplot(df_plot[df_plot[target] == 1]
                    [feature], label='1', shade=True)
    g = sns.kdeplot(df_plot[df_plot[target] == 0]
                    [feature], label='0', shade=True)

    # Add legend
    g.legend()

    # Set title and labels
    g.set_title("Feature Distribution by Target", fontsize=20)
    g.set_xlabel(feature, fontsize=18)
    g.set_ylabel("Density", fontsize=18)

    # Show plot
    plt.show()

def remove_outliers_iqr(data, features, multiplier=2.5):
    """
    Removes outliers from numeric features in a dataset using the IQR method.

    Parameters:
        data (pd.DataFrame): The dataset to clean.
        multiplier (float): The IQR multiplier to identify outliers. Default is 3.

    Returns:
        pd.DataFrame: The dataset with outliers removed from numeric columns, retaining all other columns.
    """
    cleaned_data = data.copy()
    
    for feature in features:
        Q1 = cleaned_data[feature].quantile(0.25)
        Q3 = cleaned_data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (multiplier * IQR)
        upper_bound = Q3 + (multiplier * IQR)
        
        # Remove rows with outliers in the current feature
        cleaned_data = cleaned_data[
            (cleaned_data[feature] >= lower_bound) & (cleaned_data[feature] <= upper_bound)
        ]
    
    return cleaned_data

def date_features(df, day_column='day'):
    df["date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(df[day_column] - 1, unit="D")

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




def create_lag_features(df, groupby_column="day", lag_features=None, lags=[1, 2, 3]):
    """
    Creates lag features for all '_day' aggregated features and merges only the new lag features,
    ensuring no duplication in the dataframe and excluding the current values as lag features.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - groupby_column (str): The column to use for lagging (e.g., 'day').
    - lag_features (list): List of feature names to create lags for (default: all '_day' features).
    - lags (list): List of lag periods (e.g., [1, 2, 3] for 1-day, 2-day, 3-day lags).
    
    Returns:
    - pd.DataFrame: Original dataframe with only new lag features merged.
    """
    # Sort dataframe by the 'day' column to ensure correct lagging order
    df = df.sort_values(by=groupby_column)
    
    # Identify all '_day' features if not explicitly provided
    if lag_features is None:
        lag_features = [col for col in df.columns if col.endswith("_day")]
    
    # Select only relevant columns for lagging
    df_lag = df[[groupby_column] + lag_features].copy().drop_duplicates()
    
    # Generate lagged features
    lagged_columns = [groupby_column]  # Keep day column for merging
    for lag in lags:
        for feature in lag_features:
            lag_col = f"{feature}_lag{lag}"
            df_lag[lag_col] = df_lag[feature].shift(lag)  # Create lag
            lagged_columns.append(lag_col)

    # Merge lagged features back to the original dataframe without including the original features
    df = df.merge(df_lag[lagged_columns], on=groupby_column, how="left")
    
    return df


def create_rolling_features(df, groupby_column, rolling_features=None, windows=[3, 7, 14], agg_funcs=["mean", "std"]):
    """
    Creates rolling window features for specified numerical columns.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - groupby_column (str): The column used to group data before applying rolling calculations.
    - rolling_features (list): List of feature names to create rolling features for (default: all numerical features).
    - windows (list): List of rolling window sizes (e.g., [3, 6, 12] for 3, 6, and 12 time-period rolling windows).
    - agg_funcs (list): List of aggregation functions to apply (e.g., ["mean", "std", "min", "max"]).
    
    Returns:
    - pd.DataFrame: Original dataframe with only new rolling features merged.
    """
    df = df.sort_values(by=groupby_column)
    # Identify all numerical features if not explicitly provided
    if rolling_features is None:
        rolling_features = [col for col in df.columns if col.endswith("_day")]
    
    # Create a dataframe for rolling calculations
    df_roll = df[[groupby_column] + rolling_features].copy().drop_duplicates()

    # Generate rolling features
    rolling_columns = [groupby_column]  # Keep grouping column for merging
    for window in windows:
        for feature in rolling_features:
            for agg_func in agg_funcs:
                roll_col = f"{feature}_rolling_{agg_func}_{window}"
                df_roll[roll_col] = df_roll[feature].transform(lambda x: x.shift(1).rolling(window).agg(agg_func))
                rolling_columns.append(roll_col)

    # Merge rolling features back to the original dataframe without including the original features
    df = df.merge(df_roll[rolling_columns], on=groupby_column, how="left")

    return df

def create_interaction_features(df):
    df['temperature_to_maxtemp_ratio'] = df['temparature'] / df['maxtemp']
    df['temperature_to_mintemp_ratio'] = df['temparature'] / df['mintemp']
    df['temperature_to_dewpoint_ratio'] = df['temparature'] / df['dewpoint']
    df['mintemp_to_maxtemp_ratio'] = df['mintemp'] / df['maxtemp']
    df['sunshine_x_humidity'] = df['sunshine'] * df['humidity']
    df['sunshine_x_dewpoint'] = df['sunshine'] * df['dewpoint']
    df['sunshine_x_cloud'] = df['sunshine'] * df['cloud']
    df['sunshine_x_windspeed'] = df['sunshine'] * df['windspeed']
    df['sunshine_x_winddirection'] = df['sunshine'] * df['winddirection']
    df['sunshine_x_temparature'] = df['sunshine'] * df['temparature']
    df['pressure_to_temperature_ratio'] = df['pressure'] / df['temparature']
    df['pressure_to_windspeed_ratio'] = df['pressure'] / df['windspeed']
    df['pressure_to_dewpoint_ratio'] = df['pressure'] / df['dewpoint']
    df['humidity_x_dewpoint'] = df['humidity'] * df['dewpoint']
    df['humidity_x_cloud'] = df['humidity'] * df['cloud']
    df['humidity_x_temparature'] = df['humidity'] * df['temparature']
    df['humidity_x_winddirection'] = df['humidity'] * df['winddirection']
    df['humidity_x_windspeed'] = df['humidity'] * df['windspeed']
    df["cloud_x_windspeed"] = df["cloud"] * df["windspeed"]
    df["cloud_x_winddirection"] = df["cloud"] * df["winddirection"]
    df["cloud_x_temparature"] = df["cloud"] * df["temparature"]
    df["cloud_x_maxtemp"] = df["cloud"] * df["maxtemp"]
    df["cloud_x_mintemp"] = df["cloud"] * df["mintemp"]
    df["cloud_x_dewpoint"] = df["cloud"] * df["dewpoint"]
    df['wind_x'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
    df['wind_y'] = df['windspeed'] * np.sin(np.radians(df['winddirection']))

    return df

def create_ranks(df):
    df['pressure_max_rank'] = df['pressure']/df['pressure_max_day']
    df['maxtemp_max_rank'] = df['maxtemp']/df['maxtemp_max_day']
    df['temparature_max_rank'] = df['temparature']/df['temparature_max_day']
    df['mintemp_max_rank'] = df['mintemp']/df['mintemp_max_day']
    df['dewpoint_max_rank'] = df['dewpoint']/df['dewpoint_max_day']
    df['humidity_max_rank'] = df['humidity']/df['humidity_max_day']
    df['cloud_max_rank'] = df['cloud']/df['cloud_max_day']
    df['windspeed_max_rank'] = df['windspeed']/df['windspeed_max_day']
    df['sunshine_max_rank'] = np.where(df['sunshine_max_day'] == 0, 0, df['sunshine']/df['sunshine_max_day'])
    #
    df['pressure_min_rank'] = df['pressure_min_day']/df['pressure']
    df['maxtemp_min_rank'] = df['maxtemp_min_day']/df['maxtemp']
    df['temparature_min_rank'] = df['temparature_min_day']/df['temparature']
    df['mintemp_min_rank'] = df['mintemp_min_day']/df['mintemp']
    df['dewpoint_min_rank'] = df['dewpoint_min_day']/df['dewpoint']
    df['humidity_min_rank'] = df['humidity_min_day']/df['humidity']
    df['cloud_min_rank'] = df['cloud_min_day']/df['cloud']
    df['windspeed_min_rank'] = df['windspeed_min_day']/df['windspeed']
    df['sunshine_min_rank'] = np.where(df['sunshine_min_day'] == 0, 0, df['sunshine_min_day']/df['sunshine'])

    return df




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


def find_min_max_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = df[date_col].astype(str)
    # min
    ts_min = df[date_col].min()
    ts_min = ts_min.replace("-", "")
    print(f'Min date in df is: {ts_min}')
    date_time_min = datetime.strptime(ts_min, '%Y%m%d')
    # max
    ts_max = df[date_col].max()
    ts_max = ts_max.replace("-", "")
    print(f'Max date in df is: {ts_max}')
    date_time_max = datetime.strptime(ts_max, '%Y%m%d')
    # difference between dates in timedelta
    delta = date_time_max - date_time_min
    print(f'Difference is {delta.days} days')
    return delta.days, ts_min, ts_max

def get_dates(total_days: int,
              max_date: str,
              date_format1: str = "%Y%m%d",
              date_format2: str = "%Y-%m-%d") -> list:
    """
    Given total number of days, maximum date and date format, returns a list of dates in reverse chronological order.

    Parameters:
    total_days (int): total number of days
    max_date (str): maximum date in format specified by date_format
    date_format (str): format of the date. Default is "%Y-%m-%d"

    Returns:
    list : list of dates in order

    """
    # convert max date
    date_time_max = datetime.strptime(max_date, date_format1)
    # create search list
    search_date_list = []
    for i in range(total_days, -1, -1):
        date_str = (date_time_max - timedelta(days=i)).strftime(date_format2)
        search_date_list.append(date_str)
    search_date_list = search_date_list[1:]
    return search_date_list
    
def find_validation_test_dates(
        total_days: int,
        test_days: int,
        validation_days: int,
        search_date_list: list
) -> Tuple[str, str]:
    """
    Given total number of days, number of days for testing and validation, and the maximum date,
    returns the validation date.

    Parameters:
        total_days (int): total number of days
        test_days (int): number of days for testing
        validation_days (int): number of days for validation
        search_date_list (list): list of total days in scope

    Returns:
        Tuple of result. Can be either tuple of validation and test date or error
    """
    try:
        # check if test_days + validation_days > total_days
        if test_days + validation_days > total_days:
            raise ValueError(
                f"test_days({test_days}) + validation_days({validation_days}) > total_days({total_days})")

        # generate validation index
        if test_days > 0:
            test_num = len(search_date_list) - test_days
        else:
            test_num = len(search_date_list) - 1
        if validation_days > 0:
            validation_num = test_num - validation_days
        else:
            validation_num = 0
        # bring test date
        validation_date = search_date_list[validation_num]
        # bring test date
        test_date = search_date_list[test_num]
        return validation_date, test_date
    except Exception as e:
        # logging.error(f"Error occurred: {e}")
        return "", f"Error occurred: {e}"


# Calculate the inverse of the target frequency
def compute_scale_pos_weight(target):
    """
    Compute the scale_pos_weight based on the inverse frequency of the target.
    Args:
        target (pd.Series or np.ndarray): The target values (binary classification: 0s and 1s).
    Returns:
        float: The scale_pos_weight value.
    """
    class_counts = np.bincount(target)
    if len(class_counts) != 2:
        raise ValueError("Target must be binary (contain only 0s and 1s).")
    negative_count, positive_count = class_counts
    return negative_count / positive_count


def catboost_select_feats_classifier(x_trn, y_trn, x_val, y_val, num_feats):
    """
    Selects top `num_feats` features using CatBoostClassifier and SHAP-based recursive feature selection.

    Parameters:
    - x_trn (pd.DataFrame): Training features
    - y_trn (pd.Series): Training labels
    - x_val (pd.DataFrame): Validation features
    - y_val (pd.Series): Validation labels
    - num_feats (int): Number of features to select

    Returns:
    - list: Selected feature names
    """
    feats_list = list(x_trn.columns)
    train_pool = Pool(x_trn, y_trn, feature_names=feats_list)
    test_pool = Pool(x_val, y_val, feature_names=feats_list)

    model = CatBoostClassifier(iterations=150, random_seed=0)
    
    summary = model.select_features(
        train_pool,
        eval_set=test_pool,
        features_for_select=feats_list,
        num_features_to_select=num_feats,
        steps=10,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=True,
        logging_level='Silent',
        plot=True
    )

    selected = summary['selected_features_names']
    return selected

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



class LightgbmPipeline:

    @classmethod
    def pipeline(cls,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 hyper_space,
                 scoring,
                 number_of_evals):

        # run optimization
        optimization = fmin(fn=partial(cls.to_minimize,
                               scoring=scoring,
                               x_train=x_train,
                               y_train=y_train),
                    space=hyper_space,
                    algo=tpe.suggest,
                    trials=Trials(),
                    max_evals=number_of_evals)

        # fit model
        best_params = space_eval(hyper_space, optimization)
        #
        lgbm_chosen = cls.run_lgb(
            x_train, y_train, x_test, y_test, best_params)

        return lgbm_chosen, best_params

    @classmethod
    def run_lgb(cls, x_train, y_train, x_test, y_test, best_params):
        # Init params for lgbm
        params_lgbm = {'boosting_type': 'gbdt',
                       'max_depth': 8,
                       'objective': 'binary',
                       'num_leaves': 120,
                       'learning_rate': 0.15,
                       'verbose': -1,
                       'metric': {'auc'},
                       'eval_metric': {'auc'}
                       }

        #
        params_lgbm['objective'] = 'binary'
        params_lgbm['metric'] = 'auc'
        params_lgbm['boosting'] = 'gbdt'
        params_lgbm['silent'] = True
        params_lgbm['verbose'] = -1
        params_lgbm['scale_pos_weight'] = best_params['scale_pos_weight']
        params_lgbm['extra_trees'] = best_params['extra_trees']
        params_lgbm['n_estimators'] = best_params['n_estimators']
        params_lgbm['max_depth'] = best_params['max_depth']
        params_lgbm['num_leaves'] = best_params['num_leaves']
        params_lgbm['subsample'] = best_params['subsample']
        params_lgbm['colsample_bytree'] = best_params['colsample_bytree']
        params_lgbm['learning_rate'] = best_params['learning_rate']
        params_lgbm['reg_alpha'] = best_params['reg_alpha']
        params_lgbm['reg_lambda'] = best_params['reg_lambda']
        params_lgbm['max_bin'] = best_params['max_bin']
        params_lgbm['min_child_samples'] = best_params['min_child_samples']
        params_lgbm['min_child_weight'] = best_params['min_child_weight']
        params_lgbm['bagging_fraction'] = best_params['bagging_fraction']
        params_lgbm['feature_fraction'] = best_params['feature_fraction']
        params_lgbm['min_split_gain'] = best_params['min_split_gain']

        # define dataset
        train_data_df = lgb.Dataset(x_train, label=y_train)
        test_data_df = lgb.Dataset(
            x_test, label=y_test, reference=train_data_df)
        evals_results_df = {}
        print("Training the model...")

        lgbm_best = lgb.train(params_lgbm,
                              train_data_df,
                              valid_sets=[train_data_df, test_data_df],
                              valid_names=['train', 'test'],
                              #   evals_result=evals_results_df,
                              num_boost_round=best_params['n_estimators'],
                              callbacks=[lgb.early_stopping(
                                  stopping_rounds=400)],
                              feval=OptimizationFunctions.gini_lgb,
                              #   verbose_eval=True
                              )

        return lgbm_best

    @classmethod
    def to_minimize(cls, hyperparameters, scoring, x_train, y_train):
        # create an instance of the model
        clf = LGBMClassifier(**hyperparameters)
        # evaluate pipeline
        cvrep = RepeatedStratifiedKFold(
            n_splits=10, n_repeats=2, random_state=1)
        # train with cross-validation
        result = cross_val_score(estimator=clf,
                                 X=x_train,
                                 y=y_train,
                                 scoring=scoring,
                                 cv=cvrep,
                                 n_jobs=-1,
                                 error_score='raise')

        return -result.mean()

    @classmethod
    def time_series_to_minimize(cls,
                                hyperparameters,
                                scoring,
                                time_train,
                                x_train,
                                y_train):

        folds = 5
        res_vec = np.zeros((folds, 1))
        prv = np.zeros((x_train.shape[0], folds))
        for (ii, (id0, id1)) in enumerate(GroupTimeSeriesSplit(n_splits=folds).split(x_train, groups=pd.DataFrame(time_train)['order'])):
            x0, x1 = x_train.iloc[id0], x_train.iloc[id1]
            y0, y1 = y_train[id0], y_train[id1]

            model = LGBMClassifier(**hyperparameters)

            model.fit(x0, y0, eval_metric=scoring,
                      eval_set=[(x0, y0), (x1, y1)])

            val_preds = model.predict(x1)
            # validation score
            score = roc_auc_score(y1, val_preds)
            print("validation score: " + str(score))
            res_vec[ii] = score

            del model, x0, x1, y0, y1
        return -res_vec.mean()
    
class PlotMetrics:

    @classmethod
    def plot_confusion_matrix(cls, y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=[],
               yticks=[],
               # ... and label them with the respective list entries
               # xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    @classmethod
    def plot_precision_recall_vs_thresholds(cls, precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(bbox_to_anchor=(1.05, 1),
                   loc='upper left', borderaxespad=0.)
        plt.grid(which="both", axis="both", color='gray',
                 linestyle='-', linewidth=1)
        plt.title('Precision-Recall Thresholds')

    @classmethod
    def plot_predictions(cls, x, y, clf):
        # Predict on test set
        # Predicting proba
        predictions_clf_prob = clf.predict(x)
        # Turn probability to 0-1 binary output
        predictions_clf_01 = np.where(predictions_clf_prob > 0.5, 1, 0)

        print("accuracy is:", accuracy_score(y, predictions_clf_01))
        print("\n")
        print("confusion matrix is:", confusion_matrix(y, predictions_clf_01))
        print("\n")
        print("fbeta is:", fbeta_score(y, predictions_clf_01, beta=2))
        print("\n")
        print(classification_report(y, predictions_clf_01))

        # Generate ROC curve values: fpr, tpr, thresholds
        fpr, tpr, thresholds = roc_curve(y, predictions_clf_prob)

        # Plot ROC curve
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

        # Plot non-normalized confusion matrix
        cls.plot_confusion_matrix(y, predictions_clf_01,
                                  title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        cls.plot_confusion_matrix(y, predictions_clf_01, normalize=True,
                                  title='Normalized confusion matrix')
        plt.show()
        # Plot Precision Recall Curve
        precisions, recalls, thresholds = precision_recall_curve(
            y, predictions_clf_prob)
        cls.plot_precision_recall_vs_thresholds(
            precisions, recalls, thresholds)
        plt.show()

        return predictions_clf_prob, predictions_clf_01

    @classmethod
    def plot_importance(cls, model, X, num=10):
        feature_imp = pd.DataFrame(
            {'Value': model.feature_importance(), 'Feature': X.columns})
        plt.figure(figsize=(20, 10))
        sns.set(font_scale=1)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[0:num])
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importance.png')
        plt.show()
        

class OptimizationFunctions:

    '''Function for lgbm evaluation metric similar to roc auc'''
    @classmethod
    def gini(cls, y, pred):
        g = np.asarray(np.c_[y, pred, np.arange(len(y))], dtype=float)
        g = g[np.lexsort((g[:, 2], -1*g[:, 1]))]
        gs = g[:, 0].cumsum().sum() / g[:, 0].sum()
        gs -= (len(y) + 1) / 2.
        return gs / len(y)

    '''Function for lgbm evaluation metric'''
    @classmethod
    def gini_lgb(cls, y_hat, data):
        y = list(data.get_label())
        score = cls.gini(y, y_hat) / cls.gini(y, y)
        return 'gini', score, True

    '''Function for lgbm evaluation metric'''
    def lgb_fbeta_score(cls, y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
        return 'fbeta', fbeta_score(y_true, y_hat, average='binary', beta=0.5), True

    '''Function for lgbm evaluation metric'''
    def lgb_precision_score(cls, y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat)
        return 'precision', precision_score(y_true, y_hat), True
    
def train_lgb(df, best_params, selected_features):
    
    X = df[selected_features]
    y = df[TARGET]
    skf = StratifiedKFold(n_splits=NUM_MODELS, random_state=42, shuffle = True)
    model = LGBMClassifier(**best_params)
    models = []
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = LGBMClassifier(**best_params)
        model.fit(X_train, y_train)
        models.append(model)

        y_pred_probs = model.predict_proba(X_val)[:, 1]            
        score = roc_auc_score(y_val, y_pred_probs)
        fold_scores.append(score)
        print(f'Fold {fold_idx+1} ROC AUC: {score:.4f}')

        del model, X_train, y_train, X_val, y_val, y_pred_probs
        gc.collect() 
    
    return models

def infer_lgb(data, models):
    return np.mean([model.predict_proba(data)[:, 1] for model in models], axis=0)


train_synthetic, train_extra, test = import_data()
train = pd.concat([train_synthetic, train_extra], axis=0, ignore_index=True)


train.head()


train_extra.head()


test.head()


summary_df(train)


summary_df(test)


plot_count_distribution(train, TARGET)


plot_distribution(train, 'day')


plot_distribution(test, 'day')


plot_target_time(train, target_column="rainfall", time_freq="W")


plot_target_time(train, target_column="rainfall", time_freq="M")


plot_correlation_matrix(train)


for feature in MAIN_FEATURES:
    plot_target_time(train, target_column=feature, time_freq="M")


for feature in MAIN_FEATURES:
    plot_target_time(test, target_column=feature, time_freq="M")


for feature in MAIN_FEATURES:
    evaluate_feature_density_by_target(train, feature, TARGET, log_feat = False)


# Clean using IQR
# print("Original Train length is: ",len(train))
# train = remove_outliers_iqr(train, MAIN_FEATURES)
# print("Train cleaned length is: ",len(train))



# Define the features to aggregate
# Apply the function
train = aggregate_features(train, groupby_columns=["day"], agg_features=AGGREGATE_FEATURES, agg_funcs=AGGREGATION_METHODS)
# Create lag features
train = create_lag_features(train, groupby_column="day", lags = LAGS)
# Date Features
train = date_features(train, day_column='day')
# Rolling Features
train = create_rolling_features(train, groupby_column="day", windows = WINDOWS)
# Rest Features
train = create_interaction_features(train)
# Create Ranks
train = create_ranks(train)


train.tail(3)


x = train.drop(TARGET,axis=1)
y = train[TARGET]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42, stratify=y
)


# feature selection
numeric_features = x_train.select_dtypes(include=['number']).columns.tolist()

selected_feats = catboost_select_feats_classifier(x_train[numeric_features], y_train,
                                       x_test[numeric_features], y_test,
                                       90
                                       )


if 'id' in selected_feats:
    selected_feats.remove('id')
    
print("Selected Features are: ", selected_feats)


x_train = x_train[selected_feats]
x_test = x_test[selected_feats]
# Choose Best Lightgbm model
lgbm_chosen, best_params = LightgbmPipeline.pipeline(x_train,
                                                     y_train,
                                                     x_test,
                                                     y_test,
                                                     LGBM_HYPER_SPACE,
                                                     SCORING,
                                                     NUM_OF_EVALS,
                                                     )



print("Best parameters are: ", best_params)


PlotMetrics.plot_predictions(x_train, y_train, lgbm_chosen)
PlotMetrics.plot_predictions(x_test, y_test, lgbm_chosen)


PlotMetrics.plot_importance(lgbm_chosen, x_test, num=20)


lgbm_rainfall = LGBMClassifier(**best_params)
lgbm_rainfall.fit(x[selected_feats], y)


cat_rainfall = CatBoostClassifier(
    bagging_temperature=0.7,
    depth=12,
    iterations=100,
    l2_leaf_reg=8,
    learning_rate=0.03,
    random_strength=4,  
    eval_metric="AUC",
    verbose=0,
    random_seed=42,
    auto_class_weights="Balanced"
)
cat_rainfall.fit(x[selected_feats], y)


xgb_rainfall = XGBClassifier(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=1.0,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=best_params["scale_pos_weight"]
)

xgb_rainfall.fit(x[selected_feats], y)


voting = VotingClassifier(
    estimators=[('lgbm', lgbm_rainfall), ('cat', cat_rainfall), ('xgb', xgb_rainfall)],
    voting='soft'
)

voting.fit(x_train, y_train)


lgbm_models = train_lgb(train, best_params, selected_feats)


# Apply the function
test['is_synthetic'] = 1
test = aggregate_features(test, groupby_columns=["day"], agg_features=AGGREGATE_FEATURES, agg_funcs=AGGREGATION_METHODS)
# Create lag features
test = create_lag_features(test, groupby_column="day", lags=LAGS)
# Date Features
test = date_features(test, day_column='day')
# Rolling Features
test = create_rolling_features(test, groupby_column="day", windows=WINDOWS)
# Rest Features
test = create_interaction_features(test)
# Create Ranks
test = create_ranks(test)
# Sort Values
test = test.sort_values(by="id", ascending=True)


test.head(5)


test_vote_preds = voting.predict_proba(test[selected_feats])[:, 1]
test_preds = infer_lgb(test[selected_feats], lgbm_models)
submission = pd.DataFrame({"id": test['id'], "rainfall": test_preds})
submission.to_csv("submission.csv", index=False)
submission.head()


# Plot distribution
plt.figure(figsize=(10, 6))
sns.kdeplot(test_preds, label="Averaging LGBM", shade=True, color="blue")
sns.kdeplot(test_vote_preds, label="Voting Mechanism", shade=True, color="red")

plt.title("Distribution of Probabilities (Prob1 vs Prob2)")
plt.xlabel("Probability Value")
plt.ylabel("Density")
plt.legend()
plt.show()

