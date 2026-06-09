%%time
import os
import gc
import random
import pandas as pd
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from math import log, sqrt
from scipy.stats import (ttest_1samp, pearsonr, spearmanr, f_oneway, shapiro, kstest, norm, chi2)
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import kaggle_evaluation.jane_street_inference_server
import warnings

warnings.filterwarnings('ignore')


feature_cols = [f"feature_{idx:02d}" for idx in range(79)]+ [f"responder_{idx}_lag_1" for idx in range(9)]
target_col = "responder_6"
selected_features = ["symbol_id", "time_id"] + feature_cols+[target_col]


responders = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/responders.csv')
features = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/features.csv')
sample_submission = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/sample_submission.csv')

train = (
    pl.read_parquet('/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=9/part-0.parquet')
)

train = train.to_pandas()


print("Shape of the dataset:", train.shape)
print(train.info())
#print(train.describe())


%%time
train_data = pl.scan_parquet(f"/kaggle/input/js24-dataset-stats-with-lags/training.parquet/").collect().to_pandas()
val_data = pl.scan_parquet(f"/kaggle/input/js24-dataset-stats-with-lags/validation.parquet/").collect().to_pandas()


def reduce_memory_usage(df):
    """ 
    iterate through all the columns of a dataframe and 
    modify the data type to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**3
    print(('Memory usage of dataframe is {:.2f}' 
                     'GB').format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max <\
                  np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max <\
                   np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max <\
                   np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max <\
                   np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max <\
                   np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max <\
                   np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**3
    print(('Memory usage after optimization is: {:.2f}' 
                              'GB').format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) 
                                             / start_mem))
    
    return df

def percentage_missing_values(df):
    missing_values_count = df.isnull().sum()
    total_cells = np.product(df.shape)
    total_missing = missing_values_count.sum()
    print ("Percentage of Missing Data = ",(total_missing/total_cells) * 100,"%")


%%time
train_data = reduce_memory_usage(train_data)


features_has_nan=[]
features_not_nan=[]
for col in train_data.columns:
    if train_data[col].isna().sum()>0:
        features_has_nan+=[col]
        print(f'{col}: {train_data[col].isna().sum()}')
    else:
        features_not_nan+=[col]


fig, ax = plt.subplots(figsize=(15, 5))
balance= pd.Series(train_data[target_col]).cumsum()
ax.set_xlabel("Trade", fontsize=18)
ax.set_ylabel("Cumulative resp", fontsize=18);
balance.plot(lw=3);
del balance
gc.collect();


fig, ax = plt.subplots(figsize=(15, 5))
balance= pd.Series(val_data[target_col]).cumsum()
ax.set_xlabel("Trade", fontsize=18)
ax.set_ylabel("Cumulative resp", fontsize=18);
balance.plot(lw=3);
del balance
gc.collect();


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gc

def plot_weighted_resps(df, cols=['responder_6'], marker_day=85, start_date=None, use_weight=True):
    """
    Plots the cumulative daily return for weighted or unweighted responses over specified time horizons.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data with 'weight' and specified horizon columns.
        cols (list): List of horizon column names (e.g., 'responder_6', etc.).
        marker_day (int): Day to mark with a vertical line and shaded region.
        start_date (int or None): Start date for the x-axis. If None, defaults to the minimum date in the data.
        use_weight (bool): If True, uses weight in the calculation; otherwise, plots unweighted cumulative return.

    Returns:
        None: Displays the plot.
    """
    if start_date is None:
        start_date = df['date_id'].min()  # Default to the minimum date in the dataset
        marker_day+=start_date

    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Compute and plot cumulative daily returns for each column in `cols`
    for horizon in cols:
        if use_weight:
            weighted_resp_col = f'weight_{horizon}'
            if weighted_resp_col not in df.columns:
                # Compute weighted response if not already in DataFrame
                df[weighted_resp_col] = df['weight'] * df[horizon]
            col_to_plot = weighted_resp_col
        else:
            col_to_plot = horizon
        
        # Group by date, compute mean, and calculate cumulative product
        cumulative_return = pd.Series(1 + df.groupby('date_id')[col_to_plot].mean()).cumprod()
        cumulative_return.index -= start_date  # Adjust the x-axis to start from `start_date`
        label = f"{horizon} {'x weight' if use_weight else ''}"
        cumulative_return.plot(lw=3, label=label, ax=ax)
        
        # Clean up memory
        del cumulative_return
        gc.collect()
    
    # Add labels and title
    ax.set_xlabel("Day", fontsize=18)
    ax.set_title(f"Cumulative daily return for {'weighted' if use_weight else 'unweighted'} responses over time horizons", fontsize=18)
    
    # Adjust marker day to align with the new x-axis starting point
    adjusted_marker_day = marker_day - start_date
    ax.axvline(x=adjusted_marker_day, linestyle='--', alpha=0.3, c='red', lw=1, label=f'Marker Day {marker_day}')
    ax.axvspan(0, adjusted_marker_day, color=sns.xkcd_rgb['grey'], alpha=0.1)
    
    # Add legend
    plt.legend(loc="lower left")
    plt.show()


plot_weighted_resps(train_data,cols=[target_col],use_weight=True)


plot_weighted_resps(train_data,cols=[target_col],use_weight=False)


plot_weighted_resps(val_data,marker_day=10,use_weight=True)


plot_weighted_resps(val_data,marker_day=10,use_weight=False)


missing_values = train.isnull().sum()
missing_percentage = (missing_values / train.shape[0]) * 100

missing_data = pd.DataFrame({
    "Feature": train.columns,
    "MissingCount": missing_values,
    "MissingPercentage": missing_percentage
}).sort_values(by="MissingPercentage", ascending=False)

# threshold > 0% 33 features
# threshold > 1% 13 features
# threshold > 5% 4 features
missing_data_filtered = missing_data[missing_data["MissingPercentage"] > 0]

plt.figure(figsize=(14, 8))
sns.barplot(
    x="MissingPercentage",
    y="Feature",
    data=missing_data_filtered,
    palette="vlag"
)
plt.title("Missing Values Analysis", fontsize=16)
plt.xlabel("Percentage of Missing Values", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.show()

print("Number of features with missing values : ", len(missing_data_filtered))
missing_data_filtered


plt.figure(figsize=(8, 5))
sns.histplot(train['responder_6'], bins=100, kde=True,fill=False)
plt.title("Distribution of Responder_6")
plt.xlabel("Responder_6")
plt.ylabel("Frequency")
plt.show()

all_response = train['responder_6'].mean() * 100
print(f"Proportion of all Responder_6: {all_response:.2f}%")

zero_response = (train['responder_6'] == 0).mean() * 100
print(f"Proportion of exact zeros in Responder_6: {zero_response:.2f}%")


daily_mean = train.groupby('date_id')['responder_6'].mean()
plt.figure(figsize=(12, 6))
plt.plot(daily_mean)
plt.title("Date_id Mean of Responder_6 Over Time")
plt.xlabel("Date ID")
plt.ylabel("Mean Responder_6")
plt.grid()
plt.show()


daily_std = train.groupby('date_id')['responder_6'].std()
plt.figure(figsize=(12, 6))
plt.plot(daily_std)
plt.title("Date_id Std of Responder_6 Over Time")
plt.xlabel("Date ID")
plt.ylabel("Mean Responder_6")
plt.grid()
plt.show()



intraday_mean = train.groupby('time_id')['responder_6'].mean()
plt.figure(figsize=(12, 6))
plt.plot(intraday_mean)
plt.title("Time_id Mean of Responder_6")
plt.xlabel("Time ID")
plt.ylabel("Mean Responder_6")
plt.grid()
plt.show()


print("Unique date-id:", train['date_id'].nunique())
print("Total Unique time_id steps:", train['time_id'].nunique())


time_per_day = train.groupby('date_id')['time_id'].nunique()
print("\nTime Steps per Day:")
print(time_per_day.describe()) 


### time_id per date_id
rows_per_day = train['date_id'].value_counts()
print("\nRows per Day:")
print(rows_per_day.describe())

plt.figure(figsize=(10, 6))
sns.histplot(rows_per_day, bins=100, kde=True)
plt.title("Number of Tick per Date_id")
plt.xlabel("Number of Rows")
plt.ylabel("Frequency")
plt.grid()
plt.show()


specific_date = 1530
train_one_day = train[train['date_id'] == specific_date]
print(f"Number of time_id for date_id 1530: {train_one_day.shape[0]}")

print("from ", train_one_day['time_id'].min(), "to", train_one_day['time_id'].max())

plt.figure(figsize=(12, 6))
plt.plot(train_one_day['time_id'], train_one_day['responder_6'])
plt.title(f"Responder_6 Over Time for date_id {specific_date}")
plt.xlabel("Time ID")
plt.ylabel("Responder_6")
plt.grid()
plt.show()



plt.figure(figsize=(8, 5))
sns.histplot(train['weight'], bins=100, kde=True,fill=False)
plt.title("Distribution of Weights")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.show()


train['id'] = train.index.values

plt.figure(figsize=(16, 8))
for symbol_id in train['symbol_id'].unique():
    xx = train[train['symbol_id'] == symbol_id]['id']
    yy = train[train['symbol_id'] == symbol_id]['responder_6']
    plt.plot(xx, yy.cumsum(), label=f'Symbol ID {symbol_id}', linewidth=0.5)

plt.title('Cumulative responder_6 for All Symbol IDs', fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Cumulative Returns", fontsize=12)
plt.grid(color='lightgray', linewidth=0.5)
plt.axhline(0, color='red', linestyle='-', linewidth=0.7)
plt.legend(loc='upper left', fontsize=8)
plt.show()


responders = ['responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5', 'responder_6', 'responder_7', 'responder_8']
plt.figure(figsize=(16, 8))

for responder in responders:
    cumulative_sum = train.groupby('id')[responder].sum().cumsum()
    plt.plot(cumulative_sum, label=f'{responder}', linewidth=0.8)

plt.title('Cumulative Sum of All Responders', fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Cumulative Returns", fontsize=12)
plt.grid(color='lightgray', linewidth=0.5)
plt.axhline(0, color='red', linestyle='-', linewidth=0.7)
plt.legend(loc='upper left', fontsize=8)
plt.show()



responder_data = train['responder_6'].dropna()
mean_responder = responder_data.mean()
std_responder = responder_data.std()


ks_statistic, p_value = kstest(responder_data, 'norm', args=(mean_responder, std_responder))
print(f"KS Test Statistic: {ks_statistic:.9f}")
print(f"P-value: {p_value:.9e}")


if p_value > 0.05:
    print("Fail to reject H0: The data appears to follow a normal distribution.")
else:
    print("Reject H0: The data does not follow a normal distribution.")


feature_columns = [col for col in train.columns if 'feature' in col]

print("Total Features:", len(feature_columns))
print("Feature Columns:", feature_columns)


%%time
feature_corr = train[feature_columns].corr(method ='pearson')

plt.figure(figsize=(24, 20))
sns.heatmap(feature_corr, cmap="coolwarm", annot=False, center=0)
plt.title("Feature-to-Feature Correlation Heatmap", fontsize=16)
plt.show()


%%time
full_corr_s = train.corr(method='spearman')

plt.figure(figsize=(24, 20))
sns.heatmap(full_corr_s, cmap="coolwarm", annot=False, center=0)
plt.title("Full Correlation Heatmap", fontsize=16)
plt.show()


correlation_with_responder = train[feature_columns + ['responder_6']].corr()

responder_corr = correlation_with_responder['responder_6'].drop('responder_6').sort_values(ascending=True)

plt.figure(figsize=(12, 6))
sns.barplot(x=responder_corr.index, y=responder_corr.values)
plt.xticks(rotation=90)
plt.title("Correlation of Features with Responder_6", fontsize=16)
plt.xlabel("Features")
plt.ylabel("Correlation Coefficient")
plt.show()


%%time
full_corr_p = train.corr(method ='pearson')

plt.figure(figsize=(24, 20))
sns.heatmap(full_corr_p, cmap="coolwarm", annot=False, center=0)
plt.title("Full Correlation Heatmap", fontsize=16)
plt.show()


combined_corr =  full_corr_s
#combined_corr = (full_corr_p + full_corr_p) / 2

combined_corr_responder = combined_corr['responder_6'].drop('responder_6').abs().sort_values(ascending=False)
top_10_features = combined_corr_responder.head(15)

print("Top 15 Features with highest mean (spearman) correlation:")
print(top_10_features)


combined_corr = full_corr_p 
#combined_corr = (full_corr_p + full_corr_p) / 2

combined_corr_responder = combined_corr['responder_6'].drop('responder_6').abs().sort_values(ascending=False)
top_10_features = combined_corr_responder.head(15)

print("Top 15 Features with highest mean (pearson) correlation:")
print(top_10_features)


combined_corr = (full_corr_p + full_corr_s) / 2
#combined_corr = (full_corr_p + full_corr_p) / 2

combined_corr_responder = combined_corr['responder_6'].drop('responder_6').abs().sort_values(ascending=False)
top_10_features = combined_corr_responder.head(10)
bottom_10_features = combined_corr_responder.tail(10)

print("Top 10 Features with highest mean (spearman + pearson) correlation:")
print(top_10_features)
print("Bottom 10 Features with lowest mean (spearman + pearson) correlation:")
print(bottom_10_features)


threshold = 0.9
high_corr_pairs = []

for i in range(len(combined_corr.columns)):
    for j in range(i):
        if combined_corr.iloc[i, j] > threshold:
            high_corr_pairs.append((combined_corr.columns[i], combined_corr.columns[j], combined_corr.iloc[i, j]))


print(f"Highly Correlated Column Pairs (|corr| > {threshold}:")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.2f}")


combined_corr_responder = combined_corr['responder_6'].drop('responder_6').abs().sort_values(ascending=False)
combined_corr_features = combined_corr_responder[combined_corr_responder.index.str.contains('feature')].abs().sort_values(ascending=False)

top_9_features = combined_corr_features.head(9).index.tolist()
print("Top 9 Features Most Correlated with Responder_6 (features only):", top_9_features)

plt.figure(figsize=(16, 16))
for i, feature in enumerate(top_9_features, 1):
    plt.subplot(3, 3, i)
    plt.hexbin(train['responder_6'], train[feature],gridsize=1000, bins='log', cmap='inferno')
    plt.xlabel(feature, fontsize=10)
    plt.ylabel('Responder 6', fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title(f"Responder_6 vs {feature}", fontsize=12)

plt.tight_layout()
plt.show()


top_10_features.index.tolist()


alpha = 0.05
rho_1 = 0.1
responder_name = "responder_6"

responders_to_test = top_10_features.index.tolist()

results_list = []

for feature_name in responders_to_test:
    df_pair = train[[responder_name, feature_name]].dropna()
    X = df_pair[feature_name].values
    Y = df_pair[responder_name].values
    n = len(X)

    r = np.corrcoef(X, Y)[0,1]
    Z_obs = 0.5 * log((1+r)/(1-r))
    z_alpha_standard = norm.ppf(1 - alpha)
    z_alpha = z_alpha_standard * sqrt(1/(n-3))
    reject_H0 = (Z_obs > z_alpha)
    mean_Z_h1 = 0.5 * log((1+rho_1)/(1-rho_1))
    power = 1 - norm.cdf(z_alpha, loc=mean_Z_h1, scale=sqrt(1/(n-3)))
    results_list.append({
        'Feature': feature_name,
        'N': n,
        'Correlation': r,
        'Z_Obs': Z_obs,
        'Z_Critical': z_alpha,
        'Reject_H0': reject_H0,
        'Power_at_rho1': power
    })

np_results = pd.DataFrame(results_list)
np_results


responder_name = "responder_6"
responders_to_test = top_10_features.index.tolist()

results_list = []

for feature_name in responders_to_test:
    df_pair = train[[responder_name, feature_name]].dropna()
    X = df_pair[feature_name].values
    Y = df_pair[responder_name].values
    n = len(X)
    
    X_with_const = sm.add_constant(X)
    model = sm.OLS(Y, X_with_const).fit()
    
    slope = model.params[1]
    se = model.bse[1]
    wald_stat = (slope ** 2) / (se ** 2)  
    p_val = 1 - chi2.cdf(wald_stat, df=1)  
    
    alpha = 0.05
    reject_H0 = p_val < alpha

    results_list.append({
        'Feature': feature_name,
        'N': n,
        'Slope': slope,
        'SE': se,
        'Wald_Stat': wald_stat,
        'P_Value': p_val,
        'Reject_H0': reject_H0
    })

wald_results = pd.DataFrame(results_list)
wald_results


X = train[top_10_features.index.tolist()]
y = train['responder_6']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)

baseline_mse = mean_squared_error(y_test, lgb_model.predict(X_test))
print(f"Baseline MSE: {baseline_mse}")

perm_importance = permutation_importance(lgb_model, X_test, y_test, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': perm_importance.importances_mean,
    'Std_Dev': perm_importance.importances_std
}).sort_values(by='Importance', ascending=False)
print("Permutation Importance sklearn:")
print(perm_importance_df)

results_list = []
for feature_name in top_10_features.index.tolist():

    X_test_permuted = X_test.copy()
    X_test_permuted[feature_name] = np.random.permutation(X_test_permuted[feature_name])
    
    y_pred_permuted = lgb_model.predict(X_test_permuted)
    permuted_mse = mean_squared_error(y_test, y_pred_permuted)
    delta_mse = permuted_mse - baseline_mse
    
    results_list.append({
        'Feature': feature_name,
        'Baseline_MSE': baseline_mse,
        'Permuted_MSE': permuted_mse,
        'Delta_MSE': delta_mse
    })

custom_perm_df = pd.DataFrame(results_list).sort_values(by='Delta_MSE', ascending=False)
print("\nCustom Permutation Importance:")
print(custom_perm_df)


%%time

responders = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/responders.csv')
features = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/features.csv')
sample_submission = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/sample_submission.csv')

train = (
    pl.read_parquet('/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=9/part-0.parquet')
).to_pandas()

lgb_params = {
    "boosting_type": "gbdt",
    "metric": "rmse",
    "random_state": 9,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "num_leaves": 64,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "min_child_weight": 10,
    "device": "gpu", 
}

important_features = [
    'symbol_id', 'feature_16', 'feature_17', 'feature_36', 
    'responder_3', 'responder_7', 'responder_8'
]

train['feature_16_17_interaction'] = train['feature_16'] * train['feature_17']
train['responder_avg'] = (train['responder_3'] + train['responder_7'] + train['responder_8']) / 3
train['sin_time_id'] = np.sin(2 * np.pi * train['time_id'] / 967)
train['cos_time_id'] = np.cos(2 * np.pi * train['time_id'] / 967)
train['feature_36_squared'] = train['feature_36'] ** 2
train['feature_16_36_interaction'] = train['feature_16'] * train['feature_36']
train['feature_ratio'] = train['feature_16'] / (train['feature_17'] + 1e-9)
train['responder_sum'] = train[['responder_3', 'responder_7', 'responder_8']].sum(axis=1)
train['feature_16_rolling_mean'] = train['feature_16'].rolling(window=5, min_periods=1).mean()
train['feature_16_rolling_std'] = train['feature_16'].rolling(window=5, min_periods=1).std()

final_feature = important_features + [
    'feature_16_17_interaction', 'responder_avg', 'sin_time_id', 'cos_time_id',
    'feature_36_squared', 'feature_16_36_interaction', 'feature_ratio', 'responder_sum',
    'feature_16_rolling_mean', 'feature_16_rolling_std'
]
train = train[['responder_6'] + final_feature]

lgb_model = LGBMRegressor(**lgb_params)
lgb_model.fit(train[final_feature], train['responder_6'])

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    global lags_
    if lags is not None:
        lags_ = lags

    test_df = test.to_pandas()

    for col in important_features + ['time_id']:
        if col not in test_df.columns:
            test_df[col] = 0

    test_df['feature_16_17_interaction'] = test_df['feature_16'] * test_df['feature_17']
    test_df['responder_avg'] = (test_df['responder_3'] + test_df['responder_7'] + test_df['responder_8']) / 3
    test_df['sin_time_id'] = np.sin(2 * np.pi * test_df['time_id'] / 967)
    test_df['cos_time_id'] = np.cos(2 * np.pi * test_df['time_id'] / 967)
    test_df['feature_36_squared'] = test_df['feature_36'] ** 2
    test_df['feature_16_36_interaction'] = test_df['feature_16'] * test_df['feature_36']
    test_df['feature_ratio'] = test_df['feature_16'] / (test_df['feature_17'] + 1e-9)
    test_df['responder_sum'] = test_df[['responder_3', 'responder_7', 'responder_8']].sum(axis=1)
    test_df['feature_16_rolling_mean'] = test_df['feature_16'].rolling(window=5, min_periods=1).mean()
    test_df['feature_16_rolling_std'] = test_df['feature_16'].rolling(window=5, min_periods=1).std()

    test_df = test_df[final_feature].fillna(-1)
    predictions = lgb_model.predict(test_df)
    eps = 1e-10
    predictions = np.clip(predictions, -5+eps, 5-eps)

    predictions_df = test.select(
        'row_id',
        pl.Series('responder_6', predictions),
    )

    assert isinstance(predictions_df, (pl.DataFrame, pd.DataFrame)), "Predictions must be a Polars or Pandas DataFrame."
    print("Predictions are a Polars or Pandas DataFrame.")

    assert list(predictions_df.columns) == ['row_id', 'responder_6'], "Predictions must have columns ['row_id', 'responder_6']."
    print("Predictions have the correct columns ['row_id', 'responder_6'].")

    assert len(predictions_df) == len(test), "Predictions must have the same number of rows as the test data."
    print("Predictions have the same number of rows as the test data.")

    return predictions_df

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )


print('EOF')

