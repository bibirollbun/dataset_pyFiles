# import requiered libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from xgboost import XGBRegressor
import random
from tqdm import tqdm
import dateutil.easter as easter
import holidays
rc_params = {'legend.fontsize': 6,
         'axes.labelsize': 8,
         'axes.titlesize':8,
         'xtick.labelsize':6,
         'ytick.labelsize':6,
         'figure.figsize': [8, 6]}
plt.rcParams.update(rc_params)

random.seed(1368)

# import warnings
# warnings.filterwarnings('ignore')
from warnings import simplefilter
simplefilter("ignore")

pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
print('All Done!')


# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
%config InlineBackend.figure_format = 'retina'


# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

# difine costum fourier feature function
def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features, index=index)

# Compute Fourier features to the 4th order (8 new features) for a
# series y with daily observations and annual seasonality:
#
# fourier_features(df.index, freq=365.25*2, order=4)

# find best values for freq and order of dp and fourier features
def find_best_freq_order(df):
    score_dict = {}
    for freq in list([1, 2, 3, 4, 0.5, 0.25, 1/6]):
        for order in list([1, 2, 3, 4]):
            for dp_order in list([1, 2]):
                dp = DeterministicProcess(
                    index=df.index,
                    constant=True,               # dummy feature for bias (y-intercept)
                    order=dp_order,              # trend (order 1 means linear)
                    seasonal=True,               # weekly seasonality (indicators)
                    drop=True,                   # drop terms to avoid collinearity
                )
                         
                X_1 = dp.in_sample()  # create features for dates in tunnel.index
                X_1 = pd.concat([X_1, fourier_features(df.index, freq=365.25*freq, order=order)], axis=1)
                y = df['num_sold']
                X_train_1, X_valid_1, y_train, y_valid = train_test_split(X_1, y, test_size=365, shuffle=False)
                model = LinearRegression(fit_intercept=False)
                model.fit(X_train_1, y_train)
                y_fit_1 = pd.Series(model.predict(X_train_1), index=X_train_1.index)
                y_pred_1 = pd.Series(model.predict(X_valid_1), index=X_valid_1.index)
                
                X_2 = pd.DataFrame(train_df.index.unique(), index=train_df.index.unique())
                X_2 = get_holidays(X_2, country)
                X_2 = feature_engineer(X_2)
                X_2 = X_2.drop(['date'], axis=1)
                X_train_2, X_valid_2 = train_test_split(X_2, test_size=365, shuffle=False)
              
                # Create residuals (the collection of detrended series) from the training set
                y_resid = y_train - y_fit_1
                
                # Train XGBoost on the residuals
                xgb = XGBRegressor()
                xgb.fit(X_train_2, y_resid)
                
                # Add the predicted residuals onto the predicted trends
                y_fit_total = xgb.predict(X_train_2) + y_fit_1
                y_pred_total = xgb.predict(X_valid_2) + y_pred_1
                                
                score_dict[f'{freq}_{order}_{dp_order}'] = mean_absolute_percentage_error(y_pred_total, y_valid)
    return score_dict

def feature_engineer(df):
    new_df = df.copy()
    
    new_df['year'] = df.index.year - 2010
    new_df['quarter'] = df.index.quarter
    new_df["month"] = df.index.month
    new_df["month_sin"] = np.sin(new_df['month'] * (2 * np.pi / 12))
    new_df["day"] = df.index.day
    new_df["day_sin"] = np.sin(new_df['day'] * (2 * np.pi / 12))
    new_df["day_of_week"] = df.index.dayofweek
    new_df["day_of_year"] = df.index.dayofyear
    new_df['week'] = df.index.week
    new_df['week'] = new_df['week'].apply(lambda x: x if x != 53 else 52)
    new_df['is_weekend'] = new_df.apply(lambda x: 1 if (x['day_of_week'] >= 5) else 0, axis=1)
    
    # new_df["day_of_year"] = new_df.apply(lambda x: x["day_of_year"]-1 if (x.index > pd.Timestamp("2020-02-29") and x.index < pd.Timestamp("2021-01-01"))  else x["day_of_year"], axis=1)

    important_dates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 124, 125, 126, 127, 140, 141, 167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 
                       180, 181, 203, 230, 231, 232, 233, 234, 282, 289, 290, 307, 308, 309, 310, 311, 312, 313, 317, 318, 319, 320, 360, 361, 362, 363, 364, 365]

    new_df["important_dates"] = new_df["day_of_year"].apply(lambda x: x if x in important_dates else 0)
    
    # easter_date = new_df.index.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    # for day in list(range(-5, 5)) + list(range(40, 48)):
    #     new_df[f'easter_{day}'] = (new_df.index - easter_date).days.eq(day)
    new_df = new_df.drop(columns=["month","day", "day_of_year"])
    
    # for col in new_df.columns :
    #     if 'easter' in col :
    #         new_df = pd.get_dummies(new_df, columns = [col], drop_first=True)
    
    return new_df


def get_holidays(df, country_name):
    years_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    
    country_map = {
                'Canada': 'CA',
                'Finland': 'FI',
                'Kenya': 'KE',
                'Italy': 'IT',
                'Norway': 'NO',
                'Singapore': 'SI'
                }

    holiday_ = holidays.CountryHoliday(country_map[country_name], years = years_list)

    df['holiday_name'] = df.index.map(holiday_)
    df['is_holiday'] = np.where(df['holiday_name'].notnull(), 1, 0)
    df = df.drop('holiday_name', axis=1)

    return df


# import datasets
data_path = '/kaggle/input/playground-series-s5e1/'
train_df = pd.read_csv(data_path + 'train.csv', index_col='date', parse_dates=['date'])
test_df = pd.read_csv(data_path + 'test.csv', index_col='date', parse_dates=['date'])
sub_df = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

train_df = train_df.to_period("D")
train_df.name = 'train data'
test_df = test_df.to_period("D")
test_df.name = 'test data'
data_list = [train_df, test_df]
print('All datasets completely imported!')


# define some parameteres for training models
DEVICE = 'cpu'
TARGET = 'num_sold'
NUM_BOOST_ROUNDS = 2
EARLY_STOPPING_ROUNDS = 2
VERBOSE_EVAL = 2
N_SPLITS = 2
N_REPEATS = 2
N_TRIALS = 2
print('All parameters are in right place!')


# datasets first few rows
for data in data_list:
    print(f'\n{data.name} shape: ', data.shape)
    display(data.head(4))
print('\n sample submission data shape: ', sub_df.shape)
display(sub_df.head(4))


# datasets basic information
object_cols = [col for col in train_df.columns if train_df[col].dtype==object]
for data in data_list:
    for col in object_cols:
        data[col] = data[col].astype('category')
        
for data in data_list:
    print(10*'* '+f'{data.name} information'+10*' *')
    display(data.info())
    print('\n')


# numerical features statistics
for data in data_list:
    print(10*'* '+f'{data.name} numerical features statistics'+10*' *')
    display(data.describe())


# categorical features details
for data in data_list:
    print(10*'* '+f'{data.name} categorical columns unique values'+10*' *')
    for col in data.select_dtypes('category'):
        print(f'for column "{col}" we have {data[col].nunique()} unique values and their value counts is:')
        display(data[col].value_counts())
        print('\n')
    print('\n')


# missing values
for data in data_list:
    print(10*'* '+f'{data.name} missing values'+10*' *')
    display(data.isna().sum())
    print('\n')


# categorical features violin plot
fig, ax = plt.subplots(1, 3, figsize=(8, 3.5))
fig.suptitle('Categorical Features Distribution')
for col, ax in zip([col for col in train_df.columns if train_df[col].dtype == 'category'], ax.ravel()):
    sns.violinplot(data=train_df, x=col, y=TARGET, cut=0, scale='count', ax=ax)
    ax.tick_params('x', rotation=30)
    # ax.set_ylabel('')
plt.tight_layout()
plt.show()


# train hybrid model on fragmented dataset
rc_params = {
            'axes.titlesize':5,
}
plt.rcParams.update(rc_params)

train_df.fillna(method='bfill', inplace=True)

fig, ax = plt.subplots(18, 5, figsize=(10, 40))

for country, i in zip(train_df['country'].unique(), range(6)):
    for store, k in zip(train_df['store'].unique(), range(3)):
        for product, j in zip(train_df['product'].unique(), range(5)):
            df = train_df.loc[(train_df['country']==country) & (train_df['store']==store) & (train_df['product']==product), ['num_sold']]
            
            score_dict = find_best_freq_order(df)
            freq, order, dp_order = min(score_dict, key=score_dict.get).split('_')
            freq = float(freq)
            order = int(order)
            dp_order = int(dp_order)
            
            dp = DeterministicProcess(
                        index=df.index,
                        constant=True,               # dummy feature for bias (y-intercept)
                        order=dp_order,              # trend (order 1 means linear)
                        seasonal=True,               # weekly seasonality (indicators)
                        drop=True,                   # drop terms to avoid collinearity
                    )

            X_1 = dp.in_sample()  # create features for dates in tunnel.index
            X_1 = pd.concat([X_1, fourier_features(df.index, freq=365.25*freq, order=order)], axis=1)
            y = df['num_sold']
            model = LinearRegression(fit_intercept=False)
            model.fit(X_1, y)
            y_pred_1 = pd.Series(model.predict(X_1), index=y.index)
            X_fore_1 = dp.out_of_sample(steps=1095)
            X_fore_1 = pd.concat([X_fore_1, fourier_features(X_fore_1.index, freq=365.25*freq, order=order)], axis=1)
            y_fore_1 = pd.Series(model.predict(X_fore_1), index=X_fore_1.index)
            X_2 = pd.DataFrame(train_df.index.unique(), index=train_df.index.unique())
            X_2 = get_holidays(X_2, country)
            X_2 = feature_engineer(X_2)
            X_2 = X_2.drop(['date'], axis=1)
            
            X_fore_2 = pd.DataFrame(test_df.index.unique(), index=test_df.index.unique())
            X_fore_2 = get_holidays(X_fore_2, country)
            X_fore_2 = feature_engineer(X_fore_2)
            X_fore_2 = X_fore_2.drop(['date'], axis=1)
            
            # Create residuals (the collection of detrended series) from the training set
            y_resid = y - y_pred_1
            
            # Train XGBoost on the residuals
            xgb = XGBRegressor()
            xgb.fit(X_2, y_resid)
            
            # Add the predicted residuals onto the predicted trends
            y_pred_total = xgb.predict(X_2) + y_pred_1
            y_fore_total = xgb.predict(X_fore_2) + y_fore_1
            
            y.plot(ax=ax[i*3+k, j], color='0.95', style='.-',markerfacecolor="0.25", markersize=6, title=f'{country} {store} {product}',label='Actual Values', legend=True, xlabel='')
            y_pred_total.plot(ax=ax[i*3+k, j], linewidth=0.1, label="Trend fitted", legend=True, xlabel='', alpha=0.8)
            y_fore_total.plot(ax=ax[i*3+k, j], linewidth=0.1, label="Trend Forecasted", color="C3", legend=True, xlabel='')
            test_df.loc[(test_df['country']==country) & (test_df['store']==store) & (test_df['product']==product), ['num_sold']] = y_fore_total
plt.tight_layout()
plt.show()


# df = train_df.loc[(train_df['country']=='Canada') & (train_df['store']=='Discount Stickers') & (train_df['product']=='Kaggle'), ['num_sold']]
# df_test = test_df.loc[(test_df['country']=='Canada') & (test_df['store']=='Discount Stickers') & (test_df['product']=='Kaggle'), ['num_sold']]
# df.head()


# fig, ax = plt.subplots()
# df_rolling = df.rolling(
#     window = 365,
#     min_periods = 185,
#     center=True,
# ).mean()
# ax = df['num_sold'].plot(style='.', color='0.5')
# df_rolling.plot(ax=ax, linewidth=3, title="Num Solds - 365-Day Moving Average", legend=False,)
# plt.show()


# X = df.copy()

# # days within a week
# X["day"] = X.index.dayofweek  # the x-axis (freq)
# X["week"] = X.index.week  # the seasonal period (period)

# # days within a year
# X["dayofyear"] = X.index.dayofyear
# X["year"] = X.index.year
# fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
# seasonal_plot(X, y="num_sold", period="week", freq="day", ax=ax0)
# seasonal_plot(X, y="num_sold", period="year", freq="dayofyear", ax=ax1);


# plot_periodogram(df.num_sold)


# def find_best_freq_order(df):
#     score_dict = {}
#     for freq in list([1, 2, 3, 4, 0.5, 0.25, 1/6]):
#         for order in list([1, 2, 3]):
#             dp = DeterministicProcess(
#                 index=df.index,
#                 constant=True,               # dummy feature for bias (y-intercept)
#                 order=1,                     # trend (order 1 means linear)
#                 seasonal=True,               # weekly seasonality (indicators)
#                 drop=True,                   # drop terms to avoid collinearity
#             )
            
#             X = dp.in_sample()  # create features for dates in tunnel.index
#             X = pd.concat([X, fourier_features(df.index, freq=365.25*freq, order=order)], axis=1)
#             y = df['num_sold']
#             X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=365, shuffle=False)
            
#             model = LinearRegression(fit_intercept=False)
            
    
#             model.fit(X_train, y_train)
#             y_eval = pd.Series(model.predict(X_valid), index=X_valid.index)
#             mape = mean_absolute_percentage_error(y_eval, y_valid)
#             score_dict[f'{freq}_{order}'] = mape
#     return score_dict
    
# dp = DeterministicProcess(
#             index=df.index,
#             constant=True,               # dummy feature for bias (y-intercept)
#             order=1,                     # trend (order 1 means linear)
#             seasonal=True,               # weekly seasonality (indicators)
#             drop=True,                   # drop terms to avoid collinearity
#         )
# score_dict = find_best_freq_order(df)
# freq, order = min(score_dict, key=score_dict.get).split('_')
# freq = int(freq)
# order = int(order)
# X = dp.in_sample()
# X = pd.concat([X, fourier_features(df.index, freq=365.25*freq, order=order)], axis=1)
# y = df['num_sold']
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=365, shuffle=False)
# model = LinearRegression(fit_intercept=False)
# model.fit(X_train, y_train)
# y_fit = pd.Series(model.predict(X_train), index=X_train.index)
# y_pred = pd.Series(model.predict(X_valid), index=X_valid.index)
# X_fore = dp.out_of_sample(steps=1095)
# X_fore = pd.concat([X_fore, fourier_features(X_fore.index, freq=365.25*freq, order=order)], axis=1)
# y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
# # ax = y.plot(color='0.25', style='.', title="Num Sold - Seasonal Forecast")
# # ax = y_pred.plot(ax=ax, label="Seasonal", alpha=0.8)
# # ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3', alpha=0.8)
# # ax.legend()

# X_2 = get_holidays(df, country)
# X_2 = feature_engineer(X_2)
# X_fore_2 = get_holidays(df_test, country)
# X_fore_2 = feature_engineer(X_fore_2)

# X_2 = X_2.drop(['num_sold'], axis=1)
# X_fore_2 = X_fore_2.drop(['num_sold'], axis=1)
# # Create residuals (the collection of detrended series) from the training set
# y_resid = y_train - y_fit
# X_train_2, X_valid_2 = train_test_split(X_2, test_size=365, shuffle=False)

# # Train XGBoost on the residuals
# xgb = XGBRegressor()
# xgb.fit(X_train_2, y_resid)

# # Add the predicted residuals onto the predicted trends
# y_fit_boosted = xgb.predict(X_train_2) + y_fit
# y_pred_boosted = xgb.predict(X_valid_2) + y_pred
# y_fore_total = xgb.predict(X_fore_2) + y_fore

# axs = y_train.plot(color='0.25', subplots=True, sharex=True)
# axs = y_valid.plot(color='0.25', subplots=True, sharex=True, ax=axs)
# axs = y_fit_boosted.plot(color='C0', subplots=True, sharex=True, ax=axs)
# axs = y_pred_boosted.plot(color='C3', subplots=True, sharex=True, ax=axs)
# axs = y_fore_total.plot(color='C3', subplots=True, sharex=True, ax=axs)
# for ax in axs: ax.legend([])
# _ = plt.suptitle("Trends")

# plt.show()


sub_df['num_sold'] = np.array(test_df['num_sold'])
sub_df.to_csv('submission.csv')
sub_df.head(4)

