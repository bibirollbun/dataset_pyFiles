import shap
import holidays
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import STL

from catboost import CatBoostRegressor, Pool

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'], index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'], index_col='id')

monthly_train = train.groupby(['country', 'store', 'product', pd.Grouper(key='date', freq='MS')])['num_sold'].sum().rename('num_sold').reset_index()
weekly_train = train.groupby(['country', 'store', 'product', pd.Grouper(key='date', freq='W')])['num_sold'].sum().rename('num_sold').reset_index()

gdp = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')


train[['country', 'store', 'product']].drop_duplicates()


train.info()


# Time interval of the train and test sets

print(f'--> Train set is expanding from {train["date"].min()} to {train["date"].max()}')
print(f'--> Test set is expanding from {test["date"].min()} to {test["date"].max()}')


train['country'].unique().tolist()


train['store'].unique().tolist()


train.groupby('country')['store'].unique().reset_index()


train['product'].unique().tolist()


train.groupby(['country', 'store'])['product'].nunique().reset_index()


train.isnull().sum()


num_of_rows = train.groupby(['country', 'store', 'product'])['num_sold'].count()
num_of_missing_rows = num_of_rows.loc[num_of_rows != 2557]
num_of_missing_rows = num_of_missing_rows.reset_index()
num_of_missing_rows['num_missing_rows'] = 2557 - num_of_missing_rows['num_sold']
num_of_missing_rows


missing_data = num_of_missing_rows[num_of_missing_rows['num_missing_rows'] > 200]

combinations = missing_data.drop_duplicates(subset=['country', 'store', 'product'])
combinations = combinations.reset_index(drop=True)

fig, axes = plt.subplots(3, 2, figsize=(20, 10))  
axes = axes.flatten()

for i, row in combinations.iterrows():
    if i >= len(axes):  
        break
    
    filtered_df = train[(train['country'] == row['country']) &
                        (train['store'] == row['store']) &
                        (train['product'] == row['product'])]
    
    filtered_df = filtered_df.set_index('date')

    missing_indices = filtered_df[filtered_df.isna().any(axis=1)].index

    axes[i].plot(filtered_df['num_sold'])
    axes[i].set_title(f"{row['country']}, {row['store']}, {row['product']}")
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Num Sold')

    for idx in missing_indices:
        axes[i].axvline(x=idx, color='red', linestyle='--', linewidth=1)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


country_sales_ratio_over_time = (train.groupby(['date', 'country'])['num_sold'].sum() / train.groupby(['date'])['num_sold'].sum()).reset_index()


## Discount Stickers & Holographic Goose

filtered_df = monthly_train[(monthly_train['store'] == 'Discount Stickers') &
                            (monthly_train['product'] == 'Holographic Goose')]

plt.figure(figsize=(20, 4))

plt.subplot(1, 2, 1)
sns.lineplot(data=filtered_df, x='date', y='num_sold', hue='country')
plt.title('Discount Stickers - Holographic Goose (Montly Total Sales by Country)')

plt.subplot(1, 2, 2)
sns.lineplot(data=country_sales_ratio_over_time, x='date', y='num_sold', hue='country')
plt.title('Ratio of total sales over time by each country')
plt.show()

## Stickers for Less & Holographic Goose

filtered_df = monthly_train[(monthly_train['store'] == 'Stickers for Less') &
                            (monthly_train['product'] == 'Holographic Goose')]

plt.figure(figsize=(20, 4))

plt.subplot(1, 2, 1)
sns.lineplot(data=filtered_df, x='date', y='num_sold', hue='country')
plt.title('Stickers for Less - Holographic Goose (Montly Total Sales by Country)')

plt.subplot(1, 2, 2)
sns.lineplot(data=country_sales_ratio_over_time, x='date', y='num_sold', hue='country')
plt.title('Ratio of total sales over time by each country')
plt.show()


grouped_by_country = train.groupby(["date", "country"])["num_sold"].sum().reset_index()

plt.figure(figsize=(20, 6))
sns.lineplot(data=grouped_by_country, x='date', y='num_sold', hue='country')
plt.title('Total Daily Sales by Country')

del grouped_by_country


grouped_by_store = train.groupby(['date', 'store'])['num_sold'].sum().reset_index()

plt.figure(figsize=(20, 6))
sns.lineplot(data=grouped_by_store, x='date', y='num_sold', hue='store')
plt.title('Total Daily Sales by Store')

del grouped_by_store


grouped_by_product = train.groupby(["date", "product"])["num_sold"].sum().reset_index()

plt.figure(figsize=(20, 6))
sns.lineplot(data=grouped_by_product, x='date', y='num_sold', hue='product')
plt.title('Total Daily Sales by Product')

del grouped_by_product


def get_holiday_flag(df):
    country_holidays = {
            'Canada': holidays.CountryHoliday('CA'),
            'Finland': holidays.CountryHoliday('FI'),
            'Italy': holidays.CountryHoliday('IT'),
            'Kenya': holidays.CountryHoliday('KE'),
            'Norway': holidays.CountryHoliday('NO'),
            'Singapore': holidays.CountryHoliday('SG')
        }
    
    df['holiday_flag'] = df.apply(
        lambda row: row['date'] in country_holidays.get(row['country'], []), axis=1
    )
    
    return df


train_copy = train.copy()

train_copy['year'] = train['date'].dt.year.astype(str)
train_copy['month'] = train['date'].dt.month
train_copy['quarter'] = train['date'].dt.quarter.astype(str)
train_copy['day_of_week'] = train['date'].dt.dayofweek.astype(str)
train_copy['day_of_year'] = train['date'].dt.dayofyear
train_copy['day_month'] = train_copy['date'].dt.strftime('%d.%m')
train_copy = get_holiday_flag(train_copy)

train_copy.head()


train_copy_agg = train_copy.groupby(['date'])['num_sold'].sum().reset_index()

train_copy_agg['year'] = train_copy_agg['date'].dt.year.astype(str)
train_copy_agg['month'] = train_copy_agg['date'].dt.month
train_copy_agg['quarter'] = train_copy_agg['date'].dt.quarter.astype(str)
train_copy_agg['day_of_week'] = train_copy_agg['date'].dt.dayofweek.astype(str)
train_copy_agg['day_of_year'] = train_copy_agg['date'].dt.dayofyear
train_copy_agg['day_month'] = train_copy_agg['date'].dt.strftime('%d.%m')

date_feature_corr = train_copy_agg[['year', 'month', 'quarter', 'day_of_week', 'day_of_year', 'day_month', 'num_sold']].corr()
date_feature_corr = date_feature_corr.round(3)

plt.figure(figsize=(6, 6))
sns.heatmap(date_feature_corr, annot=True)


product_mean_yearly = train_copy.groupby(['year', 'product'])['num_sold'].mean().reset_index()
prodcut_mean_monthly = train_copy.groupby(['month', 'product'])['num_sold'].mean().reset_index()
product_mean_daily = train_copy.groupby(['day_of_week', 'product'])['num_sold'].mean().reset_index()
product_mean_quarterly = train_copy.groupby(['quarter', 'product'])['num_sold'].mean().reset_index()

fig, axs = plt.subplots(2, 2, figsize=(20, 10))

sns.lineplot(x='year', y='num_sold', hue='product', data=product_mean_yearly, ax=axs[0, 0])
axs[0, 0].set_ylabel('Average Number of Sales');
axs[0, 0].set_xlabel('Year');

sns.lineplot(x='month', y='num_sold', hue='product', data=prodcut_mean_monthly, ax=axs[0, 1])
axs[0, 1].set_ylabel('Average Number of Sales');
axs[0, 1].set_xlabel('Month');

sns.lineplot(x='day_of_week', y='num_sold', hue='product', data=product_mean_daily, ax=axs[1, 0])
axs[1, 0].set_ylabel('Average Number of Sales');
axs[1, 0].set_xlabel('Weekday');

sns.lineplot(x='quarter', y='num_sold', hue='product', data=product_mean_quarterly, ax=axs[1,1])
axs[1, 1].set_ylabel('Average Number of Sales');
axs[1, 1].set_xlabel('Quarter');

del product_mean_yearly, prodcut_mean_monthly, product_mean_daily, product_mean_quarterly


fig, ax = plt.subplots(3,2,figsize=(12, 6))
ax = ax.flatten()

for idx, product in enumerate(train_copy['product'].unique()):
    product_df = train_copy.set_index('date')
    product_weekday_avg = product_df[product_df['product'] == product]['num_sold'].resample('D').mean().to_frame()
    product_weekday_avg = product_weekday_avg.groupby(product_weekday_avg.index.dayofweek)['num_sold'].mean()

    ax[idx].plot(product_weekday_avg)
    ax[idx].set_title(product)

for j in range(idx + 1, len(axes)):
    ax[j].axis('off')

plt.tight_layout()


fig, ax = plt.subplots(3,2,figsize=(12, 6))
ax = ax.flatten()

for idx, product in enumerate(product_df['product'].unique()):
    product_df = train_copy.set_index('date')
    product_weekly_avg = product_df[product_df['product'] == product]['num_sold'].resample('W').mean().to_frame()
    product_weekly_avg = product_weekly_avg.groupby(product_weekly_avg.index.isocalendar().week)['num_sold'].mean()

    ax[idx].plot(product_weekly_avg)
    ax[idx].set_title(product)

for j in range(idx + 1, len(axes)):
    ax[j].axis('off')

plt.tight_layout()


store_mean_yearly = train_copy.groupby(['year', 'store'])['num_sold'].mean().reset_index()
store_mean_montly = train_copy.groupby(['month', 'store'])['num_sold'].mean().reset_index()
store_mean_day = train_copy.groupby(['day_of_week', 'store'])['num_sold'].mean().reset_index()
store_mean_quarter = train_copy.groupby(['quarter', 'store'])['num_sold'].mean().reset_index()

fig, axs = plt.subplots(1, 4, figsize=(24, 5))

sns.lineplot(x='year', y='num_sold', hue='store', data=store_mean_yearly, ax=axs[0])
axs[0].set_ylabel('Average Number of Sales');
axs[0].set_xlabel('Year');

sns.lineplot(x='month', y='num_sold', hue='store', data=store_mean_montly, ax=axs[1])
axs[1].set_ylabel('Average Number of Sales');
axs[1].set_xlabel('Month');

sns.lineplot(x='day_of_week', y='num_sold', hue='store', data=store_mean_day, ax=axs[2])
axs[2].set_ylabel('Average Number of Sales');
axs[2].set_xlabel('Weekday');

sns.lineplot(x='quarter', y='num_sold', hue='store', data=store_mean_quarter, ax=axs[3])
axs[3].set_ylabel('Average Number of Sales');
axs[3].set_xlabel('Quarter');


country_ratios = (train_copy.groupby(['date', 'country'])['num_sold'].mean() / train_copy.groupby(['date'])['num_sold'].mean()).reset_index()
country_ratios

plt.figure(figsize=(20, 4))
sns.lineplot(data=country_ratios, x='date', y='num_sold', hue='country');


store_ratios = (train_copy.groupby(['date', 'store'])['num_sold'].mean() / train_copy.groupby(['date'])['num_sold'].mean()).reset_index()
store_ratios

plt.figure(figsize=(20, 4))
sns.lineplot(data=store_ratios, x='date', y='num_sold', hue='store');


product_ratios = (train_copy.groupby(['date', 'product'])['num_sold'].mean() / train_copy.groupby(['date'])['num_sold'].mean()).reset_index()
product_ratios

plt.figure(figsize=(20, 4))
sns.lineplot(data=product_ratios, x='date', y='num_sold', hue='product');


avg_num_sold = (
    train_copy.groupby(["country", "day_of_year"])["num_sold"].mean().reset_index()
)
holidays_df = train_copy[train_copy["holiday_flag"]].drop_duplicates(
    ["country", "day_of_year"]
)

g = sns.FacetGrid(
    avg_num_sold,
    col="country",
    col_wrap=2,
    height=4,
    aspect=2,
    sharex=False,
    sharey=False,
)
g.map_dataframe(sns.lineplot, x="day_of_year", y="num_sold")


def format_day_month(x, pos):
    if 1 <= x <= 366:
        date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=x - 1)
        return date.strftime("%d.%m")
    return ""


for ax in g.axes.flat:
    ax.xaxis.set_major_formatter(FuncFormatter(format_day_month))

for ax, country in zip(g.axes.flat, avg_num_sold["country"].unique()):
    country_holidays = holidays_df[holidays_df["country"] == country]["day_of_year"]
    for holiday in country_holidays:
        ax.axvline(x=holiday, color="red", linestyle="--", alpha=0.7, label="Holiday")

g.set_axis_labels("Day.Month", "Average Sales")
g.set_titles(col_template="{col_name}")
plt.suptitle("Average Number of Sales for Each Day of the Year by Country")
plt.tight_layout()
plt.show()


def get_gdp_data(countries, year_start, year_end):
    gdp = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')

    gdp.drop(['Code'], axis=1, inplace=True)
    years_to_select = [str(year) for year in range(year_start, year_end+1)]
    
    gdp = gdp.loc[:, ['Country Name'] + years_to_select]
    gdp = gdp[gdp['Country Name'].isin(countries)]
    gdp = gdp.melt(id_vars='Country Name', var_name='Year', value_name='Value')
    
    gdp = gdp.reset_index(drop=True)
    gdp.columns = ['country', 'year', 'gdp']
    return gdp
    
def get_country_gdp_ratio(gdp):
    gdp_ratios = (gdp.groupby(['year', 'country'])['gdp'].sum() / gdp.groupby('year')['gdp'].sum()).reset_index()

    months = pd.DataFrame({'month': [f'{m:02d}' for m in range(1, 13)]})
    years = gdp_ratios['year'].unique()
    month_year_df = pd.DataFrame([(y, m) for y in years for m in months['month']], columns=['year', 'month'])
    gdp_ratios = gdp_ratios.merge(month_year_df, on='year')
    
    gdp_ratios['date'] = pd.to_datetime(gdp_ratios['year'].str[:4] + '-' + gdp_ratios['month'] + '-01')
    gdp_ratios.drop(columns=['year', 'month'], inplace=True)
    gdp_ratios = gdp_ratios[['date', 'country', 'gdp']]
    return gdp_ratios

def get_daily_gdp_data(gdp_data):
    df = gdp_data.copy()
    df = df.set_index('date')

    daily_index = pd.date_range(start='2010-01-01', end='2016-12-31', freq='D')
    daily_df = df.reindex(daily_index).ffill()
    daily_df.reset_index(inplace=True)
    daily_df.rename(columns={'index': 'date'}, inplace=True)
    return daily_df.set_index('date')


train_set_gdp = get_gdp_data(train['country'].unique().tolist(), 2010, 2016)
gdp_ratios = get_country_gdp_ratio(train_set_gdp)


plt.figure(figsize=(20, 8))
sns.lineplot(data=country_sales_ratio_over_time, x='date', y='num_sold', hue='country')
sns.lineplot(data=gdp_ratios, x='date', y='gdp', hue='country', linestyle='--', linewidth=1.2, palette = ["black"]*6, legend = False)
plt.title('Ratio of total sales over time by each country & GDP')
plt.show()


def impute_missing_values_by_using_a_reference_country(df, reference_country, target_country, store, product, gdp_ratios_df):
    target_df = df[(df['country'] == target_country) &
                (df['store'] == store) &
                (df['product'] == product)]

    reference_df = df[(df['country'] == reference_country) &
                (df['store'] == store) &
                (df['product'] == product)]
    
    target_gdp = gdp_ratios_df[gdp_ratios_df['country'] == target_country].set_index('date')['gdp'].reset_index()
    reference_gdp = gdp_ratios_df[gdp_ratios_df['country'] == reference_country].set_index('date')['gdp'].reset_index()

    target_gdp_daily = get_daily_gdp_data(target_gdp)
    reference_gdp_daily = get_daily_gdp_data(reference_gdp)

    imputed_value = (reference_df.set_index('date')['num_sold'] * target_gdp_daily['gdp']) / reference_gdp_daily['gdp']
    imputed_value = imputed_value.rename('num_sold')
    imputed_value.index = target_df.index

    target_df['num_sold'] = target_df['num_sold'].fillna(imputed_value)
    return target_df

def plot_imputation_results(target_df, reference_df, imputed_target_df, title, plot_reference=True):
    
    if plot_reference:
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 2, 1)
        sns.lineplot(data=reference_df, x='date', y='num_sold', label=reference_df['country'].values[0])
        sns.lineplot(data=target_df, x='date', y='num_sold', label=target_df['country'].values[0])
    
        plt.subplot(1, 2, 2)
        sns.lineplot(data=imputed_target_df, x='date', y='num_sold', label='Imputed')
        sns.lineplot(data=target_df, x='date', y='num_sold', label=target_df['country'].values[0])
    else:
        plt.figure(figsize=(10, 4))
        sns.lineplot(data=imputed_target_df, x='date', y='num_sold', label='Imputed')
        if not target_df['num_sold'].isnull().all():
            sns.lineplot(data=target_df, x='date', y='num_sold', label=target_df['country'].values[0])
        
    plt.suptitle(title)
    plt.show()


reference_country = 'Norway'

missing_data = num_of_missing_rows[num_of_missing_rows['num_missing_rows'] > 0]
combinations = missing_data.drop_duplicates(subset=['country', 'store', 'product'])
combinations = combinations.reset_index(drop=True)

for i, row in combinations.iterrows():
    target_country_df = train[(train['country'] == row['country']) &
                            (train['store'] == row['store']) &
                            (train['product'] == row['product'])]

    reference_country_df = train[(train['country'] == reference_country) &
                                (train['store'] == row['store']) &
                                (train['product'] == row['product'])]

    
    imputed_target = impute_missing_values_by_using_a_reference_country(train_copy,
                                                                       reference_country,
                                                                       row['country'],
                                                                       row['store'],
                                                                       row['product'],
                                                                       gdp_ratios)
    
    train_copy.loc[target_country_df.index, 'num_sold'] = imputed_target['num_sold']
    
    plot_imputation_results(target_country_df,
                            reference_country_df,
                            imputed_target,
                            title=f'Impute {row["country"]} - {row["store"]} - {row["product"]}',
                            plot_reference=False)


def impute_missing_values_by_using_a_reference_country(df, reference_country, target_country, store, product, gdp_ratios_df):
    target_df = df[(df['country'] == target_country) &
                (df['store'] == store) &
                (df['product'] == product)]

    reference_df = df[(df['country'] == reference_country) &
                (df['store'] == store) &
                (df['product'] == product)]
    
    target_gdp = gdp_ratios_df[gdp_ratios_df['country'] == target_country].set_index('date')['gdp'].reset_index()
    reference_gdp = gdp_ratios_df[gdp_ratios_df['country'] == reference_country].set_index('date')['gdp'].reset_index()

    target_gdp_daily = get_daily_gdp_data(target_gdp)
    reference_gdp_daily = get_daily_gdp_data(reference_gdp)

    imputed_value = (reference_df.set_index('date')['num_sold'] * target_gdp_daily['gdp']) / reference_gdp_daily['gdp']
    imputed_value = imputed_value.rename('num_sold')
    imputed_value.index = target_df.index

    target_df['num_sold'] = target_df['num_sold'].fillna(np.round(imputed_value))
    return target_df


def get_holidays(countries, start_year, end_year):
    all_holidays = []
    for country in countries:
        country_holidays = holidays.CountryHoliday(country, years=range(start_year, end_year + 1))
        
        for date, name in country_holidays.items():
            all_holidays.append({
                'date': date,
                'country': country
            })
    
    holidays_df = pd.DataFrame(all_holidays)
    holidays_df['tmp'] = 1

    holidays_df['country'] = holidays_df['country'].map({'CA': 'Canada', 'FI': 'Finland', 'IT': 'Italy', 'KE': 'Kenya', 'NO': 'Norway', 'SG': 'Singapore'})
    return holidays_df.sort_values(by='date').reset_index(drop=True)

def add_offset_to_holidays(holiday_df, max_offset):
    result_list = []
    columns = []
    for i in range(0, max_offset):
        column = f'holiday_{i}'
        new_df = holiday_df.rename(columns={'tmp': column})
        new_df['date'] = new_df['date'] + dt.timedelta(days=i)
        new_df['date'] = pd.to_datetime(new_df['date'])
        result_list.append(new_df)
        columns.append(column)
        
    return result_list, columns


def generate_date_features(df, holiday_df_list, holiday_cols, aggregated_df=False):
    df_new = df.copy()

    df_new['year'] = df_new['date'].dt.year
    df_new['month'] = df_new['date'].dt.month
    df_new['quarter'] = df_new['date'].dt.quarter
    df_new['week'] = df_new['date'].dt.isocalendar().week
    df_new['day'] = df_new['date'].dt.day
    df_new['day_of_week'] = df_new['date'].dt.dayofweek
    df_new['day_of_year'] = df_new['date'].dt.dayofyear
    
    df_new['sine_day'] = np.sin(2 * np.pi * df_new['day'] / 31)
    df_new['cos_day'] = np.cos(2 * np.pi * df_new['day'] / 31)

    df_new['year_even'] = df_new['year'] % 2

    # df_new['product_year_even'] = df_new['product'].astype(str) + '-' + df_new['year_even'].astype(str)
    # df_new['product_year_even'] = df_new['product_year_even'].astype('category')
    
    for i in [1, 2]:
        sine_name = f'sine_year_{i}'
        cos_name = f'cos_year_{i}'
            
        df_new[sine_name] = np.sin(i * np.pi * df_new['day_of_year'] / 365)
        df_new[cos_name] = np.cos(i * np.pi * df_new['day_of_year'] / 365)

    df_new['isFriday'] = (df_new['day_of_week'] == 4).astype(int)
    df_new['isWeekend'] = df_new['day_of_week'].isin([5, 6]).astype(int)
    df_new['week16'] = (df_new['week'] == 16).astype(int)

    # Special New Year's Dates
    for day in range(25, 32):
        column_name = f'december_{day}'
        df_new[column_name] = ((df_new['month'] == 12) & (df_new['day'] == day)).astype(int)

    for day in range(1, 12):
        column_name = f'january_{day}'
        df_new[column_name] = ((df_new['month'] == 1) & (df_new['day'] == day)).astype(int)

    holiday_agg_cols = ['date'] if aggregated_df else ['country', 'date']
    for idx, hols_df in enumerate(holiday_df_list):
        df_new = df_new.merge(
            hols_df,
            on=holiday_agg_cols,
            how='left'
        )
        column = holiday_cols[idx]
        df_new[column] = df_new[column].fillna(0).astype(int)
    
    return df_new


def define_catorical_features(df, columns):
    df_new = df.copy()
    df_new[columns] = df_new[columns].astype('category')

    df_new['store_product'] = df_new['store'].astype(str) + '-' + df_new['product'].astype(str)
    df_new['store_product'] = df_new['store_product'].astype('category')
    return df_new


def add_gdp_ratio_feature(df, gdp_ratio_df):
    df_new = df.copy()
    gdp = gdp_ratio_df.copy()
    
    gdp['year'] = gdp['date'].dt.year
    gdp['month'] = gdp['date'].dt.month
    
    df_new = df_new.merge(gdp.drop(['date'], axis=1), on=['year', 'month', 'country'], how='inner')
    df_new.index = df.index
    return df_new


def custom_imputation(df, ratios, country, product, reference_store, target_stores):
    df_new = df.copy()
    
    # Filter out main df with specified classes of country and product 
    filtered_df = df[(df['country'] == country) & 
                     (df['product'] == product)]

    # Reference store within same settings
    reference_ratio = ratios[ratios['store'] == reference_store]['num_sold']
    reference_values = filtered_df[filtered_df['store'] == reference_store]['num_sold']

    for target in target_stores:
        target_ratio = ratios[ratios['store'] == target]['num_sold']
        values_to_use = (reference_values.values * target_ratio.values) / reference_ratio.values

        imputation_df = pd.DataFrame(data=values_to_use,
                                     index=filtered_df[filtered_df['store'] == target].index,
                                     columns=['num_sold'])
        
        df_new.loc[filtered_df[filtered_df['store'] == target].index] = filtered_df.loc[filtered_df[filtered_df['store'] == target].index].fillna(imputation_df)
    return df_new


def get_gdp_data(countries, year_start, year_end):
    gdp = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')

    gdp.drop(['Code'], axis=1, inplace=True)
    years_to_select = [str(year) for year in range(year_start, year_end+1)]
    
    gdp = gdp.loc[:, ['Country Name'] + years_to_select]
    gdp = gdp[gdp['Country Name'].isin(countries)]
    gdp = gdp.melt(id_vars='Country Name', var_name='Year', value_name='Value')
    
    gdp = gdp.reset_index(drop=True)
    gdp.columns = ['country', 'year', 'gdp']
    return gdp
    
def get_country_gdp_ratio(gdp):
    gdp_ratios = (gdp.groupby(['year', 'country'])['gdp'].sum() / gdp.groupby('year')['gdp'].sum()).reset_index()

    months = pd.DataFrame({'month': [f'{m:02d}' for m in range(1, 13)]})
    years = gdp_ratios['year'].unique()
    month_year_df = pd.DataFrame([(y, m) for y in years for m in months['month']], columns=['year', 'month'])
    gdp_ratios = gdp_ratios.merge(month_year_df, on='year')
    
    gdp_ratios['date'] = pd.to_datetime(gdp_ratios['year'].str[:4] + '-' + gdp_ratios['month'] + '-01')
    gdp_ratios.drop(columns=['year', 'month'], inplace=True)
    gdp_ratios = gdp_ratios[['date', 'country', 'gdp']]
    return gdp_ratios

def get_daily_gdp_data(gdp_data):
    df = gdp_data.copy()
    df = df.set_index('date')

    daily_index = pd.date_range(start='2010-01-01', end='2016-12-31', freq='D')
    daily_df = df.reindex(daily_index).ffill()
    daily_df.reset_index(inplace=True)
    daily_df.rename(columns={'index': 'date'}, inplace=True)
    return daily_df.set_index('date')


def decompose_timeseries(df):
    '''
    Decompose the time series into different components based on the idea 
    that each country, store, and product operates at distinct levels of `num_sales`.
    '''
    df_new = df.copy()
    
    df_new['day_of_week'] = df_new['date'].dt.dayofweek
    df_new['year'] = df_new['date'].dt.year

    # Day of week ratio
    day_of_week_sale_ratios = (df_new.groupby(['day_of_week'])['num_sold'].mean() / df_new.groupby('day_of_week')['num_sold'].mean().mean()).rename('day_of_week_ratios').reset_index()

    # Country Ratios
    countries = df_new['country'].unique().tolist()
    country_sales_ratios = get_country_gdp_ratio(get_gdp_data(countries, df_new['year'].min(), df_new['year'].max()))
    country_sales_ratios['year'] = country_sales_ratios['date'].dt.year
    country_sales_ratios = country_sales_ratios.groupby(['year', 'country'])['gdp'].mean().reset_index()
    
    # Store Ratios
    store_sales_ratios = (df_new.groupby(['store'])['num_sold'].sum() / df_new['num_sold'].sum()).rename('store_sales_ratios').reset_index()

    # Product Ratios
    product_sales_ratios = (df_new.groupby(['date', 'product'])['num_sold'].sum() / df_new.groupby(['date'])['num_sold'].sum()).rename('product_sales_ratios').reset_index()

    df_new = df_new.groupby(['date'])['num_sold'].sum().reset_index()
    
    return df_new, country_sales_ratios, store_sales_ratios, product_sales_ratios, day_of_week_sale_ratios


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'], index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'], index_col='id')


gdp_ratios = get_country_gdp_ratio(get_gdp_data(train['country'].unique().tolist(), 2010, 2016))


## Rules:

apply_missing_value_imputation = True
generate_datetype_features = True
generate_product_based_cycle_features = False
define_categorical_features = False
add_gdp_country_ratio_data = False
apply_custom_imputation = False
apply_scaling = False
apply_ohe_encoding = False
apply_transformation = True
apply_decomposition = True


if apply_missing_value_imputation:
    num_of_rows = train.groupby(['country', 'store', 'product'])['num_sold'].count()
    num_of_missing_rows = num_of_rows.loc[num_of_rows != 2557]
    num_of_missing_rows = num_of_missing_rows.reset_index()
    num_of_missing_rows['num_missing_rows'] = 2557 - num_of_missing_rows['num_sold']
    num_of_missing_rows['ratio'] = num_of_missing_rows['num_missing_rows'] / 2557
    
    combinations = num_of_missing_rows.drop_duplicates(subset=['country', 'store', 'product'])
    combinations = combinations.reset_index(drop=True)

    REFERENCE_COUNTRY = 'Norway'
    for i, row in combinations.iterrows():
        target_country_df = train[(train['country'] == row['country']) &
                                  (train['store'] == row['store']) &
                                  (train['product'] == row['product'])]
    
        reference_country_df = train[(train['country'] == REFERENCE_COUNTRY) &
                                    (train['store'] == row['store']) &
                                    (train['product'] == row['product'])]
    
        
        imputed_target = impute_missing_values_by_using_a_reference_country(train,
                                                                           REFERENCE_COUNTRY,
                                                                           row['country'],
                                                                           row['store'],
                                                                           row['product'],
                                                                           gdp_ratios)
        
        train.loc[imputed_target.index, 'num_sold'] = imputed_target['num_sold']


if apply_custom_imputation:
    ### Canada - Holographic Goose
    canada_ratio_df = train[(train['country'] == 'Canada') & 
                     (train['product'] == 'Kerneler Dark Mode')]
    
    canada_ratios = (canada_ratio_df.groupby(['date', 'store'])['num_sold'].sum() / canada_ratio_df.groupby(['date'])['num_sold'].sum()).reset_index()
    
    train = custom_imputation(train, canada_ratios, 'Canada', 'Holographic Goose',
                              'Premium Sticker Mart', ['Stickers for Less', 'Discount Stickers'])
    
    # --------------------------------------------------------------------------------------
    
    ### Kenya - Holographic Goose
    kenya_ratio_df = train[(train['country'] == 'Kenya') & 
                           (train['product'] == 'Kerneler Dark Mode')]
    
    kenya_ratios = (kenya_ratio_df.groupby(['date', 'store'])['num_sold'].sum() / kenya_ratio_df.groupby(['date'])['num_sold'].sum()).reset_index()
    
    train = custom_imputation(train, kenya_ratios, 'Kenya', 'Holographic Goose',
                              'Premium Sticker Mart', ['Stickers for Less', 'Discount Stickers'])


if apply_decomposition:
    train, country_sales_ratios, store_sales_ratios, product_sales_ratios, day_of_week_sale_ratios = decompose_timeseries(train)

    ### Adjust num_sold by day of week sales ratios
    train['day_of_week'] = train['date'].dt.dayofweek
    train = train.merge(day_of_week_sale_ratios, on='day_of_week', how='left')
    train['num_sold'] = train['num_sold'] / train['day_of_week_ratios']
    train.drop(['day_of_week', 'day_of_week_ratios'], axis=1, inplace=True)

    ### Adjust product ratios for test data
    product_sales_ratios_2017 = product_sales_ratios.loc[product_sales_ratios["date"].dt.year == 2015].copy()
    product_sales_ratios_2018 = product_sales_ratios.loc[product_sales_ratios["date"].dt.year == 2016].copy()
    product_sales_ratios_2019 = product_sales_ratios.loc[product_sales_ratios["date"].dt.year == 2015].copy()
    
    product_sales_ratios_2017["date"] = product_sales_ratios_2017["date"] + pd.DateOffset(years=2)
    product_sales_ratios_2018["date"] = product_sales_ratios_2018["date"] + pd.DateOffset(years=2)
    product_sales_ratios_2019["date"] =  product_sales_ratios_2019["date"] + pd.DateOffset(years=4)
    
    test_product_sales_ratios = pd.concat([product_sales_ratios_2017, product_sales_ratios_2018, product_sales_ratios_2019])

    #
    test = test.groupby('date')['date'].first().rename('tmp').reset_index().drop(['tmp'], axis=1)


if generate_datetype_features:
    # TRAIN SET
    holiday_df = get_holidays(['CA', 'FI', 'IT', 'KE', 'NO', 'SG'], 2010, 2017)
    train_holiday_list, train_hols_cols = add_offset_to_holidays(holiday_df, 14)

    if apply_decomposition:
        for idx, hols_df in enumerate(train_holiday_list):
            agg_hols_df = hols_df.groupby(['date'])[train_hols_cols[idx]].sum().reset_index()
            train_holiday_list[idx] = agg_hols_df
        aggregated_df = True
    
    train = generate_date_features(train, train_holiday_list, train_hols_cols, aggregated_df=aggregated_df)

    # TEST SET
    holiday_df = get_holidays(['CA', 'FI', 'IT', 'KE', 'NO', 'SG'], 2017, 2019)
    test_holiday_list, test_hol_cols = add_offset_to_holidays(holiday_df, 14)
    
    if apply_decomposition:
        for idx, hols_df in enumerate(test_holiday_list):
            agg_hols_df = hols_df.groupby(['date'])[test_hol_cols[idx]].sum().reset_index()
            test_holiday_list[idx] = agg_hols_df
        aggregated_df = True
    
    test = generate_date_features(test, test_holiday_list, test_hol_cols, aggregated_df=aggregated_df)


if define_categorical_features:
    if not apply_decomposition:
        train = define_catorical_features(train, ['country', 'store', 'product'])
        test = define_catorical_features(test, ['country', 'store', 'product'])


if add_gdp_country_ratio_data:
    countries = train['country'].unique().tolist()
    gdp_ratios_train = get_country_gdp_ratio(get_gdp_data(countries, 2010, 2016))
    gdp_ratios_test = get_country_gdp_ratio(get_gdp_data(countries, 2017, 2019))
    
    train = add_gdp_ratio_feature(train, gdp_ratios_train)
    test = add_gdp_ratio_feature(test, gdp_ratios_test)


categorical_columns = ['country', 'store', 'product', 'product_year_even', 'store_product']


if apply_ohe_encoding:
    rename_cols = []
    for col in categorical_columns:
            values = train[col].unique().tolist()
            rename_cols += values

    # Train
    train_ohe_result = pd.get_dummies(train[categorical_columns], dtype='int', drop_first=False)
    train_ohe_result.columns = rename_cols
    train = train.drop(categorical_columns, axis=1)
    train = pd.concat([train, train_ohe_result], axis=1)

    # Test
    test_ohe_result = pd.get_dummies(test[categorical_columns], dtype='int', drop_first=False)
    test_ohe_result.columns = rename_cols
    test = test.drop(categorical_columns, axis=1)
    test = pd.concat([test, test_ohe_result], axis=1)


if generate_datetype_features and generate_product_based_cycle_features:
    product_list = ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode']
    cyclical_features = ['sine_year_1', 'cos_year_1', 'sine_year_2', 'cos_year_2']
    for product in product_list:
        prod_name = ('_'.join(product.split(' '))).lower()
        for ftr in cyclical_features:
            train[f'{prod_name.lower()}_{ftr}'] = train[ftr] * train[product]
            test[f'{prod_name.lower()}_{ftr}'] = test[ftr] * test[product]


scaler = None
if apply_scaling:
    cols_to_scale = [col for col in train.select_dtypes(exclude='object').columns.to_list() if col not in ['date', 'num_sold', 'product', 'store', 'product']]
    
    scaler = StandardScaler()
    train[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])
    test[cols_to_scale] = scaler.transform(test[cols_to_scale])


if not apply_missing_value_imputation:
    train = train.dropna()


if apply_transformation:
    train['num_sold'] = np.log(train['num_sold'])
    
    if add_gdp_country_ratio_data:
        train['log_gdp'] = np.log(train['gdp'])
        test['log_gdp'] = np.log(test['gdp'])


train.head()


def linear_model_trainer(train_df,
                        test_df,
                        feature_set,
                        target,
                        scorer,
                        log=True):

    val_scores = []
    unseen_preds = []
    val_preds = []
    train_preds = []
    models = []
    residuals = []
    fold = 1

    cv = KFold(n_splits=5, shuffle=True)

    train_df[feature_set] = train_df[feature_set].astype(np.float32)
    test_df[feature_set] = test_df[feature_set].astype(np.float32)    

    for train_idx, val_idx in cv.split(train_df):
        X_train, y_train = train_df.iloc[train_idx][feature_set], train_df.iloc[train_idx][target]
        X_val, y_val = train_df.iloc[val_idx][feature_set], train_df.iloc[val_idx][target]

        # y_train -= train_df.iloc[train_idx]['log_gdp']
        # y_val -= train_df.iloc[val_idx]['log_gdp']
        
        # lr = Ridge(alpha=0.001)
        lr = Lasso(tol=1e-2, max_iter=1000000, random_state=0)

        lr.fit(X_train, y_train)
        models.append(lr)

        # Validation Prediction
        val_pred = lr.predict(X_val)
        if apply_transformation:
            val_pred = np.exp(val_pred) # + train_df.iloc[val_idx]['log_gdp'])

        # Train Prediction
        train_pred = lr.predict(X_train)
        if apply_transformation:
            train_pred = np.exp(train_pred) # + train_df.iloc[train_idx]['log_gdp'])

        val_preds.append(val_pred)
        train_preds.append(train_pred)

        # Unseen test pred
        unseen_pred = lr.predict(test_df[feature_set])
        if apply_transformation:
            unseen_pred = np.exp(unseen_pred) # + test_df['log_gdp'])
        unseen_preds.append(unseen_pred)

        # Scores
        if apply_transformation:
            y_train = np.exp(y_train) # + train_df.iloc[train_idx]['log_gdp'])
            y_val = np.exp(y_val) # + train_df.iloc[val_idx]['log_gdp'])
            
        train_score = scorer(y_train, train_pred)
        val_score = scorer(y_val, val_pred)

        val_scores.append(val_score)

        residuals.append(y_val - val_pred)
        
        if log:   
            print(f'\nFOLD {fold}')
            print(f'Train Score: {train_score}')
            print(f'Validation Score: {val_score}')
            # print(f'Predicted Mean:{np.mean(unseen_pred)}')
            print('_'*50)

        fold += 1
        
    print("Mean MAPE:", np.mean(val_scores),"Std MAPE:",np.std(val_scores))
    return val_scores, models, unseen_preds, residuals, scaler


feature_set = [col for col in train.columns if col not in ['date', 'num_sold', 'gdp', 'log_gdp', 'year',
                                                           'day_of_week_ratios', 'country_sales_ratios',
                                                           'store_sales_ratios', 'product_sales_ratios',
                                                           'num_sold_adjusted']]

val_scores, models, unseen_preds, residuals, scaler = linear_model_trainer(train,
                                                                           test,
                                                                           feature_set,
                                                                           'num_sold',
                                                                           mean_absolute_percentage_error,
                                                                           log=True)

avg_unseen_preds = np.mean(unseen_preds, axis=0)


# plt.figure(figsize=(20, 8))

# country = 'Canada'
# store = 'Discount Stickers'
# product = 'Kaggle Tiers'

# deneme = train[(train[country] > 0) &
#                  (train[store] > 0) &
#                  (train[product] > 0)]

# preds = []
# for model in models:
#     pred = model.predict(deneme[feature_set])
#     preds.append(np.exp(pred + train.iloc[deneme.index]['log_gdp']))

# avg_pred = np.mean(preds, axis=0)

# real = deneme['num_sold'].values 

# sns.scatterplot(x=deneme['date'], y=np.exp(real), label='real', color='red')
# sns.lineplot(x=deneme['date'], y=avg_pred, label=f'Prediction')


model = Lasso(tol=1e-2, max_iter=1000000, random_state=0)

x, y = train[feature_set], train['num_sold']

model.fit(x, y)

final_prediction = model.predict(test[feature_set])
test['num_sold'] = np.exp(final_prediction)


submission = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])
submission = submission.merge(test[['date', 'num_sold']], on='date', how='left')

submission['year'] = submission['date'].dt.year
submission['day_of_week'] = submission['date'].dt.dayofweek

countries = submission['country'].unique().tolist()
test_country_sales_ratios = get_country_gdp_ratio(get_gdp_data(countries, 2017, 2019))
test_country_sales_ratios['year'] = test_country_sales_ratios['date'].dt.year
test_country_sales_ratios = test_country_sales_ratios.groupby(['year', 'country'])['gdp'].mean().reset_index()

submission = submission.merge(test_country_sales_ratios, on=['year', 'country'], how='inner')
submission = submission.merge(store_sales_ratios, on=['store'], how='inner')
submission = submission.merge(test_product_sales_ratios, on=['date', 'product'], how='inner')
submission = submission.merge(day_of_week_sale_ratios, on=['day_of_week'], how='left')
submission['num_sold'] = submission['num_sold'] * submission['gdp'] * submission['store_sales_ratios'] * submission['product_sales_ratios'] * submission['day_of_week_ratios']

# submission = submission.drop_duplicates(subset=['id'])
submission.head()


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sample_submission['num_sold'] = np.round(submission['num_sold'])
sample_submission.to_csv('aggregated_timeseries.csv', index=False)
sample_submission


def catboost_trainer(train_df,
                     test_df,
                     model_params,
                     feature_set,
                     target,
                     scorer,
                     cat_features,
                     log=True):
    val_scores = []
    unseen_preds = []
    val_preds = []
    train_preds = []
    models = []
    fold = 1

    cv = KFold(n_splits=5, shuffle=True)
    # tscv = TimeSeriesSplit(n_splits=5)
    
    for train_idx, val_idx in cv.split(train_df):
        X_train, y_train = train_df.iloc[train_idx][feature_set], train_df.iloc[train_idx][target]
        X_val, y_val = train_df.iloc[val_idx][feature_set], train_df.iloc[val_idx][target]

        # y_train -= train_df.iloc[train_idx]['log_gdp']
        # y_val -= train_df.iloc[val_idx]['log_gdp']

        model = CatBoostRegressor(**model_params)
    
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=200,
                  cat_features=cat_features)

        models.append(model)
        
        # Validation Pred
        val_pred = model.predict(X_val)
        if apply_transformation:
            val_pred = np.exp(val_pred) # + train_df.iloc[val_idx]['log_gdp'])

        # Training Preds
        train_pred = model.predict(X_train)
        if apply_transformation:
            train_pred = np.exp(train_pred) # + train_df.iloc[train_idx]['log_gdp'])

        val_preds.append(val_pred)
        train_preds.append(train_pred)

        # Unseen Test Pred
        unseen_pred = model.predict(test_df[feature_set])
        if apply_transformation:
            unseen_pred = np.exp(unseen_pred) # + test_df['log_gdp'])
        unseen_preds.append(unseen_pred)

        # Score
        if apply_transformation:
            y_train = np.exp(y_train) #  + train_df.iloc[train_idx]['log_gdp'])
            y_val = np.exp(y_val) #  + train_df.iloc[val_idx]['log_gdp'])

        
        train_score = scorer(y_train, train_pred)
        val_score = scorer(y_val, val_pred)

        val_scores.append(val_score)

        if log:   
            print(f'\nFOLD {fold}')
            print(f'Train Score: {train_score}')
            print(f'Validation Score: {val_score}')
            # print(f'Predicted Mean:{np.mean(unseen_pred)}')
            print('_'*50)

        fold += 1
        
    print("Mean MAPE:", np.mean(val_scores),"Std MAPE:",np.std(val_scores))
    return val_scores, models, unseen_preds


params = {
    'loss_function': 'MAPE',
    'iterations': 10000,
    'learning_rate': 0.05,
    'l2_leaf_reg': 4.75,
    'depth': 8,
    'eval_metric': 'MAPE',
    'random_state': 42,
    'allow_writing_files': False,
    'use_best_model': True,
    'early_stopping_rounds': 500,
    # 'task_type': 'GPU'
}

feature_set = [col for col in train.columns if col not in ['date', 'num_sold', 'gdp', 'log_gdp', 'year',
                                                           'day_of_week_ratios', 'country_sales_ratios',
                                                           'store_sales_ratios', 'product_sales_ratios',
                                                           'num_sold_adjusted', 'year_even']]


val_scores, models, unseen_preds = catboost_trainer(train,
                                                   test,
                                                   params,
                                                   feature_set,
                                                   'num_sold',
                                                   mean_absolute_percentage_error,
                                                   # ['country', 'store', 'product', 'holiday_flag'],
                                                    [],
                                                   log=True)

avg_unseen_preds = np.mean(unseen_preds, axis=0)


importances = [model.get_feature_importance() for model in models]
importances = np.mean(importances, axis=0)

importance_df = pd.DataFrame(index=feature_set,
                            data=importances,
                            columns=['Importance']).sort_values(by='Importance', ascending=False).reset_index()

importance_df = importance_df[:20]

sns.barplot(y='index', x='Importance', data=importance_df)


# plt.figure(figsize=(20, 8))

# country = 'Canada'
# store = 'Discount Stickers'
# product = 'Kaggle Tiers'

# deneme = train[(train[country] > 0) &
#                  (train[store] > 0) &
#                  (train[product] > 0)]

# preds = []
# for model in models:
#     pred = model.predict(deneme[feature_set])
#     preds.append(np.exp(pred + train.iloc[deneme.index]['log_gdp']))

# avg_pred = np.mean(preds, axis=0)

# real = deneme['num_sold'].values 

# sns.scatterplot(x=deneme['date'], y=np.exp(real), label='real', color='red')
# sns.lineplot(x=deneme['date'], y=avg_pred, label=f'Prediction')


test['num_sold'] = avg_unseen_preds


submission = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])
submission = submission.merge(test[['date', 'num_sold']], on='date', how='left')

submission['year'] = submission['date'].dt.year
submission['day_of_week'] = submission['date'].dt.dayofweek

countries = submission['country'].unique().tolist()
test_country_sales_ratios = get_country_gdp_ratio(get_gdp_data(countries, 2017, 2019))
test_country_sales_ratios['year'] = test_country_sales_ratios['date'].dt.year
test_country_sales_ratios = test_country_sales_ratios.groupby(['year', 'country'])['gdp'].mean().reset_index()

submission = submission.merge(test_country_sales_ratios, on=['year', 'country'], how='inner')
submission = submission.merge(store_sales_ratios, on=['store'], how='inner')
submission = submission.merge(test_product_sales_ratios, on=['date', 'product'], how='inner')
submission = submission.merge(day_of_week_sale_ratios, on=['day_of_week'], how='left')
submission['num_sold'] = submission['num_sold'] * submission['gdp'] * submission['store_sales_ratios'] * submission['product_sales_ratios'] * submission['day_of_week_ratios']

# submission = submission.drop_duplicates(subset=['id'])
submission.head()


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sample_submission['num_sold'] = np.round(submission['num_sold'])
sample_submission.to_csv('catboost_aggregated_timeseries.csv', index=False)
sample_submission




