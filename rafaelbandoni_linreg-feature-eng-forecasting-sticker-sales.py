import random
random.seed(42)


import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import holidays
import optuna

from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

from lightgbm import LGBMRegressor


sns.set_theme()


sample = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])


sample.head()


test.head()


train.head()


train['num_sold'].isna().sum() / len(train['num_sold'])


for col in train.columns:
    n = train[col].isna().sum() / len(train[col])
    print(f'{col} null: {n.round(3)}')

print('\n####\n')
for col in test.columns:
    n = test[col].isna().sum() / len(test[col])
    print(f'{col} null: {n.round(3)}')


for col in train.columns:
    n = len(train[col].unique())
    print(f'{col} nunique: {n}')
print('\n####\n')
for col in test.columns:
    n = len(test[col].unique())
    print(f'{col} nunique: {n}')


plt.hist(train['num_sold'], bins=20)
plt.show()


plt.hist(np.log1p(train['num_sold']), bins=20)
plt.show()


plt.figure(figsize=(12,8))
sns.lineplot(
    data=train[~train['num_sold'].isna()].groupby(['date', 'country'], as_index=False).sum(),
    x='date',
    y='num_sold',
    hue='country'
)
plt.ylabel('Total sales')
plt.show()


country_ratio_over_time = (train.groupby(["date","country"])["num_sold"].sum() / train.groupby(["date"])["num_sold"].sum()).reset_index()
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = country_ratio_over_time, x="date", y="num_sold", hue="country");
ax.set_ylabel("Proportion of sales");


monthly_df = train.groupby(["country","store", "product", pd.Grouper(key="date", freq="MS")])["num_sold"].sum().rename("num_sold").reset_index()

f,axes = plt.subplots(3,2,figsize=(25,25))
                      # , sharex = True, sharey=True)
f.tight_layout()
for n,prod in enumerate(monthly_df["product"].unique()):
    plot_df = monthly_df.loc[monthly_df["product"] == prod]
    sns.lineplot(data=plot_df, x="date", y="num_sold", hue="country", style="store",ax=axes[n//2,n%2])
    axes[n//2,n%2].set_title("Product: "+str(prod))


categorical = ['country', 'store', 'product']
df = train[train['num_sold'].isna()]
for item in categorical:
    to_plot = df.groupby(item, as_index=False).count()
    sns.barplot(
        data=to_plot[to_plot['id'] > 0],
        x=item,
        y='id'
    )
    plt.title(item)
    plt.show()


counts = train.groupby(["country","store","product"])["num_sold"].count().rename("num_rows")
missing_data = counts.loc[counts != train.groupby(["country","store","product"])["num_sold"].count().rename("num_rows").max()]

f,axs = plt.subplots(9,1, figsize=(20,50))
for i, (country, store, product) in enumerate(missing_data.index):
    plot_df = train.loc[(train["country"] == country) & (train["store"] == store) & (train["product"] == product)]
    missing_vals = plot_df.loc[plot_df["num_sold"].isna()]
    sns.lineplot(
        data=plot_df[~plot_df['num_sold'].isna()],
        x="date",
        y="num_sold",
        ax=axs[i]
    )
    for missing_date in missing_vals["date"]:
        axs[i].axvline(missing_date, color='red',  linestyle='-', linewidth=1, alpha=0.2)
    axs[i].set_title(f"{country} - {store} - {product}")


train[train['num_sold'].isna()][['country', 'store', 'product']].drop_duplicates()


gdp_per_capita_df = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv")

years =  ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
gdp_per_capita_filtered_df = gdp_per_capita_df.loc[gdp_per_capita_df["Country Name"].isin(train["country"].unique()), ["Country Name"] + years].set_index("Country Name")
gdp_data = gdp_per_capita_filtered_df.copy()

gdp_per_capita_filtered_df["2010_ratio"] = gdp_per_capita_filtered_df["2010"] / gdp_per_capita_filtered_df.sum()["2010"]

for year in years:
    gdp_per_capita_filtered_df[f"{year}_ratio"] = gdp_per_capita_filtered_df[year] / gdp_per_capita_filtered_df.sum()[year]
gdp_per_capita_filtered_ratios_df = gdp_per_capita_filtered_df[[i+"_ratio" for i in years]]
gdp_per_capita_filtered_ratios_df.columns = [int(i) for i in years]
gdp_per_capita_filtered_ratios_df = gdp_per_capita_filtered_ratios_df.unstack().reset_index().rename(columns = {"level_0": "year", 0: "ratio", "Country Name": "country"})
gdp_per_capita_filtered_ratios_df['year'] = pd.to_datetime(gdp_per_capita_filtered_ratios_df['year'], format='%Y')

gdp_per_capita_filtered_ratios_df_2 = gdp_per_capita_filtered_ratios_df.copy()
gdp_per_capita_filtered_ratios_df_2["year"] = pd.to_datetime(gdp_per_capita_filtered_ratios_df_2['year'].astype(str)) + pd.offsets.YearEnd(1)
gdp_per_capita_filtered_ratios_df = pd.concat([gdp_per_capita_filtered_ratios_df, gdp_per_capita_filtered_ratios_df_2]).reset_index()
gdp_per_capita_filtered_ratios_df_2["year"] = gdp_per_capita_filtered_ratios_df_2["year"].dt.year


gdp_data = gdp_data.unstack().reset_index().rename(columns = {"level_0": "year", 0: "gdp", "Country Name": "country"})
gdp_data['year'] = gdp_data['year'].astype('datetime64[ns]').dt.year


imputed_train = train.copy()
imputed_train["year"] = imputed_train["date"].dt.year
for year in imputed_train["year"].unique():
    # Impute Time Series 1 (Canada, Discount Stickers, Holographic Goose)
    target_ratio = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Norway"), "ratio"].values[0] # Using Norway as should have the best precision
    current_raito = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Canada"), "ratio"].values[0]
    ratio_can = current_raito / target_ratio
    imputed_train.loc[(imputed_train["country"] == "Canada") & (imputed_train["store"] == "Discount Stickers") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year), "num_sold"] = (imputed_train.loc[(imputed_train["country"] == "Norway") & (imputed_train["store"] == "Discount Stickers") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year), "num_sold"] * ratio_can).values
    
    # Impute Time Series 2 (Only Missing Values)
    current_ts =  imputed_train.loc[(imputed_train["country"] == "Canada") & (imputed_train["store"] == "Premium Sticker Mart") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    imputed_train.loc[(imputed_train["country"] == "Canada") & (imputed_train["store"] == "Premium Sticker Mart") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year) & (imputed_train["date"].isin(missing_ts_dates)), "num_sold"] = (imputed_train.loc[(imputed_train["country"] == "Norway") & (imputed_train["store"] == "Premium Sticker Mart") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year) & (imputed_train["date"].isin(missing_ts_dates)), "num_sold"] * ratio_can).values

    # Impute Time Series 3 (Only Missing Values)
    current_ts =  imputed_train.loc[(imputed_train["country"] == "Canada") & (imputed_train["store"] == "Stickers for Less") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    imputed_train.loc[(imputed_train["country"] == "Canada") & (imputed_train["store"] == "Stickers for Less") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year) & (imputed_train["date"].isin(missing_ts_dates)), "num_sold"] = (imputed_train.loc[(imputed_train["country"] == "Norway") & (imputed_train["store"] == "Stickers for Less") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year) & (imputed_train["date"].isin(missing_ts_dates)), "num_sold"] * ratio_can).values
    
    # Impute Time Series 4 (Kenya, Discount Stickers, Holographic Goose)
    current_raito = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Kenya"), "ratio"].values[0]
    ratio_ken = current_raito / target_ratio
    imputed_train.loc[(imputed_train["country"] == "Kenya") & (imputed_train["store"] == "Discount Stickers") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year), "num_sold"] = (imputed_train.loc[(imputed_train["country"] == "Norway") & (imputed_train["store"] == "Discount Stickers") & (imputed_train["product"] == "Holographic Goose")& (imputed_train["year"] == year), "num_sold"] * ratio_ken).values

    # Impute Time Series 5 (Only Missing Values)
    current_ts = imputed_train.loc[(imputed_train["country"] == "Kenya") & (imputed_train["store"] == "Premium Sticker Mart") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    imputed_train.loc[(imputed_train["country"] == "Kenya") & (imputed_train["store"] == "Premium Sticker Mart") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year) & (imputed_train["date"].isin(missing_ts_dates)), "num_sold"] = (imputed_train.loc[(imputed_train["country"] == "Norway") & (imputed_train["store"] == "Premium Sticker Mart") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year) & (imputed_train["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values

    # Impute Time Series 6 (Only Missing Values)
    current_ts = imputed_train.loc[(imputed_train["country"] == "Kenya") & (imputed_train["store"] == "Stickers for Less") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    imputed_train.loc[(imputed_train["country"] == "Kenya") & (imputed_train["store"] == "Stickers for Less") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year) & (imputed_train["date"].isin(missing_ts_dates)), "num_sold"] = (imputed_train.loc[(imputed_train["country"] == "Norway") & (imputed_train["store"] == "Stickers for Less") & (imputed_train["product"] == "Holographic Goose") & (imputed_train["year"] == year) & (imputed_train["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values

    # Impute Time Series 7 (Only Missing Values)
    current_ts = imputed_train.loc[(imputed_train["country"] == "Kenya") & (imputed_train["store"] == "Discount Stickers") & (imputed_train["product"] == "Kerneler") & (imputed_train["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    imputed_train.loc[(imputed_train["country"] == "Kenya") & (imputed_train["store"] == "Discount Stickers") & (imputed_train["product"] == "Kerneler") & (imputed_train["year"] == year) & (imputed_train["date"].isin(missing_ts_dates)), "num_sold"] = (imputed_train.loc[(imputed_train["country"] == "Norway") & (imputed_train["store"] == "Discount Stickers") & (imputed_train["product"] == "Kerneler") & (imputed_train["year"] == year) & (imputed_train["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values


missing_rows = imputed_train.loc[imputed_train["num_sold"].isna()]
imputed_train.loc[imputed_train["id"] == 23719, "num_sold"] = 4
imputed_train.loc[imputed_train["id"] == 207003, "num_sold"] = 195


imputed_train[imputed_train['num_sold'].isna()][['country', 'store', 'product']].drop_duplicates()


def add_holiday_feature(df):
    country_holidays = {
        'Canada': holidays.CountryHoliday('CA'),
        'Finland': holidays.CountryHoliday('FI'),
        'Italy': holidays.CountryHoliday('IT'),
        'Kenya': holidays.CountryHoliday('KE'),
        'Norway': holidays.CountryHoliday('NO'),
        'Singapore': holidays.CountryHoliday('SG')
    }

    df['date'] = pd.to_datetime(df['date'])

    df['is_holiday'] = df.apply(
        lambda row: row['date'] in country_holidays.get(row['country'], []), axis=1
    )
    return df


def make_sales_ratios(train_df):
    ratio_df = train_df.copy()
    ratio_df['date'] = ratio_df['date'].astype('datetime64[ns]')
    ratio_df_to_test = ratio_df.copy()
    ratio_df_to_test = ratio_df_to_test[ratio_df_to_test['date'] >= '2014-01-01']
    ratio_df_to_test['date'] = ratio_df_to_test['date'] + pd.offsets.DateOffset(years=3)
    ratio_df = pd.concat([ratio_df, ratio_df_to_test])
    
    ratio_df['year'] = ratio_df['date'].dt.year
    ratio_df['month'] = ratio_df['date'].dt.month
    
    product_ratio = pd.DataFrame(ratio_df.groupby(['year', 'month', 'product'])['num_sold'].sum() / ratio_df.groupby(['year', 'month'])['num_sold'].sum()).reset_index().rename(columns={'num_sold' : 'product_ratio'})
    store_ratio = pd.DataFrame(ratio_df.groupby(['year', 'month', 'store'])['num_sold'].sum() / ratio_df.groupby(['year', 'month'])['num_sold'].sum()).reset_index().rename(columns={'num_sold' : 'store_ratio'})
    country_ratio = pd.DataFrame(ratio_df.groupby(['year', 'month', 'country'])['num_sold'].sum() / ratio_df.groupby(['year', 'month'])['num_sold'].sum()).reset_index().rename(columns={'num_sold' : 'country_ratio'})
    return product_ratio, store_ratio, country_ratio


def make_gdp_ratios(gdp_data):
    gdp_ratios = gdp_data.copy()
    gdp_ratios = pd.DataFrame(gdp_ratios.groupby(['year', 'country'])['gdp'].sum() / gdp_ratios.groupby('year')['gdp'].sum()).reset_index().rename(columns={'gdp' : 'gdp_ratio'})
    return gdp_ratios


def feature_engineering(train, to_predict):
    to_predict = to_predict.copy()
    train_df = train.copy()
    product_sales_ratio, store_sales_ratio, country_sales_ratio = make_sales_ratios(train)
    gdp_ratios = make_gdp_ratios(gdp_data)

    train_df = add_holiday_feature(train_df)
    to_predict = add_holiday_feature(to_predict)

    for df in [train_df, to_predict]:
        df['date'] = df['date'].astype('datetime64[ns]')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week   
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df["day_of_year"] = df['date'].apply(
            lambda x: x.timetuple().tm_yday if not (x.is_leap_year and x.month > 2) else x.timetuple().tm_yday - 1
        )
        df.drop('date', axis=1, inplace=True)

        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30.0)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4.0)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4.0)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
        df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52.0)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52.0)

        df['day_sin1'] = np.sin(0.5 * np.pi * df['day'] / 30.0)
        df['day_cos1'] = np.cos(0.5 * np.pi * df['day'] / 30.0)
        df['day_of_year_sin1'] = np.sin(0.5 * np.pi * df['day_of_year'] / 365.0)
        df['day_of_year_cos1'] = np.cos(0.5 * np.pi * df['day_of_year'] / 365.0)
        df['month_sin1'] = np.sin(0.5 * np.pi * df['month'] / 12.0)
        df['month_cos1'] = np.cos(0.5 * np.pi * df['month'] / 12.0)
        df['quarter_sin1'] = np.sin(0.5 * np.pi * df['quarter'] / 4.0)
        df['quarter_cos1'] = np.cos(0.5 * np.pi * df['quarter'] / 4.0)
        df['day_of_week_sin1'] = np.sin(0.5 * np.pi * df['day_of_week'] / 7.0)
        df['day_of_week_cos1'] = np.cos(0.5 * np.pi * df['day_of_week'] / 7.0)
        df['week_of_year_sin1'] = np.sin(0.5 * np.pi * df['week_of_year'] / 52.0)
        df['week_of_year_cos1'] = np.cos(0.5 * np.pi * df['week_of_year'] / 52.0)

        df['day_sin2'] = np.sin(4 * np.pi * df['day'] / 30.0)
        df['day_cos2'] = np.cos(4 * np.pi * df['day'] / 30.0)
        df['day_of_year_sin2'] = np.sin(4 * np.pi * df['day_of_year'] / 365.0)
        df['day_of_year_cos2'] = np.cos(4 * np.pi * df['day_of_year'] / 365.0)
        df['month_sin2'] = np.sin(4 * np.pi * df['month'] / 12.0)
        df['month_cos2'] = np.cos(4 * np.pi * df['month'] / 12.0)
        df['quarter_sin2'] = np.sin(4 * np.pi * df['quarter'] / 4.0)
        df['quarter_cos2'] = np.cos(4 * np.pi * df['quarter'] / 4.0)
        df['day_of_week_sin2'] = np.sin(4 * np.pi * df['day_of_week'] / 7.0)
        df['day_of_week_cos2'] = np.cos(4 * np.pi * df['day_of_week'] / 7.0)
        df['week_of_year_sin2'] = np.sin(4 * np.pi * df['week_of_year'] / 52.0)
        df['week_of_year_cos2'] = np.cos(4 * np.pi * df['week_of_year'] / 52.0)

        df['day_sin3'] = np.sin(6 * np.pi * df['day'] / 30.0)
        df['day_cos3'] = np.cos(6 * np.pi * df['day'] / 30.0)
        df['day_of_year_sin3'] = np.sin(6 * np.pi * df['day_of_year'] / 365.0)
        df['day_of_year_cos3'] = np.cos(6 * np.pi * df['day_of_year'] / 365.0)
        df['month_sin3'] = np.sin(6 * np.pi * df['month'] / 12.0)
        df['month_cos3'] = np.cos(6 * np.pi * df['month'] / 12.0)
        df['quarter_sin3'] = np.sin(6 * np.pi * df['quarter'] / 4.0)
        df['quarter_cos3'] = np.cos(6 * np.pi * df['quarter'] / 4.0)
        df['day_of_week_sin3'] = np.sin(6 * np.pi * df['day_of_week'] / 7.0)
        df['day_of_week_cos3'] = np.cos(6 * np.pi * df['day_of_week'] / 7.0)
        df['week_of_year_sin3'] = np.sin(6 * np.pi * df['week_of_year'] / 52.0)
        df['week_of_year_cos3'] = np.cos(6 * np.pi * df['week_of_year'] / 52.0)

        df['day_sin4'] = np.sin(8 * np.pi * df['day'] / 30.0)
        df['day_cos4'] = np.cos(8 * np.pi * df['day'] / 30.0)
        df['day_of_year_sin4'] = np.sin(8 * np.pi * df['day_of_year'] / 365.0)
        df['day_of_year_cos4'] = np.cos(8 * np.pi * df['day_of_year'] / 365.0)
        df['month_sin4'] = np.sin(8 * np.pi * df['month'] / 12.0)
        df['month_cos4'] = np.cos(8 * np.pi * df['month'] / 12.0)
        df['quarter_sin4'] = np.sin(8 * np.pi * df['quarter'] / 4.0)
        df['quarter_cos4'] = np.cos(8 * np.pi * df['quarter'] / 4.0)
        df['day_of_week_sin4'] = np.sin(8 * np.pi * df['day_of_week'] / 7.0)
        df['day_of_week_cos4'] = np.cos(8 * np.pi * df['day_of_week'] / 7.0)
        df['week_of_year_sin4'] = np.sin(8 * np.pi * df['week_of_year'] / 52.0)
        df['week_of_year_cos4'] = np.cos(8 * np.pi * df['week_of_year'] / 52.0)
        
        df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7

        df['month_country'] = df['month'].astype(str) + "_" + df['country']
        df['month_store'] = df['month'].astype(str) + "_" + df['store']
        df['month_product'] = df['month'].astype(str) + "_" + df['product']

        df['country_store'] = df['country'] + "_" + df['store']
        df['country_product'] = df['country'] + "_" + df['product']
        df['store_product'] = df['store'] + "_" + df['product']

        df['month_country_store'] = df['month'].astype(str) + "_" + df['country'] + "_" + df['store']
        df['month_country_product'] = df['month'].astype(str) + "_" + df['country'] + "_" + df['product']
        df['month_store_product'] = df['month'].astype(str) + "_" + df['store'] + "_" + df['product']
        df['country_store_product'] = df['country'] + "_" + df['store'] + "_" + df['product']
        

    train_df = train_df.merge(product_sales_ratio, on=['year', 'month', 'product'], how='left')
    train_df = train_df.merge(store_sales_ratio, on=['year', 'month', 'store'], how='left')
    train_df = train_df.merge(country_sales_ratio, on=['year', 'month', 'country'], how='left')

    to_predict = to_predict.merge(product_sales_ratio, on=['year', 'month', 'product'], how='left')
    to_predict = to_predict.merge(store_sales_ratio, on=['year', 'month', 'store'], how='left')
    to_predict = to_predict.merge(country_sales_ratio, on=['year', 'month', 'country'], how='left')
    
    train_df = train_df.merge(gdp_data, on=['country', 'year'], how='left')
    to_predict = to_predict.merge(gdp_data, on=['country', 'year'], how='left')

    train_df = train_df.merge(gdp_ratios, on=['country', 'year'], how='left')
    to_predict = to_predict.merge(gdp_ratios, on=['country', 'year'], how='left')

    cat_columns = [
        'country', 'store', 'product',
        'month_country', 'month_store','month_product',
        'country_store', 'country_product', 'store_product',
        'month_country_store', 'month_country_product', 'month_store_product',
        'country_store_product'
    ]
    
    train_df = pd.get_dummies(train_df, columns=cat_columns, drop_first=True)
    to_predict = pd.get_dummies(to_predict, columns=cat_columns, drop_first=True)
    
    train_df['num_sold'] = np.log1p(train_df['num_sold'])
    return train_df, to_predict


train_df, to_predict = feature_engineering(imputed_train, test)


train_df.head()


to_predict.head()


train_df.drop('id', axis=1, inplace=True)
target = 'num_sold'

X_train = train_df[train_df['year'] < 2016].drop(target, axis=1)
y_train = train_df[train_df['year'] < 2016][target]

X_test = train_df[train_df['year'] == 2016].drop(target, axis=1)
y_test = train_df[train_df['year'] == 2016][target]


def objective(trial):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mape',  # We'll evaluate on MAPE
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        # 'random_state': 42,
        'verbose': -1,
        'device': 'gpu'
    }
    
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(y_pred))
    return mape

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best parameters and MAPE
print("Best parameters:", study.best_params)
print("Best MAPE:", study.best_value)


X = train_df.drop(columns=['num_sold'])
y = train_df['num_sold']

tscv = TimeSeriesSplit(n_splits=7)

lgb_params = study.best_params
lgb_params.update({
    'device': 'gpu',                # Use GPU for training
    'n_jobs': -1,                   # Use all available CPU threads
})
model = LGBMRegressor(**lgb_params)

for train_index, val_index in tscv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    
    mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(predictions))
    print(f'MAPE: {mape}')


y_to_submit = np.expm1(model.predict(to_predict.drop('id', axis=1)))


to_submit = pd.DataFrame(
    data={
        'id' : to_predict['id'],
        'num_sold' : y_to_submit
    }
)
to_submit.head()


to_submit.to_csv('submission.csv', index=False)

