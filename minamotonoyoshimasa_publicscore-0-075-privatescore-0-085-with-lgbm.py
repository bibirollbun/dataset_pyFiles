!pip install optuna


pip install seaborn==0.13.2


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


 display(train.head())
 display(test.head())


def valueCount_plot(df, column_name):
  ValueCounts = df[column_name].value_counts().reset_index().set_index(column_name)
  ValueCounts['Percentage'] = df[column_name].value_counts(normalize=True) * 100
  plt.figure(figsize=(10,5))
  colors = sns.color_palette('husl')
  sns.barplot(x=ValueCounts.index, y=ValueCounts['count'], palette = colors)
  plt.xticks(rotation=0)
  plt.xlabel(column_name)
  plt.ylabel('Count')
  plt.title('Value Counts')
  for i, v in enumerate(ValueCounts['count']):
      percentage = ValueCounts['Percentage'][i]
      plt.text(i, v, f"{v}({percentage:.2f}%)", ha='center', va='bottom')
  plt.show()


valueCount_plot(train, 'country')


valueCount_plot(train, 'store')


valueCount_plot(train, 'product')


counts = train.groupby(["country","store","product"])["id"].count().rename("num_rows").reset_index()
counts


counts_val_counts = counts["num_rows"].value_counts().rename("Count").reset_index().rename(columns={"index": "length"})
display(counts_val_counts.head(10))


print(f"Number of missing values: {train.isnull().sum()}") 


counts = train.groupby(["country","store","product"])["num_sold"].count().rename("num_rows")
Missing_data = counts.loc[counts!= 2557]
Missing_data_df = Missing_data.reset_index()
Missing_data_df['missing_rows'] = 2557 - Missing_data_df['num_rows']
Missing_data_df


f, axs = plt.subplots(9,1, figsize = (20,50))
for i, (country, store, product) in enumerate(Missing_data.index):
  plot_df = train.loc[(train['country'] == country) & (train['store'] == store) & (train['product'] == product)]
  missing_df = plot_df.loc[plot_df['num_sold'].isna()]
  sns.lineplot(data = plot_df, x = 'date', y = 'num_sold', ax = axs[i])
  for missing_data in missing_df['date']:
    axs[i].axvline(x = missing_data, color = 'red', linestyle = '-')
    axs[i].set_title(f"Country: {country}, Store: {store}, Product: {product}")


gdp_per_capita_df = pd.read_csv('/kaggle/input/gdp-per-capita/gdp_per_capita.csv')
gdp_per_capita_df.head()


train_df = train.copy()
country_weights = train_df.groupby("country")["num_sold"].sum()/train_df["num_sold"].sum()

country_ratio_over_time = (train_df.groupby(["date","country"])["num_sold"].sum() / train_df.groupby(["date"])["num_sold"].sum()).reset_index()
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = country_ratio_over_time, x="date", y="num_sold", hue="country");
ax.set_ylabel("Proportion of sales");


years =  ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
gdp_per_capita_filtered_df = gdp_per_capita_df.loc[gdp_per_capita_df["Country Name"].isin(train_df["country"].unique()), ["Country Name"] + years].set_index("Country Name")
gdp_per_capita_filtered_df["2010_ratio"] = gdp_per_capita_filtered_df["2010"] / gdp_per_capita_filtered_df.sum()["2010"]
for year in years:
    gdp_per_capita_filtered_df[f"{year}_ratio"] = gdp_per_capita_filtered_df[year] / gdp_per_capita_filtered_df.sum()[year]
gdp_per_capita_filtered_ratios_df = gdp_per_capita_filtered_df[[i+"_ratio" for i in years]]
gdp_per_capita_filtered_ratios_df.columns = [int(i) for i in years]
gdp_per_capita_filtered_ratios_df = gdp_per_capita_filtered_ratios_df.unstack().reset_index().rename(columns = {"level_0": "year", 0: "ratio", "Country Name": "country"})
gdp_per_capita_filtered_ratios_df['year'] = pd.to_datetime(gdp_per_capita_filtered_ratios_df['year'], format='%Y')
# For plotting purposes
gdp_per_capita_filtered_ratios_df_2 = gdp_per_capita_filtered_ratios_df.copy()
gdp_per_capita_filtered_ratios_df_2["year"] = pd.to_datetime(gdp_per_capita_filtered_ratios_df_2['year'].astype(str)) + pd.offsets.YearEnd(1)
gdp_per_capita_filtered_ratios_df = pd.concat([gdp_per_capita_filtered_ratios_df, gdp_per_capita_filtered_ratios_df_2]).reset_index()

country_ratio_over_time['date'] = pd.to_datetime(country_ratio_over_time['date'])
f,ax = plt.subplots(figsize=(20,15))
sns.lineplot(data = country_ratio_over_time, x="date", y="num_sold", hue="country");
sns.lineplot(data = gdp_per_capita_filtered_ratios_df, x="year", y = "ratio", hue="country", palette = ["black"]*6, legend = False)
ax.set_ylabel("Proportion of sales");


gdp_per_capita_filtered_ratios_df_2["year"] = gdp_per_capita_filtered_ratios_df_2["year"].dt.year


train_df_imputed = train_df.copy()
print(f"Missing values remaining: {train_df_imputed['num_sold'].isna().sum()}")

train_df_imputed["date"] = pd.to_datetime(train_df_imputed["date"])
train_df_imputed["year"] = train_df_imputed["date"].dt.year
for year in train_df_imputed["year"].unique():
    # Impute Time Series 1 (Canada, Discount Stickers, Holographic Goose)
    target_ratio = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Norway"), "ratio"].values[0] # Using Norway as should have the best precision
    current_raito = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Canada"), "ratio"].values[0]
    ratio_can = current_raito / target_ratio
    train_df_imputed.loc[(train_df_imputed["country"] == "Canada") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year), "num_sold"] * ratio_can).values
    # Impute Time Series 2 (Only Missing Values)
    current_ts =  train_df_imputed.loc[(train_df_imputed["country"] == "Canada") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[(train_df_imputed["country"] == "Canada") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_can).values
    # Impute Time Series 3 (Only Missing Values)
    current_ts =  train_df_imputed.loc[(train_df_imputed["country"] == "Canada") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[(train_df_imputed["country"] == "Canada") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_can).values
    # Impute Time Series 4 (Kenya, Discount Stickers, Holographic Goose)
    current_raito = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Kenya"), "ratio"].values[0]
    ratio_ken = current_raito / target_ratio
    train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Holographic Goose")& (train_df_imputed["year"] == year), "num_sold"] * ratio_ken).values
    # Impute Time Series 5 (Only Missing Values)
    current_ts = train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Premium Sticker Mart") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values
    # Impute Time Series 6 (Only Missing Values)
    current_ts = train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Stickers for Less") & (train_df_imputed["product"] == "Holographic Goose") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values
    # Impute Time Series 7 (Only Missing Values)
    current_ts = train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Kerneler") & (train_df_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[(train_df_imputed["country"] == "Kenya") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Kerneler") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_df_imputed.loc[(train_df_imputed["country"] == "Norway") & (train_df_imputed["store"] == "Discount Stickers") & (train_df_imputed["product"] == "Kerneler") & (train_df_imputed["year"] == year) & (train_df_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values

print(f"Missing values remaining: {train_df_imputed['num_sold'].isna().sum()}")



missing_rows = train_df_imputed.loc[train_df_imputed["num_sold"].isna()]
display(missing_rows)
train_df_imputed.loc[train_df_imputed["id"] == 23719, "num_sold"] = 4
train_df_imputed.loc[train_df_imputed["id"] == 207003, "num_sold"] = 195

print(f"Missing values remaining: {train_df_imputed['num_sold'].isna().sum()}")


store_ratio_over_time = (train_df_imputed.groupby(["date","store"])["num_sold"].sum() / train_df_imputed.groupby(["date"])["num_sold"].sum()).reset_index()
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = store_ratio_over_time, x="date", y="num_sold", hue="store");
ax.set_ylabel("Proportion of sales");


# Update monthly_df with our imputed data:
weekly_df = train_df_imputed.groupby(["country","store", "product", pd.Grouper(key="date", freq="W")])["num_sold"].sum().rename("num_sold").reset_index()
monthly_df = train_df_imputed.groupby(["country","store", "product", pd.Grouper(key="date", freq="MS")])["num_sold"].sum().rename("num_sold").reset_index()


product_df = train_df_imputed.groupby(["date","product"])["num_sold"].sum().reset_index()
product_ratio_df = product_df.pivot(index="date", columns="product", values="num_sold")
product_ratio_df = product_ratio_df.apply(lambda x: x/x.sum(),axis=1)
product_ratio_df = product_ratio_df.stack().rename("ratios").reset_index()
product_ratio_2017_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2015].copy()
product_ratio_2018_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2016].copy()
product_ratio_2019_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2015].copy()

product_ratio_2017_df["date"] = product_ratio_2017_df["date"] + pd.DateOffset(years=2)
product_ratio_2018_df["date"] = product_ratio_2018_df["date"] + pd.DateOffset(years=2)
product_ratio_2019_df["date"] =  product_ratio_2019_df["date"] + pd.DateOffset(years=4)

forecasted_ratios_df = pd.concat([product_ratio_2017_df, product_ratio_2018_df, product_ratio_2019_df])
temp_df = pd.concat([product_ratio_df,forecasted_ratios_df]).reset_index(drop=True)
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data=temp_df, x="date", y="ratios", hue="product");
ax.axvline(pd.to_datetime("2017-01-01"), color='black', linestyle='--');


train_copy = train_df_imputed.copy()
train_copy_year = train_copy.groupby([pd.Grouper(key = "date", freq = "Y")])["num_sold"].sum().reset_index()
f, ax = plt.subplots(figsize = (10,6))
sns.lineplot(data = train_copy_year, x = "date", y = "num_sold");


train_copy_month = train_copy.groupby([pd.Grouper(key = "date", freq = "MS")])["num_sold"].sum().reset_index()
f, ax = plt.subplots(figsize = (10,6))
sns.lineplot(data = train_copy_month, x = "date", y = "num_sold");


train_copy_week = train_copy.groupby([pd.Grouper(key = "date", freq = "W")])["num_sold"].sum().reset_index()
f, ax = plt.subplots(figsize = (10,6))
sns.lineplot(data = train_copy_week[1:-1], x = "date", y = "num_sold");


train_copy_day = train_copy.groupby([pd.Grouper(key = "date", freq = "D")])["num_sold"].sum().reset_index()
f, ax = plt.subplots(figsize = (10,6))
sns.lineplot(data = train_copy_day, x = "date", y = "num_sold");


train_seasonality = train_df_imputed.copy()
train_seasonality = train_seasonality.groupby("date")["num_sold"].sum().reset_index()
train_seasonality['month'] = train_seasonality['date'].dt.month
f, ax = plt.subplots(figsize = (10, 6))
sns.lineplot(data = train_seasonality, x = "month", y = "num_sold")
ax.set_title("Month Seasonality")


train_seasonality['day_of_week'] = train_seasonality['date'].dt.dayofweek
f, ax = plt.subplots(figsize = (10, 6))
sns.lineplot(data = train_seasonality, x = "day_of_week", y = "num_sold")
ax.set_title("day_of_week Seasonality")


train_seasonality['day_of_month'] = train_seasonality['date'].dt.day
f, ax = plt.subplots(figsize = (10, 6))
sns.lineplot(data = train_seasonality, x = "day_of_month", y = "num_sold")
ax.set_title("day_of_month Seasonality")


train_seasonality['day_of_year'] = train_seasonality['date'].dt.dayofyear
f, ax = plt.subplots(figsize = (10, 6))
sns.lineplot(data = train_seasonality, x = "day_of_year", y = "num_sold")
ax.set_title("day_of_year Seasonality")


from statsmodels.tsa.seasonal import seasonal_decompose
for periods in [30, 90, 180, 360]:
  decomposition = seasonal_decompose(train_seasonality['num_sold'], model='multiplicative', period=periods)

  plt.figure(figsize=(20, 5))
  decomposition.seasonal.plot()
  plt.title(f'Seasonal Component for {periods} period')
  plt.show()


import holidays

# Mapping full country names to ISO country codes
country_mapping = {
    'Finland': 'FI',
    'Canada': 'CA',
    'Italy': 'IT',
    'Kenya': 'KE',
    'Singapore': 'SG',
    'Norway': 'NO'
}

# Function to check if a date is a holiday in a given country
def is_holiday(row):
    country_code = country_mapping.get(row['country'])  # Convert country name to ISO code
    if country_code:
        country_holidays = holidays.country_holidays(country_code)
        return row['date'] in country_holidays
    return False  # Return False if the country is not mapped



def feature_engineer(df):
    new_df = df.copy()
    new_df["quarter"] = new_df["date"].dt.quarter
    new_df["month"] = new_df["date"].dt.month
    new_df["month_sin"] = np.sin(new_df['month'] * (2 * np.pi / 12))
    new_df["month_cos"] = np.cos(new_df['month'] * (2 * np.pi / 12))
    new_df["day_of_week"] = df["date"].dt.dayofweek
    new_df["day_of_week"] = new_df["day_of_week"].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3))))
    new_df["day_of_month"] = new_df["date"].dt.day
    new_df["day_of_month_sin_.5"] = np.sin(new_df['day_of_month'] * (1 * np.pi / new_df['date'].dt.daysinmonth))
    new_df["day_of_month_cos_.5"] = np.cos(new_df['day_of_month'] * (1 * np.pi / new_df['date'].dt.daysinmonth))
    new_df["day_of_month_sin"] = np.sin(new_df['day_of_month'] * (2 * np.pi / new_df['date'].dt.daysinmonth))
    new_df["day_of_month_cos"] = np.cos(new_df['day_of_month'] * (2 * np.pi / new_df['date'].dt.daysinmonth))
    new_df["day_of_month_sin_2"] = np.sin(new_df['day_of_month'] * (4 * np.pi / new_df['date'].dt.daysinmonth))
    new_df["day_of_month_cos_2"] = np.cos(new_df['day_of_month'] * (4 * np.pi / new_df['date'].dt.daysinmonth))
    new_df["day_of_month_sin_3"] = np.sin(new_df['day_of_month'] * (6 * np.pi / new_df['date'].dt.daysinmonth))
    new_df["day_of_month_cos_3"] = np.cos(new_df['day_of_month'] * (6 * np.pi / new_df['date'].dt.daysinmonth))
    new_df["day_of_month_sin_4"] = np.sin(new_df['day_of_month'] * (8 * np.pi / new_df['date'].dt.daysinmonth))
    new_df["day_of_month_cos_4"] = np.cos(new_df['day_of_month'] * (8 * np.pi / new_df['date'].dt.daysinmonth))

    new_df["day_of_year"] = df['date'].apply(
        lambda x: x.timetuple().tm_yday if not (x.is_leap_year and x.month > 2) else x.timetuple().tm_yday - 1
    )
    new_df['day_sin_.5'] = np.sin(new_df['day_of_year'] * (1 * np.pi /  365.0))
    new_df['day_cos_.5'] = np.cos(new_df['day_of_year'] * (1 * np.pi /  365.0))
    new_df['day_sin_1'] = np.sin(new_df['day_of_year'] * (2 * np.pi /  365.0))
    new_df['day_cos_1'] = np.cos(new_df['day_of_year'] * (2 * np.pi /  365.0))
    new_df['day_sin_2'] = np.sin(new_df['day_of_year'] * (4 * np.pi /  365.0))
    new_df['day_cos_2'] = np.cos(new_df['day_of_year'] * (4 * np.pi /  365.0))
    new_df['day_sin_3'] = np.sin(new_df['day_of_year'] * (6 * np.pi /  365.0))
    new_df['day_cos_3'] = np.cos(new_df['day_of_year'] * (6 * np.pi /  365.0))
    new_df['day_sin_4'] = np.sin(new_df['day_of_year'] * (8 * np.pi /  365.0))
    new_df['day_cos_4'] = np.cos(new_df['day_of_year'] * (8 * np.pi /  365.0))

    new_df["year"] = (df["date"].dt.year - 2010)/6
    new_df['holiday'] = new_df.apply(is_holiday, axis=1)
    new_df['end_of_month'] = new_df['date'].dt.day >= 25
    new_df['end_of_year'] = new_df['date'].apply(lambda x: (x.month == 12 and x.day >= 24) or (x.month == 1 and x.day <= 5))


    new_df = new_df.drop(columns=["date","day_of_year", "id", "month", "day_of_month", "day_of_year"])
    return new_df


country_weights = train_df_imputed.groupby("country")["num_sold"].sum()/train_df_imputed["num_sold"].sum()
country_ratio_overtime = (train_df_imputed.groupby(["date","country"])["num_sold"].sum() / train_df_imputed.groupby(["date"])["num_sold"].sum()).reset_index()
product_weights = train_df_imputed.groupby("product")["num_sold"].sum()/train_df_imputed["num_sold"].sum()
product_ratio_overtime = (train_df_imputed.groupby(["date","product"])["num_sold"].sum() / train_df_imputed.groupby(["date"])["num_sold"].sum()).reset_index()


train_df_added = train_df_imputed.copy()
train_df_added['store_ratio'] = None
train_df_added.set_index(['store'], inplace=True)
store_ratio_over_time.set_index(['store'], inplace=True)
train_df_added['store_ratio'] = store_ratio_over_time.groupby(['store'])['num_sold'].mean()
train_df_added.reset_index(inplace=True)
store_ratio_over_time.reset_index(inplace=True)

train_df_added['country_ratio'] = None
train_df_added['product_ratio'] = None
gdp_per_capita_filtered_ratios_df_train = gdp_per_capita_filtered_ratios_df.copy()
gdp_per_capita_filtered_ratios_df_train['year'] = gdp_per_capita_filtered_ratios_df['year'].dt.year
train_df_added.set_index(['year', 'country'], inplace=True)
gdp_per_capita_filtered_ratios_df_train.set_index(['year', 'country'], inplace=True)
gdp_per_capita_filtered_ratios_df_train = gdp_per_capita_filtered_ratios_df_train.reset_index().drop_duplicates().set_index(['year', 'country'])
train_df_added['country_ratio'].update(gdp_per_capita_filtered_ratios_df_train['ratio'])
train_df_added.reset_index(inplace=True)
gdp_per_capita_filtered_ratios_df_train.reset_index(inplace=True)

train_df_added.set_index(['date', 'product'], inplace=True)
product_ratio_overtime.set_index(['date', 'product'], inplace=True)
train_df_added['product_ratio'].update(product_ratio_overtime['num_sold'])
train_df_added.reset_index(inplace=True)
product_ratio_overtime.reset_index(inplace=True)


train_total_sales_df = feature_engineer(train_df_added)
train_total_sales_df['country_ratio'] = train_total_sales_df['country_ratio'].astype(float)
train_total_sales_df['product_ratio'] = train_total_sales_df['product_ratio'].astype(float)
train_total_sales_df['store_ratio'] = train_total_sales_df['store_ratio'].astype(float)
train_total_sales_df['country'] = train_total_sales_df['country'].astype('category')
train_total_sales_df['product'] = train_total_sales_df['product'].astype('category')
train_total_sales_df['store'] = train_total_sales_df['store'].astype('category')



train_total_sales_df.head()


train_df_x = train_total_sales_df.drop(columns = ['num_sold'])
train_df_y = train_total_sales_df['num_sold']


train_df_x.columns


from scipy.stats import skew
from scipy.stats import kurtosis
target_skewness = skew(train_total_sales_df['num_sold'])
target_kurtosis = kurtosis(train_total_sales_df['num_sold'])
print(f"Skewness of the target variable: {target_skewness}")
print(f"kurtosis of the target variable: {target_kurtosis}")


# Split the dataset into 70% training and 30% validation while maintaining order
split_index = int(len(train_total_sales_df) * 0.7)  # Calculate 70% index

# Split data in order (not randomly)
train_df_x, vali_df_x = train_df_x.iloc[:split_index], train_df_x.iloc[split_index:]
train_df_y, vali_df_y = train_df_y.iloc[:split_index], train_df_y.iloc[split_index:]


train_df_y_norm = np.sqrt(train_df_y)
target_skewness = skew(train_df_y_norm)
target_kurtosis = kurtosis(train_df_y_norm)
print(f"Skewness of the target variable: {target_skewness}")
print(f"kurtosis of the target variable: {target_kurtosis}")




# Function to optimize
def objective(trial):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mape',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'random_state': 42,
        'verbose': -1,
        'device': 'cpu'
    }

    # Train LightGBM model
    model = lgb.LGBMRegressor(**params)
    score = cross_val_score(model, train_df_x, train_df_y_norm, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_absolute_percentage_error')

    return -score.mean()  # Optuna minimizes the function, so negate MAPE

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Get best parameters
best_params = study.best_params  # Best hyperparameters found
best_score = study.best_value  # Best score (lowest MAPE)

print("Best Parameters:", best_params)
print(f"Best MAPE: {best_score:.4f}")



lgb_params = study.best_params

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(train_df_x, train_df_y_norm)
vali_df_result = lgb_model.predict(vali_df_x)
mape_score = mean_absolute_percentage_error(vali_df_y, np.square(vali_df_result))
print(f"MAPE: {mape_score}")



lgb.plot_importance(lgb_model, max_num_features=10)


# Merge features with the target for correlation analysis
df_corr = pd.concat([train_df_x, train_df_y_norm], axis=1)

df_corr_encoded = df_corr.copy()

# Convert categorical columns to numeric using label encoding
for col in df_corr.select_dtypes(include=['object', 'category']).columns:
    df_corr_encoded[col] = df_corr_encoded[col].astype('category').cat.codes  # Convert to category codes

# Compute correlation matrix
corr_matrix = df_corr_encoded.corr()
corr_with_target = corr_matrix.iloc[-1].sort_values(ascending=False)
print(corr_with_target)


train_df_x['country_ratio'] = np.log1p(train_df_x['country_ratio'])
train_df_x['product_ratio'] = np.log1p(train_df_x['product_ratio'])
train_df_x['store_country_interaction'] = train_df_x['store_ratio'] * train_df_x['country_ratio']
train_df_x['product_country_interaction'] = train_df_x['product_ratio'] * train_df_x['country_ratio']
train_df_x['product_store_interaction'] = train_df_x['product_ratio'] * train_df_x['store_ratio']
vali_df_x['country_ratio'] = np.log1p(vali_df_x['country_ratio'])
vali_df_x['product_ratio'] = np.log1p(vali_df_x['product_ratio'])
vali_df_x['store_country_interaction'] = vali_df_x['store_ratio'] * vali_df_x['country_ratio']
vali_df_x['product_country_interaction'] = vali_df_x['product_ratio'] * vali_df_x['country_ratio']
vali_df_x['product_store_interaction'] = vali_df_x['product_ratio'] * vali_df_x['store_ratio']


drop_features_1 = ["quarter", "day_of_month_sin_.5", "day_sin_2", "day_of_month_sin_2",
                 "day_of_month_cos_3", "day_of_month_cos_2", "day_of_month_cos_4",
                 "day_sin_4", "day_of_month_cos_.5", "end_of_month", "day_of_month_sin",
                 "day_cos_4", "day_cos_.5", "day_cos_2"]

train_df_x = train_df_x.drop(columns=drop_features_1)
vali_df_x = vali_df_x.drop(columns=drop_features_1)


# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Get best parameters
best_params = study.best_params  # Best hyperparameters found
best_score = study.best_value  # Best score (lowest MAPE)

print("Best Parameters:", best_params)
print(f"Best MAPE: {best_score:.4f}")


lgb_params = study.best_params

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(train_df_x, train_df_y_norm)
vali_df_result = lgb_model.predict(vali_df_x)
mape_score = mean_absolute_percentage_error(vali_df_y, np.square(vali_df_result))
print(f"MAPE: {mape_score}")


test['date'] = pd.to_datetime(test['date'])
test_df_added = test.copy()
test_df_added['store_ratio'] = None
test_df_added.set_index(['store'], inplace=True)
store_ratio_over_time.set_index(['store'], inplace=True)
test_df_added['store_ratio'] = store_ratio_over_time.groupby(['store'])['num_sold'].mean()
test_df_added.reset_index(inplace=True)
store_ratio_over_time.reset_index(inplace=True)

gdp_per_capita_filtered_ratios_df_test = gdp_per_capita_filtered_ratios_df.copy()
test_df_added['year'] = test_df_added['date'].dt.year
test_df_added['country_ratio'] = None
gdp_per_capita_filtered_ratios_df_test['year'] = gdp_per_capita_filtered_ratios_df['year'].dt.year
test_df_added.set_index(['year', 'country'], inplace=True)
gdp_per_capita_filtered_ratios_df_test.set_index(['year', 'country'], inplace=True)
gdp_per_capita_filtered_ratios_df_test = gdp_per_capita_filtered_ratios_df_test.reset_index().drop_duplicates().set_index(['year', 'country'])
test_df_added['country_ratio'].update(gdp_per_capita_filtered_ratios_df_test['ratio'])
test_df_added.reset_index(inplace=True)
gdp_per_capita_filtered_ratios_df_test.reset_index(inplace=True)

test_df_added['product_ratio'] = None
temp_df_test = temp_df.copy()
test_df_added.set_index(['date', 'product'], inplace=True)
temp_df_test.set_index(['date', 'product'], inplace=True)
temp_df_test = temp_df_test.reset_index().drop_duplicates(subset=['date', 'product']).set_index(['date', 'product'])
test_df_added['product_ratio'].update(temp_df_test['ratios'])
test_df_added.reset_index(inplace=True)
temp_df_test.reset_index(inplace=True)


test_df = feature_engineer(test_df_added)
test_df['product_ratio'] = test_df['product_ratio'].astype(float)
test_df['country_ratio'] = test_df['country_ratio'].astype(float)
test_df['store_ratio'] = test_df['store_ratio'].astype(float)
test_df['country'] = test_df['country'].astype('category')
test_df['product'] = test_df['product'].astype('category')
test_df['store'] = test_df['store'].astype('category')
test_df['country_ratio'] = np.log1p(test_df['country_ratio'])
test_df['product_ratio'] = np.log1p(test_df['product_ratio'])
test_df['store_country_interaction'] = test_df['store_ratio'] * test_df['country_ratio']
test_df['product_country_interaction'] = test_df['product_ratio'] * test_df['country_ratio']
test_df['product_store_interaction'] = test_df['product_ratio'] * test_df['store_ratio']

drop_features_1 = ["quarter", "day_of_month_sin_.5", "day_sin_2", "day_of_month_sin_2",
                 "day_of_month_cos_3", "day_of_month_cos_2", "day_of_month_cos_4",
                 "day_sin_4", "day_of_month_cos_.5", "end_of_month", "day_of_month_sin",
                 "day_cos_4", "day_cos_.5", "day_cos_2"]

test_df = test_df.drop(columns=drop_features_1)

test_df.head()


test_df_result = np.square(lgb_model.predict(test_df))


test_df_result = pd.DataFrame({'id': range(230130, 230130 + len(test_df_result)), 'num_id': test_df_result})


test_df_result = test_df_result.round()


test_df_result.to_csv('/kaggle/working/submission.csv', index=False)


test_df_result.head()

