import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=["date"])
original_train_df = train_df.copy()
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=["date"])


gdp_per_capita_df = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv")

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



gdp_per_capita_filtered_ratios_df_2["year"] = gdp_per_capita_filtered_ratios_df_2["year"].dt.year


train_df_imputed = train_df.copy()
missing_value_ids = train_df.loc[train_df["num_sold"].isna(), "id"].values
print(f"Missing values remaining: {train_df_imputed['num_sold'].isna().sum()}")

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


store_weights = train_df_imputed.groupby("store")["num_sold"].sum()/train_df_imputed["num_sold"].sum()
store_weights


product_df = train_df_imputed.groupby(["date","product"])["num_sold"].sum().reset_index()
product_ratio_df = product_df.pivot(index="date", columns="product", values="num_sold")
product_ratio_df = product_ratio_df.apply(lambda x: x/x.sum(),axis=1)
product_ratio_df = product_ratio_df.stack().rename("ratios").reset_index()
product_ratio_df.head()


train_df_imputed


original_train_df_imputed = train_df_imputed.copy()
train_df_imputed = train_df_imputed.groupby(["date"])["num_sold"].sum().reset_index()


train_df_imputed


train_df_imputed["day_of_week"] = train_df_imputed["date"].dt.dayofweek
day_of_week_ratio = (train_df_imputed.groupby("day_of_week")["num_sold"].mean() / train_df_imputed.groupby("day_of_week")["num_sold"].mean().mean()).rename("day_of_week_ratios")
display(day_of_week_ratio)
train_df_imputed = pd.merge(train_df_imputed, day_of_week_ratio, how="left", on="day_of_week")
train_df_imputed["num_sold"] = train_df_imputed["num_sold"] / train_df_imputed["day_of_week_ratios"]
train_df_imputed = train_df_imputed.drop("day_of_week_ratios",axis=1)# we don't need it in training.


test_total_sales_df = test_df.groupby(["date"])["id"].first().reset_index().drop(columns="id")
test_total_sales_dates = test_total_sales_df[["date"]]


def feature_engineer(df):
    new_df = df.copy()
    new_df["month"] = df["date"].dt.month
    new_df["month_sin"] = np.sin(new_df['month'] * (2 * np.pi / 12))
    new_df["month_cos"] = np.cos(new_df['month'] * (2 * np.pi / 12))
    new_df["day_of_week"] = df["date"].dt.dayofweek
    new_df["day_of_week"] = new_df["day_of_week"].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3))))    
    new_df["day_of_year"] = df['date'].apply(
        lambda x: x.timetuple().tm_yday if not (x.is_leap_year and x.month > 2) else x.timetuple().tm_yday - 1
    )

    new_df['day_sin4'] = np.sin(new_df['day_of_year'] * (8 * np.pi /  365.0))
    new_df['day_cos4'] = np.cos(new_df['day_of_year'] * (8 * np.pi /  365.0))
    new_df['day_sin3'] = np.sin(new_df['day_of_year'] * (6 * np.pi /  365.0))
    new_df['day_cos3'] = np.cos(new_df['day_of_year'] * (6 * np.pi /  365.0))
    new_df['day_sin2'] = np.sin(new_df['day_of_year'] * (4 * np.pi /  365.0))
    new_df['day_cos2'] = np.cos(new_df['day_of_year'] * (4 * np.pi /  365.0))
    new_df['day_sin'] = np.sin(new_df['day_of_year'] * (2 * np.pi /  365.0))
    new_df['day_cos'] = np.cos(new_df['day_of_year'] * (2 * np.pi /  365.0)) 
    new_df['day_sin_0.5'] = np.sin(new_df['day_of_year'] * (1 * np.pi /  365.0))
    new_df['day_cos_0.5'] = np.cos(new_df['day_of_year'] * (1 * np.pi /  365.0))    
    new_df["important_dates"] = new_df["day_of_year"].apply(lambda x: x if x in [1,2,3,4,5,6,7,8,9,10,99, 100, 101, 125,126,355,256,357,358,359,360,361,362,363,364,365] else 0)
    
    new_df = new_df.drop(columns=["date","month","day_of_year"])
    new_df = pd.get_dummies(new_df, columns = ["important_dates","day_of_week"], drop_first=True)
    
    return new_df

train_total_sales_df = feature_engineer(train_df_imputed)
test_total_sales_df = feature_engineer(test_total_sales_df)


import holidays
train_df_tmp = train_df.copy()
test_df_tmp = test_df.copy()
alpha2 = dict(zip(np.sort(train_df.country.unique()), ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']))
h = {c: holidays.country_holidays(a, years=range(2010, 2020)) for c, a in alpha2.items()}
train_df_tmp['is_holiday'] = 0
test_df_tmp['is_holiday'] = 0
for c in alpha2:
    train_df_tmp.loc[train_df_tmp.country==c, 'is_holiday'] = train_df_tmp.date.isin(h[c]).astype(int)
    test_df_tmp.loc[test_df_tmp.country==c, 'is_holiday'] = test_df_tmp.date.isin(h[c]).astype(int)


train_total_sales_df['is_holiday'] = (train_df_tmp.groupby(["date"])["is_holiday"].sum().reset_index())["is_holiday"]
test_total_sales_df['is_holiday'] = (test_df_tmp.groupby(["date"])["is_holiday"].sum().reset_index())["is_holiday"]


train_total_sales_df.info()


cat_features = []

bool_cat_cols = [
    "important_dates_1", "important_dates_2", "important_dates_3", "important_dates_4",
    "important_dates_5", "important_dates_6", "important_dates_7", "important_dates_8",
    "important_dates_9", "important_dates_10", "important_dates_99", "important_dates_100",
    "important_dates_101", "important_dates_125", "important_dates_126", "important_dates_256",
    "important_dates_355", "important_dates_357", "important_dates_358", "important_dates_359",
    "important_dates_360", "important_dates_361", "important_dates_362", "important_dates_363",
    "important_dates_364", "important_dates_365",
    "day_of_week_1", "day_of_week_2", "day_of_week_3",
    "is_holiday", "month_sin", "month_cos", "day_sin4", "day_cos4",
    "day_sin3", "day_cos3", "day_sin2", "day_cos2",
    "day_sin", "day_cos", "day_sin_0.5", "day_cos_0.5"
]

cat_features.extend(bool_cat_cols)
print("Updated cat_features:", cat_features)


y = train_total_sales_df["num_sold"]
X = train_total_sales_df.drop(columns="num_sold")
X_test = test_total_sales_df


#y_log = np.log1p(y)


"""
#max_iter=1000000，0.0701264047555749
#alpha=0.1, max_iter=10000，0.0701190665329048
model = Lasso(tol=1e-2, alpha=0.1, max_iter=10000, random_state=42)
model.fit(X, y)
#model.fit(X, y_log)
preds = model.predict(X_test)
#preds = np.expm1(model.predict(X_test))
test_total_sales_dates["num_sold"] = preds
mean_absolute_percentage_error(y,model.predict(X))
"""


lasso_model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
lasso_model.fit(X, y)
train_preds = lasso_model.predict(X)
preds = lasso_model.predict(X_test)
test_total_sales_dates["num_sold"] = preds
mean_absolute_percentage_error(y, train_preds)


product_ratio_2017_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2015].copy()
product_ratio_2018_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2016].copy()
product_ratio_2019_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2015].copy()

product_ratio_2017_df["date"] = product_ratio_2017_df["date"] + pd.DateOffset(years=2)
product_ratio_2018_df["date"] = product_ratio_2018_df["date"] + pd.DateOffset(years=2)
product_ratio_2019_df["date"] =  product_ratio_2019_df["date"] + pd.DateOffset(years=4)

forecasted_ratios_df = pd.concat([product_ratio_2017_df, product_ratio_2018_df, product_ratio_2019_df])


store_weights_df = store_weights.reset_index()
test_sub_df = pd.merge(test_df, test_total_sales_dates, how="left", on="date")
test_sub_df = test_sub_df.rename(columns = {"num_sold":"day_num_sold"})
# Adding in the product ratios
test_sub_df = pd.merge(test_sub_df, store_weights_df, how="left", on="store")
test_sub_df = test_sub_df.rename(columns = {"num_sold":"store_ratio"})
# Adding in the country ratios
test_sub_df["year"] = test_sub_df["date"].dt.year
test_sub_df = pd.merge(test_sub_df, gdp_per_capita_filtered_ratios_df_2, how="left", on=["year", "country"])
test_sub_df = test_sub_df.rename(columns = {"ratio":"country_ratio"})
# Adding in the product ratio
test_sub_df = pd.merge(test_sub_df, forecasted_ratios_df, how="left", on=["date", "product"])
test_sub_df = test_sub_df.rename(columns = {"ratios":"product_ratio"})

# Adding in the week ratio
test_sub_df["day_of_week"] = test_sub_df["date"].dt.dayofweek
test_sub_df = pd.merge(test_sub_df, day_of_week_ratio.reset_index(), how="left", on="day_of_week")


# Disaggregating the forecast
test_sub_df.loc[test_sub_df['country'] == 'Kenya', 'country_ratio'] -= 0.0007/2

test_sub_df["num_sold"] = test_sub_df["day_num_sold"] *test_sub_df["day_of_week_ratios"]* test_sub_df["store_ratio"] * test_sub_df["country_ratio"] * test_sub_df["product_ratio"]
test_sub_df["num_sold"] = test_sub_df["num_sold"].round()
display(test_sub_df.head(2))


submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
submission["num_sold"] = test_sub_df["num_sold"]
display(submission.head())


submission.to_csv('submission.csv', index = False)

