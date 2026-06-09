import numpy as np
import pandas as pd
import re
from itertools import chain
from joblib import Parallel, delayed
from scipy.special import inv_boxcox
from scipy.stats import boxcox, norm as stats_norm
from sklearn.metrics import r2_score
from sklearn.neighbors import BallTree
from sklearn.preprocessing import PowerTransformer


# set path
project_path = "/kaggle/input/prediction-interval-competition-ii-house-price"
macro_path = "/kaggle/input/usa-macroeconomic-rate-of-changes-1993-2025"


# load dataset
df_train = pd.read_csv(project_path + "/dataset.csv")
print(df_train.shape)
df_train.head(1)


# load test
df_test = pd.read_csv(project_path + "/test.csv")
print(df_test.shape)
df_test.head(1)


# load macro
df_macro = pd.read_csv(macro_path + "/macro_monthly.csv")
df_macro.columns = [re.sub(r'[^a-z0-9_]', '', col.lower().replace(' ', '_')) for col in df_macro.columns]
print(df_macro.shape)
df_macro.head(1)


df_train.info()


# preprocess the dataframe
def preprocess(data):
    """ preprocess the data columns """
    data['sale_date'] = pd.to_datetime(data['sale_date'], format='mixed')
    data['sale_nbr'] = data['sale_nbr'].fillna(0).astype(int)
    data['year_reno'] = np.where(data['year_reno'] == 0, data['year_built'], data['year_reno'])
    data[['subdivision', 'submarket']] = data[['subdivision', 'submarket']].fillna("Unknown")
    return data

df_train = preprocess(df_train)
df_test = preprocess(df_test)
df_train.info()


# create engineering features
def engineering_features(data):
    """ enrich the dataframe to create meaningful features """
    #### main feature engineering ####
    # years
    data['join_to_sale'] = data['join_year'] - data['sale_date'].dt.year
    data['sale_to_built'] = data['sale_date'].dt.year - data['year_built']
    data['sale_to_renovate'] = data['sale_date'].dt.year - data['year_reno']
    data['is_renovated'] = np.where(data['year_built'] == data['year_reno'], 0, 1)
    data['renovate_to_built'] = data['year_reno'] - data['year_built']
    # total area
    data['living_area'] = data['sqft'] + data["sqft_1"] + data['sqft_fbsmt']
    data["total_sqft"] = data["sqft_lot"] + data["sqft"] + data["sqft_1"] + data["sqft_fbsmt"]
    data["total_bath"] = data["bath_full"] + (3/4)*data["bath_3qtr"] + (1/2)*data["bath_half"]
    data["total_gar"] = data["garb_sqft"] + data["gara_sqft"]
    # multiplications
    data["val_mult"] = np.log1p(data["land_val"]) * np.log1p(data["imp_val"])
    data["val_add"] = data["land_val"] + data["imp_val"]
    for i in ["area", "living_area"]:
        data[f"{i}_land"] = np.log1p(data[i]) * np.log1p(data["land_val"])
        data[f"{i}_imp"] = np.log1p(data[i]) * np.log1p(data["imp_val"])
        data[f"{i}_val_mult"] = np.log1p(data[i]) * data["val_mult"]
        data[f"{i}_val_add"] = np.log1p(data[i]) * np.log1p(data["val_add"])
    # total view
    view_cols = [i for i in data.columns if i.startswith("view_")]
    data["total_view"] = data[view_cols].sum(axis=1)
    # high cardinality columns
    data["zoning_2c"] = data["zoning"].astype(str).str.slice(0, 2)
    data["subdivision_1w"] = data["subdivision"].astype(str).str.split().str[0]
    data["subdivision_2w"] = data["subdivision"].astype(str).str.split().str[:2].str.join(" ")
    # sale warning codes
    sale_warning_split = data["sale_warning"].astype(str).str.split()
    unique_codes = sorted(set(chain.from_iterable(sale_warning_split)))
    unique_codes = [code for code in unique_codes if code.strip().isdigit()]
    for code in unique_codes:
        data[f"sale_warning_{code}"] = sale_warning_split.apply(lambda lst: int(code in lst))
    #### other feature engineering ####
    # time
    data = engineer_date_features(data)
    # macro
    data = data.merge(df_macro, on=["year", "month"], how="left")
    # knn sale prices
    data = run_parallel_knn_features(data, k_values=[250, 1000, 2500], n_jobs=3)
    # fillna
    data = data.fillna(data.mean(numeric_only=True))
    return data

def run_parallel_knn_features(df, k_values, n_jobs=-1):
    # combine all features
    results = Parallel(n_jobs=n_jobs)(delayed(knn_known_price_features)(df, k=k) for k in k_values)
    combined = pd.concat(results, axis=1)
    df = pd.concat([df.reset_index(drop=True), combined.reset_index(drop=True)], axis=1)
    return df

def knn_known_price_features(df, k=5):
    df = df.copy()
    df_coords_rad = np.radians(df[["latitude", "longitude"]].values)

    # create BallTree on only rows with sale_price
    known_mask = df["sale_price"].notnull()
    known_coords = np.radians(df.loc[known_mask, ["latitude", "longitude"]].values)
    known_price_indexes = df.loc[known_mask, "sale_price"].values / df.loc[known_mask, "living_area"].values

    tree = BallTree(known_coords, metric='haversine')
    dist, ind = tree.query(df_coords_rad, k=k+1)  # +1 because first is self

    # get neighbor sale price indexes
    knn_price_indexes = np.array([known_price_indexes[i[1:]] for i in ind])  # exclude self

    # compute aggregate statistics
    df[f'knn_{k}_mean_price_index'] = knn_price_indexes.mean(axis=1)
    df[f'knn_{k}_std_price_index'] = knn_price_indexes.std(axis=1)
    return df[[f'knn_{k}_mean_price_index', f'knn_{k}_std_price_index']]

def engineer_date_features(df):
    # extract date components
    df['year'] = df["sale_date"].dt.year
    df['month'] = df["sale_date"].dt.month
    df['day'] = df["sale_date"].dt.day
    df['dayofweek'] = df["sale_date"].dt.dayofweek
    df['weekofyear'] = df["sale_date"].dt.isocalendar().week
    df['quarter'] = df["sale_date"].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # months since first sale
    first_sale_month = df['sale_date'].dt.to_period('M').min()
    df['since_first_sale'] = (df['sale_date'].dt.to_period('M') - first_sale_month).apply(lambda x: x.n)

    # season
    def get_season(month):
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        else: return 'fall'
    df['season'] = df['month'].apply(get_season)

    # aggregation columns
    df['avg_price_per_year'] = df.groupby('year')["sale_price"].transform('mean')
    df['avg_price_per_month'] = df.groupby(['year', 'month'])["sale_price"].transform('mean')
    df['avg_price_per_month'] = df['avg_price_per_month'].fillna(df['avg_price_per_year'])

    return df

# apply
df_full = pd.concat([df_train, df_test], ignore_index=True)
df_full = engineering_features(df_full)
df_train, df_test = df_full.iloc[:200000], df_full.iloc[200000:]


df_train.info()

