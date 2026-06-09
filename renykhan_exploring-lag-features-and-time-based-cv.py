import pandas as pd
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Ridge

from tqdm import tqdm
from colorama import Fore, Style, init

import warnings
warnings.filterwarnings('ignore')


sns.set_style('darkgrid')


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates = ['date'])
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates = ['date'])

sample = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


def display_info(df, df_name):
    '''Displays head, info, describe, missing values of df'''
    for data, label in zip([df], [f'{df_name}']):
        print(Style.BRIGHT + Fore.BLUE + f'\n{label} head \n')
        print(f"length of {label} : ", len(data))
        print(f"start timestamp of {label} : ", df.date.min())
        print(f"End timestamp of {label} : ", df.date.max())
        
        print(Style.BRIGHT + Fore.GREEN + f'\n{label} head \n' + Style.RESET_ALL)
        display(data.head())

        print(Style.BRIGHT + Fore.BLUE + f'\n{label} info \n')
        display(data.info())

        print(Style.BRIGHT + Fore.GREEN + f'\n{label} describe \n' + Style.RESET_ALL)
        display(data.describe().T)

        print(Style.BRIGHT + Fore.BLUE + f'\n{label} missing values \n')
        display(data.isnull().sum())
        print("------------------------------------------------------------------")


# display_info(train, "Train")


uniques = {}
for col in ['country', 'store', 'product']:
    uniques[col] = train[col].unique()
    print(f'Unique values in {col} : ', uniques[col], " and number = ", len(uniques[col]))


country_weights = train.groupby("country")["num_sold"].sum()/train["num_sold"].sum()

country_ratio_over_time = (train.groupby(["date","country"])["num_sold"].sum() / train.groupby(["date"])["num_sold"].sum()).reset_index()
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = country_ratio_over_time, x="date", y="num_sold", hue="country");
ax.set_ylabel("Proportion of sales");


gdp_per_capita_df = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv")

years =  ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

gdp_per_capita_filtered_df = gdp_per_capita_df.loc[gdp_per_capita_df["Country Name"].isin(train["country"].unique()), ["Country Name"] + years].set_index("Country Name")
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


train_imputed = train.copy()
print(f"Missing values remaining: {train_imputed['num_sold'].isna().sum()}")

train_imputed["year"] = train_imputed["date"].dt.year
for year in train_imputed["year"].unique():
    # Impute Time Series 1 (Canada, Discount Stickers, Holographic Goose)
    target_ratio = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Norway"), "ratio"].values[0] # Using Norway as should have the best precision
    current_raito = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Canada"), "ratio"].values[0]
    ratio_can = current_raito / target_ratio
    train_imputed.loc[(train_imputed["country"] == "Canada") & (train_imputed["store"] == "Discount Stickers") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year), "num_sold"] = (train_imputed.loc[(train_imputed["country"] == "Norway") & (train_imputed["store"] == "Discount Stickers") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year), "num_sold"] * ratio_can).values
    
    # Impute Time Series 2 (Only Missing Values)
    current_ts =  train_imputed.loc[(train_imputed["country"] == "Canada") & (train_imputed["store"] == "Premium Sticker Mart") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_imputed.loc[(train_imputed["country"] == "Canada") & (train_imputed["store"] == "Premium Sticker Mart") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year) & (train_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_imputed.loc[(train_imputed["country"] == "Norway") & (train_imputed["store"] == "Premium Sticker Mart") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year) & (train_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_can).values

    # Impute Time Series 3 (Only Missing Values)
    current_ts =  train_imputed.loc[(train_imputed["country"] == "Canada") & (train_imputed["store"] == "Stickers for Less") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_imputed.loc[(train_imputed["country"] == "Canada") & (train_imputed["store"] == "Stickers for Less") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year) & (train_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_imputed.loc[(train_imputed["country"] == "Norway") & (train_imputed["store"] == "Stickers for Less") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year) & (train_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_can).values
    
    # Impute Time Series 4 (Kenya, Discount Stickers, Holographic Goose)
    current_raito = gdp_per_capita_filtered_ratios_df_2.loc[(gdp_per_capita_filtered_ratios_df_2["year"] == year) & (gdp_per_capita_filtered_ratios_df_2["country"] == "Kenya"), "ratio"].values[0]
    ratio_ken = current_raito / target_ratio
    train_imputed.loc[(train_imputed["country"] == "Kenya") & (train_imputed["store"] == "Discount Stickers") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year), "num_sold"] = (train_imputed.loc[(train_imputed["country"] == "Norway") & (train_imputed["store"] == "Discount Stickers") & (train_imputed["product"] == "Holographic Goose")& (train_imputed["year"] == year), "num_sold"] * ratio_ken).values

    # Impute Time Series 5 (Only Missing Values)
    current_ts = train_imputed.loc[(train_imputed["country"] == "Kenya") & (train_imputed["store"] == "Premium Sticker Mart") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_imputed.loc[(train_imputed["country"] == "Kenya") & (train_imputed["store"] == "Premium Sticker Mart") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year) & (train_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_imputed.loc[(train_imputed["country"] == "Norway") & (train_imputed["store"] == "Premium Sticker Mart") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year) & (train_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values

    # Impute Time Series 6 (Only Missing Values)
    current_ts = train_imputed.loc[(train_imputed["country"] == "Kenya") & (train_imputed["store"] == "Stickers for Less") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_imputed.loc[(train_imputed["country"] == "Kenya") & (train_imputed["store"] == "Stickers for Less") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year) & (train_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_imputed.loc[(train_imputed["country"] == "Norway") & (train_imputed["store"] == "Stickers for Less") & (train_imputed["product"] == "Holographic Goose") & (train_imputed["year"] == year) & (train_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values

    # Impute Time Series 7 (Only Missing Values)
    current_ts = train_imputed.loc[(train_imputed["country"] == "Kenya") & (train_imputed["store"] == "Discount Stickers") & (train_imputed["product"] == "Kerneler") & (train_imputed["year"] == year)]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_imputed.loc[(train_imputed["country"] == "Kenya") & (train_imputed["store"] == "Discount Stickers") & (train_imputed["product"] == "Kerneler") & (train_imputed["year"] == year) & (train_imputed["date"].isin(missing_ts_dates)), "num_sold"] = (train_imputed.loc[(train_imputed["country"] == "Norway") & (train_imputed["store"] == "Discount Stickers") & (train_imputed["product"] == "Kerneler") & (train_imputed["year"] == year) & (train_imputed["date"].isin(missing_ts_dates)), "num_sold"] * ratio_ken).values
    
print(f"Missing values remaining: {train_imputed['num_sold'].isna().sum()}")


# It seems a bit overkill to replace the entire timeseries for the remaining 2 missing values, 
# I'll just fill them in manually using the graphs from earlier:


missing_rows = train_imputed.loc[train_imputed["num_sold"].isna()]
display(missing_rows)
train_imputed.loc[train_imputed["id"] == 23719, "num_sold"] = 4
train_imputed.loc[train_imputed["id"] == 207003, "num_sold"] = 195

print(f"Missing values remaining: {train_imputed['num_sold'].isna().sum()}")


# Update monthly_df with our imputed data:
weekly_df = train_imputed.groupby(["country","store", "product", pd.Grouper(key="date", freq="W")])["num_sold"].sum().rename("num_sold").reset_index()
monthly_df = train_imputed.groupby(["country","store", "product", pd.Grouper(key="date", freq="MS")])["num_sold"].sum().rename("num_sold").reset_index()


store_weights = train_imputed.groupby("store")["num_sold"].sum()/train_imputed["num_sold"].sum()
store_ratio_over_time = (train_imputed.groupby(["date","store"])["num_sold"].sum() / train_imputed.groupby(["date"])["num_sold"].sum()).reset_index()


product_df = train_imputed.groupby(["date","product"])["num_sold"].sum().reset_index()


product_ratio_df = product_df.pivot(index="date", columns="product", values="num_sold")
product_ratio_df = product_ratio_df.apply(lambda x: x/x.sum(),axis=1)
product_ratio_df = product_ratio_df.stack().rename("ratios").reset_index()


original_train_imputed = train_imputed.copy()
train_imputed = train_imputed.groupby(["date"])["num_sold"].sum().reset_index()
# This is the time series we need to forecast


f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = train_imputed, x="date", y="num_sold");


weekly_df = train.groupby([pd.Grouper(key="date", freq="W")])["num_sold"].sum().rename("num_sold").reset_index()
monthly_df = train.groupby([pd.Grouper(key="date", freq="MS")])["num_sold"].sum().rename("num_sold").reset_index()
yearly_df = train.groupby([pd.Grouper(key="date", freq="YS")])["num_sold"].sum().rename("num_sold").reset_index()


train_imputed.tail()


explore_df = train_imputed.copy()


def lagplot(x, y=None, lag=1, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    y_ = y if y is not None else x
    
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    
    ax = sns.regplot(x=x_, y=y_, line_kws=dict(color='C3', ), lowess=True, ax=ax, **kwargs)
    at = AnchoredText(f"{corr:.2f}", prop=dict(size="large"), frameon=True,loc="upper left")
    
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
            
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    
    return fig

def add_lag(df : pd.DataFrame, col_name, lag_no):
    return pd.concat([
        df,
        pd.DataFrame({
            f'{col_name}_lag_{lag_no}' : df[col_name].shift(lag_no)
        })
    ], axis = 1)



from statsmodels.graphics.tsaplots import plot_pacf

_ = plot_lags(explore_df.num_sold, lags=12, nrows=2)
_ = plot_pacf(explore_df.num_sold, lags=12)


test_lag_df = pd.concat({
    'num_sold_lag_6' : explore_df.iloc[[i for i in range(2551, 2557)], 1],
    'num_sold_lag_7' : explore_df.iloc[[i for i in range(2550, 2557)], 1]
}, axis = 1)


test_lag_df = test_lag_df.reset_index()
test_lag_df.drop('index', axis = 1, inplace = True)



#get the dates to forecast for
test_total_sales_df = test.groupby(["date"])["id"].first().reset_index().drop(columns="id")
#keep dates for later
test_total_sales_dates = test_total_sales_df[["date"]]


test_total_sales_df = pd.concat([test_total_sales_df, test_lag_df], axis = 1)


# explore_df = explore_df.copy()
explore_df = add_lag(explore_df, "num_sold", 6)
explore_df = add_lag(explore_df, "num_sold", 7)


explore_df.dropna(inplace = True)
explore_df = explore_df.reset_index()


explore_df.drop(['index'], axis = 1, inplace = True)
explore_df


def feature_engineer(df):
    
    new_df = df.copy()
    new_df["month"] = df["date"].dt.month
    # new_df['year'] = df['date'].dt.year
    
    new_df["month_sin"] = np.sin(new_df['month'] * (2 * np.pi / 12))
    new_df["month_cos"] = np.cos(new_df['month'] * (2 * np.pi / 12))
    new_df["day_of_week"] = df["date"].dt.dayofweek
    new_df["day_of_week"] = new_df["day_of_week"].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3))))
    
    new_df["day_of_year"] = df['date'].apply(
        lambda x: x.timetuple().tm_yday if not (x.is_leap_year and x.month > 2) else x.timetuple().tm_yday - 1
    )
    new_df['day_sin'] = np.sin(new_df['day_of_year'] * (2 * np.pi /  365.0))
    new_df['day_cos'] = np.cos(new_df['day_of_year'] * (2 * np.pi /  365.0))

    #new_df['week_of_year'] = new_df['date'].dt.isocalendar().week
    new_df['is_month_end'] = new_df['date'].dt.is_month_end.astype(int)
    
    new_df["important_dates"] = new_df["day_of_year"].apply(lambda x: x if x in [1,2,3,4,5,6,7,8,9,10,99, 100, 101, 125,126,355,256,357,358,359,360,361,362,363,364,365] else 0)
    #new_df["year"] = df["date"].dt.year - 2010
    
    new_df = new_df.drop(columns=["date","month","day_of_year"])
    new_df = pd.get_dummies(new_df, columns = ["important_dates","day_of_week"], drop_first=True)
    
    return new_df


train_total_sales_df = feature_engineer(explore_df)
test_total_sales_df = feature_engineer(test_total_sales_df)


SEED = 42
target = 'num_sold'

n_splits = 10
n_repeats = 1

# OOF_preds = pd.DataFrame()
TEST_preds = pd.DataFrame()
scores_df = pd.DataFrame(columns = ['Score'])



X = train_total_sales_df.drop(target, axis = 1)
y = train_total_sales_df[target]
test_df = test_total_sales_df


def predict(model, test_df):
    test_df_copy = test_df.copy()
    
    n = len(test_df_copy)
    prediction = []
    
    for i in test_df_copy.index :
        pred_value = model.predict(test_df_copy.iloc[[i]])
        prediction.append(pred_value)
        if i + 6 < n :
            # test_df_copy.iloc[i+6, 'num_sold_lag_6'] = pred_value
            test_df_copy.iloc[i+6, test_df_copy.columns.get_loc('num_sold_lag_6')] = pred_value
        if i + 7 < n :
            # test_df_copy.iloc[i+7, 'num_sold_lag_7'] = pred_value
            test_df_copy.iloc[i+7, test_df_copy.columns.get_loc('num_sold_lag_7')] = pred_value

    # make prediction a np array
    prediction = np.array(prediction)

    return prediction


folds = TimeSeriesSplit(n_splits = n_splits)

model_name = "Ridge_baseline1"

for n_fold, (train_idx, val_idx) in enumerate(tqdm(folds.split(X, y), desc = "Training Folds", total = n_splits)): 
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # y_train_log = np.log1p(y_train)
    # y_val_log = np.log1p(y_val)

    # oof_preds = pd.DataFrame(columns = [model_name], index = X_val.index)
    test_preds = pd.DataFrame(columns = [model_name], index = test_df.index)

    model = Ridge(tol=1e-2, max_iter=1000000, random_state=SEED)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    test_pred = predict(model, test_df)

    # y_train_pred = np.expm1(y_train_log_pred)
    # y_val_pred = np.expm1(y_val_log_pred)
    # test_pred = np.expm1(test_log_pred)

    # oof_preds[model_name] = y_val_pred
    test_preds[model_name] = test_pred

    train_score = mean_absolute_percentage_error(y_train, y_train_pred)
    val_score = mean_absolute_percentage_error(y_val, y_val_pred)

    print(f"Fold {n_fold+1} - Train MAPE: {train_score:.4f}, Validation MAPE: {val_score:.4f}")

    scores_df.loc[f'{model_name}', f'{n_fold + 1}'] = val_score
    # OOF_preds = pd.concat([OOF_preds, oof_preds], axis = 0, ignore_index = False)
    TEST_preds = pd.concat([TEST_preds, test_preds], axis = 0, ignore_index = False)

# OOF_preds = OOF_preds.groupby(level = 0).mean()
TEST_preds = TEST_preds.groupby(level = 0).mean()

scores_df.loc[f'{model_name}', 'Score'] = scores_df.loc[f'{model_name}'][1:].mean()
scores_df.sort_values('Score')

    


TEST_preds.head(10)


test_total_sales_dates["num_sold"] = TEST_preds 


test_total_sales_dates.head()


f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = pd.concat([train_imputed,test_total_sales_dates]).reset_index(drop=True), x="date", y="num_sold", linewidth=0.6);


product_ratio_2017_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2015].copy()
product_ratio_2018_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2016].copy()
product_ratio_2019_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2015].copy()

product_ratio_2017_df["date"] = product_ratio_2017_df["date"] + pd.DateOffset(years=2)
product_ratio_2018_df["date"] = product_ratio_2018_df["date"] + pd.DateOffset(years=2)
product_ratio_2019_df["date"] =  product_ratio_2019_df["date"] + pd.DateOffset(years=4)

forecasted_ratios_df = pd.concat([product_ratio_2017_df, product_ratio_2018_df, product_ratio_2019_df])


# Adding in the store ratios
store_weights_df = store_weights.reset_index()
test_sub_df = pd.merge(test, test_total_sales_dates, how="left", on="date")
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

# Disaggregating the forecast
test_sub_df["num_sold"] = test_sub_df["day_num_sold"] * test_sub_df["store_ratio"] * test_sub_df["country_ratio"] * test_sub_df["product_ratio"]
test_sub_df["num_sold"] = test_sub_df["num_sold"].round()
display(test_sub_df.head(2))


### Submission.....
sample["num_sold"] = test_sub_df["num_sold"]

sample.to_csv('submission.csv', index = False)
sample.head()




