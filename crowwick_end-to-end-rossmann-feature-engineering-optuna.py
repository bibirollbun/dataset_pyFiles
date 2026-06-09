# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Time Series Analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import optuna

# Utilities
import warnings
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")

# Global settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# Load Dataset
train_df = pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv")
store_df = pd.read_csv("/kaggle/input/rossmann-store-sales/store.csv")
test_df = pd.read_csv("/kaggle/input/rossmann-store-sales/test.csv")
display(Markdown("### DataFrames Shapes  \n"
                 f"- **Train Data:** {train_df.shape}  \n"
                 f"- **Store Data:** {store_df.shape}  \n"
                 f"- **Test Data:** {test_df.shape}  \n"))


# General Information

display(Markdown("### Train Head"))
display(train_df.head())

display(Markdown("### Train Info"))
display(train_df.dtypes.to_frame("dtype"))

display(Markdown("### Train Statistics"))
display(train_df.describe())

display(Markdown("### Missing Values"))
display(train_df.isnull().sum().to_frame("missing"))


train_df["Date"] = pd.to_datetime(train_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])


# Histogram with KDE
def plot_histogram(data, column, bins=50):
    sns.histplot(data[column], bins=bins, kde=True, color='royalblue')
    plt.title(f'Distribution of {column}', fontsize=14)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# Sales distribution
plot_histogram(train_df, 'Sales')


# Customers distribution
plot_histogram(train_df, 'Customers')


#Boxplot
def plot_boxplot(data, column, by=None):
    if by:
        sns.boxplot(x=by, y=data[column], data=data, palette='Set3')
        plt.title(f'Boxplot of {column} by {by}', fontsize=14)
        plt.xlabel(by)
    else:
        sns.boxplot(y=data[column], color='skyblue')
        plt.title(f'Boxplot of {column}', fontsize=14)
        plt.ylabel(column)
    plt.show()


plot_boxplot(train_df, 'Sales', 'DayOfWeek')
plot_boxplot(train_df, 'Sales', 'Promo')
plot_boxplot(train_df, 'Sales', 'SchoolHoliday')


train_store = pd.merge(train_df, store_df, on="Store", how="left")
plot_boxplot(train_store, 'Sales', 'StoreType')


corr = train_df.corr(numeric_only=True, method='pearson')
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=.5)
plt.title("Correlation Heatmap Pearson", fontsize=14)
plt.show()


def plot_time_series(data, store_id=None, column="Sales"):
    plt.figure(figsize=(12,6))

    if store_id:
        ts = data[data["Store"] == store_id].set_index("Date")[column]
        title = f"{column} over time - Store {store_id}"
    else:
        ts = data.groupby("Date")[column].sum()
        title = f"Total {column} over time (all stores)"
    
    ts.plot()
    plt.title(title, fontsize=14)
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.grid(True)
    plt.show()


plot_time_series(train_df)
plot_time_series(train_df, store_id=1)
plot_time_series(train_df, store_id=2)


# Global sales by date
global_sales = train_df.groupby('Date')['Sales'].sum()

#Decomposition
decomposition = seasonal_decompose(global_sales, model='additive', period=365)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()


plt.figure(figsize=(12,4))
plot_acf(global_sales, lags=60, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)")
plt.show()

plt.figure(figsize=(12,4))
plot_pacf(global_sales, lags=60, ax=plt.gca(), method="ywm")
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()


from statsmodels.tsa.stattools import adfuller
def adf_report(series, name):
    series = series.dropna()
    result = adfuller(series, autolag='AIC')
    stat, pval, usedlag, nobs = result[0], result[1], result[2], result[3]
    cvals = result[4]
    text = (
        f"**ADF - {name}**  \n"
        f"- Test statistic: {stat:.3f}  \n"
        f"- p-value: {pval:.4f}  \n"
        f"- Used lags: {usedlag} | N obs: {nobs}  \n"
        f"- Critical values: " +
        ", ".join([f"{k}: {v:.3f}" for k, v in cvals.items()]) + "  \n"
        f"- Decision (α=0.05): " +
        ("Stationary ✅ (reject H₀)" if pval < 0.05 else "Non-stationary ❌ (fail to reject H₀)")
    )
    display(Markdown(text))
    return pval

display(Markdown("## 4.4 Stationarity Test (ADF) - Global Sales"))
p_raw = adf_report(global_sales, "Raw series")


from statsmodels.tsa.stattools import kpss

def adf_test(series, alpha=0.05):
    result = adfuller(series.dropna(), autolag="AIC")
    stat, pvalue, lags, nobs, critical, _ = result
    decision = "Stationary ✅" if pvalue < alpha else "Non-stationary ❌"
    return {"ADF Stat": stat, "p-value": pvalue, "Decision ADF": decision}

def kpss_test(series, alpha=0.05, regression="c"):
    try:
        stat, pvalue, lags, critical = kpss(series.dropna(), regression=regression, nlags="auto")
        decision = "Non-stationary ❌" if pvalue < alpha else "Stationary ✅"
    except:
        stat, pvalue, decision = np.nan, np.nan, "Test failed"
    return {"KPSS Stat": stat, "p-value": pvalue, "Decision KPSS": decision}



# Select a balanced set of stores for training
store_ids = [3, 7, 21] + list(train_store["Store"].drop_duplicates().sample(7, random_state=42))


results = []
for sid in store_ids:
    sales_series = train_df.loc[train_df["Store"] == sid, "Sales"]
    adf_res = adf_test(sales_series)
    kpss_res = kpss_test(sales_series)

    results.append({
        "Store": sid,
        "ADF Stat": adf_res["ADF Stat"],
        "ADF p-value": adf_res["p-value"],
        "Decision ADF": adf_res["Decision ADF"],
        "KPSS Stat": kpss_res["KPSS Stat"],
        "KPSS p-value": kpss_res["p-value"],
        "Decision KPSS": kpss_res["Decision KPSS"]
    })


# summary
stationarity_df = pd.DataFrame(results)
stationarity_df.reset_index(drop=True, inplace=True)
display(Markdown("### Stationarity Test Results (Sample of Stores)"))
display(stationarity_df)


import holidays
de_holidays = holidays.Germany()
def create_features(df: pd.DataFrame, lags=[7, 14, 28], windows=[7, 14, 28]) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    # Calendar features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['day'] = df['Date'].dt.day
    df['is_holiday'] = df['Date'].isin(de_holidays).astype(int)
    df['Promo_DayOfWeek'] = df['Promo'] * df['dayofweek']

    # Lags per store
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
        df[f'pct_change_lag_{lag}'] = (
            df.groupby('Store')['Sales'].shift(1).pct_change(periods=lag)
        )

    # Rolling per store
    for window in windows:
        df[f'rmean_{window}'] = (
            df.groupby('Store')['Sales']
              .transform(lambda s: s.shift(1).rolling(window).mean())
        )
        df[f'rstd_{window}'] = (
            df.groupby('Store')['Sales']
              .transform(lambda s: s.shift(1).rolling(window).std())
        )
        df[f'median_{window}'] = (
            df.groupby('Store')['Sales']
              .transform(lambda s: s.shift(1).rolling(window).median())
        )
        df[f'rmin_{window}'] = (
            df.groupby('Store')['Sales']
              .transform(lambda s: s.shift(1).rolling(window).min())
        )
        df[f'rmax_{window}'] = (
            df.groupby('Store')['Sales']
              .transform(lambda s: s.shift(1).rolling(window).max())
        )

    # 4) Target transformation
    df['Sales_log'] = np.log1p(df['Sales'])

    return df


train_fe = create_features(train_df)
display(
    train_fe.query('Store == 1')[['Date','Sales','lag_7','rmean_7','rstd_7']].tail(20)
)


# Order the DataFrame
train_fe = train_fe.sort_values(["Store", "Date"]).reset_index(drop=True)
# We will use the last 6 weeks of train as validation
val_start_date = train_fe["Date"].max() - pd.Timedelta(weeks=6)
# Split the data
train_df = train_fe[train_fe["Date"] < val_start_date].copy()
val_df   = train_fe[train_fe["Date"] >= val_start_date].copy()

display(Markdown(f"Train hasta: {train_df['Date'].max()}  \n"
                 f"Validación desde: {val_df['Date'].min()} hasta: {val_df['Date'].max()}"))
# Target
y_train = train_fe[train_fe['Date'] < val_start_date]["Sales_log"]
y_val   = train_fe[train_fe['Date'] >= val_start_date]["Sales_log"]

drop_cols = ["Sales", "Sales_log", "Date"]
X_train = train_fe[train_fe['Date'] < val_start_date].drop(columns=drop_cols)
X_val   = train_fe[train_fe['Date'] >= val_start_date].drop(columns=drop_cols)

# We validate that columns that are not of object type
for col in X_train.select_dtypes(include="object").columns:
    X_train[col] = X_train[col].astype("category")
    X_val[col]   = X_val[col].astype(pd.CategoricalDtype(categories=X_train[col].cat.categories))

# Identify numeric and categorical columns
num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(include='category').columns.tolist()

# Clean the data for XGBoost
X_train_xgb = X_train.copy()
X_val_xgb   = X_val.copy()
X_train_xgb[num_cols] = X_train_xgb[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
X_val_xgb[num_cols]   = X_val_xgb[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

display(Markdown(f"X_train: {X_train.shape}  \n"
                 f"X_val: {X_val.shape}"))



# LightGBM
lgb_model = lgb_model = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.14149080338407782,
    num_leaves=55,
    max_depth=6,
    n_jobs=-1,
    random_state=42,
    verbose =-1
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[early_stopping(stopping_rounds=100)]
)


# CatBoost
cat_model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.16780567752854458,
    depth=10,
    l2_leaf_reg=7.69136102843618,
    thread_count=-1,
    random_seed=42,
    verbose=0
)
cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_cols,
    early_stopping_rounds=100
)


xgb_model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.07721710330379798,
    max_depth=8,
    subsample=0.9515651074625635,
    colsample_bytree=0.8491614704487909,
    n_jobs=-1,
    enable_categorical=True,
    random_state=42
)
xgb_model.fit(
    X_train_xgb, y_train,
    eval_set=[(X_val_xgb, y_val)],
    verbose=0
)


y_pred_lgb = np.expm1(lgb_model.predict(X_val))
y_pred_cat = np.expm1(cat_model.predict(X_val))
y_pred_xgb = np.expm1(xgb_model.predict(X_val_xgb))

y_pred_blend = (y_pred_lgb + y_pred_cat + y_pred_xgb) / 3

def rmspe(y_true, y_pred):
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

rmspe_val = rmspe(val_df["Sales"].values, y_pred_blend)
display(f"RMSPE Blended: {rmspe_val:.4%}")

