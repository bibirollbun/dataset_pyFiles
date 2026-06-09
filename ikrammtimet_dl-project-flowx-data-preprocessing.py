import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install py7zr


import os
import numpy as np 
import pandas as pd
import py7zr
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


for dirname, _, filenames in os.walk('/kaggle/input/favorita-grocery-sales-forecasting'):
    for filename in filenames:
        archive = py7zr.SevenZipFile(os.path.join(dirname, filename), mode='r')
        archive.extractall(path="/kaggle/working")
        archive.close()

print(check_output(["ls", "../working"]).decode("utf8"))


items = pd.read_csv("../working/items.csv")
holiday_events = pd.read_csv("../working/holidays_events.csv", parse_dates=['date'])
stores = pd.read_csv("../working/stores.csv")
oil = pd.read_csv("../working/oil.csv", parse_dates=['date'])
transactions = pd.read_csv("../working/transactions.csv", parse_dates=['date'])
train = pd.read_csv('../working/train.csv', parse_dates = ['date'])
test = pd.read_csv('../working/test.csv', parse_dates = ['date'])


print(f"Train: {train.shape}")
print(f"Items: {items.shape}")
print(f"Stores: {stores.shape}")
print(f"Oil: {oil.shape}")
print(f"Holidays: {holiday_events.shape}")
print(f"Transactions: {transactions.shape}")


print(f"Train: {train.info()}")
print(f"Items: {items.info()}")
print(f"Stores: {stores.info()}")
print(f"Oil: {oil.info()}")
print(f"Holidays: {holiday_events.info()}")
print(f"Transactions: {transactions.info()}")


train.head()


train.isnull().sum()


train['onpromotion'] = train['onpromotion'].fillna(False)
train['onpromotion'] = train['onpromotion'].astype(bool).astype(int)

train.isnull().sum()


train.tail()


train.duplicated().sum()


TOP_STORES = [44, 47, 45, 46, 3, 48, 8, 49, 50, 11]
YEARS = [2016, 2017]
OUTPUT_PATH = "/kaggle/working/"
# Filter years & stores
train = train[
    (train["date"].dt.year.isin(YEARS)) &
    (train["store_nbr"].isin(TOP_STORES))
]

print("Sales shape:", train.shape)

train.to_csv(OUTPUT_PATH + "dashboard_sales.csv", index=False)


# Aggregate unit sales by date
daily_sales = train.groupby("date")["unit_sales"].sum().reset_index()

# Plot line graph
plt.figure(figsize=(12, 6))
plt.plot(daily_sales["date"], daily_sales["unit_sales"], label="Total Daily Sales", color="blue")

# Formatting
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Daily Sales Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()


daily_sales = train.groupby('date')['unit_sales'].sum().reset_index()
daily_sales['rolling_avg'] = daily_sales['unit_sales'].rolling(window=30).mean()

# Plot the trend
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(daily_sales['date'], daily_sales['unit_sales'], label='Daily Sales', alpha=0.5)
plt.plot(daily_sales['date'], daily_sales['rolling_avg'], label='30-Day Rolling Avg', color='red')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# Filter only 2017 data
daily_sales_2017 = daily_sales[daily_sales["date"].dt.year == 2017].copy()

# Set index for decomposition
daily_sales_2017 = daily_sales_2017.set_index("date")

# Decompose the time series — weekly seasonality
decomposition = seasonal_decompose(
    daily_sales_2017["unit_sales"],
    model="additive",
    period=7
)

plt.figure(figsize=(50, 8))
decomposition.plot()
plt.show()


oil.head()


oil.isnull().sum()


oil['dcoilwtico'] = oil['dcoilwtico'].interpolate()
oil['dcoilwtico'] = oil['dcoilwtico'].fillna(method='bfill')

oil.isnull().sum()


oil = oil[oil["date"].dt.year.isin(YEARS)]
oil.to_csv(
    OUTPUT_PATH + "dashboard_oil.csv",
    index=False
)


# Aggregate daily sales
sales_oil = train.groupby("date")["unit_sales"].sum().reset_index()  
sales_oil = sales_oil.merge(oil, on="date", how="left") 

plt.figure(figsize=(14, 6))

# Plot Sales
plt.plot(sales_oil["date"], sales_oil["unit_sales"], label="Total Unit Sales", color="blue", alpha=0.6)

# Plot Oil Prices on a second y-axis
plt.twinx()
plt.plot(sales_oil["date"], sales_oil["dcoilwtico"], label="Oil Prices", color="red", alpha=0.6)

plt.title("Daily Sales vs Oil Prices")
plt.xlabel("Date")
plt.legend(["Unit Sales", "Oil Prices"])
plt.show()


correlation = sales_oil["unit_sales"].corr(sales_oil["dcoilwtico"])
print(f"Correlation between oil price and sales: {correlation:.4f}")


# Create lag features for oil prices
sales_oil["oil_lag_7"] = sales_oil["dcoilwtico"].shift(7)  # Lag by 7 days
sales_oil["oil_lag_3"] = sales_oil["dcoilwtico"].shift(3)  # Lag by 3 days

# Compute correlation with lagged values
correlation_lag_7 = sales_oil["unit_sales"].corr(sales_oil["oil_lag_7"])
correlation_lag_3 = sales_oil["unit_sales"].corr(sales_oil["oil_lag_3"])

print(f"Correlation with 7-day lag: {correlation_lag_7:.4f}")
print(f"Correlation with 3-day lag: {correlation_lag_3:.4f}")


holiday_events.tail()


holiday_events.isnull().sum()


holiday_events = holiday_events[
    (holiday_events["date"].dt.year.isin(YEARS)) &
    (holiday_events["transferred"] == False)
]
holiday_events.to_csv(
    OUTPUT_PATH + "dashboard_holidays.csv",
    index=False
)


holiday_events['type'].value_counts()


holiday_events['description'].value_counts()


sns.set_theme(style="whitegrid")
colName="type"
val = holiday_events[colName].value_counts().sort_values(ascending=False)
f, axs = plt.subplots(1, 1, figsize=(10, 5))
ax = sns.countplot(holiday_events, x=colName # or y
                   , palette="pastel"
                   , order=val.index
                   ,ax=axs
                   )

summ = val.sum()

title = val.index
lbls = [f'{p[0] * 100 / summ:.1f}%' for p in zip(val)]


# hor label
for p, label in zip(ax.patches, lbls):
    ax.annotate(label, (p.get_x(), p.get_height() + 0.15))
    
plt.title("holiday distribution",
          fontsize = 20)    

plt.tight_layout()
plt.show()


# Ensure date column is datetime
holiday_events['date'] = pd.to_datetime(holiday_events['date'])

# Keep only relevant holiday types
# Exclude transferred days (these are not actually celebrated on this day)
holidays_actual = holiday_events[
    (holiday_events['type'].isin(['Holiday', 'Additional', 'Bridge'])) & 
    (holiday_events['transferred'] == False)
].copy()

# simplify description
holidays_actual['holiday_flag'] = 1


holidays_actual.head()


# Aggregate total unit sales per day
daily_sales = train.groupby('date')['unit_sales'].sum().reset_index()

daily_sales = daily_sales.merge(
    holidays_actual[['date', 'holiday_flag']],
    on='date',
    how='left'
)

# Fill NaN (non-holiday) with 0
daily_sales['holiday_flag'].fillna(0, inplace=True)


daily_sales['holiday_label'] = daily_sales['holiday_flag'].map({1: 'Holiday', 0: 'Non-Holiday'})
avg_sales = daily_sales.groupby('holiday_label')['unit_sales'].mean().reset_index()

plt.figure(figsize=(6,4))
sns.barplot(x='holiday_label', y='unit_sales', data=avg_sales, palette=['skyblue','lightgreen'])
plt.title("Average Daily Sales: Holidays vs Non-Holidays (All Stores, All Years)")
plt.ylabel("Average Unit Sales")
plt.xlabel("")
plt.show()


stores.tail(11)


stores.isnull().sum()


stores = stores[stores["store_nbr"].isin(TOP_STORES)]

stores.to_csv(
    OUTPUT_PATH + "dashboard_stores.csv",
    index=False
)


items.head()


items.isnull().sum()


items.to_csv(
    OUTPUT_PATH + "dashboard_items.csv",
    index=False
)


items['family'].value_counts()


colName="family"

sns.set_theme(style="whitegrid")
val = items[colName].value_counts().sort_values(ascending=False)
plt.figure(figsize=(8, 10))

ax = sns.countplot(items, y=colName # or y
                   , palette="pastel"
                   ,hue ="perishable"
                   # , stat="percent"
                   , order=val.index
                   )
plt.rcParams["figure.figsize"] = (20,25)
summ = val.sum()

title = val.index
lbls = [f'{p[0] * 100 / summ:.1f}%' for p in zip(val)]

#vetical label
ax.bar_label(container=ax.containers[0], labels=lbls)

plt.legend(["not perishable", "perishable"],loc='center left', bbox_to_anchor=(1, 0.5))#out
plt.title("Distribution of perishable item",
          fontsize = 18)
plt.tight_layout()
plt.show()


transactions.head()


transactions.isnull().sum()


transactions = transactions[
    (transactions["date"].dt.year.isin(YEARS)) &
    (transactions["store_nbr"].isin(TOP_STORES))
]

transactions.to_csv(
    OUTPUT_PATH + "dashboard_transactions.csv",
    index=False
)


grouped = (
    transactions.groupby("store_nbr")["transactions"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 10))

plt.barh(grouped.index.astype(str), grouped.values)

plt.gca().invert_yaxis()

plt.xlabel("Total Transactions", fontsize=12)
plt.ylabel("Store Number (ID)", fontsize=12)
plt.title("Total Transactions per Store", fontsize=14)

plt.tight_layout()
plt.show()


store_id = 44

# Filter by store AND by year 2017
store_2017 = train[
    (train["store_nbr"] == store_id) &
    (train["date"].dt.year == 2017)
]

# Aggregate unit sales by date
daily_sales = (
    store_2017.groupby("date")["unit_sales"]
    .sum()
    .reset_index()
)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(
    daily_sales["date"],
    daily_sales["unit_sales"],
    label=f"Store {store_id} — Daily Sales (2017)"
)

# Formatting
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title(f"Daily Sales Over Time — Store {store_id} in 2017")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()


train_df['date'] = pd.to_datetime(train_df['date'])
train_df = train_df[(train_df['date'].dt.year >= 2016) & (train_df['date'].dt.year <= 2017)]

train_df = train_df.reset_index(drop=True)
train_df['id'] = train_df.index + 1


train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['weekday'] = train_df['date'].dt.dayofweek  # Monday=0, Sunday=6
train_df['is_weekend'] = train_df['weekday'].isin([5, 6]).astype(int)


holiday_simple = holidays_actual.groupby("date").agg({
    "holiday_flag": "max"   
}).reset_index()

train_df = train_df.merge(holiday_simple,on="date",how="left")


train_df['holiday_flag'] = train_df['holiday_flag'].fillna(0)


train_df = pd.merge(train_df, items[['item_nbr', 'family','class', 'perishable']], on='item_nbr', how='left')

train_df = pd.merge(train_df, oil[['date', 'dcoilwtico']], on='date', how='left')


tx_44 = transactions[transactions["store_nbr"] == 44]

train_df = train_df.merge(tx_44[['date', 'transactions']], on=['date'], how='left')


# Forward fill missing values 
train_df['dcoilwtico'] = train_df['dcoilwtico'].fillna(method='ffill')

# Fill missing transaction counts by interpolation
train_df["transactions"] = train_df.groupby("store_nbr")["transactions"].transform(
    lambda x: x.interpolate().bfill().ffill()
)


train_df.isnull().sum()


train_df.info()


train_df.drop(columns=['store_nbr','id'], inplace=True)
train_df.head()


train_df.shape


#train_df.to_pickle("train_df.pkl")


test.head()


test_df = test[test["store_nbr"] == 44].reset_index(drop=True)


test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['weekday'] = test_df['date'].dt.dayofweek  # Monday=0, Sunday=6
test_df['is_weekend'] = test_df['weekday'].isin([5, 6]).astype(int)


test_df = test_df.merge(
    items[['item_nbr', 'family', 'class', 'perishable']],
    on='item_nbr',
    how='left'
)

test_df = test_df.merge(holiday_simple[['date', 'holiday_flag']],on='date',how='left')
test_df['holiday_flag'] = test_df['holiday_flag'].fillna(0)

test_df = test_df.merge(oil[['date', 'dcoilwtico']], on='date', how='left')
test_df['dcoilwtico'] = test_df['dcoilwtico'].fillna(method='ffill')


test_df.drop(columns=['store_nbr','id'], inplace=True)
test_df.head()


test_df.shape


#test_df.to_pickle("test_df.pkl")


#test_df = pd.read_pickle("test_df.pkl")


trainre= pd.read_csv("/kaggle/input/trainre/FlowXTransformer Reduced Train Data.csv")
trainre.head()


trainre.info()


trainre.columns


categorical_cols = ["family", "holiday_type", "description"]
numerical_cols = ["item_nbr", "onpromotion", "class", "perishable",
                  "dcoilwtico", "day_of_week", "day_of_month", "month", "year"]


print("Categorical Columns Info:\n")
for col in categorical_cols:
    unique_vals = sorted(trainre[col].dropna().unique())
    print(f"Column: {col}")
    print(f"Unique Values ({len(unique_vals)}): {unique_vals}\n")

# Extract min-max for numerical columns
print("Numerical Columns Info:\n")
for col in numerical_cols:
    col_min = trainre[col].min()
    col_max = trainre[col].max()
    print(f"Column: {col}")
    print(f"Min: {col_min}, Max: {col_max}\n")


import pandas as pd
import json

# --------------------------
# Load your datasets
# --------------------------
sales = pd.read_csv("/kaggle/working/dashboard_sales.csv")        # your sales file
stores = pd.read_csv("/kaggle/working/dashboard_stores.csv")      # stores file

# --------------------------
# Filter data for Top-10 stores and 2016-2017
# --------------------------
TOP_STORES = [44, 47, 45, 46, 3, 48, 8, 49, 50, 11]

# Ensure 'date' column is datetime
sales['date'] = pd.to_datetime(sales['date'])
sales = sales[(sales['date'].dt.year.isin([2016, 2017])) & (sales['store_nbr'].isin(TOP_STORES))]

# --------------------------
# Compute KPIs
# --------------------------

# 1. Total Sales
total_sales = sales['unit_sales'].sum()

# 2. Number of Active Stores
num_stores = sales['store_nbr'].nunique()

# 3. Number of Products (SKUs)
num_products = sales['item_nbr'].nunique()

# 4. Average Daily Sales
avg_daily_sales = sales.groupby('date')['unit_sales'].sum().mean()

# 5. Promotion Sales Ratio
promo_ratio = sales[sales['onpromotion'] == True]['unit_sales'].sum() / sales['unit_sales'].sum()

# 6. Sales Volatility (Std Dev)
sales_volatility = sales.groupby('date')['unit_sales'].sum().std()

# 7. Average Sales per Store
avg_sales_per_store = sales.groupby('store_nbr')['unit_sales'].sum().mean()

# 8. Top Product Contribution (%)
top_product_share = sales.groupby('item_nbr')['unit_sales'].sum().sort_values(ascending=False).iloc[0] / sales['unit_sales'].sum()

# --------------------------
# Prepare JSON
# --------------------------
kpis = {
    "total_sales": round(total_sales, 2),
    "num_stores": int(num_stores),
    "num_products": int(num_products),
    "avg_daily_sales": round(avg_daily_sales, 2),
    "promo_ratio": round(promo_ratio, 4),  # 4 decimals for ratio
    "sales_volatility": round(sales_volatility, 2),
    "avg_sales_per_store": round(avg_sales_per_store, 2),
    "top_product_share": round(top_product_share, 4)
}

# Save to JSON
with open('kpis.json', 'w') as f:
    json.dump(kpis, f, indent=4)

print("✅ KPIs JSON file created successfully!")



# ============================================
# FlowX Model Insights JSON (Dashboard Version)
# ============================================

import pandas as pd
import json

# Load reduced training dataset
train = pd.read_csv("/kaggle/input/trainre/FlowXTransformer Reduced Train Data.csv")

# Convert date column
train['date'] = pd.to_datetime(train['date'])

# Filter years 2016–2017
train = train[
    (train['date'].dt.year >= 2016) &
    (train['date'].dt.year <= 2017)
]

# Select top 10 stores by total transactions
top_10_stores = (
    train.groupby('store_nbr')['transactions']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
    .tolist()
)

train = train[train['store_nbr'].isin(top_10_stores)]

# Dataset size information
dataset_size = {
    "rows": int(train.shape[0]),
    "columns": int(train.shape[1]),
    "years_used": [2016, 2017],
    "stores_used": sorted(train['store_nbr'].unique().tolist()),
    "number_of_stores": train['store_nbr'].nunique(),
    "number_of_products": train['item_nbr'].nunique()
}

# Target statistics
sales_statistics = train['unit_sales'].describe().to_dict()

# Final Model Insights JSON
model_insights = {
    "project": {
        "title": "FlowX – Transformer-Based Grocery Sales Forecasting",
        "description": (
            "FlowX is an AI-driven demand forecasting system designed to predict daily "
            "product sales in grocery stores. The project focuses on capturing complex "
            "temporal patterns and external influences to support inventory planning "
            "and decision-making."
        )
    },

    "dataset": {
        "name": "Corporación Favorita Grocery Sales Forecasting",
        "source": "Kaggle Competition Dataset",
        "competition_data": True,
        "overview": (
            "The dataset originates from a Kaggle competition hosted by Corporación "
            "Favorita, Ecuador’s largest grocery retailer. It contains historical daily "
            "sales enriched with promotions, holidays, oil prices, and store transaction data."
        ),
        "time_scope": {
            "start_year": 2016,
            "end_year": 2017
        },
        "store_selection": {
            "strategy": "Top 10 stores with highest transaction volume",
            "store_ids": dataset_size["stores_used"]
        },
        "size": {
            "rows": dataset_size["rows"],
            "columns": dataset_size["columns"],
            "unique_products": dataset_size["number_of_products"],
            "unique_stores": dataset_size["number_of_stores"]
        },
        "target_variable": {
            "name": "unit_sales",
            "statistics": sales_statistics
        }
    },

    "model": {
        "name": "FlowX Transformer",
        "strategy": (
            "FlowX adopts a multimodal learning strategy, combining numerical, categorical, "
            "temporal, and textual data to model sales dynamics more accurately."
        ),

        "inputs": {
            "numerical_features": [
                "historical sales",
                "oil prices",
                "store transactions"
            ],
            "categorical_features": [
                "store",
                "product family",
                "holiday type"
            ],
            "text_features": {
                "source": "holiday descriptions",
                "embedding_model": "MiniLM",
                "embedding_dimension": 384
            },
            "temporal_features": [
                "day of week (cyclical encoding)",
                "month (cyclical encoding)"
            ]
        },

        "architecture": {
            "type": "Transformer (Built from Scratch)",
            "attention_mechanism": "Scaled Dot-Product Attention (QKᵀ / √dₖ)",
            "multi_head_attention": {
                "number_of_heads": 8,
                "purpose": "Capture multiple temporal patterns in parallel"
            },
            "encoder": {
                "layers": 4,
                "input_window": "30-day historical sequence"
            },
            "decoder": {
                "layers": 2,
                "forecast_horizon": "15 days"
            },
            "fusion_module": "Gated Residual Network for adaptive multimodal fusion",
            "positional_encoding": "Sinusoidal positional encoding"
        },

        "output": {
            "type": "Quantile Regression",
            "quantiles": [0.1, 0.5, 0.9],
            "description": (
                "The model predicts multiple quantiles to estimate uncertainty and provide "
                "confidence intervals for future sales."
            )
        }
    },

    "explainability": {
        "methods": [
            "Attention heatmaps",
            "Cross-attention analysis"
        ],
        "description": (
            "Explainability is achieved through attention visualization, highlighting "
            "which past days and events most influence future predictions."
        )
    },

    "training": {
        "optimization": {
            "mixed_precision": "Automatic Mixed Precision (AMP)",
            "learning_rate_schedule": "Cosine annealing",
            "gradient_clipping": True
        },
        "regularization": {
            "early_stopping": "Enabled with patience"
        },
        "goal": (
            "Ensure stable training, faster convergence, and improved generalization."
        )
    }
}

# Save JSON file
with open("model_insights.json", "w") as f:
    json.dump(model_insights, f, indent=4)

print("✅ model_insights.json generated successfully for the dashboard!")



# ============================================================
# FlowX Dashboard Data Generation (2016–2017, Top-10 Stores)
# ============================================================

import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
TOP_STORES = [44, 47, 45, 46, 3, 48, 8, 49, 50, 11]
OUTPUT_PATH = "/kaggle/working/"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(
    "/kaggle/input/trainre/FlowXTransformer Reduced Train Data.csv",
    parse_dates=["date"]
)

# -----------------------------
# FILTER DATA
# -----------------------------
df = df[
    (df["date"].dt.year >= 2016) &
    (df["date"].dt.year <= 2017) &
    (df["store_nbr"].isin(TOP_STORES))
].copy()

# Safety
df["unit_sales"] = df["unit_sales"].clip(lower=0)

print(f"Filtered data shape: {df.shape}")

# ============================================================
# 1️⃣ TOTAL SALES OVER TIME
# ============================================================
total_sales_time = (
    df.groupby("date", as_index=False)["unit_sales"]
      .sum()
)

total_sales_time.to_csv(
    OUTPUT_PATH + "dashboard_total_sales_over_time.csv",
    index=False
)

# ============================================================
# 2️⃣ SALES VS OIL PRICE
# ============================================================
sales_oil = (
    df.groupby("date", as_index=False)
      .agg({
          "unit_sales": "sum",
          "dcoilwtico": "mean"
      })
)

sales_oil.to_csv(
    OUTPUT_PATH + "dashboard_sales_vs_oil_price.csv",
    index=False
)

# ============================================================
# 3️⃣ SALES VS HOLIDAYS
# ============================================================
sales_holiday = (
    df.groupby("holiday_type", as_index=False)["unit_sales"]
      .sum()
      .sort_values("unit_sales", ascending=False)
)

sales_holiday.to_csv(
    OUTPUT_PATH + "dashboard_sales_vs_holidays.csv",
    index=False
)

# ============================================================
# 4️⃣ SALES PER PRODUCT PER STORE (INTERACTIVE)
# ============================================================
sales_product_store = (
    df.groupby(
        ["date", "store_nbr", "item_nbr"],
        as_index=False
    )["unit_sales"]
    .sum()
)

sales_product_store.to_csv(
    OUTPUT_PATH + "dashboard_sales_product_store.csv",
    index=False
)

# ============================================================
# 5️⃣ TOP PRODUCTS
# ============================================================
top_products = (
    df.groupby("item_nbr", as_index=False)["unit_sales"]
      .sum()
      .sort_values("unit_sales", ascending=False)
      .head(100)
)

top_products.to_csv(
    OUTPUT_PATH + "dashboard_top100_products.csv",
    index=False
)

# ============================================================
#  STORE PERFORMANCE
# ============================================================
store_performance = (
    df.groupby("store_nbr", as_index=False)["unit_sales"]
      .sum()
      .sort_values("unit_sales", ascending=False)
)

store_performance.to_csv(
    OUTPUT_PATH + "dashboard_store_performance.csv",
    index=False
)

# ============================================================
#  WEEKLY SEASONALITY
# ============================================================
weekly_pattern = (
    df.groupby("day_of_week", as_index=False)["unit_sales"]
      .mean()
)

weekly_pattern["day_name"] = weekly_pattern["day_of_week"].map({
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
})

weekly_pattern.to_csv(
    OUTPUT_PATH + "dashboard_weekly_pattern.csv",
    index=False
)

print("✅ All dashboard CSV files generated successfully!")




