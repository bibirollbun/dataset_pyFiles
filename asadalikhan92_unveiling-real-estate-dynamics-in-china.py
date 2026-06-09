# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ====================================================
# China Real Estate Demand Prediction - Advanced EDA
# ====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="tab10")

# ----------------------------------------
# 1. Load Data
# ----------------------------------------
base_path = "/kaggle/input/china-real-estate-demand-prediction/"
files = {
    "sample_submission": base_path + "sample_submission.csv",
    "test": base_path + "test.csv",
    "city_search_index": base_path + "train/city_search_index.csv",
    "land_transactions_nearby": base_path + "train/land_transactions_nearby_sectors.csv",
    "new_house_transactions_nearby": base_path + "train/new_house_transactions_nearby_sectors.csv",
    "city_indexes": base_path + "train/city_indexes.csv",
    "pre_owned_house_transactions": base_path + "train/pre_owned_house_transactions.csv",
    "new_house_transactions": base_path + "train/new_house_transactions.csv",
    "land_transactions": base_path + "train/land_transactions.csv",
    "sector_POI": base_path + "train/sector_POI.csv",
    "pre_owned_nearby": base_path + "train/pre_owned_house_transactions_nearby_sectors.csv"
}
data = {name: pd.read_csv(path) for name, path in files.items()}

# Ensure datetime where needed
for key in ["pre_owned_house_transactions", "new_house_transactions", "land_transactions"]:
    if "month" in data[key].columns:
        data[key]["month"] = pd.to_datetime(data[key]["month"], errors="coerce")

# ----------------------------------------
# 2. Tabular EDA Summary
# ----------------------------------------
print("\nðŸ”Ž Missing Values Summary")
missing_df = pd.DataFrame({k: v.isna().sum() for k,v in data.items()}).T
print(missing_df)

print("\nðŸ“Š Dataset Shapes")
shapes = pd.DataFrame({k: v.shape for k,v in data.items()}, index=["Rows","Cols"]).T
print(shapes)

print("\nðŸ“ˆ Pre-Owned Transactions Summary")
print(data["pre_owned_house_transactions"].describe(include="all").T)

print("\nðŸ“ˆ New House Transactions Summary")
print(data["new_house_transactions"].describe(include="all").T)

print("\nðŸ“ˆ Land Transactions Summary")
print(data["land_transactions"].describe(include="all").T)

# ----------------------------------------
# 3. Correlation Analysis (Numerical Features)
# ----------------------------------------
def plot_corr(df, title):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", annot=False, cbar_kws={'label': 'Correlation'})
    plt.title(title, fontsize=16, weight="bold")
    plt.show()

plot_corr(data["pre_owned_house_transactions"], "Correlation: Pre-Owned Transactions")
plot_corr(data["new_house_transactions"], "Correlation: New House Transactions")
plot_corr(data["land_transactions"], "Correlation: Land Transactions")

# ----------------------------------------
# 4. Pre-Owned Houses (Advanced Visuals)
# ----------------------------------------
df_pre = data["pre_owned_house_transactions"]
top_pre = (
    df_pre.groupby("sector")["area_pre_owned_house_transactions"]
    .sum().sort_values(ascending=False).head(10).index
)

# Trend (rolling mean)
df_pre["rolling_price"] = df_pre.groupby("sector")["price_pre_owned_house_transactions"]\
                               .transform(lambda x: x.rolling(3,1).mean())

plt.figure(figsize=(14,7))
sns.lineplot(
    data=df_pre[df_pre["sector"].isin(top_pre)],
    x="month", y="rolling_price", hue="sector", linewidth=2.5
)
plt.title("Pre-Owned House Prices (Top 10 Sectors, 3-Month Rolling Avg)", fontsize=16, weight="bold")
plt.xlabel("Month"); plt.ylabel("Price (Yuan/sqm)")
plt.xticks(rotation=45); plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout(); plt.show()

# Boxplot distribution
plt.figure(figsize=(12,6))
sns.boxplot(data=df_pre[df_pre["sector"].isin(top_pre)], x="sector", y="price_pre_owned_house_transactions")
plt.title("Price Distribution of Pre-Owned Houses (Top 10 Sectors)", fontsize=16, weight="bold")
plt.xticks(rotation=45); plt.show()

# ----------------------------------------
# 5. New House Transactions
# ----------------------------------------
df_new = data["new_house_transactions"]
top_new = (
    df_new.groupby("sector")["num_new_house_transactions"]
    .sum().sort_values(ascending=False).head(10).index
)

df_new["rolling_price"] = df_new.groupby("sector")["price_new_house_transactions"]\
                                .transform(lambda x: x.rolling(3,1).mean())

plt.figure(figsize=(14,7))
sns.lineplot(
    data=df_new[df_new["sector"].isin(top_new)],
    x="month", y="rolling_price", hue="sector", linewidth=2.5
)
plt.title("New House Prices (Top 10 Sectors, Smoothed)", fontsize=16, weight="bold")
plt.xlabel("Month"); plt.ylabel("Price (Yuan/sqm)")
plt.xticks(rotation=45); plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout(); plt.show()

# ----------------------------------------
# 6. Land Transactions
# ----------------------------------------
df_land = data["land_transactions"]
top_land = df_land.groupby("sector")["transaction_amount"].sum().sort_values(ascending=False).head(5).index
df_land["rolling_amount"] = df_land.groupby("sector")["transaction_amount"]\
                                   .transform(lambda x: x.rolling(3,1).mean())

plt.figure(figsize=(14,7))
sns.lineplot(data=df_land[df_land["sector"].isin(top_land)], 
             x="month", y="rolling_amount", hue="sector", linewidth=2.5)
plt.title("Land Transaction Amounts (Top 5 Sectors, Smoothed)", fontsize=16, weight="bold")
plt.xlabel("Month"); plt.ylabel("Amount (10,000 Yuan)")
plt.xticks(rotation=45); plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout(); plt.show()

# ----------------------------------------
# 7. Sector POI (Demographics & Prices)
# ----------------------------------------
df_poi = data["sector_POI"]

# Top by population
top_pop = df_poi.nlargest(10, "resident_population")
plt.figure(figsize=(12,6))
sns.barplot(data=top_pop, x="sector", y="resident_population", palette="viridis")
plt.title("Top 10 Sectors by Resident Population", fontsize=16, weight="bold")
plt.xticks(rotation=45); plt.show()

# Population vs price
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_poi, x="resident_population", y="surrounding_housing_average_price",
                size="number_of_shops", hue="commercial_area", alpha=0.7)
plt.title("Population vs Housing Price (Bubble ~ Shops, Color ~ Commercial Area)", fontsize=14, weight="bold")
plt.show()

# ----------------------------------------
# 8. City Indexes (Macro Economy)
# ----------------------------------------
df_city = data["city_indexes"]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_city, x="gdp_100m", y="real_estate_development_investment_completed_10k",
                size="year_end_resident_population_10k", alpha=0.7)
plt.title("GDP vs Real Estate Investment (Bubble ~ Population)", fontsize=14, weight="bold")
plt.xlabel("GDP (100m Yuan)"); plt.ylabel("Investment (10k Yuan)")
plt.show()

# ----------------------------------------
# 9. Search Index (Demand Signals)
# ----------------------------------------
df_search = data["city_search_index"]
top_keywords = df_search.groupby("keyword")["search_volume"].sum().sort_values(ascending=False).head(15)

plt.figure(figsize=(12,6))
sns.barplot(x=top_keywords.index, y=top_keywords.values, palette="Spectral")
plt.title("Top 15 Real Estate Search Keywords", fontsize=16, weight="bold")
plt.xticks(rotation=30, ha="right"); plt.ylabel("Search Volume")
plt.show()

# ----------------------------------------
# 10. Summary Table of Key Drivers
# ----------------------------------------
summary = pd.DataFrame({
    "Dataset": ["Pre-Owned Houses", "New Houses", "Land Transactions", "POI", "City Indexes", "Search Index"],
    "Main Drivers": [
        "Area, Price, Num Transactions",
        "Price, Units Available, Sell-through",
        "Land Value, Area, Planned Building",
        "Population, Shops, Commercial Activity",
        "GDP, Income, Urbanization",
        "Search Demand (leading indicator)"
    ]
})
print("\nðŸ“Œ Key Driver Summary")
print(summary.to_string(index=False))

print("""
âœ… Deliverables:
- Clean EDA tables: shapes, missing, descriptive stats
- Correlation matrices
- Rolling smoothed HD line charts
- Distribution boxplots
- Bubble scatterplots (population, GDP vs housing)
- Heatmaps for time-series by sector
""")


