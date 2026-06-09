import numpy as np
import pandas as pd

import os

# Duyá»‡t toÃ n bá»™ thÆ° má»¥c con bÃªn trong thÆ° má»¥c gá»‘c (file zip)
for dirname, _, filenames in os.walk('/kaggle/input/favorita-grocery-sales-forecasting'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install py7zr


import py7zr
from subprocess import check_output

for dirname, _, filenames in os.walk('/kaggle/input/favorita-grocery-sales-forecasting'):
    for filename in filenames:
        # Má»Ÿ láº§n lÆ°á»£t tá»«ng file zip
        archive = py7zr.SevenZipFile(os.path.join(dirname, filename), mode='r')
        # Giáº£i nÃ©n láº§n lÆ°á»£t tá»«ng file zip rá»“i xuáº¥t ra output
        archive.extractall(path="/kaggle/working")
        archive.close()

# Giáº£i nÃ©n thÃ nh cÃ´ng thÃ¬ in ra mÃ n hÃ¬nh Ä‘á»ƒ thÃ´ng bÃ¡o
print(check_output(["ls", "../working"]).decode("utf8"))


# Ä�á»�c tá»«ng file csv
train = pd.read_csv('../working/train.csv', parse_dates=['date'], low_memory = False)
test = pd.read_csv('../working/test.csv', parse_dates=['date'])
oil = pd.read_csv('../working/oil.csv', parse_dates=['date'])
stores = pd.read_csv('../working/stores.csv')
items = pd.read_csv('../working/items.csv')
transactions = pd.read_csv('../working/transactions.csv', parse_dates=['date'])
holidays = pd.read_csv('../working/holidays_events.csv', parse_dates=['date'])


train.head()


train.tail()


# Kiá»ƒm tra tá»«ng cá»™t trong tá»«ng bá»™ dá»¯ liá»‡u xem thá»­ cá»™t nÃ o xuáº¥t hiá»‡n missing values
print("Nulls in Oil columns: {0} => {1}".format(oil.columns.values,oil.isnull().any().values))
print("="*70)
print("Nulls in holiday_events columns: {0} => {1}".format(holidays.columns.values,holidays.isnull().any().values))
print("="*70)
print("Nulls in stores columns: {0} => {1}".format(stores.columns.values,stores.isnull().any().values))
print("="*70)
print("Nulls in transactions columns: {0} => {1}".format(transactions.columns.values,transactions.isnull().any().values))
print("="*70)
print("Nulls in train columns: {0} => {1}".format(train.columns.values,train.isnull().any().values))


datasets = {
    'train': train,
    'test': test,
    'oil': oil,
    'stores': stores,
    'items': items,
    'transactions': transactions,
    'holidays': holidays
}

for name, df in datasets.items():
    print(f"\n===== Missing values in {name} =====")
    print(df.isna().sum())



oil['dcoilwtico'] = oil['dcoilwtico'].interpolate()


oil['dcoilwtico'] = oil['dcoilwtico'].bfill()


oil.isnull().sum()


train.isnull().sum()


train['onpromotion'] = train['onpromotion'].fillna('False')


train.isnull().sum()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib import gridspec
from matplotlib.dates import DateFormatter


# === Cáº¥u hÃ¬nh ===
train_path = "/kaggle/working/train.csv"
sample_frac = 0.1
chunksize = 2_000_000
seed = 1234

# === Ä�á»�c theo khá»‘i vÃ  láº¥y máº«u ===
sample_list = []
reader = pd.read_csv(
    train_path,
    usecols=['id', 'date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion'],
    parse_dates=['date'],
    chunksize=chunksize,
    low_memory=False
)

for i, chunk in enumerate(reader, start=1):
    sample_chunk = chunk.sample(frac=sample_frac, random_state=seed)
    sample_list.append(sample_chunk)
    print(f"Ä�Ã£ xá»­ lÃ½ xong chunk {i}")

# === GhÃ©p toÃ n bá»™ máº«u ===
train_sample = pd.concat(sample_list, ignore_index=True)

print(f"\n HoÃ n táº¥t! Sá»‘ dÃ²ng sau khi láº¥y máº«u: {len(train_sample):,}")
print(train_sample.head())
train_sample.info(memory_usage='deep')



train.info()


import pandas as pd
import numpy as np

# TÃ¡ch nhÃ³m
num_cols = train_sample.select_dtypes(include=[np.number]).columns
train_sample[num_cols] = train_sample[num_cols].replace([np.inf, -np.inf], np.nan)

# Ä�Æ°a vá»� sá»‘ dá»… nháº­n xÃ©t
pd.options.display.float_format = '{:,.0f}'.format

desc_num = train_sample.describe()
desc_cat = train_sample.describe(include=['object', 'bool'])

print("\nThá»‘ng kÃª mÃ´ táº£")
print(desc_num)
print("="*10)
print(desc_cat)


test.info()


# TÃ¡ch nhÃ³m dá»¯ liá»‡u
num_cols = test.select_dtypes(include=[np.number]).columns
test[num_cols] = test[num_cols].replace([np.inf, -np.inf], np.nan)

# Hiá»ƒn thá»‹ sá»‘
pd.options.display.float_format = '{:,.0f}'.format

# Thá»‘ng kÃª mÃ´ táº£
desc_num = test.describe()                      
desc_cat = test.describe(include=['object', 'bool'])  # cho bool/object

print("Thá»‘ng kÃª mÃ´ táº£")
print(desc_num)

print("="*10)
print(desc_cat)


oil.info()


# TÃ¡ch nhÃ³m dá»¯ liá»‡u
num_cols = oil.select_dtypes(include=[np.number]).columns

# Hiá»ƒn thá»‹ sá»‘
pd.options.display.float_format = '{:,.0f}'.format

# Thá»‘ng kÃª mÃ´ táº£
desc_num = oil.describe()      

print("Thá»‘ng kÃª mÃ´ táº£")
print(desc_num)


stores.info()


# TÃ¡ch nhÃ³m dá»¯ liá»‡u
num_cols = stores.select_dtypes(include=np.number).columns
cat_cols = stores.select_dtypes(include=['object']).columns

# Hiá»ƒn thá»‹ sá»‘
pd.options.display.float_format = '{:,.0f}'.format

# Thá»‘ng kÃª mÃ´ táº£
desc_num = stores[num_cols].describe()
print(desc_num)
print("="*10)
desc_cat = stores[cat_cols].describe()
print(desc_cat)


items.info()


# TÃ¡ch nhÃ³m theo kiá»ƒu dá»¯ liá»‡u
num_cols = items.select_dtypes(include=np.number).columns
cat_cols = items.select_dtypes(include=['object']).columns

# Hiá»ƒn thá»‹ sá»‘
pd.options.display.float_format = '{:,.0f}'.format

# Thá»‘ng kÃª mÃ´ táº£
desc_num = items[num_cols].describe()
print(desc_num)
print("="*10)
desc_cat = items[cat_cols].describe()
print(desc_cat)


transactions.info()


# TÃ¡ch nhÃ³m dá»¯ liá»‡u
num_cols = transactions.select_dtypes(include=[np.number]).columns
transactions[num_cols] = transactions[num_cols].replace([np.inf, -np.inf], np.nan)

# Hiá»ƒn thá»‹ sá»‘
pd.options.display.float_format = '{:,.0f}'.format

# Thá»‘ng kÃª mÃ´ táº£
desc_num = transactions.describe()      

print("Thá»‘ng kÃª mÃ´ táº£")
print(desc_num)


holidays.info()


# TÃ¡ch nhÃ³m dá»¯ liá»‡u
num_cols = holidays.select_dtypes(include=[np.number]).columns
holidays[num_cols] = holidays[num_cols].replace([np.inf, -np.inf], np.nan)

# Hiá»ƒn thá»‹ sá»‘
pd.options.display.float_format = '{:,.0f}'.format

# Thá»‘ng kÃª mÃ´ táº£
desc_num = holidays.describe()                      
desc_cat = holidays.describe(include=['object', 'bool'])  # cho bool/object

print("Thá»‘ng kÃª mÃ´ táº£")
print(desc_num)

print("="*10)
print(desc_cat)


# Giáº£ sá»­ báº¡n Ä‘Ã£ cÃ³ train_sample (Ä‘Ã£ parse date vÃ  cÃ³ unit_sales)
df = train_sample.copy()

# ----------- PLOT 1: Tá»•ng doanh sá»‘ theo ngÃ y -----------
daily_sales = (
    df.groupby('date', as_index=False)['unit_sales']
    .sum()
    .rename(columns={'unit_sales': 'sales'})
)

plt.figure(figsize=(14, 5))
plt.plot(daily_sales['date'], daily_sales['sales'], color='blue')
plt.title("Daily Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Unit Sales")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------- PLOT 2: Heatmap doanh sá»‘ trung bÃ¬nh theo thá»© vÃ  thÃ¡ng -----------
# Táº¡o cá»™t weekday vÃ  month
df['wday'] = df['date'].dt.day_name()
df['month'] = df['date'].dt.month_name()

# Chuáº©n hÃ³a thá»© tá»± thá»© trong tuáº§n
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# TÃ­nh trung bÃ¬nh doanh sá»‘ theo thá»© vÃ  thÃ¡ng
heatmap_data = (
    df.groupby(['wday', 'month'], as_index=False)['unit_sales']
    .mean()
    .rename(columns={'unit_sales': 'mean_sales'})
)

# Pivot Ä‘á»ƒ hiá»ƒn thá»‹ heatmap
pivot_table = heatmap_data.pivot(index='wday', columns='month', values='mean_sales')
pivot_table = pivot_table.reindex(index=weekday_order, columns=month_order)

plt.figure(figsize=(14, 6))
sns.heatmap(pivot_table, cmap='Spectral', linewidths=0.5)
plt.title("Average Unit Sales by Day of Week and Month")
plt.xlabel("Month of the Year")
plt.ylabel("Day of the Week")
plt.tight_layout()
plt.show()


df = train_sample.copy()

# Tá»•ng doanh sá»‘ theo ngÃ y
daily_sales = (
    df.groupby('date', as_index=False)['unit_sales']
    .sum()
    .rename(columns={'unit_sales': 'total_sales'})
)

# TÃ­nh rolling mean (30 ngÃ y)
daily_sales['rolling_30'] = daily_sales['total_sales'].rolling(window=30, min_periods=1).mean()

# Váº½ biá»ƒu Ä‘á»“
plt.figure(figsize=(14, 6))
plt.plot(
    daily_sales['date'], daily_sales['total_sales'],
    color='lightgray', label='Daily Total Sales', alpha=0.6
)
plt.plot(
    daily_sales['date'], daily_sales['rolling_30'],
    color='red', linewidth=2, label='30-Day Rolling Average'
)
plt.title('30-Day Rolling Average â€“ Smoothed Sales Trend', fontsize=14, pad=10)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Unit Sales', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Giáº£ sá»­ báº¡n Ä‘ang dÃ¹ng báº£n train_sample (10%)
df = train_sample.copy()

# Chá»�n 1 sáº£n pháº©m vÃ  1 cá»­a hÃ ng cá»¥ thá»ƒ
item_id = 1503844
store_id = 1

# Kiá»ƒm tra xem item vÃ  store cÃ³ tá»“n táº¡i trong máº«u khÃ´ng
if (item_id in df['item_nbr'].unique()) and (store_id in df['store_nbr'].unique()):
    # Lá»�c dá»¯ liá»‡u
    sample = df[(df['item_nbr'] == item_id) & (df['store_nbr'] == store_id)]

    if not sample.empty:
        # TÃ­nh tá»•ng tÃ­ch lÅ©y theo ngÃ y
        sales_by_date = (
            sample.groupby('date', as_index=False)['unit_sales']
            .sum()
            .rename(columns={'unit_sales': 'daily_sales'})
        )
        sales_by_date['cumulative_sales'] = sales_by_date['daily_sales'].cumsum()

        # Váº½ biá»ƒu Ä‘á»“ tÃ­ch lÅ©y
        plt.figure(figsize=(14, 6))
        plt.plot(sales_by_date['date'], sales_by_date['cumulative_sales'],
                 color='blue', linewidth=2)
        plt.title(f'Cumulative Sales Over Time â€“ Item {item_id} at Store {store_id}', fontsize=14, pad=10)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Unit Sales', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print(f"KhÃ´ng tÃ¬m tháº¥y dá»¯ liá»‡u cho item {item_id} táº¡i store {store_id} trong sample.")
else:
    print(f"Item {item_id} hoáº·c store {store_id} khÃ´ng cÃ³ trong sample (10%).")


train_dates = train[['date']].drop_duplicates().copy()
train_dates['dset'] = 'train'

test_dates = test[['date']].drop_duplicates().copy()
test_dates['dset'] = 'test'

foo = pd.concat([train_dates, test_dates], ignore_index=True)
foo['year'] = foo['date'].dt.year


def safe_replace_year(x):
    try:
        return x.replace(year=2017)
    except ValueError:
        return datetime(2017, 2, 28)

foo['date'] = foo['date'].apply(safe_replace_year)
foo = foo.dropna(subset=['date'])

# =====================
# Váº½ biá»ƒu Ä‘á»“ timeline
# =====================
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(14, 4))

sns.scatterplot(
    data=foo,
    x='date',
    y='year',
    hue='dset',
    palette={'train': '#3182bd', 'test': '#de2d26'},
    s=180,
    marker='|',
    ax=ax
)


# Hiá»ƒn thá»‹ Ä‘áº§y Ä‘á»§ cÃ¡c nÄƒm
years_sorted = sorted(foo['year'].unique())
ax.set_yticks(years_sorted)
ax.set_yticklabels(years_sorted)

# Trá»¥c X hiá»ƒn thá»‹ thÃ¡ng
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%B"))
plt.xticks(rotation=45, ha='right')

# TiÃªu Ä‘á»� & nhÃ£n
plt.title("Timeline of Train/Test Date Coverage", fontsize=14, weight='bold', pad=20)
plt.xlabel("Month (2017)", labelpad=10)
plt.ylabel("Year")

# Ä�áº£o trá»¥c Y
ax.invert_yaxis()

# === Legend gÃ³c pháº£i dÆ°á»›i, náº±m ngoÃ i khung ===
plt.legend(
    title="Data set",
    loc='lower right',
    bbox_to_anchor=(1.15, -0.05),  # ğŸ”¹ Ä‘áº©y legend ra khá»�i vÃ¹ng plot
    frameon=True,
    facecolor='white',
    edgecolor='gray'
)

# === TÄƒng khoáº£ng trá»‘ng Ä‘á»ƒ khÃ´ng bá»‹ cáº¯t ===
plt.subplots_adjust(right=0.85, bottom=0.2, top=0.9)

plt.show()


foo = (
    train[['item_nbr', 'unit_sales']]
    .merge(items[['item_nbr', 'family', 'class']], on='item_nbr', how='left')
)

# --- Tá»•ng doanh sá»‘ theo family ---
sales_by_family = (
    foo.groupby('family', as_index=False)['unit_sales']
    .sum()
    .sort_values('unit_sales', ascending=False)
)
sales_by_family['family_short'] = sales_by_family['family'].str.slice(0, 19)

# --- Top 10 class bÃ¡n cháº¡y nháº¥t ---
sales_by_class = (
    foo.groupby('class', as_index=False)['unit_sales']
    .sum()
    .sort_values('unit_sales', ascending=False)
    .head(10)
)

# === Váº½ layout 2 biá»ƒu Ä‘á»“ ngang ===
sns.set(style="whitegrid")
fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# --- Biá»ƒu Ä‘á»“ 1: Sales theo Family (náº±m ngang + log scale) ---
ax1 = plt.subplot(gs[0])
sns.barplot(
    data=sales_by_family.head(20),  # hiá»ƒn thá»‹ top 20 family cho dá»… nhÃ¬n
    y='family_short',
    x='unit_sales',
    palette='crest',
    ax=ax1
)
ax1.set_xscale('log')
ax1.set_title("Top 20 Item Families by Total Sales", fontsize=14, weight='bold', pad=10)
ax1.set_xlabel("Total Sales (log scale)", fontsize=12)
ax1.set_ylabel("Item Family", fontsize=12)
ax1.tick_params(axis='y', labelsize=9)
sns.despine(ax=ax1, left=True, bottom=True)

# --- Biá»ƒu Ä‘á»“ 2: Top 10 Class ---
ax2 = plt.subplot(gs[1])
sns.barplot(
    data=sales_by_class,
    y='class',
    x='unit_sales',
    palette='coolwarm',
    ax=ax2
)
ax2.set_title("Top 10 Best-Selling Item Classes", fontsize=13, pad=10)
ax2.set_xlabel("Total Sales", fontsize=12)
ax2.set_ylabel("Item Class", fontsize=12)
ax2.tick_params(axis='y', labelsize=9)
sns.despine(ax=ax2, left=True, bottom=True)

# --- Ä�iá»�u chá»‰nh bá»‘ cá»¥c ---
plt.tight_layout(pad=3)
plt.subplots_adjust(right=0.95, bottom=0.08, top=0.93)
plt.show()


# --- Tá»•ng sales theo ngÃ y ---
foo = train.groupby('date', as_index=False)['unit_sales'].sum().rename(columns={'unit_sales': 'sales'})

# --- Lá»�c dá»¯ liá»‡u oil theo thá»�i gian tÆ°Æ¡ng á»©ng ---
oil_back = oil[oil['date'] > foo['date'].min()].copy()

# --- Chuáº©n hÃ³a oilprice theo tá»· lá»‡ sales ---
min_sales, max_sales = foo['sales'].min(), foo['sales'].max()
min_oil, max_oil = oil_back['dcoilwtico'].min(), oil_back['dcoilwtico'].max()

oil_back['oilprice_scaled'] = (
    min_sales +
    (oil_back['dcoilwtico'] - min_oil) / (max_oil - min_oil) * (max_sales - min_sales)
)

# --- Váº½ biá»ƒu Ä‘á»“ ---
plt.figure(figsize=(14,6))

# Tá»•ng doanh sá»‘ (Ä‘en)
plt.plot(foo['date'], foo['sales'], color='black', label='Total Sales')

# GiÃ¡ dáº§u Ä‘Ã£ scale (xanh)
plt.plot(oil_back['date'], oil_back['oilprice_scaled'], color='blue', label='Oil Price (scaled)')

# TiÃªu Ä‘á»� vÃ  format
plt.title("Total Sales (black) with Oil Price (blue)", fontsize=14, weight='bold')
plt.xlabel("Date")
plt.ylabel("Total Sales (scaled units)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


print(train)
print("=")
print(oil)


foo = (
    train.groupby('date', as_index=False)['unit_sales']
    .sum()
    .rename(columns={'unit_sales': 'sales'})
)


bar = (
    foo.merge(oil, on='date', how='left')
    .assign(
        oil1=lambda df: df['dcoilwtico'] - df['dcoilwtico'].shift(1),
        oil7=lambda df: df['dcoilwtico'] - df['dcoilwtico'].shift(7),
        oil30=lambda df: df['dcoilwtico'] - df['dcoilwtico'].shift(30),
        sales1=lambda df: df['sales'] - df['sales'].shift(1),
        sales7=lambda df: df['sales'] - df['sales'].shift(7),
        sales30=lambda df: df['sales'] - df['sales'].shift(30)
    )
    .dropna(subset=['oil1', 'oil7', 'oil30', 'sales1', 'sales7', 'sales30'])
)


cols = ['oil1', 'oil7', 'oil30', 'sales1', 'sales7', 'sales30']
corr_matrix = bar[cols].corr(method='spearman')

corr_matrix = corr_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)
corr_matrix = corr_matrix.fillna(0)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    mask=mask,                # chá»‰ hiá»‡n tam giÃ¡c dÆ°á»›i
    annot=True, fmt=".2f",    # hiá»ƒn thá»‹ sá»‘, lÃ m trÃ²n 2 chá»¯ sá»‘
    cmap="coolwarm", center=0,
    cbar=True, square=True,
    linewidths=0.5,
    annot_kws={"size": 10, "weight": "bold"}
)
plt.title("Spearman Correlation: Oil vs Sales Changes (1, 7, 30 days)", fontsize=14, weight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# --- 1. TÃ­nh median doanh sá»‘ cho tá»«ng sáº£n pháº©m theo tráº¡ng thÃ¡i khuyáº¿n mÃ£i ---
foo = (
    train.groupby(['item_nbr', 'onpromotion'], as_index=False)['unit_sales']
    .median()
    .rename(columns={'unit_sales': 'med_sales'})
    .dropna(subset=['onpromotion'])
)

# --- 2. Pivot Ä‘á»ƒ tÃ¡ch TRUE / FALSE ---
pivot = foo.pivot(index='item_nbr', columns='onpromotion', values='med_sales').reset_index()
pivot.columns.name = None
pivot = pivot.rename(columns={True: 'promo_true', False: 'promo_false'})

# --- 3. Giá»¯ láº¡i chá»‰ cÃ¡c sáº£n pháº©m cÃ³ cáº£ 2 loáº¡i dá»¯ liá»‡u ---
pivot = pivot.dropna(subset=['promo_true', 'promo_false'])

# --- 4. Gom láº¡i dá»¯ liá»‡u Ä‘á»ƒ váº½ ---
foo_long = pivot.melt(
    id_vars='item_nbr',
    value_vars=['promo_true', 'promo_false'],
    var_name='promo',
    value_name='med_sales'
)
foo_long['promo'] = foo_long['promo'].replace({'promo_true': 'Promotion', 'promo_false': 'No Promotion'})

# --- 5. Káº¿t há»£p vá»›i thÃ´ng tin sáº£n pháº©m ---
bar = (
    foo_long.merge(items, on='item_nbr', how='left')
)

# --- 6. Biá»ƒu Ä‘á»“ 1: So sÃ¡nh median sales giá»¯a cÃ³ vÃ  khÃ´ng khuyáº¿n mÃ£i ---
plt.figure(figsize=(8,6))
sns.boxplot(data=foo_long, x='promo', y='med_sales', palette='Set2')
plt.yscale('log')
plt.title("Median Sales Distribution: With vs Without Promotion", fontsize=14, weight='bold')
plt.xlabel("Promotion Status")
plt.ylabel("Median Unit Sales (log scale)")
plt.tight_layout()
plt.show()

# --- 7. Biá»ƒu Ä‘á»“ 2: So sÃ¡nh theo nhÃ³m sáº£n pháº©m (family) ---
bar['family_short'] = bar['family'].str.slice(0, 19)

plt.figure(figsize=(14,7))
sns.boxplot(
    data=bar,
    x='family_short',
    y='med_sales',
    hue='promo',
    palette='coolwarm'
)
plt.yscale('log')
plt.title("Median Sales by Product Family (Promotion vs No Promotion)", fontsize=14, weight='bold')
plt.xlabel("Product Family (truncated)")
plt.ylabel("Median Unit Sales (log scale)")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Promotion")
plt.tight_layout()
plt.show()


sns.set(style="whitegrid")

# --- Biá»ƒu Ä‘á»“ 1: Sá»‘ lÆ°á»£ng ngÃ y lá»… theo loáº¡i (type) ---
plt.figure(figsize=(6,4))
sns.countplot(data=holidays, x='type', palette='Set2')
plt.title("Number of Holidays by Type", fontsize=13, weight='bold')
plt.xlabel("Holiday Type")
plt.ylabel("Count")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# --- Biá»ƒu Ä‘á»“ 2: Sá»‘ lÆ°á»£ng ngÃ y lá»… theo khu vá»±c (locale) ---
plt.figure(figsize=(6,4))
sns.countplot(data=holidays, x='locale', palette='coolwarm')
plt.title("Number of Holidays by Locale", fontsize=13, weight='bold')
plt.xlabel("Locale (National / Regional / Local)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- Biá»ƒu Ä‘á»“ 3: 12 mÃ´ táº£ ngÃ y lá»… phá»• biáº¿n nháº¥t ---
desc_freq = (
    holidays.groupby('description')
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
    .head(12)
)

print(desc_freq.head())  # giá»� sáº½ ra ['description', 'count']
plt.figure(figsize=(8,5))
sns.barplot(
    data=desc_freq,
    y='description',
    x='count',
    color='skyblue'
)
plt.title("Top 12 Most Frequent Holiday Descriptions", fontsize=13, weight='bold')
plt.xlabel("Frequency")
plt.ylabel("Holiday Description")
plt.tight_layout()
plt.show()



import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Gá»™p dá»¯ liá»‡u train vá»›i thÃ´ng tin cá»­a hÃ ng ---
foo = (
    train[['store_nbr', 'date', 'unit_sales']]
    .merge(stores[['store_nbr', 'cluster']], on='store_nbr', how='left')
)

# --- Biá»ƒu Ä‘á»“ 1: Total Sales by Store Cluster ---
sales_by_cluster = (
    foo.groupby('cluster', as_index=False)['unit_sales']
    .sum()
    .sort_values('unit_sales', ascending=False)
)

plt.figure(figsize=(8,5))
sns.barplot(
    data=sales_by_cluster,
    x='cluster',
    y='unit_sales',
    palette='crest'
)
plt.title("Total Sales by Store Cluster", fontsize=14, weight='bold')
plt.xlabel("Store Cluster")
plt.ylabel("Total Unit Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Biá»ƒu Ä‘á»“ 2: Sales Trend by Cluster (Time Series) ---
sales_trend = (
    foo.groupby(['date', 'cluster'], as_index=False)['unit_sales']
    .sum()
)

plt.figure(figsize=(12,6))
sns.lineplot(
    data=sales_trend,
    x='date',
    y='unit_sales',
    hue='cluster',
    palette='tab10'
)
plt.title("Sales Trend by Cluster (Time Series)", fontsize=14, weight='bold')
plt.xlabel("Date")
plt.ylabel("Total Unit Sales")
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --- Gá»™p dá»¯ liá»‡u train + items ---
foo = (
    train[['item_nbr', 'date', 'unit_sales']]
    .merge(items[['item_nbr', 'family']], on='item_nbr', how='left')
)

# ---  Top 10 Product Families by Total Sales ---
sales_by_family = (
    foo.groupby('family', as_index=False)['unit_sales']
    .sum()
    .sort_values('unit_sales', ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
sns.barplot(
    data=sales_by_family,
    x='unit_sales',
    y='family',
    palette='crest'
)
plt.title("Top 10 Product Families by Total Sales", fontsize=14, weight='bold')
plt.xlabel("Total Unit Sales")
plt.ylabel("Product Family")
plt.tight_layout()
plt.show()

# ---  Seasonal Pattern by Family (family Ã— month) ---
foo['month'] = foo['date'].dt.month_name()

# TÃ­nh doanh sá»‘ trung bÃ¬nh theo thÃ¡ng vÃ  family
seasonal_pattern = (
    foo.groupby(['family', 'month'], as_index=False)['unit_sales']
    .mean()
)

# Sáº¯p xáº¿p thÃ¡ng Ä‘Ãºng thá»© tá»±
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
seasonal_pattern['month'] = pd.Categorical(seasonal_pattern['month'], categories=month_order, ordered=True)

plt.figure(figsize=(14,6))
sns.heatmap(
    seasonal_pattern.pivot_table(index='family', columns='month', values='unit_sales'),
    cmap='YlGnBu',
    linewidths=0.5
)
plt.title("Seasonal Pattern by Product Family (Average Monthly Sales)", fontsize=14, weight='bold')
plt.xlabel("Month")
plt.ylabel("Product Family")
plt.tight_layout()
plt.show()


# === Giáº£ sá»­ báº¡n Ä‘Ã£ cÃ³: train_sample (10%) vÃ  stores ===
# train_sample: cá»™t ['date', 'store_nbr', 'unit_sales']
# stores: cá»™t ['store_nbr', 'state', 'city', 'cluster', ...]

# Gá»™p dá»¯ liá»‡u vá»›i thÃ´ng tin cá»­a hÃ ng
train_store = train_sample.merge(stores, on='store_nbr', how='left')

# Lá»�c ra 2 tá»‰nh chá»‹u áº£nh hÆ°á»Ÿng Ä‘á»™ng Ä‘áº¥t
eq_states = ['Manabi', 'Esmeraldas']
region_sales = (
    train_store[train_store['state'].isin(eq_states)]
    .groupby('date', as_index=False)['unit_sales']
    .sum()
)

# TÃ­nh trung bÃ¬nh trÆ°á»£t (14 ngÃ y)
region_sales['rolling_mean'] = region_sales['unit_sales'].rolling(window=14, center=True).mean()

# === Váº½ biá»ƒu Ä‘á»“ ===
plt.figure(figsize=(12,6))
sns.lineplot(data=region_sales, x='date', y='unit_sales', color='tomato', alpha=0.5, label='Daily Sales')
sns.lineplot(data=region_sales, x='date', y='rolling_mean', color='red', linewidth=2, label='14-day Rolling Avg')

# NgÃ y xáº£y ra Ä‘á»™ng Ä‘áº¥t
earthquake_date = pd.to_datetime('2016-04-16')

# Ä�Ã¡nh dáº¥u trÃªn biá»ƒu Ä‘á»“
plt.axvline(earthquake_date, color='black', linestyle='--', linewidth=1.3)
plt.text(earthquake_date, region_sales['unit_sales'].max()*0.9,
         'Earthquake\n2016-04-16', rotation=90, ha='right', fontsize=9, color='black')

# TÃ¹y chá»‰nh trá»¥c & style
plt.title('Impact of 2016 Earthquake on Sales (ManabÃ­ & Esmeraldas)', fontsize=14, weight='bold')
plt.xlabel('Date')
plt.ylabel('Total Unit Sales (10% Sample)')
plt.grid(alpha=0.3)
plt.legend()
plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y'))
plt.tight_layout()
plt.show()



# === Ä�á»�c dá»¯ liá»‡u cáº§n thiáº¿t ===
oil_path = "/kaggle/working/oil.csv"
holidays_path = "/kaggle/working/holidays_events.csv"

oil = pd.read_csv(oil_path, parse_dates=['date'])
holidays = pd.read_csv(holidays_path, parse_dates=['date'])

# --- Giáº£ sá»­ báº¡n Ä‘Ã£ cÃ³ train_sample = 10% dá»¯ liá»‡u ---
# train_sample: ['date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion']

# Tá»•ng doanh sá»‘ theo ngÃ y (chá»‰ tá»« máº«u 10%)
sales_daily = (
    train_sample.groupby('date', as_index=False)['unit_sales']
    .sum()
    .rename(columns={'unit_sales': 'total_sales'})
)

# ThÃªm giÃ¡ dáº§u
merged = sales_daily.merge(oil, on='date', how='left')

# TÃ­nh tá»•ng sá»‘ sáº£n pháº©m Ä‘ang Ä‘Æ°á»£c khuyáº¿n mÃ£i má»—i ngÃ y
promo_daily = (
    train_sample.groupby('date', as_index=False)['onpromotion']
    .sum()
    .rename(columns={'onpromotion': 'total_promos'})
)
merged = merged.merge(promo_daily, on='date', how='left')

# ThÃªm cá»™t Ä‘Ã¡nh dáº¥u ngÃ y lá»…
holiday_days = holidays['date'].unique()
merged['holiday_flag'] = merged['date'].isin(holiday_days).astype(int)

# Chuáº©n hÃ³a tÃªn cá»™t (Ä‘á»ƒ dá»… nhÃ¬n khi váº½)
merged = merged.rename(columns={'dcoilwtico': 'oil_price'})

# --- TÃ­nh tÆ°Æ¡ng quan ---
corr = merged[['total_sales', 'oil_price', 'total_promos', 'holiday_flag']].corr()

# --- Váº½ Heatmap ---
plt.figure(figsize=(7,5))
sns.heatmap(
    corr, annot=True, cmap='coolwarm', fmt=".2f", center=0,
    linewidths=0.5, cbar_kws={"shrink": 0.8}
)
plt.title("Correlation between Sales and External Factors (10% Sample)", fontsize=13, weight='bold')
plt.show()



train['day_of_week'] = train['date'].dt.dayofweek 
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train['week_of_year'] = train['date'].dt.isocalendar().week
train['is_weekend'] = train['day_of_week'].isin([5, 6]).astype(int)


print(train)


train_sample['sales_lag_7'] = train.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(7)
oil['oil_price_lag_7'] = oil['dcoilwtico'].shift(7)
transactions['transactions_lag_7'] = transactions['transactions'].shift(7)

