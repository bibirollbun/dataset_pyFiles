import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
import matplotlib.patheffects as pe

from sklearn.preprocessing import LabelEncoder
import calendar
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import xgboost as xgb

pd.options.display.max_columns = 500

import os
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


real_train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv', parse_dates=['Date'])
real_test = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv', parse_dates=['Date'])
real_store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

data_train = real_train.copy()
data_test = real_test.copy()
data_store = real_store.copy()


print("="*25, '\n Data Train')
display(data_train.head(2))
display(data_train.shape)

print("="*25,  '\n Data Test')
display(data_test.head(2))
display(data_test.shape)

print("="*25,  '\n Store Data')
display(data_store.head(2))
display(data_store.shape)


# === Data Profiling ===
def data_quality_metric(df):
    print('=== Data Count ===')
    print('Count of data length:', df.shape)
    print()
    
    print('=== Completeness ===')
    print('Number of null data:')
    print(df.isnull().sum())
    print()
    
    print('=== Consistency ===')
    for col in df.columns:
        if col != 'Id':
            display(df[col].value_counts().sort_values(ascending=False))
    print()
        
    print("=== Duplication ===")
    print('Count of duplicated:')
    print(df.duplicated().sum())
    print()
    
    print("=== Number of unique ===")
    print('Count of uniqueness:')
    print(data_train.nunique())
    print()


data_quality_metric(data_train)


data_quality_metric(data_test)


data_quality_metric(data_store)


ts_data = pd.merge(data_train, data_store, how='left', left_on=['Store'], right_on=['Store']).copy()


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates

ts_data_one = ts_data.groupby(['Date'])['Sales'].mean()
result = seasonal_decompose(ts_data_one, model='additive')

fig = result.plot()
fig.set_size_inches(12, 8)
fig.suptitle('Time Series Decomposition', fontsize=16)

# Set x-axis to show years only
for ax in fig.axes:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.show()


max_year = ts_data.Date.dt.year.max()
min_observed_date = ts_data[ts_data.Date.dt.year == max_year].Date.min()
max_observed_date = ts_data[ts_data.Date.dt.year == max_year].Date.max()

previous_observed_date = ts_data[(ts_data.Date >= (min_observed_date - pd.Timedelta(days=365))) & (ts_data.Date <= (max_observed_date - pd.Timedelta(days=365)))]
This_year_observed_date = ts_data[(ts_data.Date >= min_observed_date) & (ts_data.Date <= max_observed_date)]
This_year_sales_comparison = (This_year_observed_date.Sales.mean() - previous_observed_date.Sales.mean())
This_year_sales_comparison


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Compute mean sales
mean_prev = previous_observed_date['Sales'].mean()
mean_this = This_year_observed_date['Sales'].mean()

# Create a DataFrame for plotting
comparison_df = pd.DataFrame({
    'Tahun': ['Tahun Lalu', 'Tahun Ini'],
    'Rata-rata Penjualan': [mean_prev, mean_this]
})

# Plot
plt.figure(figsize=(8, 6))
sns.set_style("white")  # Clean white background, no grid

barplot = sns.barplot(
    data=comparison_df,
    x='Tahun',
    y='Rata-rata Penjualan',
    palette='pastel',
    width=0.6
)

# Annotate values
for i, value in enumerate(comparison_df['Rata-rata Penjualan']):
    plt.text(i, value + value * 0.05, f'{value:,.0f}', 
             ha='center', va='bottom', fontsize=12, weight='bold', color='#333')

# Set y-axis limit to 20% higher for breathing room
plt.ylim(0, max(comparison_df['Rata-rata Penjualan']) * 1.2)

# Titles and formatting
plt.title('Perbandingan Rata-rata Penjualan per Tahun', fontsize=16, weight='bold')
plt.xlabel('')
plt.ylabel('')
sns.despine(left=True, bottom=True)  # Remove borders

plt.xticks(fontsize=12, weight='medium')
plt.yticks([])  # Remove y-ticks for clean look
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

ts_data_weekday = ts_data[ts_data['Date'].dt.weekday <= 5]

plt.figure(figsize=(30, 6))
sns.set_theme(style="whitegrid")
sns.lineplot(
    data=ts_data_weekday.groupby('Date')['Sales'].mean().reset_index(),
    x='Date',
    y='Sales',
    color='#007acc',
    linewidth=2,
    alpha=0.9,
    label='Daily Sales'
)
plt.title('Tren Penjualan Harian', fontsize=20, weight='bold')
plt.xlabel('Tanggal', fontsize=14)
plt.ylabel('Penjualan', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(visible=True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

years = sorted(ts_data_weekday.Date.dt.year.unique())
sns.set_style("whitegrid")
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

colors = sns.color_palette("pastel", n_colors=len(years))

for idx, year in enumerate(years):
    # Create full business-day date range for the year
    full_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='B')
    full_df = pd.DataFrame({'Date': full_dates})

    # Aggregate existing data
    yearly_data = ts_data_weekday[ts_data_weekday.Date.dt.year == year].copy()
    daily_sales = yearly_data.groupby('Date')['Sales'].mean().reset_index()

    # Merge actual data with full date range
    full_sales = pd.merge(full_df, daily_sales, on='Date', how='left')

    # Ensure Sales is float so NaN doesn't break plotting
    full_sales['Sales'] = full_sales['Sales'].astype(float)

    # Plot
    plt.figure(figsize=(14, 5))
    ax = sns.lineplot(
        data=full_sales,
        x='Date',
        y='Sales',
        color=colors[idx],
        linewidth=1.8
    )

    # Force x-axis to full range
    ax.set_xlim([f'{year}-01-01', f'{year}-12-31'])

    plt.title(f'Penjualan Harian Tahun {year}', weight='bold')
    plt.xlabel('Tanggal')
    plt.ylabel('Total Penjualan')
    plt.grid(False)
    sns.despine()
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure date column is datetime
ts_data['Date'] = pd.to_datetime(ts_data['Date'])

# Add helper columns
ts_data['Year'] = ts_data['Date'].dt.year
ts_data['DayOfWeek'] = ts_data['Date'].dt.dayofweek + 1

import seaborn as sns
import matplotlib.pyplot as plt

def plot_grouped_line(df, x, y='Sales', group='Year', title=None):
    # Group and aggregate
    grouped_df = df.groupby([group, x])[y].mean().reset_index()

    # Plot
    plt.figure(figsize=(16, 6))
    palette = sns.color_palette("Set2", n_colors=grouped_df[x].nunique())

    sns.lineplot(
        data=grouped_df,
        x=x,
        y=y,
        hue=group,
        palette=palette,
        linewidth=2
    )

    plt.title(title or f'Average {y} by {x} and {x}', fontsize=16, weight='bold')
    plt.xlabel(x, fontsize=12)
    plt.ylabel(f'Average {y}', fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.legend(title=x, loc='upper right', frameon=False)
    plt.show()


# Ensure 'Year' and 'DayOfWeek' are extracted first
ts_data['DayOfWeek'] = ts_data['Date'].dt.dayofweek
ts_data['Year'] = ts_data['Date'].dt.year

plot_grouped_line(ts_data, x='DayOfWeek', y='Sales', title='Average Sales by Day of Week per Year')


del ts_data_weekday, This_year_observed_date, This_year_sales_comparison, mean_prev, mean_this, yearly_data, years, previous_observed_date


store_type_merged = pd.merge(left=ts_data.groupby(['StoreType']).Sales.mean().reset_index(), right=ts_data.StoreType.value_counts().reset_index().rename(columns={'count':'StoreCount'}), on='StoreType', how='inner')
store_type_merged.columns = ['StoreType', 'AvgSales', 'StoreCount']

store_type_merged


store_type_merged = pd.merge(left=ts_data.groupby(['StoreType']).Sales.mean().reset_index(), right=ts_data.StoreType.value_counts().reset_index().rename(columns={'count':'StoreCount'}), on='StoreType', how='inner')
store_type_merged.columns = ['StoreType', 'AvgSales', 'StoreCount']

sns.set_theme(style="whitegrid", context='talk')
palette = sns.color_palette("pastel")
fig, ax = plt.subplots(figsize=(10, 6))

barplot = sns.barplot(
    data=store_type_merged,
    x='StoreType',
    y='AvgSales',
    palette=palette,
    edgecolor='black',
    ax=ax
)

for index, row in store_type_merged.iterrows():
    ax.text(
        index,
        row['AvgSales'] + 0.02 * store_type_merged['AvgSales'].max(),
        f"{row['StoreCount']} stores",
        ha='center',
        va='bottom',
        fontsize=11,
        weight='semibold',
        color='#333333',
        path_effects=[pe.withStroke(linewidth=2, foreground='white')]  # subtle outline for visibility
    )

ax.set_title('Rata-rata Penjualan per Tipe Toko', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('Tipe Toko', fontsize=13)
ax.set_ylabel('Rata-rata Penjualan', fontsize=13)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.set_ylim(0, store_type_merged['AvgSales'].max() * 1.15)

plt.tight_layout()
plt.show()


# Create bins: 0-500, 501-1000, ..., 4001–4500
bin_edges = list(range(0, 8000, 500))
bin_labels = [f"{bin_edges[i]}–{bin_edges[i+1]-1}" for i in range(len(bin_edges)-1)]
ts_data['DistanceBin'] = pd.cut(ts_data['CompetitionDistance'], bins=bin_edges, labels=bin_labels, right=False)

# Prepare plot
sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharey=True)
axes = axes.flatten()

# Plot each bin
for i, bin_label in enumerate(bin_labels):
    ax = axes[i]
    
    bin_df = ts_data[ts_data['DistanceBin'] == bin_label]
    grouped = (
        bin_df.groupby('CompetitionDistance')['Sales']
        .mean()
        .reset_index()
        .sort_values('CompetitionDistance')
    )
    
    if not grouped.empty:
        grouped['SmoothedSales'] = grouped['Sales'].rolling(window=5, center=True).mean()
        sns.lineplot(
            data=grouped,
            x='CompetitionDistance',
            y='SmoothedSales',
            color=sns.color_palette("Set2")[i % len(sns.color_palette("Set2"))],
            linewidth=2.2,
            ax=ax
        )
    
    ax.set_title(f"Jarak {bin_label} m", fontsize=13, weight='bold')
    ax.set_xlabel("Kompetitor (m)", fontsize=11)
    ax.set_ylabel("Rata-rata Penjualan", fontsize=11)
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

# Final layout
plt.tight_layout()
fig.suptitle('Tren Penjualan berdasarkan Jarak Kompetitor (per 500m)', fontsize=18, weight='bold', y=1.02)
plt.show()


assortment_level = pd.merge(left=ts_data.groupby(['Assortment']).Sales.mean().reset_index(), right=ts_data.Assortment.value_counts().reset_index().rename(columns={'count': 'AssortmentCount'}), on='Assortment', how='inner')
assortment_level.columns = ['AssortmentType', 'AvgSales', 'AssortmentCount']

sns.set_theme(style="whitegrid", context='talk')
palette = sns.color_palette("pastel")
fig, ax = plt.subplots(figsize=(10, 6))

barplot = sns.barplot(
    data=assortment_level,
    x='AssortmentType',
    y='AvgSales',
    palette=palette,
    edgecolor='black',
    ax=ax
)

for index, row in assortment_level.iterrows():
    ax.text(
        index,
        row['AvgSales'] + 0.02 * assortment_level['AvgSales'].max(),
        f"{row['AssortmentCount']}",
        ha='center',
        va='bottom',
        fontsize=11,
        weight='semibold',
        color='#333333',
        path_effects=[pe.withStroke(linewidth=2, foreground='white')]  # subtle outline for visibility
    )

ax.set_title('Rata-rata Penjualan per Level Assortment', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('Level Assortment', fontsize=13)
ax.set_ylabel('Rata-rata Penjualan', fontsize=13)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.set_ylim(0, assortment_level['AvgSales'].max() * 1.15)

plt.tight_layout()
plt.show()


# Step 1: Create HolidayStatus column
ts_data['HolidayStatus'] = ts_data.apply(
    lambda row: 'Holiday' if (row['StateHoliday'] == 1 or row['SchoolHoliday'] == 1) else 'Non-Holiday',
    axis=1
)

# Step 2: Downsample if dataset is large
sample_size = 3000
sampled_data = ts_data.sample(n=sample_size, random_state=42) if len(ts_data) > sample_size else ts_data

# Step 3: Limit sales range to exclude extreme outliers (for clarity)
sales_cap = sampled_data['Sales'].quantile(0.99)

# Step 4: Plot setup
sns.set_theme(style="whitegrid", context='notebook')
fig, ax = plt.subplots(figsize=(10, 6))

# Custom color palette
palette = {'Holiday': '#FFA07A', 'Non-Holiday': '#87CEFA'}

# Swarmplot
sns.swarmplot(
    data=sampled_data,
    x='HolidayStatus',
    y='Sales',
    palette=palette,
    size=4,
    edgecolor='gray',
    ax=ax
)

# Title and labels
ax.set_title(
    'Distribusi Penjualan: Holiday vs Non-Holiday',
    fontsize=18,
    weight='bold',
    pad=20
)
ax.set_xlabel('Status Hari', fontsize=14)
ax.set_ylabel('Penjualan (dalam unit)', fontsize=14)

# Axes styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_ylim(0, sales_cap)
ax.set_axisbelow(True)

# Annotation: sample size
for label in ['Holiday', 'Non-Holiday']:
    count = (sampled_data['HolidayStatus'] == label).sum()
    x_pos = 0 if label == 'Holiday' else 1
    ax.text(
        x_pos,
        sales_cap * 0.95,
        f'n = {count}',
        ha='center',
        fontsize=11,
        weight='semibold',
        color='#333333',
        path_effects=[pe.withStroke(linewidth=2, foreground='white')]
    )

plt.tight_layout()
plt.show()


# Step 1: Create Promo Category
ts_data['PromoCategory'] = ts_data.apply(
    lambda row: 'Both Promo1 & Promo2' if row['Promo'] == 1 and row['Promo2'] == 1
    else 'No Promo' if row['Promo'] == 0 and row['Promo2'] == 0
    else 'Promo1 only' if row['Promo'] == 1 else 'Promo2 only',
    axis=1
)

# Step 2: Group by and calculate mean sales
promo_summary = ts_data.groupby('PromoCategory')['Sales'].mean().reset_index().sort_values('Sales', ascending=False)

# Step 3: Plot
sns.set_theme(style="whitegrid", context="notebook")
fig, ax = plt.subplots(figsize=(10, 6))

barplot = sns.barplot(
    data=promo_summary,
    x='PromoCategory',
    y='Sales',
    palette=['#FF9999', '#66B2FF', '#FFD700', '#B0C4DE'],
    ax=ax
)

# Step 4: Add data labels
for p in barplot.patches:
    barplot.annotate(
        f'{p.get_height():,.0f}',
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='semibold',
        color='black'
    )

# Step 5: Aesthetic settings
ax.set_title('Rata-Rata Penjualan Berdasarkan Status Promo', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('')
ax.set_ylabel('Rata-Rata Penjualan', fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


ts_data['PromoStatus'] = ts_data.apply(
    lambda row: 'Promotion' if row['Promo'] == 1 or row['Promo2'] == 1 else 'No Promotion',
    axis=1
)
daily_sales = ts_data.groupby(['Date', 'PromoStatus'])['Sales'].mean().reset_index()
promo_counts = ts_data['PromoStatus'].value_counts().rename(index={1: 'With Promo', 0: 'No Promo'})

plt.figure(figsize=(14, 6))
sns.set_theme(style='whitegrid', context='talk')
explode = (0.05, 0)

plt.pie(
    promo_counts,
    labels=promo_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    explode=explode,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)

# Aesthetic Tweaks
plt.title('Proporsi Penjualan Dengan Promo vs Tanpa Promo', fontsize=16, weight='bold', pad=20)
plt.tight_layout()
plt.show()


# Clean the data
promo_interval_data = ts_data.dropna(subset=['PromoInterval'])
promo_interval_data['PromoInterval'] = promo_interval_data['PromoInterval'].astype(str)

# Set visual style
sns.set_theme(style="white", context="talk")

# Plot with narrow figure size
plt.figure(figsize=(8, 6))  # reduced width = tighter category spacing
sns.stripplot(
    data=promo_interval_data,
    x='PromoInterval',
    y='Sales',
    jitter=0.1,         # minimal horizontal jitter
    alpha=0.5,
    size=4,
    palette='Set2'
)

# Add median markers
medians = promo_interval_data.groupby('PromoInterval')['Sales'].median()
for i, (label, median) in enumerate(medians.items()):
    plt.plot(i, median, 'D', color='black', markersize=6, label='Median' if i == 0 else "")

# Polish
plt.grid(False)
plt.title('Tight Dot Plot Penjualan Berdasarkan PromoInterval', fontsize=16, weight='bold', pad=15)
plt.xlabel('PromoInterval', fontsize=13)
plt.ylabel('Penjualan', fontsize=13)
plt.tight_layout()
sns.despine()
plt.show()


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def plot_correlation_heatmap(df: pd.DataFrame, data_type: str = 'numerical', drop_cols: list = None):
    drop_cols = drop_cols or []
    
    if data_type == 'numerical':
        data = df.select_dtypes(include=['number']).drop(columns=drop_cols, errors='ignore')
    elif data_type == 'categorical':
        data = pd.get_dummies(df.select_dtypes(include=['object', 'category', 'bool']), drop_first=True)
        data['Sales'] = df.Sales
    else:
        raise ValueError("data_type must be 'numerical' or 'categorical'")

    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(16, 12))
    sns.set_theme(style='white')
    sns.heatmap(
        corr, mask=mask, cmap='coolwarm', center=0, square=True,
        linewidths=0.5, annot=True, fmt='.2f',
        cbar_kws={'shrink': 0.7, 'label': 'Correlation'}
    )
    plt.title(f'{data_type.capitalize()} Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()

df = ts_data.copy()

# For numerical
plot_correlation_heatmap(df, data_type='numerical', drop_cols=['Id', 'Sales', 'Promo2'])

# For categorical
plot_correlation_heatmap(df, data_type='categorical')


data_train.info()


data_train['Date'] = pd.to_datetime(data_train['Date'])
data_test['Date'] = pd.to_datetime(data_test['Date'])

# display(data_train.drop(columns=['Date']).groupby(data_train.DayOfWeek).sum())
display(data_train.Promo.shift(-1).value_counts())
display(data_train.Promo.shift(1).value_counts())


real_train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv', parse_dates=['Date'])
real_test = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv', parse_dates=['Date'])
real_store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

data_train = real_train.copy()
data_test = real_test.copy()
data_store = real_store.copy()


def monthToNum(value):
    if(value=='Sept'):
        value='Sep'
    return list(calendar.month_abbr).index(value)

join_with = data_store['PromoInterval'].str.split(',').apply(pd.Series)
join_with.columns = join_with.columns.map(lambda x: str(x) + '_PromoInterval')
data_store = data_store.join(join_with) #joining splits

data_store['0_PromoInterval'] = data_store['0_PromoInterval'].map(lambda x: monthToNum(x) if str(x) != 'nan' else np.nan)
data_store['1_PromoInterval'] = data_store['1_PromoInterval'].map(lambda x: monthToNum(x) if str(x) != 'nan' else np.nan)
data_store['2_PromoInterval'] = data_store['2_PromoInterval'].map(lambda x: monthToNum(x) if str(x) != 'nan' else np.nan)
data_store['3_PromoInterval'] = data_store['3_PromoInterval'].map(lambda x: monthToNum(x) if str(x) != 'nan' else np.nan)

competition_open = []
for index, value in data_store[['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']].iterrows():
    try: 
        year, month = int(value['CompetitionOpenSinceYear']), int(value['CompetitionOpenSinceMonth'])
        date = pd.to_datetime(f'{year}-{month}-01', format='%Y-%m')
        competition_open.append(date)
    except:
        competition_open.append(np.nan)

competition_open = pd.Series(competition_open)
data_store['CompetitionOpen'] = competition_open #converted int to datetime
data_store['CompetitionOpen'] = pd.to_datetime(data_store['CompetitionOpen'], errors='coerce')
data_store['CompetitionOpen'] = data_store.CompetitionOpen.dt.strftime('%Y%m%d')

promo = []
for index, value in data_store[['Promo2SinceWeek', 'Promo2SinceYear']].iterrows():
    try:
        year, week = int(value['Promo2SinceYear']), int(value['Promo2SinceWeek'])
        date = pd.to_datetime("{}-{}-01".format(year, week), format='%Y%W')
        promo.append(date)
    except:
        promo.append(np.nan)

promo = pd.to_datetime(pd.Series(promo))
data_store['PromoSince'] = promo #converted int to datetime
data_store['PromoSince'] = pd.to_datetime(data_store['PromoSince'], errors='coerce')
data_store['PromoSince'] = data_store.PromoSince.dt.strftime('%Y%m%d')

for col in ['0_PromoInterval', '1_PromoInterval', '2_PromoInterval', '3_PromoInterval', 'CompetitionOpen', 'PromoSince']:
    data_store[col] = data_store[col].fillna(-999)

data_store.drop(inplace=True, columns=['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear'])


def feature_engineering(df, df_2=None, data_store=None, viz=False):    
    data_model = df.loc[~((df['Open'] == 1) & (df['Sales'] == 0))].copy()
    data_predict = df_2.copy() if df_2 is not None else None
    
    data_model = pd.merge(data_model, data_store, on='Store', how='left')
    if data_predict is not None:
        data_predict = pd.merge(data_predict, data_store, on='Store', how='left')
    
    data_model['is_train'] = 1
    data_predict['is_train'] = 0
        
    df_all = pd.concat([data_model, data_predict], axis=0)
    
    df_all['StateHoliday'] = df_all['StateHoliday'].map({0: '0', '0': '0', 'a': 'a', 'b': 'b', 'c': 'c'})    
            
    sales_per_store_day = df_all.groupby([df_all['Store']])['Sales'].sum()
    store_per_customer_day = df_all.groupby('Store')['Customers'].sum()
    open_store = df_all.groupby([df_all['Store']])['Open'].count()
    
    store_data_customers = store_per_customer_day / open_store
    store_data_customers.name = 'CustomersPerDay'
    store_data_customers = store_data_customers.reset_index()
    
    sales_per_store = sales_per_store_day / open_store
    sales_per_store.name = 'SalesPerDay'
    sales_per_store = sales_per_store.reset_index()
    
    sales_per_store_per_cust = pd.DataFrame({
        "Store": sales_per_store['Store'],
        "SalesPerCustomersPerDay": sales_per_store['SalesPerDay'] / store_data_customers['CustomersPerDay']
    })
    
    df_all = pd.merge(df_all, store_data_customers, on='Store', how='left')
    df_all = pd.merge(df_all, sales_per_store, on='Store', how='left')
    df_all = pd.merge(df_all, sales_per_store_per_cust, on='Store', how='left')
        
    data_model = df_all[df_all['is_train'] == 1].copy()
    data_predict = df_all[df_all['is_train'] == 0].copy()
        
    columns_to_drop = ['Customers', 'is_train', 'Id', 'Date', 
                       '1_PromoInterval', '2_PromoInterval', '3_PromoInterval',
                       'Promo2SinceWeek', 'Promo2SinceYear',
                       'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                       'Promo2', 'PromoInterval', 'CompetitionDistance',]
    
    for df in [data_model, data_predict]:
        if df is not None:
            df['DateWeek'] = df.Date.dt.isocalendar().week.astype(int)
            df['DateMonth'] = df.Date.dt.month
            df['DateYear'] = df.Date.dt.year
            df["DateDay"] = df.Date.dt.day
            df['DateDayOfYear'] = df.Date.dt.dayofyear
            df['CompetitionOpen'] = df.CompetitionOpen.map(int)
            df['PromoSince'] = df.PromoSince.map(int)
            df['PromoTommorrow'] = df.Promo.shift(-1)
            df['PromoYesterday'] = df.Promo.shift(1)
            
            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    columns_to_drop2 = ['Sales']
    data_predict.drop(columns=columns_to_drop2, inplace=True)        

    for col in data_model.select_dtypes(include=['category', 'object', 'bool']):
        le = LabelEncoder()
        data_model[col] = le.fit_transform(data_model[col].astype(str))
        if data_predict is not None and col in data_predict.columns:
            data_predict[col] = le.transform(data_predict[col].astype(str))
    return (data_model, data_predict) if data_predict is not None else data_model

submisison_data_train, prediction_data = feature_engineering(data_train, data_test, data_store)
submisison_data_train['Sales'] = np.log(1+submisison_data_train['Sales'])
submisison_data_train = submisison_data_train[submisison_data_train['Open'] == 1][sorted(submisison_data_train.columns)]
X = submisison_data_train.drop(columns='Sales').sort_index()
y = submisison_data_train['Sales']
display(submisison_data_train.columns.sort_values())
display(submisison_data_train.head(2))
display(prediction_data.head(2))


# Forum Created
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

num_round = 20000
evallist = [(dtrain, 'train'), (dtest, 'test')]

params = {
    'eta': 0.05,                # Learning rate (lower = slower, better generalization)
    'max_depth': 9,             # Tree depth (controls model complexity)
    'subsample': 0.8,           # % of rows per tree (adds randomness, prevents overfit)
    'colsample_bytree': 0.8,    # % of features per tree (same purpose as above)
    'lambda': 1.0,              # L2 regularization
    'alpha': 3,               # L1 regularization
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist'   # GPU-accelerated histogram-based tree building
}

plst = list(params.items())

model = xgb.train(plst, dtrain, num_round, evallist, feval=rmspe_xg, verbose_eval=100, early_stopping_rounds=20)


#Print Feature Importance
plt.figure(figsize=(18,8))
plot_importance(model)
plt.show()


dprediction_data = xgb.DMatrix(prediction_data[X_train.columns])
prediction = model.predict(dprediction_data)


submission['Sales'] = prediction
submission['Sales'] = (np.exp(submission['Sales']) - 1) * 0.985
submission.to_csv('solution1.csv', index=False)
pd.read_csv('solution1.csv').head(5)

