import pathlib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

input_path = pathlib.Path('/kaggle/input/playground-series-s5e1')

train_df = pd.read_csv(input_path / 'train.csv', index_col='id')
test_df = pd.read_csv(input_path / 'test.csv', index_col='id')
sample_submission = pd.read_csv(input_path / 'sample_submission.csv')
gdp_per_capita = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')

train_df.shape, test_df.shape


train_df['date'] = pd.to_datetime(train_df['date'])
train_df = train_df.set_index('date')

test_df['date'] = pd.to_datetime(test_df['date'])
test_df = test_df.set_index('date')


train_df.head(5)


train_df_na = pd.DataFrame(train_df.isna().sum()).transpose()
train_df_na


test_df.head(5)


test_df_na = pd.DataFrame(test_df.isna().sum()).transpose()
test_df_na


# Get dtypes for train and test
train_df_dtypes = pd.DataFrame(train_df.dtypes)
test_df_dtypes = pd.DataFrame(test_df.dtypes)

# Combined dtypes for inspection
dtypes_combined = pd.concat([train_df_dtypes, test_df_dtypes], axis=1)
dtypes_combined


train_date_tup = train_df.index.min().date().strftime('%Y-%m-%d'), train_df.index.max().date().strftime('%Y-%m-%d')
test_date_tup = test_df.index.min().date().strftime('%Y-%m-%d'), test_df.index.max().date().strftime('%Y-%m-%d')

print(f'Training data ranges from {train_date_tup[0]} till {train_date_tup[1]}')
print(f'Testing data ranges from {test_date_tup[0]} till {test_date_tup[1]}')


train_df.groupby(['country','store','product']).size().rename('n_records').to_frame().reset_index()


90 * 2557, train_df.shape[0]


train_missings_df = train_df.groupby(['country','store','product'])['num_sold'].count().rename("n_records").reset_index()
train_missings_df['n_records'] = np.abs(train_missings_df['n_records'] - 2557)
train_missings_df = train_missings_df.loc[train_missings_df.n_records > 0]
train_missings_df


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(5, 2, figsize=(18, 12))
ax = ax.flatten()

for i, (combi, df) in enumerate(train_missings_df.groupby(['country','store','product'])):
    country, store, product = combi

    combi_with_missings = train_df.loc[
        (train_df.country==country) & (train_df.store==store) & (train_df['product']==product)
    ]

    combi_missing_vals = combi_with_missings.loc[
        combi_with_missings.num_sold.isna()
    ]

    sns.lineplot(combi_with_missings, ax=ax[i])
    ax[i].set_xlim(train_df.index.min(), train_df.index.max())

    for missing_val in combi_missing_vals.index:
        ax[i].axvline(missing_val, color='green', linestyle='-', linewidth=1, alpha=0.2)

plt.tight_layout()


country_store_sales = train_df.groupby(['country', 'store', 'product']).num_sold.mean().unstack(level='store')
country_store_sales


store_pct = pd.concat([
    country_store_sales['Discount Stickers'] / np.sum(country_store_sales, axis=1),
    country_store_sales['Premium Sticker Mart'] / np.sum(country_store_sales, axis=1),
    country_store_sales['Stickers for Less'] / np.sum(country_store_sales, axis=1)
], axis=1).rename(
    columns={
        0: 'Discount Stickers',
        1: 'Premium Sticker Mart',
        2: 'Stickers for Less'
    }
)

store_pct


sold_column = train_df.groupby([train_df.index, 'product']).num_sold.sum().reset_index().pivot(index="date", columns='product', values='num_sold')

for product in np.unique(train_df['product']):
    sold_column.divide(sold_column.sum(axis=1), axis=0)[product].plot()


 country_store_sales_ratio = pd.concat([
     country_store_sales['Premium Sticker Mart'] / country_store_sales['Discount Stickers'],
     country_store_sales['Stickers for Less'] / country_store_sales['Discount Stickers'],
     country_store_sales['Premium Sticker Mart'] / country_store_sales['Stickers for Less']
 ], axis=1).rename(
     columns={
         0: 'ratio_premium_discount',
         1: 'ratio_less_discount',
         2: 'ratio_premium_less'
     }
 )

country_store_sales_ratio


import itertools

pd.options.display.max_rows=10

product_ratio_year = train_df.groupby(['country','store','product', train_df.index.year])['num_sold'].mean().unstack(level='product')
product_ratio_month = train_df.groupby(['country','store','product', train_df.index.month])['num_sold'].mean().unstack(level='product')

ratio_columns = []

for p1, p2 in itertools.combinations(product_ratio_year.columns, 2):
    column_name = f"{p1}/{p2}"
    ratio_columns.append(column_name)
    product_ratio_year[column_name] = product_ratio_year[p1] / product_ratio_year[p2]
    product_ratio_month[column_name] = product_ratio_month[p1] / product_ratio_month[p2]

product_ratio_year = product_ratio_year[ratio_columns]
product_ratio_month = product_ratio_month[ratio_columns]


import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ratios(ratio_df):

    n_cols = 2
    n_rows = math.ceil(len(ratio_columns) / n_cols)
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(18,12))
    ax = ax.flatten()
    
    for i, ratio_col in enumerate(ratio_columns):
    
        sns.lineplot(
            ratio_df[ratio_col].reset_index(), x='date', y=ratio_col, hue='store', ax=ax[i]
        )
    
    plt.tight_layout()


plot_ratios(product_ratio_year)


plot_ratios(product_ratio_month)


no_goose_df = train_df.loc[
    ~(((train_df.country == 'Canada') & (train_df.store=='Discount Stickers') & (train_df['product']=='Holographic Goose')) |
    ((train_df.country == 'Kenya') & (train_df.store=='Discount Stickers') & (train_df['product']=='Holographic Goose')))
]


plt.figure(figsize=(18,72))
for i, (combi, df) in enumerate(no_goose_df.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(30, 3, i+1, ymargin=0.5)
    ax.hist(df.num_sold, bins=50, color='pink')
    ax.set_title(combi)
plt.tight_layout(h_pad=3.0)
plt.show()


plt.figure(figsize=(18,82))
for i, (combi, df) in enumerate(no_goose_df.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(30, 3, i+1, ymargin=0.5)
    sns.lineplot(df, x='date', y='num_sold')
    ax.set_title(combi)
plt.tight_layout(h_pad=3.0)
plt.show()


sold_column = train_df.groupby([train_df.index, 'country']).num_sold.sum().reset_index().pivot(index="date", columns='country', values='num_sold')

fig, ax = plt.subplots(figsize=(10,5))

for product in np.unique(train_df['country']):
    sold_column.divide(sold_column.sum(axis=1), axis=0)[product].plot()


sold_column = train_df.groupby([train_df.index, 'store']).num_sold.sum().reset_index().pivot(index="date", columns='store', values='num_sold')

fig, ax = plt.subplots(figsize=(10,5))

for product in np.unique(train_df['store']):
    sns.lineplot(sold_column.divide(sold_column.sum(axis=1), axis=0)[product])


sold_column = train_df.groupby([train_df.index, 'product']).num_sold.sum().reset_index().pivot(index="date", columns='product', values='num_sold')

fig, ax = plt.subplots(figsize=(10,5))

for product in np.unique(train_df['product']):
    sns.lineplot(sold_column.divide(sold_column.sum(axis=1), axis=0)[product])


fig, ax = plt.subplots(3,2,figsize=(18, 12))
ax = ax.flatten()

for i, product in enumerate(np.unique(train_df['product'])):
    weekly_product = train_df.loc[train_df['product']==product].resample('W')['num_sold'].mean().to_frame()
    weekly_product = weekly_product.groupby(weekly_product.index.isocalendar().week)['num_sold'].mean()
    ax[i].plot(weekly_product)
    ax[i].set_title(product)
    ax[i].axvline(16, color='red')
    ax[i].axvline(51, color='red')
plt.tight_layout()


fig, ax = plt.subplots(3,2,figsize=(18, 12))
ax = ax.flatten()

for i, product in enumerate(np.unique(train_df['product'])):
    weekday_product = train_df.loc[train_df['product']==product].resample('D')['num_sold'].mean().to_frame()
    weekday_product = weekday_product.groupby(weekday_product.index.dayofweek)['num_sold'].mean()
    ax[i].plot(weekday_product)
    ax[i].set_title(product)
    ax[i].axvline(3, color='orange')
    ax[i].axvline(5, color='orange')
    ax[i].axvline(6, color='purple')
plt.tight_layout()


from sklearn.preprocessing import RobustScaler

def decompose(train, c):
    df = train_df.groupby(['date', c])['num_sold'].sum().reset_index()
    df['num_sold_scaled'] = df.groupby('date')['num_sold'].transform(lambda x: RobustScaler().fit_transform(x.values.reshape(-1,1)).flatten())
    for i,m in enumerate(np.sort(df[c].unique())):
        mask = df[c]==m
        ax[i].plot(df[mask]['date'], df[mask]['num_sold_scaled'], label=m)
        ax[i].legend(bbox_to_anchor=(1,1))
    plt.tight_layout()


gdp = list()
years = np.arange(2010, 2017).astype(str)
alpha3s = ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP']

for c in np.unique(train_df.country):
    gdp_c = gdp_per_capita[gdp_per_capita['Country Name']==c][years].values.ravel()
    gdp.append(gdp_c)

gdp = pd.DataFrame(
    RobustScaler().fit_transform(np.array(gdp)), index=alpha3s, columns=years
)

gdp


df = train_df.reset_index()[['date', 'country']]
df['alpha3'] = df['country'].map(dict(zip(np.sort(np.unique(df['country'])), alpha3s)))
df['year'] = df['date'].dt.year.astype(str)
df['gdp'] = df.apply(lambda s: gdp.loc[s['alpha3'], s['year']], axis=1)


fig, ax = plt.subplots(6, 1, figsize=(9, 10))
decompose(train_df, 'country')
for i, country in enumerate(df['country'].unique()):
    mask = df['country']==country
    ax[i].plot(df[mask]['date'], df[mask]['gdp'], 'k--', color='red')


df = train_df.reset_index().groupby(['date','country'])[['num_sold']].sum().reset_index().join(
    train_df.reset_index().groupby('date')[['num_sold']].sum(), on='date',rsuffix='_global'
)

df['fractions'] = df['num_sold'] / df['num_sold_global']


import seaborn as sns

ratio_by_country = (train_df.groupby([train_df.index, train_df.country])['num_sold'].sum() / train_df.groupby(train_df.index)['num_sold'].sum()).reset_index()

sns.lineplot(ratio_by_country, x='date', y='num_sold', hue='country')


# List all the unique countries in the training data
countries = np.unique(train_df.country)

# List all the years in the training and testing data
gdp_years = list(np.arange(2010,2020).astype('str'))

# Select only the relevant countries
gdp_per_capita = gdp_per_capita.loc[gdp_per_capita['Country Name'].isin(train_df.country.unique())]

# Select only the country name column and relevant years
gdp_per_capita = gdp_per_capita[['Country Name'] + gdp_years].set_index('Country Name')

# Calculate the ratio by dividing on the sum of the column
gdp_ratio_years = gdp_per_capita / gdp_per_capita.sum(axis=0)

# Unstack so we can easily join back on the training data
gdp_ratio_years = gdp_ratio_years.unstack().reset_index().rename(
    columns={'level_0': 'year','Country Name':'country', 0: 'gdp_ratio'}
)

gdp_total_years = gdp_per_capita.unstack().reset_index().rename(
    columns={'level_0':'year','Country Name': 'country', 0: 'gdp'}
)

gdp_pivot = gdp_total_years.pivot(index='year', columns='country', values='gdp')


train_df_imputed = train_df.copy().reset_index()
train_df_imputed['year'] = train_df_imputed['date'].dt.year.astype('str')
train_df_imputed


def impute_missing_multiply_by_gdp_ratio(impute_df, gdp_df, missings_df):
    """
    Impute missings values based on the ratio of GDP per country. It 
    handles both complete missing time series and partly missing data
    in a time series. 

    Args:
        impute_df: The dataframe on which we want to impute data.
        gdp_df: The dataframe with the GDP data per country.
        missings_df: The dataframe with all the missing combinations.

    Returns:
        A new dataframe with 
    """

    df = impute_df.copy()

    for row in missings_df.itertuples():
        for year in np.unique(df['year']):
            # Get missing timestamp for each combination of missing and year
            missing_ts = df.loc[
                (df['country']==row.country) & 
                (df['store']==row.store) &
                (df['product']==row.product) &
                (df['year']==year) &
                (df['num_sold'].isna()),
                'date'
            ]

            # Get ratio for current Country
            current_ratio = gdp_df.loc[
                (gdp_df['year']==year) & (gdp_df['country']==row.country), 'gdp_ratio'
            ].values[0]

            # We always use Norway as the baseline. Also try other countries to see impact, but is should not matter.
            target_ratio = gdp_df.loc[
                (gdp_df['year']==year) & (gdp_df['country']=='Norway'), 'gdp_ratio'
            ].values[0]
            
            multiply_ratio = current_ratio / target_ratio

            shape_missings = missing_ts.shape[0]

            if shape_missings > 0:

                print(f'Imputing combination: {row.country} - {row.store} - {row.product} - {year} - {shape_missings} records')
                
                df.loc[
                    (df['country']==row.country) & 
                    (df['store']==row.store) &
                    (df['product']==row.product) &
                    (df['date'].isin(missing_ts)),
                    'num_sold'
                ] = np.round(
                    df.loc[
                    (df['country']=='Norway') & 
                    (df['store']==row.store) &
                    (df['product']==row.product) &
                    (df['date'].isin(missing_ts)),
                    'num_sold'
                ] * multiply_ratio).values

    return df

train_df_imputed = impute_missing_multiply_by_gdp_ratio(train_df_imputed, gdp_ratio_years, train_missings_df)


def get_gdp(row):
    return gdp_pivot.loc[str(row.date.year), row.country]


import holidays
from dateutil.easter import easter
from datetime import timedelta

def feature_engineering(df, gdp_ratios):
    """
    Create new features for a given dataframe. It is used to create
    a standardized way to create the same features between the training
    and testing data.

    Args:
        df: The dataframe on which we want to create new features

    Returns:
        A new dataframe with the add features.
    """

    new_df = pd.DataFrame({
        'gdp': np.log(df.apply(get_gdp, axis=1)),
        'is_friday': (df.date.dt.weekday == 4).astype('int'),
        'is_saturday': (df.date.dt.weekday == 5).astype('int'),
        'is_sunday': (df.date.dt.weekday == 6).astype('int')
    })

    new_df = pd.concat([new_df, pd.DataFrame({
        f'dec_{d}': (df.date.dt.month == 12) & (df.date.dt.day == d) for d in range(25, 32)
    })], axis=1)

    new_df = pd.concat([new_df, pd.DataFrame({
        f'jan{d}': (df.date.dt.month == 1) & (df.date.dt.day == d) for d in range(1, 12)
    })], axis=1)

    cols_to_drop = ['country','store','product','year','date']

    new_df['year'] = df['date'].dt.year.astype('str')
    day_of_year = df['date'].dt.dayofyear

    # OHE for country
    for country in ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway']:
        country_mask = (df.country == country).astype('int')
        new_df[country] = country_mask

    # OHE for store
    for store in ['Discount Stickers','Premium Sticker Mart']:
        store_mask = (df.store == store).astype('int')
        new_df[store] = store_mask

    # OHE for product
    for product in ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler','Kerneler Dark Mode']:
        product_mask = (df['product'] == product).astype('int')
        new_df[product] = product_mask

    # Easter
    easter_date = df.date.apply(lambda date: pd.Timestamp(easter(date.year)))
    new_df = pd.concat([new_df, pd.DataFrame({
        f"easter{d}": (df.date - easter_date == np.timedelta64(d, "D")) for d in list(range(-2, 11))
    })], axis=1)
    
    for k in range(1, 3):
        new_df[f'sin_{k}'] = np.sin(day_of_year / 365.0 * 2 * math.pi * k)
        new_df[f'cos_{k}'] = np.cos(day_of_year / 365.0 * 2 * math.pi * k)
        
        new_df[f'sin_hy_{k}'] = np.sin(day_of_year / 365.0 * 4 * math.pi * k)
        new_df[f'cos_hy_{k}'] = np.cos(day_of_year / 365.0 * 4 * math.pi * k)

        new_df[f'sin_qt_{k}'] = np.sin(day_of_year / 365.0 * 8 * math.pi * k)
        new_df[f'cos_qt_{k}'] = np.cos(day_of_year / 365.0 * 8 * math.pi * k)

        # Year fourier terms
        new_df[f'hgs_sin_{k}'] = new_df[f'sin_{k}'] * new_df['Holographic Goose']
        new_df[f'hgs_cos_{k}'] = new_df[f'cos_{k}'] * new_df['Holographic Goose']
    
        new_df[f'kag_sin_{k}'] = new_df[f'sin_{k}'] * new_df['Kaggle']
        new_df[f'kag_cos_{k}'] = new_df[f'cos_{k}'] * new_df['Kaggle']
    
        new_df[f'kgt_sin_{k}'] = new_df[f'sin_{k}'] * new_df['Kaggle Tiers']
        new_df[f'kgt_cos_{k}'] = new_df[f'cos_{k}'] * new_df['Kaggle Tiers']
    
        new_df[f'krn_sin_{k}'] = new_df[f'sin_{k}'] * new_df['Kerneler']
        new_df[f'krn_cos_{k}'] = new_df[f'cos_{k}'] * new_df['Kerneler']
    
        new_df[f'kdm_sin_{k}'] = new_df[f'sin_{k}'] * new_df['Kerneler Dark Mode']
        new_df[f'kdm_cos_{k}'] = new_df[f'cos_{k}'] * new_df['Kerneler Dark Mode']


    alpha2 = dict(zip(np.sort(train_df.country.unique()), ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']))
    h = {c: holidays.country_holidays(a, years=range(2010, 2020)) for c, a in alpha2.items()}
    new_df['is_holiday'] = 0
    for c in alpha2:
        new_df.loc[df.country==c, 'is_holiday'] = df.date.isin(h[c]).astype(int)

    # Canada day
    holiday_df = pd.DataFrame([(c, d,n) for c,e in h.items() for d, n in e.items()], columns=[
        'country','date','holiday_name'
    ])
    
    holiday_df['date'] = pd.to_datetime(holiday_df['date'])
    holiday_df['year'] = holiday_df['date'].dt.year.astype('str')

    df['year'] = df.date.dt.year.astype('str')
    
    testen = df.merge(
        holiday_df.loc[(holiday_df.holiday_name=='Canada Day') & (holiday_df.country=='Canada')],
        how='left', left_on=['country', 'year'], right_on=['country', 'year']
    )
    
    test_date = testen.date_y
    
    new_df = pd.concat([new_df, pd.DataFrame({f'can_cd_{d}': testen.date_x - test_date == np.timedelta64(d, 'D') for d in list(range(1,11))})], axis=1)

    return new_df


# Add date back as normal column and do feature engineering on test
test_df = test_df.reset_index()
test_df = feature_engineering(test_df, gdp_total_years)

# Extract num_sold before feature engineering
num_sold = train_df_imputed['num_sold'].values

# Do feature engineering on the training data
train_df_imputed = feature_engineering(train_df_imputed, gdp_total_years)
train_df_imputed['num_sold'] = num_sold

features = [c for c in test_df.columns]


for df in [train_df_imputed, test_df]:
    df[features] = df[features].astype(np.float32)


from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

X_train = train_df_imputed[features]
y_train = train_df_imputed['num_sold'].values.reshape(-1,1)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_df)


model = Ridge(alpha=0.001)

model.fit(X_train_scaled, np.log(y_train).ravel())

test_pred = np.exp(model.predict(X_test_scaled))


inspect_df = pd.DataFrame({
    'row_id': 0, 
    'date': pd.date_range('2010-01-01', '2016-12-31', freq='D'),
    'country': 'Canada',
    'store': 'Stickers for Less',
    'product': 'Holographic Goose'
})

inspect_df = feature_engineering(inspect_df, gdp_total_years)

inspect_df = inspect_df[test_df.columns]

inspect_df['num_sold'] = np.exp(model.predict(scaler.transform(inspect_df)))


train_subset = train_df_imputed.loc[
    (train_df_imputed['Canada']==1) & (train_df_imputed['Discount Stickers']==0) & (train_df_imputed['Premium Sticker Mart']==0) & (train_df_imputed['Holographic Goose']==1)
]


plt.figure(figsize=(20, 6))
plt.plot(np.arange(len(inspect_df)), inspect_df['num_sold'], label='prediction')
plt.scatter(np.arange(len(train_subset)), train_subset.num_sold, color='red', label='true', alpha=0.5, s=3)
plt.legend()


single_combi = train_df.loc[
    (train_df['country']=='Norway') & (train_df['store']=='Stickers for Less') & (train_df['product']=='Holographic Goose')
].reset_index()


plt.figure(figsize=(20, 6))
plt.plot(single_combi.date, single_combi.num_sold)


pred_vs_real = pd.DataFrame({
    'date': train_df.index.to_list(),
    'pred': np.exp(model.predict(X_train_scaled)),
    'num_sold': train_df_imputed['num_sold'].values,
    'country': train_df.country.values
})

by_date = pred_vs_real.groupby('date')

residuals = (by_date.pred.sum() - by_date.num_sold.sum()) / (by_date.pred.sum() + by_date.num_sold.sum()) * 200


plt.figure(figsize=(20, 6))
plt.scatter(residuals.index, residuals, s=1, color='k')
plt.vlines(pd.date_range('2010-01-01', '2017-01-01', freq='M'),
           plt.ylim()[0], plt.ylim()[1], alpha=0.5)
plt.vlines(pd.date_range('2010-01-01', '2017-01-01', freq='Y'),
           plt.ylim()[0], plt.ylim()[1], alpha=0.5)


from datetime import date, timedelta
from matplotlib.ticker import MaxNLocator

def plot_around_date(residuals, m, d, w):
    """Plot residuals in an interval of with 2*w around month=m and day=d"""
    plt.figure()
    plt.title(f"Residuals around m={m} d={d}")
    for y in np.arange(2010, 2017):
        d0 = pd.Timestamp(date(y, m, d))
        residual_range = residuals[(residuals.index > d0 - timedelta(w)) & 
                                   (residuals.index < d0 + timedelta(w))]
        plt.plot([(r - d0).days for r in residual_range.index], residual_range, label=str(y))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
    plt.legend()
    plt.show()


plot_around_date(residuals, m=5, d=1, w=50)


sres = train_df.loc[(train_df.index>='2010-04-20') & (train_df.index<'2010-06-01')]


sres = sres.groupby([sres.index, 'country'])['num_sold'].mean().reset_index()


fig, ax = plt.subplots(3,2,figsize=(15,10))
ax = ax.flatten()

for i, cntry in enumerate(np.unique(sres.country)):
    sns.lineplot(sres.loc[sres.country==cntry], x='date', y='num_sold', ax=ax[i], hue='country')


from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

X_train = train_df_imputed[features]
y_train = train_df_imputed['num_sold'].values.reshape(-1,1)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_df)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

def fit_model_and_cross_validate(model, X, y):

    X = pd.concat([train_df.reset_index()['date'], X], axis=1)

    features = [c for c in X.columns if c != 'date']

    folds = [
        {'trn_start': '2010-01-01', 'trn_end': '2011-12-31', 'val_start': '2012-01-01', 'val_end': '2012-12-31'},
        {'trn_start': '2011-01-01', 'trn_end': '2012-12-31', 'val_start': '2013-01-01', 'val_end': '2013-12-31'},
        {'trn_start': '2012-01-01', 'trn_end': '2013-12-31', 'val_start': '2014-01-01', 'val_end': '2014-12-31'},
        {'trn_start': '2013-01-01', 'trn_end': '2014-12-31', 'val_start': '2015-01-01', 'val_end': '2015-12-31'},
        {'trn_start': '2014-01-01', 'trn_end': '2015-12-31', 'val_start': '2016-01-01', 'val_end': '2016-12-31'}
    ]

    scores = []
    all_residuals = []

    for i, fold in enumerate(folds):
        train_idx = X.loc[
            (X['date'] >= pd.to_datetime(fold['trn_start'])) &
            (X['date'] <= pd.to_datetime(fold['trn_end']))
        ].index
        
        val_idx = X.loc[
            (X['date'] >= pd.to_datetime(fold['val_start'])) &
            (X['date'] <= pd.to_datetime(fold['val_end']))
        ].index

        X_train, X_val = X[features].iloc[train_idx], X[features].iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = Ridge()

        model.fit(X_train, np.log(y_train))

        y_pred = np.exp(model.predict(X_val))
        mape_score = mean_absolute_percentage_error(y_val, y_pred)

        print(mape_score)

        scores.append(mape_score)

        residuals = (y_val.ravel() - y_pred.ravel()) / (y_val.ravel() + y_pred.ravel()) * 200
    
        residual_df = pd.DataFrame({
            'date': X.iloc[val_idx]['date'].values,
            'fold': i + 1,
            'residual': residuals
        })
        all_residuals.append(residual_df)

    print('-------------------------------')
    print(f'Average MAPE: {np.mean(scores)}')

    return scores, all_residuals

scores, residuals = fit_model_and_cross_validate(model, X_train, y_train)


#0.07147842119343477


residuals_df = pd.concat(residuals)


residuals_df


fig, ax = plt.subplots(5, 1, figsize=(18, 40))

for i in range(5):
    f1 = residuals_df.loc[residuals_df.fold == i + 1]
    sns.lineplot(f1, x='date', y='residual', marker='o', ax=ax[i])
plt.show()
plt.tight_layout()


model = Ridge()

model.fit(X_train_scaled, np.log(y_train).ravel())

test_pred = np.exp(model.predict(X_test_scaled))


sample_submission['num_sold'] = np.ceil(test_pred)
sample_submission.head()


sample_submission.to_csv('submission.csv', index=False)


train_df


import pandas as pd

train_df = pd.read_csv(input_path / 'train.csv', index_col='id')
train_df['date'] = pd.to_datetime(train_df.date)
gdp_per_capita = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')
gdp_per_capita = gdp_per_capita.loc[gdp_per_capita['Country Name'].isin(np.unique(train_df.country))]
gdp_per_capita = gdp_per_capita[['Country Name']+np.unique(train_df.date.dt.year.astype(str)).tolist()]
gdp_per_capita = gdp_per_capita.set_index('Country Name').stack().reset_index().rename(
    columns={'Country Name':'country','level_1':'year', 0: 'gdp'}
)


train_df['year'] = train_df.date.dt.year.astype(str)


train_df = train_df.merge(gdp_per_capita, how='left', left_on=['country','year'], right_on=['country','year'])


train_df['num_sold'] /= train_df['gdp']


store_ratio = train_df.groupby('store')['num_sold'].mean().to_dict()


train_df['store_ratio'] = train_df.store.map(store_ratio)


train_df['num_sold'] /= train_df['store_ratio']


t1 = train_df#.loc[train_df['product']=='Holographic Goose']
t1 = t1.loc[t1.date.dt.year == 2011]


t1


import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(6, 1, figsize=(15, 20))
ax = ax.flatten()

t1 = t1.groupby(['date', 'country'])['num_sold'].mean().reset_index()

alpha2 = dict(zip(np.sort(train_df.country.unique()), ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']))
h = {c: holidays.country_holidays(a, years=range(2010, 2020)) for c, a in alpha2.items()}

for i, c in enumerate(np.unique(t1.country)):
    plt_df = t1.loc[t1.country==c]
    plt_df['7d_avg'] = plt_df['num_sold'].rolling(window=7, center=True, min_periods=1).mean()
    plt_df['holiday'] = plt_df['date'].isin(h[c])
    
    sns.lineplot(plt_df, x='date', y='num_sold', ax=ax[i], label=c)
    sns.lineplot(plt_df, x='date', y='7d_avg', ax=ax[i])

    important_dates = plt_df[plt_df['holiday']]
    for d in important_dates['date']:
        ax[i].axvline(d, color='green', linestyle='--')
        ax[i].axvline(easter(d.year), color='red')


holiday_df = pd.DataFrame([(c, d,n) for c,e in h.items() for d, n in e.items()], columns=[
    'country','date','holiday_name'
])

holiday_df['date'] = pd.to_datetime(holiday_df['date'])
holiday_df['year'] = holiday_df['date'].dt.year.astype('str')


testen = train_df.merge(
    holiday_df.loc[(holiday_df.holiday_name=='Canada Day') & (holiday_df.country=='Canada')],
    how='left', left_on=['country', 'year'], right_on=['country', 'year']
)

test_date = testen.date_y

pd.DataFrame({f'can_cd_{d}': testen.date_x - test_date == np.timedelta64(d, 'D') for d in list(range(1,11))})


pd.DataFrame({f'cd_{d}': testen.date_x - test_date == np.timedelta64(d, 'D') for d in list(range(1,11))})

