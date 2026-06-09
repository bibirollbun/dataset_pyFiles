import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

import warnings 
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=['date'])
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=['date'])


train.head()


train.head()


def cat_univar_plots(data, col):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    counts = data[col].value_counts()
    colors = [tuple(np.random.choice(256, size=3)/255) for _ in range(len(counts))]

    # Pie Chart
    ax[1].pie(counts.values, labels=counts.index, autopct='%.1f%%', colors=colors, wedgeprops={'edgecolor': 'black'})
    ax[1].set_title('Pie Chart')

    # Bar Chart
    ax[0].bar(counts.index, counts.values, color=colors, edgecolor='black')
    ax[0].set_title('Bar Chart')
    ax[0].set_xlabel('Categories')  
    ax[0].set_ylabel('Counts')     
    ax[0].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


cat_univar_plots(train, 'country')


cat_univar_plots(train, 'store')


cat_univar_plots(train, 'product')


counts = train.groupby(['country', 'store', 'product'])['id'].count().rename('num_rows').reset_index()
counts_val_counts = counts['num_rows'].value_counts().rename('Count').reset_index().rename(columns={'index': 'length'})
counts_val_counts.head()


print('Number of missing value in each columns :\n',train.isnull().sum())
# train.isnull().sum()


counts = train.groupby(['country', 'store', 'product'])['num_sold'].count().rename('num_rows')
missing_data = counts.loc[counts!=2557]
missing_data.reset_index()


train.head()


fig, axes = plt.subplots(9, 1, figsize=(20, 50))
for idx, (country, store, product) in enumerate(missing_data.index):
    plot_df = train.loc[(train['country']==country) & (train['store']==store) & (train['product']==product)]
    missing_df = plot_df.loc[plot_df['num_sold'].isna()]
    sns.lineplot(data=plot_df.dropna(subset=['num_sold']), x='date', y='num_sold', ax=axes[idx])
    for missing_date in missing_df['date']:
        axes[idx].axvline(missing_date, color='red', linestyle='-', linewidth=0.8, alpha=0.2)
    axes[idx].set_title(f'{country} - {store} - {product}')


print('Train - Earliest date: ', train['date'].min())
print('Train - Latest date: ', test['date'].max())
print('Test - Earliest date: ', train['date'].min())
print('Test - Latest date: ', test['date'].max())


weekly_data = train.groupby(['country', 'store', 'product', pd.Grouper(key='date', freq='W')])['num_sold'].sum().rename('num_sold').reset_index()
monthly_data= train.groupby(['country', 'store', 'product', pd.Grouper(key='date', freq='MS')])['num_sold'].sum().rename('num_sold').reset_index()


def plot_all(df):
    fig, ax = plt.subplots(5,1, figsize=(25,25), sharex=True, sharey=True)
    fig.tight_layout()
    for i, prod in enumerate(df['product'].unique()):
        plot_df = df.loc[df['product']==prod]
        sns.lineplot(data=plot_df, x='date', y='num_sold', hue='country', style='store', ax=ax[i], alpha=1)
        ax[i].set_title("Product: "+str(prod))


plot_all(monthly_data)


country_weights = train.groupby(['country'])['num_sold'].sum()/train['num_sold'].sum()
country_ratio_over_time = (train.groupby(['date', 'country'])['num_sold'].sum()/train.groupby(['date'])['num_sold'].sum()).reset_index()

fig, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(data=country_ratio_over_time, x='date', y='num_sold', hue='country')
ax.set_ylabel('Proportion of Sales:')
plt.show()


gdp_per_capita_df = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv")
years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']

# gpc = gdp_per_capita
filtered_gpc_df = gdp_per_capita_df.loc[gdp_per_capita_df['Country Name'].isin(train['country'].unique()), ['Country Name']+years].set_index('Country Name')

for year in years: 
    filtered_gpc_df[f'{year}_ratio'] = filtered_gpc_df[year]/filtered_gpc_df.sum()[year]

filtered_gpc_ratio_df = filtered_gpc_df[[year+'_ratio' for year in years]]
filtered_gpc_ratio_df.columns = [int(i) for i in years]
filtered_gpc_ratio_df = filtered_gpc_ratio_df.unstack().reset_index().rename(columns={"level_0":"year", 0: 'ratio', "Country Name": 'country'})
filtered_gpc_ratio_df['year'] = pd.to_datetime(filtered_gpc_ratio_df['year'], format='%Y')

filtered_gpc_ratio_2 = filtered_gpc_ratio_df.copy()
filtered_gpc_ratio_2['year'] = pd.to_datetime(filtered_gpc_ratio_2['year'].astype('str'))+pd.offsets.YearEnd(1)

filtered_gpc_ratio_new_df = pd.concat([filtered_gpc_ratio_df, filtered_gpc_ratio_2]).reset_index()


fig, ax = plt.subplots(figsize=(20,15))
sns.lineplot(data=country_ratio_over_time, x='date', y='num_sold', hue='country')
sns.lineplot(data=filtered_gpc_ratio_new_df, x='year', y='ratio', hue='country', palette=['black']*6, legend=False)
ax.set_ylabel('Proportion of Sales')
plt.show()


filtered_gpc_ratio_new_df["year"] = filtered_gpc_ratio_new_df["year"].dt.year
def plot_adjust_country(df):
    new_df = df.copy()
    new_df["year"] = new_df["date"].dt.year
    
    for country in new_df["country"].unique():
        for year in new_df["year"].unique():
            
            new_df.loc[(new_df["country"] == country) & (new_df["year"] == year), "num_sold"] = new_df.loc[(new_df["country"] == country) & (new_df["year"] == year), "num_sold"] / filtered_gpc_ratio_new_df.loc[(filtered_gpc_ratio_new_df['country']==country) & (filtered_gpc_ratio_new_df['year']==year), 'ratio'].values[0]
    plot_all(new_df)


plot_adjust_country(monthly_data)


train_imputed = train.copy()
missing_value_idx = train.loc[train['num_sold'].isna(), 'id'].values

print('Missing value Before Imputation: ', train['num_sold'].isna().sum())

train_imputed['year'] = train_imputed['date'].dt.year

for year in train_imputed['year'].unique():
    
    # Imputation of 1st time series
    curr_ratio = filtered_gpc_ratio_new_df.loc[
    (filtered_gpc_ratio_new_df['year']==year) & 
    (filtered_gpc_ratio_new_df['country']=='Canada'), 'ratio'].values[0]
    
    target_ratio=filtered_gpc_ratio_new_df.loc[(filtered_gpc_ratio_new_df['year']==year) & (filtered_gpc_ratio_new_df['country']=='Norway'), 'ratio'].values[0]   # Using norway gi ves best precision

    ratio_can = curr_ratio/target_ratio
    train_imputed.loc[(train_imputed['year']==year) & 
    (train_imputed['country']=='Canada') & 
    (train_imputed['product']=='Holographic Goose') & 
    (train_imputed['store']=='Discount Stickers'), 
    'num_sold']=(train_imputed.loc[(train_imputed['year']==year) & 
     (train_imputed['country']=='Norway') & 
     (train_imputed['product']=='Holographic Goose') & 
     (train_imputed['store']=='Discount Stickers'), 'num_sold']*ratio_can).values

    # Imputation of 2nd time series
    curr_ts = train_imputed.loc[
        (train_imputed['year'] == year) & 
        (train_imputed['country'] == 'Canada') & 
        (train_imputed['product'] == 'Holographic Goose') & 
        (train_imputed['store'] == 'Premium Sticker Mart')
    ]
    missing_ts = curr_ts.loc[curr_ts['num_sold'].isna(), 'date']
    
    if not missing_ts.empty:
        norway_values = train_imputed.loc[
            (train_imputed['year'] == year) & 
            (train_imputed['country'] == 'Norway') & 
            (train_imputed['product'] == 'Holographic Goose') & 
            (train_imputed['store'] == 'Premium Sticker Mart') & 
            (train_imputed['date'].isin(missing_ts)), 
            'num_sold'
        ].values
    
        if len(norway_values) > 0:
            train_imputed.loc[
                (train_imputed['year'] == year) & 
                (train_imputed['country'] == 'Canada') & 
                (train_imputed['product'] == 'Holographic Goose') & 
                (train_imputed['store'] == 'Premium Sticker Mart') & 
                (train_imputed['date'].isin(missing_ts)), 
                'num_sold'
            ] = norway_values * ratio_can
        else:
            print(f"No data for Premium Sticker Mart in Norway for year {year}. Skipping.")
    else:
        print(f"No missing dates for Premium Sticker Mart in Canada for year {year}.")


     # Imputation of 3rd time series
    curr_ts = train_imputed.loc[
        (train_imputed['year'] == year) & 
        (train_imputed['country'] == 'Canada') & 
        (train_imputed['product'] == 'Holographic Goose') & 
        (train_imputed['store'] == 'Stickers for Less')
    ]
    missing_ts = curr_ts.loc[curr_ts['num_sold'].isna(), 'date']
    
    if not missing_ts.empty:
        norway_values = train_imputed.loc[
            (train_imputed['year'] == year) & 
            (train_imputed['country'] == 'Norway') & 
            (train_imputed['product'] == 'Holographic Goose') & 
            (train_imputed['store'] == 'Stickers for Less') & 
            (train_imputed['date'].isin(missing_ts)), 
            'num_sold'
        ].values
    
        if len(norway_values) > 0:
            train_imputed.loc[
                (train_imputed['year'] == year) & 
                (train_imputed['country'] == 'Canada') & 
                (train_imputed['product'] == 'Holographic Goose') & 
                (train_imputed['store'] == 'Stickers for Less') & 
                (train_imputed['date'].isin(missing_ts)), 
                'num_sold'
            ] = norway_values * ratio_can
        else:
            print(f"No data for Premium Sticker Mart in Norway for year {year}. Skipping.")
    else:
        print(f"No missing dates for Premium Sticker Mart in Canada for year {year}.")


     # Imputation of 4th time series
    curr_ratio = filtered_gpc_ratio_new_df.loc[
    (filtered_gpc_ratio_new_df['year']==year) & 
    (filtered_gpc_ratio_new_df['country']=='Kenya'), 'ratio'].values[0]
    
    target_ratio=filtered_gpc_ratio_new_df.loc[(filtered_gpc_ratio_new_df['year']==year) & (filtered_gpc_ratio_new_df['country']=='Norway'), 'ratio'].values[0]   # Using norway gi ves best precision

    ratio_can = curr_ratio/target_ratio
    train_imputed.loc[(train_imputed['year']==year) & 
    (train_imputed['country']=='Kenya') & 
    (train_imputed['product']=='Holographic Goose') & 
    (train_imputed['store']=='Discount Stickers'), 
    'num_sold']=(train_imputed.loc[(train_imputed['year']==year) & 
     (train_imputed['country']=='Norway') & 
     (train_imputed['product']=='Holographic Goose') & 
     (train_imputed['store']=='Discount Stickers'), 'num_sold']*ratio_can).values

    # Imputation of 5th time series
    curr_ts = train_imputed.loc[
        (train_imputed['year'] == year) & 
        (train_imputed['country'] == 'Kenya') & 
        (train_imputed['product'] == 'Holographic Goose') & 
        (train_imputed['store'] == 'Premium Sticker Mart')
    ]
    missing_ts = curr_ts.loc[curr_ts['num_sold'].isna(), 'date']
    
    if not missing_ts.empty:
        norway_values = train_imputed.loc[
            (train_imputed['year'] == year) & 
            (train_imputed['country'] == 'Norway') & 
            (train_imputed['product'] == 'Holographic Goose') &
            (train_imputed['store'] == 'Premium Sticker Mart') & 
            (train_imputed['date'].isin(missing_ts)), 
            'num_sold'
        ].values
    
        if len(norway_values) > 0:
            train_imputed.loc[
                (train_imputed['year'] == year) & 
                (train_imputed['country'] == 'Kenya') & 
                (train_imputed['product'] == 'Holographic Goose') & 
                (train_imputed['store'] == 'Premium Sticker Mart') & 
                (train_imputed['date'].isin(missing_ts)), 
                'num_sold'
            ] = norway_values * ratio_can
        else:
        
            print(f"No data for Premium Sticker Mart in Norway for year {year}. Skipping.")
    else:
        print(f"No missing dates for Premium Sticker Mart in Canada for year {year}.")


      # Imputation of 6th time series
    curr_ts = train_imputed.loc[
        (train_imputed['year'] == year) & 
        (train_imputed['country'] == 'Kenya') & 
        (train_imputed['product'] == 'Holographic Goose') & 
        (train_imputed['store'] == 'Stickers for Less')
    ]
    missing_ts = curr_ts.loc[curr_ts['num_sold'].isna(), 'date']
    
    if not missing_ts.empty:
        norway_values = train_imputed.loc[
            (train_imputed['year'] == year) & 
            (train_imputed['country'] == 'Norway') & 
            (train_imputed['product'] == 'Holographic Goose') & 
            (train_imputed['store'] == 'Stickers for Less') & 
            (train_imputed['date'].isin(missing_ts)), 
            'num_sold'
        ].values
    
        if len(norway_values) > 0:
            train_imputed.loc[
                (train_imputed['year'] == year) & 
                (train_imputed['country'] == 'Kenya') & 
                (train_imputed['product'] == 'Holographic Goose') & 
                (train_imputed['store'] == 'Stickers for Less') & 
                (train_imputed['date'].isin(missing_ts)), 
                'num_sold'
            ] = norway_values * ratio_can
        else:
            print(f"No data for Premium Sticker Mart in Norway for year {year}. Skipping.")
    else:
        print(f"No missing dates for Premium Sticker Mart in Canada for year {year}.")

    # Imputation of 7th time series
    curr_ts = train_imputed.loc[
        (train_imputed['year'] == year) & 
        (train_imputed['country'] == 'Kenya') & 
        (train_imputed['product'] == 'Kerneler') & 
        (train_imputed['store'] == 'Discount Stickers')
    ]
    missing_ts = curr_ts.loc[curr_ts['num_sold'].isna(), 'date']
    
    if not missing_ts.empty:
        norway_values = train_imputed.loc[
            (train_imputed['year'] == year) & 
            (train_imputed['country'] == 'Norway') & 
            (train_imputed['product'] == 'Kerneler') & 
            (train_imputed['store'] == 'Discount Stickers') & 
            (train_imputed['date'].isin(missing_ts)), 
            'num_sold'
        ].values
    
        if len(norway_values) > 0:
            train_imputed.loc[
                (train_imputed['year'] == year) & 
                (train_imputed['country'] == 'Kenya') & 
                (train_imputed['product'] == 'Kerneler') & 
                (train_imputed['store'] == 'Discount Stickers') & 
                (train_imputed['date'].isin(missing_ts)), 
                'num_sold'
            ] = norway_values * ratio_can
        else:
            print(f"No data for Premium Sticker Mart in Norway for year {year}. Skipping.")
    else:
        print(f"No missing dates for Premium Sticker Mart in Canada for year {year}.")



print('Missing value after Imputation: ', train_imputed['num_sold'].isna().sum())


missing_rows = train_imputed.loc[train_imputed['num_sold'].isna()]
missing_rows


# these remaining two values are imputed manually
train_imputed.loc[train_imputed['id']==23719, 'num_sold']=4
train_imputed.loc[train_imputed['id']==207003, 'num_sold']=200
print('Missing value after Imputation: ', train_imputed['num_sold'].isna().sum())


# Update the monthly and weekly data:
weekly_data = train_imputed.groupby(['store', 'product', 'country', pd.Grouper(key='date', freq='W')])['num_sold'].sum().rename('num_sold').reset_index()
monthly_data= train_imputed.groupby(['store', 'product', 'country', pd.Grouper(key='date', freq='MS')])['num_sold'].sum().rename('num_sold').reset_index()


store_weights = train_imputed.groupby('store')['num_sold'].sum()/train_imputed['num_sold'].sum()
store_weights['Discount Stickers']/store_weights


store_ratio_over_time = (train_imputed.groupby(['date', 'store'])['num_sold'].sum()/train_imputed.groupby(['date'])['num_sold'].sum()).reset_index()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data=store_ratio_over_time, x='date', y='num_sold', hue='store')
ax.set_ylabel('Proportion of sales: ')
plt.show()


def plot_adjusted_store(df):
    new_df = df.copy()
    weights = store_weights.loc['Premium Sticker Mart']/store_weights
    print(weights)
    for store in weights.index:
        new_df.loc[new_df['store']==store, 'num_sold'] = new_df.loc[new_df['store']==store, 'num_sold']*weights[store]
    plot_all(new_df)


plot_adjusted_store(monthly_data)


product_data = train_imputed.groupby(['date', 'product'])['num_sold'].sum().reset_index()

fig, ax = plt.subplots(figsize=(20, 15))
sns.lineplot(data=product_data, x='date', y='num_sold', hue='product')


product_ratio_data = product_data.pivot(index='date', columns='product', values='num_sold')
product_ratio_data = product_ratio_data.apply(lambda x: x/x.sum(), axis=1)
product_ratio_data = product_ratio_data.stack().rename('ratios').reset_index()
product_ratio_data.head()


fig, ax = plt.subplots(figsize=(25, 15))
sns.lineplot(data=product_ratio_data, x='date', y='ratios', hue='product')
plt.show()


original_train_imputed=train_imputed.copy()
train_imputed = train_imputed.groupby(['date'])['num_sold'].sum().reset_index()

fig, ax = plt.subplots(figsize=(25, 15))
sns.lineplot(data=train_imputed, x='date', y='num_sold')
plt.show()


weekly_data = train_imputed.groupby([pd.Grouper(key='date', freq='W')])['num_sold'].sum().rename('num_sold').reset_index()
monthly_data = train_imputed.groupby([pd.Grouper(key='date', freq='MS')])['num_sold'].sum().rename('num_sold').reset_index()


fig, ax = plt.subplots(figsize=(25,10))
sns.lineplot(data=monthly_data, x='date', y='num_sold')
plt.grid('True')
plt.show()


fig, ax = plt.subplots(figsize=(25, 12))
sns.lineplot(data=weekly_data, x='date', y='num_sold')
plt.grid('True')
plt.show()


def plot_seasonality(df, x_axis):
    df['month'] = df['date'].dt.month
    df['week_day'] = df['date'].dt.dayofweek
    df['year_day'] = df['date'].apply(lambda x: x.timetuple().tm_yday if not (x.is_leap_year and x.month>2) else x.timetuple().tm_yday-1)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.lineplot(data=df, x=x_axis, y='num_sold', ax=ax)
    ax.set_title("{} Seasonality".format(x_axis))
    plt.grid('True')
    plt.show()


plot_seasonality(train_imputed, "month")


plot_seasonality(train_imputed, 'week_day')


plot_seasonality(train_imputed, 'year_day')


fig, ax = plt.subplots(figsize=(25, 10))
sns.lineplot(data=train_imputed, x='date', y='num_sold')
plt.grid('True')
plt.show()


# get the dates to forecasts for 
test_total_sales_df = test.groupby(['date'])['id'].first().reset_index().drop(columns='id')

# keep dates for later
test_total_sales_date = test_total_sales_df[['date']]


def feature_engineer(df):
    new_df = df.copy()
    new_df['month'] = df['date'].dt.month
    new_df['week_day'] = df['date'].dt.dayofweek
    new_df['year_day'] = df['date'].apply(lambda x: x.timetuple().tm_yday if not (x.is_leap_year and x.month>2) else x.timetuple().tm_yday-1)
    
    new_df['month_sin'] = np.sin(2 * np.pi * new_df['month']/12)
    new_df['month_cos'] = np.cos(2 * np.pi * new_df['month']/12)

    new_df['day_sin'] = np.sin(2 * np.pi * new_df['year_day']/365.0)
    new_df['day_cos'] = np.cos(2 * np.pi * new_df['year_day']/365.0)

    # from the seasonality plot of week day
    new_df['week_day'] = new_df['week_day'].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3)))).astype('int')

    # from the seasonality plot of year day
    new_df['important_dates'] = new_df['year_day'].apply(lambda x:x if x in [1,2,3,4,5,6,7,8,9,10,11,99, 100, 101,102, 124,125,126, 355,356,357,358,359,360,361,362,363,364,365] else 0).astype('int')
    new_df['year'] = new_df['date'].dt.year
    new_df['year'] = new_df['year']-2010

    new_df = new_df.drop(columns=['date', 'year_day', 'month'])
    new_df = pd.get_dummies(new_df, columns=['important_dates', 'week_day'], drop_first=True)
    # new_df = new_df.astype(int)

    for col in new_df.select_dtypes(include=['bool']).columns:
        new_df[col] = new_df[col].astype(int)
    return new_df


new_train_total_sales = feature_engineer(train_imputed)
new_test_total_sales = feature_engineer(test_total_sales_df)


display(new_train_total_sales.head(2))
display(new_test_total_sales.head(2))


y = new_train_total_sales['num_sold']
X = new_train_total_sales.drop(columns=['num_sold'])
X_test = new_test_total_sales


model = Ridge(tol=1e-2, max_iter=1000000, random_state=0)
model.fit(X, y)
preds = model.predict(X_test)
test_total_sales_date['num_sold'] = preds


fig, ax = plt.subplots(figsize=(25, 15))
sns.lineplot(data=pd.concat([train_imputed,test_total_sales_date]), x='date', y='num_sold')
plt.grid('True')
plt.show()


temp_df = pd.concat([product_ratio_data,forecasted_ratios_df]).reset_index(drop=True)
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data=temp_df, x="date", y="ratios", hue="product");
ax.axvline(pd.to_datetime("2017-01-01"), color='black', linestyle='--');






















