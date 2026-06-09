import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
#plt.style.use('default')


train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)


train.head(10)


train.info()


# 'country'
train['country'].value_counts()


# 'store'
train['store'].value_counts()


# 'country'
train['product'].value_counts()


train['date'].value_counts()


# Convert 'Date' to datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])


print('First day:', train['date'].iloc[0])
print('Last day:', train['date'].iloc[-1])


sales_per_day = train.groupby('date')[['num_sold']].sum()
#sales_per_day.head()


# Total sales per day plot

plt.figure(figsize=(12, 6))

plt.plot(sales_per_day)

plt.xlabel('Dates')
plt.ylabel('Total sales')
plt.title('Total sales')
plt.savefig('images/Total sales')
plt.show()


sales_per_day_country = train.groupby(['date', 'country'])['num_sold'].sum().to_frame()
#sales_per_day_country.head(12)


# Total sales per country plot

plt.figure(figsize=(12, 8))
    
for country in train['country'].unique():
    plt.plot(sales_per_day_country[sales_per_day_country.index.get_level_values('country') == country].index.get_level_values('date'), 
             sales_per_day_country[sales_per_day_country.index.get_level_values('country') == country]['num_sold'],
             label=country,
             alpha=0.7)

plt.xlabel('Dates')
plt.ylabel('Total sales')
plt.title('Total sales per Country')
plt.legend(loc='upper center', ncol=len(train['country'].unique()))
plt.savefig('images/Total sales per country')
plt.show()


sales_day_country_store = train.groupby(['date', 'country', 'store'])['num_sold'].sum().to_frame()
sales_day_country_product = train.groupby(['date', 'country', 'product'])['num_sold'].sum().to_frame()
#sales_day_country_store.head(18)
#sales_day_country_product.head(30)


# Sales per store plot

for country in train['country'].unique():
    plt.figure(figsize=(12, 4))
    
    for store in train['store'].unique():
        plt.plot(sales_day_country_store[(sales_day_country_store.index.get_level_values('country') == country)
                          & (sales_day_country_store.index.get_level_values('store') == store)].index.get_level_values('date'), 
                 sales_day_country_store[(sales_day_country_store.index.get_level_values('country') == country)
                          & (sales_day_country_store.index.get_level_values('store') == store)]['num_sold'],
                label=store,
                alpha=0.7)

    plt.xlabel('Dates')
    plt.ylabel('Total sales')
    plt.title('{} Total sales per Store'.format(country))
    plt.legend(loc='upper center', ncol=len(train['store'].unique()))
    plt.savefig('images/{} Total sales per store'.format(country))
    plt.show()


# Sales per product plot

for country in train['country'].unique():
    plt.figure(figsize=(12, 4))
    
    for product in train['product'].unique():
        plt.plot(sales_day_country_product[(sales_day_country_product.index.get_level_values('country') == country)
                          & (sales_day_country_product.index.get_level_values('product') == product)].index.get_level_values('date'), 
                 sales_day_country_product[(sales_day_country_product.index.get_level_values('country') == country)
                          & (sales_day_country_product.index.get_level_values('product') == product)]['num_sold'],
                label=product,
                alpha=0.7)

    plt.xlabel('Dates')
    plt.ylabel('Total sales')
    plt.title('{} Total sales per Product'.format(country))
    plt.legend(loc='upper center', ncol=len(train['product'].unique()))
    plt.savefig('images/{} Total sales per Product'.format(country))
    plt.show()


alpha3 = {'Finland': 'FIN', 'Canada': 'CAN', 'Italy': 'IT', 'Kenya': 'KEN', 'Singapore': 'SGP', 'Norway': 'NOR'}
years= list(range(2010, 2020))

def get_gdp_per_capita(country,year):
    url="https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json".format(alpha3[country], year)
    response = requests.get(url).json()
    return response[1][0]['value']


# Get GDP data
gdp = np.array([[get_gdp_per_capita(country, year) for year in years] for country in train['country'].unique()])
gdp_df = pd.DataFrame(gdp, index=train.country.unique(), columns=years)
gdp_df


# Incorporate the new gdp datato train and test sets

train['gdp_factor'] = None
for year in years[:7]:
    for country in train['country'].unique():
        train.loc[(train.country == country) & (train['date'].dt.year == year), 'gdp_factor'] = gdp_df.loc[country, year]
        
test['gdp_factor'] = None
for year in years[7:]:
    for country in test['country'].unique():
        test.loc[(test.country == country) & (test['date'].dt.year == year), 'gdp_factor'] = gdp_df.loc[country, year]


gdp_per_day_country =  train.groupby(['date', 'country'])['gdp_factor'].first().to_frame()
#gdp_per_day_country.head(12)


# Plot GDP 

plt.figure(figsize=(12, 6))

for country in train['country'].unique():
    
    plt.plot(gdp_per_day_country[gdp_per_day_country.index.get_level_values('country') == country].index.get_level_values('date'),
             gdp_per_day_country[gdp_per_day_country.index.get_level_values('country') == country]['gdp_factor'],
                                     label=country)
    
plt.xlabel('Dates')
plt.ylabel('GDP')
plt.title('GDP per Country')
plt.legend(loc='upper center', ncol=len(train['country'].unique()))
plt.yticks(np.arange(0, 140000, 20000))
plt.savefig('images/GDP per country')
plt.show()


# Fisrt Plot .............................................

fig, ax1 = plt.subplots(figsize=(12,8))
plt.title('Sales and GDP per Country')
    
for country in train['country'].unique():
    ax1.plot(sales_per_day_country[sales_per_day_country.index.get_level_values('country') == country].index.get_level_values('date'), 
             sales_per_day_country[sales_per_day_country.index.get_level_values('country') == country]['num_sold'],
             label=country+' Sales',
             alpha=0.5)

ax1.set_xlabel('Dates')
ax1.set_ylabel('Total sales')
ax1.legend(loc='upper left', ncol=3, fontsize=9)

# Second Plot .............................................

ax2 = ax1.twinx() 

for country in train['country'].unique():
    
    ax2.plot(gdp_per_day_country[gdp_per_day_country.index.get_level_values('country') == country].index.get_level_values('date'),
             gdp_per_day_country[gdp_per_day_country.index.get_level_values('country') == country]['gdp_factor'],
                                     label=country+' GDP',
                                     linewidth=4)
    
ax2.set_xlabel('Dates')
ax2.set_ylabel('GDP')
ax2.legend(loc='upper right', ncol=3, fontsize=9)
ax2.set_yticks(np.arange(0, 170000, 20000))
plt.savefig('images/Sales vs country')
plt.show()
















