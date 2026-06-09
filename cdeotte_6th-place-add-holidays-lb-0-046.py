import holidays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.rcParams.update({'font.size': 14})

TRAIN_CSV = '/kaggle/input/playground-series-s5e1/train.csv'
TEST_CSV = '/kaggle/input/playground-series-s5e1/test.csv'

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# DISPLAY TEST PREDICTIONS
USE_TEST = False
if USE_TEST:
    test_df['num_sold'] = \
        pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv").num_sold.values

train_df.date = pd.DatetimeIndex(train_df.date)
train_df['test'] = 0
test_df.date = pd.DatetimeIndex(test_df.date)
test_df['test'] = 1

class CFG:
    years_train = train_df.date.dt.year.unique()
    years_test = test_df.date.dt.year.unique()    
    years = np.concatenate((train_df.date.dt.year.unique(), test_df.date.dt.year.unique()))
    
    validation_year = 2000 # non-existing year or the year we use in validation - fill free to change this to 2016 or whatever you like
    
    countries = train_df.country.unique()
    stores = train_df.store.unique()
    products = train_df['product'].unique()

    alpha3 = {'Finland': 'FIN', 'Canada': 'CAN', 'Italy': 'IT', 'Kenya': 'KEN', 'Singapore': 'SGP', 'Norway': 'NOR'}

    fft_filter_width = 8

    countries_2l = {'Finland': 'FI', 'Canada': 'CA', 'Italy': 'IT', 'Kenya': 'KE', 'Singapore': 'SG', 'Norway': 'NO'}
    holiday_response_len = 10


df = pd.concat((train_df, test_df))
df.date = pd.DatetimeIndex(df.date)
df['year'] = df['date'].dt.year
df['weekday'] = df['date'].dt.weekday
df['dayofyear'] = df['date'].dt.dayofyear
df['daynum'] = (df.date - df.date.iloc[0]).dt.days
df['weeknum'] = df['daynum'] // 7
df['month'] = df.date.dt.month

# Sinusoidal features
daysinyear = (df.groupby('year').id.count() / len(CFG.countries) / len(CFG.stores) / len(CFG.products)).rename('daysinyear').astype(int).to_frame()
df = df.join(daysinyear, on='year', how='left')
df['partofyear'] = (df['dayofyear'] - 1) / df['daysinyear']
df['partof2year'] = df['partofyear'] + df['year'] % 2
df['partof2year'] = df['partofyear'] + df['year'] % 2

CFG.sincoscol = [f'sin t', f'cos t', f'sin t/2', f'cos t/2']
CFG.sincoscol2 = [f'sin 2t', f'cos 2t', f'sin t', f'cos t', f'sin t/2', f'cos t/2']
df['sin 4t'] = np.sin(8 * np.pi * df['partofyear'])
df['cos 4t'] = np.cos(8 * np.pi * df['partofyear'])
df['sin 3t'] = np.sin(6 * np.pi * df['partofyear'])
df['cos 3t'] = np.cos(6 * np.pi * df['partofyear'])
df['sin 2t'] = np.sin(4 * np.pi * df['partofyear'])
df['cos 2t'] = np.cos(4 * np.pi * df['partofyear'])
df['sin t'] = np.sin(2 * np.pi * df['partofyear'])
df['cos t'] = np.cos(2 * np.pi * df['partofyear'])
df['sin t/2'] = np.sin(np.pi * df['partof2year'])
df['cos t/2'] = np.cos(np.pi * df['partof2year'])
df.drop(['daysinyear', 'partofyear', 'partof2year'], axis=1, inplace=True)


import requests
def get_gdp_per_capita(country,year):
    url="https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json".format(CFG.alpha3[country],year)
    response = requests.get(url).json()
    return response[1][0]['value']

# gdp[counties x years]
gdp = np.array([[get_gdp_per_capita(country, year) for year in CFG.years] for country in CFG.countries])
gdp_df = pd.DataFrame(gdp, index=train_df.country.unique(), columns=CFG.years)
    
df['gdp_factor'] = None
for year in CFG.years:
    for country in CFG.countries:
        df.loc[(df.country == country) & (df.year == year), 'gdp_factor'] = gdp_df.loc[country, year]

gdp_df


df_no_can_ken = df[~df.country.isin(('Canada', 'Kenya'))]

store_df = df_no_can_ken.groupby(by='store').num_sold.mean().rename('store_factor').to_frame()
df = df.drop('store_factor', axis=1, errors='ignore').join(store_df, on='store', how='left')


store_df


from sklearn.linear_model import Ridge

df_no_can_ken = df[~df.country.isin(('Canada', 'Kenya'))].copy()

total = df_no_can_ken.groupby(by='date').num_sold.sum().rename('num_sold_total')
df_no_can_ken = df_no_can_ken.join(total, on='date', how='left')
df_no_can_ken['num_sold_ratio'] = df_no_can_ken['num_sold'] / df_no_can_ken['num_sold_total']

plt.figure(figsize=(24, 6))
df['product_factor'] = None
for product in CFG.products:
    df_no_can_ken_date = df_no_can_ken[(df_no_can_ken['product'] == product) & (df_no_can_ken['test'] == 0)].groupby(by='date')
    x = df_no_can_ken_date[CFG.sincoscol].mean().to_numpy()
    y = df_no_can_ken_date.num_sold_ratio.sum().to_numpy()

    reg = Ridge()
    reg.fit(x, y)
    p = reg.predict(x)
    df.loc[(df['product'] == product), 'product_factor'] = reg.predict(df.loc[(df['product'] == product), CFG.sincoscol].to_numpy())
   
    plt.plot(y, 'b')
    plt.plot(p, 'r')
plt.show();


df['holiday'] = 0

for country in CFG.countries:
    days = [str(day) for day in holidays.CountryHoliday(CFG.countries_2l[country], years=CFG.years)]
    df.loc[(df.country==country) & (df.date.isin(days)), 'holiday'] = 1

num_sold_per_week_country_weekday = df.groupby(['weeknum', 'country', 'weekday'])['num_sold'].sum().reset_index().pivot(index=['weeknum', 'country'], columns='weekday')
ratio_sold_per_week_country_weekday = num_sold_per_week_country_weekday.apply(lambda row: row/sum(row), axis=1).reset_index()

ratio_weekday = pd.DataFrame(columns=CFG.countries, data=[[0, ]*len(CFG.countries)]*7)

for n, country in enumerate(CFG.countries):
    for d in range(7):
        dt = ratio_sold_per_week_country_weekday.loc[ratio_sold_per_week_country_weekday.country == country, ('num_sold', d)][:-60]
        ratio_weekday.loc[d, country] = dt.median()

ratio_weekday_mean = ratio_weekday.mean(axis=1)
ratio_weekday['mean'] = ratio_weekday_mean

df['weekday_factor'] = df.weekday.map(ratio_weekday_mean)

# The total ratio taking into account all factors
df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor']

# The total sold items taking into account all factors
df['total'] = df['num_sold'] / df['ratio']


ratio_weekday_mean


# Exclude holidays
df_holidays = df.copy()
df_holidays['holiday_response'] = 0
for country in CFG.countries:
    for holiday, _ in holidays.CountryHoliday(CFG.countries_2l[country], years=CFG.years).items():
        df_holidays.loc[(df_holidays.country==country) & df_holidays.date.isin(pd.date_range(holiday, periods=CFG.holiday_response_len)), 'holiday_response'] = 1

fig = plt.figure(figsize=(24,6))
data = pd.DataFrame()
for n, country in enumerate(CFG.countries):
    dt = df_holidays[(df_holidays.country==country) & (df_holidays.holiday_response == 0)].groupby(['dayofyear']).total.median()
    data[country]= dt
    plt.plot(dt, label=country)
data['median'] = data.median(axis=1)

# Linear regression on fourier series
x = data.index.to_numpy()
y = data['median'].to_numpy()
fourier = lambda t: np.array([np.sin(2*np.pi/365*t), np.cos(2*np.pi/365*t)])

year_ratio = Ridge(alpha=0.01).fit(fourier(x).T, y.T).predict(fourier(np.arange(1, 366)).T)
year_ratio = np.append(year_ratio, year_ratio[-1])

df['dayofyear_factor'] = df.dayofyear.map(dict(zip(np.arange(1, 367), year_ratio)))

# The total ratio taking into account all factors
df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['dayofyear_factor']

# The total sold items taking into account all factors
df['total'] = df['num_sold'] / df['ratio']

plt.plot(year_ratio, 'k', linewidth=4)
plt.legend();


fig = plt.figure(figsize=(24,6))
data = pd.DataFrame()
for n, country in enumerate(CFG.countries):
    if not USE_TEST:
        dt = df_holidays[(df_holidays.test == 0) & (df_holidays.country==country) & (df_holidays.holiday_response == 0)].groupby(['date']).total.median()
    else:
        dt = df_holidays[(df_holidays.country==country) & (df_holidays.holiday_response == 0)].groupby(['date']).total.median()
    data[country]= dt
    plt.plot(dt, label=country)
data['median'] = data.median(axis=1)


CFG.sincoscol2 = ['sin 4t', 'cos 4t', 'sin 3t', 'cos 3t', 'sin 2t', 'cos 2t', 'sin t', 'cos t', 'sin t/2', 'cos t/2']

# Linear regression on fourier series
if not USE_TEST:
    dfsc = df[df.test == 0].groupby('date')[CFG.sincoscol2].mean()#.to_numpy()
else:
    dfsc = df.groupby('date')[CFG.sincoscol2].mean()#.to_numpy()
dfsc['median'] = data['median']

x = dfsc[~pd.isna(dfsc['median'])][CFG.sincoscol2].to_numpy()
y = dfsc[~pd.isna(dfsc['median'])]['median'].to_numpy()

reg = Ridge(alpha=0.01, fit_intercept=True)
reg.fit(x, y)

fig = plt.figure(figsize=(24,6))
plt.plot(y, 'k')
plt.plot(reg.predict(x), 'r')

df['sincos_factor'] = reg.intercept_ + (df[CFG.sincoscol2] * reg.coef_).sum(axis=1)


# The total ratio taking into account all factors
df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor']

# The total sold items taking into account all factors
df['total'] = df['num_sold'] / df['ratio']

fig = plt.figure(figsize=(24,6))
for c in CFG.countries:
    df_p = df[(df.country == c) & (df['product'] == 'Kaggle')].groupby('date').total.sum().to_numpy()
    plt.plot(df_p, label=c)

plt.legend();


country_factor = df[(df['product'] == 'Kaggle')].groupby('country').total.sum().rename('country_factor')
country_factor = country_factor / country_factor.median()
df = df.join(country_factor, on='country', how='left')


country_factor


df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor'] * df['country_factor']

# The total sold items taking into account all factors
df['total'] = df['num_sold'] / df['ratio']

fig = plt.figure(figsize=(24,6))
for c in CFG.countries:
    df_p = df[(df.country == c) & (df['product'] == 'Kaggle')].groupby('date').total.sum().to_numpy()
    plt.plot(df_p, label=c)

plt.legend();


npdoy_factor = df.groupby('dayofyear').total.median().rename('npdoy_factor')
df = df.join(npdoy_factor, on='dayofyear', how='left')


TOP_YEAR = 2017
if USE_TEST: TOP_YEAR = 2020


from datetime import datetime, timedelta

def get_sorted_holidays_reversed(country, year):
    # Dynamically get the holiday class using getattr
    holiday_class = getattr(holidays, country)
    
    # Get the holidays for the given year
    country_holidays = holiday_class(years=year)
    
    # Reverse key-value pairs and sort by the new keys (which are the holiday names)
    if country=='Italy':
        reversed_holidays = {name.split(";")[-1].strip(): date for date, name in sorted(country_holidays.items(), key=lambda x: x[0])}
    else:
        reversed_holidays = {name: date for date, name in sorted(country_holidays.items(), key=lambda x: x[0])}
    
    return reversed_holidays
    
#get_sorted_holidays_reversed('Kenya', 2010)


CFG.countries


CCC = CFG.countries[0]
print(CCC)
names = [k for k,v in get_sorted_holidays_reversed(CCC, 2010).items()]
[f"{i} {n}" for i,n in enumerate(names)]


IDX = -1

AA = 0
BB = 8

USE = [(2,0,8),(3,0,9),(-2,0,8)]

df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor']\
    * df['country_factor'] * df['npdoy_factor']  
df['total'] = df['num_sold'] / df['ratio']

bumps = []
bumps2 = []; names2 = []
for k in range(len(USE)):
    bumps2.append([])

for ccc in [CCC]:
    
    print("#"*25)
    print(f"### {ccc} ###")
    print("#"*25)
    
    c = ccc
    for y in range(2010,TOP_YEAR):
        df2 = df.loc[(df.year==y)]
        fig = plt.figure(figsize=(24,4))
        for cc in CFG.countries:
            if cc!=c: continue
            df_p = df2[(df2.country == c) & (df2['product'] == 'Kaggle')].groupby('date').total.sum() #.to_numpy()
            plt.plot(df_p, label=c)

        vv=""
        if IDX != -1:
            if IDX != -2:
                vv = names[IDX].replace(" (observed)","")+" (observed)"
                dd = get_sorted_holidays_reversed(ccc, y)
                if vv in dd: cny = dd[vv]
                else: cny = dd[vv.replace(" (observed)","")]
            else:
                vv = f"{y}-05-01"
                cny = pd.to_datetime(vv)
                
            old_date = cny + timedelta(days=AA)
            new_date = cny + timedelta(days=BB)
            plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='red')
            plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='red')
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps.append( tmp.groupby("date").total.median().values / df2.total.median() )

        for ii,(x1,x2,x3) in enumerate(USE):
            if x1 != -2:
                vv2 = names[x1].replace(" (observed)","")+" (observed)"
                dd = get_sorted_holidays_reversed(ccc, y)
                if vv2 in dd: cny = dd[vv2]
                else: cny = dd[vv2.replace(" (observed)","")]
            else:
                vv2 = "05-01"
                cny = pd.to_datetime(f"{y}-{vv2}")
            if y==2010: names2.append(vv2.replace(" (observed)",""))
            old_date = cny + timedelta(days=x2)
            new_date = cny + timedelta(days=x3)
            if IDX==-1:
                plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='black')
                plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='black')
    
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps2[ii].append( tmp.groupby("date").total.median().values / df2.total.median() )

        plt.legend();
        plt.title(f"Year {y} - {vv}")
        plt.show()

if IDX!=-1:
    x = np.vstack(bumps)
    print(x.shape)
    dd = {x:y for x,y in zip(range(0,x.shape[-1]),np.median(x,axis=0))}
    print(dd)

print('#'*25)
dds = []
for ii,bumps in enumerate(bumps2):
    x = np.vstack(bumps)
    aa = USE[ii][1]
    bb = USE[ii][2]
    dd = {x:y for x,y in zip(range(aa,bb+1),np.median(x,axis=0))}
    dds.append(dd)
    print(f"({USE[ii][0]})({names2[ii]})",end="")
    print(dd)


df[f"{CCC}_holidays"] = 1.0
for y in range(2010,2020):

    for ii,(x1,x2,x3) in enumerate(USE):
        if y==2010:
            print(f"({names2[ii]} from {x2} to {x3})")
        if x1 != -2:
            vv2 = names[x1].replace(" (observed)","")+" (observed)"
            dd = get_sorted_holidays_reversed(ccc, y)
            if vv2 in dd: cny = dd[vv2]
            else: cny = dd[vv2.replace(" (observed)","")]
        else:
            vv2 = names2[ii]
            cny = pd.to_datetime(f"{y}-{vv2}")
        for k in range(x2,x3+1):
            new_date = cny + timedelta(days=k)
            df.loc[(df.date == pd.to_datetime(new_date))&(df.country==CCC),f"{CCC}_holidays"] = dds[ii][k]


CCC = CFG.countries[1]
print(CCC)
names = [k for k,v in get_sorted_holidays_reversed(CCC, 2010).items()]
[f"{i} {n}" for i,n in enumerate(names)]


IDX = -1

AA = 0
BB = 8

USE = [(1,2,8),(2,1,12),(6,2,8),(7,1,8),(8,1,9),(10,2,7),(11,2,7),(12,1,7)]

df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor']\
    * df['country_factor'] * df['npdoy_factor']  
df['total'] = df['num_sold'] / df['ratio']

bumps = []
bumps2 = []; names2 = []
for k in range(len(USE)):
    bumps2.append([])

for ccc in [CCC]:
    
    print("#"*25)
    print(f"### {ccc} ###")
    print("#"*25)
    
    c = ccc
    for y in range(2010,TOP_YEAR):
        df2 = df.loc[(df.year==y)]
        fig = plt.figure(figsize=(24,4))
        for cc in CFG.countries:
            if cc!=c: continue
            df_p = df2[(df2.country == c) & (df2['product'] == 'Kaggle')].groupby('date').total.sum() #.to_numpy()
            plt.plot(df_p, label=c)

        vv=""
        if IDX != -1:
            vv = names[IDX].replace(" (observed)","")+" (observed)"
            dd = get_sorted_holidays_reversed(ccc, y)
            if vv in dd: cny = dd[vv]
            else: cny = dd[vv.replace(" (observed)","")]
                
            old_date = cny + timedelta(days=AA)
            new_date = cny + timedelta(days=BB)
            plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='red')
            plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='red')
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps.append( tmp.groupby("date").total.median().values / df2.total.median() )

        for ii,(x1,x2,x3) in enumerate(USE):
            vv2 = names[x1].replace(" (observed)","")+" (observed)"
            if y==2010: names2.append(vv2.replace(" (observed)",""))
            dd = get_sorted_holidays_reversed(ccc, y)
            if vv2 in dd: cny = dd[vv2]
            else: cny = dd[vv2.replace(" (observed)","")]
            old_date = cny + timedelta(days=x2)
            new_date = cny + timedelta(days=x3)
            if IDX==-1:
                plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='black')
                plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='black')
    
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps2[ii].append( tmp.groupby("date").total.median().values / df2.total.median() )

        plt.legend();
        plt.title(f"Year {y} - {vv}")
        plt.show()

if IDX!=-1:
    x = np.vstack(bumps)
    print(x.shape)
    dd = {x:y for x,y in zip(range(0,x.shape[-1]),np.median(x,axis=0))}
    print(dd)

print('#'*25)
dds = []
for ii,bumps in enumerate(bumps2):
    x = np.vstack(bumps)
    aa = USE[ii][1]
    bb = USE[ii][2]
    dd = {x:y for x,y in zip(range(aa,bb+1),np.median(x,axis=0))}
    dds.append(dd)
    print(f"({USE[ii][0]})({names2[ii]})",end="")
    print(dd)


df[f"{CCC}_holidays"] = 1.0
for y in range(2010,2020):

    for ii,(x1,x2,x3) in enumerate(USE):
        if y==2010:
            print(f"({names2[ii]} from {x2} to {x3})")
        vv2 = names[x1].replace(" (observed)","")+" (observed)"
        dd = get_sorted_holidays_reversed(ccc, y)
        if vv2 in dd: cny = dd[vv2]
        else: cny = dd[vv2.replace(" (observed)","")]
        for k in range(x2,x3+1):
            new_date = cny + timedelta(days=k)
            df.loc[(df.date == pd.to_datetime(new_date))&(df.country==CCC),f"{CCC}_holidays"] = dds[ii][k]


CCC = CFG.countries[2]
print(CCC)
names = [k for k,v in get_sorted_holidays_reversed(CCC, 2010).items()]
[f"{i} {n}" for i,n in enumerate(names)]


IDX = -1

AA = -5
BB = 4

USE = [(1,0,8),(3,1,8),(6,2,9),(7,1,8),(8,2,7),(9,2,7),(-2,-5,4)]

df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor']\
    * df['country_factor'] * df['npdoy_factor']  
df['total'] = df['num_sold'] / df['ratio']

bumps = []
bumps2 = []; names2 = []
for k in range(len(USE)):
    bumps2.append([])

for ccc in [CCC]:
    
    print("#"*25)
    print(f"### {ccc} ###")
    print("#"*25)
    
    c = ccc
    for y in range(2010,TOP_YEAR):
        df2 = df.loc[(df.year==y)]
        fig = plt.figure(figsize=(24,4))
        for cc in CFG.countries:
            if cc!=c: continue
            df_p = df2[(df2.country == c) & (df2['product'] == 'Kaggle')].groupby('date').total.sum() #.to_numpy()
            plt.plot(df_p, label=c)

        vv=""
        if IDX != -1:
            if IDX != -2:
                vv = names[IDX].replace(" (observed)","")+" (observed)"
                dd = get_sorted_holidays_reversed(ccc, y)
                if vv in dd: cny = dd[vv]
                else: cny = dd[vv.replace(" (observed)","")]
            else:
                vv = f"{y}-05-01"
                cny = pd.to_datetime(vv)
                
            old_date = cny + timedelta(days=AA)
            new_date = cny + timedelta(days=BB)
            plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='red')
            plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='red')
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps.append( tmp.groupby("date").total.median().values / df2.total.median() )

        for ii,(x1,x2,x3) in enumerate(USE):
            if x1 != -2:
                vv2 = names[x1].replace(" (observed)","")+" (observed)"
                dd = get_sorted_holidays_reversed(ccc, y)
                if vv2 in dd: cny = dd[vv2]
                else: cny = dd[vv2.replace(" (observed)","")]
            else:
                vv2 = "05-01"
                cny = pd.to_datetime(f"{y}-{vv2}")
            if y==2010: names2.append(vv2.replace(" (observed)",""))
            old_date = cny + timedelta(days=x2)
            new_date = cny + timedelta(days=x3)
            if IDX==-1:
                plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='black')
                plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='black')
    
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps2[ii].append( tmp.groupby("date").total.median().values / df2.total.median() )

        plt.legend();
        plt.title(f"Year {y} - {vv}")
        plt.show()

if IDX!=-1:
    x = np.vstack(bumps)
    print(x.shape)
    dd = {x:y for x,y in zip(range(0,x.shape[-1]),np.median(x,axis=0))}
    print(dd)

print('#'*25)
dds = []
for ii,bumps in enumerate(bumps2):
    x = np.vstack(bumps)
    aa = USE[ii][1]
    bb = USE[ii][2]
    dd = {x:y for x,y in zip(range(aa,bb+1),np.median(x,axis=0))}
    dds.append(dd)
    print(f"({USE[ii][0]})({names2[ii]})",end="")
    print(dd)


df[f"{CCC}_holidays"] = 1.0
for y in range(2010,2020):

    for ii,(x1,x2,x3) in enumerate(USE):
        if y==2010:
            print(f"({names2[ii]} from {x2} to {x3})")
        if x1 != -2:
            vv2 = names[x1].replace(" (observed)","")+" (observed)"
            dd = get_sorted_holidays_reversed(ccc, y)
            if vv2 in dd: cny = dd[vv2]
            else: cny = dd[vv2.replace(" (observed)","")]
        else:
            vv2 = names2[ii]
            cny = pd.to_datetime(f"{y}-{vv2}")
        for k in range(x2,x3+1):
            new_date = cny + timedelta(days=k)
            df.loc[(df.date == pd.to_datetime(new_date))&(df.country==CCC),f"{CCC}_holidays"] = dds[ii][k]


CCC = CFG.countries[3]
print(CCC)
names = [k for k,v in get_sorted_holidays_reversed(CCC, 2010).items()]
[f"{i} {n}" for i,n in enumerate(names)]


IDX = -1

AA = 3
BB = 9

USE = [(0,0,6),(1,1,10),(4,3,7),(5,1,8),(6,3,6),(-2,3,9)]

df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor']\
    * df['country_factor'] * df['npdoy_factor']  
df['total'] = df['num_sold'] / df['ratio']

bumps = []
bumps2 = []; names2 = []
for k in range(len(USE)):
    bumps2.append([])

for ccc in [CCC]:
    
    print("#"*25)
    print(f"### {ccc} ###")
    print("#"*25)
    
    c = ccc
    for y in range(2010,TOP_YEAR):
        df2 = df.loc[(df.year==y)]
        fig = plt.figure(figsize=(24,4))
        for cc in CFG.countries:
            if cc!=c: continue
            df_p = df2[(df2.country == c) & (df2['product'] == 'Kaggle')].groupby('date').total.sum() #.to_numpy()
            plt.plot(df_p, label=c)

        vv=""
        if IDX != -1:
            if IDX != -2:
                vv = names[IDX].replace(" (observed)","")+" (observed)"
                dd = get_sorted_holidays_reversed(ccc, y)
                if vv in dd: cny = dd[vv]
                else: cny = dd[vv.replace(" (observed)","")]
            else:
                vv = f"{y}-10-09"
                cny = pd.to_datetime(vv)
                
            old_date = cny + timedelta(days=AA)
            new_date = cny + timedelta(days=BB)
            plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='red')
            plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='red')
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps.append( tmp.groupby("date").total.median().values / df2.total.median() )

        for ii,(x1,x2,x3) in enumerate(USE):
            if x1 != -2:
                vv2 = names[x1].replace(" (observed)","")+" (observed)"
                dd = get_sorted_holidays_reversed(ccc, y)
                if vv2 in dd: cny = dd[vv2]
                else: cny = dd[vv2.replace(" (observed)","")]
            else:
                vv2 = "10-09"
                cny = pd.to_datetime(f"{y}-{vv2}")
            if y==2010: names2.append(vv2.replace(" (observed)",""))
            old_date = cny + timedelta(days=x2)
            new_date = cny + timedelta(days=x3)
            if IDX==-1:
                plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='black')
                plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='black')
    
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps2[ii].append( tmp.groupby("date").total.median().values / df2.total.median() )

        plt.legend();
        plt.title(f"Year {y} - {vv}")
        plt.show()

if IDX!=-1:
    x = np.vstack(bumps)
    print(x.shape)
    dd = {x:y for x,y in zip(range(0,x.shape[-1]),np.median(x,axis=0))}
    print(dd)

print('#'*25)
dds = []
for ii,bumps in enumerate(bumps2):
    x = np.vstack(bumps)
    aa = USE[ii][1]
    bb = USE[ii][2]
    dd = {x:y for x,y in zip(range(aa,bb+1),np.median(x,axis=0))}
    dds.append(dd)
    print(f"({USE[ii][0]})({names2[ii]})",end="")
    print(dd)


df[f"{CCC}_holidays"] = 1.0
for y in range(2010,2020):

    for ii,(x1,x2,x3) in enumerate(USE):
        if y==2010:
            print(f"({names2[ii]} from {x2} to {x3})")
        if x1 != -2:
            vv2 = names[x1].replace(" (observed)","")+" (observed)"
            dd = get_sorted_holidays_reversed(ccc, y)
            if vv2 in dd: cny = dd[vv2]
            else: cny = dd[vv2.replace(" (observed)","")]
        else:
            vv2 = names2[ii]
            cny = pd.to_datetime(f"{y}-{vv2}")
        for k in range(x2,x3+1):
            new_date = cny + timedelta(days=k)
            df.loc[(df.date == pd.to_datetime(new_date))&(df.country==CCC),f"{CCC}_holidays"] = dds[ii][k]


CCC = CFG.countries[4]
print(CCC)
names = [k for k,v in get_sorted_holidays_reversed(CCC, 2010).items()]
[f"{i} {n}" for i,n in enumerate(names)]


IDX = -1

AA = 0
BB = 8

USE = [(2,-1,11),(8,-9,10),(-2,0,8)]

df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor']\
    * df['country_factor'] * df['npdoy_factor']  
df['total'] = df['num_sold'] / df['ratio']

bumps = []
bumps2 = []; names2 = []
for k in range(len(USE)):
    bumps2.append([])

for ccc in [CCC]:
    
    print("#"*25)
    print(f"### {ccc} ###")
    print("#"*25)
    
    c = ccc
    for y in range(2010,TOP_YEAR):
        df2 = df.loc[(df.year==y)]
        fig = plt.figure(figsize=(24,4))
        for cc in CFG.countries:
            if cc!=c: continue
            df_p = df2[(df2.country == c) & (df2['product'] == 'Kaggle')].groupby('date').total.sum() #.to_numpy()
            plt.plot(df_p, label=c)

        vv=""
        if IDX != -1:
            if IDX != -2:
                vv = names[IDX].replace(" (observed)","")+" (observed)"
                dd = get_sorted_holidays_reversed(ccc, y)
                if vv in dd: cny = dd[vv]
                else: cny = dd[vv.replace(" (observed)","")]
            else:
                vv = f"{y}-05-17"
                cny = pd.to_datetime(vv)
                
            old_date = cny + timedelta(days=AA)
            new_date = cny + timedelta(days=BB)
            plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='red')
            plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='red')
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps.append( tmp.groupby("date").total.median().values / df2.total.median() )

        for ii,(x1,x2,x3) in enumerate(USE):
            if x1 != -2:
                vv2 = names[x1].replace(" (observed)","")+" (observed)"
                dd = get_sorted_holidays_reversed(ccc, y)
                if vv2 in dd: cny = dd[vv2]
                else: cny = dd[vv2.replace(" (observed)","")]
            else:
                vv2 = "05-17"
                cny = pd.to_datetime(f"{y}-{vv2}")
            if y==2010: names2.append(vv2.replace(" (observed)",""))
            old_date = cny + timedelta(days=x2)
            new_date = cny + timedelta(days=x3)
            if IDX==-1:
                plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='black')
                plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='black')
    
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps2[ii].append( tmp.groupby("date").total.median().values / df2.total.median() )

        plt.legend();
        plt.title(f"Year {y} - {vv}")
        plt.show()

if IDX!=-1:
    x = np.vstack(bumps)
    print(x.shape)
    dd = {x:y for x,y in zip(range(0,x.shape[-1]),np.median(x,axis=0))}
    print(dd)

print('#'*25)
dds = []
for ii,bumps in enumerate(bumps2):
    x = np.vstack(bumps)
    aa = USE[ii][1]
    bb = USE[ii][2]
    dd = {x:y for x,y in zip(range(aa,bb+1),np.median(x,axis=0))}
    dds.append(dd)
    print(f"({USE[ii][0]})({names2[ii]})",end="")
    print(dd)


df[f"{CCC}_holidays"] = 1.0
for y in range(2010,2020):

    for ii,(x1,x2,x3) in enumerate(USE):
        if y==2010:
            print(f"({names2[ii]} from {x2} to {x3})")
        if x1 != -2:
            vv2 = names[x1].replace(" (observed)","")+" (observed)"
            dd = get_sorted_holidays_reversed(ccc, y)
            if vv2 in dd: cny = dd[vv2]
            else: cny = dd[vv2.replace(" (observed)","")]
        else:
            vv2 = names2[ii]
            cny = pd.to_datetime(f"{y}-{vv2}")
        for k in range(x2,x3+1):
            new_date = cny + timedelta(days=k)
            df.loc[(df.date == pd.to_datetime(new_date))&(df.country==CCC),f"{CCC}_holidays"] = dds[ii][k]


CCC = CFG.countries[5]
print(CCC)
names = [k for k,v in get_sorted_holidays_reversed(CCC, 2010).items()]
[f"{i} {n}" for i,n in enumerate(names)]


IDX = -1

AA = 0
BB = 8

USE = [(1,-1,9),(3,1,7),(5,1,8),(6,1,7),(7,1,9),(8,2,8),(9,1,8)]

df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] * df['sincos_factor']\
    * df['country_factor'] * df['npdoy_factor']  
df['total'] = df['num_sold'] / df['ratio']

bumps = []
bumps2 = []; names2 = []
for k in range(len(USE)):
    bumps2.append([])

for ccc in [CCC]:
    
    print("#"*25)
    print(f"### {ccc} ###")
    print("#"*25)
    
    c = ccc
    for y in range(2010,TOP_YEAR):
        df2 = df.loc[(df.year==y)]
        fig = plt.figure(figsize=(24,4))
        for cc in CFG.countries:
            if cc!=c: continue
            df_p = df2[(df2.country == c) & (df2['product'] == 'Kaggle')].groupby('date').total.sum() #.to_numpy()
            plt.plot(df_p, label=c)

        vv=""
        if IDX != -1:
            vv = names[IDX].replace(" (observed)","")+" (observed)"
            dd = get_sorted_holidays_reversed(ccc, y)
            if vv in dd: cny = dd[vv]
            else: cny = dd[vv.replace(" (observed)","")]
                
            old_date = cny + timedelta(days=AA)
            new_date = cny + timedelta(days=BB)
            plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='red')
            plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='red')
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps.append( tmp.groupby("date").total.median().values / df2.total.median() )

        for ii,(x1,x2,x3) in enumerate(USE):
            vv2 = names[x1].replace(" (observed)","")+" (observed)"
            if y==2010: names2.append(vv2.replace(" (observed)",""))
            dd = get_sorted_holidays_reversed(ccc, y)
            if vv2 in dd: cny = dd[vv2]
            else: cny = dd[vv2.replace(" (observed)","")]
            old_date = cny + timedelta(days=x2)
            new_date = cny + timedelta(days=x3)
            if IDX==-1:
                plt.plot([old_date,old_date],[2.5,3.5],linewidth=3,color='black')
                plt.plot([new_date,new_date],[2.5,3.5],linewidth=3,color='black')
    
            tmp = df.loc[(df.date>=pd.to_datetime(old_date))&(df.date<=pd.to_datetime(new_date))&(df['country']==ccc)]
            bumps2[ii].append( tmp.groupby("date").total.median().values / df2.total.median() )

        plt.legend();
        plt.title(f"Year {y} - {vv}")
        plt.show()

if IDX!=-1:
    x = np.vstack(bumps)
    print(x.shape)
    dd = {x:y for x,y in zip(range(0,x.shape[-1]),np.median(x,axis=0))}
    print(dd)

print('#'*25)
dds = []
for ii,bumps in enumerate(bumps2):
    x = np.vstack(bumps)
    aa = USE[ii][1]
    bb = USE[ii][2]
    dd = {x:y for x,y in zip(range(aa,bb+1),np.median(x,axis=0))}
    dds.append(dd)
    print(f"({USE[ii][0]})({names2[ii]})",end="")
    print(dd)


df[f"{CCC}_holidays"] = 1.0
for y in range(2010,2020):

    for ii,(x1,x2,x3) in enumerate(USE):
        if y==2010:
            print(f"({names2[ii]} from {x2} to {x3})")
        vv2 = names[x1].replace(" (observed)","")+" (observed)"
        dd = get_sorted_holidays_reversed(ccc, y)
        if vv2 in dd: cny = dd[vv2]
        else: cny = dd[vv2.replace(" (observed)","")]
        for k in range(x2,x3+1):
            new_date = cny + timedelta(days=k)
            df.loc[(df.date == pd.to_datetime(new_date))&(df.country==CCC),f"{CCC}_holidays"] = dds[ii][k]


df['ratio'] = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor']\
    * df['sincos_factor'] * df['country_factor'] * df['npdoy_factor']\
    * df['Canada_holidays'] * df['Finland_holidays'] * df['Italy_holidays']\
    * df['Kenya_holidays'] * df['Norway_holidays'] * df['Singapore_holidays']

df['total'] = df['num_sold'] / df['ratio']
const_factor = df['total'].median() * 1.06

df['prediction'] = const_factor * df['ratio']
mape_train = mean_absolute_percentage_error(df[(df.test == 0) & (~pd.isna(df.num_sold))].num_sold, df[(df.test == 0) & (~pd.isna(df.num_sold))].prediction)

print(f'MAPE = {mape_train=}')


df['prediction'] = np.round(df['prediction'].astype(float)).astype(int)


submission = df[(df.test == 1)][['id', 'prediction']].rename(columns={'prediction': 'num_sold'})
submission.to_csv('submission.csv', index=False)

!head submission.csv

