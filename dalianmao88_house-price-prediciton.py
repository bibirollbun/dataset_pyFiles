#library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
from sklearn.cluster import KMeans


#read in data
df=pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test=pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')


#describe
df.describe().T


#EDA numerical data
def plot_num_cols(num_cols):
    n=len(num_cols)
    cols=4
    rows=math.ceil(n/cols)
    plt.figure(figsize=(cols*5,rows*4))
    for i,col in enumerate(num_cols):
        plt.subplot(rows,cols,i+1)
        df[col].hist(bins=20)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Freq')
    plt.tight_layout()
    plt.show()
    plt.close()

num_cols=df.select_dtypes(include=[np.number]).columns.to_list()
plot_num_cols(num_cols)


#EDA categorical data
def plot_cat_cols(cat_cols):
    n=len(cat_cols)
    cols=4
    rows=math.ceil(n/cols)
    plt.figure(figsize=(cols*5,rows*4))
    for i,col in enumerate(cat_cols):
        plt.subplot(rows,cols,i+1)
        df[col].value_counts().plot(kind='bar')
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Freq')
    plt.tight_layout()
    plt.show()
    plt.close()

cat_cols=df.select_dtypes(include=['object','category']).columns.to_list()
plot_cat_cols(cat_cols)



#data preprocess
## fill na sale_nbr
df['sale_nbr']=df['sale_nbr'].fillna(0)
##1  year calculate
df['built_to_sale_year']=pd.to_datetime(df['sale_date']).dt.year-df['year_built']
df=df.drop(['sale_date','year_built','join_year'],axis=1)


##2  sale_warning var
all_warnings = set()
for val in df['sale_warning'].dropna():
    parts = re.split(r'[;,	\s]+', str(val))
    for p in parts:
        p = p.strip()
        if p != '':
            all_warnings.add(p)
all_warnings = sorted(all_warnings, key=str)
dummy_cols = []
for warn in all_warnings:
    colname = f'sale_warning_{warn}'
    df[colname] = df['sale_warning'].apply(
        lambda x: int(str(x) and warn in re.split(r'[;,	\s]+', str(x)))
    )
    dummy_cols.append(colname)

keep_cols = []
other_cols = []
n = len(df)
for col in dummy_cols:
    if df[col].sum() / n > 0.01:
        keep_cols.append(col)
    else:
        other_cols.append(col)
if other_cols:
    df['sale_warning_other'] = df[other_cols].sum(axis=1)
    df = df.drop(other_cols, axis=1)
df = df.drop('sale_warning', axis=1)
plot_num_cols(keep_cols)
#plot_cat_cols(['sale_warning_other'])


##3 join_status
status_dummies = pd.get_dummies(df['join_status'], prefix='join_status', dummy_na=False)
df = pd.concat([df, status_dummies], axis=1)
df=df.drop(['join_status'],axis=1)


##4  latitude & longitude
coords = df[['latitude', 'longitude']].dropna()
kmeans = KMeans(n_clusters=10, random_state=42)
df.loc[coords.index, 'geo_cluster'] = kmeans.fit_predict(coords)
df['geo_cluster'] = df['geo_cluster'].fillna(-1).astype(int)
# 可选：做one-hot
geo_dummies = pd.get_dummies(df['geo_cluster'], prefix='geo_cluster')
df = pd.concat([df, geo_dummies], axis=1)
df = df.drop(['geo_cluster','latitude', 'longitude'], axis=1)


##5 city
city_counts = df['city'].value_counts(normalize=True)
major_cities = city_counts[city_counts > 0.03].index.tolist()
df['city_processed'] = df['city'].apply(lambda x: x if x in major_cities else 'other')
city_dummies = pd.get_dummies(df['city_processed'], prefix='city', dummy_na=False)
df = pd.concat([df, city_dummies], axis=1)
df=df.drop(['city'],axis=1)

# city_counts = df['city'].value_counts()
# major_cities = city_counts[city_counts > 1000].index.tolist()
# df['city_processed'] = df['city'].apply(lambda x: x if x in major_cities else 'other')
# city_dummies = pd.get_dummies(df['city_processed'], prefix='city', dummy_na=False)
# df = pd.concat([df.drop(['city', 'city_processed'], axis=1), city_dummies], axis=1)



##6 zoning
zoning = df['zoning'].value_counts(normalize=True)
major_zoning = zoning[zoning > 0.03].index.tolist()
df['zoning_processed'] = df['zoning'].apply(lambda x: x if x in major_zoning else 'other')
zoning_dummies = pd.get_dummies(df['zoning_processed'], prefix='zoning', dummy_na=False)
df = pd.concat([df, zoning_dummies], axis=1)
df=df.drop(['zoning'],axis=1)


##7 subdivision
df=df.drop(['subdivision'],axis=1)


##8 present_use
present_use_dummies=pd.get_dummies(df['present_use'],prefix='present_use',dummy_na=False)
df=pd.concat([df,present_use_dummies],axis=1)
df=df.drop(['present_use'],axis=1)



##9 log transform
log_cols = ['sale_price', 'land_val', 'imp_val', 'sqft_lot', 'sqft', 'sqft_1', 'sqft_fbsmt', 'garb_sqft', 'gara_sqft']
log_cols=['sale_price','land_val','imp_val']
plot_num_cols(log_cols)


df['sqft_over_sqftlot']=df['sqft']/df['sqft_lot']
df['sqft1_over_sqftlot']=df['sqft_1']/df['sqft_lot']
df['sqft_fbsmt_over_sqftlot']=df['sqft_fbsmt']/df['sqft_lot']
df['garb_sqft_over_sqftlot']=df['garb_sqft']/df['sqft_lot']
df['gara_sqft_over_sqftlot']=df['gara_sqft']/df['sqft_lot']
df=df.drop(['sqft','sqft_lot','sqft_1','sqft_fbsmt','garb_sqft','gara_sqft'],axis=1)


df.describe().T


num_cols=df.select_dtypes(include=[np.number]).columns.to_list()
plot_num_cols(num_cols)


cat_cols=df.select_dtypes(include=['object','category']).columns.to_list()
plot_cat_cols(cat_cols)

