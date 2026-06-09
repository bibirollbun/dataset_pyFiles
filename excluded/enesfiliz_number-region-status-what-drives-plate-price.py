import supplemental_english as sup  
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from statsmodels.tsa.seasonal import seasonal_decompose




# Plotting configuration
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
sample_submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')




print("Train Data Shape:", train.shape)
print(train.head())

print("Test Data Shape:", test.shape)
print(test.head())

print("Sample Submission Preview:")
print(sample_submission.head())


print(train.info())
print(train.isnull().sum())
print(test.isnull().sum())
print(train.info())


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

for df in [train, test]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday

print(train[['date', 'year', 'month', 'day', 'weekday']].head())


def extract_region(plate):
    return plate[-2:]

train['region_code'] = train['plate'].apply(extract_region)
test['region_code'] = test['plate'].apply(extract_region)

train['region_code'] = pd.to_numeric(train['region_code'], errors='coerce')
test['region_code'] = pd.to_numeric(test['region_code'], errors='coerce')

region_map = {}
for region_name, codes in sup.REGION_CODES.items():
    for c in codes:
        region_map[int(c)] = region_name

train['region_name'] = train['region_code'].map(region_map)
test['region_name'] = test['region_code'].map(region_map)

print(train[['plate', 'region_code', 'region_name']].head())


def is_government_plate(plate):
    return int(plate in sup.GOVERNMENT_CODES)

train['is_gov'] = train['plate'].apply(is_government_plate)
test['is_gov'] = test['plate'].apply(is_government_plate)

print(train[['plate', 'is_gov']].head())


train.info()
print(train.isnull().sum())
train['price'] = train.groupby('region_name')['price'].transform(lambda x: x.fillna(x.median()))



def datareview(dataframe):
    print("******top 10 observations******")
    print(dataframe.head(10))
    
    print("******Variable names******")
    print(dataframe.columns)
    
    print("******Descriptive statistics********")
    # tranzpozunu alÄ±yoruz daha okunaklÄ± hale getirmek iÃ§in
    print(dataframe.describe().T)
    
    print("*****Missing Values********")
    print(dataframe.isnull().sum())

    print("******Variable types, reviews.********")
    print(dataframe.info())

    print(dataframe.nunique())

datareview(train)


num_cols = ['price', 'year', 'month', 'day', 'weekday', 'region_code']
train[num_cols].hist(bins=30, figsize=(15, 10))



from scipy.stats import zscore
train['price_z'] = zscore(train['price'])
train[train['price_z'].abs() > 3]



train.groupby('region_code')['price'].describe()



train['date'] = pd.to_datetime(train['date'])

monthly_stats = train.set_index('date').resample('M').agg({
    'price': ['mean', 'count']
})
monthly_stats.columns = ['avg_price', 'count']
monthly_stats.plot(secondary_y='avg_price', title="AylÄ±k Ortalama Fiyat ve Ä°ÅŸlem SayÄ±sÄ±")



top_regions = train['region_name'].value_counts().head(10).index
sns.boxplot(data=train[train['region_name'].isin(top_regions)], x='region_name', y='price')
plt.xticks(rotation=45)



sns.violinplot(data=train, x='is_gov', y='price')
sns.boxplot(data=train, x='weekday', y='price')



train['is_weekend'] = train['weekday'].isin([5, 6]).astype(int)
train['age_of_vehicle'] = 2025 - train['year']
train['region_freq'] = train['region_code'].map(train['region_code'].value_counts())



corr_matrix = train.corr(numeric_only=True).drop('is_gov', axis=0).drop('is_gov', axis=1)


plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()



X = train[['price', 'region_code', 'year']].dropna()
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42).fit(X_scaled)
train['cluster'] = kmeans.labels_

sns.pairplot(train, hue='cluster', vars=['price', 'year', 'region_code'])



result = seasonal_decompose(train.set_index('date')['price'].resample('W').mean(), model='additive')
result.plot()


