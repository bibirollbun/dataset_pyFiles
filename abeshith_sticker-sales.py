pip install --upgrade tpot


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from tpot import TPOTRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import warnings
warnings.simplefilter(action = 'ignore')


train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train_data.head()


train_data.isnull().sum()


train_data.info()


train_data.columns


data = train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

target_mean = data['num_sold'].mean()
data['num_sold'].fillna(target_mean, inplace=True)


data['date'] = pd.to_datetime(data['date'])

data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

data['country'].unique()


plt.figure(figsize=(10, 6))
sns.barplot(data=train_data, x='product', y='num_sold', hue='store', ci=None)
plt.title('Product Sales Across Stores')
plt.xlabel('Product')
plt.ylabel('Number Sold')
plt.show()


country_sales = train_data.groupby('country')['num_sold'].sum().reset_index()

# choropleth map
fig = px.choropleth(
    country_sales,
    locations='country',
    locationmode='country names',
    color='num_sold',
    title='Total Sales by Country',
    color_continuous_scale='Viridis'
)
fig.show()


monthly_sales = train_data.groupby(['year', 'month'])['num_sold'].sum().unstack()

plt.figure(figsize=(10, 6))
sns.heatmap(monthly_sales, cmap='coolwarm', annot=True, fmt='.0f')
plt.title('Monthly Sales Heatmap')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x='month', y='num_sold', hue='store', style='country')
plt.title('Monthly Sales Distribution by Store and Country')
plt.xlabel('Month')
plt.ylabel('Number Sold')
plt.show()


country_date_sales = train_data.groupby(['country', 'date']).agg({'num_sold': 'sum'}).reset_index()
country_coords = {
    'Norway': [60.472, 8.4689],
    'Sweden': [60.128, 18.6435],
    'Finland': [61.924, 25.7482],
    'Canada': [56.1304, -106.3468],
    'Italy': [41.8719, 12.5674],
    'Kenya': [-1.286389, 36.817223],
    'Singapore': [1.352083, 103.819836]
}

country_date_sales['latitude'] = country_date_sales['country'].map(lambda x: country_coords[x][0])
country_date_sales['longitude'] = country_date_sales['country'].map(lambda x: country_coords[x][1])

# Bubble map
fig = px.scatter_geo(
    country_date_sales,
    lat='latitude',
    lon='longitude',
    size='num_sold',
    color='country',
    animation_frame='date',
    title='Sales Over Time (Bubble Map)',
    projection='natural earth'
)
fig.show()


sunburst_data = train_data.groupby(['country', 'store', 'product']).agg({'num_sold': 'sum'}).reset_index()
fig = px.sunburst(
    sunburst_data,
    path=['country', 'store', 'product'],
    values='num_sold',
    title='Sales Breakdown by Country, Store, and Product',
    color='num_sold',
    color_continuous_scale='RdBu',
)
fig.show()


train_data['num_sold'].fillna(train_data['num_sold'].mean(), inplace=True)


def preprocess_date(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Apply sine and cosine transformations to cyclic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df.drop(columns=['date'], inplace=True)


preprocess_date(train_data)
preprocess_date(test_data)


label_encoder = LabelEncoder()

train_data['country'] = label_encoder.fit_transform(train_data['country'])
test_data['country'] = label_encoder.transform(test_data['country'])
train_data['store'] = label_encoder.fit_transform(train_data['store'])
test_data['store'] = label_encoder.transform(test_data['store'])
train_data['product'] = label_encoder.fit_transform(train_data['product'])
test_data['product'] = label_encoder.transform(test_data['product'])


X = train_data.drop(columns=['id', 'num_sold'])
y = train_data['num_sold']
X_test = test_data.drop(columns=['id'])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.pipeline import Pipeline
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20, random_state=42)


tpot.fit(X_train, y_train)


val_score = tpot.score(X_val, y_val)
print(f"Validation R^2 Score: {val_score}")


test_predictions = tpot.predict(X_test)


sample_submission['num_sold'] = test_predictions
sample_submission.to_csv('submission.csv', index=False)







