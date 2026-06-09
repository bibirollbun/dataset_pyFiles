import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio  
from plotly.subplots import make_subplots
import plotly.subplots as sp
from wordcloud import WordCloud
import optuna
import lightgbm as lgb
from IPython.display import display, HTML
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")
pio.renderers.default = 'iframe_connected'


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train.head()


train.info()


train.describe().T


numerical_column_names = train.select_dtypes(include=['number']).columns
print("Numerical Column Names:", numerical_column_names.tolist())


object_column_names = train.select_dtypes(include=['object']).columns
print("Object Column Names:", object_column_names.tolist())


print("Null Values in Data:")
train.isnull().sum()


print("Number of Rows in Train Data:",train.shape[0])
print("-"*20)
print("Number of Columns in Train Data:",train.shape[1])
print("-"*20)
print("Number of Duplicated Rows in Train Data:",train.duplicated().sum())


train = train.dropna()  


train.nunique()


cat_cols = ['country','store','product']


for cat in cat_cols:
    print(f"Distribution for '{cat}':\n")
    print(train[cat].value_counts(), '\n')


def transform_date(df, col):
    df[col] = pd.to_datetime(df[col])
    
    df['year'] = df[col].dt.year.astype('int')
    df['quarter'] = df[col].dt.quarter.astype('int')
    df['month'] = df[col].dt.month.astype('int')
    df['day'] = df[col].dt.day.astype('int')
    df['day_of_week'] = df[col].dt.dayofweek.astype('int')
    df['week_of_year'] = df[col].dt.isocalendar().week.astype('int')
    
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7)
    
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
    
    return df


train = transform_date(train, 'date')
test = transform_date(test, 'date')


train.head()


sns.set(style="whitegrid")

plt.figure(figsize=(7, 6))
sns.histplot(train['num_sold'], kde=True, bins=30, color='violet')

plt.title('Distribution of Sticker Sales (num_sold)', fontsize=16)
plt.xlabel('Number of Stickers Sold')
plt.ylabel('Frequency')

plt.show()


country_counts = train['country'].value_counts().reset_index()
country_counts.columns = ['country', 'count']

fig = px.bar(
    country_counts,
    x='country',
    y='count',
    color='count',
    title="Country-wise Sticker Sales",
    color_continuous_scale='YlGn',  
    labels={'count': 'Sales Count', 'country': 'Country'}
)

fig.update_layout(
    width=750,
    height=500,
    xaxis_title="Country",
    yaxis_title="Sales Count",
)

fig.show()


store_counts = train['store'].value_counts().reset_index()
store_counts.columns = ['store', 'count']  

fig = px.bar(
    store_counts,
    x='store',
    y='count',
    color='count',
    title="Store-wise Sticker Sales",
    color_continuous_scale='YlGn',  
    labels={'count': 'Sales Count', 'store': 'Store'}
)

fig.update_layout(
    width=750,
    height=500,
    xaxis_title="Store",
    yaxis_title="Sales Count",
)

fig.show()


product_counts = train['product'].value_counts().reset_index()
product_counts.columns = ['product', 'count']

fig = px.bar(
    product_counts,
    x='product',
    y='count',
    color='count',
    title="Product-wise Sticker Sales",
    color_continuous_scale='YlGn',
    labels={'count': 'Sales Count', 'product': 'Product'}
)

fig.update_layout(
    width=750,
    height=500,
    xaxis_title="Product",
    yaxis_title="Sales Count",
)

fig.show()


grouped_data = train.groupby(['country', 'store']).size().reset_index(name='count')

fig = px.bar(
    grouped_data,
    x='country',
    y='count',
    color='store',
    title='Sticker Sales by Country and Store',
    labels={'count': 'Sales Count', 'country': 'Country'},
    barmode='group',
    text='count',
    color_discrete_sequence=px.colors.sequential.Greens  
)

fig.update_layout(
    width=750,
    height=500,
    legend_title="Store Type",
    xaxis_title="Country",
    yaxis_title="Sales Count",
    template="plotly"
)

fig.show()


grouped_data = train.groupby(['country', 'product']).size().reset_index(name='count')

fig = px.bar(
    grouped_data,
    x='country',
    y='count',
    color='product',
    color_continuous_scale='greens',
    title='Country vs Product Distribution',
    labels={'count': 'Number of Records', 'country': 'Country'},
    text='count'
)

fig.update_layout(
    xaxis_title='Country',
    yaxis_title='Count',
    legend_title='Product',
    width=800,
    height=500
)

fig.show()


total_sales = train.groupby('date')['num_sold'].sum().reset_index()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=total_sales['date'],
        y=total_sales['num_sold'],
        mode='lines+markers',
        line=dict(color='green'),
        name='Total Sales'
    )
)

fig.update_layout(
    title='Total Sales Over Time',
    xaxis_title='Date',
    yaxis_title='Number of Products Sold',
    width=800,
    height=500
)

fig.show()


country_sales = train.groupby('country')['num_sold'].sum().reset_index()
country_sales.columns = ['Country', 'Total Sales'] 

country_sales = country_sales.sort_values(by='Total Sales', ascending=False)

fig = px.bar(
    country_sales,
    x='Country',  
    y='Total Sales', 
    text='Total Sales',
    color='Total Sales',
    color_continuous_scale=px.colors.sequential.Greens,
    title="Total Sales by Country"
)

fig.update_layout(
    xaxis_title="Country",
    yaxis_title="Total Sales",
    width=750,
    height=500,
)

fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')

fig.show()


total_sales_by_store = train.groupby('store')['num_sold'].sum().reset_index()

fig = px.bar(
    total_sales_by_store,
    x='store',
    y='num_sold',
    text='num_sold',
    color='num_sold',
    color_continuous_scale=px.colors.sequential.Greens,  
    title="Total Sales by Store"
)

fig.update_layout(
    xaxis_title="Store Type",
    yaxis_title="Total Sales",
    width=750,
    height=500
)

fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')

fig.show()


grouped_data = train.groupby(['year', 'country'])['num_sold'].sum().reset_index()

fig = px.line(
    grouped_data,
    x='year',
    y='num_sold',
    color='country',
    title='Sales Trends by Country (Year-wise)',
    labels={'year': 'year', 'num_sold': 'Number of Products Sold'},
    line_shape='linear',
)

fig.update_layout(
    legend_title='Country',
    width=750,
    height=500
)

fig.show()


grouped_data = train.groupby(['year', 'product'])['num_sold'].sum().reset_index()

fig = px.line(
    grouped_data,
    x='year',
    y='num_sold',
    color='product',  
    title='Sales Trends by Product (Year-wise)',
    labels={'year': 'Year', 'num_sold': 'Number of Products Sold'},
    line_shape='linear',
)

fig.update_layout(
    legend_title='Product',  
    width=750,
    height=500
)

fig.show()


grouped_data = train.groupby(['year', 'store'])['num_sold'].sum().reset_index()

fig = px.line(
    grouped_data,
    x='year',
    y='num_sold',
    color='store',  
    title='Sales Trends by Store Type (Year-wise)',
    labels={'year': 'Year', 'num_sold': 'Number of Products Sold'},
    line_shape='linear',
)

fig.update_layout(
    legend_title='Store Type',  
    width=750,
    height=500
)

fig.show()


text = ' '.join(train['product'].astype(str).values)
wordcloud = WordCloud(width=600, height=350, background_color='white').generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Product Names', fontsize=16)
plt.show()


products = ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode']
stores = ['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart']
countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']


num_subplots = len(stores) * len(countries)

cols = 3  
rows = (num_subplots // cols) + (num_subplots % cols > 0)  

for product in products:
    print(f"\n--- {product} ---\n")
    fig = plt.figure(figsize=(15, 5 * rows), dpi=100)  
    fig.subplots_adjust(hspace=0.25)
    
    for i, store in enumerate(stores):
        for j, country in enumerate(countries):
            ax = fig.add_subplot(rows, cols, i * len(countries) + j + 1)  
            selection = (train['country'] == country) & (train['store'] == store) & (train['product'] == product)
            selected = train[selection]
            selected.set_index('date').groupby('year')['num_sold'].mean().plot(ax=ax)
            ax.set_title(f"{country}:{store}")
    
    plt.show()


for product in ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode']:
    fig = plt.figure(figsize=(20, 30), dpi=100)
    fig.subplots_adjust(hspace=0.25)
    for i, store in enumerate(['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart']):
        for j, country in enumerate(['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']):
            ax = fig.add_subplot(6, 3, (i * 6 + j + 1))
            selection = (train['country'] == country) & (train['store'] == store) & (train['product'] == product)
            selected = train[selection].copy()
            selected['year'] = selected['date'].dt.year  
            selected['month'] = selected['date'].dt.month  
            
            for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016]:
                selected[selected['year'] == year].set_index('date').groupby('month')['num_sold'].mean().plot(ax=ax, label=year)
            ax.set_title(f"{product} | {country}:{store}")
            ax.legend()
    plt.show()


train = train.drop(columns=['date'])


cat_cols = ['country','store','product']


label_encoders = {}  
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le


label_encoders = {}  
for col in cat_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])
    label_encoders[col] = le


train.head()


corr_matrix = train.corr()

plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True)
plt.title('Correlation Matrix - Train Data')
plt.show()


train['num_sold'] = np.log1p(train['num_sold'])


sns.set(style="whitegrid")

plt.figure(figsize=(7, 6))
sns.histplot(train['num_sold'], kde=True, bins=30, color='violet')

plt.title('Distribution of Sticker Sales (num_sold)', fontsize=16)
plt.xlabel('Number of Stickers Sold')
plt.ylabel('Frequency')

plt.show()


X = train.drop(columns=['num_sold'])
y = train['num_sold']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


best_params = {
    'n_estimators': 4409,
    'learning_rate': 0.056211580299301386,
    'max_depth': 13,
    'reg_alpha': 0.2206493050847489,
    'reg_lambda': 0.08403278270607403,
    'min_child_samples': 29,
    'colsample_bytree': 0.502532761451806,
    'subsample': 0.9718829536052142,
    'objective': 'regression',  # Assuming regression task
    'metric': 'mape',  # Use MAPE for evaluation metric
    'random_state': 42
}

model = lgb.LGBMRegressor(**best_params)
model.fit(X, y)


test = test.drop(columns=['date'])


test.head()


submission_ids = test['id']
predictions = model.predict(test)


predictions = np.expm1(predictions)


submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': predictions 
})


submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())

