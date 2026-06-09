import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/housing-price-prediction/Housing.csv")

df.head()


df.tail()


df.info()


df.describe()


df.describe(include='object')


df.isnull().sum()


sns.histplot(df['price'], kde=True, bins=30)
plt.title('Price Distribution')
plt.show()


sns.histplot(df['area'], kde=True, bins=30)
plt.title('Area Distribution')
plt.show()


sns.histplot(df['bedrooms'], kde=True, bins=30)
plt.title('Bedrooms Distribution')
plt.show()


sns.histplot(df['stories'], kde=True, bins=30)
plt.title('Stories Distribution')
plt.show()


sns.countplot(x='bedrooms', data=df, palette='Set2')
plt.show()


sns.heatmap(df[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']].corr(), annot=True, cmap='coolwarm')


sns.scatterplot(x='area', y='price', data=df)
plt.title('Price vs Area')
plt.show()


sns.scatterplot(x='bedrooms', y='price', data=df)
plt.title('Price vs Bedroom')
plt.show()


sns.scatterplot(x='stories', y='price', data=df)
plt.title('Price vs stories')
plt.show()


cat_features = ['airconditioning', 'mainroad', 'hotwaterheating', 'furnishingstatus', 'basement', 'guestroom']

for cat in cat_features:
    sns.boxplot(data=df, x=cat, y='price')
    plt.title(f'Price by Air {cat}')
    plt.show()


df.head()


ct = pd.crosstab(df['guestroom'], df['basement'])
ct.plot(kind='bar', stacked=True, colormap='viridis')


ct = pd.crosstab(df['airconditioning'], df['guestroom'])
ct.plot(kind='bar', stacked=True, colormap='viridis')


ct = pd.crosstab(df['airconditioning'], df['mainroad'])
ct.plot(kind='bar', stacked=True, colormap='viridis')


ct = pd.crosstab(df['airconditioning'], df['hotwaterheating'])
ct.plot(kind='bar', stacked=True, colormap='viridis')


ct = pd.crosstab(df['airconditioning'], df['furnishingstatus'])
ct.plot(kind='bar', stacked=True, colormap='viridis')


df.head()


sns.pairplot(df, vars=['area', 'price', 'bedrooms', 'stories', 'bathrooms', 'parking'])
plt.suptitle('Pairplot ')
plt.show()


sns.boxplot(x=df['price'])




