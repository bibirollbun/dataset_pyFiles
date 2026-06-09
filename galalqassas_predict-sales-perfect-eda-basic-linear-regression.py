import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

pd.set_option('display.float_format', lambda x: '%.3f' % x)
palette = sns.color_palette('Spectral')
pastel = sns.color_palette('pastel')
# This lets us see all of the columns, preventing Juptyer from redacting them.
pd.set_option('display.max_columns', None)


df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df


df.info()


df.describe()


df.isna().sum()


df.isna().sum() / df.shape[0]


df.duplicated().sum()


df[df.isna().any(axis=1)]


df.num_sold = df.num_sold.fillna(df.num_sold.median())
df.isna().sum()


sns.countplot(data=df, x='country')


sns.countplot(data=df, x='store')


sns.countplot(data=df, x='product')


df.num_sold.hist()


q1, q3 = df.num_sold.quantile([.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr


outliers = df[(df.num_sold < lower) | (df.num_sold > upper)]
outliers


df['num_sold'].plot.box()


df['num_sold'] = (
    df.num_sold
    .mask((df.num_sold < lower) | (df.num_sold > upper))  # mark outliers as NaN
    .interpolate(method='polynomial', order=3)  # fill NaN gaps with linear interpolation
)


df['num_sold'].plot.box()


num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(include='object').columns


corr = df[num_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='jet', annot=True)


df[cat_cols].nunique()


df.date = pd.to_datetime(df.date)
df['year'] = df.date.dt.year
df['month'] = df.date.dt.month
df['day'] = df.date.dt.day

for col in ['month', 'day']:
    max_val = 12 if col == 'month' else 31
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/max_val)


df['day_of_week'] = df['date'].dt.dayofweek  
df['is_weekend'] = (df['date'].dt.weekday >= 5).astype(np.uint8)
df['quarter'] = df['date'].dt.quarter 


df = df.drop('date', axis=1)
df


df = pd.get_dummies(df, dtype=np.uint8)


from sklearn.model_selection import train_test_split
y = df.num_sold
X = df.drop('num_sold', axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=11, shuffle=True)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
lin_reg.score(X_train, y_train), lin_reg.score(X_val, y_val)


from sklearn.preprocessing import PolynomialFeatures
for i in range(1, 4):
    poly = PolynomialFeatures(degree=i)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    lin2 = LinearRegression()
    lin2.fit(X_train_poly, y_train)
    print(lin2.score(X_train_poly, y_train), lin2.score(X_val_poly, y_val))

