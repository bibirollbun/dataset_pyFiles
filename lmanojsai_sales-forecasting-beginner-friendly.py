# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=['date'])
df.head()


df.info()


df.describe()


df.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


f, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.countplot(data=df, x='country', ax=axes[0], color='#4682B4')
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(data=df, x='store', ax=axes[1], color='#20B2AA')
axes[1].tick_params(axis='x', rotation=45)

sns.countplot(data=df, x='product', ax=axes[2], color='#BA55D3')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.kdeplot(df['num_sold'].dropna(), fill=True, color='#4682B4')
plt.title(f'Histogram of num_sold')
    
plt.subplot(1, 2, 2)
stats.probplot(df['num_sold'].dropna(), dist="norm", plot=plt)
plt.title(f'Q-Q Plot of num_sold')
    
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 3))
sns.boxplot(data=df, x='num_sold')
plt.title(f'Boxplot of num_sold')

plt.show()


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.barplot(data=df, y='num_sold', x='product', palette='coolwarm')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, y='num_sold', x='country')
plt.tight_layout()
plt.show()


sns.violinplot(data=df, x='store', y='num_sold', palette="Pastel2")


daily_sales = df.groupby('date')['num_sold'].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=daily_sales, x='date', y='num_sold', color='#4682B4', alpha=0.5, label='Daily Sales')

daily_sales['7_day_ma'] = daily_sales['num_sold'].rolling(window=7).mean()
sns.lineplot(data=daily_sales, x='date', y='7_day_ma', color='red', linewidth=2, label='7-day Moving Average')

plt.tight_layout()
plt.show()


monthly_sales = df.groupby(pd.Grouper(key='date', freq='M'))['num_sold'].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_sales, x='date', y='num_sold', color='#20B2AA', linewidth=2)
plt.title('Monthly Sales Over Time', pad=15, fontsize=14, fontweight='bold')
monthly_sales['3_month_ma'] = monthly_sales['num_sold'].rolling(window=3).mean()
sns.lineplot(data=monthly_sales, x='date', y='3_month_ma', color='red', linewidth=2, label='3-month Moving Average')

plt.tight_layout()
plt.show()


df.dropna(inplace=True)


# Finding the IQR
percentile25 = df['num_sold'].quantile(0.25)
percentile75 = df['num_sold'].quantile(0.75)

iqr = percentile75 - percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

df = df.copy()

df['num_sold'] = np.where(
    df['num_sold'] > upper_limit,
    upper_limit,
    np.where(
        df['num_sold'] < lower_limit,
        lower_limit,
        df['num_sold']
    )
)


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df.drop(columns=['date', 'id'], inplace=True)


X = df.drop(columns='num_sold')
y = df['num_sold']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from category_encoders import BinaryEncoder 

transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', sparse_output=False), ['country', 'store', 'product'])
], remainder='passthrough')

scaler = StandardScaler()
xgb_regr = XGBRegressor(
        n_estimators=300,
        learning_rate=0.2,
        gamma=0.1,
        max_depth=10,
        min_child_weight=3,
        reg_alpha = 0.1,
        reg_lambda = 0.1,
        subsample = 0.9,
        colsample_bytree = 0.9
)

pipe = Pipeline([
    ('transformer', transformer),
    ('scaler', scaler),
    ('model', xgb_regr)
])


pipe.fit(X_train, y_train)


from sklearn.metrics import r2_score, mean_absolute_percentage_error, get_scorer_names

predictions = pipe.predict(X_test)
print(f"R2 SCORE: {r2_score(y_test, predictions)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, predictions)}")


# cross validation using cross_val_score
from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2').mean()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=['date'])
test_df.sample(5)


test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day

id_series = test_df['id']
test_df.drop(columns=['date', 'id'], inplace=True)


preds = pipe.predict(test_df)


submission = pd.DataFrame({
    "id": id_series,
    "num_sold": preds
})
submission.to_csv('submission.csv', index=False)




