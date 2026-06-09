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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


train_df.head(10)


train_df.info()


train_df.nunique()


train_df.isnull().count()


train_df.isnull().sum()


train_df.describe()


pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


# sns.lineplot(data=train_df)


list(train_df.columns)


# Examining the Distribution of Sales (num_sold)
plt.figure(figsize=(10,6))
sns.histplot(train_df['num_sold'], bins=60, kde=True)
plt.title("Distribution of Stricker Sales")
plt.xlabel("Number of Stricker Sold")
plt.ylabel("Frequency")
plt.show()


# Ploting num_sold over time to observe trends.
plt.figure(figsize=(12,6))
train_df.groupby('date')['num_sold'].sum().plot()
plt.title('Total Stricker Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Stricker Sold')
plt.show()


# plt.figure(figsize=(12,6))
# sns.lineplot(data=train_df, x='date', y='num_sold', hue='country')
# plt.title('Total Stricker Sales Over Time by Country')
# plt.xlabel('Date')
# plt.ylabel('Number of Strickers Sold')
# plt.xticks(rotation=45)
# plt.legend(title='Country')
# plt.show()


# # Creating a FacetGrid with separate line plots for each country
# g = sns.FacetGrid(train_df, col="country", col_wrap=1, height=10, aspect=2)
# g.map(sns.lineplot, 'date', 'num_sold')

# g.set_titles("{col_name}")
# g.set_axis_labels("Date", "Number of Strickers Sold")
# g.set_xticklabels(rotation=45, ha='right')

# # Adjusting the layout for readability
# g.tight_layout()

# plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(x='country', y='num_sold', data=train_df)
plt.title('Stricker Sales Distribution by Coutnry')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='store', y='num_sold', data=train_df)
plt.title('Stricker Sales Distribution by Store')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='product', y='num_sold', data=train_df)
plt.title('Stricker Sales Distribution by Product')
plt.show()


train_df['day_of_week'] = pd.to_datetime(train_df['date']).dt.dayofweek
plt.figure(figsize=(10,6))
sns.boxplot(x='day_of_week', y='num_sold', data=train_df)
plt.title('Stricker Sales Distribution by Day of the Week')
plt.show()


train_df['date'] = pd.to_datetime(train_df['date'])
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_week'] = train_df['date'].dt.dayofweek


# train_df['lag_1'] = train_df['num_sold'].shift(1)
# train_df['lag_7'] = train_df['num_sold'].shift(7)
# train_df['lag_30'] = train_df['num_sold'].shift(30)


# train_df['rolling_mean_7'] = train_df['num_sold'].rolling(window=7).mean()
# train_df['rolling_std_7'] = train_df['num_sold'].rolling(window=7).std()


train_df.fillna(0.1, inplace=True)


train_df = pd.get_dummies(train_df, columns=['country', 'store', 'product'])


train_df.head()


train_df.isnull().sum()


from sklearn.model_selection import train_test_split

train_set, val_set = train_test_split(train_df, test_size=0.2, shuffle=True)

X_train = train_set.drop(['date', 'num_sold'], axis=1)
y_train = train_set['num_sold']
X_val = val_set.drop(['date', 'num_sold'], axis=1)
y_val = val_set['num_sold']


from sklearn.ensemble import RandomForestRegressor 

model = RandomForestRegressor(n_estimators=200, random_state=42)


model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_percentage_error

y_pred = model.predict(X_val)
mape = mean_absolute_percentage_error(y_val, y_pred)
print(f'MAPE on validation set: {mape:}')


test_df['date'] = pd.to_datetime(test_df['date'])
test_df['day_of_week'] = test_df['date'].dt.dayofweek
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df = pd.get_dummies(test_df, columns=['country', 'store', 'product'])


test_df.isnull().sum()


X_test = test_df.drop(['date'], axis=1)
predictions = model.predict(X_test)


submission = pd.DataFrame({'id': test_df['id'], 'num_sold': predictions})
submission.to_csv('submission.csv', index=False)

