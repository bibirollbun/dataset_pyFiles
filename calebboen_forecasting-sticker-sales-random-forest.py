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


import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


# load data
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=["date"])
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=["date"])


train.head()


test.head()


train.info()


train["num_sold"].isnull().sum()


train = train.dropna(subset=["num_sold"])
train.info()


# Import libraries for Ploting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# Group data by month and country
train['month'] = train['date'].dt.to_period('M').astype(str)
time_df = train.groupby(['month', 'country'], as_index=False)['num_sold'].sum()

# List of unique countries
countries = time_df['country'].unique()

# Loop through each country and create a line plot
for country in countries:
    country_data = time_df[time_df['country'] == country]

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=country_data, x='month', y='num_sold')

    plt.title(f'Monthly Sales in {country}', fontsize=14, pad=15)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Num Sold', fontsize=12)
    plt.xticks(ticks=range(0, len(country_data['month']), 3), labels=country_data['month'][::3], rotation=45)

    plt.tight_layout()
    plt.show()

# clean train
train.drop('month',axis=1,inplace=True)


combined = pd.concat([train.drop(columns=['num_sold']), test], axis=0, ignore_index=True)
combined = combined.drop(columns=['date'])
train_correlation=train.copy()


# label encode the values
for col in ['country', 'store', 'product']:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    train_correlation[col] = le.fit_transform(train_correlation[col])


correlation_data = train_correlation.drop(columns=['id', 'date'])

# Compute correlation matrix
correlation_matrix = correlation_data.corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


X_train = combined.iloc[:len(train), :]
y_train = train['num_sold']
X_test = combined.iloc[len(train):, :]


# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Evaluate on validation set
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model on the training split
model.fit(X_train_split, y_train_split)

# Predict on the validation set
val_predictions = model.predict(X_val)

# Calculate Mean Squared Error
mse = mean_squared_error(y_val, val_predictions)
print("Mean Squared Error:", mse)


# Predict on the test set
test_predictions = model.predict(X_test)

# Prepare submission dataframe
submission = pd.DataFrame({'id': test['id'], 'num_sold': test_predictions})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")




