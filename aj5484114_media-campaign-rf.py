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


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor



df_media = pd.read_csv('/kaggle/input/playground-series-s3e11/train.csv')
df_media_test = pd.read_csv('/kaggle/input/playground-series-s3e11/test.csv')


df_media.head()


df_media.dtypes


df_media_test.dtypes


df_media.isnull().sum()


target_variable = 'cost'

plt.figure(figsize=(10, 6))
sns.histplot(df_media[target_variable],kde=True,bins=50)
plt.title(f'Distribution of {target_variable}')
plt.xlabel(target_variable)
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(df_media.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


numerical_cols = df_media.select_dtypes(include=['float64', 'int64']).columns

for col in numerical_cols:
    if col != 'store_sales(in millions)':
        sns.boxplot(x=col,data = df_media)
        plt.title(f"Boxplot of {col}")
        plt.show()


binary_cols = ['recyclable_package', 'low_fat', 'coffee_bar', 'video_store', 
               'salad_bar', 'prepared_food', 'florist']

for col in binary_cols:
    sns.barplot(x=col, y='store_sales(in millions)', data=df_media)
    plt.title(f"Avg Sales by {col}")
    plt.show()


important_numerics = ['unit_sales(in millions)', 'store_sqft', 'avg_cars_at home(approx).1']

for col in important_numerics:
    sns.scatterplot(x=df_media[col], y=df_media['store_sales(in millions)'])
    plt.title(f"Sales vs {col}")
    plt.show()


df_media['num_amenities'] = df_media[['coffee_bar', 'video_store', 'salad_bar', 'prepared_food', 'florist']].sum(axis=1)



amenity_cols = [col for col in df_media.columns if any(key in col.lower() for key in ['coffee', 'video', 'salad', 'food', 'florist'])]

df_media.drop(columns=amenity_cols, inplace=True)



df_media.head()


X = df_media.drop(columns=['id', 'cost'])
y = df_media['cost']


df_media_test['num_amenities'] = df_media_test[['coffee_bar', 'video_store', 'salad_bar', 'prepared_food', 'florist']].sum(axis=1)



amenity_cols = [col for col in df_media_test.columns if any(key in col.lower() for key in ['coffee', 'video', 'salad', 'food', 'florist'])]

df_media_test.drop(columns=amenity_cols, inplace=True)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


test_ids = df_media_test['id']

X_test = df_media_test.drop(columns=['id'])

X_test_scaled = scaler.transform(X_test)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)


predictions = model.predict(X_test_scaled)


submission = pd.DataFrame({
    'id': test_ids,
    'cost': predictions
})

submission.to_csv('media_submission.csv', index=False)




