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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

X1 = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
Y = X1['sale_price']
X1 = X1.drop(['sale_price'], axis=1)
X2 = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
X1['is_train'] = 1
X2['is_train'] = 0

combined = pd.concat([X1, X2], axis=0, ignore_index=True)


plt.figure(figsize=(12, 6))
sns.heatmap(combined.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


combined.info()


drop_features = ['sale_nbr','subdivision','sale_warning']
combined = combined.drop(drop_features, axis=1)


combined.head(30)


combined['sale_year']=pd.to_datetime(combined['sale_date'])



combined.head(10)


combined['sale_year'] = combined['sale_year'].dt.year


combined.head(10)


combined = combined.drop(['sale_date'], axis=1)


combined.head(10)


num_cols = combined.select_dtypes(include=['number']).columns
for col in num_cols:
    combined[col].fillna(combined[col].mean(), inplace=True)


cat_cols = combined.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    mode = combined[col].mode(dropna=True)
    if not mode.empty:
        combined[col].fillna(mode[0], inplace=True)
        
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
combined_encoded.head(10)


X_train = combined_encoded[combined_encoded['is_train'] == 1].drop(columns='is_train')
X_test = combined_encoded[combined_encoded['is_train'] == 0].drop(columns='is_train')

lower_model = GradientBoostingRegressor(loss='quantile', alpha=0.1)
upper_model = GradientBoostingRegressor(loss='quantile', alpha=0.9)
mean_model  = GradientBoostingRegressor(loss='squared_error')

lower_model.fit(X_train, Y)
upper_model.fit(X_train, Y)
mean_model.fit(X_train, Y)




y_lower = lower_model.predict(X_test)
y_upper = upper_model.predict(X_test)
y_mean  = mean_model.predict(X_test)


output = pd.DataFrame({'id': X_test.id, 'pi_lower': y_lower, 'pi_upper': y_upper})
output.to_csv('submission.csv', index=False)
output.head(20)

