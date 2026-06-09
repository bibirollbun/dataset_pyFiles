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


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


for n in df.columns:
    print(n , df[n].nunique())


df = df.dropna()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#df['date'] = pd.to_datetime(df['date'])
df['country_encoded'] = le.fit_transform(df['country'])
df['country_store'] = le.fit_transform(df['store'])
df['country_product'] = le.fit_transform(df['product'])

df.head()


x_train = df[['country_encoded','country_store','country_product' ]]
y_train = df[['num_sold']]



import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)



scaler_y =StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)


test_data['date'] = pd.to_datetime(test_data['date'])



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

test_data['date'] = pd.to_datetime(test_data['date'])
test_data['country_encoded'] = le.fit_transform(test_data['country'])
test_data['country_store'] = le.fit_transform(test_data['store'])
test_data['country_product'] = le.fit_transform(test_data['product'])

test_data.head()


x_test = test_data[['country_encoded','country_store','country_product' ]]



scaler_x = StandardScaler()
x_test_scaled = scaler_x.fit_transform(x_test)




from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


kmeans = KMeans(n_clusters=9, random_state=42)
x_train['Cluster'] = kmeans.fit_predict(x_train)



x_train = x_train[['country_encoded','country_store','country_product' ,'Cluster']]



import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder




model = RandomForestRegressor(n_estimators=200 ,max_features=7 , max_depth=10, random_state=42)
model.fit(x_train_scaled, y_train_scaled)



x_test.head()


kmeans = KMeans(n_clusters=9, random_state=42)
x_test['Cluster'] = kmeans.fit_predict(x_test)



x_test.head()


x_test = x_test[['country_encoded','country_store','country_product' ,'Cluster']]



scaler_x = StandardScaler()
x_test_scaled = scaler_x.fit_transform(x_test)



y_pred = model.predict(x_test_scaled)



y_pred



predictions_reshaped = y_pred.ravel().reshape(-1, 1)
predictions_original = scaler_y.inverse_transform(predictions_reshaped)



predictions_original


y_pred_series = pd.Series(predictions_original.ravel())



aligned_y_train, aligned_y_pred = y_train.align(y_pred_series, axis=0, join='inner')



from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(aligned_y_train, aligned_y_pred) 
print(f"MAPE: {mape}")




submission = pd.DataFrame({'id': test_data['id'], 'num_sold': y_pred_series})
submission.to_csv('submission.csv', index=False)





