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


from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train.info()


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

test['Brand_numeric'] = pd.factorize(test['Brand'])[0]
test['Material_numeric'] = pd.factorize(test['Material'])[0]
test['Size_numeric'] = pd.factorize(test['Size'])[0]
test['Laptop Compartment_numeric'] = pd.factorize(test['Laptop Compartment'])[0]
test['Waterproof_numeric'] = pd.factorize(test['Waterproof'])[0]
test['Style_numeric'] = pd.factorize(test['Style'])[0]
test['Color_numeric'] = pd.factorize(test['Color'])[0]


train['Brand_numeric'] = pd.factorize(train['Brand'])[0]
train['Material_numeric'] = pd.factorize(train['Material'])[0]
train['Size_numeric'] = pd.factorize(train['Size'])[0]
train['Laptop Compartment_numeric'] = pd.factorize(train['Laptop Compartment'])[0]
train['Waterproof_numeric'] = pd.factorize(train['Waterproof'])[0]
train['Style_numeric'] = pd.factorize(train['Style'])[0]
train['Color_numeric'] = pd.factorize(train['Color'])[0]


numeric_features = train.select_dtypes(include=['int64','float64']).columns


# train.info()


df = train[numeric_features]


y = train['Price']
features = ['Compartments', 'Weight Capacity (kg)', 'Brand_numeric',
       'Material_numeric', 'Size_numeric', 'Laptop Compartment_numeric',
       'Waterproof_numeric', 'Style_numeric', 'Color_numeric']
X = train[features]

X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)





df.corr()['Price']


set1 = df[['Size_numeric','Material_numeric','Brand_numeric','Waterproof_numeric','Laptop Compartment_numeric']]
# set1


input_shape = [set1.shape[1]]

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),    
    layers.Dense(1)])

model.compile(
    optimizer='adam',
    loss='mae')

history = model.fit(
    set1, y,
    validation_split = 0.15,
    batch_size=512,
    epochs=10)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));


from sklearn.neighbors import KNeighborsRegressor


knn = KNeighborsRegressor()
knn.fit(set1, y)


knn.predict()


set1

