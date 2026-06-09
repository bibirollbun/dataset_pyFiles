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
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df.info()


print(f'Data Missing : {df.isnull().sum().sum()}')
print(f'Data Size: {df.size}')
print(f'Data Missing Rate: {(df.isnull().sum().sum() / df.size) * 100:.2f}%')



msno.matrix(df, width_ratios=(5, 5))


df.select_dtypes(include='object').isnull().sum()


cat_col_with_missing = df.select_dtypes(include='object').columns


cat_col_with_missing


df.select_dtypes(exclude='object').isnull().sum()


for col in cat_col_with_missing:
    df[col] = df[col].fillna(value=df[col].mode)


df[cat_col_with_missing].isnull().sum()


df.isnull().sum()


df['Weight Capacity (kg)'].fillna((df['Weight Capacity (kg)'].mean()), inplace=True)


df.isnull().sum()


df = pd.get_dummies(df, dtype='int', drop_first=True)


plt.figure(figsize=[10, 10])
sns.heatmap(df.corr(), lw=.1, cmap='coolwarm')


from sklearn.model_selection import train_test_split


X = df.drop(['id', 'Price'], axis=1)
y = df['Price']


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, random_state=0)


test_df.isnull().sum()


for col in test_df.select_dtypes(include='object').columns:
    test_df[col] = test_df[col].fillna(value=test_df[col].mode)


test_df.select_dtypes(include='object').isnull().sum()


test_df.select_dtypes(exclude='object').isnull().sum()


test_df['Weight Capacity (kg)'] = test_df['Weight Capacity (kg)'].fillna(value=test_df['Weight Capacity (kg)'].mean())


test_df['Weight Capacity (kg)'].isnull().sum()


test_df.isnull().sum()


test_df = pd.get_dummies(test_df, dtype='int', drop_first=True)


test_df.columns


test_df.head()





X_test = test_df.drop('id', axis=1)


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks


X_train.shape


early_stopping = callbacks.EarlyStopping(
    min_delta=.001,
    patience=10,
    restore_best_weights=True
)


model = Sequential([
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dropout(.3),
    
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(.3),
    
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    
    layers.BatchNormalization(),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss= 'mse',
    metrics=[keras.metrics.RootMeanSquaredError()]
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64,
    epochs=50,
    callbacks=[early_stopping]
)


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print(("Minimum Validation Loss: {:0.4f}").format(history_df['root_mean_squared_error'].min()))


history_df


preds = model.predict(X_test)


preds


output = pd.DataFrame(
    {
        'id': test_df['id'],
        'Price': preds.reshape(-1)
    }
)


output.to_csv('submission.csv', index=False)


output




