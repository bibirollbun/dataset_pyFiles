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


train = pd.read_csv('/kaggle/input/molecular-machine-learning/train.csv')
train.info()


test = pd.read_csv('/kaggle/input/molecular-machine-learning/test.csv')


train.select_dtypes(include=['int64','float64']).columns


train.info()


y = train['T80']
features = ['PrimeState', 'SDOS4.5', 'SDOS2.5', 'O19', 'O1', 'SurfaceCharge',
       'SDOS3.7', 'SDOS2.6', 'TDOS1.5', 'O10', 'TDOS1.6', 'O12',
       'LUMO(eV)', 'HAcceptors', 'SDOS2.7']
X = train[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_X, val_X, train_y, val_y = train_test_split(X_scaled, y, random_state=1)


input_shape = [train_X.shape[1]]

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),    
    layers.Dense(1)])

model.compile(
    optimizer='adam',
    loss='mae')

history = model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    batch_size=512,
    epochs=79)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));


test_X = test[features]
test_X_scaled = scaler.fit_transform(test_X)

#scaler = StandardScaler()
#test_X_scaled = scaler.fit_transform(test_X)

input_shape = [test_X_scaled.shape[1]]

predictions = model.predict(test_X)


#Generic submission formatting 

output = pd.DataFrame({'Batch_ID': test['Batch_ID'],
                       'T80': predictions.flatten()})
output.to_csv('submission.csv', index=False)


output

