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


from tensorflow.keras import models, layers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df.head()


df['BMI'] = df['Weight']/df['Height'] ** 2


X = pd.get_dummies(df[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp', 'BMI']])
y = df['Calories']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # X should be a NumPy array or DataFrame


model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='relu')
])


def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1))))


model.compile(
    optimizer='adam',
    loss=rmsle,
    metrics=['mae']
)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(
    X_scaled, y,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, lr_schedule],
    verbose=1
)


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('RMSLE')
plt.ylim([0, 0.1])
plt.legend(loc='upper right')


test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_df['BMI'] = test_df['Weight']/test_df['Height'] ** 2
test = pd.get_dummies(test_df[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp', 'BMI']])
test_scaled = scaler.transform(test)


pred = model.predict(test_scaled)
test_df = pd.DataFrame({'id': test_df['id'].values, 'Calories': pred[:, 0]})
test_df.to_csv('predictions.csv', index=False)
print("Prediction results saved to predictions.csv")


test_df.head()

