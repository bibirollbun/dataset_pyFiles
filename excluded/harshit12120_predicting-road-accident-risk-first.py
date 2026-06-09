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


import seaborn as sns 
import matplotlib.pyplot as plt


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')


df.info()


df.sample(10)


df.size


df.isnull().sum()


df['time_of_day'].value_counts()


df['school_season'] = df['school_season'].astype(int)
df['holiday'] = df['holiday'].astype(int)
df['public_road'] = df['public_road'].astype(int)
df['road_signs_present'] = df['road_signs_present'].astype(int)


df = pd.get_dummies(df, columns=['road_type','lighting','weather','time_of_day'],drop_first=True,dtype=int)


df.info()


df = df.drop(columns = ['id'])


corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)


X = df.drop('accident_risk', axis=1)
y = df['accident_risk']



df.describe().T



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam, SGD, Adadelta


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=16))
model.add(Dropout(0.2))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='mse',           
    metrics=['mae']       
)


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',    
    patience=10,           
    restore_best_weights=True,  
    verbose=1
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    verbose=1,
    callbacks=[early_stop]
)



import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['mae '], label='Validation Loss', linewidth=2)
plt.title('Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()



test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


test['school_season'] = df['school_season'].astype(int)
test['holiday'] = df['holiday'].astype(int)
test['public_road'] = df['public_road'].astype(int)
test['road_signs_present'] = df['road_signs_present'].astype(int)


test = pd.get_dummies(test, columns=['road_type','lighting','weather','time_of_day'],drop_first=True,dtype=int)


X_test_final = test.drop('id', axis=1) 
y_pred = model.predict(X_test_final)


submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

submission['accident_risk'] = y_pred  
submission.to_csv('submission1.csv', index=False)
print("✅ Submission file created: submission.csv")

print(submission.shape)
print(submission.head())




print(submission.shape)
print(submission.head())





