import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error 

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MSE


X_pred=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
df=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')


df.info()


df.describe()


X=df.drop(['id','BeatsPerMinute'],axis=1)

y=df['BeatsPerMinute']
id=X_pred.pop('id')


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


model=keras.Sequential([
    layers.Dense(units=9,input_shape=(9,)),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    layers.Dense(units=4, activation= 'relu'),
    layers.Dropout(rate=0.3),
    layers.Dense(units=2,activation= 'relu'),
    layers.Dense (units=1, activation= 'relu')
                 
])


model.compile(loss = "MSE",
                  optimizer = Adam(learning_rate=0.001),
                  metrics =["MSE"]
                  )


early_stopping = EarlyStopping(
  	   min_delta=1, # minimium amount of change to count as an improvement
   	  patience=10, # how many epochs to wait before stopping
   	  restore_best_weights=True,
)


history = model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    batch_size=512,
    epochs=100,
    callbacks=[early_stopping], # put your callbacks in a list
)


history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['MSE', 'val_MSE']].plot();


val_pred=model.predict(val_X)


print(mean_squared_error(val_y,val_pred,squared=False))


prediction=model.predict(X_pred)


predictions = prediction.flatten()


output = pd.DataFrame({ 'id':id,
                       'Target': predictions})


output.set_index('id')


output.to_csv('submission.csv', index=False)




