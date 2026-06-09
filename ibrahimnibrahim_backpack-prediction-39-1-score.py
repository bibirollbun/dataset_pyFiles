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


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
train_ex = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",index_col='id')
Id=test.index


train=pd.concat([train, train_ex], axis =0)


train.head()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for col in ['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color']:
    train[col]=le.fit_transform(train[col])
    test[col]=le.transform(test[col])
train.head()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler=StandardScaler()


x=train.drop('Price',axis=1)
y=train['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test=scaler.transform(test)


x_train = np.nan_to_num(x_train, nan=np.nanmean(x_train))
y_train = np.nan_to_num(y_train, nan=np.nanmean(y_train))

x_test = np.nan_to_num(x_train, nan=np.nanmean(x_test))
y_test = np.nan_to_num(y_train, nan=np.nanmean(y_test))



import tensorflow as tf

class SquaredDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):  
        # Get kernel_regularizer from kwargs or set to None
        self.kernel_regularizer = kwargs.pop('kernel_regularizer', None)
        # Pass remaining kwargs to super()
        super(SquaredDense, self).__init__(**kwargs)  
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.a = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 regularizer=self.kernel_regularizer)  # Apply regularizer to 'a'
        self.b = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 regularizer=self.kernel_regularizer)  # Apply regularizer to 'b'
        self.c = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        x = tf.matmul(tf.square(inputs), self.a) + tf.matmul(inputs, self.b) + self.c
        x = self.activation(x) if self.activation is not None else x
        return x


from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout

model = Sequential([
    Input(shape=(x_train.shape[1],)),  
    Dense(256, kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Activation('swish'),
    Dropout(0.1),
    
    Dense(256, kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Activation('swish'),
    Dropout(0.1),
    
    Dense(256, kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Activation('swish'),
    SquaredDense(1)
])
# swish

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[RootMeanSquaredError()])

# Show the model summary
model.summary()


from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

es=EarlyStopping(patience=5,restore_best_weights=True,monitor="val_root_mean_squared_error",mode="min")
lr = ReduceLROnPlateau(monitor='val_root_mean_squared_error', mode = 'min', patience=2, factor=0.5, min_lr=1e-30, verbose = 2)
callback=[es,lr]


history=model.fit(x_train,y_train,epochs=15,validation_data=(x_test,y_test),callbacks=[callback],batch_size=1024)


model.evaluate(x_test,y_test)


y_test_pred = model.predict(test)
y_test_pred = np.nan_to_num(y_test_pred, nan=np.nanmean(y_test_pred))
y_test_pred = np.array(y_test_pred).flatten() 
y_test_pred = pd.Series(y_test_pred).fillna(y_test_pred.mean()) 
submission = pd.DataFrame({'id': Id, 'Price': y_test_pred})

submission.to_csv('submission.csv', index=False)
display(submission)

