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


df=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv').set_index('id')
df.head()


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.05, random_state = 101)
train.shape,test.shape


x_train=train.drop(['day','rainfall','winddirection'],axis=1)
x_test=test.drop(['day','rainfall','winddirection'],axis=1)

y_train=train['rainfall']
y_test=test['rainfall']


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


import tensorflow as tf

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    verbose=1,
    mode='max',
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-10,
    verbose=1
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(9, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy', # Use binary cross-entropy for binary classification
              metrics=['accuracy'])

# Train the model
model.fit(x_train_scaled, y_train, epochs=100, batch_size=64, validation_data=(x_test_scaled, y_test),callbacks=[early_stopping,reduce_lr])


model.evaluate(x_test_scaled, y_test)


df_test= pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_test = df_test.drop(['winddirection','day'],axis=1)
test = scaler.transform(df_test)
df_subm= pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
y_pred_keras = model.predict(test).flatten()
# Save Submission
df_subm['rainfall'] = y_pred_keras
df_subm.to_csv('submission.csv', index=False)


display(df_subm)




