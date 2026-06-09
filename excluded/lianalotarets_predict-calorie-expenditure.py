import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


df


def sex_encoder(df_orig, name_column: str):
    df = df_orig.copy()
    new_column = []
    for value in df[name_column]:
        if value == 'male':
            new_column.append(1)
        elif value == 'female':  
            new_column.append(0)

    df[name_column] = new_column
    return df


df = sex_encoder(df, 'Sex')
df_test = sex_encoder(df_test, 'Sex')


df


df.plot.box(
    column='Age',
)
df.plot.box(
    column='Height',
)
df.plot.box(
    column='Weight',
)
df.plot.box(
    column='Duration',
)
df.plot.box(
    column='Heart_Rate',
)
df.plot.box(
    column='Body_Temp',
)
plt.show()


X = df.drop(columns=['Calories', 'id'])
y = df['Calories']


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)


def rmsle(y_true, y_pred):
    y_true = tf.maximum(y_true, 0)  
    y_pred = tf.maximum(y_pred, 0)  
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log(y_true + 1) - tf.math.log(y_pred + 1))))


model = Sequential(
    [
    Input(shape=(X_train.shape[1],)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    # Dense(64, activation='relu'),
    # Dense(32, activation='relu'),
    Dense(1)
    ]
)

model.compile(optimizer=Adam(learning_rate=1e-4), loss=rmsle)


early_stop = EarlyStopping(
    monitor="val_loss", 
    patience=10, 
    min_delta=1e-4,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1
)

history = model.fit(
    X_train, y_train, 
    epochs=10,               
    batch_size=16,            
    validation_data=(X_test, y_test), 
    callbacks=[early_stop, reduce_lr],
    verbose=1                 
)


plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='train loss', color='green')
plt.plot(history.history['val_loss'], label='val loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


y_pred_test = model.predict(scaler.transform(df_test.drop(columns=['id'])))


df_sample_submission['id'] = df_test['id']
df_sample_submission['Calories'] = y_pred_test


df_sample_submission


df_sample_submission.to_csv('submission.csv', index=False)

