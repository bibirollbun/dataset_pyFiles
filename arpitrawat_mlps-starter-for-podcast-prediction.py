%load_ext cudf.pandas
import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.info()


test.info()


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


columns_to_encode=test.select_dtypes('object').columns.to_list()
le_dict = {}
for col in columns_to_encode:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = test[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    le_dict[col] = le



columns_to_impute=['Episode_Length_minutes','Guest_Popularity_percentage','Number_of_Ads']
imputer = SimpleImputer()
train[columns_to_impute] = imputer.fit_transform(train[columns_to_impute])
test[columns_to_impute] = imputer.transform(test[columns_to_impute])  # Use transform, NOT fit_transform



from sklearn.model_selection import train_test_split
X=train.drop(columns='Listening_Time_minutes')
y=train['Listening_Time_minutes']


scaler=StandardScaler()
X=scaler.fit_transform(X)
test=scaler.transform(test)
X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2)


import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l2  
import keras.backend as K


model = Sequential([
    Input(shape=(11,)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.1)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.5)),
    Dropout(0.1),
    Dense(1, activation='linear')
])



early_stopping = EarlyStopping(
    monitor='val_loss',  # Stop when validation loss stops improving
    patience=5,          # Number of epochs with no improvement before stopping
    restore_best_weights=True,  # Restore model to best weights
    verbose=1
)



model.compile(optimizer='adam',loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])


model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping],  
    verbose=1
)


y_pred = model.predict(test)



y_pred = y_pred.flatten()
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission["Listening_Time_minutes"] = y_pred
submission.to_csv('submission.csv', index=False)




