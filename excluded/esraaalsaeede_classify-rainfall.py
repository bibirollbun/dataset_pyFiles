import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.shape


train.head(5)


print("train.isnull")
print(train.isnull().sum())


print("tset.isnull:")
print(test.isnull().sum())


train.drop(columns=['id', 'day'], inplace=True)
test.drop(columns=['id', 'day'], inplace=True)


X = train.drop(columns=['rainfall']) 
y = train['rainfall']  

X.fillna(X.median(), inplace=True)
test.fillna(test.median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.3, random_state=42, shuffle=False)


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_AUC', patience=10, restore_best_weights=True)



model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),  
    layers.Dense(1024, activation='swish'),
    layers.Dropout(0.25), 
    
    layers.Dense(512, activation='swish'),
    layers.Dropout(0.25),  
    
    layers.Dense(128, activation='swish'),
    layers.Dropout(0.5),
    
    layers.Dense(64, activation='swish'),
    layers.Dropout(0.5),
    
    layers.Dense(16, activation='swish'),
    layers.Dropout(0.5),
    
    layers.Dense(1, activation='sigmoid') 
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['AUC'])

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=32,callbacks=[early_stopping],verbose=1)



y_pred_prob = model.predict(X_valid).flatten()

auc_score = roc_auc_score(y_valid, y_pred_prob)
print("Validation AUC Score:", auc_score)


test_predictions = model.predict(test_scaled).flatten()


submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

submission['rainfall'] = test_predictions

submission.to_csv("submission.csv", index=False)


submission

