
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/train.csv')
test=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/test.csv')


train


idcol=test.id
test=test.drop('id',axis=1)


y=train.target
X=train.drop('target',axis=1)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import pandas as pd

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X, y, test_size=0.1, random_state=42
)

from sklearn.preprocessing import MinMaxScaler
import pandas as pd



standard_scaler =  MinMaxScaler()

X_train_scaled = standard_scaler.fit_transform(X_train_split)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)

X_val_scaled = standard_scaler.transform(X_val_split)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)

X_test_scaled = standard_scaler.transform(test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=test.columns)


print(f"Training shape : {X_train_scaled.shape}")
print(f"Validation shape : {X_val_scaled.shape}")
print(f"Test shape : {X_test_scaled.shape}")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
def r2_keras(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - (SS_res / (SS_tot + tf.keras.backend.epsilon()))
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1) 
])


model.compile(optimizer='adam', loss='mse', metrics=['mae', r2_keras])

early_stopping = EarlyStopping(
    monitor='val_loss',    
    patience=5,          
    restore_best_weights=True,  
    verbose=1
)

history = model.fit(X_train_scaled, y_train_split, 
                    epochs=100, 
                    batch_size=32, 
                    validation_data=(X_val_scaled, y_val_split),
                    callbacks=[early_stopping],  
                    verbose=1)

y_pred_val_nn = model.predict(X_val_scaled).flatten()

r2_nn = r2_score(y_val_split, y_pred_val_nn)

print(f"Optimized Neural Network R² Score: {r2_nn:.4f}")



y_test_pred = model.predict(X_test_scaled)

if y_test_pred.ndim > 1:
    y_test_pred = y_test_pred.flatten()
submission = pd.DataFrame({
    'id': idcol,    
    'target': y_test_pred 
})


submission.to_csv('submission.csv', index=False)

print("Submission file saved as submission.csv.")

