# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train_df.head()


x = train_df.drop(["id","Calories"],axis = 1)
y = train_df["Calories"]


x["Sex"] = x["Sex"].map({"male":0,"female":1})
test_df["Sex"] = test_df["Sex"].map({"male":0,"female":1})
x_test = test_df.drop(["id"],axis =1)


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_test_scaled = scaler.fit_transform(x_test)


x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size = 0.2,random_state = 42)


model = Sequential([
    Dense(64,activation = "relu", input_shape =(x_train.shape[1],)),
    Dropout(0.1),
    Dense(64,activation="relu"),
    Dropout(0.1),
    Dense(16,activation="relu"),
    Dense(1)
    
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mse'])




history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test), 
                    epochs=50, batch_size=32, verbose=1)

# ðŸ“‰ DeÄŸerlendirme
val_preds = model.predict(x_test)
mae = mean_absolute_error(y_test, val_preds)
print(f"Validation MAE: {mae:.2f}")

# ðŸ“¤ Tahminleri oluÅŸtur ve gÃ¶nderim dosyasÄ±na yaz



preds = model.predict(x_test_scaled)
submission['Calories'] = preds
submission.to_csv('submission.csv', index=False)

