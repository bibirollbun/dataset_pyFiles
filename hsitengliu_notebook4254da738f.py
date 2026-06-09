import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_log_error, mean_absolute_error, r2_score, 
    accuracy_score, f1_score, recall_score, precision_score
)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping






train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv").drop("id", axis=1)
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train.head()





# Define which features are numerical for subsequent engineering steps.
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']



# === 4. Label Encoding for Categorical Features ===
# Encode 'Sex' feature as integer labels.
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex']  = le.transform(test['Sex'])


train.head()


test.head()


# === 6. Define Features and Target Variable ===
X = train.drop(columns=["Calories"])
y = train["Calories"]
X_test = test.drop("id", axis=1)


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


X_scaled.shape


X_scaled[0].shape


def root_mean_squared_log_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(1+y_pred) - K.log(1+y_true))))


model = keras.Sequential([
    layers.Input(shape=(X_scaled.shape[1],)),
    layers.Dense(6, activation='tanh'),
    layers.Dense(4, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
              loss="mean_squared_logarithmic_error")

history = model.fit(
    X_scaled, y, validation_split=0.2,
    epochs=250, batch_size=1024, verbose=1, 
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)


history.history.keys()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


results = model.evaluate(X_scaled, y, batch_size=1024)


results


test_preds = model.predict(X_test_scaled)


test_preds


y.max()


y.min()


test_preds.max()


test_preds.min()


calories = test_preds.reshape(-1)


submission = pd.DataFrame({ "id": test["id"].to_numpy(), "Calories": calories})


submission.head()


model.save("calories_prediction.keras")


submission.to_csv("submission.csv", index=False)

