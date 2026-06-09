import numpy as np
import pandas as pd 
import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt





df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df


df = pd.get_dummies(df)
df = df.map(lambda x: 1 if x is True else (0 if x is False else x))
df.drop(columns=["id"], inplace=True)
df


X = df.drop(columns = ['diagnosed_diabetes'])
y = df['diagnosed_diabetes']
X


y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.6, random_state=4212356789)


print(y_train)


# model = Sequential()
# model.add(Dense(64, input_dim=X_train.shape[1], activation='relu')) 
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# model.compile(optimizer='adam',  
#               loss='binary_crossentropy',
#               metrics=['precision']) 

# early_stop = tf.keras.callbacks.EarlyStopping(
#     monitor='loss',
#     patience=5,    
#     restore_best_weights=True 
# )

# model.fit(X_train, y_train, epochs=80, batch_size=400, callbacks=[early_stop])


# model.compile(optimizer='adam',  
#               loss='binary_crossentropy',
#               metrics=['recall']) 

# early_stop = tf.keras.callbacks.EarlyStopping(
#     monitor='loss',
#     patience=5,    
#     restore_best_weights=True 
# )

# model.fit(X_train, y_train, epochs=80, batch_size=400, callbacks=[early_stop])


# def objective(trial):
#     n_layers = trial.suggest_int("n_layers", 3, 7)
#     learning_rate = trial.suggest_float("learning_rate", 0.01, 0.05, log=True)

#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Input(shape=(42, )))

#     for i in range(n_layers):
#         num_hidden = trial.suggest_int(f"n_units_l{i}", 96, 256, step=10)
#         model.add(tf.keras.layers.Dense(num_hidden, activation="relu"))

#         dropout_rate = trial.suggest_float(f"dropout_l{i}", 0.1, 0.5)
#         model.add(tf.keras.layers.Dropout(dropout_rate))

#     model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#         loss="binary_crossentropy",
#         metrics=['accuracy']
#     )

#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
#         TFKerasPruningCallback(trial, "accuracy")
#     ]

#     result = model.fit(
#         X_train, y_train,
#         epochs=80,
#         batch_size=400,
#         callbacks=callbacks,
#         verbose=1
#     )

#     return max(result.history["accuracy"])

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# print(f"Best parameters: {study.best_params}")


params = {
    'n_layers': 3,
    'learning_rate': 0.01255,
    'n_units_l0': 256,
    'dropout_l0': 0.2817,
    'n_units_l1': 226,
    'dropout_l1': 0.2356,
    'n_units_l2': 236,
    'dropout_l2': 0.2332,
    # 'n_units_l3': 224,
    # 'dropout_l3': 0.1982
}

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(42,)))


model.add(tf.keras.layers.Dense(params['n_units_l0'], activation="relu"))
model.add(tf.keras.layers.Dropout(params['dropout_l0']))

model.add(tf.keras.layers.Dense(params['n_units_l1'], activation="relu"))
model.add(tf.keras.layers.Dropout(params['dropout_l1']))

model.add(tf.keras.layers.Dense(params['n_units_l2'], activation="relu"))
model.add(tf.keras.layers.Dropout(params['dropout_l2']))

# model.add(tf.keras.layers.Dense(params['n_units_l3'], activation="relu"))
# model.add(tf.keras.layers.Dropout(params['dropout_l3']))

model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
model.compile(optimizer=optimizer,
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=80,
    batch_size=400,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ],
    verbose=1
)


y_pred = model.predict(X_test)


y_pred_classes = (y_pred > 0.5).astype("int32") #rounds them to binary values
y_pred_classes


print(confusion_matrix(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))
recall = recall_score(y_test, y_pred_classes)
print(f"\nFinal Recall Score: {recall:.2f}")


ss = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


ss


test


test = pd.get_dummies(test)
test = test.map(lambda x: 1 if x is True else (0 if x is False else x))
test.drop(columns=["id"], inplace=True)
test


submission_raw = model.predict(test)
submission = submission_raw#(submission_raw > 0.5).astype("int32")
counter = 0


submission


output = pd.DataFrame({
    'id': range(700000, 1000000),
    'diagnosed_diabetes': submission.ravel()
})
output


output.to_csv('submission.csv', index=False)

