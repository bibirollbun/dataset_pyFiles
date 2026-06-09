import pandas as pd
import numpy as numpy
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import optuna

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization


df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_subm = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


display(df_train.shape)
display(df_test.shape)


display(df_train.isnull().sum())
display(df_test.isnull().sum())


for col in df_train.columns[:-1]:
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_train, y=col, x='rainfall', palette="coolwarm")
    plt.subplot(1, 2, 2)
    sns.histplot(data=df_train, x=col, hue='rainfall', kde=True)
    plt.show()


plt.figure(figsize=(12, 10))
sns.heatmap(data=df_train.corr(), annot=True, linewidths=0.2);


df_test["winddirection"] = df_test["winddirection"].fillna((df_test["winddirection"].shift(1) + df_test["winddirection"].shift(-1)) / 2)


X = df_train.drop(["rainfall"], axis=1)
y = df_train["rainfall"]


X_train = df_train.drop(columns=['day','rainfall'])
y_train = df_train["rainfall"]
X_test = df_test.drop(columns="day")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)


model = Sequential([
    Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dense(1, activation='sigmoid'),
])


optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])


history=model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)


y_pred_keras = model.predict(X_test_scaled).flatten()

# Save Submission
df_subm['rainfall'] = y_pred_keras
df_subm.to_csv("nn_submission.csv", index=False)
df_subm.head()




