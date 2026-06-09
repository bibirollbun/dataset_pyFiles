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


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


df_sub = df_test[["id"]]


df_train = df_train.drop(columns=["id", "day"])
df_test = df_test.drop(columns=["id", "day"])


plt.figure(figsize=(10, 6))
sns.heatmap(df_train.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Values Heatmap")
plt.show()


df_train = df_train.fillna(df_train.median())
df_test = df_test.fillna(df_test.median())


plt.figure(figsize=(12, 8))
sns.heatmap(df_train.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train.drop(columns=["rainfall"]))
plt.xticks(rotation=45)
plt.title("Boxplot of Weather Features")
plt.show()


important_features = ["pressure", "humidity", "maxtemp", "mintemp", "rainfall"]
sns.pairplot(df_train[important_features], hue="rainfall")
plt.show()


X = df_train.drop(columns=["rainfall"])
y = df_train["rainfall"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(df_test)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = keras.Sequential([
    keras.layers.Dense(512, activation="relu", input_shape=(X_train.shape[1],)),  
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")  # Binary classification (0 or 1)
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",  
    metrics=["accuracy"]
)


history = model.fit(
    X_train, y_train,  
    epochs=50,  
    batch_size=32,  
    validation_data=(X_val, y_val),  
    verbose=1  
)


plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()


val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}")


df_sub["rainfall"] = (model.predict(test_scaled) > 0.5).astype(int)


df_sub.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

