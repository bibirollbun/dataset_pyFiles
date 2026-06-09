# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


df.head()


df.info()


df.describe()


print(df.isnull().sum())


sns.countplot(x=df["rainfall"])
plt.show()


df = df.drop(columns=['id','day'])


corr_matrix= df.corr()
sns.heatmap(corr_matrix, annot=True)


df = df.drop(columns=['pressure','windspeed','winddirection','mintemp','maxtemp','dewpoint'])


X = df.drop(columns=['rainfall'])
y = df['rainfall']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42, stratify=y)


from sklearn.utils import class_weight
class_weights=class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = dict(enumerate(class_weights))


print(class_weights_dict)
print(type(list(class_weights_dict.keys())[0]))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42, stratify=y)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


y_train.head()


from tensorflow.keras import Input
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu', kernel_initializer= 'he_normal'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode= 'auto', restore_best_weights= True)


y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


history = model.fit(
    X_train, y_train,
    class_weight=class_weights_dict,
    validation_data=(X_test, y_test),
    epochs= 100,
    batch_size= 64,
    callbacks=[early_stopping],
    verbose = 1
)


test_loss,test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
# Plot Training History
plt.figure(figsize=(12, 4))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Over Epochs")

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Over Epochs")

plt.show()



test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_ids = test_df["id"].copy()
test_df = test_df.drop(columns=['id','day','pressure','windspeed','winddirection','mintemp','maxtemp','dewpoint'])

X_test = scaler.transform(test_df)

y_pred_prob = model.predict(test_df)
y_pred = (y_pred_prob > 0.8).astype(int)

submission = pd.DataFrame({"id": test_ids, "rainfall": y_pred.flatten()})
submission.to_csv("rainfall_predictions.csv", index=False)


submission


counts = submission["rainfall"].value_counts()
print(counts)







