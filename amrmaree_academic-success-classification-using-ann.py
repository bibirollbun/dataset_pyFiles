import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")


df.head()


df.info()


df.drop(columns=['id'], inplace= True)


x = df.drop(columns=['Target'])
y = df['Target']


x


y


encoder = LabelEncoder()
y = encoder.fit_transform(y)


y


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential([
    Dense(128, activation='relu', input_dim = X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')


test_df = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')
test_ids = test_df['id']
test_df.drop(columns=['id'], inplace=True)


X_test_final = scaler.transform(test_df)


predictions = model.predict(X_test_final)
pred_labels = np.argmax(predictions, axis=1)


submission = pd.DataFrame({'id': test_ids, 'Target': encoder.inverse_transform(pred_labels)})
submission.to_csv('submission.csv', index=False)

