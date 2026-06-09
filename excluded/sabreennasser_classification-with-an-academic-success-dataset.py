import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


df=pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")


df.head()


df.info()


df.isna().sum()


df.duplicated().sum()


df = df.drop(columns=["id"])


from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(df.drop(columns=["Target"]))
outlier_percentage = (outlier_labels == -1).sum() / len(df) * 100
print(f"total outlier: {outlier_percentage:.2f}%")


X = df.drop(columns=["Target"])
y = df["Target"]



X


y


y.value_counts()


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.values.reshape(-1,1))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)




from sklearn.preprocessing import StandardScaler
numerical_columns = X_train.select_dtypes(include=['number']).columns

scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])


print(pd.get_dummies(X_train).shape)


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Dense(128, activation='relu',input_dim=X_train .shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



history = model.fit(X_train, y_train, epochs=100, validation_split=0.2,batch_size=8)



tr_loss = history.history['loss']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]

Epochs = [i+1 for i in range(len(tr_loss))]
loss_label = f'best epoch= {str(index_loss + 1)}'

# Plot training history
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout
plt.show()


y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score
y_pred_classes = np.round(y_pred) 
acc = accuracy_score(y_test, y_pred_classes)
print("Accuracy:",acc)

