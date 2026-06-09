import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")


df.head()


df.drop(columns=['id'], inplace=True)


df.info()


encoder= LabelEncoder()
df['Target']= encoder.fit_transform(df['Target'])


df


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from tensorflow.keras.optimizers import Adam


model = Sequential([
    
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    
    Dense(3, activation='softmax')
])
model.compile(optimizer="adam",loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50)


model.evaluate(X_train, y_train)


model.evaluate(X_test, y_test)

