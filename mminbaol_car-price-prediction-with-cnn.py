import pandas as pd
df_train =pd.read_csv('/kaggle/input/autoam-car-price-prediction/train.csv')
df_test =pd.read_csv('/kaggle/input/autoam-car-price-prediction/test.csv')


import numpy as np


df = pd.concat([df_train,df_test])


df.head()


df.shape


df.info()


df.describe()


df.isnull().sum()


df.drop("Id",axis=1,inplace=True)


df.corr(numeric_only=True)


#Formül: mil = kilometre × (0,621371)


# km -> miles
df.loc[df["running"].str.contains("km"),"running"] = df.loc[df["running"].str.contains("km"),"running"].str.replace("km","").str.strip().astype("int") * 0.621371


df.loc[df["running"].str.contains("miles",case=False, na=False),"running"] = df.loc[df["running"].str.contains("miles",case=False, na=False),"running"].str.replace("miles","").str.strip()


df.sample(5)


df["running"] = df["running"].astype(float)


df.head()


#age of cars
df['age'] = df['year'].apply(lambda x: 2025 - x)
df.drop("year",axis=1,inplace=True)





df['status'].value_counts()


df['status'] = df['status'].map({'excellent': 4, 'good': 3, 'normal': 2, 'crashed': 1, 'new': 5})


df['motor_type'].value_counts()


import seaborn as sns


ax=sns.countplot(x=df['motor_type'])
ax.bar_label(ax.containers[0]);


ax=sns.countplot(x=df['motor_type'],hue=df['type'])
ax.bar_label(ax.containers[0]);


sns.heatmap(df.corr(numeric_only=True),annot=True)


abs(df.corr(numeric_only=True)['price'].sort_values(ascending=False))


df = pd.get_dummies(df, drop_first=True)


from sklearn.preprocessing import MinMaxScaler

exclude = ["price"]

scale_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(exclude)

scaler = MinMaxScaler()

df[scale_cols] = scaler.fit_transform(df[scale_cols])


df.sample(5)


df.info()


df_train = df[df["price"].notna()]
df_test = df[df["price"].isna()]


df_test.drop("price",axis=1,inplace=True)


# Tüm x_train ve x_test sütunlarını float32 yap
df_train = df_train.astype(np.float32)
df_test = df_test.astype(np.float32)



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


x = df_train.drop('price', axis=1)  # Bağımsız değişkenler (özellikler)
y = df_train['price']  # Bağımlı değişken (hedef değişken)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Modeli oluştur
model = Sequential()


input_shape = x_train.shape[1]

# Girdi katmanı + ilk gizli katman
model.add(Dense(128, input_dim=input_shape, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# İkinci gizli katman
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Üçüncü gizli katman
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Dördüncü gizli katman
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Beşinci gizli katman
model.add(Dense(64, activation='relu'))

# Çıkış katmanı (fiyat tahmini için tek nöron, lineer aktivasyon)
model.add(Dense(1, activation='linear'))

# Modeli derleme
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


model.summary()


history=model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=200,epochs=500)


import matplotlib.pyplot as plt

plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])


tahmin=model.predict(x_test)


from sklearn.metrics import mean_squared_error, r2_score


r2_score(y_test,tahmin)


mean_squared_error(y_test,tahmin)**.5


import pickle

# Modeli kaydetme
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)


import joblib
loaded_model = joblib.load('/kaggle/working/model.pkl')


predictions = loaded_model.predict(df_test)


predictions = predictions.flatten()  # Diziyi 1 boyutlu hale getirir


id = pd.read_csv('/kaggle/input/autoam-car-price-prediction/test.csv')["Id"]


submission_df = pd.DataFrame({'Id': id, 'price': predictions})
submission_df.to_csv('submission.csv', index=False)


submission_df




