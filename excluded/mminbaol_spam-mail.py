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


df_train = pd.read_csv("/kaggle/input/spam-emails12345/train.csv")
df_test = pd.read_csv("/kaggle/input/spam-emails12345/test.csv")
sample_submission = pd.read_csv("/kaggle/input/spam-emails12345/sample_submission.csv")


df_train.info()
df_test.info()
sample_submission.info()


df = pd.concat([df_train,df_test],axis=0)


df.info()


#EDA


df.sample(10)


df["Balance"].value_counts().sort_index()


import pandas as pd

def categorize_balance(x):
    if x == 0:
        return "No Balance"
    elif x <= 100000:
        return "Low Balance"
    elif x > 100000:
        return "High Balance"

df["balance_category"] = df["Balance"].apply(categorize_balance)


df.head()


df.drop(["CustomerId","Surname","Balance"],axis=1,inplace=True)


from sklearn.preprocessing import MinMaxScaler

scaled_cols = ["CreditScore","Age","Tenure","EstimatedSalary"]

scaler = MinMaxScaler()

df[scaled_cols] = scaler.fit_transform(df[scaled_cols])


df.head()


categoric_cols = ["Geography","Gender","balance_category"]

df = pd.get_dummies(df, columns = categoric_cols)


df.sample(4)


df_train = df[df["Exited"].notna()]
df_test = df[df["Exited"].isna()]


x = df_train.drop(["id","Exited"],axis=1)
y = df_train[["Exited"]]


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


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
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.summary()


from tensorflow.keras.callbacks import EarlyStopping

# EarlyStopping callback'i oluştur
early_stopper = EarlyStopping(
    monitor='val_loss',  # izlenecek metrik
    patience=20,          # iyileşme olmazsa kaç epoch sonra duracak
    restore_best_weights=True  # en iyi ağırlıkları geri yükle
)


history = model.fit(x_train,y_train,
                    epochs=200,
                    batch_size=32,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopper],
                    verbose=1)


id = df_test["id"]
df_test.drop(["id","Exited"],axis=1,inplace=True)


predictions = model.predict(df_test)


y_pred = (predictions > 0.5).astype(float)


submission = pd.DataFrame({"id":id,"Exited":y_pred.flatten()})


submission.head()


submission.to_csv("submission.csv",index=False)




