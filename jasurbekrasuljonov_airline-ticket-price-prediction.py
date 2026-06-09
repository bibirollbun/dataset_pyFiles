# Kerakli kutubxonlarni yuklab olamiz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Pandas yordamida csv faylidagi ma'lumotlarni yuklab olamiz
df = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv', index_col="id")
df.head()


# DataFrame haqida ma'lumotlarni ko'rish
df.info()


#DataFrame ni number ustunlari haqida ma'lumotlarni ko'rish
df.describe()


# Ma'lumotlarni distributsiyani ko'rish
%matplotlib inline
df.hist(bins=50, figsize=(10, 7))
plt.show()


#Transformer yasash object columnlar uchun OneHotEncoder preprocessingni,number ustunlari uchun MinMaxScalerlar preprocessingni qo'llaymiz
cat_attr = ['airline','source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
num_attr = ['duration', 'days_left']
column_transformer = ColumnTransformer([
    ('standard_scaler', StandardScaler(), num_attr),
    ('one_hot_encoder', OneHotEncoder(), cat_attr)
])


#DataFramedan labelni ajratib olamiz
X = df.drop('price', axis=1)
y = df['price'].copy()

#DataFrameni X_train, X_test, y_train, y_test larga ajratib olamiz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Train ma'lumotlarni o'qitish uchun tayyorlash
X_train_prepared = column_transformer.fit_transform(X_train).toarray()


#LinearRegression modelini o'qitish
from sklearn.linear_model import LinearRegression

LR_model = LinearRegression()
LR_model.fit(X_train_prepared, y_train)


#Test datani endi bashorat qilamiz RMSE va MAE hisoblaymiz
X_test_prepared = column_transformer.transform(X_test).toarray()
X_test_predict = LR_model.predict(X_test_prepared)

mse = mean_squared_error(y_test, X_test_predict)
mae = mean_absolute_error(y_test, X_test_predict)

print({"RMSE", np.sqrt(mse)})
print("MAE:", mae)


#RandomForestRegressor modelni o'qitish
from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor()
RF_model.fit(X_train_prepared, y_train)


X_test_prepared = column_transformer.transform(X_test).toarray()
X_test_predict = RF_model.predict(X_test_prepared)

mse = mean_squared_error(y_test, X_test_predict)
mae = mean_absolute_error(y_test, X_test_predict)

print("RMSE", np.sqrt(mse))
print("MAE:", mae)


test_df = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv', index_col="id")
test_df_prepared = column_transformer.transform(test_df).toarray()
test_predict = RF_model.predict(test_df_prepared)

predict_df = pd.DataFrame({'price': test_predict}, index=test_df.index)
predict_df.to_csv("/kaggle/working/solution.csv")

