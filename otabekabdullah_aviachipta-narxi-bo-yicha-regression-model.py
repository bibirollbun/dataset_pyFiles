import numpy as np # linear algebra
import pandas as pd # data preprocessing
import matplotlib.pyplot as plt # data visualization by plot
%matplotlib inline
import seaborn as sns # data visualization
import sklearn # machine learning algorithms

# kerakli kutubxonalarni o'rnatamiz

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv")
df_test = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv")

# kerakli dataframes'ni yuklab olamiz


df_train.head()


df_test.head()


print(f"Train Dataframe'ning qator soni-{df_train.shape[0]} ta va ustunlari soni-{df_train.shape[1]} ta")
print(f"Train Dataframe'ning qator soni-{df_test.shape[0]} ta va ustunlari soni-{df_test.shape[1]} ta")

# dataframe ichidagi qator va ustunlarini aniqlash


df_train.info()


print(df_train.isnull().sum())
print(df_test.isnull().sum())

# null qiymatlar soni


airlane = df_train["airline"].unique()
flight = df_train["flight"].unique()
source_city = df_train["source_city"].unique()
departure_time = df_train["departure_time"].unique()
stops = df_train["stops"].unique()
arrival_time = df_train["arrival_time"].unique()
destination_city = df_train["destination_city"].unique()
class_type = df_train["class"].unique()

# takrorlanmagan elementlarni aniqlash
# buning sababi, biz qo'lda Machine leraning algoritmisiz elementlarni Category turidan Integer turiga o'tkazamiz


print(f"airlane ustuni ichida - {airlane} qiymatlar mavjud")
print(f"flight ustuni ichida - {flight} qiymatlar mavjud")
print(f"source_city ustuni ichida - {source_city} qiymatlar mavjud")
print(f"departure_time ustuni ichida - {departure_time} qiymatlar mavjud")
print(f"stops ustuni ichida - {stops} qiymatlar mavjud")
print(f"arrival_time ustuni ichida - {arrival_time} qiymatlar mavjud")
print(f"destination_city ustuni ichida - {destination_city} qiymatlar mavjud")
print(f"class_type ustuni ichida - {class_type} qiymatlar mavjud")


# yuqoridagi natijadan ko'rinib turibdiki: "stop" va "class_type" ustuniga o'zgartirish kiritish mumkin

df_train["stops"] = df_train["stops"].replace({
    "zero" : 1,
    "one" : 1,
    "two_or_more" : 2
    })

df_train["class"] = df_train["class"].replace({
    "Economy" : 0,
    "Business" : 1,
    })

df_train.sample(5)


airlane = df_test["airline"].unique()
flight = df_test["flight"].unique()
source_city = df_test["source_city"].unique()
departure_time = df_test["departure_time"].unique()
stops = df_test["stops"].unique()
arrival_time = df_test["arrival_time"].unique()
destination_city = df_test["destination_city"].unique()
class_type = df_test["class"].unique()

# takrorlanmagan elementlarni aniqlash
# buning sababi, biz qo'lda Machine leraning algoritmisiz elementlarni Category turidan Integer turiga o'tkazamiz


# yuqoridagi natijadan ko'rinib turibdiki: "stop" va "class_type" ustuniga o'zgartirish kiritish mumkin

df_test["stops"] = df_test["stops"].replace({
    "zero" : 1,
    "one" : 1,
    "two_or_more" : 2
    })

df_test["class"] = df_test["class"].replace({
    "Economy" : 0,
    "Business" : 1,
    })

df_test.sample(5)


df_train.info()


from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in ['airline', 'source_city', 'destination_city', 'class', "departure_time"]:
    label_encoded = LabelEncoder()
    
    df_train[column] = label_encoded.fit_transform(df_train[column])
    df_test[column] = label_encoded.transform(df_test[column])
    label_encoders[column] = label_encoded

# Categorical Columns Labelencoder orqali float ko'rinishiga olindi


df_train = df_train.drop(columns = ['stops' , 'arrival_time'])


df_train = df_train.drop(columns = ["flight"])


df_train.info()


df_train.describe()


correlation = df_train.corr().sort_values(by = "price", ascending = False)
correlation

# Price uchun muhim bo'lgan bog'liq ustunlar


correlation["price"]

# Price uchun muhim bo'lgan bog'liq ustunlar


df_train.plot(
    kind = "scatter",
    x = "price",
    y = "class",
    title = "Class bo'yicha Narxining bog'liqligi",
    figsize = (12, 4)
)
plt.show()

# demak Business Class uchun ko'proq mijozlar mavjud ekan


X = df_train.drop(columns = ['id' ,"price"])
y = df_train['price']

# X-paramet va y-label ajratish


print(f"Parametr yoki Input qismi: {X}")
print(f"Label qismi: {y}")

# dataset X-parametr va y-label ajratildi


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12)

# train_set va test_set ajratildi


print(
    f"Umumiy Parametr hajmi: {X.shape}\n"
    )
print(
    f"Umumiy  Label hajmi: {y.shape}\n"
    )
print(
    f"X-parametrlar uchun\n"
        f"Train set o'lchovi: {X_train.shape} va\n"
        f"Test set o'lchovi: {X_test.shape}\n"
    )
print(
    f"y-Label uchun\n"
        f"Train set o'lchovi: {y_train.shape} va\n"
        f"Test set o'lchovi: {y_test.shape}\n"
    )


from sklearn.ensemble import RandomForestRegressor

random_forest_regression_model = RandomForestRegressor( n_estimators = 100, random_state = 42)
random_forest_regression_model.fit(X_train, y_train)

# RandomForestRegressor Classification Supervised Machine Learning Model


y_prediction = random_forest_regression_model.predict(X_test)

# X_test nisbatan bashorat qilish


print(y_test.shape, y_prediction.shape)


from sklearn.metrics import mean_squared_error, mean_absolute_error

test_mse = mean_squared_error(y_test, y_prediction)
test_mae = mean_absolute_error(y_test, y_prediction)


print(f"Mean_squared_error of the training data: {np.sqrt(test_mse)} xatolikda bashorat qilyapti")
print(f"Mean_absolute_error of the training data: {test_mae} xatolikda bashorat qilyapti")

# modelimizni tekshirish


from sklearn.metrics import r2_score

# R² hisoblash
r2 = r2_score(y_test, y_prediction)
print(f"R^2 (foiz): {r2 * 100}% aniqlikda ishlamoqda")


ids = df_test['id'].values
X_test = df_test.drop("id", axis=1).values


submission = pd.DataFrame({
    'id': ids,
    'price' : y_prediction.reshape(-1)
})


submission.sample(10)


submission.to_csv("Flight Price - Regression Model.csv", index = False)

