import pandas as pd # data cleaning
import numpy as np # linear algebra
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization
%matplotlib inline

import sklearn # sci-kit learn for Machine Learning Models

# kerakli kutubxonalar


df_train = pd.read_csv("/kaggle/input/aviakompaniya/train_dataset.csv")
df_test= pd.read_csv("/kaggle/input/aviakompaniya/test_dataset.csv")

# kerakli dataframe'ni yuklash


df_train["satisfaction"].value_counts()

# eng muhim Column qiymatlarini: 0 - unsatisfaction; 1 - satisfaction


df_train.shape

# train_dataset o'lchami: 10000 - Rows; 23 - Columns


df_test.shape

# test_dataset o'lchami: 10000 - Rows; 23 - Columns


df_train.isnull().sum()

# faqat "Arrival Delay in Minutes" ustunida 28 null qiymat mavjud


df_test.isnull().sum()

# faqat "Arrival Delay in Minutes" ustunida 19 null qiymat mavjud


df_train["Arrival Delay in Minutes"] = df_train["Arrival Delay in Minutes"].fillna(value = df_train["Arrival Delay in Minutes"].mean())

# mean metodi orqali null qiymatlar to'ldirildi


df_test["Arrival Delay in Minutes"] = df_test["Arrival Delay in Minutes"].fillna(value = df_test["Arrival Delay in Minutes"].mean())

# mean metodi orqali null qiymatlar to'ldirildi


df_train.isnull().sum()

# yana null qiymatlar tekshirib olindi


df_test.isnull().sum()

# yana null qiymatlar tekshirib olindi


df_train.columns

# ustunlar nomlari aniqlashtirildi


df_train.info()

# data type aniqlash va Categorical types ustunlar aniqlandi


df_test.info()

# data type aniqlash va Categorical types ustunlar aniqlandi


df_train["Gender"].value_counts()


df_train["Gender"] = df_train["Gender"].replace({
    "Male" : 1,
    "Female" : 0
})


df_train["Customer Type"].value_counts()


df_train["Customer Type"] = df_train["Customer Type"].replace({
    "Loyal Customer" : 1,
    "disloyal Customer" : 0
})


df_train["Type of Travel"].value_counts()


df_train["Type of Travel"] = df_train["Type of Travel"].replace({
    "Business travel" : 1,
    "Personal Travel" : 0
})


df_train["Class"].value_counts()


df_train["Class"] = df_train["Class"].replace({
    "Business" : 0,
    "Eco" : 1,
    "Eco Plus" : 2
})


from sklearn.preprocessing import OrdinalEncoder

ordin_encoded = OrdinalEncoder()

df_test[['Gender','Customer Type','Type of Travel','Class']]  = ordin_encoded.fit_transform(df_test[['Gender','Customer Type','Type of Travel','Class']])
df_test.head()


df_test.info()

# yuqoridagi o'zgartirishlar natijasida yana Data types tekshirildi


df_train.describe()

# dataset'ning umumiy statistikasi aniqlandi


df_test.describe()


corr_train = df_train.corrwith(df_train['satisfaction'], numeric_only=True).abs().sort_values(ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(x=corr_train.to_frame()[0], y=corr_train.to_frame().index, palette='Blues_r')
plt.title('Correlation of Features with Satisfaction')
plt.show()

# Train dataset uchun "Satisfaction" ustuni bilan yuqori Correlation bo'lgan ustunlar aniqlandi


not_corr_train_columns = ["On-board service", "Leg room service", "Cleanliness", "Flight Distance", "Inflight wifi service",
                    "Baggage handling", "Inflight service", "Checkin service", "Food and drink", "Customer Type",
                    "Ease of Online booking", "Age", "Departure/Arrival time convenient", "Arrival Delay in Minutes",
                    "Departure Delay in Minutes", "Gate location", "Gender"]

df_train = df_train.drop(not_corr_train_columns, axis = 1)

# Train_dataset uchun Correlation juda past bo'lganlar .drop qilindi


not_corr_test_columns = ["On-board service", "Leg room service", "Cleanliness", "Flight Distance", "Inflight wifi service",
                    "Baggage handling", "Inflight service", "Checkin service", "Food and drink", "Customer Type",
                    "Ease of Online booking", "Age", "Departure/Arrival time convenient", "Arrival Delay in Minutes",
                    "Departure Delay in Minutes", "Gate location", "Gender"]

df_test = df_test.drop(not_corr_test_columns, axis = 1)

# Train_dataset uchun Correlation juda past bo'lganlar .drop qilindi


print(f"Keraksiz ustunlar soni: {len(not_corr_train_columns)} ta")
print(f"Kerakli ustunlar soni: {len(df_train.columns)} ta")
print(f"Jami ustunlar soni: {len(not_corr_train_columns) + len(df_train.columns)} ta")


correlation = df_train.select_dtypes(include = "number").corr().sort_values(by = "satisfaction")

corr_satisfaction = correlation["satisfaction"].sort_values(ascending = False)
corr_satisfaction

# faqat "satisfaction" ustuni uchun eng katta bog'liq bo'lgan ustunlar


plt.figure(figsize = (16, 8))
sns.countplot(data = df_train, x = "satisfaction", hue = "Online boarding", color = "blue")
plt.show()

# Aviaqatnovdan Qoniqish yoki qoniqmaslik, Online boarding - qoniqishni 5 shkalada baholangan


plt.figure(figsize = (16, 10))
sns.countplot(data = df_train, x = "satisfaction", hue = "Type of Travel", color = "red")
plt.show()

# Aviaqatnovdan Qoniqish yoki qoniqmaslik, Type of Travel orqali tahlil qilingan
# demak, business travelda Qoniqish ancha yuqori ekan, Personal'ga nisbatan


df_train.sample(5)

# dataset'ning eng kerakli qismi qoldi va bu Mashine Learning processing uchun tayyor


df_test.sample(5)

# dataset'ning eng kerakli qismi qoldi va bu Mashine Learning processing uchun tayyor


X = df_train.drop("satisfaction", axis = 1)
y = df_train["satisfaction"]

print(f"Parametr yoki Input qismi: {X}")
print(f"Label qismi: {y}")

# dataset X-parametr va y-label ajratildi


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 12)

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


from sklearn.linear_model import LogisticRegression

logistic_reg = LogisticRegression()

logistic_reg.fit(X_train, y_train)

# Logistik Regression Classification Model


X_test.shape


y_prediction = logistic_reg.predict(X_test)

# X_test nisbatan bashorat qilish


y_test.head()


print(y_test.shape, y_prediction.shape)


from sklearn.metrics import accuracy_score

test_data_accuracy = accuracy_score(y_test, y_prediction)

print(f"Accuracy score of the training data: {test_data_accuracy * 100}% aniqlikda bashorat qilyapti")

# modelimizni accuracy score orqali tekshirish


from sklearn import metrics

print(metrics.classification_report(y_test, y_prediction))
print("Model aniqligi:", metrics.accuracy_score(y_test, y_prediction))

conf_mat = metrics.confusion_matrix(y_test, y_prediction)
sns.heatmap(conf_mat, annot=True, fmt="g")
plt.show()


ids = df_test['id'].values
X_test = df_test.drop("id", axis=1).values


submission = pd.DataFrame({
    'id': ids,
    'satisfaction' : y_prediction.reshape(-1)
})


submission.sample(10)


submission.to_csv("submission_ML_flight_satisfaction.csv",index=False)

