import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
start_time = time.time()


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


df.info()


df.describe()


df.head()


df.isnull().sum()


df_value = df.nunique()/df.count()
df_value = df_value.rename_axis("feature").reset_index(name="count")

plt.figure(figsize=(10,8))
sns.barplot(df_value, x="feature", y="count")
plt.xticks(rotation = 45)
plt.xlabel("Fertilizer's Features")
plt.ylabel("Percentage of Unique Values")
plt.show()


df.drop(columns="id", inplace=True)


df_value = df.nunique()/df.count()
df_value = df_value.rename_axis("feature").reset_index(name="count")

plt.figure(figsize=(10,8))
sns.barplot(df_value, x="feature", y="count")
plt.xticks(rotation = 45)
plt.xlabel("Fertilizer's Features")
plt.ylabel("Percentage of Unique Values")
plt.show()


df_num = df.select_dtypes(include=["number"]).columns

mms = MinMaxScaler()
df[df_num] = mms.fit_transform(df[df_num])


df_cat = df.select_dtypes(include=["object"]).columns


for i in df_cat:
    print(df[i].value_counts())


for i in df_num:
    plt.figure(figsize=(10,4))
    sns.histplot(data=df[i], bins=15)
    plt.show()

for i in df_num:
    plt.figure(figsize=(5,5))
    sns.boxplot(data=df[i])
    plt.ylabel(f"{df[i].name}")
    plt.show()


df_cat_col = ["Soil Type", "Crop Type"]


oe = OrdinalEncoder()


df_cat_trans = Pipeline(
    steps=[("encoder", OrdinalEncoder())])

df_preprocess = ColumnTransformer(
    transformers=[("cat", df_cat_trans, df_cat_col)])

pipe = Pipeline(
    steps=[("preprocessor", df_preprocess), ("classifier", RandomForestClassifier())])


X = df.drop(columns="Fertilizer Name")
Y = df["Fertilizer Name"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)


pipe.fit(X_train, Y_train)


predict = pipe.predict(X_test)
accuracy = accuracy_score(Y_test, predict)
print(f"Accuracy Score: {(accuracy*100):.2f}%")


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time Required: {elapsed_time:.2f} seconds")


df_test.drop(columns="id")

test_prediction = pipe.predict(df_test)

submission = pd.DataFrame({"id": df_test["id"], "Fertilizer Name": test_prediction})
submission.to_csv("submission.csv", index=False)

