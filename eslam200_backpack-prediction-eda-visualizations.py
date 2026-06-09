import pandas as pd
import numpy as np

# Visulization 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


train_path = "/kaggle/input/playground-series-s5e2/train.csv"
test_path = "/kaggle/input/playground-series-s5e2/test.csv"
sample_sub_path = "/kaggle/input/playground-series-s5e2/sample_submission.csv"


df = pd.read_csv(train_path, index_col = "id")
df.head()


df.info()


(df.isna().sum() / len(df) * 100).sort_values(ascending=False)


df.describe()


df["Size"].value_counts()


df["Material"].value_counts()


plt.boxplot(
    data=df,
    x="Price",
    vert=False,
)

plt.xlabel("Price")
plt.title("Distribution of Price")

plt.show()


plt.scatter(
    data=df.sample(2000),
    x="Price",
    y="Weight Capacity (kg)"
)
plt.xlabel("Price")
plt.ylabel("Weight Capacity (kg)")
plt.title("Correlation Between Price and Backpack Weight Capacity")

plt.show()


df["Price"].corr(df["Weight Capacity (kg)"])


plt.scatter(
    data=df.sample(2000),
    x="Price",
    y="Compartments"
)

plt.xlabel("Price")
plt.ylabel("Compartments")
plt.title("Correlation Between Price and Backpack Num of Compartments")

plt.show()


df["Price"].corr(df["Compartments"])


df.groupby("Brand")["Price"].mean().sort_values().plot(kind="bar")

plt.xlabel("Brand")
plt.ylabel("Mean Price")
plt.title("Brand average price")

plt.show()


sns.countplot(y=df["Size"])

plt.xlabel("Count")
plt.ylabel("Size")
plt.title("Size Count")

plt.show()


df.groupby("Size")["Price"].mean().sort_values().plot(kind="bar")

plt.xlabel("Size")
plt.ylabel("Mean Price")
plt.title("Brand average price")

plt.show()


sns.countplot(y=df["Material"])

plt.xlabel("Count")
plt.ylabel("Material")
plt.title("Material Count")

plt.show()


df.groupby("Material")["Price"].mean().sort_values().plot(kind="bar")

plt.xlabel("Size")
plt.ylabel("Mean Price")
plt.title("Brand average price")

plt.show()


sns.barplot(x='Price', y='Material', hue='Size', data=df);
plt.title("Data Distribution - Material Size Price")
plt.show()

