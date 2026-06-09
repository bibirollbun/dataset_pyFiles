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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_df.head()


test_df.head()


print("Missing values in train:\n", train_df.isnull().sum())
print("Missing values in test:\n", test_df.isnull().sum())


test_df["winddirection"].fillna(test_df["winddirection"].median(), inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns


def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


train_df = remove_outliers(train_df, "windspeed")


# Data Visualization
# Distribution of numerical features
plt.figure(figsize=(12, 8))
train_df.hist(figsize=(12, 8), bins=30)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()


# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# Rainfall class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="rainfall", data=train_df, palette="viridis")
plt.title("Rainfall Class Distribution")
plt.show()


# K-means clustering on rainfall
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


X = train_df.drop(columns=["rainfall"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o", linestyle="--")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()


# Choose an optimal K (let's say K=3 based on the elbow method)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
train_df["cluster"] = kmeans.fit_predict(X_scaled)


# Visualize Clusters (Pairplot)
sns.pairplot(train_df, hue="cluster", diag_kind="hist", palette="viridis")
plt.show()

# Analyze how clusters relate to rainfall
cluster_rainfall_counts = train_df.groupby(["cluster", "rainfall"]).size().unstack()
print(cluster_rainfall_counts)


!pip install h2o


# Using AutoML
import h2o
from h2o.automl import H2OAutoML
from sklearn.preprocessing import StandardScaler


features = [col for col in train_df.columns if col not in ["rainfall", "cluster"]]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(train_df[features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
train_df["cluster"] = kmeans.fit_predict(X_train_scaled)

test_df = test_df[[col for col in features if col in test_df.columns]]  

X_test_scaled = scaler.transform(test_df)

test_df["cluster"] = kmeans.predict(X_test_scaled)

print(train_df.head())
print(test_df.head())


h2o.init()


train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

target = "rainfall"
features = [col for col in train_df.columns if col != target]


train_h2o[target] = train_h2o[target].log1p()
aml = H2OAutoML(max_models=100, max_runtime_secs=3600, seed=42)
aml.train(x=features, y=target, training_frame=train_h2o)

lb = aml.leaderboard
print(lb.head())

preds = aml.leader.predict(test_h2o)
submission = test_df[["id"]].copy()
submission["rainfall"] = preds.as_data_frame().values

submission.to_csv("h2o_predictions.csv", index=False)
print("Predictions saved to h2o_predictions.csv")




