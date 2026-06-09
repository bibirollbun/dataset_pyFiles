import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/kaggle/input/penguin-clustering-analysis/penguins.csv")
df.head()


features = ["culmen_length_mm", "culmen_depth_mm", 
            "flipper_length_mm", "body_mass_g"]

data = df[features]
data.head()


data = data.dropna()


scaler = StandardScaler()
scaled = scaler.fit_transform(data)


SSE = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, init="random", random_state=42)
    km.fit(scaled)
    SSE.append(km.inertia_)

plt.plot(range(1,10), SSE, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()


# pilih fitur
features = ["culmen_length_mm", "culmen_depth_mm", 
            "flipper_length_mm", "body_mass_g"]

# drop NA dari df
df = df.dropna(subset=features)

# ambil data numeric
data = df[features]

# scaling
scaler = StandardScaler()
scaled = scaler.fit_transform(data)

# model
kmeans = KMeans(n_clusters=3, init='random', random_state=42, n_init=10)
kmeans.fit(scaled)

labels = kmeans.predict(scaled)

df["cluster"] = labels
df.head()


# Buat kolom id dari index dataframe
df["id"] = df.index

# Buat dataframe submission
submission = df[["id", "cluster"]]

# Simpan ke CSV
submission.to_csv("submission.csv", index=False)

submission.head()

