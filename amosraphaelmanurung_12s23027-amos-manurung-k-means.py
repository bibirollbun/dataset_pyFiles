import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

plt.pyplot = plt


df = pd.read_csv("/kaggle/input/penguin-clustering-analysis/penguins.csv")
df.head()


print("Info dataset:")
print(df.info())
print("\nStatistik deskriptif:")
display(df.describe(include="all"))

print("\nJumlah missing value tiap kolom:")
print(df.isna().sum())


# Cek korelasi antar fitur numerik untuk membantu pemilihan fitur
numeric_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]

corr = df[numeric_cols].corr()
corr


data = df.copy()

# Encode kolom sex: FEMALE -> 0, MALE -> 1
data["sex"] = data["sex"].map({"FEMALE": 0, "MALE": 1})

# Isi missing value
# - Numerik: median
# - Sex: modus (nilai yang paling sering muncul)
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

data["sex"] = data["sex"].fillna(data["sex"].mode()[0])

# Pastikan tidak ada missing lagi
print(data.isna().sum())


# Fitur akhir
features = numeric_cols + ["sex"]

X = data[features].values

# Standarisasi supaya semua fitur punya skala mirip (penting untuk K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled[:5]


inertias = []          # SSE (sum of squared error / within-cluster sum of squares)
sil_scores = []        # silhouette score
K_range = range(2, 10) # kita coba K dari 2 sampai 9

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Plot Elbow (SSE vs K)
plt.figure()
plt.plot(list(K_range), inertias, marker="o")
plt.title("Elbow Method (SSE vs K)")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Inertia / SSE")
plt.grid(True)
plt.show()

# Plot Silhouette Score vs K
plt.figure()
plt.plot(list(K_range), sil_scores, marker="o")
plt.title("Silhouette Score vs K")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Tampilkan nilai numeriknya
for k, sse, sil in zip(K_range, inertias, sil_scores):
    print(f"K={k}: SSE={sse:.2f}, Silhouette={sil:.4f}")


best_k = K_range[np.argmax(sil_scores)]
print("K terbaik menurut silhouette:", best_k)



# final_k = 3
final_k = best_k

kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

data["cluster"] = cluster_labels
data.head()


# 1) SSE / inertia (dari K-Means)
sse = kmeans_final.inertia_

# 2) Silhouette Score
sil = silhouette_score(X_scaled, cluster_labels)

# 3) Davies–Bouldin Index (semakin kecil semakin baik)
db = davies_bouldin_score(X_scaled, cluster_labels)

# 4) Calinski–Harabasz Score (semakin besar semakin baik)
ch = calinski_harabasz_score(X_scaled, cluster_labels)

print(f"Jumlah cluster (K): {final_k}")
print(f"SSE (Within-Cluster Sum of Squares)        : {sse:.2f}")
print(f"Silhouette Score                           : {sil:.4f}")
print(f"Davies-Bouldin Index (lebih kecil lebih baik): {db:.4f}")
print(f"Calinski-Harabasz Score (lebih besar lebih baik): {ch:.2f}")


submission = pd.DataFrame({
    "Id": data.index,          
    "Cluster": data["cluster"]
})

submission.head()


# Simpan ke file CSV untuk di-upload ke Kaggle
submission.to_csv("submission.csv", index=False)
print("File submission.csv berhasil dibuat.")

