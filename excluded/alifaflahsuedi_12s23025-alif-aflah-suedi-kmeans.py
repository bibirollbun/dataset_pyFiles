import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Library Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Library Evaluasi (Sesuai Rubrik: Menggunakan berbagai metrik)
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Konfigurasi Tampilan
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


path = "/kaggle/input/penguin-clustering-analysis/penguins.csv"
if not os.path.exists(path):
    print("File tidak ditemukan.")
else:
    df = pd.read_csv(path)
    print("Data berhasil dimuat. Dimensi:", df.shape)


# Kita hanya memilih fitur numerik yang relevan untuk pengukuran fisik penguin.
# Kolom 'sex' (kategorikal) dan 'id' tidak digunakan untuk jarak Euclidean di K-Means.
selected_features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
print(f"\nFitur yang dipilih untuk klasterisasi: {selected_features}")

df_clean = df.copy()

# Handling Missing Values (Imputasi Median untuk Robustness)
for col in selected_features:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)

# Handling Missing Values untuk Sex (Modus) - sekadar pelengkap data
mode_sex = df_clean["sex"].mode()[0]
df_clean["sex"] = df_clean["sex"].fillna(mode_sex)

# Handling Outlier (Metode IQR)
Q1 = df_clean[selected_features].quantile(0.25)
Q3 = df_clean[selected_features].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

for col in selected_features:
    df_clean[col] = df_clean[col].clip(lower_bound[col], upper_bound[col])

# Scaling Data (Standarisasi) - Penting untuk K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[selected_features])


sse = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, sse, marker="o", linestyle='--', color='teal')
plt.xlabel("Jumlah Klaster (k)")
plt.ylabel("SSE (Sum of Squared Error)")
plt.title("Elbow Method: Menentukan k Optimal")
plt.grid(True)
plt.show()

# Berdasarkan Elbow (biasanya patahan di k=3), kita set k optimal
k_opt = 3
print(f"Jumlah klaster optimal yang dipilih: k = {k_opt}")


kmeans_final = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Menambahkan hasil label ke dataframe
df_clean["cluster"] = cluster_labels


# Metrik 1: SSE (Inertia)
sse_final = kmeans_final.inertia_

# Metrik 2: Silhouette Score (Kohesi & Separasi)
sil_score = silhouette_score(X_scaled, cluster_labels)

# Metrik 3: Davies-Bouldin Index (Semakin kecil semakin baik)
db_score = davies_bouldin_score(X_scaled, cluster_labels)

print("\n" + "="*40)
print("HASIL EVALUASI MODEL")
print("="*40)
print(f"1. SSE (Inertia)        : {sse_final:.2f} (Mengukur kepadatan dalam klaster)")
print(f"2. Silhouette Score     : {sil_score:.4f} (Mendekati 1 = klaster terpisah dengan baik)")
print(f"3. Davies-Bouldin Index : {db_score:.4f} (Mendekati 0 = pemisahan klaster sangat baik)")

# Interpretasi Profil Klaster (Analisis Mendalam)
print("\n" + "="*40)
print("INTERPRETASI PROFIL KLASTER")
print("="*40)
numeric_cols = selected_features + ["cluster"]
cluster_summary = df_clean[numeric_cols].groupby("cluster").mean()

print(cluster_summary)

print("\nAnalisis:")
for i in range(k_opt):
    mass = cluster_summary.loc[i, "body_mass_g"]
    flipper = cluster_summary.loc[i, "flipper_length_mm"]
    print(f"- Klaster {i}: Rata-rata Berat {mass:.1f}g, Sirip {flipper:.1f}mm.")


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df_clean["pca1"] = X_pca[:, 0]
df_clean["pca2"] = X_pca[:, 1]

plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df_clean, x="pca1", y="pca2", 
    hue="cluster", palette="viridis", s=100, alpha=0.8
)
plt.title(f"Visualisasi K-Means Clustering (k={k_opt}) menggunakan PCA")
plt.legend(title="Cluster ID")
plt.show()


submission = pd.DataFrame({
    "id": df_clean.index,
    "cluster": df_clean["cluster"]
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nFile submission.csv berhasil disimpan.")
print(submission.head())

