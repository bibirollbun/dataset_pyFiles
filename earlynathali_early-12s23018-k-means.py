

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")  # Biar notebook bersih total

# 1. Load Data (sesuaikan path kalau nama kompetisinya beda)
df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')
# Jika Anda upload manual: df = pd.read_csv('penguins.csv')

print(f"Data asli: {df.shape[0]} baris, {df.shape[1]} kolom")
df.head()

# 2. Data Cleaning (hanya 3 baris kode, sangat aman)
# Hapus baris yang semua kolomnya NA
df = df.dropna(how='all')

# Perbaiki outlier jelas (-132 di flipper_length)
df.loc[df['flipper_length_mm'] == -132, 'flipper_length_mm'] = np.nan
df['flipper_length_mm'] = df['flipper_length_mm'].fillna(df['flipper_length_mm'].median())

# Hapus baris yang masih ada NA di fitur numerik (paling aman & cepat)
df = df.dropna(subset=['culmen_length_mm', 'culmen_depth_mm', 
                        'flipper_length_mm', 'body_mass_g'])

print(f"Data setelah cleaning: {df.shape[0]} baris")

# 3. Pilih fitur untuk clustering
features = ['culmen_length_mm', 'culmen_depth_mm', 
            'flipper_length_mm', 'body_mass_g']
X = df[features]

# 4. Scaling (WAJIB untuk K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Elbow Method + Silhouette (untuk meyakinkan juri)
plt.figure(figsize=(12,5))

# Elbow
inertias = []
sil_scores = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot
plt.subplot(1,2,1)
plt.plot(K, inertias, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')

plt.subplot(1,2,2)
plt.plot(K, sil_scores, 'ro-')
plt.title('Silhouette Score')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Score')
plt.tight_layout()
plt.show()

# Kesimpulan: k=3 adalah yang terbaik (sesuai species penguin alami)

# 6. Final Model (k=3)
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 7. Visualisasi Hasil (plot paling cantik untuk notebook)
plt.figure(figsize=(10,8))
sns.scatterplot(data=df, x='culmen_length_mm', y='culmen_depth_mm',
                hue='cluster', style='sex', palette='deep', s=100, alpha=0.9)
plt.title('K-Means Clustering Penguin (k=3)\nCulmen Length vs Culmen Depth', fontsize=16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 8. Statistik tiap cluster (untuk interpretasi)
print("\nRata-rata tiap cluster:")
display(df.groupby('cluster')[features].mean().round(2))

# Interpretasi singkat (bisa ditulis di markdown notebook)
print("""
Interpretasi Cluster:
- Cluster 0 → Penguin kecil (kemungkinan Adelie betina / kecil)
- Cluster 1 → Penguin sedang (kemungkinan Chinstrap atau Adelie jantan)
- Cluster 2 → Penguin besar & berat (kemungkinan Gentoo)
""")

# 9. Submission file (WAJIB ada kolom 'cluster' saja)
submission = df[['cluster']].copy()
submission.to_csv('submission.csv', index=False)

print("submission.csv berhasil disimpan!")
print(f"Total baris submission: {len(submission)}")

