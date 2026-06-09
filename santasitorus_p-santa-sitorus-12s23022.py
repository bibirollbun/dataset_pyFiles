# Cell 1 - imports umum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, confusion_matrix, classification_report

sns.set(style="whitegrid")
%matplotlib inline




import warnings
warnings.filterwarnings('ignore')




df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')
df.head()



import os

for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        print(os.path.join(root, file))



# Cell 3 - cepat EDA
df.info()
df.describe(include='all')



# Cell 4 - cek missing
df.isnull().sum()
# lihat baris dengan missing untuk memahami pola
df[df.isnull().any(axis=1)].head()



# Cell 5 - pilih fitur 
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
X = df[features].copy()
# simpan label (jika tersedia) untuk evaluasi & interpretasi
y = df['species'] if 'species' in df.columns else None



# Cell 6 - preprocessing: imputasi + encoding sex + scaling
# Imputasi numerik dengan median
num_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
imp_num = SimpleImputer(strategy='median')
X[num_cols] = imp_num.fit_transform(X[num_cols])

# encoding sex: ordinal encode (Male=1,Female=0) atau one-hot
if X['sex'].isnull().any():
    # isi missing sex dengan 'Unknown' (opsional), atau drop baris
    X['sex'] = X['sex'].fillna('Unknown')

enc = OrdinalEncoder(dtype=np.int64)
X[['sex']] = enc.fit_transform(X[['sex']])

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[num_cols + ['sex']])



# Cell 7 - visualisasi awal (pairplot)
sns.pairplot(pd.concat([pd.DataFrame(X_scaled, columns=num_cols+['sex']), y], axis=1),
             hue='species' if y is not None else None, diag_kind='kde', corner=True)
plt.suptitle('Pairplot (setelah scaling)', y=1.02)



# Cell 8 - optional: PCA 2D untuk visualisasi
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
if y is not None:
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='deep', s=60)
    plt.title('PCA 2D colored by true species (jika tersedia)')
else:
    plt.scatter(X_pca[:,0], X_pca[:,1], s=50)
    plt.title('PCA 2D of features')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.show()



# Cell 9 - tentukan K: Elbow + Silhouette scores
inertia = []
sil_scores = []
K_range = range(2,8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# plot
fig, ax1 = plt.subplots(1,2, figsize=(12,4))
ax1[0].plot(K_range, inertia, marker='o')
ax1[0].set_title('Elbow: inertia vs k')
ax1[0].set_xlabel('k'); ax1[0].set_ylabel('Inertia')

ax1[1].plot(K_range, sil_scores, marker='o')
ax1[1].set_title('Silhouette score vs k')
ax1[1].set_xlabel('k'); ax1[1].set_ylabel('Silhouette score')
plt.show()



# Cell 10 - pilih k (misal 3) dan fit final KMeans
k = 3  # tentukan berdasarkan hasil elbow & silhouette
kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters



# Cell 11 - evaluation: internal metrics
sil = silhouette_score(X_scaled, clusters)
db = davies_bouldin_score(X_scaled, clusters)
print(f"Silhouette score: {sil:.4f}")
print(f"Davies-Bouldin score: {db:.4f}")



# Cell 12 - evaluation jika label benar tersedia (external metrics)
if y is not None:
    ari = adjusted_rand_score(y, clusters)
    print(f"Adjusted Rand Index (ARI) against true species: {ari:.4f}")
    # confusion-style: map cluster -> dominant species
    ct = pd.crosstab(df['cluster'], df['species'])
    display(ct)



# Cell 13 - visualisasi cluster di ruang PCA
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='tab10', s=60)
# tambahkan centroids (transform centroid to PCA space)
centroids = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids)
plt.scatter(centroids_pca[:,0], centroids_pca[:,1], marker='X', s=200, c='black', label='centroids')
plt.legend(title='cluster')
plt.title('Clusters (K-Means) in PCA 2D')
plt.show()



# Cell 14 - summary stats per cluster (numeric + sex dalam satu tabel)

agg_dict = {col: ['mean', 'median', 'count'] for col in num_cols}
agg_dict['sex'] = ['count']   # categorical → hanya count

cluster_summary = df.groupby('cluster').agg(agg_dict)

display(cluster_summary)

# Boxplots
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='cluster', y=col, data=df)
    plt.title(f'{col} by cluster')
    plt.show()



# Cell 15 - interpretasi otomatis (bantuan)
for c in sorted(df['cluster'].unique()):
    print(f"\nCluster {c} (n={len(df[df['cluster']==c])}):")
    display(df[df['cluster']==c][num_cols + ['sex']].describe().T)
    if y is not None:
        print("Species distribution in this cluster:")
        display(df[df['cluster']==c]['species'].value_counts())



# Cell 16 - simpan notebook & output penting
df.to_csv('penguins_with_clusters.csv', index=False)





