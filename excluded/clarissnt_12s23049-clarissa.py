import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

# Agar plot langsung muncul di notebook
%matplotlib inline


# Kalau Anda upload file di Kaggle, pakai path ini:
df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')

# Kalau pakai file dari attachment/chat ini:
# df = pd.read_csv('penguins.csv')

print("Shape :", df.shape)
df.head(10)


print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nValue counts kolom sex:\n", df['sex'].value_counts(dropna=False))
df.describe()


# 1. Fix flipper_length_mm yang jelas salah (-132 dan 5000)
df.loc[df['flipper_length_mm'] < 0, 'flipper_length_mm'] = np.nan
df.loc[df['flipper_length_mm'] > 300, 'flipper_length_mm'] = np.nan   # Gentoo max ~231 mm

# 2. Fix nilai sex yang aneh (titik)
df['sex'] = df['sex'].replace({'.': np.nan, 'NA': np.nan})

# 3. Hapus baris yang semua fitur numeriknya NA (hanya 2 baris)
df = df.dropna(how='all', 
               subset=['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g'])

print("Shape setelah hapus baris kosong total:", df.shape)


# Kolom numerik
num_cols = ['culmen_length_mm', 'culmen_depth_mm', 
            'flipper_length_mm', 'body_mass_g']

# Imputasi dengan median (lebih robust daripada mean)
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Kolom sex → imputasi dengan modus
mode_sex = df['sex'].mode()[0]
df['sex'] = df['sex'].fillna(mode_sex)

print("Missing values setelah imputasi:")
print(df.isnull().sum())


print("Data akhir shape:", df.shape)
df.head()
df.describe()


X = df[num_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ubah kembali ke DataFrame biar mudah dilihat
X_scaled_df = pd.DataFrame(X_scaled, columns=num_cols)
X_scaled_df.head()


inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_range, inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()


silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k={k} → Silhouette Score = {score:.4f}")

plt.figure(figsize=(8,5))
plt.plot(range(2,11), silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Score')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


final_kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = final_kmeans.fit_predict(X_scaled)

print("Clustering selesai!")
print("Silhouette Score akhir:", silhouette_score(X_scaled, df['cluster']).round(4))


cluster_means = df.groupby('cluster')[num_cols].mean().round(2)
cluster_means


pd.crosstab(df['cluster'], df['sex'], margins=True)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]

plt.figure(figsize=(10,7))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, 
                palette='deep', s=100, alpha=0.8)
plt.title('Penguin Clusters (PCA 2D Projection)')
plt.legend(title='Cluster')
plt.show()


sns.pairplot(df, vars=num_cols, hue='cluster', palette='tab10', diag_kind='kde')
plt.suptitle('Pairplot semua fitur berdasarkan Cluster', y=1.02)
plt.show()


# Contoh submission: rata-rata tiap cluster
submission = cluster_means.reset_index()
submission.to_csv('submission.csv', index=False)

print("File submission.csv sudah disimpan!")
submission

