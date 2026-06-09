#Import library yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#Mengatur seed random
np.random.seed(42)


# Hilangkan warning floating format
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.float_format", lambda x: f"{x:.3f}")


# Load dataset
df = pd.read_csv("/kaggle/input/penguin-clustering-analysis/penguins.csv")


# Perbaikan baru: ubah inf → NaN tanpa warning
df = df.replace([np.inf, -np.inf], np.nan)


# Tampilkan data
df.head()


# Cek missing values
print(df.isna().sum())

print("\nStatistik deskriptif:")
display(df.describe())


# Drop baris kosong (umum untuk dataset ini)
df = df.dropna()


# Pilih fitur numerik
features = ["culmen_length_mm", "culmen_depth_mm",
            "flipper_length_mm", "body_mass_g"]

X = df[features]


# Normalisasi / Standarisasi (penting untuk K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#Menyiapkan list untuk menyimpan nilai inertia
inertia_list = []

#Membuat rentang nilai k yang akan dicoba
K = range(1, 11)


#Melakukan K-Means untuk setiap k
for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)

plt.figure(figsize=(7,4)) #Membuat gambar plot
plt.plot(K, inertia_list, marker='o') #Menggambar grafik Elbow Method

#Memberi label dan judul grafik
plt.xlabel("Jumlah Cluster (k)") 
plt.ylabel("Inertia")
plt.title("Elbow Method")

#Menampilkan grafik
plt.grid(True)
plt.show()


#Silhouette untuk berbagai k
silhouette_scores = {}

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    silhouette_scores[k] = silhouette_score(X_scaled, labels)

plt.figure(figsize=(7, 4))
plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='s')
plt.title("Silhouette Score untuk berbagai k")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

print("\nSilhouette Score per k:")
for k, score in silhouette_scores.items():
    print(f"k = {k}: {score:.3f}")


#Pemilihan k terbaik
k_opt = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nJumlah cluster optimal berdasarkan Silhouette Score = {k_opt}")


#Final K-Means Model
model = KMeans(n_clusters=k_opt, random_state=42)
labels = model.fit_predict(X_scaled)

df["cluster"] = labels
print("\nCluster berhasil ditambahkan ke dataset.")



#Interpretasi Hasil Cluster (Centroid)
centroids = pd.DataFrame(
    model.cluster_centers_,
    columns=features
)


# Kembalikan ke skala asli
centroids_original = pd.DataFrame(
    scaler.inverse_transform(model.cluster_centers_),
    columns=features
)

print("\nCentroid cluster (skala asli):")
display(centroids_original)


#Visualisasi Pairplot
sns.pairplot(df, vars=features, hue="cluster", palette="tab10")
plt.show()


# Visualisasi Cluster 2D 
plt.figure(figsize=(7,5))
plt.scatter(
    df["flipper_length_mm"], df["body_mass_g"],
    c=df["cluster"], s=50
)
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.title("Visualisasi Cluster 2D")
plt.grid(True)
plt.show()

