import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


data=pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')


data.head(3)


data.columns


mean_spectrum = data.drop(columns=["planet_id"]).mean()
plt.figure(figsize=(14, 6))
plt.plot(range(1, len(mean_spectrum) + 1), mean_spectrum.values)
plt.title("Mean Spectrum Across All Planets")
plt.xlabel("Wavelength Index (wl_1 to wl_283)")
plt.ylabel("Mean Reflectance")
plt.grid(True)
plt.tight_layout()
plt.show()


sample_df = data.drop(columns=["planet_id"]).T
plt.figure(figsize=(18,7))

for i in range(sample_df.shape[1]):
    plt.plot(sample_df.index.str.extract('(\d+)').astype(int)[0], sample_df.iloc[:, i], label=f'Sample {i+1}')

plt.title("Spectra of Planets")
plt.xlabel("Wavelength Index")
plt.ylabel("Reflectance")
plt.tight_layout()
plt.show()


X = data.drop(columns=["planet_id"])
iso_forest = IsolationForest(contamination=0.01, random_state=42)
outlier_labels = iso_forest.fit_predict(X)

normal_spectra = X[outlier_labels == 1].T
outlier_spectra = X[outlier_labels == -1].T
wavelength_indices = normal_spectra.index.str.extract('(\d+)').astype(int)[0]

plt.figure(figsize=(18, 7))
for i in range(normal_spectra.shape[1]):
    plt.plot(wavelength_indices, normal_spectra.iloc[:, i], alpha=0.03, color='darkblue')

for i in range(outlier_spectra.shape[1]):
    plt.plot(wavelength_indices, outlier_spectra.iloc[:, i], alpha=0.8, color='red', linewidth=1.2)

plt.title("Spectra of Planets with Outliers Highlighted in Red")
plt.xlabel("Wavelength Index")
plt.ylabel("Reflectance")
plt.tight_layout()
plt.show()


df_with_labels = data.copy()
df_with_labels["outlier"] = outlier_labels
outlier_planets = df_with_labels[df_with_labels["outlier"] == -1]
outlier_details = outlier_planets.drop(columns=["outlier"])
outlier_spectra = outlier_details.drop(columns=["planet_id"]).T
wavelength_indices = outlier_spectra.index.str.extract('(\d+)').astype(int)[0]

plt.figure(figsize=(18, 8))
for i in range(outlier_spectra.shape[1]):
    plt.plot(wavelength_indices, outlier_spectra.iloc[:, i], label=f'Planet ID: {outlier_details.iloc[i]["planet_id"]}')

plt.title("Spectral Signatures of Outlier Planets")
plt.xlabel("Wavelength Index")
plt.ylabel("Reflectance")
plt.legend()
plt.tight_layout()
plt.show()


X = data.drop(columns=["planet_id"])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

kmeans = KMeans(n_clusters=4, random_state=42)
pca_df["Cluster"] = kmeans.fit_predict(pca_result)

plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", alpha=0.8)
plt.title("KMeans Clustering of Planet Spectra in PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()




