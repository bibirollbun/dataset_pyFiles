!pip install umap-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap.umap_ as umap
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway, kruskal
from scipy.stats import levene, bartlett
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


DATA_DIR = '/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot'
CSV_PATH = '/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv'

df = pd.read_csv(CSV_PATH)

data_list = []
labels = []

expected_shape = (128, 128, 125) 

for _, row in df.iterrows():
    file_path = os.path.join(DATA_DIR, row['id'])
    try:
        cube = np.load(file_path)

        if cube.shape != expected_shape:
            continue  

        mean_spectrum = cube.reshape(-1, cube.shape[2]).mean(axis=0)
        data_list.append(mean_spectrum)
        labels.append(row['label'])

    except Exception as e:
        print(f"Error with {file_path}: {e}")

X = np.array(data_list)
y = np.array(labels)

reducer = umap.UMAP(random_state=42)
X_embedded = reducer.fit_transform(X)

plt.figure(figsize=(10, 8), constrained_layout=True)
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette='tab10')
plt.title('UMAP Projection of Hyperspectral Data by Label')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("umap_plot.png")
plt.show()




df_spectra = pd.DataFrame(X)
df_spectra['id'] = y

plt.figure(figsize=(12, 6))
for label in df_spectra['id'].unique():
    mean_spectrum = df_spectra[df_spectra['id'] == label].drop('id', axis=1).mean()
    plt.plot(mean_spectrum, label=label)

plt.title('Mean Reflectance Spectra per Class')
plt.xlabel('Bands (Spectral Channels)')
plt.ylabel('Reflectance')
plt.legend(title='Class Label')
plt.tight_layout()
plt.show()



df_plot = df_spectra.copy()
selected_bands = [10, 50, 100] 

for band in selected_bands:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_plot, x='id', y=band)
    plt.title(f'Boxplot for Band {band}')
    plt.xlabel('Class')
    plt.ylabel(f'Reflectance at Band {band}')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df_plot, x='id', y=band)
    plt.title(f'Violinplot for Band {band}')
    plt.xlabel('Class')
    plt.ylabel(f'Reflectance at Band {band}')
    plt.tight_layout()
    plt.show


correlation_matrix = df_spectra.drop('id', axis=1).corr()

corr_unstacked = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
corr_pairs = corr_unstacked.unstack().dropna()
top_corr = corr_pairs.abs().sort_values(ascending=False).head(10)

print("Top 10 most correlated band pairs (by absolute correlation):")
for (band1, band2), corr_val in top_corr.items():
    print(f"Bands {band1} & {band2}: correlation = {corr_val:.3f}")

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True)
plt.title('Spectral Band Correlation Heatmap')
plt.xlabel('Band')
plt.ylabel('Band')
plt.tight_layout()
plt.show()


mean_spectra_by_class = df_spectra.groupby('id').mean()
linked = linkage(mean_spectra_by_class, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked, labels=mean_spectra_by_class.index.tolist(), leaf_rotation=90)
plt.title('Dendrogram of Class Mean Spectra')
plt.xlabel('Class')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()



R_band, G_band, B_band = 90, 60, 30

def create_rgb_image_for_class(class_label):
    sample_path = os.path.join(DATA_DIR, df[df['label'] == class_label]['id'].iloc[0])
    cube = np.load(sample_path)

    rgb_image = np.stack([
        cube[:, :, R_band],
        cube[:, :, G_band],
        cube[:, :, B_band]
    ], axis=-1)

    
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    return rgb_image


class_labels = df['label'].unique()[:4]  
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for ax, class_label in zip(axes.flatten(), class_labels):
    rgb_image = create_rgb_image_for_class(class_label)
    ax.imshow(rgb_image)
    ax.set_title(f'Class: {class_label}')
    ax.axis('off')

plt.tight_layout()
plt.show()


anova_results = {}
for band in range(X.shape[1]): 
    groups = [X[y == label, band] for label in np.unique(y)]  
    f_stat, p_value = f_oneway(*groups)
    anova_results[band] = p_value


significant_bands_anova = {band: p for band, p in anova_results.items() if p < 0.05}
print("ANOVA significant bands:", significant_bands_anova)

kruskal_results = {}
for band in range(X.shape[1]):
    groups = [X[y == label, band] for label in np.unique(y)] 
    h_stat, p_value = kruskal(*groups)
    kruskal_results[band] = p_value

significant_bands_kruskal = {band: p for band, p in kruskal_results.items() if p < 0.05}
print("Kruskal-Wallis significant bands:", significant_bands_kruskal)

plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

for band in range(X.shape[1]):
    _, p_value = shapiro(X[:, band])
    print(f"Shapiro-Wilk test for Band {band}: p-value = {p_value}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10')
plt.title('PCA Projection')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.tight_layout()
plt.show()


class1_data = X[y == df['label'].unique()[0]]  
class2_data = X[y == df['label'].unique()[1]] 


t_stat, p_value_t = ttest_ind(class1_data, class2_data, axis=0)

u_stat, p_value_u = mannwhitneyu(class1_data.flatten(), class2_data.flatten())

print("t-test p-values:", p_value_t)
print("Mann-Whitney U test p-value:", p_value_u)


levene_results = {}
for band in range(X.shape[1]):
    groups = [X[y == label, band] for label in np.unique(y)]
    stat, p_value = levene(*groups)
    levene_results[band] = p_value

bartlett_results = {}
for band in range(X.shape[1]):
    groups = [X[y == label, band] for label in np.unique(y)]
    stat, p_value = bartlett(*groups)
    bartlett_results[band] = p_value

significant_bands_levene = {band: p for band, p in levene_results.items() if p < 0.05}
significant_bands_bartlett = {band: p for band, p in bartlett_results.items() if p < 0.05}

print("Levene Test significant bands:", significant_bands_levene)
print("Bartlett Test significant bands:", significant_bands_bartlett)



def calculate_ndvi(cube, red_band=30, nir_band=90):
    red = cube[:, :, red_band]
    nir = cube[:, :, nir_band]
    return (nir - red) / (nir + red)

sample_path = os.path.join(DATA_DIR, df[df['label'] == df['label'].unique()[0]]['id'].iloc[0])
cube = np.load(sample_path)
ndvi_image = calculate_ndvi(cube)

plt.imshow(ndvi_image, cmap='RdYlGn')
plt.title("NDVI (NIR-Red)/(NIR+Red)")
plt.colorbar()
plt.show()



from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=25, random_state=42)
kmeans.fit(X)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("KMeans Clustering")
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(label='Cluster')
plt.show()



def calculate_snr(X, y, class_label):
    class_data = X[y == class_label]
    mean_signal = class_data.mean(axis=0)
    noise = class_data.std(axis=0)
    snr = mean_signal / noise
    return snr

snr_values = {}
for label in np.unique(y):
    snr_values[label] = calculate_snr(X, y, label)

for label, snr in snr_values.items():
    print(f"SNR for class {label}: {snr[:5]}")  


