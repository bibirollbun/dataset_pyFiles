%%capture
!pip install umap-learn


import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import time


df = pd.read_csv("/kaggle/input/clustering-physical-activity-data/Physical_Activity_Monitoring_unlabeled.csv")

df.head()


df.describe()


df.info()


x = df.copy().dropna()


models_to_evaluate = {
    "GaussianMixture": GaussianMixture(n_components=8, n_init=3),
    "KMeans": KMeans(n_init=10),
    # "AgglomerativeClustering": AgglomerativeClustering(),  # Need to much memory, will be trained separately
}


total_train_time = time.perf_counter()
for model_name, model in models_to_evaluate.items():
    print(f"Start to train model {model_name}")
    train_time = time.perf_counter()
    model.fit(x)
    print(f"Train took {(time.perf_counter() - train_time):.4f} (s)")
print(f"Total train time {(time.perf_counter() - total_train_time):.4f}")


NUM_SAMPLES = 50_000

x_cut = x[:NUM_SAMPLES]

y_pred_ac = AgglomerativeClustering().fit_predict(x_cut)


silhouette = silhouette_score(x_cut, y_pred_ac)
calinski = calinski_harabasz_score(x_cut, y_pred_ac)
davies = davies_bouldin_score(x_cut, y_pred_ac)

print(f"Metrics for AgglomerativeClustering:")
print(f"silhouette_score: {silhouette:.4f}")
print(f"calinski_harabasz_score: {calinski:.4f}")
print(f"davies_bouldin_score: {davies:.4f}")
print("\n*******\n")


for model_name, model in models_to_evaluate.items():
    y_pred = model.predict(x_cut)
    silhouette = silhouette_score(x_cut, y_pred)
    calinski = calinski_harabasz_score(x_cut, y_pred)
    davies = davies_bouldin_score(x_cut, y_pred)
        
    print(f"Metrics for {model_name}:")
    print(f"silhouette_score: {silhouette:.4f}")
    print(f"calinski_harabasz_score: {calinski:.4f}")
    print(f"davies_bouldin_score: {davies:.4f}")
    print("\n*******\n")


pca_2d = PCA(n_components=2)
pca_x_2d = pca_2d.fit_transform(x_cut)

manifold = umap.UMAP(n_jobs=-1).fit(x_cut)
umap_x_2d = manifold.transform(x_cut)


fig, axes = plt.subplots(nrows=len(models_to_evaluate) + 1, ncols=2, figsize=(25,30))

pca_2d_data = pd.DataFrame(pca_x_2d,columns=['PC1','PC2'])
umap_2d_data = pd.DataFrame(umap_x_2d,columns=['UMAP1','UMAP2']) 

pca_2d_data['cluster'] = pd.Categorical(y_pred_ac)
umap_2d_data['cluster'] = pd.Categorical(y_pred_ac)
axes[0, 0].scatter(pca_2d_data['PC1'], pca_2d_data['PC2'], c=pca_2d_data['cluster'], cmap=plt.colormaps["nipy_spectral"].resampled(20), edgecolors="none",
                       alpha=0.4)
axes[0, 0].title.set_text(f"PCA {model_name}")
axes[0, 0].grid(None)

axes[0, 1].scatter(umap_2d_data['UMAP1'], umap_2d_data['UMAP2'], c=umap_2d_data['cluster'], cmap=plt.colormaps["nipy_spectral"].resampled(20), edgecolors="none",
                   alpha=0.4)
axes[0, 1].title.set_text(f"UMAP {model_name}")
axes[0, 1].grid(None)

for i, (model_name, model) in enumerate(models_to_evaluate.items()):
    pred = model.predict(x_cut)
    pca_2d_data['cluster'] = pd.Categorical(pred)
    umap_2d_data['cluster'] = pd.Categorical(pred)

    axes[i + 1, 0].scatter(pca_2d_data['PC1'], pca_2d_data['PC2'], c=pca_2d_data['cluster'], cmap=plt.colormaps["nipy_spectral"].resampled(20), edgecolors="none",
                       alpha=0.4)
    axes[i + 1, 0].title.set_text(f"PCA {model_name}")
    axes[i + 1, 0].grid(None)
    
    axes[i + 1, 1].scatter(umap_2d_data['UMAP1'], umap_2d_data['UMAP2'], c=umap_2d_data['cluster'], cmap=plt.colormaps["nipy_spectral"].resampled(20), edgecolors="none",
                       alpha=0.4)
    axes[i + 1, 1].title.set_text(f"UMAP {model_name}")
    axes[i + 1, 1].grid(None)
plt.show()

