import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE



# Read the metadata (dataframe with columns 'hash_id' and 'family')
metadata = pd.read_csv('/kaggle/input/biotrove-clustering/metadata.csv')
families = np.unique(metadata.family)



# Read the embeddings of the images
with open(f'/kaggle/input/cbtd-bioclip-embeddings/embedding.pickle', 'rb') as f:
    embeddings = pickle.load(f) # array of shape (49633, 768)



%%time
# Cluster the images to genus clusters
# Plot the clusters for the first few families
# The 2d embedding is computed by T-SNE
# The colors are the clusters of AgglomerativeClustering on the original BioCLIP embedding
genus_cluster = []
n_clusters_list = []
_, axs = plt.subplots(4, 4, figsize=(15, 15))
axs = axs.ravel()
for i in range(len(families)):
    family = families[i]
    family_mask = metadata.family == family
    embeddings_f = embeddings[family_mask]

    # Guess the number of genera
    model = AgglomerativeClustering(n_clusters=3,
                                    metric='euclidean',
                                    linkage='ward',
                                    compute_distances=True)
    model.fit(embeddings_f)
    diff_distances = np.diff(model.distances_)[:-1]
    n_clusters = len(model.distances_) - np.argmax(diff_distances)
    n_clusters = min(n_clusters, len(embeddings_f) // 20) # every genus has at least 20 images
    if n_clusters >= 7:
        print(f"{family:20} {n_clusters} genera")
    if n_clusters * 20 > len(embeddings_f):
        raise ValueError(f"{family} cannot have {n_clusters} clusters for {len(embeddings_f)} images.")
    n_clusters_list.append(n_clusters)

    # Cluster the images for submission
    model.set_params(n_clusters=n_clusters)
    clusters = model.fit_predict(embeddings_f)

    # Plot the clusters for the first few families
    if i < len(axs):
        tsne = TSNE()
        embeddings_2d = tsne.fit_transform(embeddings_f)
        ax = axs[i]
        ax.scatter(embeddings_2d.T[0], embeddings_2d.T[1], s=8, c=clusters, cmap='brg')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(family)

    # Format the output for submission
    clusters = [f"{family}_g_{c}" for c in clusters]
    genus_cluster.append(pd.Series(clusters, index=metadata.hash_id[family_mask]))

plt.tight_layout()
plt.show()

# Show the distribution of n_clusters; median should be 5
print(f"Median genus clusters: {np.median(n_clusters_list)}")
vc = np.unique(n_clusters_list, return_counts=True)
plt.bar(*vc)
plt.xlabel('# genus clusters')
plt.ylabel('count')
plt.show()

genus_cluster = pd.concat(genus_cluster) # Series with hash_id as index and genus cluster as value
genus_cluster.name = 'genus_cluster'



# Cluster the images to species clusters

def cluster_for_factor(factor):
    species_cluster = []
    chs_list, dbs_list, ss_list = [], [], []
    for family in families:
        family_mask = metadata.family == family
        embeddings_f = embeddings[family_mask]
    
        model = AgglomerativeClustering(n_clusters=int(len(embeddings_f) * factor + 0.5),
                                        metric='euclidean',
                                        linkage='ward')
        clusters = model.fit_predict(embeddings_f)
        clusters = [f"{family}_s_{c}" for c in clusters]
        species_cluster.append(pd.Series(clusters, index=metadata.hash_id[family_mask]))
        # chs = calinski_harabasz_score(embeddings_f, clusters) # higher is better
        # dbs = davies_bouldin_score(embeddings_f, clusters) # lower is better
        # ss = silhouette_score(embeddings_f, clusters, metric='cosine') # higher is better
        # chs_list.append(chs)
        # dbs_list.append(dbs)
        # ss_list.append(ss)
        # print(f"{family:25} {model.n_clusters_:3}/{family_mask.sum():3} {chs:8.3f}") # n_clusters/n_images

    # print(f"{factor=:.3f} chs={np.mean(chs_list):5.2f} (higher is better)")
    # print(f"{factor=:.3f} dbs={np.mean(dbs_list):5.2f} (lower is better)")
    # print(f"{factor=:.3f} ss={np.mean(ss_list):7.5f} (higher is better)")
    
    species_cluster = pd.concat(species_cluster) # Series with hash_id as index and species cluster as value
    species_cluster.name = 'species_cluster'
    return species_cluster
    
factor = 0.1
species_cluster = cluster_for_factor(factor)


# Write the submission file

sub = pd.DataFrame({
    'hash_id': metadata.hash_id,
    'family_cluster': metadata.family,
    # 'genus_cluster': metadata.family,
    # 'species_cluster': np.arange(len(metadata))
})
sub = sub.join(genus_cluster, on='hash_id', how='inner', validate='1:1')
sub = sub.join(species_cluster, on='hash_id', how='inner', validate='1:1')
display(sub)
sub.to_csv('submission.csv', index=False)
print()
!head submission.csv


len(np.unique(sub.family_cluster)), len(np.unique(sub.genus_cluster)), len(np.unique(sub.species_cluster))




