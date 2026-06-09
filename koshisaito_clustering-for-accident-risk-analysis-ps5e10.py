import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')


train_df.head()


train_df.info()


save_dir = "/kaggle/working/"
os.makedirs(save_dir, exist_ok=True)

num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
cat_cols = ['road_type','lighting','weather','road_signs_present',
            'public_road','time_of_day','holiday','school_season']

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X = preprocess.fit_transform(train_df.drop(columns=['id','accident_risk']))
print(X.shape)
X[0]


# PCA for visualization
pca_vis = PCA(n_components=2, random_state=42)
X_2d = pca_vis.fit_transform(X)
print(X_2d.shape)
X_2d[0]


# Range of k values for clustering
K_range = range(2, 6)
inertias, silhouettes = [], []

for k in K_range:
    print(f"start k={k}")
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    train_df[f"cluster_k{k}"] = labels  # Save cluster assignments as a new column

    # Save evaluation metrics
    print("Saving evaluation metrics")
    inertias.append(km.inertia_)
    # silhouettes.append(silhouette_score(X, labels))

    sil = silhouette_score(X, labels, sample_size=10000, random_state=42)
    silhouettes.append(sil)

    # ---- Visualization ----
    # (a) Colored by cluster ID
    print("Visualizing clusters by ID")
    plt.figure(figsize=(8,8))
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="tab10", alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.title(f"PCA-2D Clusters (k={k})")
    plt.savefig(os.path.join(save_dir, f"clusters_k{k}_byID.png"), dpi=80, bbox_inches="tight")
    # plt.show()
    plt.close()

# (b) Colored by accident_risk (continuous gradient)
print("Visualizing accident risk distribution")
plt.figure(figsize=(8,8))
plt.scatter(X_2d[:,0], X_2d[:,1], 
            c=train_df['accident_risk'], 
            cmap="coolwarm", alpha=0.6)
plt.colorbar(label="Accident Risk (0=Blue, 1=Red)")
plt.title(f"PCA-2D Colored by Accident Risk")
plt.savefig(os.path.join(save_dir, f"clusters_byRisk.png"), dpi=80, bbox_inches="tight")
# plt.show()
plt.close()

# ---- Elbow method & Silhouette score summary ----
plt.figure(figsize=(8,8))

plt.subplot(1,2,1)
plt.plot(K_range, inertias, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method")

plt.subplot(1,2,2)
plt.plot(K_range, silhouettes, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "elbow_silhouette.png"), dpi=80, bbox_inches="tight")
# plt.show()
plt.close()


train_df.head()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

img1 = mpimg.imread("/kaggle/working/clusters_k4_byID.png")
img2 = mpimg.imread("/kaggle/working/clusters_byRisk.png")

axes[0].imshow(img1)
axes[0].axis("off")
axes[0].set_title("Clusters by ID")

axes[1].imshow(img2)
axes[1].axis("off")
axes[1].set_title("Clusters by Risk")

plt.tight_layout()
plt.show()


k = 4
cluster_summary = (
    train_df.groupby(f"cluster_k{k}")["accident_risk"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .reset_index()
)
print(cluster_summary)

sns.boxplot(x=f"cluster_k{k}", y="accident_risk", data=train_df)
plt.title(f"Accident Risk Distribution by Cluster (k={k})")
plt.show()


cluster_features = train_df.groupby(f"cluster_k{k}")[["num_lanes","curvature","speed_limit","num_reported_accidents"]].mean()
cluster_features


cluster_features = (
    train_df.groupby(f"cluster_k{k}")[["num_lanes","curvature","speed_limit","num_reported_accidents"]]
    .agg(["mean","std","min","max","median"])
)
cluster_features


cat_summary = train_df.groupby(f"cluster_k{k}")[["road_type","lighting","weather","road_signs_present","public_road","time_of_day","holiday","school_season"]].agg(lambda x: x.mode()[0])
cat_summary


for col in cat_cols:
    print(f"\n=== {col} distribution by cluster (k={k}) ===")
    ctab = pd.crosstab(train_df[f"cluster_k{k}"], train_df[col], normalize="index")
    print(ctab.round(3))


for col in cat_cols:
    print(f"\n=== {col} distribution by cluster ===")
    dist = (
        train_df.groupby(f"cluster_k{k}")[col]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )
    print(dist)

