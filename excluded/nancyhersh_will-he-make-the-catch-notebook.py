import glob, os

BASE_DIR = "/kaggle/input/nfl-big-data-bowl-2026-analytics"
input_files = sorted(glob.glob(f"{BASE_DIR}/**/input_2023_w*.csv", recursive=True))



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1) Load tracking data (using input_files you already defined)
df_tracking = pd.concat((pd.read_csv(f, low_memory=False) for f in input_files), ignore_index=True)

# 2) Filter to pass catchers to predict (WR/TE/RB)
receivers = df_tracking[
    (df_tracking["player_to_predict"] == True) &
    (df_tracking["player_position"].isin(["WR", "TE", "RB"]))
].copy()

# 3) Receiver-level features
receiver_features = receivers.groupby("nfl_id").agg(
    player_name=("player_name", "first"),
    player_position=("player_position", "first"),
    player_height=("player_height", "first"),
    player_weight=("player_weight", "first"),
    s_mean=("s", "mean"),
    s_max=("s", "max"),
    a_mean=("a", "mean"),
    a_max=("a", "max"),
    play_count=("nfl_id", "size"),
).reset_index()

def height_to_inches(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    if "-" not in s:
        return np.nan
    ft, inch = s.split("-", 1)
    try:
        return int(ft) * 12 + int(inch)
    except:
        return np.nan

receiver_features["height_inches"] = receiver_features["player_height"].apply(height_to_inches)

# 4) Min plays + impute
MIN_PLAYS = 6
rf = receiver_features[receiver_features["play_count"] >= MIN_PLAYS].copy()

feature_cols = ["height_inches", "player_weight", "s_mean", "s_max", "a_mean", "a_max"]
for c in feature_cols:
    rf[c] = rf[c].astype(float)
    rf[c] = rf[c].fillna(rf[c].median())

X = rf[feature_cols].values
X_scaled = StandardScaler().fit_transform(X)

# 5) Cluster
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
rf["cluster"] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, rf["cluster"])
print(f"Receivers clustered: {len(rf):,} | k={k} | silhouette={sil:.3f}")

# 6) PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
rf["pc1"] = X_pca[:, 0]
rf["pc2"] = X_pca[:, 1]

pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100
print(f"PCA variance explained: PC1={pc1_var:.1f}%, PC2={pc2_var:.1f}%, total={pc1_var+pc2_var:.1f}%")

plt.figure(figsize=(8, 6))
plt.scatter(rf["pc1"], rf["pc2"], c=rf["cluster"], s=12)
plt.title("Receiver archetypes (PCA of scaled features, colored by KMeans cluster)")
plt.xlabel(f"PC1 ({pc1_var:.1f}% variance)")
plt.ylabel(f"PC2 ({pc2_var:.1f}% variance)")
plt.grid(alpha=0.25)
plt.show()

# 7) Outputs
output_df = rf[["nfl_id", "player_name", "player_position", "play_count", "cluster", "pc1", "pc2"]].copy()

cluster_summary = (
    output_df.groupby(["cluster", "player_position"])
             .size()
             .unstack(fill_value=0)
)
cluster_summary["n_players"] = cluster_summary.sum(axis=1)
cluster_summary = cluster_summary.sort_index()

display(cluster_summary)



from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(
    "/kaggle/input/probability-of-catch-by-quintile-across-5-clusters/take5_all_clusters_combined.png"
)

plt.figure(figsize=(14, 7))
plt.imshow(img)
plt.axis("off")
plt.show()


