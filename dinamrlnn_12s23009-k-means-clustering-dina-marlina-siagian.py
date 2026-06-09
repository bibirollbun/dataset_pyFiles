# CELL 1 - IMPORTS & SETTINGS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# plotting style
sns.set(style='whitegrid')
plt.rcParams.update({'figure.dpi':110, 'font.size':11})



# CELL 2 - LOAD DATA (try common Kaggle path, fallback local)
candidates = [
    '/kaggle/input/penguin-clustering-analysis/penguins.csv',
    '/kaggle/input/penguins/penguins.csv',
    '/mnt/data/penguins.csv',
    'penguins.csv'
]
data_path = next((p for p in candidates if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError(f"penguins.csv not found. Tried: {candidates}")
df = pd.read_csv(data_path)
print("Loaded:", data_path)
print("Initial shape:", df.shape)
display(df.head(8))



# CELL 3 - INSPEKSI SINGKAT
print("Info:")
display(df.info())
print("\nMissing values per column:")
display(df.isnull().sum())
# show basic describe for numeric
display(df.describe(include='all'))



# CELL 4 - CLEANING: COERCE NUMERIC & IQR OUTLIER REMOVAL
def coerce_numeric(df, exclude=['sex']):
    df2 = df.copy()
    for col in df2.columns:
        if col not in exclude:
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
    return df2

def remove_outliers_iqr(df, cols, k=1.5):
    """
    Remove rows that are outside [Q1 - k*IQR, Q3 + k*IQR] for any given cols.
    Returns cleaned DataFrame (copy).
    """
    df2 = df.copy()
    for col in cols:
        q1 = df2[col].quantile(0.25)
        q3 = df2[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - k * iqr
        high = q3 + k * iqr
        df2 = df2[(df2[col] >= low) & (df2[col] <= high)]
    return df2

# Apply coercion
df = coerce_numeric(df, exclude=['sex'])

# Choose numeric features for clustering
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Drop rows with NA in our features (we'll keep original df for potential label reference)
df_features = df[features].copy().dropna().reset_index(drop=True)

# Remove outliers using IQR method (safer than fixed thresholds)
df_clean = remove_outliers_iqr(df_features, features, k=1.5).reset_index(drop=True)

print("Shape before outlier removal:", df_features.shape)
print("Shape after IQR-based outlier removal:", df_clean.shape)
display(df_clean.describe().round(2))



# CELL 5 - EDA SINGKAT (pairplot + distribusi)
sns.pairplot(df_clean, diag_kind='kde', plot_kws={'alpha':0.6})
plt.suptitle('Pairplot features (cleaned)', y=1.02)
plt.show()

# Show correlation matrix
plt.figure(figsize=(6,4))
sns.heatmap(df_clean.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()



# CELL 6 - SCALING
X = df_clean[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled: mean (should ~0):", np.round(X_scaled.mean(axis=0), 6))
print("Scaled: std  (should =1):", np.round(X_scaled.std(axis=0), 6))



# CELL 7 - EVALUATE RANGE K (print all metric values per k)
def evaluate_k_range(X_scaled, k_min=2, k_max=10, random_state=42):
    records = []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        rec = {
            'k': k,
            'inertia': km.inertia_,
            'silhouette': silhouette_score(X_scaled, labels) if k>1 else np.nan,
            'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
            'davies_bouldin': davies_bouldin_score(X_scaled, labels)
        }
        records.append(rec)
    return pd.DataFrame(records)

metrics_df = evaluate_k_range(X_scaled, 2, 10)
display(metrics_df)

# Plot metrics side-by-side to help choose k
K = metrics_df['k'].values
plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.plot(K, metrics_df['inertia'], 'o-'); plt.title('Elbow (Inertia)'); plt.xlabel('k'); plt.grid(True)
plt.subplot(1,3,2)
plt.plot(K, metrics_df['silhouette'], 'o-'); plt.title('Silhouette'); plt.xlabel('k'); plt.grid(True)
plt.subplot(1,3,3)
plt.plot(K, metrics_df['davies_bouldin'], 'o-'); plt.title('Davies-Bouldin'); plt.xlabel('k'); plt.grid(True)
plt.tight_layout(); plt.show()

# Print metrics per k for explicit justification
for _, row in metrics_df.iterrows():
    print(f"k={int(row.k)}: inertia={row.inertia:.1f}, silhouette={row.silhouette:.4f}, db={row.davies_bouldin:.4f}, ch={row.calinski_harabasz:.1f}")



# CELL 8 - PEMILIHAN K: auto by silhouette + domain knowledge fallback
k_best_sil = int(metrics_df.loc[metrics_df['silhouette'].idxmax(),'k'])
k_elbow_candidate = 3  # domain knowledge: penguin species = 3 (use as interpretability candidate)

print("k with max silhouette:", k_best_sil)
print("Interpretability / elbow candidate k:", k_elbow_candidate)

# Choose final_k variable (you can set to k_best_sil or k_elbow_candidate)
# We'll compute results for both and compare.
final_candidates = [k_elbow_candidate, k_best_sil]
final_candidates = sorted(list(set(final_candidates)))
final_candidates



# CELL 9 - FIT KMEANS FOR CANDIDATES & SHOW RESULTS (centroids in original units)
results = {}
for k in final_candidates:
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    # centroid in original units
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    sizes = pd.Series(labels).value_counts().sort_index()
    results[k] = {'model': km, 'labels': labels, 'inertia': inertia, 'silhouette': sil,
                  'calinski_harabasz': ch, 'davies_bouldin': db, 'centroids_orig': centroids_orig,
                  'sizes': sizes}
    print(f"--- k={k} ---")
    print(f"inertia={inertia:.1f}, silhouette={sil:.4f}, DB={db:.4f}, CH={ch:.1f}")
    print("Cluster sizes:\n", sizes.to_string())
    display(pd.DataFrame(centroids_orig, columns=features).round(3))



# CELL 10 - PCA VISUALIZATION for chosen k (both candidates)
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X_scaled)

for k in final_candidates:
    labels = results[k]['labels']
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=labels, palette='tab10', legend='full', s=50, alpha=0.8, edgecolor='k')
    cent_pca = pca.transform(results[k]['model'].cluster_centers_)
    plt.scatter(cent_pca[:,0], cent_pca[:,1], c='red', marker='X', s=200, label='centroids', edgecolors='k')
    plt.title(f'PCA scatter - k={k}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    plt.legend()
    plt.grid(True)
    plt.show()



# CELL 11 - SILHOUETTE PER-SAMPLE PLOT for chosen k (show for each candidate)
def plot_silhouette(X_scaled, labels, k):
    sil_vals = silhouette_samples(X_scaled, labels)
    y_lower = 10
    plt.figure(figsize=(8,5))
    for i in range(k):
        ith = np.sort(sil_vals[labels==i])
        size_i = ith.shape[0]
        if size_i == 0:
            continue
        y_upper = y_lower + size_i
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, alpha=0.7)
        plt.text(-0.03, y_lower + 0.5*size_i, str(i))
        y_lower = y_upper + 10
    avg = silhouette_score(X_scaled, labels)
    plt.axvline(x=avg, color='red', linestyle='--', label=f'avg silhouette={avg:.3f}')
    plt.title(f'Silhouette plot (k={k})')
    plt.xlabel('Silhouette coefficient')
    plt.ylabel('Cluster label')
    plt.legend()
    plt.show()

for k in final_candidates:
    print("Silhouette plot for k=", k)
    plot_silhouette(X_scaled, results[k]['labels'], k)



# CELL 12 - CHOOSE FINAL k and assign to df (use copy to avoid SettingWithCopyWarning)
# Decision: choose k_elbow_candidate (3) for interpretability by default; change if you prefer silhouette-opt
chosen_k = k_elbow_candidate  # or set to k_best_sil to use silhouette-best
chosen = results[chosen_k]
labels_final = chosen['labels']

# prepare output DataFrame copy
out_df = df_clean.copy().reset_index(drop=True).copy()   # explicit copy
out_df['cluster_k'+str(chosen_k)] = labels_final
print(f"Assigned clusters (k={chosen_k}). Cluster sizes:")
display(chosen['sizes'])



# CELL 13 - CENTROIDS (original units) & per-cluster means (data-based)
centroids_df = pd.DataFrame(chosen['centroids_orig'], columns=features)
centroids_df.index = [f'cluster_{i}' for i in centroids_df.index]
print("Centroids (original units):")
display(centroids_df.round(3))

print("\nPer-cluster observed means (from actual cleaned data):")
cluster_means = out_df.groupby('cluster_k'+str(chosen_k))[features].mean()
cluster_means.index = [f'cluster_{i}' for i in cluster_means.index]
display(cluster_means.round(3))



# CELL 14 - IF GROUND TRUTH 'species' EXISTS: compute ARI & NMI (automated check)
# We attempt to find a species column in the original dataframe (df) and align indices if possible.
if 'species' in df.columns:
    # Need to align species values to rows kept in df_clean (we processed only features)
    # Approach: find rows in original df that match feature rows (risky if duplicates) - better if original index preserved.
    # If original df had no index mapping, best to merge by features + small tolerance.
    print("Found 'species' column in original df. Attempting to align...")
    # Merge on features (exact) to retrieve species for cleaned rows
    df_original = df.copy().reset_index().rename(columns={'index':'orig_index'})
    merged = df_original.merge(out_df.reset_index().rename(columns={'index':'clean_idx'}),
                               left_on=features, right_on=features, how='inner', suffixes=('_orig','_clean'))
    if merged.shape[0] >= out_df.shape[0]:
        # If merge produces at least as many rows, try to get species per cleaned row
        species_mapped = merged['species'].values[:out_df.shape[0]]
        try:
            ari = adjusted_rand_score(species_mapped, labels_final)
            nmi = normalized_mutual_info_score(species_mapped, labels_final)
            print(f"Adjusted Rand Index (ARI): {ari:.4f}")
            print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
        except Exception as e:
            print("Could not compute ARI/NMI. Error:", e)
    else:
        print("Could not reliably align species column to cleaned rows (merge returned fewer matches).")
else:
    print("No 'species' column found in original df — skipping ARI/NMI.")



# CELL 15 - EVALUASI FINAL & INTERPRETASI (format rapih untuk laporan)
print("EVALUASI FINAL (k = {})".format(chosen_k))
print("="*60)
print(f"Inertia (SSE)           : {chosen['inertia']:.2f}")
print(f"Silhouette score        : {chosen['silhouette']:.4f}")
print(f"Davies-Bouldin index    : {chosen['davies_bouldin']:.4f}")
print(f"Calinski-Harabasz index : {chosen['calinski_harabasz']:.2f}")
print("\nCluster sizes:")
display(chosen['sizes'])

print("\nCentroids (original units):")
display(centroids_df.round(3))

print("\nInterpretasi singkat per cluster (gunakan centroid & means di atas):")
for i, row in centroids_df.iterrows():
    print(f"{i} -> " +
          f"culmen_len={row['culmen_length_mm']:.1f}, culmen_depth={row['culmen_depth_mm']:.1f}, "
          f"flipper={row['flipper_length_mm']:.1f}, body_mass={row['body_mass_g']:.1f}")
print("\nCatatan: Jika ingin memetakan cluster ke spesies, perlihatkan ARI/NMI (jika label ada) dan contoh observasi tiap cluster.")



# CELL 16 - SAVE RESULTS (safe path)
if os.path.exists('/kaggle/working'):
    save_dir = '/kaggle/working'
else:
    save_dir = '.'
output_file = os.path.join(save_dir, f'penguin_clusters_k{chosen_k}.csv')
out_df.to_csv(output_file, index=False)
print("Saved clustered data to:", output_file)



# CELL 17 - OPTIONAL: TRY KMeans FROM-SCRATCH (nilai tambah, optional)
# Implementasi singkat untuk pembelajaran (tidak dipakai di pipeline akhir)
class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=42):
        self.k = k; self.max_iters = max_iters; self.tol = tol; self.random_state = random_state
    def fit(self, X):
        np.random.seed(self.random_state)
        n = X.shape[0]
        self.centroids = X[np.random.choice(n, self.k, replace=False)]
        for it in range(self.max_iters):
            dists = np.linalg.norm(X[:,None] - self.centroids[None,:], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else self.centroids[i] for i in range(self.k)])
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids
        self.labels_ = labels
        self.inertia_ = np.sum((X - self.centroids[labels])**2)
        return self

# quick test (on scaled data)
km_scratch = KMeansFromScratch(k=chosen_k, random_state=42)
km_scratch.fit(X_scaled)
print("KMeansFromScratch inertia:", km_scratch.inertia_)


