# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings


df = pd.read_csv("/kaggle/input/penguin-clustering-analysis/penguins.csv")


features = ["culmen_length_mm", "flipper_length_mm"]
data = df[features].copy()
nan_mask = data.isna().any(axis=1)  # Simpan mask NaN di awal!


imputer = SimpleImputer(strategy='median')
data_imputed = imputer.fit_transform(data)
scaler = StandardScaler()
scaled = scaler.fit_transform(data_imputed)
df["culmen_length_scaled"] = scaled[:, 0]
df["flipper_length_scaled"] = scaled[:, 1]


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled)
df["cluster"] = cluster_labels
df.loc[nan_mask, "cluster"] = 2


import matplotlib.pyplot as plt
import pandas as pd

print("Cluster distribution:")
print(df["cluster"].value_counts().sort_index())
print("\n" + "="*50)
print("5 baris pertama dengan cluster:")
print("="*50)

# Tampilkan data asli DAN data scaled untuk perbandingan
display_df = df[["culmen_length_mm", "flipper_length_mm", 
                 "culmen_length_scaled", "flipper_length_scaled", 
                 "cluster"]].head(10)  # Tampilkan 10 baris untuk lebih jelas
print(display_df.to_string())

print("\n" + "="*50)
print("Statistik per cluster (data scaled yang digunakan untuk plotting):")
print("="*50)
stats = df.groupby("cluster")[["culmen_length_scaled", "flipper_length_scaled"]].agg(['mean', 'count'])
print(stats)

# Visualisasi
markers = {0: "o", 1: "s", 2: "X"}  # X besar untuk lebih jelas
labels = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}
colors = {0: "C0", 1: "C1", 2: "C2"}

plt.figure(figsize=(10, 7))

for c in sorted(df["cluster"].unique()):
    subset = df[df["cluster"] == c]
    
    # Sesuaikan ukuran marker berdasarkan jumlah data
    marker_size = 100 if len(subset) < 10 else 50
    
    plt.scatter(
        subset["culmen_length_scaled"],
        subset["flipper_length_scaled"],
        marker=markers.get(c, "o"),
        label=f'{labels.get(c, f"Cluster {c}")} (n={len(subset)})',
        color=colors.get(c),
        alpha=0.7,
        s=marker_size,
        edgecolors='black',
        linewidths=0.5
    )

plt.title("Penguin K-Means Clustering (Scaled Features)", fontsize=14, fontweight='bold')
plt.xlabel("Culmen Length (scaled)", fontsize=12)
plt.ylabel("Flipper Length (scaled)", fontsize=12)
plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
plt.tight_layout()
plt.show()

# Verifikasi: Tampilkan semua data dari Cluster 2
print("\n" + "="*50)
print(f"SEMUA data dari Cluster 2 (total: {len(df[df['cluster']==2])}):")
print("="*50)
cluster_2_data = df[df["cluster"] == 2][["culmen_length_mm", "flipper_length_mm", 
                                          "culmen_length_scaled", "flipper_length_scaled"]]
print(cluster_2_data.to_string())


submission = pd.DataFrame({
    "id": range(len(df)),
    "cluster": df["cluster"].astype(int)
})

submission.to_csv("submission.csv", index=False)
print("\n" + "="*50)
print("SUBMISSION FILE (10 baris pertama):")
print("="*50)
print(submission.head(10).to_string())

print("\n" + "="*50)
print("VERIFIKASI: Baris dengan NaN (index 3)")
print("="*50)
print(f"Index 3 masuk ke Cluster: {submission.loc[3, 'cluster']}")

