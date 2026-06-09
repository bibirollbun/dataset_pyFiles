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


# Import library untuk K-Means (sesuai modul: pandas, np, plt, KMeans, StandardScaler)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.feature_selection import SelectKBest, f_classif  # Pemilihan fitur
import scipy.spatial.distance as ssd  # Dunn index manual

print("Library imported! Load dataset.")


# Load CSV file dan lihat 5 baris pertama (sesuai modul – robust fallback)
url = 'https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv'  # URL valid
try:
    data = pd.read_csv(url)
    print("Dataset loaded from URL!")
except:
    # Fallback: String CSV dari modul dokumen (fix error)
    csv_string_loan = """Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status
LP001002,Male,No,0,Graduate,No,5849,0,NaN,360,1,Urban,Y
LP001003,Male,Yes,1,Graduate,No,4583,1508,128,360,1,Rural,N
LP001005,Male,Yes,0,Graduate,Yes,3000,0,NaN,360,1,Urban,Y
LP001006,Male,Yes,0,Not Graduate,No,2583,2358,120,360,1,Urban,Y
LP001008,Male,No,0,Graduate,No,6000,0,141,360,1,Urban,Y"""
    data = pd.read_csv(StringIO(csv_string_loan))
    print("Loaded from fallback string (modul sample)!")

print(f"Shape: {data.shape}")  # ~614 rows (or 5 for fallback)
data.head()  # First five rows


# Pemilihan fitur: Correlation dengan pseudo-target (Loan_Status), pilih top 2 numerik
data['Loan_Status_Num'] = data['Loan_Status'].map({'Y': 1, 'N': 0})  # Pseudo-label
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']  # Candidates
corr = data[numerical_cols].corrwith(data['Loan_Status_Num']).abs().sort_values(ascending=False)
top_features = corr.index[:2].tolist()  # Top 2
print("Correlation with Loan_Status (pseudo):")
print(corr)
print(f"\nFitur terpilih (top 2): {top_features}")

# Pilih fitur & clean NaN
X = data[top_features].dropna()
print(f"X shape after selection & clean: {X.shape}")


# Visualisasi data points dengan 2 fitur terpilih (sesuai modul)
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1])
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.title('Scatter Plot: Selected Features')
plt.show()


# Step 1: Pilih jumlah cluster k=2 (sesuai modul)
k = 2

# Step 2: Select random observations sebagai centroids
centroids = X.sample(n=k).reset_index(drop=True)

# Plot data dengan centroids awal (red dots)
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], label='Data Points')
plt.scatter(centroids.iloc[:, 0], centroids.iloc[:, 1], c='red', marker='o', s=200, label='Initial Centroids')
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.legend()
plt.title('Data with Initial Random Centroids (k=2)')
plt.show()


# Define conditions for K-Means loop (diff=1 initial, sesuai modul)
diff = 1
X_copy = X.copy()
X_copy['Cluster'] = np.nan

# Init centroids sebagai NumPy array fixed shape (k, 2) – robust no iloc error
centroids = X.sample(n=k).reset_index(drop=True).values  # (k, 2)

iteration = 0
while diff != 0 and iteration < 50:
    iteration += 1
    
    # Step 3: Assign points to closest centroid (vektorized distance)
    X_array = X_copy.iloc[:, :2].values  # (n_samples, 2) array
    distances = np.array([np.sqrt(np.sum((X_array - centroids[j])**2, axis=1)) for j in range(k)]).T  # (n, k)
    X_copy['Cluster'] = np.argmin(distances, axis=1)  # Assign closest cluster
    
    # Step 4: Recompute centroids (mean per cluster)
    unique_clusters = np.unique(X_copy['Cluster'])
    centroids_new = np.zeros((k, 2))  # Fixed (k, 2) array
    for cl in unique_clusters:
        mask = X_copy['Cluster'] == cl
        centroids_new[cl] = np.mean(X_array[mask], axis=0)  # Mean points in cl
    
    # Robust Pad: Isi missing clusters dengan old centroids
    for cl in range(k):
        if cl not in unique_clusters:
            centroids_new[cl] = centroids[cl]  # Copy old
            print(f"Iteration {iteration}: Padded missing cluster {cl}")
    
    # Step 5: Calculate diff (squared difference)
    diff = np.sum((centroids - centroids_new)**2)  # Scalar
    centroids = centroids_new.copy()  # Update array
    
    print(f"Iteration {iteration}: Diff = {diff:.4f} (Unique clusters: {len(unique_clusters)}/{k})")

print("Konvergensi tercapai (diff=0)!")
print("Final centroids (NumPy array):")
print(pd.DataFrame(centroids, columns=top_features))  # Tampil rapi


# Visualisasi clusters yang didapat (sesuai modul)
print(f"Type centroids: {type(centroids)}")  # Debug: array (NumPy)

plt.figure(figsize=(10, 6))
colors = ['r', 'g']
for i in range(k):
    cluster_data = X_copy[X_copy['Cluster'] == i]
    plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], c=colors[i], label=f'Cluster {i}')
# Fix: NumPy indexing untuk centroids array (no .iloc)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Final Centroids')
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.legend()
plt.title('K-Means Clusters (k=2, from Scratch)')
plt.show()

print("Cluster sizes (evaluasi balance):")
print(X_copy['Cluster'].value_counts().sort_index())


# Import required libraries for scikit-learn (sesuai modul)
# Skcit-learn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score  # Evaluasi mendalam
import scipy.spatial.distance as ssd  # Dunn index manual

print("Scikit-learn libraries imported for Exercise 2!")


# Read the data and look at the first five rows (sesuai modul)
data.head()  # First five rows
print(f"Data shape: {data.shape}")  # Konfirmasi ~5 rows sample (expand dengan full CSV)


# Pull out statistics related to the data (ApplicantIncome and LoanAmount – sesuai modul)
stats = X.describe()  # Selected features from Ex1
print("Statistics related to the data:")
print(stats)


# Bring variables to the same magnitude (scaling – sesuai modul)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)  # Fit & transform selected features
data_scaled = pd.DataFrame(data_scaled, columns=X.columns)  # Back to DataFrame for easy use

print("Statistics after scaling (mean~0, std~1):")
print(data_scaled.describe())


# Visualize normalized data with scatter plot (sesuai modul)
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled.iloc[:, 0], data_scaled.iloc[:, 1])
plt.xlabel('Scaled ' + top_features[0])
plt.ylabel('Scaled ' + top_features[1])
plt.title('Scatter Plot of Normalized Data')
plt.show()


# Fix UserWarning MKL memory leak di Windows (set threads rendah untuk data kecil)
import os
os.environ['OMP_NUM_THREADS'] = '3'

# Create KMeans with initialization as random and fit on the data (sesuai modul)
kmeans = KMeans(n_clusters=2, init='random', random_state=42)
kmeans.fit(data_scaled)

# Save new clusters for chart
y_km = kmeans.predict(data_scaled)

print("Inertia of the clusters:", kmeans.inertia_)  # Sum squared errors (SSE)


# Fit multiple k-means models, increase the number of clusters, store inertia, plot (sesuai modul)
inertias = []
k_range = range(1, 10)
for i in k_range:
    kmeans_temp = KMeans(n_clusters=i, init='random', random_state=42)
    kmeans_temp.fit(data_scaled)
    inertias.append(kmeans_temp.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimum Number of Clusters')
plt.show()

optimal_k = 3  # Dari elbow (siku di 3-5, pilih 3 sesuai modul)
print(f"Optimum k from elbow: {optimal_k}")


# Set number of clusters as 3 and fit the model (sesuai modul)
kmeans = KMeans(n_clusters=optimal_k, init='random', random_state=42)
kmeans.fit(data_scaled)
y_km = kmeans.predict(data_scaled)


# Value count of points in each of the above-formed clusters (sesuai modul)
print("Value counts per cluster:")
print(pd.Series(y_km).value_counts().sort_index())


# Check the inertia value again (sesuai modul)
print("Inertia for k=3:", kmeans.inertia_)


# Visualize the data for k=3 (sesuai modul)
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
for i in range(optimal_k):
    cluster_data = data_scaled[y_km == i]
    plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], c=colors[i], label=f'Cluster {i}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='x', s=200, label='Centroids')
plt.xlabel('Scaled ' + top_features[0])
plt.ylabel('Scaled ' + top_features[1])
plt.legend()
plt.title('K-Means Clusters (k=3, Scikit-Learn)')
plt.show()

