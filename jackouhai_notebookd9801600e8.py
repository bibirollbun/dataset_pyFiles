import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# LOAD DATA
data = pd.read_csv('/kaggle/input/clustering-cugb/Mall_Customers.csv')
print(data.head())


# ============================================
# 1. KHÁM PHÁ DỮ LIỆU (EDA)
# ============================================
print("=" * 60)
print("THÔNG TIN CƠ BẢN VỀ DỮ LIỆU")
print("=" * 60)
print(f"\nKích thước dữ liệu: {data.shape}")
print(f"\nCác cột: {list(data.columns)}")
print(f"\nThông tin chi tiết:")
print(data.info())
print(f"\nThống kê mô tả:")
print(data.describe())
print(f"\nGiá trị null:\n{data.isnull().sum()}")


# CHỌN FEATURES
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# CHUẨN HÓA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# K-MEANS
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)


# VISUALIZE
plt.figure(figsize=(10, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], 
            c=data['Cluster'], cmap='rainbow', s=100, alpha=0.6, edgecolors='black')

centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', 
            s=400, edgecolors='black', linewidths=3, label='Centers')

plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.title('Mall Customer Segmentation', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nVị trí Cluster Centers:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: Income={center[0]:.1f}k$, Spending={center[1]:.1f}")


print(f"\nClusters: {data['Cluster'].value_counts().sort_index().to_dict()}")

