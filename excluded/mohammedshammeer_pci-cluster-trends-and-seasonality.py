import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose


import warnings
warnings.filterwarnings('ignore')


# Load the dataset
train = pd.read_csv(r'/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv', index_col="id", parse_dates=["sale_date"])
train.head()


train_ids = train.index
sale_date = train.pop('sale_date')
sale_price = train.pop('sale_price')

# To avoid higher dimensions
train.drop(["zoning", "sale_warning", "subdivision"], axis=1, inplace=True)

# Fill missing values
train['sale_nbr'].fillna(train['sale_nbr'].median(), inplace=True)
# train['subdivision'].fillna(train['subdivision'].mode()[0], inplace=True)
train['submarket'].fillna(train['submarket'].mode()[0], inplace=True)

# Identify numeric and categorical columns
num_cols_data = train.select_dtypes(include=np.number).columns.tolist()
cat_cols_data = train.select_dtypes(include='object').columns.tolist()


train = pd.get_dummies(train, columns=cat_cols_data, drop_first=True)
def scaler(df, columns):
    scaled = {}
    for col in columns:
        std = StandardScaler()
        df[col] = std.fit_transform(df[[col]])
        scaled[col] = std
    return df, scaled

train, scaled = scaler(train, num_cols_data)


pca = PCA(n_components=2)
train_pca = pca.fit_transform(train)

print(f"\nShape of PCA-transformed data: {train_pca.shape}")


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(train_pca)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()


n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters, n_init=10)
train['cluster'] = kmeans.fit_predict(train_pca)

print(f"\nNumber of samples per cluster:\n{train['cluster'].value_counts()}")


scatter = plt.scatter(train_pca[:,0],train_pca[:,1],c = train['cluster'])
plt.legend(*scatter.legend_elements(), loc="upper right", title="clusters")
plt.title("Cluster Analysis for PCA Data")
plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')
plt.show()


print("\nPerforming Seasonality Analysis for Each Cluster...")

train["sale_price"] = sale_price
train["sale_date"] = sale_date

for cluster_id in sorted(train['cluster'].unique()):
    print(f"\n--- Analyzing Cluster {cluster_id} ---")
    cluster_df = train[train['cluster'] == cluster_id].copy()

    # Ensure sale_date is datetime and set as index
    cluster_df['sale_date'] = pd.to_datetime(cluster_df['sale_date'])
    cluster_df = cluster_df.sort_values(by='sale_date')
    cluster_df.set_index('sale_date', inplace=True)

    monthly_sales = cluster_df['sale_price'].resample('M').sum()

    analysis = seasonal_decompose(monthly_sales, period=12)

    # Plotting
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.5)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :2])
    ax3 = fig.add_subplot(gs[1, 2:])

    # Plot monthly sales
    sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, ax=ax1)
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax1.set_title(f'Monthly Sale Price for Cluster {cluster_id}')

    # Plot trend and seasonal components
    sns.lineplot(x=analysis.trend.index, y=analysis.trend.values, ax=ax2)
    ax2.set_title(f'Trend Component for Cluster {cluster_id}')

    sns.lineplot(x=analysis.seasonal.index, y=analysis.seasonal.values, ax=ax3)
    ax3.set_title(f'Seasonal Component for Cluster {cluster_id}')

    plt.show()

