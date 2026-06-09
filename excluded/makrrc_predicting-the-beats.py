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


# !pip install --upgrade pip setuptools wheel


# !python -m pip install --upgrade pip


# !pip cache purge


!pip install --upgrade --quiet scipy scikit-learn


!pip install --upgrade --quiet numpy==1.26.4


# pip install --upgrade numpy scipy


# !pip uninstall scikit-learn
# !pip install scikit-learn


# !pip install pandas-profiling


# from ydata_profiling import ProfileReport


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


# from ydata_profiling import ProfileReport

# profile = ProfileReport(train, title="Relatório do Dataset", explorative=True)
# profile.to_file("relatorio.html")


# profile = ProfileReport(
#     train.sample(50000, random_state=42),  # usa só uma amostra
#     title="Relatório do Dataset (amostra)",
#     explorative=True
# )
# profile.to_notebook_iframe()


train.info()


train.isnull().sum()


train.describe()


train['BeatsPerMinute'].unique()


import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_normal_distribution(df, column, bins=50):
    """
    Plota a distribuição de uma coluna de um DataFrame 
    com a curva da Normal ajustada.

    Args:
        df (pd.DataFrame): dataset
        column (str): nome da coluna numérica
        bins (int): quantidade de intervalos do histograma
    """
    data = df[column].dropna()
    mu, std = data.mean(), data.std()

    # Histograma
    plt.figure(figsize=(8,5))
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black')

    # Curva normal ajustada
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2)

    plt.title(f"Distribuição de {column}\nMédia = {mu:.2f}, Desvio Padrão = {std:.2f}")
    plt.xlabel(column)
    plt.ylabel("Densidade")
    plt.show()


train.columns


for columns in train.columns:
    plot_normal_distribution(train, columns, bins=50)


from sklearn.decomposition import PCA


X = train
X


# Initialize PCA with 2 components
pca = PCA(n_components=2)

# Fit PCA to the data and transform it
X_pca = pca.fit_transform(X)

print("Original data shape:", X.shape)
print("Transformed data shape (PCA):", X_pca.shape)
print("Explained variance ratio of each component:", pca.explained_variance_ratio_)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt


# Assuming 'data' is your original DataFrame and 'target_column' is your class label
# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(train.drop(columns=['BeatsPerMinute']))

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Target'] = train['BeatsPerMinute'] # Add the target column for coloring

# Visualize using Seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Target', data=pca_df, palette='viridis')
plt.title('PCA of Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()


# Create a boxplot
plt.boxplot(train, labels=train.columns)
plt.title("Basic Boxplot")
plt.ylabel("Value")
plt.xticks(rotation=90)
plt.show()


len(train.columns)


train


sns.heatmap(train.corr())


from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
print(scaler.fit(train))


scaled_data = scaler.fit_transform(train)


scaled_data_test = scaler.fit_transform(test)


scaled_data


scaled_data = pd.DataFrame(scaled_data)
scaled_data


scaled_data_test = pd.DataFrame(scaled_data_test)
scaled_data_test


# scaled_data = scaled_data.drop(columns='cluster')
# scaled_data


sns.heatmap(scaled_data.corr())


sns.heatmap(train.corr())


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs




# Dados de exemplo

sse = []
k_values = range(1, 10)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(train)
    sse.append(km.inertia_)  # SSE (inertia)

plt.plot(k_values, sse, 'bo-')
plt.xlabel('Número de clusters (k)')
plt.ylabel('SSE')
plt.title('Método do Cotovelo')
plt.show()


# Create a KMeans object with 2 clusters
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)


# Fit the model to the data
kmeans.fit(train)


# Get cluster labels for each data point
scaled_data['cluster'] = kmeans.fit_predict(scaled_data)


# Get cluster labels for each data point
scaled_data_test['cluster'] = kmeans.fit_predict(scaled_data_test)


scaled_data.dtypes


# scaled_data['cluster'] = kmeans.fit_predict(scaled_data)


train


print(f"Cluster labels: {scaled_data['cluster']}")


print(f"Cluster labels: {scaled_data_test['cluster']}")


scaled_data


scaled_data.cluster.unique()


train['cluster'] = scaled_data.cluster.values


train


test['cluster'] = scaled_data_test.cluster.values


test


sns.heatmap(train.corr())


train


# --- Gráfico de dispersão com cores para clusters ---
plt.figure(figsize=(8,6))
for c in train['cluster'].unique():
    cluster_points = train[train['cluster'] == c]
    plt.scatter(cluster_points['RhythmScore'], cluster_points['Energy'], label=f'Cluster {c}', alpha=0.6)

plt.xlabel('RhythmScore')
plt.ylabel('Energy')
plt.title('Visualização de Clusters K-means')
plt.legend()
plt.grid(True)
plt.show()


train.isnull().sum()


train


train['MoodScore']


sns.pairplot(train)


train['RhythmScore'].unique().size


test.to_csv('/kaggle/working/test_eda_v1.csv')


train.to_csv('/kaggle/working/eda_v1.csv')

