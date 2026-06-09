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
# !pip install --upgrade --quiet scipy scikit-learn
# !pip install --upgrade --quiet numpy==1.26.4

# pip install --upgrade numpy scipy
# !pip uninstall scikit-learn
# !pip install scikit-learn
# !pip install pandas-profiling


# from ydata_profiling import ProfileReport


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


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


train['accident_risk'].unique()


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


# for columns in train.columns:
#     plot_normal_distribution(train, columns, bins=50)


from sklearn.decomposition import PCA


train.columns[ train.dtypes == object ]


train[ train.columns[ train.dtypes == object ] ]['road_type'].unique()


pd.get_dummies(train[ 'road_type' ])


type(train)


train = pd.concat( [ train , pd.get_dummies(train[ [ 'road_type', 'lighting', 'weather', 'time_of_day' ] ]) ], axis=1 )


test = pd.concat( [ test , pd.get_dummies(test[ [ 'road_type', 'lighting', 'weather', 'time_of_day' ] ]) ], axis=1 )


train.columns


train.loc[:,train.dtypes=='object'].columns


train = train.drop(train.loc[:,train.dtypes=='object'].columns, axis=1)
train


test = test.drop(test.loc[:,test.dtypes=='object'].columns, axis=1)
test


train.dtypes == 'bool'


train.shape


train[ train.loc[:,train.dtypes=='bool'].columns ] = train[ train.loc[:,train.dtypes=='bool'].columns ].replace({True:1, False:0})
train


test[ test.loc[:,test.dtypes=='bool'].columns ] = test[ test.loc[:,test.dtypes=='bool'].columns ].replace({True:1, False:0})
test


train.shape


X = train[train.columns]
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
scaled_data = scaler.fit_transform(train.drop(columns=['accident_risk']))

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Target'] = train['accident_risk'] # Add the target column for coloring

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


train


len(train.columns)


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


scaled_data_test.columns


scaled_data.columns


# Get cluster labels for each data point
scaled_data['cluster'] = kmeans.fit_predict(scaled_data)


scaled_data = scaled_data.drop(columns='cluster')
scaled_data


# Get cluster labels for each data point
scaled_data_test['cluster'] = kmeans.fit_predict(scaled_data_test)


scaled_data.dtypes


scaled_data['cluster'] = kmeans.fit_predict(scaled_data)


print(f"Cluster labels: {scaled_data['cluster']}")


print(f"Cluster labels: {scaled_data_test['cluster']}")


scaled_data.cluster.unique()


train['cluster'] = scaled_data.cluster.values


test['cluster'] = scaled_data_test.cluster.values


# sns.heatmap(train.corr())


# --- Gráfico de dispersão com cores para clusters ---
plt.figure(figsize=(8,6))
for c in train['cluster'].unique():
    cluster_points = train[train['cluster'] == c]
    plt.scatter(cluster_points['accident_risk'], cluster_points['num_lanes'], label=f'Cluster {c}', alpha=0.6)

plt.xlabel('accident_risk')
plt.ylabel('num_lanes')
plt.title('Visualização de Clusters K-means')
plt.legend()
plt.grid(True)
plt.show()


train.isnull().sum()


train['num_lanes']


# sns.pairplot(train)


train['accident_risk'].unique().size


train.columns


test.columns


train.shape


test.shape


from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split


y=train['accident_risk']
y


X = train.iloc[:, 1:len(train.columns)]


train['id']


X = X.drop('accident_risk', axis='columns')


X


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


X_train.columns


X_test.columns


from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression


from sklearn.linear_model import LinearRegression


model = LinearRegression()


model.fit(X_train,y_train)


result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)


# 📈 Criar DataFrame ordenado pela importância
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean
}).sort_values(by="Importance", ascending=False)


# 🖼️ Plotar gráfico de barras
plt.figure(figsize=(8, 5))
plt.barh(importances["Feature"], importances["Importance"], color="royalblue")
plt.xlabel("Importância média (Permutation Importance)")
plt.ylabel("Variáveis (Features)")
plt.title("Importância das Features — Linear Regression")
plt.gca().invert_yaxis()  # Inverter eixo para mostrar a mais importante no topo
plt.grid(alpha=0.3)
plt.show()

# 📋 Exibir tabela
print(importances)


test = test[['id','curvature','speed_limit','lighting_night','lighting_dim','lighting_daylight','weather_clear','weather_foggy','weather_rainy','num_reported_accidents']]


train = train[['id','curvature','speed_limit','lighting_night','lighting_dim','lighting_daylight','weather_clear','weather_foggy','weather_rainy','num_reported_accidents', 'accident_risk']]


test.to_csv('/kaggle/working/test_eda_predict_road_accident_risk_v2_feature_importance.csv')


train.to_csv('/kaggle/working/eda_predict_road_accident_risk_v2_feature_importance.csv')

