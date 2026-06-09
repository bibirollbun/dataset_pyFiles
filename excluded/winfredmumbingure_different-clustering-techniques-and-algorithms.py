import pandas as pd
df=pd.read_csv("/kaggle/input/tabular-playground-series-jul-2022/data.csv")
ss=pd.read_csv("/kaggle/input/tabular-playground-series-jul-2022/sample_submission.csv")


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(24,20)})
sns.heatmap(df.corr(),annot=True,fmt='.2f')


sns.set(rc={'figure.figsize':(15,15)})
for i, column in enumerate(list(df.columns), 1):
    plt.subplot(5,6,i)
    p=sns.histplot(x=column,data=df.sample(1000),stat='count',kde=True,color='green')


from numpy import unique, where
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

# Initialize the dataset
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# Define the KMeans model
kmeans_model = KMeans(n_clusters=2, random_state=4, n_init=10)

# Assign each data point to a cluster
kmeans_result = kmeans_model.fit_predict(training_data)

# Get all unique clusters
kmeans_clusters = unique(kmeans_result)

# Plot the KMeans clusters
for cluster in kmeans_clusters:
    # Get data points in the current cluster
    index = where(kmeans_result == cluster)
    # Plot the data points
    plt.scatter(training_data[index, 0], training_data[index, 1],label=f'Cluster {cluster}')

# Show the KMeans plot
plt.legend()
plt.title("KMeans Clustering")
plt.show()



from numpy import unique, where
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

# Initialize the dataset
training_data, _ = make_classification(
    n_samples=1000,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# Define the KMeans model
kmeans_model = KMeans(n_clusters=5, random_state=4, n_init=10)

# Assign each data point to a cluster
kmeans_result = kmeans_model.fit_predict(training_data)

# Get all unique clusters
kmeans_clusters = unique(kmeans_result)

# Plot the KMeans clusters
for cluster in kmeans_clusters:
    # Get data points in the current cluster
    index = where(kmeans_result == cluster)
    # Plot the data points
    plt.scatter(training_data[index, 0], training_data[index, 1],label=f'Cluster {cluster}')

# Show the KMeans plot
plt.legend()
plt.title("KMeans Clustering")
plt.show()



from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
dbscan_model = DBSCAN(eps=0.24, min_samples=9)

# train the model
dbscan_model.fit(training_data)

# assign each data point to a cluster
dbscan_result = dbscan_model.fit_predict(training_data)
# get all of the unique clusters
dbscan_clusters = unique(dbscan_result)

# plot the DBSCAN clusters
for dbscan_cluster in dbscan_clusters:
    # get data points that fall in this cluster
    index = where(dbscan_result == dbscan_cluster)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1], label=f'Cluster {dbscan_cluster}')

# show the DBSCAN plot
plt.title("DBSCAN Clustering")
plt.legend(loc="upper right")
plt.show()



from numpy import unique, where
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN

# Initialize the dataset
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# Define the DBSCAN model
dbscan_model = DBSCAN(eps=0.25, min_samples=9)

# Train the model
dbscan_model.fit(training_data)

# Assign each data point to a cluster
dbscan_result = dbscan_model.labels_  # Use `.labels_`, NOT `.predict()`

# Get all unique clusters (-1 represents noise)
dbscan_clusters = unique(dbscan_result)

# Plot the DBSCAN clusters
for dbscan_cluster in dbscan_clusters:
    # Get data points in this cluster
    index = where(dbscan_result == dbscan_cluster)
    
    # Choose color for noise points (-1)
    color = 'black' if dbscan_cluster == -1 else None  

    # Plot the data points
    plt.scatter(training_data[index, 0], training_data[index, 1], label=f'Cluster {dbscan_cluster}', color=color)

# Add title and legend
plt.title("DBSCAN Clustering")
plt.legend()
plt.show()



from numpy import unique, where
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture

# Initialize the dataset
training_data, _ = make_classification(
    n_samples=10000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# Define the Gaussian Mixture Model
gaussian_model = GaussianMixture(n_components=4, random_state=4)

# Train the model
gaussian_model.fit(training_data)

# Assign each data point to a cluster
gaussian_result = gaussian_model.predict(training_data)

# Get all unique clusters
gaussian_clusters = unique(gaussian_result)

# Plot the Gaussian Mixture clusters
for gaussian_cluster in gaussian_clusters:
    # Get data points in this cluster
    index = where(gaussian_result == gaussian_cluster)
    
    # Plot the data points
    plt.scatter(training_data[index, 0], training_data[index, 1], label=f'Cluster {gaussian_cluster}')

# Add title and legend
plt.title("Gaussian Mixture Model Clustering")
plt.legend()
plt.show()




# Import required libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch

# Generating 600 samples using make_blobs
dataset, _ = make_blobs(n_samples=600, centers=8, cluster_std=0.75, random_state=0)

# Creating the BIRCH clustering model
model = Birch(branching_factor=50, n_clusters=None, threshold=1.5)

# Fit the model (training)
model.fit(dataset)

# Get cluster labels (Birch does not have predict without n_clusters)
pred = model.labels_

# Creating a scatter plot
plt.scatter(dataset[:, 0], dataset[:, 1], c=pred, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title("BIRCH Clustering")
plt.show()



from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
model = AffinityPropagation(damping=0.7)

# train the model
model.fit(training_data)

# assign each data point to a cluster
result = model.predict(training_data)

# get all of the unique clusters
clusters = unique(result)

# plot the clusters
for cluster in clusters:
    # get data points that fall in this cluster
    index = where(result == cluster)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the plot
pyplot.show()


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import MeanShift

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
mean_model = MeanShift()

# assign each data point to a cluster
mean_result = mean_model.fit_predict(training_data)

# get all of the unique clusters
mean_clusters = unique(mean_result)

# plot Mean-Shift the clusters
for mean_cluster in mean_clusters:
    # get data points that fall in this cluster
    index = where(mean_result == mean_cluster)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the Mean-Shift plot
pyplot.show()


from numpy import unique, where
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import OPTICS

# Initialize the dataset
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# Define the OPTICS model
optics_model = OPTICS(min_samples=50)  # eps is optional for OPTICS

# Train the model and get cluster assignments
optics_result = optics_model.fit_predict(training_data)

# Get unique clusters
optics_clusters = unique(optics_result)  # Corrected this line

# Plot the OPTICS clusters
for optics_cluster in optics_clusters:
    # Get data points in this cluster
    index = where(optics_result == optics_cluster)
    
    # Plot the data points
    plt.scatter(training_data[index, 0], training_data[index, 1], label=f'Cluster {optics_cluster}')

# Add title and legend
plt.title("OPTICS Clustering")
plt.legend()
plt.show()



from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
agglomerative_model = AgglomerativeClustering(n_clusters=2)

# assign each data point to a cluster
agglomerative_result = agglomerative_model.fit_predict(training_data)

# get all of the unique clusters
agglomerative_clusters = unique(agglomerative_result)

# plot the clusters
for agglomerative_cluster in agglomerative_clusters:
    # get data points that fall in this cluster
    index = where(agglomerative_result == agglomerative_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the Agglomerative Hierarchy plot
pyplot.show()


from numpy import unique, where
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering

# Initialize the dataset
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# Define the Agglomerative Clustering model
agglomerative_model = AgglomerativeClustering(n_clusters=2)

# Train the model and get cluster assignments
agglomerative_result = agglomerative_model.fit_predict(training_data)

# Get unique clusters
agglomerative_clusters = unique(agglomerative_result)

# Plot the Agglomerative Clustering clusters
for agglomerative_cluster in agglomerative_clusters:
    # Get data points in this cluster
    index = where(agglomerative_result == agglomerative_cluster)
    
    # Plot the data points
    plt.scatter(training_data[index, 0], training_data[index, 1], label=f'Cluster {agglomerative_cluster}')

# Add title and legend
plt.title("Agglomerative Clustering")
plt.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs

# Generate sample data
np.random.seed(42)
X, _ = make_blobs(n_samples=100, centers=6, cluster_std=1.0)

# Perform hierarchical clustering using divisive method (DIANA)
Z = linkage(X, method='ward')  # 'ward' approximates divisive clustering

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("DIANA - Divisive Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Cluster Distance")
plt.show()

# Assign cluster labels (choose a cutoff distance)
max_d = 10  # Adjust based on dendrogram observation
clusters = fcluster(Z, max_d, criterion='distance')

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='rainbow', edgecolors='k')
plt.title("DIANA - Divisive Clustering")
plt.show()



"""
# Compute diana()
library(cluster)
res.diana <- diana(USArrests, stand = TRUE)

# Plot the dendrogram
library(factoextra)
fviz_dend(res.diana, cex = 0.5,
          k = 4, # Cut in four groups
          palette = "jco" # Color palette
          )

"""





pip install scikit-fuzzy




from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# Define three cluster centers
centers = [[4, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(500) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(500) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(500) * i))

# Visualize the test data
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.',
             color=colors[label])
ax0.set_title('Test data: 200 points x3 clusters.')


from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# Define three cluster centers
centers = [[4, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts, ypts, labels = [], [], []
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts.append(np.random.standard_normal(200) * xsigma + xmu)
    ypts.append(np.random.standard_normal(200) * ysigma + ymu)
    labels.append(np.ones(200) * i)

# Convert to NumPy arrays
xpts = np.concatenate(xpts)
ypts = np.concatenate(ypts)
labels = np.concatenate(labels).astype(int)

# Visualize the test data
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.scatter(xpts[labels == label], ypts[labels == label], c=colors[label], marker='.')
ax0.set_title('Test data: 200 points x3 clusters.')
plt.show()


