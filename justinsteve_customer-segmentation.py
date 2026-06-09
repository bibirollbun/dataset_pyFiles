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


!pip install scikit-learn-extra

import pandas as pd
import numpy as np

# libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# libraries for scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# libraries for clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/classifying-customers-into-segments/Train.csv')
train.head()


train.shape


# Checking our datatypes
train.info()


# Looking at summary statistics
train.describe().T


# checking for missing data
train.isna().sum()


train.nunique()


# looking at our categorical columns
cat_columns = train.select_dtypes(include =['object'])
cat_columns.describe().T


#Checking values in the categorical columns
for col in cat_columns:
    print(f"Unique values in '{col}'':")
    print('*' * 50)
    print(train[col].unique())
    print()


train = train.dropna(subset=['Profession'])
train = train.dropna(subset=['Var_1'])


# checking for duplicated entries
train.duplicated().sum()


# addressing missing values in binary columns (Ever_Married, Graduated)
train['Ever_Married'] = train['Ever_Married'].fillna("No")
train['Graduated'] = train['Graduated'].fillna('No')


#Rechecking values in the categorical columns
for col in cat_columns:
    print(f"Unique values in '{col}'':")
    print('*' * 50)
    print(train[col].unique())
    print()



# Dropping NaN values to prevent biasing the dataset with imputation

train.dropna()


train.isna().sum()


train=train.dropna(subset=['Work_Experience'])
train=train.dropna(subset=['Family_Size'])


train.isna().sum()


#Checking the data distribution of each feature

num_data=train.select_dtypes(include=['number'])

for col in num_data.columns:
    print(col)
    print("Skew:", round(num_data[col].skew(), 2))
    plt.figure(figsize=(15,4))
    plt.subplot(1,2,1)
    num_data[col].hist(bins=10, grid=False)
    plt.ylabel('count')
    plt.subplot(1,2,2)
    sns.boxplot(x=num_data[col])
    plt.show()


# Checking for correlations between features

sns.heatmap(num_data.corr(), annot=True, cmap='viridis')


!pip install kmodes


from kmodes.kprototypes import KPrototypes

categorical=['Gender', "Ever_Married",'Graduated','Profession','Spending_Score','Var_1']
numeric=['ID','Age','Work_Experience','Family_Size']

cat_data=train[categorical].astype(str)
number_data = train[numeric].astype(float)

combo_data=np.hstack([number_data.values, cat_data.values])
cat_indicies = list(range(len(numeric), combo_data.shape[1]))

# Fit K-Prototypes model
kproto = KPrototypes(n_clusters=4, init='Huang', random_state=42)
clusters=kproto.fit_predict(combo_data, categorical=cat_indicies)

train['cluster'] = clusters


# viewing our clusters
from sklearn.decomposition import PCA

# Encoding our data
if isinstance(cat_data, pd.DataFrame):
    cat_data = cat_data.values
encoded_data = np.array([np.unique(cat_data[:,i], return_inverse=True)[1] for i in range(cat_data.shape[1])]).T

pca_input = np.hstack([number_data, encoded_data])
pca=PCA(n_components=2)
pca_result = pca.fit_transform(pca_input)

# Add PCA results back to Dataframe
train['pca1']=pca_result[:,0]
train['pca2']=pca_result[:,1]

# Plotting the clusters
plt.figure(figsize=(10,8))
sns.scatterplot(
    x='pca1',
    y='pca2',
    hue='cluster',
    palette='viridis',
    data=train,
    s=100
)

plt.title("Cluster Visualization")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.legend(title='Cluster')
plt.show()


# Comparing PCA with T-SNE dimensionality reduction

from sklearn.manifold import TSNE

tsne=TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_result = tsne.fit_transform(pca_input)

# Cluster Visualization

plt.figure(figsize=(10,7))
scatter=plt.scatter(tsne_result[:,0], tsne_result[:,1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title("T-SNE Visualization of Clusters")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")
plt.grid(True)
plt.show()


# Evaluating Silhouette Score of Dimensional Reduction Techniques

cluster_labels = kproto.labels_

silhouette_pca = silhouette_score(pca_result, cluster_labels)
print("Silhouette Score for PCA: ", silhouette_pca)

silhouette_tsne = silhouette_score (tsne_result, cluster_labels)
print("Silhouette Score for TSNE: ", silhouette_tsne)


test=pd.read_csv('/kaggle/input/classifying-customers-into-segments/Test.csv')
test.head()


# Checking for missing values
test.isna().sum()


test.duplicated().sum()


# Checking summary statistics

test.describe().T


# Treating missing values

test['Ever_Married'] = test['Ever_Married'].fillna("No")
test['Graduated']=test['Graduated'].fillna("No")
test['Work_Experience']=test['Work_Experience'].fillna(test['Work_Experience'].mean())
test['Family_Size']=test['Family_Size'].fillna(test['Family_Size'].mean())
test['Profession'] = test['Profession'].fillna("Unknown")
test['Var_1'] = test['Var_1'].fillna("Unknown")



test.isna().sum()


# Using our Trained K-Prototype model to predict on our Test Set

categorical=['Gender', "Ever_Married",'Graduated','Profession','Spending_Score','Var_1']
numeric=['ID','Age','Work_Experience','Family_Size']

test_cat_data=test[categorical].astype(str)
test_number_data = test[numeric].astype(float)

test_combo_data=np.hstack([test_number_data.values, test_cat_data.values])
cat_indicies = list(range(len(numeric), test_combo_data.shape[1]))

# Predict using K-Prototypes model

test_clusters=kproto.predict(test_combo_data, categorical=cat_indicies)

test['cluster'] = test_clusters

print(test.head())


# viewing our  Testclusters


# Encoding our data
if isinstance(test_cat_data, pd.DataFrame):
    test_cat_data = test_cat_data.values
test_encoded_data = np.array([np.unique(test_cat_data[:,i], return_inverse=True)[1] for i in range(test_cat_data.shape[1])]).T

test_pca_input = np.hstack([test_number_data, test_encoded_data])
pca=PCA(n_components=2)
test_pca_result = pca.fit_transform(test_pca_input)

# Add PCA results back to Dataframe
test['pca1']=test_pca_result[:,0]
test['pca2']=test_pca_result[:,1]

# Plotting the clusters
plt.figure(figsize=(10,8))
sns.scatterplot(
    x='pca1',
    y='pca2',
    hue='cluster',
    palette='viridis',
    data=test,
    s=100
)

plt.title("Cluster Visualization")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.legend(title='Cluster')
plt.show()


# Evaluating our model on the Test set

cat_indicies = list(range(len(numeric), test_combo_data.shape[1]))

test_clusters = kproto.predict(test_combo_data, categorical =cat_indicies)

test_silhouette_pca = silhouette_score(test_pca_result, test_clusters)
print("Silhouette Score for PCA: ", test_silhouette_pca)


test['cluster'].unique()


# replacing cluster values with the correct nomenclature
values = {0:'A', 1: 'B', 2: 'C', 3: 'D'}
test['Segmentation']=test['cluster'].replace(values)


test.head()


submission = test.drop(columns=['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size','Var_1','cluster','pca1','pca2'])


submission.head()


submission.to_csv('submission.csv')

