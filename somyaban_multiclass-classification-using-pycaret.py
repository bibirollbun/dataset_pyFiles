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


!pip install pycaret


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pycaret.clustering import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
import joblib
warnings.filterwarnings("ignore")
# Directory to store models
MODEL_DIR = "cluster_models"
os.makedirs(MODEL_DIR, exist_ok=True)
import pickle


train=pd.read_csv('/kaggle/input/thapar-kaggle-hack-v-01/X_train.csv')
test = pd.read_csv("/kaggle/input/thapar-kaggle-hack-v-01/X_test.csv")


print("Train Data Info:")
print(train.info())
print("\nTest Data Info:")
print(test.info())


# Check for missing values
print("\nMissing Values in Train Set:\n", train.isnull().sum())
print("\nMissing Values in Test Set:\n", test.isnull().sum())


# Display summary statistics
print("\nTrain Data Statistics:")
print(train.describe())


# Visualize target distribution
plt.figure(figsize=(8, 4))
sns.countplot(x="target", data=train)
plt.title("Target Distribution")
plt.show()


# Fill missing values
for col in train.columns:
    if train[col].dtype == 'object':  # Handle categorical columns
        mode_value = train[col].mode()[0]
        train[col].fillna(mode_value, inplace=True)
        if col in test.columns:  # Ensure the column exists in test before filling
            test[col].fillna(mode_value, inplace=True)
    else:  # Handle numerical columns
        median_value = train[col].median()
        train[col].fillna(median_value, inplace=True)
        if col in test.columns:  # Ensure the column exists in test before filling
            test[col].fillna(median_value, inplace=True)

# Encoding categorical features
for col in train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Standardize numerical features
scaler = StandardScaler()
numeric_cols = train.select_dtypes(include=[np.number]).columns.drop("target")
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])


# Remove the target column for clustering
clustering_data = train.drop(columns=['target'])


cluster_setup = setup(clustering_data, normalize=True, session_id=42)
x = create_model('kmeans')
plot_model(x, plot = 'elbow')


# Initialize PyCaret clustering
def run_clustering(data, num_clusters=5):
    print(f"Running clustering for {num_clusters} clusters")
    cluster_setup = setup(data, normalize=True, session_id=42)
    kmeans = create_model('kmeans', num_clusters=num_clusters)
    return kmeans

# Find optimal clusters
wcss = []
k_values = range(2, 13)
for i in k_values:
    model = run_clustering(clustering_data, num_clusters=i)
    wcss.append(model.inertia_)
    print(f"Cluster {i}: WCSS = {model.inertia_}")

# Plot Distortion Score
plt.figure(figsize=(10, 5))
plt.plot(k_values, wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion Score (WCSS)')
plt.title('Distortion Score vs. Number of Clusters')
plt.show()


cluster_setup = setup(clustering_data, normalize=True, session_id=42)
x = create_model('kmeans', num_clusters=5)
#plot_model(x, plot = 'tsne')
# Re-run the code again for different parameters
# normalize_method = {zscore, minmax, maxabs, robust}


cluster_setup = setup(clustering_data, transformation = True, session_id=42)
x = create_model('kmeans')
plot_model(x, plot = 'elbow')
# transformation_method = {yeo-johnson, quantile}


cluster_setup = setup(clustering_data, transformation = True, normalize=True, session_id=42)

print("For Cluster = 4")
x = create_model('kmeans', num_clusters = 4)

print("For Cluster = 5")
x = create_model('kmeans', num_clusters = 5)

print("For Cluster = 6")
x = create_model('kmeans', num_clusters = 6)

print("For Cluster = 7")
x = create_model('kmeans', num_clusters = 7)

print("For Cluster = 8")
x = create_model('kmeans', num_clusters = 8)


cluster_setup = setup(clustering_data, pca = True, session_id=42)
x = create_model('kmeans')
plot_model(x, plot = 'elbow')
# pca_method = {linear, kernel, incremental}


cluster_setup = setup(clustering_data, pca = True, session_id=42)

print("For Cluster = 4")
x = create_model('kmeans', num_clusters = 4)

print("For Cluster = 5")
x = create_model('kmeans', num_clusters = 5)

print("For Cluster = 6")
x = create_model('kmeans', num_clusters = 6)

print("For Cluster = 7")
x = create_model('kmeans', num_clusters = 7)

print("For Cluster = 8")
x = create_model('kmeans', num_clusters = 8)


cluster_setup = setup(clustering_data, normalize= True, pca = True, session_id=42)
x = create_model('kmeans')
plot_model(x, plot = 'elbow')


cluster_setup = setup(clustering_data, normalize= True, pca = True, session_id=42)

print("For Cluster = 4")
x = create_model('kmeans', num_clusters = 4)

print("For Cluster = 5")
x = create_model('kmeans', num_clusters = 5)

print("For Cluster = 6")
x = create_model('kmeans', num_clusters = 6)

print("For Cluster = 7")
x = create_model('kmeans', num_clusters = 7)

print("For Cluster = 8")
x = create_model('kmeans', num_clusters = 8)


cluster_setup = setup(clustering_data, normalize= True, pca= True, session_id=42)

x = create_model('hclust',num_clusters=4)
#plot_model(x, plot = 'elbow')


# Initialize PyCaret clustering
def run_clustering(data, num_clusters=4):
    print(f"Running clustering for {num_clusters} clusters")
    cluster_setup = setup(data, normalize=True, pca=True, session_id=42)
    model = create_model('hclust', num_clusters=num_clusters)
    return model

model = run_clustering(clustering_data, num_clusters=4)

# # Find optimal clusters
# wcss = []
# values = range(3,6)
# for i in values:
#     model = run_clustering(clustering_data, num_clusters=i)

# Plot Distortion Score using PyCaret
#plot_model(model, plot='elbow')


# Assign clusters to train data
train['Cluster'] = run_clustering(clustering_data, num_clusters=4).labels_
test['Cluster'] = run_clustering(test, num_clusters=4).labels_


from pycaret.classification import *
# Dictionary to store models for each cluster
cluster_models = {}

# Train separate models for each cluster
unique_clusters = train['Cluster'].unique()

print(f"Number of clusters: {len(unique_clusters)}")
print("Training separate models for each cluster...")

selected_models = ['lightgbm', 'knn', 'qda', 'rf', 'et', 'dt']

def train_cluster_specific_model(cluster_data, cluster_num):
    """
    Train a classification model for a specific cluster
    """
    print(f"\nTraining model for Cluster {cluster_num}")
    print(f"Cluster size: {len(cluster_data)}")
    
    # Setup PyCaret classification environment
    clf_setup = setup(
    data=cluster_data,
    target='target',
    session_id=42,
    use_gpu=True,  
    normalize=False,
    transformation=False,
    ignore_features=['Cluster'],
    n_jobs=-1 
     )
    
    # Compare models and get the best one from the selected list
    best_model = compare_models(n_select=1, include=selected_models, turbo=True)
    
    # Tune the best model
    #tuned_model = tune_model(best_model)
    
    # Finalize model
    final_model = finalize_model(best_model)

    # Save model
    with open(f'cluster_model_{cluster_num}.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    
    return final_model

for cluster in unique_clusters:
    cluster_data = train[train['Cluster'] == cluster].copy()
    
    if len(cluster_data) < 10:
        print(f"Warning: Cluster {cluster} has only {len(cluster_data)} samples")
        continue
        
    try:
        cluster_models[cluster] = train_cluster_specific_model(cluster_data, cluster)
        print(f"Successfully trained model for cluster {cluster}")
    except Exception as e:
        print(f"Error training model for cluster {cluster}: {str(e)}")

print("\nAll models have been trained and saved successfully.")


# Load trained models
def load_models():
    loaded_models = {}
    for cluster in unique_clusters:
        try:
            with open(f'cluster_model_{cluster}.pkl', 'rb') as f:
                loaded_models[cluster] = pickle.load(f)
        except FileNotFoundError:
            print(f"Model for cluster {cluster} not found.")
    return loaded_models

# Predict on test set
loaded_models = load_models()
test['Prediction'] = np.nan  # Initialize predictions as NaN

def predict_for_test(row):
    cluster = row['Cluster']
    if cluster in loaded_models:
        model = loaded_models[cluster]
        features = row.drop(['Cluster'])
        return predict_model(model, data=pd.DataFrame([features]))['prediction_label'].values[0]
    else:
        return np.nan  # Assign NaN if model is missing (e.g., Cluster 2)

test['Prediction'] = test.apply(predict_for_test, axis=1)




