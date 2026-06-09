import pandas as pd
import pyarrow.parquet as pq
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 


# Define the directory where your training and test parquet files are located
train_parquet_dir = '/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet/'
test_parquet_dir = '/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet/'

# Load your training data to extract mean values
train_df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')

# Initialize a DataFrame to store mean values
mean_values_list = []


for idx, row in train_df.iterrows():
    file_id = row['id']
    parquet_file_path = os.path.join(train_parquet_dir, f"id={file_id}", "part-0.parquet")

    if os.path.exists(parquet_file_path): 
        data = pq.read_table(parquet_file_path).to_pandas()

        feature_columns = [col for col in data.columns if col not in ['id', 'sii']]

        if feature_columns: 
            mean_values = data[feature_columns].mean().values
            mean_values_list.append(mean_values)


mean_values_df = pd.DataFrame(mean_values_list)
mean_values_df.head()


#sum of squared errors 
sse = []
k_range = range(1, 11) 

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(mean_values_df)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.xticks(k_range)
plt.grid()
plt.show()


optimal_k = 2 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(mean_values_df)

mean_values_df['Cluster'] = clusters

pca = PCA(n_components=2)
mean_values_reduced = pca.fit_transform(mean_values_df.drop('Cluster', axis=1))




mean_values_df_1 = pd.DataFrame(mean_values_reduced)
mean_values_df_1.head(10)


plt.figure(figsize=(10, 6))
plt.scatter(mean_values_reduced[:, 0], mean_values_reduced[:, 1], c=mean_values_df['Cluster'], cmap='viridis', alpha=0.6)
plt.title('K-Means Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.grid()
plt.show()


# Load test.csv to get ALL test IDs
test_df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
all_test_ids = test_df['id'].tolist()  # List of all test IDs from test.csv


#ensure PCA is fitted before clustering 
pca = PCA(n_components=2) # Set to 2 to be equal as the PCA value used in training
mean_values_pca = pca.fit_transform(mean_values_df.drop('Cluster', axis=1)) # Fit PCA on training features


# Fit K-Means on the PCA-reduced training data
optimal_k = 2 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(mean_values_pca)  # Now K-Means is trained on the reduced feature space


submission_data = []


# Loop through all test IDs
for test_file_id in all_test_ids:
    parquet_file_path = os.path.join(test_parquet_dir, f"id={test_file_id}", "part-0.parquet")

    # Check if the test parquet file exists
    if os.path.exists(parquet_file_path):
        # Load test parquet data
        test_data = pq.read_table(parquet_file_path).to_pandas()

        # Extract feature columns (excluding 'id')
        feature_columns = [col for col in test_data.columns if col not in ['id']]

        if feature_columns:  # Ensure there are feature columns
            # Compute mean values for features
            test_mean_values = test_data[feature_columns].mean().values

            #Apply the same PCA transformations you did in training
            test_mean_values_pca = pca.transform([test_mean_values])  # Transform test data to 2D
            
            # Predict the cluster label using K-Means
            cluster_label = kmeans.predict(test_mean_values_pca)[0]

        else: 
            cluster_label = np.nan  # Assign NaN if no valid features
    else:
        # If the Parquet file doesn’t exist, assigns the most frequent cluster (mode() of the training clusters).
        cluster_label = mean_values_df['Cluster'].mode()[0]# 

    #Append the prediction 
    submission_data.append([test_file_id, cluster_label])  # Avoid empty rows, replace later



# Convert results into a DataFrame
submission_df = pd.DataFrame(submission_data, columns=['id', 'sii'])

# Handle missing values (if any)
#submission_df['sii'].fillna(submission_df['sii'].mode()[0], inplace=True)  # Replace NaNs with most common cluster
submission_df['sii'] = submission_df['sii'].fillna(submission_df['sii'].mode()[0])
# Save the submission file
submission_file_path = 'submission.csv'
submission_df.to_csv(submission_file_path, index=False)

print("✅ Submission file created:", submission_file_path)




