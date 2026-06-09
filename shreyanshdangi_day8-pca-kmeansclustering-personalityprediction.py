import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


df.info()


df.shape


df['Personality'].value_counts()


df.head(5)


print(df['Stage_fear'].unique())
print(df['Drained_after_socializing'].unique())


mapping = {'Yes': 1, 'No': 0}
df['Stage_fear'] = df['Stage_fear'].map(mapping)

mapping1 = {'Yes': 1, 'No': 0}
df['Drained_after_socializing'] = df['Drained_after_socializing'].map(mapping1)

mapping2 = {'Extrovert': 1, 'Introvert': 0}
df['Personality'] = df['Personality'].map(mapping2)


df.head(5)


df.info()


df.isnull().sum()


from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

num_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing',
            'Friends_circle_size', 'Post_frequency']

imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


df.isnull().sum()


import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

X = df[num_cols]

def objective_pca(trial):
    n_components = trial.suggest_int('n_components', 2, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    n_clusters = trial.suggest_int('n_clusters', 2, 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    if len(set(labels)) == 1:
        return -1.0

    score = silhouette_score(X_pca, labels)
    return score

study_pca = optuna.create_study(direction='maximize')
study_pca.optimize(objective_pca, n_trials=30)

best_params = study_pca.best_params
best_score = study_pca.best_value

print(f"\n✅ Best PCA Components: {best_params['n_components']}")
print(f"✅ Best KMeans Clusters: {best_params['n_clusters']}")
print(f"✅ Best Silhouette Score: {best_score:.4f}")

pca_final = PCA(n_components=best_params['n_components'], random_state=42)
X_pca_final = pca_final.fit_transform(X)

kmeans_final = KMeans(n_clusters=best_params['n_clusters'], random_state=42, n_init=10)
df['KMeans_PCA_Cluster'] = kmeans_final.fit_predict(X_pca_final)

try:
    import optuna.visualization as vis
    vis.plot_optimization_history(study_pca).show()
    vis.plot_param_importances(study_pca).show()
except:
    print("Optuna visualization not available in this environment.")

plt.figure(figsize=(8,5))
plt.bar(range(1, best_params['n_components']+1), 
        pca_final.explained_variance_ratio_ * 100, color='teal')
plt.title('PCA Explained Variance (%) per Component')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(X_pca_final[:,0], X_pca_final[:,1],
            c=df['KMeans_PCA_Cluster'], cmap='viridis', s=15)
plt.title(f"PCA (Optuna optimized: {best_params['n_components']} components, {best_params['n_clusters']} clusters)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label='Cluster ID')
plt.show()

if best_params['n_components'] >= 3:
    fig = px.scatter_3d(
        x=X_pca_final[:,0],
        y=X_pca_final[:,1],
        z=X_pca_final[:,2],
        color=df['KMeans_PCA_Cluster'].astype(str),
        title='3D PCA Cluster Visualization',
        labels={'x': 'PCA 1', 'y': 'PCA 2', 'z': 'PCA 3'}
    )
    fig.show()

silhouette_vals = silhouette_samples(X_pca_final, df['KMeans_PCA_Cluster'])
y_lower = 10
plt.figure(figsize=(8,6))
for i in range(best_params['n_clusters']):
    cluster_silhouette = silhouette_vals[df['KMeans_PCA_Cluster'] == i]
    cluster_silhouette.sort()
    y_upper = y_lower + len(cluster_silhouette)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette)
    plt.text(-0.05, y_lower + 0.5 * len(cluster_silhouette), str(i))
    y_lower = y_upper + 10
plt.axvline(x=np.mean(silhouette_vals), color='red', linestyle='--')
plt.title('Silhouette Plot per Cluster')
plt.xlabel('Silhouette Coefficient')
plt.ylabel('Cluster Label')
plt.show()

cluster_means = df.groupby('KMeans_PCA_Cluster')[num_cols].mean()
plt.figure(figsize=(10,6))
sns.heatmap(cluster_means, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Cluster Feature Means (Scaled Values)')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='KMeans_PCA_Cluster', hue='Personality', data=df, palette='Set2')
plt.title('Personality Distribution across Clusters')
plt.show()

