import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os 
import warnings
warnings.filterwarnings('ignore')


# Load the data
print("Loading training data...")
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
print(f"Training data info: {train_df.info}")



spread = train_df['bid_qty'] - train_df['ask_qty']
plt.figure(figsize = (20, 20))
plt.subplot(3, 1, 1)
plt.plot(spread)
plt.subplot(3, 1, 2)
plt.plot(train_df['label'])
sns.displot(train_df['label'], kind='kde')


# Lag features from public quantities
extras = []
for col in ["bid_qty", "ask_qty", "buy_qty", "sell_qty"]:
    train_df[f"{col}_lag1"] = train_df[col].shift(1)
    extras.append(f"{col}_lag1")

# ðŸš« Drop rows with NaNs from lag/indicators
train_df.dropna(inplace=True)

common_features = ['X175', 'X179', 'X137', 'X197', 'X22', 'X40', 'X181', 
                   'X28', 'X169', 'X198', 'X173', 'X21', 'X752', 'bid_qty',
                    "X598", "X385", "X674",
                    "X415", "X345", "X174", "X302", "X178", "X168", "X612",
                  'ask_qty', 'buy_qty', 'sell_qty'] + extras #taken from https://www.kaggle.com/code/ahsuna123/anonymized-features-importance-selection-eda
fig = plt.figure(figsize=(20, 5))
plt.subplot(1,3,1)
plt.title("correlation of common eda features")
sns.heatmap(train_df[common_features].corr(), annot=False)
plt.subplot(1,3,2) 
sns.heatmap(train_df[common_features].cov(), annot=False)
plt.title("covariance of common eda features")
plt.subplot(1,3,3)
plt.bar(common_features, height=train_df[common_features].var(axis=0))
plt.title('variance of common eda features')
plt.xticks(rotation=45)




"""from statsmodels.tsa.stattools import adfuller
season_length = len(train_df['label'])/2
values = [train_df.reset_index().loc[x*season_length:season_length+x*season_length, 'label'].values for x in range(2)]
print(values)
for i in values:
    res = adfuller(i)
    
    # Printing the statistical result of the adfuller test
    print('Augmneted Dickey_fuller Statistic: %f' % res[0])
    print('p-value: %f' % res[1])
    
    # printing the critical values at different alpha levels.
    print('critical values at different levels:')
    for k, v in res[4].items():
        print('\t%s: %.3f' % (k, v))
"""


import xgboost as xgb
import gc
from sklearn.model_selection import GridSearchCV
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
xgb_label = train_df['label']
xgb_data = train_df.drop('label', axis=1)[common_features]
scaler = StandardScaler()

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from itertools import product

xgb_label = train_df['label']
xgb_data = train_df.drop('label', axis=1)[common_features]
scaler = StandardScaler()

def xgb_model_selection(xgb_data, xgb_label, scaler):
    param_grid = {
        'max_depth': [2, 4, 6],
        'n_estimators': [50, 100, 200],
        'min_child_weight': [2.5, 5, 7.5],
        'gamma': [0.5, 1.0, 1.5],
        'reg_alpha': [50, 100, 200],
        'reg_lambda': [50, 100, 200],
    }

    all_param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    best_score = -np.inf
    best_params = None

    tscv = TimeSeriesSplit(n_splits=5)

    for combo in all_param_combinations:
        param_dict = dict(zip(param_names, combo))
        scores = []

        model = xgb.XGBRegressor(
            **param_dict,
            n_jobs=multiprocessing.cpu_count() // 2,
            tree_method="hist",
            objective="reg:squarederror",
            booster="gbtree"
        )

        for train_index, test_index in tscv.split(xgb_data):
            X_train_scaled = scaler.fit_transform(xgb_data.iloc[train_index]) if scaler else xgb_data.iloc[train_index]
            X_test_scaled = scaler.transform(xgb_data.iloc[test_index]) if scaler else xgb_data.iloc[test_index]

            model.fit(X_train_scaled, xgb_label.iloc[train_index])
            pred = model.predict(X_test_scaled)
            corr = np.corrcoef(pred, xgb_label.iloc[test_index])[1, 0]
            scores.append(corr)

        avg_score = np.mean(scores)

        if avg_score > best_score:
            best_score = avg_score
            best_params = param_dict

    return best_score, best_params

best_score, best_params = xgb_model_selection(xgb_data, xgb_label, scaler)

print(f"Best average correlation score: {best_score}")
print(f"Best parameters: {best_params}")

# Train final model
xgb_model = xgb.XGBRegressor(
    **best_params,
    n_jobs=multiprocessing.cpu_count() // 2,
    tree_method="hist",
    objective="reg:squarederror",
    booster="gbtree"
)

X_scaled = scaler.fit_transform(xgb_data)
xgb_model.fit(X_scaled, xgb_label)

# Get feature importances (by weight, gain, or cover â€” here we use 'weight')
importances = xgb_model.feature_importances_
def plot_xgb(importances, xgb_data):
    feature_names = xgb_data.columns
    
    # Create a DataFrame to sort and select top 10
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance descending and select top 10
    top_10 = importance_df.sort_values(by='importance', ascending=False).head(10)
    
    # Plot
    top_10[::-1].plot.barh(x='feature', y='importance', legend=False, figsize=(8, 6))
    plt.xlabel("Importance")
    plt.title("Top 10 XGBoost Feature Importances (Weight)")
    plt.tight_layout()
    plt.show()

plot_xgb(importances, xgb_data)
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
y_scaled = scaler.transform(test_df[common_features])
prediction = pd.DataFrame(xgb_model.predict(y_scaled))
prediction.to_csv('output1.csv')



outliers = {col: [] for col in train_df.columns}
for col in train_df.select_dtypes(include=[np.number]).columns:  # Only check numeric columns
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Append column outliers to output
        outliers[col].append(train_df.loc[(train_df[col] < lower_bound) | (train_df[col] > upper_bound), col].count())
        
        



df = pd.DataFrame(outliers)
df.transpose().sort_values(by=0, ascending=False).head(20)



import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt


columns = [f'X{i}' for i in range(697, 719)]
columns.append('label')
print("\n" + "="*50)
print("3. DIMENSIONALITY REDUCTION")
print("="*50)
#dropping columns with no valid values

X_features = train_df.drop('label', axis=1)
X_clean = X_features.select_dtypes(include=[np.number])
# 2. Fill missing values with column median
# 3. Downcast to float32 to save memory
X_clean = X_clean.astype(np.float32)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_clean)
temp = np.empty((1, 85), dtype=np.float32)
print(temp)
chunk_size = 10000
ipca = IncrementalPCA(n_components=85)
for i in range(0, X_clean.shape[0], chunk_size):
    # 5. Perform Incremental PCA in batches
    ipca.partial_fit(X_scaled[i:i+chunk_size])
    #transform scaled data set
    X_pca = ipca.transform(X_scaled[i:i+chunk_size])
    temp = np.vstack((temp, X_pca))

full_scaled = np.delete(temp, 1, axis=0)
# 6. Explained variance analysis
explained_variance_ratio = ipca.explained_variance_ratio_
cumsum_variance = np.cumsum(explained_variance_ratio)
n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1

print(f"Number of components needed for 90% variance: {n_components_90}")
print(f"Number of components needed for 95% variance: {n_components_95}")

# 7. Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1),
         explained_variance_ratio, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA - Individual Explained Variance')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumsum_variance) + 1),
         cumsum_variance, 'ro-')
plt.axhline(y=0.90, color='g', linestyle='--', label='90% variance')
plt.axhline(y=0.95, color='b', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





pca_df = pd.DataFrame(full_scaled).dropna().to_numpy()
y_aligned = xgb_label.iloc[:pca_df.shape[0]]


print("\n" + "="*50)
print("5. CLUSTERING ANALYSIS")
print("="*50)


# K-means clustering on PCA-reduced data
print("Performing K-means clustering...")
n_clusters_range = range(2, 11)
inertias = []

for k in n_clusters_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_df)  # Use first 50 PCA components
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('K-means Clustering - Elbow Method')
plt.grid(True)
plt.show()



optimal_k = 3  # You can adjust based on elbow curve
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans_final.fit_predict(pca_df)
y=y_aligned
# Analyze clusters
cluster_analysis = pd.DataFrame({
    'cluster': clusters,
    'target': y
})

print(f"\nCluster analysis with {optimal_k} clusters:")
cluster_stats = cluster_analysis.groupby('cluster')['target'].agg(['count', 'mean', 'std'])
print(cluster_stats)

# Visualize clusters using first 2 PCA components
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(pca_df[:, 0], pca_df[:, 1], c=clusters, alpha=0.6, cmap='tab10')
plt.colorbar(scatter, label='Cluster')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-means Clusters in PCA Space')

plt.subplot(1, 2, 2)
scatter = plt.scatter(pca_df[:, 0], pca_df[:, 1], c=y, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Target Value')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Target Values in PCA Space')

plt.tight_layout()
plt.show()





#add distance to cluster centroid as a feature
from scipy.stats import pearsonr

from scipy.spatial.distance import cdist
distances = cdist(pca_df, kmeans_final.cluster_centers_, metric='euclidean')  # shape: (n_samples, n_clusters)
pca_final = pd.DataFrame(pca_df)
for i in range(distances.shape[1]):
    pca_final[f'distance_to_centroid_{i}'] = distances[:, i]
    corr, pval = pearsonr(pca_final[f'distance_to_centroid_{i}'], y_aligned)
    print(f"cluster {i} Pearson correlation: r = {corr:.4f}, p = {pval:.4e}")
mean_distances = distances.mean(axis=1)  # shape: (n_samples,)
pca_final['mean_distance_to_centroids'] = mean_distances
corr, pval = pearsonr(pca_final['mean_distance_to_centroids'], y_aligned)
print(f"mean Pearson correlation: r = {corr:.4f}, p = {pval:.4e}")
pca_final['cluster_label'] = clusters
pd.get_dummies(pca_final, columns=['cluster_label'], prefix='cluster')



#prediction with pca and clustering
best_score, best_params = xgb_model_selection(pca_data, y_aligned, None)

print(f"Best average correlation score: {best_score}")
print(f"Best parameters: {best_params}")

# Train final model
xgb_model = xgb.XGBRegressor(
    **best_params,
    n_jobs=multiprocessing.cpu_count() // 2,
    tree_method="hist",
    objective="reg:squarederror",
    booster="gbtree"
)

xgb_model.fit(pca_final, y_aligned)

# Get feature importances (by weight, gain, or cover â€” here we use 'weight')
importances = xgb_model.feature_importances_

    # Sort by importance descending and select top 10
    top_10 = importance_df.sort_values(by='importance', ascending=False).head(10)
    
    # Plot
    top_10[::-1].plot.barh(x='feature', y='importance', legend=False, figsize=(8, 6))
    plt.xlabel("Importance")
    plt.title("Top 10 XGBoost Feature Importances (Weight)")
    plt.tight_layout()
    plt.show()

plot_xgb(importances, pca_final)
y_scaled = scaler.transform(test_df)
y_pca = ipca.fit_transform(y_scaled)
prediction = pd.DataFrame(xgb_model.predict(y_scaled))
prediction.to_csv('output2.csv')


X_clean = X_clean.iloc[:pca_df.shape[0]]

for i in range(distances.shape[1]):
    X_clean[f'distance_to_centroid_{i}'] = distances[:, i]
    corr, pval = pearsonr(X_clean[f'distance_to_centroid_{i}'], y_aligned)
    print(f"cluster {i} Pearson correlation: r = {corr:.4f}, p = {pval:.4e}")
mean_distances = distances.mean(axis=1)  # shape: (n_samples,)
X_clean['mean_distance_to_centroids'] = mean_distances


xgb_data = X_clean.drop('label', axis=1)[common_features+
                                        [f'distance_to_centroid_{i}' for i in range(distances.shape[1])]+
                                        ['mean_distance_to_centroids']]
scaler = StandardScaler()
best_score, best_params = xgb_model_selection(xgb_data, y_aligned, scaler)

print(f"Best average correlation score: {best_score}")
print(f"Best parameters: {best_params}")

# Train final model
xgb_model = xgb.XGBRegressor(
    **best_params,
    n_jobs=multiprocessing.cpu_count() // 2,
    tree_method="hist",
    objective="reg:squarederror",
    booster="gbtree"
)

X_scaled = scaler.fit_transform(xgb_data)
xgb_model.fit(X_scaled, xgb_label)

# Get feature importances (by weight, gain, or cover â€” here we use 'weight')
importances = xgb_model.feature_importances_

    # Sort by importance descending and select top 10
    top_10 = importance_df.sort_values(by='importance', ascending=False).head(10)
    
    # Plot
    top_10[::-1].plot.barh(x='feature', y='importance', legend=False, figsize=(8, 6))
    plt.xlabel("Importance")
    plt.title("Top 10 XGBoost Feature Importances (Weight)")
    plt.tight_layout()
    plt.show()

plot_xgb(importances, xgb_data)


