import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
from statsmodels.graphics.tsaplots import plot_acf
warnings.filterwarnings('ignore')


# Load the data
print("Loading training data...")
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
print(f"Training data shape: {train_df.shape}")



print("1. EXPLORATORY DATA ANALYSIS")

# Basic info about the dataset
print(f"Dataset shape: {train_df.shape}")
print(f"Number of anonymized features: {len([col for col in train_df.columns if col.startswith('X')])}")

# Check for missing values
missing_values = train_df.isnull().sum()
print(f"\nMissing values summary:")
print(f"Total features with missing values: {(missing_values > 0).sum()}")
if (missing_values > 0).sum() > 0:
    print("Features with most missing values:")
    print(missing_values[missing_values > 0].sort_values(ascending=False).head(10))

# Separate known features from anonymized features
known_features = ['timestamp', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
anonymized_features = [col for col in train_df.columns if col.startswith('X')]
target = 'label'

print(f"\nKnown features: {len(known_features)}")
print(f"Anonymized features: {len(anonymized_features)}")

# Statistical summary of anonymized features
X_features = train_df[anonymized_features]
print(f"\nAnonymized features statistical summary:")
print(X_features.describe().T.head(10))

# Check data types
print(f"\nData types of anonymized features:")
print(X_features.dtypes.value_counts())

# Distribution analysis of first few anonymized features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of First 6 Anonymized Features', fontsize=16)
for i, feature in enumerate(anonymized_features[:6]):
    row, col = i // 3, i % 3
    X_features[feature].hist(bins=50, ax=axes[row, col], alpha=0.7)
    axes[row, col].set_title(f'{feature}')
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')
plt.tight_layout()
plt.show()


print("\n" + "="*50)
print("2. FEATURE RELATIONSHIPS ANALYSIS")
print("="*50)

# Correlation analysis for a subset of features (first 50 for visualization)
subset_features = anonymized_features[:50]
correlation_matrix = train_df[subset_features  + [target]].corr()

# Plot correlation heatmap
plt.figure(figsize=(32, 20))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            square=True, fmt='.2f',annot=True, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix - First 50 Anonymized Features + Target')
plt.tight_layout()
plt.show()


def find_high_correlations(corr_matrix, threshold=0.8):
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    return pd.DataFrame(high_corr_pairs)

high_corr_df = find_high_correlations(correlation_matrix, threshold=0.98)
print(f"High correlation pairs (|corr| > 0.98):")
print(high_corr_df.sort_values('correlation', key=abs, ascending=False).head(50))

# Correlation with target variable
target_correlations = train_df[anonymized_features + [target]].corr()[target].abs().sort_values(ascending=False)
print(f"\nTop 20 features most correlated with target:")
print(target_correlations.head(51)[1:])  # Exclude target itself


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt

print("\n" + "="*50)
print("3. DIMENSIONALITY REDUCTION")
print("="*50)

# 1. Use only numeric columns and limit to 200 features
X_numeric = X_features.select_dtypes(include=[np.number]).iloc[:, :200]

# 2. Fill missing values with column median
X_clean = X_numeric.fillna(X_numeric.median())

# 3. Downcast to float32 to save memory
X_clean = X_clean.astype(np.float32)

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# 5. Perform Incremental PCA
print("Performing Incremental PCA...")
ipca = IncrementalPCA(n_components=50, batch_size=200)
X_pca = ipca.fit_transform(X_scaled)

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



# t-SNE visualization (on a sample for computational efficiency)
print("Performing t-SNE on sample data...")
sample_size = min(5000, len(X_clean))
sample_indices = np.random.choice(len(X_clean), sample_size, replace=False)
X_sample = X_scaled[sample_indices]
y = train_df['label']  # <-- Replace 'target' with your actual label column name
y_sample = y.iloc[sample_indices]

# Use PCA first to reduce dimensions before t-SNE
pca_pre = PCA(n_components=50)
X_pca_sample = pca_pre.fit_transform(X_sample)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_pca_sample)

plt.figure(figsize=(30, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Target Value')
plt.title('t-SNE Visualization of Anonymized Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()


from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

print("\n" + "="*50)
print("4. FEATURE SELECTION")
print("="*50)

# Univariate feature selection
print("Performing univariate feature selection...")
selector_univariate = SelectKBest(score_func=f_regression, k=100)
X_selected_univariate = selector_univariate.fit_transform(X_clean, y)

# Get feature scores
anonymized_features = X_clean.columns  # <-- Make sure this matches the full feature list
feature_scores = pd.DataFrame({
    'feature': anonymized_features,
    'score': selector_univariate.scores_
}).sort_values('score', ascending=False)

print("Top 20 features by univariate selection:")
print(feature_scores.head(20))



from sklearn.ensemble import RandomForestRegressor

print("\nTraining Random Forest for feature importance (with speed-up)...")

rf = RandomForestRegressor(
    n_estimators=100,       # Reduce trees from 100 â†’ 50 for faster training
    max_depth=10,          # Limit depth to prevent very deep trees
    max_features='sqrt',   # Use sqrt of features at each split (default)
    n_jobs=-1,             # Use all CPU cores
    random_state=42
)

rf.fit(X_clean, y)

feature_importance = pd.DataFrame({
    'feature': anonymized_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 20 features by Random Forest importance:")
print(feature_importance.head(20))



from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE

print("\nPerforming Recursive Feature Elimination with Lasso...")

# Initialize Lasso with a small alpha (regularization strength) and enough iterations
lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)

# RFE with step=50 (removes 50 features at a time) to speed up
rfe = RFE(estimator=lasso, n_features_to_select=100, step=50)

# Fit RFE on your data
rfe.fit(X_clean, y)

# Collect feature selection results
rfe_features = pd.DataFrame({
    'feature': anonymized_features,
    'selected': rfe.support_,
    'ranking': rfe.ranking_
}).sort_values('ranking')

selected_features_rfe = rfe_features[rfe_features['selected']]['feature'].tolist()
print(f"RFE selected {len(selected_features_rfe)} features")
print("Top 20 RFE selected features:")
print(rfe_features.head(20))



print("\n" + "="*50)
print("5. CLUSTERING ANALYSIS")
print("="*50)

# K-means clustering on PCA-reduced data
print("Performing K-means clustering...")
n_clusters_range = range(2, 11)
inertias = []

for k in n_clusters_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca[:, :50])  # Use first 50 PCA components
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('K-means Clustering - Elbow Method')
plt.grid(True)
plt.show()


# Apply optimal clustering
optimal_k = 5  # You can adjust based on elbow curve
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans_final.fit_predict(X_pca[:, :50])

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
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, alpha=0.6, cmap='tab10')
plt.colorbar(scatter, label='Cluster')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-means Clusters in PCA Space')

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Target Value')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Target Values in PCA Space')

plt.tight_layout()
plt.show()



print("\n" + "="*80)
print("SUMMARY INSIGHTS")
print("="*80)

print(f"1. Dataset contains {890} anonymized features")
print(f"2. {n_components_90} components explain 90% of variance, {n_components_95} explain 95%")
print(f"3. Top correlated feature with target: {target_correlations.index[1]} (corr: {target_correlations.iloc[1]:.4f})")
print(f"4. {len(high_corr_df)} feature pairs have high correlation (>0.7)")
print(f"5. Optimal number of clusters appears to be around {optimal_k}")

# Save important features for further analysis
important_features = {
    'top_univariate': feature_scores.head(50)['feature'].tolist(),
    'top_rf_importance': feature_importance.head(50)['feature'].tolist(),
    'rfe_selected': selected_features_rfe,
    'high_target_corr': target_correlations.head(50).index[1:].tolist()
}

common_features = set(important_features['top_univariate']) & \
                 set(important_features['top_rf_importance']) & \
                 set(important_features['rfe_selected'])
print(f"  Common Features selected by all 3 methods: {(common_features)}")



try:
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], unit='s')
except:
    pass  # Leave it if it's already in datetime format or anonymized

# ğŸ“Š 1. Distribution of the label
plt.figure(figsize=(10, 5))
sns.histplot(train_df['label'], kde=True, bins=100, color='steelblue')
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.grid(True)
plt.show()

print("Label Summary Statistics:")
print(train_df['label'].describe())
print("\nNumber of Unique Labels:", train_df['label'].nunique())


# ğŸ“‰ 2. Sign Distribution (Up, Down, Neutral)
train_df['label_sign'] = train_df['label'].apply(lambda x: 'up' if x > 0 else ('down' if x < 0 else 'neutral'))

plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='label_sign', order=['up', 'neutral', 'down'], palette='Set2')
plt.title('Label Direction Distribution')
plt.xlabel('Movement')
plt.ylabel('Count')
plt.grid(True)
plt.show()



train_df.columns


train_df['ask_qty']


train_df = train_df.sort_index()  # sorts by timestamp index

rolling_mean = train_df['label'].rolling(window=1000).mean()
rolling_std = train_df['label'].rolling(window=1000).std()

plt.figure(figsize=(12, 5))
plt.plot(rolling_mean, label='Rolling Mean (1000)', color='blue')
plt.plot(rolling_std, label='Rolling Std (1000)', color='orange')
plt.title('Rolling Statistics of Label Over Time')
plt.xlabel('Time')
plt.ylabel('Label Value')
plt.legend()
plt.grid(True)
plt.show()



# ğŸ”� 4. Autocorrelation (Noise Check)
plt.figure(figsize=(10, 4))
plot_acf(train_df['label'].dropna(), lags=50, title="Autocorrelation of Label")
plt.tight_layout()
plt.show()


# ğŸ“Š 5. Correlation with Market Features
market_feats = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
corr = train_df[market_feats + ['label']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation with Label')
plt.show()



# âš™ï¸� Optional: Discretize Label for Classification Experiment
train_df['label_class'] = pd.qcut(train_df['label'], q=3, labels=['down', 'neutral', 'up'])

plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='label_class', palette='viridis')
plt.title('Label Quantile Bins')
plt.xlabel('Label Class')
plt.ylabel('Count')
plt.grid(True)
plt.show()


print(f"\nRECOMMENDED NEXT STEPS:")
print(f"1. Focus on the top {len(common_features)} features identified by multiple selection methods")
print(f"2. Use {n_components_90}-{n_components_95} PCA components for dimensionality reduction")
print(f"3. Consider cluster-based features as additional engineered features")
print(f"4. Remove highly correlated features to reduce redundancy")
print(f"5. Use the identified important features for model training")


!pip install koolbox scikit-learn==1.5.2


from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from sklearn.base import clone
from koolbox import Trainer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import optuna
import joblib
import gc

warnings.filterwarnings("ignore")


class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    target = "label"
    n_folds = 5
    seed = 42

    run_optuna = True
    n_optuna_trials = 250


!pip install koolbox scikit-learn==1.5.2



from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from sklearn.base import clone
from koolbox import Trainer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import optuna
import joblib
import gc

warnings.filterwarnings("ignore")


class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    target = "label"
    n_folds = 5
    seed = 42

    run_optuna = True
    n_optuna_trials = 250


def reduce_mem_usage(dataframe, dataset):    
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe


def add_features(df):
    data = df.copy()
    features_df = pd.DataFrame(index=data.index)
    
    features_df['bid_ask_spread_proxy'] = data['ask_qty'] - data['bid_qty']
    features_df['total_liquidity'] = data['bid_qty'] + data['ask_qty']
    features_df['trade_imbalance'] = data['buy_qty'] - data['sell_qty']
    features_df['total_trades'] = data['buy_qty'] + data['sell_qty']
    
    features_df['volume_per_trade'] = data['volume'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    features_df['buy_volume_ratio'] = data['buy_qty'] / (data['volume'] + 1e-8)
    features_df['sell_volume_ratio'] = data['sell_qty'] / (data['volume'] + 1e-8)
    
    features_df['buying_pressure'] = data['buy_qty'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    features_df['selling_pressure'] = data['sell_qty'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    
    features_df['order_imbalance'] = (data['bid_qty'] - data['ask_qty']) / (data['bid_qty'] + data['ask_qty'] + 1e-8)
    features_df['order_imbalance_abs'] = np.abs(features_df['order_imbalance'])
    features_df['bid_liquidity_ratio'] = data['bid_qty'] / (data['volume'] + 1e-8)
    features_df['ask_liquidity_ratio'] = data['ask_qty'] / (data['volume'] + 1e-8)
    features_df['market_depth'] = data['bid_qty'] + data['ask_qty']
    features_df['depth_imbalance'] = features_df['market_depth'] - data['volume']
    
    features_df['buy_sell_ratio'] = data['buy_qty'] / (data['sell_qty'] + 1e-8)
    features_df['bid_ask_ratio'] = data['bid_qty'] / (data['ask_qty'] + 1e-8)
    features_df['volume_liquidity_ratio'] = data['volume'] / (data['bid_qty'] + data['ask_qty'] + 1e-8)

    features_df['buy_volume_product'] = data['buy_qty'] * data['volume']
    features_df['sell_volume_product'] = data['sell_qty'] * data['volume']
    features_df['bid_ask_product'] = data['bid_qty'] * data['ask_qty']
    
    features_df['market_competition'] = (data['buy_qty'] * data['sell_qty']) / ((data['buy_qty'] + data['sell_qty']) + 1e-8)
    features_df['liquidity_competition'] = (data['bid_qty'] * data['ask_qty']) / ((data['bid_qty'] + data['ask_qty']) + 1e-8)
    
    total_activity = data['buy_qty'] + data['sell_qty'] + data['bid_qty'] + data['ask_qty']
    features_df['market_activity'] = total_activity
    features_df['activity_concentration'] = data['volume'] / (total_activity + 1e-8)
    
    features_df['info_arrival_rate'] = (data['buy_qty'] + data['sell_qty']) / (data['volume'] + 1e-8)
    features_df['market_making_intensity'] = (data['bid_qty'] + data['ask_qty']) / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    features_df['effective_spread_proxy'] = np.abs(data['buy_qty'] - data['sell_qty']) / (data['volume'] + 1e-8)
    
    lambda_decay = 0.95
    ofi = data['buy_qty'] - data['sell_qty']
    features_df['order_flow_imbalance_ewm'] = ofi.ewm(alpha=1-lambda_decay).mean()

    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    return features_df


cols_to_drop = [
    'X697', 'X698', 'X699', 'X700', 'X701', 'X702', 'X703', 'X704', 'X705', 'X706', 
    'X707', 'X708', 'X709', 'X710', 'X711', 'X712', 'X713', 'X714', 'X715', 'X716',
    'X717', 'X864', 'X867', 'X869', 'X870', 'X871', 'X872', 'X104', 'X110', 'X116',
    'X122', 'X128', 'X134', 'X146', 'X152', 'X158', 'X164', 'X170', 'X176',
    'X182', 'X351', 'X357', 'X363', 'X369', 'X375', 'X381', 'X387', 'X393', 'X399',
    'X405', 'X411', 'X417', 'X423', 'X429', 'X46',  'X50', 'X45', 'X49', 'X40',
    'X44', 'X39', 'X43', 'X6', 'X8', 'X34', 'X38', 'X35', 'X16', 'X1','X14' 
]


train = pd.read_parquet(CFG.train_path).reset_index(drop=True)
test = pd.read_parquet(CFG.test_path).reset_index(drop=True)

train = train.drop(columns=cols_to_drop)
test = test.drop(columns=["label"] + cols_to_drop)

train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test


X = pd.concat([add_features(X), X], axis=1)
X_test = pd.concat([add_features(X_test), X_test], axis=1)


def _pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.5625888953382505,
    "learning_rate": 0.029312951475451557,
    "min_child_samples": 63,
    "min_child_weight": 0.11456572852335424,
    "n_estimators": 126,
    "n_jobs": -1,
    "num_leaves": 37,
    "random_state": 42,
    "reg_alpha": 85.2476527854083,
    "reg_lambda": 99.38305361388907,
    "subsample": 0.450669817684892,
    "verbose": -1
}

lgbm_goss_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.34695458228489784,
    "learning_rate": 0.031023014900595287,
    "min_child_samples": 30,
    "min_child_weight": 0.4727729225033618,
    "n_estimators": 220,
    "n_jobs": -1,
    "num_leaves": 58,
    "random_state": 42,
    "reg_alpha": 38.665994901468224,
    "reg_lambda": 92.76991677464294,
    "subsample": 0.4810891284493255,
    "verbose": -1
}

xgb_params = {
    "colsample_bylevel": 0.4778015829774066,
    "colsample_bynode": 0.362764358742407,
    "colsample_bytree": 0.7107423488010493,
    "gamma": 1.7094857725240398,
    "learning_rate": 0.02213323588455387,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 39.352415706891264,
    "reg_lambda": 75.44843704068275,
    "subsample": 0.06566669853471274,
    "verbosity": 0
}


scores = {}
oof_preds = {}
test_preds = {}



lgbm_trainer = Trainer(
    LGBMRegressor(**lgbm_params),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)

lgbm_trainer.fit(X, y)

scores["LightGBM (gbdt)"] = lgbm_trainer.fold_scores
oof_preds["LightGBM (gbdt)"] = lgbm_trainer.oof_preds
test_preds["LightGBM (gbdt)"] = lgbm_trainer.predict(X_test)


lgbm_goss_trainer = Trainer(
    LGBMRegressor(**lgbm_goss_params),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)

lgbm_goss_trainer.fit(X, y)

scores["LightGBM (goss)"] = lgbm_goss_trainer.fold_scores
oof_preds["LightGBM (goss)"] = lgbm_goss_trainer.oof_preds
test_preds["LightGBM (goss)"] = lgbm_goss_trainer.predict(X_test)



xgb_trainer = Trainer(
    XGBRegressor(**xgb_params),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)

xgb_trainer.fit(X, y)

scores["XGBoost"] = xgb_trainer.fold_scores
oof_preds["XGBoost"] = xgb_trainer.oof_preds
test_preds["XGBoost"] = xgb_trainer.predict(X_test)


def plot_weights(weights, title):
    sorted_indices = np.argsort(weights[0])[::-1]
    sorted_coeffs = np.array(weights[0])[sorted_indices]
    sorted_model_names = np.array(list(oof_preds.keys()))[sorted_indices]

    plt.figure(figsize=(10, weights.shape[1] * 0.5))
    ax = sns.barplot(x=sorted_coeffs, y=sorted_model_names, palette="RdYlGn_r")

    for i, (value, name) in enumerate(zip(sorted_coeffs, sorted_model_names)):
        if value >= 0:
            ax.text(value, i, f"{value:.3f}", va="center", ha="left", color="black")
        else:
            ax.text(value, i, f"{value:.3f}", va="center", ha="right", color="black")

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.1 * abs(xlim[0]), xlim[1] + 0.1 * abs(xlim[1]))

    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


X = pd.DataFrame(oof_preds)
X_test = pd.DataFrame(test_preds)


joblib.dump(X, "oof_preds.pkl")
joblib.dump(X_test, "test_preds.pkl")


def objective(trial):    
    params = {
        "random_state": CFG.seed,
        "alpha": trial.suggest_float("alpha", 0, 1000),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "positive": trial.suggest_categorical("positive", [True, False])
    }

    trainer = Trainer(
        Ridge(**params),
        cv=KFold(n_splits=5, shuffle=False),
        metric=_pearsonr,
        task="regression",
        verbose=False
    )
    trainer.fit(X, y)
    
    return np.mean(trainer.fold_scores)

if CFG.run_optuna:
    sampler = optuna.samplers.TPESampler(seed=CFG.seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=CFG.n_optuna_trials, n_jobs=-1, catch=(ValueError,))
    best_params = study.best_params

    ridge_params = {
        "random_state": CFG.seed,
        "alpha": best_params["alpha"],
        "tol": best_params["tol"],
        "fit_intercept": best_params["fit_intercept"],
        "positive": best_params["positive"]
    }
else:
    ridge_params = {
        "random_state": CFG.seed
    }


ridge_trainer = Trainer(
    Ridge(**ridge_params),
    cv=KFold(n_splits=5, shuffle=False),
    metric=_pearsonr,
    task="regression",
    metric_precision=6
)

ridge_trainer.fit(X, y)

scores["Ridge (ensemble)"] = ridge_trainer.fold_scores
ridge_test_preds = ridge_trainer.predict(X_test)


ridge_coeffs = np.zeros((1, X.shape[1]))
for m in ridge_trainer.estimators:
    ridge_coeffs += m.coef_
ridge_coeffs = ridge_coeffs / len(ridge_trainer.estimators)

plot_weights(ridge_coeffs, "Ridge Coefficients")


sub = pd.read_csv(CFG.sample_sub_path)
sub["prediction"] = ridge_test_preds
sub.to_csv(f"sub_ridge_{np.mean(scores['Ridge (ensemble)']):.6f}.csv", index=False)
sub.head()


scores = pd.DataFrame(scores)
mean_scores = scores.mean().sort_values(ascending=False)
order = scores.mean().sort_values(ascending=False).index.tolist()

min_score = mean_scores.min()
max_score = mean_scores.max()
padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding

fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.5))

boxplot = sns.boxplot(data=scores, order=order, ax=axs[0], orient="h", color="grey")
axs[0].set_title(f"Fold Score")
axs[0].set_xlabel("")
axs[0].set_ylabel("")

barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], color="grey")
axs[1].set_title(f"Average Score")
axs[1].set_xlabel("")
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel("")

for i, (score, model) in enumerate(zip(mean_scores.values, mean_scores.index)):
    color = "cyan" if "ensemble" in model.lower() else "grey"
    barplot.patches[i].set_facecolor(color)
    boxplot.patches[i].set_facecolor(color)
    barplot.text(score, i, round(score, 6), va="center")

plt.tight_layout()
plt.show()

