from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import hdbscan
import category_encoders as ce


from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
import seaborn as sns

sns.set()


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col = 0)
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col=0)


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds ratio-based and aggregate features to a DataFrame.
    Assumes input is a DataFrame with specific base columns.
    """

    
    def __init__(self):
        self.columns_ = None

    
    def _clean_column_names(self, cols):
        """Remove 'cat__' and 'num__' prefixes from column names."""
        
        return [col.replace('cat__', '').replace('num__', '') for col in cols]

    
    def fit(self, X, y=None):
        """Store final column names after feature creation."""
        
        base_cols = self._clean_column_names(X.columns)
        new_cols = [
            'Total_score', 'Alone_Event_ratio', 'Post_per_Event', 'Post_per_Friend',
            'Event_per_Friend', 'Event_Outside_ratio', 'Outside_Alone_ratio',
            'Drained_Total_ratio', 'Fear_Total_ratio', 'Post_Total_ratio',
            'Alone_Total_ratio', 'Friend_Total_ratio', 'Social_Total_ratio'
        ]
        self.columns_ = base_cols + new_cols
        return self

    
    def transform(self, X, y=None):
        """Add new features to input DataFrame."""
        
        X = X.copy()
        X.columns = self._clean_column_names(X.columns)

        X['Total_score'] = X.sum(axis=1)
        X['Alone_Event_ratio'] = X['Time_spent_Alone'] / (X['Social_event_attendance'] + 1)
        X['Post_per_Event'] = X['Post_frequency'] / (X['Social_event_attendance'] + 1)
        X['Post_per_Friend'] = X['Post_frequency'] / (X['Friends_circle_size'] + 1)
        X['Event_per_Friend'] = X['Social_event_attendance'] / (X['Friends_circle_size'] + 1)
        X['Event_Outside_ratio'] = X['Social_event_attendance'] / (X['Going_outside'] + 1)
        X['Outside_Alone_ratio'] = X['Going_outside'] / (X['Time_spent_Alone'] + 1)
        X['Drained_Total_ratio'] = X['Drained_after_socializing'] / X['Total_score']
        X['Fear_Total_ratio'] = X['Stage_fear'] / X['Total_score']
        X['Post_Total_ratio'] = X['Post_frequency'] / X['Total_score']
        X['Alone_Total_ratio'] = X['Time_spent_Alone'] / X['Total_score']
        X['Friend_Total_ratio'] = X['Friends_circle_size'] / X['Total_score']
        X['Social_Total_ratio'] = X['Social_event_attendance'] / X['Total_score']
        return X

    
    def get_feature_names_out(self, input_features=None):
        """Return output column names."""
        
        return self.columns_


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

cat_cols = ['Stage_fear', 'Drained_after_socializing']


rounder = FunctionTransformer(func=lambda X: pd.DataFrame(np.round(X), columns=X.columns), feature_names_out='one-to-one')


numeric_pipeline = Pipeline([
    ('imput_nums',KNNImputer(n_neighbors=20, add_indicator=True, weights='distance').set_output(transform='pandas')),
    ('round_values', rounder),
])


categorical_pipeline = Pipeline([
    ('encode', ce.OrdinalEncoder(handle_unknown='impute', handle_missing='return_nan').set_output(transform='pandas')),
    ('imput_nums', KNNImputer(n_neighbors=20, add_indicator=True, weights='distance').set_output(transform='pandas')),
    ('round_values', rounder)
])


num_cat_preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols),
], remainder='passthrough').set_output(transform='pandas')  


preprocessing_pipeline = Pipeline([
    ('encode_imput_preprocess', num_cat_preprocessor),
    ('add_features', FeatureCreator()),  
    ('scaler', MinMaxScaler()),
    ])


X = train.drop('Personality', axis=1)
X_scaled = preprocessing_pipeline.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns = preprocessing_pipeline.get_feature_names_out(), index = train.index)
y = train['Personality'].map({"Introvert": 0, "Extrovert": 1})


def plot_feature_distributions(df, target_col='Personality', cols_to_plot=None):
    if cols_to_plot is None:
        cols_to_plot = [col for col in df.columns if col != target_col]

    num_cols = 2
    num_rows = (len(cols_to_plot) + 1) // num_cols

    plt.figure(figsize=(num_cols * 6, num_rows * 4))

    for i, col in enumerate(cols_to_plot, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.histplot(data=df, x=col, hue=target_col, kde=True, bins=50, palette=['#FF7518', '#51158C'], element='step', stat='density')
        plt.title(f'Distribution of {col}')
        plt.xlabel("")
        plt.ylabel('Density')

    plt.tight_layout()
    plt.show()


drop_cols = [col for col in X_scaled_df.columns if 'missing' in col]
plot_feature_distributions(pd.concat([X_scaled_df.drop(drop_cols, axis=1), train['Personality']], axis=1), target_col='Personality')


def find_metrics(model_name, y_val, y_pred, y_proba):
    results_df = pd.DataFrame([{
                'Model': model_name,
                'Accuracy': accuracy_score(y_val, y_pred),
                'Precision': precision_score(y_val, y_pred, zero_division=0),
                'Recall': recall_score(y_val, y_pred),
                'F1 Score': f1_score(y_val, y_pred),
                'PR AUC': average_precision_score(y_val, y_proba)
                }])
    return results_df   


k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Store fold results
fold_accuracies = []
oof_pred_svm = np.zeros(len(y))
oof_proba_svm = np.zeros(len(y))
all_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y), 1):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # SVM model
    modelsv = SVC(kernel='rbf', class_weight='balanced', probability=True) 
    modelsv.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = modelsv.predict(X_val)
    y_proba = modelsv.predict_proba(X_val)[:, 1]
    oof_pred_svm[val_idx] = y_pred
    oof_proba_svm[val_idx] = y_proba

    # Accuracy
    acc = accuracy_score(y_val, y_pred)
    fold_accuracies.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")

    # Other metrics
    fold_result = find_metrics('SVC', y_val, y_pred, y_proba)
    fold_result['Fold'] = fold
    all_results.append(fold_result)

# Overall result
print(f"\nAverage Accuracy across {k} folds: {np.mean(fold_accuracies):.4f}")


results_df = pd.concat(all_results, axis=0, ignore_index=True)
results_df 


svm_results = pd.DataFrame(oof_pred_svm, index=y.index, columns = ['oof_pred'])
svm_results['oof_proba'] = oof_proba_svm
svm_results['y_true'] = y
svm_results['misclassified_points'] = (svm_results['oof_pred'] != svm_results['y_true']).astype(int)


miss_percent = svm_results.groupby(['y_true', 'misclassified_points']).agg(count=('misclassified_points', 'count'),
                                                           percent=('misclassified_points', lambda x: 100 * len(x) / svm_results.shape[0])
                                                           ).reset_index()
miss_percent 


values = miss_percent.percent.values  
labels = ['Introvert_correct_classified', 'Introvert_misclassified', 'Extrovert_correct_classified',  'Extrovert_misclassified']
new_order = [0, 2, 3, 1] 
values_reordered = [values[i] for i in new_order]
labels_reordered = [labels[i] for i in new_order]
colors_reordered = ['#B163FF', '#FFC985', '#FFD1B3', '#9D9DCC'] 

plt.pie(values_reordered, labels = labels_reordered, autopct='%1.1f%%', startangle=2, counterclock=True, colors = colors_reordered,  explode = [0.01, 0.01, 0.5, 0.5])
plt.title('Correct vs Misclassified Samples per Class')
plt.tight_layout()
plt.show()


X_reduced = PCA(n_components=2).fit_transform(X_scaled_df.drop(drop_cols, axis=1))

svc = SVC(kernel='rbf', class_weight='balanced')
svc.fit(X_reduced, y)
y_pred = svc.predict(X_reduced)

h = .02
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap='Spectral', alpha=0.5)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='Spectral', alpha=0.7)
plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=100, facecolors='none', label='Support Vectors', alpha=0.5)
plt.title("SVM Decision Boundary (PCA-Reduced Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()


plt.figure(figsize=(10, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='Spectral', alpha=0.1, s=30)

# Outliers
outlier_mask = svm_results['misclassified_points'] == 1
y_mask = svm_results[svm_results['misclassified_points'] == 1]['y_true']
plt.scatter(X_reduced[outlier_mask, 0], X_reduced[outlier_mask, 1], c=y_mask, cmap='Spectral', s=40, label='True Class Label')
plt.scatter(X_reduced[outlier_mask, 0], X_reduced[outlier_mask, 1], facecolors='none', edgecolors='#222831', s=60, label='Model Errors')
plt.title("PCA Projection with Misclassified Points Highlighted")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()


tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled_df.drop([c for c in X_scaled_df if 'missing' in c], axis=1))

tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['Personality'] = train['Personality']  


plt.figure(figsize=(10, 8))
sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Personality', palette=['#FF7518', '#51158C'], alpha=0.5)
plt.title('t-SNE projection of feature space')
plt.grid(True)
plt.show()


X_pca = pd.DataFrame(PCA(n_components=6).fit_transform(X_scaled_df.drop([c for c in X_scaled_df if 'missing' in c], axis=1)), columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'], index = train.index)
X_features = pd.DataFrame(X_scaled, columns =[x for x in range(X_scaled.shape[1])], index=train.index)
X_outlier_full = pd.concat([X_features, X_pca, svm_results.drop('y_true', axis=1)], axis=1) 

all_svc_errors = X_outlier_full[X_outlier_full.misclassified_points == 1]
sample_non_outl =  X_outlier_full[~(X_outlier_full.misclassified_points == 1)].sample(500)

graph_df =  pd.concat([all_svc_errors, sample_non_outl], axis=0)

# lot
g = sns.pairplot(
    graph_df[['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'misclassified_points']],
    hue='misclassified_points',
    plot_kws={'alpha': 0.5}
)
g.fig.suptitle("SVC Misclassifications Across Different PCA Projections with Sampled In-Class Examples", y=1.01)


df_pca = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'], index=train.index)
df_pca['label'] = y

df_intro = df_pca[df_pca['label'] == 0].copy()
df_extro = df_pca[df_pca['label'] == 1].copy()


# dbscan_intro = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=200, allow_single_cluster=True)
# labels = dbscan_intro.fit_predict(df_intro[['PC1', 'PC2']])

dbscan_intro = DBSCAN(eps=0.6, min_samples=1000)
labels = dbscan_intro.fit_predict(df_intro[['PC1', 'PC2']])  

df_intro['cluster'] = labels


n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")


# Plot
plt.figure(figsize=(10, 8))
core_points_mask = labels != -1
noise_points_mask = labels == -1

for cluster_id in np.unique(labels):
    cluster_mask = df_intro['cluster'] == cluster_id
    if cluster_id == -1:
        # Noise
        plt.scatter(df_intro.loc[cluster_mask, 'PC1'], df_intro.loc[cluster_mask, 'PC2'], facecolors='none', edgecolors='gray', s=50, label='Noise')
    else:
        # Clusters
        plt.scatter(df_intro.loc[cluster_mask, 'PC1'], df_intro.loc[cluster_mask, 'PC2'], s=30, label=f'Cluster {cluster_id}', cmap='Spectral')

plt.title("DBSCAN Clustering on INTROVERT Class True Labels (PCA Space)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()


dbscan_extro = DBSCAN(eps=0.4, min_samples=1000)
labels = dbscan_extro.fit_predict(df_extro[['PC1', 'PC2']]) 
df_extro['cluster'] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")

core_points_mask = labels != -1
noise_points_mask = labels == -1

# Plot
plt.figure(figsize=(10, 8))
core_points_mask = labels != -1
noise_points_mask = labels == -1

for cluster_id in np.unique(labels):
    cluster_mask = df_extro['cluster'] == cluster_id
    if cluster_id == -1:
        # Noise
        plt.scatter(df_extro.loc[cluster_mask, 'PC1'], df_extro.loc[cluster_mask, 'PC2'], facecolors='none', edgecolors='gray', s=50, label='Noise')
    else:
        # Clusters
        plt.scatter(df_extro.loc[cluster_mask, 'PC1'], df_extro.loc[cluster_mask, 'PC2'], s=30, label=f'Cluster {cluster_id}', cmap='Spectral')

plt.title("DBSCAN Clustering on EXTROVERT Class True Labels (PCA Space)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()


clusters_df = pd.concat([df_intro, df_extro], axis=0)
clusters_df['svc_erroros'] = svm_results['misclassified_points']
clusters_df.groupby(['cluster', 'svc_erroros']).agg({'svc_erroros': 'count'})


plt.figure(figsize=(12, 8))

# SVC errors
errors = clusters_df['svc_erroros'] == 1
plt.scatter(clusters_df.loc[errors, 'PC1'], clusters_df.loc[errors, 'PC2'],  c='violet', cmap='coolwarm', alpha=0.6, label='SVC errors')

#  DBSCAN outliers
outliers = clusters_df['cluster'] == -1
plt.scatter(clusters_df.loc[outliers, 'PC1'], clusters_df.loc[outliers, 'PC2'], facecolors='none', edgecolors='black', s=80, linewidths=1.5, label='DBSCAN Outliers')

plt.title("Overlay: SVC Error vs. DBSCAN Outliers")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()


full_df = pd.concat([train, svm_results.sort_index()], axis=1)

misclassified_extroverts = full_df[(full_df['y_true'] == 1) & (full_df['oof_pred'] == 0)].copy()
misclassified_introverts = full_df[(full_df['y_true'] == 0) & (full_df['oof_pred'] == 1)].copy()


misclassified_extroverts 


misclassified_introverts


features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

full_df['Group'] = full_df['Personality']
misclassified_introverts['Group'] = 'Misclassified Introvert'
misclassified_extroverts['Group'] = 'Misclassified Extrovert'
combined = pd.concat([full_df, misclassified_introverts, misclassified_extroverts])


for feature in features:
    ct = pd.crosstab(combined[feature], combined['Group'])
    ct_norm = ct.divide(ct.sum(axis=0), axis=1)
    
    ax = ct_norm.T.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='tab20')
    
    plt.title(f'Stacked Bar Plot of {feature} by Group (Proportions)')
    plt.xlabel('Group')
    plt.ylabel('Proportion')
    plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


for f in features:
    combined[f"{f}_scaled"] = X_scaled_df[f]

features_scaled = [c for c in combined.columns if 'scaled' in c]
df_intr = combined[combined['Group'].isin(['Extrovert','Misclassified Introvert'])].groupby('Group')[features_scaled].mean()
df_ext = combined[combined['Group'].isin(['Introvert', 'Misclassified Extrovert'])].groupby('Group')[features_scaled].mean()


def plot_spider(data, title, ax=None, colors=None):
    """
    Plots a radar (spider) chart for comparing multiple groups on multiple features.
    Returns:
    - ax: The matplotlib axis object with the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    labels = data.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for idx, group in enumerate(data.index):
        values = data.loc[group].tolist()
        values += values[:1]
        color = colors[idx] if colors else None
        ax.plot(angles, values, label=group, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title(title, pad=50)

    return ax


fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))

plot_spider(df_intr, 'Extroverts vs Misclassified Introverts', ax=axes[0], colors =['#B34600', '#E85D5E'] )
plot_spider(df_ext, 'Introverts vs Misclassified Extroverts', ax=axes[1], colors =['#51158C', '#B163FF'] )
['#B163FF', '#FFC985', '#FFD1B3', '#9D9DCC'] 
fig.legend(loc='upper right', bbox_to_anchor=(1.1, 0.9))
plt.tight_layout()
plt.show()


dfs = {
    'Misclassified Extroverts': misclassified_extroverts,
    'Misclassified Introverts': misclassified_introverts
}

for name, df in dfs.items():
    num_nan_rows = df.isna().any(axis=1).sum()
    percent_nan_rows = 100 * num_nan_rows / len(df)
    
    print(f"{name} has {num_nan_rows} rows with at least one NaN, "
          f"which is {percent_nan_rows:.2f}% of the data.")

