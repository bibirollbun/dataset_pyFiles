# !pip install pillow
# from PIL import Image

# image_path = 'picture1.png'
# img = Image.open(image_path)
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# image_path = 'picture2.png'
# img = Image.open(image_path)
# plt.imshow(img)
# plt.axis('off')
# plt.show()


!pip install catboost
!pip install utilsforecast
!pip install category_encoders


# %autosave 60

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# ===========================
# Ğ‘Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğµ Ğ±Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞºĞ¸ Ğ¸ ÑƒÑ‚Ğ¸Ğ»Ğ¸Ñ‚Ñ‹
# ===========================
import os
import time
import datetime
import warnings
from itertools import product

import numpy as np
import pandas as pd

# ===========================
# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ¸ Ğ¸ Ğ¾Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğµ
# ===========================
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

from IPython.display import display
%matplotlib inline

warnings.filterwarnings('ignore')

# ===========================
# Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
# ===========================
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# utilsforecast
from utilsforecast.plotting import plot_series

# ===========================
# ĞŸÑ€Ğ¾Ğ³Ñ€ĞµÑ�Ñ�-Ğ±Ğ°Ñ€Ñ‹
# ===========================
from tqdm import tqdm

# ===========================
# ĞœĞ°ÑˆĞ¸Ğ½Ğ½Ğ¾Ğµ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ¸ Ğ¿Ñ€ĞµĞ¿Ñ€Ğ¾Ñ†ĞµÑ�Ñ�Ğ¸Ğ½Ğ³
# ===========================
# sklearn - Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ¸ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸
from sklearn.cluster import (
    DBSCAN, KMeans, MeanShift, OPTICS, AgglomerativeClustering
)
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import (
    GridSearchCV, ParameterGrid, RandomizedSearchCV,
    TimeSeriesSplit, train_test_split, StratifiedKFold
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# yellowbrick
from yellowbrick.cluster import KElbowVisualizer

# scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import estimate_bandwidth

# ===========================
# Boosting Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
# ===========================
from catboost import CatBoostClassifier, CatBoostRegressor
import catboost as cb

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

# ===========================
# Ğ“Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¸Ñ‡ĞµÑ�ĞºĞ¸Ğ¹ Ğ¿Ğ¾Ğ¸Ñ�Ğº
# ===========================
from hyperopt import fmin, tpe, Trials, STATUS_OK, hp

# ===========================
# ĞšĞ¾Ğ´Ğ¸Ñ€Ğ¾Ğ²Ñ‰Ğ¸ĞºĞ¸ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
# ===========================
import category_encoders as ce

# ===========================
# Ğ˜Ğ½Ñ‚ĞµÑ€Ğ¿Ñ€ĞµÑ‚Ğ¸Ñ€ÑƒĞµĞ¼Ğ¾Ñ�Ñ‚ÑŒ (SHAP)
# ===========================
import shap


# uploaded = files.upload()


# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
# sample_submission = pd.read_csv('sample_submission.csv')

train = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')
test = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')
sample_submission = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/sample_submission.csv')


train


# ĞŸĞµÑ€ĞµĞ¸Ğ¼ĞµĞ½Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ°
train['target'] = train['Exited']

# Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ Ğ½ĞµĞ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¸Ğ²Ğ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
drop_cols = ['id', 'CustomerId', 'Surname', 'Exited']
model_features = [col for col in train.columns if col not in drop_cols and col != 'target']

X_train, X_val, y_train, y_val = train_test_split(
    train[model_features], train['target'],
    test_size=0.2, random_state=42, stratify=train['target']
)

# Ñ�Ñ‚Ñ€Ğ¾ĞºĞ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
categorical_features = ['Geography', 'Gender']


from catboost import CatBoostClassifier


model_catboost = CatBoostClassifier(
    iterations=200,
    random_state=14,
    verbose=0
)

model_catboost.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_val, y_val),
    verbose=10
)

predictions_proba_val = model_catboost.predict_proba(X_val)[:, 1]


from sklearn.metrics import roc_auc_score
print("ROC-AUC:", roc_auc_score(y_val, predictions_proba_val))


X_test = test[model_features]
predictions_proba_test = model_catboost.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test['id'],
    'Exited': predictions_proba_test
})
submission.to_csv('submission_best.csv', index=False)


# Shrink model to first 176 iterations.
# ROC-AUC: 0.9375905896743512


# ĞŸĞµÑ€ĞµĞ¸Ğ¼ĞµĞ½Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚Ğ°
train['target'] = train['Exited']

# Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ Ğ½ĞµĞ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¸Ğ²Ğ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
drop_cols = ['id', 
             # 'CustomerId', 
             # 'Surname', 
             'Exited']
model_features = [col for col in train.columns if col not in drop_cols and col != 'target']

X_train, X_val, y_train, y_val = train_test_split(
    train[model_features], train['target'],
    test_size=0.2, random_state=42, stratify=train['target']
)

# Ñ�Ñ‚Ñ€Ğ¾ĞºĞ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
categorical_features = ['Geography', 'Gender', 'Surname']

model_catboost = CatBoostClassifier(
    iterations=200,
    random_state=14,
    verbose=0
)

model_catboost.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_val, y_val),
    verbose=10
)

predictions_proba_val = model_catboost.predict_proba(X_val)[:, 1]

print("ROC-AUC:", roc_auc_score(y_val, predictions_proba_val))

X_test = test[model_features]
predictions_proba_test = model_catboost.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'id': test['id'],
    'Exited': predictions_proba_test
})
submission.to_csv('submission3.csv', index=False)


# ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ°Ğ»Ğ³Ğ¾Ñ€Ğ¸Ñ‚Ğ¼Ğ° KMeans++
kmeans_params = {
    'init': 'k-means++',
    'n_init': 10,
    'max_iter': 300,
    'tol': 0.0001,
    'random_state': 42,
    'algorithm': 'elkan'
}

# Ğ‘ĞµÑ€Ñ‘Ğ¼ Ğ½ÑƒĞ¶Ğ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ğ¸Ğ· Ğ´Ğ°Ñ‚Ğ°Ñ„Ñ€ĞµĞ¹Ğ¼Ğ° df
X3 = train[['CreditScore', 'EstimatedSalary', 'Age']].values

# Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ´Ğ»Ñ� Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ� inertia (Ñ�ÑƒĞ¼Ğ¼Ğ° ĞºĞ²Ğ°Ğ´Ñ€Ğ°Ñ‚Ğ¾Ğ² Ñ€Ğ°Ñ�Ñ�Ñ‚Ğ¾Ñ�Ğ½Ğ¸Ğ¹ Ğ´Ğ¾ Ñ†ĞµĞ½Ñ‚Ñ€Ğ¾Ğ² ĞºĞ»Ğ°Ñ�Ñ‚ĞµÑ€Ğ¾Ğ²)
inertia = []
for n in range(1, 11):
    algorithm = KMeans(n_clusters=n, **kmeans_params)
    algorithm.fit(X3)
    inertia.append(algorithm.inertia_)

# Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� "Ğ»Ğ¾ĞºÑ‚Ñ�" (elbow method) Ğ´Ğ»Ñ� Ğ¾Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ� Ñ‡Ğ¸Ñ�Ğ»Ğ° ĞºĞ»Ğ°Ñ�Ñ‚ĞµÑ€Ğ¾Ğ²
plt.figure(figsize=(15, 6))
plt.plot(np.arange(1, 11), inertia, 'o-', alpha=0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()


model = KMeans(**kmeans_params)
visualizer = KElbowVisualizer(model, k=(2,10))

visualizer.fit(X3)
visualizer.show()
plt.show()


N_CLUSTERS = 4


algorithm = KMeans(n_clusters = N_CLUSTERS, **kmeans_params)
algorithm.fit(X3)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_

y_kmeans = algorithm.fit_predict(X3)
train['cluster_kmeans'] = y_kmeans
train.groupby('cluster_kmeans').size()


def plot_3d(colname):
    trace1 = go.Scatter3d(
        x=train['Age'],
        y=train['EstimatedSalary'],
        z=train['CreditScore'],
        mode='markers',
        marker=dict(
            color=train[colname],  # Ñ†Ğ²ĞµÑ‚ Ğ¿Ğ¾ Ğ²Ñ‹Ğ±Ñ€Ğ°Ğ½Ğ½Ğ¾Ğ¹ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞµ
            size=10,
            line=dict(
                color=train[colname],
                width=2  # Ğ¾Ğ±Ñ‹Ñ‡Ğ½Ğ¾ Ğ»Ğ¸Ğ½Ğ¸Ñ� Ğ½Ğµ Ñ‚Ğ°ĞºĞ°Ñ� Ñ‚Ğ¾Ğ»Ñ�Ñ‚Ğ°Ñ�
            ),
            opacity=0.8
        )
    )
    data = [trace1]
    layout = go.Layout(
        title='Clusters with Age, EstimatedSalary and CreditScore',
        scene=dict(
            xaxis=dict(title='Age'),
            yaxis=dict(title='EstimatedSalary'),
            zaxis=dict(title='CreditScore')
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)


plot_3d('cluster_kmeans')


# Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼ Ğ¼ĞµÑ‚Ğ¾Ğ´ Ward, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğ¹ Ğ¾Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµÑ‚ Ğ¿Ğ¾ ĞºÑ€Ğ¸Ñ‚ĞµÑ€Ğ¸Ñ� Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğ¹ Ñ�ÑƒĞ¼Ğ¼Ñ‹ ĞºĞ²Ğ°Ğ´Ñ€Ğ°Ñ‚Ğ¾Ğ² Ñ€Ğ°Ñ�Ñ�Ñ‚Ğ¾Ñ�Ğ½Ğ¸Ğ¹ Ğ¼ĞµĞ¶Ğ´Ñƒ Ğ¿Ğ°Ñ€Ğ°Ğ¼Ğ¸ Ñ‚Ğ¾Ñ‡ĞµĞº Ğ´Ğ²ÑƒÑ… ĞºĞ»Ğ°Ñ�Ñ‚ĞµÑ€Ğ¾Ğ²
H_cluster = linkage(X3,'ward')

# Ğ¿Ğ¾Ñ�Ñ‚Ñ€Ğ¾Ğ¸Ğ¼ Ğ³Ñ€Ğ°Ñ„Ğ¸Ğº
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('num of clusters')
plt.ylabel('distance')

dendrogram(
    H_cluster,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    orientation='right'
)
plt.show()


# ĞšĞ°ĞºĞ¾Ğ¹ ÑƒÑ€Ğ¾Ğ²ĞµĞ½ÑŒ Ñ€Ğ°Ğ·Ğ±Ğ¸ĞµĞ½Ğ¸Ñ� Ğ²Ñ‹Ğ±Ñ€Ğ°Ñ‚ÑŒ?

# ĞŸĞ¾Ğ¿Ñ€Ğ¾Ğ±ÑƒĞµĞ¼ Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ¸Ñ‚ÑŒÑ�Ñ� Ğº silhouette score


from sklearn.metrics import silhouette_score


# Ğ¿ĞµÑ€ĞµĞ±ĞµÑ€ĞµĞ¼ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° ĞºĞ»Ğ°Ñ�Ñ‚ĞµÑ€Ğ¾Ğ² Ğ¾Ñ‚ 2 Ğ´Ğ¾ 4
params = list(range(2, 4))
num_clusters = []
sil_score = []

for n_clusters in params:
    aglom_clust = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(X3)
    num_clusters.append(len(np.unique(aglom_clust.labels_)))
    sil_score.append(silhouette_score(X3, aglom_clust.labels_))

fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.set_ylabel('silhouette score', color=color)  # we already handled the x-label with ax1
ax1.plot(params, sil_score, color=color)
ax1.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

n_clusters = params[np.argmax(sil_score)]
print(f"n_clusters: {n_clusters}")


aglom_clust = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(X3)

train['cluster_agglomerative'] = aglom_clust.labels_

train.groupby('cluster_agglomerative').size()


plot_3d('cluster_agglomerative')


# # Ğ¿ĞµÑ€ĞµĞ·Ğ°Ğ³Ñ€ÑƒĞ¶Ñƒ
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
# sample_submission = pd.read_csv('sample_submission.csv')

train = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')
test = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')
sample_submission = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/sample_submission.csv')


# Ğ£Ğ´Ğ°Ğ»Ñ�ĞµĞ¼ Ğ»Ğ¸ÑˆĞ½Ğ¸Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸
excluded = ['CustomerId', 'Surname', 'id']
columns_to_plot = [col for col in train.columns if col not in excluded]

# ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ñ�ĞµÑ‚ĞºĞ¸
n_cols = 3
n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols

plt.figure(figsize=(n_cols * 6, n_rows * 4))

for i, col in enumerate(columns_to_plot):
    plt.subplot(n_rows, n_cols, i + 1)

    if train[col].dtype == 'object' or train[col].nunique() < 20:
        # ĞšĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¸ Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
        sns.countplot(x=train[col], palette='pastel', edgecolor='black')
        plt.ylabel('Count')
    else:
        # Ğ§Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
        sns.histplot(
            data=train,
            x=col,
            kde=False,
            edgecolor="black",
            alpha=1,
            shrink=1,
            stat="density",
            color='#00833f',
            bins=30
        )
        sns.kdeplot(data=train, x=col, color="#c20430", linewidth=2)

        # Ğ’ĞµÑ€Ñ‚Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ»Ğ¸Ğ½Ğ¸Ğ¸
        plt.axvline(train[col].median(), color='orange', linestyle='--', label='ĞœĞµĞ´Ğ¸Ğ°Ğ½Ğ°')
        plt.axvline(train[col].mean(), color='blue', linestyle=':', label='Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞµ')

        plt.legend(title="Ğ¡Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ°", fontsize=8, title_fontsize=9)

        plt.ylabel('ĞŸĞ»Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ñ‚ÑŒ')

    # Ğ�Ğ±Ñ‰Ğ¸Ğµ Ğ½Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ¸
    plt.title(col, fontsize=11, fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)

plt.tight_layout()

plt.show()


# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ¸ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ñ…
y_var = 'CreditScore'
excluded = ['CustomerId', 'Surname', 'id', y_var]

# Ğ‘ĞµÑ€Ñ‘Ğ¼ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ (Ğ¸ Ğ½Ğµ Ğ¸Ñ�ĞºĞ»Ñ�Ñ‡Ñ‘Ğ½Ğ½Ñ‹Ğµ)
x_vars = [col for col in train.select_dtypes(include='number').columns
          if col not in excluded and train[col].nunique() > 2]

# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ¸ Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¾Ğ²
n_cols = 3
n_rows = (len(x_vars) + n_cols - 1) // n_cols

plt.figure(figsize=(n_cols * 6, n_rows * 5))
sns.set_style("whitegrid")
palette = sns.color_palette("coolwarm", as_cmap=True)

for i, x_var in enumerate(x_vars):
    plt.subplot(n_rows, n_cols, i + 1)

    # Scatterplot Ñ� hue Ğ¸ size = CreditScore
    sns.scatterplot(
        data=train,
        x=x_var,
        y=y_var,
        hue=y_var,
        palette=palette,
        size=y_var,
        sizes=(20, 100),
        s=80,
        alpha=0.5,
        edgecolor="black",
        linewidth=0.5,
        legend=False
    )

    # Ğ›Ğ¸Ğ½Ğ¸Ñ� Ñ‚Ñ€ĞµĞ½Ğ´Ğ°
    sns.regplot(
        data=train,
        x=x_var,
        y=y_var,
        scatter=False,
        lowess=True,
        color='red',
        line_kws={'linewidth': 2, 'linestyle': '--'}
    )

    plt.title(f'{y_var} vs {x_var}', fontsize=11, fontweight='bold')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))

sns.histplot(
    data=train,
    x='Exited',
    palette=['#00833f', '#c20430'],  # 0 â€” Ğ¾Ñ�Ñ‚Ğ°Ğ»Ñ�Ñ�, 1 â€” ÑƒÑˆÑ‘Ğ»
    discrete=True,
    shrink=0.7,
    alpha=1,
    edgecolor='black',
    stat='count'
)

plt.xlabel('Ğ¡Ñ‚Ğ°Ñ‚ÑƒÑ� ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ° (Exited)', fontsize=12)
plt.ylabel('ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ¾Ğ²', fontsize=12)
plt.title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ¾Ğ² Ğ¿Ğ¾ Ñ�Ñ‚Ğ°Ñ‚ÑƒÑ�Ñƒ Exited', fontsize=14, fontweight='bold')
plt.xticks([0, 1], ['Ğ�Ñ�Ñ‚Ğ°Ğ»Ñ�Ñ�', 'Ğ£ÑˆÑ‘Ğ»'])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Ğ£Ğ´Ğ°Ğ»Ñ�ĞµĞ¼ Ğ»Ğ¸ÑˆĞ½Ğ¸Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸
excluded = ['CustomerId', 'Surname', 'id']
columns_to_plot = [col for col in test.columns if col not in excluded]

# ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ñ�ĞµÑ‚ĞºĞ¸
n_cols = 3
n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols

plt.figure(figsize=(n_cols * 6, n_rows * 4))

for i, col in enumerate(columns_to_plot):
    plt.subplot(n_rows, n_cols, i + 1)

    if test[col].dtype == 'object' or test[col].nunique() < 20:
        # ĞšĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¸ Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
        sns.countplot(x=train[col], palette='pastel', edgecolor='black')
        plt.ylabel('Count')
    else:
        # Ğ§Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
        sns.histplot(
            data=test,
            x=col,
            kde=False,
            edgecolor="black",
            alpha=1,
            shrink=1,
            stat="density",
            color='#00833f',
            bins=30
        )
        sns.kdeplot(data=test, x=col, color="#c20430", linewidth=2)

        # Ğ’ĞµÑ€Ñ‚Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ»Ğ¸Ğ½Ğ¸Ğ¸
        plt.axvline(test[col].median(), color='orange', linestyle='--', label='ĞœĞµĞ´Ğ¸Ğ°Ğ½Ğ°')
        plt.axvline(test[col].mean(), color='blue', linestyle=':', label='Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞµ')

        plt.legend(title="Ğ¡Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ°", fontsize=8, title_fontsize=9)

        plt.ylabel('ĞŸĞ»Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ñ‚ÑŒ')

    # Ğ�Ğ±Ñ‰Ğ¸Ğµ Ğ½Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ¸
    plt.title(col, fontsize=11, fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)

plt.tight_layout()

plt.show()


# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ¸ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ñ…
y_var = 'CreditScore'
excluded = ['CustomerId', 'Surname', 'id', y_var]

# Ğ‘ĞµÑ€Ñ‘Ğ¼ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ (Ğ¸ Ğ½Ğµ Ğ¸Ñ�ĞºĞ»Ñ�Ñ‡Ñ‘Ğ½Ğ½Ñ‹Ğµ)
x_vars = [col for col in test.select_dtypes(include='number').columns
          if col not in excluded and test[col].nunique() > 2]

# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ¸ Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¾Ğ²
n_cols = 3
n_rows = (len(x_vars) + n_cols - 1) // n_cols

plt.figure(figsize=(n_cols * 6, n_rows * 5))
sns.set_style("whitegrid")
palette = sns.color_palette("coolwarm", as_cmap=True)

for i, x_var in enumerate(x_vars):
    plt.subplot(n_rows, n_cols, i + 1)

    # Scatterplot Ñ� hue Ğ¸ size = CreditScore
    sns.scatterplot(
        data=test,
        x=x_var,
        y=y_var,
        hue=y_var,
        palette=palette,
        size=y_var,
        sizes=(20, 100),
        s=80,
        alpha=0.5,
        edgecolor="black",
        linewidth=0.5,
        legend=False
    )

    # Ğ›Ğ¸Ğ½Ğ¸Ñ� Ñ‚Ñ€ĞµĞ½Ğ´Ğ°
    sns.regplot(
        data=test,
        x=x_var,
        y=y_var,
        scatter=False,
        lowess=True,
        color='red',
        line_kws={'linewidth': 2, 'linestyle': '--'}
    )

    plt.title(f'{y_var} vs {x_var}', fontsize=11, fontweight='bold')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)

plt.tight_layout()
plt.show()


# ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¸ Ğ´Ğ¾Ğ»Ñ� Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ² Ğ¿Ğ¾ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ°Ğ¼
missing_info = train.isna().sum()
missing_info

# Ğ½ĞµÑ‚ Ñ�Ğ¼Ñ‹Ñ�Ğ»Ğ° Ñ‡Ñ‚Ğ¾-Ñ‚Ğ¾ Ğ´ĞµĞ»Ğ°Ñ‚ÑŒ Ñ� Ñ�Ñ‚Ğ¸Ğ¼, Ñ‚Ğ°Ğº ĞºĞ°Ğº Ğ½ÑƒĞ»Ğ¸ Ğ·Ğ°Ñ‡Ğ°Ñ�Ñ‚ÑƒÑ� Ğ¸Ğ¼ĞµÑ�Ñ‚ Ğ·Ğ° Ñ�Ğ¾Ğ±Ğ¾Ğ¹ Ğ»Ğ¾Ğ³Ğ¸ĞºÑƒ


train


# ÑƒĞ´Ğ°Ğ»Ğ¸Ğ¼ Ğ»Ğ¸ÑˆĞ½ĞµĞµ


cols_to_drop = ['id', 'CustomerId', 'Surname']

train = train.drop(columns=cols_to_drop, errors='ignore')
test = test.drop(columns=cols_to_drop, errors='ignore')


# Ñ�Ğ¿Ğ»Ğ¸Ñ‚


y = train['Exited']
X = train.drop(columns=['Exited'])

# Ğ¡Ğ¿Ğ»Ğ¸Ñ‚ Ñ� Ñ�Ñ‚Ñ€Ğ°Ñ‚Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸ĞµĞ¹ Ğ¿Ğ¾ y
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    test_size=0.2,
    random_state=42
)


y_train


# ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº: ĞºĞ»Ğ¸ĞµĞ½Ñ‚ Ñ� Ğ½ÑƒĞ»ĞµĞ²Ñ‹Ğ¼ Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ¾Ğ¼
X_train['balance_is_zero'] = (X_train['Balance'] == 0).astype(int)
X_test['balance_is_zero'] = (X_test['Balance'] == 0).astype(int)


# Ğ�Ğ±ÑŠÑ�Ğ²Ğ¸Ğ¼ Ñ�Ğ¿Ğ¸Ñ�ĞºĞ¸ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ñ…
log_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
poly_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

# Ğ›Ğ¾Ğ³Ğ°Ñ€Ğ¸Ñ„Ğ¼Ñ‹
for col in log_features:
    X_train[f'{col}_log'] = np.log1p(X_train[col])
    X_test[f'{col}_log'] = np.log1p(X_test[col])

# ĞŸĞ¾Ğ»Ğ¸Ğ½Ğ¾Ğ¼Ñ‹ Ñ�Ñ‚ĞµĞ¿ĞµĞ½Ğ¸ 2 Ğ¸ 3
for col in poly_features:
    X_train[f'{col}_sq'] = X_train[col] ** 2
    X_train[f'{col}_cube'] = X_train[col] ** 3
    X_test[f'{col}_sq'] = X_test[col] ** 2
    X_test[f'{col}_cube'] = X_test[col] ** 3


### Ğ˜Ğ´ĞµĞ¼ Ğ´Ğ°Ğ»ĞµĞµ


# ĞšĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸
cat_cols = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']

# ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²ĞµĞ½Ğ½Ñ‹Ğµ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ Ğ¸ Ñ�Ğ¾Ğ¾Ñ‚Ğ²ĞµÑ‚Ñ�Ñ‚Ğ²ÑƒÑ�Ñ‰Ğ¸Ğµ Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ¸
stats_dict = {
    'CreditScore': ['mean', 'median', 'sum'],
    'Age': ['mean', 'median', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan],
    'Tenure': ['mean', 'median', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan],
    'Balance': ['mean', 'median', 'sum'],
    'EstimatedSalary': ['mean', 'median', 'sum']
}

# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ñ‘Ğ¼ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸Ğº Ğ¿Ğ¾ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ğ¼
for cat_col in cat_cols:
    for num_col, stats in stats_dict.items():
        grouped = X_train.groupby(cat_col)[num_col].agg(stats)
        grouped.columns = [f'{num_col}_{cat_col}_{stat if isinstance(stat, str) else "mode"}' for stat in stats]
        X_train = X_train.merge(grouped, how='left', left_on=cat_col, right_index=True)
        X_test = X_test.merge(grouped, how='left', left_on=cat_col, right_index=True)

# Ğ¡Ñ‡Ğ¸Ñ‚Ğ°ĞµĞ¼ Ğ¿Ñ€Ğ¾Ğ¿Ğ¾Ñ€Ñ†Ğ¸Ğ¸ Ğ¸ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ id Ğ¿Ğ¾ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�Ğ¼
for cat_col in cat_cols:
    # ĞŸÑ€Ğ¾Ğ¿Ğ¾Ñ€Ñ†Ğ¸Ğ¸
    proportions = X_train[cat_col].value_counts(normalize=True).rename(f'{cat_col}_proportion')
    X_train = X_train.merge(proportions, how='left', left_on=cat_col, right_index=True)
    X_test = X_test.merge(proportions, how='left', left_on=cat_col, right_index=True)
    
    # ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ id
    counts = X_train[cat_col].value_counts().rename(f'{cat_col}_count')
    X_train = X_train.merge(counts, how='left', left_on=cat_col, right_index=True)
    X_test = X_test.merge(counts, how='left', left_on=cat_col, right_index=True)


X_train


# Ğ¡Ñ€ĞµĞ´Ğ½Ğ¸Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ¿Ğ¾ Geography Ğ¸ Gender â€” Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ğ¿Ğ¾ train
avg_credit_score_geo_train = X_train.groupby('Geography')['CreditScore'].mean()
avg_balance_geo_train = X_train.groupby('Geography')['Balance'].mean()
avg_salary_gender_train = X_train.groupby('Gender')['EstimatedSalary'].mean()

# ĞŸĞ¾Ñ€Ğ¾Ğ³ Ğ´Ğ»Ñ� Ğ±Ğ¾Ğ³Ğ°Ñ‚Ñ‹Ñ… (Ğ¿Ğ¾ train)
high_balance_threshold = X_train['Balance'].quantile(0.90)

# ĞœĞ°Ğ¿Ğ¿Ğ¸Ğ½Ğ³ Ğ¿Ğ¾ train
X_train['avg_credit_score_geo'] = X_train['Geography'].map(avg_credit_score_geo_train)
X_train['avg_balance_geo'] = X_train['Geography'].map(avg_balance_geo_train)
X_train['avg_salary_gender'] = X_train['Gender'].map(avg_salary_gender_train)

# Ğ�Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
X_train['balance_to_salary_ratio'] = X_train['Balance'] / (X_train['EstimatedSalary'] + 1)
X_train['tenure_per_age'] = X_train['Tenure'] / (X_train['Age'] + 1)
X_train['balance_minus_salary'] = X_train['Balance'] - X_train['EstimatedSalary']
X_train['salary_to_age_ratio'] = X_train['EstimatedSalary'] / (X_train['Age'] + 1)
X_train['balance_to_tenure_ratio'] = X_train['Balance'] / (X_train['Tenure'] + 1)

X_train['credit_score_gap_from_avg_geo'] = X_train['CreditScore'] - X_train['avg_credit_score_geo']
X_train['balance_gap_from_avg_geo'] = X_train['Balance'] - X_train['avg_balance_geo']
X_train['salary_gap_from_avg_gender'] = X_train['EstimatedSalary'] - X_train['avg_salary_gender']

X_train['is_young_and_rich_flag'] = ((X_train['Age'] < 30) & (X_train['Balance'] > high_balance_threshold)).astype(int)

X_train['ltv_proxy'] = X_train['EstimatedSalary'] * X_train['Tenure']
X_train['is_churn_risk_profile'] = ((X_train['IsActiveMember'] == 0) & (X_train['Tenure'] < 2)).astype(int)
X_train['potential_profitability_index'] = X_train['EstimatedSalary'] / (X_train['NumOfProducts'] + 1)

# Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸
X_train.drop(columns=['avg_credit_score_geo', 'avg_balance_geo', 'avg_salary_gender'], inplace=True)


# ĞœĞ°Ğ¿Ğ¿Ğ¸Ğ½Ğ³ Ğ¿Ğ¾ train
X_test['avg_credit_score_geo'] = X_test['Geography'].map(avg_credit_score_geo_train)
X_test['avg_balance_geo'] = X_test['Geography'].map(avg_balance_geo_train)
X_test['avg_salary_gender'] = X_test['Gender'].map(avg_salary_gender_train)

# Ğ�Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
X_test['balance_to_salary_ratio'] = X_test['Balance'] / (X_test['EstimatedSalary'] + 1)
X_test['tenure_per_age'] = X_test['Tenure'] / (X_test['Age'] + 1)
X_test['balance_minus_salary'] = X_test['Balance'] - X_test['EstimatedSalary']
X_test['salary_to_age_ratio'] = X_test['EstimatedSalary'] / (X_test['Age'] + 1)
X_test['balance_to_tenure_ratio'] = X_test['Balance'] / (X_test['Tenure'] + 1)

X_test['credit_score_gap_from_avg_geo'] = X_test['CreditScore'] - X_test['avg_credit_score_geo']
X_test['balance_gap_from_avg_geo'] = X_test['Balance'] - X_test['avg_balance_geo']
X_test['salary_gap_from_avg_gender'] = X_test['EstimatedSalary'] - X_test['avg_salary_gender']

X_test['is_young_and_rich_flag'] = ((X_test['Age'] < 30) & (X_test['Balance'] > high_balance_threshold)).astype(int)

X_test['ltv_proxy'] = X_test['EstimatedSalary'] * X_test['Tenure']
X_test['is_churn_risk_profile'] = ((X_test['IsActiveMember'] == 0) & (X_test['Tenure'] < 2)).astype(int)
X_test['potential_profitability_index'] = X_test['EstimatedSalary'] / (X_test['NumOfProducts'] + 1)

# Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸
X_test.drop(columns=['avg_credit_score_geo', 'avg_balance_geo', 'avg_salary_gender'], inplace=True)


### Ğ´Ğ°Ğ»ĞµĞµ


# Binning tenure
X_train['tenure_stage'] = pd.cut(X_train['Tenure'], 
                                 bins=[-1, 1, 4, np.inf], 
                                 labels=['new', 'developing', 'established'])

# Ğ˜Ğ½Ğ´ĞµĞºÑ� Ğ±Ğ¾Ğ³Ğ°Ñ‚Ñ�Ñ‚Ğ²Ğ°
X_train['wealth_index'] = (X_train['Balance'] + X_train['EstimatedSalary']) / (X_train['Age'] + 1)

# Ğ¡ĞºĞ¾Ñ€Ñ€ĞµĞºÑ‚Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğ¹ Ñ€Ğ¸Ñ�Ğº Ğ¿Ğ¾ Ğ²Ğ¾Ğ·Ñ€Ğ°Ñ�Ñ‚Ñƒ
X_train['risk_age_score'] = X_train['CreditScore'] / np.log1p(X_train['Age'])

# ĞŸĞ¾Ñ‚ĞµĞ½Ñ†Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Ğ°Ğ¿Ñ�ĞµĞ¹Ğ» Ğ¿Ğ¾ ĞºÑ€ĞµĞ´Ğ¸Ñ‚ĞºĞµ
X_train['activation_opportunity'] = ((X_train['HasCrCard'] == 0) & (X_train['IsActiveMember'] == 1)).astype(int)

# Ğ“Ñ€ÑƒĞ¿Ğ¿Ğ¾Ğ²Ñ‹Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ¿Ğ¾ train
geo_credit_mean_train = X_train.groupby('Geography')['CreditScore'].mean()
geo_balance_std_train = X_train.groupby('Geography')['Balance'].std()

# Ğ˜Ğ½Ğ´ĞµĞºÑ� ĞºÑ€ĞµĞ´Ğ¸Ñ‚Ğ½Ğ¾Ğ³Ğ¾ Ñ€Ğ¸Ñ�ĞºĞ° Ğ¿Ğ¾ Ğ³ĞµĞ¾Ğ³Ñ€Ğ°Ñ„Ğ¸Ğ¸
X_train['geo_credit_risk_index'] = X_train['Geography'].map(geo_credit_mean_train)

# Ğ¡Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğµ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ğµ Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ° Ğ¿Ğ¾ Ğ³ĞµĞ¾Ğ³Ñ€Ğ°Ñ„Ğ¸Ğ¸
X_train['geo_balance_std'] = X_train['Geography'].map(geo_balance_std_train)

# Binning Ğ¿Ğ¾ CreditScore
X_train['credit_score_band'] = pd.cut(X_train['CreditScore'],
                                      bins=[-np.inf, 600, 750, np.inf],
                                      labels=['low', 'medium', 'high'])


# Binning tenure
X_test['tenure_stage'] = pd.cut(X_test['Tenure'], 
                                bins=[-1, 1, 4, np.inf], 
                                labels=['new', 'developing', 'established'])

# Ğ˜Ğ½Ğ´ĞµĞºÑ� Ğ±Ğ¾Ğ³Ğ°Ñ‚Ñ�Ñ‚Ğ²Ğ°
X_test['wealth_index'] = (X_test['Balance'] + X_test['EstimatedSalary']) / (X_test['Age'] + 1)

# Ğ¡ĞºĞ¾Ñ€Ñ€ĞµĞºÑ‚Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğ¹ Ñ€Ğ¸Ñ�Ğº Ğ¿Ğ¾ Ğ²Ğ¾Ğ·Ñ€Ğ°Ñ�Ñ‚Ñƒ
X_test['risk_age_score'] = X_test['CreditScore'] / np.log1p(X_test['Age'])

# ĞŸĞ¾Ñ‚ĞµĞ½Ñ†Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Ğ°Ğ¿Ñ�ĞµĞ¹Ğ» Ğ¿Ğ¾ ĞºÑ€ĞµĞ´Ğ¸Ñ‚ĞºĞµ
X_test['activation_opportunity'] = ((X_test['HasCrCard'] == 0) & (X_test['IsActiveMember'] == 1)).astype(int)

# Ğ˜Ğ½Ğ´ĞµĞºÑ� ĞºÑ€ĞµĞ´Ğ¸Ñ‚Ğ½Ğ¾Ğ³Ğ¾ Ñ€Ğ¸Ñ�ĞºĞ° Ğ¿Ğ¾ Ğ³ĞµĞ¾Ğ³Ñ€Ğ°Ñ„Ğ¸Ğ¸ (Ğ¿Ğ¾ train)
X_test['geo_credit_risk_index'] = X_test['Geography'].map(geo_credit_mean_train)

# Ğ¡Ñ‚Ğ°Ğ½Ğ´Ğ°Ñ€Ñ‚Ğ½Ğ¾Ğµ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½ĞµĞ½Ğ¸Ğµ Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ° Ğ¿Ğ¾ Ğ³ĞµĞ¾Ğ³Ñ€Ğ°Ñ„Ğ¸Ğ¸ (Ğ¿Ğ¾ train)
X_test['geo_balance_std'] = X_test['Geography'].map(geo_balance_std_train)

# Binning Ğ¿Ğ¾ CreditScore
X_test['credit_score_band'] = pd.cut(X_test['CreditScore'],
                                     bins=[-np.inf, 600, 750, np.inf],
                                     labels=['low', 'medium', 'high'])


missing_train = X_train.isna().mean()
missing_train = missing_train[missing_train > 0]
print("ĞŸÑ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸ Ğ² X_train (Ğ´Ğ¾Ğ»Ñ�):")
print(missing_train)

print("\n" + "-"*40 + "\n")

missing_test = X_test.isna().mean()
missing_test = missing_test[missing_test > 0]
print("ĞŸÑ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸ Ğ² X_test (Ğ´Ğ¾Ğ»Ñ�):")
print(missing_test)


object_cols = X_train.select_dtypes(include='object').columns.tolist()
print(object_cols)


float_cols = X_train.select_dtypes(include='float64').columns.tolist()
print(float_cols)


cat_features = ['Geography', 'Gender']


# Ğ�Ğ±ÑƒÑ‡Ğ°ĞµĞ¼ target encoding Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ğ½Ğ° Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
target_enc = ce.CatBoostEncoder(cols=cat_features)
target_enc.fit(X_train[cat_features], y_train)

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ X_train Ğ¸ X_test
X_train_cb = target_enc.transform(X_train[cat_features]).add_suffix('_cb')
X_test_cb = target_enc.transform(X_test[cat_features]).add_suffix('_cb')

# ĞŸÑ€Ğ¸Ñ�Ğ¾ĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ğ·Ğ°ĞºĞ¾Ğ´Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ğº Ğ¸Ñ�Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğ¼
X_train = X_train.join(X_train_cb)
X_test = X_test.join(X_test_cb)


cat_features = ['Geography_cb', 'Gender_cb']
numeric_features = float_cols
model_features = numeric_features + cat_features


numeric_features


def permutation_importance(
    X_train, y_train, X_test, y_test,
    n_repeats=10, threshold=0.01, metric='accuracy',
    n_top_features=None, plot=True
):
    """
    Ğ ĞµĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ°Ğ»Ğ³Ğ¾Ñ€Ğ¸Ñ‚Ğ¼Ğ° Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ Ğ¿ĞµÑ€ĞµÑ�Ñ‚Ğ°Ğ½Ğ¾Ğ²Ğ¾Ğº (permutation importance)
    Ñ� Ğ²Ğ½ĞµÑˆĞ½Ğ¸Ğ¼Ğ¸ train/test Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ°Ğ¼Ğ¸.
    """

    if metric not in ['accuracy', 'roc_auc']:
        raise ValueError("ĞœĞµÑ‚Ñ€Ğ¸ĞºĞ° Ğ´Ğ¾Ğ»Ğ¶Ğ½Ğ° Ğ±Ñ‹Ñ‚ÑŒ 'accuracy' Ğ¸Ğ»Ğ¸ 'roc_auc'")


    model = CatBoostClassifier(
        depth=5,
        learning_rate=0.1,
        iterations=1000,
        od_type='Iter',
        od_wait=50,
        random_seed=42,
        verbose=False
    )    
    model.fit(X_train, y_train)

    # ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞµ
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if metric == 'roc_auc' else None

    # Ğ’Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµĞ¼ Ğ±Ğ°Ğ·Ğ¾Ğ²ÑƒÑ� Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºÑƒ
    if metric == 'accuracy':
        baseline_score = accuracy_score(y_test, y_pred)
    else:
        baseline_score = roc_auc_score(y_test, y_pred_proba)

    print(f"Ğ‘Ğ°Ğ·Ğ¾Ğ²Ğ°Ñ� {metric} Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸: {baseline_score:.4f}")

    feature_importance = {}
    feature_scores = {}

    for feature in X_test.columns:
        importance_scores = []
        permuted_scores = []

        for _ in range(n_repeats):
            X_test_permuted = X_test.copy()
            X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature].values)

            y_pred_permuted = model.predict(X_test_permuted)

            if metric == 'accuracy':
                permuted_score = accuracy_score(y_test, y_pred_permuted)
            else:
                y_pred_proba_permuted = model.predict_proba(X_test_permuted)[:, 1]
                permuted_score = roc_auc_score(y_test, y_pred_proba_permuted)

            permuted_scores.append(permuted_score)
            importance_scores.append(baseline_score - permuted_score)

        feature_importance[feature] = np.mean(importance_scores)
        feature_scores[feature] = np.mean(permuted_scores)

    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values()),
        'permuted_score': list(feature_scores.values())
    }).sort_values('importance', ascending=False)

    importance_df['percent_decrease'] = (importance_df['importance'] / baseline_score) * 100

    def get_justification(row):
        if row['importance'] <= 0:
            return "ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº Ğ½Ğµ Ğ²Ğ»Ğ¸Ñ�ĞµÑ‚ Ğ½Ğ° ĞºĞ°Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ¸Ğ»Ğ¸ ĞµĞ³Ğ¾ Ğ¿ĞµÑ€ĞµĞ¼ĞµÑˆĞ¸Ğ²Ğ°Ğ½Ğ¸Ğµ ÑƒĞ»ÑƒÑ‡ÑˆĞ°ĞµÑ‚ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ"
        elif row['percent_decrease'] < 1:
            return "Ğ�ĞµĞ·Ğ½Ğ°Ñ‡Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾Ğµ Ğ²Ğ»Ğ¸Ñ�Ğ½Ğ¸Ğµ Ğ½Ğ° Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ (Ğ¼ĞµĞ½ĞµĞµ 1% Ñ�Ğ½Ğ¸Ğ¶ĞµĞ½Ğ¸Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸)"
        elif row['percent_decrease'] < 5:
            return "Ğ£Ğ¼ĞµÑ€ĞµĞ½Ğ½Ğ¾Ğµ Ğ²Ğ»Ğ¸Ñ�Ğ½Ğ¸Ğµ Ğ½Ğ° Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ (1-5% Ñ�Ğ½Ğ¸Ğ¶ĞµĞ½Ğ¸Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸)"
        elif row['percent_decrease'] < 10:
            return "Ğ¡ÑƒÑ‰ĞµÑ�Ñ‚Ğ²ĞµĞ½Ğ½Ğ¾Ğµ Ğ²Ğ»Ğ¸Ñ�Ğ½Ğ¸Ğµ Ğ½Ğ° Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ (5-10% Ñ�Ğ½Ğ¸Ğ¶ĞµĞ½Ğ¸Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸)"
        else:
            return "ĞšÑ€Ğ¸Ñ‚Ğ¸Ñ‡ĞµÑ�ĞºĞ¸ Ğ²Ğ°Ğ¶Ğ½Ñ‹Ğ¹ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°Ğº (Ğ±Ğ¾Ğ»ĞµĞµ 10% Ñ�Ğ½Ğ¸Ğ¶ĞµĞ½Ğ¸Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸)"

    importance_df['justification'] = importance_df.apply(get_justification, axis=1)

    if n_top_features is not None:
        selected_features = importance_df.head(n_top_features)['feature'].tolist()
        importance_df['selected'] = importance_df['feature'].isin(selected_features)
    else:
        selected_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
        importance_df['selected'] = importance_df['importance'] > threshold

    print("\nĞ’Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² (permutation importance):")
    print(importance_df)

    print(f"\nĞ’Ñ‹Ğ±Ñ€Ğ°Ğ½Ğ¾ {len(selected_features)} Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²:")
    for feature in selected_features:
        row = importance_df[importance_df['feature'] == feature].iloc[0]
        print(f"- {feature}: Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ = {row['importance']:.4f}, "
              f"Ñ�Ğ½Ğ¸Ğ¶ĞµĞ½Ğ¸Ğµ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸ = {row['percent_decrease']:.2f}%, "
              f"Ğ¾Ğ±Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ: {row['justification']}")

    if plot and not importance_df.empty:
        plt.figure(figsize=(12, 8))
        colors = ['green' if selected else 'gray' for selected in importance_df['selected']]
        sorted_df = importance_df.sort_values('importance')
        plt.barh(sorted_df['feature'], sorted_df['importance'], color=colors)
        if n_top_features is None:
            plt.axvline(x=threshold, color='red', linestyle='--', label=f'ĞŸĞ¾Ñ€Ğ¾Ğ³Ğ¾Ğ²Ğ¾Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ ({threshold})')
            plt.legend()
        plt.xlabel('Ğ’Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ (Ñ�Ğ½Ğ¸Ğ¶ĞµĞ½Ğ¸Ğµ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸)')
        plt.ylabel('ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸')
        plt.title('Permutation Importance Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show("permutation_importance.png")
        print("Ğ“Ñ€Ğ°Ñ„Ğ¸Ğº Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½ ĞºĞ°Ğº 'permutation_importance.png'")

    return importance_df, selected_features


# permutation_importance(
#     X_train=X_train[model_features],
#     y_train=y_train,
#     X_test=X_test[model_features],
#     y_test=y_test,
#     n_repeats=50,
#     threshold=0.01,
#     metric='roc_auc',
#     n_top_features=None,
#     plot=True
# )


importance_df, selected_features = permutation_importance(
    X_train=X_train[model_features], 
    y_train=y_train, 
    X_test=X_test[model_features], 
    y_test=y_test,
    metric='roc_auc',
    n_repeats=10,
    threshold=0.01,
    n_top_features=None,
    plot=True
)


important_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()


important_features


model_features = important_features


importance_df, selected_features = permutation_importance(
    X_train=X_train[model_features], 
    y_train=y_train, 
    X_test=X_test[model_features], 
    y_test=y_test,
    metric='roc_auc',
    n_repeats=50,
    threshold=0.001,
    n_top_features=None,
    plot=True
)


importance_df


important_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()


important_features


model_features = important_features


model_features = ['balance_gap_from_avg_geo',
 'Age',
 'Age_cube',
 'Age_sq',
 'credit_score_gap_from_avg_geo',
 'CreditScore',
 'EstimatedSalary_NumOfProducts_sum',
 'Age_Gender_median',
 'Tenure_NumOfProducts_mode',
 'wealth_index',
 'potential_profitability_index',
 'salary_gap_from_avg_gender',
 'Age_log',
 'EstimatedSalary_log',
 'risk_age_score',
 'CreditScore_cube',
 'Tenure_IsActiveMember_mean',
 'NumOfProducts_proportion',
 'salary_to_age_ratio',
 'tenure_per_age',
 'Balance',
 'Age_NumOfProducts_mode',
 'ltv_proxy',
 'Balance_log',
 'EstimatedSalary_NumOfProducts_median',
 'EstimatedSalary_IsActiveMember_median',
 'balance_to_salary_ratio',
 'CreditScore_NumOfProducts_sum',
 'Age_NumOfProducts_median',
 'Balance_cube',
 'EstimatedSalary_cube',
 'Tenure_NumOfProducts_mean',
 'CreditScore_IsActiveMember_sum',
 'Balance_NumOfProducts_median',
 'CreditScore_NumOfProducts_median',
 'NumOfProducts',
 'IsActiveMember',
 'Balance_NumOfProducts_sum',
 'EstimatedSalary_Geography_mean',
 'Balance_IsActiveMember_mean',
 'Tenure_Geography_mean',
 'CreditScore_IsActiveMember_mean',
 'Balance_IsActiveMember_sum',
 'EstimatedSalary_Geography_sum',
 'CreditScore_NumOfProducts_mean',
 'EstimatedSalary_NumOfProducts_mean',
 'CreditScore_IsActiveMember_median',
 'Age_HasCrCard_mode',
 'CreditScore_HasCrCard_mean',
 'EstimatedSalary_IsActiveMember_mean',
 'EstimatedSalary_sq',
 'CreditScore_Geography_sum',
 'CreditScore_Geography_mean',
 'IsActiveMember_proportion',
 'CreditScore_HasCrCard_median',
 'Age_Geography_median',
 'geo_balance_std',
 'EstimatedSalary_HasCrCard_sum']


importance_df, selected_features = permutation_importance(
    X_train=X_train[model_features], 
    y_train=y_train, 
    X_test=X_test[model_features], 
    y_test=y_test,
    metric='roc_auc',
    n_repeats=50,
    threshold=0.001,
    n_top_features=None,
    plot=True
)


final_important_features = importance_df['feature'].tolist()


important_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
important_features
model_features = important_features


model_features


model_catboost = CatBoostClassifier(
    iterations=200,
    random_state=14,
    verbose=0
)

model_catboost.fit(
    X_train[model_features], y_train,
    eval_set=(X_test[model_features], y_test),
    verbose=10
)

predictions_proba_val = model_catboost.predict_proba(X_test[model_features])[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, predictions_proba_val))


import shap
# compute the SHAP values for every prediction in the validation dataset
explainer = shap.TreeExplainer(model_catboost)
shap_values = explainer.shap_values(X_test[model_features])
shap.summary_plot(shap_values, X_test[model_features], plot_type="bar")


### Ğ Ğ°Ğ½ĞµĞµ Ñ� Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ» val Ñ�ĞµÑ‚. Ğ� Ñ‡Ñ‚Ğ¾ ĞµÑ�Ğ»Ğ¸ Ğ¿Ğ¾Ğ´Ğ¾Ğ±Ñ€Ğ°Ñ‚ÑŒ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ¿Ğ¾ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ¼Ñƒ roc-auc Ğ¿Ğ¾ Ñ„Ğ¾Ğ»Ğ´Ğ°Ğ¼ Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ�ÑŒ Ğ½Ğ° Ğ²Ñ�ĞµĞ¼ train?


def cross_validation_classification(
    X_train, y_train, X_test, y_test,
    X_oot=None, y_oot=None,
    model=None, n_folds=5, random_state=42,
    metrics=('accuracy', 'f1', 'roc_auc'),
    cv_type='stratified'  # 'stratified' Ğ¸Ğ»Ğ¸ 'kfold'
):
    """
    Ğ¤ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ğ´Ğ»Ñ� Ğ¿Ñ€Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ� ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸ Ğ¿Ğ¾ Ğ·Ğ°Ğ´Ğ°Ñ‡Ğµ Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ğ¾Ğ¹ ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸

    ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:
    X_train - Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸ (pandas DataFrame)
    y_train - Ñ†ĞµĞ»ĞµĞ²Ğ°Ñ� Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ°Ñ� Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸ (pandas Series Ğ¸Ğ»Ğ¸ numpy array)
    X_test - Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸ (pandas DataFrame)
    y_test - Ñ†ĞµĞ»ĞµĞ²Ğ°Ñ� Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ°Ñ� Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸ (pandas Series Ğ¸Ğ»Ğ¸ numpy array)
    X_oot - Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ out-of-time Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸ (pandas DataFrame), Ğ¾Ğ¿Ñ†Ğ¸Ğ¾Ğ½Ğ°Ğ»ÑŒĞ½Ğ¾
    y_oot - Ñ†ĞµĞ»ĞµĞ²Ğ°Ñ� Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ°Ñ� out-of-time Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸ (pandas Series Ğ¸Ğ»Ğ¸ numpy array), Ğ¾Ğ¿Ñ†Ğ¸Ğ¾Ğ½Ğ°Ğ»ÑŒĞ½Ğ¾
    model - Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ¼Ğ°ÑˆĞ¸Ğ½Ğ½Ğ¾Ğ³Ğ¾ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� (Ğ¿Ğ¾ ÑƒĞ¼Ğ¾Ğ»Ñ‡Ğ°Ğ½Ğ¸Ñ� RandomForestClassifier)
    n_folds - ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ„Ğ¾Ğ»Ğ´Ğ¾Ğ² Ğ´Ğ»Ñ� ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸
    random_state - Ñ„Ğ¸ĞºÑ�Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ğ¾Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ Ğ´Ğ»Ñ� Ğ²Ğ¾Ñ�Ğ¿Ñ€Ğ¾Ğ¸Ğ·Ğ²Ğ¾Ğ´Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¾Ğ²
    metrics - ĞºĞ¾Ñ€Ñ‚ĞµĞ¶ Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº ('accuracy', 'f1', 'roc_auc')
    cv_type - Ñ‚Ğ¸Ğ¿ ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸: 'kfold' Ğ¸Ğ»Ğ¸ 'stratified'

    Ğ’Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚:
    cv_results - Ñ�Ğ»Ğ¾Ğ²Ğ°Ñ€ÑŒ Ñ� Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ°Ğ¼Ğ¸ ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸
    final_model - Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ½Ğ°Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ½Ğ° Ğ²Ñ�ĞµÑ… Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train Ğ´Ğ¾Ğ»Ğ¶ĞµĞ½ Ğ±Ñ‹Ñ‚ÑŒ pandas DataFrame")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test Ğ´Ğ¾Ğ»Ğ¶ĞµĞ½ Ğ±Ñ‹Ñ‚ÑŒ pandas DataFrame")
    if X_oot is not None and not isinstance(X_oot, pd.DataFrame):
        raise TypeError("X_oot Ğ´Ğ¾Ğ»Ğ¶ĞµĞ½ Ğ±Ñ‹Ñ‚ÑŒ pandas DataFrame")
    if not set(X_train.columns) == set(X_test.columns):
        raise ValueError("ĞšĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸ Ğ² X_train Ğ¸ X_test Ğ´Ğ¾Ğ»Ğ¶Ğ½Ñ‹ Ñ�Ğ¾Ğ²Ğ¿Ğ°Ğ´Ğ°Ñ‚ÑŒ")
    if X_oot is not None and not set(X_train.columns) == set(X_oot.columns):
        raise ValueError("ĞšĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸ Ğ² X_train Ğ¸ X_oot Ğ´Ğ¾Ğ»Ğ¶Ğ½Ñ‹ Ñ�Ğ¾Ğ²Ğ¿Ğ°Ğ´Ğ°Ñ‚ÑŒ")

    if model is None:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    # Ğ’Ñ‹Ğ±Ğ¾Ñ€ Ñ‚Ğ¸Ğ¿Ğ° ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸
    if cv_type == 'stratified':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif cv_type == 'kfold':
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        raise ValueError("cv_type Ğ´Ğ¾Ğ»Ğ¶ĞµĞ½ Ğ±Ñ‹Ñ‚ÑŒ 'stratified' Ğ¸Ğ»Ğ¸ 'kfold'")

    fold_metrics = {m: [] for m in metrics}
    val_preds = np.zeros(len(y_train))
    val_probas = np.zeros(len(y_train))

    print("Ğ—Ğ°Ğ¿ÑƒÑ�Ğº ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸ (Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ğ°Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ñ�)...")

    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train, y_train) 
                                                     if cv_type == 'stratified'
                                                     else kf.split(X_train), total=n_folds, desc="Ğ¤Ğ¾Ğ»Ğ´Ñ‹")):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
        y_fold_val = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]

        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)
        try:
            probas = model.predict_proba(X_fold_val)[:, 1]
        except AttributeError:
            probas = preds

        val_preds[val_idx] = preds
        val_probas[val_idx] = probas

        for m in metrics:
            if m == 'accuracy':
                fold_metrics[m].append(accuracy_score(y_fold_val, preds))
            elif m == 'f1':
                fold_metrics[m].append(f1_score(y_fold_val, preds))
            elif m == 'roc_auc':
                fold_metrics[m].append(roc_auc_score(y_fold_val, probas))

        print(f"Ğ¤Ğ¾Ğ»Ğ´ {fold+1}/{n_folds}: " +
              ", ".join([f"{m}: {fold_metrics[m][-1]:.4f}" for m in metrics]))

    cv_metrics = {f'val_{m}': np.mean(fold_metrics[m]) for m in metrics}

    final_model = model.fit(X_train, y_train)

    test_preds = final_model.predict(X_test)
    try:
        test_probas = final_model.predict_proba(X_test)[:, 1]
    except AttributeError:
        test_probas = test_preds

    test_metrics = {}
    for m in metrics:
        if m == 'accuracy':
            test_metrics[m] = accuracy_score(y_test, test_preds)
        elif m == 'f1':
            test_metrics[m] = f1_score(y_test, test_preds)
        elif m == 'roc_auc':
            test_metrics[m] = roc_auc_score(y_test, test_probas)

    oot_metrics = None
    oot_preds = None
    if X_oot is not None and y_oot is not None:
        oot_preds = final_model.predict(X_oot)
        try:
            oot_probas = final_model.predict_proba(X_oot)[:, 1]
        except AttributeError:
            oot_probas = oot_preds

        oot_metrics = {}
        for m in metrics:
            if m == 'accuracy':
                oot_metrics[m] = accuracy_score(y_oot, oot_preds)
            elif m == 'f1':
                oot_metrics[m] = f1_score(y_oot, oot_preds)
            elif m == 'roc_auc':
                oot_metrics[m] = roc_auc_score(y_oot, oot_probas)

    cv_results = {
        'fold_metrics': fold_metrics,
        'val_metrics': cv_metrics,
        'test_metrics': test_metrics,
        'oot_metrics': oot_metrics,
        'val_predictions': val_preds,
        'test_predictions': test_preds,
        'oot_predictions': oot_preds,
    }

    print("\nĞ˜Ñ‚Ğ¾Ğ³Ğ¾Ğ²Ñ‹Ğµ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸:")
    print("Ğ’Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ� (CV):", cv_metrics)
    print("Ğ¢ĞµÑ�Ñ‚:", test_metrics)
    if oot_metrics is not None:
        print("OOT:", oot_metrics)

    return cv_results, final_model


def run_hyperopt_classification_catboost(
    X_train, y_train, X_test, y_test,
    space,
    n_iter=30,
    random_state=42,
    n_splits_cv=5,
    early_stopping_rounds=20,
    cat_features=None
):
    log_rows = []
    trials = Trials()
    pbar = tqdm(total=n_iter, desc="Hyperopt", position=0, leave=True)
    iteration = 0

    def get_model_params(params):
        dct = {
            'iterations': int(params['iterations']),
            'depth': int(params['depth']),
            'learning_rate': params['learning_rate'],
            'l2_leaf_reg': int(params['l2_leaf_reg']),
            'random_strength': params['random_strength'],
            'bagging_temperature': params['bagging_temperature'],
            'random_seed': random_state,
            'verbose': 0
        }
        if 'bootstrap_type' in params:
            dct['bootstrap_type'] = params['bootstrap_type']
        if 'subsample' in params and dct.get('bootstrap_type', 'Bayesian') == 'Bernoulli':
            dct['subsample'] = params['subsample']
        return dct

    def objective(params):
        nonlocal iteration
        model_params = get_model_params(params)
        skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            model = CatBoostClassifier(**model_params)
            fit_params = dict()
            if cat_features is not None:
                fit_params['cat_features'] = cat_features
            model.fit(
                X_train.iloc[train_idx], y_train.iloc[train_idx],
                eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                early_stopping_rounds=early_stopping_rounds,
                use_best_model=True,
                **fit_params
            )
            val_pred_proba = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            roc_auc = roc_auc_score(y_train.iloc[val_idx], val_pred_proba)
            val_scores.append(roc_auc)

        mean_cv_auc = np.mean(val_scores)

        # Ğ¤Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ½Ğ° Ğ²Ñ�Ñ‘Ğ¼ Ñ‚Ñ€ĞµĞ¹Ğ½Ğµ
        model = CatBoostClassifier(**model_params)
        fit_params = dict()
        if cat_features is not None:
            fit_params['cat_features'] = cat_features
        model.fit(X_train, y_train, **fit_params)
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        test_roc_auc = roc_auc_score(y_test, test_pred_proba)
        test_pred = (test_pred_proba >= 0.5).astype(int)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)

        row = {
            'iteration': iteration + 1,
            **model_params,
            'cv_roc_auc': mean_cv_auc,
            'test_f1': test_f1,
            'test_accuracy': test_accuracy,
            'test_roc_auc': test_roc_auc
        }
        log_rows.append(row)

        iteration += 1
        pbar.update(1)
        return {'loss': -mean_cv_auc, 'status': STATUS_OK}

    start_time = time.time()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n_iter,
        trials=trials,
        rstate=np.random.default_rng(random_state),
        verbose=0
    )
    pbar.close()

    if len(log_rows) == 0:
        print("Ğ�ĞµÑ‚ ÑƒÑ�Ğ¿ĞµÑˆĞ½Ñ‹Ñ… Ğ¸Ñ‚ĞµÑ€Ğ°Ñ†Ğ¸Ğ¹ Hyperopt.")
        return None

    results_df = pd.DataFrame(log_rows)
    duration = time.time() - start_time

    best_cv_idx = np.argmax([t['result']['loss'] for t in trials.trials])
    best_params_cv = log_rows[best_cv_idx]
    best_test_idx = results_df['test_f1'].idxmax()
    best_params_test = results_df.loc[best_test_idx].to_dict()

    final_params = {k: v for k, v in best_params_test.items()
                    if k not in ['iteration', 'cv_roc_auc', 'test_f1', 'test_accuracy', 'test_roc_auc']}
    best_model = CatBoostClassifier(**final_params)
    fit_params = dict()
    if cat_features is not None:
        fit_params['cat_features'] = cat_features
    best_model.fit(X_train, y_train, **fit_params)

    print(f"\nĞ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ¿Ğ¾ test_f1: {best_params_test}")
    print(f"Hyperopt Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½ Ğ·Ğ° {duration:.2f} Ñ�ĞµĞºÑƒĞ½Ğ´")
    print("\nĞ¢Ğ¾Ğ¿-10 Ñ�Ñ‚Ñ€Ğ¾Ğº Ğ»Ğ¾Ğ³Ğ°:")
    display(results_df.sort_values('test_f1', ascending=False).head(10).style.background_gradient(
        subset=['cv_roc_auc', 'test_f1', 'test_accuracy', 'test_roc_auc'], cmap='Blues')
           )

    return {
        'best_model': best_model,
        'best_params_cv': best_params_cv,
        'best_params_test': best_params_test,
        'log_df': results_df,
        'duration': duration,
    }


space = {
    'iterations': hp.quniform('iterations', 100, 500, 50),
    'depth': hp.quniform('depth', 4, 8, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
    'l2_leaf_reg': hp.quniform('l2_leaf_reg', 3, 20, 1),
    'random_strength': hp.uniform('random_strength', 0, 1.0),
    'bagging_temperature': hp.uniform('bagging_temperature', 0, 1.0),
    'bootstrap_type': hp.choice('bootstrap_type', ['Bayesian'])
    # 'subsample': hp.uniform('subsample', 0.6, 1.0)
}


result = run_hyperopt_classification_catboost(
    X_train[model_features], 
    y_train, 
    X_test[model_features], 
    y_test,
    space=space,
    n_iter=30,
    random_state=42,
    n_splits_cv=5,
    early_stopping_rounds=10,
    cat_features=None
)

log_df = result['log_df']


# Hyperopt:   0%|                                                                                 | 0/30 [05:42<?, ?it/s]
# Hyperopt: 100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 30/30 [07:15<00:00, 14.53s/it]

# Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ¿Ğ¾ test_f1: {'iteration': 8, 'iterations': 250, 'depth': 5, 'learning_rate': 0.08470806691352324, 'l2_leaf_reg': 16, 'random_strength': 0.06428686330954925, 'bagging_temperature': 0.7375817486746217, 'random_seed': 42, 'verbose': 0, 'bootstrap_type': 'Bayesian', 'cv_roc_auc': 0.9324115413484749, 'test_f1': 0.7482142857142857, 'test_accuracy': 0.906, 'test_roc_auc': 0.9368309336366423}
# Hyperopt Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½ Ğ·Ğ° 435.85 Ñ�ĞµĞºÑƒĞ½Ğ´



log_df


def plot_hyperopt_param_influence(
    trials_df,
    params_to_plot=None,
    metric_col='test_roc_auc',
    title="Ğ—Ğ°Ğ²Ğ¸Ñ�Ğ¸Ğ¼Ğ¾Ñ�Ñ‚ÑŒ ROC-AUC Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğµ Ğ¾Ñ‚ Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ²"
):
    """
    Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ²Ğ»Ğ¸Ñ�Ğ½Ğ¸Ñ� Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ² Ğ½Ğ° ROC-AUC (Ğ¸Ğ»Ğ¸ Ğ´Ñ€ÑƒĞ³ÑƒÑ� Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºÑƒ) Ğ¿Ğ¾ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ°Ğ¼ Hyperopt.

    ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:
    -----------
    trials_df : pd.DataFrame
        Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ñ‹ Ğ¿Ğ¾Ğ´Ğ±Ğ¾Ñ€Ğ° Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ² (Ğ´Ğ¾Ğ»Ğ¶ĞµĞ½ Ğ²ĞºĞ»Ñ�Ñ‡Ğ°Ñ‚ÑŒ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ² Ğ¸ Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº).
    params_to_plot : list, optional
        Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ² Ğ´Ğ»Ñ� Ğ¾Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�. Ğ•Ñ�Ğ»Ğ¸ None â€” Ğ°Ğ²Ñ‚Ğ¾Ğ¼Ğ°Ñ‚Ğ¸Ñ‡ĞµÑ�ĞºĞ¸ Ğ²Ñ‹Ğ±Ñ€Ğ°Ñ‚ÑŒ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹.
    metric_col : str
        Ğ�Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ° Ñ� Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ¹ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¾Ğ¹ (Ğ¿Ğ¾ ÑƒĞ¼Ğ¾Ğ»Ñ‡Ğ°Ğ½Ğ¸Ñ� 'test_roc_auc').
    title : str
        Ğ—Ğ°Ğ³Ğ¾Ğ»Ğ¾Ğ²Ğ¾Ğº Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ°.
    """

    # Ğ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ�Ğ»ÑƒĞ¶ĞµĞ±Ğ½Ñ‹Ñ… ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ğ½Ğµ Ğ½ÑƒĞ¶Ğ½Ğ¾ Ğ²Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ¸Ñ€Ğ¾Ğ²Ğ°Ñ‚ÑŒ
    service_cols = {
        'iteration', 
        'cv_f1', 
        'test_f1', 
        'test_accuracy', 
        'test_roc_auc',
        'test_threshold', 
        'random_seed', 
        'verbose', 
        'train_auc',
        'cv_roc_auc'
    }

    # Ğ•Ñ�Ğ»Ğ¸ Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ² Ğ½Ğµ Ğ·Ğ°Ğ´Ğ°Ğ½ â€” Ğ²Ñ‹Ğ±Ğ¸Ñ€Ğ°ĞµĞ¼ Ğ²Ñ�Ğµ Ğ¿Ğ¾Ğ´Ñ…Ğ¾Ğ´Ñ�Ñ‰Ğ¸Ğµ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹
    if params_to_plot is None:
        params_to_plot = [
            col for col in trials_df.columns
            if col not in service_cols and pd.api.types.is_numeric_dtype(trials_df[col])
        ]

    n = len(params_to_plot)
    if n == 0:
        raise ValueError("Ğ�Ğµ Ğ½Ğ°Ğ¹Ğ´ĞµĞ½Ğ¾ Ğ½Ğ¸ Ğ¾Ğ´Ğ½Ğ¾Ğ³Ğ¾ Ğ¿Ğ¾Ğ´Ñ…Ğ¾Ğ´Ñ�Ñ‰ĞµĞ³Ğ¾ Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ° Ğ´Ğ»Ñ� Ğ¾Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�.")

    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows))
    axes = axes.flatten()
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    sns.set_style("whitegrid")
    palette = sns.color_palette("coolwarm", as_cmap=True)

    for i, param in enumerate(params_to_plot):
        ax = axes[i]
        sns.scatterplot(
            data=trials_df,
            x=param,
            y=metric_col,
            hue=metric_col,
            palette=palette,
            size=metric_col,
            sizes=(20, 100),
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
            legend=False
        )
        sns.regplot(
            data=trials_df,
            x=param,
            y=metric_col,
            scatter=False,
            lowess=True,
            color='red',
            line_kws={'linewidth': 2, 'linestyle': '--'},
            ax=ax
        )
        ax.set_title(f'{metric_col} vs {param}', fontsize=13, fontweight='bold')
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel(metric_col, fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    # Ğ£Ğ´Ğ°Ğ»ĞµĞ½Ğ¸Ğµ Ğ»Ğ¸ÑˆĞ½Ğ¸Ñ… Ğ¿ÑƒÑ�Ñ‚Ñ‹Ñ… Ğ¿Ğ¾Ğ´Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¾Ğ²
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


params_to_plot = [
    'iterations',
    'depth',
    'learning_rate',
    'l2_leaf_reg',
    'random_strength',
    'bagging_temperature'
]

plot_hyperopt_param_influence(
    trials_df=log_df,
    params_to_plot=params_to_plot,
    metric_col='test_roc_auc',
    title="Ğ—Ğ°Ğ²Ğ¸Ñ�Ğ¸Ğ¼Ğ¾Ñ�Ñ‚ÑŒ ROC-AUC Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğµ Ğ¾Ñ‚ Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ²"
)


# train_final = pd.read_csv('train.csv')
# test_final = pd.read_csv('test.csv')
# sample_submission = pd.read_csv('sample_submission.csv')

train_final = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')
test_final = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')
sample_submission = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/sample_submission.csv')


def engineer_features(df_main, y_train=None, is_train=True, stat_refs=None):
    import numpy as np
    import pandas as pd
    import category_encoders as ce

    df = df_main.copy()
    
    # Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ Ğ»Ğ¸ÑˆĞ½Ğ¸Ğµ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ñ‹
    cols_to_drop = ['id', 'CustomerId', 'Surname']
    df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº: ĞºĞ»Ğ¸ĞµĞ½Ñ‚ Ñ� Ğ½ÑƒĞ»ĞµĞ²Ñ‹Ğ¼ Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ¾Ğ¼
    df['balance_is_zero'] = (df['Balance'] == 0).astype(int)

    # Ğ›Ğ¾Ğ³Ğ°Ñ€Ğ¸Ñ„Ğ¼Ñ‹ Ğ¸ Ğ¿Ğ¾Ğ»Ğ¸Ğ½Ğ¾Ğ¼Ñ‹
    log_poly_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    for col in log_poly_cols:
        df[f'{col}_log'] = np.log1p(df[col])
        df[f'{col}_sq'] = df[col] ** 2
        df[f'{col}_cube'] = df[col] ** 3

    # ĞšĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ
    cat_cols = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    stats_dict = {
        'CreditScore': ['mean', 'median', 'sum'],
        'Age': ['mean', 'median', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan],
        'Tenure': ['mean', 'median', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan],
        'Balance': ['mean', 'median', 'sum'],
        'EstimatedSalary': ['mean', 'median', 'sum']
    }

    if is_train:
        stats_by_cat = {}
        for cat in cat_cols:
            for num_col, stats in stats_dict.items():
                grouped = df.groupby(cat)[num_col].agg(stats)
                grouped.columns = [f'{num_col}_{cat}_{s if isinstance(s, str) else "mode"}' for s in stats]
                df = df.merge(grouped, how='left', left_on=cat, right_index=True)
                stats_by_cat[(cat, num_col)] = grouped
        # Ğ¢Ğ°ĞºĞ¶Ğµ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ğ¸Ğ¼ proportions/counts
        prop_count = {}
        for cat in cat_cols:
            prop = df[cat].value_counts(normalize=True).rename(f'{cat}_proportion')
            count = df[cat].value_counts().rename(f'{cat}_count')
            df = df.merge(prop, how='left', left_on=cat, right_index=True)
            df = df.merge(count, how='left', left_on=cat, right_index=True)
            prop_count[cat] = (prop, count)

        # Ğ“Ñ€ÑƒĞ¿Ğ¿Ğ¾Ğ²Ñ‹Ğµ Ñ�Ñ€ĞµĞ´Ğ½Ğ¸Ğµ
        avg_credit_score_geo = df.groupby('Geography')['CreditScore'].mean()
        avg_balance_geo = df.groupby('Geography')['Balance'].mean()
        avg_salary_gender = df.groupby('Gender')['EstimatedSalary'].mean()
        high_balance_threshold = df['Balance'].quantile(0.9)

        # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼
        stat_refs = {
            'stats_by_cat': stats_by_cat,
            'prop_count': prop_count,
            'avg_geo_score': avg_credit_score_geo,
            'avg_geo_balance': avg_balance_geo,
            'avg_salary_gender': avg_salary_gender,
            'high_balance_threshold': high_balance_threshold,
            'geo_credit_mean': df.groupby('Geography')['CreditScore'].mean(),
            'geo_balance_std': df.groupby('Geography')['Balance'].std()
        }

    else:
        for (cat, num_col), grouped in stat_refs['stats_by_cat'].items():
            df = df.merge(grouped, how='left', left_on=cat, right_index=True)
        for cat, (prop, count) in stat_refs['prop_count'].items():
            df = df.merge(prop, how='left', left_on=cat, right_index=True)
            df = df.merge(count, how='left', left_on=cat, right_index=True)

    # ĞœĞ°Ğ¿Ğ¿Ğ¸Ğ½Ğ³
    df['avg_credit_score_geo'] = df['Geography'].map(stat_refs['avg_geo_score'])
    df['avg_balance_geo'] = df['Geography'].map(stat_refs['avg_geo_balance'])
    df['avg_salary_gender'] = df['Gender'].map(stat_refs['avg_salary_gender'])

    df['balance_to_salary_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['tenure_per_age'] = df['Tenure'] / (df['Age'] + 1)
    df['balance_minus_salary'] = df['Balance'] - df['EstimatedSalary']
    df['salary_to_age_ratio'] = df['EstimatedSalary'] / (df['Age'] + 1)
    df['balance_to_tenure_ratio'] = df['Balance'] / (df['Tenure'] + 1)

    df['credit_score_gap_from_avg_geo'] = df['CreditScore'] - df['avg_credit_score_geo']
    df['balance_gap_from_avg_geo'] = df['Balance'] - df['avg_balance_geo']
    df['salary_gap_from_avg_gender'] = df['EstimatedSalary'] - df['avg_salary_gender']

    df['is_young_and_rich_flag'] = ((df['Age'] < 30) & (df['Balance'] > stat_refs['high_balance_threshold'])).astype(int)
    df['ltv_proxy'] = df['EstimatedSalary'] * df['Tenure']
    df['is_churn_risk_profile'] = ((df['IsActiveMember'] == 0) & (df['Tenure'] < 2)).astype(int)
    df['potential_profitability_index'] = df['EstimatedSalary'] / (df['NumOfProducts'] + 1)

    df.drop(columns=['avg_credit_score_geo', 'avg_balance_geo', 'avg_salary_gender'], inplace=True)

    df['tenure_stage'] = pd.cut(df['Tenure'], bins=[-1, 1, 4, np.inf], labels=['new', 'developing', 'established'])
    df['wealth_index'] = (df['Balance'] + df['EstimatedSalary']) / (df['Age'] + 1)
    df['risk_age_score'] = df['CreditScore'] / np.log1p(df['Age'])
    df['activation_opportunity'] = ((df['HasCrCard'] == 0) & (df['IsActiveMember'] == 1)).astype(int)

    df['geo_credit_risk_index'] = df['Geography'].map(stat_refs['geo_credit_mean'])
    df['geo_balance_std'] = df['Geography'].map(stat_refs['geo_balance_std'])
    df['credit_score_band'] = pd.cut(df['CreditScore'], bins=[-np.inf, 600, 750, np.inf], labels=['low', 'medium', 'high'])

    # Target encoding
    cat_enc_cols = ['Geography', 'Gender']
    if is_train:
        encoder = ce.CatBoostEncoder(cols=cat_enc_cols)
        encoder.fit(df[cat_enc_cols], y_train)
        stat_refs['encoder'] = encoder
    else:
        encoder = stat_refs['encoder']

    df_cb = encoder.transform(df[cat_enc_cols]).add_suffix('_cb')
    df = df.join(df_cb)

    # Ğ’Ñ‹Ğ´ĞµĞ»Ğ¸Ğ¼ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
    float_cols = df.select_dtypes(include='float64').columns.tolist()
    cat_features = [col for col in df.columns if col.endswith('_cb')]
    model_features = float_cols + cat_features

    return df, model_features, stat_refs if is_train else None


# Ğ Ğ°Ğ·Ğ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ½Ğ° Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ğ¸ Ñ†ĞµĞ»ĞµĞ²ÑƒÑ� Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½ÑƒÑ�
X_train_final = train_final.drop(columns=['Exited'])
y_train_final = train_final['Exited']


X_train_fe, model_features, stat_refs = engineer_features(X_train_final, y_train_final, is_train=True)


X_test_fe, _, _ = engineer_features(test_final, is_train=False, stat_refs=stat_refs)


model_features = ['Age_sq',
 'balance_gap_from_avg_geo',
 'Age_Gender_median',
 'salary_gap_from_avg_gender',
 'potential_profitability_index',
 'risk_age_score',
 'wealth_index',
 'CreditScore',
 'credit_score_gap_from_avg_geo',
 'Tenure_NumOfProducts_mode',
 'EstimatedSalary_log',
 'Age',
 'ltv_proxy',
 'Age_cube',
 'Age_log',
 'CreditScore_NumOfProducts_mean',
 'Balance_log',
 'CreditScore_NumOfProducts_median',
 'EstimatedSalary_cube',
 'tenure_per_age',
 'CreditScore_cube',
 'IsActiveMember_proportion',
 'CreditScore_NumOfProducts_sum',
 'salary_to_age_ratio',
 'Balance_cube',
 'EstimatedSalary_NumOfProducts_mean',
 'Tenure_NumOfProducts_mean',
 'balance_to_salary_ratio',
 'Balance',
 'EstimatedSalary_NumOfProducts_median',
 'EstimatedSalary_IsActiveMember_mean',
 'Age_NumOfProducts_mode',
 'Balance_NumOfProducts_median',
 'NumOfProducts_proportion',
 'Tenure_IsActiveMember_mean',
 'EstimatedSalary_NumOfProducts_sum',
 'IsActiveMember',
 'CreditScore_IsActiveMember_median',
 'CreditScore_IsActiveMember_mean',
 'CreditScore_Geography_mean',
 'Age_HasCrCard_mode',
 'Age_NumOfProducts_median',
 'Balance_IsActiveMember_mean',
 'NumOfProducts',
 'Tenure_Geography_mean',
 'CreditScore_HasCrCard_median',
 'EstimatedSalary_IsActiveMember_median',
 'EstimatedSalary_sq',
 'CreditScore_HasCrCard_mean',
 'Age_Geography_median',
 'EstimatedSalary_Geography_mean',
 'geo_balance_std',
 'Balance_NumOfProducts_sum',
 'Balance_IsActiveMember_sum']


space = {
    'iterations': hp.quniform('iterations', 100, 500, 50),
    'depth': hp.quniform('depth', 4, 8, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
    'l2_leaf_reg': hp.quniform('l2_leaf_reg', 3, 20, 1),
    'random_strength': hp.uniform('random_strength', 0, 1.0),
    'bagging_temperature': hp.uniform('bagging_temperature', 0, 1.0),
    'bootstrap_type': hp.choice('bootstrap_type', ['Bayesian'])
    # 'subsample': hp.uniform('subsample', 0.6, 1.0)
}

result = run_hyperopt_classification_catboost(
    X_train[model_features], 
    y_train, 
    X_test[model_features], 
    y_test,
    space=space,
    n_iter=30,
    random_state=42,
    n_splits_cv=5,
    early_stopping_rounds=10,
    cat_features=None
)


# # best_params = {
# #     'iterations': 450,
# #     'depth': 8,
# #     'learning_rate': 0.04987948839600423,
# #     'l2_leaf_reg': 4,
# #     'random_strength': 0.6990785947415082,
# #     'bagging_temperature': 0.9477077386530258,
# #     'random_seed': 42,
# #     'verbose': 0,
# #     'bootstrap_type': 'Bayesian'
# #     # 'auto_class_weights': 'SqrtBalanced'
# # }

best_params = {
    "iterations": 500,
    "depth": 8,
    "learning_rate": 0.013713,
    "l2_leaf_reg": 8,
    "random_strength": 0.034703,
    "bagging_temperature": 0.552650,
    "random_seed": 42,
    "verbose": 0,
    "bootstrap_type": "Bayesian"
    # 'auto_class_weights': 'SqrtBalanced'
}


model = CatBoostClassifier(**best_params)

cv_results, final_model = cross_validation_classification(
    X_train[model_features], 
    y_train, 
    X_test[model_features], 
    y_test,
    model=model,
    n_folds=5,
    random_state=42,
    metrics=('accuracy', 'f1', 'roc_auc'),
    cv_type='stratified'
)

# ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹ Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğµ
y_test_proba = final_model.predict_proba(X_test_fe[model_features])[:, 1]

# ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğµ
y_test_pred = final_model.predict(X_test_fe[model_features])


# # X_test = test[model_features]
# predictions_proba_test = y_test_proba
# submission = pd.DataFrame({
#     'id': test_final['id'],
#     'Exited': predictions_proba_test
# })
# submission.to_csv('submission.csv', index=False)


# ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
best_params = {
    "iterations": 500,
    "depth": 8,
    "learning_rate": 0.013713,
    "l2_leaf_reg": 8,
    "random_strength": 0.034703,
    "bagging_temperature": 0.552650,
    "random_seed": 42,
    "verbose": 0,
    "bootstrap_type": "Bayesian",
    'auto_class_weights': 'SqrtBalanced'
}

model = CatBoostClassifier(**best_params)

# Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ½Ğ° Ğ²Ñ�ĞµÑ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
model.fit(X_train_fe[model_features], y_train_final)

# ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğµ Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹ Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğµ
y_test_proba = model.predict_proba(X_test_fe[model_features])[:, 1]

# Ğ¤Ğ¾Ñ€Ğ¼Ğ¸Ñ€ÑƒĞµĞ¼ Ñ�Ğ°Ğ±Ğ¼Ğ¸Ñ‚
submission = pd.DataFrame({
    'id': test_final['id'],
    'Exited': y_test_proba
})

submission.to_csv('submission.csv', index=False)
print("Ğ¡Ğ°Ğ±Ğ¼Ğ¸Ñ‚ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ‘Ğ½ Ğ² 'submission.csv'")


# importance_df, selected_features = permutation_importance(
#     X_train=X_train[model_features], 
#     y_train=y_train, 
#     X_test=X_test[model_features], 
#     y_test=y_test,
#     metric='roc_auc',
#     n_repeats=50,
#     threshold=0.001,
#     n_top_features=None,
#     plot=True
# )


# final_important_features = importance_df['feature'].tolist()
# important_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
# important_features
# model_features = important_features


# importance_df, selected_features = permutation_importance(
#     X_train=X_train[model_features], 
#     y_train=y_train, 
#     X_test=X_test[model_features], 
#     y_test=y_test,
#     metric='roc_auc',
#     n_repeats=50,
#     threshold=0.0001,
#     n_top_features=None,
#     plot=True
# )


# final_important_features = importance_df['feature'].tolist()
# important_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
# important_features
# model_features = important_features


# importance_df, selected_features = permutation_importance(
#     X_train=X_train[model_features], 
#     y_train=y_train, 
#     X_test=X_test[model_features], 
#     y_test=y_test,
#     metric='roc_auc',
#     n_repeats=50,
#     threshold=0.0001,
#     n_top_features=None,
#     plot=True
# )


# final_important_features = importance_df['feature'].tolist()
# important_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
# important_features
# model_features = important_features


# importance_df, selected_features = permutation_importance(
#     X_train=X_train[model_features], 
#     y_train=y_train, 
#     X_test=X_test[model_features], 
#     y_test=y_test,
#     metric='roc_auc',
#     n_repeats=50,
#     threshold=0.0001,
#     n_top_features=None,
#     plot=True
# )


# final_important_features = importance_df['feature'].tolist()
# important_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
# important_features
# model_features = important_features


# importance_df, selected_features = permutation_importance(
#     X_train=X_train[model_features], 
#     y_train=y_train, 
#     X_test=X_test[model_features], 
#     y_test=y_test,
#     metric='roc_auc',
#     n_repeats=10,
#     threshold=0.0001,
#     n_top_features=None,
#     plot=True
# )


# final_important_features = importance_df['feature'].tolist()
# important_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
# important_features
# model_features = important_features


# importance_df, selected_features = permutation_importance(
#     X_train=X_train[model_features], 
#     y_train=y_train, 
#     X_test=X_test[model_features], 
#     y_test=y_test,
#     metric='roc_auc',
#     n_repeats=10,
#     threshold=0.0001,
#     n_top_features=None,
#     plot=True
# )


# final_important_features = importance_df['feature'].tolist()
# important_features = importance_df[importance_df['importance'] > 0.0041]['feature'].tolist()
# important_features
# model_features = important_features


# model_catboost = CatBoostClassifier(
#     iterations=200,
#     random_state=14,
#     verbose=0
# )

# model_catboost.fit(
#     X_train[model_features], y_train,
#     eval_set=(X_test[model_features], y_test),
#     verbose=10
# )

# predictions_proba_val = model_catboost.predict_proba(X_test[model_features])[:, 1]


# from sklearn.metrics import roc_auc_score
# print("ROC-AUC:", roc_auc_score(y_test, predictions_proba_val))


# # compute the SHAP values for every prediction in the validation dataset
# explainer = shap.TreeExplainer(model_catboost)
# shap_values = explainer.shap_values(X_test[model_features])
# shap.summary_plot(shap_values, X_test[model_features], plot_type="bar")


# model_features


model_features = ['balance_gap_from_avg_geo',
 'IsActiveMember',
 'Age_log',
 'Age_NumOfProducts_mode',
 'Age_Gender_median',
 'Balance_cube',
 'salary_gap_from_avg_gender',
 'ltv_proxy',
 'Age_sq',
 'CreditScore_NumOfProducts_mean',
 'risk_age_score',
 'CreditScore_NumOfProducts_median',
 'potential_profitability_index',
 'CreditScore']


# result = run_hyperopt_classification_catboost(
#     X_train[model_features], 
#     y_train, 
#     X_test[model_features], 
#     y_test,
#     space=space,
#     n_iter=100,
#     random_state=42,
#     n_splits_cv=5,
#     early_stopping_rounds=30,
#     cat_features=None
# )

# Hyperopt: 100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 100/100 [21:30<00:00, 12.90s/it]

# Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ¿Ğ¾ test_f1: {'iteration': 64, 'iterations': 450, 'depth': 4, 'learning_rate': 0.07147593205479658, 'l2_leaf_reg': 17, 'random_strength': 0.007639422481463765, 'bagging_temperature': 0.30863445630094466, 'random_seed': 42, 'verbose': 0, 'bootstrap_type': 'Bayesian', 'cv_roc_auc': 0.9327940760602292, 'test_f1': 0.7453416149068323, 'test_accuracy': 0.9043333333333333, 'test_roc_auc': 0.9372930923189643}
# Hyperopt Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½ Ğ·Ğ° 1290.21 Ñ�ĞµĞºÑƒĞ½Ğ´


# log_df.query('test_roc_auc == 0.9380458298135246')


# params_to_plot = [
#     'iterations',
#     'depth',
#     'learning_rate',
#     'l2_leaf_reg',
#     'random_strength',
#     'bagging_temperature'
# ]

# plot_hyperopt_param_influence(
#     trials_df=log_df,
#     params_to_plot=params_to_plot,
#     metric_col='test_roc_auc',
#     title="Ğ—Ğ°Ğ²Ğ¸Ñ�Ğ¸Ğ¼Ğ¾Ñ�Ñ‚ÑŒ ROC-AUC Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğµ Ğ¾Ñ‚ Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ²"
# )

