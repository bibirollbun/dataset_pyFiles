import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import skew
from lightgbm import LGBMClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, SelectKBest, SelectFromModel
from sklearn.metrics import f1_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans,DBSCAN
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


data = np.load("/kaggle/input/linear-regression-competitive-ml-letovo/task_data.npz")
sub = pd.read_csv('/kaggle/input/linear-regression-competitive-ml-letovo/sample_submission.csv')

X = data["X"]
p_vasya = data["p_vasya"]

print(X.shape, p_vasya.shape)
df = pd.concat( [pd.DataFrame(X), pd.DataFrame(p_vasya)],axis=1)


df_X = pd.DataFrame(X)
axes = df_X.plot(
    subplots=True,
    figsize=(9, 14),
    kind='hist',
    bins=50,        
    sharex=True,     
    legend=False,    
    title='Distribuition')

for ax, col_name in zip(axes, df_X.columns):
    ax.set_title(f'feature_{col_name}')
    ax.set_ylabel("Frequency")
    
plt.xlabel('Values') 
plt.tight_layout() 
plt.show()


def argumented_data() -> pd.DataFrame():
    gm = GaussianMixture(n_components=2)
    gm.fit(df)
    syntetic_data, _ = gm.sample(300)
    X_new = pd.concat( [pd.DataFrame(X), pd.DataFrame(syntetic_data[:,:15])])
    return X_new


kmeans = KMeans(n_clusters=2, max_iter=500,random_state=42, n_init='auto')
kmeans.fit(X)
cluster_labels = kmeans.fit_predict(X)

unique, counts = np.unique(cluster_labels, return_counts=True)
print(f"Clusters: {dict(zip(unique, counts))}")

counts = np.bincount(cluster_labels)
minority_label = np.argmin(counts)
pseudo_labels = (cluster_labels == minority_label).astype(int)



clf = LogisticRegression(max_iter=1000,
                         class_weight='balanced',
                         penalty='l2',
                         solver='liblinear',
                         C=0.8)
clf.fit(X, pseudo_labels)
probs = clf.predict_proba(X)[:,1]


sub['Target'] = (probs >= 0.25).astype(int)
sub.to_csv('submission.csv',index=False)


sub['Target'].value_counts()

