import os
import sys
import subprocess
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

try:
    from ydata_profiling import ProfileReport, compare
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ydata-profiling"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from ydata_profiling import ProfileReport, compare


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.filterwarnings("ignore")
sns.set_theme()
%matplotlib inline


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col="id")


# Need to separate data before homogeneity analysis
train["Target"] = 0
test["Target"] = 1


total_data = pd.concat([train.drop(["Calories"], axis=1), test], axis=0)
total_data = total_data.sample(frac=1, random_state=42) # Shuffle data before analysis


train_profile = ProfileReport(train.drop("Target", axis=1), title="Train data report")
test_profile = ProfileReport(test.drop("Target", axis=1), title="Test data report")
total_data_profile = ProfileReport(total_data, title="Full data report")
compare_profile = compare([train_profile, test_profile]) # For idea of compare report thx for https://www.kaggle.com/rahul713


train_profile.to_notebook_iframe()


test_profile.to_notebook_iframe()


total_data_profile.to_notebook_iframe()


compare_profile.to_notebook_iframe()


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay


total_data_copy = total_data.copy()

total_data_copy["Sex"] = total_data_copy["Sex"].map({"female": 0, "male": 1})

print("Target value counts:")
print(total_data_copy["Target"].value_counts())

scaler = StandardScaler()

X = total_data_copy.drop("Target", axis=1)
X = scaler.fit_transform(X)
y = total_data_copy["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


clf = LogisticRegression(random_state=42, max_iter=5000, n_jobs=-1) 
clf.fit(X_train, y_train)

RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()


# Let's reduce the data while maintaining the distribution of the target variable through binning
train_copy = train.copy()
train_copy = train_copy.drop("Target", axis=1)
train_copy["target_bin"] = pd.qcut(train_copy["Calories"], q=10, labels=False)
train_sample, _ = train_test_split(train_copy, train_size=0.1, stratify=train_copy["target_bin"], random_state=42)
train_sample = train_sample.drop("target_bin", axis=1)

plt.figure(figsize=(12, 12))
sns.pairplot(train_sample, corner=True, hue="Sex", diag_kind="hist")
plt.show()


fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(14, 12))
axes = axes.flatten()

cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

for i, col in enumerate(cols):
    sns.regplot(
        data=train_sample,
        x=col,
        y="Calories",
        ax=axes[i],
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"}
    )
    axes[i].set_title(f"{col} vs Calories")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
sns.histplot(data=train_sample, x="Calories", hue="Sex", bins=50)
plt.show()


train_sample_scaled = scaler.fit_transform(train_sample[cols])

train_sample_scaled_df = pd.DataFrame(train_sample_scaled, columns=cols)

plt.figure(figsize=(16, 8))
sns.boxplot(data=train_sample_scaled_df)
plt.xticks(rotation=45)
plt.title("Outliers data")
plt.show()


fig = px.scatter(train_sample, x="Height", y="Weight", color="Sex", size="Calories")
fig.show()


fig = px.scatter(train_sample, x="Heart_Rate", y="Body_Temp", color="Sex", size="Calories")
fig.show()


tm = train_sample.copy()
tm["HDivW"] = tm["Height"] / tm["Weight"]
fig = px.scatter(tm, x="HDivW", y="Calories", color="Sex", size="Calories")
fig.show()


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


train_sample_v2 = train_sample.copy()
train_sample_v2["Sex"] = train_sample_v2["Sex"].map({"female": 0, "male": 1})
train_sample_scaled_v2 = scaler.fit_transform(train_sample_v2)
train_sample_scaled_df_v2 = pd.DataFrame(train_sample_scaled_v2, columns=train_sample.columns)
train_sample_scaled_df_v2.columns


inertias = []

for i in range(2, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(train_sample_scaled_df_v2)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.plot(range(2, 10), inertias)
plt.xlabel("Clusters count")
plt.ylabel("Inertia")


sample_data_ = train_sample_scaled_df_v2.sample(frac=1, random_state=42)

kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(sample_data_)

tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=500)
X_embedded = tsne.fit_transform(sample_data_)

plt.figure(figsize=(14, 10))

sns.scatterplot(
    x=X_embedded[:, 0], 
    y=X_embedded[:, 1],
    hue=clusters,
    palette="viridis",
    alpha=0.8,
    s=50
)

plt.title("t-SNE visualization + K-means", fontsize=16)
plt.xlabel("t-SNE 1")
plt.ylabel("t-S")

