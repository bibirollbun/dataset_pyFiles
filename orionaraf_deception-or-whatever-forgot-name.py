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


train = pd.read_csv("/kaggle/input/deceptive-points-aicc-round-0/train.csv")
train


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


X = train.drop("target", axis = 1)
y = train["target"]


pca = PCA(n_components=1)
X_train_pca = pca.fit_transform(X)


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 7))
plt.scatter(X_train_pca, y, alpha=0.6)
plt.title("Corrupted Data Visualization", fontsize=16)
plt.xlabel("PC1")
plt.ylabel("target")
plt.show()


features = X.columns.tolist()
features


import seaborn as sns


for col in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(X[col], color="skyblue", kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()


from sklearn.mixture import GaussianMixture
from sklearn.linear_model import HuberRegressor


scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)


gmm = GaussianMixture(
    n_components=2,
    random_state=42,
    covariance_type="full",
    n_init=100,
    max_iter=200,
    tol=1e-4
)
gmm.fit(train_scaled)


cluster_probs = gmm.predict_proba(train_scaled)

teacher_id = -1
mx = -1


for i in range(2):
    weights = cluster_probs[:, i]
    X_mean_effort = X.mean(axis=1)

    weighted_mean_x = np.average(X_mean_effort, weights=weights)
    weighted_mean_y = np.average(y, weights=weights)

    weighted_cov = np.average((X_mean_effort - weighted_mean_x) * (y - weighted_mean_y), weights=weights)
    weighted_std_x = np.sqrt(np.average((X_mean_effort - weighted_mean_x)**2, weights=weights))
    weighted_std_y = np.sqrt(np.average((y - weighted_mean_y)**2, weights=weights))

    correlation = weighted_cov / (weighted_std_x * weighted_std_y) 
    print(f"  Cluster {i} Linearity Score (Weighted Correlation): {correlation:.4f}")

    if correlation > mx:
        mx = correlation
        teacher_id = i


print(teacher_id)


sample_weights = cluster_probs[:, teacher_id]
model = HuberRegressor()
model.fit(X, y, sample_weight=sample_weights)


test = pd.read_csv("/kaggle/input/deceptive-points-aicc-round-0/test.csv")
X_test = test.drop("ID", axis = 1)
test_ids = test["ID"]


y_pred = model.predict(X_test)

submission_df = pd.DataFrame({
    "ID": test_ids,
    "Target": y_pred
})
submission_df.to_csv("last_submission.csv", index=False)




