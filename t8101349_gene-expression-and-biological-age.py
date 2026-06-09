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


bioAge_path = "/kaggle/input/gene-expression-and-biological-age/bioAge.csv"
genExp_path = "/kaggle/input/gene-expression-and-biological-age/genExp.csv"


bioAge_df = pd.read_csv(bioAge_path)
genExp_df = pd.read_csv(genExp_path)


bioAge_df


genExp_df


genExp_df.isnull().sum()


bioAge_df = bioAge_df.drop(['colID'], axis=1)


genExp_df = genExp_df.drop(['colID'], axis=1)


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_valid, y_train, y_valid = train_test_split(genExp_df, bioAge_df, test_size=0.2, random_state=42)

# 建立 MLPRegressor 模型
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(100, 50), 
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)

mlp_reg.fit(X_train, y_train)

y_pred = mlp_reg.predict(X_valid)

print("MSE:", mean_squared_error(y_valid, y_pred))
print("R^2 Score:", r2_score(y_valid, y_pred))



train_df = pd.concat([genExp_df, bioAge_df], axis=1)



import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = train_df.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()


# pca
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

X_train, X_valid, y_train, y_valid = train_test_split(genExp_df, bioAge_df, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.90)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = mlp_reg.predict(X_valid)

print("MSE:", mean_squared_error(y_valid, y_pred))
print("R^2 Score:", r2_score(y_valid, y_pred))



# CNN/Unet


# K-means Cluster   
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score



scaler = StandardScaler()
scaled_data = scaler.fit_transform(genExp_df)

silhouette_scores = []
possible_ks = range(2,16)

for k in possible_ks:
    kmeans = KMeans(n_clusters= k, random_state = 12)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)
    print(f"k={k},score = {score:.4f}")

best_k = possible_ks[silhouette_scores.index(max(silhouette_scores))]
print(f"best_k={best_k}")


scaler = StandardScaler()
final_data = scaler.fit_transform(train_df)

#final_data = pd.concat([scaled_data, scaled_age_data], axis=1)

silhouette_scores = []
possible_ks = range(2,16)

for k in possible_ks:
    kmeans = KMeans(n_clusters= k, random_state = 12)
    labels = kmeans.fit_predict(final_data)
    score = silhouette_score(final_data, labels)
    silhouette_scores.append(score)
    print(f"k={k},score = {score:.4f}")

best_k = possible_ks[silhouette_scores.index(max(silhouette_scores))]
print(f"best_k={best_k}")


bioAge_df = pd.read_csv(bioAge_path)
genExp_df = pd.read_csv(genExp_path)


n_clusters = 5
final_kmeans = KMeans(n_clusters= n_clusters, random_state = 12)
final_labels = kmeans.fit_predict(final_data)

result_df = pd.DataFrame({
    "colID":genExp_df.colID,
    "cluster":final_labels,
})

result_df.to_csv("submission.csv", index = False)
print("ok")

