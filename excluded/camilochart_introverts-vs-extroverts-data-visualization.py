import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
df_train.head()


df_train.info()


df_train.describe()


df_train.isnull().sum()


df_train.nunique()


target_column = 'Personality'


str_columns = list(df_train.select_dtypes(include=["object"]).columns)
index_remove = str_columns.index(target_column)
str_columns.pop(index_remove)
str_columns


from matplotlib.colors import LinearSegmentedColormap

colors = ['orange', 'white', 'blue']
custom_cmap = LinearSegmentedColormap.from_list('CustomMap', colors)


df_train[target_column].value_counts().plot.bar(color=['orange','blue'])


for col in str_columns:
    df_train.groupby([col,target_column]).size().unstack().plot(kind='bar', stacked=True,cmap=custom_cmap)
    plt.title('Feature: ' + col)
    plt.show()


numerical_columns = list(df_train.select_dtypes(include=["int64", "float64"]).columns)
numerical_columns


for col in numerical_columns:
    df_train.groupby([col,target_column]).size().unstack().plot(kind='bar', stacked=True,cmap=custom_cmap)
    plt.title('Feature: ' + col)
    plt.show()


df_train[target_column + '_code'] = pd.Categorical(df_train[target_column]).codes


variables = numerical_columns
n_vars = len(variables)

fig, axes = plt.subplots(n_vars, n_vars, figsize=(10, 10))

for i, var_i in enumerate(variables):
    for j, var_j in enumerate(variables):
        ax = axes[i, j]

        if i == j:  # Diagonal plots (histograms)
            ax.hist(df_train[var_i], bins=15, edgecolor='black')
            ax.set_title(f'Distribution of {var_i}')
        else:  # Off-diagonal plots (scatter plots)
            ax.scatter(df_train[var_j], df_train[var_i],c = df_train[target_column + '_code'], cmap=custom_cmap)
            ax.set_xlabel(var_j)
            ax.set_ylabel(var_i)

# Adjust layout and display
plt.tight_layout()
plt.show()


df_train_encode = df_train.copy()
for col in str_columns:
    df_train_encode[col] = pd.Categorical(df_train_encode[col]).codes

df_train_encode = df_train_encode.fillna(0)
df_train_encode = df_train_encode.drop(labels=[target_column],axis=1)


from sklearn.decomposition import PCA


pca = PCA(n_components=2)
df_train_pca = pca.fit_transform(df_train_encode)


df_pca = pd.DataFrame(data = {'C1':df_train_pca[:,0] , 'C2': df_train_pca[:,1]})


plt.scatter(df_pca['C1'], df_pca['C2'], c=df_train[target_column + '_code'] ,cmap=custom_cmap)


pca = PCA(n_components=3)
df_train_pca_3d = pca.fit_transform(df_train_encode)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(df_train_pca_3d[:,0], df_train_pca_3d[:,1],df_train_pca_3d[:,2], c=df_train[target_column + '_code'] ,cmap=custom_cmap)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

