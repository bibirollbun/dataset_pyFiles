import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


cmap = plt.cm.tab10
plt.style.use('dark_background')


data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")


data.head()


data.shape


data.info()


data.isna().sum()


num_features = data.select_dtypes(include=[np.number])
cat_features = data.select_dtypes(exclude=[np.number])


print(num_features.columns, "\n",cat_features.columns)


cat_features.shape


num_columns = cat_features.shape[1]
num_rows = num_cols = int(np.ceil(np.sqrt(num_columns)))
fig, axs = plt.subplots(ncols=num_cols, nrows=num_rows,figsize=(5*num_cols,4*num_rows))
axs = axs.flatten()
for i,col in enumerate(cat_features.columns):
    value_count = cat_features[col].value_counts()
    value_count.plot(kind="bar",ax=axs[i],rot=0, color=cmap(np.arange(len(value_count))))
    axs[i].set_title(col)

for i in range(num_columns, num_rows * num_cols):
    fig.delaxes(axs[i])
    
fig.suptitle("Value counts for categorical columns",fontsize=16)
plt.tight_layout()
plt.show()


for col in num_features.columns:
    print(f"{col} : {num_features[col].skew()}")


num_features.columns


fig, axs = plt.subplots(ncols=3,nrows=2,figsize=(5*3,4*2))
axs = axs.flatten()
for i,col in enumerate(['speed_limit','num_reported_accidents','num_lanes','curvature','accident_risk']):
    value_count = num_features[col].value_counts()
    value_count.plot(kind="bar",ax=axs[i],rot=0, color=cmap(np.arange(len(value_count))))
    axs[i].set_title(col)
    if col in ['curvature','accident_risk']:
        axs[i].set_xticks([0,value_count.shape[0]],labels=[0, 1])
    

fig.delaxes(axs[5])
    
fig.suptitle("Value counts for numerical columns",fontsize=16)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(ncols=3,nrows=2,figsize=(5*3,4*2))
axs = axs.flatten()
for i,col in enumerate(['speed_limit','num_reported_accidents','num_lanes','curvature']):
    axs[i].scatter(y=data['accident_risk'],x=data[col])
    axs[i].set_title(col)
    axs[i].set_ylabel('accident risk')
    axs[i].set_xlabel(col)

fig.delaxes(axs[5])
fig.delaxes(axs[4])
    
fig.suptitle("Value counts for numerical columns",fontsize=16)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(ncols=3,nrows=2,figsize=(5*3,4*2))
axs = axs.flatten()
for i,col in enumerate(['speed_limit','num_reported_accidents','num_lanes','curvature']):
    axs[i].boxplot(data[col])
    axs[i].set_title(col)

fig.delaxes(axs[5])
fig.delaxes(axs[4])
    
fig.suptitle("Value counts for numerical columns",fontsize=16)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(
    num_features.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=.5 
)
plt.plot()
plt.show()


cat_features.columns


fig, axs = plt.subplots(ncols=3,nrows=3,figsize=(5*3,4*3))
axs = axs.flatten()
for i,col in enumerate(cat_features.columns):
    vals = cat_features[col].unique()
    risk_mean = []
    for val in vals:
        risk_mean.append((data[cat_features[col]==val]['accident_risk']).mean())
    axs[i].plot(vals,risk_mean)
    axs[i].set_title(col)
    axs[i].set_ylabel('accident risk')
    axs[i].set_xticks(range(0,len(risk_mean)),labels=vals)

fig.delaxes(axs[8])
    
fig.suptitle("Accident risk for categorical columns",fontsize=16)
plt.tight_layout()
plt.show()    


plt.figure(figsize=(8, 6))
sns.heatmap(
    num_features.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=.5 
)
plt.plot()
plt.show()


# One-Hot Encode the categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(cat_features)
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_features.columns))

# Combine numerical and encoded features
combined_df = pd.concat([num_features, encoded_df], axis=1)


#running pca to see redundant attributes
# Step 1 : Standarisation of data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(combined_df)

# Step 2 : Run Pca
pca = PCA(n_components=2)
pca.fit(df_scaled)

pca_df = pd.DataFrame(
    data=pca.transform(df_scaled),
    columns=['Principal Component 1', 'Principal Component 2']
)


plt.figure(figsize=(10, 8))
plt.scatter(
    x=pca_df['Principal Component 1'],
    y=pca_df['Principal Component 2'],
    alpha=0.6
)
plt.title('2D PCA for Clustering Tendency')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
plt.grid(True)
plt.show()


# Step 2 : Run Pca
pca = PCA(n_components=None)
pca.fit(df_scaled)

explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio for each component:")
print(explained_variance)


cumulative_variance = np.cumsum(explained_variance)

# Plot the explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

