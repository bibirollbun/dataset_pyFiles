import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")


dir=r"/kaggle/input/k-means-clustering-for-heart-disease-analysis"
data=pd.read_csv(os.path.join(dir,"heart_disease.csv"))
submission=pd.read_csv(os.path.join(dir,"sample.csv"))
ID=data["id"]
data.drop(columns=["id"],inplace=True)
data


data.info()


data.isnull().sum()


from sklearn.compose import make_column_selector as selector
numeric_columns=selector(dtype_exclude=object)(data)
object_columns=selector(dtype_include=object)(data)
data[numeric_columns].isnull().sum()


for col in numeric_columns:
    plt.hist(data[col],bins="auto",edgecolor="black")
    plt.title(col)
    plt.show()


data[numeric_columns]=data[numeric_columns].fillna(data[numeric_columns].mean())
data[numeric_columns].isnull().sum()


for col in object_columns:
    object_distributions=data[col].value_counts()
    plt.pie(object_distributions.values,labels=object_distributions.index,autopct="%.1f%%")
    plt.title(col)
    plt.show()


data[object_columns].isnull().sum()


# 缺失值填充，策略是：缺失值多，就把 NAN视为一类，缺失值少就按最多类补齐
# 布尔类型值随机填充，但要保持与原数据分布情况一致
col_bool=["exang","fbs"]
for col in col_bool:
    # 获取缺失值索引
    idx=data[data[col].isnull()].index
    # 计算比例
    distributions=data[col].value_counts(normalize=True)  # 直接获取比例，而不是数目
    probs=[distributions.get(True,0),distributions.get(False,0)]
    data.loc[idx,col]=np.random.choice([True,False],size=len(idx),p=probs)
data[col_bool].isnull().sum()


# 填充  “restecg”，按最多类填充
data["restecg"]=data["restecg"].fillna("normal")


# 处理剩下的列
data[["thal","slope"]]=data[["thal","slope"]].fillna("unknown")
data[object_columns].isnull().sum()


from sklearn.preprocessing import OrdinalEncoder

data[object_columns]=OrdinalEncoder().fit_transform(data[object_columns])


data


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias=[]
sse=[]
for n_cluster in range(2,20):
    kmeans =KMeans(n_clusters=n_cluster,n_init="auto",random_state=42).fit(data)
    inertias.append(kmeans.inertia_)
    sse.append(silhouette_score(data,kmeans.labels_))
_,ax=plt.subplots(nrows=1,ncols=2,figsize=(8,4))
ax[0].plot(np.arange(2,20),inertias,"b-")
ax[0].set_title("inertials")
ax[1].plot(np.arange(2,20),sse,"r-")
ax[1].set_title("sse score")
plt.tight_layout()
plt.show()


for n in range(2,6):
    temp=KMeans(n_clusters=n).fit(data)
    submission["cluster"]=temp.labels_[submission["id"]]
    submission.to_csv(f"submission_v{n}.csv",index=False)


from sklearn.preprocessing import OneHotEncoder

data=OneHotEncoder().fit_transform(data)
data


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias=[]
sse=[]
for n_cluster in range(2,20):
    kmeans =KMeans(n_clusters=n_cluster,n_init="auto",random_state=42).fit(data)
    inertias.append(kmeans.inertia_)
    sse.append(silhouette_score(data,kmeans.labels_))
_,ax=plt.subplots(nrows=1,ncols=2,figsize=(8,4))
ax[0].plot(np.arange(2,20),inertias,"b-")
ax[0].set_title("inertials")
ax[1].plot(np.arange(2,20),sse,"r-")
ax[1].set_title("sse score")
plt.tight_layout()
plt.show()




