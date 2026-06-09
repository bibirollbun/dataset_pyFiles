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


# -----------------------------------------------------
# 环境准备：导入常用库 & 设置显示参数
# -----------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pd.set_option("display.max_columns", None)
sns.set(style="whitegrid", font_scale=1.1)



# -----------------------------------------------------
# 读取航司会员数据集
# -----------------------------------------------------
path_train = "/kaggle/input/sa-customer-segmentation/flight_train.csv"
train = pd.read_csv(path_train)

print("数据维度:", train.shape)
train.head()



# -----------------------------------------------------
# 数据清洗：处理缺失值与日期格式
# -----------------------------------------------------
# 删除无用编号列
if "MEMBER_NO" in train.columns:
    train = train.drop(columns=["MEMBER_NO"])

# 转换日期列
date_cols = ["FFP_DATE", "FIRST_FLIGHT_DATE", "LOAD_TIME", "LAST_FLIGHT_DATE"]
for col in date_cols:
    train[col] = pd.to_datetime(train[col], errors="coerce")

# 计算会员入会年限
train["MEMBER_YEARS"] = (train["LOAD_TIME"] - train["FFP_DATE"]).dt.days / 365

# 性别缺失填充
train["GENDER"] = train["GENDER"].fillna("Unknown")

# 年龄缺失填中位数
if "AGE" in train.columns:
    train["AGE"] = train["AGE"].fillna(train["AGE"].median())

# 其余缺失值填 0
train = train.fillna(0)

train.head()



# -----------------------------------------------------
# 选择用于聚类的特征（行为 + 价值类变量）
# -----------------------------------------------------
features = [
    "FFP_TIER", "AGE", "SEG_KM_SUM", "LAST_TO_END",
    "AVG_INTERVAL", "MAX_INTERVAL", "EXCHANGE_COUNT",
    "avg_discount", "Points_Sum", "Point_NotFlight", "MEMBER_YEARS"
]

X = train[features].copy()

# 标准化数值特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("标准化后维度:", X_scaled.shape)



# -----------------------------------------------------
# 使用 PCA 降维到二维，便于可视化客户分布
# -----------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], s=6, alpha=0.5)
plt.title("PCA Projection of Customers")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()



# -----------------------------------------------------
# 使用肘部法 (Elbow Method) 选择最优聚类数 K
# -----------------------------------------------------
sse = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.plot(range(2,10), sse, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE")
plt.show()



# -----------------------------------------------------
# 聚类训练 & 将结果添加回原表
# -----------------------------------------------------
best_k = 4  # 你可以根据上一步的图调整这个值
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
train["Cluster"] = kmeans.fit_predict(X_scaled)

print(train["Cluster"].value_counts())
train.head()



# -----------------------------------------------------
# 查看每个聚类的均值特征（客户画像基础）
# -----------------------------------------------------
cluster_summary = train.groupby("Cluster")[features].mean().round(2)
cluster_summary["Count"] = train["Cluster"].value_counts().sort_index()
cluster_summary



# -----------------------------------------------------
# 关键指标可视化（例如：总飞行里程、积分、折扣）
# -----------------------------------------------------
plt.figure(figsize=(10,5))
sns.boxplot(data=train, x="Cluster", y="SEG_KM_SUM")
plt.title("Total Flight KM per Cluster")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(data=train, x="Cluster", y="Points_Sum")
plt.title("Points Sum per Cluster")
plt.show()



# -----------------------------------------------------
# 根据聚类均值 + 业务理解进行解释（示例）
# -----------------------------------------------------
cluster_profile = {
    0: "高价值常旅客（频繁飞行 + 高积分）",
    1: "普通活跃客户（中等飞行频率 + 稳定积分）",
    2: "低频休闲客户（偶尔出行，积分低）",
    3: "沉睡客户（长期未飞行，低活跃）"
}

train["Cluster_Profile"] = train["Cluster"].map(cluster_profile)
train[["Cluster", "Cluster_Profile"]].head(10)



# === PCA explained variance ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

evr = pca.explained_variance_ratio_
print("Explained variance ratio:", evr)
print(f"PC1: {evr[0]:.4f}, PC2: {evr[1]:.4f},  Cum: {evr[:2].sum():.4f}")

plt.figure(figsize=(5,3.5))
plt.bar(["PC1","PC2"], evr, alpha=0.8)
plt.title("Explained Variance Ratio")
plt.ylabel("Ratio")
plt.ylim(0, 1)
plt.show()



# === PCA loadings (feature contributions) ===
loadings = pd.DataFrame(
    pca.components_.T,     # shape: [n_features, 2]
    index=X.columns,
    columns=["PC1","PC2"]
).sort_index()

# Top |loading| 特征（绝对值排序）
top_n = 10
top_pc1 = loadings.reindex(loadings["PC1"].abs().sort_values(ascending=False).head(top_n).index)
top_pc2 = loadings.reindex(loadings["PC2"].abs().sort_values(ascending=False).head(top_n).index)

print("Top features for PC1 (by |loading|):")
display(top_pc1)

print("Top features for PC2 (by |loading|):")
display(top_pc2)

# 可视化：柱状图（按绝对值排序）
def plot_loading_bar(series, title):
    order = series.abs().sort_values(ascending=True).index
    plt.figure(figsize=(7,4))
    plt.barh(order, series.loc[order].values)
    plt.title(title)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.show()

plot_loading_bar(top_pc1["PC1"], "PC1 Loadings (top by |value|)")
plot_loading_bar(top_pc2["PC2"], "PC2 Loadings (top by |value|)")



# === Correlation between original features and PCs ===
# 注意：X_scaled 是标准化后的 ndarray，需要转成 DataFrame 以便相关性计算
Xs_df = pd.DataFrame(X_scaled, columns=X.columns, index=getattr(X, "index", None))
pc_df = pd.DataFrame(X_pca, columns=["PC1","PC2"], index=Xs_df.index)

corr_pc = Xs_df.join(pc_df)[["PC1","PC2"]].corrwith(Xs_df, axis=0)  # 每个原特征与PC的相关性
# 上面一行取法会返回层级索引，改两次分别算更清楚：
pc1_corr = Xs_df.corrwith(pc_df["PC1"]).sort_values(key=np.abs, ascending=False).head(12)
pc2_corr = Xs_df.corrwith(pc_df["PC2"]).sort_values(key=np.abs, ascending=False).head(12)

print("Top correlations with PC1:")
display(pc1_corr.to_frame("corr_PC1"))

print("Top correlations with PC2:")
display(pc2_corr.to_frame("corr_PC2"))



# === Scatter in PCA space, colored by cluster (if available) ===
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
if "Cluster" in getattr(train, "columns", []):
    plt.scatter(X_pca[:,0], X_pca[:,1], c=train["Cluster"], s=8, alpha=0.6, cmap="tab10")
    plt.title("PCA Projection (colored by Cluster)")
else:
    plt.scatter(X_pca[:,0], X_pca[:,1], s=8, alpha=0.6)
    plt.title("PCA Projection (no cluster labels found)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# （可选）把 KMeans 的中心也投影进来（若 kmeans 存在）
try:
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.figure(figsize=(6,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=train["Cluster"], s=6, alpha=0.3, cmap="tab10")
    plt.scatter(centers_pca[:,0], centers_pca[:,1], marker="X", s=160, edgecolor="k")
    for i, (x,y) in enumerate(centers_pca):
        plt.text(x, y, f"C{i}", fontsize=10, weight="bold", ha="center", va="center")
    plt.title("PCA Projection with Cluster Centers")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.show()
except Exception as e:
    pass



# === Per-cluster distribution on PCs ===
import seaborn as sns

if "Cluster" in getattr(train, "columns", []):
    pc_plot_df = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1], "Cluster": train["Cluster"]})

    plt.figure(figsize=(7,4))
    sns.boxplot(x="Cluster", y="PC1", data=pc_plot_df, showfliers=False)
    plt.title("PC1 by Cluster")
    plt.show()

    plt.figure(figsize=(7,4))
    sns.boxplot(x="Cluster", y="PC2", data=pc_plot_df, showfliers=False)
    plt.title("PC2 by Cluster")
    plt.show()



# === Simple biplot (scores + a few feature vectors) ===
import numpy as np

scores = X_pca
# 选 |loading| 最大的前 k 个特征作箭头
k = 6
top_feats = loadings.assign(absPC1=loadings["PC1"].abs()+loadings["PC2"].abs()) \
                    .sort_values("absPC1", ascending=False).head(k).index.tolist()

plt.figure(figsize=(6,5))
plt.scatter(scores[:,0], scores[:,1], s=6, alpha=0.25)
scale = 3.0  # 箭头放缩比，视图面板可调
for feat in top_feats:
    xv, yv = loadings.loc[feat, ["PC1","PC2"]].values * scale
    plt.arrow(0, 0, xv, yv, head_width=0.2, head_length=0.3, length_includes_head=True)
    plt.text(xv*1.05, yv*1.05, feat, fontsize=9)
plt.title("PCA Biplot (top feature vectors)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
plt.tight_layout()
plt.show()



# === Auto textual summary for PCA semantics ===
def pc_summary_text(loadings_df, pc="PC1", top=6):
    s = loadings_df[pc].sort_values(key=np.abs, ascending=False)
    pos = s[s > 0].head(top).index.tolist()
    neg = s[s < 0].head(top).index.tolist()
    txt = []
    if pos:
        txt.append(f"{pc} 正向主要由：{', '.join(pos)} 提升；")
    if neg:
        txt.append(f"{pc} 负向主要由：{', '.join(neg)} 降低；")
    return " ".join(txt)

print(f"Explained variance — PC1: {evr[0]:.2%}, PC2: {evr[1]:.2%} (Cum: {evr[:2].sum():.2%})")
print(pc_summary_text(loadings, "PC1", top=5))
print(pc_summary_text(loadings, "PC2", top=5))



# -----------------------------------------------------
# 导出包含聚类标签的结果，便于下游分析
# -----------------------------------------------------
train.to_csv("airline_customer_clusters.csv", index=False)
print("✅ 文件已保存：airline_customer_clusters.csv")


