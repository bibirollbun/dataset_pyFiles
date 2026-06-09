# 重新加载数据（确保路径正确）
train_df = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])

print("数据加载验证:")
print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)
print("训练集列名:", list(train_df.columns))

# 检查关键列是否存在
assert 'target' in train_df.columns, "target列不存在！"
assert 'first_active_month' in train_df.columns, "first_active_month列不存在！"


import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)


!ls ../input/


train_df = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("Number of rows and columns in train set : ",train_df.shape)
print("Number of rows and columns in test set : ",test_df.shape)

train_df["active_duration"] = (pd.to_datetime("2018-01-01") - train_df["first_active_month"]).dt.days
test_df["active_duration"] = (pd.to_datetime("2018-01-01") - test_df["first_active_month"]).dt.days


train_df.head()


target_col = "target"

plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df[target_col].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.show()


plt.figure(figsize=(12,8))
sns.distplot(train_df[target_col].values, bins=50, kde=False, color="red")
plt.title("Histogram of Loyalty score")
plt.xlabel('Loyalty score', fontsize=12)
plt.show()


(train_df[target_col]<-30).sum()


cnt_srs = train_df['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in train set")
plt.show()

cnt_srs = test_df['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in test set")
plt.show()


# Feature 1
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_1", y=target_col, data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 1 distribution")
plt.show()

# Feature 2
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_2", y=target_col, data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 2', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 2 distribution")
plt.show()

# Feature 3
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_3", y=target_col, data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 3', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 3 distribution")
plt.show()


hist_df = pd.read_csv("../input/historical_transactions.csv")
hist_df.head()

from sklearn import preprocessing
for col in ["category_1", "category_2", "category_3"]:
    lbl = preprocessing.LabelEncoder()
    # 先填充缺失值为字符串"-1"，再统一转为字符串类型
    hist_df[col] = hist_df[col].fillna("-1").astype(str)  # 把所有值转为字符串，包括原来的数字
    lbl.fit(hist_df[col])  # 现在都是字符串，可正常排序和编码
    hist_df[col] = lbl.transform(hist_df[col])

# 检查每个类别列的数据类型和唯一值
for col in ["category_1", "category_2", "category_3"]:
    print(f"列 {col} 数据类型：", hist_df[col].dtype)
    print(f"列 {col} 唯一值：", hist_df[col].unique())
    print("-" * 20)


# 序号13：历史交易数据多维度聚合（替换原单一交易次数统计）
agg_funcs = {
    # 交易金额相关统计量
    "purchase_amount": ["sum", "mean", "std", "min", "max", "count"],  # count就是原交易次数
    # 分期付款相关统计量
    "installments": ["sum", "mean", "std", "min", "max"],
    # 已编码的类别特征统计量（新增，捕捉类别分布）
    "category_1": ["mean", "count"],  # 类别1的均值（可理解为某类别占比）、非空数量
    "category_2": ["mean", "nunique"],  # 类别2的均值、不同类别数量
    "category_3": ["mean", "nunique"]   # 类别3的均值、不同类别数量
}

# 按card_id分组聚合，计算每个用户的统计特征
hist_agg = hist_df.groupby("card_id").agg(agg_funcs).reset_index()

# 重命名列名（避免多级列名，让特征名更清晰）
hist_agg.columns = ["card_id"] + [f"hist_{col[0]}_{col[1]}" for col in hist_agg.columns[1:]]

# 合并聚合特征到训练集和测试集（左连接，保留所有用户）
train_df = pd.merge(train_df, hist_agg, on="card_id", how="left")
test_df = pd.merge(test_df, hist_agg, on="card_id", how="left")

# 可选：填充可能的缺失值（无历史交易的用户，特征值设为0）
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# 验证合并结果
print("训练集合并后列数：", train_df.shape[1])
print("新增的历史交易特征：", [col for col in train_df.columns if col.startswith("hist_")])


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Kaggle默认支持的英文黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 1. 确保历史交易次数特征存在（兜底处理）
if "hist_purchase_amount_count" not in train_df.columns:
    print("特征缺失，临时生成历史交易次数...")
    hist_count = train_df.groupby("card_id").size().reset_index(name="hist_purchase_amount_count")
    train_df = pd.merge(train_df, hist_count, on="card_id", how="left")

# 2. 按交易次数分组，计算平均忠诚度评分
cnt_srs = train_df.groupby("hist_purchase_amount_count")["target"].mean().sort_index()

# 3. 筛选数据（避免极端值，用简单切片，适配旧版本Pandas）
# 只保留交易次数<=200的数据（避免少数极端值让图表变形）
cnt_srs = cnt_srs[cnt_srs.index <= 200]

# 4. 用Matplotlib绘制散点图
plt.figure(figsize=(10, 6)) 
plt.scatter(
    cnt_srs.index,  # x轴：交易次数
    cnt_srs.values, # y轴：平均忠诚度评分
    color='orange', 
    s=50,  
    alpha=0.7  
)

plt.title('Loyalty Score by Number of Historical Transactions', fontsize=14, pad=20)
plt.xlabel('Number of Historical Transactions', fontsize=12)
plt.ylabel('Average Loyalty Score', fontsize=12)
plt.grid(True, alpha=0.3)  
plt.tight_layout()  

plt.show()


bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 10000]
train_df['binned_num_hist_transactions'] = pd.cut(train_df['hist_purchase_amount_count'], bins)
cnt_srs = train_df.groupby("binned_num_hist_transactions")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_num_hist_transactions", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_num_hist_transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("binned_num_hist_transactions distribution")
plt.show()


gdf = hist_df.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")


bins = np.percentile(train_df["hist_purchase_amount_sum"], range(0,101,10))
train_df['binned_sum_hist_trans'] = pd.cut(train_df['hist_purchase_amount_sum'], bins)

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_hist_trans", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('Binned Sum of Historical Transaction Value', fontsize=12)  
plt.ylabel('Loyalty Score', fontsize=12)
plt.title("Sum of Historical Transaction Value (Binned) Distribution", fontsize=14)
plt.tight_layout() 
plt.show()


bins = np.percentile(train_df["hist_purchase_amount_mean"], range(0,101,10))
train_df['binned_mean_hist_trans'] = pd.cut(train_df['hist_purchase_amount_mean'], bins)

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_mean_hist_trans", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('Binned Mean Historical Transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Mean of historical transaction value (Binned) distribution")
plt.tight_layout() 
plt.show()


new_trans_df = pd.read_csv("../input/new_merchant_transactions.csv")
new_trans_df.head()


new_trans_df = pd.read_csv("../input/new_merchant_transactions.csv")
new_trans_df.head()

gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "new_num_merch_transactions"]
# 合并时添加缺失值填充（无新商户交易的用户，交易次数设为0）
train_df = pd.merge(train_df, gdf, on="card_id", how="left").fillna({"new_num_merch_transactions": 0})
test_df = pd.merge(test_df, gdf, on="card_id", how="left").fillna({"new_num_merch_transactions": 0})


bins = [0, 10, 20, 30, 40, 50, 75, 10000]
train_df['binned_num_merch_transactions'] = pd.cut(train_df['new_num_merch_transactions'], bins)
cnt_srs = train_df.groupby("binned_num_merch_transactions")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_num_merch_transactions", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('Binned Number of New Merchant Transactions', fontsize=12)  
plt.ylabel('Loyalty Score', fontsize=12)
plt.title("Number of New Merchant Transactions (Binned) Distribution", fontsize=14)
plt.tight_layout() 
plt.show()


gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = [
    "card_id", 
    "new_sum_merch_trans",    # 新商户交易金额总和
    "new_mean_merch_trans",   # 新商户交易金额均值
    "new_std_merch_trans",    # 新商户交易金额标准差
    "new_min_merch_trans",    # 新商户交易金额最小值
    "new_max_merch_trans"     # 新商户交易金额最大值
]
train_df = pd.merge(train_df, gdf, on="card_id", how="left").fillna({
    "new_sum_merch_trans": 0,
    "new_mean_merch_trans": 0,
    "new_std_merch_trans": 0,
    "new_min_merch_trans": 0,
    "new_max_merch_trans": 0
})
test_df = pd.merge(test_df, gdf, on="card_id", how="left").fillna({
    "new_sum_merch_trans": 0,
    "new_mean_merch_trans": 0,
    "new_std_merch_trans": 0,
    "new_min_merch_trans": 0,
    "new_max_merch_trans": 0
})


bins = np.nanpercentile(train_df["new_sum_merch_trans"], range(0,101,10))
train_df['binned_sum_merch_trans'] = pd.cut(train_df['new_sum_merch_trans'], bins)

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_merch_trans", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('Binned Sum of New Merchant Transactions', fontsize=12) 
plt.ylabel('Loyalty Score', fontsize=12)
plt.title("Sum of New Merchant Transaction Value (Binned) Distribution", fontsize=14)
plt.tight_layout()  
plt.show()


bins = np.nanpercentile(train_df["new_mean_merch_trans"], range(0,101,10))
train_df['binned_mean_merch_trans'] = pd.cut(train_df['new_mean_merch_trans'], bins)

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_mean_merch_trans", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('Binned Mean of New Merchant Transactions', fontsize=12)  
plt.ylabel('Loyalty Score', fontsize=12)
plt.title("Mean of New Merchant Transaction Value (Binned) Distribution", fontsize=14)
plt.tight_layout()  
plt.show()


import lightgbm as lgb
from sklearn.model_selection import KFold
import numpy as np

def run_lgb(train_df, test_df, target_col, cols_to_use, n_folds=5):
    # 1. 简化参数（去掉可能冲突的类别特征、复杂正则，回归基础稳定配置）
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,  
        "min_child_samples": 20,  
        "learning_rate": 0.05,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_freq": 3,
        "verbosity": -1,
        "n_estimators": 1000,  # 适当减少迭代次数，加快训练
        "boosting_type": "gbdt",
        "device_type": "cpu",
        "seed": 2025,
        "early_stopping_round": 100 
    }
    
    # 2. 初始化变量
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2025)
    pred_test = np.zeros(test_df.shape[0])
    models = []
    fold_scores = []
    
    # 3. 5折交叉验证（简化逻辑，去掉冗余）
    for n_fold, (train_idx, val_idx) in enumerate(kf.split(train_df[cols_to_use])):
        train_X, val_X = train_df[cols_to_use].iloc[train_idx], train_df[cols_to_use].iloc[val_idx]
        train_y, val_y = train_df[target_col].iloc[train_idx], train_df[target_col].iloc[val_idx]
        
        # 构建数据集（不指定reference，避免依赖冲突）
        train_data = lgb.Dataset(train_X, label=train_y)
        val_data = lgb.Dataset(val_X, label=val_y)
        
        # 训练模型（简化参数传递）
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            verbose_eval=200,
            num_boost_round=params["n_estimators"]
        )
        
        models.append(model)
        fold_scores.append(model.best_score["valid_0"]["rmse"])
        pred_test += model.predict(test_df[cols_to_use], num_iteration=model.best_iteration) / kf.n_splits
        
        print(f"第{n_fold+1}折 验证集RMSE：{fold_scores[-1]:.4f}")
    
    # 4. 输出结果
    print("="*50)
    print(f"5折平均RMSE：{np.mean(fold_scores):.4f}")
    print(f"RMSE标准差：{np.std(fold_scores):.4f}")
    print("="*50)
    
    return pred_test, models

# 定义特征：排除非数值列，避免数据类型冲突
cols_to_use = [col for col in train_df.columns if col not in ["card_id", "target", "first_active_month"]]
# 额外过滤：只保留数值型特征（彻底避免类别特征冲突）
cols_to_use = [col for col in cols_to_use if train_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
print("用于建模的特征数：", len(cols_to_use))
print("前10个特征：", cols_to_use[:10])

# 调用训练
pred_test, models = run_lgb(train_df, test_df, target_col="target", cols_to_use=cols_to_use)


fig, ax = plt.subplots(figsize=(12,18))
# 使用 models 列表中的最后一个模型
lgb.plot_importance(models[-1], max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance (last)", fontsize=15)
plt.show()


submission = pd.DataFrame({
    "card_id": test_df["card_id"],
    "target": pred_test
})
submission.to_csv("submission.csv", index=False)

import os
print(os.listdir("/kaggle/working"))

