import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split



# 导入数据集路径
path = Path('/kaggle/input/tabular-playground-series-nov-2022/')

# 加载测试集和训练集
sample_submission = pd.read_csv(path / 'sample_submission.csv', index_col='id')
train_labels = pd.read_csv(path / 'train_labels.csv', index_col='id')
# 验证是否成功
print(sample_submission.head())
print(train_labels.head())

# 加载数据集
submission_files = sorted([
    str(path / 'submission_files' / f) 
    for f in os.listdir(path / 'submission_files') 
    if f.endswith('.csv')
])
dfs = [pd.read_csv(f).set_index('id') for f in submission_files]
# 验证是否成功
print(f"数: {len(submission_files)}")
print(submission_files[:5]) 

# 打包数据集 将5000个一维数组水平堆叠成二维矩阵
pred_matrix = np.column_stack([df['pred'].values for df in dfs])
# 验证是否成功
print(f"pred_matrix shape: {pred_matrix.shape}")

# 洲练集和测试集标签
train_id = train_labels.index
test_id = sample_submission.index
# 洲练集个数
train_num = 20000

# 真实标签值
train_true_value = train_labels.loc[train_id]['label'].values

# 截取前20000行数据
train_data_np = pred_matrix[:train_num]
train_data_pd = pd.DataFrame(pred_matrix[:train_num], index=train_id)
train_data_pd_value = train_data_pd.loc[train_id].values


# Logit Average 模型
# 定义安全 Logit Average
def safe_logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))

def logit_average(pred_matrix):
    logits = safe_logit(pred_matrix)
    avg_logits = np.mean(logits, axis=1)
    return 1 / (1 + np.exp(-avg_logits))  # sigmoid

# 执行 Logit Average
logit_preds = logit_average(train_data_np)

# 构建索引对齐的 Series，并提取训练集 ======
pred_series = pd.Series(logit_preds, index=train_id)
train_preds = pred_series.loc[train_id].values

# 评估 Log Loss
logit_loss = log_loss(train_true_value, train_preds)
print(f"Logit Average Model Log Loss: {logit_loss:.6f}")


# Stacking（堆叠法）模型
# 二级模型：Logistic 回归
meta_model = LogisticRegression(max_iter=200, solver='liblinear', random_state=42)

# 交叉验证（建议使用 Stratified）
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_meta = np.zeros(len(train_id))  # 存储 out-of-fold 预测

for train_idx, val_idx in kf.split(train_data_pd_value, train_true_value):
    X_tr, X_val = train_data_pd_value[train_idx], train_data_pd_value[val_idx]
    y_tr, y_val = train_true_value[train_idx], train_true_value[val_idx]

    meta_model.fit(X_tr, y_tr)
    oof_meta[val_idx] = meta_model.predict_proba(X_val)[:, 1]

# 评估
stack_loss = log_loss(train_true_value, np.clip(oof_meta, 1e-5, 1 - 1e-5))
print(f"Stacking Model Log Loss: {stack_loss:.6f}")


# Blending（混合法）模型
# 使用 holdout 进行划分（最后 20% 作为 meta model 的训练数据）
split_point = int(train_num * 0.8)
X_base = train_data_pd_value[:split_point]
y_base = train_true_value[:split_point]
X_holdout = train_data_pd_value[split_point:]
y_holdout = train_true_value[split_point:]

# 基础模型预测已固定为 pred_matrix，不需再训练；我们只训练二级模型
# 这里使用 Logistic Regression 作为融合器
meta_model = LogisticRegression(max_iter=200, solver='liblinear', random_state=42)
meta_model.fit(X_holdout, y_holdout)

# 在 holdout 上预测（模拟 test）
blend_preds = meta_model.predict_proba(X_holdout)[:, 1]

# 评估 logloss
blend_loss = log_loss(y_holdout, np.clip(blend_preds, 1e-5, 1 - 1e-5))
print(f"Blending Model Log Loss (hold out 20%): {blend_loss:.6f}")


# Meta-Ensemble 模型 Blending + Weighted Average + Logit Average 的融合集成
# 1. Weighted Average
weights = np.linspace(1, 0.1, train_data_np.shape[1])
weights /= weights.sum()
pred_weighted = np.average(train_data_np, axis=1, weights=weights)

# 2. Logit Average
def safe_logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))

logits = safe_logit(train_data_np)
avg_logits = np.mean(logits, axis=1)
pred_logit = 1 / (1 + np.exp(-avg_logits))  # sigmoid

# 3. Blending 模型（Logistic Regression）
# 划分 blending 训练集和验证集
X_blend_train, X_blend_val, y_blend_train, y_blend_val = train_test_split(
    train_data_np, train_true_value, test_size=0.3, random_state=42
)

blend_model = LogisticRegression(max_iter=200, solver='liblinear')
blend_model.fit(X_blend_train, y_blend_train)
pred_blend = blend_model.predict_proba(train_data_np)[:, 1]

# 4. 构建 Meta-Ensemble 特征集
X_meta = np.vstack([pred_weighted, pred_logit, pred_blend]).T

# 使用一个轻量模型作为次级集成器（可以换成 GBDT、MLP 等）
meta_model = LogisticRegression(max_iter=200, solver='liblinear')
meta_model.fit(X_meta, train_true_value)
meta_preds = meta_model.predict_proba(X_meta)[:, 1]

# 5. 评估最终融合模型
meta_loss = log_loss(train_true_value, np.clip(meta_preds, 1e-5, 1 - 1e-5))
print(f"Meta-Ensemble Fusion Log Loss: {meta_loss:.6f}")


# Boosting（提升法）模型
# LightGBM 数据集对象
dtrain = lgb.Dataset(train_data_pd_value, label=train_true_value)

# LightGBM 参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'learning_rate': 0.05,
    'num_leaves': 15,
    'feature_fraction': 0.9,
    'seed': 42,
}

# 运行 CV，不带 verbose_eval
cv_results = lgb.cv(
    params,
    dtrain,
    num_boost_round=500,
    nfold=5,
    stratified=True,
    seed=42
)

# 查看 key 来判断正确的字段名
print(f"cv_results keys: {cv_results.keys()}")

# 自动查找 logloss 列名
logloss_key = [k for k in cv_results.keys() if "logloss" in k]
assert len(logloss_key) > 0, "无法在 CV 结果中找到 logloss key"
logloss_key = logloss_key[0]

# 找出最佳轮数并训练
best_iter = np.argmin(cv_results[logloss_key]) + 1
print(f"Best iteration from CV: {best_iter}")

# 最终训练
final_model = lgb.train(params, dtrain, num_boost_round=best_iter)

# 模型预测
preds_boost = final_model.predict(train_data_pd_value)
boost_loss = log_loss(train_true_value, np.clip(preds_boost, 1e-5, 1 - 1e-5))
print(f"Boosting Fusion Model Log Loss: {boost_loss:.6f}")


# 原始顺序下的各模型 Log Loss（请替换成你自己的实际结果）
results = {
    "Logit Average": logit_loss,
    "Blending Model": blend_loss,
    "Stacking Model": stack_loss,
    "Meta-Ensemble Fusion": meta_loss,
    "Boosting Fusion Model": boost_loss,
}

# 保留原始顺序
model_names = list(results.keys())
log_losses = list(results.values())

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(model_names, log_losses, marker='o', linestyle='-', color='indigo', linewidth=2, markersize=8)

# 添加数据标签
for i, loss in enumerate(log_losses):
    plt.text(i, loss + 0.0003, f"{loss:.6f}", ha='center', fontsize=9)

# 图形设置
plt.title("Log Loss Comparison of Ensemble Models", fontsize=14)
plt.ylabel("Log Loss (lower is better)", fontsize=12)
plt.xticks(rotation=15)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



print("Blending（混合法）模型虽然最优，但存在过拟合现象，所以综合考虑选取Meta-Ensemble 模型")

# 测试集模型输出矩阵
pred_matrix_test = pred_matrix[20000:40000]
# 1. 加权平均
pred_weighted_test = np.average(pred_matrix_test, axis=1, weights=weights)

# 2. Logit Average
def safe_logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))
logits_test = safe_logit(pred_matrix_test)
avg_logits_test = np.mean(logits_test, axis=1)
pred_logit_test = 1 / (1 + np.exp(-avg_logits_test))

# 3. Blending 模型输出（注意这是 5000维 → scalar）
pred_blend_test = blend_model.predict_proba(pred_matrix_test)[:, 1]

# 4. 构造 Meta-Ensemble 特征（3维：加权平均、logit平均、blending输出）
X_meta_test = np.vstack([
    pred_weighted_test,
    pred_logit_test,
    pred_blend_test
]).T

# 使用训练好的 meta_model 做最终预测
meta_preds_test = meta_model.predict_proba(X_meta_test)[:, 1]

# 保存预测结果
submission_df = pd.DataFrame({
    'id': test_id,     # 确保 test_id 是长度为 20000 的对应 id
    'pred': meta_preds_test
})
submission_df.to_csv('meta_ensemble_submission.csv', index=False)
print("预测并保存完成：meta_ensemble_submission.csv")

