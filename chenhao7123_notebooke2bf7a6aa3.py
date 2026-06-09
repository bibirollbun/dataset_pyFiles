# 学号: 2024423310229 姓名: 张杰

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib
import gc
import os

# ===== 函数定义: MAP@5计算 =====
def map_at_k(y_true, y_pred_prob, k=5):
    """计算MAP@k指标"""
    map_score = 0
    y_true = np.array(y_true)
    for i in range(len(y_true)):
        topk_idx = np.argsort(y_pred_prob[i])[-k:][::-1]
        for j in range(k):
            if topk_idx[j] == y_true[i]:
                map_score += 1 / (j + 1)
                break
    return map_score / len(y_true)

# ===== 1. 定义文件路径 =====
TRAIN_PATH = '/kaggle/input/playground-series-s5e6/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e6/test.csv'

# ===== 2. 加载数据 =====
print("正在加载数据...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"训练数据形状: {train.shape}")
print(f"测试数据形状: {test.shape}")

# ===== 3. 精简特征工程 =====
print("\n进行精简特征工程...")

train_engineered = train.copy()
test_engineered = test.copy()

# 保留核心特征
train_engineered['NP_ratio'] = train['Nitrogen'] / (train['Phosphorous'] + 1e-8)
test_engineered['NP_ratio'] = test['Nitrogen'] / (test['Phosphorous'] + 1e-8)

train_engineered['Temp_Humidity'] = train['Temparature'] * train['Humidity'] / 100
test_engineered['Temp_Humidity'] = test['Temparature'] * test['Humidity'] / 100

# 组合特征标签编码
train_engineered['Soil_Crop'] = train['Soil Type'] + "_" + train['Crop Type']
test_engineered['Soil_Crop'] = test['Soil Type'] + "_" + test['Crop Type']
le_soil_crop = LabelEncoder()
combined = pd.concat([train_engineered['Soil_Crop'], test_engineered['Soil_Crop']])
le_soil_crop.fit(combined)
train_engineered['Soil_Crop_Code'] = le_soil_crop.transform(train_engineered['Soil_Crop'])
test_engineered['Soil_Crop_Code'] = le_soil_crop.transform(test_engineered['Soil_Crop'])

# ===== 4. 数据预处理 =====
print("\n进行数据预处理...")

# 标签编码肥料名称
le = LabelEncoder()
train_engineered['target'] = le.fit_transform(train_engineered['Fertilizer Name'])
print(f"肥料种类: {len(le.classes_)}种")

# 核心特征列
features = [
    'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 
    'Potassium', 'Phosphorous', 'NP_ratio',
    'Temp_Humidity', 'Soil_Crop_Code'
]

X = train_engineered[features]
y = train_engineered['target']
X_test = test_engineered[features]

# ===== 5. GPU加速参数设置 =====
print("\n启用GPU加速训练...")
params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 35,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'min_child_samples': 30,
    'lambda_l1': 0.4,
    'lambda_l2': 0.4,
    'verbose': -1,
    
    # 核心GPU设置
    'device': 'gpu',  # 启用GPU加速
    'gpu_platform_id': 0,  # 平台ID
    'gpu_device_id': 0,   # 设备ID
}

# ===== 6. 3折交叉验证训练 =====
print("\n开始3折交叉验证(GPU加速)...")
n_folds = 3
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
test_preds = np.zeros((X_test.shape[0], len(le.classes_)))
val_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n======= 第{fold+1}折训练 =======")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 创建GPU加速数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    # GPU加速训练
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,  # 减少迭代次数
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),  # 减少早停轮数
            lgb.log_evaluation(period=50)  # 减少日志频率
        ]
    )
    
    # 验证集评估
    val_preds = model.predict(X_val)
    score = map_at_k(y_val, val_preds, k=5)
    val_scores.append(score)
    print(f"Fold {fold+1} MAP@5: {score:.5f}")
    
    # 测试集预测
    test_preds += model.predict(X_test) / n_folds
    gc.collect()  # 释放内存

# 平均验证分数
mean_score = np.mean(val_scores)
print(f"\n平均 MAP@5: {mean_score:.5f}")

# ===== 7. 生成提交文件 =====
print("\n生成提交文件...")
top5_preds = []
for probs in test_preds:
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_names = le.inverse_transform(top5_idx)
    top5_preds.append(" ".join(top5_names))

submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': top5_preds
})
submission.to_csv('submission.csv', index=False)
print("提交文件已生成")

