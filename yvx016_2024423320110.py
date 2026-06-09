# 2024423320110 李晓文
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

# 设置随机种子确保结果可复现
np.random.seed(42)

# 1. 加载数据 - 使用更健壮的路径处理
try:
    # 尝试从指定路径加载数据
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
except:
    # 如果找不到文件，使用当前目录（适合Kaggle环境）
    print("无法从指定路径加载数据，尝试从当前目录加载...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

# 2. 数据探索与预处理
print(f"训练集形状: {train.shape}")
print(f"测试集形状: {test.shape}")

# 检查缺失值
print("\n训练集缺失值统计:")
print(train.isnull().sum())
print("\n测试集缺失值统计:")
print(test.isnull().sum())

# 处理缺失值（如果有）
train = train.fillna(train.mean())
test = test.fillna(test.mean())

# 3. 编码类别特征
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

train['Soil Type'] = soil_encoder.fit_transform(train['Soil Type'])
test['Soil Type'] = soil_encoder.transform(test['Soil Type'])

train['Crop Type'] = crop_encoder.fit_transform(train['Crop Type'])
test['Crop Type'] = crop_encoder.transform(test['Crop Type'])

train['Fertilizer Name'] = fertilizer_encoder.fit_transform(train['Fertilizer Name'])

# 4. 特征和标签
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)  # 测试集没有目标列

# 5. 使用StratifiedKFold进行交叉验证
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros_like(y)
test_preds = np.zeros((len(test), len(np.unique(y))))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== 第 {fold + 1}/{n_splits} 折 =====")

    # 划分训练集和验证集
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 模型定义
    model = lgb.LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        n_estimators=1000,  # 增加树的数量
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # 模型训练，添加早停策略
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50
    )

    # 验证集预测
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds

    # 记录验证集准确率
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"第 {fold + 1} 折验证集准确率: {val_accuracy:.4f}")

    # 测试集预测
    test_preds += model.predict_proba(X_test) / n_splits

# 计算整体交叉验证分数
cv_accuracy = accuracy_score(y, oof_preds)
print(f"\n整体交叉验证准确率: {cv_accuracy:.4f}")

# 6. Top5 预测
top5_pred_indices = np.argsort(test_preds, axis=1)[:, -5:][:, ::-1]
top5_pred_labels = np.array([fertilizer_encoder.inverse_transform(row) for row in top5_pred_indices])

# 7. 保存结果 - 使用更健壮的路径处理
output = pd.DataFrame(top5_pred_labels, columns=[f'Top{i + 1}' for i in range(5)])
output['id'] = test['id']
output = output[['id', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5']]

# 创建输出目录（如果不存在）
os.makedirs('output', exist_ok=True)
output_path = 'output/top5_predictions.csv'
output.to_csv(output_path, index=False)

print(f"✅ Top5预测结果已生成并保存为 {output_path}")

# 8. 特征重要性分析
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n特征重要性:")
print(feature_importance)

