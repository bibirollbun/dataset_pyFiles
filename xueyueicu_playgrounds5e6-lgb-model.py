import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取CSV文件
file_path = "/kaggle/input/playground-series-s5e6/train.csv"  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 2. 检查数值列的最大值/最小值
numeric_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
numeric_stats = df[numeric_cols].agg(["min", "max"])

print("数值列的最大值和最小值:")
print(numeric_stats)

# 4. 统计分类列的种类个数及所有种类
categorical_cols = ["Soil Type", "Crop Type", "Fertilizer Name"]

print("\n分类列的种类统计:")
for col in categorical_cols:
    unique_values = df[col].unique()
    print(f"\n{col}:")
    print(f"  种类个数: {len(unique_values)}")
    print(f"  所有种类: {', '.join(map(str, unique_values))}")


# 导入必要的库
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 准备特征和目标变量
X = df[numeric_cols + ["Soil Type", "Crop Type"]]
y = df["Fertilizer Name"]

# 对分类特征进行编码
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

X["Soil Type"] = le_soil.fit_transform(X["Soil Type"])
X["Crop Type"] = le_crop.fit_transform(X["Crop Type"])
y = le_fertilizer.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置模型参数
params = {
    'objective': 'multiclass',
    'num_class': len(le_fertilizer.classes_),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 训练模型
model = lgb.train(params,
                 train_data,
                 num_boost_round=1000,
                 valid_sets=[test_data],
                 callbacks=[lgb.early_stopping(stopping_rounds=10)])

# 预测并评估模型
y_pred = model.predict(X_test)
y_pred_class = y_pred.argmax(axis=1)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred_class)
print(f"模型准确率: {accuracy:.4f}")

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred_class, 
                          target_names=le_fertilizer.classes_))

# 特征重要性
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance()
})
importance = importance.sort_values('importance', ascending=False)
print("\n特征重要性:")
print(importance)



# 读取测试数据
import numpy as np
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 对测试数据进行相同的预处理
X_test_new = test_df[numeric_cols + ["Soil Type", "Crop Type"]].copy()  # 创建副本避免SettingWithCopyWarning
X_test_new["Soil Type"] = le_soil.transform(X_test_new["Soil Type"])
X_test_new["Crop Type"] = le_crop.transform(X_test_new["Crop Type"])

# 确保数据类型为数值型
X_test_new = X_test_new.astype(float)

# 获取预测概率
y_pred_proba = model.predict(X_test_new)

# 获取每个样本的top3预测结果
top3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]

# 将二维数组转换为一维数组进行转换
top3_fertilizers = []
for row in top3_indices:
    row_fertilizers = le_fertilizer.inverse_transform(row)
    top3_fertilizers.append(row_fertilizers)

# 创建提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'fertilizer': [' '.join(ferts) for ferts in top3_fertilizers]
})

# 保存结果
submission.to_csv('submission3.csv', index=False)


