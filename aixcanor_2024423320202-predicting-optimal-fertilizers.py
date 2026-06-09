import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置Kaggle数据路径
kaggle_path = '/kaggle/input'

# 显示Kaggle输入目录中的所有数据集
print("Kaggle输入目录中的所有数据集：")
print("=" * 50)

datasets = os.listdir(kaggle_path)
for dataset in datasets:
    dataset_path = os.path.join(kaggle_path, dataset)
    print(f"\n数据集: {dataset}")
    print(f"路径: {dataset_path}")
    
    # 列出数据集中的文件
    try:
        files = os.listdir(dataset_path)
        print(f"包含 {len(files)} 个文件:")
        for file in files:
            file_path = os.path.join(dataset_path, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file} ({file_size:.2f} KB)")
    except Exception as e:
        print(f"  无法读取数据集内容: {e}")

print("\n" + "=" * 50)
print(f"总共找到 {len(datasets)} 个数据集")

# 设置训练和测试数据路径
train_path = '/kaggle/input/playground-series-s5e6/train.csv'
test_path = '/kaggle/input/playground-series-s5e6/test.csv'

print(f"\n训练数据路径: {train_path}")
print(f"测试数据路径: {test_path}")

# 加载数据
try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("\n数据加载成功!")
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    # 显示训练集和测试集的前几行
    print("\n训练集预览:")
    print(train_df.head(3))
    print("\n测试集预览:")
    print(test_df.head(3))
    
    # 检查目标变量是否存在
    if 'Fertilizer Name' not in train_df.columns:
        raise KeyError("训练集中缺少目标变量 'Fertilizer Name'")
    
    # 打印数据集的列名
    print("\n训练集列名:")
    print(train_df.columns)
    
except Exception as e:
    print(f"\n数据加载失败: {e}")
    raise SystemExit("无法加载数据，程序退出。")

# 数据探索分析
print("\n数据探索分析:")
print("1. 目标变量分布:")
plt.figure(figsize=(12, 6))
sns.countplot(x='Fertilizer Name', data=train_df)
plt.title('肥料类型分布')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('target_distribution.png')
plt.show()

# 数值特征分析
print("\n2. 数值特征分布:")
num_features = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temparature', 'Humidity', 'Moisture']
if set(num_features).issubset(train_df.columns):
    train_df[num_features].hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    plt.savefig('numerical_features_distribution.png')
    plt.show()
else:
    print("警告: 缺少部分数值特征")

# 类别特征分析
print("\n3. 类别特征分布:")
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='Soil Type', data=train_df)
plt.title('土壤类型分布')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.countplot(x='Crop Type', data=train_df)
plt.title('作物类型分布')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('categorical_features_distribution.png')
plt.show()

# 特征与目标变量的关系
print("\n4. 特征与目标变量的关系:")
if set(['Nitrogen', 'Phosphorous', 'Potassium']).issubset(train_df.columns):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.boxplot(x='Fertilizer Name', y='Nitrogen', data=train_df)
    plt.xticks(rotation=45)
    plt.title('氮含量分布')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(x='Fertilizer Name', y='Phosphorous', data=train_df)
    plt.xticks(rotation=45)
    plt.title('磷含量分布')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(x='Fertilizer Name', y='Potassium', data=train_df)
    plt.xticks(rotation=45)
    plt.title('钾含量分布')
    
    plt.tight_layout()
    plt.savefig('nutrient_distribution_by_fertilizer.png')
    plt.show()

# 特征列表
soil_features = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type']

# 分离特征和目标变量
X = train_df[soil_features]
y_str = train_df['Fertilizer Name']  # 保存原始字符串标签

# 编码目标变量
print("\n编码目标变量...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_str)
print(f"类别数量: {len(label_encoder.classes_)}")
print("类别映射:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i}: {class_name}")

# 保存类别映射
class_mapping = pd.DataFrame({
    'encoded_label': range(len(label_encoder.classes_)),
    'fertilizer_type': label_encoder.classes_
})
class_mapping.to_csv('class_mapping.csv', index=False)
print("类别映射已保存为 'class_mapping.csv'")

# 处理缺失值
print("\n处理缺失值...")
imputer = SimpleImputer(strategy='mean')
num_cols = [col for col in num_features if col in X.columns]
if num_cols:
    X[num_cols] = imputer.fit_transform(X[num_cols])
    print("数值特征缺失值处理完成")

# 编码类别特征
print("\n编码类别特征...")
X = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'])
print(f"编码后特征数量: {X.shape[1]}")

# 标准化数值特征
print("\n标准化数值特征...")
scaler = StandardScaler()
if num_cols:
    X[num_cols] = scaler.fit_transform(X[num_cols])
    print("数值特征标准化完成")

# 划分训练集和验证集
print("\n划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")

# 准备测试集
print("\n准备测试集...")
X_test = test_df[soil_features].copy()

# 处理测试集缺失值
if num_cols:
    X_test[num_cols] = imputer.transform(X_test[num_cols])

# 编码类别特征
X_test = pd.get_dummies(X_test, columns=['Soil Type', 'Crop Type'])

# 确保测试集有相同的列
for col in X.columns:
    if col not in X_test.columns:
        X_test[col] = 0

# 按训练集的列顺序排序
X_test = X_test[X.columns]

# 标准化数值特征
if num_cols:
    X_test[num_cols] = scaler.transform(X_test[num_cols])

print(f"测试集形状: {X_test.shape}")
print("\n数据预处理完成!")

# 模型训练
print("\n开始模型训练...")
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1),
    "CatBoost": CatBoostClassifier(iterations=300, learning_rate=0.1, random_state=42, verbose=False)
}


# 训练模型并评估
results = {}
for name, model in models.items():
    print(f"\n训练 {name} 模型...")
    model.fit(X_train, y_train)
    
    # 验证集预测
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    results[name] = accuracy
    
    print(f"{name} 验证集准确率: {accuracy:.4f}")
    print(f"{name} 分类报告:\n{classification_report(y_val, y_pred, target_names=label_encoder.classes_)}")

# 选择最佳模型
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n最佳模型: {best_model_name}, 准确率: {results[best_model_name]:.4f}")

# 在测试集上进行预测
print("\n在测试集上进行预测...")
test_probs = best_model.predict_proba(X_test)
test_preds = best_model.predict(X_test)

# 解码预测结果
decoded_preds = label_encoder.inverse_transform(test_preds)

# 创建提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': decoded_preds
})

# 保存提交文件
submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f"\n提交文件已保存: {submission_path}")
print(f"提交文件行数: {len(submission)}")
print(f"id范围: {submission['id'].min()} - {submission['id'].max()}")

# 显示预测结果示例
print("\n预测结果示例:")
sample_indices = np.random.choice(len(test_probs), 5, replace=False)
for idx in sample_indices:
    sample_id = test_df.iloc[idx]['id']
    pred_class = decoded_preds[idx]
    prob = np.max(test_probs[idx])
    print(f"id={sample_id}: 预测肥料 = {pred_class}, 置信度 = {prob:.4f}")


print("\n特征重要性分析:")
if hasattr(best_model, 'feature_importances_'):
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
    plt.title(f'{best_model_name} - 特征重要性 (Top 20)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    print("Top 10 重要特征:")
    print(feature_importances.head(10))
else:
    print(f"{best_model_name} 模型不支持特征重要性分析")

print("\n模型训练和预测完成!")

