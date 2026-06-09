# 学号 2024423310132 姓名 钟帅邦

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


# 学号 2024423310132 姓名 钟帅邦

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


print("=" * 50)
print("数据处理阶段")
print("=" * 50)


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
train_df.head()


print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")


# 数据基本信息可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 目标变量分布
target_counts = train_df['Fertilizer Name'].value_counts()
axes[0,0].pie(target_counts.values[:10], labels=target_counts.index[:10], autopct='%1.1f%%')
axes[0,0].set_title('Top 10 Fertilizer Distribution')

# 土壤类型分布
soil_counts = train_df['Soil Type'].value_counts()
axes[0,1].bar(range(len(soil_counts)), soil_counts.values)
axes[0,1].set_title('Soil Type Distribution')
axes[0,1].set_xticks(range(len(soil_counts)))
axes[0,1].set_xticklabels(soil_counts.index, rotation=45)

# 作物类型分布
crop_counts = train_df['Crop Type'].value_counts()
axes[1,0].bar(range(len(crop_counts)), crop_counts.values)
axes[1,0].set_title('Crop Type Distribution')
axes[1,0].set_xticks(range(len(crop_counts)))
axes[1,0].set_xticklabels(crop_counts.index, rotation=45)

# 数值特征相关性热图
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
corr_matrix = train_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
axes[1,1].set_title('Numeric Features Correlation')

plt.tight_layout()
plt.show()


print("\n数据预处理...")


# 编码目标变量
target = 'Fertilizer Name'  # 指定目标变量列名
label_encoder = LabelEncoder()  # 创建编码器实例

# 对训练数据中的目标变量进行拟合和转换
y_full = label_encoder.fit_transform(train_df[target])

print(f"Original unique values: {label_encoder.classes_}")
print(f"目标变量类别数: {len(label_encoder.classes_)}")


# 独热编码对应类型的特征比如土壤种类、农作物种类
categorical_features = ['Soil Type', 'Crop Type']  

# 创建编码器实例，设置参数
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 拟合训练数据并转换
one_hot_encoder.fit(train_df[categorical_features])
train_encoded_features = one_hot_encoder.transform(train_df[categorical_features])


test_encoded_features = one_hot_encoder.transform(test_df[categorical_features])


# 获取编码后的特征列名
encoded_cols = one_hot_encoder.get_feature_names_out(categorical_features)

# 创建带列名和索引的训练集DataFrame
train_encoded_df = pd.DataFrame(
    train_encoded_features,           # 编码后的特征矩阵
    columns=encoded_cols,             # 使用生成的列名
    index=train_df.index              # 保留原始数据的索引
)

# 同样处理测试集
test_encoded_df = pd.DataFrame(
    test_encoded_features, 
    columns=encoded_cols, 
    index=test_df.index
)




# 1. 移除不需要的列（分类特征、目标变量和ID）
X_full = train_df.drop(columns=categorical_features + [target, 'id'])
X_full = pd.concat([X_full, train_encoded_df], axis=1)

X_test = test_df.drop(columns=categorical_features + ['id'])
X_test = pd.concat([X_test, test_encoded_df], axis=1)

# 划分训练验证集用于评估
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)


print(f"训练集形状: {X_train.shape}")
print(f"验证集形状: {X_val.shape}")


# store ids to use in the output file
ids = test_df['id']


def map_at_k(y_true, y_pred_proba, k=5):
    """计算MAP@K评分"""
    num_samples = len(y_true)
    map_score = 0.0
    
    for i in range(num_samples):
        # 获取前k个预测类别
        top_k_preds = np.argsort(y_pred_proba[i])[-k:][::-1]
        true_label = y_true[i]
        
        # 计算精度
        precision_sum = 0.0
        for j, pred in enumerate(top_k_preds):
            if pred == true_label:
                precision_sum += 1.0 / (j + 1)
                break
        
        map_score += precision_sum
    
    return map_score / num_samples


print("\n" + "=" * 50)
print("XGBoost模型训练")
print("=" * 50)


# 初始化 XGBoost 分类器
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',          
    num_class=len(label_encoder.classes_), 
    eval_metric='mlogloss',               # 评估指标：多类对数损失
    use_label_encoder=False,              
    n_estimators=300,                     # 树的数量
    max_depth=5,                        
    learning_rate=0.1,                    # 学习率
    subsample=0.8,                        # 训练样本采样比例
    colsample_bytree=0.8,                 # 特征采样比例
    random_state=42                       
)

# 训练模型
xgb_model.fit(X_train, y_train)


# 获取模型预测的概率矩阵
#y_probs_xgb = xgb_model.predict_proba(X_test)

y_val_proba_xgb = xgb_model.predict_proba(X_val)
y_test_proba_xgb = xgb_model.predict_proba(X_test)  # 重命名原始变量

# XGBoost评估
xgb_map5 = map_at_k(y_val, y_val_proba_xgb, k=5)
print(f"XGBoost MAP@5: {xgb_map5:.4f}")




print("\n" + "=" * 50)
print("随机森林模型训练")
print("=" * 50)


rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)


# 获取随机森林模型的预测概率
y_val_proba_rf = rf_model.predict_proba(X_val)
y_test_proba_rf = rf_model.predict_proba(X_test)  # 重命名原始变量

# 随机森林评估
rf_map5 = map_at_k(y_val, y_val_proba_rf, k=5)
print(f"随机森林 MAP@5: {rf_map5:.4f}")




print("\n" + "=" * 50)
print("LightGBM模型训练")
print("=" * 50)


X_train_lgb = train_df.drop(columns=[target, 'id'])
X_test_lgb = test_df.drop(columns=['id'])

# Method 1: Use .loc[] with the actual indices (recommended)
X_val_lgb = X_train_lgb.loc[X_val.index].copy()
X_train_lgb = X_train_lgb.loc[X_train.index].copy()

# Alternative Method 2: Reset index before using iloc
# X_train_lgb_reset = X_train_lgb.reset_index(drop=True)
# train_positions = [X_train_lgb_reset.index[X_train_lgb_reset.index.get_loc(idx)] for idx in X_train.index if idx in X_train_lgb.index]
# val_positions = [X_train_lgb_reset.index[X_train_lgb_reset.index.get_loc(idx)] for idx in X_val.index if idx in X_train_lgb.index]
# X_train_lgb = X_train_lgb_reset.iloc[train_positions].copy()
# X_val_lgb = X_train_lgb_reset.iloc[val_positions].copy()

# 合并训练集和测试集以便统一处理
combined_lgb = pd.concat([X_train_lgb, X_val_lgb, X_test_lgb], axis=0)

# 将分类特征转换为pandas的category类型
for col in categorical_features:  # ['Soil Type', 'Crop Type']
    combined_lgb[col] = combined_lgb[col].astype('category')

# 分离回训练集和测试集
X_train_lgb = combined_lgb.iloc[:len(X_train_lgb), :].copy()
X_val_lgb = combined_lgb.iloc[len(X_train_lgb):len(X_train_lgb)+len(X_val_lgb), :].copy()
X_test_lgb = combined_lgb.iloc[len(X_train_lgb)+len(X_val_lgb):, :].copy()

# 验证形状
print(f"X_train_lgb shape: {X_train_lgb.shape}")
print(f"X_val_lgb shape: {X_val_lgb.shape}")
print(f"X_test_lgb shape: {X_test_lgb.shape}")

# 验证索引匹配
print(f"X_train indices match: {X_train_lgb.index.equals(X_train.index)}")
print(f"X_val indices match: {X_val_lgb.index.equals(X_val.index)}")


# 创建 LightGBM 训练数据集，指定分类特征
lgb_train = lgb.Dataset(
    X_train_lgb,                    # 修改后的训练数据
    label=y_train,          
    # 特征矩阵（包含 category 类型特征）
    categorical_feature=categorical_features  # 指定分类特征列名
)

lgb_val = lgb.Dataset(X_val_lgb, label=y_val, categorical_feature=categorical_features, reference=lgb_train)

# 设置模型参数
params = {
    'objective': 'multiclass',    
    'num_class': len(label_encoder.classes_),  
    'metric': 'multi_logloss',      # 评估指标：多类对数损失
    'learning_rate': 0.1,           # 学习率
    'num_leaves': 31,           
    'random_state': 42,             # 随机种子
    'verbosity': -1                 
}

# 训练模型
lgb_model = lgb.train(
    params,                 # 参数字典
    lgb_train,              # 训练数据集
    num_boost_round=300,     # 提升轮数（树的数量）
    valid_sets=[lgb_val], 
    callbacks=[lgb.early_stopping(50)]
)


# 获取模型预测的概率矩阵
y_val_proba_lgb = lgb_model.predict(X_val_lgb)
y_test_proba_lgb = lgb_model.predict(X_test_lgb)  # 重命名原始变量

# LightGBM评估
lgb_map5 = map_at_k(y_val, y_val_proba_lgb, k=5)
print(f"LightGBM MAP@5: {lgb_map5:.4f}")


print("\n" + "=" * 50)
print("模型集成和最终评估")
print("=" * 50)


assert y_test_proba_lgb.shape == y_test_proba_xgb.shape == y_test_proba_rf.shape
# 模型集成
y_val_proba_ensemble = (y_val_proba_xgb + y_val_proba_rf + y_val_proba_lgb) / 3
y_test_proba_ensemble = (y_test_proba_xgb + y_test_proba_rf + y_test_proba_lgb) / 3

# 集成模型评估
ensemble_map5 = map_at_k(y_val, y_val_proba_ensemble, k=5)
print(f"集成模型 MAP@5: {ensemble_map5:.4f}")

models = ['XGBoost', 'Random Forest', 'LightGBM', 'Ensemble']
map5_scores = [xgb_map5, rf_map5, lgb_map5, ensemble_map5]

plt.figure(figsize=(12, 8))

# MAP@5分数对比
plt.subplot(2, 2, 1)
bars = plt.bar(models, map5_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Model Performance Comparison (MAP@5)')
plt.ylabel('MAP@5 Score')
plt.ylim(0, max(map5_scores) * 1.1)
for bar, score in zip(bars, map5_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{score:.4f}', ha='center', va='bottom')

# 特征重要性(XGBoost)
plt.subplot(2, 2, 2)
feature_importance = xgb_model.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:]
top_features_names = [X_train.columns[i] for i in top_features_idx]
top_features_scores = feature_importance[top_features_idx]

plt.barh(range(len(top_features_names)), top_features_scores)
plt.yticks(range(len(top_features_names)), top_features_names)
plt.title('XGBoost Feature Importance (Top 10)')
plt.xlabel('Importance Score')

# 预测概率分布
plt.subplot(2, 2, 3)
ensemble_preds = np.argmax(y_val_proba_ensemble, axis=1)
accuracy = accuracy_score(y_val, ensemble_preds)
plt.hist(np.max(y_val_proba_ensemble, axis=1), bins=30, alpha=0.7, color='skyblue')
plt.title(f'Ensemble Model Prediction Confidence\nValidation Accuracy: {accuracy:.4f}')
plt.xlabel('Max Prediction Probability')
plt.ylabel('Number of Samples')

# 混淆矩阵(部分类别)
plt.subplot(2, 2, 4)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, ensemble_preds)
# 只显示前10个类别的混淆矩阵
cm_subset = cm[:10, :10]
sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues')
plt.title('Ensemble Model Confusion Matrix (Top 10 Classes)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()




print("\n" + "=" * 60)
print("最终模型性能总结")
print("=" * 60)
print(f"XGBoost MAP@5:     {xgb_map5:.4f}")
print(f"随机森林 MAP@5:    {rf_map5:.4f}")
print(f"LightGBM MAP@5:    {lgb_map5:.4f}")
print(f"集成模型 MAP@5:    {ensemble_map5:.4f}")
print(f"最佳模型: {'集成模型' if ensemble_map5 == max(map5_scores) else models[map5_scores.index(max(map5_scores))]}")


top_3_preds = np.argsort(y_test_proba_ensemble, axis=1)[:, -3:][:, ::-1]
top_3_labels = np.vectorize(lambda idx: label_encoder.classes_[idx])(top_3_preds)
top_3_strs = [' '.join(row) for row in top_3_labels]

submission_ensemble = pd.DataFrame({
    'id': ids,
    'Fertilizer Name': top_3_strs
})

# submission_ensemble.to_csv('/kaggle/working/submission_ensemble.csv', index=False)  # 原始代码，修改文件名
submission_ensemble.to_csv('/kaggle/working/submission_ensemble_final.csv', index=False)

print(f"\n最终提交文件已保存: submission_ensemble_final.csv")
print(f"提交文件形状: {submission_ensemble.shape}")

