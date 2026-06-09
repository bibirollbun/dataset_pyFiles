#学号：2024423310225 姓名：叶铎盛
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


# Kaggle Fertilizer Prediction Solution - Fixed for LightGBM Compatibility
# Designed for Playground Series S5E6

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import joblib
import os
import lightgbm as lgb  # 导入lightgbm原生接口

# 自定义MAP@5计算函数
def map_at_5(y_true, y_pred_probs):
    """
    计算Mean Average Precision @ 5
    """
    # 将真实标签转换为二进制矩阵
    n_classes = y_pred_probs.shape[1]
    y_true_binary = np.zeros_like(y_pred_probs)
    for i, label in enumerate(y_true):
        y_true_binary[i, label] = 1
    
    # 计算MAP@5
    return label_ranking_average_precision_score(y_true_binary, y_pred_probs)

# Kaggle环境设置
KAGGLE_INPUT_PATH = '/kaggle/input/playground-series-s5e6'
KAGGLE_WORKING_PATH = '/kaggle/working'

# 确保工作目录存在
os.makedirs(KAGGLE_WORKING_PATH, exist_ok=True)

# 1. 加载数据
print("Loading data...")
train_df = pd.read_csv(f'{KAGGLE_INPUT_PATH}/train.csv')
test_df = pd.read_csv(f'{KAGGLE_INPUT_PATH}/test.csv')
sample_submission = pd.read_csv(f'{KAGGLE_INPUT_PATH}/sample_submission.csv')

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# 2. 数据预处理
print("Preprocessing data...")

# 合并数据集以便统一处理
full_df = pd.concat([train_df, test_df], axis=0)

# 处理缺失值 - 用中位数填充数值列
numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in numeric_cols:
    full_df[col].fillna(full_df[col].median(), inplace=True)

# 编码分类特征
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

full_df['Soil Type'] = le_soil.fit_transform(full_df['Soil Type'])
full_df['Crop Type'] = le_crop.fit_transform(full_df['Crop Type'])

# 仅对训练集的肥料名称进行编码
train_df_processed = full_df[full_df['id'].isin(train_df['id'])].copy()
train_df_processed['Fertilizer Name'] = le_fertilizer.fit_transform(train_df_processed['Fertilizer Name'])

# 3. 特征工程
print("Feature engineering...")
# 添加新特征
full_df['Nutrient_Ratio_NP'] = full_df['Nitrogen'] / (full_df['Phosphorous'] + 1e-6)
full_df['Nutrient_Ratio_NK'] = full_df['Nitrogen'] / (full_df['Potassium'] + 1e-6)
full_df['Nutrient_Ratio_PK'] = full_df['Phosphorous'] / (full_df['Potassium'] + 1e-6)
full_df['Total_Nutrients'] = full_df['Nitrogen'] + full_df['Phosphorous'] + full_df['Potassium']

# 4. 准备训练和测试数据
print("Preparing datasets...")

# 分离训练集和测试集
X_train = full_df[full_df['id'].isin(train_df['id'])].drop(['id', 'Fertilizer Name'], axis=1)
X_test = full_df[full_df['id'].isin(test_df['id'])].drop(['id', 'Fertilizer Name'], axis=1)
y_train = train_df_processed['Fertilizer Name']

# 划分验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 5. 使用原生LightGBM接口训练模型（解决兼容性问题）
print("Training model using native LightGBM interface...")

# 创建数据集
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# 设置参数
params = {
    'objective': 'multiclass',
    'num_class': len(le_fertilizer.classes_),
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'metric': 'multi_logloss',
    'verbosity': -1  # 减少日志输出
}

# 训练模型
model = lgb.train(
    params,
    train_data,
    num_boost_round=1500,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

# 6. 模型评估
print("Evaluating model...")
# 在验证集上预测概率
y_val_probs = model.predict(X_val, num_iteration=model.best_iteration)

# 计算MAP@5
map_score = map_at_5(y_val, y_val_probs)
print(f"Validation MAP@5: {map_score:.6f}")

# 7. 生成测试集预测
print("Generating predictions...")
# 获取测试集的概率预测
test_probs = model.predict(X_test, num_iteration=model.best_iteration)

# 获取Top5预测
top5_indices = np.argsort(-test_probs, axis=1)[:, :5]
top5_fertilizers = le_fertilizer.inverse_transform(top5_indices.flatten()).reshape(top5_indices.shape)

# 8. 创建提交文件
print("Creating submission file...")
# 生成Top3预测字符串
top3_preds = []
for preds in top5_fertilizers:
    top3_preds.append(' '.join(preds[:3]))

# 创建提交DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': top3_preds
})

# 保存提交文件
submission_path = f'{KAGGLE_WORKING_PATH}/submission.csv'
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")

# 9. 特征重要性分析
print("\nFeature Importances:")
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)

print(feature_importances.head(10))

# 10. 保存模型和编码器（可选）
print("Saving artifacts...")
model.save_model(f'{KAGGLE_WORKING_PATH}/fertilizer_model.txt')
joblib.dump(le_fertilizer, f'{KAGGLE_WORKING_PATH}/fertilizer_encoder.pkl')
joblib.dump(le_soil, f'{KAGGLE_WORKING_PATH}/soil_encoder.pkl')
joblib.dump(le_crop, f'{KAGGLE_WORKING_PATH}/crop_encoder.pkl')

print("Process completed successfully!")


# Kaggle Fertilizer Prediction Solution - Fixed Import Issue
# Designed for Playground Series S5E6

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import label_ranking_average_precision_score  # 修复缺失的导入
import lightgbm as lgb
import joblib
import os

# Kaggle环境设置
KAGGLE_INPUT_PATH = '/kaggle/input/playground-series-s5e6'
KAGGLE_WORKING_PATH = '/kaggle/working'

# 确保工作目录存在
os.makedirs(KAGGLE_WORKING_PATH, exist_ok=True)

# 1. 加载数据
print("Loading data...")
train_df = pd.read_csv(f'{KAGGLE_INPUT_PATH}/train.csv')
test_df = pd.read_csv(f'{KAGGLE_INPUT_PATH}/test.csv')
sample_submission = pd.read_csv(f'{KAGGLE_INPUT_PATH}/sample_submission.csv')

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# 2. 数据预处理
print("Preprocessing data...")

# 合并数据集以便统一处理
full_df = pd.concat([train_df, test_df], axis=0)

# 处理缺失值 - 用中位数填充数值列
numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in numeric_cols:
    full_df[col].fillna(full_df[col].median(), inplace=True)

# 编码分类特征
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

full_df['Soil Type'] = le_soil.fit_transform(full_df['Soil Type'])
full_df['Crop Type'] = le_crop.fit_transform(full_df['Crop Type'])

# 仅对训练集的肥料名称进行编码
train_df_processed = full_df[full_df['id'].isin(train_df['id'])].copy()
train_df_processed['Fertilizer Name'] = le_fertilizer.fit_transform(train_df_processed['Fertilizer Name'])

# 3. 特征工程
print("Feature engineering...")
# 添加新特征
full_df['Nutrient_Ratio_NP'] = full_df['Nitrogen'] / (full_df['Phosphorous'] + 1e-6)
full_df['Nutrient_Ratio_NK'] = full_df['Nitrogen'] / (full_df['Potassium'] + 1e-6)
full_df['Nutrient_Ratio_PK'] = full_df['Phosphorous'] / (full_df['Potassium'] + 1e-6)
full_df['Total_Nutrients'] = full_df['Nitrogen'] + full_df['Phosphorous'] + full_df['Potassium']

# 4. 准备训练和测试数据
print("Preparing datasets...")

# 分离训练集和测试集
X_train = full_df[full_df['id'].isin(train_df['id'])].drop(['id', 'Fertilizer Name'], axis=1)
X_test = full_df[full_df['id'].isin(test_df['id'])].drop(['id', 'Fertilizer Name'], axis=1)
y_train = train_df_processed['Fertilizer Name']

# 划分验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 5. 使用原生LightGBM接口训练模型
print("Training model using native LightGBM interface...")

# 创建数据集
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# 设置参数
params = {
    'objective': 'multiclass',
    'num_class': len(le_fertilizer.classes_),
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'metric': 'multi_logloss',
    'verbosity': -1  # 减少日志输出
}

# 训练模型
model = lgb.train(
    params,
    train_data,
    num_boost_round=1500,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

# 6. 模型评估
print("Evaluating model...")
# 在验证集上预测概率
y_val_probs = model.predict(X_val, num_iteration=model.best_iteration)

# 自定义MAP@5计算函数
def map_at_5(y_true, y_pred_probs):
    """
    计算Mean Average Precision @ 5
    """
    # 将真实标签转换为二进制矩阵
    n_classes = y_pred_probs.shape[1]
    y_true_binary = np.zeros_like(y_pred_probs)
    for i, label in enumerate(y_true):
        y_true_binary[i, label] = 1
    
    # 计算MAP@5
    return label_ranking_average_precision_score(y_true_binary, y_pred_probs)

# 计算MAP@5
map_score = map_at_5(y_val, y_val_probs)
print(f"Validation MAP@5: {map_score:.6f}")

# 7. 生成测试集预测
print("Generating predictions...")
# 获取测试集的概率预测
test_probs = model.predict(X_test, num_iteration=model.best_iteration)

# 获取Top5预测
top5_indices = np.argsort(-test_probs, axis=1)[:, :5]
top5_fertilizers = le_fertilizer.inverse_transform(top5_indices.flatten()).reshape(top5_indices.shape)

# 8. 创建提交文件
print("Creating submission file...")
# 生成Top3预测字符串
top3_preds = []
for preds in top5_fertilizers:
    top3_preds.append(' '.join(preds[:3]))

# 创建提交DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': top3_preds
})

# 保存提交文件
submission_path = f'{KAGGLE_WORKING_PATH}/submission.csv'
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")

# 9. 特征重要性分析
print("\nFeature Importances:")
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)

print(feature_importances.head(10))

# 10. 保存模型和编码器（可选）
print("Saving artifacts...")
model.save_model(f'{KAGGLE_WORKING_PATH}/fertilizer_model.txt')
joblib.dump(le_fertilizer, f'{KAGGLE_WORKING_PATH}/fertilizer_encoder.pkl')
joblib.dump(le_soil, f'{KAGGLE_WORKING_PATH}/soil_encoder.pkl')
joblib.dump(le_crop, f'{KAGGLE_WORKING_PATH}/crop_encoder.pkl')

print("Process completed successfully!")


