import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import time
from tqdm import tqdm
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 设置可视化样式
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


# 步骤1: 解压并加载数据
print("解压文件中...")
data_path = "/kaggle/input/allstate-purchase-prediction-challenge/"
working_path = "/kaggle/working/"

# 解压文件
with zipfile.ZipFile(data_path + "train.csv.zip", 'r') as zip_ref:
    zip_ref.extractall(working_path)
    
with zipfile.ZipFile(data_path + "test_v2.csv.zip", 'r') as zip_ref:
    zip_ref.extractall(working_path)


# 加载数据
print("加载数据中...")
train = pd.read_csv(working_path + "train.csv")
test = pd.read_csv(working_path + "test_v2.csv")
sample_submission = pd.read_csv(data_path + "sampleSubmission.csv")


# 基本信息
print("\n数据集信息:")
print(f"训练集: {train.shape[0]} 行, {train.shape[1]} 列")
print(f"测试集: {test.shape[0]} 行, {test.shape[1]} 列")
print(f"提交样本: {sample_submission.shape[0]} 行, {sample_submission.shape[1]} 列")


# 查看列类型
print("\n列类型分布:")
print(train.dtypes.value_counts())


# 识别保险选项列
option_columns = [col for col in train.columns if col.upper() in "ABCDEFG" and len(col) == 1]
print("\n识别到的保险选项列:", option_columns)


# 识别保险选项列
option_columns = [col for col in train.columns if col.upper() in "ABCDEFG" and len(col) == 1]
print("\n识别到的保险选项列：", option_columns)


# 分析各选项的值分布
for col in option_columns:
    print(f"\n选项 {col} 取值分布:")
    value_counts = train[col].value_counts(normalize=True)
    print(value_counts)


# 确定购买记录
# 检查是否有record_type列
if 'record_type' in train.columns:
    # 使用record_type标记的购买点
    purchases = train[train['record_type'] == 1].copy()
    print(f"\n使用record_type=1找到 {len(purchases)} 个购买记录")
else:
    # 假设每个客户的最后一个报价是购买点
    last_pts = train.groupby('customer_ID')['shopping_pt'].max().reset_index()
    purchases = train.merge(last_pts, on=['customer_ID', 'shopping_pt'])
    print(f"\n使用最后一个购物点找到 {len(purchases)} 个假设的购买记录")


# 创建客户级特征
print("\n创建客户级特征...")
customer_features = []

for customer_id in tqdm(train['customer_ID'].unique()):
    customer_data = train[train['customer_ID'] == customer_id]
    
    # 基本信息
    feature = {'customer_ID': customer_id}
    
    # 购物点数量
    feature['shopping_pt_count'] = len(customer_data)
    
    # 选项变化特征
    for col in option_columns:
        # 变化次数和比率
        changes = (customer_data[col] != customer_data[col].shift(1)).sum()
        feature[f'{col}_change_count'] = changes
        feature[f'{col}_change_rate'] = changes / (len(customer_data) - 1) if len(customer_data) > 1 else 0
        
        # 最常见的选项值
        if len(customer_data) > 0:
            most_common = customer_data[col].mode()[0]
            feature[f'{col}_most_common'] = most_common
    
    # 添加静态特征 (如果存在)
    static_features = ['state', 'location', 'group_size', 'homeowner', 'car_age', 'car_value', 
                     'risk_factor', 'age_oldest', 'age_youngest', 'married_couple', 'C_previous', 'duration_previous']
    
    for col in static_features:
        if col in customer_data.columns and len(customer_data) > 0:
            feature[col] = customer_data[col].iloc[0]  # 假设这些特征在客户内部不变
    
    customer_features.append(feature)

# 转换为DataFrame
customer_feature_df = pd.DataFrame(customer_features)


# 创建方案流行度特征
print("\n创建方案流行度特征...")
plan_counts = train.groupby(option_columns).size()
plan_counts = plan_counts / len(train)
plan_popularity = plan_counts.reset_index(name='plan_popularity')


# 将特征与购买数据合并
print("\n合并特征...")
training_data = purchases.merge(customer_feature_df, on='customer_ID', how='left')


# 处理分类特征
print("\n处理分类特征...")
categorical_cols = [col for col in training_data.columns if 
                  training_data[col].dtype == 'object' or 
                  (training_data[col].dtype == 'int64' and 
                   training_data[col].nunique() < 20 and 
                   col not in ['customer_ID', 'shopping_pt'] + option_columns)]

print(f"识别到的分类特征: {categorical_cols}")


# 分割训练集和验证集
print("\n分割训练集和验证集...")
# 选择特征列，排除ID和目标列
feature_cols = [col for col in training_data.columns if 
               col not in option_columns + ['customer_ID', 'shopping_pt', 'record_type']]

X = training_data[feature_cols]
targets = {col: training_data[col] for col in option_columns}


# 创建训练/验证分割
X_train, X_val = {}, {}
y_train, y_val = {}, {}

for col in option_columns:
    X_train[col], X_val[col], y_train[col], y_val[col] = train_test_split(
        X, targets[col], test_size=0.2, random_state=42)
    print(f"选项 {col} 的训练集大小: {len(X_train[col])}, 验证集大小: {len(X_val[col])}")


# 处理测试数据
print("\n处理测试数据...")
# 提取测试集中每个客户的最后一个购物点
test_last_pts = test.groupby('customer_ID')['shopping_pt'].max().reset_index()
test_final = test.merge(test_last_pts, on=['customer_ID', 'shopping_pt'])


# 合并客户特征
test_final = test_final.merge(customer_feature_df, on='customer_ID', how='left')


# 定义预处理函数处理所有数据集
def preprocess_features(X_train, X_val, X_test, option_columns):
    """
    统一处理训练集、验证集和测试集的特征，确保特征一致性
    
    参数:
    - X_train: 训练特征集字典，键为选项名，值为DataFrame
    - X_val: 验证特征集字典，键为选项名，值为DataFrame
    - X_test: 测试特征集DataFrame
    - option_columns: 保险选项列表
    
    返回:
    - X_train: 处理后的训练特征集
    - X_val: 处理后的验证特征集
    - X_test: 处理后的测试特征集
    """
    print("\n开始预处理所有数据集...")
    
    # 获取所有训练集中的特征列
    all_cols = set()
    for col in option_columns:
        all_cols.update(X_train[col].columns)
    
    # 确保测试集包含所有需要的列
    for col_name in all_cols:
        if col_name not in X_test.columns:
            print(f"测试集缺少特征列: {col_name}，添加空列")
            X_test[col_name] = np.nan
    
    # 处理所有数据集中的非数值特征
    object_cols = []
    for col in option_columns:
        object_cols.extend(X_train[col].select_dtypes(include=['object']).columns.tolist())
    object_cols = list(set(object_cols))  # 去重
    
    # 也检查测试集中的对象类型列
    test_object_cols = X_test.select_dtypes(include=['object']).columns.tolist()
    object_cols = list(set(object_cols + test_object_cols))
    
    print(f"发现 {len(object_cols)} 个需要处理的非数值类型列")
    
    # 处理每个非数值列
    for obj_col in object_cols:
        print(f"处理列: {obj_col}")
        
        # 检查是否是时间格式
        has_time_format = False
        
        # 从训练集或测试集中获取样本检查
        sample_vals = []
        for col in option_columns:
            if obj_col in X_train[col].columns:
                sample_vals = X_train[col][obj_col].dropna().astype(str).head(5).values
                break
        
        if len(sample_vals) == 0 and obj_col in X_test.columns:
            sample_vals = X_test[obj_col].dropna().astype(str).head(5).values
        
        # 检查样本值是否有时间格式
        if len(sample_vals) > 0:
            has_time_format = any(':' in str(val) and 
                                  any(part.isdigit() for part in str(val).split(':')) 
                                  for val in sample_vals)
        
        if has_time_format:
            print(f"  检测到时间格式，创建小时和分钟特征")
            
            # 处理训练集和验证集
            for k in option_columns:
                if obj_col in X_train[k].columns:
                    # 创建小时特征
                    X_train[k][f'{obj_col}_hour'] = X_train[k][obj_col].astype(str).apply(
                        lambda x: int(x.split(':')[0]) if ':' in str(x) and x.split(':')[0].isdigit() else np.nan)
                    # 创建分钟特征
                    X_train[k][f'{obj_col}_minute'] = X_train[k][obj_col].astype(str).apply(
                        lambda x: int(x.split(':')[1]) if ':' in str(x) and 
                                                        len(x.split(':')) > 1 and 
                                                        x.split(':')[1].isdigit() else np.nan)
                    # 删除原始列
                    X_train[k] = X_train[k].drop(columns=[obj_col])
                
                if obj_col in X_val[k].columns:
                    X_val[k][f'{obj_col}_hour'] = X_val[k][obj_col].astype(str).apply(
                        lambda x: int(x.split(':')[0]) if ':' in str(x) and x.split(':')[0].isdigit() else np.nan)
                    X_val[k][f'{obj_col}_minute'] = X_val[k][obj_col].astype(str).apply(
                        lambda x: int(x.split(':')[1]) if ':' in str(x) and 
                                                        len(x.split(':')) > 1 and 
                                                        x.split(':')[1].isdigit() else np.nan)
                    X_val[k] = X_val[k].drop(columns=[obj_col])
            
            # 处理测试集
            if obj_col in X_test.columns:
                X_test[f'{obj_col}_hour'] = X_test[obj_col].astype(str).apply(
                    lambda x: int(x.split(':')[0]) if ':' in str(x) and x.split(':')[0].isdigit() else np.nan)
                X_test[f'{obj_col}_minute'] = X_test[obj_col].astype(str).apply(
                    lambda x: int(x.split(':')[1]) if ':' in str(x) and 
                                                    len(x.split(':')) > 1 and 
                                                    x.split(':')[1].isdigit() else np.nan)
                X_test = X_test.drop(columns=[obj_col])
        else:
            print(f"  使用标签编码处理分类特征")
            
            # 收集所有数据集中的唯一值
            all_values = set()
            
            # 从训练集和验证集收集
            for k in option_columns:
                if obj_col in X_train[k].columns:
                    X_train[k][obj_col] = X_train[k][obj_col].fillna('missing')
                    all_values.update(X_train[k][obj_col].astype(str).unique())
                
                if obj_col in X_val[k].columns:
                    X_val[k][obj_col] = X_val[k][obj_col].fillna('missing')
                    all_values.update(X_val[k][obj_col].astype(str).unique())
            
            # 从测试集收集
            if obj_col in X_test.columns:
                X_test[obj_col] = X_test[obj_col].fillna('missing')
                all_values.update(X_test[obj_col].astype(str).unique())
            
            # 标签编码
            le = LabelEncoder()
            le.fit(list(all_values))
            
            # 应用到训练集和验证集
            for k in option_columns:
                if obj_col in X_train[k].columns:
                    X_train[k][f'{obj_col}_encoded'] = le.transform(X_train[k][obj_col].astype(str))
                    X_train[k] = X_train[k].drop(columns=[obj_col])
                
                if obj_col in X_val[k].columns:
                    X_val[k][f'{obj_col}_encoded'] = le.transform(X_val[k][obj_col].astype(str))
                    X_val[k] = X_val[k].drop(columns=[obj_col])
            
            # 应用到测试集
            if obj_col in X_test.columns:
                X_test[f'{obj_col}_encoded'] = le.transform(X_test[obj_col].astype(str))
                X_test = X_test.drop(columns=[obj_col])
    
    # 确保特征列一致性
    all_train_features = set()
    for k in option_columns:
        all_train_features.update(X_train[k].columns)
    
    # 检查测试集是否有所有训练集特征
    missing_in_test = all_train_features - set(X_test.columns)
    if missing_in_test:
        print(f"测试集缺少 {len(missing_in_test)} 个训练集中使用的特征:")
        for feat in missing_in_test:
            print(f"  添加缺失特征: {feat}")
            X_test[feat] = 0  # 使用0填充缺失特征
    
    # 对于训练集中没有但测试集有的特征，从测试集中删除
    extra_in_test = set(X_test.columns) - all_train_features - set(['customer_ID', 'shopping_pt'])
    if extra_in_test:
        print(f"测试集包含 {len(extra_in_test)} 个训练集中未使用的特征:")
        for feat in extra_in_test:
            print(f"  删除多余特征: {feat}")
            X_test = X_test.drop(columns=[feat])
    
    print("特征处理完成，所有数据集现在具有一致的特征集")
    return X_train, X_val, X_test


# 预处理所有数据集
X_train, X_val, test_final = preprocess_features(X_train, X_val, test_final, option_columns)



# 存储模型和性能
models = {}
model_accuracies = {}

# 对每个选项训练模型
for col in option_columns:
    print(f"\n训练选项 {col} 的模型...")
    start_time = time.time()
    
    # 创建模型
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    
    # 训练模型
    model.fit(X_train[col], y_train[col])
    
    # 验证性能
    y_pred = model.predict(X_val[col])
    accuracy = accuracy_score(y_val[col], y_pred)
    
    print(f"选项 {col} 的验证准确率: {accuracy:.4f}")
    print(f"训练时间: {time.time() - start_time:.2f} 秒")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_train[col].columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"前5个重要特征:")
    print(feature_importance.head(5))
    
    # 存储模型和准确率
    models[col] = model
    model_accuracies[col] = accuracy


# 计算整体方案准确率 (所有选项都必须正确)
all_correct = np.ones(len(X_val[option_columns[0]])).astype(bool)
for col in option_columns:
    y_pred = models[col].predict(X_val[col])
    all_correct = all_correct & (y_pred == y_val[col].values)

overall_accuracy = all_correct.mean()
print(f"\n整体方案预测准确率: {overall_accuracy:.4f}")


# 确保使用正确的特征列
feature_cols = list(X_train[option_columns[0]].columns)
print(f"\n使用 {len(feature_cols)} 个特征进行预测")



# 预测测试集
predictions = {}
for col in option_columns:
    predictions[col] = models[col].predict(test_final[feature_cols])
    print(f"选项 {col} 的预测分布:")
    print(pd.Series(predictions[col]).value_counts(normalize=True))



# 创建提交文件
submission = pd.DataFrame()
submission['customer_ID'] = test_final['customer_ID']

# 合并所有选项的预测为一个方案字符串
submission['plan'] = ''
for col in option_columns:
    submission['plan'] += predictions[col].astype(str)

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("\n提交文件已保存。前几行:")
print(submission.head())

