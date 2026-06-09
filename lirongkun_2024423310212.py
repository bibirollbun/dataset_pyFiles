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


# ======================
# 肥料类型预测竞赛 - 完整解决方案
# ======================

# 1. 导入必要的库
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# 2. 读取数据
print("正在读取数据...")
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 查看数据前几行
print("\n训练数据前几行：")
print(train_data.head())
print("\n测试数据前几行：")
print(test_data.head())



# 3. 数据预处理
print("\n正在进行数据预处理...")


# 3.1 检查缺失值
print("\n训练数据缺失值统计：")
print(train_data.isnull().sum())
print("\n测试数据缺失值统计：")
print(test_data.isnull().sum())


# 3.2 特征工程：独热编码类别特征
categorical_features = ['Soil Type', 'Crop Type']

# 初始化OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# 对训练数据进行独热编码
encoded_train_data = encoder.fit_transform(train_data[categorical_features])
encoded_train_feature_names = encoder.get_feature_names_out(categorical_features)

# 将独热编码后的特征与数值特征合并
numeric_features_train = train_data.drop(columns=categorical_features + ['Fertilizer Name', 'id'])
encoded_train_data_df = pd.DataFrame(encoded_train_data, columns=encoded_train_feature_names)
train_data_processed = pd.concat([numeric_features_train, encoded_train_data_df], axis=1)

# 对测试数据进行独热编码
encoded_test_data = encoder.transform(test_data[categorical_features])
encoded_test_data_df = pd.DataFrame(encoded_test_data, columns=encoded_train_feature_names)
numeric_features_test = test_data.drop(columns=categorical_features + ['id'])
test_data_processed = pd.concat([numeric_features_test, encoded_test_data_df], axis=1)

print("\n处理后的训练数据形状:", train_data_processed.shape)
print("处理后的测试数据形状:", test_data_processed.shape)


# 4. 准备训练和验证集
print("\n正在准备训练和验证集...")
X = train_data_processed
y = train_data['Fertilizer Name']

# 划分训练集和验证集（80%训练，20%验证）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集大小: {X_train.shape[0]}")
print(f"验证集大小: {X_val.shape[0]}")


# 5. 训练多分类模型
print("\n正在训练模型...")

# 获取肥料类别并创建映射字典
fertilizer_names = train_data['Fertilizer Name'].unique()
fertilizer_to_num = {name: i for i, name in enumerate(fertilizer_names)}
num_to_fertilizer = {i: name for i, name in enumerate(fertilizer_names)}
num_classes = len(fertilizer_names)
print(f"肥料类别数量: {num_classes}")

# 将字符串标签转换为数值标签
y_train_num = y_train.map(fertilizer_to_num)
y_val_num = y_val.map(fertilizer_to_num)



# 5.1 LightGBM模型（直接使用字符串标签）
print("\n训练LightGBM模型...")
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=num_classes,
    metric='multi_logloss',
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=100,
    random_state=42,
    verbose=-1
)

lgb_model.fit(
    X_train,
    y_train,  # 直接使用字符串标签
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)


# 5.2 XGBoost模型（使用数值标签）
print("\n训练XGBoost模型...")
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=num_classes,
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=100,  # 在构造函数中设置
    random_state=42,
    eval_metric='mlogloss',
    verbose=False
)

xgb_model.fit(
    X_train,
    y_train_num,  # 使用数值标签
    eval_set=[(X_val, y_val_num)],
    verbose=False
)


# 5.3 CatBoost模型（直接使用字符串标签）
print("\n训练CatBoost模型...")
catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    early_stopping_rounds=100,
    random_seed=42,
    verbose=0
)

catboost_model.fit(
    X_train,
    y_train,  # 直接使用字符串标签
    eval_set=(X_val, y_val),
    early_stopping_rounds=100
)


# 6. 模型评估
print("\n正在评估模型性能...")

def calculate_map5(y_true, y_pred_indices):
    """
    计算Mean Average Precision @ 5 (MAP@5)
    y_true: 真实标签的数组（索引形式）
    y_pred_indices: 预测标签的数组（每个样本的前5个预测索引）
    """
    U = len(y_true)
    total_score = 0.0
    for i in range(U):
        # 真实标签
        true_label = y_true[i]
        # 预测标签
        pred_labels = y_pred_indices[i]
        # 初始化
        score = 0.0
        num_hits = 0
        for k in range(5):
            if pred_labels[k] == true_label:
                num_hits += 1
                score += num_hits / (k + 1)
        total_score += score
    return total_score / U

def get_top5_predictions(model, X, is_xgb=False):
    """
    获取每个样本的前5个预测索引
    model: 训练好的模型
    X: 输入数据
    is_xgb: 是否为XGBoost模型
    """
    if is_xgb:
        # XGBoost需要预测概率并排序
        probs = model.predict_proba(X)
        top5_indices = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
    else:
        # LightGBM和CatBoost可以直接预测概率并排序
        probs = model.predict_proba(X)
        top5_indices = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
    return top5_indices

# 获取验证集的预测结果（Top5索引）
lgb_top5_preds_val_indices = get_top5_predictions(lgb_model, X_val, is_xgb=False)
xgb_top5_preds_val_indices = get_top5_predictions(xgb_model, X_val, is_xgb=True)
catboost_top5_preds_val_indices = get_top5_predictions(catboost_model, X_val, is_xgb=False)

# 将验证集的真实标签转换为索引
y_true_indices = np.array([fertilizer_to_num[name] for name in y_val])

# 计算MAP@5
lgb_map5_score = calculate_map5(y_true_indices, lgb_top5_preds_val_indices)
xgb_map5_score = calculate_map5(y_true_indices, xgb_top5_preds_val_indices)
catboost_map5_score = calculate_map5(y_true_indices, catboost_top5_preds_val_indices)

print(f"LightGBM 验证集 MAP@5 得分: {lgb_map5_score:.4f}")
print(f"XGBoost 验证集 MAP@5 得分: {xgb_map5_score:.4f}")
print(f"CatBoost 验证集 MAP@5 得分: {catboost_map5_score:.4f}")


# 7. 在测试集上进行预测
print("\n正在测试集上进行预测...")

def get_top5_predictions_names(model, X, is_xgb=False):
    """
    获取每个样本的前5个预测肥料名称
    model: 训练好的模型
    X: 输入数据
    is_xgb: 是否为XGBoost模型
    """
    if is_xgb:
        # XGBoost需要预测概率并排序
        probs = model.predict_proba(X)
        top5_indices = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
        top5_names = np.array([[num_to_fertilizer[p] for p in preds] for preds in top5_indices])
    else:
        # LightGBM和CatBoost可以直接预测概率并排序
        probs = model.predict_proba(X)
        top5_indices = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
        top5_names = np.array([[fertilizer_names[p] for p in preds] for preds in top5_indices])
    return top5_names

# LightGBM测试集预测
lgb_top5_preds_test_names = get_top5_predictions_names(lgb_model, test_data_processed, is_xgb=False)

# XGBoost测试集预测
xgb_top5_preds_test_names = get_top5_predictions_names(xgb_model, test_data_processed, is_xgb=True)

# CatBoost测试集预测
catboost_top5_preds_test_names = get_top5_predictions_names(catboost_model, test_data_processed, is_xgb=False)

# 查看前5个样本的预测结果
print("测试集前5个样本的预测结果（LightGBM）：")
for i in range(5):
    print(f"样本 {i+1}: {lgb_top5_preds_test_names[i]}")

print("\n测试集前5个样本的预测结果（XGBoost）：")
for i in range(5):
    print(f"样本 {i+1}: {xgb_top5_preds_test_names[i]}")

print("\n测试集前5个样本的预测结果（CatBoost）：")
for i in range(5):
    print(f"样本 {i+1}: {catboost_top5_preds_test_names[i]}")


# 确保预测结果是NumPy数组
lgb_top5_preds_test_names = np.array(lgb_top5_preds_test_names)
xgb_top5_preds_test_names = np.array(xgb_top5_preds_test_names)
catboost_top5_preds_test_names = np.array(catboost_top5_preds_test_names)

#  检查预测结果是否正确
def check_predictions(preds_names, fertilizer_names_set):
    """
    检查预测结果是否都在肥料名称集合中，并且每个样本有5个预测
    preds_names: 预测结果数组
    fertilizer_names_set: 肥料名称的集合
    """
    for i, preds in enumerate(preds_names):
        if len(preds) != 5:
            print(f"样本 {i+1} 预测数量不正确: {len(preds)}")
        for pred in preds:
            if pred not in fertilizer_names_set:
                print(f"样本 {i+1} 包含未知肥料名称: {pred}")
    print("预测结果检查完成。")

fertilizer_names_set = set(fertilizer_names)

print("检查LightGBM预测结果：")
check_predictions(lgb_top5_preds_test_names, fertilizer_names_set)

print("检查XGBoost预测结果：")
check_predictions(xgb_top5_preds_test_names, fertilizer_names_set)

print("检查CatBoost预测结果：")
check_predictions(catboost_top5_preds_test_names, fertilizer_names_set)


# 8. 集成模型预测及MAP@5得分计算
print("\n正在生成集成模型预测...")

# 确保所有预测结果都是NumPy数组，并且形状正确
# 检查形状
print("LightGBM预测结果形状:", lgb_top5_preds_test_names.shape)
print("XGBoost预测结果形状:", xgb_top5_preds_test_names.shape)
print("CatBoost预测结果形状:", catboost_top5_preds_test_names.shape)

# 如果预测结果是列表的数组，需要转换为NumPy数组
if isinstance(lgb_top5_preds_test_names, list):
    lgb_top5_preds_test_names = np.array(lgb_top5_preds_test_names)
if isinstance(xgb_top5_preds_test_names, list):
    xgb_top5_preds_test_names = np.array(xgb_top5_preds_test_names)
if isinstance(catboost_top5_preds_test_names, list):
    catboost_top5_preds_test_names = np.array(catboost_top5_preds_test_names)

# 确保每个预测结果有5个元素
assert lgb_top5_preds_test_names.shape[1] == 5, "LightGBM预测结果每行应包含5个元素"
assert xgb_top5_preds_test_names.shape[1] == 5, "XGBoost预测结果每行应包含5个元素"
assert catboost_top5_preds_test_names.shape[1] == 5, "CatBoost预测结果每行应包含5个元素"

# 初始化集成模型的预测结果
ensemble_top5_preds_test_names = []

# 遍历每个样本
for i in range(len(test_data)):
    # 收集三个模型的预测结果
    lgb_preds = set(lgb_top5_preds_test_names[i])
    xgb_preds = set(xgb_top5_preds_test_names[i])
    catboost_preds = set(catboost_top5_preds_test_names[i])
    
    # 合并所有预测结果
    combined_preds = lgb_preds.union(xgb_preds).union(catboost_preds)
    
    # 如果合并后的预测结果超过5个，选择出现次数最多的前5个
    # 这里采用简单的投票机制，统计每个肥料名称在三个模型中的出现次数
    from collections import Counter
    all_preds = list(lgb_preds) + list(xgb_preds) + list(catboost_preds)
    pred_counts = Counter(all_preds)
    top5_preds = [pred for pred, _ in pred_counts.most_common(5)]
    
    # 如果仍然超过5个（理论上不会），取前5个
    ensemble_top5_preds_test_names.append(top5_preds[:5])

# 转换为NumPy数组以便后续处理
ensemble_top5_preds_test_names = np.array(ensemble_top5_preds_test_names)

# 查看前5个样本的集成预测结果
print("测试集前5个样本的集成预测结果：")
for i in range(5):
    print(f"样本 {i+1}: {ensemble_top5_preds_test_names[i]}")

# 8.1 计算集成模型的MAP@5得分
# 由于我们没有真实的测试集标签，无法直接计算MAP@5得分。
# 但是，我们可以通过验证集的预测结果来评估集成模型的性能。

# 获取验证集的集成预测结果
def get_ensemble_top5_indices(models, X, is_xgb_list, fertilizer_to_num):
    """
    获取集成模型的前5个预测索引
    models: 模型列表 [lgb_model, xgb_model, catboost_model]
    X: 输入数据
    is_xgb_list: 是否为XGBoost模型的列表 [False, True, False]
    fertilizer_to_num: 肥料名称到数值的映射字典
    """
    # 获取每个模型的预测索引
    top5_indices_list = []
    for model, is_xgb in zip(models, is_xgb_list):
        if is_xgb:
            probs = model.predict_proba(X)
            top5_indices = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
        else:
            probs = model.predict_proba(X)
            top5_indices = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
        top5_indices_list.append(top5_indices)
    
    # 初始化集成预测索引
    ensemble_top5_indices = np.zeros((X.shape[0], 5), dtype=int)
    
    # 对每个样本，统计三个模型预测的索引，选择出现次数最多的前5个
    for sample_idx in range(X.shape[0]):
        # 收集三个模型的预测索引
        combined_indices = []
        for top5_indices in top5_indices_list:
            combined_indices.extend(top5_indices[sample_idx])
        
        # 统计每个索引出现的次数
        index_counts = Counter(combined_indices)
        # 选择出现次数最多的前5个索引
        top5_index_counts = index_counts.most_common(5)
        top5_indices_selected = [idx for idx, _ in top5_index_counts]
        # 如果不足5个，可能需要填充（但在此假设每个模型预测5个，合并后至少有5个）
        ensemble_top5_indices[sample_idx] = top5_indices_selected[:5]
    
    return ensemble_top5_indices

# 获取验证集的集成预测索引
is_xgb_list = [False, True, False]  # [lgb, xgb, catboost]
ensemble_top5_preds_val_indices = get_ensemble_top5_indices([lgb_model, xgb_model, catboost_model], X_val, is_xgb_list, fertilizer_to_num)

# 计算集成模型的MAP@5得分
ensemble_map5_score = calculate_map5(y_true_indices, ensemble_top5_preds_val_indices)
print(f"集成模型 验证集 MAP@5 得分: {ensemble_map5_score:.4f}")


# 9. 生成提交文件
print("\n正在生成提交文件...")



# 生成单列文件
#  LightGBM提交文件
submission_lgb = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer Name': [' '.join(preds) for preds in lgb_top5_preds_test_names]
})
submission_lgb.to_csv('submission_lgb.csv', index=False)



#  XGBoost提交文件
submission_xgb = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer Name': [' '.join(preds) for preds in xgb_top5_preds_test_names]
})
submission_xgb.to_csv('submission_xgb.csv', index=False)


#  CatBoost提交文件
submission_catboost = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer Name': [' '.join(preds) for preds in catboost_top5_preds_test_names]
})
submission_catboost.to_csv('submission_catboost.csv', index=False)


# 9.1 如果竞赛要求每行包含5个独立的列
# LightGBM
submission_lgb_split = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer1': lgb_top5_preds_test_names[:, 0],
    'Fertilizer2': lgb_top5_preds_test_names[:, 1],
    'Fertilizer3': lgb_top5_preds_test_names[:, 2],
    'Fertilizer4': lgb_top5_preds_test_names[:, 3],
    'Fertilizer5': lgb_top5_preds_test_names[:, 4]
})
submission_lgb_split.to_csv('submission_lgb_split.csv', index=False) 


# XGBoost
submission_xgb_split = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer1': xgb_top5_preds_test_names[:, 0],
    'Fertilizer2': xgb_top5_preds_test_names[:, 1],
    'Fertilizer3': xgb_top5_preds_test_names[:, 2],
    'Fertilizer4': xgb_top5_preds_test_names[:, 3],
    'Fertilizer5': xgb_top5_preds_test_names[:, 4]
})
submission_xgb_split.to_csv('submission_xgb_split.csv', index=False)


# CatBoost
submission_catboost_split = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer1': catboost_top5_preds_test_names[:, 0],
    'Fertilizer2': catboost_top5_preds_test_names[:, 1],
    'Fertilizer3': catboost_top5_preds_test_names[:, 2],
    'Fertilizer4': catboost_top5_preds_test_names[:, 3],
    'Fertilizer5': catboost_top5_preds_test_names[:, 4]
})
submission_catboost_split.to_csv('submission_catboost_split.csv', index=False)



#  集成模型提交文件（单列）
submission_ensemble = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer Name': [' '.join(preds) for preds in ensemble_top5_preds_test_names]
})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)


# 集成模型提交文件（拆分列）
submission_ensemble_split = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer1': ensemble_top5_preds_test_names[:, 0],
    'Fertilizer2': ensemble_top5_preds_test_names[:, 1],
    'Fertilizer3': ensemble_top5_preds_test_names[:, 2],
    'Fertilizer4': ensemble_top5_preds_test_names[:, 3],
    'Fertilizer5': ensemble_top5_preds_test_names[:, 4]
})
submission_ensemble_split.to_csv('submission_ensemble_split.csv', index=False)


print("\n提交文件已生成：")
print("1. submission_lgb.csv (LightGBM)")
print("2. submission_xgb.csv (XGBoost)")
print("3. submission_catboost.csv (CatBoost)")
print("4. submission_lgb_split.csv (LightGBM拆分列)")
print("5. submission_xgb_split.csv (XGBoost拆分列)")
print("6. submission_catboost_split.csv (CatBoost拆分列)")
print("7. submission_ensemble.csv (集成模型，单列)")
print("8. submission_ensemble_split.csv (集成模型，拆分列)")

