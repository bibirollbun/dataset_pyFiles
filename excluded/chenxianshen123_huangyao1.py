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


#2024423310207***黄耀
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# 设置随机种子保证结果可复现
np.random.seed(42)

def load_data(train_path='train.csv', test_path='test.csv'):
    """加载训练集和测试集数据"""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        print(f"训练集形状: {train_data.shape}, 测试集形状: {test_data.shape}")
        return train_data, test_data
    except:
        # 模拟数据加载（实际使用时替换为真实数据路径）
        print("使用模拟数据进行演示...")
        # 生成模拟特征
        n_train, n_test = 1000, 500
        features = ['N', 'P', 'K', 'pH', 'temperature', 'humidity', 'soil_type']
        train_data = pd.DataFrame({
            'N': np.random.randint(50, 200, n_train),
            'P': np.random.randint(30, 150, n_train),
            'K': np.random.randint(40, 180, n_train),
            'pH': np.random.uniform(4.5, 8.5, n_train),
            'temperature': np.random.uniform(15, 35, n_train),
            'humidity': np.random.uniform(30, 90, n_train),
            'soil_type': np.random.choice(['砂质土', '黏质土', '壤土', '腐殖土'], n_train),
            'fertilizer': np.random.choice(
                ['NPK_10-20-10', 'Urea', 'DAP', 'MOP', 'NPK_20-20-20', 
                 'NPK_15-15-15', 'Compost', 'Manure', 'SNP'], n_train)
        })
        test_data = pd.DataFrame({
            'N': np.random.randint(50, 200, n_test),
            'P': np.random.randint(30, 150, n_test),
            'K': np.random.randint(40, 180, n_test),
            'pH': np.random.uniform(4.5, 8.5, n_test),
            'temperature': np.random.uniform(15, 35, n_test),
            'humidity': np.random.uniform(30, 90, n_test),
            'soil_type': np.random.choice(['砂质土', '黏质土', '壤土', '腐殖土'], n_test),
            'id': range(1, n_test+1)
        })
        return train_data, test_data

def preprocess_data(train_data, test_data):
    """数据预处理：处理缺失值、特征编码、标准化"""
    # 1. 处理缺失值（简单示例：用均值/众数填充）
    num_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = train_data.select_dtypes(include=['object']).columns.tolist()
    
    # 数值特征用均值填充
    train_data[num_cols] = train_data[num_cols].fillna(train_data[num_cols].mean())
    test_data[num_cols] = test_data[num_cols].fillna(test_data[num_cols].mean())
    
    # 类别特征用众数填充
    for col in cat_cols:
        if col != 'fertilizer':  # 排除标签列
            train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
            test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
    
    # 2. 特征编码：数值特征标准化，类别特征标签编码
    scaler = StandardScaler()
    train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
    test_data[num_cols] = scaler.transform(test_data[num_cols])
    
    # 标签编码类别特征
    for col in cat_cols:
        if col != 'fertilizer':
            le = LabelEncoder()
            train_data[col] = le.fit_transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
    
    # 3. 分离特征和标签
    X = train_data.drop(['fertilizer'], axis=1)
    y = train_data['fertilizer']
    test_X = test_data.drop(['id'], axis=1) if 'id' in test_data.columns else test_data
    
    print("数据预处理完成")
    return X, y, test_X, test_data

def split_dataset(X, y, test_size=0.2):
    """划分训练集和验证集"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def build_xgboost_model(X_train, y_train, X_val, y_val, num_classes):
    """构建XGBoost多分类模型"""
    # 标签编码肥料类型
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    
    # 转换为DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
    dval = xgb.DMatrix(X_val, label=y_val_encoded)
    
    # 模型参数
    params = {
        'objective': 'multi:softprob',  # 多分类任务
        'num_class': num_classes,       # 类别数量
        'eval_metric': ['mlogloss', 'merror'],  # 替换为多分类适用的评估指标
        'learning_rate': 0.05,
        'max_depth': 5,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'seed': 42,
    }
    
    # 训练模型
    print("开始训练XGBoost模型...")
    model = xgb.train(
        params, dtrain, 
        num_boost_round=300,  # 明确指定迭代次数
        evals=[(dval, 'validation')], 
        early_stopping_rounds=50,
        verbose_eval=50
    )
    print("模型训练完成")
    return model, le

def predict_top5(model, X, le, top_n=5):
    """预测Top5肥料类型"""
    # 生成概率预测
    dtest = xgb.DMatrix(X)
    proba = model.predict(dtest)
    
    # 获取Top5预测结果的索引
    top_indices = np.argsort(proba, axis=1)[:, -top_n:][:, ::-1]
    
    # 对每个样本的预测索引分别进行逆变换
    top_labels = []
    for indices in top_indices:
        top_labels.append(le.inverse_transform(indices))
    
    # 转换为numpy数组
    top_labels = np.array(top_labels)
    
    return top_labels, proba

def calculate_map_at_k(y_true, y_pred_topk, k=5):
    """计算MAP@k指标"""
    map_score = 0.0
    n_samples = len(y_true)
    
    for i in range(n_samples):
        true_label = y_true[i]
        pred_labels = y_pred_topk[i]
        relevant = 0
        sample_map = 0.0
        
        for j in range(min(k, len(pred_labels))):
            if pred_labels[j] == true_label:
                relevant += 1
                sample_map += relevant / (j + 1)
        
        if relevant > 0:
            map_score += sample_map / relevant
    
    if n_samples > 0:
        map_score /= n_samples
    
    return map_score

def plot_feature_importance(model, feature_names, top_n=10):
    """绘制特征重要性图"""
    importance = model.get_score(importance_type='weight')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # 取前top_n个特征
    features = [f[0] for f in importance[:top_n]]
    scores = [f[1] for f in importance[:top_n]]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=features)
    plt.title('特征重要性排名')
    plt.xlabel('重要性得分')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

def generate_submission(test_data, top5_predictions, output_file='submission.csv'):
    """生成提交文件"""
    if 'id' not in test_data.columns:
        test_data['id'] = range(1, len(test_data) + 1)
    
    # 构建提交数据格式
    submission = test_data[['id']].copy()
    for i in range(5):
        submission[f'Prediction_{i+1}'] = top5_predictions[:, i]
    
    # 保存提交文件
    submission.to_csv(output_file, index=False)
    print(f"提交文件已生成: {output_file}")
    return submission

def main():
    """主函数：执行完整的预测流程"""
    # 1. 加载数据
    train_data, test_data = load_data()
    
    # 2. 数据预处理
    X, y, test_X, test_original = preprocess_data(train_data, test_data)
    
    # 3. 划分训练集和验证集
    X_train, X_val, y_train, y_val = split_dataset(X, y)
    
    # 4. 获取类别数量
    num_classes = len(y.unique())
    print(f"肥料类别数量: {num_classes}")
    
    # 5. 构建并训练模型
    model, le = build_xgboost_model(X_train, y_train, X_val, y_val, num_classes)
    
    # 6. 在验证集上预测Top5
    val_top5, val_proba = predict_top5(model, X_val, le)
    
    # 7. 计算MAP@5
    map_score = calculate_map_at_k(y_val.values, val_top5)
    print(f"验证集MAP@5得分: {map_score:.4f}")
    
    # 8. 特征重要性分析
    plot_feature_importance(model, X.columns)
    
    # 9. 在测试集上预测Top5
    test_top5, test_proba = predict_top5(model, test_X, le)
    
    # 10. 生成提交文件
    generate_submission(test_original, test_top5)
    
    print("所有任务完成！")

if __name__ == "__main__":
    main()

