# 学号: 2024423320109, 姓名: 敬李强
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import os

# 1. 数据加载与预处理（合并训练集和测试集进行编码）
def load_and_preprocess_data(train_path, test_path):
    """
    合并训练集和测试集进行编码，避免测试集中出现未知类别
    返回处理后的训练集、测试集特征，以及训练集标签
    """
    # 加载数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # 标记数据来源以便后续拆分
    train_data['_data_source'] = 'train'
    test_data['_data_source'] = 'test'
    
    # 保存测试集ID列（用于最终提交）
    test_ids = test_data['id']
    
    # 合并数据（训练集特征 + 测试集特征）
    # 测试集没有'Fertilizer Name'列，使用concat的join='outer'自动处理
    combined_data = pd.concat([
        train_data.drop(['id', 'Fertilizer Name'], axis=1, errors='ignore'),
        test_data.drop(['id'], axis=1, errors='ignore')
    ], ignore_index=True)
    
    # 对类别特征进行编码
    cat_cols = ['Soil Type', 'Crop Type']
    for col in cat_cols:
        le = LabelEncoder()
        combined_data[col] = le.fit_transform(combined_data[col])
    
    # 拆分为训练集和测试集特征
    X_train = combined_data[combined_data['_data_source'] == 'train'].drop('_data_source', axis=1)
    X_test = combined_data[combined_data['_data_source'] == 'test'].drop('_data_source', axis=1)
    
    # 获取训练集标签
    y_train = train_data['Fertilizer Name']
    
    return X_train, y_train, X_test, test_ids

# 2. 模型训练
def train_model(X_train, y_train):
    """训练LightGBM多分类模型"""
    model = LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# 3. 生成测试集Top5预测结果
def generate_test_top5(model, X_test, test_ids):
    """获取测试集每个样本的Top5预测类别（保留id对应）"""
    # 预测概率
    probabilities = model.predict_proba(X_test)
    
    # 获取每个样本概率最高的5个类别索引，并按概率从高到低排序
    top5_indices = np.argsort(probabilities, axis=1)[:, -5:][:, ::-1]
    
    # 构建结果DataFrame（按id对应）
    results = []
    for i, (test_id, top5_idx) in enumerate(zip(test_ids, top5_indices)):
        top5_labels = [model.classes_[idx] for idx in top5_idx]
        results.append([test_id] + top5_labels)
    
    # 转为DataFrame
    result_df = pd.DataFrame(
        results, 
        columns=['id', 'Fertilizer_Top1', 'Fertilizer_Top2', 
                 'Fertilizer_Top3', 'Fertilizer_Top4', 'Fertilizer_Top5']
    )
    return result_df

# 4. 保存预测结果
def save_submission(result_df, save_path):
    """保存预测结果为Kaggle提交格式"""
    result_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    # Kaggle竞赛数据路径
    train_path = "/kaggle/input/playground-series-s5e6/train.csv"
    test_path = "/kaggle/input/playground-series-s5e6/test.csv"
    
    # 加载并预处理数据
    X_train, y_train, X_test, test_ids = load_and_preprocess_data(train_path, test_path)
    
    # 训练模型
    print("开始训练模型...")
    trained_model = train_model(X_train, y_train)
    
    # 生成测试集Top5预测结果
    print("生成预测结果...")
    top5_result_df = generate_test_top5(trained_model, X_test, test_ids)
    
    # 保存提交文件
    output_dir = "/kaggle/working/submission"
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")
    save_submission(top5_result_df, submission_path)
    
    print(f"预测结果已保存至: {submission_path}")
    print("前5条预测示例:")
    print(top5_result_df.head())

