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


import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import os

# 1. 数据加载与预处理（合并训练集和测试集进行编码）
def prepare_data(train_file, test_file):
    """
    合并训练集和测试集进行特征编码，确保测试集中的类别都在训练集中出现过
    返回处理后的训练数据、测试数据特征，以及训练数据标签
    """
    # 读取数据
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # 标记数据来源以便后续分离
    train_df['_source'] = 'train'
    test_df['_source'] = 'test'
    
    # 保存测试集ID列（用于最终提交）
    test_id = test_df['id']
    
    # 合并数据（训练集特征 + 测试集特征）
    # 测试集没有'Fertilizer Name'列，使用concat的join='outer'自动处理
    combined_df = pd.concat([
        train_df.drop(['id', 'Fertilizer Name'], axis=1, errors='ignore'),
        test_df.drop(['id'], axis=1, errors='ignore')
    ], ignore_index=True)
    
    # 对类别特征进行编码
    categorical_cols = ['Soil Type', 'Crop Type']
    for col in categorical_cols:
        encoder = LabelEncoder()
        combined_df[col] = encoder.fit_transform(combined_df[col])
    
    # 分离为训练集和测试集特征
    X_training = combined_df[combined_df['_source'] == 'train'].drop('_source', axis=1)
    X_testing = combined_df[combined_df['_source'] == 'test'].drop('_source', axis=1)
    
    # 获取训练集标签
    y_training = train_df['Fertilizer Name']
    
    return X_training, y_training, X_testing, test_id

# 2. 模型训练与优化
def build_model(X_train, y_train):
    """训练LightGBM多分类模型并进行优化"""
    model = LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        n_estimators=600,
        learning_rate=0.04,
        random_state=42,
        num_leaves=31,
        min_child_samples=20
    )
    model.fit(X_train, y_train)
    return model

# 3. 生成测试集Top5预测结果
def predict_top_five(model, X_test, test_id):
    """获取测试集每个样本的Top5预测类别（保留id对应）"""
    # 预测概率
    probs = model.predict_proba(X_test)
    
    # 获取每个样本概率最高的5个类别索引，并按概率从高到低排序
    top_indices = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
    
    # 构建结果DataFrame（按id对应）
    predictions = []
    for i, (id_val, top_idx) in enumerate(zip(test_id, top_indices)):
        top_labels = [model.classes_[idx] for idx in top_idx]
        predictions.append([id_val] + top_labels)
    
    # 转为DataFrame
    result_dataframe = pd.DataFrame(
        predictions, 
        columns=['id', 'Fertilizer_Top1', 'Fertilizer_Top2', 
                 'Fertilizer_Top3', 'Fertilizer_Top4', 'Fertilizer_Top5']
    )
    return result_dataframe

# 4. 保存预测结果
def save_results(result_df, output_path):
    """保存预测结果为CSV格式"""
    result_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Kaggle竞赛数据路径
    train_data_path = "/kaggle/input/playground-series-s5e6/train.csv"
    test_data_path = "/kaggle/input/playground-series-s5e6/test.csv"
    
    # 加载并预处理数据
    X_train, y_train, X_test, test_ids = prepare_data(train_data_path, test_data_path)
    
    # 训练模型
    print("开始训练模型...")
    model = build_model(X_train, y_train)
    
    # 生成测试集Top5预测结果
    print("生成预测结果...")
    top5_df = predict_top_five(model, X_test, test_ids)
    
    # 保存提交文件
    output_directory = "/kaggle/working/submission"
    os.makedirs(output_directory, exist_ok=True)
    submission_file = os.path.join(output_directory, "submission.csv")
    save_results(top5_df, submission_file)
    
    print(f"预测结果已保存至: {submission_file}")
    print("前5条预测示例:")
    print(top5_df.head())

