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


# 学号: 2024423310110, 姓名: 邓肖东

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import os
import time

# ================
# 1. 数据加载与预处理
# ================

def load_data():
    """快速加载并验证竞赛数据"""
    try:
        if os.path.exists('/kaggle/input'):
            data_dir = '/kaggle/input/playground-series-s5e6'
            train = pd.read_csv(f'{data_dir}/train.csv')
            test = pd.read_csv(f'{data_dir}/test.csv')
        else:
            train = pd.read_csv('train.csv')
            test = pd.read_csv('test.csv')
            
        print(f"训练集形状: {train.shape}, 测试集形状: {test.shape}")
        print(f"训练集列名: {train.columns.tolist()}")
        return train, test, 'Fertilizer Name'
    
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None, None, None

def preprocess_data(train, test, target_col):
    """优化的数据预处理与特征工程"""
    print("开始快速数据预处理...")
    
    # 分离特征和目标
    X = train.drop([target_col, 'id'], axis=1)
    y = train[target_col]
    test_id = test['id']
    test = test.drop(['id'], axis=1)
    
    # 定义候选特征列表
    candidate_features = [
        'Nitrogen', 'Phosphorous', 'Potassium', 'pH', 
        'Soil Type', 'Crop Type', 'Moisture'
    ]
    
    # 动态检测可用特征
    available_features = [col for col in candidate_features if col in X.columns]
    print(f"可用特征: {available_features}")
    
    # 添加关键特征工程
    X['NPK_ratio'] = X['Nitrogen'] / (X['Phosphorous'] + X['Potassium'] + 1e-6)
    test['NPK_ratio'] = test['Nitrogen'] / (test['Phosphorous'] + test['Potassium'] + 1e-6)
    
    # 确保至少有一个特征
    if not available_features:
        raise ValueError("没有找到任何可用特征，请检查数据列名")
    
    # 仅选择可用特征
    selected_features = available_features + ['NPK_ratio']
    X = X[selected_features]
    test = test[selected_features]
    
    # 快速编码类别特征
    categorical_cols = ['Soil Type', 'Crop Type']
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    
    if categorical_cols:
        encoder = OrdinalEncoder()
        X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
        test[categorical_cols] = encoder.transform(test[categorical_cols])
    
    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"处理后的特征数: {X.shape[1]}")
    return X, y_encoded, test, test_id, label_encoder

# ================
# 2. 极速模型训练
# ================

def train_ultra_fast_xgboost_model(X, y):
    """超快速训练XGBoost模型"""
    print("开始超快速训练XGBoost模型...")
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 极致性能优化参数
    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'n_estimators': 150,         # 进一步减少树的数量
        'learning_rate': 0.15,       # 增加学习率
        'max_depth': 5,              # 减少树的深度
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'n_jobs': -1,                # 启用所有CPU核心
        'random_state': 42,
        'early_stopping_rounds': 15,
        'verbosity': 0,              # 减少输出
        'predictor': 'gpu_predictor' # 启用GPU预测
    }
    
    # 尝试启用GPU训练
    if hasattr(XGBClassifier, 'get_params') and 'gpu_id' in XGBClassifier().get_params():
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
        print("已启用GPU加速")
    
    model = XGBClassifier(**params)
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    end_time = time.time()
    
    print(f"模型训练完成，耗时: {end_time - start_time:.2f}秒")
    return model, X_val, y_val

# ================
# 3. 优化的模型评估
# ================

def calculate_map_at_k_vectorized(y_true, y_pred_proba, k=5):
    """向量化计算MAP@k，提高效率"""
    n_samples = len(y_true)
    pred_ranking = np.argsort(-y_pred_proba, axis=1)[:, :k]  # 降序排列
    
    # 创建掩码矩阵，标记预测正确的位置
    correct_mask = pred_ranking == y_true[:, np.newaxis]
    
    # 计算每个样本的AP@k
    ap_scores = []
    for i in range(n_samples):
        correct = correct_mask[i]
        if not np.any(correct):
            ap_scores.append(0)
            continue
            
        # 计算累积精度
        correct_indices = np.where(correct)[0] + 1  # 转为1-based索引
        ap = np.mean((np.arange(len(correct_indices)) + 1) / correct_indices)
        ap_scores.append(ap)
    
    return np.mean(ap_scores)

def evaluate_model(model, X_val, y_val, model_name="XGBoost"):
    """评估模型性能"""
    print(f"开始评估{model_name}模型...")
    
    start_time = time.time()
    y_pred_proba = model.predict_proba(X_val)
    map5_score = calculate_map_at_k_vectorized(y_val, y_pred_proba, 5)
    map3_score = calculate_map_at_k_vectorized(y_val, y_pred_proba, 3)
    end_time = time.time()
    
    print(f"{model_name}模型评估结果:")
    print(f"MAP@5: {map5_score:.4f}")
    print(f"MAP@3: {map3_score:.4f}")
    print(f"评估耗时: {end_time - start_time:.2f}秒")
    
    return map5_score

def plot_feature_importance(model, X, top_n=5):
    """轻量级特征重要性可视化"""
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100)  # 降低图片质量以节省时间

# ================
# 4. 结果生成与提交
# ================

def generate_submission(model, test, test_id, label_encoder, model_name="ultra_fast"):
    """快速生成提交文件"""
    print(f"生成{model_name}模型的提交文件...")
    
    start_time = time.time()
    test_proba = model.predict_proba(test)
    
    # 向量化获取Top5
    test_top5 = np.argsort(-test_proba, axis=1)[:, :5]
    test_pred = label_encoder.inverse_transform(test_top5.reshape(-1)).reshape(-1, 5)
    
    # 创建提交DataFrame
    submission = pd.DataFrame({
        'ID': test_id,
        'Fertilizer Type 1': test_pred[:, 0],
        'Fertilizer Type 2': test_pred[:, 1],
        'Fertilizer Type 3': test_pred[:, 2],
        'Fertilizer Type 4': test_pred[:, 3],
        'Fertilizer Type 5': test_pred[:, 4]
    })
    
    submission_path = f'submission_{model_name}.csv'
    submission.to_csv(submission_path, index=False, header=True)
    print(f"提交文件已保存至: {submission_path}")
    print(f"生成耗时: {time.time() - start_time:.2f}秒")
    
    return submission

# ================
# 5. 主函数
# ================

def main():
    # 禁用警告
    import warnings
    warnings.filterwarnings('ignore')
    
    # 开始计时
    total_start_time = time.time()
    print("开始执行肥料类型预测程序...")
    
    # 数据加载与预处理
    train, test, target_col = load_data()
    if train is None:
        print("数据加载失败，程序退出")
        return
    
    X, y, test, test_id, label_encoder = preprocess_data(train, test, target_col)
    
    # 训练超快速XGBoost模型
    xgb_model, X_val, y_val = train_ultra_fast_xgboost_model(X, y)
    
    # 评估模型
    xgb_map5 = evaluate_model(xgb_model, X_val, y_val, "XGBoost")
    
    # 特征重要性分析
    plot_feature_importance(xgb_model, X)
    
    # 生成提交文件
    generate_submission(xgb_model, test, test_id, label_encoder)
    
    # 总耗时
    total_time = time.time() - total_start_time
    print(f"程序全部完成，总耗时: {total_time:.2f}秒")

if __name__ == "__main__":
    main()

