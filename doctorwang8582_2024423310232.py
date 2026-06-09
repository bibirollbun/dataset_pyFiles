# 导入必要的库
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


print("学号: 2024423310232, 姓名: 冼佳浩")

# 1. 数据加载与预处理
def load_and_preprocess_data():
    """加载数据并进行预处理"""
    
    train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    
    # 查看数据基本信息
    print("训练数据形状:", train_data.shape)
    print("测试数据形状:", test_data.shape)
    print("训练数据前5行:\n", train_data.head())
    
    # 分离特征与标签（注意：假设test.csv中包含id列）
    X = train_data.drop('Fertilizer Name', axis=1)
    y = train_data['Fertilizer Name']
    
    # 提取测试集的id列（用于提交文件）
    test_ids = test_data['id']  # 假设test.csv包含id列
    
    # 处理类别特征：使用LabelEncoder编码
    categorical_cols = X.select_dtypes(include='object').columns
    categorical_mappings = {}  # 保存所有类别列的映射
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        categorical_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    # 处理数值特征：标准化
    numerical_cols = X.select_dtypes(include='number').columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # 对测试集进行相同的预处理
    X_test = test_data.copy()
    for col in categorical_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].map(categorical_mappings.get(col, {}))
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # 划分训练集与验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, y_train, y_val, X_test, test_ids, categorical_cols, numerical_cols

# 2. 模型训练与评估
def train_xgboost_model(X_train, X_val, y_train, y_val, categorical_cols):
    """训练XGBoost多分类模型并优化MAP@5"""
    # 标签编码
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    n_classes = len(le.classes_)
    
    # 准备DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
    dval = xgb.DMatrix(X_val, label=y_val_encoded)
    
    # 设置参数：多分类任务，优化MAP@5
    params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': ['mlogloss'],  # 移除map@5评估，使用手动计算
        'learning_rate': 0.05,
        'n_estimators': 500,
        'max_depth': 5,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'seed': 42,
        'use_label_encoder': False,
    }
    
    # 训练模型
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(params, dtrain, evals=watchlist, verbose_eval=50)
    
    # 在验证集上评估
    y_val_pred_proba = model.predict(dval)
    y_val_pred = np.argsort(y_val_pred_proba, axis=1)[:, -5:][:, ::-1]  # 前5预测
    
    # 计算MAP@5（使用手动实现的函数）
    map5_score = calculate_map_at_k(y_val_encoded, y_val_pred_proba, k=5)
    print(f"验证集MAP@5分数: {map5_score:.4f}")
    
    return model, le

# 手动实现的MAP@k计算函数
def calculate_map_at_k(y_true, y_pred_proba, k=5):
    """
    计算多分类问题的MAP@k分数
    
    参数:
    y_true: 一维数组，真实标签索引（0到n_classes-1）
    y_pred_proba: 二维数组，预测概率矩阵（样本数×类别数）
    k: 计算前k个预测的精度
    
    返回:
    map_score: MAP@k分数
    """
    n_samples = len(y_true)
    map_score = 0.0
    
    for i in range(n_samples):
        true_label = y_true[i]
        pred_proba = y_pred_proba[i]
        
        # 获取前k个预测的类别索引（降序排列）
        top_k_indices = np.argsort(pred_proba)[::-1][:k]
        
        # 计算每个前k预测位置的精度
        precision_at_k = 0.0
        relevant_count = 0
        
        for j, idx in enumerate(top_k_indices):
            if idx == true_label:
                relevant_count += 1
                # 计算当前位置的精度（正确预测数/已预测数）
                precision_at_k += relevant_count / (j + 1)
        
        # 如果真实标签在前k个预测中，累加AP；否则加0
        ap = precision_at_k / min(k, relevant_count) if relevant_count > 0 else 0.0
        map_score += ap
    
    # 计算所有样本的平均AP
    map_score /= n_samples
    return map_score

# 3. 特征重要性分析（排除id列）
def analyze_feature_importance(model, X_train, feature_names):
    """分析特征重要性（排除id列）"""
    importance = model.get_fscore()
    
    # 过滤掉id列（如果存在）
    feature_names = [f for f in feature_names if f != 'id']
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': [importance.get(f, 0) for f in feature_names]
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return importance_df

# 4. 生成预测结果（修正列名为Fertilizer Name）
def generate_predictions(model, X_test, le, test_ids, n_top=5):
    """生成前n个预测结果（修正列名）"""
    dtest = xgb.DMatrix(X_test)
    y_test_pred_proba = model.predict(dtest)
    
    # 获取前n个预测
    y_test_pred_indices = np.argsort(y_test_pred_proba, axis=1)[:, -n_top:][:, ::-1]
    y_test_pred = [le.inverse_transform(indices) for indices in y_test_pred_indices]
    
    # 转换为Kaggle提交格式
    predictions = []
    for pred in y_test_pred:
        predictions.append(' '.join(pred))
    
    # 创建提交文件（列名改为id和Fertilizer Name）
    submission = pd.DataFrame({
        'id': test_ids,  # 使用测试集的真实id
        'Fertilizer Name': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("预测结果已保存至submission.csv")
    
    return submission

# 5. 主函数
def main():
    """主函数：执行完整流程"""
    print("开始数据加载与预处理...")
    X_train, X_val, y_train, y_val, X_test, test_ids, categorical_cols, numerical_cols = load_and_preprocess_data()
    
    print("开始模型训练...")
    model, le = train_xgboost_model(X_train, X_val, y_train, y_val, categorical_cols)
    
    print("进行特征重要性分析...")
    feature_names = X_train.columns.tolist()
    importance_df = analyze_feature_importance(model, X_train, feature_names)
    print("前5个重要特征:\n", importance_df.head(5))
    
    print("生成预测结果...")
    submission = generate_predictions(model, X_test, le, test_ids, n_top=5)
    print("完整流程执行完毕！")

if __name__ == "__main__":
    main()

