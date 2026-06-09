#学号：2024423310226 姓名：曾垂凯
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import early_stopping

# 设置随机种子确保结果可复现
np.random.seed(42)


# 模拟生成土壤特征数据和肥料类型数据
def generate_sample_data(n_samples=1000):
    """生成模拟数据用于模型训练和测试"""
    # 土壤特征数据
    data = {
        'nitrogen': np.random.normal(50, 15, n_samples),
        'phosphorus': np.random.normal(40, 12, n_samples),
        'potassium': np.random.normal(45, 10, n_samples),
        'ph_value': np.random.normal(6.5, 1.0, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'humidity': np.random.normal(70, 15, n_samples),
        'organic_matter': np.random.normal(3.5, 1.2, n_samples),
        'water_retention': np.random.normal(55, 10, n_samples),
    }

    df = pd.DataFrame(data)

    # 根据特征值创建目标变量（肥料类型）
    def assign_fertilizer(row):
        if row['nitrogen'] < 40 and row['phosphorus'] < 35:
            return 'NPK_20-20-20'
        elif row['nitrogen'] > 60 and row['potassium'] < 40:
            return 'Urea'
        elif row['phosphorus'] > 50 and row['ph_value'] < 6.0:
            return 'DAP'
        elif row['potassium'] > 55 and row['temperature'] > 28:
            return 'MOP'
        elif row['organic_matter'] < 2.5:
            return 'Compost'
        else:
            return 'NPK_15-15-15'

    df['fertilizer_type'] = df.apply(assign_fertilizer, axis=1)

    return df


# 数据预处理
def preprocess_data(df):
    """数据预处理：分割特征和目标变量，编码分类特征"""
    # 特征列
    X = df.drop('fertilizer_type', axis=1)

    # 目标变量（肥料类型）
    y = df['fertilizer_type']

    # 对分类特征进行编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 保存编码映射，以便后续解码
    class_mapping = dict(zip(le.classes_, range(len(le.classes_))))

    return X, y_encoded, le, class_mapping


# 训练模型
def train_model(X, y):
    """训练LightGBM多分类模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建并训练模型
    model = LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    )

    # 使用callbacks参数替代early_stopping_rounds
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=True)]
    )

    return model, X_train, X_test, y_train, y_test


# 获取top-k预测
def get_top_k_predictions(model, X_test, k=5):
    """获取每个样本的top-k预测类别及其概率"""
    # 获取预测概率
    probabilities = model.predict_proba(X_test)

    # 获取top-k预测的类别索引（按概率从高到低排序）
    top_k_indices = np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]

    # 获取top-k预测的概率值
    top_k_probs = np.take_along_axis(probabilities, top_k_indices, axis=1)

    return top_k_indices, top_k_probs


# 计算Precision@k
def calculate_precision_at_k(y_true, top_k_indices, k=5):
    """计算Precision@k评估指标"""
    precision_scores = []

    for i in range(len(y_true)):
        # 检查真实类别是否在top-k预测中
        relevant_items = np.sum(top_k_indices[i, :k] == y_true[i])
        precision_scores.append(relevant_items / k)

    return np.mean(precision_scores)


# 可视化特征重要性
def plot_feature_importance(model, X):
    """可视化模型特征重要性"""
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()


# 主函数
def main():
    # 生成模拟数据
    print("生成模拟数据...")
    df = generate_sample_data(n_samples=5000)

    # 数据预处理
    print("数据预处理...")
    X, y, label_encoder, class_mapping = preprocess_data(df)

    # 训练模型
    print("训练模型...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # 获取top-5预测
    print("获取Top-5预测...")
    top_5_indices, top_5_probs = get_top_k_predictions(model, X_test, k=5)

    # 计算评估指标
    accuracy = accuracy_score(y_test, model.predict(X_test))
    precision_at_3 = calculate_precision_at_k(y_test, top_5_indices, k=3)
    precision_at_5 = calculate_precision_at_k(y_test, top_5_indices, k=5)

    print(f"模型准确率: {accuracy:.4f}")
    print(f"Precision@3: {precision_at_3:.4f}")
    print(f"Precision@5: {precision_at_5:.4f}")

    # 可视化特征重要性
    plot_feature_importance(model, X)

    # 示例：显示前5个样本的预测结果
    print("\n示例预测结果 (样本索引, 真实类别, 预测类别及概率):")
    for i in range(5):
        true_class = label_encoder.inverse_transform([y_test[i]])[0]
        pred_classes = label_encoder.inverse_transform(top_5_indices[i])

        print(f"\n样本 {i+1}:")
        print(f"  真实肥料类型: {true_class}")
        print("  预测的Top-5肥料类型及概率:")
        for j in range(5):
            print(f"    {j + 1}. {pred_classes[j]} ({top_5_probs[i, j]:.4f})")


if __name__ == "__main__":
    main()

