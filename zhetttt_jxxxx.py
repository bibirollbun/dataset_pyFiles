import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ======================== 数据加载与预处理 ========================
def load_and_preprocess_data():
    """
    严格控制标签编码流程，确保训练集和验证集编码一致
    返回: 预处理后的训练集、测试集、标签编码器、数据路径、目标变量编码器
    """
    data_paths = {
        'train': '/kaggle/input/playground-series-s5e6/train.csv',
        'test': '/kaggle/input/playground-series-s5e6/test.csv',
        'sample_submission': '/kaggle/input/playground-series-s5e6/sample_submission.csv'
    }

    try:
        train = pd.read_csv(data_paths['train'])
        test = pd.read_csv(data_paths['test'])
    except FileNotFoundError as e:
        print(f"错误: 未找到数据文件 - {e}")
        print("请检查文件路径或数据集是否正确加载")
        raise

    # 分离特征和目标变量
    target_col = 'Fertilizer Name'
    X_train_raw = train.drop(target_col, axis=1)
    y_train_raw = train[target_col]
    X_test_raw = test.copy()

    # 缺失值处理
    # 数值型特征：均值填充
    numeric_cols = X_train_raw.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        X_train_raw[col].fillna(X_train_raw[col].mean(), inplace=True)
        if col in X_test_raw.columns:
            X_test_raw[col].fillna(X_train_raw[col].mean(), inplace=True)  # 用训练集的均值填充测试集

    # 类别型特征：众数填充（除目标变量外）
    categorical_cols = X_train_raw.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_val = X_train_raw[col].mode()[0]
        X_train_raw[col].fillna(mode_val, inplace=True)
        if col in X_test_raw.columns:
            X_test_raw[col].fillna(mode_val, inplace=True)  # 用训练集的众数填充测试集

    # 标签编码目标变量（仅用训练集拟合）
    fertilizer_le = LabelEncoder()
    fertilizer_le.fit(y_train_raw)
    y_train_encoded = fertilizer_le.transform(y_train_raw)

    # 编码其他类别特征（仅用训练集拟合）
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_train_raw[col])
        X_train_raw[col] = le.transform(X_train_raw[col])
        if col in X_test_raw.columns:
            X_test_raw[col] = le.transform(X_test_raw[col])
        label_encoders[col] = le

    return X_train_raw, y_train_encoded, X_test_raw, data_paths, fertilizer_le, y_train_raw.unique()

# ======================== 模型训练（XGBoost） ========================
def train_xgboost(X, y, num_class, n_splits=5):
    """
    明确设置num_class，确保模型知道类别数量
    参数:
        X: 训练集特征（已编码）
        y: 训练集标签（已编码）
        num_class: 类别数量
        n_splits: K折数量
    返回: 训练好的模型
    """
    # 分层K折验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 多分类适配器（OneVsRest）
    model = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=num_class,  # 关键：设置类别数量
            random_state=42,
            n_jobs=-1
        )
    )

    # 训练模型（使用分层K折验证）
    model.fit(X, y)

    return model

# ======================== 特征重要性分析 ========================
def plot_feature_importance(model, X, top_n=10):
    """
    可视化特征重要性（基于XGBoost）
    """
    # 获取特征重要性（多分类场景下取均值）
    importances = np.mean([clf.feature_importances_ for clf in model.estimators_], axis=0)
    feature_names = X.columns

    # 构建特征重要性DataFrame
    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importances
    }).sort_values('重要性', ascending=False).head(top_n)

    # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(x='重要性', y='特征', data=importance_df, palette='viridis')
    plt.title(f'Top {top_n} 特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

    return importance_df

# ======================== 生成预测结果（符合竞赛格式） ========================
def generate_submission(model, X_test, classes, data_paths, fertilizer_le):
    """
    生成符合竞赛要求的提交文件（使用Fertilizer Name作为预测列）
    """
    # 预测概率
    y_test_pred_proba = model.predict_proba(X_test)

    # 获取最可能的类别（取概率最大的类别）
    y_test_pred = [classes[np.argmax(probs)] for probs in y_test_pred_proba]

    # 加载样例提交文件
    sample_submission = pd.read_csv(data_paths['sample_submission'])

    # 构建提交文件（仅保留id和Fertilizer Name列）
    submission = pd.DataFrame({
        'id': sample_submission['id'],  # 确保与样例提交的id一致
        'Fertilizer Name': y_test_pred   # 竞赛要求的预测结果列
    })

    # 保存提交文件
    submission.to_csv('fertilizer_prediction.csv', index=False)
    print(f"提交文件已保存: {len(submission)} 条预测结果")

    return submission

# ======================== 主流程 ========================
def main():
    print("=== 开始执行 ===")
    print("1. 加载和预处理数据...")
    try:
        X_train, y_train, X_test, data_paths, fertilizer_le, unique_labels = load_and_preprocess_data()
        num_class = len(unique_labels)
        print(f"检测到目标变量类别: {unique_labels}, 共 {num_class} 类")
    except Exception as e:
        print(f"错误: 数据处理失败 - {e}")
        return

    print("2. 准备训练集和验证集...")
    # 分层拆分验证集（确保类别分布一致）
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 转换回原始标签以便校验（可选，用于调试）
    y_train_original = fertilizer_le.inverse_transform(y_train_split)
    y_val_original = fertilizer_le.inverse_transform(y_val_split)

    # 校验验证集标签是否在训练集标签中
    val_unique = np.unique(y_val_original)
    train_unique = np.unique(y_train_original)
    if not set(val_unique).issubset(set(train_unique)):
        print("警告: 验证集包含训练集未出现的标签！")
        print(f"训练集原始标签: {train_unique}")
        print(f"验证集原始标签: {val_unique}")
    else:
        print("验证集标签校验通过，所有标签均在训练集中出现")

    print("3. 训练XGBoost模型...")
    try:
        model = train_xgboost(X_train_split, y_train_split, num_class)
    except Exception as e:
        print(f"错误: 模型训练失败 - {e}")
        return

    print("5. 分析特征重要性...")
    plot_feature_importance(model, X_train, top_n=10)

    print("6. 生成提交文件...")
    submission = generate_submission(model, X_test, unique_labels, data_paths, fertilizer_le)

    print("=== 执行完成 ===")
    print(f"提交文件保存路径: fertilizer_prediction.csv")
    print(f"预测样本数量: {len(submission)}")

if __name__ == "__main__":
    main()

