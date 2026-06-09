import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import gzip
import os

def load_local_fashion_mnist(data_dir='./datasets/fashion_mnist/'):
    """从本地加载Fashion-MNIST数据集"""
    def parse_idx_file(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data
    
    def parse_idx_label(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        raise Exception(f"请先将数据集文件放在{data_dir}目录下")
    
    train_images = parse_idx_file(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = parse_idx_label(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    test_images = parse_idx_file(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = parse_idx_label(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    train_images = train_images.reshape(-1, 28, 28)
    test_images = test_images.reshape(-1, 28, 28)
    return (train_images, train_labels), (test_images, test_labels)

# 加载数据
try:
    (X_train, y_train), (X_test, y_test) = load_local_fashion_mnist()
    print("成功从本地加载Fashion-MNIST数据集")
except:
    from tensorflow.keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print("成功从网络加载Fashion-MNIST数据集")

# 数据探索
print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
print(f"训练集标签分布: {np.bincount(y_train)}")
print(f"测试集标签分布: {np.bincount(y_test)}")

# 可视化样本
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    idx = np.where(y_train == i)[0][0]
    plt.imshow(X_train[idx], cmap='gray')
    plt.title(class_names[y_train[idx]])
    plt.axis('off')
plt.tight_layout()
plt.savefig('fashion_mnist_samples.png', dpi=300)
plt.show()



def svd_implementation(X, n_components=None):
    """实现SVD分解并返回降维后数据与模型"""
    # 数据预处理
    X_flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    
    # 标准化数据
    print("正在标准化数据...")
    X_scaled = scaler.fit_transform(X_flat)
    
    # 执行SVD
    if n_components is None:
        n_components = min(X_scaled.shape) - 1
    print(f"执行SVD分解，目标维度: {n_components}")
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X_scaled)
    
    return X_reduced, svd, X_scaled


def svd_implementation(X, n_components=None):
    """实现SVD分解并返回降维后数据与模型"""
    X_flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    max_components = min(X_scaled.shape) - 1
    if n_components is None:
        n_components = max_components
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X_scaled)
    return X_reduced, svd, X_scaled

# 执行SVD并获取模型
X_train_reduced, svd_model, X_train_scaled = svd_implementation(X_train)
print(f"SVD分解完成，分解后维度: {X_train_reduced.shape[1]}")

# 绘制奇异值衰减曲线（确保svd_model已定义）
plt.figure(figsize=(10, 6))
plt.plot(svd_model.singular_values_)
plt.title('Singular Values Decay Curve')
plt.xlabel('Component Index')
plt.ylabel('Singular Value')
plt.grid(True)
plt.savefig('singular_values_decay.png', dpi=300)
plt.show()


# 计算累积解释方差
cumulative_variance = np.cumsum(explained_variance_ratio)

# 绘制累积解释方差曲线
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance)
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
plt.axhline(y=0.99, color='m', linestyle='--', label='99% Variance')

# 计算达到特定方差所需维度
n_comp_90 = np.argmax(cumulative_variance >= 0.9) + 1
n_comp_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_comp_99 = np.argmax(cumulative_variance >= 0.99) + 1

plt.axvline(x=n_comp_90, color='r', linestyle='--')
plt.axvline(x=n_comp_95, color='g', linestyle='--')
plt.axvline(x=n_comp_99, color='m', linestyle='--')

plt.title('Cumulative Explained Variance Curve')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.legend()
plt.grid(True)
plt.savefig('cumulative_variance.png', dpi=300)
plt.show()

# 输出结果
print(f"保留90%方差所需维度: {n_comp_90}")
print(f"保留95%方差所需维度: {n_comp_95}")
print(f"保留99%方差所需维度: {n_comp_99}")


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import pandas as pd

def dimension_performance_test(dimensions, X_train, y_train, X_test, y_test, n_neighbors=5):
    """测试不同维度下的KNN性能"""
    results = {
        'dimension': [],
        'accuracy': [],
        'train_time': [],
        'predict_time': []
    }
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))
    
    # 测试每个维度
    for dim in dimensions:
        print(f"测试维度: {dim}")
        # SVD降维
        svd = TruncatedSVD(n_components=dim)
        X_train_reduced = svd.fit_transform(X_train_scaled)
        X_test_reduced = svd.transform(X_test_scaled)
        
        # KNN分类
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        # 测量训练时间
        start_time = time.time()
        knn.fit(X_train_reduced, y_train)
        train_time = time.time() - start_time
        
        # 测量预测时间
        start_time = time.time()
        y_pred = knn.predict(X_test_reduced)
        predict_time = time.time() - start_time
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        # 存储结果
        results['dimension'].append(dim)
        results['accuracy'].append(accuracy)
        results['train_time'].append(train_time)
        results['predict_time'].append(predict_time)
        
        print(f"  准确率: {accuracy:.4f}, 训练时间: {train_time:.4f}s, 预测时间: {predict_time:.4f}s")
    
    return pd.DataFrame(results)

# 定义测试维度
test_dimensions = [10, 25, 50, 100, 200, 300]
# 执行实验
results_df = dimension_performance_test(test_dimensions, X_train, y_train, X_test, y_test, n_neighbors=5)
# 保存结果
results_df.to_csv('dimension_performance.csv', index=False)
print("实验结果已保存至 dimension_performance.csv")


# 绘制准确率-维度关系图
plt.figure(figsize=(10, 6))
plt.plot(results_df['dimension'], results_df['accuracy'], 'o-', linewidth=2)
plt.xlabel('Number of Dimensions')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. Dimensionality')
plt.grid(True)
plt.savefig('accuracy_vs_dimension.png', dpi=300)
plt.show()

# 绘制时间-维度关系图
plt.figure(figsize=(10, 6))
plt.plot(results_df['dimension'], results_df['train_time'], 'o-', label='Training Time')
plt.plot(results_df['dimension'], results_df['predict_time'], 's-', label='Prediction Time')
plt.xlabel('Number of Dimensions')
plt.ylabel('Time (Seconds)')
plt.title('KNN Time Consumption vs. Dimensionality')
plt.legend()
plt.grid(True)
plt.savefig('time_vs_dimension.png', dpi=300)
plt.show()

# 输出结果表格
print("\n维度对KNN性能影响实验结果:")
print(results_df.to_string(index=False))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def knn_k_optimization(X_train, y_train, X_test, y_test, best_dim, k_range=range(1, 21)):
    """
    优化KNN分类器的k值参数
    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        best_dim: 最佳降维维度
        k_range: 要测试的k值范围
    返回:
        k_results: 包含不同k值下的准确率和时间消耗的DataFrame
    """
    # 初始化SVD和标准化器
    svd = TruncatedSVD(n_components=best_dim)
    scaler = StandardScaler()
    
    # 数据预处理：展平图像
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 标准化处理（使用同一个标准化器）
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # SVD降维（训练集和测试集使用相同的SVD模型）
    X_train_reduced = svd.fit_transform(X_train_scaled)
    X_test_reduced = svd.transform(X_test_scaled)
    
    # 存储不同k值的结果
    k_results = {'k': [], 'accuracy': [], 'train_time': [], 'predict_time': []}
    
    # 测试每个k值
    print(f"开始K值优化实验，测试维度: {best_dim}")
    for k in k_range:
        print(f"  测试k值: {k}")
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # 测量训练时间
        start_time = time.time()
        knn.fit(X_train_reduced, y_train)
        train_time = time.time() - start_time
        
        # 测量预测时间
        start_time = time.time()
        y_pred = knn.predict(X_test_reduced)
        predict_time = time.time() - start_time
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        # 存储结果
        k_results['k'].append(k)
        k_results['accuracy'].append(accuracy)
        k_results['train_time'].append(train_time)
        k_results['predict_time'].append(predict_time)
        
        print(f"  结果: 准确率={accuracy:.4f}, 训练时间={train_time:.4f}s, 预测时间={predict_time:.4f}s")
    
    return pd.DataFrame(k_results)

# 执行k值优化（假设已加载X_train, y_train, X_test, y_test）
best_dim = 100  # 基于2.3节确定的最佳维度
k_results = knn_k_optimization(X_train, y_train, X_test, y_test, best_dim)

# 输出结果表格
print("\nK值优化实验结果汇总:")
print(k_results.to_string(index=False))

# 绘制k值-准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(k_results['k'], k_results['accuracy'], 'o-', color='#00A1FF', linewidth=2)
plt.xlabel('k Value', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN Classification Accuracy vs. k Value (Dimension=100)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 标记最佳k值
best_k = k_results.loc[k_results['accuracy'].idxmax(), 'k']
best_accuracy = k_results.loc[k_results['accuracy'].idxmax(), 'accuracy']
plt.scatter(best_k, best_accuracy, color='red', s=100, zorder=5)
plt.annotate(f'Best k={best_k}\nAccuracy={best_accuracy:.4f}', 
             xy=(best_k, best_accuracy), 
             xytext=(best_k+2, best_accuracy-0.05),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12)

plt.tight_layout()
plt.savefig('knn_k_optimization.png', dpi=300, bbox_inches='tight')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

def knn_k_optimization(X_train, y_train, X_test, y_test, best_dim, k_range=range(1, 21)):
    """优化KNN的k值参数并返回结果"""
    # 函数实现...（同3.1节完整代码）
    svd = TruncatedSVD(n_components=best_dim)
    scaler = StandardScaler()
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    X_train_reduced = svd.fit_transform(X_train_scaled)
    X_test_reduced = svd.transform(X_test_scaled)
    
    k_results = {'k': [], 'accuracy': [], 'train_time': [], 'predict_time': []}
    
    for k in k_range:
        print(f"测试k值: {k}")
        knn = KNeighborsClassifier(n_neighbors=k)
        start_time = time.time()
        knn.fit(X_train_reduced, y_train)
        train_time = time.time() - start_time
        start_time = time.time()
        y_pred = knn.predict(X_test_reduced)
        predict_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        k_results['k'].append(k)
        k_results['accuracy'].append(accuracy)
        k_results['train_time'].append(train_time)
        k_results['predict_time'].append(predict_time)
    
    return pd.DataFrame(k_results)

def load_fashion_mnist_local(data_dir='./datasets/fashion_mnist/'):
    """本地加载Fashion-MNIST数据集"""
    # 数据加载函数（同之前实现）
    def parse_idx_file(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data
    
    def parse_idx_label(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        raise Exception(f"请先将数据集文件放在{data_dir}目录下")
    
    train_images = parse_idx_file(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = parse_idx_label(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    test_images = parse_idx_file(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = parse_idx_label(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    train_images = train_images.reshape(-1, 28, 28)
    test_images = test_images.reshape(-1, 28, 28)
    return (train_images, train_labels), (test_images, test_labels)

# 主执行流程
try:
    # 加载数据集
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist_local()
    
    # 执行K值优化
    best_dim = 100
    k_results = knn_k_optimization(X_train, y_train, X_test, y_test, best_dim)
    
    # 确保k_results有效
    if k_results is not None and not k_results.empty:
        # 获取最佳k值
        best_k = k_results.loc[k_results['accuracy'].idxmax(), 'k']
        print(f"最佳k值: {best_k}")
        
        # 数据预处理与降维
        svd = TruncatedSVD(n_components=best_dim)
        scaler = StandardScaler()
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        X_train_reduced = svd.fit_transform(X_train_scaled)
        X_test_reduced = svd.transform(X_test_scaled)
        
        # 训练最佳k值的KNN模型
        knn_best = KNeighborsClassifier(n_neighbors=best_k)
        knn_best.fit(X_train_reduced, y_train)
        
        # 预测测试集
        y_pred = knn_best.predict(X_test_reduced)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        # 确保cm非空
        if cm.size == 0:
            raise ValueError("混淆矩阵为空，请检查预测结果")
        
        # 绘制混淆矩阵热力图
        plt.figure(figsize=(12, 10))
        sns.set(font_scale=1.2)  # 调整字体大小
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                             xticklabels=class_names, yticklabels=class_names)
        heatmap.set_xlabel('Predicted Label', fontsize=14)
        heatmap.set_ylabel('True Label', fontsize=14)
        heatmap.set_title(f'Confusion Matrix (k={best_k}, dim={best_dim})', fontsize=16)
        
        # 确保图像保存路径存在
        save_dir = './results/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
    else:
        print("错误：k_results为空或未正确生成，无法绘制混淆矩阵")

except Exception as e:
    print(f"执行过程中出错: {e}")
    # 错误处理：使用默认参数生成示例混淆矩阵（用于测试）
    print("生成示例混淆矩阵...")
    cm = np.random.randint(0, 100, size=(10, 10))
    class_names = ['Class'+str(i) for i in range(10)]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Example Confusion Matrix (For Testing)')
    plt.savefig('example_confusion_matrix.png', dpi=300)
    plt.show()


def train_final_model(X_train, y_train, best_dim, best_k):
    """训练最终模型"""
    # 数据预处理与降维
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    
    svd = TruncatedSVD(n_components=best_dim)
    X_train_reduced = svd.fit_transform(X_train_scaled)
    
    # 训练KNN模型
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_reduced, y_train)
    
    return knn, svd, scaler

# 训练最终模型
best_dim = 100
best_k = 5
final_knn, final_svd, final_scaler = train_final_model(X_train, y_train, best_dim, best_k)
print("最终模型训练完成")


def generate_prediction(X_test, model, svd, scaler, best_dim):
    """生成测试集预测并保存"""
    # 数据预处理
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_test_scaled = scaler.transform(X_test_flat)
    X_test_reduced = svd.transform(X_test_scaled)
    
    # 预测
    y_pred = model.predict(X_test_reduced)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'ImageId': range(1, len(y_pred) + 1),
        'Label': y_pred
    })
    
    # 保存为CSV
    submission.to_csv('fashion_mnist_submission.csv', index=False)
    print(f"预测完成，结果已保存至 fashion_mnist_submission.csv")
    print(f"预测标签分布: {np.bincount(y_pred)}")
    
    return submission

# 生成预测
submission = generate_prediction(X_test, final_knn, final_svd, final_scaler, best_dim)


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def compare_classifiers(X_train, y_train, X_test, y_test, best_dim):
    """对比不同分类器性能"""
    # 数据预处理与降维
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    svd = TruncatedSVD(n_components=best_dim)
    X_train_reduced = svd.fit_transform(X_train_scaled)
    X_test_reduced = svd.transform(X_test_scaled)
    
    # 定义分类器
    classifiers = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(gamma='scale', random_state=42)
    }
    
    results = {'classifier': [], 'accuracy': [], 'train_time': [], 'predict_time': []}
    
    # 测试每个分类器
    for name, clf in classifiers.items():
        print(f"测试分类器: {name}")
        
        # 测量训练时间
        start_time = time.time()
        clf.fit(X_train_reduced, y_train)
        train_time = time.time() - start_time
        
        # 测量预测时间
        start_time = time.time()
        y_pred = clf.predict(X_test_reduced)
        predict_time = time.time() - start_time
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        # 存储结果
        results['classifier'].append(name)
        results['accuracy'].append(accuracy)
        results['train_time'].append(train_time)
        results['predict_time'].append(predict_time)
        
        print(f"  准确率: {accuracy:.4f}, 训练时间: {train_time:.4f}s, 预测时间: {predict_time:.4f}s")
    
    return pd.DataFrame(results)

# 执行对比实验
comparison_results = compare_classifiers(X_train, y_train, X_test, y_test, best_dim=100)
print("\n分类器对比实验结果:")
print(comparison_results.to_string(index=False))

# 绘制对比图表
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(comparison_results['classifier'], comparison_results['accuracy'])
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy Comparison')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(comparison_results['classifier'], comparison_results['train_time'], label='Train Time')
plt.bar(comparison_results['classifier'], comparison_results['predict_time'], bottom=comparison_results['train_time'], label='Predict Time')
plt.xlabel('Classifier')
plt.ylabel('Time (Seconds)')
plt.title('Classifier Time Consumption Comparison')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('classifier_comparison.png', dpi=300)
plt.show()

