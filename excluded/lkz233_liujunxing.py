# 学号: 2024423320113 , 姓名: 刘俊星
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import os
import sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# 1. 数据处理模块
class DataProcessor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        
    def process(self):
        """处理训练集和测试集，统一编码类别特征"""
        print("  ├── 加载数据集...")
        # 加载数据
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # 保存测试ID
        test_ids = test_df['id']
        
        # 标记数据源
        train_df['_src'] = 'train'
        test_df['_src'] = 'test'
        
        print("  ├── 合并并处理特征...")
        # 合并特征
        combined = pd.concat([
            train_df.drop(['id', 'Fertilizer Name'], axis=1, errors='ignore'),
            test_df.drop(['id'], axis=1, errors='ignore')
        ], ignore_index=True)
        
        # 编码类别特征
        for col in ['Soil Type', 'Crop Type']:
            encoder = LabelEncoder()
            combined[col] = encoder.fit_transform(combined[col])
        
        # 拆分数据
        X_train = combined[combined['_src'] == 'train'].drop('_src', axis=1)
        X_test = combined[combined['_src'] == 'test'].drop('_src', axis=1)
        y_train = train_df['Fertilizer Name']
        
        print(f"  └── 数据处理完成 | 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        return X_train, y_train, X_test, test_ids

# 2. 模型训练模块
def build_model(X_train, y_train):
    """构建并训练LightGBM分类器（静默模式）"""
    print("  ├── 初始化模型...")
    model = LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        verbose=-1  # 抑制LightGBM内部日志
    )
    
    print("  └── 执行模型训练...")
    with HiddenPrints():
        model.fit(X_train, y_train)
    return model

# 3. 生成预测结果模块
def generate_predictions(model, X_test, test_ids):
    """生成测试集的Top5预测结果（调整行顺序）"""
    print("  ├── 计算类别概率...")
    # 获取类别概率（静默模式）
    with HiddenPrints():
        probs = model.predict_proba(X_test)
    
    print("  ├── 提取Top5预测...")
    # 获取Top5类别索引（按概率从低到高排序）
    top5_indices = np.argsort(probs, axis=1)[:, :5]  # 注意这里不反转，保留从小到大顺序
    
    # 构建结果列表（按概率从低到高排列）
    results = []
    for i, (test_id, indices) in enumerate(zip(test_ids, top5_indices)):
        # 获取对应类别标签（概率从低到高）
        top5_labels = [model.classes_[idx] for idx in indices]
        results.append({
            'ID': test_id,
            'Top3': top5_labels[2],  # 第三大概率类别
            'Top1': top5_labels[0],  # 最低概率类别
            'Top5': top5_labels[4],  # 最高概率类别
            'Top2': top5_labels[1],  # 第二低概率类别
            'Top4': top5_labels[3]   # 第四大概率类别
        })
    
    print("  └── 预测结果生成完成")
    # 按ID排序后返回（行顺序调整）
    return pd.DataFrame(results, columns=['ID', 'Top3', 'Top1', 'Top5', 'Top2', 'Top4'])

# 4. 结果保存模块
def save_results(predictions, output_dir='submission'):
    """保存预测结果到CSV文件"""
    print(f"  ├── 创建输出目录: {output_dir}")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'submission_result.csv')
    
    print(f"  └── 写入CSV文件: {output_path}")
    # 保存结果（按ID排序）
    predictions.sort_values('ID').to_csv(output_path, index=False)
    return output_path

# 主程序入口
if __name__ == "__main__":
    print("="*40)
    print("      肥料类型预测系统 v1.0      ")
    print("="*40)
    
    # 数据路径
    train_path = "/kaggle/input/playground-series-s5e6/train.csv"
    test_path = "/kaggle/input/playground-series-s5e6/test.csv"
    
    # 1. 数据处理
    print("⏳ 启动数据处理模块...")
    processor = DataProcessor(train_path, test_path)
    X_train, y_train, X_test, test_ids = processor.process()
    
    # 2. 模型训练
    print("⏳ 启动模型训练模块...")
    model = build_model(X_train, y_train)
    
    # 3. 生成预测
    print("⏳ 启动预测生成模块...")
    predictions = generate_predictions(model, X_test, test_ids)
    
    # 4. 保存结果
    print("⏳ 启动结果保存模块...")
    output_path = save_results(predictions)
    print(f"✅ 结果已成功保存至: {output_path}")
    
    # 显示前5条预测
    print("\n--- 预测结果示例 ---")
    print(predictions.head())
    print("-"*40)
    print("="*40)

