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


# 学号: XXX, 姓名: XXX

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

# 1. 生成更合理的模拟数据
def generate_improved_data(n_samples=1000):
    np.random.seed(42)
    
    # 生成更符合实际的土壤特征
    data = {
        'Nitrogen': np.clip(np.random.normal(25, 8, n_samples), 5, 50),
        'Phosphorus': np.clip(np.random.normal(15, 5, n_samples), 5, 30),
        'Potassium': np.clip(np.random.normal(20, 6, n_samples), 10, 40),
        'pH': np.clip(np.random.normal(6.5, 0.8, n_samples), 5.0, 8.5),
        'Temperature': np.clip(np.random.normal(25, 5, n_samples), 15, 35),
        'Humidity': np.clip(np.random.normal(60, 15, n_samples), 30, 90)
    }
    
    # 定义肥料类型及其特性
    fertilizer_types = {
        'NPK_30-10-10': {'N': 30, 'P': 10, 'K': 10},
        'NPK_20-20-20': {'N': 20, 'P': 20, 'K': 20},
        'NPK_15-15-15': {'N': 15, 'P': 15, 'K': 15},
        'Urea': {'N': 46, 'P': 0, 'K': 0},
        'DAP': {'N': 18, 'P': 46, 'K': 0},
        'MOP': {'N': 0, 'P': 0, 'K': 60},
        'SSP': {'N': 0, 'P': 20, 'K': 0},
        'TSP': {'N': 0, 'P': 46, 'K': 0}
    }
    
    # 更智能的肥料分配规则
    def assign_fertilizer(row):
        n, p, k, ph, temp, hum = row
        
        # 根据土壤条件分配肥料
        if ph < 6.0 and hum > 70 and n > 25:
            return 'Urea'
        elif ph > 7.5 and p < 15:
            return 'TSP'
        elif k < 15 and temp > 30:
            return 'MOP'
        elif n > 30 and p < 15:
            return 'NPK_30-10-10'
        elif n < 20 and p > 25:
            return 'DAP'
        elif k > 25 and temp < 20:
            return 'NPK_15-15-15'
        elif p > 15 and k > 15 and n > 15:
            return 'NPK_20-20-20'
        else:
            return 'SSP'
    
    # 应用分配规则
    features = np.column_stack([data['Nitrogen'], data['Phosphorus'], 
                              data['Potassium'], data['pH'], 
                              data['Temperature'], data['Humidity']])
    data['Fertilizer'] = [assign_fertilizer(x) for x in features]
    
    # 添加ID列
    data['id'] = range(1, n_samples+1)
    
    return pd.DataFrame(data), fertilizer_types

# 2. 修正的MAP@5评估指标
def improved_map_at_5(y_true, y_pred_top5, fertilizer_info=None):
    n = len(y_true)
    total_map = 0.0
    correct_counts = np.zeros(5)
    position_scores = np.zeros(5)
    class_analysis = {}
    
    # 初始化类别分析
    for cls in np.unique(y_true):
        class_analysis[cls] = {'count': 0, 'correct': np.zeros(5)}
    
    # 计算MAP@5
    for i in range(n):
        true_class = y_true[i]
        class_analysis[true_class]['count'] += 1
        
        ap = 0.0
        hits = 0
        
        for k in range(5):
            if y_pred_top5[i, k] == true_class:
                hits += 1
                ap += hits / (k + 1)
                correct_counts[k] += 1
                position_scores[k] += 1 / (k + 1)
                class_analysis[true_class]['correct'][k] += 1
        
        total_map += ap / min(5, 1)
    
    # 计算各类别统计
    class_stats = []
    for cls, info in class_analysis.items():
        total = info['count']
        correct = info['correct']
        recall_at_5 = np.sum(correct) / total if total > 0 else 0
        avg_position = np.sum([(k+1)*correct[k] for k in range(5)]) / np.sum(correct) if np.sum(correct) > 0 else 0
        
        if fertilizer_info and cls in fertilizer_info:
            n_val = fertilizer_info[cls]['N']
            p_val = fertilizer_info[cls]['P']
            k_val = fertilizer_info[cls]['K']
            class_stats.append([cls, n_val, p_val, k_val, total, recall_at_5, avg_position])
        else:
            class_stats.append([cls, np.nan, np.nan, np.nan, total, recall_at_5, avg_position])
    
    # 创建统计DataFrame
    stats_df = pd.DataFrame(class_stats, columns=[
        'Fertilizer', 'N%', 'P%', 'K%', 'Count', 'Recall@5', 'AvgPosition'
    ]).sort_values('Count', ascending=False)
    
    # 计算总体指标
    map_score = total_map / n if n > 0 else 0
    accuracy_at_k = correct_counts / n if n > 0 else np.zeros(5)
    precision_at_k = correct_counts / (np.arange(1, 6) * n) if n > 0 else np.zeros(5)
    
    return map_score, stats_df, accuracy_at_k, precision_at_k, position_scores

# 3. 改进的可视化
def improved_visualization(data, fertilizer_info):
    plt.figure(figsize=(18, 12))
    
    # 1. 特征分布
    plt.subplot(2, 3, 1)
    sns.boxplot(data=data[['Nitrogen', 'Phosphorus', 'Potassium']])
    plt.title('Nutrient Distribution')
    
    plt.subplot(2, 3, 2)
    sns.histplot(data['pH'], bins=20, kde=True, color='gold')
    plt.title('pH Distribution')
    
    plt.subplot(2, 3, 3)
    sns.scatterplot(data=data, x='Temperature', y='Humidity', hue='Fertilizer', 
                    palette='viridis', alpha=0.6)
    plt.title('Temp vs Humidity by Fertilizer')
    
    # 2. 肥料成分热图
    plt.subplot(2, 3, 4)
    fert_comp = pd.DataFrame(fertilizer_info).T[['N', 'P', 'K']]
    sns.heatmap(fert_comp, annot=True, cmap='YlGnBu', fmt='g')
    plt.title('Fertilizer NPK Composition')
    
    # 3. 肥料类型分布
    plt.subplot(2, 3, 5)
    fert_counts = data['Fertilizer'].value_counts()
    plt.pie(fert_counts, labels=fert_counts.index, autopct='%1.1f%%', 
            colors=sns.color_palette('pastel'), startangle=90)
    plt.title('Fertilizer Distribution')
    
    # 4. 养分与肥料类型关系
    plt.subplot(2, 3, 6)
    sample_data = data.sample(100)
    sns.scatterplot(data=sample_data, x='Nitrogen', y='Phosphorus', 
                    hue='Fertilizer', size='Potassium', sizes=(20, 200), 
                    alpha=0.7, palette='Set2')
    plt.title('N-P-K by Fertilizer Type')
    
    plt.tight_layout()
    plt.savefig('fertilizer_analysis.png', dpi=300)
    plt.close()

# 4. 改进的随机森林分类器
class ImprovedRandomForest:
    def __init__(self, n_estimators=50, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)
        
        for _ in range(self.n_estimators):
            # 自助采样
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X.iloc[sample_indices]
            y_sample = y[sample_indices]
            
            # 构建决策树
            tree = self._build_tree(X_sample, y_sample, depth=0)
            self.trees.append(tree)
            
            # 更新特征重要性
            self._update_feature_importance(tree)
        
        # 归一化特征重要性
        self.feature_importances_ /= self.n_estimators
    
    def _build_tree(self, X, y, depth):
        unique_classes, counts = np.unique(y, return_counts=True)
        
        if depth >= self.max_depth or len(unique_classes) == 1:
            return {'is_leaf': True, 'pred': unique_classes[np.argmax(counts)], 'samples': len(y)}
        
        # 随机选择特征子集
        n_features = X.shape[1]
        feature_indices = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        
        best_feature, best_threshold = None, None
        best_gini = np.inf
        
        for feature in feature_indices:
            unique_values = np.unique(X.iloc[:, feature])
            if len(unique_values) > 10:
                thresholds = np.percentile(X.iloc[:, feature], [25, 50, 75])
            else:
                thresholds = unique_values
            
            for threshold in thresholds:
                left_mask = X.iloc[:, feature] <= threshold
                right_mask = ~left_mask
                
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                
                # 计算基尼指数
                gini = (self._gini_impurity(y[left_mask]) * sum(left_mask) + 
                        self._gini_impurity(y[right_mask]) * sum(right_mask)) / len(y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        if best_feature is None:
            return {'is_leaf': True, 'pred': unique_classes[np.argmax(counts)], 'samples': len(y)}
        
        left_mask = X.iloc[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node = {
            'is_leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth+1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth+1),
            'samples': len(y),
            'gini': best_gini
        }
        return node
    
    def _update_feature_importance(self, tree):
        if tree['is_leaf']:
            return
        
        # 重要性计算
        impurity_reduction = (tree['samples'] * tree['gini'] - 
                             tree['left']['samples'] * tree['left'].get('gini', 0) - 
                             tree['right']['samples'] * tree['right'].get('gini', 0))
        
        self.feature_importances_[tree['feature']] += impurity_reduction
        
        # 递归更新左右子树
        self._update_feature_importance(tree['left'])
        self._update_feature_importance(tree['right'])
    
    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _predict_tree(self, x, tree):
        if tree['is_leaf']:
            return tree['pred']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self.classes_)))
        
        for i in range(n_samples):
            x = X.iloc[i]
            votes = np.zeros(len(self.classes_))
            
            for tree in self.trees:
                pred = self._predict_tree(x, tree)
                class_idx = np.where(self.classes_ == pred)[0][0]
                votes[class_idx] += 1
            
            proba[i] = votes / self.n_estimators
        
        return proba
    
    def predict_top5(self, X):
        proba = self.predict_proba(X)
        top5_indices = np.argsort(proba, axis=1)[:, -5:][:, ::-1]
        return np.array([[str(self.classes_[i]) for i in row] for row in top5_indices])

# 5. 改进的主程序
def improved_main():
    print("="*70)
    print("FERTILIZER TYPE PREDICTION SYSTEM".center(70))
    print("="*70 + "\n")
    
    # 1. 数据生成
    print("[1/4] Generating improved synthetic data...")
    data, fertilizer_info = generate_improved_data(1500)
    
    # 显示数据样例
    print("\nSample data (first 3 rows):")
    display(data.head(3))
    
    # 2. 数据可视化
    print("\n[2/4] Creating improved visualizations...")
    improved_visualization(data, fertilizer_info)
    print("Visualizations saved as 'fertilizer_analysis.png'")
    
    # 3. 模型训练
    print("\n[3/4] Training improved random forest model...")
    X = data.drop(['id', 'Fertilizer'], axis=1)
    y = data['Fertilizer']
    
    # 编码目标变量
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # 训练模型
    model = ImprovedRandomForest(n_estimators=50, max_depth=5)
    model.fit(X_train, y_train)
    
    # 显示特征重要性
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    display(importance_df)
    
    # 4. 评估与预测
    print("\n[4/4] Evaluating model performance...")
    val_preds_top5 = model.predict_top5(X_val)
    
    # 改进的评估
    map_score, stats_df, accuracy_at_k, precision_at_k, _ = improved_map_at_5(
        le.inverse_transform(y_val), val_preds_top5, fertilizer_info)
    
    print(f"\nOverall MAP@5 Score: {map_score:.4f}")
    
    print("\nAccuracy at each position:")
    for k in range(5):
        print(f"Position {k+1}: {accuracy_at_k[k]:.2%}")
    
    print("\nDetailed Fertilizer Performance:")
    display(stats_df)
    
    # 5. 生成测试集预测
    test_data = data.sample(200, random_state=123)
    X_test = test_data.drop(['id', 'Fertilizer'], axis=1)
    test_ids = test_data['id']
    test_actual = test_data['Fertilizer']
    
    test_preds_top5 = model.predict_top5(X_test)
    test_proba = model.predict_proba(X_test)
    
    # 准备详细的结果报告
    detailed_results = []
    for i in range(len(test_data)):
        actual = test_actual.iloc[i]
        preds = test_preds_top5[i]
        probs = test_proba[i][np.argsort(test_proba[i])[-5:][::-1]]
        
        correct_pos = np.where(np.array(preds) == actual)[0]
        is_correct = len(correct_pos) > 0
        correct_pos = correct_pos[0] + 1 if is_correct else "N/A"
        
        detailed_results.append([
            test_ids.iloc[i],
            actual,
            ", ".join(preds),
            ", ".join([f"{p:.2f}" for p in probs]),
            str(is_correct),
            str(correct_pos)
        ])
    
    results_df = pd.DataFrame(detailed_results, columns=[
        'ID', 'Actual', 'Top5 Predictions', 'Probabilities', 'Correct', 'Position'
    ])
    
    # 保存结果
    submission = pd.DataFrame({
        'id': test_ids,
        'Fertilizer_1': test_preds_top5[:, 0],
        'Fertilizer_2': test_preds_top5[:, 1],
        'Fertilizer_3': test_preds_top5[:, 2],
        'Fertilizer_4': test_preds_top5[:, 3],
        'Fertilizer_5': test_preds_top5[:, 4]
    })
    
    submission.to_csv('submission.csv', index=False)
    results_df.to_csv('detailed_results.csv', index=False)
    
    # 显示部分详细结果
    print("\nSample Detailed Predictions (first 5 rows):")
    display(results_df.head(5))
    
    # 计算并显示整体统计
    correct_in_top5 = results_df[results_df['Correct'] == 'True'].shape[0]
    accuracy_top1 = results_df[results_df['Position'] == '1'].shape[0] / len(results_df)
    accuracy_top5 = correct_in_top5 / len(results_df)
    
    print("\nFinal Statistics:")
    print(f"- Top-1 Accuracy: {accuracy_top1:.2%}")
    print(f"- Top-5 Accuracy: {accuracy_top5:.2%}")
    
    if correct_in_top5 > 0:
        avg_pos = results_df[results_df['Correct'] == 'True']['Position'].apply(float).mean()
        print(f"- Mean Position of Correct Predictions: {avg_pos:.2f}")
    else:
        print("- No correct predictions in top-5")
    
    print("\nSubmission file saved as 'submission.csv'")
    print("Detailed results saved as 'detailed_results.csv'")
    print("\n" + "="*70)
    print("PROCESS COMPLETED SUCCESSFULLY".center(70))
    print("="*70)

if __name__ == "__main__":
    improved_main()

