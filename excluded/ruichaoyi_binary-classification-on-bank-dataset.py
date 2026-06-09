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


# 导入必要库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')

# 加载数据
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

print("训练集形状:", train.shape)
print("测试集形状:", test.shape)
print("\n训练集信息:")
print(train.info())
print("\n训练集前几行:")
print(train.head())
print("\n缺失值统计:")
print(train.isnull().sum())


# 数据探索性分析
def exploratory_analysis(df, title):
    print(f"\n=== {title} ===")
    print(f"数据集形状: {df.shape}")
    
    # 数值特征和分类特征分离
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n数值特征 ({len(numeric_cols)}个): {numeric_cols}")
    print(f"分类特征 ({len(categorical_cols)}个): {categorical_cols}")
    
    # 目标变量分布（仅训练集）
    if 'y' in df.columns:
        print(f"\n目标变量分布:")
        print(df['y'].value_counts(normalize=True))
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        df['y'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Target Variable Distribution (Count)')
        plt.xlabel('y')
        plt.ylabel('count')
        
        plt.subplot(1, 2, 2)
        df['y'].value_counts(normalize=True).plot(kind='pie', autopct='%1.1f%%', 
                                                   colors=['lightblue', 'lightcoral'])
        plt.title('Target Variable Distribution (Proportion)')
        plt.ylabel('')
        plt.tight_layout()
        plt.show()
    
    return numeric_cols, categorical_cols

# 对训练集进行探索性分析
numeric_cols, categorical_cols = exploratory_analysis(train, "训练集分析")


# 特征工程和预处理
def preprocess_data(train_df, test_df):
    # 保存ID列
    train_ids = train_df['id'] if 'id' in train_df.columns else None
    test_ids = test_df['id'] if 'id' in test_df.columns else None
    
    # 分离特征和目标变量
    if 'y' in train_df.columns:
        y = train_df['y']
        train_df = train_df.drop('y', axis=1)
    
    # 合并数据集以便统一预处理
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # 删除ID列（如果有）
    if 'id' in combined.columns:
        combined = combined.drop('id', axis=1)
    
    # 处理分类特征 - Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        if col != 'is_train':
            le = LabelEncoder()
            # 处理训练和测试集中可能出现的未知类别
            combined[col] = combined[col].astype(str)
            le.fit(combined[col])
            combined[col] = le.transform(combined[col])
            label_encoders[col] = le
    
    # 特征缩放 - 标准化数值特征
    scaler = StandardScaler()
    numeric_features = [col for col in numeric_cols if col not in ['id', 'y', 'is_train']]
    
    if numeric_features:
        combined[numeric_features] = scaler.fit_transform(combined[numeric_features])
    
    # 分离回训练集和测试集
    train_processed = combined[combined['is_train'] == 1].drop('is_train', axis=1)
    test_processed = combined[combined['is_train'] == 0].drop('is_train', axis=1)
    
    # 重新添加目标变量
    if 'y' in locals():
        train_processed['y'] = y.values
    
    return train_processed, test_processed, label_encoders, scaler

# 应用预处理
train_processed, test_processed, label_encoders, scaler = preprocess_data(train, test)

print("预处理后的训练集形状:", train_processed.shape)
print("预处理后的测试集形状:", test_processed.shape)
print("\n预处理后的训练集前几行:")
print(train_processed.head())


# 特征重要性分析
def analyze_feature_importance(X, y):
    # 使用随机森林评估特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # 获取特征重要性
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    bars = plt.barh(importance['feature'][:20], importance['importance'][:20])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importance (Random Forest)')
    plt.gca().invert_yaxis()
    
    # 添加重要性数值
    for i, (value, bar) in enumerate(zip(importance['importance'][:20], bars)):
        plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()
    
    return importance

# 准备特征和目标变量
X = train_processed.drop('y', axis=1)
y = train_processed['y']

# 分析特征重要性
feature_importance = analyze_feature_importance(X, y)
print("Top 10 The most important feature:")
print(feature_importance.head(10))


# 模型训练和评估
def train_and_evaluate_models(X, y):
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"正样本比例 (训练集): {y_train.mean():.3f}")
    print(f"正样本比例 (验证集): {y_val.mean():.3f}")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, random_state=42, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        model.fit(X_train, y_train)
        
        # 预测概率
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # 计算AUC分数
        auc_score = roc_auc_score(y_val, y_pred_proba)
        results[name] = {
            'model': model,
            'auc': auc_score,
            'predictions': y_pred_proba
        }
        
        print(f"{name} AUC: {auc_score:.4f}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_val, result['predictions'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc"]:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Comparison of ROC Curves Across Different Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 显示结果表格
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'AUC Score': [results[m]['auc'] for m in results.keys()]
    }).sort_values('AUC Score', ascending=False)
    
    print("\n=== 模型性能比较 ===")
    print(results_df.to_string(index=False))
    
    return results, results_df

# 训练和评估模型
model_results, results_df = train_and_evaluate_models(X, y)


# 交叉验证
def perform_cross_validation(X, y, n_folds=5):
    print(f"执行 {n_folds}-折交叉验证...")
    
    # 选择最佳模型（基于之前的评估）
    best_model_name = results_df.iloc[0]['Model']
    best_model = model_results[best_model_name]['model']
    
    # 设置交叉验证
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 计算交叉验证分数
    cv_scores = cross_val_score(
        best_model, X, y, 
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print(f"\n{best_model_name} 交叉验证结果:")
    print(f"各折AUC分数: {cv_scores}")
    print(f"平均AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 绘制交叉验证结果
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_folds+1), cv_scores, 'o-', linewidth=2, markersize=10)
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'平均值 = {cv_scores.mean():.4f}')
    plt.xlabel('n_folds')
    plt.ylabel('AUC scores')
    plt.title(f'{best_model_name} {n_folds}-Fold cross-validation results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return cv_scores

# 执行交叉验证
cv_scores = perform_cross_validation(X, y, n_folds=5)


# 生成预测结果
def generate_predictions(best_model, X_train, y_train, test_data, test_ids):
    print("在完整训练集上重新训练最佳模型...")
    
    # 使用完整训练集重新训练最佳模型
    best_model.fit(X_train, y_train)
    
    print("生成测试集预测...")
    # 预测概率
    test_predictions = best_model.predict_proba(test_data)[:, 1]
    
    # 创建提交文件
    submission = pd.DataFrame({
        'id': test_ids,
        'y': test_predictions
    })
    
    # 检查预测分布
    print(f"\n预测统计:")
    print(f"最小值: {submission['y'].min():.4f}")
    print(f"最大值: {submission['y'].max():.4f}")
    print(f"平均值: {submission['y'].mean():.4f}")
    print(f"标准差: {submission['y'].std():.4f}")
    
    # 绘制预测分布
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(submission['y'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Test Set Prediction Distribution')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(submission['y'])
    plt.ylabel('Prediction Probability')
    plt.title('Test Set Prediction Box Plot')
    
    plt.tight_layout()
    plt.show()
    
    return submission

# 获取最佳模型
best_model_name = results_df.iloc[0]['Model']
best_model = model_results[best_model_name]['model']

# 生成预测
submission = generate_predictions(
    best_model, 
    X, y,
    test_processed,
    test['id'] if 'id' in test.columns else None
)

# 保存提交文件
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)
print(f"\n提交文件已保存为: {submission_file}")
print(f"文件形状: {submission.shape}")
print("\n提交文件前几行:")
print(submission.head())

