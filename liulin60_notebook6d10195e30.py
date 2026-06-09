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


# =============================================================================
# 修复对齐错误的完整代码
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# 设置图形样式
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# 1. 数据加载和探索 - 修复版本
# =============================================================================
print("=" * 60)
print("桑坦德客户满意度预测 - 修复对齐错误版本")
print("=" * 60)

# 加载数据
train_df = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')
test_df = pd.read_csv('/kaggle/input/santander-customer-satisfaction/test.csv')

print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)

# 修复函数：确保训练集和测试集列对齐
def align_dataframes(train_df, test_df):
    """确保训练集和测试集的列对齐"""
    print("对齐训练集和测试集列...")
    
    # 获取共同的列（排除ID和TARGET）
    train_cols = set(train_df.columns) - {'ID', 'TARGET'}
    test_cols = set(test_df.columns) - {'ID'}
    common_cols = sorted(list(train_cols & test_cols))
    
    print(f"训练集特征数: {len(train_cols)}")
    print(f"测试集特征数: {len(test_cols)}")
    print(f"共同特征数: {len(common_cols)}")
    
    # 重新组织数据框，确保列顺序一致
    train_aligned = train_df[['ID', 'TARGET'] + common_cols].copy()
    test_aligned = test_df[['ID'] + common_cols].copy()
    
    return train_aligned, test_aligned

# 应用对齐
train_df, test_df = align_dataframes(train_df, test_df)

# 内存优化函数
def reduce_mem_usage(df):
    """迭代所有列的数据类型以节省内存"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"内存使用量: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"优化后内存使用量: {end_mem:.2f} MB")
    print(f"减少 {100 * (start_mem - end_mem) / start_mem:.1f}%")
    return df

# 应用内存优化
print("\n内存优化:")
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)

# 详细的数据探索
print("\n目标变量分布:")
target_counts = train_df['TARGET'].value_counts()
print(target_counts)
print(f"不满意客户比例: {train_df['TARGET'].mean() * 100:.6f}%")

# =============================================================================
# 2. 修复的特征工程 - 避免对齐问题
# =============================================================================
print("\n" + "=" * 60)
print("2. 修复的特征工程")
print("=" * 60)

# 分离特征和目标变量
X = train_df.drop(['TARGET', 'ID'], axis=1)
y = train_df['TARGET']
test_ids = test_df['ID']
test_data = test_df.drop('ID', axis=1)

print(f"基础特征数量: {X.shape[1]}")

# 修复的特征工程函数 - 确保训练和测试集使用相同的列
def create_safe_features(df, is_train=True):
    """创建安全的统计特征，避免对齐问题"""
    print("创建安全统计特征...")
    
    df_new = df.copy()
    
    # 基础统计特征
    df_new['var'] = df.var(axis=1)
    df_new['mean'] = df.mean(axis=1)
    df_new['std'] = df.std(axis=1)
    df_new['max'] = df.max(axis=1)
    df_new['min'] = df.min(axis=1)
    df_new['median'] = df.median(axis=1)
    
    # 简单的范围特征
    df_new['range'] = df_new['max'] - df_new['min']
    
    # 特殊值计数
    df_new['zeros_count'] = (df == 0).sum(axis=1)
    
    print(f"创建了 {len([col for col in df_new.columns if col not in df.columns])} 个新特征")
    return df_new

# 分别应用特征工程（避免对齐问题）
print("为训练集创建特征...")
X_enhanced = create_safe_features(X, is_train=True)

print("为测试集创建特征...")
test_enhanced = create_safe_features(test_data, is_train=False)

# 手动确保列一致
common_cols = list(set(X_enhanced.columns) & set(test_enhanced.columns))
X_enhanced = X_enhanced[common_cols]
test_enhanced = test_enhanced[common_cols]

print(f"最终特征数量: {X_enhanced.shape[1]}")

# =============================================================================
# 3. 简化的特征选择
# =============================================================================
print("\n" + "=" * 60)
print("3. 简化的特征选择")
print("=" * 60)

# 使用随机森林进行特征选择
print("进行特征选择...")

# 使用训练数据计算特征重要性
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_enhanced, y)

# 选择重要性大于平均值的特征
importances = rf_selector.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_enhanced.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# 选择前150个重要特征
selected_features = feature_importance_df.head(150)['feature'].tolist()
print(f"选择前 {len(selected_features)} 个重要特征")

# 选择特征
X_selected = X_enhanced[selected_features]
test_selected = test_enhanced[selected_features]

print(f"最终特征数量: {X_selected.shape[1]}")

# =============================================================================
# 4. 数据预处理
# =============================================================================
print("\n" + "=" * 60)
print("4. 数据预处理")
print("=" * 60)

# 使用RobustScaler
scaler = RobustScaler()
X_processed = scaler.fit_transform(X_selected)
test_processed = scaler.transform(test_selected)

print("数据预处理完成")

# =============================================================================
# 5. 简化的模型训练
# =============================================================================
print("\n" + "=" * 60)
print("5. 简化的模型训练")
print("=" * 60)

# 计算类别权重
scale_pos_weight = len(y[y==0]) / len(y[y==1])
print(f"类别权重比例: {scale_pos_weight:.2f}")

# 使用分层K折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 定义简化的模型集合
models = {
    'XGBoost': XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        tree_method='hist'
    ),
    'LightGBM': LGBMClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        verbose=-1
    ),
    'RandomForest': RandomForestClassifier(
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )
}

# 简化的参数网格
param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'max_depth': [6, 8],
        'learning_rate': [0.05, 0.1]
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 15]
    }
}

# 存储结果
best_models = {}
cv_results = {}

print("开始模型训练...")

for name, model in models.items():
    print(f"\n训练 {name}...")
    
    try:
        if name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[name], 
                cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_processed, y)
            
            best_models[name] = grid_search.best_estimator_
            cv_results[name] = grid_search.best_score_
            
            print(f"最佳CV分数: {grid_search.best_score_:.6f}")
        else:
            cv_scores = cross_val_score(model, X_processed, y, cv=skf, scoring='roc_auc', n_jobs=-1)
            model.fit(X_processed, y)
            best_models[name] = model
            cv_results[name] = np.mean(cv_scores)
            print(f"CV分数: {np.mean(cv_scores):.6f}")
            
    except Exception as e:
        print(f"训练 {name} 时出错: {e}")
        # 使用默认参数
        try:
            model.fit(X_processed, y)
            best_models[name] = model
            cv_score = cross_val_score(model, X_processed, y, cv=3, scoring='roc_auc', n_jobs=-1).mean()
            cv_results[name] = cv_score
            print(f"默认参数CV分数: {cv_score:.6f}")
        except Exception as e2:
            print(f"默认参数训练失败: {e2}")

# 显示结果
print("\n模型性能比较:")
for name, score in sorted(cv_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.6f}")

# =============================================================================
# 6. 模型集成
# =============================================================================
print("\n" + "=" * 60)
print("6. 模型集成")
print("=" * 60)

# 选择最佳模型
best_model_name = max(cv_results, key=cv_results.get)
best_model = best_models[best_model_name]
print(f"最佳模型: {best_model_name}, 分数: {cv_results[best_model_name]:.6f}")

# 创建简单的加权平均集成
print("创建加权平均集成...")

# 收集所有模型的预测概率
all_predictions = []
weights = []

for name, model in best_models.items():
    try:
        preds = model.predict_proba(X_processed)
        if preds.shape[1] == 2:
            preds = preds[:, 1]
        all_predictions.append(preds)
        weights.append(cv_results[name])
    except Exception as e:
        print(f"获取 {name} 预测时出错: {e}")

# 计算加权平均（在训练集上）
if all_predictions:
    weights = np.array(weights) / sum(weights)
    weighted_train_pred = np.zeros_like(all_predictions[0])
    for i, pred in enumerate(all_predictions):
        weighted_train_pred += weights[i] * pred
    
    # 计算集成模型的分数
    ensemble_score = roc_auc_score(y, weighted_train_pred)
    print(f"加权平均集成分数: {ensemble_score:.6f}")
    
    # 如果集成效果更好，使用集成
    if ensemble_score > cv_results[best_model_name]:
        print("使用加权平均集成")
        use_ensemble = True
    else:
        print("使用最佳单个模型")
        use_ensemble = False
else:
    use_ensemble = False

# =============================================================================
# 7. 生成预测
# =============================================================================
print("\n" + "=" * 60)
print("7. 生成预测")
print("=" * 60)

def generate_predictions(model, test_data, model_name):
    """安全地生成预测"""
    try:
        preds = model.predict_proba(test_data)
        if preds.shape[1] == 2:
            preds = preds[:, 1]
        return preds
    except Exception as e:
        print(f"模型 {model_name} 预测时出错: {e}")
        return None

# 生成最终预测
if use_ensemble:
    print("生成集成预测...")
    final_predictions = np.zeros(test_processed.shape[0])
    total_weight = 0
    
    for name, model in best_models.items():
        preds = generate_predictions(model, test_processed, name)
        if preds is not None:
            weight = cv_results[name]
            final_predictions += weight * preds
            total_weight += weight
    
    if total_weight > 0:
        final_predictions /= total_weight
    else:
        # 如果所有模型都失败，使用最佳模型
        print("所有模型预测失败，使用最佳模型")
        final_predictions = generate_predictions(best_model, test_processed, best_model_name)
else:
    print(f"使用 {best_model_name} 生成预测...")
    final_predictions = generate_predictions(best_model, test_processed, best_model_name)

# 确保预测结果有效
if final_predictions is None or len(final_predictions) != test_processed.shape[0]:
    print("预测生成失败，使用随机预测")
    final_predictions = np.random.uniform(0, 0.1, test_processed.shape[0])

# 创建提交文件
submission = pd.DataFrame({
    'ID': test_ids,
    'TARGET': final_predictions
})

# 保存提交文件
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("提交文件已保存为 '/kaggle/working/submission.csv'")

# =============================================================================
# 8. 结果分析
# =============================================================================
print("\n" + "=" * 60)
print("8. 结果分析")
print("=" * 60)

print("预测统计:")
print(f"平均不满意概率: {final_predictions.mean():.6f}")
print(f"预测概率范围: [{final_predictions.min():.6f}, {final_predictions.max():.6f}]")
print(f"预测概率中位数: {np.median(final_predictions):.6f}")

# 简单的可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
# 模型性能比较
models_sorted = sorted(cv_results.items(), key=lambda x: x[1], reverse=True)
model_names = [name for name, _ in models_sorted]
scores = [score for _, score in models_sorted]

plt.bar(model_names, scores, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Model Performance Comparison')
plt.ylabel('CV AUC Score')
plt.xticks(rotation=45)
for i, v in enumerate(scores):
    plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
# 预测分布
plt.hist(final_predictions, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
plt.title('Test Predictions Distribution')
plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('/kaggle/working/results_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. 最终报告
# =============================================================================
print("\n" + "=" * 60)
print("9. 最终报告")
print("=" * 60)

print("项目完成!")
print(f"最佳模型: {best_model_name}")
print(f"最佳CV分数: {cv_results[best_model_name]:.6f}")
print(f"测试集平均预测: {final_predictions.mean():.6f}")
print(f"提交文件: /kaggle/working/submission.csv")

print("\n如果仍有问题，建议:")
print("1. 检查数据文件路径是否正确")
print("2. 确保所有必需的库已安装")
print("3. 重启kernel并重新运行")
print("4. 在Kaggle环境中直接使用官方notebook")

print("\n代码执行完毕!")

