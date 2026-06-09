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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

class CFG:
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e7/sample_submission.csv'
    target = 'Personality'
    n_folds = 5
    seed = 42

# 数据读取
train = pd.read_csv(CFG.train_path, index_col='id')
test = pd.read_csv(CFG.test_path, index_col='id')

print("数据集形状:")
print(f"训练集: {train.shape}")
print(f"测试集: {test.shape}")
print("\n训练集基本信息:")
print(train.info())
print("\n测试集基本信息:")
print(test.info())

# 数据分析 - 数据可视化
plt.figure(figsize=(15, 10))

# 1. 目标变量分布
plt.subplot(2, 3, 1)
train[CFG.target].value_counts().plot(kind='bar')
plt.title('目标变量分布')
plt.xlabel('Personality')
plt.ylabel('数量')

# 2. 数值特征分布
plt.subplot(2, 3, 2)
sns.histplot(train['Time_spent_Alone'], kde=True)
plt.title('独处时间分布')

plt.subplot(2, 3, 3)
sns.histplot(train['Social_event_attendance'], kde=True)
plt.title('社交活动参与度分布')

# 3. 相关性热图（7个主要特征）- 已修复
plt.subplot(2, 3, 4)
# 只选择7个数值特征，不包含目标列
selected_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                    'Friends_circle_size', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']
# 创建副本并转换分类特征为数值用于热图显示
temp_df = train[selected_features].copy()
# 将分类特征映射为数值
temp_df['Stage_fear'] = temp_df['Stage_fear'].map({'No': 0, 'Yes': 1})
temp_df['Drained_after_socializing'] = temp_df['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
corr_matrix = temp_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('7个特征的相关性热图')

# 4. 分类特征分析
plt.subplot(2, 3, 5)
train['Stage_fear'].value_counts().plot(kind='bar')
plt.title('舞台恐惧分布')

plt.subplot(2, 3, 6)
train['Drained_after_socializing'].value_counts().plot(kind='bar')
plt.title('社交后疲惫分布')

plt.tight_layout()
plt.show()

# 数据预处理
def preprocess_data(train_df, test_df):
    # 复制数据避免修改原始数据
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # 分类特征编码
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in categorical_cols:
        train_processed[col] = train_processed[col].map({'No': 0, 'Yes': 1})
        test_processed[col] = test_processed[col].map({'No': 0, 'Yes': 1})
    
    # 目标变量编码
    if CFG.target in train_processed.columns:
        train_processed[CFG.target] = train_processed[CFG.target].map({'Extrovert': 0, 'Introvert': 1})
    
    # 异常值处理 - 使用IQR方法
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency']
    
    for col in numeric_cols:
        Q1 = train_processed[col].quantile(0.25)
        Q3 = train_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 将异常值替换为边界值
        train_processed[col] = np.clip(train_processed[col], lower_bound, upper_bound)
        test_processed[col] = np.clip(test_processed[col], lower_bound, upper_bound)
    
    # 缺失值处理
    for col in train_processed.columns:
        if col == CFG.target:
            continue
            
        if col in categorical_cols:
            # 分类特征用众数填充
            mode_val = train_processed[col].mode()[0]
            train_processed[col].fillna(mode_val, inplace=True)
            test_processed[col].fillna(mode_val, inplace=True)
        else:
            # 数值特征用中位数填充
            median_val = train_processed[col].median()
            train_processed[col].fillna(median_val, inplace=True)
            test_processed[col].fillna(median_val, inplace=True)
    
    # 高级特征工程
    # 特征组合：社交倾向指数
    train_processed['social_tendency_index'] = train_processed['Social_event_attendance'] / (train_processed['Time_spent_Alone'] + 1)
    test_processed['social_tendency_index'] = test_processed['Social_event_attendance'] / (test_processed['Time_spent_Alone'] + 1)
    
    # 特征交互：社交疲劳交互项
    train_processed['social_fatigue_interaction'] = train_processed['Stage_fear'] * train_processed['Drained_after_socializing']
    test_processed['social_fatigue_interaction'] = test_processed['Stage_fear'] * test_processed['Drained_after_socializing']
    
    return train_processed, test_processed

# 执行数据预处理
train_processed, test_processed = preprocess_data(train, test)

print("预处理后的特征:")
print(train_processed.columns.tolist())

# 准备训练数据
X = train_processed.drop(CFG.target, axis=1)
y = train_processed[CFG.target]
X_test = test_processed

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 模型建立和训练
def train_and_predict(X, y, X_test):
    # 交叉验证设置
    cv = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    
    # 存储预测结果
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    # 特征重要性
    feature_importance = pd.DataFrame(index=X.columns)
    
    # 存储每折的ROC曲线数据 - 已修复
    fprs = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f'正在训练第 {fold+1} 折...')
        
        # 划分训练集和验证集
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 使用随机森林作为基础模型
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=CFG.seed + fold,
            n_jobs=-1
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 验证集预测
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        # 测试集预测
        test_preds += model.predict_proba(X_test_scaled)[:, 1] / CFG.n_folds
        
        # 计算AUC分数
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
        print(f'第 {fold+1} 折 AUC: {fold_auc:.6f}')
        
        # 计算ROC曲线 - 已修复
        fpr, tpr, _ = roc_curve(y_val, val_preds)
        
        # 对tpr进行插值，使其与mean_fpr长度一致
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        
        fprs.append(mean_fpr)  # 使用统一的mean_fpr
        tprs.append(tpr_interp)  # 使用插值后的tpr
        aucs.append(fold_auc)
        
        # 记录特征重要性
        feature_importance[f'fold_{fold}'] = model.feature_importances_
    
    # 计算整体OOF AUC
    oof_auc = roc_auc_score(y, oof_preds)
    print(f'\n整体OOF AUC: {oof_auc:.6f}')
    print(f'各折AUC: {fold_scores}')
    print(f'平均AUC: {np.mean(fold_scores):.6f}')
    
    # 显示特征重要性
    feature_importance['mean'] = feature_importance.mean(axis=1)
    feature_importance = feature_importance.sort_values('mean', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importance['mean'], y=feature_importance.index)
    plt.title('特征重要性')
    plt.tight_layout()
    plt.show()
    
    # 绘制ROC曲线 - 已修复
    plt.figure(figsize=(10, 8))
    
    # 绘制各折ROC曲线
    for i in range(CFG.n_folds):
        plt.plot(fprs[i], tprs[i], lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {aucs[i]:.4f})')
    
    # 绘制平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'平均 ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})', lw=2, alpha=0.8)
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='随机分类器', alpha=0.8)
    
    # 设置图形属性
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC曲线 - 交叉验证结果')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return oof_preds, test_preds, oof_auc

# 训练模型并预测
oof_preds, test_preds, oof_auc = train_and_predict(X, y, X_test)

# 生成提交文件
submission = pd.read_csv(CFG.sample_sub_path)
submission[CFG.target] = (test_preds > 0.5).astype(int)
submission[CFG.target] = submission[CFG.target].map({0: "Extrovert", 1: "Introvert"})

# 保存提交文件
submission_file = f'submission_auc_{oof_auc:.6f}.csv'
submission.to_csv(submission_file, index=False)

print(f"\n提交文件已保存: {submission_file}")
print("预测结果分布:")
print(submission[CFG.target].value_counts())

# 最终模型性能评估
if oof_auc >= 0.96:
    print(f"\n成功达到目标AUC! 最终AUC: {oof_auc:.6f}")
else:
    print(f"\n当前AUC: {oof_auc:.6f}, 接近目标值0.96")

print("\n代码执行完成!")

