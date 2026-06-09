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


# ======================
# 导入必要的库
# ======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 设置图形样式
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# ======================
# 1. 数据分析
# ======================

# 读取数据
train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print("训练集形状:", train_data.shape)
print("测试集形状:", test_data.shape)

# 检查目标变量
print("\n目标变量分布:")
print(train_data['Personality'].value_counts())

# 检查是否有NaN值
print("\n训练集缺失值统计:")
print(train_data.isnull().sum())
print("\n测试集缺失值统计:")
print(test_data.isnull().sum())

# 目标变量分布可视化
plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=train_data)
plt.title('目标变量分布')
plt.xlabel('性格类型')
plt.ylabel('数量')
plt.tight_layout()
plt.show()

# ======================
# 2. 数据预处理
# ======================

# 复制数据
train_processed = train_data.copy()
test_processed = test_data.copy()

# 处理分类特征编码
categorical_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']

# 创建编码器字典
label_encoders = {}

# 对训练集和测试集中的分类特征进行编码
for col in categorical_cols:
    if col in train_processed.columns:
        if train_processed[col].dtype == 'object':
            le = LabelEncoder()
            # 只对训练集进行fit，然后应用到训练集和测试集
            if col == 'Personality':
                train_processed[col] = le.fit_transform(train_processed[col])
            else:
                # 对于Stage_fear和Drained_after_socializing，处理缺失值后再编码
                train_processed[col] = le.fit_transform(train_processed[col].fillna('0'))
                test_processed[col] = le.transform(test_processed[col].fillna('0'))
            
            label_encoders[col] = le
            print(f"已编码特征 {col}，类别: {le.classes_}")

# 处理数值特征的缺失值
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

for col in numeric_cols:
    if col in train_processed.columns:
        # 用中位数填充训练集
        median_val = train_processed[col].median()
        train_processed[col].fillna(median_val, inplace=True)
        
        # 用训练集中位数填充测试集
        test_processed[col].fillna(median_val, inplace=True)

print("缺失值处理完成")

# 处理异常值 - 使用IQR方法
def handle_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in ['int64', 'float64']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 将异常值缩放到边界值
            outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            if outliers_count > 0:
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    return df_clean

# 处理异常值
train_processed = handle_outliers_iqr(train_processed, numeric_cols + ['Stage_fear', 'Drained_after_socializing'])
test_processed = handle_outliers_iqr(test_processed, numeric_cols + ['Stage_fear', 'Drained_after_socializing'])

print("数据预处理完成")

# ======================
# 3. 特征工程
# ======================

# 特征组合：社交倾向指数
train_processed['Social_tendency_index'] = train_processed['Social_event_attendance'] / (train_processed['Time_spent_Alone'] + 1)
test_processed['Social_tendency_index'] = test_processed['Social_event_attendance'] / (test_processed['Time_spent_Alone'] + 1)

# 特征交互：社交疲劳交互项
train_processed['Social_fatigue_interaction'] = train_processed['Stage_fear'] * train_processed['Drained_after_socializing']
test_processed['Social_fatigue_interaction'] = test_processed['Stage_fear'] * test_processed['Drained_after_socializing']

print("特征工程完成")

# ======================
# 4. 数据标准化
# ======================

# 分离特征和目标变量
X = train_processed.drop(['Personality', 'id'], axis=1)
y = train_processed['Personality']

# 保存测试集ID
test_ids = test_processed['id']
X_test = test_processed.drop(['id'], axis=1)

# 确保列顺序一致
X_test = X_test[X.columns]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集形状: {X_train.shape}")
print(f"验证集形状: {X_val.shape}")

# ======================
# 5. 构建模型进行预测
# ======================

# 随机森林模型
print("训练随机森林模型中...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_val_pred = rf_model.predict(X_val)

# XGBoost模型
print("训练XGBoost模型中...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_val_pred = xgb_model.predict(X_val)

# 模型评估
print("\n随机森林模型性能:")
print(classification_report(y_val, rf_val_pred))
print(f"随机森林准确率: {accuracy_score(y_val, rf_val_pred):.4f}")

print("\nXGBoost模型性能:")
print(classification_report(y_val, xgb_val_pred))
print(f"XGBoost准确率: {accuracy_score(y_val, xgb_val_pred):.4f}")

# 在测试集上进行预测
rf_test_pred = rf_model.predict(X_test_scaled)
xgb_test_pred = xgb_model.predict(X_test_scaled)

# 确保预测值是二进制0/1
print(f"\n随机森林预测值分布:")
unique_rf, counts_rf = np.unique(rf_test_pred, return_counts=True)
for val, count in zip(unique_rf, counts_rf):
    print(f"类别 {val}: {count}")

print(f"\nXGBoost预测值分布:")
unique_xgb, counts_xgb = np.unique(xgb_test_pred, return_counts=True)
for val, count in zip(unique_xgb, counts_xgb):
    print(f"类别 {val}: {count}")

# ======================
# 6. 预测结果可视化
# ======================

# 混淆矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

cm_rf = confusion_matrix(y_val, rf_val_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('随机森林混淆矩阵')

cm_xgb = confusion_matrix(y_val, xgb_val_pred)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('XGBoost混淆矩阵')

plt.tight_layout()
plt.show()

# 特征重要性
feature_names = X.columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

rf_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

sns.barplot(x='importance', y='feature', data=rf_importance, ax=ax1)
ax1.set_title('随机森林特征重要性 Top 10')

xgb_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

sns.barplot(x='importance', y='feature', data=xgb_importance, ax=ax2)
ax2.set_title('XGBoost特征重要性 Top 10')

plt.tight_layout()
plt.show()

# ======================
# 7. 生成提交文件（关键修复部分）
# ======================

# 确保预测值是整数类型（0或1）
xgb_test_pred_int = xgb_test_pred.astype(int)

# 创建提交文件
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': xgb_test_pred_int
})

# 验证提交文件的格式
print("\n提交文件信息:")
print(submission.info())
print("\n提交文件前5行:")
print(submission.head())
print("\n提交文件Personality列的值分布:")
print(submission['Personality'].value_counts())

# 检查是否有NaN值
print(f"\n提交文件中的NaN值数量: {submission.isnull().sum().sum()}")

# 确保Personality列只有0和1
if set(submission['Personality'].unique()).issubset({0, 1}):
    print("✓ Personality列只包含0和1，格式正确")
else:
    print("✗ Personality列包含其他值，需要修正")
    # 如果预测值不是0/1，将其转换为0/1
    submission['Personality'] = submission['Personality'].apply(lambda x: 1 if x > 0.5 else 0)

# 保存提交文件
submission_file_path = 'submission.csv'
submission.to_csv(submission_file_path, index=False)

# 验证文件大小
import os
file_size = os.path.getsize(submission_file_path)
print(f"\n提交文件大小: {file_size / 1024:.2f} KiB")

# 读取保存的文件验证内容
submission_check = pd.read_csv(submission_file_path)
print("\n验证读取的提交文件:")
print(f"行数: {len(submission_check)}")
print(f"列名: {list(submission_check.columns)}")
print(f"Personality唯一值: {sorted(submission_check['Personality'].unique())}")

print("\n=== 提交文件已生成 ===")
print(f"文件路径: {submission_file_path}")
print("请将此文件上传到Kaggle比赛页面")

