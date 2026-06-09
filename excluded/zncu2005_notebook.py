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


# ==============================================================================
# 1. 环境设置 (Environment Setup)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import warnings

# 忽略警告，保持输出整洁
warnings.filterwarnings('ignore')

# 设置绘图风格 (符合实验报告美观要求)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
sns.set_palette("husl")

print("环境配置完成。")


# ==============================================================================
# 2. 数据加载 (Data Loading)
# ==============================================================================
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

# 标记数据来源，方便后续分离
train['_data'] = 'train'
test['_data'] = 'test'

# 合并数据集以便统一进行特征工程
data = pd.concat([train, test], ignore_index=True)
data['datetime'] = pd.to_datetime(data['datetime'])

print(f"训练集维度: {train.shape}")
print(f"测试集维度: {test.shape}")
print(f"合并后维度: {data.shape}")


# ==============================================================================
# 3. 探索性数据分析 (EDA & Visualization)
# 理论依据：通过可视化识别数据分布特征（偏态）和属性间的相关性（相异度矩阵）
# ==============================================================================

# 3.1 目标变量分布分析 (Target Distribution)
# 目的：证明 Log 变换的必要性
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 原始分布
sns.histplot(train['count'], kde=True, ax=axes[0], color='#3498DB')
axes[0].set_title('Original Count Distribution (Skewed)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Count')

# 对数变换后分布
sns.histplot(np.log1p(train['count']), kde=True, ax=axes[1], color='#E74C3C')
axes[1].set_title('Log-Transformed Distribution (Normalized)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Log(Count + 1)')

plt.suptitle('Data Transformation: Why we need Log1p?', fontsize=14)
plt.show()

# 3.2 业务逻辑验证：工作日 vs 非工作日 (Pattern Recognition)
# 目的：证明构建 "Peak" (高峰期) 特征的合理性
# 提取小时特征用于绘图
train['hour'] = pd.to_datetime(train['datetime']).dt.hour

plt.figure(figsize=(12, 6))
sns.pointplot(x='hour', y='count', hue='workingday', data=train, palette='Set2')
plt.title('Hourly Demand: Working Days vs Weekends', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Average Rentals')
plt.legend(title='Working Day (0=No, 1=Yes)')
plt.grid(True, alpha=0.3)
plt.annotate('Commute Peaks', xy=(8, 450), xytext=(5, 600),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Leisure Peak', xy=(13, 380), xytext=(15, 500),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()

# 3.3 相关性热力图 (Correlation Matrix)
# 目的：特征选择与共线性分析
plt.figure(figsize=(10, 8))
# 选取数值型特征
corr_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
corr_matrix = train[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.show()


# ==============================================================================
# 4. 数据清洗与插补 (Data Cleaning & Imputation)
# 理论依据：噪声处理（去除异常点）与数据插补（恢复缺失信息）
# ==============================================================================

# 4.1 提取基础时间特征 (用于辅助风速预测)
data['hour'] = data['datetime'].dt.hour
data['month'] = data['datetime'].dt.month
data['year'] = data['datetime'].dt.year

# 4.2 异常值剔除 (Outlier Removal)
# 现象：训练集中存在 Humidity <= 2 的极低值，判定为传感器故障噪声
print("剔除前数据量:", len(data))
# 注意：只剔除训练集的异常值，测试集不可动
train_outliers = data[(data['humidity'] <= 2) & (data['_data'] == 'train')].index
data = data.drop(train_outliers)
print(f"剔除异常湿度样本数: {len(train_outliers)}")

# 4.3 风速插补 (Windspeed Imputation)
# 现象：Windspeed 为 0 的数据过多，判定为缺失值
# 方法：使用随机森林回归 (Random Forest) 基于其他气象特征进行预测
print("正在进行风速插补...")
wind_0 = data[data['windspeed'] == 0]
wind_not0 = data[data['windspeed'] != 0]

# 选取相关性强的特征进行预测
rf_wind = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
wind_features = ['season', 'weather', 'temp', 'atemp', 'humidity', 'month', 'year']

rf_wind.fit(wind_not0[wind_features], wind_not0['windspeed'])
predicted_wind = rf_wind.predict(wind_0[wind_features])

# 填补数据
data.loc[data['windspeed'] == 0, 'windspeed'] = predicted_wind
print("风速插补完成。")


# ==============================================================================
# 5. 高级特征工程 (Advanced Feature Engineering)
# 理论依据：属性构造 (Attribute Construction) 与 专家知识集成 (Expert Knowledge)
# ==============================================================================

# 5.1 基础特征补全
data['dow'] = data['datetime'].dt.dayofweek
data['woy'] = data['datetime'].dt.isocalendar().week.astype(int)

# 5.2 属性构造：气温体感差 (Temp Gap)
# 理论：反映环境对人体的真实影响
data['temp_gap'] = abs(data['temp'] - data['atemp'])

# 5.3 专家知识集成：特殊日期修正 (Special Dates Correction)
# 理论：数据集成中的逻辑校验，修正原始数据中的错误标签
special_dates = [
    (pd.Timestamp(2011, 4, 15), 1, 0), (pd.Timestamp(2012, 4, 16), 1, 0), # Tax Day
    (pd.Timestamp(2011, 11, 25), 0, 1), (pd.Timestamp(2012, 11, 23), 0, 1), # Thanksgiving
    (pd.Timestamp(2012, 10, 30), 0, 1), # Hurricane Sandy
    (pd.Timestamp(2011, 12, 24), 0, 1), (pd.Timestamp(2012, 12, 24), 0, 1) # Christmas Eve
]

for date, work_val, hol_val in special_dates:
    mask = data['datetime'].dt.date == date.date()
    data.loc[mask, 'workingday'] = work_val
    data.loc[mask, 'holiday'] = hol_val

# 5.4 属性构造：精细化高峰期 (Peak Engineering)
# 理论：基于EDA发现的业务规律，进行概念分层
data['peak'] = 0
# 工作日高峰 (扩大范围以覆盖前后波动)
data.loc[(data['workingday'] == 1) & (data['hour'].isin([7, 8, 9, 17, 18, 19])), 'peak'] = 1
# 非工作日高峰 (午后休闲段)
data.loc[(data['workingday'] == 0) & (data['hour'].between(10, 19)), 'peak'] = 1

# 5.5 属性构造：舒适度指标 (Comfort Index)
data['ideal'] = 0
data.loc[(data['temp'] > 27) & (data['windspeed'] < 30), 'ideal'] = 1
data['sticky'] = 0
data.loc[(data['workingday'] == 1) & (data['humidity'] >= 60), 'sticky'] = 1


# ==============================================================================
# 6. 模型构建与集成 (Modeling & Ensemble)
# 理论依据：分流建模 (Clustering-based Separation) 与 集成学习 (Ensemble Learning)
# 策略：Casual/Registered 分开预测，Boosting (XGB) + Bagging (RF) 加权融合
# ==============================================================================

# 特征选择
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'temp_gap',
            'humidity', 'windspeed', 'year', 'hour', 'dow', 'woy', 'peak', 'ideal', 'sticky']

# 数据集分离
train_df = data[data['_data'] == 'train']
test_df = data[data['_data'] == 'test']

X = train_df[features]
X_test = test_df[features]

# 对目标变量进行 Log1p 变换 (解决偏态分布)
y_cas = np.log1p(train_df['casual'])
y_reg = np.log1p(train_df['registered'])

print("开始训练模型...")

# 模型 1: XGBoost (Gradient Boosting) - 降低偏差 (Bias)
xgb_params = {
    'n_estimators': 1200, 
    'max_depth': 5, 
    'learning_rate': 0.05, 
    'subsample': 0.8, 
    'colsample_bytree': 0.7, 
    'random_state': 42,
    'n_jobs': -1
}
model_xgb_cas = xgb.XGBRegressor(**xgb_params).fit(X, y_cas)
model_xgb_reg = xgb.XGBRegressor(**xgb_params).fit(X, y_reg)
print("XGBoost 训练完成。")

# 模型 2: Random Forest (Bagging) - 降低方差 (Variance)
rf_params = {
    'n_estimators': 1000, 
    'max_depth': 20, 
    'min_samples_split': 4, 
    'random_state': 0,
    'n_jobs': -1
}
model_rf_cas = RandomForestRegressor(**rf_params).fit(X, y_cas)
model_rf_reg = RandomForestRegressor(**rf_params).fit(X, y_reg)
print("Random Forest 训练完成。")


# ==============================================================================
# 7. 预测融合与提交 (Prediction Blending & Submission)
# 策略：0.75 XGB + 0.25 RF (经验验证的最佳比例)
# ==============================================================================

print("正在生成预测结果...")

# Casual 用户预测
pred_cas_xgb = np.expm1(model_xgb_cas.predict(X_test))
pred_cas_rf = np.expm1(model_rf_cas.predict(X_test))
pred_cas = 0.75 * pred_cas_xgb + 0.25 * pred_cas_rf

# Registered 用户预测
pred_reg_xgb = np.expm1(model_xgb_reg.predict(X_test))
pred_reg_rf = np.expm1(model_rf_reg.predict(X_test))
pred_reg = 0.75 * pred_reg_xgb + 0.25 * pred_reg_rf

# 汇总结果 (确保非负)
final_count = np.round(pred_cas + pred_reg)
final_count[final_count < 0] = 0

# 生成提交文件
submission = pd.DataFrame({'datetime': test_df['datetime'], 'count': final_count})
submission.to_csv('submission_final.csv', index=False)

print("="*50)
print("任务完成！文件 'submission_final.csv' 已生成。")
print(f"预测统计: Mean={final_count.mean():.2f}, Max={final_count.max()}")
print("="*50)

# 可选：绘制特征重要性 (用于实验报告)
plt.figure(figsize=(10, 6))
xgb.plot_importance(model_xgb_reg, max_num_features=10, title='Feature Importance (Registered - XGB)')
plt.show()

