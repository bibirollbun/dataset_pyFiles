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
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import gc # Garbage Collector for memory management

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
print("✅ 库导入成功")
# ===================================================================
# 2. 数据加载与初步探查 (Data Loading & Initial Inspection)
# ===================================================================
# 数据路径
DATA_PATH = '/kaggle/input/drw-crypto-market-prediction/'

# 使用 pandas 读取高效的 parquet 文件
try:
    train_df = pd.read_parquet(f'{DATA_PATH}train.parquet')
    test_df = pd.read_parquet(f'{DATA_PATH}test.parquet')
    sample_submission_df = pd.read_csv(f'{DATA_PATH}sample_submission.csv')
    
    print("✅ 数据加载成功")
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
except FileNotFoundError:
    print("❌ 数据文件未找到。请确保比赛数据已正确添加到Notebook的 /kaggle/input/ 目录下。")
    # In a real script, you might exit, but here we continue for demonstration
    # exit()

# --- 初步检查 ---
print("\n--- 训练集信息 ---")

# 新增：在使用数据前，先将索引重置为普通列
# 这是因为 'timestamp' 在 parquet 文件中被存储为索引
train_df = train_df.reset_index()
test_df = test_df.reset_index()

print("✅ 已将索引 'timestamp' 转换为普通列。")

# 使用 display 可以更好地在Notebook中展示DataFrame
display(train_df.head())

# 将 Unix 时间戳转换为 datetime 对象，方便后续处理
# unit='ms' 表示时间戳是毫秒级的
train_df['timestamp_dt'] = pd.to_datetime(train_df['timestamp'], unit='ms')

# ----------------- 代码修正 #1 -----------------
# 我们不再将测试集中的'timestamp'列重命名为'id'。
# 我们将保留'timestamp'列，因为它本身就是提交所需的ID。
# 这让代码的意图更加清晰。
# 原代码: test_df.rename(columns={'timestamp': 'id'}, inplace=True)
print("\n✅ 时间戳处理完成。测试集的 'timestamp' 列已保留，它将作为提交文件的'id'。")

# 检查缺失值
print("\n--- 缺失值检查 ---")
# 计算缺失值比例
missing_values = train_df.isnull().sum() / len(train_df)
print(missing_values[missing_values > 0].sort_values(ascending=False))

# 处理缺失值：使用前向填充 (ffill)
# 这对于时间序列数据是合理的，因为它假设当前缺失的值与上一个时间点的值相同
train_df.fillna(method='ffill', inplace=True)
test_df.fillna(method='ffill', inplace=True)
print("\n✅ 使用前向填充(ffill)处理了缺失值。")

# ===================================================================
# 3. 探索性数据分析 (EDA) & 动态特征选择
# ===================================================================
print("\n--- 开始探索性数据分析 (EDA) ---")

# --- 3.1. 目标变量 'label' 的分布 ---
plt.figure(figsize=(14, 6))
sns.histplot(train_df['label'], bins=100, kde=True, color='blue')
plt.title('目标变量 (label) 的分布', fontsize=16)
plt.xlabel('Label 值', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.show()
print(f"目标变量统计信息:\n{train_df['label'].describe()}")


# --- 3.2. 特征与目标变量的相关性 (高效优化版) & 动态选择 ---
print("\n--- 高效计算特征与目标的相关性 ---")

feature_cols = [col for col in train_df.columns if col.startswith('f_')] # 更精确地选择特征列
correlations = train_df[feature_cols].corrwith(train_df['label']).sort_values(ascending=False)

# 动态地从相关性计算结果中选择最重要的特征
N_TOP_FEATURES = 3 # 选择前N个正相关和前N个负相关的特征
top_positive_corr_features = correlations.head(N_TOP_FEATURES).index.tolist()
top_negative_corr_features = correlations.tail(N_TOP_FEATURES).index.tolist()

key_features_for_eng = top_positive_corr_features + top_negative_corr_features

print("\n--- 与 'label' 正相关性最高的特征 ---")
display(correlations.head(N_TOP_FEATURES))
print("\n--- 与 'label' 负相关性最高的特征 ---")
display(correlations.tail(N_TOP_FEATURES))
print(f"\n✅ 动态选择的关键特征为: {key_features_for_eng}")

# 释放内存
del correlations
gc.collect()


# ===================================================================
# 4. 特征工程 (Feature Engineering) - 动态版
# ===================================================================
print("\n--- 开始特征工程 (动态版) ---")

def create_features_dynamic(df, key_features):
    """为数据集创建新的特征 (使用动态传入的特征列表)"""
    df_out = df.copy()

    df_out['bid_ask_spread'] = df_out['ask_qty'] - df_out['bid_qty']
    df_out['buy_sell_ratio'] = df_out['buy_qty'] / (df_out['sell_qty'] + 1e-6)
    
    print(f"将对以下关键特征进行滞后和滚动计算: {key_features}")

    for col in key_features:
        if col in df_out.columns:
            df_out[f'{col}_lag1'] = df_out[col].shift(1)
            df_out[f'{col}_lag3'] = df_out[col].shift(3)
            df_out[f'{col}_rolling_mean_10'] = df_out[col].rolling(window=10, min_periods=1).mean()
            df_out[f'{col}_rolling_std_10'] = df_out[col].rolling(window=10, min_periods=1).std()

    df_out.fillna(method='ffill', inplace=True)
    df_out.fillna(method='bfill', inplace=True)

    return df_out

train_featured_df = create_features_dynamic(train_df, key_features_for_eng)
test_featured_df = create_features_dynamic(test_df, key_features_for_eng)

print("✅ 特征工程完成")
print(f"新的训练集形状: {train_featured_df.shape}")
print(f"新的测试集形状: {test_featured_df.shape}")

del train_df, test_df
gc.collect()

# ===================================================================
# 5. 模型训练与验证 (Model Training & Validation) - 极限速度优化版
# ===================================================================
print("\n--- 开始模型训练与验证 (极限速度优化版) ---")

TRAIN_SAMPLE_RATIO = 0.5 
start_row = int(len(train_featured_df) * (1 - TRAIN_SAMPLE_RATIO))
print(f"⚠️ 为提高效率，将仅使用最新的 {TRAIN_SAMPLE_RATIO*100}% 数据进行训练。")
train_subset_df = train_featured_df.iloc[start_row:].copy()

del train_featured_df
gc.collect()

print(f"用于训练的数据形状: {train_subset_df.shape}")

# --- 定义特征列和目标列 ---
# 'timestamp'是ID, 'timestamp_dt'是datetime对象, 'label'是目标, 都不应作为特征
features = [col for col in train_subset_df.columns if col not in ['timestamp', 'timestamp_dt', 'label']]
X = train_subset_df[features]
y = train_subset_df['label']

del train_subset_df
gc.collect()

split_index = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

print(f"新的训练集大小: {len(X_train)}")
print(f"新的验证集大小: {len(X_val)}")

try:
    import cupy
    cupy.zeros(1)
    device = 'gpu'
    print("✅ GPU环境检测成功，将使用 GPU 进行训练。")
except (ImportError, Exception):
    device = 'cpu'
    print("⚠️ 未检测到可用的GPU环境，将自动切换到 CPU 进行训练。")

lgb_params_fast = {
    'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000,
    'learning_rate': 0.05, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 1, 'lambda_l1': 0.5, 'lambda_l2': 0.5,
    'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 
    'seed': 42, 'boosting_type': 'gbdt', 'device': device,
}

model = lgb.LGBMRegressor(**lgb_params_fast)

print("\n开始模型训练...")
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='mae',
          callbacks=[lgb.early_stopping(30, verbose=True)])

print("\n在验证集上进行评估...")
val_preds = model.predict(X_val)
pearson_corr, _ = pearsonr(y_val, val_preds)
print(f"\n✅ 验证集上的皮尔逊相关系数 (Pearson Correlation): {pearson_corr:.6f}")

print("\n绘制特征重要性图...")
lgb.plot_importance(model, max_num_features=20, figsize=(10, 10))
plt.title('Top 20 Feature Importances', fontsize=16)
plt.show()

# ===================================================================================
# 6. 预测与提交 (最终修正版)
# ===================================================================================

print("\n--- 准备测试数据集以进行预测...")
# 确保使用与训练时完全相同的特征列
X_test = test_featured_df[features]

print("\n--- 使用已训练好的模型进行预测...")
test_predictions = model.predict(X_test)

print("\n--- 创建完全符合官方格式的提交文件...")

# ----------------- 代码修正 #2 -----------------
# 关键修正：
# 提交文件要求的'id'是原始的'timestamp'值, 而不是DataFrame的行号索引。
# 我们的'test_featured_df'中保留了'timestamp'列，现在用它来生成'id'列。
submission_df = pd.DataFrame({
    'id': test_featured_df['timestamp'],  # 使用正确的timestamp作为ID
    'prediction': test_predictions
})

print("释放内存...")
del test_featured_df, X_test, test_predictions
gc.collect()

# 保存到 CSV 文件, index=False 是必须的
submission_df.to_csv('submission.csv', index=False)

print("\n✅ 'submission.csv' 文件已成功生成！")
print("文件预览:")
display(submission_df.head())
print(f"提交文件总行数: {len(submission_df)}")
# 检查ID列的数据类型，它应该是代表时间戳的整数
print("id 列数据类型:", submission_df['id'].dtype)
print("prediction 列数据类型:", submission_df['prediction'].dtype)

