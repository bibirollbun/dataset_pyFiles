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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_pinball_loss
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据获取
train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')

# 2. 数据探索性分析
print("训练集形状:", train.shape)
print("测试集形状:", test.shape)

# 缺失值分析
print("\n缺失值分析:")
missing_train = train.isna().sum()
missing_test = test.isna().sum()
print("训练集缺失值统计:")
print(missing_train[missing_train > 0])
print("\n测试集缺失值统计:")
print(missing_test[missing_test > 0])

# 重复值分析
print("\n重复值分析:")
print(f"训练集重复行数量: {train.duplicated().sum()}")
print(f"测试集重复行数量: {test.duplicated().sum()}")


# 目标变量分析
print("\n目标变量分析 (sale_price):")
print(train['sale_price'].describe())
plt.figure(figsize=(10, 6))
sns.histplot(train['sale_price'], kde=True)
plt.title('Sale Price Distribution')
plt.show()

# 数值型变量分析
num_cols = train.select_dtypes(include=['float64', 'int64']).columns
print("\n数值型变量分析:")
for col in num_cols:
    if col != 'sale_price':  # 跳过目标变量
        # 绘制分布图
        plt.figure(figsize=(8, 4))
        sns.histplot(train[col], kde=True)
        plt.title(f'{col} Distribution')
        plt.show()

# 类别型变量分析
cat_cols = train.select_dtypes(include='object').columns
print("\n类别型变量分析:")
for col in cat_cols:
    # 绘制计数图
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=train)
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=45)
    plt.show()


# 3. 数据预处理
def preprocess_data(df):
    df = df.copy()
    
    # 处理日期特征
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['sale_year'] = df['sale_date'].dt.year
        df['sale_month'] = df['sale_date'].dt.month
        df['sale_day'] = df['sale_date'].dt.day
        df['sale_dayofweek'] = df['sale_date'].dt.dayofweek
        df['sale_season'] = df['sale_month'] % 12 // 3 + 1  # 添加季节特征
        df = df.drop('sale_date', axis=1)

    # 处理所有分类特征：进行频率编码
    cat_cols = ['sale_warning', 'join_status', 'city', 'zoning', 'subdivision', 'submarket']
    for col in cat_cols:
        if col in df.columns:
            # 使用频率编码替代LabelEncoder
            freq_encoding = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq_encoding)
            df[col].fillna(0, inplace=True)  # 处理新类别
    
    # 特定缺失值处理
    for col in ['subdivision', 'submarket', 'sale_nbr']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        
    # 计算总价值
    if all(col in df.columns for col in ['land_val', 'imp_val']):
        df['total_val'] = df['land_val'] + df['imp_val']
    
    # 计算房屋年龄
    if 'join_year' in df.columns and 'sale_year' in df.columns:
        df['year_gap'] = df['join_year'] - df['sale_year']
    if 'sale_year' in df.columns and 'year_built' in df.columns:
        df['year_diff'] = df['sale_year'] - df['year_built']

    # 创建总景观评分
    view_cols = [c for c in df.columns if 'view_' in c]
    if view_cols:
        df['total_view_score'] = df[view_cols].sum(axis=1)

    # 创建总平方英尺
    sqft_cols = [c for c in df.columns if 'sqft_' in c]
    if sqft_cols:
        df['total_sqft'] = df[sqft_cols].sum(axis=1)
    
    return df

print("\n数据预处理...")
train = preprocess_data(train).drop_duplicates().reset_index(drop=True)
test = preprocess_data(test)


# 4. 特征工程
# 分离特征和目标
X = train.drop(['sale_price'], axis=1, errors='ignore')
y = train['sale_price']
X_test = test.drop(['sale_price'], axis=1, errors='ignore')


# 5. 数据变换
# 对数值特征进行对数变换（避免0值）
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    if X[col].min() > 0:  # 确保所有值大于0
        X[col] = np.log1p(X[col])
        X_test[col] = np.log1p(X_test[col])

# 对目标变量进行对数变换
y = np.log1p(y)


# 6. 划分数据集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7. 模型训练
# Winkler评分函数
def winkler_score(y_true, lower, upper, alpha=0.1):
    width = upper - lower
    penalty = np.zeros_like(y_true)
    
    below = y_true < lower
    penalty[below] = (2 / alpha) * (lower[below] - y_true[below])
    
    above = y_true > upper
    penalty[above] = (2 / alpha) * (y_true[above] - upper[above])
    
    return np.mean(width + penalty)

# 创建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# 分位数回归参数
params = {
    'objective': 'quantile',
    'metric': 'quantile',
    'alpha': 0.05,
    'num_leaves': 50,
    'max_depth': 8,
    'min_data_in_leaf': 30,
    'lambda_l1': 0.15,
    'lambda_l2': 0.15,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'learning_rate': 0.01,
    'num_iterations': 3000,
    'quantile_dalpha': 0.2,
    'path_smooth': 100,
    'verbose': -1,
    'seed': 42
}

# 训练5%分位数模型
print("\n训练5%分位数模型...")
model_lower = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False)
    ]
)

# 更新参数为95%分位数
params['alpha'] = 0.95

# 训练95%分位数模型
print("\n训练95%分位数模型...")
model_upper = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False)
    ]
)



# 8. 模型评估
# 验证集预测
lower_val = model_lower.predict(X_val)
upper_val = model_upper.predict(X_val)

# 转换回原始空间
lower_val = np.expm1(lower_val)
upper_val = np.expm1(upper_val)
y_val_orig = np.expm1(y_val)

# 计算验证集Winkler分数
winkler_val = winkler_score(y_val_orig, lower_val, upper_val)
print(f"\nValidation Winkler Score (α=0.1): {winkler_val:.2f}")

# 计算覆盖率
coverage = np.mean((y_val_orig >= lower_val) & (y_val_orig <= upper_val)) * 100
print(f"Coverage: {coverage:.2f}%")

# 绘制预测区间可视化
plt.figure(figsize=(12, 6))
sample_size = min(100, len(y_val_orig))
indices = np.random.choice(len(y_val_orig), sample_size, replace=False)
sorted_idx = np.argsort(y_val_orig.values[indices])

plt.plot(y_val_orig.values[indices][sorted_idx], 'o-', label='Actual Price')
plt.plot(lower_val[indices][sorted_idx], 'r--', label='Lower Bound')
plt.plot(upper_val[indices][sorted_idx], 'g--', label='Upper Bound')
plt.fill_between(range(sample_size), 
                 lower_val[indices][sorted_idx], 
                 upper_val[indices][sorted_idx], 
                 color='gray', alpha=0.3)
plt.title('Prediction Intervals vs Actual Prices')
plt.xlabel('Sample Index')
plt.ylabel('Sale Price')
plt.legend()
plt.show()



# 9. 预测与提交
# 测试集预测
lower_test = model_lower.predict(X_test)
upper_test = model_upper.predict(X_test)

# 转换回原始空间
lower_test = np.expm1(lower_test)
upper_test = np.expm1(upper_test)

# 创建提交文件
submission = pd.DataFrame({
    'id': test['id'],
    'pi_lower': lower_test,
    'pi_upper': upper_test
})

# 保存结果
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully!")

