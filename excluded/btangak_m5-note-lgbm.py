import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')


# 配置路径
SALES_PATH = '/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv'
CALENDAR_PATH = '/kaggle/input/m5-forecasting-accuracy/calendar.csv'
SELL_PRICES_PATH = '/kaggle/input/m5-forecasting-accuracy/sell_prices.csv'
CHUNKSIZE = 5000


# Function to downcast dtypes for memory saving
def downcast(df):
    start_mem = df.memory_usage().sum() / 1024**2
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i, t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif 'object' in str(t):
            if cols[i] != 'date':
                df[cols[i]] = df[cols[i]].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is {end_mem:.2f} MB')
    print(f'Decreased by {(100 * (start_mem - end_mem) / start_mem):.1f}%')
    return df


calendar = pd.read_csv(CALENDAR_PATH)
calendar['date'] = pd.to_datetime(calendar['date'])
calendar = downcast(calendar)


sell_prices = pd.read_csv(SELL_PRICES_PATH)
sell_prices = downcast(sell_prices)


sales_iter = pd.read_csv(SALES_PATH, chunksize=CHUNKSIZE)
processed_chunks = []
for chunk in sales_iter:
    id_vars = [col for col in chunk.columns if not col.startswith('d_')]
    chunk_melted = pd.melt(chunk, id_vars=id_vars, var_name='d', value_name='sales')
    processed_chunks.append(chunk_melted)
    del chunk_melted

sales = pd.concat(processed_chunks, ignore_index=True)
del processed_chunks, chunk
gc.collect()

sales = downcast(sales)


sales = sales.merge(
    calendar[['d', 'date', 'wm_yr_wk', 'weekday', 'month', 'year', 
              'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
              'snap_CA', 'snap_TX', 'snap_WI']],
    on='d', how='left'
)

sales['snap_flag'] = 0
sales.loc[sales['state_id'] == 'CA', 'snap_flag'] = sales['snap_CA']
sales.loc[sales['state_id'] == 'TX', 'snap_flag'] = sales['snap_TX']
sales.loc[sales['state_id'] == 'WI', 'snap_flag'] = sales['snap_WI']

sales.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1, inplace=True)


LAGS = [7, 28]      
WINDOWS = [7, 28]  
FIRST = 1942        
LENGTH = 28        

# 需求特征构造函数
def demand_features(df):
    """构造滞后特征和滚动均值特征"""
    for lag in LAGS:
        # 前lag天的demand
        df[f'lag_t{lag}'] = df.groupby('id')['demand'].transform(
            lambda x: x.shift(lag)
        ).astype("float32")
        
        for w in WINDOWS:
            # 前lag天到前lag+w天的均值
            df[f'rolling_mean_lag{lag}_w{w}'] = df.groupby('id')[f'lag_t{lag}'].transform(
                lambda x: x.rolling(w).mean()
            ).astype("float32")
    return df

# 检查并准备数据
print("\n检查数据...")
print(f"当前列名: {sales.columns.tolist()}")
print(f"数据形状: {sales.shape}")

# 确保 'd' 列是数值型
if sales['d'].dtype == 'object' or sales['d'].dtype == 'category':
    print("转换 'd' 列为数值型...")
    sales['d'] = sales['d'].astype(str).str.replace('d_', '').astype(np.int16)
else:
    print(f"'d' 列已经是数值型: {sales['d'].dtype}")

# 确保 'demand' 列存在
if 'sales' in sales.columns and 'demand' not in sales.columns:
    sales = sales.rename(columns={'sales': 'demand'})
    print("已将 'sales' 列重命名为 'demand'")
sales['demand'] = sales['demand'].astype("float32")

# 构造需求特征
print("\n构造滞后特征和滚动均值特征...")
sales = demand_features(sales)

# 删除有缺失值的行
drop_d = 1000 - LENGTH 
threshold = drop_d + max(LAGS) + max(WINDOWS) 
print(f"\n删除 d <= {threshold} 的行（移除缺失值）...")
print(f"删除前数据形状: {sales.shape}")
sales = sales[sales['d'] > threshold]
print(f"删除后数据形状: {sales.shape}")

# 清理内存
gc.collect()

# 显示特征工程结果
print("\n" + "=" * 60)
print("特征工程完成")
print("=" * 60)
print(f"最终数据形状: {sales.shape}")
print(f"特征列表:")
feature_cols = [col for col in sales.columns if col.startswith('lag_') or col.startswith('rolling_')]
for col in feature_cols:
    print(f"  - {col}")

print(f"\n保留的日期范围: d > {threshold}")
print(f"数据中 d 的范围: {sales['d'].min()} 到 {sales['d'].max()}")

# 检查是否有缺失值
print(f"\n各列缺失值数量:")
missing = sales.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(missing)
else:
    print("  无缺失值")

sales.head()


sales = sales.merge(
    sell_prices, 
    how="left", 
    on=["store_id", "item_id", "wm_yr_wk"]
)

print(f"\n合并后 sales 形状: {sales.shape}")
print(f"sell_price 缺失值数量: {sales['sell_price'].isnull().sum()}")
print(f"sell_price 非空数量: {sales['sell_price'].notna().sum()}")

print(sales[['id', 'item_id', 'store_id', 'wm_yr_wk', 'sell_price']].head(10))

sales = sales.drop(["wm_yr_wk"], axis=1)

print(f"\n最终 sales 形状: {sales.shape}")
print(f"最终列名: {sales.columns.tolist()}")

gc.collect()


import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

LAGS = [7, 28]
WINDOWS = [7, 28]
FIRST = 1942       
LENGTH = 28        

print(f"\n当前数据形状: {sales.shape}")
print(f"当前列名: {sales.columns.tolist()}")

print("\n开始编码类别变量...")
categorical_cols = ["item_id", "store_id", "state_id", "dept_id", "cat_id"]

for col in categorical_cols:
    print(f"  编码 {col}...")
    encoder = OrdinalEncoder(dtype="int")
    sales[col] = encoder.fit_transform(sales[[col]]).astype("int16") + 1

print("编码完成！")
print("\n编码后样本:")
print(sales[categorical_cols].head())

# 定义特征列
print("\n定义特征列...")
exclude_cols = {'id', 'd', 'demand', 'date'}  
x = [col for col in sales.columns if col not in exclude_cols]
print(f"特征列数量: {len(x)}")
print(f"特征列: {x}")

# 准备测试集
print("\n准备测试集...")
test_threshold = FIRST - max(LAGS) - max(WINDOWS) - LENGTH
print(f"测试集阈值: d >= {test_threshold}")
test = sales[sales.d >= test_threshold].copy()
print(f"测试集形状: {test.shape}")

# 准备训练集
print("\n准备训练集...")
train_data = sales[sales.d < FIRST].copy()
print(f"训练集形状: {train_data.shape}")

# 随机划分训练集和验证集
print("\n随机划分训练集和验证集（9:1）...")
xtrain, xvalid, ytrain, yvalid = train_test_split(
    train_data[x], 
    train_data["demand"], 
    test_size=0.1, 
    shuffle=True, 
    random_state=2024
)

print(f"训练集大小: {len(xtrain)}")
print(f"验证集大小: {len(xvalid)}")

# 创建 LightGBM Dataset
print("\n创建 LightGBM Dataset...")
train = lgb.Dataset(xtrain, label=ytrain)
valid = lgb.Dataset(xvalid, label=yvalid)

# 清理内存
del train_data, xtrain, xvalid, ytrain, yvalid
gc.collect()

print("\n数据准备完成！")


def fit_model(train, valid):

    params = {
        'metric': 'rmse',
        'objective': 'poisson',
        'seed': 42,                
        'force_row_wise': True,     
        'learning_rate': 0.08,       
        'lambda': 0.1,
        'num_leaves': 64,
        'sub_row': 0.7,
        'bagging_freq': 1,
        'colsample_bytree': 0.7
    }
    
    print("\n模型参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\n开始训练...")
    print(f"训练轮数: 2000")
    
    fit = lgb.train(
        params, 
        train, 
        num_boost_round=2000, 
        valid_sets=[valid],
        valid_names=['valid'],  
        callbacks=[lgb.log_evaluation(period=100)],
    )

    # 画图：按 gain 显示特征重要性
    lgb.plot_importance(fit, importance_type="gain", precision=0, height=0.5, figsize=(6, 10))
    
    return fit

print("\n开始训练 LightGBM 模型...")
fit = fit_model(train, valid)

print("\n" + "=" * 60)
print("模型训练完成！")
print("=" * 60)


# 显示特征重要性
print("\n特征重要性（Top 20）:")
importance = pd.DataFrame({
    'feature': x,
    'importance': fit.feature_importance()
}).sort_values('importance', ascending=False)

print(importance.head(20))

gc.collect()


duplicates = test.duplicated(subset=['id', 'd']).sum()
print(f"  重复的 (id, d) 组合数: {duplicates}")

if duplicates > 0:
    print(f" 发现重复数据，进行去重...")
    # 保留每个 (id, d) 的第一条记录
    test = test.drop_duplicates(subset=['id', 'd'], keep='first').reset_index(drop=True)
    print(f"  去重后 test 形状: {test.shape}")
    gc.collect()

# 预测函数
def demand_features_eval(df):
    """每个 id 都取最后一天的数据"""
    return df.groupby('id', sort=False).last().reset_index()

def pred_all(fit, test, x):
    """逐天滚动预测"""
    print("\n开始逐天预测...")
    print("-" * 60)
    
    for day in tqdm(range(FIRST, FIRST + LENGTH)):
        # 筛选窗口数据
        test_day = demand_features_eval(
            test[(test.d <= day) & (test.d >= day - max(LAGS) - max(WINDOWS))]
        )
        
        predictions = fit.predict(test_day[x])
        
        test_day['demand_pred'] = predictions
        
        # 更新 test 中的 demand
        test = test.merge(
            test_day[['id', 'demand_pred']], 
            on='id', 
            how='left', 
            suffixes=('', '_new')
        )
        
        # 只更新当天的数据
        mask = (test.d == day)
        test.loc[mask, 'demand'] = test.loc[mask, 'demand_pred']
        
        # 删除临时列
        test = test.drop(columns=['demand_pred'])
    
    print(f"\n 预测完成！")
    print("-" * 60)
    
    return test

# 执行预测

pred = pred_all(fit, test, x)


# 生成提交文件
def pred_to_csv(test, file):
    """将预测结果转换为 Kaggle 提交格式"""
    print("\n转换预测结果为提交格式...")
    
    # 创建副本避免修改原始数据
    test = test.copy()
    test['id'] = test['id'].astype(str)
    test['d'] = test['d'].astype(int)
    
    # 添加后缀
    test['id_suffix'] = np.where(test.d < FIRST, "validation", "evaluation")
    test['id'] = test['id'] + "_" + test['id_suffix']
    
    # 添加 F 列
    test['F'] = "F" + (test.d - FIRST + 1).astype(str)
    
    print(f"\n数据转换后:")
    print(f"  形状: {test.shape}")
    print(f"  唯一 ID 数: {test['id'].nunique()}")
    print(f"  唯一 F 数: {test['F'].nunique()}")
    print(f"  示例 ID: {test['id'].iloc[0]}")
    print(f"  示例 F: {test['F'].iloc[0]}")
    
    # Pivot
    print(f"\n执行 pivot...")
    submission = test.pivot(
        index="id", 
        columns="F", 
        values="demand"
    ).reset_index()
    
    # 填充缺失值
    submission = submission.fillna(0)
    
    print(f"  Pivot 后形状: {submission.shape}")
    
    # 确保列顺序正确
    f_cols = [f'F{i}' for i in range(1, 29)]
    
    # 检查是否所有列都存在
    missing_cols = set(f_cols) - set(submission.columns)
    if missing_cols:
        print(f"缺失的列: {missing_cols}")
        for col in missing_cols:
            submission[col] = 0
    
    # 重新排序列
    submission = submission[['id'] + f_cols]
    
    # 保存
    submission.to_csv(file, index=False)
    
    print(f"\n提交文件已保存: {file}")
    print(f"  最终形状: {submission.shape}")
    print(f"  期望形状: (60980, 29)")
    print(f"  匹配: {submission.shape == (60980, 29)}")
    
    return submission

# 执行生成
submission = pred_to_csv(pred, file="submission.csv")

# 验证提交文件
print(f"\n" + "=" * 60)
print("验证提交文件")
print("=" * 60)

print(f"\n基本信息:")
print(f"  形状: {submission.shape}")
print(f"  列数: {len(submission.columns)}")
print(f"  行数: {len(submission)}")

print(f"\n预测值统计:")
f_cols = [f'F{i}' for i in range(1, 29)]
all_predictions = submission[f_cols].values.flatten()
print(f"  均值: {all_predictions.mean():.4f}")
print(f"  中位数: {np.median(all_predictions):.4f}")
print(f"  最小值: {all_predictions.min():.4f}")
print(f"  最大值: {all_predictions.max():.4f}")
print(f"  零值比例: {(all_predictions == 0).mean():.2%}")

print(f"\n提交文件预览:")
print(submission.head())

print(f"\nID 格式检查:")
sample_ids = submission['id'].head(3).tolist()
for sid in sample_ids:
    print(f"  {sid}")

print("\n提交文件: /kaggle/working/submission.csv")

