!pip install pmdarima


# --- 基础库 ---
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style
import warnings

# --- 模型库 ---
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import pmdarima as pm # 引入pmdarima

# --- 设置 ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
sns.set_style('whitegrid')


def custom_score(y_true, y_pred, eps=1e-12):
    """比赛定义的评分函数"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        raise ValueError('empty array')

    if (y_true < 0).any():
        raise ValueError('negative y_true')

    if (~ np.isfinite(y_pred)).any():
        raise ValueError('infinite y_pred')

    ape = np.abs((y_true - y_pred) / np.maximum(y_true, eps))
    good_mask = ape <= 1.0
    good_rate = good_mask.mean()

    if good_rate < 0.7:
        return {'score': 0, 'good_rate': good_rate, 'str': f"{Fore.RED}score={0:.3f} {good_rate=:.3f}{Style.RESET_ALL}"}

    good_ape = ape[good_mask]
    mape = np.mean(good_ape)
    scaled_mape = mape / good_rate
    score = 1 - scaled_mape
    return {'score': score, 'good_rate': good_rate, 'str': f"{score=:.3f} {good_rate=:.3f}"}

month_codes = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}


# 读取训练集
DATA_PATH = '/kaggle/input/china-real-estate-demand-prediction/train/'

ci = pd.read_csv(f'{DATA_PATH}city_indexes.csv')
ci.rename(columns={'city_indicator_data_year': 'year'}, inplace=True)
csi = pd.read_csv(f'{DATA_PATH}city_search_index.csv')
sp = pd.read_csv(f'{DATA_PATH}sector_POI.csv')

train_lt = pd.read_csv(f'{DATA_PATH}land_transactions.csv')
train_ltns = pd.read_csv(f'{DATA_PATH}land_transactions_nearby_sectors.csv')
train_pht = pd.read_csv(f'{DATA_PATH}pre_owned_house_transactions.csv')
train_phtns = pd.read_csv(f'{DATA_PATH}pre_owned_house_transactions_nearby_sectors.csv')
train_nht = pd.read_csv(f'{DATA_PATH}new_house_transactions.csv')
train_nhtns = pd.read_csv(f'{DATA_PATH}new_house_transactions_nearby_sectors.csv')

# 读取测试集
test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv') 
print("Data loaded successfully.")
print(train_lt.shape, train_ltns.shape, train_pht.shape, train_phtns.shape, train_nht.shape, train_nhtns.shape)
# print(train_nht.head())


# 训练集：2019年1日-2024年7月，96个区块，但多个训练集文件中各个区块的数据存在较多缺失, 在选择特征时需注意
np.unique((test['id'].str.split().str[-1])).size


test_id = test.id.str.split('_', expand=True)
test['month'] = test_id[0]
test['sector'] = test_id[1]
del test_id

all_dfs = {
    'train_lt': train_lt, 'train_ltns': train_ltns, 
    'train_pht': train_pht, 'train_phtns': train_phtns, 
    'train_nht': train_nht, 'train_nhtns': train_nhtns, 
    'csi': csi, 'sp': sp, 'test': test
}

for name, df in all_dfs.items():
    if 'sector' in df.columns:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
    
    if 'month' in df.columns:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month_num'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month_num'] - 1
        print(f"Processed {name}: time range {df['time'].min()} - {df['time'].max()}")

print("\nInitial processing complete.")
print(test.head())


sectors = np.arange(1, 97)
times = np.arange(train_nht['time'].min(), train_nht['time'].max() + 1)

grid = []
for sector in sectors:
    for time in times:
        grid.append({'sector_id': sector, 'time': time})

df_train = pd.DataFrame(grid)

# 将目标变量合并进来
df_train = pd.merge(df_train, train_nht[['sector_id', 'time', 'amount_new_house_transactions']], on=['sector_id', 'time'], how='left')

# 缺失值填充为0，因为NaN代表该时间点没有交易记录
df_train['amount_new_house_transactions'] = df_train['amount_new_house_transactions'].fillna(0)

print("基础训练集构建完成:")
print(df_train.head())


# 总体交易量趋势
plt.figure(figsize=(16, 6))
total_transactions = df_train.groupby('time')['amount_new_house_transactions'].sum()
total_transactions.plot()

plt.title('Total New House Transactions Over Time (All Sectors)', fontsize=16)
plt.xlabel('Time (Months since Jan 2019)', fontsize=12)
plt.ylabel('Total Transaction Amount', fontsize=12)
plt.xticks(np.arange(0, 70, 6))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()


# 查看不同板块的交易量差异
sector_total_transactions = df_train.groupby('sector_id')['amount_new_house_transactions'].sum().sort_values(ascending=False)

plt.figure(figsize=(16, 6))
sector_total_transactions.plot(kind='bar')
plt.title('Total Transactions per Sector', fontsize=16)
plt.xlabel('Sector ID', fontsize=12)
plt.ylabel('Total Transaction Amount', fontsize=12)
plt.xticks(rotation=90)
plt.show()

print("交易量最高的板块:")
print(sector_total_transactions.head())
print("\n交易量最低的板块 (包括0交易):")
print(sector_total_transactions.tail())


# 为了简化合并，先对每个数据源进行聚合，确保每个 (sector_id, time) 组合只有一行记录
# 对于有多个记录的，我们取均值
train_pht_agg = train_pht.groupby(['sector_id', 'time']).agg({'amount_pre_owned_house_transactions': 'mean'}).reset_index()
train_lt_agg = train_lt.groupby(['sector_id', 'time']).agg({'transaction_amount': 'mean'}).reset_index().rename(columns={'transaction_amount': 'amount_land_transactions'})

# 合并二手房和土地交易数据
df_train = pd.merge(df_train, train_pht_agg, on=['sector_id', 'time'], how='left')
df_train = pd.merge(df_train, train_lt_agg, on=['sector_id', 'time'], how='left')

# 合并城市搜索指数 (csi) - 这个数据没有sector_id，是城市级别的，所以直接按time合并
csi_agg = csi.groupby('time').agg(csi_mean=('search_volume', 'mean')).reset_index()
df_train = pd.merge(df_train, csi_agg, on='time', how='left')

# 合并城市宏观指标 (ci) - 这个数据是年度的，需要处理
df_train['year'] = (df_train['time'] // 12) + 2019
df_train = pd.merge(df_train, ci, on='year', how='left')

# 合并板块POI (sp) - 这个数据没有时间维度，直接按sector_id合并
df_train = pd.merge(df_train, sp.drop(columns=['sector']), on='sector_id', how='left')

# 填充合并后产生的NaN值
# 对于交易数据，NaN意味着没有交易，填0
df_train['amount_pre_owned_house_transactions'] = df_train['amount_pre_owned_house_transactions'].fillna(0)
df_train['amount_land_transactions'] = df_train['amount_land_transactions'].fillna(0)
# 对于POI数据，NaN可能意味着该区域没有这类设施，也填0
poi_cols = sp.columns.drop(['sector', 'sector_id'])
df_train[poi_cols] = df_train[poi_cols].fillna(0)

print("数据合并完成:")
print(df_train.head())
print("\n合并后数据的缺失值情况:")
print(df_train.isnull().sum()[df_train.isnull().sum() > 0])


# 选择数值类型的列进行相关性分析
numeric_cols = df_train.select_dtypes(include=np.number).columns
correlation_matrix = df_train[numeric_cols].corr()

plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False) # annot=True 会太密集，所以关闭
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.show()

# 单独查看与目标变量的相关性
target_correlation = correlation_matrix['amount_new_house_transactions'].sort_values(ascending=False)
print("Features correlated with 'amount_new_house_transactions':")
print(target_correlation.head(15))


fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Key Feature vs. Target Variable', fontsize=20)

# 1. 二手房 vs 新房
sns.scatterplot(ax=axes[0, 0], data=df_train, x='amount_pre_owned_house_transactions', y='amount_new_house_transactions', alpha=0.5)
axes[0, 0].set_title('Pre-owned vs. New House Transactions')
axes[0, 0].set_xscale('log') # 使用对数坐标轴，因为数据分布不均
axes[0, 0].set_yscale('log')

# 2. 土地交易 vs 新房
sns.scatterplot(ax=axes[0, 1], data=df_train, x='amount_land_transactions', y='amount_new_house_transactions', alpha=0.5)
axes[0, 1].set_title('Land vs. New House Transactions')
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')

# 3. 搜索指数 vs 新房
sns.lineplot(ax=axes[1, 0], data=df_train, x='time', y='csi_mean', label='Search Index', color='orange')
ax2 = axes[1, 0].twinx()
sns.lineplot(ax=ax2, data=df_train, x='time', y='amount_new_house_transactions', label='New House Trans.', color='blue')
axes[1, 0].set_title('Search Index and Transactions over Time')

# 4. GDP vs 新房
sns.lineplot(ax=axes[1, 1], data=df_train, x='time', y='gdp_100m', label='GDP', color='green')
ax3 = axes[1, 1].twinx()
sns.lineplot(ax=ax3, data=df_train, x='time', y='amount_new_house_transactions', label='New House Trans.', color='blue')
axes[1, 1].set_title('GDP and Transactions over Time')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


df_train['month'] = df_train['time'] % 12 + 1
# 'year' 列在合并ci时已经创建了

print("时间特征已添加:")
print(df_train[['time', 'year', 'month']].head())


# 按 sector_id 分组，然后对目标变量进行移位
lags = [1, 2, 3, 6, 12]
for lag in lags:
    df_train[f'lag_{lag}'] = df_train.groupby('sector_id')['amount_new_house_transactions'].shift(lag)

print(f"滞后特征 {lags} 已添加。")
print(df_train[df_train['sector_id']==1][['time', 'amount_new_house_transactions', 'lag_1', 'lag_12']].head(15))


windows = [3, 6, 12]
for window in windows:
    # 我们在shift(1)上计算，以避免包含当前值，防止数据泄露
    shifted_series = df_train.groupby('sector_id')['amount_new_house_transactions'].shift(1)
    
    df_train[f'rolling_mean_{window}'] = shifted_series.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    df_train[f'rolling_std_{window}'] = shifted_series.rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
    df_train[f'rolling_max_{window}'] = shifted_series.rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
    df_train[f'rolling_min_{window}'] = shifted_series.rolling(window, min_periods=1).min().reset_index(level=0, drop=True)

# 填充滚动特征产生的NaN值
for col in df_train.columns:
    if 'rolling' in col:
        df_train[col] = df_train[col].fillna(0)

print(f"滚动窗口特征 {windows} 已添加。")
print(df_train[df_train['sector_id']==1][['time', 'amount_new_house_transactions', 'rolling_mean_3', 'rolling_std_3']].head())


# 合并邻近板块新房交易数据
nhtns_agg = train_nhtns.groupby(['sector_id', 'time']).agg(
    transaction_amount_new_house_nearby=('amount_new_house_transactions_nearby_sectors', 'sum'),
    num_new_house_transactions_nearby=('num_new_house_transactions_nearby_sectors', 'sum')
).reset_index()
df_train = pd.merge(df_train, nhtns_agg, on=['sector_id', 'time'], how='left')

# 合并邻近板块二手房交易数据
phtns_agg = train_phtns.groupby(['sector_id', 'time']).agg(
    transaction_amount_pre_owned_house_nearby=('amount_pre_owned_house_transactions_nearby_sectors', 'sum'),
    num_pre_owned_house_transactions_nearby=('num_pre_owned_house_transactions_nearby_sectors', 'sum')
).reset_index()
df_train = pd.merge(df_train, phtns_agg, on=['sector_id', 'time'], how='left')

# 填充NaN
nearby_cols = ['transaction_amount_new_house_nearby', 'num_new_house_transactions_nearby', 
               'transaction_amount_pre_owned_house_nearby', 'num_pre_owned_house_transactions_nearby']
df_train[nearby_cols] = df_train[nearby_cols].fillna(0)

print("邻近板块特征已添加。")
print(df_train[df_train['sector_id']==1][['time'] + nearby_cols].head())


# 填充特征工程中产生的NaN值
df_train = df_train.fillna(0)

# 定义特征和目标
target = 'amount_new_house_transactions'
features = [col for col in df_train.columns if col not in [target]]
# 'year'列和ci中的其他列是时间相关的，但在我们的'time'特征中已经包含了，为了避免冗余和潜在的数据泄露，先去掉
features = [col for col in features if col not in ci.columns or col == 'year'] # 重新加回year
features.remove('year') # year列在df_train中，但我们用time和month，所以去掉

# 将分类特征转换为category类型，以便LGBM高效处理
categorical_features = ['sector_id', 'month']
for col in categorical_features:
    df_train[col] = df_train[col].astype('category')

# 交叉验证框架
N_SPLITS = 5
TEST_SIZE = int(0.1*len(df_train))
cv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

print(f"使用的特征数量: {len(features)}")
print("部分特征:", features[:10])


oof_preds = []
oof_trues = []


df_train_copy = df_train[df_train[target]>0]
feature_selected = ['sector_id', 'month', 'time', 'lag_1', 'csi_mean',
       'transaction_amount_new_house_nearby',
       'amount_pre_owned_house_transactions', 'rolling_mean_12',
       'num_new_house_transactions_nearby',
       'transaction_amount_pre_owned_house_nearby']

feature_importances = pd.DataFrame(index=feature_selected)
for fold, (train_idx, val_idx) in enumerate(cv.split(df_train_copy)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    
    # 划分训练集和验证集
    X_train = df_train_copy.iloc[train_idx][feature_selected]
    y_train = np.log1p(df_train_copy.iloc[train_idx][target])
    X_val = df_train_copy.iloc[val_idx][feature_selected]
    y_val = np.log1p(df_train_copy.iloc[val_idx][target])
    
    # 定义模型参数
    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }
    
    # 训练模型
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(3000, verbose=False)])
    
    # 预测验证集
    val_preds = model.predict(X_val)
    # 预测值不能为负
    val_preds[val_preds < 0] = 0
    
    # 保存OOF结果
    oof_preds.append(val_preds)
    oof_trues.append(y_val)
    
    # 保存特征重要性
    feature_importances[f'fold_{fold+1}'] = model.feature_importances_
    
    # 打印当前折的分数
    score = custom_score(y_val, val_preds)
    print(f"Fold {fold+1} Score: {score['str']}")

# 计算总体OOF分数
oof_preds_all = np.concatenate(oof_preds)
oof_trues_all = np.concatenate(oof_trues)
overall_score = custom_score(oof_trues_all, oof_preds_all)
print(f"\n--- Overall OOF Score ---")
print(overall_score['str'])


plt.plot(y_train)


feature_importances['mean'] = feature_importances.mean(axis=1)
feature_importances.sort_values('mean', ascending=False, inplace=True)

plt.figure(figsize=(10, 12))
sns.barplot(x='mean', y=feature_importances.index[:30], data=feature_importances.head(30))
plt.title('Top 30 Feature Importances (LGBM)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


df_train[features]


# # Prophet 需要特定的列名：ds (datestamp) 和 y (target)
# # 我们需要将 time 转换为日期格式
# df_prophet = df_train.copy()
# df_prophet['ds'] = df_prophet['time'].apply(lambda x: pd.to_datetime('2019-01-01') + pd.DateOffset(months=x))
# df_prophet.rename(columns={'amount_new_house_transactions': 'y'}, inplace=True)

# # 定义要用作额外回归量的特征
# # regressors = [
# #     'amount_pre_owned_house_transactions',
# #     'amount_land_transactions',
# #     'csi_mean', # 'search_index'
# #     'gdp_100m', # 'GDP'
# #     'urban_consumer_price_index_previous_year_100', # 'CPI'
# #     'number_of_shops', # 'num_shopping'
# #     'transportation_station', # 'num_transportation'
# #     'education', # 'num_education'
# #     'medical_health', # 'num_medical'
# #     'transaction_amount_new_house_nearby',
# #     'transaction_amount_pre_owned_house_nearby'
# # ]
# regressors = feature_selected
# oof_preds_prophet = []
# oof_trues_prophet = []

# for fold, (train_idx, val_idx) in enumerate(cv.split(df_train)):
#     print(f"--- Prophet Fold {fold+1}/{N_SPLITS} ---")
#     fold_preds = []
#     fold_trues = []
    
#     # 在每个板块上循环
#     for sector_id in df_prophet['sector_id'].unique():
#         # 提取当前板块的数据
#         sector_data = df_prophet[df_prophet['sector_id'] == sector_id]
        
#         # 划分训练集和验证集
#         train_data = sector_data[sector_data.index.isin(train_idx)]
#         val_data = sector_data[sector_data.index.isin(val_idx)]
        
#         if len(train_data) == 0 or train_data['y'].sum() == 0:
#             # 如果没有训练数据或交易量全为0，则预测为0
#             preds = np.zeros(len(val_data))
#         else:
#             # 实例化并训练模型
#             m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
#             valid_regressors = [reg for reg in regressors if reg in train_data.columns]
#             for regressor in valid_regressors:
#                 m.add_regressor(regressor)
            
#             m.fit(train_data[['ds', 'y'] + valid_regressors])
            
#             if len(val_data) == 0:
#                 # If no validation data for this sector in this fold, append empty arrays
#                 preds = np.array([])
#             elif len(train_data) == 0 or train_data['y'].sum() == 0:
#                 # If no training data or all transactions are 0, predict 0 for validation data
#                 preds = np.zeros(len(val_data))
#             else:
#                 # Instantiate and train model
#                 # m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
#                 # valid_regressors = [reg for reg in regressors if reg in train_data.columns]
#                 # for regressor in valid_regressors:
#                 #     m.add_regressor(regressor)
                
#                 # m.fit(train_data[['ds', 'y'] + valid_regressors])
                
#                 # Create future dataframe for prediction using val_data's ds and regressors
#                 future = val_data[['ds']].copy()
#                 if valid_regressors:
#                     for regressor in valid_regressors:
#                         future[regressor] = val_data[regressor]
                
#                 forecast = m.predict(future)
#                 preds = forecast['yhat'].values
        
#         preds[preds < 0] = 0
#         fold_preds.append(preds)
#         fold_trues.append(val_data['y'].values)
        
#     # 将当前折叠的所有板块预测结果合并
#     fold_preds_flat = np.concatenate(fold_preds)
#     fold_trues_flat = np.concatenate(fold_trues)
    
#     oof_preds_prophet.append(fold_preds_flat)
#     oof_trues_prophet.append(fold_trues_flat)
    
#     score = custom_score(fold_trues_flat, fold_preds_flat)
#     print(f"Prophet Fold {fold+1} Score: {score['str']}")

# # 计算总体OOF分数
# oof_preds_prophet_all = np.concatenate(oof_preds_prophet)
# oof_trues_prophet_all = np.concatenate(oof_trues_prophet)
# overall_score_prophet = custom_score(oof_trues_prophet_all, oof_preds_prophet_all)
# print(f"\n--- Prophet Overall OOF Score ---")
# print(overall_score_prophet['str'])


from statsmodels.tsa.statespace.sarimax import SARIMAX

oof_preds_arima = []
oof_trues_arima = []

for fold, (train_idx, val_idx) in enumerate(cv.split(df_train)):
    print(f"--- ARIMA Fold {fold+1}/{N_SPLITS} ---")
    fold_preds = []
    fold_trues = []
    
    # 在每个板块上循环
    for sector_id in df_train['sector_id'].unique():
        # 提取当前板块的数据
        sector_data = df_train[df_train['sector_id'] == sector_id]
        
        # 划分训练集和验证集
        train_y = sector_data[sector_data.index.isin(train_idx)][target]
        val_y = sector_data[sector_data.index.isin(val_idx)][target]
        
        if len(train_y) < 12 or train_y.sum() == 0:
            # 如果训练数据太少或全为0，则预测为0
            preds = np.zeros(len(val_y))
        else:
            try:
                sector_data = np.log1p(sector_data)
                # 定义并训练SARIMA模型
                # 参数 (p,d,q)(P,D,Q,s) s=12 代表年度季节性
                model = SARIMAX(train_y, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), 
                                enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit(disp=False)
                preds = results.forecast(steps=len(val_y))
            except Exception as e:
                # 如果模型拟合失败，则预测为0
                # print(f"ARIMA failed for sector {sector_id} in fold {fold+1}: {e}")
                preds = np.zeros(len(val_y))
        
        preds[preds < 0] = 0
        fold_preds.append(np.expm1(preds))
        fold_trues.append(val_y.values)
        
    # 将当前折叠的所有板块预测结果合并
    fold_preds_flat = np.concatenate(fold_preds)
    fold_trues_flat = np.concatenate(fold_trues)
    
    oof_preds_arima.append(fold_preds_flat)
    oof_trues_arima.append(fold_trues_flat)
    
    score = custom_score(fold_trues_flat, fold_preds_flat)
    print(f"ARIMA Fold {fold+1} Score: {score['str']}")

# 计算总体OOF分数
oof_preds_arima_all = np.concatenate(oof_preds_arima)
oof_trues_arima_all = np.concatenate(oof_trues_arima)
overall_score_arima = custom_score(oof_trues_arima_all, oof_preds_arima_all)
print(f"\n--- ARIMA Overall OOF Score ---")
print(overall_score_arima['str'])


import optuna
# 在Jupyter中禁用Optuna的日志记录，以保持输出整洁
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    """定义Optuna的优化目标函数"""
    # 定义超参数搜索空间
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'seed': 42,
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    
    oof_preds = []
    oof_trues = []
    
    # 使用相同的时间序列交叉验证
    cv_tuner = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    
    for fold, (train_idx, val_idx) in enumerate(cv_tuner.split(df_train)):
        X_train = df_train.iloc[train_idx][feature_selected]
        y_train = np.log1p(df_train.iloc[train_idx][target])
        X_val = df_train.iloc[val_idx][feature_selected]
        y_val = np.log1p(df_train.iloc[val_idx][target])
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='mae',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        val_preds = model.predict(X_val)
        val_preds[val_preds < 0] = 0
        
        oof_preds.append(val_preds)
        oof_trues.append(y_val)
        
    oof_preds_all = np.concatenate(oof_preds)
    oof_trues_all = np.concatenate(oof_trues)
    
    # 我们的目标是最大化 custom_score
    score = custom_score(oof_trues_all, oof_preds_all)['score']
    
    return score


# # 创建一个study对象，并指定优化方向为“最大化”
# study = optuna.create_study(direction='maximize')

# # 运行优化，n_trials是尝试的次数
# # 注意：这可能会花费一些时间
# study.optimize(objective, n_trials=50) # 为了演示，我们只运行50次

# print("\nOptimization Finished!")
# print("Best trial:")
# trial = study.best_trial

# print(f"  Value: {trial.value}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")

# # 保存最佳参数以备后用
# best_params = trial.params


# trial.params


best_params = {'n_estimators': 1143,
 'learning_rate': 0.07425322198950735,
 'num_leaves': 52,
 'max_depth': 11,
 'feature_fraction': 0.6314889798103858,
 'bagging_fraction': 0.6016004365273729,
 'bagging_freq': 3,
 'lambda_l1': 0.021329634360916467,
 'lambda_l2': 0.06276748612295736}


# 使用Optuna找到的最佳参数，并加入一些固定参数
final_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 1000, # 使用早停，所以这个值可以大一些
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'seed': 42,
    'verbose': -1,
}
final_params.update(best_params)

# 在所有训练数据上训练最终模型
print("Training final model on all training data...")
X_train_full = df_train[df_train['time'] < 67][feature_selected]
y_train_full = np.log1p(df_train[df_train['time'] < 67][target])

final_model = lgb.LGBMRegressor(**final_params)
# 这里我们就不需要验证集和早停了，因为我们想用所有数据训练到指定的迭代次数
# 为了稳妥，我们还是用一个象征性的早停，但迭代次数可以由n_estimators控制
final_model.fit(X_train_full, y_train_full)

print("Final model trained.")


# 准备测试集框架
test_times = np.arange(67, 79)
df_test = []
for sector in sectors:
    for time in test_times:
        df_test.append({'sector_id': sector, 'time': time})
df_test = pd.DataFrame(df_test)

# 将测试集与训练集合并，方便统一创建特征
df_full = pd.concat([df_train, df_test], ignore_index=True)

# 重新创建所有特征
# (这里为了代码简洁，我们重复一遍特征创建过程，实际项目中可以封装成函数)
df_full['month'] = df_full['time'] % 12 + 1
df_full['year'] = (df_full['time'] // 12) + 2019

# 合并外部数据
df_full = pd.merge(df_full, csi_agg, on='time', how='left')
df_full = pd.merge(df_full, ci, on='year', how='left')
df_full = pd.merge(df_full, sp.drop(columns=['sector']), on='sector_id', how='left')
df_full = pd.merge(df_full, csi_agg, on='time', how='left')
# df_full[poi_cols] = df_full[poi_cols].fillna(0)

categorical_features = ['sector_id', 'month']
for col in categorical_features:
    df_full[col] = df_full[col].astype('category')
    
# 1. 创建滞后和滚动特征 (基于已知和已预测的值)
for lag in lags:
    df_full[f'lag_{lag}'] = df_full.groupby('sector_id')['amount_new_house_transactions'].shift(lag)
    df_full[f'lag_{lag}'].fillna(0, inplace=True)
for window in windows:
    shifted_series = df_full.groupby('sector_id')['amount_new_house_transactions'].shift(1)
    df_full[f'rolling_mean_{window}'] = shifted_series.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    df_full[f'rolling_std_{window}'] = shifted_series.rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
    df_full[f'rolling_max_{window}'] = shifted_series.rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
    df_full[f'rolling_min_{window}'] = shifted_series.rolling(window, min_periods=1).min().reset_index(level=0, drop=True)


df_full_raw = df_full.copy()
# 用历史均值替换为测试集
for feature in ['csi_mean', 'transaction_amount_new_house_nearby', 'amount_pre_owned_house_transactions', 'rolling_mean_12', 'num_new_house_transactions_nearby', 'transaction_amount_pre_owned_house_nearby']:
    df_full[feature] = df_full.groupby(['sector_id', 'month'])[feature].transform('mean')

# 递归预测
for t in test_times:
    print(f"Predicting for time = {t}")
    # 2. 预测当前时间步
    current_X = df_full[df_full['time'] == t][feature_selected]
    current_preds = final_model.predict(current_X[feature_selected])
    current_preds[current_preds < 0] = 0
    
    # 3. 将预测值写回DataFrame，用于下一步的特征生成
    df_full.loc[df_full['time'] == t, 'amount_new_house_transactions'] = np.expm1(current_preds)

print("Recursive prediction complete.")


# 计算平均值
mean_amounts = df_full.groupby(['sector_id', 'month'])['amount_new_house_transactions'].transform('mean')
df_full_copy = df_full.copy()
# df_full_copy.loc[df_full['amount_new_house_transactions']<1, 'amount_new_house_transactions'] = mean_amounts[df_full['amount_new_house_transactions']<1]

# 提取测试集的预测结果
df_submission = df_full_copy[df_full_copy['time'] >= 67].copy()
df_submission['month'] = df_submission['month'].astype('int64')

# 整理成提交格式
df_submission['month_str'] = df_submission['month'].map({v: k for k, v in month_codes.items()})
df_submission['id'] = df_submission['year'].astype(str) + ' ' + df_submission['month_str'] + '_sector ' + df_submission['sector_id'].astype(str)
df_submission.rename(columns={'amount_new_house_transactions': 'new_house_transaction_amount'}, inplace=True)
# df_submission['new_house_transaction_amount'] = np.expm1(df_submission['new_house_transaction_amount'])

# 创建提交文件
submission_file = df_submission[['id', 'new_house_transaction_amount']]
submission_file.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
print(submission_file.head())


# --- Helper function for fallback prediction ---
def predict_with_fallback(series, steps, lookback_months=6):
    """
    Predict using geometric mean of recent positive values.
    """
    recent_data = series.tail(lookback_months)
    positive_values = recent_data[recent_data > 0]
    
    if len(positive_values) > 0:
        pred_value = np.exp(np.log(positive_values).mean())
    else:
        pred_value = 0
        
    return np.full(steps, pred_value)

# --- Main Prediction Loop ---
fold_preds = []
target = 'amount_new_house_transactions'
predict_months = 12
zero_check_months = 3 # Champion strategy from reference script

for sector_id in df_train['sector_id'].unique():
    # Extract data for the current sector
    sector_data = df_train[df_train['sector_id'] == sector_id][target].copy()
    
    preds = np.zeros(predict_months) # Default prediction is 0
    
    # Condition for attempting SARIMA: sufficient data and non-zero values
    # Increased required length for auto_arima to be more stable
    if len(sector_data) >= 36 and sector_data.sum() > 0:
        try:
            # Use auto_arima to find the best model
            # We use log1p transformation for stability
            model = pm.auto_arima(np.log1p(sector_data),
                                  start_p=1, start_q=1,
                                  test='adf',
                                  max_p=3, max_q=3,
                                  m=12, # seasonal period
                                  start_P=0, seasonal=True,
                                  d=None, D=1, trace=False,
                                  error_action='ignore',  
                                  suppress_warnings=True, 
                                  stepwise=True)
            
            # Forecast
            preds_log = model.predict(n_periods=predict_months)
            # Inverse transform from log1p
            preds = np.expm1(preds_log)
            print(f"SARIMA success for sector {sector_id}. Model: {model.order}, {model.seasonal_order}")

        except Exception as e:
            # If auto_arima fails, use the fallback method
            print(f"SARIMA failed for sector {sector_id}: {e}. Using fallback.")
            preds = predict_with_fallback(sector_data, predict_months)
    else:
        # If data is insufficient, use the fallback method directly
        print(f"Insufficient data for sector {sector_id}. Using fallback.")
        preds = predict_with_fallback(sector_data, predict_months)

    # --- Post-processing: Apply "recent zero" rule ---
    if (sector_data.tail(zero_check_months).min() == 0):
        print(f"Applying zero rule for sector {sector_id}.")
        preds[:] = 0 # Set all future predictions to 0

    # Ensure predictions are non-negative
    preds[preds < 0] = 0
    
    # Note: The prediction 'preds' is already in the original scale. 
    # No need for np.expm1 here as it's handled inside the try-except block.
    fold_preds.append(preds)


# # --- Helper function for fallback prediction ---
# def predict_with_fallback(series, steps, lookback_months=6):
#     """
#     Predict using geometric mean of recent positive values.
#     """
#     recent_data = series.tail(lookback_months)
#     positive_values = recent_data[recent_data > 0]
    
#     if len(positive_values) > 0:
#         pred_value = np.exp(np.log(positive_values).mean())
#     else:
#         pred_value = 0
        
#     return np.full(steps, pred_value)

# # --- Cross-Validation Setup ---
# N_SPLITS = 4
# TEST_SIZE = 12 # Predict 12 months into the future
# cv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

# oof_preds = []
# oof_trues = []
# target = 'amount_new_house_transactions'
# zero_check_months = 3 # Champion strategy from reference script

# # --- Main Cross-Validation Loop ---
# for fold, (train_idx, val_idx) in enumerate(cv.split(df_train[df_train.sector_id==1])): # split based on one sector's timeline
#     print(f"--- FOLD {fold+1}/{N_SPLITS} ---")
    
#     fold_preds = []
#     fold_trues = []

#     # Iterate over each sector
#     for sector_id in df_train['sector_id'].unique():
        
#         # --- Data Preparation for this fold and sector ---
#         full_sector_data = df_train[df_train['sector_id'] == sector_id]
#         train_data = full_sector_data.iloc[train_idx]
#         val_data = full_sector_data.iloc[val_idx]
        
#         train_series = train_data[target]
        
#         preds = np.zeros(len(val_data)) # Default prediction is 0

#         # --- Model Training and Prediction ---
#         if len(train_series) >= 24 and train_series.sum() > 0:
#             try:
#                 model = pm.auto_arima(np.log1p(train_series),
#                                       start_p=1, start_q=1, test='adf', max_p=3, max_q=3,
#                                       m=12, start_P=0, seasonal=True, d=None, D=1, 
#                                       trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
                
#                 preds_log = model.predict(n_periods=len(val_data))
#                 preds = np.expm1(preds_log)
#                 # print(f"SARIMA success for sector {sector_id} in fold {fold+1}.")

#             except Exception as e:
#                 # print(f"SARIMA failed for sector {sector_id} in fold {fold+1}: {e}. Using fallback.")
#                 preds = predict_with_fallback(train_series, len(val_data))
#         else:
#             # print(f"Insufficient data for sector {sector_id} in fold {fold+1}. Using fallback.")
#             preds = predict_with_fallback(train_series, len(val_data))

#         # --- Post-processing ---
#         if (train_series.tail(zero_check_months).min() == 0):
#             preds[:] = 0

#         preds[preds < 0] = 0
        
#         fold_preds.append(preds)
#         fold_trues.append(val_data[target].values)

#     # --- Evaluate Fold ---
#     fold_preds_flat = np.concatenate(fold_preds)
#     fold_trues_flat = np.concatenate(fold_trues)
    
#     oof_preds.append(fold_preds_flat)
#     oof_trues.append(fold_trues_flat)
    
#     score = custom_score(fold_trues_flat, fold_preds_flat)
#     print(f"Fold {fold+1} Score: {score['str']}")

# # --- Final OOF Score ---
# oof_preds_all = np.concatenate(oof_preds)
# oof_trues_all = np.concatenate(oof_trues)
# overall_score = custom_score(oof_trues_all, oof_preds_all)
# print(f"\n--- Overall OOF Score ---")
# print(overall_score['str'])

# # --- Final Model Training on All Data ---
# print("\n--- Training Final Model on All Data ---")
# final_predictions = []
# for sector_id in df_train['sector_id'].unique():
#     sector_data = df_train[df_train['sector_id'] == sector_id][target].copy()
#     preds = np.zeros(TEST_SIZE)
#     if len(sector_data) >= 24 and sector_data.sum() > 0:
#         try:
#             model = pm.auto_arima(np.log1p(sector_data),
#                                   start_p=1, start_q=1, test='adf', max_p=3, max_q=3,
#                                   m=12, start_P=0, seasonal=True, d=None, D=1, 
#                                   trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
#             preds_log = model.predict(n_periods=TEST_SIZE)
#             preds = np.expm1(preds_log)
#             print(f"Final model training success for sector {sector_id}.")
#         except Exception as e:
#             print(f"Final model training failed for sector {sector_id}: {e}. Using fallback.")
#             preds = predict_with_fallback(sector_data, TEST_SIZE)
#     else:
#         print(f"Insufficient data for final model on sector {sector_id}. Using fallback.")
#         preds = predict_with_fallback(sector_data, TEST_SIZE)
    
#     if (sector_data.tail(zero_check_months).min() == 0):
#         preds[:] = 0
        
#     preds[preds < 0] = 0
#     final_predictions.append(preds)


# from statsmodels.tsa.statespace.sarimax import SARIMAX

# oof_preds_arima = []
# oof_trues_arima = []


# fold_preds = []
# fold_trues = []

# # 在每个板块上循环
# predict_months = 12
# for sector_id in df_train['sector_id'].unique():
#     # 提取当前板块的数据
#     sector_data = df_train[df_train['sector_id'] == sector_id][target]
    
#     if len(sector_data) < 12 or sector_data.sum() == 0:
#         # 如果训练数据太少或全为0，则预测为0
#         preds = np.zeros(predict_months)
#         print(sector_id, ' equal 0!')
        
#     else:
#         try:
#             # log转换
#             sector_data = np.log1p(sector_data)
#             # 定义并训练SARIMA模型
#             # 参数 (p,d,q)(P,D,Q,s) s=12 代表年度季节性
#             model = SARIMAX(sector_data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), 
#                             enforce_stationarity=False, enforce_invertibility=False)
#             results = model.fit(disp=False)
#             preds = results.forecast(steps=predict_months)
#         except Exception as e:
#             # 如果模型拟合失败，则预测为0
#             print(f"ARIMA failed for sector {sector_id} in fold {fold+1}: {e}")
#             preds = np.zeros(predict_months)
    
#     preds[preds < 0] = 0
#     fold_preds.append(np.expm1(preds))
# # np.array(fold_preds).ravel().shape


df_train[df_train['sector_id'] == 95][target]


# 创建提交文件
submission_file['new_house_transaction_amount'] = np.array(fold_preds).ravel()
submission_file.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
print(submission_file.head())


df_submission[ 'new_house_transaction_amount'].plot()
df_submission[ 'new_house_transaction_amount'][df_submission[ 'new_house_transaction_amount']<10].size


mean_amounts.plot(figsize=(20,6))


df_full['amount_new_house_transactions'].plot(figsize=(20,6))


# --- 基础库 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from itertools import product

# --- 模型库 ---
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from colorama import Fore, Style

# --- 比赛官方评分函数 ---
def custom_score(y_true, y_pred, eps=1e-12):
    """比赛定义的评分函数"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0: return {'score': 0, 'good_rate': 0, 'str': "Empty array"}
    if (y_true < 0).any(): raise ValueError('negative y_true')
    if (~ np.isfinite(y_pred)).any(): raise ValueError('infinite y_pred')
    ape = np.abs((y_true - y_pred) / np.maximum(y_true, eps))
    good_mask = ape <= 1.0
    good_rate = good_mask.mean()
    if good_rate < 0.7:
        return {'score': 0, 'good_rate': good_rate, 'str': f"{Fore.RED}score={0:.3f} good_rate={good_rate:.3f}{Style.RESET_ALL}"}
    good_ape = ape[good_mask]
    mape = np.mean(good_ape)
    scaled_mape = mape / good_rate
    score = 1 - scaled_mape
    return {'score': score, 'good_rate': good_rate, 'str': f"score={score:.3f} good_rate={good_rate:.3f}"}

# --- 设置 ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
sns.set_style('whitegrid')

# 定义文件路径
DATA_PATH = '/kaggle/input/china-real-estate-demand-prediction/train/'

# --- 0. 数据加载 ---
print("--- 0. Loading Data ---")
all_dfs_raw = {
    'train_nht': pd.read_csv(f'{DATA_PATH}new_house_transactions.csv'),
    'train_pht': pd.read_csv(f'{DATA_PATH}pre_owned_house_transactions.csv'),
    'train_lt': pd.read_csv(f'{DATA_PATH}land_transactions.csv'),
    'train_nhtns': pd.read_csv(f'{DATA_PATH}new_house_transactions_nearby_sectors.csv'),
    'train_phtns': pd.read_csv(f'{DATA_PATH}pre_owned_house_transactions_nearby_sectors.csv'),
    'sp': pd.read_csv(f'{DATA_PATH}sector_POI.csv'),
    'ci': pd.read_csv(f'{DATA_PATH}city_indexes.csv'),
    'csi': pd.read_csv(f'{DATA_PATH}city_search_index.csv'),
    'test': pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv'),
    'sample_submission': pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv')
}
print("Data loaded successfully.")

# --- 1. 数据预处理与整合 ---
print("\n--- 1. Data Preprocessing & Integration ---")
test_id_split = all_dfs_raw['test']['id'].str.split('_', expand=True)
all_dfs_raw['test']['month_str'] = test_id_split[0]
all_dfs_raw['test']['sector'] = test_id_split[1]
month_codes = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
for name, df in all_dfs_raw.items():
    if 'sector' in df.columns: df['sector_id'] = df['sector'].str.slice(7, None).astype(int)
    month_col = 'month' if 'month' in df.columns else 'month_str'
    if month_col in df.columns:
        df['year'] = df[month_col].str.slice(0, 4).astype(int)
        df['month_num'] = df[month_col].str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month_num'] - 1
sectors = np.arange(1, 97)
times = np.arange(all_dfs_raw['train_nht']['time'].min(), all_dfs_raw['test']['time'].max() + 1)
grid = pd.MultiIndex.from_product([sectors, times], names=['sector_id', 'time'])
df_full = pd.DataFrame(index=grid).reset_index()
df_full = pd.merge(df_full, all_dfs_raw['train_nht'][['sector_id', 'time', 'amount_new_house_transactions']], on=['sector_id', 'time'], how='left')
sector_mean = df_full.groupby('sector_id')['amount_new_house_transactions'].transform('mean')
df_full['amount_new_house_transactions'] = df_full['amount_new_house_transactions'].fillna(sector_mean)
df_full['amount_new_house_transactions'] = df_full['amount_new_house_transactions'].fillna(0)
print("Preprocessing complete.")

# --- 2. 特征工程 ---
print("\n--- 2. Feature Engineering ---")
df_full['month'] = df_full['time'] % 12 + 1
df_full['year'] = (df_full['time'] // 12) + 2019
df_full['date'] = pd.to_datetime(df_full['year'].astype(str) + '-' + df_full['month'].astype(str))
for k in range(1, 7):
    df_full[f'fourier_sin_{k}'] = np.sin(2 * np.pi * k * df_full['time'] / 12)
    df_full[f'fourier_cos_{k}'] = np.cos(2 * np.pi * k * df_full['time'] / 12)
sp_clean = all_dfs_raw['sp'].copy()
sp_clean.columns = [re.sub(r'[\W]+', '_', col) for col in sp_clean.columns]
poi_features = [col for col in sp_clean.columns if col not in ['sector_id', 'sector']]
sector_avg_amount = df_full.groupby('sector_id')['amount_new_house_transactions'].mean().reset_index()
sector_avg_amount.columns = ['sector_id', 'historical_avg_amount']
base_features_df = pd.merge(sp_clean, sector_avg_amount, on='sector_id', how='left').fillna(0)
base_model = lgb.LGBMRegressor(random_state=42)
base_model.fit(base_features_df[poi_features], base_features_df['historical_avg_amount'])
base_value_preds = base_model.predict(base_features_df[poi_features])
sector_base_value = pd.DataFrame({'sector_id': base_features_df['sector_id'], 'base_value': base_value_preds})
df_full = pd.merge(df_full, sector_base_value, on='sector_id', how='left')
csi_agg = all_dfs_raw['csi'].groupby('time').agg(csi_mean=('search_volume', 'mean')).reset_index()
csi_future = csi_agg.copy(); csi_future['time'] += 12; csi_future.rename(columns={'csi_mean': 'csi_mean_future'}, inplace=True)
csi_agg = pd.merge(csi_agg, csi_future[['time', 'csi_mean_future']], on='time', how='left')
csi_agg['csi_mean'] = csi_agg['csi_mean'].fillna(csi_agg['csi_mean_future'])
df_full = pd.merge(df_full, csi_agg[['time', 'csi_mean']], on='time', how='left')
ci_prefixed = all_dfs_raw['ci'].add_prefix('ci_')
df_full = pd.merge(df_full, ci_prefixed, left_on='year', right_on='ci_city_indicator_data_year', how='left')
for col in ci_prefixed.columns: df_full[col] = df_full[col].fillna(method='ffill')
df_full.drop(columns=['ci_city_indicator_data_year'], inplace=True)
df_full.fillna(0, inplace=True)
print("Feature engineering complete.")

# --- 3. 使用LGBM进行特征选择 ---
print("\n--- 3. Feature Selection with LightGBM ---")
df_train_fs = df_full[df_full['time'] < 67].copy()
exog_features = [col for col in df_full.columns if col not in ['sector_id', 'time', 'amount_new_house_transactions', 'date', 'year', 'month']]
fs_model = lgb.LGBMRegressor(random_state=42)
fs_model.fit(df_train_fs[exog_features], df_train_fs['amount_new_house_transactions'])
importances = pd.DataFrame({'feature': exog_features, 'importance': fs_model.feature_importances_}).sort_values('importance', ascending=False)
final_features = importances.head(15)['feature'].tolist()
print("Top 15 most important features selected:", final_features)

# --- 4. SARIMAX + Prophet 混合模型预测 ---
print("\n--- 4. Per-Sector Forecasting with Model Selection ---")
all_predictions = []
sectors_to_process = df_full['sector_id'].unique()

for i, sector in enumerate(sectors_to_process):
    print(f"\nProcessing Sector {sector} ({i+1}/{len(sectors_to_process)})...")
    sector_df = df_full[df_full['sector_id'] == sector].copy()
    sector_df.drop_duplicates(subset=['date'], inplace=True)
    sector_df.set_index('date', inplace=True)
    sector_df = sector_df.asfreq('MS')
    static_features = ['base_value']
    for col in static_features:
        if col in sector_df.columns:
            first_valid_idx = sector_df[col].first_valid_index()
            if first_valid_idx is not None:
                valid_value = sector_df.loc[first_valid_idx, col]
                sector_df[col].fillna(valid_value, inplace=True)
    sector_df['year'] = sector_df.index.year; sector_df['month'] = sector_df.index.month
    sector_df['time'] = (sector_df['year'] - 2019) * 12 + sector_df['month'] - 1
    sector_df.ffill(inplace=True); sector_df.bfill(inplace=True)
    sector_df.fillna(0, inplace=True)

    train_data = sector_df[sector_df['time'] < 67]
    test_data = sector_df[sector_df['time'] >= 67]

    # --- Cross-Validation ---
    print(f"--- Running Cross-Validation for Sector {sector} ---")
    cv = TimeSeriesSplit(n_splits=3, test_size=12)
    sarima_fold_scores, prophet_fold_scores, avg_fold_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(train_data)):
        cv_train, cv_val = train_data.iloc[train_idx], train_data.iloc[val_idx]
        if cv_train.empty or cv_val.empty or len(cv_train) < 6: continue

        y_cv_train, X_cv_train_exog = cv_train['amount_new_house_transactions'], cv_train[final_features]
        y_cv_val, X_cv_val_exog = cv_val['amount_new_house_transactions'], cv_val[final_features]

        # 6-Month Average Model
        avg_pred_cv = pd.Series([y_cv_train.tail(6).mean()] * len(cv_val), index=cv_val.index)
        avg_fold_scores.append(custom_score(y_cv_val, avg_pred_cv)['score'])

        # SARIMAX Model
        try:
            sarima_model_cv = SARIMAX(np.log1p(y_cv_train), exog=X_cv_train_exog, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            forecast_log_cv = sarima_model_cv.get_forecast(steps=len(cv_val), exog=X_cv_val_exog)
            sarima_preds_cv = np.expm1(forecast_log_cv.predicted_mean).replace([np.inf, -np.inf], y_cv_train.mean()).fillna(y_cv_train.mean())
            sarima_fold_scores.append(custom_score(y_cv_val, sarima_preds_cv)['score'])
        except Exception: sarima_fold_scores.append(0)

        # Prophet Model
        try:
            prophet_df_train_cv = cv_train[['amount_new_house_transactions']].reset_index().rename(columns={'date': 'ds', 'amount_new_house_transactions': 'y'})
            prophet_df_train_cv['y'] = np.log1p(prophet_df_train_cv['y'])
            prophet_df_train_cv = pd.concat([prophet_df_train_cv, X_cv_train_exog.reset_index(drop=True)], axis=1)
            model_cv = Prophet(); [model_cv.add_regressor(f) for f in final_features]
            model_cv.fit(prophet_df_train_cv, algorithm='LBFGS')
            future_df_cv = model_cv.make_future_dataframe(periods=len(cv_val), freq='MS')
            future_df_cv = pd.concat([future_df_cv, X_cv_val_exog.reset_index(drop=True)], axis=1).fillna(0)
            forecast_log_cv = model_cv.predict(future_df_cv)
            prophet_preds_cv = np.expm1(forecast_log_cv['yhat'].iloc[-len(cv_val):]).replace([np.inf, -np.inf], y_cv_train.mean()).fillna(y_cv_train.mean())
            prophet_fold_scores.append(custom_score(y_cv_val, prophet_preds_cv)['score'])
        except Exception: prophet_fold_scores.append(0)

    avg_sarima_score = np.mean(sarima_fold_scores) if sarima_fold_scores else 0
    avg_prophet_score = np.mean(prophet_fold_scores) if prophet_fold_scores else 0
    avg_6m_score = np.mean(avg_fold_scores) if avg_fold_scores else 0
    print(f"--- CV Results: SARIMAX={avg_sarima_score:.4f}, Prophet={avg_prophet_score:.4f}, 6m_Avg={avg_6m_score:.4f} ---")

    # --- Final Model Selection and Training ---
    models = {'SARIMAX': avg_sarima_score, 'Prophet': avg_prophet_score, '6m_Avg': avg_6m_score}
    best_model_name = max(models, key=models.get)
    best_model_score = models[best_model_name]

    models_sort = sorted(models, key=lambda x:models[x])
    second_model_score = models[models_sort[1]]
    y_train = train_data['amount_new_house_transactions']
    X_train_exog = train_data[final_features]
    X_test_exog = test_data[final_features]
    final_preds = None

    if best_model_score < 0.4 and best_model_score<1.5*second_model_score:
        print(f"-> Best model ({best_model_name}) score {best_model_score:.4f} is below threshold. Using 6-month average fallback.")
        best_model_name = '6m_Avg' # Force fallback

    if best_model_name == 'SARIMAX':
        print(f"-> Selecting SARIMAX for final prediction (Score: {best_model_score:.4f}).")
        try:
            final_model = SARIMAX(np.log1p(y_train), exog=X_train_exog, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            forecast_log = final_model.get_forecast(steps=len(test_data), exog=X_test_exog)
            final_preds = np.expm1(forecast_log.predicted_mean).replace([np.inf, -np.inf], y_train.mean()).fillna(y_train.mean())
        except Exception as e: print(f"  Final SARIMAX failed: {e}")

    elif best_model_name == 'Prophet':
        print(f"-> Selecting Prophet for final prediction (Score: {best_model_score:.4f}).")
        try:
            prophet_df_train = train_data[['amount_new_house_transactions']].reset_index().rename(columns={'date': 'ds', 'amount_new_house_transactions': 'y'})
            prophet_df_train['y'] = np.log1p(prophet_df_train['y'])
            prophet_df_train = pd.concat([prophet_df_train, X_train_exog.reset_index(drop=True)], axis=1)
            final_model = Prophet(); [final_model.add_regressor(f) for f in final_features]
            final_model.fit(prophet_df_train, algorithm='LBFGS')
            future_df = final_model.make_future_dataframe(periods=len(test_data), freq='MS')
            future_df = pd.concat([future_df, X_test_exog.reset_index(drop=True)], axis=1).fillna(0)
            forecast_log = final_model.predict(future_df)
            final_preds = np.expm1(forecast_log['yhat'].iloc[-len(test_data):]).replace([np.inf, -np.inf], y_train.mean()).fillna(y_train.mean())
        except Exception as e: print(f"  Final Prophet failed: {e}")

    if final_preds is None or best_model_name == '6m_Avg':
        if best_model_name == '6m_Avg': print(f"-> Selecting 6-month average for final prediction (Score: {best_model_score:.4f}).")
        fallback_value = y_train.tail(6).mean()
        final_preds = pd.Series([fallback_value] * len(test_data), index=test_data.index)

    sector_preds_df = pd.DataFrame({'sector_id': sector, 'time': test_data['time'], 'amount_new_house_transactions': final_preds.values})
    all_predictions.append(sector_preds_df)

print("\nAll sectors processed.")
df_test_pred = pd.concat(all_predictions)

# --- 5. 后处理与提交 ---
print("\n--- 5. Post-Processing & Submission ---")
df_test_pred['amount_new_house_transactions'] = df_test_pred['amount_new_house_transactions'].clip(lower=0)
df_train_fs = df_full[df_full['time'] < 67].copy() # Redefine for post-processing
last_6m_data = df_train_fs[df_train_fs['time'] >= 61]
zero_amount_sectors = last_6m_data.groupby('sector_id')['amount_new_house_transactions'].sum()
zero_amount_sectors = zero_amount_sectors[zero_amount_sectors == 0].index.tolist()
print(f"Found {len(zero_amount_sectors)} sectors with zero transactions in the last 6 months. Overriding their predictions to 0.")
df_test_pred.loc[df_test_pred['sector_id'].isin(zero_amount_sectors), 'amount_new_house_transactions'] = 0
df_test_pred['amount_new_house_transactions'].fillna(0, inplace=True)
submission_df = pd.merge(all_dfs_raw['test'][['id', 'sector_id', 'time']], df_test_pred, on=['sector_id', 'time'])
submission_df = submission_df[['id', 'amount_new_house_transactions']]
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully: submission.csv")
print(submission_df.head())

