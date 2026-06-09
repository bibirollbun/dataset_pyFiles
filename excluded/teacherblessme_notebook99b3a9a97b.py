import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 自定义RMSLE损失函数
def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def lgb_rmsle_loss(y_pred, y_true):
    y_true = y_true.get_label()
    y_pred = np.maximum(y_pred, 0)
    loss = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return 'RMSLE', loss, False

# 目标编码
def target_encoding(df, feat, target, cv=5):
    te_df = df.copy()
    tscv = TimeSeriesSplit(n_splits=cv)
    te_df[f'{feat}_te'] = np.nan
    for train_idx, val_idx in tscv.split(te_df):
        train_mean = te_df.iloc[train_idx].groupby(feat)[target].mean()
        te_df.iloc[val_idx, te_df.columns.get_loc(f'{feat}_te')] = te_df.iloc[val_idx][feat].map(train_mean)
    global_mean = te_df[target].mean()
    te_df[f'{feat}_te'] = te_df[f'{feat}_te'].fillna(global_mean)
    return te_df[f'{feat}_te']

# 数据加载与预处理
BASE_PATH = '/kaggle/input/bike-sharing-demand/'
train_df = pd.read_csv(f'{BASE_PATH}train.csv')
test_df = pd.read_csv(f'{BASE_PATH}test.csv')
submission_df = pd.read_csv(f'{BASE_PATH}sampleSubmission.csv')

df_all = pd.concat([train_df.assign(_is_train=1), test_df.assign(_is_train=0)], ignore_index=True)
df_all['datetime'] = pd.to_datetime(df_all['datetime'])

# 时序特征精细化
df_all['year'] = df_all['datetime'].dt.year
df_all['month'] = df_all['datetime'].dt.month
df_all['hour'] = df_all['datetime'].dt.hour
df_all['weekday'] = df_all['datetime'].dt.weekday
df_all['day'] = df_all['datetime'].dt.day
df_all['is_weekend'] = (df_all['weekday'] >= 5).astype(int)

df_all['month_sin'] = np.sin(2 * np.pi * df_all['month'] / 12)
df_all['month_cos'] = np.cos(2 * np.pi * df_all['month'] / 12)
df_all['weekday_sin'] = np.sin(2 * np.pi * df_all['weekday'] / 7)
df_all['weekday_cos'] = np.cos(2 * np.pi * df_all['weekday'] / 7)
df_all['hour_sin'] = np.sin(2 * np.pi * df_all['hour'] / 24)
df_all['hour_cos'] = np.cos(2 * np.pi * df_all['hour'] / 24)

# 特征交互
df_all['temp_humidity'] = df_all['temp'] * df_all['humidity']
df_all['hour_workingday'] = df_all['hour'] * df_all['workingday']
df_all['season_temp'] = df_all['season'] * df_all['temp']
df_all['wind_humidity'] = df_all['windspeed'] * df_all['humidity']

# 异常值处理
train_mask = df_all['_is_train'] == 1
count_mean = df_all.loc[train_mask, 'count'].mean()
count_std = df_all.loc[train_mask, 'count'].std()
df_all = df_all[~(train_mask & (df_all['count'] > count_mean + 3 * count_std))]

df_all.loc[df_all['humidity'] == 0, 'humidity'] = df_all['humidity'].median()
df_wind_not_zero = df_all[df_all['windspeed'] > 0].copy()
df_wind_zero = df_all[df_all['windspeed'] == 0].copy()
if not df_wind_zero.empty and not df_wind_not_zero.empty:
    rf_wind = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    wind_cols = ['temp', 'season', 'weather', 'humidity', 'month', 'hour']
    rf_wind.fit(df_wind_not_zero[wind_cols], df_wind_not_zero['windspeed'])
    df_all.loc[df_all['windspeed'] == 0, 'windspeed'] = rf_wind.predict(df_wind_zero[wind_cols])

# 目标编码
train_df_te = df_all[df_all['_is_train'] == 1].copy()
test_df_te = df_all[df_all['_is_train'] == 0].copy()
train_df_te['hour_weekday_te'] = target_encoding(train_df_te, 'hour_workingday', 'count')
hour_weekday_mean = train_df_te.groupby('hour_workingday')['count'].mean()
test_df_te['hour_weekday_te'] = test_df_te['hour_workingday'].map(hour_weekday_mean).fillna(train_df_te['count'].mean())
df_all = pd.concat([train_df_te, test_df_te], ignore_index=True)

# 类别特征处理
categorical_features = ['year', 'month', 'hour', 'weekday', 'season', 'weather', 'holiday', 'workingday', 'is_weekend']
for col in categorical_features:
    df_all[col] = df_all[col].astype(int)

# 冗余特征移除
drop_cols = ['atemp', 'datetime', 'day', 'casual', 'registered']
df_all = df_all.drop(drop_cols, axis=1, errors='ignore')

# 数据拆分
X_train = df_all[df_all['_is_train'] == 1].drop(['_is_train', 'count'], axis=1)
y_train = df_all[df_all['_is_train'] == 1]['count']
X_test = df_all[df_all['_is_train'] == 0].drop(['_is_train', 'count'], axis=1)

y_train_log = np.log1p(y_train)

tscv = TimeSeriesSplit(n_splits=5)

# 模型参数
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# 初始化预测结果
lgb_oof = np.zeros(len(X_train))
lgb_test_pred = np.zeros(len(X_test))
ridge_oof = np.zeros(len(X_train))
ridge_test_pred = np.zeros(len(X_test))

print("开始时序交叉验证训练")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
    
    # LightGBM训练
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False), lgb.log_evaluation(0)]
    )
    lgb_oof[val_idx] = lgb_model.predict(X_val)
    lgb_test_pred += lgb_model.predict(X_test) / tscv.n_splits
    
    # Ridge回归训练
    ridge_model = Ridge(alpha=50, random_state=42)
    ridge_model.fit(X_tr, y_tr)
    ridge_oof[val_idx] = ridge_model.predict(X_val)
    ridge_test_pred += ridge_model.predict(X_test) / tscv.n_splits
    
    val_rmsle = rmsle(np.expm1(y_val), np.expm1(lgb_oof[val_idx]))
    print(f"Fold {fold+1} - LightGBM 验证集RMSLE: {val_rmsle:.4f}")

# 输出各模型整体验证分数
lgb_rmsle = rmsle(np.expm1(y_train_log), np.expm1(lgb_oof))
ridge_rmsle = rmsle(np.expm1(y_train_log), np.expm1(ridge_oof))
print(f"\n模型验证分数")
print(f"LightGBM RMSLE: {lgb_rmsle:.4f}")
print(f"Ridge RMSLE: {ridge_rmsle:.4f}")

# 智能模型融合
def blend_weights(weights):
    w1, w2 = weights
    blended_oof = w1 * lgb_oof + w2 * ridge_oof
    return rmsle(np.expm1(y_train_log), np.expm1(blended_oof))

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1), (0, 1)]
init_weights = [0.8, 0.2]
opt_result = minimize(blend_weights, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_w1, best_w2 = opt_result.x

print(f"\n最优融合权重")
print(f"LightGBM: {best_w1:.3f}, Ridge: {best_w2:.3f}")

# 融合测试集预测结果
blended_test_pred = best_w1 * lgb_test_pred + best_w2 * ridge_test_pred
final_pred = np.expm1(blended_test_pred)
final_pred = np.maximum(final_pred, 0).astype(int)

# 生成提交文件
submission = pd.DataFrame({
    'datetime': test_df['datetime'],
    'count': final_pred
})
submission.to_csv('submission_optimized.csv', index=False)

print(f"\n✅ 优化完成！最终提交文件已生成：submission_optimized.csv")
print(f"融合后验证集RMSLE: {blend_weights([best_w1, best_w2]):.4f}")

