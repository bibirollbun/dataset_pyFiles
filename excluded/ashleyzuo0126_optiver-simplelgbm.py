import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgbm
from lightgbm import early_stopping
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import gc
import warnings
warnings.filterwarnings('ignore')



# 设置随机种子以保证结果可复现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# 1. 数据加载
print("开始加载数据...")

def load_data():
    # 加载训练集
    train = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
    # 加载测试集
    test = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv')
    # 加载目标变量
    revealed_targets = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/revealed_targets.csv')
    
    return train, test, revealed_targets

train, test, revealed_targets = load_data()



# 2. 数据探索
print("开始数据探索...")

def explore_data(train, test, revealed_targets):
    print(f"训练集形状: {train.shape}")
    print(f"测试集形状: {test.shape}")
    print(f"已公开目标值形状: {revealed_targets.shape}")
    
    print("\n训练集列名:")
    print(train.columns.tolist())
    
    print("\n训练集数值统计描述:")
    print(train.describe())
    
    print("\n检查缺失值:")
    missing_train = train.isnull().sum().sum()
    print(f"训练集中的缺失值总数: {missing_train}")
    
    # 检查数据的时间序列特性
    print("\n数据的时间分布:")
    print(train['date_id'].value_counts().sort_index().head())

     # 检查目标值分布
    if 'target' in train.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(train['target'], bins=50)
        plt.title('目标变量分布')
        plt.xlabel('目标值')
        plt.ylabel('频率')
        plt.show()
    
    return
    
# 执行数据探索
explore_data(train, test, revealed_targets)


# 3. 特征工程
print("开始特征工程...")

def feature_engineering(df, is_train=True):
    """
    对数据进行特征工程
    """
    result_df = df.copy()
         
    # 基本特征：计算每个股票代码的基本统计量
    numeric_features = [col for col in result_df.columns if col not in ['stock_id', 'date_id', 'row_id', 'target', 'time_id']]
    
    # 时间特征
    result_df['day_of_week'] = result_df['date_id'] % 5  # 假设交易日为周一至周五
    
    # 统计特征 - 波动性和趋势
    for col in numeric_features:
        # 加入原始特征的幂次项
        result_df[f'{col}_squared'] = result_df[col] ** 2
        result_df[f'{col}_sqrt'] = np.sqrt(np.abs(result_df[col]))

    # 特征间的交互项
    # 选择一些重要的特征进行交互
    result_df['bid_plus_ask_sizes'] = result_df['bid_size'] + result_df['ask_size'] 
     
    result_df['imbalance_ratio'] = result_df['imbalance_size'] / result_df['matched_size']
    
    result_df['imb_s1'] = result_df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    result_df['imb_s2'] = result_df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')

    result_df['ask_x_size'] = result_df.eval('ask_size*ask_price')
    result_df['bid_x_size'] = result_df.eval('bid_size*bid_price')
        
    result_df['ask_minus_bid'] = result_df['ask_x_size'] - result_df['bid_x_size'] 
    
    result_df["bid_size_over_ask_size"] = result_df["bid_size"].div(df["ask_size"])
    result_df["bid_price_over_ask_price"] = result_df["bid_price"].div(df["ask_price"])
    
    # 历史滚动特征 (按date_id和stock_id分组)
    if 'date_id' in result_df.columns and 'stock_id' in result_df.columns:
        # 按时间和股票分组后排序
        result_df = result_df.sort_values(['date_id', 'stock_id'])
        
        # 按股票分组，计算滚动统计量
        for feat in numeric_features[:3]:  # 选择前3个重要特征
            # 按股票ID分组，计算移动平均
            result_df[f'{feat}_stock_rolling_mean'] = result_df.groupby('stock_id')[feat].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
            
            # 差分特征
            result_df[f'{feat}_diff_1'] = result_df.groupby('stock_id')[feat].diff(1)
    
    # 标准化某些特征
    for feat in numeric_features:
        mean = result_df[feat].mean()
        std = result_df[feat].std()
        result_df[f'{feat}_normalized'] = (result_df[feat] - mean) / (std + 1e-8)
    
    # 删除缺失值过多的列
    result_df = result_df.loc[:, result_df.isnull().mean() < 0.8]
    
    # 填充缺失值
    result_df = result_df.fillna(0)

    return result_df

# 应用特征工程
train_fe = feature_engineering(train, is_train=True)
test_fe = feature_engineering(test, is_train=False)

# 检查特征工程后的数据
print(f"特征工程后训练集形状: {train_fe.shape}")
print(f"特征工程后测试集形状: {test_fe.shape}")


# 4. 使用Optuna调优LGBM超参数

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna的目标函数，用于优化LGBM超参数
    """
    param = {
        'objective': 'mae',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'seed': RANDOM_SEED,
        
        # 超参数搜索空间
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
     # 训练模型
    model = lgbm.LGBMRegressor(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=50)]
    )
    
    # 预测并计算MAE
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    
    return mae


def optimize_hyperparameters(X, y, n_trials=20):
    """
    使用Optuna优化LGBM超参数
    """
    print("开始超参数优化...")
    
    # 使用时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 获取最后一次分割用于验证
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 创建Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials
    )
    
    print(f"最佳超参数: {study.best_params}")
    print(f"最佳MAE: {study.best_value}")
    
    return study.best_params


def train_and_evaluate_model(train_data, test_data, revealed_targets, best_params):
    """
    使用最佳超参数训练LGBM模型并评估
    """
    print("开始模型训练与评估...")
    
    # 准备特征和目标变量
    if 'target' in train_data.columns:
        y_train = train_data['target']
        X_train = train_data.drop(['target', 'row_id'], axis=1, errors='ignore')
    else:
        # 如果训练集没有目标列，可能需要从revealed_targets合并
        train_with_targets = train_data.merge(revealed_targets, on='row_id', how='left')
        y_train = train_with_targets['target']
        X_train = train_data.drop(['row_id'], axis=1, errors='ignore')
    
    # 准备测试集
    X_test = test_data.drop(['row_id'], axis=1, errors='ignore')
    
    # 移除非数值列
    non_numeric_cols = ['date_id', 'stock_id', 'time_id']
    X_train = X_train.drop(non_numeric_cols, axis=1, errors='ignore')
    X_test = X_test.drop(non_numeric_cols, axis=1, errors='ignore')
    
    # 确保训练集和测试集有相同的列
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    print(f"训练特征形状: {X_train.shape}")
    print(f"测试特征形状: {X_test.shape}")
    
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"训练折叠 {fold + 1}/5...")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 创建模型
        model = lgbm.LGBMRegressor(**best_params)
        
        # 训练模型
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(100),
                log_evaluation(100)  # 每100轮输出一次日志，等价于 verbose=100
            ]
        )
        
        # 预测验证集
        val_preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        cv_scores.append(mae)
        
        print(f"Fold {fold + 1} MAE: {mae}")
        
        # 绘制真实值vs预测值散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val, val_preds, alpha=0.5)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'折叠 {fold + 1} - 真实值 vs 预测值')
        plt.show()
        
        # 绘制特征重要性
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title(f'折叠 {fold + 1} - 前20个重要特征')
        plt.tight_layout()
        plt.show()
    
    print(f"平均交叉验证 MAE: {np.mean(cv_scores)}")
    print(f"交叉验证 MAE 标准差: {np.std(cv_scores)}")

    # 在整个训练集上训练最终模型
    final_model = lgbm.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    # 预测测试集
    test_preds = final_model.predict(X_test)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'row_id': test_data['row_id'],
        'target': test_preds
    })
    
    # 保存提交文件
    submission.to_csv('submission.csv', index=False)
    print("生成的提交文件已保存为 'submission.csv'")
    
    return final_model, submission, np.mean(cv_scores)


# 主流程
if __name__ == "__main__":
    # 只选取数值特征进行优化和训练
    feature_cols = [col for col in train_fe.columns if col not in ['row_id', 'target', 'date_id', 'stock_id', 'time_id']]
    X = train_fe[feature_cols]
    y = train_fe['target'] if 'target' in train_fe.columns else revealed_targets['target']
    
    # 优化超参数
    best_params = optimize_hyperparameters(X, y, n_trials=20)
    
    # 训练和评估模型
    final_model, submission, cv_mae = train_and_evaluate_model(train_fe, test_fe, revealed_targets, best_params)
    
    print(f"模型训练完成! 平均 MAE: {cv_mae}")
    

