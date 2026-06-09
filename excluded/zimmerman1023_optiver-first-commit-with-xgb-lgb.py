import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = '/kaggle/input/optiver-trading-at-the-close'

train_data = pd.read_csv(f'{DATA_PATH}/train.csv')

test_data = pd.read_csv(f'{DATA_PATH}/example_test_files/test.csv')
revealed_targets = pd.read_csv(f'{DATA_PATH}/example_test_files/revealed_targets.csv')
sample_submit = pd.read_csv(f'{DATA_PATH}/example_test_files/sample_submission.csv')


train_data.shape


train_data[train_data['stock_id'] == 0]


test_data.head()


revealed_targets.head()


sample_submit.head()


train_data.info()


revealed_targets.info()


revealed_targets.info()


train_data.isnull().mean()


plt.figure(figsize=(10, 7))
sns.heatmap(train_data.iloc[:, :-1].corr())


train_data.iloc[:, :-1].corr().target.sort_values().iloc[:-1]


train_data[train_data['stock_id'] == 0]['ask_size'].reset_index(drop=True).plot(figsize=(10, 7))


(
    train_data
    .query('stock_id ==0 & date_id ==0')
    [['seconds_in_bucket','bid_price','ask_price', 'wap']]
    .replace(0, np.nan)
    .set_index('seconds_in_bucket')
    .plot(title='Stock 0 on Day 0 - How the order book pricing changes during the auction',
         figsize=(10, 7))
)


import polars as pl
import pandas as pd
from itertools import combinations
import gc

# 定义权重 DataFrame
weight_df = pd.DataFrame()
weight_df['stock_id'] = list(range(200))
weight_df['weight'] = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
    0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
    0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
    0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
    0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
    0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
    0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
    0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
    0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
    0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
    0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
    0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
    0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
    0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
    0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
    0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
]
# 转换为 Polars 以便后续 Join
weight_df_pl = pl.from_pandas(weight_df)
# 修复: 显式转换 stock_id 为 Int16 以匹配主 DataFrame
weight_df_pl = weight_df_pl.with_columns(pl.col("stock_id").cast(pl.Int16))

# 预先计算好的全局中位数 (基于训练集)
global_median_values = None

def generate_features(df, global_medians=None):
    # 1. 输入处理：如果是 Pandas，转为 Polars
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    
    # 2. 启用 Lazy 模式 (关键优化：允许 Polars 优化查询计划并减少内存使用)
    if isinstance(df, pl.DataFrame):
        df = df.lazy()
        
    # 3. 类型强制转换 (修复 ComputeError 和 内存优化)
    # 显式将所有数值列转换为 Float32。这解决了 "arithmetic on string and numeric" 错误，
    # 并确保内存占用最小。
    numeric_cols = ['imbalance_size', 'reference_price', 'matched_size', 'far_price', 'near_price', 
                    'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap', 'target', 
                    'imbalance_buy_sell_flag', 'seconds_in_bucket']
    
    # 获取当前 DataFrame 的列名 (LazyFrame 也有 columns 属性)
    current_cols = df.columns
    
    # 批量转换存在的数值列
    cast_exprs = []
    for col in numeric_cols:
        if col in current_cols:
            cast_exprs.append(pl.col(col).cast(pl.Float32, strict=False))
            
    if 'stock_id' in current_cols:
        cast_exprs.append(pl.col('stock_id').cast(pl.Int16))
        
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # 合并权重
    if "weight" not in current_cols:
        df = df.join(weight_df_pl.lazy(), on=['stock_id'], how='left')
    
    # 定义特征列表
    feas_list = ['stock_id','seconds_in_bucket','imbalance_size','imbalance_buy_sell_flag',
               'reference_price','matched_size','far_price','near_price','bid_price','bid_size',
                'ask_price','ask_size','wap','scale_imbalance_size','scale_matched_size','scale_bid_size','scale_ask_size'
                 ,'auc_bid_size','auc_ask_size']
    
    # --- 使用传入的全局中位数进行 Scale ---
    size_cols = ['imbalance_size','matched_size','bid_size','ask_size']
    
    if global_medians is not None:
        # 如果 global_medians 是 DataFrame，转为 Lazy
        if isinstance(global_medians, pl.DataFrame):
            global_medians = global_medians.lazy()
        
        # global_medians 的 stock_id 也是 Int16
        global_medians = global_medians.with_columns(pl.col("stock_id").cast(pl.Int16))
            
        df = df.join(global_medians, on="stock_id", how="left")
        df = df.with_columns([
            (pl.col(col) / pl.col(f"median_{col}")).alias(f"scale_{col}") 
            for col in size_cols
        ])
    else:
        df = df.with_columns([
            (pl.col(col) / pl.col(col).median().over('stock_id')).alias(f"scale_{col}") 
            for col in size_cols
        ])
    
    # auc_bid_size / auc_ask_size 初始化
    df = df.with_columns([
        pl.col('matched_size').alias('auc_bid_size'),
        pl.col('matched_size').alias('auc_ask_size')
    ])
    
    # 根据 imbalance flag 调整
    df = df.with_columns([
        pl.when(pl.col('imbalance_buy_sell_flag') == 1)
          .then(pl.col('auc_bid_size') + pl.col('imbalance_size'))
          .otherwise(pl.col('auc_bid_size')).alias('auc_bid_size'),
        pl.when(pl.col('imbalance_buy_sell_flag') == -1)
          .then(pl.col('auc_ask_size') + pl.col('imbalance_size'))
          .otherwise(pl.col('auc_ask_size')).alias('auc_ask_size')
    ])

    # 基础特征
    df = df.with_columns([
        (pl.col('ask_size') * pl.col('ask_price')).alias("ask_money"),
        (pl.col('bid_size') * pl.col('bid_price')).alias("bid_money"),
        (pl.col('ask_size') + pl.col("auc_ask_size")).alias("ask_size_all"),
        (pl.col('bid_size') + pl.col("auc_bid_size")).alias("bid_size_all"),
        (pl.col('ask_size') + pl.col("auc_ask_size") + pl.col('bid_size') + pl.col("auc_bid_size")).alias("volumn_size_all"),
        (pl.col('reference_price') * pl.col('auc_ask_size')).alias("ask_auc_money"),
        (pl.col('reference_price') * pl.col('auc_bid_size')).alias("bid_auc_money"),
        (pl.col('ask_size') * pl.col('ask_price') + pl.col('bid_size') * pl.col('bid_price')).alias("volumn_money"),
        (pl.col('ask_size') + pl.col('bid_size')).alias('volume_cont'),
        (pl.col('ask_size') - pl.col('bid_size')).alias('diff_ask_bid_size'),
        (pl.col('imbalance_size') + 2 * pl.col('matched_size')).alias('volumn_auc'),
        ((pl.col('imbalance_size') + 2 * pl.col('matched_size')) * pl.col("reference_price")).alias('volumn_auc_money'),
        ((pl.col('ask_price') + pl.col('bid_price'))/2).alias('mid_price'),
        ((pl.col('near_price') + pl.col('far_price'))/2).alias('mid_price_near_far'),
        (pl.col('ask_price') - pl.col('bid_price')).alias('price_diff_ask_bid'),
        (pl.col('ask_price') / pl.col('bid_price')).alias('price_div_ask_bid'),
        (pl.col('imbalance_buy_sell_flag') * pl.col('scale_imbalance_size')).alias('flag_scale_imbalance_size'),
        (pl.col('imbalance_buy_sell_flag') * pl.col('imbalance_size')).alias('flag_imbalance_size'),
        (pl.col('imbalance_size') / pl.col('matched_size') * pl.col('imbalance_buy_sell_flag')).alias("div_flag_imbalance_size_2_balance"),
        ((pl.col('ask_price') - pl.col('bid_price')) * pl.col('imbalance_size')).alias('price_pressure'),
        ((pl.col('ask_price') - pl.col('bid_price')) * pl.col('imbalance_size') * pl.col('imbalance_buy_sell_flag')).alias('price_pressure_v2'),
        ((pl.col("ask_size") - pl.col("bid_size")) / (pl.col("far_price") - pl.col("near_price"))).alias("depth_pressure"),
        (pl.col("bid_size") / pl.col("ask_size")).alias("div_bid_size_ask_size"),
    ])
    
    feas_list.extend(['ask_money', 'bid_money', 'ask_auc_money','bid_auc_money',"ask_size_all","bid_size_all","volumn_size_all",
                      'volumn_money','volume_cont',"volumn_auc","volumn_auc_money","mid_price",
                      'mid_price_near_far','price_diff_ask_bid',"price_div_ask_bid","flag_imbalance_size","div_flag_imbalance_size_2_balance",
                     "price_pressure","price_pressure_v2","depth_pressure","flag_scale_imbalance_size","diff_ask_bid_size"])

    # 各种 Ratio
    add_cols = []
    for col1, col2 in [
        ("imbalance_size","bid_size"),
        ("imbalance_size","ask_size"),
        ("matched_size","bid_size"),
        ("matched_size","ask_size"),
        ("imbalance_size","volume_cont"),
        ("matched_size","volume_cont"),
        ("auc_bid_size","bid_size"),
        ("auc_ask_size","ask_size"),
        ("bid_auc_money","bid_money"),
        ("ask_auc_money","ask_money"),
    ]:
        add_cols.append((pl.col(col1) / pl.col(col2)).alias(f"div_{col1}_2_{col2}"))
        feas_list.append(f"div_{col1}_2_{col2}")        
    df = df.with_columns(add_cols)

    # Imbalance 特征
    add_cols = []
    for pair1,pair2 in [
        ('ask_size','bid_size'),
        ('ask_money','bid_money'),
        ('volumn_money','volumn_auc_money'),
        ('volume_cont','volumn_auc'),
        ('imbalance_size','matched_size'),
        ('auc_ask_size','auc_bid_size'),
        ("ask_size_all",'bid_size_all')
    ]:
        col_imb = f"imb1_{pair1}_{pair2}"
        add_cols.extend([
            ((pl.col(pair1) - pl.col(pair2)) / (pl.col(pair1) + pl.col(pair2))).alias(col_imb),
        ])
        feas_list.extend([col_imb])
    df = df.with_columns(add_cols)
    
    # Price Imbalance
    fea_append_list = []
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap","mid_price"]
    for c in combinations(prices, 2):
        fea_append_list.append(((pl.col(c[0]) - pl.col(c[1])) / (pl.col(c[0]) + pl.col(c[1]))).alias(f"imb1_{c[0]}_{c[1]}"))
        feas_list.extend([f"imb1_{c[0]}_{c[1]}"])
    df = df.with_columns(fea_append_list)
    
    # Market Urgency
    df = df.with_columns([
        ((pl.col("imb1_ask_size_bid_size") + 2) * (pl.col("imb1_ask_price_bid_price") + 2) * (pl.col("imb1_auc_ask_size_auc_bid_size")+2)).alias("market_urgency_v2"),
        (pl.col('price_diff_ask_bid') * (pl.col('imb1_ask_size_bid_size'))).alias('market_urgency'),
        (pl.col('imb1_ask_price_bid_price') * (pl.col('imb1_ask_size_bid_size'))).alias('market_urgency_v3'),
    ])
    feas_list.extend([f"market_urgency_v3",'market_urgency','market_urgency_v2'])
    
    # Rolling Features (需要历史数据)
    # 极度简化: 只保留一个窗口 [10] 以最大化节省内存
    add_cols = []
    for col in ["bid_auc_money","imb1_reference_price_wap","bid_size_all",
                "imb1_auc_ask_size_auc_bid_size","div_flag_imbalance_size_2_balance",
                "imb1_ask_size_all_bid_size_all","flag_imbalance_size","imb1_reference_price_mid_price"]:
        for window in [10]: # 极度简化
            add_cols.append(pl.col(col).rolling_mean(window_size=window,min_periods=1).over('stock_id','date_id').alias(f'rolling{window}_mean_{col}'))
            add_cols.append(pl.col(col).rolling_std(window_size=window,min_periods=1).over('stock_id','date_id').alias(f'rolling{window}_std_{col}'))
            feas_list.extend([f'rolling{window}_mean_{col}',f'rolling{window}_std_{col}'])
    
    df = df.with_columns(add_cols)
    
    # Diff / Momentum
    df = df.with_columns([
        pl.col("flag_imbalance_size").diff().over('stock_id','date_id').alias("imbalance_momentum_unscaled"),
        pl.col("price_diff_ask_bid").diff().over('stock_id','date_id').alias("spread_intensity"),
    ])
    feas_list.extend(["imbalance_momentum_unscaled","spread_intensity"])
    
    df = df.with_columns([
        (pl.col("imbalance_momentum_unscaled")/pl.col("matched_size")).alias("imbalance_momentum")
    ])
    feas_list.extend(["imbalance_momentum"])

    # Diff features
    add_cols = []
    for col in ['ask_price', 'bid_price', 'imb1_reference_price_near_price', 'bid_size', 'scale_bid_size', 
                'mid_price', 'ask_size', 'price_div_ask_bid', 'div_bid_size_ask_size', 'market_urgency', 
                'wap', 'imbalance_momentum']:
        for window in [10]: # 极度简化
            add_cols.append((pl.col(col).diff(window).over('stock_id','date_id')).alias(f"{col}_diff_{window}"))
            feas_list.append(f"{col}_diff_{window}")
    df = df.with_columns(add_cols)
    
    # --- 修复2: 纯 Polars 实现 Target Mock (去除 Pandas 转换) ---
    for mock_period in [6]: # 极度简化
        df = df.with_columns([
            pl.col("wap").shift(-mock_period).over("stock_id","date_id").alias(f"wap_shift_n{mock_period}")
        ])
        df = df.with_columns([
            (pl.col(f"wap_shift_n{mock_period}")/pl.col("wap")).alias("target_single")
        ])
        
        # Weight handling: 如果 target_single 是 null，weight 设为 0
        df = df.with_columns([
            pl.when(pl.col("target_single").is_null())
              .then(0.0)
              .otherwise(pl.col("weight"))
              .alias("weight_tmp")
        ])

        df = df.with_columns([
            (((pl.col("weight_tmp") * pl.col("target_single")).sum().over("date_id","seconds_in_bucket")) / ((pl.col("weight_tmp")).sum().over("date_id","seconds_in_bucket"))).alias("index_target_mock")
        ])

        df = df.with_columns([
            ((pl.col("target_single") - pl.col("index_target_mock"))*10000).alias("target_mock")
        ])

        df = df.with_columns([
            pl.col("target_mock").shift(mock_period).over("stock_id","date_id").alias(f"target_mock_shift{mock_period}"),
        ])
        
    add_cols = []
    for col in ['target_mock_shift6']:
        for window in [10]: # 极度简化
            add_cols.append(pl.col(col).rolling_mean(window_size=window,min_periods=1).over('stock_id','date_id').alias(f'rolling{window}_mean_{col}'))
    df = df.with_columns(add_cols)
    
    keep_cols_new = ['rolling10_mean_target_mock_shift6'] 
    feas_list.extend(keep_cols_new)
    
    # Shift / Div Shift / Diff Shift
    add_cols = []
    for col in ["imb1_auc_ask_size_auc_bid_size","flag_imbalance_size","price_pressure_v2","scale_matched_size"]:
        for window_size in [6]: # 极度简化
            add_cols.append(pl.col(col).shift(window_size).over('stock_id','date_id').alias(f'shift{window_size}_{col}'))
            add_cols.append((pl.col(col) / pl.col(col).shift(window_size).over('stock_id','date_id')).alias(f'div_shift{window_size}_{col}'))
            add_cols.append((pl.col(col) - pl.col(col).shift(window_size).over('stock_id','date_id')).alias(f'diff_shift{window_size}_{col}'))
    
    feas_list.extend(['div_shift6_imb1_auc_ask_size_auc_bid_size', 'diff_shift6_price_pressure_v2', 
     'diff_shift6_flag_imbalance_size', 'shift6_flag_imbalance_size'])
    df = df.with_columns(add_cols)
    
    # Global Features (移除以节省内存)
    # MACD (移除以节省内存)
    
    # 4. 返回 LazyFrame，由调用者决定何时 collect()
    return df, feas_list

TRAINING = True
if TRAINING:
    # 5. 内存优化：使用 scan_csv 惰性读取数据
    # scan_csv 不会立即读取文件，而是创建一个执行计划
    q = pl.scan_csv(f'{DATA_PATH}/train.csv')
    
    # 过滤 (Lazy)
    q = q.filter(pl.col('target').is_not_null())
    
    # --- 计算全局中位数 (需要部分 collect) ---
    # 为了计算中位数，我们需要执行聚合。这里我们只读取需要的列。
    size_cols = ['imbalance_size','matched_size','bid_size','ask_size']
    
    # 创建一个专门用于计算中位数的 LazyFrame 分支
    median_q = q.select(['stock_id'] + size_cols)
    
    global_medians_pl = median_q.group_by('stock_id').agg([
        pl.col(col).median().alias(f"median_{col}") for col in size_cols
    ]).collect() # 这里必须 collect，因为后续 join 需要它 (或者作为 LazyFrame 传入)
    
    # --- 保存为 CSV ---
    global_medians_pl.write_csv('global_medians.csv')
    print("global_medians.csv saved locally.")

    # 生成特征 (传入 LazyFrame q)
    # generate_features 会返回一个 LazyFrame，包含了所有特征生成的计算图
    df_lazy, features = generate_features(q, global_medians=global_medians_pl)
    features = list(set(features))
    print(f"Feature count: {len(features)}")
    
    # 6. 内存优化：只选择需要的列并执行 (Collect)
    # 只有在这里，Polars 才会真正读取数据并计算特征。
    # 优化器会确保只读取和计算生成这些列所需的数据。
    df_ = df_lazy.select(features + ['target']).collect()
    
    # 显式清理
    del q
    gc.collect()


df_.head()


import lightgbm as lgb 
import xgboost as xgb 
import numpy as np 
import joblib 
import os
import gc
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

if not os.path.exists('models'):
    os.makedirs('models')

N_fold = 5

# --- PyTorch 模型定义 ---
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layers(x)

class PyTorchRegressor:
    def __init__(self, model_class, device='cuda' if torch.cuda.is_available() else 'cpu', epochs=10, batch_size=2048, lr=0.001):
        self.model_class = model_class
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.scaler = None # 保存 scaler 以便推理使用
        
    def fit(self, X, y, eval_set=None, **kwargs):
        input_dim = X.shape[1]
        self.model = self.model_class(input_dim).to(self.device)
        criterion = nn.L1Loss() # MAE
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation
        X_val, y_val = None, None
        if eval_set:
            X_val = torch.tensor(eval_set[0][0], dtype=torch.float32).to(self.device)
            y_val = torch.tensor(eval_set[0][1], dtype=torch.float32).view(-1, 1).to(self.device)
            
        print(f"Training PyTorch Model on {self.device}...")
        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Detailed logging
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val)
                    val_loss = criterion(val_preds, y_val)
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f}")
                
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
        return preds

# --- 训练逻辑 ---

if TRAINING and 'df_' in globals():
    print("Starting memory-optimized training...")
    
    # 0. 确保 target 没有 NaN 或 Inf (关键修复)
    df_ = df_.filter(pl.col("target").is_not_null() & pl.col("target").is_finite())

    # 1. 添加 fold 列
    try:
        df_ = df_.with_row_index("index")
    except AttributeError:
        df_ = df_.with_row_count("index")
        
    df_ = df_.with_columns((pl.col("index") % N_fold).alias("fold"))
    
    # 2. 将数据分块保存到磁盘
    print("Partitioning data to disk...")
    for fold in range(N_fold):
        fold_file = f"train_fold_{fold}.parquet"
        df_.filter(pl.col("fold") == fold).select(features + ['target']).write_parquet(fold_file)
        print(f"Saved {fold_file}")
        
    # 3. 彻底清理内存
    del df_
    gc.collect()
    print("Full dataset removed from memory.")
elif TRAINING:
    print("df_ not found in memory, assuming data already partitioned to disk.")

models = []
model_scores = {} # 记录每个模型的 CV 分数
last_fold_data = {} # 记录最后一个 Fold 的数据用于可视化

# 配置模型字典 (满足课程要求)
model_dict = {
    'lgb': lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, device='gpu'),
    'xgb': xgb.XGBRegressor(tree_method='gpu_hist', objective='reg:absoluteerror', n_estimators=500, early_stopping_rounds=100),
    'rf': lgb.LGBMRegressor(boosting_type='rf', bagging_freq=1, bagging_fraction=0.7, feature_fraction=0.7, n_estimators=100, device='gpu'),
    'nn_simple': PyTorchRegressor(SimpleNN, epochs=10, batch_size=2048),
    'nn_deep': PyTorchRegressor(DeepNN, epochs=10, batch_size=2048)
}

# 初始化分数记录
for k in model_dict.keys():
    model_scores[k] = []

def train_model(model_name, i, X_train, y_train, X_valid, y_valid):
    print(f"Training {model_name} - Fold {i}")
    model = model_dict[model_name]
    
    # 确保 y 没有异常值
    y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0).astype(np.float32)
    y_valid = np.nan_to_num(y_valid, nan=0, posinf=0, neginf=0).astype(np.float32)
    
    # 区分处理 GBDT/RF 和 NN
    if 'nn' in model_name:
        # NN 对数值敏感，不能用 -9e9
        # 1. 填充 NaN 为 0
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # 2. 标准化 (StandardScaler)
        # 注意：这里每次 Fold 都会重新 fit scaler，这是正确的。
        # 但在推理时，我们需要保存这个 scaler。
        # 为了简化，我们假设 PyTorchRegressor 内部可以处理，或者我们在这里处理。
        # 由于 PyTorchRegressor 是我们自定义的类，我们可以把 scaler 存进去。
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        
        # 将 scaler 绑定到 model 实例上，以便后续推理使用 (虽然这里是 list 中的 model，但引用是同一个)
        model.scaler = scaler
        
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    else:
        # 树模型对数值不敏感，可以用 -9e9 标记缺失值
        X_train = np.nan_to_num(X_train, nan=np.float32(-9e9), posinf=np.float32(9e9), neginf=np.float32(-9e9))
        X_valid = np.nan_to_num(X_valid, nan=np.float32(-9e9), posinf=np.float32(9e9), neginf=np.float32(-9e9))
        
        # 使用 callbacks 替代 deprecated 参数 (针对 LGBM)
        if isinstance(model, lgb.LGBMRegressor):
            callbacks = [
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
            model.fit(X_train, y_train, 
                      eval_set=[(X_valid, y_valid)], 
                      callbacks=callbacks
                     )
        else:
            # XGBoost
            model.fit(X_train, y_train, 
                  eval_set=[(X_valid, y_valid)], 
                  verbose=100
                 )
    
    # 计算验证集分数
    # 注意：如果是 NN，predict 内部需要用到 scaler 吗？
    # 我们的 PyTorchRegressor.predict 接收的是 X。
    # 如果我们在外部做了 scaling，传进去的 X_valid 已经是 scaled 的了。
    # 所以这里直接 predict(X_valid) 是没问题的。
    val_preds = model.predict(X_valid)
    score = mean_absolute_error(y_valid, val_preds)
    print(f"{model_name} Fold {i} MAE: {score:.4f}")
    
    # 保存模型 (如果是 NN，还需要保存 scaler)
    if 'nn' in model_name:
        # 保存整个对象 (包含 scaler)
        joblib.dump(model, f'./models/{model_name}_{i}.model')
    else:
        joblib.dump(model, f'./models/{model_name}_{i}.model')
        
    return model, score, val_preds

if TRAINING:
    for i in range(N_fold):
        print(f"\n=== Processing Fold {i} ===")
        
        # --- 加载验证集 ---
        print(f"Loading validation fold {i}...")
        valid_df = pl.read_parquet(f"train_fold_{i}.parquet")
        valid_df = valid_df.filter(pl.col("target").is_not_null() & pl.col("target").is_finite())
        X_valid = valid_df.select(features).to_numpy()
        y_valid = valid_df.select('target').to_numpy().flatten()
        del valid_df
        gc.collect()
        
        # 记录最后一个 Fold 的真实值用于可视化
        if i == N_fold - 1:
            last_fold_data['y_true'] = y_valid
            last_fold_data['preds'] = {}
        
        # --- 加载训练集 ---
        print(f"Loading training folds...")
        train_files = [f"train_fold_{f}.parquet" for f in range(N_fold) if f != i]
        
        q = pl.concat([pl.scan_parquet(f) for f in train_files])
        q = q.filter(pl.col("target").is_not_null() & pl.col("target").is_finite())
        train_df = q.collect()
        
        X_train = train_df.select(features).to_numpy()
        y_train = train_df.select('target').to_numpy().flatten()
        del train_df
        gc.collect()
        
        # --- 训练所有模型 ---
        for model_type in model_dict.keys():
             # 注意：我们需要为每个 fold 创建新的模型实例，或者重置模型
             # 对于 sklearn 接口的 LGB/XGB，fit 会重置。
             # 对于我们的 PyTorchRegressor，fit 会重新初始化 model。
             # 但是为了安全起见，最好是 clone 一份，或者确保 fit 内部重置。
             # 我们的 PyTorchRegressor.fit 确实重新创建了 self.model = ...
             # 所以是可以复用的。
             
             model, score, preds = train_model(model_type, i, X_train, y_train, X_valid, y_valid)
             
             # 注意：这里 models 列表可能会存同一个对象的引用
             # 如果我们复用 model_dict 中的对象，models 列表里全是同一个对象
             # 这在后续分析 feature importance 时没问题 (取最后一个状态)
             # 但如果我们要保存所有 fold 的模型，最好是 copy
             import copy
             models.append(copy.deepcopy(model))
             
             model_scores[model_type].append(score)
             
             # 记录最后一个 Fold 的预测值
             if i == N_fold - 1:
                 last_fold_data['preds'][model_type] = preds
        
        # --- 清理当前 Fold 的数据 ---
        del X_train, y_train, X_valid, y_valid
        gc.collect()
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # --- 输出最终评估结果 ---
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS (MAE)")
    print("="*30)
    results_df = pd.DataFrame(model_scores)
    results_df.index = [f'Fold {i}' for i in range(N_fold)]
    results_df.loc['Average'] = results_df.mean()
    print(results_df)
    print("="*30)
        
else:
    # 推理模式下加载模型
    for i in range(N_fold):
        for model_type in model_dict.keys():
            models.append(joblib.load(f'models/{model_type}_{i}.model'))

# 清理临时文件 (可选)
if TRAINING:
    for fold in range(N_fold):
        try:
            os.remove(f"train_fold_{fold}.parquet")
        except:
            pass


# --- 结果可视化与分析 ---
if TRAINING and 'last_fold_data' in globals() and last_fold_data:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("Generating Evaluation Plots...")
    
    # 1. 模型性能对比 (Bar Plot)
    plt.figure(figsize=(10, 6))
    avg_scores = pd.DataFrame(model_scores).mean()
    sns.barplot(x=avg_scores.index, y=avg_scores.values)
    plt.title('Average MAE by Model (Lower is Better)')
    plt.ylabel('MAE')
    plt.xlabel('Model Type')
    plt.show()
    
    # 2. 特征重要性 (仅针对 LGBM)
    # 获取第一个 LGBM 模型 (Fold 0)
    lgb_model = [m for m in models if isinstance(m, lgb.LGBMRegressor)][0]
    if hasattr(lgb_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': lgb_model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Top 20 Feature Importance (LightGBM)')
        plt.show()
        
    # 3. 预测值分布对比 (Density Plot)
    # 使用最后一个 Fold 的数据
    y_true = last_fold_data['y_true']
    preds_dict = last_fold_data['preds']
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(y_true, label='True Target', fill=True, alpha=0.3, color='black')
    
    for model_name, preds in preds_dict.items():
        sns.kdeplot(preds, label=f'{model_name} Preds', alpha=0.3)
        
    plt.title('Distribution of Predictions vs True Target (Last Fold)')
    plt.xlim(-100, 100) # 限制范围以便观察核心分布
    plt.legend()
    plt.show()
    
    # 4. 散点图 (True vs Preds - Sample)
    # 为了避免点太多，只随机采样 1000 个点
    sample_idx = np.random.choice(len(y_true), 1000, replace=False)
    
    fig, axes = plt.subplots(1, len(preds_dict), figsize=(20, 4))
    if len(preds_dict) == 1: axes = [axes]
    
    for ax, (model_name, preds) in zip(axes, preds_dict.items()):
        ax.scatter(y_true[sample_idx], preds[sample_idx], alpha=0.5, s=10)
        ax.plot([-100, 100], [-100, 100], 'r--') # 对角线
        ax.set_title(f'{model_name} vs True')
        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        
    plt.tight_layout()
    plt.show()


import optiver2023
import pandas as pd
import numpy as np
import os

# 1. 初始化环境
env = optiver2023.make_env()
iter_test = env.iter_test()

# 缓存，用于存储当天的历史数据
cache = pd.DataFrame()

# 尝试加载全局中位数
if 'global_medians_pl' not in globals():
    possible_paths = [
        'global_medians.csv', 
        '/kaggle/input/optiver-global-medians/global_medians.csv',
        '/kaggle/input/global-medians/global_medians.csv'
    ]
    
    global_medians_pl = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading global medians from {path}")
            global_medians_pl = pl.read_csv(path)
            break
            
    if global_medians_pl is None:
        print("Warning: global_medians.csv not found. Using batch median fallback.")

# 2. 遍历测试集
counter = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    # test: 当前时间步的测试数据 (DataFrame)
    
    # --- 缓存管理 ---
    if not cache.empty:
        if test['date_id'].iloc[0] != cache['date_id'].iloc[-1]:
            cache = pd.DataFrame()
            
    cache = pd.concat([cache, test], ignore_index=True)
    
    # --- A. 数据预处理与特征工程 ---
    try:
        # generate_features 现在返回 LazyFrame
        feat_df_lazy, current_feas = generate_features(cache, global_medians=global_medians_pl)
        
        # 执行计算 (Collect) 并转回 Pandas
        # 注意：在推理阶段数据量小，collect() 开销可控
        feat_df = feat_df_lazy.collect().to_pandas()
        
    except Exception as e:
        # Fallback: 如果出错，尝试不带 global_medians 运行
        # print(f"Error in feature generation: {e}")
        feat_df_lazy, current_feas = generate_features(cache, global_medians=None)
        feat_df = feat_df_lazy.collect().to_pandas()
    
    # 确保使用训练时的特征列表 (优先使用全局 features，否则使用当前生成的)
    if 'features' in globals():
        use_features = features
    else:
        use_features = list(set(current_feas))

    # 取出最后一部分
    current_len = len(test)
    current_feat_df = feat_df.iloc[-current_len:][use_features]
    
    # --- B. 模型推理 (区分 NN 和 Tree) ---
    preds = []
    for model in models:
        # 检查是否是我们的 PyTorchRegressor (通过是否有 scaler 属性判断)
        if hasattr(model, 'scaler') and model.scaler is not None:
            # NN: 缺失值填 0，然后标准化
            # 注意：StandardScaler 需要 2D array
            # 关键修复：必须使用 .values 转为 numpy array，否则 torch.tensor(DataFrame) 会报错
            X_input = current_feat_df.fillna(0).values
            X_input = model.scaler.transform(X_input)
            preds.append(model.predict(X_input))
        else:
            # Tree Models: 缺失值填 -9e9
            # 关键修复：必须使用 .values 转为 numpy array
            X_input = current_feat_df.fillna(-9e9).values
            preds.append(model.predict(X_input))
            
    current_pred = np.mean(preds, axis=0)
    
    # --- C. 填充预测值 ---
    sample_prediction['target'] = current_pred
    
    # --- D. 提交 ---
    env.predict(sample_prediction)
    
    counter += 1




