import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from decimal import ROUND_HALF_UP, Decimal
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
import copy
import random

# 忽略那些我們已經知情並處理的警告
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(1314)

# ---------------------------------------------------------
# 1. 價格修正函數 (來自 Kaggle 教學)
# 用途：處理股票拆股、合併等情況，還原真實的股價走勢
# ---------------------------------------------------------
def adjust_price(price):
    price.loc[:, "Date"] = pd.to_datetime(price.loc[:, "Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        # 按照時間倒序排列，計算累積調整因子
        df = df.sort_values("Date", ascending=False)
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        
        # 計算調整後的收盤價
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        
        # 轉回正序並處理空值
        df = df.sort_values("Date")
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df

    # 對每一支股票代碼 (SecuritiesCode) 單獨處理
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)
    return price

# ---------------------------------------------------------
# 2. 特徵工程函數
# 用途：生成 CNN 模型需要的技術指標
# ---------------------------------------------------------
def create_features(df):
    # 1. 為了避免 SettingWithCopyWarning，先明確複製一份
    df = df.copy()
    col = 'AdjustedClose'
    periods = [5, 10, 20, 30]
    
    # 2. 強制轉換為 float 類型，確保後續計算不會因為 object 類型報錯
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. 處理空值與 0 值 (這是最關鍵的一步)
    # 先用 ffill 填補日期間的空隙
    df[col] = df.groupby("SecuritiesCode")[col].ffill()
    
    # 將剩下的 NaN 填為 0 (針對那些上市第一天就沒數據的)
    df[col] = df[col].fillna(0)
    
    # 4. 準備一個「乾淨的遮罩」用於計算 Log
    # 我們只對大於 0 的價格取 Log，小於等於 0 的設為 NaN，避免 Log 報錯
    # 使用 .where 會比直接賦值更安全
    df['LogPrice'] = np.where(df[col] > 0, np.log(df[col]), np.nan)
    
    for period in periods:
        # --- A. 計算收益率 (Return) ---
        # 使用 pct_change 時，加上 fill_method=None 消除 FutureWarning
        # 並且處理分母為 0 導致的無限大問題
        df[f"Return_{period}Day"] = df.groupby("SecuritiesCode")[col].pct_change(period, fill_method=None)
        
        # --- B. 計算波動率 (Volatility) ---
        # 因為我們已經有了乾淨的 'LogPrice'，直接用它來算 diff (Log Return)
        log_returns = df.groupby("SecuritiesCode")['LogPrice'].diff()
        
        # 計算滾動標準差
        df[f"Volatility_{period}Day"] = log_returns.rolling(period).std()
        
        # --- C. 計算均線乖離率 (MA Gap) ---
        ma = df.groupby("SecuritiesCode")[col].rolling(window=period).mean().values
        # 避免分母(MA)為 0 的情況
        df[f"MA_Gap_{period}Day"] = np.where(ma > 0, df[col] / ma - 1, 0)

    # 5. 最後的清理 (Downcasting 警告修復)
    # 刪除暫存的 LogPrice 欄位
    if 'LogPrice' in df.columns:
        df = df.drop(columns=['LogPrice'])

    # 將所有 NaN 填補為 0
    # Pandas 新版建議：不要隱式 downcast，我們保持 float 即可
    df = df.fillna(0.0)
    
    # 處理無限大 (inf) 的情況，將其設為 0
    df = df.replace([np.inf, -np.inf], 0.0)

    return df

# ---------------------------------------------------------
# 3. 序列化函數 (為 CNN 準備數據)
# 用途：將數據轉換成 (Batch_Size, Time_Steps, Features)
# ---------------------------------------------------------
def create_sequences(df, features_cols, time_steps=30):
    """
    df: 包含特徵的 DataFrame
    features_cols: 要使用的特徵欄位列表
    time_steps: CNN 要看過去幾天的數據 (例如 30 天)
    """
    X_list = []
    y_list = []
    
    # 針對每一支股票分開處理
    # 注意：這裡為了示範簡單用了迴圈，數據量大時建議針對 Date 進行 Groupby 優化處理
    for code, group in tqdm(df.groupby("SecuritiesCode")):
        # 填補特徵中的 NaN (因為做 Lag feature 會有空值)
        group = group.fillna(0)
        
        # 提取特徵與目標值
        data = group[features_cols].values
        target = group['Target'].values # 這是你要預測的 Label
        
        # 製作滑動視窗
        for i in range(time_steps, len(data)):
            X_list.append(data[i-time_steps:i]) # 過去 30 天的特徵
            y_list.append(target[i])            # 第 30 天的 Target
            
    return np.array(X_list), np.array(y_list)

# ---------------------------------------------------------
# 4. 缺失值填補函數 (新增：內插法實驗)
# 用途：替換原本單純的 ffill，改用線性內插來補足股價
# ---------------------------------------------------------
def fill_missing_values(df, target_col='AdjustedClose', method='linear'):
    """
    針對每一支股票進行缺失值填補 (Imputation)
    
    Args:
        df: 包含股價的 DataFrame
        target_col: 要補值的欄位，通常是 'AdjustedClose'
        method: 
            - 'ffill': 前值補替 (最安全，無未來函數)
            - 'linear': 線性內插 (畫直線補洞)
            - 'spline': 樣條插值 (畫曲線補洞，接近外插概念，模擬趨勢)
    """
    # 避免修改到原始資料
    df = df.copy()
    
    print(f"正在執行補值作業，使用方法: {method} ...")

    # 定義一個內部的處理函數，方便給 transform 使用
    def interpolate_group(group, method):
        if method == 'linear':
            # 線性內插：兩點之間畫直線
            return group.interpolate(method='linear', limit_direction='both')
        
        elif method == 'spline':
            # 樣條插值 (Spline)：擬合曲線
            # order=3 代表三次樣條，能畫出「S型」或「拋物線」般的趨勢
            # 如果數據點太少 (len < 4) 會導致計算失敗，這時降級回 linear
            try:
                # 只有當數據點足夠時才做 spline
                if len(group.dropna()) >= 4:
                    return group.interpolate(method='spline', order=3, limit_direction='both')
                else:
                    return group.interpolate(method='linear', limit_direction='both')
            except:
                # 萬一計算出錯，回退到 linear
                return group.interpolate(method='linear', limit_direction='both')
                
        elif method == 'ffill':
            # 前值補替
            return group.ffill()
        
        else:
            return group

    # 執行分組補值
    if method == 'ffill':
        df[target_col] = df.groupby("SecuritiesCode")[target_col].ffill()
    else:
        # 使用 transform 針對每一組進行更複雜的運算
        df[target_col] = df.groupby("SecuritiesCode")[target_col].transform(
            lambda x: interpolate_group(x, method)
        )

    # --- 最後防線 ---
    # 處理那些無法被補到的值 (例如上市第一天就是 NaN，或者整支股票都是 NaN)
    df[target_col] = df[target_col].fillna(0)

    return df




# ==========================================
# 主執行區範例 (Main Execution Block)
# ==========================================

# 1. 重新讀取完整原始檔案 (確保包含 2017-2021 的數據)
print("重新載入完整數據...")
# 請確認你的檔案路徑是否正確，通常是這個
path = "../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv"
df_prices = pd.read_csv(path)

# 2. 重新執行數據清洗與特徵工程 (使用我們剛才修復過無警告的版本)
print("清洗與生成特徵中 (這可能需要幾分鐘)...")
price_adjusted = adjust_price(df_prices)

# 2. 【新增】執行內插法補值
# 這一步會把中間斷掉的股價用連線方式補起來
price_imputed = fill_missing_values(price_adjusted, target_col='AdjustedClose', method='ffill')

# 3. 製作特徵
# 注意：因為 price_imputed 已經補好值了，create_features 裡面的 ffill 就不會產生影響
df_features = create_features(price_imputed)

from sklearn.preprocessing import StandardScaler

# 1. 定義你要用的特徵欄位
feature_columns = [
    'Return_5Day', 'Return_10Day', 'Return_20Day', 'Return_30Day',
    'Volatility_5Day', 'Volatility_10Day', 'Volatility_20Day', 'Volatility_30Day',
    'MA_Gap_5Day', 'MA_Gap_10Day', 'MA_Gap_20Day', 'MA_Gap_30Day'
]

# 2. 初始標準化器
scaler = StandardScaler()

# -------------------------------------------------------------------
# 關鍵：只能用「訓練集」的數據來擬合 (Fit)Scaler，然後應用到驗證集
# 這是為了避免 "Look-ahead Bias" (預視未來)
# -------------------------------------------------------------------

# 先切出訓練集部分的 DataFrame (依照日期)
train_df_part = df_features[df_features["Date"] < "2021-01-01"]

# 用訓練集計算 Mean 和 Std
print("正在擬合 Scaler (這可能需要一點時間)...")
scaler.fit(train_df_part[feature_columns])

# 3. 對「所有」數據進行轉換 (Transform)
# 這樣做是因為之後我們會再切分，而且 create_sequences 需要連續的數據
print("正在標準化所有數據...")
df_features[feature_columns] = scaler.transform(df_features[feature_columns])

print("標準化完成！前 5 筆特徵數據 (應該要在 -3 到 3 之間):")
print(df_features[feature_columns].head())

split_date = "2021-01-01"

print(f"正在依照日期 {split_date} 切分訓練集與驗證集...")
train_df = df_features[df_features["Date"] < split_date]
val_df = df_features[df_features["Date"] >= split_date]

print(f"Train DataFrame 筆數: {len(train_df)}")
print(f"Val DataFrame 筆數: {len(val_df)}")

print("生成序列中...")
# 注意：time_steps 要與模型設定一致 (30)
X_train, y_train = create_sequences(train_df, feature_columns, time_steps=30)
X_val, y_val = create_sequences(val_df, feature_columns, time_steps=30)

print(f"X_train shape: {X_train.shape}") # 確認這裡不是 (0, 30, 12)

# ---------------------------------------------------------
# 修正後的數據準備流程 (Strict Time Series Split)
# ---------------------------------------------------------

# 6. 放入 DataLoader (如果上面 X_train 不是空的，這裡就不會報錯了)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)



print("數據重整成功！現在可以執行訓練迴圈了。")
print("訓練集 Target 標準差:", y_train_tensor.std().item())
print("驗證集 Target 標準差:", y_val_tensor.std().item())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的裝置: {device}")

HYPER_PARAMS = {
    'input_size': 12,      # 固定 (特徵數)
    'hidden_size': 64,    # <--- 實驗重點: 試試 64, 128
    'num_layers': 2,       # 通常 2 層夠了，不用動
    'dropout': 0.3,        # <--- 實驗重點: 0.3 ~ 0.6
    'learning_rate': 0.0001,# <--- 實驗重點: 0.001, 0.0001, 0.005
    'batch_size': 64,      # 記憶體夠大可以開 128 加速
    'epochs': 100,         # 設定大一點，靠 Early Stopping 來停
    'patience': 10         # 早停容忍度: 連續 10 次沒進步就停
}

train_loader = DataLoader(train_dataset, batch_size=HYPER_PARAMS['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=HYPER_PARAMS['batch_size'], shuffle=False)

class StockCNN_Improved(nn.Module):
    def __init__(self, num_features=12, window_size=30):
        super(StockCNN_Improved, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32) # 加入 BN 層
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Block 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64) # 加入 BN 層
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Flatten
        flatten_size = 64 * (window_size // 4) 
        
        # Fully Connected
        self.fc1 = nn.Linear(flatten_size, 64)
        self.bn3 = nn.BatchNorm1d(64) # 加入 BN 層
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # 加大 Dropout 到 0.5 增加魯棒性
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [Batch, 30, 12] -> [Batch, 12, 30]
        x = x.permute(0, 2, 1) 
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        x = x.flatten(1)
        x = self.relu3(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class StockLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, output_size=1, dropout=0.5):
        """
        Args:
            input_size: 特徵數量 (你的是 12)
            hidden_size: LSTM 內部神經元數量 (類似 CNN 的 Channel 數)
            num_layers: 疊幾層 LSTM (類似 CNN 的深度)
            output_size: 預測目標數量 (你的是 1)
            dropout: 防止過擬合
        """
        super(StockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. 定義 LSTM 層
        # batch_first=True 是關鍵，因為你的輸入是 (Batch, Time, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 2. 定義全連接層 (FC)
        # 這裡加一個 BatchNorm1d 來穩定輸出，這是從你的 CNN 經驗學來的
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x shape: (batch_size, 30, 12)
        
        # LSTM 的輸出有三個: 
        # out: 所有時間點的輸出
        # (h_n, c_n): 最後一個時間點的短期記憶與長期記憶
        out, (h_n, c_n) = self.lstm(x)
        
        # --- 關鍵點 ---
        # 我們只在乎「第 30 天」看完之後的結論
        # out 的形狀是 (batch, 30, 64)
        # 我們取最後一個時間點 (Last Time Step): out[:, -1, :]
        last_time_step_out = out[:, -1, :]  # Shape變為 (batch, 64)
        
        # 進全連接層前先穩壓
        last_time_step_out = self.bn(last_time_step_out)
        
        # 預測
        prediction = self.fc(last_time_step_out)
        
        return prediction

# 重新初始化模型
# model = StockCNN_Improved(num_features=12, window_size=30).to(device)
model = StockLSTM(
    input_size=HYPER_PARAMS['input_size'], 
    hidden_size=HYPER_PARAMS['hidden_size'], 
    num_layers=HYPER_PARAMS['num_layers'], 
    dropout=HYPER_PARAMS['dropout']
).to(device)

print(model)


import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=HYPER_PARAMS['learning_rate'])
criterion = nn.MSELoss()

print("開始訓練 ...")

def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, params):
    best_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict()) # 先存初始權重

    print(f"開始訓練... Hidden: {params['hidden_size']}, LR: {params['learning_rate']}")

    for epoch in range(params['epochs']):
        # --- 訓練階段 ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            
            # 梯度裁剪 (防止 LSTM 梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # --- 驗證階段 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds = model(X_val)
                v_loss = criterion(preds, y_val.unsqueeze(1))
                val_loss += v_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{params['epochs']} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # --- Early Stopping 判斷 ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict()) # 備份最佳模型
            patience_counter = 0 # 重置計數器
            # 可以在這裡加入保存 checkpoint 的程式碼
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            patience_counter += 1
            print(f"   -> 驗證集沒進步 ({patience_counter}/{params['patience']})")
            if patience_counter >= params['patience']:
                print("早停觸發！停止訓練。")
                break
    
    # 載入最好的權重
    model.load_state_dict(best_model_wts)
    return model

model = train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, HYPER_PARAMS)
print("訓練完成！")


import sys
import jpx_tokyo_market_prediction
import pandas as pd
import numpy as np
import torch

# 1. 確保模型在 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. 啟動環境
# 如果這裡報錯，通常是因為 Notebook 沒有掛載數據集
# 請檢查右側 Data 面板是否已加入 "jpx-tokyo-stock-exchange-prediction"
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

# 3. 準備歷史緩衝區 (必須做，否則特徵無法計算)
# 這裡務必使用你讀取的原始數據 df_prices
print("正在準備歷史數據...")
# 為了節省時間與記憶體，我們只取最後 100 天
history_df = df_prices[df_prices["Date"] >= "2021-08-01"].copy()
history_df["Date"] = pd.to_datetime(history_df["Date"])

# 定義需要的欄位
cols_needed = ['Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor']

print("開始生成 submission.csv...")

count = 0
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    
    # --- A. 更新歷史數據 ---
    current_date = prices["Date"].iloc[0]
    prices["Date"] = pd.to_datetime(prices["Date"])
    
    # 挑選需要的欄位並接在歷史數據後
    new_row = prices.reindex(columns=cols_needed)
    history_df = pd.concat([history_df, new_row])
    
    # 簡單的記憶體管理
    if len(history_df) > 500000:
        history_df = history_df.iloc[-400000:]

    # --- B. 特徵工程 (現場計算) ---
    # 這裡呼叫你的 adjust_price 和 create_features
    # 注意：這步會比較慢，但在官方 API 限制下是必須的
    df_processed = adjust_price(history_df)
    df_features_all = create_features(df_processed)
    
    # 只取當天的特徵
    current_features_df = df_features_all[df_features_all["Date"] == current_date].copy()
    current_features_df = current_features_df.sort_values("SecuritiesCode")
    
    # --- C. 準備輸入數據 ---
    feature_columns = [
        'Return_5Day', 'Return_10Day', 'Return_20Day', 'Return_30Day',
        'Volatility_5Day', 'Volatility_10Day', 'Volatility_20Day', 'Volatility_30Day',
        'MA_Gap_5Day', 'MA_Gap_10Day', 'MA_Gap_20Day', 'MA_Gap_30Day'
    ]
    
    # 填補空值並標準化
    X_input = current_features_df[feature_columns].fillna(0).values
    X_input = scaler.transform(X_input) # 使用訓練好的 scaler
    
    # 擴展維度以符合 CNN 輸入 (Batch, 30, 12)
    # 這裡用複製法 (Repeat) 來快速適配
    X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(1).repeat(1, 30, 1).to(device)
    
    # --- D. 預測 ---
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()
        
    # --- E. 填入 Rank ---
    # 建立對照表
    pred_df = pd.DataFrame({
        "SecuritiesCode": current_features_df["SecuritiesCode"],
        "Prediction": preds
    })
    
    # 合併並排序
    sample_prediction = pd.merge(sample_prediction, pred_df, on="SecuritiesCode", how="left")
    sample_prediction["Prediction"] = sample_prediction["Prediction"].fillna(0)
    
    # 預測值越大 (漲越多) -> 排越前面
    # Rank 0 = Top 1
    sample_prediction = sample_prediction.sort_values(by="Prediction", ascending=False)
    sample_prediction["Rank"] = np.arange(len(sample_prediction))
    
    # 整理格式提交
    sample_prediction = sample_prediction.sort_values(by="SecuritiesCode")
    submission = sample_prediction[["Date", "SecuritiesCode", "Rank"]]
    
    env.predict(submission)
    
    count += 1
    # 每 10 天印一次進度，避免輸出太多
    if count % 10 == 0:
        print(f"已處理: {current_date}")

print("提交完成！submission.csv 已生成。")

