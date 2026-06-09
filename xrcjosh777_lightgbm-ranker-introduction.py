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


%time df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
print(df.shape)
df.head()


df["Target"] = (
    df.groupby("Date")["Target"]
    .rank("dense", ascending=False)#排序
    .astype("Int64")  # 注意是大寫 I，可以容納 NaN
)


df["Target"] = pd.qcut(df.Target, 30).cat.codes


print(df.Date.agg(['min', 'max']))


# Just some arbitrary dates
time_config = {'train_split_date': '2021-12-06',
               'val_split_date'  : '2022-02-10',
               'test_split_date' : '2022-02-20'}

train = df[(df.Date >= time_config['train_split_date']) & (df.Date < time_config['val_split_date'])]
val = df[(df.Date >= time_config['val_split_date']) & (df.Date < time_config['test_split_date'])]
test = df[(df.Date >= time_config['test_split_date'])]

print(train.shape)
print(val.shape)
print(test.shape)

col_use = [c for c in df.columns if c not in ["RowId","Date", "Target"]]


query_train = [train.shape[0] /2000] * 2000 #Because we have 2000 stock in each time group
query_val = [val.shape[0] / 2000] * 2000
query_test = [test.shape[0] / 2000] *2000


from lightgbm import LGBMRanker

model_return = LGBMRanker(n_estimators=15000,
                          random_state=42,
                          num_leaves=41,
                          learning_rate=0.002,
                          #max_bin =20,
                          #subsample_for_bin=20000,
                          colsample_bytree=0.7,
                          n_jobs=2)
model_return.fit(train[col_use], train['Target'],
             group = query_train,
             verbose=100,
             early_stopping_rounds=200,
             eval_set=[(val[col_use], val['Target'])],
             eval_group=[query_val],
             eval_at=[3] #Make evaluation for target=1 ranking, I choosed arbitrarily
                )


test["pred"] = model_return.predict(test[col_use])
test["pred"] # So our output is not ranks, yet..


test


# load Time Series API
import jpx_tokyo_market_prediction
# make Time Series API environment (this function can be called only once in a session)
env = jpx_tokyo_market_prediction.make_env()
# get iterator to fetch data day by day
iter_test = env.iter_test()


for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    try:
        sample_prediction['Rank'] = model_return.predict(prices[col_use]) * -1
        # Get the ranks from prediction first and for the duplicated ones, just rank again
        sample_prediction['Rank'] = sample_prediction.groupby("Date")["Rank"].rank("dense", 
                                                                                   ascending=False).astype(int)
        sample_prediction['Rank'] = sample_prediction.groupby("Date")["Rank"].rank("first").astype(int) - 1
    except:
        sample_prediction['Rank'] = 0
    sample_prediction = sample_prediction.replace([-np.inf, np.inf], np.nan).fillna(0.0)
    # register your predictions
    env.predict(sample_prediction)
    display(sample_prediction)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lightgbm import LGBMRanker
import warnings
warnings.filterwarnings('ignore')

# 設置隨機種子確保結果可重現
np.random.seed(42)

# 生成更豐富的模擬數據
users = np.repeat(['u1', 'u2', 'u3', 'u4', 'u5'], 5)
videos = [f'v{i}' for i in range(1, 26)]  # 每個用戶看不同的影片
durations = np.random.randint(30, 300, size=25)
is_popular = np.random.randint(0, 2, size=25)
watch_time = durations * np.random.uniform(0.3, 1.0, size=25)

# 添加更多特徵來提升模型表現
video_category = np.random.randint(1, 6, size=25)  # 影片類別 1-5
user_age = np.random.randint(18, 65, size=25)  # 用戶年齡
upload_days = np.random.randint(1, 365, size=25)  # 上傳天數

df = pd.DataFrame({
    "user_id": users,
    "video_id": videos,
    "duration": durations,
    "is_popular": is_popular,
    "watch_time": watch_time,
    "video_category": video_category,
    "user_age": user_age,
    "upload_days": upload_days
})

# 創建更多工程特徵
df['watch_ratio'] = df['watch_time'] / df['duration']  # 觀看完成率
df['duration_log'] = np.log1p(df['duration'])  # 對數變換
df['recency_score'] = 1 / (1 + df['upload_days'] / 30)  # 新鮮度分數
df['is_long_video'] = (df['duration'] > 180).astype(int)  # 是否為長影片

# 建立目標變量：基於watch_ratio進行排名（更合理的排序目標）
df['target'] = df.groupby("user_id")["watch_ratio"].rank("dense", ascending=False).astype(int)

print("數據概覽：")
print(df.head())
print(f"\n各用戶的排名分佈：")
print(df.groupby('user_id')['target'].value_counts().sort_index())


df


# 選擇特徵（避免直接用watch_time相關特徵）
features = [
    "duration", "is_popular", "video_category", 
    "user_age", "upload_days", "duration_log", 
    "recency_score", "is_long_video"
]

X = df[features]
y = df["target"]

# 特徵標準化（可選，但通常能提升表現）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# 每個使用者的影片數量作為group
group = df.groupby("user_id").size().to_list()

print(f"\nGroup sizes: {group}")
print(f"Total samples: {len(X)}")

# 優化後的LightGBM排序模型參數
model = LGBMRanker(
    objective='lambdarank',  # 明確指定排序目標
    n_estimators=100,        # 降低估計器數量避免過擬合
    learning_rate=0.1,       # 提高學習率
    num_leaves=31,           # 適中的葉子數
    max_depth=5,             # 限制深度
    min_data_in_leaf=1,      # 最小葉子樣本數
    subsample=0.8,           # 樣本採樣率
    colsample_bytree=0.8,    # 特徵採樣率
    reg_alpha=0.1,           # L1正則化
    reg_lambda=0.1,          # L2正則化
    random_state=42,
    n_jobs=2,
    verbose=-1               # 關閉詳細輸出
)

# 訓練模型
print("\n開始訓練模型...")
model.fit(
    X_scaled, y,
    group=group,
    eval_at=[1, 3, 5],  # 評估前1、3、5名的排序品質
    callbacks=[],       # 移除early_stopping避免訓練不充分
)

# 預測分數
df["score"] = model.predict(X_scaled)

print(f"\nScore統計信息：")
print(f"Score範圍: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
print(f"Score標準差: {df['score'].std():.4f}")
print(f"是否有0分數: {(df['score'] == 0).any()}")

# 檢視每個用戶的排序情況
print("\n各用戶排序結果：")
for user in df['user_id'].unique():
    user_data = df[df.user_id == user].sort_values("score", ascending=False)
    print(f"\n{user} 的影片排序：")
    print(user_data[["video_id", "watch_ratio", "target", "score"]].round(4))

# 計算排序相關性（Spearman相關係數）
from scipy.stats import spearmanr
print("\n排序品質評估：")
for user in df['user_id'].unique():
    user_data = df[df.user_id == user]
    corr, p_value = spearmanr(-user_data['target'], user_data['score'])  # 負號因為target越小排名越高
    print(f"{user}: Spearman相關係數 = {corr:.4f} (p={p_value:.4f})")

# 特徵重要性
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n特徵重要性：")
print(feature_importance)

# 如果仍有score為0的問題，使用MinMaxScaler進行後處理
if (df["score"] == 0).any() or df["score"].std() < 1e-6:
    print("\n檢測到score變異性過低，進行後處理...")
    score_scaler = MinMaxScaler(feature_range=(0.1, 1.0))
    df["score_normalized"] = score_scaler.fit_transform(df[["score"]]).flatten()
    
    print("後處理後的Score統計：")
    print(f"Score範圍: [{df['score_normalized'].min():.4f}, {df['score_normalized'].max():.4f}]")
    print(f"Score標準差: {df['score_normalized'].std():.4f}")




