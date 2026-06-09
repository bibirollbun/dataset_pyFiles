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
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from sklearn.metrics import mean_squared_error
import warnings
import gc # ãƒ¡ãƒ¢ãƒªè§£æ”¾ã�®ã�Ÿã‚�ã�«è¿½åŠ 
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', '{:.2f}'.format)

# --- è¨­å®š ---
# å®Ÿè¡Œç’°å¢ƒã�«å�ˆã‚�ã�›ã�¦ãƒ•ã‚¡ã‚¤ãƒ«ã�®ãƒ‘ã‚¹ã‚’å¤‰æ›´ã�—ã�¦ã��ã� ã�•ã�„
DATA_PATH = '../input/m5-forecasting-accuracy/' 
# äºˆæ¸¬å¯¾è±¡ã�®å•†å“� (å…¨åº—èˆ—ã�®å�ˆè¨ˆè²©å£²æ•°)
TARGET_ITEM = 'FOODS_3_090'
# TARGET_STORE = 'CA_1' # ğŸ‘ˆ å…¨åº—èˆ—å¯¾å¿œã�®ã�Ÿã‚�ã€�ã�“ã�®å¤‰æ•°ã�¯ä»¥é™�ä½¿ç”¨ã�—ã�¾ã�›ã‚“
# å®Ÿç¸¾æœŸé–“ã�®æœ€çµ‚æ—¥ (d_1913)
LAST_TRAIN_DAY = 1913 
# äºˆæ¸¬æœŸé–“ã�®æ—¥æ•°
FORECAST_PERIOD = 28 

# --- ãƒ‡ãƒ¼ã‚¿ãƒ­ãƒ¼ãƒ‰ ---
print("ãƒ‡ãƒ¼ã‚¿ã‚’ãƒ­ãƒ¼ãƒ‰ä¸­...")
try:
    sales_df = pd.read_csv(f'{DATA_PATH}sales_train_validation.csv')
    cal_df = pd.read_csv(f'{DATA_PATH}calendar.csv')
    price_df = pd.read_csv(f'{DATA_PATH}sell_prices.csv')
except FileNotFoundError as e:
    print(f"ã‚¨ãƒ©ãƒ¼: {e}")
    print("DATA_PATHã�®è¨­å®šã‚’ç¢ºèª�ã�—ã�¦ã��ã� ã�•ã�„ã€‚ä¾‹: '../input/m5-forecasting-accuracy/'")
    raise e


# -------------------------------------------------------------
# --- 2. ãƒ‡ãƒ¼ã‚¿ã�®å‰�å‡¦ç�†ã�¨å¤–éƒ¨ãƒªã‚°ãƒ¬ãƒƒã‚µã�®æº–å‚™ ---
# -------------------------------------------------------------

print("ãƒ‡ãƒ¼ã‚¿ã�®å‰�å‡¦ç�†ã�¨ç‰¹å¾´é‡�ã‚¨ãƒ³ã‚¸ãƒ‹ã‚¢ãƒªãƒ³ã‚°ã‚’é–‹å§‹...")

# 1. äºˆæ¸¬å¯¾è±¡ã�®å£²ä¸Šãƒ‡ãƒ¼ã‚¿ã�®æº–å‚™ (ds, yå½¢å¼�)
sales_target = sales_df[
    (sales_df['item_id'] == TARGET_ITEM)
]

day_cols = [col for col in sales_target.columns if col.startswith('d_')]
sales_melted = sales_target[day_cols].sum(axis=0).reset_index()
sales_melted.columns = ['d', 'y']

cal_dates = cal_df[['d', 'date']].rename(columns={'date': 'ds'})
cal_dates['ds'] = pd.to_datetime(cal_dates['ds'])
df_prophet = sales_melted.merge(cal_dates, on='d', how='left')

df_prophet_train = df_prophet[df_prophet['d'].str.replace('d_', '').astype(int) <= LAST_TRAIN_DAY].copy()
df_prophet_train.drop(columns=['d'], inplace=True)


# 2-1. å¤–éƒ¨ãƒªã‚°ãƒ¬ãƒƒã‚µï¼ˆå…¨åº—èˆ—ã�®å¹³å�‡ä¾¡æ ¼ï¼‰ã�®æº–å‚™
price_cal = price_df.merge(cal_df[['d', 'date', 'wm_yr_wk']], on='wm_yr_wk', how='left')
price_cal.rename(columns={'date': 'ds'}, inplace=True)
price_cal['ds'] = pd.to_datetime(price_cal['ds'])

price_cal_all_stores = price_cal[
    (price_cal['item_id'] == TARGET_ITEM)
].copy()

avg_price_per_day = price_cal_all_stores.groupby('ds')['sell_price'].mean().reset_index()
avg_price_per_day.rename(columns={'sell_price': 'avg_price'}, inplace=True)
gc.collect()

# 2-2. å¤–éƒ¨ãƒªã‚°ãƒ¬ãƒƒã‚µï¼ˆå…¨ç±³ã�®SNAPã‚¤ãƒ™ãƒ³ãƒˆï¼‰ã�®æº–å‚™
# SNAPãƒ•ãƒ©ã‚°ã�¯çµ�å�ˆå¾Œã�«å�ˆè¨ˆã�™ã‚‹ã�Ÿã‚�ã€�ã�“ã�“ã�§ã�¯æ—¥ä»˜ã�¨ãƒ•ãƒ©ã‚°ã�®ã�¿ç”¨æ„�
snap_ca_df = cal_df[['date', 'snap_CA']].rename(columns={'date': 'ds', 'snap_CA': 'is_snap_CA'})
snap_tx_df = cal_df[['date', 'snap_TX']].rename(columns={'date': 'ds', 'snap_TX': 'is_snap_TX'})
snap_wi_df = cal_df[['date', 'snap_WI']].rename(columns={'date': 'ds', 'snap_WI': 'is_snap_WI'})

snap_ca_df['ds'] = pd.to_datetime(snap_ca_df['ds'])
snap_tx_df['ds'] = pd.to_datetime(snap_tx_df['ds'])
snap_wi_df['ds'] = pd.to_datetime(snap_wi_df['ds'])


# 2-3. å¤–éƒ¨ãƒªã‚°ãƒ¬ãƒƒã‚µï¼ˆã‚¤ãƒ™ãƒ³ãƒˆã�®æœ‰ç„¡ãƒ•ãƒ©ã‚°ï¼‰ã�®æº–å‚™
event_flag_df = cal_df[['date', 'event_name_1']].rename(columns={'date': 'ds'})
event_flag_df['ds'] = pd.to_datetime(event_flag_df['ds'])
event_flag_df['has_event'] = event_flag_df['event_name_1'].apply(lambda x: 0 if pd.isna(x) else 1)
event_flag_df = event_flag_df[['ds', 'has_event']]


# 3. è¨“ç·´ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ã�«å…¨ãƒªã‚°ãƒ¬ãƒƒã‚µã‚’çµ�å�ˆ

# ãƒ™ãƒ¼ã‚¹ãƒ‡ãƒ¼ã‚¿ã�«å¹³å�‡ä¾¡æ ¼ã‚’çµ�å�ˆ
df_train_final = df_prophet_train.merge(avg_price_per_day, on='ds', how='left')

# SNAPã‚¤ãƒ™ãƒ³ãƒˆã‚’çµ�å�ˆ
df_train_final = df_train_final.merge(snap_ca_df, on='ds', how='left')
df_train_final = df_train_final.merge(snap_tx_df, on='ds', how='left')
df_train_final = df_train_final.merge(snap_wi_df, on='ds', how='left')

# ã‚¤ãƒ™ãƒ³ãƒˆãƒ•ãƒ©ã‚°ã‚’çµ�å�ˆ
df_train_final = df_train_final.merge(event_flag_df, on='ds', how='left')


# # 4. æœ€çµ‚çš„ã�ªNaNå€¤ã�®è£œé–“ (ä¾¡æ ¼: ffill, ã‚¤ãƒ™ãƒ³ãƒˆ: 0)

# # å¹³å�‡ä¾¡æ ¼ã�®NaNå€¤è£œé–“
# df_train_final['avg_price'].fillna(method='ffill', inplace=True)

# # SNAPã‚¤ãƒ™ãƒ³ãƒˆã�¨ã‚¤ãƒ™ãƒ³ãƒˆãƒ•ãƒ©ã‚°ã�®NaNå€¤è£œé–“
# snap_cols = ['is_snap_CA', 'is_snap_TX', 'is_snap_WI']
# df_train_final[snap_cols + ['has_event']] = df_train_final[snap_cols + ['has_event']].fillna(0)

# 4. æœ€çµ‚çš„ã�ªNaNå€¤ã�®è£œé–“ (ä¾¡æ ¼: ffill, ã‚¤ãƒ™ãƒ³ãƒˆ: 0)

# å¹³å�‡ä¾¡æ ¼ã�®NaNå€¤è£œé–“ï¼ˆå¤‰å‹•ç�‡è¨ˆç®—ã�®ã�Ÿã‚�ã�«å¿…é ˆï¼‰
df_train_final['avg_price'].fillna(method='ffill', inplace=True) 

# SNAPã‚¤ãƒ™ãƒ³ãƒˆã�¨ã‚¤ãƒ™ãƒ³ãƒˆãƒ•ãƒ©ã‚°ã�®NaNå€¤è£œé–“
snap_cols = ['is_snap_CA', 'is_snap_TX', 'is_snap_WI']
df_train_final[snap_cols + ['has_event']] = df_train_final[snap_cols + ['has_event']].fillna(0)


# 4-2. ã€�ä¾¡æ ¼ã�®çµ¶å¯¾å€¤ã€‘ã�‹ã‚‰ã€�é��å�»æ•°é€±é–“ï¼ˆ28æ—¥é–“ï¼‰ã�®ç§»å‹•å¹³å�‡ã�‹ã‚‰ã�®ä¹–é›¢ç�‡ã€‘ã�¸ã�®å¤‰æ›´ ğŸ‘ˆ ä¿®æ­£

# 1. é��å�»28æ—¥é–“ã�®ç§»å‹•å¹³å�‡ä¾¡æ ¼ã‚’è¨ˆç®— (Rolling Mean)
# 'avg_price_28d_ma'
window_size = 28 # 4é€±é–“ (28æ—¥)
df_train_final['avg_price_28d_ma'] = df_train_final['avg_price'].rolling(
    window=window_size,
    min_periods=1, # ãƒ‡ãƒ¼ã‚¿ã�Œä¸�è¶³ã�—ã�¦ã�„ã‚‹åˆ�æ—¥ã�ªã�©ã�¯ã€�åˆ©ç”¨å�¯èƒ½ã�ªãƒ‡ãƒ¼ã‚¿ã�§è¨ˆç®—
    closed='left' # å½“æ—¥ã‚’å�«ã‚�ã�šã€�é��å�»28æ—¥é–“ã�®å¹³å�‡ã‚’è¨ˆç®—
).mean()

# 2. ç§»å‹•å¹³å�‡ã�‹ã‚‰ã�®ä¹–é›¢ç�‡ã‚’è¨ˆç®—ã�—ã€�æ–°ã�—ã�„åˆ—ã‚’ä½œæˆ�
# ä¹–é›¢ç�‡ = (å½“æ—¥ä¾¡æ ¼ - é��å�»å¹³å�‡) / é��å�»å¹³å�‡
# Note: é��å�»å¹³å�‡ã�Œ0ã�«ã�ªã‚‹ã�“ã�¨ã�¯ç¨€ã�§ã�™ã�Œã€�å¿µã�®ã�Ÿã‚�0é™¤ç®—ã‚’é�¿ã�‘ã‚‹å‡¦ç�†ã‚’ã�—ã�¦ã‚‚è‰¯ã�„ã�§ã�™ã€‚
df_train_final['price_deviation_rate'] = (
    (df_train_final['avg_price'] - df_train_final['avg_price_28d_ma']) / df_train_final['avg_price_28d_ma']
)

# 3. åˆ�æœŸã�®NaNå€¤ã�®è£œé–“
# ç§»å‹•å¹³å�‡ã�Œè¨ˆç®—ã�•ã‚Œã‚‹ã�¾ã�§ã�®æœ€åˆ�ã�®æ•°æ—¥ï¼ˆwindow_size-1æ—¥åˆ†ï¼‰ã�¯ã€�ä¹–é›¢ç�‡ã�ŒNaNã�¾ã�Ÿã�¯ä¸�å®‰å®šã�«ã�ªã‚‹å�¯èƒ½æ€§ã�Œã�‚ã‚Šã�¾ã�™ã€‚
# ã�“ã�“ã�§ã�¯ã€�ä¹–é›¢ç�‡ã�®NaNã‚’0ã�§è£œé–“ã�—ã�¾ã�™ï¼ˆã�¤ã�¾ã‚Šã€�é��å�»ã�®å¹³å�‡ã�Œã�ªã�„æœŸé–“ã�¯ä¹–é›¢ã�ªã�—ã�¨è¦‹ã�ªã�™ï¼‰ã€‚
df_train_final['price_deviation_rate'].fillna(0, inplace=True)


# 4. å…ƒã�®çµ¶å¯¾ä¾¡æ ¼ã�®åˆ—ã�¨ç§»å‹•å¹³å�‡ã�®åˆ—ã‚’å‰Šé™¤
# ãƒªã‚°ãƒ¬ãƒƒã‚µã�¨ã�—ã�¦ä½¿ç”¨ã�™ã‚‹ã�®ã�¯ä¹–é›¢ç�‡ã�®ã�¿
df_train_final.drop(columns=['avg_price', 'avg_price_28d_ma'], inplace=True)


# 5. SNAPåˆ—ã�®çµ±å�ˆ (is_snap_CA, is_snap_TX, is_snap_WI ã‚’å�ˆè¨ˆã�—ã€�snap_countã‚’ä½œæˆ�) ğŸ‘ˆ è¿½è¨˜ãƒ»ä¿®æ­£

# 3ã�¤ã�®SNAPãƒ•ãƒ©ã‚°ã‚’å�ˆè¨ˆã�—ã�¦æ–°ã�—ã�„åˆ—ã‚’ä½œæˆ�
df_train_final['snap_count'] = df_train_final[snap_cols].sum(axis=1).astype(np.int8)

# å…ƒã�®3ã�¤ã�®SNAPåˆ—ã‚’å‰Šé™¤
df_train_final.drop(columns=snap_cols, inplace=True)


print(f"è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®å½¢çŠ¶: {df_train_final.shape}")
print("è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®åˆ—æ•°:", len(df_train_final.columns))
print("è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®åˆ—:")
print(df_train_final.columns.tolist())
print("\nè¨“ç·´ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®å…ˆé ­:")
print(df_train_final.head())



import matplotlib.pyplot as plt
import seaborn as sns

# ã‚°ãƒ©ãƒ•ã�®ã‚¹ã‚¿ã‚¤ãƒ«è¨­å®š (ã‚ªãƒ—ã‚·ãƒ§ãƒ³)
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12

# ã‚°ãƒ©ãƒ•ã�®æ��ç”»
plt.figure(figsize=(15, 6))

# æ™‚ç³»åˆ—ãƒ—ãƒ­ãƒƒãƒˆ
# xè»¸ã�«æ—¥ä»˜ ('ds')ã€�yè»¸ã�«ä¹–é›¢ç�‡ ('price_deviation_rate') ã‚’æŒ‡å®š
plt.plot(df_train_final['ds'], df_train_final['price_deviation_rate'], 
         label='Price Deviation Rate (vs 28-day MA)', 
         color='tab:blue', 
         alpha=0.8)

# ã‚¼ãƒ­ãƒ©ã‚¤ãƒ³ï¼ˆä¹–é›¢ã�ªã�—ï¼‰ã‚’è¿½åŠ 
plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Zero Deviation (Average Price)')

# ã‚¿ã‚¤ãƒˆãƒ«ã�¨ãƒ©ãƒ™ãƒ«ã�®è¨­å®š
plt.title('Time Series Plot of Price Deviation Rate (vs 28-day Moving Average)')
plt.xlabel('Date (ds)')
plt.ylabel('Deviation Rate')
plt.legend()
plt.grid(True)
plt.tight_layout() # ãƒ¬ã‚¤ã‚¢ã‚¦ãƒˆã�®èª¿æ•´
plt.show()

# è£œè¶³ã�¨ã�—ã�¦ã€�ä¹–é›¢ç�‡ã�®åˆ†å¸ƒã‚‚ç¢ºèª�ã�™ã‚‹ã�¨æœ‰ç›Šã�§ã�™ã€‚
plt.figure(figsize=(8, 5))
sns.histplot(df_train_final['price_deviation_rate'], kde=True, bins=50)
plt.title('Distribution of Price Deviation Rate')
plt.xlabel('Deviation Rate')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# -------------------------------------------------------------
# --- 3. Prophetãƒ¢ãƒ‡ãƒ«ã�®æ§‹ç¯‰ã€�å¤–éƒ¨ãƒªã‚°ãƒ¬ãƒƒã‚µã�®è¿½åŠ ã€�è¨“ç·´ ---
# -------------------------------------------------------------

from prophet.make_holidays import make_holidays_df
# ã‚¢ãƒ¡ãƒªã‚«ã�®ç¥�æ—¥ã‚’ DataFrame ã�«å¤‰æ�›ï¼ˆ2011ã€œ2016å¹´ï¼‰
us_holidays = make_holidays_df(year_list=[2011, 2012, 2013, 2014, 2015, 2016], country='US')

# Prophetãƒ¢ãƒ‡ãƒ«ã�®ã‚¤ãƒ³ã‚¹ã‚¿ãƒ³ã‚¹åŒ–
model = Prophet(
    yearly_seasonality=True,           # å¹´æ¬¡å­£ç¯€æ€§ï¼ˆ1å¹´å‘¨æœŸï¼‰
    weekly_seasonality=True,           # é€±æ¬¡å­£ç¯€æ€§ï¼ˆæ›œæ—¥ã�”ã�¨ã�®å¤‰å‹•ï¼‰
    daily_seasonality=False,           # æ—¥æ¬¡å­£ç¯€æ€§ï¼ˆé€šå¸¸ã�¯ä¸�è¦�ï¼‰
    # holidays=us_holidays,            # ã‚¢ãƒ¡ãƒªã‚«ã�®ç¥�æ—¥ã‚’è€ƒæ…® (ç�¾åœ¨ã‚³ãƒ¡ãƒ³ãƒˆã‚¢ã‚¦ãƒˆ)
    seasonality_mode='additive',       # å¤‰å‹•ã�Œä¸€å®šã�ªã‚‰ additiveã€�å¤‰å‹•ã�Œå¢—æ¸›ã�™ã‚‹ã�ªã‚‰ multiplicative
    seasonality_prior_scale=7.0,       # å­£ç¯€æ€§ã�®æŸ”è»Ÿæ€§ï¼ˆå¤§ã��ã�„ã�»ã�©è¤‡é›‘ã�ªæ³¢å½¢ã‚’è¨±å®¹ï¼‰
    holidays_prior_scale=13.0,         # ç¥�æ—¥åŠ¹æ�œã�®å¼·ã�•ï¼ˆå¤§ã��ã�„ã�»ã�©ç¥�æ—¥ã�«ã‚ˆã‚‹å¤‰å‹•ã‚’å¼·ã��å��æ˜ ï¼‰
    changepoint_prior_scale=0.4        # ãƒˆãƒ¬ãƒ³ãƒ‰å¤‰åŒ–ã�«æ•�æ„Ÿ
)

# å¿…è¦�ã�ªã‚‰è¿½åŠ ã�®å­£ç¯€æ€§ã‚‚åŠ ã�ˆã‚‹
model.add_seasonality(name='weekly_detail', period=7, fourier_order=6)

# --- å¤–éƒ¨ãƒªã‚°ãƒ¬ãƒƒã‚µã�®è¿½åŠ  (add_regressor) ---
# å¤–éƒ¨ãƒªã‚°ãƒ¬ãƒƒã‚µã�¨ã�—ã�¦ä½¿ç”¨ã�™ã‚‹åˆ—å��ã�«å�ˆã‚�ã�›ã�¦ä¿®æ­£

# # 1. å¹³å�‡ä¾¡æ ¼ (avg_price): ä¾¡æ ¼ã�¯å£²ä¸Šã�«ä¹—æ³•çš„ã�«å½±éŸ¿ã�™ã‚‹ã�“ã�¨ã�Œå¤šã�„ã�Ÿã‚�ã€�multiplicativeãƒ¢ãƒ¼ãƒ‰ã�§è¿½åŠ 
# model.add_regressor('avg_price', prior_scale=30.0, mode='multiplicative')

# # 2. ã‚¤ãƒ™ãƒ³ãƒˆãƒ•ãƒ©ã‚° (has_event): ã‚¤ãƒ™ãƒ³ãƒˆã�®æœ‰ç„¡ã�¯å£²ä¸Šã�«åŠ æ³•çš„ã�«å½±éŸ¿ã�™ã‚‹ï¼ˆãƒ‡ãƒ•ã‚©ãƒ«ãƒˆã�®additiveï¼‰
# model.add_regressor('has_event', prior_scale=20.0) # å¼·ã‚�ã�®äº‹å‰�åˆ†å¸ƒã‚’è¨­å®š

# # 3. SNAPæ”¯çµ¦å·�ã�®æ•° (snap_count): æ”¯çµ¦å·�ã�®æ•°ã�¯å£²ä¸Šã�«åŠ æ³•çš„ã�«å½±éŸ¿ã�™ã‚‹ï¼ˆãƒ‡ãƒ•ã‚©ãƒ«ãƒˆã�®additiveï¼‰
# model.add_regressor('snap_count', prior_scale=30.0)


# æ–°ã�—ã�„ãƒªã‚°ãƒ¬ãƒƒã‚µã‚’ç™»éŒ²
model.add_regressor('price_deviation_rate', prior_scale=30.0, mode='multiplicative')
model.add_regressor('has_event', prior_scale=20.0)
model.add_regressor('snap_count', prior_scale=30.0)

# ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’
print("Prophetãƒ¢ãƒ‡ãƒ«ã‚’å­¦ç¿’ä¸­...")
# df_train_final ã�«ã�¯ ds, y, avg_price, has_event, snap_count ã�®ã�¿ã�Œå�«ã�¾ã‚Œã�¦ã�„ã‚‹ã�“ã�¨ã‚’å‰�æ��ã�¨ã�™ã‚‹
model.fit(df_train_final) 
print("ãƒ¢ãƒ‡ãƒ«å­¦ç¿’ã�Œå®Œäº†ã�—ã�¾ã�—ã�Ÿã€‚")

# â€» äºˆæ¸¬ï¼ˆforecastï¼‰éƒ¨åˆ†ã�¯ã€�ã‚¯ãƒ­ã‚¹ãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³ã�¨è©•ä¾¡ã�«é›†ä¸­ã�™ã‚‹ã�Ÿã‚�çœ�ç•¥ã�—ã�¾ã�™ã€‚


# -------------------------------------------------------------
# --- 4. ã‚«ã‚¹ã‚¿ãƒ ã‚«ãƒƒãƒˆã‚ªãƒ•ã�«ã‚ˆã‚‹ã‚¯ãƒ­ã‚¹ãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³ã�¨æ€§èƒ½è©•ä¾¡ ---
# -------------------------------------------------------------
import matplotlib.ticker as ticker

print("ã‚«ã‚¹ã‚¿ãƒ ã‚«ãƒƒãƒˆã‚ªãƒ•ã‚’ä½¿ç”¨ã�—ã�¦ã‚¯ãƒ­ã‚¹ãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³ã‚’å®Ÿè¡Œä¸­...")

# ãƒ¦ãƒ¼ã‚¶ãƒ¼æŒ‡å®šã�®ã‚«ãƒƒãƒˆã‚ªãƒ•ã�¨ãƒ›ãƒ©ã‚¤ã‚ºãƒ³
cutoffs = pd.to_datetime(['2012-12-31', '2013-12-31', '2014-12-31'])
horizons = ['365 days'] * len(cutoffs) # 3ã�¤ã�®ã‚«ãƒƒãƒˆã‚ªãƒ•å…¨ã�¦ã�§365æ—¥å…ˆã‚’äºˆæ¸¬

dfs = []
for cutoff, horizon in zip(cutoffs, horizons):
    # å�„ã‚«ãƒƒãƒˆã‚ªãƒ•/ãƒ›ãƒ©ã‚¤ã‚ºãƒ³ã�®çµ„ã�¿å�ˆã‚�ã�›ã�§cvã‚’å®Ÿè¡Œ
    # Prophetã�¯è‡ªå‹•çš„ã�«å¤–éƒ¨ãƒªã‚°ãƒ¬ãƒƒã‚µã‚’å‡¦ç�†ã�—ã�¾ã�™ã€‚
    # modelã�¯å‰�ã‚¹ãƒ†ãƒƒãƒ—ã�§å­¦ç¿’æ¸ˆã�¿ã�§ã�‚ã‚‹ã�“ã�¨ã‚’å‰�æ��ã�¨ã�—ã�¾ã�™ã€‚
    df = cross_validation(model, cutoffs=[cutoff], horizon=horizon, parallel="processes")
    dfs.append(df)

# çµ�æ�œã‚’çµ�å�ˆ
df_cv = pd.concat(dfs, ignore_index=True)
print("ã‚¯ãƒ­ã‚¹ãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³ã�Œå®Œäº†ã�—ã�¾ã�—ã�Ÿã€‚")

# --- ãƒ‘ãƒ•ã‚©ãƒ¼ãƒ�ãƒ³ã‚¹æŒ‡æ¨™ã�®ç®—å‡º ---
df_p = performance_metrics(df_cv)
print("\n--- ãƒ‘ãƒ•ã‚©ãƒ¼ãƒ�ãƒ³ã‚¹æŒ‡æ¨™ (365æ—¥äºˆæ¸¬) ---")
print(df_p.head())

# --- MSEã�¨Coverageã�®ã‚°ãƒ©ãƒ•åŒ– ---

# a. MSE (å¹³å�‡äºŒä¹—èª¤å·®) ã�®ã‚°ãƒ©ãƒ•
fig_mse = plot_cross_validation_metric(df_cv, metric='mse')
# ä¿®æ­£ç‚¹1: ã‚°ãƒ©ãƒ•ã‚¿ã‚¤ãƒˆãƒ«ã�‹ã‚‰TARGET_STOREã�®è¨€å�Šã‚’å‰Šé™¤
fig_mse.suptitle(f'{TARGET_ITEM} (å…¨åº—å�ˆè¨ˆ) MSE vs. äºˆæ¸¬æœŸé–“ (ã‚«ã‚¹ã‚¿ãƒ CV)', fontsize=16) 
ax_mse = fig_mse.axes[0] 
ax_mse.set_xlabel('horizon_days')
ax_mse.set_ylabel('MSE')

# æŒ‡æ•°è¡¨è¨˜ã‚’ç„¡åŠ¹åŒ–ã�™ã‚‹è¨­å®š
formatter = ticker.ScalarFormatter(useMathText=False)
formatter.set_scientific(False) 
ax_mse.yaxis.set_major_formatter(formatter)

# ç¸¦è»¸ã�®ä¸Šé™�è¨­å®š
ax_mse.set_ylim(0, 1000000) 
plt.show()

# b. Coverage (ä¿¡é ¼åŒºé–“ã�®ã‚«ãƒ�ãƒ¬ãƒƒã‚¸) ã�®ã‚°ãƒ©ãƒ•
fig_coverage = plot_cross_validation_metric(df_cv, metric='coverage')
# ä¿®æ­£ç‚¹1: ã‚°ãƒ©ãƒ•ã‚¿ã‚¤ãƒˆãƒ«ã�‹ã‚‰TARGET_STOREã�®è¨€å�Šã‚’å‰Šé™¤
fig_coverage.suptitle('horizon_days', fontsize=16)
fig_coverage.axes[0].set_xlabel('horizon_days')
fig_coverage.axes[0].set_ylabel('Coverage')
plt.show()

