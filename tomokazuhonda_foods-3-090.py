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

from sklearn.preprocessing import LabelEncoder
import warnings
import gc # メモリ解放（ガベージコレクション）のためのライブラリ
warnings.filterwarnings('ignore') 

# --- 1. データロードとメモリ最適化 ---

# ✅ ファイルパス: 実行環境に合わせて変更してください
DATA_PATH = '../input/m5-forecasting-accuracy/'

def reduce_mem_usage(df):
    """データ（表）を軽くしてパソコンへの負担を減らす関数"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8) 
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else: # float型
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16) 
    return df

print("データのロードを開始...")
sales_df = pd.read_csv(DATA_PATH + 'sales_train_validation.csv')
cal_df = pd.read_csv(DATA_PATH + 'calendar.csv')               
price_df = pd.read_csv(DATA_PATH + 'sell_prices.csv')          
submission_df = pd.read_csv(DATA_PATH + 'sample_submission.csv') 
eval_df = pd.read_csv(DATA_PATH + 'sales_train_evaluation.csv')

# メモリダイエットを実行
sales_df = reduce_mem_usage(sales_df)
cal_df = reduce_mem_usage(cal_df)
price_df = reduce_mem_usage(price_df)
eval_df = reduce_mem_usage(eval_df)


print("--- sales_train_validation の先頭5行---")
print(sales_df.head())

print("\n--- calendarの先頭5行---")
print(cal_df.head())

print("\n--- sell_prices先頭5行---")
print(price_df.head())

print("\n--- sample_submission先頭5行---")
print(submission_df.head())

print("--- sales_train_evaluation の先頭5行---")
print(eval_df.head())


# ターゲットIDを全アイテム・全店舗のリストとして取得
all_ids = sales_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].copy()

# ----------------------------------------------------------
## 1. 過去の実績データ (d_1〜d_1913) の縦長変換
# ----------------------------------------------------------
print("✅ 過去の実績データを縦長リストに変換中...")
dates_past = [f'd_{i}' for i in range(1, 1914)] 
train_df = pd.melt(
    sales_df,
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
    value_vars=dates_past, 
    var_name='d', 
    value_name='sales' 
)
del sales_df # 元の大きなデータフレームはここで削除

# ----------------------------------------------------------
## 2. 検証期間の実績値 (d_1914〜d_1941) の結合
# ----------------------------------------------------------
print("✅ 検証期間(d_1914〜d_1941)の実績値を結合中...")
EVAL_DATES = [f'd_{i}' for i in range(1914, 1942)] 

# evaluationデータを縦長に変換し、IDをvalidationに変換
df_eval_melt = pd.melt(
    eval_df, # sales_train_evaluation.csv
    id_vars=['id'],
    value_vars=EVAL_DATES,
    var_name='d',
    value_name='sales' # カラム名は 'sales' に統一
)
# IDを合わせる ('XXX_evaluation' -> 'XXX_validation')
df_eval_melt['id'] = df_eval_melt['id'].str.replace('_evaluation', '_validation')

# 過去データ (train_df) の下部に検証期間の実績値 (df_eval_melt) を連結
df_all_actuals = pd.concat([train_df, df_eval_melt], ignore_index=True)

# ----------------------------------------------------------
## 3. 未来の予測枠 (d_1942〜d_1969) の作成と最終結合
# ----------------------------------------------------------
print("✅ 未来の予測枠を作成し、全データを最終結合...")
TEST_DAYS = 28
TEST_EVALUATION_DAYS = [f'd_{i}' for i in range(1942, 1942 + TEST_DAYS)] 

future_data_base = []
for d_id in TEST_EVALUATION_DAYS:
    temp_df = all_ids.copy()
    temp_df['d'] = d_id
    temp_df['sales'] = np.nan # 未来の売上は空欄
    future_data_base.append(temp_df)
    
future_df = pd.concat(future_data_base, ignore_index=True)

# 実績データ (d_1〜d_1941) と未来枠 (d_1942〜d_1969) を結合
df = pd.concat([df_all_actuals, future_df], ignore_index=True)

# --- 最終クリーンアップ ---
df['sales'] = df['sales'].astype(np.float16) 
del train_df, df_eval_melt, future_df, all_ids, temp_df, eval_df
# del df_all_actuals
gc.collect()

print("=========================================================")
print("✅ 全データの前処理が完了しました。")
print(f"最終データサイズ (df): {df.memory_usage().sum() / 1024**3:.2f} GB")
print(f"期間: {df['d'].min()} (d_1) 〜 {df['d'].max()} (d_1969)")
print("=========================================================")

# --- （以降、特徴量生成、モデル訓練のコードが続く） ---


print(df.head())


top_items = (
    df[df['sales'].notna()]  # NaN（未来予測枠）を除外
    .groupby('item_id')['sales']
    .sum()
    .sort_values(ascending=False)
)

print(top_items.head(10))  # 上位10商品



# FOODS_3_090 のみ抽出し、NaN（未来予測枠）を除外
# target_df = df[(df['item_id'] == 'FOODS_3_090') & (df['sales'].notna())]

# target_df = df[(df['item_id'] == 'FOODS_3_090')]
target_df = df[(df['item_id'] == 'FOODS_3_586')]


# 日付ごとに販売数を合計
daily_sales = target_df.groupby('d')['sales'].sum().reset_index()

# 1. d列の文字列から、'_'より後の数字部分を抽出
daily_sales['day_num'] = daily_sales['d'].str.replace('d_', '').astype(int)

# 2. 数字（day_num）でソート
daily_sales = daily_sales.sort_values(by='day_num').reset_index(drop=True)

# 3. day_num列は不要なので削除
daily_sales = daily_sales.drop(columns='day_num')

print(daily_sales.head())


daily_sales = daily_sales.merge(cal_df[['d', 'date']], on='d', how='left')
daily_sales.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
daily_sales = daily_sales[['ds','y']]
daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])


# daily_sales の行数を確認
print(f"daily_sales の行数: {len(daily_sales)} 行")


daily_sales.head()


!pip install japanize-matplotlib


import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib


#　FOODS_3_586でエラーになったので追加したコード 
# 'ds'列を日付時刻型に変換する
daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])

# 'y'列（販売数）がfloat16だった場合、float64に変換する
daily_sales['y'] = daily_sales['y'].astype('float64')


sns.lineplot(x='ds', y='y', data=daily_sales)
plt.title('「FOODS_3_090」 の日ごとの販売数推移')
plt.xlabel('日付')
plt.ylabel('販売数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 'ds' 列を datetime 型に変換
daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])

# 曜日と月を抽出
# 曜日は日本語表記に変換（0:月, 6:日）
daily_sales['weekday'] = daily_sales['ds'].dt.dayofweek
weekday_labels = ['月', '火', '水', '木', '金', '土', '日']
daily_sales['weekday_name'] = daily_sales['weekday'].map(lambda x: weekday_labels[x])

# 月を抽出
daily_sales['month'] = daily_sales['ds'].dt.month


plt.figure(figsize=(8, 5))
# 曜日順に並べるため、'weekday_name'ではなく'weekday'でソートし、'weekday_name'をラベルに使用
sns.boxplot(x='weekday_name', y='y', data=daily_sales, 
            order=weekday_labels, # 曜日の順番を保証
            palette='viridis') 

plt.title('曜日別の販売数傾向')
plt.xlabel('曜日')
plt.ylabel('販売数 (y)')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
#


plt.figure(figsize=(10, 5))
# 月を数値順（1月〜12月）に並べるため、'month'をx軸に使用
sns.boxplot(x='month', y='y', data=daily_sales, palette='plasma')

plt.title('月別の販売数傾向')
plt.xlabel('月')
plt.ylabel('販売数 (y)')
plt.xticks(rotation=0) # 月のラベルは回転不要
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


from prophet import Prophet
model = Prophet()


model.fit(daily_sales)



future = model.make_future_dataframe(periods=28, freq='D')
future


forecast = model.predict(future)
forecast


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


fig_forecast = model.plot(forecast)


#  予測の評価


from prophet.diagnostics import cross_validation

cutoffs = pd.to_datetime(['2012-12-31', '2013-12-31', '2014-12-31'])
horizons = ['365 days', '365 days', '365 days']

dfs = []
for cutoff, horizon in zip(cutoffs, horizons):
    df = cross_validation(model, cutoffs=[cutoff], horizon=horizon)
    # df = cross_validation(model, cutoffs=[cutoff], horizon=horizon,period='90 days') 0~30が表示されるか？
    dfs.append(df)
# 結果を結合
df_cv = pd.concat(dfs, ignore_index=True)


df_cv


from prophet.diagnostics import performance_metrics

# 追加(0~30が表示されるか？)
df_p = performance_metrics(
    df_cv, 
    # rolling_windowを非常に小さい値（例：0.01）に設定することで、
    # 評価ポイントが増加し、より短いhorizonから表示されるようになります。
    rolling_window=0.01
) 


df_p = performance_metrics(df_cv)
df_p.head()


df_p['horizon_days'] = df_p['horizon'].dt.days


sns.lineplot(x='horizon_days', y='mse', data=df_p)
plt.ticklabel_format(style='plain', axis='y')
plt.show()


sns.lineplot(x='horizon_days', y='coverage', data=df_p)


from prophet.make_holidays import make_holidays_df

# アメリカの祝日を DataFrame に変換（2011〜2016年）
us_holidays = make_holidays_df(year_list=[2011, 2012, 2013, 2014, 2015, 2016], country='US')

# モデルに祝日を追加
model_2 = Prophet(
    yearly_seasonality=True,         # 年次季節性（1年周期）
    weekly_seasonality=True,         # 週次季節性（曜日ごとの変動）
    daily_seasonality=False,         # 日次季節性（通常は不要）
    holidays=us_holidays,            # アメリカの祝日を考慮
    seasonality_mode='additive',     # 変動が一定なら additive、変動が増減するなら multiplicative
    seasonality_prior_scale=7.0,    # 季節性の柔軟性（大きいほど複雑な波形を許容）
    holidays_prior_scale=13.0,       # 祝日効果の強さ（大きいほど祝日による変動を強く反映）
    changepoint_prior_scale =0.4     #トレンド変化に敏感
)
# 必要なら追加の季節性も加える
model_2.add_seasonality(name='weekly_detail', period=7, fourier_order=6)

# 学習
model_2.fit(daily_sales)

future_2 = model_2.make_future_dataframe(periods=28, freq='D')
future_2


forecast_2 = model_2.predict(future_2)
forecast_2


forecast_2 = model_2.predict(future_2)
forecast_2


forecast_2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


fig_forecast_2 = model_2.plot(forecast_2)


from prophet.diagnostics import cross_validation

cutoffs = pd.to_datetime(['2012-12-31', '2013-12-31', '2014-12-31'])
horizons = ['365 days', '365 days', '365 days']

dfs_2 = []
for cutoff, horizon in zip(cutoffs, horizons):
    df_2 = cross_validation(model_2, cutoffs=[cutoff], horizon=horizon)
    dfs_2.append(df_2)

# 結果を結合
df_cv_2 = pd.concat(dfs_2, ignore_index=True)


df_cv_2


df_p_2 = performance_metrics(df_cv_2)
df_p_2.head()


df_p_2['horizon_days'] = df_p_2['horizon'].dt.days


sns.lineplot(x='horizon_days', y='mse', data=df_p_2)


sns.lineplot(x='horizon_days', y='coverage', data=df_p_2)

