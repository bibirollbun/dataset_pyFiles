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



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder # 今回はOne-Hot Encodingを使用しますが、LabelEncoderも参考に記載
from sklearn.preprocessing import MinMaxScaler
!apt-get -y install fonts-ipaexfont > /dev/null

# 2. 必要なライブラリをインポート
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 3. フォントを設定（IPAexゴシック）
plt.rcParams['font.family'] = 'IPAexGothic'
train_df = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/test.csv')


train_df


train_df.columns


train_df.describe()


sample_submission_df = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/sample_submission.csv')



# 日付をdatetimeに変換
train_df['date'] = pd.to_datetime(train_df['date'])

# カテゴリ変数をLabel Encoding
label_encoders = {}
for col in ['country', 'store', 'product']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le  # 後で逆変換したいとき用

# num_soldの正規化（0〜1）
scaler = MinMaxScaler()
train_df['num_sold_normalized'] = scaler.fit_transform(train_df[['num_sold']])

# 整形後のtrain_df表示（上位5行）
print("整形・正規化後のデータ:")
print(train_df.head())

# 国別の売上合計を表示（逆変換して国名に戻す）
train_df['country_name'] = label_encoders['country'].inverse_transform(train_df['country'])
country_sales = train_df.groupby('country_name')['num_sold'].sum().reset_index().sort_values(by='num_sold', ascending=False)

print("\n国別の売上数（num_sold合計）:")
print(country_sales)


import matplotlib.pyplot as plt

# 国別売上のグループ化（前と同じ）
country_sales = train_df.groupby('country_name')['num_sold'].sum().sort_values(ascending=False)

# グラフ描画
plt.figure(figsize=(8, 5))
country_sales.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('num_sold')
plt.xlabel('country')
plt.ylabel('num_sold')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


print(train_df)


# Yearly Sales Total
train_df['year'] = train_df['date'].dt.year
yearly_sales = train_df.groupby('year')['num_sold'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(yearly_sales['year'], yearly_sales['num_sold'], color='lightcoral', edgecolor='black')
plt.title('Total Sales by Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Sales (num_sold)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(yearly_sales['num_sold']):
    plt.text(yearly_sales['year'].iloc[i], v + v*0.01, f'{v:,.0f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

print("Total Sales by Year:")
print(yearly_sales)


train_df['month'] = train_df['date'].dt.month
monthly_sales = train_df.groupby('month')['num_sold'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['month'], monthly_sales['num_sold'], marker='o', linewidth=2, markersize=8, color='steelblue')
plt.title('Total Sales by Month', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales (num_sold)', fontsize=12)
plt.xticks(range(1, 13))
plt.grid(True, linestyle='--', alpha=0.7)
for i, v in enumerate(monthly_sales['num_sold']):
    plt.annotate(f'{v:,.0f}', (monthly_sales['month'].iloc[i], v), 
                textcoords="offset points", xytext=(0,10), ha='center')
plt.tight_layout()
plt.show()

print("Total Sales by Month:")
print(monthly_sales)


# Seasonal Sales Total
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:  # 12, 1, 2
        return 'Winter'

train_df['season'] = train_df['month'].apply(get_season)
seasonal_sales = train_df.groupby('season')['num_sold'].sum().reset_index()

# Season order setting
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
seasonal_sales['season'] = pd.Categorical(seasonal_sales['season'], categories=season_order, ordered=True)
seasonal_sales = seasonal_sales.sort_values('season')

colors = ['lightgreen', 'gold', 'orange', 'lightblue']
plt.figure(figsize=(10, 6))
bars = plt.bar(seasonal_sales['season'], seasonal_sales['num_sold'], color=colors, edgecolor='black')
plt.title('Total Sales by Season', fontsize=16)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Total Sales (num_sold)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(seasonal_sales['num_sold']):
    plt.text(i, v + v*0.01, f'{v:,.0f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

print("Total Sales by Season:")
print(seasonal_sales)


# Store Sales Total
store_sales = train_df.groupby('store')['num_sold'].sum().reset_index().sort_values(by='num_sold', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(store_sales['store'].astype(str), store_sales['num_sold'], color='lightseagreen', edgecolor='black')
plt.title('Total Sales by Store', fontsize=16)
plt.xlabel('Store ID', fontsize=12)
plt.ylabel('Total Sales (num_sold)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(store_sales['num_sold']):
    plt.text(i, v + v*0.01, f'{v:,.0f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

print("Total Sales by Store:")
print(store_sales)


# Product Sales Total
product_sales = train_df.groupby('product')['num_sold'].sum().reset_index().sort_values(by='num_sold', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(product_sales['product'].astype(str), product_sales['num_sold'], color='mediumpurple', edgecolor='black')
plt.title('Total Sales by Product', fontsize=16)
plt.xlabel('Product ID', fontsize=12)
plt.ylabel('Total Sales (num_sold)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(product_sales['num_sold']):
    plt.text(i, v + v*0.01, f'{v:,.0f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

print("Total Sales by Product:")
print(product_sales)


import seaborn as sns

# Data preparation for violin plots (sampling for speed)
sample_df = train_df.sample(n=min(10000, len(train_df)), random_state=42)

# Create 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribution of num_sold - Violin Plots', fontsize=20)

# 1. Violin plot by Country
sns.violinplot(data=sample_df, x='country_name', y='num_sold', ax=axes[0,0])
axes[0,0].set_title('Distribution of num_sold by Country', fontsize=14)
axes[0,0].set_xlabel('Country', fontsize=12)
axes[0,0].set_ylabel('num_sold', fontsize=12)
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Violin plot by Season
sns.violinplot(data=sample_df, x='season', y='num_sold', ax=axes[0,1], order=['Spring', 'Summer', 'Autumn', 'Winter'])
axes[0,1].set_title('Distribution of num_sold by Season', fontsize=14)
axes[0,1].set_xlabel('Season', fontsize=12)
axes[0,1].set_ylabel('num_sold', fontsize=12)

# 3. Violin plot by Year
sns.violinplot(data=sample_df, x='year', y='num_sold', ax=axes[1,0])
axes[1,0].set_title('Distribution of num_sold by Year', fontsize=14)
axes[1,0].set_xlabel('Year', fontsize=12)
axes[1,0].set_ylabel('num_sold', fontsize=12)

# 4. Violin plot by Month
sns.violinplot(data=sample_df, x='month', y='num_sold', ax=axes[1,1])
axes[1,1].set_title('Distribution of num_sold by Month', fontsize=14)
axes[1,1].set_xlabel('Month', fontsize=12)
axes[1,1].set_ylabel('num_sold', fontsize=12)

plt.tight_layout()
plt.show()

# Display statistical information
print("Statistical Information of num_sold by Category:")
print("\n=== Country Statistics ===")
print(sample_df.groupby('country_name')['num_sold'].describe())
print("\n=== Season Statistics ===")
print(sample_df.groupby('season')['num_sold'].describe())
print("\n=== Year Statistics ===")
print(sample_df.groupby('year')['num_sold'].describe())
print("\n=== Month Statistics ===")
print(sample_df.groupby('month')['num_sold'].describe())


# セル1: 予測用の追加ライブラリインポート
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("予測用ライブラリのインポート完了")

# ========================================



# セル2: テストデータの読み込みと基本確認
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/test.csv')

print("テストデータの形状:", test_df.shape)
print("\nテストデータの最初の5行:")
print(test_df.head())

print("\nテストデータの基本情報:")
print(test_df.info())

print("\nテストデータの欠損値確認:")
print(test_df.isnull().sum())

# ========================================



# セル3: テストデータの前処理
# 日付をdatetimeに変換
test_df['date'] = pd.to_datetime(test_df['date'])

# 同じ特徴量エンジニアリング
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['dayofweek'] = test_df['date'].dt.dayofweek
test_df['quarter'] = test_df['date'].dt.quarter
test_df['season'] = test_df['month'].apply(get_season)

# カテゴリ変数をLabel Encoding（trainと同じencoderを使用）
for col in ['country', 'store', 'product']:
    test_df[col] = label_encoders[col].transform(test_df[col])

print("テストデータの前処理完了")
print("前処理後のテストデータ:")
print(test_df.head())

# ========================================


# セル4: 追加特徴量の作成と特徴量準備
# まず訓練データに不足している特徴量を追加
train_df['day'] = train_df['date'].dt.day
train_df['dayofweek'] = train_df['date'].dt.dayofweek
train_df['quarter'] = train_df['date'].dt.quarter

# テストデータにも同じ特徴量を追加
test_df['day'] = test_df['date'].dt.day
test_df['dayofweek'] = test_df['date'].dt.dayofweek
test_df['quarter'] = test_df['date'].dt.quarter

# 使用する特徴量を定義
feature_columns = ['country', 'store', 'product', 'year', 'month', 'day', 'dayofweek', 'quarter']

# 既存のデータに存在する特徴量のみを確認
print("train_dfの列名:")
print(train_df.columns.tolist())
print("\ntest_dfの列名:")
print(test_df.columns.tolist())

# 訓練データとテストデータの特徴量を準備
X_train = train_df[feature_columns]
y_train = train_df['num_sold']
X_test = test_df[feature_columns]

print("\n特徴量の準備完了")
print(f"訓練データの特徴量形状: {X_train.shape}")
print(f"目的変数の形状: {y_train.shape}")
print(f"テストデータの特徴量形状: {X_test.shape}")

print("\n使用する特徴量:")
print(feature_columns)

print("\n各特徴量のデータ型:")
print(X_train.dtypes)

# ========================================


# セル5: 訓練・検証データの分割
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print("データ分割完了")
print(f"訓練データ: {X_train_split.shape}")
print(f"検証データ: {X_val_split.shape}")

# ========================================



# セル6: 複数モデルの定義と学習
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression()
}

model_results = {}

print("複数モデルの学習・評価を開始...")

for name, model in models.items():
    print(f"\n=== {name}の学習中 ===")
    
    # モデル学習
    model.fit(X_train_split, y_train_split)
    
    # 予測
    y_pred_train = model.predict(X_train_split)
    y_pred_val = model.predict(X_val_split)
    
    # 評価指標計算
    train_rmse = np.sqrt(mean_squared_error(y_train_split, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val_split, y_pred_val))
    val_mae = mean_absolute_error(y_val_split, y_pred_val)
    val_r2 = r2_score(y_val_split, y_pred_val)
    
    print(f"訓練RMSE: {train_rmse:.4f}")
    print(f"検証RMSE: {val_rmse:.4f}")
    print(f"検証MAE: {val_mae:.4f}")
    print(f"検証R²: {val_r2:.4f}")
    
    # 結果を保存
    model_results[name] = {
        'model': model,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }

print("\n全モデルの学習・評価完了")

# ========================================


# セル7: 最良モデルの選択と評価
# RMSEが最小のモデルを選択
best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['val_rmse'])
best_model = model_results[best_model_name]['model']

print(f"最良モデル: {best_model_name}")
print(f"最良モデルの検証RMSE: {model_results[best_model_name]['val_rmse']:.4f}")

# 最良モデルでの詳細評価
print(f"\n=== {best_model_name} 詳細評価 ===")
for metric, value in model_results[best_model_name].items():
    if metric != 'model':
        print(f"{metric}: {value:.4f}")

# ========================================



# セル8: 交差検証による性能評価
print(f"\n=== {best_model_name}の交差検証評価 ===")

# 5-fold交差検証
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, 
                           scoring='neg_mean_squared_error', n_jobs=-1)
cv_rmse_scores = np.sqrt(-cv_scores)

print(f"5-Fold交差検証 RMSE:")
print(f"平均: {cv_rmse_scores.mean():.4f}")
print(f"標準偏差: {cv_rmse_scores.std():.4f}")
print(f"各フォールド: {cv_rmse_scores}")

# ========================================


# セル9: 特徴量重要度の分析（Random Forestの場合）
if best_model_name == 'RandomForest':
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特徴量重要度:")
    print(feature_importance)
    
    # 特徴量重要度の可視化
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('特徴量重要度')
    plt.xlabel('重要度')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ========================================


# セル10: 最終モデルの学習
print("最終モデルの学習（全訓練データ使用）...")

# 全訓練データで最終モデルを学習
final_model = best_model
final_model.fit(X_train, y_train)

print("最終モデルの学習完了")

# 訓練データでの最終性能確認
y_train_pred = final_model.predict(X_train)
final_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
final_train_r2 = r2_score(y_train, y_train_pred)

print(f"最終モデル（全訓練データ）:")
print(f"訓練RMSE: {final_train_rmse:.4f}")
print(f"訓練R²: {final_train_r2:.4f}")

# ========================================


# セル11: テストデータでの予測
print("テストデータでの予測開始...")

# 予測実行
test_predictions = final_model.predict(X_test)

print("予測完了!")
print(f"\n予測結果の統計:")
print(f"最小値: {test_predictions.min():.2f}")
print(f"最大値: {test_predictions.max():.2f}")
print(f"平均値: {test_predictions.mean():.2f}")
print(f"中央値: {np.median(test_predictions):.2f}")
print(f"標準偏差: {test_predictions.std():.2f}")


# セル12: 提出ファイルの作成
# まずテストデータのrow_idを確認
print("テストデータのrow_id確認:")
print(f"row_id列の存在: {'row_id' in test_df.columns}")
if 'row_id' in test_df.columns:
    print(f"row_idの範囲: {test_df['row_id'].min()} - {test_df['row_id'].max()}")
    print(f"row_idの件数: {len(test_df['row_id'])}")
    print("最初の10個のrow_id:")
    print(test_df['row_id'].head(10).tolist())
    print("最後の10個のrow_id:")
    print(test_df['row_id'].tail(10).tolist())
    
    # 正しいrow_idを使用
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'num_sold': test_predictions
    })
else:
    # row_id列がない場合は1から始まる連番を作成
    print("row_id列が見つかりません。1から始まる連番を作成します。")
    submission = pd.DataFrame({
        'row_id': range(1, len(test_predictions) + 1),
        'num_sold': test_predictions
    })

# CSVファイルに保存
submission.to_csv('submission.csv', index=False)

print("\n提出ファイル作成完了!")
print("\nsubmission.csvの内容（最初の10行）:")
print(submission.head(10))
print("\nsubmission.csvの内容（最後の10行）:")
print(submission.tail(10))

print(f"\n提出ファイルの形状: {submission.shape}")
print(f"row_idの範囲: {submission['row_id'].min()} - {submission['row_id'].max()}")
print("DONE :)")

# ========================================



# セル13: 予測結果の可視化
plt.figure(figsize=(15, 10))

# 1. 予測値の分布
plt.subplot(2, 3, 1)
plt.hist(test_predictions, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('テストデータ予測値の分布')
plt.xlabel('num_sold')
plt.ylabel('頻度')

# 2. 訓練データ vs 予測値の散布図
plt.subplot(2, 3, 2)
plt.scatter(y_train, y_train_pred, alpha=0.5, s=1)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('訓練データ: 実際 vs 予測')

# 3. 残差プロット
plt.subplot(2, 3, 3)
residuals = y_train - y_train_pred
plt.scatter(y_train_pred, residuals, alpha=0.5, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('予測値')
plt.ylabel('残差')
plt.title('残差プロット')

# 4. 予測値の箱ひげ図
plt.subplot(2, 3, 4)
plt.boxplot([y_train, test_predictions], labels=['Train実際値', 'Test予測値'])
plt.title('訓練実際値 vs テスト予測値')
plt.ylabel('num_sold')

# 5. 月別予測値の平均
test_df_with_pred = test_df.copy()
test_df_with_pred['predicted_num_sold'] = test_predictions
monthly_pred = test_df_with_pred.groupby('month')['predicted_num_sold'].mean()

plt.subplot(2, 3, 5)
plt.plot(monthly_pred.index, monthly_pred.values, 'o-', linewidth=2, markersize=8)
plt.title('月別予測値の平均')
plt.xlabel('月')
plt.ylabel('予測num_sold平均')
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)

# 6. 国別予測値の平均
test_df_with_pred['country_name'] = label_encoders['country'].inverse_transform(test_df_with_pred['country'])
country_pred = test_df_with_pred.groupby('country_name')['predicted_num_sold'].mean().sort_values(ascending=False)

plt.subplot(2, 3, 6)
plt.bar(range(len(country_pred)), country_pred.values, color='lightcoral')
plt.title('国別予測値の平均')
plt.xlabel('国')
plt.ylabel('予測num_sold平均')
plt.xticks(range(len(country_pred)), country_pred.index, rotation=45)

plt.tight_layout()
plt.show()

print("可視化完了!")


test_df


# セル14: テストデータ予測結果の中身確認

# 予測結果を含むデータフレームの作成
test_results = test_df.copy()
test_results['predicted_num_sold'] = test_predictions

# カテゴリ変数を元の名前に戻す
test_results['country_name'] = label_encoders['country'].inverse_transform(test_results['country'])

print("=== テストデータ予測結果の確認 ===")
print(f"予測データ数: {len(test_predictions)}")
print(f"データ形状: {test_results.shape}")

print("\n=== 予測結果の最初の20行 ===")
display_cols = ['date', 'country_name', 'store', 'product', 'predicted_num_sold']
print(test_results[display_cols].head(20))

print("\n=== 予測結果の統計情報 ===")
print(f"最小値: {test_predictions.min():.2f}")
print(f"最大値: {test_predictions.max():.2f}")
print(f"平均値: {test_predictions.mean():.2f}")
print(f"中央値: {np.median(test_predictions):.2f}")
print(f"標準偏差: {test_predictions.std():.2f}")
print(f"第1四分位数 (25%): {np.percentile(test_predictions, 25):.2f}")
print(f"第3四分位数 (75%): {np.percentile(test_predictions, 75):.2f}")

print("\n=== 予測値の分布（範囲別件数） ===")
ranges = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, float('inf'))]
for start, end in ranges:
    if end == float('inf'):
        count = sum(test_predictions >= start)
        print(f"{start}以上: {count}件")
    else:
        count = sum((test_predictions >= start) & (test_predictions < end))
        print(f"{start}-{end}: {count}件")

print("\n=== 国別予測結果の集計 ===")
country_summary = test_results.groupby('country_name')['predicted_num_sold'].agg(['count', 'mean', 'min', 'max', 'std']).round(2)
print(country_summary)

print("\n=== 月別予測結果の集計 ===")
monthly_summary = test_results.groupby('month')['predicted_num_sold'].agg(['count', 'mean', 'min', 'max', 'std']).round(2)
print(monthly_summary)

print("\n=== 年別予測結果の集計 ===")
yearly_summary = test_results.groupby('year')['predicted_num_sold'].agg(['count', 'mean', 'min', 'max', 'std']).round(2)
print(yearly_summary)

print("\n=== 店舗別予測結果の集計 ===")
store_summary = test_results.groupby('store')['predicted_num_sold'].agg(['count', 'mean', 'min', 'max', 'std']).round(2)
print(store_summary)

print("\n=== 商品別予測結果の集計 ===")
product_summary = test_results.groupby('product')['predicted_num_sold'].agg(['count', 'mean', 'min', 'max', 'std']).round(2)
print(product_summary)

print("\n=== 予測値の上位10件 ===")
top_predictions = test_results.nlargest(10, 'predicted_num_sold')[display_cols]
print(top_predictions)

print("\n=== 予測値の下位10件 ===")
bottom_predictions = test_results.nsmallest(10, 'predicted_num_sold')[display_cols]
print(bottom_predictions)

print("\n=== 最終提出ファイル（submission.csv）の中身確認 ===")
print("提出ファイルの最初の10行:")
print(submission.head(10))
print(f"\n提出ファイルの最後の10行:")
print(submission.tail(10))
print(f"\n提出ファイルの形状: {submission.shape}")

print("\n=== 予測完了サマリー ===")
print(f"・予測対象レコード数: {len(test_predictions)}")
print(f"・予測値の範囲: {test_predictions.min():.2f} - {test_predictions.max():.2f}")
print(f"・平均予測値: {test_predictions.mean():.2f}")
print(f"・提出ファイル: submission.csv ({submission.shape[0]}行)")
print("・予測完了!")



