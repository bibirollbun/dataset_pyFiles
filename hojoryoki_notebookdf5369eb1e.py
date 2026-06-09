# --- 0. ライブラリのインポート ---
import numpy as np
import pandas as pd
import lightgbm as lgb
import ast # 文字列をPythonのオブジェクトとして評価するためのライブラリ
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
!pip install japanize-matplotlib
import japanize_matplotlib # 日本語表示のためにインポート


# --- 1. データの読み込み ---
# ファイルの場所を指定
train_df = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv')
print("Data loaded successfully!")


# --- データの準備 ---
# まず、グラフ作成の元となる訓練データを読み込む
try:
    train_df = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')
except FileNotFoundError:
    print("ファイルが見つかりません。'train.csv'へのパスを修正してください。")

# --- グラフ1：予算と興行収入の関係（散布図） ---
print("グラフ1：予算と興行収入の散布図を作成します...")

plt.figure(figsize=(12, 7)) # グラフのサイズを横長に設定
# 散布図を作成
# x軸に'budget'、y軸に'revenue'を指定
# alpha=0.3で点を半透明にして、重なり具合が分かるように
sns.scatterplot(x='budget', y='revenue', data=train_df[train_df['budget'] > 0], alpha=0.3)

# グラフのタイトルと軸ラベルを設定
plt.title('映画の予算と世界興行収入の関係', fontsize=16)
plt.xlabel('予算 (Budget)', fontsize=12)
plt.ylabel('興行収入 (Revenue)', fontsize=12)
# 軸の数値を読みやすくするために、指数表記ではなく通常の表記に
plt.ticklabel_format(style='plain', axis='both')
plt.grid(True) # グリッド線を表示
plt.savefig('budget_revenue_scatter.png', dpi=300, bbox_inches='tight') # ファイル名と解像度を指定
plt.show() # グラフを表示


# --- グラフ2：主要な製作会社ごとの興行収入（箱ひげ図） ---
print("\nグラフ2：主要な製作会社ごとの興行収入の箱ひげ図を作成します...")

# 'production_companies'カラムはJSON形式の文字列なので、Pythonのリストに変換
train_df['production_companies_list'] = train_df['production_companies'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# 各行を、製作会社ごとに分割（展開）
# これにより、映画と製作会社のペアのデータへ
exploded_companies = train_df.explode('production_companies_list')

# 各製作会社の名前を抽出
exploded_companies['company_name'] = exploded_companies['production_companies_list'].apply(
    lambda x: x['name'] if isinstance(x, dict) else None
)

# 映画の製作本数が多い上位10社を特定
top_10_companies = exploded_companies['company_name'].value_counts().nlargest(10).index

# 上位10社のデータのみを抽出
top_companies_df = exploded_companies[exploded_companies['company_name'].isin(top_10_companies)]

# 箱ひげ図を作成
plt.figure(figsize=(14, 8)) # グラフのサイズを大きめに設定
sns.boxplot(x='company_name', y='revenue', data=top_companies_df)

# グラフのタイトルと軸ラベルを設定
plt.title('主要製作会社ごとの興行収入の分布', fontsize=16)
plt.xlabel('製作会社', fontsize=12)
plt.ylabel('興行収入 (Revenue)', fontsize=12)
# X軸のラベル（会社名）が長いので、90度回転させて重ならないように
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y') # Y軸の数値を通常の表記に
plt.grid(True, axis='y') # Y軸方向のグリッド線を表示
plt.tight_layout() # レイアウトを自動調整
plt.savefig('company_revenue_scatter.png', dpi=300, bbox_inches='tight') # ファイル名と解像度を指定
plt.show() # グラフを表示


# # --- 2. 簡単な特徴量作成 ---
print("\n--- 2. Feature Engineering ---")

# trainとtestを一度結合して、まとめて処理を行う
# これにより、trainとtestでコードの重複がなくなる
all_df = pd.concat([train_df.drop(['revenue'], axis=1), test_df], axis=0)

# JSON形式のカラムをパースする関数
def parse_json(column_str):
    if isinstance(column_str, str):
        try:
            return ast.literal_eval(column_str)
        except:
            return []
    return []

json_columns = ['belongs_to_collection', 'genres', 'production_companies', 
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

for col in json_columns:
    all_df[col] = all_df[col].apply(parse_json)

# --- 基本的な特徴量 ---
print("Creating basic features...")
all_df['has_collection'] = all_df['belongs_to_collection'].apply(lambda x: 1 if x else 0)
all_df['has_homepage'] = all_df['homepage'].notna().astype(int)
all_df['has_tagline'] = all_df['tagline'].notna().astype(int)

# # --- 日付に関する特徴量 ---（精度向上のため使用しない）
# print("Creating date features...")
# all_df['release_date'] = pd.to_datetime(all_df['release_date'])
# all_df['release_year'] = all_df['release_date'].dt.year
# all_df['release_month'] = all_df['release_date'].dt.month
# all_df['release_dayofweek'] = all_df['release_date'].dt.dayofweek
# all_df['release_quarter'] = all_df['release_date'].dt.quarter

# --- 数値に関する特徴量 ---
print("Creating numerical features...")
all_df['budget'] = all_df['budget'].replace(0, np.nan) # 0を欠損値として扱う
# budgetとruntimeの欠損値をrelease_yearの中央値で埋める
all_df['budget'] = all_df['budget'].fillna(all_df['budget'].median())
all_df['runtime'] = all_df['runtime'].fillna(all_df['runtime'].median())

# --- JSONカラムから数を数える特徴量 ---
print("Creating count features from JSON...")
for col in ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']:
    all_df[f'num_{col}'] = all_df[col].apply(len)

print("Extracting Team & Background features...")

# 汎用的な抽出関数
def get_names_from_list(data_list, key='name', top_n=5):
    if not isinstance(data_list, list): return ['Unknown'] * top_n
    names = [item.get(key, 'Unknown') for item in data_list]
    return (names + ['Unknown'] * top_n)[:top_n]

def get_job_name(crew_list, job_title):
    if not isinstance(crew_list, list): return 'Unknown'
    return next((member['name'] for member in crew_list if member.get('job') == job_title), 'Unknown')

# 主要な名前を抽出
all_df['first_genre'] = all_df['genres'].apply(lambda x: x[0]['name'] if x else 'Unknown')
all_df['production_company'] = all_df['production_companies'].apply(lambda x: x[0]['name'] if x else 'Unknown')
all_df['director'] = all_df['crew'].apply(get_job_name, job_title='Director')
all_df['writer'] = all_df['crew'].apply(get_job_name, job_title='Writer')
all_df['producer'] = all_df['crew'].apply(get_job_name, job_title='Producer')
all_df['lead_actor_name'] = all_df['cast'].apply(lambda x: x[0]['name'] if x else 'Unknown')

# --- テキスト特徴量 ---
print("Creating text features...")
all_df['overview_word_count'] = all_df['overview'].str.split().str.len().fillna(0)
all_df['title_char_count'] = all_df['title'].str.len().fillna(0)

# --- カテゴリカル変数をすべて数値に変換 ---
print("Encoding all categorical features...")
categorical_cols = [
    'first_genre', 'production_company', 'director', 'writer', 'producer',
    'lead_actor_name', 'original_language'
]
for col in categorical_cols:
    all_df[f'{col}_code'], _ = pd.factorize(all_df[col])

# --- 最終的なデータセットの準備 ---
train_processed = all_df[:len(train_df)]
test_processed = all_df[len(train_df):]


# # --- 3. モデルの学習準備 ---
print("\n--- 3. Preparing for training ---")
# 使用する特徴量をリストで定義
features = [
    # 基本数値
    'budget', 'popularity', 'runtime',
    # フラグ
    'has_collection', 'has_homepage', 'has_tagline',
    # カウント
    'num_genres', 'num_production_companies', 'num_production_countries',
    'num_spoken_languages', 'num_Keywords', 'num_cast', 'num_crew',
    # テキスト
    'overview_word_count', 'title_char_count',
    # カテゴリカル
    'first_genre_code', 'production_company_code', 'director_code', 'writer_code', 'producer_code',
    'lead_actor_name_code', 'original_language_code'
]

X_train = train_processed[features]
X_test = test_processed[features]
y_train_log = np.log1p(train_df['revenue'])

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)
print(f"Using {len(features)} features for training.")


# # --- 4. モデルの学習 ---
print("\n--- 4. Training model ---")
params = {
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 3000,
    'learning_rate': 0.003, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 31,
    'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
}
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train_log)


# # --- 5. 予測と提出ファイルの作成 ---
print("\n--- 5. Making predictions ---")
log_predictions = model.predict(X_test)
final_predictions = np.expm1(log_predictions)

submission = pd.DataFrame({
    'id': test_df['id'],
    'revenue': final_predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")
print(submission.head())


# # --- 6. 特徴量の重要度を確認 ---
print("\n--- 6. Feature Importances ---")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance_df.head(20)) # 上位20個を表示

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

