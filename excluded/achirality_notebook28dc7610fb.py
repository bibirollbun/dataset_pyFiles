import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')

# カテゴリカル特徴量の分布の可視化
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

# 1. Episode_Title: "Episode 98" → 98
train_df['Episode_Title'] = train_df['Episode_Title'].str.extract(r'(\d+)').astype(int)


# ワンホットエンコーディングを適用する列
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# ワンホットエンコーディングの適用
train_df_encoded = pd.get_dummies(train_df, columns=categorical_cols, dtype=int)


#dfを個々のポッドキャストに分割
podcast_dfs = {name: group for name, group in train_df_encoded.groupby('Podcast_Name')}
podcast_names = list(podcast_dfs.keys())



import pandas as pd
import numpy as np

# ===========================
# Train: NaN & 外れ値除去 + OOF列削除
# ===========================
drop_cols = [
    'Linear_OOF_Pred', 'ElasticNet_OOF_Pred', 'BayesianRidge_OOF_Pred',
    'LightGBM_OOF_Pred', 'XGBoost_OOF_Pred'
]

def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
    if 'Listening_Time_minutes' in numeric_cols:
        numeric_cols.remove('Listening_Time_minutes')  # ターゲットは除外

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.05)
        Q3 = df_clean[col].quantile(0.95)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean.reset_index(drop=True)

podcast_dfs_clean = {
    name: remove_outliers_iqr(
        df.drop(columns=[col for col in drop_cols if col in df.columns]).dropna()
    )
    for name, df in podcast_dfs.items()
}

# ===========================
# Test: 前処理 + NaNと外れ値を平均で補完
# ===========================

# Step 1: Episode_Title 数値化
# 数値部分だけ抽出（Seriesとして）
episode_nums = (
    test_df['Episode_Title']
    .astype(str)
    .str.extract(r'(\d+)')[0]
    .astype(float)  # 平均計算のため float に
)

# 平均で欠損値を補完し、intに変換（四捨五入する場合は round してもOK）
mean_value = round(episode_nums.mean())  # または .fillna(episode_nums.mean()) でもOK
test_df['Episode_Title'] = episode_nums.fillna(mean_value).astype(int)

# Step 2: ワンホットエンコーディング
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols, dtype=int)

# Step 3: Podcastごとに分割
podcast_test_dfs = {
    name: group.reset_index(drop=True)
    for name, group in test_df_encoded.groupby('Podcast_Name')
}

# Step 4: NaN + 外れ値補完（平均で）
def fillna_and_fix_outliers_per_podcast(podcast_test_dfs: dict) -> dict:
    filled_dfs = {}

    for name, df in podcast_test_dfs.items():
        df = df.copy()
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if 'id' in numeric_cols:
            numeric_cols.remove('id')

        for col in numeric_cols:
            mean_val = df[col].mean()

            # NaN補完
            df[col] = df[col].fillna(mean_val)

            # 外れ値補正（IQR）
            Q1 = df[col].quantile(0.05)
            Q3 = df[col].quantile(0.95)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = (df[col] < lower) | (df[col] > upper)
            df.loc[outliers, col] = mean_val

        filled_dfs[name] = df.reset_index(drop=True)

    return filled_dfs

# 適用
podcast_test_dfs_clean = fillna_and_fix_outliers_per_podcast(podcast_test_dfs)



import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge

# 使用モデル一覧（関数と接尾辞）
model_specs = {
    'xgb': XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                        objective='reg:squarederror', random_state=42, verbosity=0),
    'lgb': LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                         objective='regression', random_state=42),
    'linear': LinearRegression(),
    'elastic': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    'bayes': BayesianRidge()
}

# ターゲット・ID列
target_col = 'Listening_Time_minutes'
id_col = 'id'

# 特徴量列（事前に決められているもの）
base_features = [
    'Episode_Length_minutes', 'Episode_Sentiment_Negative', 'Episode_Sentiment_Positive', 'Episode_Title',
    'Genre_Sports', 'Guest_Popularity_percentage', 'Host_Popularity_percentage', 'Number_of_Ads',
    'Publication_Day_Monday', 'Publication_Day_Saturday', 'Publication_Day_Sunday', 'Publication_Day_Thursday',
    'Publication_Day_Tuesday', 'Publication_Time_Afternoon', 'Publication_Time_Evening', 'Publication_Time_Night'
]

# 学習・予測ループ
for podcast_name in tqdm(podcast_test_dfs_clean.keys()):
    if podcast_name not in podcast_dfs_clean:
        continue

    train_df = podcast_dfs_clean[podcast_name].copy()
    test_df = podcast_test_dfs_clean[podcast_name].copy()

    # log1p変換（学習データのみ）
    train_df[target_col] = np.log1p(train_df[target_col])

    X_train = train_df[base_features]
    y_train = train_df[target_col]
    X_test = test_df[base_features]

    # 各モデルで学習・予測
    for model_name, model in model_specs.items():
        model.fit(X_train, y_train)

        # 学習データへの予測（logスケール）
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # 元スケールに戻して保存（exp - 1）
        train_df[f'pred_{model_name}'] = np.expm1(train_pred)
        test_df[f'pred_{model_name}'] = np.expm1(test_pred)

        # モデル保存（任意）
        joblib.dump(model, f'model_{model_name}_{podcast_name}.pkl')

    # 結果を戻し入れる
    podcast_dfs_clean[podcast_name] = train_df
    podcast_test_dfs_clean[podcast_name] = test_df



from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

def remove_outliers(df, columns, z=3):
    for col in columns:
        col_zscore = (df[col] - df[col].mean()) / df[col].std()
        df = df[col_zscore.abs() < z]
    return df.reset_index(drop=True)

def reverse_one_hot(df, prefix):
    onehot_cols = [col for col in df.columns if col.startswith(prefix)]
    if not onehot_cols:
        return pd.Series(["" for _ in range(len(df))])
    onehot_data = df[onehot_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    values = onehot_data.values.argmax(axis=1)
    categories = [col.replace(prefix, '') for col in onehot_cols]
    return pd.Series([categories[i] for i in values], index=df.index)

def common_preprocessing(df):
    # One-hot逆変換
    for col in ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']:
        df[col] = reverse_one_hot(df, col + '_')

    # LabelEncoding
    for col in ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']:
        le = LabelEncoder()
        df[col + '_ID'] = le.fit_transform(df[col].astype(str))

    # 組み合わせ特徴
    df['Genre_Day'] = df['Genre'] + '_' + df['Publication_Day']

    # スケーリング
    std_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage']
    mm_cols = ['Host_Popularity_percentage', 'Number_of_Ads']

    df[[f'{c}_std' for c in std_cols]] = StandardScaler().fit_transform(df[std_cols])
    df[[f'{c}_minmax' for c in mm_cols]] = MinMaxScaler().fit_transform(df[mm_cols])

    # ビン化・派生特徴
    df['Ads_bin'] = pd.qcut(df['Number_of_Ads'], q=3, labels=False)
    df['is_weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Time_Simplified'] = df['Publication_Time'].map({
        'Morning': 'Day', 'Afternoon': 'Day',
        'Evening': 'Night', 'Night': 'Night'
    })
    if 'Episode_Sentiment_Positive' in df.columns and 'Episode_Sentiment_Negative' in df.columns:
        df['Sentiment_Diff'] = df['Episode_Sentiment_Positive'] - df['Episode_Sentiment_Negative']

    return df

# 各 podcast ごとに処理
for podcast_name, df in podcast_dfs_clean.items():
    df = df.copy()
    df['Podcast_Name'] = podcast_name
    df = remove_outliers(df, [
        'Episode_Length_minutes', 'Host_Popularity_percentage',
        'Guest_Popularity_percentage', 'Number_of_Ads'])
    df = common_preprocessing(df)
    podcast_dfs_clean[podcast_name] = df

for podcast_name, df in podcast_test_dfs_clean.items():
    df = df.copy()
    df['Podcast_Name'] = podcast_name
    df = common_preprocessing(df)
    podcast_test_dfs_clean[podcast_name] = df



from sklearn.preprocessing import LabelEncoder

for podcast_name, df in podcast_dfs_clean.items():
    df = df.copy()

    # 数値でない列を抽出
    non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns

    # 各列に対して LabelEncoder を適用
    for col in non_numeric_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 変換後に戻し保存
    podcast_dfs_clean[podcast_name] = df



podcast_dfs_clean["Athlete's Arena"]


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import joblib

# モデル保存と予測格納用
xgb_models = {}
test_preds = []

# ターゲットと除外カラム
target_col = 'Listening_Time_minutes'
id_col = 'id'

for podcast_name in podcast_test_dfs_clean.keys():
    if podcast_name in podcast_dfs_clean:
        train_df = podcast_dfs_clean[podcast_name]
        test_df = podcast_test_dfs_clean[podcast_name]

        # log1p変換（log(1 + x)で0にも安全）
        train_df['Listening_Time_minutes'] = np.log1p(train_df['Listening_Time_minutes'])

        # 変換後のDataFrameを再保存（必要に応じて元に上書きしてもOK）
        podcast_dfs_clean[podcast_name] = train_df

# train/test 共通で使う特徴量列を決定
example_podcast = next(iter(podcast_dfs_clean))
all_columns = podcast_dfs_clean[example_podcast].columns
base_features = [
    'Episode_Length_minutes', 'Episode_Sentiment_Negative', 'Episode_Sentiment_Positive', 'Episode_Title',
    'Genre_Sports', 'Guest_Popularity_percentage', 'Host_Popularity_percentage', 'Number_of_Ads',
    'Publication_Day_Monday', 'Publication_Day_Saturday', 'Publication_Day_Sunday', 'Publication_Day_Thursday',
    'Publication_Day_Tuesday', 'Publication_Time_Afternoon', 'Publication_Time_Evening', 'Publication_Time_Night'
]
# 各 Podcast ごとに処理
for podcast_name in podcast_test_dfs_clean.keys():
    if podcast_name in podcast_dfs_clean:
        train_df = podcast_dfs_clean[podcast_name]
        test_df = podcast_test_dfs_clean[podcast_name]

        # 特徴量とターゲット分離
        X_train = train_df[base_features]
        y_train = train_df[target_col]
        X_test = test_df[base_features]
        test_ids = test_df[id_col]

        # XGBoost モデル学習
        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

        # モデル保存
        xgb_models[podcast_name] = model
        joblib.dump(model, f'model_xgb_{podcast_name}.pkl')

        # 推論
        y_pred = model.predict(X_test)

        # サブミッション用に整形
        pred_df = pd.DataFrame({
            'id': test_ids,
            'Listening_Time_minutes': y_pred
        })
        test_preds.append(pred_df)

# ========================
# ✅ Submission 作成
# ========================
submission_df = pd.concat(test_preds, ignore_index=True).sort_values('id')
submission_df.to_csv('submission.csv', index=False)
print("submission.csv が出力されました")




