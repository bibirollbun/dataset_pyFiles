# ライブラリのインポート
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# データの読み込み
train = pd.read_csv("/kaggle/input/bakery-sales-prediction-summer-2025/train.csv")
test = pd.read_csv("/kaggle/input/bakery-sales-prediction-summer-2025/test.csv")
wetter = pd.read_csv("/kaggle/input/bakery-sales-prediction-summer-2025/wetter.csv")
kiwo = pd.read_csv("/kaggle/input/bakery-sales-prediction-summer-2025/kiwo.csv")
sample_submission = pd.read_csv("/kaggle/input/bakery-sales-prediction-summer-2025/sample_submission.csv")

# 日付型変換と豊富な時系列特徴量の追加
for df in [train, test]:
    df["Datum"] = pd.to_datetime(df["Datum"])
    
    # 基本的な時系列特徴量
    df["Wochentag"] = df["Datum"].dt.dayofweek
    df["Monat"] = df["Datum"].dt.month
    df["Jahr"] = df["Datum"].dt.year
    df["Tag"] = df["Datum"].dt.day
    df["Woche_im_Jahr"] = df["Datum"].dt.isocalendar().week
    df["Quartal"] = df["Datum"].dt.quarter
    df["Tag_im_Jahr"] = df["Datum"].dt.dayofyear
    
    # 曜日関連の特徴量
    df["ist_Wochenende"] = (df["Wochentag"] >= 5).astype(int)
    df["ist_Montag"] = (df["Wochentag"] == 0).astype(int)
    df["ist_Freitag"] = (df["Wochentag"] == 4).astype(int)
    df["ist_Mittwoch"] = (df["Wochentag"] == 2).astype(int)
    
    # 月関連の特徴量
    df["ist_Januar"] = (df["Monat"] == 1).astype(int)
    df["ist_Dezember"] = (df["Monat"] == 12).astype(int)
    df["ist_Sommer"] = df["Monat"].isin([6, 7, 8]).astype(int)
    df["ist_Winter"] = df["Monat"].isin([12, 1, 2]).astype(int)
    df["ist_Fruehling"] = df["Monat"].isin([3, 4, 5]).astype(int)
    df["ist_Herbst"] = df["Monat"].isin([9, 10, 11]).astype(int)
    
    # Monatsende/anfang
    df["ist_Monatsanfang"] = (df["Tag"] <= 5).astype(int)
    df["ist_Monatsende"] = (df["Tag"] >= 25).astype(int)
    df["ist_Monatsmitte"] = ((df["Tag"] >= 10) & (df["Tag"] <= 20)).astype(int)
    
    # 周期的な特徴量（三角関数）
    df["Monat_sin"] = np.sin(2 * np.pi * df["Monat"] / 12)
    df["Monat_cos"] = np.cos(2 * np.pi * df["Monat"] / 12)
    df["Wochentag_sin"] = np.sin(2 * np.pi * df["Wochentag"] / 7)
    df["Wochentag_cos"] = np.cos(2 * np.pi * df["Wochentag"] / 7)
    df["Tag_sin"] = np.sin(2 * np.pi * df["Tag"] / 31)
    df["Tag_cos"] = np.cos(2 * np.pi * df["Tag"] / 31)
    
    # 特別な日付フラグ
    df["ist_Feiertag_Naehe"] = 0  # 後で祝日データがあれば設定
    df["ist_Schulferien"] = 0     # 後で学校休暇データがあれば設定

# wetter, kiwo の日付変換
wetter["Datum"] = pd.to_datetime(wetter["Datum"])
kiwo["Datum"] = pd.to_datetime(kiwo["Datum"])

# 結合関数と追加の特徴量エンジニアリング
def merge_external_data(df):
    df = df.merge(wetter, on="Datum", how="left")
    df = df.merge(kiwo, on="Datum", how="left")
    
    # 天気データがある場合の追加特徴量
    if 'Temperatur' in df.columns:
        df['Temperatur_squared'] = df['Temperatur'] ** 2
        df['ist_heiss'] = (df['Temperatur'] > 25).astype(int)
        df['ist_kalt'] = (df['Temperatur'] < 5).astype(int)
        df['Temperatur_kategorie'] = pd.cut(df['Temperatur'], 
                                          bins=[-np.inf, 0, 10, 20, 30, np.inf], 
                                          labels=['sehr_kalt', 'kalt', 'mild', 'warm', 'heiss'])
    
    if 'Niederschlag' in df.columns:
        df['ist_regen'] = (df['Niederschlag'] > 0).astype(int)
        df['starker_regen'] = (df['Niederschlag'] > 10).astype(int)
        df['Niederschlag_log'] = np.log1p(df['Niederschlag'])
    
    if 'Windgeschwindigkeit' in df.columns:
        df['ist_windig'] = (df['Windgeschwindigkeit'] > 15).astype(int)
        df['Wind_kategorie'] = pd.cut(df['Windgeschwindigkeit'],
                                    bins=[0, 5, 15, 25, np.inf],
                                    labels=['schwach', 'maessig', 'stark', 'sehr_stark'])
    
    if 'Luftfeuchtigkeit' in df.columns:
        df['ist_feucht'] = (df['Luftfeuchtigkeit'] > 80).astype(int)
        df['ist_trocken'] = (df['Luftfeuchtigkeit'] < 40).astype(int)
    
    # KIWO データがある場合の追加特徴量
    kiwo_cols = [col for col in df.columns if 'kiwo' in col.lower() or any(x in col for x in ['Warengruppe', 'Bestellmenge'])]
    for col in kiwo_cols:
        if df[col].dtype in ['int64', 'float64']:
            df[f'{col}_log'] = np.log1p(df[col].fillna(0))
            df[f'{col}_sqrt'] = np.sqrt(df[col].fillna(0))
            df[f'{col}_ist_null'] = df[col].isnull().astype(int)
    
    # ラグ特徴量（前日、前週の情報）- 簡易版
    df = df.sort_values('Datum')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['Jahr', 'Monat', 'Tag', 'Wochentag'] and len(df[col].dropna()) > 0:
            try:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag7'] = df[col].shift(7)
                df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_rolling_std_7'] = df[col].rolling(window=7, min_periods=1).std()
            except Exception as e:
                print(f"警告: {col} のラグ特徴量作成でエラー: {e}")
    
    return df

train = merge_external_data(train)
test = merge_external_data(test)

# さらなる相互作用特徴量の追加
def add_interaction_features(df):
    # 曜日と月の相互作用
    df['Wochentag_Monat'] = df['Wochentag'].astype(str) + '_' + df['Monat'].astype(str)
    
    # 季節と曜日の相互作用
    df['Saison_Wochentag'] = ''
    df.loc[df['ist_Sommer'] == 1, 'Saison_Wochentag'] = 'Sommer_'
    df.loc[df['ist_Winter'] == 1, 'Saison_Wochentag'] = 'Winter_'
    df.loc[df['ist_Fruehling'] == 1, 'Saison_Wochentag'] = 'Fruehling_'
    df.loc[df['ist_Herbst'] == 1, 'Saison_Wochentag'] = 'Herbst_'
    df['Saison_Wochentag'] = df['Saison_Wochentag'] + df['Wochentag'].astype(str)
    
    # 天気と曜日の相互作用（天気データがある場合）
    if 'ist_regen' in df.columns:
        df['Wetter_Wochentag'] = df['ist_regen'].astype(str) + '_' + df['ist_Wochenende'].astype(str)
    
    # 数値特徴量の組み合わせ（安全に処理）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    safe_cols = [col for col in numeric_cols if not any(x in col for x in ['lag', 'rolling', 'sin', 'cos', '_times_', '_plus_'])][:5]
    
    for i, col1 in enumerate(safe_cols):
        for col2 in safe_cols[i+1:]:
            if col1 != col2:
                try:
                    # 欠損値を0で埋めてから計算
                    val1 = df[col1].fillna(0)
                    val2 = df[col2].fillna(0)
                    df[f'{col1}_times_{col2}'] = val1 * val2
                    df[f'{col1}_plus_{col2}'] = val1 + val2
                except Exception as e:
                    print(f"警告: {col1}と{col2}の相互作用特徴量作成でエラー: {e}")
    
    return df

train = add_interaction_features(train)
test = add_interaction_features(test)

# 除外したい特徴量を明示的に指定
drop_cols = ["Datum"]  # 基本的に除外する列

# 追加で除外したい特徴量があれば以下に指定
exclude_features = [
    # 例: "特徴量名1", "特徴量名2"
    # ラグ特徴量で欠損が多い場合は除外を検討
    # "Temperatur_lag1", "Niederschlag_lag7"
    # 相互作用特徴量で不要なものがあれば除外
    # "Wochentag_times_Monat", 
]

# 全ての除外列をまとめる
all_drop_cols = drop_cols + exclude_features

print(f"除外する特徴量: {all_drop_cols}")

y = train["Umsatz"]
X = train.drop(columns=["Umsatz"] + all_drop_cols, errors='ignore')
X_test = test.drop(columns=all_drop_cols, errors='ignore')

print(f"学習に使用する特徴量数: {X.shape[1]}")
print(f"使用する特徴量: {list(X.columns)}")

# 重要度に基づく特徴量選択を有効にする場合は以下をTrueに設定
USE_FEATURE_SELECTION = False
MIN_IMPORTANCE_THRESHOLD = 0.01  # この値未満の重要度の特徴量を除外

if USE_FEATURE_SELECTION:
    print(f"\n*** 重要度ベースの特徴量選択が有効です（閾値: {MIN_IMPORTANCE_THRESHOLD}） ***")
    print("まず全特徴量でモデルを学習して重要度を計算し、その後再学習します。")

# カテゴリ変数のエンコーディング（Random Forestは数値データが必要）
cat_cols = X.select_dtypes(include="object").columns.tolist()
label_encoders = {}

print(f"エンコーディング対象のカテゴリ変数: {cat_cols}")

# 訓練データとテストデータを結合してエンコーディング
all_data = pd.concat([X, X_test], axis=0, ignore_index=True)

# カテゴリ変数を安全にエンコーディング
for col in cat_cols:
    le = LabelEncoder()
    # 欠損値を文字列として処理
    all_data[col] = all_data[col].fillna('missing').astype(str)
    all_data[col] = le.fit_transform(all_data[col])
    label_encoders[col] = le

# 訓練データとテストデータに分割
X_encoded = all_data.iloc[:len(X)].copy()
X_test_encoded = all_data.iloc[len(X):].copy()

# データ型を確実に数値型にする
for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object':
        print(f"警告: {col} がまだobject型です。強制的に数値変換します。")
        X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
        X_test_encoded[col] = pd.to_numeric(X_test_encoded[col], errors='coerce')

# 欠損値の処理（数値特徴量は中央値で埋める）
print(f"エンコーディング後のデータ型確認:")
print(X_encoded.dtypes.value_counts())

# 全て数値型であることを確認
numeric_cols = X_encoded.columns
X_encoded[numeric_cols] = X_encoded[numeric_cols].fillna(X_encoded[numeric_cols].median())
X_test_encoded[numeric_cols] = X_test_encoded[numeric_cols].fillna(X_encoded[numeric_cols].median())

# 無限大値やNaNが残っていないかチェック
print(f"訓練データの無限大値: {np.isinf(X_encoded).sum().sum()}")
print(f"訓練データのNaN: {X_encoded.isnull().sum().sum()}")
print(f"テストデータの無限大値: {np.isinf(X_test_encoded).sum().sum()}")
print(f"テストデータのNaN: {X_test_encoded.isnull().sum().sum()}")

# 無限大値を中央値で置換
X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
X_test_encoded = X_test_encoded.replace([np.inf, -np.inf], np.nan)
X_encoded = X_encoded.fillna(X_encoded.median())
X_test_encoded = X_test_encoded.fillna(X_encoded.median())

# Random Forestモデルの定義
model = RandomForestRegressor(
    n_estimators=5000,          # 木の数
    max_depth=10,               # 木の深さ
    min_samples_split=5,        # 分割に必要な最小サンプル数
    min_samples_leaf=2,         # 葉ノードの最小サンプル数
    random_state=42,            # ランダムシード
    n_jobs=-1,                  # 並列処理
    oob_score=True              # Out-of-Bag score を計算
)

# 学習
model.fit(X_encoded, y)

# Out-of-Bag スコアの表示
print(f"OOB Score: {model.oob_score_:.4f}")

# 予測
preds = model.predict(X_test_encoded)

# 提出ファイルの作成
submission = sample_submission.copy()
submission["Umsatz"] = preds
submission.to_csv("submission.csv", index=False)

# 特徴量重要度の表示と分析
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特徴量重要度（上位10個）:")
print(feature_importance.head(10))

# 全特徴量の重要度をCSVで保存
feature_importance.to_csv("feature_importance.csv", index=False)

# 重要度の閾値を設定して重要な特徴量のみ選択（オプション）
threshold = 0.01  # 1%以上の重要度を持つ特徴量のみ
important_features = feature_importance[feature_importance['importance'] >= threshold]['feature'].tolist()
print(f"\n重要度{threshold*100}%以上の特徴量数: {len(important_features)}")
print(f"全特徴量数: {len(feature_importance)}")

# 累積重要度の計算
feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
print(f"\n上位10特徴量で全体の{feature_importance.head(10)['cumulative_importance'].iloc[-1]:.1%}の重要度をカバー")

# 低重要度特徴量の表示（除外候補として参考に）
low_importance_features = feature_importance[feature_importance['importance'] < 0.005]['feature'].tolist()
if low_importance_features:
    print(f"\n低重要度特徴量（0.5%未満、除外候補）:")
    print(low_importance_features[:10])  # 上位10個のみ表示

# 重要度に基づく特徴量選択が有効な場合、重要な特徴量のみで再学習
if USE_FEATURE_SELECTION:
    selected_features = feature_importance[feature_importance['importance'] >= MIN_IMPORTANCE_THRESHOLD]['feature'].tolist()
    print(f"\n選択された特徴量数: {len(selected_features)} (元: {len(feature_importance)})")
    
    # 重要な特徴量のみでデータを再作成
    X_selected = X_encoded[selected_features]
    X_test_selected = X_test_encoded[selected_features]
    
    # 新しいモデルで再学習
    model_selected = RandomForestRegressor(
        n_estimators=1000,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )
    
    model_selected.fit(X_selected, y)
    print(f"特徴量選択後のOOB Score: {model_selected.oob_score_:.4f}")
    
    # 予測も選択された特徴量で実行
    preds = model_selected.predict(X_test_selected)
    print("特徴量選択後のモデルで予測を実行しました。")
else:
    print(f"\n重要度ベースの特徴量選択は無効です。全特徴量を使用します。")

