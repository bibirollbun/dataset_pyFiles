import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





# データの読み込み
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

# データの確認
print(f"Train Data Shape: {train.shape}")
print(f"Test Data Shape: {test.shape}")
print(f"Sample Submission Shape: {sample_submission.shape}")


# カラム一覧を先頭50個だけ表示（匿名カラムの形式を特定）
print(train.columns[:50])



# 匿名特徴量（X1〜X890相当）の抽出
anon_cols = [col for col in train.columns if col.startswith('X')]

# 相関係数を算出
correlations = train[anon_cols].corrwith(train['label'])

# 相関の絶対値が高い上位50個を取得
top_corr = correlations.abs().sort_values(ascending=False).head(50)
print("Top 50 Features by Absolute Correlation:")
print(top_corr)

# 公開特徴量
public_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

# 特徴量候補：公開5 + 匿名特徴量上位50
selected_features = public_cols + top_corr.index.tolist()



import numpy as np

# === 公開特徴量（常に使う） ===
public_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

# === 派生特徴量（公開系） ===
train['bid_ask_ratio'] = train['bid_qty'] / (train['ask_qty'] + 1e-6)
train['buy_sell_ratio'] = train['buy_qty'] / (train['sell_qty'] + 1e-6)
train['buy_volume_ratio'] = train['buy_qty'] / (train['volume'] + 1e-6)
train['sell_volume_ratio'] = train['sell_qty'] / (train['volume'] + 1e-6)

# 派生特徴量の名前を保持
derived_features = [
    'bid_ask_ratio', 'buy_sell_ratio', 'buy_volume_ratio', 'sell_volume_ratio'
]

# === 匿名特徴量の上位50（すでに前セルで抽出済みの top_corr）===
# 再計算せず再利用
top_50_anon = top_corr.index.tolist()

# === 最小限の欠損/無限大処理：対象列のみ（匿名 + 公開 + 派生） ===
target_cols = public_cols + top_50_anon + derived_features
for col in target_cols:
    train[col] = train[col].replace([np.inf, -np.inf], np.nan).fillna(0)

# === 相関係数を用いた共線性除去（匿名特徴量のみ） ===
def remove_highly_correlated_features(df, features, threshold=0.95):
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    return [f for f in features if f not in drop_cols]

filtered_anon = remove_highly_correlated_features(train, top_50_anon, threshold=0.95)

# === 最終特徴量セット（公開 + 派生 + 共線性除去済み匿名） ===
final_features = public_cols + derived_features + filtered_anon
print(f"最終使用特徴量数: {len(final_features)}")



from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import numpy as np

# ✅ 全体データ数
n_total = len(train)

# ✅ スライス設定（時間順）
slices = {
    'first_50pct': (0, int(n_total * 0.5)),
    'middle_30pct': (int(n_total * 0.5), int(n_total * 0.8)),
    'last_20pct': (int(n_total * 0.8), n_total)
}

# === スコア格納 ===
val_scores = []
val_lens = []
val_preds = []
val_true = None

for slice_name, (start_idx, end_idx) in slices.items():
    print(f"\n[Slice: {slice_name}] Rows {start_idx} to {end_idx}")

    # スライス抽出
    subset = train.iloc[start_idx:end_idx].copy()

    # === 派生特徴量の再計算（データが分割されているため） ===
    subset['bid_ask_ratio'] = subset['bid_qty'] / (subset['ask_qty'] + 1e-6)
    subset['buy_sell_ratio'] = subset['buy_qty'] / (subset['sell_qty'] + 1e-6)
    subset['buy_volume_ratio'] = subset['buy_qty'] / (subset['volume'] + 1e-6)
    subset['sell_volume_ratio'] = subset['sell_qty'] / (subset['volume'] + 1e-6)

    # === 欠損補完（学習で使用する全特徴量に対して） ===
    for col in final_features:
        if col in subset.columns:
            subset[col] = subset[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            print(f"Warning: {col} not found in subset. Filling with 0.")
            subset[col] = 0

    # 特徴量と目的変数を抽出
    X = subset[final_features]
    y = subset['label']

    # 最初の1回のみ検証用ラベルを保存（可視化や解析用途）
    if val_true is None:
        val_true = y.iloc[int(len(y) * 0.8):].reset_index(drop=True)

    # 時系列分割（shuffle=False）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(" - Training XGB")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        tree_method='hist',
        enable_categorical=False,
        verbosity=0,
        early_stopping_rounds=30  # ✅ constructor側に設定（警告対策）
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # ✅ モデル保存（特徴量とともに）
    model_path = f"/kaggle/working/xgb_model_{slice_name}.pkl"
    joblib.dump((model, final_features), model_path)
    print(f"   → Model + features saved to {model_path}")

    # === 検証スコア計算 ===
    y_pred = model.predict(X_val)
    score = pearsonr(y_val, y_pred)[0]
    val_preds.append(y_pred)
    val_lens.append(len(y_val))
    val_scores.append(score)

    print(f"   → Pearson: {score:.5f} (len: {len(y_val)})")

# === スコア集計表示 ===
avg_score = np.mean(val_scores)
weighted_score = np.average(val_scores, weights=val_lens)

print(f"\n✅ Average Pearson (equal):   {avg_score:.5f}")
print(f"✅ Average Pearson (weighted by size): {weighted_score:.5f}")





import pandas as pd
import numpy as np
import joblib  # モデル読み込み用

# === テストデータの読み込み ===
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# === 派生特徴量の追加（学習と同様）===
test['bid_ask_ratio'] = test['bid_qty'] / (test['ask_qty'] + 1e-6)
test['buy_sell_ratio'] = test['buy_qty'] / (test['sell_qty'] + 1e-6)
test['buy_volume_ratio'] = test['buy_qty'] / (test['volume'] + 1e-6)
test['sell_volume_ratio'] = test['sell_qty'] / (test['volume'] + 1e-6)

# === スライス名と重み設定（後半データ重視） ===
slices = ['first_50pct', 'middle_30pct', 'last_20pct']
weights = {'first_50pct': 0.2, 'middle_30pct': 0.3, 'last_20pct': 0.5}

# === 予測結果初期化 ===
weighted_preds = np.zeros(len(test))

# === 各スライスモデルで推論して加重平均 ===
for slice_name in slices:
    model_path = f'/kaggle/working/xgb_model_{slice_name}.pkl'
    print(f"Loading model: {model_path}")
    
    # ✅ モデルと使用特徴量を読み込み
    model, final_features = joblib.load(model_path)
    
    # 特徴量がテストデータに存在するか確認・補完
    for col in final_features:
        if col not in test.columns:
            print(f"Warning: {col} not in test set. Filling with 0.")
            test[col] = 0
        else:
            test[col] = test[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 特徴量抽出と予測
    X_test = test[final_features]
    y_pred = model.predict(X_test)

    # 重み付きで加算
    weighted_preds += weights[slice_name] * y_pred

# === 提出ファイルの作成 ===
submission = pd.DataFrame({
    'ID': test['row_id'] if 'row_id' in test.columns else test.index,
    'prediction': weighted_preds
})

# 保存
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)

# === 提出ファイルの確認 ===
print("First 10 rows of the submission file:")
print(submission.head(10))

print("\nSubmission file stats:")
print(submission.describe())


