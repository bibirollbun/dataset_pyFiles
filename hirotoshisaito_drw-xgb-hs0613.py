import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import pandas as pd
import numpy as np

# データの読み込み
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

# データの確認
print(f"Train Data Shape: {train.shape}")
print(f"Test Data Shape: {test.shape}")
print(f"Sample Submission Shape: {sample_submission.shape}")

# 最初の数行を確認
print(train.head())



# データの列名を確認
print(train.columns)

# timestampをIDとして残し、その他の数値データのみを抽出
numerical_features = train.select_dtypes(include=[np.number])

# 目的変数（label）の取得
target = train['label']

# 特徴量と目的変数の相関を算出
correlations = numerical_features.corrwith(target)

# 相関係数を降順に並べ替え
sorted_correlations = correlations.sort_values(ascending=False)

# 相関係数の絶対値を取って、上位15の特徴量を選択
top_15_features = sorted_correlations.abs().sort_values(ascending=False).head(15)
print("Top 15 Features Based on Absolute Correlation with Target:")
print(top_15_features)

# 目的変数との相関が上位15の特徴量を選択
selected_features = top_15_features.index.tolist()

# 交差特徴量の作成：bid_qty / ask_qty, buy_qty / sell_qty
train['bid_qty_to_ask_qty'] = train['bid_qty'] / train['ask_qty']
train['buy_qty_to_sell_qty'] = train['buy_qty'] / train['sell_qty']

# 新たに作成した交差特徴量をcorrelationsに追加
additional_correlations = train[['bid_qty_to_ask_qty', 'buy_qty_to_sell_qty']].corrwith(target)

# 交差特徴量の相関をcorrelationsに手動で追加
correlations = pd.concat([correlations, additional_correlations])

# 交差特徴量をfinal_featuresに追加
final_features = selected_features + ['bid_qty_to_ask_qty', 'buy_qty_to_sell_qty']

# 最終的に使用する特徴量の相関を表示
final_correlations = correlations[final_features]
print("\nCorrelations of Selected Features with Target:")
print(final_correlations)



from sklearn.preprocessing import StandardScaler

# 使用する特徴量（final_features）を抽出
X = train[final_features]

# StandardScalerを使用してデータを標準化（スケーリング）
scaler = StandardScaler()

# スケーリングされた特徴量
X_scaled = scaler.fit_transform(X)

# スケーリング後のデータをDataFrameに戻す
X_scaled_df = pd.DataFrame(X_scaled, columns=final_features)

# 最初の数行を表示
print(X_scaled_df.head())



import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# 訓練データと特徴量を分割
X = train[final_features]
y = target

# 分割検証のためのKFoldを定義
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# XGBoostモデルの設定
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# クロスバリデーションでモデルを評価
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

# 評価結果（平均二乗誤差）を表示
print(f"Cross-Validation MSE scores: {-cv_scores}")
print(f"Mean CV MSE: {-cv_scores.mean()}")

# 最終的に全データでモデルを学習
model.fit(X, y)

# 学習済みモデルを使って予測を実行（訓練データで予測）
y_pred = model.predict(X)

# 平均二乗誤差（MSE）を計算
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error on Training Data: {mse}")



import pandas as pd

# 正式なテストデータの読み込み
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# 交差特徴量の作成：テストデータにも追加
test['bid_qty_to_ask_qty'] = test['bid_qty'] / test['ask_qty']
test['buy_qty_to_sell_qty'] = test['buy_qty'] / test['sell_qty']

# テストデータから、学習で使用した特徴量（final_features）を抽出
X_test = test[final_features]

# 学習したXGBoostモデルで予測を実施
y_test_pred = model.predict(X_test)

# 提出用ファイルの作成
submission = pd.DataFrame({
    'ID': test.index,  # testデータのインデックスをID列として使用
    'prediction': y_test_pred  # 予測結果をprediction列として保存
})

# 提出用ファイルをCSV形式で保存
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)

# 提出用ファイルの最初の30行を表示
print("First 30 rows of the submission file:")
print(submission.head(30))

# 提出用ファイルの記述統計を表示
print("\nDescriptive statistics of the submission file:")
print(submission.describe())


