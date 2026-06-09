# -----------------------------------------------------------------
# このセルは変更不可
# -----------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# データの読み込み
train = pd.read_csv("../input/playground-series-s4e4/train.csv")
test = pd.read_csv("../input/playground-series-s4e4/test.csv")

# 列の特定
ID_COL = "id"
OBJECT_COL = "Rings"
DROP_COLs = []

# 数値型のみ抽出
num_features = train.select_dtypes(include=['int64', 'float64'])
num_features = num_features.columns.drop(DROP_COLs + [OBJECT_COL])
print(num_features)

# 数値型特徴量のみで、中央値で補完
train[num_features] = train[num_features].fillna(train[num_features].median())
test[num_features] = test[num_features].fillna(train[num_features].median())

# 目的変数と説明変数に分割
X_train = train[num_features]
X_test = test[num_features]
y_train = train[OBJECT_COL]


# -----------------------------------------------------------------
# (A) ここで目的変数に1を足してlogを取る(log1p変換)
# その他の処理は追加不可
# -----------------------------------------------------------------

y_train = np.log1p(y_train)



# -----------------------------------------------------------------
# このセルは変更不可
# -----------------------------------------------------------------
# モデルの学習
model = RandomForestRegressor(n_estimators=50, random_state=71)
model.fit(X_train, y_train)

# テストデータの予測
preds = model.predict(X_test)


# -----------------------------------------------------------------
# (B)ここで予測値から1を引いてexpを取って元のスケールに戻す(expm1変換)
# その他の処理は追加不可
# -----------------------------------------------------------------
preds = np.expm1(preds)




# -------------------------------------------------------------
# このセルは変更不可
# -------------------------------------------------------------
# 提出用データ作成
submission = pd.DataFrame({ID_COL: test[ID_COL], OBJECT_COL: preds})
submission.to_csv("submission.csv", index=False)

print(preds)
print(np.min(preds), np.mean(preds), np.max(preds))


# -----------------------------------------------------------------
# 以下から追加コード（表示用コード）
# -----------------------------------------------------------------

# (1) 学習用レコードの件数
num_train_records = len(train)
print("学習用レコードの件数:", num_train_records)

# (2) テスト用レコードの件数
num_test_records = len(test)
print("テスト用レコードの件数:", num_test_records)

# (3) 説明変数（特徴量）の個数
# ここでは、ID列と目的変数以外の全列を「説明変数」とする
all_features = train.columns.drop([ID_COL, OBJECT_COL] + DROP_COLs)
num_features_total = len(all_features)
print("説明変数（特徴量）の個数:", num_features_total)

# (4) 説明変数のうち，数値型である変数の個数
num_features_numeric = train[all_features].select_dtypes(include=['int64', 'float64']).columns
print("説明変数（数値型）の個数:", len(num_features_numeric))

# (5) 説明変数のうち，文字列型である変数の個数
str_features = train[all_features].select_dtypes(include=['object']).columns
print("説明変数（文字列型）の個数:", len(str_features))

# (6) 目的変数の個数（学習用レコードと同じ数になる）
print("目的変数の個数:", len(y_train))



import pandas as pd

# データの読み込み
train = pd.read_csv("../input/playground-series-s4e4/train.csv")

# 列の特定
ID_COL = "id"
OBJECT_COL = "Rings"

# 説明変数（特徴量）のリストを取得
feature_columns = train.columns.drop([ID_COL, OBJECT_COL])

# 特徴量を表示
print("説明変数（特徴量）の一覧:")
print(feature_columns.tolist())


