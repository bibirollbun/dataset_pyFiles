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
y_train = np.log1p(y_train)
# その他の処理は追加不可
# -----------------------------------------------------------------


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
preds = np.expm1(preds)
# その他の処理は追加不可
# -----------------------------------------------------------------


# -------------------------------------------------------------
# このセルは変更不可
# -------------------------------------------------------------
# 提出用データ作成
submission = pd.DataFrame({ID_COL: test[ID_COL], OBJECT_COL: preds})
submission.to_csv("submission.csv", index=False)

print(preds)
print(np.min(preds), np.mean(preds), np.max(preds))

