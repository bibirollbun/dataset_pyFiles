# -----------------------------------------------------------------
# このセルは変更不可
# -----------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# データの読み込み
train = pd.read_csv("../input/playground-series-s4e4/train.csv")
test = pd.read_csv("../input/playground-series-s4e4/test.csv")


##
import pandas as pd



# 行数 (1) と (2)
print("学習用レコード件数:", train.shape[0])
print("テスト用レコード件数:", test.shape[0])
# 目的変数名を 'Rings' と仮定
all_cols = train.columns.tolist()
feature_cols = [c for c in all_cols if c != 'Rings']
print("説明変数の個数:", len(feature_cols))
# 説明変数だけを抽出
df_feat = train[feature_cols]

# 数値型
num_cols = df_feat.select_dtypes(include=['int64','float64']).columns
# 文字列型
str_cols = df_feat.select_dtypes(include=['object']).columns

print("数値型変数の個数:", len(num_cols))
print("文字列型変数の個数:", len(str_cols))
# train_y を Series として切り出した後
train_y = train['Rings']
# もし DataFrame なら train_y.shape[1] で列数
print("目的変数の個数:", 1)
##
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

