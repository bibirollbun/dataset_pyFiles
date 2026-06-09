import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

train.head()


X = train.drop(columns=["loan_paid_back", "id"])
y = train["loan_paid_back"]

# train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 目的変数と特徴量を分ける
target_col = "loan_paid_back"

# train から目的変数を除いた特徴量＋ test を縦に結合
train_features = train.drop(columns=[target_col])
test_features = test.copy()

full = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# object 型の列（カテゴリ変数）を確認
cat_cols = full.select_dtypes(include=["object"]).columns
print("Categorical columns:", list(cat_cols))

# one-hot エンコーディング（ダミー変数化）
full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=False)

# 再び train / test に分割
n_train = len(train)
full_train = full_encoded.iloc[:n_train, :]
full_test = full_encoded.iloc[n_train:, :]

# モデルに使う特徴量（id は除く）
X = full_train.drop(columns=["id"])
X_test = full_test.drop(columns=["id"])

y = train[target_col]

# train / validation スプリット
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train.shape, X_val.shape, X_test.shape



from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

val_pred = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_pred)
roc_auc



test_pred = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()


