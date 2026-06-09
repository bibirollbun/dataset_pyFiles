import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


# 1. 加载数据
train = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv")
test = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/test.csv")


# 2. 标签编码
le = LabelEncoder()
y = le.fit_transform(train["target"])
X = train.drop(columns=["id", "target"])
X_test = test.drop(columns=["id"])


# 3. 添加行统计特征 + Top3重要特征log变换
def add_features(df):
    df = df.copy()
    df["row_mean"] = df.mean(axis=1)
    df["row_std"] = df.std(axis=1)
    df["row_nonzero"] = (df != 0).sum(axis=1)
    for feat in ["feat_11", "feat_90", "feat_60"]:  # Top3重要特征
        df[f"{feat}_log"] = np.log1p(df[feat])
    return df

X_feat = add_features(X)
X_test_feat = add_features(X_test)



# 4. KFold交叉验证 + 模型预测
params = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 9,
    "seed": 42,
    "verbosity": 0
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_test_feat), 9))
valid_logloss = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_feat, y)):
    X_train, X_valid = X_feat.iloc[train_idx], X_feat.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test_feat)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    test_preds += model.predict(dtest) / skf.n_splits
    valid_logloss.append(model.best_score)

print(f"平均交叉验证 logloss: {np.mean(valid_logloss):.5f}")



# 5. 生成 submission.csv
submission = pd.DataFrame(test_preds, columns=[f"Class_{i+1}" for i in range(9)])
submission.insert(0, "id", test["id"])
submission.to_csv("/kaggle/working/submission.csv", index=False)

