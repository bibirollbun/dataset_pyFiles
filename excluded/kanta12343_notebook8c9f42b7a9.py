import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 71

# データ読み込み
train = pd.read_csv("/kaggle/input/playground-series-s3e24/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv")

# 目的変数とID
target_col = "smoking"
id_col = "id"
y = train[target_col]
train_id = train[id_col]
test_id = test[id_col]
train = train.drop(columns=[target_col, id_col])
test = test.drop(columns=[id_col])



def feature_engineering(df):
    df["Gtp_log"] = np.log1p(df["Gtp"])
    df["ALT_log"] = np.log1p(df["ALT"])
    df["AST_log"] = np.log1p(df["AST"])
    df["LDL_to_HDL"] = df["LDL"] / (df["HDL"] + 1)
    df["liver_total"] = df["ALT"] + df["AST"] + df["Gtp"]
    df["BMI_like"] = df["weight(kg)"] / ((df["height(cm)"] / 100) ** 2)
    df["hearing_diff"] = abs(df["hearing(right)"] - df["hearing(left)"])
    return df

train = feature_engineering(train)
test = feature_engineering(test)



# 欠損値補完
train = train.fillna("NA")
test = test.fillna("NA")

# Label Encoding
categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 特徴量選択
X_train = train
X_test = test



# モデル定義
xgb_model = XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric="logloss")
lgb_model = LGBMClassifier(random_state=RANDOM_SEED)
cat_model = CatBoostClassifier(verbose=0, random_seed=RANDOM_SEED)

# 学習
xgb_model.fit(X_train, y)
lgb_model.fit(X_train, y)
cat_model.fit(X_train, y)

# 予測
xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
cat_preds = cat_model.predict_proba(X_test)[:, 1]





# グリッド探索によるアンサンブル重みの最適化

best_score = 0
best_weights = (0, 0, 0)

for w1 in [0.1 * i for i in range(11)]:
    for w2 in [0.1 * j for j in range(11)]:
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1:
            continue
        blended = w1 * pred_xgb + w2 * pred_lgb + w3 * pred_cat
        score = roc_auc_score(y_val, blended)
        print(f"{w1:.1f}, {w2:.1f}, {w3:.1f} → {score:.5f}")
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

print(f"Best weights: {best_weights}, Best AUC: {best_score:.5f}")

# 3モデル平均アンサンブル
ensemble_preds = (best_weights[0]*xgb_preds + best_weights[1]*lgb_preds + best_weights[2]*cat_preds)


submission = pd.DataFrame({
    "id": test_id,
    "smoking": ensemble_preds
})
submission.to_csv("submission.csv", index=False)
submission.head()


