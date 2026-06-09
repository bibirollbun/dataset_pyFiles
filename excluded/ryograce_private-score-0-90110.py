import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool



train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.shape, test.shape


train.head()


def add_feature(df):
    df=df.copy()
    df["humidity_cloud"] = df["humidity"] * df["cloud"]
    df["humidity_temp"] = df["humidity"] * df["temparature"]




train.info()


num_cols = train.select_dtypes(include = ["int64", "float64"]).columns.tolist()
if "id" in num_cols:
    num_cols.remove("id")

n_features = len(num_cols)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(train[col].dropna(), bins=50, ax=axes[i])
    axes[i].set_title(col)
    axes[i].grid(True, axis="y")

plt.tight_layout()
plt.show()


X = train.drop(columns=["id", "rainfall"])
y = train["rainfall"]
X_test = test.drop(columns = ["id"])


print(f"特徴量数: {X.shape[1]}")


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(X), dtype=float)
scores_lgb = []
test_pred_lgb = np.zeros(len(X_test), dtype=float)  

for fold, (tr_idx, va_idx) in enumerate(skf.split(X,y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_test = lgb.Dataset(X_va, label=y_va)

    params = {
        "objective":"binary",
        "metric" : "auc",
        "learning_rate": 0.03,
        "num_leaves":63,
        "feature_fraction":0.8,
        "bagging_fraction":0.8,
        "bagging_freq":1,
        "verbose":-1,
        "device_type": "cpu"
    }

    model_lgb = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_test],
        callbacks=[lgb.early_stopping(stopping_rounds=300),
                  lgb.log_evaluation(period=500)],
    )
    

    va_pred = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
    oof_lgb[va_idx] = va_pred
    auc = roc_auc_score(y_va, va_pred)
    scores_lgb.append(auc)
    print(f"[LGBM] fold {fold}: AUC={auc:.6f}")

    test_pred_lgb += model_lgb.predict(
        X_test, num_iteration=model_lgb.best_iteration
    ) / skf.n_splits


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_cb = np.zeros(len(X), dtype=float)
test_pred_cb = np.zeros(len(X_test), dtype=float)
scores_cb = []

cat_features = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    valid_pool = Pool(X_va, y_va, cat_features=cat_features)
    test_pool  = Pool(X_test,       cat_features=cat_features)

    model_cb = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=6,
        random_strength=1.0,
        bagging_temperature=0.5,
        random_seed=42 + fold,
        verbose=500,
        od_type="Iter",
        od_wait=300,
        task_type="CPU",  
    )

    model_cb.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    va_pred = model_cb.predict_proba(valid_pool)[:, 1]
    oof_cb[va_idx] = va_pred
    auc = roc_auc_score(y_va, va_pred)
    scores_cb.append(auc)
    print(f"[CatBoost] fold {fold}: AUC={auc:.6f}")

    test_pred_cb += model_cb.predict_proba(test_pool)[:, 1] / skf.n_splits

print("CatBoost CV AUC =", roc_auc_score(y, oof_cb), "mean =", np.mean(scores_cb))


print("CatBoost CV AUC =", roc_auc_score(y, oof_cb), "mean =", np.mean(scores_cb))


w_lgb = 0.3
w_cb  = 0.7

oof_blend = w_lgb * oof_lgb + w_cb * oof_cb
test_pred_blend = w_lgb * test_pred_lgb + w_cb * test_pred_cb

print("Blend CV AUC =", roc_auc_score(y, oof_blend))


importances = model_lgb.feature_importance(importance_type="gain")
fi = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

fi["importance_pct"] = fi["importance"] / fi["importance"].sum() * 100
fi["cum_importance_pct"] = fi["importance_pct"].cumsum()

print(fi)


print("LGBM CV AUC =", roc_auc_score(y, oof_lgb), "mean =", np.mean(scores_lgb))


sub_lgb = pd.DataFrame({
    "id": test["id"],
    "rainfall": test_pred_blend
})
sub_lgb.to_csv("submission_lgbm.csv", index=False)
print("saved submission_blend.csv")

