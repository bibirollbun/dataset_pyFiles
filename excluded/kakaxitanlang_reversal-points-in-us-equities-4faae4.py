import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif

import lightgbm as lgb



TRAIN_CSV = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/train.csv"
TEST_CSV  = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/test.csv"
SAMPLE_SUB = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/sample_submission.csv"

train = pd.read_csv(TRAIN_CSV, low_memory=False)
test  = pd.read_csv(TEST_CSV,  low_memory=False)
sub   = pd.read_csv(SAMPLE_SUB, low_memory=False)

print(train.shape, test.shape)

y_raw = train["class_label"].fillna("None").astype(str)

mapping = {
    "HH": "H", "LH": "H",
    "HL": "L", "LL": "L",
    "H": "H", "L": "L",
    "None": "None", "nan": "None", "N": "None"
}

y = y_raw.replace(mapping)
print(y.value_counts())



meta_cols = ["id", "train_id", "ticker_id", "t", "class_label", "Unnamed: 0"]
features = [c for c in train.columns if c not in meta_cols]

X_train_raw = train[features].apply(pd.to_numeric, errors="coerce")
X_test_raw  = test[features].apply(pd.to_numeric, errors="coerce")

med = X_train_raw.median()
X_train_raw = X_train_raw.fillna(med)
X_test_raw  = X_test_raw.fillna(med)

# 1️⃣ 去掉常数特征
vt = VarianceThreshold(1e-6)
X_train_vt = vt.fit_transform(X_train_raw)
X_test_vt  = vt.transform(X_test_raw)

# 2️⃣ 互信息选前 1000 个
k = min(1000, X_train_vt.shape[1])
selector = SelectKBest(mutual_info_classif, k=k)
selector.fit(X_train_vt, LabelEncoder().fit_transform(y))

X_train = selector.transform(X_train_vt)
X_test  = selector.transform(X_test_vt)

print("Final feature dim:", X_train.shape)



y_stage1 = np.where(y == "None", "None", "Swing")

groups = train["ticker_id"].values
gkf = GroupKFold(n_splits=5)

oof_p_none = np.zeros(len(X_train))
test_p_none = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(gkf.split(X_train, y_stage1, groups)):
    print(f"\nStage1 Fold {fold+1}")

    le = LabelEncoder()
    y_tr = le.fit_transform(y_stage1[tr])
    y_va = le.transform(y_stage1[va])

    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw = dict(zip(np.unique(y_tr), cw))

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.05,
        class_weight=cw,
        random_state=42+fold
    )

    model.fit(
        X_train[tr], y_tr,
        eval_set=[(X_train[va], y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100)]
    )

    idx_none = list(le.classes_).index("None")
    oof_p_none[va] = model.predict_proba(X_train[va])[:, idx_none]
    test_p_none += model.predict_proba(X_test)[:, idx_none] / 5



swing_idx = np.where(y != "None")[0]
X_swing = X_train[swing_idx]
y_swing = y.iloc[swing_idx].values

oof_stage2 = np.zeros((len(swing_idx), 2))
test_stage2 = np.zeros((len(X_test), 2))

groups_swing = train.loc[swing_idx, "ticker_id"].values
gkf2 = GroupKFold(n_splits=5)

for fold, (tr, va) in enumerate(gkf2.split(X_swing, y_swing, groups_swing)):
    print(f"\nStage2 Fold {fold+1}")

    le = LabelEncoder()
    y_tr = le.fit_transform(y_swing[tr])
    y_va = le.transform(y_swing[va])

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.05,
        random_state=100+fold
    )

    model.fit(
        X_swing[tr], y_tr,
        eval_set=[(X_swing[va], y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100)]
    )

    proba_va = model.predict_proba(X_swing[va])
    if le.classes_[0] == "H":
        oof_stage2[va] = proba_va
        test_stage2 += model.predict_proba(X_test) / 5
    else:
        oof_stage2[va] = proba_va[:, ::-1]
        test_stage2 += model.predict_proba(X_test)[:, ::-1] / 5



best_f1, best_thr = -1, None
best_pred = None

for thr in np.linspace(0.2, 0.6, 41):
    pred = np.array(["None"] * len(y), dtype=object)

    for i, idx in enumerate(swing_idx):
        if oof_p_none[idx] < thr:
            pred[idx] = "H" if oof_stage2[i,0] >= oof_stage2[i,1] else "L"

    f1 = f1_score(y, pred, average="macro")
    if f1 > best_f1:
        best_f1, best_thr, best_pred = f1, thr, pred

print("Best thr:", best_thr)
print("OOF Macro F1:", best_f1)
print(pd.Series(best_pred).value_counts())



final_pred = np.array(["None"] * len(X_test), dtype=object)

for i in range(len(X_test)):
    if test_p_none[i] < best_thr:
        final_pred[i] = "H" if test_stage2[i,0] >= test_stage2[i,1] else "L"

out = pd.DataFrame({
    "id": test["id"],
    "class_label": final_pred
})

out.to_csv("submission.csv", index=False)
print(out["class_label"].value_counts())
out.head(10)


