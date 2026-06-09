import numpy as np
import pandas as pd
import xgboost as xgb

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from category_encoders import CatBoostEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ===========================
# 1. 读取数据
# ===========================
train = pd.read_csv('../input/playground-series-s5e11/train.csv')
test  = pd.read_csv('../input/playground-series-s5e11/test.csv')

y = train['loan_paid_back']
test_id = test['id']

train = train.drop(['loan_paid_back'], axis=1)

# 合并，保证特征工程和编码一致
data = pd.concat([train, test], axis=0).reset_index(drop=True)


# ===========================
# 2. 特征工程
# ===========================

# (1) log 特征
for col in ['annual_income', 'loan_amount', 'interest_rate', 'credit_score']:
    data[col + "_log"] = np.log1p(data[col])

# (2) 比例特征
data["debt_income_ratio"] = data["debt_to_income_ratio"]
data["credit_per_income"] = data["credit_score"] / (data["annual_income"] + 1)
data["loan_income_ratio"] = data["loan_amount"] / (data["annual_income"] + 1)
data["loan_credit_ratio"] = data["loan_amount"] / (data["credit_score"] + 1)
data["interest_credit_ratio"] = data["interest_rate"] / (data["credit_score"] + 1)

# (3) 交互特征
data["income_interest_mul"] = data["annual_income"] * data["interest_rate"]
data["income_credit_mul"]   = data["annual_income"] * data["credit_score"]
data["loan_credit_mul"]     = data["loan_amount"] * data["credit_score"]

# (4) 分箱特征
num_cols = ['annual_income', 'loan_amount', 'credit_score', 'interest_rate']
for col in num_cols:
    data[col + "_bin"] = pd.qcut(data[col], 10, duplicates='drop').cat.codes


# ===========================
# 3. 类别特征 CatBoost 编码
# ===========================
cate_cols = data.select_dtypes(include="object").columns.tolist()
data[cate_cols] = data[cate_cols].fillna("missing")

encoder = CatBoostEncoder(cols=cate_cols)

train_df = data.iloc[:len(train)].copy()
test_df  = data.iloc[len(train):].copy()

# 对类别列进行 target encoding
train_encoded = encoder.fit_transform(train_df[cate_cols], y)
test_encoded  = encoder.transform(test_df[cate_cols])

# 删除原始类别列，避免 object dtype
train_df = train_df.drop(columns=cate_cols)
test_df  = test_df.drop(columns=cate_cols)

# 拼上编码后的列
train_df = pd.concat([train_df.reset_index(drop=True),
                      train_encoded.reset_index(drop=True)], axis=1)
test_df  = pd.concat([test_df.reset_index(drop=True),
                      test_encoded.reset_index(drop=True)], axis=1)


# ===========================
# 4. 标准化 & 类型统一
# ===========================
all_features = train_df.columns.tolist()
all_features.remove("id")

scaler = StandardScaler()
train_df[all_features] = scaler.fit_transform(train_df[all_features]).astype("float32")
test_df[all_features]  = scaler.transform(test_df[all_features]).astype("float32")


# ===========================
# 5. 定义三种一层模型
# ===========================
params_xgb = {
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 2,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.2,
    "n_estimators": 1100,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
}

model_xgb = xgb.XGBClassifier(**params_xgb)

model_lgb = LGBMClassifier(
    n_estimators=1300,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.2,
    random_state=42,
    objective="binary",
    metric="auc"
)

model_cb = CatBoostClassifier(
    iterations=1200,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=3.0,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=False
)



# ===========================
# 6. KFold 训练一层模型，生成 oof 预测
# ===========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_cb  = np.zeros(len(train_df))

pred_xgb = np.zeros(len(test_df))
pred_lgb = np.zeros(len(test_df))
pred_cb  = np.zeros(len(test_df))

X = train_df[all_features].values
X_test = test_df[all_features].values
y_np = y.values

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_np), 1):
    print(f"Fold {fold} ...")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y_np[tr_idx], y_np[val_idx]

    # XGBoost
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    pred_xgb += model_xgb.predict_proba(X_test)[:, 1] / kf.n_splits

    # LightGBM (无 verbose 参数)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[]  # 安全关闭日志
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    pred_lgb += model_lgb.predict_proba(X_test)[:, 1] / kf.n_splits

    # CatBoost（可以保持 verbose=False）
    model_cb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_cb[val_idx] = model_cb.predict_proba(X_val)[:, 1]
    pred_cb += model_cb.predict_proba(X_test)[:, 1] / kf.n_splits


# 一层整体 AUC（看下实力）
auc_xgb = roc_auc_score(y_np, oof_xgb)
auc_lgb = roc_auc_score(y_np, oof_lgb)
auc_cb  = roc_auc_score(y_np, oof_cb)
print(f"AUC XGB: {auc_xgb:.5f} | LGB: {auc_lgb:.5f} | CB: {auc_cb:.5f}")

# 也可以看简单平均融合的 AUC
oof_mean = (oof_xgb + oof_lgb + oof_cb) / 3
print(f"AUC simple blend (level-1): {roc_auc_score(y_np, oof_mean):.5f}")


# ===========================
# 7. 二层 Stacking：用一层预测当特征
# ===========================
meta_train = np.vstack([oof_xgb, oof_lgb, oof_cb]).T   # (n_train, 3)
meta_test  = np.vstack([pred_xgb, pred_lgb, pred_cb]).T  # (n_test, 3)

meta_model = LogisticRegression(
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    n_jobs=-1
)

meta_model.fit(meta_train, y_np)

oof_meta = meta_model.predict_proba(meta_train)[:, 1]
auc_meta = roc_auc_score(y_np, oof_meta)
print(f"AUC level-2 stacking: {auc_meta:.5f}")


# ===========================
# 8. 预测 test，生成提交文件
# ===========================
final_pred = meta_model.predict_proba(meta_test)[:, 1]

submission = pd.DataFrame({
    "id": test_id,
    "loan_paid_back": final_pred
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv 已生成在 /kaggle/working/ 下，可提交。")

