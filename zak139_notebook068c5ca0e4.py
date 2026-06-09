import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


# ---------------------- 1. 数据加载 ----------------------
train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

print(" 数据加载完成（Kaggle input路径）")
print(f"训练集: {train_df.shape} | 测试集: {test_df.shape}")

# 自动识别关键列
id_col = "id"
target_col = [col for col in train_df.columns if col != id_col and train_df[col].nunique() == 2][0]
submission_col = sample_sub.columns[1]
test_ids = test_df[id_col]
print(f"\n目标列={target_col} | 提交概率列={submission_col}")



# ---------------------- 2. 数据预处理 ----------------------
X = train_df.drop([target_col, id_col], axis=1)
y = train_df[target_col]
X_test = test_df.drop(id_col, axis=1)

# 分类/数值特征识别
cat_features = [col for col in X.columns if X[col].dtype == "object" or (X[col].nunique() <= 10 and not pd.api.types.is_numeric_dtype(X[col]))]
num_features = [col for col in X.columns if col not in cat_features]

# 分类特征编码（防unseen labels）
for col in cat_features:
    le = LabelEncoder()
    all_vals = pd.concat([X[col].fillna("Unknown"), X_test[col].fillna("Unknown")]).astype(str).unique()
    le.fit(all_vals)
    X[col] = X[col].fillna("Unknown").astype(str).map(lambda x: x if x in le.classes_ else "Unknown")
    X_test[col] = X_test[col].fillna("Unknown").astype(str).map(lambda x: x if x in le.classes_ else "Unknown")
    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])

# 数值特征预处理
scaler = StandardScaler()
for col in num_features:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)
X[num_features] = scaler.fit_transform(X[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])



# ---------------------- 3. 双模型训练 ----------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM（用callbacks实现早停）
lgb_model = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.08, max_depth=5,
    num_leaves=18, subsample=0.8, colsample_bytree=0.8,
    random_state=42, objective="binary", metric="auc", verbose=-1
)
early_stopping = lgb.early_stopping(stopping_rounds=50, verbose=False)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping],
    categorical_feature=cat_features
)
lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1]
print(f" LightGBM | 验证集AUC: {roc_auc_score(y_val, lgb_model.predict_proba(X_val)[:, 1]):.4f}")

# 逻辑回归
lr_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight="balanced", solver="liblinear")
lr_model.fit(X_train, y_train)
lr_test_prob = lr_model.predict_proba(X_test)[:, 1]
print(f" 逻辑回归 | 验证集AUC: {roc_auc_score(y_val, lr_model.predict_proba(X_val)[:, 1]):.4f}")



# ---------------------- 4. 保存到Kaggle working目录 ----------------------
# 双模型融合
total_auc = roc_auc_score(y_val, lgb_model.predict_proba(X_val)[:, 1]) + roc_auc_score(y_val, lr_model.predict_proba(X_val)[:, 1])
fusion_prob = (roc_auc_score(y_val, lgb_model.predict_proba(X_val)[:, 1])/total_auc)*lgb_test_prob + (roc_auc_score(y_val, lr_model.predict_proba(X_val)[:, 1])/total_auc)*lr_test_prob

# 保存提交文件到/kaggle/working/
lgb_sub = pd.DataFrame({id_col: test_ids, submission_col: lgb_test_prob.round(4)})
lr_sub = pd.DataFrame({id_col: test_ids, submission_col: lr_test_prob.round(4)})
fusion_sub = pd.DataFrame({id_col: test_ids, submission_col: fusion_prob.round(4)})

# 保存到Kaggle working目录
lgb_sub.to_csv("/kaggle/working/lgb_submission.csv", index=False)
lr_sub.to_csv("/kaggle/working/lr_submission.csv", index=False)
fusion_sub.to_csv("/kaggle/working/submission.csv", index=False)

print("\n 文件已保存到Kaggle working目录！")
print("1. lgb_submission.csv")
print("2. lr_submission.csv")
print("3. submission.csv")

