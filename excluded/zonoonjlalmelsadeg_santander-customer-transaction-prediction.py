import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


train_df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


train_df.head()


test_df.head()


train_df.shape


test_df.shape


train_df.info()


test_df.info()


train_df.target.describe()


test_df.describe()


train_df.isnull().sum()


test_df.isnull().sum()


counts = train_df["target"].value_counts().sort_index()

plt.figure()
plt.bar(["0", "1"], counts.values)
plt.title("Target Distribution (0 vs 1)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()



num_cols = [c for c in train_df.columns if c.startswith("var_")]
np.random.seed(42)
picked = np.random.choice(num_cols, size=20, replace=False)

corr = train_df[picked].corr().values

plt.figure()
im = plt.imshow(corr, interpolation="nearest")
plt.title("Correlation Heatmap (20 features)")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(ticks=np.arange(len(picked)), labels=picked, rotation=90)
plt.yticks(ticks=np.arange(len(picked)), labels=picked)
plt.tight_layout()
plt.show()


train_df.columns


test_df.columns


feature_cols =[c for c in train_df.columns if c not in['ID_code','target']]


x = train_df[feature_cols].values
y = train_df['target'].values


x.shape


y.shape


# --- تطبيع البيانات ---
from sklearn.impute import SimpleImputer

# Impute missing values using the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(x)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# --- إعداد النموذج ---
log_reg = LogisticRegression(max_iter=1000, solver="lbfgs")

# --- Cross Validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for train_idx, val_idx in cv.split(X_scaled, y):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)

print("AUC Scores:", auc_scores)
print("Mean AUC:", np.mean(auc_scores))


import lightgbm as lgb
from sklearn.metrics import roc_auc_score

train_ds = lgb.Dataset(X_train, label=y_train)
val_ds   = lgb.Dataset(X_val, label=y_val, reference=train_ds)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "max_depth": -1,
    "verbose": -1,
    "is_unbalance": True,   # مهم لعدم التوازن
}

gbm = lgb.train(
    params,
    train_ds,
    num_boost_round=2000,
    valid_sets=[val_ds],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
print("Validation AUC (LightGBM):", roc_auc_score(y_val, preds))



# إعادة التدريب على كامل البيانات باستخدام أفضل عدد جولات
X_full = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)], axis=0).reset_index(drop=True)
y_full = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_val)], axis=0).reset_index(drop=True)
full_ds = lgb.Dataset(X_full, label=y_full)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "max_depth": -1,
    "verbose": -1,
    "is_unbalance": True,
}

best_rounds = gbm.best_iteration if gbm.best_iteration is not None else 200
final_gbm = lgb.train(params, full_ds, num_boost_round=best_rounds)

# Prepare test data
X_test = test_df[feature_cols].values
X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)
test_ids = test_df['ID_code']

# تنبؤ احتمالات على الاختبار
test_proba = final_gbm.predict(X_test_scaled, num_iteration=final_gbm.best_iteration)

# تجهيز ملف التسليم: ID_code + target (احتمالات)
submission = pd.DataFrame({
    "ID_code": test_ids,
    "target": test_proba.clip(0, 1)   # للتأكد أنها ضمن [0,1]
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv", submission.shape)




