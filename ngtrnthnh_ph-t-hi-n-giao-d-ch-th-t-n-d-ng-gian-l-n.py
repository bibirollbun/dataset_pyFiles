# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Cell 1: Import thư viện
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
sns.set_style("whitegrid")


# Cell 2: Load dữ liệu
TRAIN_PATH = "/kaggle/input/detect-fraudulent-credit-card-transactions-btl/train.csv"
TEST_PATH = "/kaggle/input/detect-fraudulent-credit-card-transactions-btl/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


# Cell 3: Kiểm tra Class
print(train.info())
print(train["Class"].value_counts())
print(train["Class"].value_counts(normalize=True))


# Cell 4: Thống kê Train,Duplicate,Missing
print(train.describe().T)

print("Duplicate rows:", train.duplicated().sum())
print("Missing values:\n", train.isna().sum().sort_values(ascending=False).head(10))


# Cell 5: Phân phối Amount & Time
fig, ax = plt.subplots(1,2, figsize=(14,5))
sns.histplot(train["Amount"], bins=100, ax=ax[0], kde=False)
ax[0].set_title("Distribution of Amount")
sns.histplot(np.log1p(train["Amount"]), bins=100, ax=ax[1], kde=False)
ax[1].set_title("Distribution of log(Amount+1)")
plt.show()

plt.figure(figsize=(12,5))
sns.histplot(train["Time"], bins=100, kde=False)
plt.title("Distribution of Time")
plt.show()


# Cell 6: Class-conditional plots
fig, ax = plt.subplots(1,2, figsize=(14,5))
sns.kdeplot(data=train, x="Amount", hue="Class", ax=ax[0], common_norm=False)
ax[0].set_title("Amount by Class")
sns.kdeplot(data=train, x="Time", hue="Class", ax=ax[1], common_norm=False)
ax[1].set_title("Time by Class")
plt.show()


# Cell 7: Tương quan
feature_cols = [c for c in train.columns if c not in ["id", "Class"]]
corrs = []
for c in feature_cols:
    r, p = pointbiserialr(train[c], train["Class"])
    corrs.append((c, r))
corr_df = pd.DataFrame(corrs, columns=["feature","corr"]).sort_values("corr", ascending=False)
print(corr_df.head(10))


# Cell 8: Time of day pattern
train["hour"] = (train["Time"] % (24*3600)) // 3600
plt.figure(figsize=(12,5))
sns.countplot(data=train, x="hour", hue="Class")
plt.title("Transaction counts by hour of day")
plt.show()


# Cell 9: Feature Engineering - Amount log
train["Amount_log"] = np.log1p(train["Amount"])
test["Amount_log"]  = np.log1p(test["Amount"])

# Time được tính bằng giây từ giao dịch đầu tiên
# => chuyển thành "giờ trong ngày" để xem pattern
train["hour"] = (train["Time"] % (24*3600)) // 3600
test["hour"]  = (test["Time"] % (24*3600)) // 3600

# cyclical encoding
train["hour_sin"] = np.sin(2*np.pi*train["hour"]/24)
train["hour_cos"] = np.cos(2*np.pi*train["hour"]/24)
test["hour_sin"]  = np.sin(2*np.pi*test["hour"]/24)
test["hour_cos"]  = np.cos(2*np.pi*test["hour"]/24)

# flag ban đêm
train["is_night"] = train["hour"].between(0,6).astype(int)
test["is_night"]  = test["hour"].between(0,6).astype(int)

# Z-score Amount theo giờ
stats = train.groupby("hour")["Amount_log"].agg(["mean","std"]).reset_index()
stats["std"].replace(0, 1e-6, inplace=True)

def apply_zscore(df, stats):
    df = df.merge(stats, on="hour", how="left")
    return (df["Amount_log"] - df["mean"]) / df["std"]

train["Amount_log_z_h"] = apply_zscore(train, stats)
test["Amount_log_z_h"]  = apply_zscore(test, stats)

# Mahalanobis distance tới class 0
from sklearn.covariance import LedoitWolf
from numpy.linalg import inv

v_cols = [c for c in train.columns if c.startswith("V")]
md_feats = v_cols + ["Amount_log"]

X0 = train.loc[train["Class"]==0, md_feats].values
mu0 = X0.mean(axis=0)
cov0 = LedoitWolf().fit(X0).covariance_
inv_cov0 = inv(cov0)

def mahalanobis_rows(X, mu, inv_cov):
    D = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", D, inv_cov, D))

train["md_neg"] = mahalanobis_rows(train[md_feats].values, mu0, inv_cov0)
test["md_neg"]  = mahalanobis_rows(test[md_feats].values,  mu0, inv_cov0)

# Kiểm tra lại dữ liệu
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("New features added: Amount_log, hour, hour_sin, hour_cos, is_night, Amount_log_z_h, md_neg")


# Cell 10: Chuẩn bị dữ liệu cho model
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# Xác định feature và target
drop_cols = ["id", "Time", "Amount", "Class"]
feature_cols = [c for c in train.columns if c not in drop_cols]

X = train[feature_cols]
y = train["Class"]
X_test = test[feature_cols]

print("Feature count:", len(feature_cols))

# Cross-validation (5-fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_scores, pr_scores = [], []
test_pred = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        C=1.0
    )
    clf.fit(X_tr_s, y_tr)

    # predict validation
    y_pred_val = clf.predict_proba(X_val_s)[:, 1]
    roc = roc_auc_score(y_val, y_pred_val)
    pr = average_precision_score(y_val, y_pred_val)

    roc_scores.append(roc)
    pr_scores.append(pr)

    print(f"Fold {fold}: ROC AUC={roc:.4f}, PR AUC={pr:.4f}")

    # predict test (trung bình qua folds)
    test_pred += clf.predict_proba(X_test_s)[:, 1] / cv.n_splits

print("\nMean ROC AUC:", np.mean(roc_scores))
print("Mean PR AUC:", np.mean(pr_scores))

# Submission file
y_pred_label = (test_pred >= 0.5).astype(int)
sub = pd.DataFrame({"id": test["id"], "class": y_pred_label})
sub.to_csv("submission.csv", index=False)
print("Submission preview:")
print(sub.head())

