# Importslib & global config
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình hiển thị
pd.set_option("display.max_columns", 50)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

DATA_DIR = Path("/kaggle/input/talkingdata-adtracking-fraud-detection")

print("Files in DATA_DIR:")
for p in DATA_DIR.iterdir():
    print("  -", p.name)



# Đọc train_sample + thống kê nhanh


sample_path = DATA_DIR / "train_sample.csv"
df = pd.read_csv(sample_path)

print("Shape:", df.shape)
print("\n5 dòng đầu:")
display(df.head())

print("\nKiểu dữ liệu:")
print(df.dtypes)

print("\nTỉ lệ thiếu (missing rate):")
print(df.isna().mean())

# Thống kê mô tả cho cột số
print("\nDescribe numeric columns:")
display(df.describe())

# Hàm tính dung lượng bộ nhớ
def memory_in_mb(df_):
    return df_.memory_usage(deep=True).sum() / 1024**2

print(f"\nMemory usage: {memory_in_mb(df):.2f} MB")



df["click_time"] = pd.to_datetime(df["click_time"])
df["hour"] = df["click_time"].dt.hour.astype("int8")
df["day"]  = df["click_time"].dt.day.astype("int8")

# Tạm thời giữ attributed_time để quan sát, lát nữa có thể drop khi train
print(df[["click_time", "hour", "day", "attributed_time"]].head())



target_col = "is_attributed"

print("Đếm số lượng mỗi lớp:")
print(df[target_col].value_counts())

print("\nTỉ lệ mỗi lớp:")
print(df[target_col].value_counts(normalize=True))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Count
df[target_col].value_counts().plot(
    kind="bar", ax=ax[0], title="Số lượng mỗi lớp (sample)"
)
ax[0].set_xticklabels(["not fraud (0)", "fraud (1)"], rotation=0)

# Tỉ lệ %
(df[target_col].value_counts(normalize=True) * 100).plot(
    kind="bar", ax=ax[1], title="Tỉ lệ mỗi lớp (%)"
)
ax[1].set_xticklabels(["not fraud (0)", "fraud (1)"], rotation=0)

plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Phân bố số click theo giờ (không phân lớp)
sns.countplot(data=df, x="hour", ax=axes[0])
axes[0].set_title("Số click theo giờ (sample)")

# Phân bố số click theo giờ, tách theo label
sns.countplot(data=df, x="hour", hue="is_attributed", ax=axes[1])
axes[1].set_title("Số click theo giờ & label (sample)")
axes[1].legend(title="is_attributed")

plt.tight_layout()
plt.show()

# Tương tự cho ngày
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.countplot(data=df, x="day", ax=axes[0])
axes[0].set_title("Số click theo ngày (sample)")

sns.countplot(data=df, x="day", hue="is_attributed", ax=axes[1])
axes[1].set_title("Số click theo ngày & label (sample)")
axes[1].legend(title="is_attributed")

plt.tight_layout()
plt.show()



# Top IP
top_ip = (
    df["ip"]
    .value_counts()
    .head(10)
    .rename_axis("ip")
    .reset_index(name="count")
)
print("Top 10 ip theo số click:")
display(top_ip)

plt.figure(figsize=(10, 4))
sns.barplot(data=top_ip, x="ip", y="count")
plt.title("Top 10 IP theo số click (sample)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Top app
top_app = (
    df["app"]
    .value_counts()
    .head(10)
    .rename_axis("app")
    .reset_index(name="count")
)
print("Top 10 app theo số click:")
display(top_app)

plt.figure(figsize=(8, 4))
sns.barplot(data=top_app, x="app", y="count")
plt.title("Top 10 App theo số click (sample)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top channel
top_channel = (
    df["channel"]
    .value_counts()
    .head(10)
    .rename_axis("channel")
    .reset_index(name="count")
)
print("Top 10 channel theo số click:")
display(top_channel)

plt.figure(figsize=(8, 4))
sns.barplot(data=top_channel, x="channel", y="count")
plt.title("Top 10 Channel theo số click (sample)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import warnings

# Tắt riêng FutureWarning liên quan tới use_inf_as_na (do seaborn gây ra)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*use_inf_as_na.*"
)



# Tỉ lệ fraud theo hour
hour_fraud = (
    df.groupby("hour")[target_col]
    .mean()
    .reset_index(name="fraud_rate")
)

plt.figure(figsize=(8, 4))
sns.lineplot(data=hour_fraud, x="hour", y="fraud_rate", marker="o")
plt.title("Tỉ lệ fraud theo giờ (sample)")
plt.ylabel("fraud_rate")
plt.tight_layout()
plt.show()

# Tỉ lệ fraud theo app (chỉ lấy top 10 app nhiều click nhất)
top10_app_ids = top_app["app"].tolist()
df_app_top = df[df["app"].isin(top10_app_ids)]

app_fraud = (
    df_app_top.groupby("app")[target_col]
    .mean()
    .reset_index(name="fraud_rate")
    .sort_values("fraud_rate", ascending=False)
)

plt.figure(figsize=(8, 4))
sns.barplot(data=app_fraud, x="app", y="fraud_rate")
plt.title("Tỉ lệ fraud theo app (top 10 app)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Bảng tỉ lệ fraud theo app (top 10):")
display(app_fraud)



# Bỏ cả click_time (datetime) và attributed_time
df_model = df.drop(columns=["click_time", "attributed_time"], errors="ignore")

target_col = "is_attributed"
feature_cols = [c for c in df_model.columns if c != target_col]

X = df_model[feature_cols]
y = df_model[target_col]

print("Feature columns:", feature_cols)
print(df_model.dtypes)
print("X shape:", X.shape, "y shape:", y.shape)



import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

target_col = "is_attributed"


feature_cols = [c for c in df_model.columns if c != target_col]

X = df_model[feature_cols]
y = df_model[target_col]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # giữ tỉ lệ fraud tương tự
)

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)
print("Positives train:", y_train.sum(), "Val:", y_val.sum())



from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import time

# Pipeline: chuẩn hóa feature + Logistic Regression
logit_clf = Pipeline([
    ("scaler", StandardScaler()),  # chuẩn hóa các feature dạng số
    ("logreg", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # xử lý mất cân bằng lớp
        n_jobs=-1,
        random_state=42
    ))
])

start = time.time()
logit_clf.fit(X_train, y_train)
train_time_logit = time.time() - start

# Dự đoán xác suất trên tập validation
y_valid_proba_logit = logit_clf.predict_proba(X_val)[:, 1]
auc_logit = roc_auc_score(y_val, y_valid_proba_logit)

print(f"Thời gian train Logistic Regression (sample 100k): {train_time_logit:.2f} giây")
print(f"Valid AUC (Logistic Regression, sample 100k): {auc_logit:.6f}")



from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr_logit, tpr_logit, _ = roc_curve(y_val, y_valid_proba_logit)

plt.figure(figsize=(6, 6))
plt.plot(fpr_logit, tpr_logit, label=f"Logistic Regression (AUC = {auc_logit:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Đường cong ROC - Baseline Logistic Regression (sample 100k)")
plt.legend()
plt.grid(True)
plt.show()



# LightGBM Dataset
train_set = lgb.Dataset(X_train, label=y_train)
val_set   = lgb.Dataset(X_val, label=y_val)

# Tính tạm scale_pos_weight cho cân bằng lớp
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / max(1, n_pos)
print("scale_pos_weight ≈", scale_pos_weight)

params_sample = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 100,
    "num_threads": 4,          
    "scale_pos_weight": scale_pos_weight,  # xử lý imbalance
    "verbose": -1,
}


evals_result_sample = {}

callbacks_sample = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=20),
    lgb.record_evaluation(evals_result_sample),  
]

start = time.time()
model = lgb.train(
    params_sample,
    train_set,
    num_boost_round=1000,
    valid_sets=[train_set, val_set],
    valid_names=["train", "valid"],
    callbacks=callbacks_sample,
)

train_time_sample = time.time() - start

print("Train time (sample):", train_time_sample, "sec")
print("Best iteration:", model.best_iteration)

y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
auc = roc_auc_score(y_val, y_val_pred)
print("Valid AUC (sample):", auc)



from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# 1) Accuracy cho Logistic Regression (baseline)
#    (đã có y_valid_proba_logit và y_val từ cell Logistic ở trên)

y_val_pred_logit = (y_valid_proba_logit >= 0.5).astype("int")
acc_logit = accuracy_score(y_val, y_val_pred_logit)

print(f"Accuracy (Logistic Regression, sample 100k): {acc_logit:.6f}")

# 2) Accuracy cho LightGBM (sample 100k)
#    (đã có y_val_pred và y_val từ cell LightGBM sample ở trên)

y_val_pred_lgb = (y_val_pred >= 0.5).astype("int")
acc_lgb = accuracy_score(y_val, y_val_pred_lgb)

print(f"Accuracy (LightGBM, sample 100k): {acc_lgb:.6f}")

# 3) Bảng so sánh kết quả
results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "LightGBM (sample 100k)"],
    "AUC": [auc_logit, auc],
    "Accuracy": [acc_logit, acc_lgb],
    "Train_time_sec": [train_time_logit, train_time_sample],
})

results_df = results_df.set_index("Model")

print("\nBảng so sánh Logistic Regression vs LightGBM (sample 100k):")
display(results_df)



#đồ thị so sánh Logistic vs LightGBM (sample 100k)

models = results_df.index.tolist()
auc_scores = results_df["AUC"].values
acc_scores = results_df["Accuracy"].values
time_scores = results_df["Train_time_sec"].values

# Biểu đồ 1: So sánh AUC & Accuracy 
x = np.arange(len(models))  # vị trí cột
width = 0.35                # độ rộng mỗi cột

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, auc_scores, width, label="AUC", color="red")
ax.bar(x + width/2, acc_scores, width, label="Accuracy", color="green")

ax.set_ylabel("Giá trị")
ax.set_title("So sánh AUC và Accuracy\n(Logistic Regression vs LightGBM, sample 100k)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylim(0.0, 1.0)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# Biểu đồ 2: So sánh thời gian train 
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(models, time_scores, color="purple")  
ax.set_ylabel("Thời gian train (giây)")
ax.set_title("So sánh thời gian huấn luyện\n(Logistic Regression vs LightGBM, sample 100k)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()



# Cell 10 — Learning curve cho sample

train_auc_sample = evals_result_sample["train"]["auc"]
valid_auc_sample = evals_result_sample["valid"]["auc"]

plt.figure(figsize=(8, 5))
plt.plot(train_auc_sample, label="train AUC")
plt.plot(valid_auc_sample, label="valid AUC")
plt.xlabel("Iteration")
plt.ylabel("AUC")
plt.title("Learning curve — sample (100k dòng)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Feature importance (gain) cho sample

def plot_importance(model, feature_names, title):
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "gain": importance,
    }).sort_values("gain", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=imp_df.head(15),
        x="gain", y="feature", orient="h"
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return imp_df

imp_sample = plot_importance(
    model,
    feature_cols,
    "Top 15 feature — sample (100k dòng)"
)
display(imp_sample.head(15))



# ROC curve cho sample

fpr_s, tpr_s, _ = roc_curve(y_val, y_val_pred)

plt.figure(figsize=(7, 7))
plt.plot(fpr_s, tpr_s, label=f"Sample (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve — sample (100k dòng)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



train_path = DATA_DIR / "train.csv"

dtypes_big = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
    "is_attributed": "uint8",
}

usecols_big = ["ip", "app", "device", "os", "channel", "click_time", "is_attributed"]

N_ROWS = 10_000_000   

start = time.time()
df_big = pd.read_csv(
    train_path,
    nrows=N_ROWS,
    usecols=usecols_big,
    dtype=dtypes_big,
    parse_dates=["click_time"],
)
load_time = time.time() - start

print("df_big shape:", df_big.shape)
print(df_big.head())

print("\nTỉ lệ thiếu:")
print(df_big.isna().mean())

print(f"\nMemory usage df_big: {memory_in_mb(df_big):.2f} MB")
print(f"Thời gian load df_big: {load_time:.2f} sec")



df_big["hour"] = df_big["click_time"].dt.hour.astype("int8")
df_big["day"]  = df_big["click_time"].dt.day.astype("int8")
df_big = df_big.drop(columns=["click_time"])

target_col_big = "is_attributed"
feature_cols_big = [c for c in df_big.columns if c != target_col_big]

Xb = df_big[feature_cols_big]
yb = df_big[target_col_big]

Xb_train, Xb_val, yb_train, yb_val = train_test_split(
    Xb, yb,
    test_size=0.1,
    random_state=42,
    stratify=yb,
)

print("Big Train shape:", Xb_train.shape, "Big Val shape:", Xb_val.shape)
print("Positives big train:", yb_train.sum(), "Val:", yb_val.sum())



train_set_big = lgb.Dataset(Xb_train, label=yb_train)
val_set_big   = lgb.Dataset(Xb_val, label=yb_val)

n_neg_big = (yb_train == 0).sum()
n_pos_big = (yb_train == 1).sum()
scale_pos_weight_big = n_neg_big / max(1, n_pos_big)
print("scale_pos_weight_big ≈", scale_pos_weight_big)

params_big = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.01,      #nhỏ hơn
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 500,    #tăng thêm regularization
    "num_threads": 4,
    "scale_pos_weight": scale_pos_weight_big,
    "verbose": -1,
}

# dùng callback record_evaluation thay cho evals_result=
evals_result_big = {}

callbacks_big = [
    lgb.early_stopping(stopping_rounds=100, verbose=True),
    lgb.log_evaluation(period=100),
    lgb.record_evaluation(evals_result_big),  # ghi log vào dict
]

start = time.time()
model_big = lgb.train(
    params_big,
    train_set_big,
    num_boost_round=2000,
    valid_sets=[train_set_big, val_set_big],
    valid_names=["train", "valid"],
    callbacks=callbacks_big, 
)

train_time_big = time.time() - start

print("Train time (big subset):", train_time_big, "sec")
print("Best iteration (big):", model_big.best_iteration)

y_val_pred_big = model_big.predict(Xb_val, num_iteration=model_big.best_iteration)
auc_big = roc_auc_score(yb_val, y_val_pred_big)
print("Valid AUC (big subset):", auc_big)



# Tính thêm Accuracy cho LightGBM big

from sklearn.metrics import accuracy_score

# y_val_pred_big là xác suất dự đoán của model_big trên Xb_val (Cell 15)
y_val_pred_label_big = (y_val_pred_big >= 0.5).astype("int")
acc_lgb_big = accuracy_score(yb_val, y_val_pred_label_big)


print(f"Valid ACC  (LightGBM, big subset): {acc_lgb_big:.6f}")
print(f"Thời gian train LightGBM (big subset): {train_time_big:.2f} giây")



from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
import time

# Xb_train, Xb_val, yb_train, yb_val đã được tạo ở Cell 14

logit_big_clf = Pipeline([
    ("scaler", StandardScaler()),  # có thể bỏ nếu lo RAM, nhưng để công bằng với sample 100k
    ("logreg", LogisticRegression(
        solver="saga",            # solver phù hợp cho dữ liệu lớn + nhiều mẫu
        max_iter=100,              # có thể tăng nếu thấy chưa hội tụ
        class_weight="balanced",  # xử lý mất cân bằng lớp
        n_jobs=-1,                # tận dụng tất cả CPU cores
        random_state=42,
        verbose=1
    ))
])

start = time.time()
logit_big_clf.fit(Xb_train, yb_train)
train_time_logit_big = time.time() - start

# Xác suất và nhãn dự đoán trên tập validation lớn
y_val_proba_logit_big = logit_big_clf.predict_proba(Xb_val)[:, 1]
auc_logit_big = roc_auc_score(yb_val, y_val_proba_logit_big)

y_val_pred_logit_big = (y_val_proba_logit_big >= 0.5).astype("int")
acc_logit_big = accuracy_score(yb_val, y_val_pred_logit_big)

print(f"Thời gian train Logistic Regression (big subset): {train_time_logit_big:.2f} giây")
print(f"Valid AUC  (Logistic Regression, big subset): {auc_logit_big:.6f}")
print(f"Valid ACC  (Logistic Regression, big subset): {acc_logit_big:.6f}")



from sklearn.metrics import roc_curve


# Learning curve (big subset)
train_auc_big = np.array(evals_result_big["train"]["auc"])
valid_auc_big = np.array(evals_result_big["valid"]["auc"])

iters = np.arange(1, len(train_auc_big) + 1)

fig, ax = plt.subplots(figsize=(8, 5))

# Đường train va valid AUC
ax.plot(
    iters,
    train_auc_big,
    label="Train AUC (big)",
    linewidth=2,
    color="red"
)
ax.plot(
    iters,
    valid_auc_big,
    label="Valid AUC (big)",
    linewidth=2,
    linestyle="--",
    color="green"
)

# Đánh dấu best_iteration
best_iter = model_big.best_iteration
ax.axvline(best_iter, linestyle=":", linewidth=1.5, color="black")
ax.text(
    best_iter,
    valid_auc_big[best_iter - 1] + 0.0003,
    f"best_iter = {best_iter}",
    ha="center",
    va="bottom",
    fontsize=9,
)

# Label & title
ax.set_xlabel("Iteration")
ax.set_ylabel("AUC")
ax.set_title("Learning curve — LightGBM big subset (10M dòng)")

# Giới hạn trục Y cho dễ nhìn (có thể chỉnh lại nếu cần)
ax.set_ylim(0.94, 0.981)

ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()

# =======================
# Feature importance (big subset)
# =======================

imp_big = plot_importance(
    model_big,
    feature_cols_big,
    "Top 15 feature — big subset"
)
display(imp_big.head(15))


# ROC curve (so sánh sample vs big)


fpr_b, tpr_b, _ = roc_curve(yb_val, y_val_pred_big)

plt.figure(figsize=(7, 7))
plt.plot(
    fpr_s,
    tpr_s,
    label=f"Sample (AUC = {auc:.4f})",
    color="red"
)
plt.plot(
    fpr_b,
    tpr_b,
    label=f"Big subset (AUC = {auc_big:.4f})",
    color="green"
)
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve — sample vs big subset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



results_big_df = pd.DataFrame({
    "Model": [
        "Logistic Regression (big)",
        "LightGBM (big subset)"
    ],
    "AUC": [
        auc_logit_big,
        auc_big
    ],
    "Accuracy": [
        acc_logit_big,
        acc_lgb_big
    ],
    "Train_time_sec": [
        train_time_logit_big,
        train_time_big
    ]
})

results_big_df = results_big_df.set_index("Model")

print("So sánh Logistic Regression vs LightGBM trên big subset (df_big):")
display(results_big_df)



models = results_big_df.index.tolist()
auc_scores = results_big_df["AUC"].values
acc_scores = results_big_df["Accuracy"].values
time_scores = results_big_df["Train_time_sec"].values

# --- Biểu đồ 1: So sánh AUC & Accuracy ---
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, auc_scores, width, label="AUC", color="red")
ax.bar(x + width/2, acc_scores, width, label="Accuracy", color="green")

ax.set_ylabel("Giá trị")
ax.set_title("So sánh AUC và Accuracy\n(Logistic Regression vs LightGBM, big subset)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylim(0.0, 1.0)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# --- Biểu đồ 2: So sánh thời gian train ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(models, time_scores,color ="blue")
ax.set_ylabel("Thời gian train (giây)")
ax.set_title("So sánh thời gian huấn luyện\n(Logistic Regression vs LightGBM, big subset)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()



import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Lấy 2 triệu dòng đầu từ df_big (đã có sẵn 10M ở trên)
N_SMALL = 2_000_000
df_small = df_big.iloc[:N_SMALL].copy()

target_col_small = "is_attributed"
feature_cols_small = [c for c in df_small.columns if c != target_col_small]

Xs = df_small[feature_cols_small]
ys = df_small[target_col_small]

Xs_train, Xs_val, ys_train, ys_val = train_test_split(
    Xs, ys, test_size=0.1, random_state=42, stratify=ys
)

train_set_s = lgb.Dataset(Xs_train, label=ys_train)
val_set_s   = lgb.Dataset(Xs_val, label=ys_val)

n_neg_s = (ys_train == 0).sum()
n_pos_s = (ys_train == 1).sum()
scale_pos_weight_s = n_neg_s / max(1, n_pos_s)

base_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 200,
    "scale_pos_weight": scale_pos_weight_s,
    "verbose": -1,
}

def run_with_threads(num_threads):
    params = base_params.copy()
    params["num_threads"] = num_threads

    evals_result = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.record_evaluation(evals_result),
    ]

    start = time.time()
    model_tmp = lgb.train(
        params,
        train_set_s,
        num_boost_round=500,
        valid_sets=[val_set_s],
        valid_names=["valid"],
        callbacks=callbacks,
    )
    t = time.time() - start
    y_pred = model_tmp.predict(Xs_val, num_iteration=model_tmp.best_iteration)
    auc_tmp = roc_auc_score(ys_val, y_pred)
    return t, auc_tmp, model_tmp.best_iteration

time_1, auc_1, it_1 = run_with_threads(1)
time_4, auc_4, it_4 = run_with_threads(4)

print("num_threads=1 -> time = {:.2f}s, AUC = {:.4f}, best_iter = {}".format(time_1, auc_1, it_1))
print("num_threads=4 -> time = {:.2f}s, AUC = {:.4f}, best_iter = {}".format(time_4, auc_4, it_4))



test_path = DATA_DIR / "test.csv"

dtypes_test = {
    "click_id": "uint32",
    "ip":       "uint32",
    "app":      "uint16",
    "device":   "uint16",
    "os":       "uint16",
    "channel":  "uint16",
}

usecols_test = ["click_id", "ip", "app", "device", "os", "channel", "click_time"]

print("Đang load test.csv ...")
test_df = pd.read_csv(
    test_path,
    usecols=usecols_test,
    dtype=dtypes_test,
    parse_dates=["click_time"],
)

print("test_df shape:", test_df.shape)
display(test_df.head())

# Feature engineering giống y train
test_df["hour"] = test_df["click_time"].dt.hour.astype("int8")
test_df["day"]  = test_df["click_time"].dt.day.astype("int8")

# Bỏ cột thời gian thô
test_df = test_df.drop(columns=["click_time"])

print("\nSau khi thêm feature:")
display(test_df.head())
print(test_df.dtypes)



print("Feature dùng để train (big):", feature_cols_big)

# Đảm bảo test_df có đủ các cột 
missing_cols = set(feature_cols_big) - set(test_df.columns)
print("Missing cols in test:", missing_cols)

if len(missing_cols) > 0:
    raise ValueError(f"Thiếu cột trong test_df: {missing_cols}")

X_test = test_df[feature_cols_big]

print("X_test shape:", X_test.shape)
display(X_test.head())

#Dự đoán xác suất is_attributed=1 bằng LightGBM 
test_pred_lgb = model_big.predict(
    X_test,
    num_iteration=model_big.best_iteration
)

print("\nVí dụ 10 giá trị dự đoán đầu (LightGBM):")
print(test_pred_lgb[:10])

#Dự đoán xác suất is_attributed=1 bằng Logistic Regression 
test_pred_logit = logit_big_clf.predict_proba(X_test)[:, 1]

print("\nVí dụ 10 giá trị dự đoán đầu (Logistic Regression):")
print(test_pred_logit[:10])

# So sánh phân bố xác suất của 2 mô hình 
plt.figure(figsize=(8, 4))
plt.hist(test_pred_lgb, bins=50, alpha=0.5, label="LightGBM")
plt.hist(test_pred_logit, bins=50, alpha=0.5, label="Logistic")
plt.title("Phân bố xác suất dự đoán trên test.csv\n(so sánh LightGBM vs Logistic)")
plt.xlabel("P(is_attributed = 1)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()



sub_path = DATA_DIR / "sample_submission.csv"
sub_base = pd.read_csv(sub_path)

print("sample_submission head:")
display(sub_base.head())

#LightGBM 
sub_lgb = sub_base.copy()
sub_lgb["is_attributed"] = test_pred_lgb

out_path_lgb = Path("submission_lightgbm_10m.csv")
sub_lgb.to_csv(out_path_lgb, index=False)
print("Đã lưu file submission LightGBM:", out_path_lgb)

#Logistic Regression
sub_logit = sub_base.copy()
sub_logit["is_attributed"] = test_pred_logit

out_path_logit = Path("submission_logit_10m.csv")
sub_logit.to_csv(out_path_logit, index=False)
print("Đã lưu file submission Logistic:", out_path_logit)



test_vis_df = test_df.copy()
test_vis_df["pred_lgb"]   = test_pred_lgb
test_vis_df["pred_logit"] = test_pred_logit

display(test_vis_df.head())






#mẫu 100k 
N_SAMPLE = 100_000
if len(test_vis_df) > N_SAMPLE:
    vis_sample = test_vis_df.sample(N_SAMPLE, random_state=42)
else:
    vis_sample = test_vis_df

plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=vis_sample,
    x="pred_logit",
    y="pred_lgb",
    alpha=0.3,
    s=10
)
plt.xlabel("P(is_attributed = 1) - Logistic")
plt.ylabel("P(is_attributed = 1) - LightGBM")
plt.title("So sánh xác suất dự đoán của 2 mô hình\n(trên mẫu từ test.csv)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# In hệ số tương quan
corr = vis_sample[["pred_logit", "pred_lgb"]].corr().iloc[0,1]
print(f"Hệ số tương quan Pearson giữa 2 model: {corr:.4f}")


