import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Cố định seed cho reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Đường dẫn dataset H&M trên Kaggle
DATA_DIR = Path("/kaggle/input/h-and-m-personalized-fashion-recommendations")
print("DATA_DIR:", DATA_DIR, "| Exists:", DATA_DIR.exists())


from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Thư mục ảnh H&M 
IMG_DIR = DATA_DIR / "images"

def get_image_path(article_id):
    """
    Trả về đường dẫn file ảnh tương ứng article_id (theo cấu trúc H&M).
    """
    aid_str = str(article_id).zfill(10)
    subdir = aid_str[:3]
    return IMG_DIR / subdir / f"{aid_str}.jpg"


def _plot_article_row(article_ids, axes_row, fontsize=8):
    """
    Vẽ một hàng ảnh trên dãy axes_row (mảng Axes).
    Nếu số cột > số article thì các ô dư sẽ tắt trục.
    """
    n_cols = len(axes_row)

    for col_idx, ax in enumerate(axes_row):
        if col_idx < len(article_ids):
            aid = article_ids[col_idx]
            try:
                img = Image.open(get_image_path(aid))
                ax.imshow(img)
            except Exception:
                # Nếu không load được ảnh thì in id ra giữa ô
                ax.text(0.5, 0.5, str(aid), ha="center", va="center")
            ax.set_title(str(aid), fontsize=fontsize)
        else:
            # Ô trống
            ax.set_title("")
        ax.axis("off")




def show_cf_svd_visual(customer_id, topk=8):
    """
    Trực quan hoá kết quả cho 1 user:
    - Hàng trên: các sản phẩm thực tế user mua trong tuần test
    - Hàng dưới: các sản phẩm được gợi ý bởi CF–SVD
    """
    true_items = true_items_dict.get(customer_id, [])[:topk]
    if len(true_items) == 0:
        print(f"[CF–SVD] Người dùng {customer_id} không có giao dịch trong tuần test.")
        return

    pred_items = recommend_cf_svd(customer_id, topk=topk)

    print(f"Người dùng: {customer_id}")
    print("  Sản phẩm thực tế (tuần test):", true_items)
    print("  Sản phẩm gợi ý bởi CF–SVD :", pred_items)

    n_cols = max(len(true_items), len(pred_items))
    if n_cols == 0:
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 6))

    # Khi n_cols == 1 thì axes là 1D, ép về (2, n_cols)
    axes = np.atleast_2d(axes)

    # Hàng 1: thực tế
    _plot_article_row(true_items, axes[0, :], fontsize=9)
    # Hàng 2: gợi ý
    _plot_article_row(pred_items, axes[1, :], fontsize=9)

    # Nhãn hàng (tiếng Việt)
    fig.text(0.5, 0.94, "LỌC CỘNG TÁC (CF) DÙNG SVD VÀ MA TRẬN TƯƠNG TÁC USER–ITEM",
             ha="center", va="center", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.70, "Thực tế", rotation=90,
             ha="center", va="center", fontsize=11, fontweight="bold")
    fig.text(0.02, 0.25, "Gợi ý", rotation=90,
             ha="center", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0.05, 0.02, 1.0, 0.92])
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from tqdm import tqdm

# =========================
# Visualize Content-based
# =========================
def show_cb_visual(customer_id, topk=8):
    """
    Trực quan hoá kết quả Content-based cho 1 user:
    - Hàng trên: sản phẩm thực tế trong tuần test
    - Hàng dưới: sản phẩm gợi ý từ ma trận đặc trưng user–item
    """
    true_items = true_items_dict.get(customer_id, [])[:topk]
    if len(true_items) == 0:
        print(f"[Content-based] Người dùng {customer_id} không có giao dịch trong tuần test.")
        return

    pred_items = recommend_content_based(customer_id, topk=topk)

    print(f"Người dùng: {customer_id}")
    print("  Sản phẩm thực tế (tuần test):", true_items)
    print("  Sản phẩm gợi ý Content-based:", pred_items)

    n_cols = max(len(true_items), len(pred_items))
    if n_cols == 0:
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 6))

    # axes có thể là (2,) nếu n_cols=1 => reshape về (2, 1)
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, n_cols)

    # Hàng 1: thực tế
    _plot_article_row(true_items, axes[0, :], fontsize=9)
    # Hàng 2: gợi ý
    _plot_article_row(pred_items, axes[1, :], fontsize=9)

    fig.text(
        0.5, 0.94,
        "GỢI Ý DỰA TRÊN NỘI DUNG (CONTENT-BASED) – MA TRẬN ĐẶC TRƯNG USER–ITEM",
        ha="center", va="center", fontsize=14, fontweight="bold"
    )
    fig.text(
        0.02, 0.70, "Thực tế",
        rotation=90, ha="center", va="center", fontsize=11, fontweight="bold"
    )
    fig.text(
        0.02, 0.25, "Gợi ý",
        rotation=90, ha="center", va="center", fontsize=11, fontweight="bold"
    )

    plt.tight_layout(rect=[0.05, 0.02, 1.0, 0.92])
    plt.show()


# =========================
# Submission helpers
# =========================
def pad_article_id(aid) -> str:
    """Kaggle submission thường dùng 10-digit article_id."""
    try:
        return str(int(aid)).zfill(10)
    except Exception:
        s = str(aid)
        return s.zfill(10) if s.isdigit() else s


def uniq_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def ensure_12(recs, fallback_12):
    recs = [pad_article_id(x) for x in recs]
    recs = uniq_keep_order(recs)
    if len(recs) < 12:
        recs = recs + [pad_article_id(x) for x in fallback_12]
        recs = uniq_keep_order(recs)
    return recs[:12]


def write_submission_stream(customer_ids, rec_fn, fallback_12, out_csv, max_customers=None):
    """
    Ghi submission dạng stream (không giữ DataFrame lớn trong RAM).
    rec_fn(cid) -> list article_id (có thể < 12)
    fallback_12: list 12 article_id dự phòng (popular)
    """
    out_csv = str(out_csv)
    n = len(customer_ids) if max_customers is None else min(len(customer_ids), max_customers)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "prediction"])
        for cid in tqdm(customer_ids[:n], total=n, desc=f"Write {Path(out_csv).name}"):
            recs = rec_fn(cid)
            recs12 = ensure_12(recs, fallback_12)
            w.writerow([cid, " ".join(recs12)])

    print("Saved:", out_csv, "| rows:", n)



def pad_article_id(aid) -> str:
    # Kaggle submission thường dùng 10-digit article_id
    try:
        return str(int(aid)).zfill(10)
    except Exception:
        s = str(aid)
        return s.zfill(10) if s.isdigit() else s

def uniq_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def ensure_12(recs, fallback_12):
    recs = [pad_article_id(x) for x in recs]
    recs = uniq_keep_order(recs)
    if len(recs) < 12:
        recs = recs + [pad_article_id(x) for x in fallback_12]
        recs = uniq_keep_order(recs)
    return recs[:12]

def write_submission_stream(customer_ids, rec_fn, fallback_12, out_csv, max_customers=None):
    out_csv = str(out_csv)
    n = len(customer_ids) if max_customers is None else min(len(customer_ids), max_customers)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "prediction"])
        for i, cid in enumerate(tqdm(customer_ids[:n], total=n, desc=f"Write {Path(out_csv).name}")):
            recs = rec_fn(cid)
            recs12 = ensure_12(recs, fallback_12)
            w.writerow([cid, " ".join(recs12)])

    print("Saved:", out_csv, "| rows:", n)


# Ta dùng 3 file chính:
# - `articles.csv`: metadata sản phẩm
# - `customers.csv`: thông tin khách hàng
# - `transactions_train.csv`: lịch sử giao dịch
#
# Chiến lược thời gian:
# - Lấy **6 tuần gần cuối** để train
# - Tuần cuối cùng làm **validation / test nội bộ**


# Đọc dữ liệu
articles = pd.read_csv(DATA_DIR / "articles.csv")
customers = pd.read_csv(DATA_DIR / "customers.csv")
transactions = pd.read_csv(
    DATA_DIR / "transactions_train.csv",
    parse_dates=["t_dat"]
)

print("articles:", articles.shape)
print("customers:", customers.shape)
print("transactions:", transactions.shape)


# Xem nhanh vài dòng
display(articles.head())
display(customers.head())
display(transactions.head())


# Xác định mốc chia thời gian
max_date = transactions["t_dat"].max()
test_start = max_date - pd.Timedelta(weeks=1)
train_start = test_start - pd.Timedelta(weeks=6)

print("Max date      :", max_date)
print("Train from    :", train_start)
print("Validation from:", test_start)

train_df = transactions[(transactions["t_dat"] >= train_start) & (transactions["t_dat"] < test_start)]
test_df  = transactions[transactions["t_dat"] >= test_start]

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)


# Map: customer_id -> set(article_id) đã mua trong tuần test
true_items_dict = (
    test_df.groupby("customer_id")["article_id"]
    .apply(list)  # dùng list để giữ thứ tự
    .to_dict()
)



# map customer -> purchased list 
test_true = test_df.groupby("customer_id")["article_id"].apply(list).to_dict()
print("test users:", len(test_true))

all_customer_ids = customers["customer_id"].astype(str).values


# - Phân bố nhóm tuổi khách hàng
# - Top product type
# - Số giao dịch theo ngày trong 6 tuần train


# Nhóm tuổi khách hàng
cust_basic = customers[["customer_id", "age"]].copy()
cust_basic["age_bucket"] = pd.cut(
    cust_basic["age"],
    bins=[0, 18, 25, 35, 50, 100],
    labels=["<18", "18-25", "25-35", "35-50", "50+"]
)

plt.figure(figsize=(6,4))
cust_basic["age_bucket"].value_counts().sort_index().plot(kind="bar")
plt.title("Phân bố nhóm tuổi khách hàng")
plt.xlabel("Nhóm tuổi")
plt.ylabel("Số khách hàng")
plt.show()


# Top product_type
plt.figure(figsize=(10,4))
articles["product_type_name"].value_counts().head(15).plot(kind="bar")
plt.title("Top 15 Product Type phổ biến")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()


# Transactions theo ngày (6 tuần train)
tmp = train_df.copy()
tmp["date"] = tmp["t_dat"].dt.date
tx_per_day = tmp.groupby("date")["t_dat"].count()

plt.figure(figsize=(10,4))
tx_per_day.plot()

plt.title("Số giao dịch mỗi ngày (6 tuần train)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Ý tưởng:
# - Đếm số lần xuất hiện của mỗi `article_id` trong `train_df`
# - Sắp xếp giảm dần → danh sách `popular_items`
# - Với mỗi customer: recommend top-12 sản phẩm phổ biến nhất.


def get_global_popular_items(train_df, topk=1000):
    item_pop = (
        train_df.groupby("article_id")["t_dat"]
        .count()
        .sort_values(ascending=False)
    )
    popular_items = item_pop.head(topk).index.tolist()
    return popular_items, item_pop

popular_items, item_pop_series = get_global_popular_items(train_df, topk=1000)
popular_12 = popular_items[:12]

print("Số item phổ biến lưu lại:", len(popular_items))
print("Top 12 article_id phổ biến:", popular_12)


def recommend_popular_for_customer(customer_id, k=12, popular_items=popular_items):
    """
    Trả về top-k sản phẩm phổ biến nhất (không phụ thuộc customer).
    """
    return popular_items[:k]

# Thử demo cho 5 khách hàng
sample_customers = customers["customer_id"].sample(5, random_state=RANDOM_STATE).tolist()
for cid in sample_customers:
    print("Customer:", cid)
    print("  Recs:", recommend_popular_for_customer(cid))
    print()




def rec_popularity(cid):
    return popular_12

# Xuất submission (full hoặc giới hạn để test nhanh)
MAX_CUSTOMERS_SUB = None   
write_submission_stream(
    all_customer_ids,
    rec_popularity,
    fallback_12=popular_12,
    out_csv=Path("/kaggle/working/submission_popularity.csv"),
    max_customers=MAX_CUSTOMERS_SUB
)


# Mục tiêu:
# - Chuẩn hoá hành vi mua sắm theo mức độ hoạt động.
# - Đánh giá / huấn luyện mô hình riêng cho từng nhóm user segment.
#
# Quy tắc phân tầng:
# - New users:     0 – 20 giao dịch
# - Medium users:  21 – 100 giao dịch
# - Frequent users:101 – 200 giao dịch
# - Heavy users:   201 – 500 giao dịch
# - Super users:   > 500 giao dịch


user_tx_counts = (
    transactions
    .groupby("customer_id")["article_id"]
    .count()
    .reset_index(name="n_transactions")
)

user_tx_counts.head()


# Hàm phân tầng theo rule đã nêu
def assign_segment(n_tx: int) -> str:
    if n_tx <= 20:
        return "new_users"
    if n_tx <= 100:
        return "medium_users"
    if n_tx <= 200:
        return "frequent_users"
    if n_tx <= 500:
        return "heavy_users"
    return "super_users"


user_tx_counts["user_segment"] = user_tx_counts["n_transactions"].apply(
    assign_segment
)



# Thứ tự dùng cho vẽ biểu đồ
segment_order = [
    "new_users",
    "medium_users",
    "frequent_users",
    "heavy_users",
    "super_users",
]


transactions_seg = transactions.merge(
    user_tx_counts[["customer_id", "user_segment"]],
    on="customer_id",
    how="left",
)

transactions_seg.head()



# Biểu đồ phân bố segment trên TOÀN BỘ dữ liệu
seg_counts = (
    user_tx_counts["user_segment"]
    .value_counts()
    .reindex(segment_order)
)

plt.figure(figsize=(8,5))
seg_counts.plot(kind="bar", color="skyblue")
plt.title("Phân bố phân tầng người dùng (toàn bộ dữ liệu)")
plt.xlabel("Phân tầng người dùng")
plt.ylabel("Số lượng khách hàng")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()



# ================= Thống kê trên TOÀN BỘ dữ liệu =================
total_tx_all = len(transactions)
unique_customers_all = transactions["customer_id"].nunique()
unique_articles_all = transactions["article_id"].nunique()

print("Thống kê trên TOÀN BỘ dữ liệu:\n")
print(f"Tổng số giao dịch: {total_tx_all:,}")
print(f"Số khách hàng duy nhất: {unique_customers_all:,}")
print(f"Số sản phẩm duy nhất: {unique_articles_all:,}\n")

# Tỷ lệ % khách hàng theo từng phân tầng trên toàn bộ dữ liệu
seg_ratio_all = seg_counts / seg_counts.sum()
print("Tỷ lệ % khách hàng theo phân tầng (toàn bộ dữ liệu):")
print((seg_ratio_all * 100).round(2).astype(str) + " %")
print("\n" + "=" * 60 + "\n")


N_PER_SEGMENT = 1000  # số user tối đa chọn trong mỗi segment

sampled_user_ids = []

for seg in segment_order:
    seg_users = user_tx_counts.loc[
        user_tx_counts["user_segment"] == seg, "customer_id"
    ]
    if len(seg_users) == 0:
        continue

    n_pick = min(N_PER_SEGMENT, len(seg_users))
    sampled = seg_users.sample(n_pick, random_state=RANDOM_STATE)
    sampled_user_ids.append(sampled)

sampled_user_ids = pd.concat(sampled_user_ids)

len(sampled_user_ids), sampled_user_ids.head()

transactions_seg = train_df.merge(
    user_tx_counts[["customer_id", "user_segment"]],
    on="customer_id",
    how="left",
)

transactions_seg.head()


# Lấy toàn bộ giao dịch của các user đã sample
sampled_tx = transactions_seg[
    transactions_seg["customer_id"].isin(sampled_user_ids)
]

sampled_tx.shape, sampled_tx["customer_id"].nunique()


# Biểu đồ phân bố segment trong SAMPLE
sample_seg_counts = (
    user_tx_counts[user_tx_counts["customer_id"].isin(sampled_user_ids)]
    ["user_segment"]
    .value_counts()
    .reindex(segment_order)
)

plt.figure(figsize=(8, 5))
sample_seg_counts.plot(kind="bar", color="skyblue")
plt.title("Phân bố phân tầng người dùng (tập dữ liệu mẫu)")
plt.xlabel("Phân tầng người dùng")
plt.ylabel("Số lượng khách hàng")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()



# ================= THỐNG KÊ TRÊN TẬP MẪU  =================
total_tx_sample = len(sampled_tx)
unique_customers_sample = sampled_tx["customer_id"].nunique()
unique_articles_sample = sampled_tx["article_id"].nunique()

print("Thống kê trên TẬP DỮ LIỆU MẪU:\n")
print(f"Tổng số giao dịch: {total_tx_sample:,}")
print(f"Số khách hàng duy nhất: {unique_customers_sample:,}")
print(f"Số sản phẩm duy nhất: {unique_articles_sample:,}\n")

seg_ratio_sample = sample_seg_counts / sample_seg_counts.sum()
print("Tỷ lệ % khách hàng theo phân tầng (tập dữ liệu mẫu):")
print((seg_ratio_sample * 100).round(2).astype(str) + " %")
print("\n" + "=" * 60 + "\n")


sampled_tx_articles = sampled_tx.merge(
    articles[["article_id", "prod_name", "product_group_name"]],
    on="article_id",
    how="left",
)

sampled_tx_articles.head()


# 1) Top 5 sản phẩm bán chạy
top_products = (
    sampled_tx_articles
    .groupby(["article_id", "prod_name"])["t_dat"]
    .count()
    .sort_values(ascending=False)
    .head(5)
    .reset_index(name="n_transactions")
)

plt.figure(figsize=(8, 5))
plt.bar(top_products["prod_name"], top_products["n_transactions"])
plt.xticks(rotation=45, ha="right")
plt.xlabel("Tên sản phẩm")
plt.ylabel("Số lượng giao dịch")
plt.title("Top 5 sản phẩm bán chạy (tập dữ liệu mẫu)")
plt.tight_layout()
plt.show()

top_products


# 2) Top 10 nhóm sản phẩm
top_groups = (
    sampled_tx_articles
    .groupby("product_group_name")["t_dat"]
    .count()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="n_transactions")
)

plt.figure(figsize=(8, 5))
plt.bar(top_groups["product_group_name"], top_groups["n_transactions"])
plt.xticks(rotation=45, ha="right")
plt.xlabel("Nhóm sản phẩm")
plt.ylabel("Số lượng giao dịch")
plt.title("Top 10 nhóm sản phẩm (tập dữ liệu mẫu)")
plt.tight_layout()
plt.show()

top_groups


# 3) Phân bố kênh bán (1 = Online, 2 = In-Store)
channel_map = {1: "Online", 2: "In-Store"}
sampled_tx = sampled_tx.copy()
sampled_tx["sales_channel"] = sampled_tx["sales_channel_id"].map(
    channel_map
)

plt.figure(figsize=(6, 4))
sampled_tx["sales_channel"].value_counts().plot(kind="bar")
plt.title("Phân bố kênh bán hàng (tập dữ liệu mẫu)")
plt.xlabel("Kênh bán hàng")
plt.ylabel("Số lượng giao dịch")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()




# 4) Phân bố tuổi khách hàng trong SAMPLE
sampled_tx_cust = sampled_tx.merge(
    customers[["customer_id", "age"]],
    on="customer_id",
    how="left",
)

plt.figure(figsize=(8, 4))
sns.histplot(sampled_tx_cust["age"].dropna(), bins=50, kde=True)
plt.title("Phân bố tuổi khách hàng (tập dữ liệu mẫu)")
plt.xlabel("Tuổi")
plt.ylabel("Tần suất")
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


# ===== Tạo ma trận tương tác USER–ITEM (0/1 đã mua) =====

# Chỉ giữ cột cần thiết
ui_df = sampled_tx[["customer_id", "article_id"]].drop_duplicates()

# Gán index số cho user và item
user_ids = ui_df["customer_id"].unique()
item_ids = ui_df["article_id"].unique()

user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
item_id_to_idx = {a: j for j, a in enumerate(item_ids)}

n_users = len(user_ids)
n_items = len(item_ids)
print("n_users:", n_users, "| n_items:", n_items)

# Ma trận tương tác nhị phân R (n_users x n_items)
R = np.zeros((n_users, n_items), dtype=np.float32)
for _, row in ui_df.iterrows():
    u_idx = user_id_to_idx[row["customer_id"]]
    i_idx = item_id_to_idx[row["article_id"]]
    R[u_idx, i_idx] = 1.0

print("Interaction matrix shape:", R.shape)


# ===== SVD: R ≈ U Σ V^T (TruncatedSVD) =====

k = 50  # số nhân tố ẩn (latent factors), có thể tune theo slide SVD
svd = TruncatedSVD(n_components=k, random_state=42)
U = svd.fit_transform(R)           # (n_users, k)
Sigma = svd.singular_values_       # (k,)
Vt = svd.components_               # (k, n_items)

# R_approx = U Σ V^T
Sigma_matrix = np.diag(Sigma)
R_hat = np.dot(np.dot(U, Sigma_matrix), Vt)  # (n_users, n_items)


# ===== Hàm gợi ý bằng CF–SVD =====

idx_to_user_id = {i: u for u, i in user_id_to_idx.items()}
idx_to_item_id = {j: a for a, j in item_id_to_idx.items()}

def recommend_cf_svd(customer_id, topk=10):
    """
    Gợi ý top-k sản phẩm cho 1 khách hàng,
    dựa trên điểm dự đoán trong R_hat.
    """
    if customer_id not in user_id_to_idx:
        # user mới hoàn toàn trong sample -> fallback: chưa xử lý
        return []

    u_idx = user_id_to_idx[customer_id]
    scores = R_hat[u_idx]  # vector điểm cho tất cả item

    # Không gợi ý lại item đã mua
    bought_items = set(
        ui_df.loc[ui_df["customer_id"] == customer_id, "article_id"].tolist()
    )
    candidate_indices = [
        j for j, aid in idx_to_item_id.items() if aid not in bought_items
    ]

    candidate_scores = [(j, scores[j]) for j in candidate_indices]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    top_items = [idx_to_item_id[j] for j, _ in candidate_scores[:topk]]
    return top_items



# Demo
sample_users = ui_df["customer_id"].drop_duplicates().sample(3, random_state=42)
for cid in sample_users:
    print("User:", cid)
    print("  CF–SVD recs:", recommend_cf_svd(cid, topk=5))
    print()


# Lấy danh sách user vừa xuất hiện trong sample (ui_df) vừa có giao dịch test
candidate_users = [
    cid for cid in ui_df["customer_id"].unique()
    if cid in true_items_dict
]

demo_users = pd.Series(candidate_users).sample(3, random_state=42).tolist()

for cid in demo_users:
    print("=" * 80)
    show_cf_svd_visual(cid, topk=8)   # hình CF


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


from gensim.models import Word2Vec
from scipy.sparse import hstack, csr_matrix

# ===== Vector hoá mô tả sản phẩm bằng Word2Vec =====

# Lấy cột mô tả chi tiết
articles_text = articles[["article_id", "detail_desc"]].copy()
articles_text["detail_desc"] = (
    articles_text["detail_desc"]
    .fillna("")
    .astype(str)
    .str.lower()
)

# Tạo corpus câu (danh sách token) cho Word2Vec
sentences = [
    desc.split()
    for desc in articles_text["detail_desc"].values
    if desc.strip() != ""
]

print("Số câu dùng train Word2Vec:", len(sentences))

# Train Word2Vec
w2v_dim = 100  # số chiều vector từ, có thể chỉnh 50/100/200
w2v_model = Word2Vec(
    sentences,
    vector_size=w2v_dim,
    window=5,
    min_count=2,
    workers=4,
    sg=1,                # skip-gram
    seed=RANDOM_STATE,
)

def get_w2v_vector(text: str, model: Word2Vec, dim: int) -> np.ndarray:
    """
    Trả về vector Word2Vec (trung bình các từ) cho 1 mô tả sản phẩm.
    Nếu không có từ nào trong vocab -> vector 0.
    """
    tokens = text.lower().split()
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)




# ===== Tạo ITEM PROFILE MATRIX (One-hot + price + Word2Vec) =====

# 1) Tính giá trung bình mỗi article từ transactions
article_price = (
    transactions.groupby("article_id")["price"]
    .mean()
    .reset_index(name="price")
)

# 2) Lấy meta từ articles, thêm detail_desc và merge giá vào
item_meta = articles[[
    "article_id",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "detail_desc",
]].copy()

item_meta = item_meta.merge(article_price, on="article_id", how="left")

# Chỉ giữ các item xuất hiện trong SAMPLE
item_meta = item_meta[item_meta["article_id"].isin(item_ids)].reset_index(drop=True)

# 3) Xử lý categorical + numerical
cat_cols = [
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
]
num_cols = ["price"]

for c in cat_cols:
    item_meta[c] = item_meta[c].fillna("UNK").astype(str)

item_meta["price"] = item_meta["price"].fillna(item_meta["price"].median())

# 4) One-hot cho categorical
ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
item_cat_ohe = ohe.fit_transform(item_meta[cat_cols])

# 5) Chuẩn hoá numeric
scaler = StandardScaler()
item_num_scaled = scaler.fit_transform(item_meta[num_cols])

# 6) Vector Word2Vec cho detail_desc
item_meta["detail_desc"] = (
    item_meta["detail_desc"]
    .fillna("")
    .astype(str)
    .str.lower()
)

w2v_vectors = np.vstack([
    get_w2v_vector(desc, w2v_model, w2v_dim)
    for desc in item_meta["detail_desc"].values
])

# Chuyển sang sparse để ghép với one-hot
w2v_sparse = csr_matrix(w2v_vectors)

# 7) Ghép thành ITEM PROFILE MATRIX
item_feature_matrix = hstack([item_cat_ohe, item_num_scaled, w2v_sparse]).tocsr()
print("Item feature matrix shape:", item_feature_matrix.shape)

# Map article_id <-> index trong item_feature_matrix
itemid_to_feat_idx = {aid: i for i, aid in enumerate(item_meta["article_id"].values)}
feat_idx_to_itemid = {i: aid for aid, i in itemid_to_feat_idx.items()}



from scipy.sparse import coo_matrix, diags

# ----- 1. Tạo ma trận user–item nhị phân sparse -----

# ui_df chỉ gồm (customer_id, article_id) trong SAMPLE
rows = np.array([user_id_to_idx[c] for c in ui_df["customer_id"].values], dtype=np.int32)
cols = np.array([item_id_to_idx[a] for a in ui_df["article_id"].values], dtype=np.int32)
data = np.ones(len(ui_df), dtype=np.float32)

n_users = len(user_ids)
n_items = len(item_ids)

UI = coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()
print("UI shape:", UI.shape)

# ----- 2. Đếm số item mỗi user đã mua -----

user_item_counts = np.asarray(UI.sum(axis=1)).ravel()   # (n_users,)
# Tránh chia cho 0 (về lý thì user trong UI đều có giao dịch, nhưng cứ phòng)
user_item_counts[user_item_counts == 0] = 1.0

# Ma trận đường chéo D^{-1}
D_inv = diags(1.0 / user_item_counts)

# ----- 3. Tính USER PROFILE MATRIX = D^{-1} * UI * ItemFeatures -----

# item_feature_matrix phải là CSR
# item_feature_matrix = hstack([...]).tocsr()  # bạn đã làm ở trên
user_profile_matrix = D_inv.dot(UI).dot(item_feature_matrix)

print("User profile matrix shape:", user_profile_matrix.shape)

# Map id <-> index cho user profile
user_profile_ids = user_ids
cid_to_profile_idx = {cid: i for i, cid in enumerate(user_profile_ids)}
profile_idx_to_cid = {i: cid for i, cid in enumerate(user_profile_ids)}



# ===== Hàm gợi ý Content-based =====

def recommend_content_based(customer_id, topk=10):
    """
    Gợi ý top-k sản phẩm dựa trên:
    - User profile: trung bình feature của item đã mua
    - Tính cosine similarity với toàn bộ item_feature_matrix
    """
    if customer_id not in cid_to_profile_idx:
        return []  # user không có profile trong sample

    u_idx = cid_to_profile_idx[customer_id]
    u_vec = user_profile_matrix[u_idx]          # (1, d)

    # Tính cosine similarity giữa user và tất cả item
    sims = cosine_similarity(u_vec, item_feature_matrix).ravel()  # (n_items,)

    # Không gợi ý lại item đã mua
    bought_items = set(user_to_items.get(customer_id, []))
    candidate_indices = [
        i for i, aid in feat_idx_to_itemid.items()
        if aid not in bought_items
    ]

    candidate_scores = [(i, sims[i]) for i in candidate_indices]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    top_items = [feat_idx_to_itemid[i] for i, _ in candidate_scores[:topk]]
    return top_items


# Map: mỗi user -> danh sách (hoặc set) các article_id đã mua trong SAMPLE
user_to_items = (
    ui_df.groupby("customer_id")["article_id"]
    .apply(list)            # hoặc .apply(set) nếu muốn set luôn
    .to_dict()
)
print("Số user trong user_to_items:", len(user_to_items))



# Demo
for cid in sample_users:
    print("User:", cid)
    print("  CB recs:", recommend_content_based(cid, topk=5))
    print()



# Lấy danh sách user vừa xuất hiện trong sample (ui_df) vừa có giao dịch test
candidate_users = [
    cid for cid in ui_df["customer_id"].unique()
    if cid in true_items_dict
]

demo_users = pd.Series(candidate_users).sample(3, random_state=42).tolist()

for cid in demo_users:
    print("=" * 80)
    show_cb_visual(cid, topk=8)       # hình Content-based


# Chuẩn bị user features
user_cols = ["customer_id", "age", "club_member_status", "fashion_news_frequency"]
item_cols = ["article_id", "product_type_name", "product_group_name",
             "graphical_appearance_name", "colour_group_name", "price"]

user_df = customers[user_cols].copy()
user_df["age"] = user_df["age"].fillna(30)
user_df["age_bucket"] = pd.cut(
    user_df["age"],
    bins=[0, 18, 25, 35, 50, 100],
    labels=["<18", "18-25", "25-35", "35-50", "50+"]
)

item_df = articles[item_cols].copy()
item_df["price"] = item_df["price"].fillna(item_df["price"].median())

display(user_df.head())
display(item_df.head())


# Label encode các cột categorical
def label_encode(df, col):
    values = df[col].fillna("UNK").astype(str).values
    uniques = pd.unique(values)
    mapping = {v: i for i, v in enumerate(uniques)}
    df[col + "_le"] = [mapping[v] for v in values]
    return mapping

user_mappings = {}
item_mappings = {}

for col in ["age_bucket", "club_member_status", "fashion_news_frequency"]:
    user_mappings[col] = label_encode(user_df, col)

for col in ["product_type_name", "product_group_name",
            "graphical_appearance_name", "colour_group_name"]:
    item_mappings[col] = label_encode(item_df, col)

display(user_df.head())
display(item_df.head())


# Đặt index để join nhanh
user_df_indexed = user_df.set_index("customer_id")
item_df_indexed = item_df.set_index("article_id")

print("Num users in user_df:", len(user_df_indexed))
print("Num items in item_df:", len(item_df_indexed))


# - Positive: tất cả (customer, article) có thật trong train_df
# - Negative: với mỗi (customer, article_pos), sample một vài article_neg khác
# - Kết quả: `train_full` chứa (customer_id, article_id, label)


# Positive
train_pairs = train_df[["customer_id", "article_id"]].drop_duplicates()
train_pairs["label"] = 1

# Negative sampling đơn giản
all_article_ids = item_df_indexed.index.values
n_negative_per_pos = 3

neg_rows = []
rng = np.random.default_rng(RANDOM_STATE)

for cust_id, art_id in train_pairs[["customer_id", "article_id"]].values:
    candidate_neg = rng.choice(all_article_ids, size=n_negative_per_pos, replace=False)
    for neg_item in candidate_neg:
        if neg_item == art_id:
            continue
        neg_rows.append((cust_id, neg_item, 0))

neg_df = pd.DataFrame(neg_rows, columns=["customer_id", "article_id", "label"])

train_full = pd.concat([train_pairs, neg_df], ignore_index=True)
print("train_pairs:", train_pairs.shape, "| neg_df:", neg_df.shape)
print("train_full:", train_full.shape)


# Loại bỏ cặp không có thông tin user / item
mask = train_full["customer_id"].isin(user_df_indexed.index) & \
       train_full["article_id"].isin(item_df_indexed.index)
train_full = train_full[mask].reset_index(drop=True)

print("train_full (after filtering):", train_full.shape)
train_full.head()


# - `HybridRecDataset`: trả về (user_cat, item_cat, item_num, label)


class HybridRecDataset(Dataset):
    def __init__(self, df, user_df_indexed, item_df_indexed):
        self.df = df
        self.user_df_idx = user_df_indexed
        self.item_df_idx = item_df_indexed
        
        # Cột categorical & numeric
        self.user_cat_cols = ["age_bucket_le", "club_member_status_le", "fashion_news_frequency_le"]
        self.item_cat_cols = ["product_type_name_le", "product_group_name_le",
                              "graphical_appearance_name_le", "colour_group_name_le"]
        self.item_num_cols = ["price"]
        
        self.user_cat = []
        self.item_cat = []
        self.item_num = []
        self.labels = []
        
        for _, row in df.iterrows():
            uid = row["customer_id"]
            iid = row["article_id"]
            label = row["label"]
            
            u = self.user_df_idx.loc[uid]
            it = self.item_df_idx.loc[iid]
            
            self.user_cat.append(u[self.user_cat_cols].values.astype(np.int64))
            self.item_cat.append(it[self.item_cat_cols].values.astype(np.int64))
            self.item_num.append(it[self.item_num_cols].values.astype(np.float32))
            self.labels.append(label)
        
        self.user_cat = torch.tensor(np.stack(self.user_cat), dtype=torch.long)
        self.item_cat = torch.tensor(np.stack(self.item_cat), dtype=torch.long)
        self.item_num = torch.tensor(np.stack(self.item_num), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.user_cat[idx],
            self.item_cat[idx],
            self.item_num[idx],
            self.labels[idx],
        )


#- `HybridRecModel`:
#   - Embedding cho mỗi cột categorical
#   - MLP cho nhánh user, MLP cho nhánh item
#   - Ghép 2 vector lại và qua MLP cuối để ra score \[0,1]



class HybridRecModel(nn.Module):
    def __init__(self, user_cardinalities, item_cardinalities, n_item_num=1, emb_dim=16):
        super().__init__()
        # Embedding user
        self.user_emb_layers = nn.ModuleDict()
        for col, card in user_cardinalities.items():
            self.user_emb_layers[col] = nn.Embedding(card, emb_dim)
        
        # Embedding item
        self.item_emb_layers = nn.ModuleDict()
        for col, card in item_cardinalities.items():
            self.item_emb_layers[col] = nn.Embedding(card, emb_dim)
        
        user_input_dim = len(user_cardinalities) * emb_dim
        item_input_dim = len(item_cardinalities) * emb_dim + n_item_num
        
        # MLP cho user branch
        self.user_mlp = nn.Sequential(
            nn.Linear(user_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # MLP cho item branch
        self.item_mlp = nn.Sequential(
            nn.Linear(item_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # MLP cuối
        self.final_mlp = nn.Sequential(
            nn.Linear(32 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_cat, item_cat, item_num):
        # user_cat: [B, n_user_cat]
        # item_cat: [B, n_item_cat]
        # item_num: [B, n_item_num]
        user_embs = []
        for i, col in enumerate(user_cardinalities.keys()):
            user_embs.append(self.user_emb_layers[col](user_cat[:, i]))
        user_emb = torch.cat(user_embs, dim=-1)
        
        item_embs = []
        for i, col in enumerate(item_cardinalities.keys()):
            item_embs.append(self.item_emb_layers[col](item_cat[:, i]))
        item_emb_cat = torch.cat(item_embs, dim=-1)
        item_input = torch.cat([item_emb_cat, item_num], dim=-1)
        
        u_vec = self.user_mlp(user_emb)
        i_vec = self.item_mlp(item_input)
        
        x = torch.cat([u_vec, i_vec], dim=-1)
        logit = self.final_mlp(x)
        score = self.sigmoid(logit).squeeze(-1)
        return score


max_train_samples = 200_000
if len(train_full) > max_train_samples:
    train_sample = train_full.sample(max_train_samples, random_state=RANDOM_STATE)
else:
    train_sample = train_full

train_sample = train_sample.reset_index(drop=True)
train_sample.shape


dataset = HybridRecDataset(train_sample, user_df_indexed, item_df_indexed)
len(dataset)


from sklearn.model_selection import train_test_split

indices = np.arange(len(dataset))
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=RANDOM_STATE)

train_subset = torch.utils.data.Subset(dataset, train_idx)
val_subset   = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_subset, batch_size=1024, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_subset, batch_size=2048, shuffle=False, num_workers=0)

len(train_loader), len(val_loader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


model = HybridRecModel(user_cardinalities, item_cardinalities, n_item_num=1, emb_dim=16)
model = model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def run_one_epoch(loader, model, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    
    total_loss = 0.0
    n_samples = 0
    
    for user_cat, item_cat, item_num, labels in loader:
        user_cat = user_cat.to(device)
        item_cat = item_cat.to(device)
        item_num = item_num.to(device)
        labels = labels.to(device)
        
        scores = model(user_cat, item_cat, item_num)
        loss = criterion(scores, labels)
        
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
    
    return total_loss / max(1, n_samples)

n_epochs = 3  # demo
for epoch in range(1, n_epochs+1):
    train_loss = run_one_epoch(train_loader, model, criterion, optimizer)
    val_loss = run_one_epoch(val_loader, model, criterion, optimizer=None)
    print(f"Epoch {epoch}/{n_epochs} - train_loss={train_loss:.4f} - val_loss={val_loss:.4f}")



# Chiến lược candidate đơn giản:
# - Với mỗi customer:
#   - Candidate = top-N popular items
#   - + các item tương tự (content-based) từ những sản phẩm họ đã mua trong train
# - Score từng (customer, item) bằng Hybrid model
# - Lấy top-12 theo score


# Map: customer -> list article_id đã mua trong train
customer_to_items_train = (
    train_df.groupby("customer_id")["article_id"]
    .apply(list)
    .to_dict()
)

def build_candidates_for_customer(customer_id, n_pop=200, n_cb_per_item=5):
    """
    Candidate items:
    - top-n_pop popular
    - + các item CB tương tự những item user đã mua
    """
    cands = set(popular_items[:n_pop])
    past_items = customer_to_items_train.get(customer_id, [])
    # Lấy tối đa 5 item gần nhất
    for aid in past_items[-5:]:
        sim_items = get_similar_items(aid, topk=n_cb_per_item)
        cands.update(sim_items)
    # Chỉ giữ items có metadata
    cands = [aid for aid in cands if aid in item_df_indexed.index]
    return cands


def score_items_for_customer(customer_id, candidate_items, model, topk=12):
    """
    Tính score (customer, item) bằng Hybrid model.
    Nếu customer không có info → fallback Popularity.
    """
    if customer_id not in user_df_indexed.index:
        return popular_items[:topk]
    
    u = user_df_indexed.loc[customer_id]
    user_cat_vals = u[["age_bucket_le", "club_member_status_le", "fashion_news_frequency_le"]].values.astype(np.int64)
    user_cat_t = torch.tensor(user_cat_vals, dtype=torch.long).unsqueeze(0).to(device)
    
    scored_items = []
    for aid in candidate_items:
        it = item_df_indexed.loc[aid]
        item_cat_vals = it[["product_type_name_le", "product_group_name_le",
                            "graphical_appearance_name_le", "colour_group_name_le"]].values.astype(np.int64)
        item_num_vals = it[["price"]].values.astype(np.float32)
        
        item_cat_t = torch.tensor(item_cat_vals, dtype=torch.long).unsqueeze(0).to(device)
        item_num_t = torch.tensor(item_num_vals, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            score = model(user_cat_t, item_cat_t, item_num_t).item()
        scored_items.append((aid, score))
    
    scored_items.sort(key=lambda x: x[1], reverse=True)
    top_items = [aid for aid, _ in scored_items[:topk]]
    return top_items


# Demo gợi ý cho 5 khách hàng trong tập test
sample_test_customers = list(true_items_dict.keys())[:5]

for cid in sample_test_customers:
    cands = build_candidates_for_customer(cid)
    recs = score_items_for_customer(cid, cands, model, topk=12)
    print("Customer:", cid)
    print("  Recs:", recs[:5], "...")
    print()


def recall_at_k_hybrid(true_items_dict, model, sample_size=2000, k=12):
    customers_list = list(true_items_dict.keys())
    if len(customers_list) > sample_size:
        customers_list = customers_list[:sample_size]
    
    hits = 0
    n_cust = 0
    
    for cid in customers_list:
        true_items = true_items_dict[cid]
        if not true_items:
            continue
        
        cands = build_candidates_for_customer(cid)
        recs = score_items_for_customer(cid, cands, model, topk=k)
        recs_set = set(recs)
        
        if len(recs_set & true_items) > 0:
            hits += 1
        n_cust += 1
    
    return hits / max(1, n_cust)

hybrid_recall = recall_at_k_hybrid(true_items_dict, model, sample_size=2000, k=12)
print("Hybrid model Recall@12 (subset):", hybrid_recall)
print("Baseline Popularity Recall@12 :", baseline_recall)


customers_all = customers["customer_id"].unique()
max_customers_submission = 50_000

sub_rows = []
for i, cid in enumerate(customers_all[:max_customers_submission]):
    cands = build_candidates_for_customer(cid)
    recs = score_items_for_customer(cid, cands, model, topk=12)
    pred_str = " ".join([str(aid) for aid in recs])
    sub_rows.append((cid, pred_str))
    
    if (i+1) % 5000 == 0:
        print(f"Đã xử lý {i+1} khách hàng")

submission_demo = pd.DataFrame(sub_rows, columns=["customer_id", "prediction"])
submission_demo.head()


submission_demo.to_csv("submission_demo.csv", index=False)
print("Đã lưu file: submission_demo.csv")

