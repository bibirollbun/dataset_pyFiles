import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu
CSV_PATH = "/kaggle/input/MABe-mouse-behavior-detection/train.csv"  # đổi đường dẫn nếu cần
df = pd.read_csv(CSV_PATH)

# ===== 1. Tổng quan =====
print("Shape:", df.shape)
print("\n== Dtypes ==")
print(df.dtypes)
print("\n== Missing values (Top 30) ==")
print(df.isnull().sum().sort_values(ascending=False).head(30))
print("\n== Numeric summary ==")
print(df.describe(include='number'))
print("\n== Categorical summary ==")
print(df.describe(include='object'))

# ===== 2. Biểu đồ Missing values =====
missing_counts = df.isnull().sum().sort_values(ascending=False)
top_missing = missing_counts.head(30)
plt.figure(figsize=(10,8))
plt.barh(top_missing.index, top_missing.values)
plt.title("Top 30 cột có giá trị thiếu nhiều nhất")
plt.xlabel("Số lượng thiếu")
plt.ylabel("Cột")
plt.show()

# ===== 3. Histogram cho các cột số =====
numeric_cols = [
    "frames_per_second", "video_duration_sec", "pix_per_cm_approx",
    "video_width_pix", "video_height_pix", "arena_width_cm", "arena_height_cm"
]
for c in [c for c in numeric_cols if c in df.columns]:
    s = df[c].dropna()
    if s.empty:
        continue
    plt.figure(figsize=(8,4))
    if s.max() / max(s.min(), 1e-9) > 1e3:
        plt.hist(s, bins=50, log=True)
        plt.title(f"Histogram (log-y) của {c}")
        plt.ylabel("Tần suất (log)")
    else:
        plt.hist(s, bins=50)
        plt.title(f"Histogram của {c}")
        plt.ylabel("Tần suất")
    plt.xlabel(c)
    plt.show()

# ===== 4. Bar chart cho một số cột phân loại =====
cat_cols = ["arena_shape", "arena_type", "tracking_method", "mouse1_strain", "mouse1_sex", "mouse1_age"]
for c in [c for c in cat_cols if c in df.columns]:
    vc = df[c].astype(str).fillna("NA").value_counts().head(20)
    plt.figure(figsize=(10,6))
    plt.barh(vc.index, vc.values)
    plt.title(f"Top 20 giá trị phổ biến của {c}")
    plt.xlabel("Số lượng")
    plt.ylabel(c)
    plt.show()

# ===== 5. Tỉ lệ có behaviors_labeled =====
if "behaviors_labeled" in df.columns:
    has_labels = df["behaviors_labeled"].notna().astype(int)
    counts = has_labels.value_counts().sort_index()
    plt.figure(figsize=(6,4))
    plt.bar(["Không có nhãn","Có nhãn"], [counts.get(0,0), counts.get(1,0)])
    plt.title("Tỉ lệ video có/không có behaviors_labeled")
    plt.ylabel("Số video")
    plt.show()

    # Tách sơ hành vi ra top action (nếu muốn)
    actions = []
    for raw in df["behaviors_labeled"].dropna().astype(str):
        s = raw.strip().strip("[]").replace('"', "").replace("'", "")
        parts = [p.strip() for p in s.split(", ")]
        for i in range(0, len(parts), 3):
            triple = parts[i:i+3]
            if len(triple) == 3:
                actions.append(triple[2])
    if actions:
        import pandas as pd
        vc_act = pd.Series(actions).value_counts().head(20)
        plt.figure(figsize=(10,6))
        plt.barh(vc_act.index, vc_act.values)
        plt.title("Top 20 hành vi trong behaviors_labeled (ước lượng)")
        plt.xlabel("Số lần xuất hiện")
        plt.ylabel("action")
        plt.show()

# ===== 6. Scatter video dims =====
if all(c in df.columns for c in ["video_width_pix","video_height_pix"]):
    plt.figure(figsize=(6,6))
    sample = df[["video_width_pix","video_height_pix"]].dropna().sample(min(3000,len(df)), random_state=0)
    plt.scatter(sample["video_width_pix"], sample["video_height_pix"], s=6)
    plt.title("Scatter: video_width_pix vs video_height_pix (sample)")
    plt.xlabel("video_width_pix")
    plt.ylabel("video_height_pix")
    plt.show()

# ===== 7. Scatter arena dims =====
if all(c in df.columns for c in ["arena_width_cm","arena_height_cm"]):
    plt.figure(figsize=(6,6))
    sample = df[["arena_width_cm","arena_height_cm"]].dropna().sample(min(3000,len(df)), random_state=0)
    plt.scatter(sample["arena_width_cm"], sample["arena_height_cm"], s=6)
    plt.title("Scatter: arena_width_cm vs arena_height_cm (sample)")
    plt.xlabel("arena_width_cm")
    plt.ylabel("arena_height_cm")
    plt.show()


