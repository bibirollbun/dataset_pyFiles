import os, sys, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ====== 1) Khai báo đường dẫn submissions ======
# Lưu ý: bạn đưa 1 dòng bị dính 2 path, mình tách sẵn ra dưới đây.
SUB_PATHS = [
    "/kaggle/input/x-ray/submission.csv",
    "/kaggle/input/pre-trained-fine-tuning-for-x-rays/sample_submission.csv",
    "/kaggle/input/kaggle-chest-x-ray-submission-notebook/submission.csv",
    "/kaggle/input/vit-imageaug-gput42/submission.csv",
    "/kaggle/input/grand-xray-slam-division-a-tpu/submission.csv",
]

# ====== 2) Định nghĩa cột ID và nhãn (khớp code train bạn đưa) ======
ID_CANDIDATES = ["Image_name", "image_name", "id", "ID", "image_id"]
TARGET_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# ====== 3) Helpers ======
def _find_id_col(df: pd.DataFrame):
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: chọn cột đầu tiên
    return df.columns[0]

def _clip01(x):
    return np.clip(x, 1e-6, 1-1e-6)

def _to_logit(p):
    p = _clip01(p)
    return np.log(p/(1-p))

def _from_logit(z):
    return 1/(1+np.exp(-z))

def _safe_read_csv(p):
    try:
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df
        else:
            print(f"[WARN] File không tồn tại: {p}")
            return None
    except Exception as e:
        print(f"[WARN] Lỗi đọc {p}: {e}")
        return None

def _ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ====== 4) Load tất cả submissions hợp lệ ======
loaded = []
for p in SUB_PATHS:
    df = _safe_read_csv(p)
    if df is None:
        continue
    id_col = _find_id_col(df)

    # Chuẩn hóa ID để tránh mismatch ký tự trắng/dtype
    if id_col in df.columns:
        df[id_col] = df[id_col].astype(str).str.strip()

    # Giữ lại ID + các label cần
    keep_cols = [id_col] + [c for c in TARGET_LABELS if c in df.columns]
    df = df[keep_cols].copy()
    df = _ensure_numeric(df, [c for c in TARGET_LABELS if c in df.columns])

    # báo nhanh tình trạng
    missing = [c for c in TARGET_LABELS if c not in df.columns]
    if missing:
        print(f"[INFO] {os.path.basename(p)} thiếu {len(missing)} nhãn (bỏ qua các cột này): {missing[:4]}{'...' if len(missing)>4 else ''}")

    # đảm bảo id đứng đầu
    df.columns = [id_col] + [c for c in df.columns if c != id_col]
    df.attrs['id_col'] = id_col
    df.attrs['path'] = p
    loaded.append(df)

if len(loaded) < 2:
    raise RuntimeError("Cần >= 2 submission hợp lệ để ensemble.")

print(f"[INFO] Đã nạp {len(loaded)} submissions hợp lệ.")

# ====== 5) Chọn submission gốc làm chuẩn thứ tự ID ======
# Ưu tiên file có đủ nhiều nhãn
loaded_sorted = sorted(loaded, key=lambda d: sum([c in d.columns for c in TARGET_LABELS]), reverse=True)
base_df = loaded_sorted[0]
base_id = base_df.attrs['id_col']
print(f"[INFO] Dùng {os.path.basename(base_df.attrs['path'])} làm chuẩn thứ tự ID (id_col='{base_id}').")

# ====== 6) Inner-join theo ID để chắc chắn các hàng trùng nhau (FIXED) ======
def project(df, id_col, cols):
    cols_use = [c for c in cols if c in df.columns]
    out = df[[id_col] + cols_use].copy()
    out[id_col] = out[id_col].astype(str).str.strip()
    return out

# giao nhãn thật sự có ở ít nhất 2 file
label_presence = {c: 0 for c in TARGET_LABELS}
for d in loaded:
    for c in TARGET_LABELS:
        if c in d.columns:
            label_presence[c] += 1
usable_labels = [c for c, v in label_presence.items() if v >= 2]
if not usable_labels:
    raise RuntimeError("Không có nhãn giao nhau giữa các submissions.")

print(f"[INFO] Số nhãn dùng để ensemble: {len(usable_labels)} / {len(TARGET_LABELS)}")

# tạo bảng hợp nhất theo ID chuẩn
merged = base_df[[base_id]].copy()
merged[base_id] = merged[base_id].astype(str).str.strip()

for i, d in enumerate(loaded):
    did = d.attrs['id_col']
    # đổi tên cột nhãn để tránh đụng tên (thêm hậu tố m{i})
    rename_map = {c: f"{c}__m{i}" for c in usable_labels if c in d.columns}
    dd = project(d, did, usable_labels).rename(columns=rename_map)

    # merge theo ID; nếu id cùng tên với base_id thì KHÔNG drop base_id
    if did == base_id:
        merged = merged.merge(dd, on=base_id, how='inner')
    else:
        merged = merged.merge(dd, left_on=base_id, right_on=did, how='inner')
        # chỉ drop cột right key khi nó KHÁC base_id
        if did != base_id and did in merged.columns:
            merged.drop(columns=[did], inplace=True)

if merged.shape[0] == 0:
    raise RuntimeError("Không còn bản ghi chung giữa các submissions sau khi join.")

print(f"[INFO] Số ảnh sau khi align: {merged.shape[0]}")

# ====== 7) Tạo các tensor xác suất theo từng nhãn ======
model_count = len(loaded)
per_label_arrays = {}  # label -> (N, M)
N = merged.shape[0]

for c in usable_labels:
    cols_c = [f"{c}__m{i}" for i in range(model_count) if f"{c}__m{i}" in merged.columns]
    if len(cols_c) == 0:
        continue
    arr = merged[cols_c].astype(float).to_numpy()
    # clip để tránh logit inf
    arr = _clip01(arr)
    per_label_arrays[c] = arr

# ====== 8) Các phương pháp ensemble ======
def ensemble_mean():
    out = pd.DataFrame({base_id: merged[base_id].values})
    for c, arr in per_label_arrays.items():
        out[c] = arr.mean(axis=1)
    return out

def ensemble_logit_mean():
    out = pd.DataFrame({base_id: merged[base_id].values})
    for c, arr in per_label_arrays.items():
        z = _to_logit(arr)
        out[c] = _from_logit(z.mean(axis=1))
    return out

def ensemble_rank_average():
    out = pd.DataFrame({base_id: merged[base_id].values})
    for c, arr in per_label_arrays.items():
        M = arr.shape[1]
        # rank theo cột (mỗi model), rồi chuẩn hóa về [0,1]
        ranks = np.zeros_like(arr, dtype=float)
        for j in range(M):
            order = arr[:, j].argsort()
            r = np.empty_like(order, dtype=float)
            r[order] = np.arange(len(order))
            r = r / (len(order)-1 if len(order) > 1 else 1)
            ranks[:, j] = r
        out[c] = ranks.mean(axis=1)
    return out

# ====== 9) Sinh ba file blend ======
blend_mean = ensemble_mean()
blend_logit = ensemble_logit_mean()
blend_rank = ensemble_rank_average()

# Bảo toàn tất cả label gốc: nếu label nào không usable thì điền 0.0
def finalize_columns(df_out):
    # giữ nguyên thứ tự ID như base_df
    df_out = base_df[[base_id]].merge(df_out, on=base_id, how='left')
    for c in TARGET_LABELS:
        if c not in df_out.columns:
            df_out[c] = 0.0
    # sắp cột: ID + labels
    df_out = df_out[[base_id] + TARGET_LABELS]
    # điền missing, ép float và clip
    for c in TARGET_LABELS:
        df_out[c] = df_out[c].fillna(0.0).astype(float)
        df_out[c] = _clip01(df_out[c])
    return df_out

blend_mean  = finalize_columns(blend_mean)
blend_logit = finalize_columns(blend_logit)
blend_rank  = finalize_columns(blend_rank)

blend_mean.to_csv("blend_mean.csv", index=False)
blend_logit.to_csv("blend_logit.csv", index=False)
blend_rank.to_csv("blend_rank.csv", index=False)

print("[DONE] Đã tạo: blend_mean.csv, blend_logit.csv, blend_rank.csv")
print("Khuyến nghị nộp: blend_logit.csv (logit-mean thường ổn định nhất cho multi-label).")


