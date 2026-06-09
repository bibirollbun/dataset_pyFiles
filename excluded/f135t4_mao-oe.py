# ============================================================
# ONE CELL: XGBoost A/B/C
#   A = Opcode(filtered) only  (from rf_gini_out/X_parts/*.npz)
#   B = Opcode(filtered) + Segment (from features_full_v4/Xtr_num.npy)
#   C = Opcode(filtered) + Segment + 2nd (sf_names) (from Xtr_num.npy)
#
# Needs:
#   /kaggle/input/rf-gini-out/rf_gini_out/X_parts/X_part_*.npz   (or /kaggle/working/...)
#   /kaggle/input/malware-features-ckpt-v5/features_full_v4/
#       train_ids.txt, y_train.npy, Xtr_num.npy, config.json
#
# Output:
#   /kaggle/working/xgb_out_A/*
#   /kaggle/working/xgb_out_B/*
#   /kaggle/working/xgb_out_C/*
# ============================================================
import os, glob, json, time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report
import xgboost as xgb

# -----------------------
# Paths
# -----------------------
FEAT_ROOT = "/kaggle/input/malware-features-ckpt-v5/features_full_v4"
TRAIN_IDS_TXT = os.path.join(FEAT_ROOT, "train_ids.txt")
Y_TRAIN_NPY   = os.path.join(FEAT_ROOT, "y_train.npy")
XTR_NUM_NPY   = os.path.join(FEAT_ROOT, "Xtr_num.npy")
CFG_JSON      = os.path.join(FEAT_ROOT, "config.json")

CAND_XPART_DIRS = [
    "/kaggle/working/rf_gini_out/X_parts",
    "/kaggle/input/rf-gini-out/rf_gini_out/X_parts",
    "/kaggle/input/rf_gini_out/rf_gini_out/X_parts",
    "/kaggle/input/**/rf_gini_out/X_parts",
]
X_PART_DIR = None
for p in CAND_XPART_DIRS:
    if "*" in p:
        hits = glob.glob(p, recursive=True)
        hits = [h for h in hits if os.path.isdir(h)]
        if hits:
            X_PART_DIR = hits[0]
            break
    else:
        if os.path.isdir(p):
            X_PART_DIR = p
            break

assert X_PART_DIR is not None, "Không tìm thấy X_parts. Hãy attach dataset rf_gini_out."
assert os.path.exists(TRAIN_IDS_TXT), "Không thấy train_ids.txt"
assert os.path.exists(Y_TRAIN_NPY), "Không thấy y_train.npy"
assert os.path.exists(XTR_NUM_NPY), "Không thấy Xtr_num.npy"
assert os.path.exists(CFG_JSON), "Không thấy config.json"

OUT_A = "/kaggle/working/xgb_out_A"
OUT_B = "/kaggle/working/xgb_out_B"
OUT_C = "/kaggle/working/xgb_out_C"
os.makedirs(OUT_A, exist_ok=True)
os.makedirs(OUT_B, exist_ok=True)
os.makedirs(OUT_C, exist_ok=True)

print("[OK] X_PART_DIR:", X_PART_DIR)
print("[OK] FEAT_ROOT:", FEAT_ROOT)

# -----------------------
# Load ids + labels (aligned)
# -----------------------
with open(TRAIN_IDS_TXT, "r", encoding="utf-8") as f:
    train_ids = [x.strip() for x in f if x.strip()]
y = np.load(Y_TRAIN_NPY).astype(np.int64)

# normalize y to 0..8
if y.min() == 1 and y.max() == 9:
    y = y - 1

n_samples = len(train_ids)
assert len(y) == n_samples, f"Mismatch: len(y)={len(y)} vs n_samples={n_samples}"
n_classes = int(y.max() + 1)
print("[INFO] n_samples:", n_samples, "| n_classes:", n_classes)

# -----------------------
# Load opcode filtered X (CSR) from X_parts
# -----------------------
xparts = sorted(glob.glob(os.path.join(X_PART_DIR, "X_part_*.npz")))
assert xparts, "Không có file X_part_*.npz trong X_PART_DIR"

t0 = time.time()
mats = []
for fp in xparts:
    Xi = sparse.load_npz(fp).tocsr()
    mats.append(Xi)
    print(f"[LOAD] {os.path.basename(fp)} shape={Xi.shape} nnz={Xi.nnz}")
X_op = sparse.vstack(mats, format="csr")
del mats
print("[X_op] shape:", X_op.shape, "| nnz:", X_op.nnz, "| load_min:", (time.time()-t0)/60)
assert X_op.shape[0] == n_samples, f"Row mismatch: X_op has {X_op.shape[0]} rows but n_samples={n_samples}"

# -----------------------
# Load numeric features: segment + 2nd from Xtr_num.npy using config.json
# -----------------------
cfg = json.load(open(CFG_JSON, "r", encoding="utf-8"))
segments = cfg.get("segments", [])
sf_names = cfg.get("sf_names", [])

n_seg = len(segments)
n_sf  = len(sf_names)
assert n_seg > 0 and n_sf > 0, "config.json thiếu segments hoặc sf_names?"

X_num = np.load(XTR_NUM_NPY)
assert X_num.shape[0] == n_samples, f"Row mismatch: X_num has {X_num.shape[0]} rows but n_samples={n_samples}"

need_cols = n_seg + n_sf
if X_num.shape[1] < need_cols:
    raise RuntimeError(f"Xtr_num.npy has {X_num.shape[1]} cols < needed {need_cols} (segments+sf_names).")

# assume layout: [segments..., sf_names...]
X_seg = X_num[:, :n_seg]
X_2nd = X_num[:, n_seg:n_seg+n_sf]

X_seg_sp = sparse.csr_matrix(X_seg, dtype=np.float32)
X_2nd_sp = sparse.csr_matrix(X_2nd, dtype=np.float32)

print("[X_num] shape:", X_num.shape)
print("[SEG] shape:", X_seg_sp.shape, "segments=", n_seg)
print("[2ND] shape:", X_2nd_sp.shape, "sf_names=", n_sf)

# -----------------------
# Build A / B / C matrices
# -----------------------
X_A = X_op
X_B = sparse.hstack([X_op, X_seg_sp], format="csr")
X_C = sparse.hstack([X_op, X_seg_sp, X_2nd_sp], format="csr")

print("[X_A] shape:", X_A.shape, "| nnz:", X_A.nnz)
print("[X_B] shape:", X_B.shape, "| nnz:", X_B.nnz)
print("[X_C] shape:", X_C.shape, "| nnz:", X_C.nnz)

# -----------------------
# Split indices (same split for A/B/C)
# -----------------------
idx = np.arange(n_samples)
tr_idx, va_idx = train_test_split(
    idx, test_size=0.20, random_state=20251226, stratify=y
)
y_tr = y[tr_idx]
y_va = y[va_idx]

# -----------------------
# Params (GIỮ NGUYÊN của bạn)
# -----------------------

params_A = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_bin": 256,

    "eta": 0.04,
    "max_depth": 2,
    "min_child_weight": 110,   # (A) bớt chặt hơn 130
    "gamma": 24,               # (A) bớt chặt hơn 29

    "subsample": 0.42,
    "colsample_bytree": 0.42,

    "lambda": 75.0,
    "alpha": 22.0,

    "seed": 20251226,
    "verbosity": 1,
}

params_B = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_bin": 256,

    "eta": 0.04,
    "max_depth": 2,
    "min_child_weight": 95,    # (B) nới tiếp
    "gamma": 20,

    "subsample": 0.48,
    "colsample_bytree": 0.48,

    "lambda": 60.0,
    "alpha": 18.0,

    "seed": 20251226,
    "verbosity": 1,
}

params_C = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_bin": 256,

    "eta": 0.04,
    "max_depth": 3,            # (C) tăng depth để ăn được interactions opcode<->numeric
    "min_child_weight": 60,    # (C) nới rõ để 2nd features có tác dụng
    "gamma": 10,

    "subsample": 0.62,         # (C) nhiều dữ liệu/cây hơn
    "colsample_bytree": 0.62,  # (C) nhiều features/cây hơn

    "lambda": 35.0,
    "alpha": 10.0,

    "seed": 20251226,
    "verbosity": 1,
}


num_boost_round = 1500
early_stopping_rounds = 50
verbose_eval = 50

def train_one(tag, X_all, out_dir, params):
    X_tr = X_all[tr_idx]
    X_va = X_all[va_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)

    print(f"\n===== TRAIN {tag} =====")
    t0 = time.time()
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval
    )
    mins = (time.time()-t0)/60
    print(f"[{tag}] fit done. minutes:", mins)

    p_va = bst.predict(dval)
    pred_va = p_va.argmax(axis=1)
    acc = accuracy_score(y_va, pred_va)
    ll  = log_loss(y_va, p_va, labels=list(range(n_classes)))

    print(f"[{tag}][VAL] accuracy={acc:.4f} | logloss={ll:.4f}\n")
    print(classification_report(y_va, pred_va, digits=4))

    model_path = os.path.join(out_dir, f"xgb_{tag}.json")
    bst.save_model(model_path)

    metrics = {
        "tag": tag,
        "val_accuracy": float(acc),
        "val_logloss": float(ll),
        "best_iteration": int(bst.best_iteration) if bst.best_iteration is not None else None,
        "best_score": float(bst.best_score) if bst.best_score is not None else None,
        "n_features": int(X_all.shape[1]),
        "params": params,
        "num_boost_round": int(num_boost_round),
        "early_stopping_rounds": int(early_stopping_rounds),
        "train_minutes": float(mins),
    }
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    imp = bst.get_score(importance_type="gain")
    rows = [(int(k[1:]), v) for k, v in imp.items()]
    pd.DataFrame(rows, columns=["feature_index", "gain"]).sort_values("gain", ascending=False)\
      .to_csv(os.path.join(out_dir, f"xgb_gain_{tag}.csv"), index=False)

    print(f"[{tag}] saved:", model_path)
    return acc, ll

accA, llA = train_one("A_opcode_only", X_A, OUT_A, params_A)
accB, llB = train_one("B_opcode_segment", X_B, OUT_B, params_B)
accC, llC = train_one("C_opcode_segment_2nd", X_C, OUT_C, params_C)

print("\n=== SUMMARY ===")
print("A (opcode)                 acc=", accA, "logloss=", llA)
print("B (opcode+segment)         acc=", accB, "logloss=", llB)
print("C (opcode+segment+2nd)     acc=", accC, "logloss=", llC)

