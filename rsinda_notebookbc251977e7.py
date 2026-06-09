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


import zipfile
with zipfile.ZipFile('/kaggle/input/mlsp-2013-birds/mlsp_contest_dataset.zip', "r") as zip_file:
    zip_file.extractall(path='/kaggle/working/')


# -------------------------------------------------
# Bird Sound Classification – LightGBM + Pseudo‑Labeling
# -------------------------------------------------
# 1. feature extraction (histogram, segment stats, wav2vec2, MFCC+deltas, raw audio stats)
# 2. 5‑fold CV with per‑class LightGBM (print mean AUC)
# 3. generate high‑confidence pseudo‑labels on the test set
# 4. re‑train on real + pseudo data (pseudo weight=0.3)
# 5. predict test set and write submission.csv
# -------------------------------------------------

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ------------------- 0. Paths -------------------
DATA_ROOT = "/kaggle/working/mlsp_contest_dataset/"
ESS = os.path.join(DATA_ROOT, "essential_data")
SUPP = os.path.join(DATA_ROOT, "supplemental_data")


# ------------------- 1. Helpers -------------------
def load_numeric_file(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace(",", " ")
            parts = line.split()
            if not parts[0].lstrip("-").isdigit():
                continue
            rows.append([int(parts[0])] + [float(p) for p in parts[1:]])
    return rows


# ------------------- 2. Meta information -------------------
# 2.1 Labels
labels_path = os.path.join(ESS, "rec_labels_test_hidden.txt")
labels = []
with open(labels_path) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("rec_id"):
            continue
        parts = line.split(",")
        rid = int(parts[0])
        labs = "?" if len(parts) == 2 and parts[1] == "?" else ",".join(parts[1:])
        labels.append({"rec_id": rid, "labels": labs})
labels_df = pd.DataFrame(labels)

# 2.2 rec_id → wav filename
id2file = {}
with open(os.path.join(ESS, "rec_id2filename.txt")) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("rec_id"):
            continue
        rid, fname = line.split(",")
        id2file[int(rid)] = fname

# 2.3 Histogram of segments (100‑dim)
hist_rows = load_numeric_file(os.path.join(SUPP, "histogram_of_segments.txt"))
hist_df = pd.DataFrame(hist_rows).rename(columns={0: "rec_id"}).set_index("rec_id")

# 2.4 Segment features → mean / std (38‑dim *2)
seg_rows = load_numeric_file(os.path.join(SUPP, "segment_features.txt"))
seg_feat = pd.DataFrame(seg_rows).rename(columns={0: "rec_id", 1: "seg_id"})
feat_cols = [c for c in seg_feat.columns if c not in ("rec_id", "seg_id")]
seg_stats_raw = seg_feat.groupby("rec_id")[feat_cols].agg(["mean", "std"])
seg_stats_raw.columns = [f"{c}_{s}" for c, s in seg_stats_raw.columns]
seg_stats = seg_stats_raw.copy()

# ------------------- 3. Train / Test split -------------------
NUM_CLASSES = 19
train_ids, test_ids = [], []
train_labels = []

for _, row in labels_df.iterrows():
    rid = row["rec_id"]
    if row["labels"] == "?":
        test_ids.append(rid)
    else:
        train_ids.append(rid)
        lbls = list(map(int, filter(None, row["labels"].split(","))))
        vec = np.zeros(NUM_CLASSES, dtype=int)
        if lbls:
            vec[lbls] = 1
        train_labels.append(vec)

train_labels = np.stack(train_labels)

# ------------------- 4. Align auxiliary tables -------------------
all_ids = set(train_ids + test_ids)

# histogram – fill missing with zeros
missing_hist = all_ids - set(hist_df.index)
for rid in missing_hist:
    hist_df.loc[rid] = np.zeros(hist_df.shape[1])
hist_df = hist_df.sort_index()

# segment stats – fill missing with zeros
missing_seg = all_ids - set(seg_stats.index)
if missing_seg:
    zero_seg = pd.DataFrame(
        np.zeros((len(missing_seg), seg_stats.shape[1])),
        index=list(missing_seg),
        columns=seg_stats.columns,
    )
    seg_stats = pd.concat([seg_stats, zero_seg])
seg_stats = seg_stats.sort_index()

# ------------------- 5. Audio helpers -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_waveform(rid):
    fname = id2file.get(rid, "")
    wav_path = os.path.join(ESS, "src_wavs", fname)
    if not os.path.isfile(wav_path):
        if not fname.lower().endswith(".wav"):
            wav_path = os.path.join(ESS, "src_wavs", fname + ".wav")
    if not os.path.isfile(wav_path):
        return None, None
    try:
        wav, sr = torchaudio.load(wav_path)
        return wav, sr
    except Exception:
        return None, None


# MFCC + delta
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=13,
    melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 40},
)


def mfcc_stats_with_delta(wav):
    mfcc = mfcc_transform(wav).squeeze(0)  # (13, T)
    mfcc = torch.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)
    delta = torchaudio.functional.compute_deltas(mfcc)
    mean = mfcc.mean(dim=1).numpy()
    std = mfcc.std(dim=1).numpy()
    d_mean = delta.mean(dim=1).numpy()
    d_std = delta.std(dim=1).numpy()
    return np.concatenate([mean, std, d_mean, d_std]).astype(np.float32)  # 52 dim


# wav2vec2 (base) – we only need the pooled mean/std
wav2vec_bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec_model = wav2vec_bundle.get_model().to(device).eval()


def wav2vec2_stats(wav):
    with torch.inference_mode():
        emb, _ = wav2vec_model(wav.to(device))
    emb = emb.squeeze(0)  # (seq_len, 768)
    mean = emb.mean(dim=0).cpu().numpy()
    std = emb.std(dim=0).cpu().numpy()
    return np.concatenate([mean, std]).astype(np.float32)  # 1536 dim


# simple raw‑audio statistics
def raw_audio_stats(wav):
    w = wav.squeeze().float()
    if w.numel() == 0:
        return np.zeros(8, dtype=np.float32)
    mean = w.mean().item()
    std = w.std().item()
    max_ = w.max().item()
    min_ = w.min().item()
    rms = torch.sqrt((w**2).mean()).item()
    zcr = ((w[:-1] * w[1:]) < 0).float().mean().item()
    energy = (w**2).mean().item()
    ptp = max_ - min_
    return np.array([mean, std, max_, min_, rms, zcr, energy, ptp], dtype=np.float32)


# ------------------- 6. Feature extraction (no augmentation) -------------------
ZERO_W2V = np.zeros(1536, dtype=np.float32)
ZERO_MFCC = np.zeros(52, dtype=np.float32)
ZERO_RAW = np.zeros(8, dtype=np.float32)


def extract_features(rid):
    wav, sr = load_waveform(rid)
    if wav is None:
        return ZERO_W2V, ZERO_MFCC, ZERO_RAW
    # resample / mono to 16kHz
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    w2v = wav2vec2_stats(wav)
    mfcc = mfcc_stats_with_delta(wav)
    raw = raw_audio_stats(wav)
    return w2v, mfcc, raw


print("Extracting training features …")
X_parts = []  # (rec_id, w2v, mfcc, raw)
y_parts = []

for rid in tqdm(sorted(train_ids)):
    w2v, mfcc, raw = extract_features(rid)
    X_parts.append((rid, w2v, mfcc, raw))
    y_parts.append(train_labels[train_ids.index(rid)])

rids_train = np.array([p[0] for p in X_parts])
w2v_arr = np.stack([p[1] for p in X_parts])
mfcc_arr = np.stack([p[2] for p in X_parts])
raw_arr = np.stack([p[3] for p in X_parts])

hist_arr = hist_df.loc[rids_train].values.astype(np.float32)
seg_arr = seg_stats.loc[rids_train].values.astype(np.float32)

X_train_raw = np.concatenate([hist_arr, seg_arr, w2v_arr, mfcc_arr, raw_arr], axis=1)
y_train = np.stack(y_parts)

print("Extracting test features …")
test_feats = [extract_features(rid) for rid in tqdm(sorted(test_ids))]
w2v_test, mfcc_test, raw_test = zip(*test_feats)
w2v_test = np.stack(w2v_test)
mfcc_test = np.stack(mfcc_test)
raw_test = np.stack(raw_test)

hist_test = hist_df.loc[sorted(test_ids)].values.astype(np.float32)
seg_test = seg_stats.loc[sorted(test_ids)].values.astype(np.float32)

X_test_raw = np.concatenate(
    [hist_test, seg_test, w2v_test, mfcc_test, raw_test], axis=1
)

# ------------------- 7. Scaling -------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# class‑wise pos weight (helps LightGBM with imbalance)
pos_counts = y_train.sum(axis=0)
neg_counts = y_train.shape[0] - pos_counts
scale_pos_weight = np.where(pos_counts == 0, 1.0, neg_counts / pos_counts)

# ------------------- 8. 5‑fold CV with LightGBM -------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    val_pred = np.zeros_like(y_val, dtype=np.float32)

    for cls in range(NUM_CLASSES):
        mdl = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.03,
            n_estimators=5000,
            num_leaves=127,
            bagging_fraction=0.8,
            feature_fraction=0.8,
            metric="auc",
            verbose=-1,
            scale_pos_weight=scale_pos_weight[cls],
        )
        mdl.fit(
            X_tr,
            y_tr[:, cls],
            eval_set=[(X_val, y_val[:, cls])],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )
        val_pred[:, cls] = mdl.predict_proba(X_val)[:, 1]

    # compute mean AUC (ignore classes without both pos & neg in val)
    aucs = [
        roc_auc_score(y_val[:, c], val_pred[:, c])
        for c in range(NUM_CLASSES)
        if len(np.unique(y_val[:, c])) > 1
    ]
    fold_auc = np.mean(aucs)
    cv_aucs.append(fold_auc)
    print(f"Fold {fold+1} – mean AUC: {fold_auc:.5f}")

print(f"\n5‑fold CV mean AUC: {np.mean(cv_aucs):.5f}")

# ------------------- 9. Train on full data (real only) -------------------
full_models = []
for cls in range(NUM_CLASSES):
    mdl = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        n_estimators=5000,
        num_leaves=127,
        bagging_fraction=0.8,
        feature_fraction=0.8,
        metric="auc",
        verbose=-1,
        scale_pos_weight=scale_pos_weight[cls],
    )
    mdl.fit(X_train, y_train[:, cls])
    full_models.append(mdl)

# initial test prediction (used for pseudo‑labeling)
test_pred_initial = np.stack(
    [mdl.predict_proba(X_test)[:, 1] for mdl in full_models], axis=1
)

# ------------------- 10. Pseudo‑label generation -------------------
POS_THRESH = 0.90
NEG_THRESH = 0.10
pseudo_labels = []
pseudo_weights = []  # 0.3 for pseudo, 1.0 for real

for i, rid in enumerate(sorted(test_ids)):
    probs = test_pred_initial[i]
    pseudo_vec = np.full(NUM_CLASSES, -1, dtype=int)  # -1 = ignore
    for cls in range(NUM_CLASSES):
        p = probs[cls]
        if p >= POS_THRESH:
            pseudo_vec[cls] = 1
        elif p <= NEG_THRESH:
            pseudo_vec[cls] = 0
    # keep only rows that have at least one confident label
    if (pseudo_vec != -1).any():
        pseudo_labels.append(pseudo_vec)
        pseudo_weights.append(0.3)  # lower weight than real samples
        # add the feature row for this test recording
        # (use the same X_test row already computed)
        # We'll concatenate later

pseudo_labels = np.array(pseudo_labels)
pseudo_weights = np.array(pseudo_weights, dtype=np.float32)

if pseudo_labels.size > 0:
    # features of pseudo samples
    pseudo_features = X_test[
        ((test_pred_initial >= POS_THRESH) | (test_pred_initial <= NEG_THRESH)).any(
            axis=1
        )
    ]
    # combine
    X_combined = np.vstack([X_train, pseudo_features])
    y_combined = np.vstack([y_train, pseudo_labels])
    # sample weight per instance (same for all classes)
    sample_weight = np.concatenate([np.ones(X_train.shape[0]), pseudo_weights])
else:
    X_combined, y_combined, sample_weight = X_train, y_train, np.ones(X_train.shape[0])

# ------------------- 11. Retrain on real + pseudo -------------------
final_models = []
for cls in range(NUM_CLASSES):
    mdl = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        n_estimators=5000,
        num_leaves=127,
        bagging_fraction=0.8,
        feature_fraction=0.8,
        metric="auc",
        verbose=-1,
        scale_pos_weight=scale_pos_weight[cls],
    )
    # Filter out samples with -1 (ignore) labels for this class
    valid_mask = y_combined[:, cls] != -1
    X_cls = X_combined[valid_mask]
    y_cls = y_combined[valid_mask, cls]
    w_cls = sample_weight[valid_mask]
    
    mdl.fit(X_cls, y_cls, sample_weight=w_cls)
    final_models.append(mdl)

# ------------------- 12. Final test prediction -------------------
test_pred = np.stack([mdl.predict_proba(X_test)[:, 1] for mdl in final_models], axis=1)

# ------------------- 13. Write submission -------------------
rows = []
for rid, probs in zip(sorted(test_ids), test_pred):
    for sp_id, prob in enumerate(probs):
        Id = rid * 100 + sp_id
        rows.append({"Id": Id, "Probability": prob})

submission = pd.DataFrame(rows).sort_values("Id")
submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)

print(f"\nSubmission written to {submission_path}")


""




