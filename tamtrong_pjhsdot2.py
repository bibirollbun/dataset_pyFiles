import os, re, glob, random, time
from collections import defaultdict  # FIX: bạn thiếu dòng này

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# =======================
# Config
# =======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

TRAIN_ROOT = "/kaggle/input/indoor-location-navigation/train"
TEST_ROOT  = "/kaggle/input/indoor-location-navigation/test"
SAMPLE_SUB = "/kaggle/input/indoor-location-navigation/sample_submission.csv"

assert os.path.isdir(TRAIN_ROOT)
assert os.path.isdir(TEST_ROOT)
assert os.path.isfile(SAMPLE_SUB)

# =======================
# Hyperparams
# =======================
FREQ_HZ = 50.0
WINDOW = 100
STRIDE = 20
BATCH = 256
EPOCHS = 6
LR = 1e-3

MAX_TRAIN_FILES = None     # tăng dần 6000/12000/None nếu muốn
VAL_RATIO = 0.2

WIFI_WIN_MS = 1000         # wifi window around waypoint
ALPHA_WIFI = 0.4           # fusion presence+RSSI in fingerprint
KNN_K = 30

GATE_SIGMA = 0.25          # fusion weight from WiFi confidence
print("OK paths")



train_files_all = glob.glob(TRAIN_ROOT + "/*/*/*.txt")
print("total train files:", len(train_files_all))

if MAX_TRAIN_FILES is not None:
    train_files_all = train_files_all[:MAX_TRAIN_FILES]
print("using train files:", len(train_files_all))

random.shuffle(train_files_all)
n_val = int(len(train_files_all) * VAL_RATIO)

val_files = train_files_all[:n_val]
tr_files  = train_files_all[n_val:]

print("train:", len(tr_files), "val:", len(val_files))
print("example train file:", tr_files[0])



TYPE_ACCEL = "TYPE_ACCELEROMETER"
TYPE_GYRO  = "TYPE_GYROSCOPE"
TYPE_WP    = "TYPE_WAYPOINT"
TYPE_WIFI  = "TYPE_WIFI"

def parse_floor_str(floor_str: str) -> int:
    s = str(floor_str).strip().upper()
    m = re.match(r"^([BF])(\d+)$", s)
    if m:
        sign = -1 if m.group(1) == "B" else 1
        return sign * int(m.group(2))
    return 0

def parse_train_path(fp: str):
    parts = fp.replace("\\", "/").split("/")
    idx = parts.index("train")
    site = parts[idx+1]
    floor_str = parts[idx+2]
    path = parts[idx+3].replace(".txt","")
    return site, parse_floor_str(floor_str), path

def read_txt(fp: str):
    acc, gyro, wp, wifi = [], [], [], []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            v = line.split("\t")
            if len(v) < 2:
                continue
            ts = int(v[0]); t = v[1]
            try:
                if t == TYPE_ACCEL and len(v) >= 5:
                    acc.append([ts, float(v[2]), float(v[3]), float(v[4])])
                elif t == TYPE_GYRO and len(v) >= 5:
                    gyro.append([ts, float(v[2]), float(v[3]), float(v[4])])
                elif t == TYPE_WP and len(v) >= 4:
                    wp.append([ts, float(v[2]), float(v[3])])
                elif t == TYPE_WIFI and len(v) >= 5:
                    wifi.append([ts, str(v[3]), float(v[4])])  # bssid,rssi
            except:
                continue

    return {
        "acc":  pd.DataFrame(acc,  columns=["ts","ax","ay","az"]),
        "gyro": pd.DataFrame(gyro, columns=["ts","gx","gy","gz"]),
        "wp":   pd.DataFrame(wp,   columns=["ts","x","y"]),
        "wifi": pd.DataFrame(wifi, columns=["ts","bssid","rssi"]),
    }



class Synchronizer:
    def __init__(self, freq_hz=50.0):
        self.dt = 1000.0 / freq_hz

    def _interp(self, df, cols, t_new):
        if df.empty or len(df) < 2:
            return None
        df = df.sort_values("ts")
        f = interp1d(df["ts"].values, df[cols].values,
                     axis=0, kind="linear",
                     fill_value="extrapolate",
                     bounds_error=False,
                     assume_sorted=True)
        return f(t_new).astype(np.float32)

    def sync(self, data):
        acc, gyro, wp = data["acc"], data["gyro"], data["wp"]
        if len(acc) < 2 or len(gyro) < 2 or len(wp) < 2:
            return None

        t0 = max(acc["ts"].min(), gyro["ts"].min(), wp["ts"].min())
        t1 = min(acc["ts"].max(), gyro["ts"].max(), wp["ts"].max())
        if t1 <= t0:
            return None

        t_new = np.arange(t0, t1+1, self.dt).astype(np.int64)

        A = self._interp(acc,  ["ax","ay","az"], t_new)
        G = self._interp(gyro, ["gx","gy","gz"], t_new)
        P = self._interp(wp,   ["x","y"],       t_new)
        if A is None or G is None or P is None:
            return None

        df = pd.DataFrame({
            "ts": t_new,
            "ax": A[:,0], "ay": A[:,1], "az": A[:,2],
            "gx": G[:,0], "gy": G[:,1], "gz": G[:,2],
            "x":  P[:,0], "y":  P[:,1],
        }).dropna()

        if len(df) < WINDOW + 5:
            return None
        return df

syncer = Synchronizer(freq_hz=FREQ_HZ)
print("syncer ready")



def wifi_fingerprint_at_ts(wifi_df: pd.DataFrame, ts: int, win_ms: int):
    if wifi_df.empty:
        return {}
    w = wifi_df[(wifi_df["ts"] >= ts-win_ms) & (wifi_df["ts"] <= ts+win_ms)]
    if w.empty:
        return {}
    return w.groupby("bssid")["rssi"].mean().to_dict()

def build_knn_anchors(files, alpha=0.4, k=30):
    site_rows = defaultdict(list)  # site -> (floor,x,y,fp_dict)

    for i, fp in enumerate(files):
        site, floor, _ = parse_train_path(fp)
        d = read_txt(fp)
        wp = d["wp"]
        wifi = d["wifi"]
        if wp.empty or wifi.empty:
            continue

        for _, r in wp.iterrows():
            fpd = wifi_fingerprint_at_ts(wifi, int(r["ts"]), WIFI_WIN_MS)
            if not fpd:
                continue
            site_rows[site].append((floor, float(r["x"]), float(r["y"]), fpd))

        if (i+1) % 500 == 0:
            print("anchors processed", i+1)

    site_data = {}
    for site, rows in site_rows.items():
        bssid_set = set()
        for _,_,_,fpd in rows:
            bssid_set.update(fpd.keys())
        b2i = {b:i for i,b in enumerate(sorted(bssid_set))}
        n, m = len(rows), len(b2i)

        indptr=[0]; idxs=[]; rssi_vals=[]; bin_vals=[]
        y_floor=np.zeros(n,np.int32)
        y_xy=np.zeros((n,2),np.float32)

        for j,(floor,x,y,fpd) in enumerate(rows):
            y_floor[j]=floor
            y_xy[j]=[x,y]
            for bssid,rssi in fpd.items():
                ii=b2i.get(bssid)
                if ii is None: 
                    continue
                idxs.append(ii)
                rssi_vals.append(float(np.clip(rssi,-100,-30)))
                bin_vals.append(1.0)
            indptr.append(len(idxs))

        Xr = csr_matrix((np.array(rssi_vals,np.float32), np.array(idxs,np.int32), np.array(indptr,np.int32)), shape=(n,m))
        Xb = csr_matrix((np.array(bin_vals,np.float32),  np.array(idxs,np.int32), np.array(indptr,np.int32)), shape=(n,m))

        Xr = normalize(Xr, axis=1)
        Xb = normalize(Xb, axis=1)
        X  = hstack([Xb.multiply(alpha), Xr.multiply(1-alpha)]).tocsr()
        X  = normalize(X, axis=1)

        knn = NearestNeighbors(n_neighbors=min(k,n), metric="cosine")
        knn.fit(X)

        site_data[site] = {"b2i": b2i, "X": X, "knn": knn, "y_floor": y_floor, "y_xy": y_xy}
        print(f"site={site} anchors={n} bssids={m}")

    return site_data

anchor_knn = build_knn_anchors(tr_files, alpha=ALPHA_WIFI, k=KNN_K)
print("anchor sites:", len(anchor_knn))



def vectorize_fusion(fpd, b2i, alpha=0.4):
    if not fpd:
        return None
    idxs=[]; rssi=[]; bins=[]
    for bssid,val in fpd.items():
        j=b2i.get(bssid)
        if j is None: 
            continue
        idxs.append(j)
        rssi.append(float(np.clip(val,-100,-30)))
        bins.append(1.0)
    if not idxs:
        return None
    m=len(b2i)
    indptr=np.array([0,len(idxs)],np.int32)
    idxs=np.array(idxs,np.int32)
    Xr=csr_matrix((np.array(rssi,np.float32), idxs, indptr), shape=(1,m))
    Xb=csr_matrix((np.array(bins,np.float32), idxs, indptr), shape=(1,m))
    Xr=normalize(Xr, axis=1)
    Xb=normalize(Xb, axis=1)
    X=hstack([Xb.multiply(alpha), Xr.multiply(1-alpha)]).tocsr()
    X=normalize(X, axis=1)
    return X

def knn_anchor_predict(site, wifi_df, ts, k=30, alpha=0.4):
    d = anchor_knn.get(site)
    if d is None:
        return 0, 0.0, 0.0, 0.0, 1.0

    fpd = wifi_fingerprint_at_ts(wifi_df, ts, WIFI_WIN_MS)
    xvec = vectorize_fusion(fpd, d["b2i"], alpha=alpha)
    if xvec is None:
        return 0, 0.0, 0.0, 0.0, 1.0

    k = min(k, d["X"].shape[0])
    dist, idx = d["knn"].kneighbors(xvec, n_neighbors=k, return_distance=True)
    dist = dist[0]; idx = idx[0]
    w = 1.0 / (dist + 1e-6)

    floors = d["y_floor"][idx]
    xy = d["y_xy"][idx]

    # weighted floor
    scores={}
    for f,ww in zip(floors.tolist(), w.tolist()):
        scores[f]=scores.get(f,0.0)+ww
    floor_pred = int(max(scores.items(), key=lambda x:x[1])[0])

    x_pred = float(np.sum(xy[:,0]*w)/np.sum(w))
    y_pred = float(np.sum(xy[:,1]*w)/np.sum(w))

    best = float(dist[0])
    conf = float(np.exp(-(best*best)/(2*GATE_SIGMA*GATE_SIGMA)))
    return floor_pred, x_pred, y_pred, conf, best



FEATURES = ["ax","ay","az","gx","gy","gz"]

def make_windows_odometry(df, window=100, stride=20):
    arr = df[FEATURES + ["x","y"]].values.astype(np.float32)
    X_list=[]; y_list=[]
    for st in range(0, len(df)-window, stride):
        ed = st+window
        chunk = arr[st:ed]
        imu = chunk[:, :6]
        x0,y0 = chunk[0,6], chunk[0,7]
        x1,y1 = chunk[-1,6], chunk[-1,7]
        X_list.append(imu)
        y_list.append([x1-x0, y1-y0])
    if not X_list:
        return None, None
    return np.stack(X_list), np.array(y_list, np.float32)

class IMUOdometryDataset(Dataset):
    def __init__(self, files):
        self.samples=[]
        for fp in files:
            d = read_txt(fp)
            df = syncer.sync(d)  # FIX: syncer đã tạo
            if df is None:
                continue
            X,y = make_windows_odometry(df, WINDOW, STRIDE)
            if X is None:
                continue
            for i in range(len(X)):
                self.samples.append((X[i], y[i]))
        print("IMU samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X,y = self.samples[idx]
        return torch.from_numpy(X), torch.from_numpy(y)

imu_train_ds = IMUOdometryDataset(tr_files[:800])   # tăng dần nếu muốn
imu_val_ds   = IMUOdometryDataset(val_files[:200])

train_loader = DataLoader(imu_train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(imu_val_ds, batch_size=BATCH, shuffle=False)



class GRUOdometry(nn.Module):
    def __init__(self, in_dim=6, hid=128):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hid, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hid, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        feat = out[:, -1, :]
        return self.head(feat)

model = GRUOdometry().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

model



def run_epoch(model, loader, train=True):
    model.train(train)
    total=0.0; n=0
    for xb,yb in loader:
        xb=xb.to(DEVICE); yb=yb.to(DEVICE)
        if train:
            opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        if train:
            loss.backward()
            opt.step()
        total += loss.item()*xb.size(0)
        n += xb.size(0)
    return total/max(n,1)

train_losses=[]; val_losses=[]
for ep in range(1, EPOCHS+1):
    tr = run_epoch(model, train_loader, train=True)
    va = run_epoch(model, val_loader, train=False)
    train_losses.append(tr); val_losses.append(va)
    print(f"Epoch {ep:02d} | train_mse={tr:.6f} | val_mse={va:.6f}")

plt.figure()
plt.plot(train_losses, marker="o", label="train")
plt.plot(val_losses, marker="o", label="val")
plt.title("GRU Odometry Loss (MSE)")
plt.xlabel("epoch"); plt.ylabel("mse")
plt.legend(); plt.show()



@torch.no_grad()
def eval_on_file(fp, max_points=300):
    site, _, _ = parse_train_path(fp)
    d = read_txt(fp)
    df = syncer.sync(d)
    if df is None:
        return None

    if len(df) > max_points:
        df = df.iloc[:max_points].copy()

    gt_xy = df[["x","y"]].values.astype(np.float32)
    ts_arr = df["ts"].values.astype(np.int64)

    Xw, _ = make_windows_odometry(df, WINDOW, STRIDE)
    if Xw is None:
        return None
    dxy = model(torch.from_numpy(Xw).to(DEVICE)).cpu().numpy()

    starts = list(range(0, len(df)-WINDOW, STRIDE))
    pred_gru=[]; pred_knn=[]; pred_fus=[]; gt_end=[]

    wifi_df = d["wifi"]

    for i, st in enumerate(starts):
        ed = st + WINDOW - 1
        x0,y0 = gt_xy[st]
        xg,yg = gt_xy[ed]
        dx,dy = dxy[i]
        x_gru, y_gru = x0+dx, y0+dy

        fl_w, x_w, y_w, conf, best = knn_anchor_predict(site, wifi_df, int(ts_arr[ed]), k=KNN_K, alpha=ALPHA_WIFI)

        a = conf
        x_f = a*x_w + (1-a)*x_gru
        y_f = a*y_w + (1-a)*y_gru

        pred_gru.append([x_gru,y_gru])
        pred_knn.append([x_w,y_w])
        pred_fus.append([x_f,y_f])
        gt_end.append([xg,yg])

    pred_gru=np.array(pred_gru); pred_knn=np.array(pred_knn); pred_fus=np.array(pred_fus); gt_end=np.array(gt_end)

    def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
    return {
        "site": site,
        "rmse_knn": rmse(pred_knn, gt_end),
        "rmse_gru": rmse(pred_gru, gt_end),
        "rmse_fus": rmse(pred_fus, gt_end),
        "pred_knn": pred_knn,
        "pred_gru": pred_gru,
        "pred_fus": pred_fus,
        "gt": gt_end
    }

results=[]
for fp in val_files[:20]:
    out = eval_on_file(fp)
    if out is not None:
        results.append(out)

df_res = pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v, np.ndarray)} for r in results])
print(df_res.describe())



if len(results):
    r = results[0]
    gt = r["gt"]
    pk = r["pred_knn"]
    pg = r["pred_gru"]
    pf = r["pred_fus"]

    plt.figure()
    plt.plot(gt[:,0], gt[:,1], marker="o", label="GT")
    plt.plot(pk[:,0], pk[:,1], marker="x", label="WiFi-kNN")
    plt.plot(pg[:,0], pg[:,1], marker="^", label="GRU")
    plt.plot(pf[:,0], pf[:,1], marker="s", label="Fusion")
    plt.title("Trajectory comparison (window endpoints)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend()
    plt.show()

    print("RMSE:", {"kNN": r["rmse_knn"], "GRU": r["rmse_gru"], "Fusion": r["rmse_fus"]})



# %%
def wifi_fingerprint_window(wifi_df: pd.DataFrame, ts: int, win_ms: int):
    if wifi_df.empty:
        return {}
    w = wifi_df[(wifi_df["ts"] >= ts-win_ms) & (wifi_df["ts"] <= ts+win_ms)]
    if w.empty:
        return {}
    return w.groupby("bssid")["rssi"].mean().to_dict()

def wifi_fingerprint_nearest(wifi_df: pd.DataFrame, ts: int, k_rows=40):
    """Take k WiFi rows closest in time to ts."""
    if wifi_df.empty:
        return {}
    w = wifi_df.copy()
    w["dt"] = (w["ts"] - ts).abs()
    w = w.nsmallest(k_rows, "dt")
    if w.empty:
        return {}
    return w.groupby("bssid")["rssi"].mean().to_dict()



# %%
def knn_anchor_predict_v2(site, wifi_df, ts, win_ms, k=30, alpha=0.4, sigma=0.25):
    """
    Return: floor, x, y, conf, best_dist
    - conf computed from best_dist (Gaussian)
    - xy weighted by Gaussian(dist) for stability
    """
    d = anchor_knn.get(site)
    if d is None:
        return 0, 0.0, 0.0, 0.0, 1.0

    # 1) fingerprint by window
    fpd = wifi_fingerprint_window(wifi_df, ts, win_ms)
    xvec = vectorize_fusion(fpd, d["b2i"], alpha=alpha)

    # 2) if no overlap -> nearest-by-time fallback
    if xvec is None:
        fpd2 = wifi_fingerprint_nearest(wifi_df, ts, k_rows=40)
        xvec = vectorize_fusion(fpd2, d["b2i"], alpha=alpha)

    # 3) final fallback
    if xvec is None:
        # site missing or no overlap
        return 0, 0.0, 0.0, 0.0, 1.0

    k = min(k, d["X"].shape[0])
    dist, idx = d["knn"].kneighbors(xvec, n_neighbors=k, return_distance=True)
    dist = dist[0]; idx = idx[0]

    # Gaussian weights (more stable than inverse-dist)
    w = np.exp(-(dist**2) / (2*sigma*sigma)) + 1e-9

    floors = d["y_floor"][idx]
    xy = d["y_xy"][idx]

    # weighted floor vote
    scores={}
    for f, ww in zip(floors.tolist(), w.tolist()):
        scores[f] = scores.get(f, 0.0) + ww
    floor_pred = int(max(scores.items(), key=lambda x: x[1])[0])

    # xy weighted mean
    x_pred = float(np.sum(xy[:,0]*w) / np.sum(w))
    y_pred = float(np.sum(xy[:,1]*w) / np.sum(w))

    best = float(dist[0])
    conf = float(np.exp(-(best*best) / (2*sigma*sigma)))
    return floor_pred, x_pred, y_pred, conf, best



# %%
def predict_ensemble(site, wifi_df, ts,
                     windows=(1000, 2000, 4000),
                     k=30, alpha=0.4, sigma=0.25):
    """
    Try multiple time windows; pick prediction with highest confidence.
    """
    best = None
    for w in windows:
        fl, x, y, conf, dist = knn_anchor_predict_v2(site, wifi_df, ts, win_ms=w, k=k, alpha=alpha, sigma=sigma)
        cand = (conf, fl, x, y, w, dist)
        if best is None or cand[0] > best[0]:
            best = cand
    # unpack
    conf, fl, x, y, w, dist = best
    return fl, x, y, conf, w, dist



# %%
sample_sub = pd.read_csv(SAMPLE_SUB)
id_col = "site_path_timestamp" if "site_path_timestamp" in sample_sub.columns else sample_sub.columns[0]

def parse_site_path_timestamp(s: str):
    a, ts = s.rsplit("_", 1)
    site, path = a.split("_", 1)
    return site, path, int(ts)

req = sample_sub[id_col].apply(parse_site_path_timestamp)
sample_sub["site"] = req.apply(lambda t: t[0])
sample_sub["path"] = req.apply(lambda t: t[1])
sample_sub["ts"]   = req.apply(lambda t: t[2])

# Map test files by PATH only
test_fp_by_path = {}
for root, _, files in os.walk(TEST_ROOT):
    for fn in files:
        if fn.endswith(".txt"):
            fp = os.path.join(root, fn)
            path = os.path.basename(fp).replace(".txt","")
            test_fp_by_path[path] = fp

missing = (sample_sub["path"].map(lambda p: p not in test_fp_by_path)).sum()
print(f"Missing rows by PATH: {missing}/{len(sample_sub)} = {missing/len(sample_sub)*100:.2f}%")

# Hyperparams to try (start with these)
ENSEMBLE_WINS = (1000, 2000, 4000)
SIGMA = 0.25
K = KNN_K          # use your KNN_K
ALPHA = ALPHA_WIFI # use your ALPHA_WIFI

pred_floor, pred_x, pred_y = [], [], []
conf_list, win_used = [], []
wifi_cache = {}

for i, row in sample_sub.iterrows():
    site, path, ts = row["site"], row["path"], int(row["ts"])
    fp = test_fp_by_path.get(path)
    if fp is None:
        pred_floor.append(0); pred_x.append(0.0); pred_y.append(0.0)
        conf_list.append(0.0); win_used.append(-1)
        continue

    if fp not in wifi_cache:
        d = read_txt(fp)
        wifi_cache[fp] = d["wifi"]
    wifi_df = wifi_cache[fp]

    fl, x, y, conf, w, dist = predict_ensemble(site, wifi_df, ts, windows=ENSEMBLE_WINS, k=K, alpha=ALPHA, sigma=SIGMA)
    pred_floor.append(int(fl)); pred_x.append(float(x)); pred_y.append(float(y))
    conf_list.append(float(conf)); win_used.append(int(w))

    if (i+1) % 5000 == 0:
        print(f"pred {i+1}/{len(sample_sub)}")

submission = pd.DataFrame({id_col: sample_sub[id_col].values, "floor": pred_floor, "x": pred_x, "y": pred_y})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv | rows:", len(submission))
submission.head()



# %%
plt.figure()
plt.hist(pred_floor, bins=30)
plt.title("Predicted floor distribution")
plt.xlabel("floor"); plt.ylabel("count")
plt.show()

plt.figure()
plt.hist(conf_list, bins=40)
plt.title("Confidence distribution")
plt.xlabel("conf"); plt.ylabel("count")
plt.show()

plt.figure()
plt.hist(win_used, bins=10)
plt.title("Chosen window (ms) distribution")
plt.xlabel("win"); plt.ylabel("count")
plt.show()


