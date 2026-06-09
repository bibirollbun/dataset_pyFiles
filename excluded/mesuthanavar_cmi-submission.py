# ==============================================
# CMI BFRB — Submission (Inference Only)
# Stage2+ artefaktlarıyla uyumlu
#  - feature_cols.json/.npy varsa birebir kanal sırası
#  - EMA/SWA/bundle fold ensemble + Gated Softmax
#  - Otomatik mimari + depthwise tespiti (shape-aware)
#  - Konvolüsyon öncesi pad maskesi (kenar sızıntısı yok)
#  - Rerun'da pre-warm yapılmaz (900s limit)
# ==============================================
import os, re, glob, json, numpy as np, pandas as pd, joblib, torch, torch.nn as nn
import polars as pl
from scipy.spatial.transform import Rotation as R
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ---------------------------
# Yollar / Ortam
# ---------------------------
ARTIFACT_DIR = "/kaggle/input/my-model-artifacts"
DATA_DIR     = "/kaggle/input/cmi-detect-behavior-with-sensor-data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IS_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
VERBOSE = not IS_RERUN
def vlog(msg):
    if VERBOSE: print(msg)

# ---------------------------
# Yardımcı: dosya bulucu
# ---------------------------
def find_in_artifacts(candidates, also_subdirs=("stage2_plus_export",)):
    for name in candidates:
        p = os.path.join(ARTIFACT_DIR, name)
        if os.path.exists(p): return p
    for sub in also_subdirs:
        for name in candidates:
            p = os.path.join(ARTIFACT_DIR, sub, name)
            if os.path.exists(p): return p
    return None

def glob_in_artifacts(patterns, also_subdirs=("stage2_plus_export",)):
    found = []
    for pat in patterns:
        found += glob.glob(os.path.join(ARTIFACT_DIR, pat))
    for sub in also_subdirs:
        for pat in patterns:
            found += glob.glob(os.path.join(ARTIFACT_DIR, sub, pat))
    return sorted(found)

# ---------------------------
# Artefaktları yükle
# ---------------------------
classes_path = find_in_artifacts([
    "stage2_plus_classes.npy", "classes.npy",
    "stage3_classes.npy", "stage4_classes.npy", "stage5_classes.npy"
])
if classes_path is None:
    raise FileNotFoundError("Sınıf dosyası (stage2_plus_classes.npy / classes.npy ...) bulunamadı.")
classes = np.load(classes_path, allow_pickle=True)
num_classes = int(len(classes))

fold_stats_path = find_in_artifacts([
    "stage2_plus_fold_stats.pkl", "fold_stats.pkl",
    "stage3_fold_stats.pkl", "stage4_fold_stats.pkl", "stage5_fold_stats.pkl"
])
if fold_stats_path is None:
    raise FileNotFoundError("Fold normalize istatistikleri (mu, sd) dosyası bulunamadı.")
cv_stats = joblib.load(fold_stats_path)
IN_DIM_TARGET = int(len(cv_stats[0][0]))

config_path = find_in_artifacts([
    "stage2_plus_config.json", "config.json",
    "stage3_config.json", "stage4_config.json", "stage5_config.json"
])
config = {}
if config_path:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

# ---------------------------
# feature_cols (NPY/JSON)
# ---------------------------
feature_cols = None
feature_cols_path_npy = find_in_artifacts([
    "stage2_plus_feature_cols.npy", "feature_cols.npy"
])
if feature_cols_path_npy and os.path.exists(feature_cols_path_npy):
    feature_cols = list(np.load(feature_cols_path_npy, allow_pickle=True))
    vlog(f"[INFO] feature_cols.npy bulundu ve kullanılacak. len={len(feature_cols)}")

if feature_cols is None:
    ch = (config.get("channels", {}) if isinstance(config, dict) else {})
    json_name = ch.get("feature_cols_json", None)
    cand_json = []
    if json_name: cand_json.append(json_name)
    cand_json += ["stage2_plus_feature_cols.json", "feature_cols.json"]
    feature_cols_path_json = find_in_artifacts(cand_json)
    if feature_cols_path_json and os.path.exists(feature_cols_path_json):
        with open(feature_cols_path_json, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        vlog(f"[INFO] feature_cols.json bulundu ve kullanılacak. len={len(feature_cols)}")

# ---------------------------
# Hiperparam/Config
# ---------------------------
MAX_LEN = int(config.get("max_len", 127))
hidden  = int(config.get("hyperparams", {}).get("hidden", 128))
layers  = int(config.get("hyperparams", {}).get("layers", 2))
dropout = float(config.get("hyperparams", {}).get("dropout", 0.3))

# ---------------------------
# Gating (A-grubu)
# ---------------------------
DEFAULT_TARGET_LABELS = {
    "Above ear - pull hair","Forehead - pull hairline","Forehead - scratch",
    "Eyebrow - pull hair","Eyelash - pull hair","Neck - pinch skin",
    "Neck - scratch","Cheek - pinch skin"
}
cfg_gate = (config.get("gating", {}) if isinstance(config, dict) else {})
group_A = set(cfg_gate.get("group_A", [])) if isinstance(cfg_gate, dict) else set()
if not group_A:
    group_A = DEFAULT_TARGET_LABELS

maskA = np.array([c in group_A for c in classes], dtype=bool)
GATE_BIAS = float(cfg_gate.get("bias", 0.0)) if isinstance(cfg_gate, dict) else 0.0
GATE_TAU  = float(cfg_gate.get("tau", 1.0))  if isinstance(cfg_gate, dict) else 1.0

def apply_gated_softmax(pm, pb, maskA):
    eps = 1e-9
    logit = np.log(np.clip(pb, eps, 1 - eps)) - np.log(np.clip(1 - pb, eps, 1 - eps))
    pb_adj = 1.0 / (1.0 + np.exp(-(logit + GATE_BIAS) / max(GATE_TAU, 1e-6)))
    pb_adj = pb_adj.reshape(-1, 1)
    out = pm.copy()
    out[:, maskA]  *= pb_adj
    out[:, ~maskA] *= (1.0 - pb_adj)
    out /= (out.sum(axis=1, keepdims=True) + 1e-9)
    return out

# ---------------------------
# FE yardımcıları
# ---------------------------
imu_raw_cols = ['acc_x','acc_y','acc_z','rot_x','rot_y','rot_z','rot_w']
thm_raw_cols = ['thm_1','thm_2','thm_3','thm_4','thm_5']

def _safe_series(x):
    return pd.Series(x).replace([np.inf,-np.inf], np.nan).fillna(0.0).astype(np.float32)

def remove_gravity_from_acc(acc_xyz: np.ndarray, quat_xyzw: np.ndarray):
    g = np.array([0,0,9.81], dtype=np.float32)
    out = np.zeros_like(acc_xyz, dtype=np.float32)
    for i in range(len(acc_xyz)):
        q = quat_xyzw[i]
        if np.any(np.isnan(q)):
            out[i] = acc_xyz[i]; continue
        try:
            Rq = R.from_quat(q)
            g_sens = Rq.apply(g, inverse=True)
            out[i] = acc_xyz[i] - g_sens
        except ValueError:
            out[i] = acc_xyz[i]
    return out

def angular_velocity_from_quat(quat_xyzw: np.ndarray, dt: float = 1/200):
    w = np.zeros((len(quat_xyzw), 3), dtype=np.float32)
    for i in range(len(quat_xyzw)-1):
        q1, q2 = quat_xyzw[i], quat_xyzw[i+1]
        if np.any(np.isnan(q1)) or np.any(np.isnan(q2)): 
            continue
        try:
            r1, r2 = R.from_quat(q1), R.from_quat(q2)
            d = r1.inv() * r2
            w[i] = d.as_rotvec() / dt
        except ValueError:
            pass
    return w

def angular_distance_from_quat(quat_xyzw: np.ndarray):
    ang = np.zeros((len(quat_xyzw),), dtype=np.float32)
    for i in range(len(quat_xyzw)-1):
        q1, q2 = quat_xyzw[i], quat_xyzw[i+1]
        if np.any(np.isnan(q1)) or np.any(np.isnan(q2)): 
            continue
        try:
            r1, r2 = R.from_quat(q1), R.from_quat(q2)
            ang[i] = np.linalg.norm((r1.inv() * r2).as_rotvec())
        except ValueError:
            pass
    return ang

def _rolling_feats(s: pd.Series, win=5):
    m = s.rolling(win, min_periods=1).mean().astype(np.float32)
    sd = s.rolling(win, min_periods=1).std(ddof=0).fillna(0.0).astype(np.float32)
    return m, sd

def get_engineered_fallback_list(target_len: int):
    base = [
        "acc_mag","acc_mag_jerk",
        "rot_angle","rot_angle_vel",
        "linear_acc_x","linear_acc_y","linear_acc_z",
        "linear_acc_mag","linear_acc_mag_jerk",
        "angular_vel_x","angular_vel_y","angular_vel_z",
        "angular_distance",
    ]  # 13
    extra = [
        "acc_mag_ma5","acc_mag_std5",
        "linear_acc_mag_ma5","linear_acc_mag_std5",
        "rot_angle_ma5","rot_angle_std5",
        "acc_x_diff","acc_y_diff","acc_z_diff",
        "linear_acc_x_diff","linear_acc_y_diff","linear_acc_z_diff",
        "ang_speed_mag","ang_speed_jerk","ang_speed_ma5","ang_speed_std5",
        "thm_mean","thm_std","thm_range",
    ]  # +19 = 32
    all_eng = base + extra
    if target_len > len(all_eng):
        raise RuntimeError(f"Fallback engineered {len(all_eng)} < hedef {target_len}. Lütfen feature_cols.json/.npy ekleyin.")
    return all_eng[:target_len]

def compute_features_df(df_seq: pd.DataFrame, want_cols: list | None, max_len: int):
    s = df_seq.copy()
    if 'sequence_counter' in s.columns:
        s = s.sort_values('sequence_counter')
    for c in set(imu_raw_cols + thm_raw_cols):
        if c not in s.columns: s[c] = 0.0

    acc = s[["acc_x","acc_y","acc_z"]].values.astype(np.float32)
    quat = s[["rot_x","rot_y","rot_z","rot_w"]].values.astype(np.float32)

    acc_mag = np.linalg.norm(acc, axis=1).astype(np.float32)
    s["acc_mag"] = acc_mag
    s["acc_mag_jerk"] = _safe_series(acc_mag).diff().fillna(0.0).astype(np.float32)

    rot_w_clip = np.nan_to_num(s["rot_w"].values, nan=0.0)
    rot_angle = (2*np.arccos(np.clip(rot_w_clip, -1, 1))).astype(np.float32)
    s["rot_angle"] = rot_angle
    s["rot_angle_vel"] = _safe_series(rot_angle).diff().fillna(0.0).astype(np.float32)

    lin = remove_gravity_from_acc(acc, quat)
    s["linear_acc_x"] = lin[:,0]; s["linear_acc_y"] = lin[:,1]; s["linear_acc_z"] = lin[:,2]
    lin_mag = np.linalg.norm(lin, axis=1).astype(np.float32)
    s["linear_acc_mag"] = lin_mag
    s["linear_acc_mag_jerk"] = _safe_series(lin_mag).diff().fillna(0.0).astype(np.float32)

    ang_vel = angular_velocity_from_quat(quat)
    s["angular_vel_x"] = ang_vel[:,0]; s["angular_vel_y"] = ang_vel[:,1]; s["angular_vel_z"] = ang_vel[:,2]
    s["angular_distance"] = angular_distance_from_quat(quat)

    s["acc_mag_ma5"], s["acc_mag_std5"] = _rolling_feats(pd.Series(acc_mag), win=5)
    s["linear_acc_mag_ma5"], s["linear_acc_mag_std5"] = _rolling_feats(pd.Series(lin_mag), win=5)
    s["rot_angle_ma5"], s["rot_angle_std5"] = _rolling_feats(pd.Series(rot_angle), win=5)

    s["acc_x_diff"] = _safe_series(s["acc_x"]).diff().fillna(0.0).astype(np.float32)
    s["acc_y_diff"] = _safe_series(s["acc_y"]).diff().fillna(0.0).astype(np.float32)
    s["acc_z_diff"] = _safe_series(s["acc_z"]).diff().fillna(0.0).astype(np.float32)
    s["linear_acc_x_diff"] = _safe_series(s["linear_acc_x"]).diff().fillna(0.0).astype(np.float32)
    s["linear_acc_y_diff"] = _safe_series(s["linear_acc_y"]).diff().fillna(0.0).astype(np.float32)
    s["linear_acc_z_diff"] = _safe_series(s["linear_acc_z"]).diff().fillna(0.0).astype(np.float32)

    ang_speed = np.linalg.norm(ang_vel, axis=1).astype(np.float32)
    s["ang_speed_mag"] = ang_speed
    s["ang_speed_jerk"] = _safe_series(ang_speed).diff().fillna(0.0).astype(np.float32)
    s["ang_speed_ma5"], s["ang_speed_std5"] = _rolling_feats(pd.Series(ang_speed), win=5)

    thm_mat = s[thm_raw_cols].astype(np.float32).values
    s["thm_mean"]  = np.nanmean(thm_mat, axis=1).astype(np.float32)
    s["thm_std"]   = np.nanstd(thm_mat,  axis=1).astype(np.float32)
    s["thm_range"] = (np.nanmax(thm_mat, axis=1) - np.nanmin(thm_mat, axis=1)).astype(np.float32)

    if want_cols is not None and len(want_cols) > 0:
        cols = []
        for c in want_cols:
            if c not in s.columns: s[c] = 0.0
            cols.append(c)
        all_cols = cols
    else:
        need_eng = IN_DIM_TARGET - len(imu_raw_cols) - len(thm_raw_cols)
        if need_eng <= 0:
            raise RuntimeError("IN_DIM_TARGET IMU+THM'den küçük/eşit görünüyor. Artefaktlar hatalı olabilir.")
        engineered_cols = get_engineered_fallback_list(need_eng)
        all_cols = imu_raw_cols + thm_raw_cols + engineered_cols
        vlog(f"[INFO] feature_cols yok. Fallback FE seti kullanılıyor. (in_dim hedef={IN_DIM_TARGET}, engineered={len(engineered_cols)})")

    X = s[all_cols].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    L, C = X.shape
    T = min(L, max_len)
    x = np.zeros((max_len, C), dtype=np.float32)
    m = np.zeros((max_len,), dtype=np.float32)
    x[:T] = X[:T]
    m[:T] = 1.0
    return x, m, all_cols

# ---------------------------
# Model mimari parçaları
# ---------------------------
class Rearrange(nn.Module):
    def __init__(self, pat): super().__init__(); self.pat = pat
    def forward(self, x):
        if self.pat == "b t c -> b c t": return x.permute(0,2,1).contiguous()
        if self.pat == "b c t -> b t c": return x.permute(0,2,1).contiguous()
        raise ValueError("pattern?")

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, s=1, p=2, dw=False):
        super().__init__()
        if dw:
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),  # depthwise
                nn.Conv1d(in_ch, out_ch, 1, bias=False),                     # pointwise
                nn.BatchNorm1d(out_ch), nn.GELU()
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, s, p, bias=False),
                nn.BatchNorm1d(out_ch), nn.GELU()
            )
        self.res_proj = (in_ch != out_ch) or (s != 1)
        if self.res_proj:
            self.proj = nn.Conv1d(in_ch, out_ch, 1, stride=s, bias=False)
    def forward(self, x):
        y = self.net(x)
        if self.res_proj: x = self.proj(x)
        return y + x

def lengths_from_mask(mask_t: torch.Tensor):
    return mask_t.sum(dim=1).long().cpu()

# --- Yeni mimari (layernorm_in ayrık, conv1 dw=False, conv2 dw=True)
class BiLSTM_Multitask_New(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.3, num_classes=18, conv_dim=128):
        super().__init__()
        self.layernorm_in = nn.LayerNorm(input_dim)
        self.fe = nn.Sequential(
            Rearrange("b t c -> b c t"),
            ConvBlock(input_dim, 128, k=5, p=2, dw=False),
            ConvBlock(128, 128, k=5, p=2, dw=True),
            nn.Conv1d(128, conv_dim, 1, bias=False),
            nn.BatchNorm1d(conv_dim), nn.GELU(),
            Rearrange("b c t -> b t c"),
        )
        self.lstm = nn.LSTM(input_size=conv_dim, hidden_size=hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=(dropout if layers > 1 else 0.0))
        self.attn = nn.Sequential(nn.Linear(hidden*2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.norm = nn.LayerNorm(hidden*2)
        self.drop = nn.Dropout(dropout)
        self.head_multi = nn.Linear(hidden*2, num_classes)
        self.head_bin   = nn.Linear(hidden*2, 1)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = self.layernorm_in(x)
        x = self.fe(x)
        if mask is not None:
            lengths = lengths_from_mask(mask)
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
            scores = self.attn(out).squeeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            feat = (out * w).sum(dim=1)
        else:
            out, _ = self.lstm(x); feat = out.mean(dim=1)
        feat = self.drop(self.norm(feat))
        return self.head_multi(feat), self.head_bin(feat).squeeze(1)

# --- Legacy mimari (LN fe.0 içinde; conv1/conv2 dw bayrakları şekilden tespit edilir)
class BiLSTM_Multitask_Legacy(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.3, num_classes=18,
                 conv1_dw=False, conv2_dw=False, conv_dim=128):
        super().__init__()
        self.fe = nn.Sequential(
            nn.LayerNorm(input_dim),                    # fe.0
            Rearrange("b t c -> b c t"),               # fe.1
            ConvBlock(input_dim, 128, k=5, p=2, dw=conv1_dw),  # fe.2
            ConvBlock(128, 128, k=5, p=2, dw=conv2_dw),        # fe.3
            nn.Conv1d(128, conv_dim, 1, bias=False),   # fe.4
            nn.BatchNorm1d(conv_dim), nn.GELU(),       # fe.5
            Rearrange("b c t -> b t c"),               # fe.6
        )
        self.lstm = nn.LSTM(input_size=conv_dim, hidden_size=hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=(dropout if layers > 1 else 0.0))
        self.attn = nn.Sequential(nn.Linear(hidden*2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.norm = nn.LayerNorm(hidden*2)
        self.drop = nn.Dropout(dropout)
        self.head_multi = nn.Linear(hidden*2, num_classes)
        self.head_bin   = nn.Linear(hidden*2, 1)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)                 # konvolüsyon öncesi pad sıfırlama
        x = self.fe(x)
        if mask is not None:
            lengths = lengths_from_mask(mask)
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
            scores = self.attn(out).squeeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            feat = (out * w).sum(dim=1)
        else:
            out, _ = self.lstm(x); feat = out.mean(dim=1)
        feat = self.drop(self.norm(feat))
        return self.head_multi(feat), self.head_bin(feat).squeeze(1)

# ---------------------------
# Ağırlık yükleme (auto-arch + shape-aware)
# ---------------------------
def parse_fold_id(path: str):
    m = re.search(r"fold(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None

def _extract_state_dict(sd):
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], (dict,)):
            return sd["state_dict"]
        keys = list(sd.keys())
        if keys and isinstance(keys[0], str) and (keys[0].startswith("fe.") or "lstm.weight_ih_l0" in keys or "head_multi.weight" in keys or "layernorm_in.weight" in keys):
            return sd
    return sd

def _has(sd, k): return (k in sd)
def _shape(sd, k): return tuple(sd[k].shape) if k in sd else None

def _detect_arch_params(sd, in_dim):
    keys = sd.keys()
    new_style = any(k.startswith("layernorm_in.") for k in keys)
    legacy_ln = any(k == "fe.0.weight" for k in keys) and (_shape(sd, "fe.0.weight") == (in_dim,))
    if new_style and not legacy_ln:
        # New: conv1 dw? (beklenen False), conv2 dw? (beklenen True), conv_dim from fe.3 or fe.4?
        conv1_dw = (_shape(sd, "fe.1.net.0.weight") is not None and _shape(sd, "fe.1.net.0.weight")[1] == 1)
        conv2_dw = (_shape(sd, "fe.2.net.0.weight") is not None and _shape(sd, "fe.2.net.0.weight")[1] == 1)
        # conv1x1 is fe.3.weight likely
        conv1x1_shape = _shape(sd, "fe.3.weight")
        conv_dim = conv1x1_shape[0] if conv1x1_shape else 128
        return ("new", conv1_dw, conv2_dw, conv_dim)
    else:
        # Legacy: LN fe.0 içinde; convblocklar fe.2 ve fe.3; 1x1 conv fe.4
        conv1_dw = (_shape(sd, "fe.2.net.0.weight") is not None and _shape(sd, "fe.2.net.0.weight")[1] == 1)
        conv2_dw = (_shape(sd, "fe.3.net.0.weight") is not None and _shape(sd, "fe.3.net.0.weight")[1] == 1)
        conv1x1_shape = _shape(sd, "fe.4.weight")
        conv_dim = conv1x1_shape[0] if conv1x1_shape else 128
        return ("legacy", conv1_dw, conv2_dw, conv_dim)

def _build_single_model_for_state(sd, in_dim, hidden, layers, dropout, num_classes):
    arch, conv1_dw, conv2_dw, conv_dim = _detect_arch_params(sd, in_dim)
    if arch == "new":
        m = BiLSTM_Multitask_New(in_dim, hidden=hidden, layers=layers, dropout=dropout, num_classes=num_classes, conv_dim=conv_dim)
    else:
        m = BiLSTM_Multitask_Legacy(in_dim, hidden=hidden, layers=layers, dropout=dropout, num_classes=num_classes,
                                    conv1_dw=conv1_dw, conv2_dw=conv2_dw, conv_dim=conv_dim)
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()
    return m, arch, conv1_dw, conv2_dw, conv_dim

def _build_models_for_fold(weight_paths, in_dim, hidden, layers, dropout, num_classes):
    models = []
    for wp in weight_paths:
        raw = torch.load(wp, map_location="cpu")
        sd = _extract_state_dict(raw)
        m, arch, d1, d2, cd = _build_single_model_for_state(sd, in_dim, hidden, layers, dropout, num_classes)
        models.append(m)
        if VERBOSE:
            print(f"[INFO] {os.path.basename(wp)} -> arch={arch}, conv1_dw={d1}, conv2_dw={d2}, conv_dim={cd}")
    return models

class Ensemble:
    def __init__(self, in_dim, hidden, layers, dropout, num_classes):
        weight_patterns = [
            "stage2_plus_ema_fold*.pt", "stage2_plus_swa_fold*.pt",
            "stage2_plus_fold*_bundle.pt",
            "stage3_fold*.pt", "stage4_fold*.pt", "stage5_fold*.pt",
            "model_fold*.pt"
        ]
        paths = glob_in_artifacts(weight_patterns)
        if not paths:
            raise FileNotFoundError("Fold ağırlıkları (*.pt) bulunamadı.")

        fold_map = {}
        for p in paths:
            f = parse_fold_id(p)
            if f is None: 
                continue
            fold_map.setdefault(f, []).append(p)

        self.folds = []
        total_models = 0
        for f in sorted(fold_map.keys()):
            order = sorted(fold_map[f], key=lambda x: (("ema" not in x), x))
            ms = _build_models_for_fold(order, in_dim, hidden, layers, dropout, num_classes)
            self.folds.append((f, ms))
            total_models += len(ms)
        vlog(f"[INFO] Fold sayısı: {len(self.folds)} (EMA/SWA/bundle birlikte {total_models} model)")

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, m: np.ndarray, mu: np.ndarray, sd: np.ndarray):
        # Normalizasyon + PAD'leri sıfırla (ek güvenlik)
        xn = (x - mu.reshape(1, -1)) / (sd.reshape(1, -1) + 1e-6)
        xn *= m.reshape(-1, 1)

        xb = torch.tensor(xn[None, ...], dtype=torch.float32, device=device)
        mb = torch.tensor(m[None,  ...], dtype=torch.float32, device=device)
        probs_folds = []
        for f, ms in self.folds:
            probs_models = []
            for model in ms:
                log_m, log_b = model(xb, mask=mb)
                pm = torch.softmax(log_m, dim=1).cpu().numpy()
                pb = torch.sigmoid(log_b).cpu().numpy()
                pm_g = apply_gated_softmax(pm, pb, maskA)
                probs_models.append(pm_g[0])
            probs_folds.append(np.mean(probs_models, axis=0))
        return np.mean(probs_folds, axis=0)

# ---------------------------
# Tek sequence tahmini
# ---------------------------
_models_cache = None
_feature_cols_used = None

def _ensure_models(feature_cols_runtime: list, cv_stats):
    global _models_cache, _feature_cols_used
    if _models_cache is not None:
        return _models_cache

    first_mu, first_sd = cv_stats[0]
    in_dim = int(len(first_mu))
    for (mu, sd) in cv_stats:
        assert len(mu) == in_dim and len(sd) == in_dim, "cv_stats boyutları tutarsız!"

    if len(feature_cols_runtime) != in_dim:
        raise AssertionError(
            f"FE kanal sayısı ({len(feature_cols_runtime)}) ile cv_stats in_dim ({in_dim}) eşit değil.\n"
            "-> Çözüm: Eğitimde 'feature_cols.json/.npy' kaydet ve artefakta koy; "
            "veya fallback FE setinin in_dim hedefiyle üretildiğini doğrula."
        )

    ens = Ensemble(in_dim=in_dim, hidden=hidden, layers=layers, dropout=dropout, num_classes=num_classes)
    _models_cache = ens
    _feature_cols_used = feature_cols_runtime
    vlog(f"[INFO] in_dim={in_dim} | MAX_LEN={MAX_LEN} | hidden={hidden} | layers={layers} | dropout={dropout}")
    return ens

@torch.no_grad()
def _predict_one(sequence_df: pd.DataFrame) -> np.ndarray:
    x, m, cols = compute_features_df(sequence_df, feature_cols, MAX_LEN)
    ens = _ensure_models(cols, cv_stats)
    probs_by_fold = []
    for i in range(len(cv_stats)):
        mu, sd = cv_stats[i]
        mu = np.asarray(mu, dtype=np.float32)
        sd = np.asarray(sd, dtype=np.float32)
        probs_by_fold.append(ens.predict_proba(x, m, mu, sd))
    return np.mean(probs_by_fold, axis=0)

# ---------------------------
# Kaggle Inference Server imzası
# ---------------------------
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df = sequence.to_pandas()
    p = _predict_one(df)
    idx = int(np.argmax(p))
    return str(classes[idx])

vlog("[INFO] predict() tanımlandı ve hazır.")

# ---------------------------
# CMI Inference Server entegrasyonu
# ---------------------------
from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

# Pre-warm SADECE lokal gateway'de (rerun'da değil)
if not IS_RERUN:
    try:
        dummy = pd.DataFrame({c: [0.0] for c in (imu_raw_cols + thm_raw_cols)})
        x, m, cols = compute_features_df(dummy, feature_cols, MAX_LEN)
        _ensure_models(cols, cv_stats)
    except Exception as e:
        vlog("[WARN] Ön ısıtma uyarısı: " + repr(e))

inference_server = CMIInferenceServer(predict)

if IS_RERUN:
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            os.path.join(DATA_DIR, "test.csv"),
            os.path.join(DATA_DIR, "test_demographics.csv"),
        )
    )
    try:
        print(pd.read_parquet("submission.parquet").head())
    except Exception as e:
        vlog("[WARN] submission.parquet okunamadı: " + repr(e))





