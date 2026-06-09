# ===========================
# Distracted Driving — SOTA-ish Tabular Baseline (2023–2025 style)
# + Pseudo-Label Upgrade: Consensus + Per-class Threshold + TTA Stability
# + (ADDED) Mean-Teacher FixMatch + Distribution Alignment for unlabeled
# (UNDERFIT→FIT): widen models, relax regularization, stronger SSL & PL
# + (ADDED) More models for SOTA-style ensemble: XGB-DART, LGB-DART, HGB, ExtraTrees, Sup-MLP
# + (ADDED) Time-aware features: hour_sin, hour_cos, hour_sin_lowvis, hour_cos_precip
# ===========================
import numpy as np, pandas as pd, warnings, os, gc
warnings.filterwarnings("ignore")

# !pip -q install lightgbm catboost  # (nếu cần)

import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False

# ---- Torch (for SSL Mean-Teacher & optional supervised MLP) ----
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

RANDOM_STATE = int(os.environ.get("SEED", "42"))
np.random.seed(RANDOM_STATE)

# ---------------------------
# 1) Load data
# ---------------------------
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input/distracted-driving-risk-detection-challenge")
train_df = pd.read_csv(f"{DATA_DIR}/kaggle_train.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/kaggle_test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/kaggle_sample_submission.csv")
print("Shapes:", train_df.shape, test_df.shape)

train_df = train_df.drop(columns=[c for c in ['label_source'] if c in train_df.columns])
test_df  = test_df.drop(columns=[c for c in ['label_source'] if c in test_df.columns])

# ---------------------------
# 2) Feature Engineering
# ---------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------------- Time-aware (NEW) ----------------
    # encode 24h cyclic and simple interactions (no new external columns required)
    hour = pd.to_numeric(df['observation_hour'], errors='coerce').fillna(0)
    df['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0)

    # Speed violation
    df['speed_violation_ratio'] = df['speed'] / (df['design_speed'] + 1e-3)
    df['is_speeding'] = (df['speed_violation_ratio'] > 1.0).astype(int)

    # Aggressive driving
    df['aggressive_driving'] = df['acceleration'] * df['throttle_position']
    df['hard_braking'] = (df['acceleration'] < -3).astype(int)
    df['rapid_acceleration'] = (df['acceleration'] > 3).astype(int)

    # Engine / efficiency
    df['engine_stress'] = df['rpm'] * df['engine_load_value'] / 100.0
    df['engine_efficiency'] = df['speed'] / (df['rpm'] + 1.0)
    df['speed_rpm_ratio'] = df['speed'] / (df['rpm'] + 1.0) * 1000.0
    df['throttle_load_diff'] = df['throttle_position'] - df['engine_load_value']
    df['speed_squared'] = df['speed'] ** 2

    # Environment & location
    df['weather_risk'] = df['current_weather'] * (1.0 / (df['visibility'] + 1.0)) * (df['precipitation'] + 1.0)
    df['location_risk'] = df['accidents_onsite'] + df['accidents_time']
    df['weather_visibility_ratio'] = df['current_weather'] / (df['visibility'] + 0.1)
    df['visibility_precipitation_interaction'] = df['visibility'] * df['precipitation']
    df['speed_weather_interaction'] = df['speed'] * df['current_weather']

    # Time-based flags (context)
    df['is_rush_hour'] = df['observation_hour'].isin([7,8,9,17,18,19]).astype(int)
    df['is_night'] = ((df['observation_hour'] >= 20) | (df['observation_hour'] <= 5)).astype(int)

    # Driver physiology
    hr_mean, hr_std = df['heart_rate'].mean(), df['heart_rate'].std(ddof=0)
    df['heart_rate_zscore'] = (df['heart_rate'] - hr_mean) / (hr_std + 1e-6)
    df['heart_rate_bin'] = pd.cut(df['heart_rate'], bins=[0, 70, 90, 110, 200], labels=False, include_lowest=True)
    df['heart_rate_bin'] = df['heart_rate_bin'].fillna(0).astype(int) + 1
    df['driver_stress'] = (df['heart_rate_zscore'] > 1.0).astype(int)

    # Flags
    df['engine_temp_normal'] = ((df['engine_temperature'] >= 80) & (df['engine_temperature'] <= 105)).astype(int)
    df['high_rpm'] = (df['rpm'] > 3000).astype(int)
    df['low_visibility'] = (df['visibility'] < 5).astype(int)
    df['heavy_precipitation'] = (df['precipitation'] > 10).astype(int)
    df['dangerous_location'] = (df['accidents_onsite'] > 50).astype(int)

    # Combined risk
    df['total_risk_score'] = df['speed_violation_ratio'] + df['weather_risk'] + df['location_risk']
    df['speed_acceleration_product'] = df['speed'] * np.abs(df['acceleration'])

    # Bins
    df['speed_bin'] = pd.cut(df['speed'], bins=[-0.001, 30, 60, 90, 200], labels=False, include_lowest=True)
    df['speed_bin'] = df['speed_bin'].fillna(0).astype(int) + 1

    # Interactions (existing)
    df['night_lowvis']    = (df['is_night'] & df['low_visibility']).astype(int)
    df['rush_lowvis']     = (df['is_rush_hour'] & df['low_visibility']).astype(int)
    df['speeding_lowvis'] = (df['is_speeding'] & df['low_visibility']).astype(int)

    # ---- NEW interactions with cyclical hour ----
    df['hour_sin_lowvis'] = df['hour_sin'] * df['low_visibility']
    df['hour_cos_precip'] = df['hour_cos'] * df['precipitation']

    return df

train_feat = create_features(train_df)
test_feat  = create_features(test_df)

feature_cols = [c for c in train_feat.columns if c != 'risk_level']
X = train_feat[feature_cols].replace([np.inf, -np.inf], np.nan)
X_test = test_feat[feature_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X.median(numeric_only=True))

y = (train_feat['risk_level'].astype(int) - 1)  # 0..3
num_class = 4
print("Num features:", X.shape[1])

scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X.values)
Xtest_scaled = scaler.transform(X_test.values)

# ---------------------------
# 2.5) Adversarial validation → sample_weight
# ---------------------------
from sklearn.linear_model import LogisticRegression
adv_y = np.concatenate([np.zeros(len(X), dtype=int), np.ones(len(X_test), dtype=int)])
adv_X = np.vstack([X_scaled, Xtest_scaled])

adv_clf = LogisticRegression(max_iter=200, n_jobs=None, random_state=RANDOM_STATE)
adv_clf.fit(adv_X, adv_y)
p_test_like = adv_clf.predict_proba(X_scaled)[:,1]
w_domain = 0.5 + p_test_like
w_class = compute_sample_weight(class_weight='balanced', y=y.values)
sample_w = w_class * w_domain

# ---------------------------
# 3) CV config
# ---------------------------
CV_MODE = os.environ.get("CV_MODE", "stratified").lower()  # stratified|group|time
N_SPLITS = int(os.environ.get("N_SPLITS","5"))

groups = None
if CV_MODE == "group":
    if "driver_id" in train_df.columns:
        groups = train_df["driver_id"].values
    else:
        print("CV_MODE=group nhưng không có driver_id, fallback Stratified.")
        CV_MODE = "stratified"

def make_splitter():
    if CV_MODE == "time":
        return TimeSeriesSplit(n_splits=N_SPLITS)
    elif CV_MODE == "group":
        return GroupKFold(n_splits=N_SPLITS)
    else:
        return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
splitter = make_splitter()

# ============================================================
# 4) OOF + Base + Extra models (dynamic registry, giữ nguyên pipeline)
# ============================================================
# ==== 4.1 Params cho từng model ====
# ↓↓↓ CHỈNH ĐỂ BỚT UNDERFIT (tăng capacity, nới reg) ↓↓↓
xgb_params = dict(
    objective='multi:softprob', num_class=num_class, eval_metric='mlogloss',
    n_estimators=20000, learning_rate=0.05, max_depth=10,
    subsample=0.95, colsample_bytree=0.95, gamma=0.0,
    min_child_weight=1.0, reg_alpha=0.0, reg_lambda=0.0,
    random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist'
)

# XGB-DART (tăng diversity) — FIX: tránh duplicate keys trong dict()
_xgb_base_for_dart = {k: v for k, v in xgb_params.items()
                      if k not in ['gamma', 'min_child_weight', 'booster', 'eval_metric']}
xgb_dart_params = {
    **_xgb_base_for_dart,
    'booster': 'dart',
    'rate_drop': 0.1,
    'skip_drop': 0.0,
    'sample_type': 'uniform',
    'normalize_type': 'tree',
    'eval_metric': 'mlogloss',
}

lgb_params = dict(
    objective='multiclass', num_class=num_class, metric='multi_logloss',
    n_estimators=30000, learning_rate=0.05, num_leaves=127,
    feature_fraction=0.95, bagging_fraction=0.95, bagging_freq=1,
    min_data_in_leaf=20, lambda_l1=0.0, lambda_l2=0.0,
    min_gain_to_split=0.0, max_depth=-1,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
)
# LGB-DART
lgb_dart_params = dict(
    **lgb_params, boosting_type='dart', drop_rate=0.1, xgboost_dart_mode=False
)

cat_params = dict(
    loss_function='MultiClass', iterations=30000, learning_rate=0.05,
    depth=10, l2_leaf_reg=2.0, random_strength=0.5, random_seed=RANDOM_STATE,
    od_type='Iter', od_wait=600, verbose=False, allow_writing_files=False
) if HAVE_CAT else None

# Sklearn tree-based
hgb_params = dict(            # HistGradientBoosting
    loss='log_loss',
    max_depth=None,
    max_leaf_nodes=63,
    learning_rate=0.05,
    max_iter=300,
    l2_regularization=0.0,
    random_state=RANDOM_STATE
)
et_params = dict(             # ExtraTreesClassifier
    n_estimators=600,
    max_depth=None,
    max_features='sqrt',
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=False,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# (Optional) MLP supervised nhỏ nếu có Torch — khác bias với tree
USE_SUP_MLP = HAVE_TORCH

if USE_SUP_MLP:
    class SupTabMLP(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, d_out),
            )
        def forward(self, x): return self.net(x)

    def fit_sup_mlp(Xtr, ytr, Xva, yva, epochs=12, lr=1e-3, wd=1e-5, bs=1024):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SupTabMLP(Xtr.shape[1], num_class).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        Xtr_t = torch.from_numpy(Xtr.astype(np.float32)); ytr_t = torch.from_numpy(ytr.astype(np.int64))
        Xva_t = torch.from_numpy(Xva.astype(np.float32)); yva_t = torch.from_numpy(yva.astype(np.int64))
        dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=bs, shuffle=True)
        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p_va = F.softmax(model(Xva_t.to(device)), dim=1).cpu().numpy()
        return model, p_va

    def predict_sup_mlp(model, X):
        device = next(model.parameters()).device
        with torch.no_grad():
            p = F.softmax(model(torch.from_numpy(X.astype(np.float32)).to(device)), dim=1).cpu().numpy()
        return p

# ==== 4.2 Đăng ký model để train OOF ====
MODEL_REGISTRY = []
MODEL_REGISTRY.append(('xgb', 'xgb', xgb_params, lambda p: xgb.XGBClassifier(**p), lambda m, X: m.predict_proba(X), True))
MODEL_REGISTRY.append(('xgb_dart', 'xgb', xgb_dart_params, lambda p: xgb.XGBClassifier(**p), lambda m, X: m.predict_proba(X), True))
MODEL_REGISTRY.append(('lgb', 'lgb', lgb_params, lambda p: lgb.LGBMClassifier(**p), lambda m, X: m.predict_proba(X), True))
MODEL_REGISTRY.append(('lgb_dart', 'lgb', lgb_dart_params, lambda p: lgb.LGBMClassifier(**p), lambda m, X: m.predict_proba(X), True))
if HAVE_CAT:
    MODEL_REGISTRY.append(('cat', 'cat', cat_params, lambda p: CatBoostClassifier(**p), lambda m, X: m.predict_proba(X), True))
MODEL_REGISTRY.append(('hgb', 'sk', hgb_params, lambda p: HistGradientBoostingClassifier(**p), lambda m, X: m.predict_proba(X), False))
MODEL_REGISTRY.append(('extratrees', 'sk', et_params, lambda p: ExtraTreesClassifier(**p), lambda m, X: m.predict_proba(X), False))
if HAVE_TORCH:
    MODEL_REGISTRY.append(('mlp', 'torch', dict(epochs=12, lr=1e-3, wd=1e-5, bs=1024), None, None, False))

X_np, y_np = X_scaled, y.values
Xtest_np = Xtest_scaled

OOF_DICT = {name: np.zeros((len(X_np), num_class), dtype=float) for (name, *_ ) in MODEL_REGISTRY}
TEST_DICT = {name: np.zeros((len(X_test), num_class), dtype=float) for (name, *_ ) in MODEL_REGISTRY}
BEST_ITERS = {name: [] for (name, *_ ) in MODEL_REGISTRY}

if CV_MODE == "group":
    folds_enum = enumerate(splitter.split(X_np, y_np, groups=groups), 1)
elif CV_MODE == "time":
    folds_enum = enumerate(splitter.split(X_np), 1)
else:
    folds_enum = enumerate(splitter.split(X_np, y_np), 1)

MLP_MODELS = [] if HAVE_TORCH else None

for fold, (tr, va) in folds_enum:
    print(f"\nFOLD {fold}")
    Xtr, Xva = X_np[tr], X_np[va]
    ytr, yva = y_np[tr], y_np[va]
    wtr = sample_w[tr]

    for (name, kind, params, make_fn, pred_fn, earlystop) in MODEL_REGISTRY:
        print(f"  -> Training {name}")
        if name == 'mlp':
            model, p_va = fit_sup_mlp(Xtr, ytr, Xva, yva, **params)
            OOF_DICT[name][va] = p_va
            TEST_DICT[name] += predict_sup_mlp(model, Xtest_np)/N_SPLITS
            BEST_ITERS[name].append(0)
            MLP_MODELS.append(model)
            continue

        model = make_fn(params)
        if kind == 'xgb':
            model.fit(Xtr, ytr, sample_weight=wtr,
                      eval_set=[(Xva, yva)], early_stopping_rounds=400, verbose=False)
            BEST_ITERS[name].append(getattr(model, "best_iteration", params.get('n_estimators', 0)) or 0)
        elif kind == 'lgb':
            model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)],
                      callbacks=[lgb.early_stopping(stopping_rounds=600, verbose=False),
                                 lgb.log_evaluation(period=0)])
            BEST_ITERS[name].append(getattr(model, "best_iteration_", params.get('n_estimators', 0)) or 0)
        else:
            if 'sample_weight' in model.fit.__code__.co_varnames:
                model.fit(Xtr, ytr, sample_weight=wtr)
            else:
                model.fit(Xtr, ytr)
            if name == 'cat':
                BEST_ITERS[name].append(int(model.tree_count_))
            else:
                BEST_ITERS[name].append(0)

        OOF_DICT[name][va] = pred_fn(model, Xva)
        TEST_DICT[name] += pred_fn(model, Xtest_np)/N_SPLITS

def acc_of(proba): return accuracy_score(y_np, proba.argmax(1))
for name in OOF_DICT:
    print(f"OOF Acc {name.upper()}: {acc_of(OOF_DICT[name])::.4f}")

# ---------------------------
# 5) Temperature scaling (global) — dùng tổng logits OOF của tất cả model
# ---------------------------
def logits_of(proba, eps=1e-9): return np.log(np.clip(proba, eps, 1.0))

base_logits_oof = 0
base_logits_test = 0
for name in OOF_DICT:
    base_logits_oof += logits_of(OOF_DICT[name])
    base_logits_test += logits_of(TEST_DICT[name])

def fit_temperature(logits, y_true):
    T = 1.0
    for _ in range(50):
        cands = np.clip(np.array([T*0.8, T*0.9, T, T*1.1, T*1.25]), 0.05, 10.0)
        losses = []
        for t in cands:
            p = (logits / t)
            p = np.exp(p - p.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)
            losses.append(log_loss(y_true, p, labels=np.arange(num_class)))
        T = cands[np.argmin(losses)]
    return float(T)

T_global = fit_temperature(base_logits_oof, y_np)

def apply_temp(logits, T):
    p = (logits / T)
    p = np.exp(p - p.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)

proba_oof_cal = apply_temp(base_logits_oof, T_global)
proba_test_cal = apply_temp(base_logits_test, T_global)
oof_ens_acc = accuracy_score(y_np, proba_oof_cal.argmax(1))
print(f"OOF Acc (Ensemble+TempCal ALL, T={T_global:.2f}): {oof_ens_acc:.4f}")

# ---------------------------
# 6) Meta-learner Stacking (LogReg) — từ logits của TẤT CẢ model
# ---------------------------
X_stack_oof = np.hstack([logits_of(OOF_DICT[name]) for name in OOF_DICT])
X_stack_test = np.hstack([logits_of(TEST_DICT[name]) for name in TEST_DICT])

meta = LogisticRegression(multi_class="multinomial", max_iter=700, C=5.0,
                          class_weight='balanced', n_jobs=None, random_state=RANDOM_STATE)
meta.fit(X_stack_oof, y_np)
oof_meta = meta.predict_proba(X_stack_oof)
test_meta = meta.predict_proba(X_stack_test)
print("OOF Acc (Stacking Meta ALL):", round(accuracy_score(y_np, oof_meta.argmax(1)),4))

oof_best = oof_meta if accuracy_score(y_np, oof_meta.argmax(1)) >= oof_ens_acc else proba_oof_cal
test_best = test_meta if oof_best is oof_meta else proba_test_cal

# ---------------------------
# 7) Train full models @ mean best iters cho model có n_estimators
# ---------------------------
FULLS = {}
for (name, kind, params, make_fn, pred_fn, earlystop) in MODEL_REGISTRY:
    print(f"Train full: {name}")
    if name == 'mlp':
        if not HAVE_TORCH:
            continue
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SupTabMLP(X_np.shape[1], num_class).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        X_t = torch.from_numpy(X_np.astype(np.float32)); y_t = torch.from_numpy(y_np.astype(np.int64))
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=1024, shuffle=True)
        model.train()
        for _ in range(15):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        FULLS[name] = model
        continue

    if kind in ['xgb','lgb'] and len(BEST_ITERS[name]) > 0 and params.get('n_estimators', 0) > 0:
        avg_iter = int(np.mean(BEST_ITERS[name]))
        model = make_fn({**params, 'n_estimators': max(avg_iter, 50)})
    elif name == 'cat' and len(BEST_ITERS[name]) > 0 and params.get('iterations', 0) > 0:
        avg_iter = int(np.mean(BEST_ITERS[name]))
        model = make_fn({**params, 'iterations': max(avg_iter, 50)})
    else:
        model = make_fn(params)

    if name in ['xgb','xgb_dart']:
        model.fit(X_np, y_np, sample_weight=sample_w, verbose=False)
    elif name in ['lgb','lgb_dart']:
        model.fit(X_np, y_np, sample_weight=sample_w, callbacks=[lgb.log_evaluation(period=0)])
    else:
        if 'sample_weight' in model.fit.__code__.co_varnames:
            model.fit(X_np, y_np, sample_weight=sample_w)
        else:
            model.fit(X_np, y_np)
    FULLS[name] = model

# ---------------------------
# Full ensemble (calibrated) — cộng logits của tất cả FULLS (+ SSL nếu có)
# ---------------------------
def predict_full_logits(models, Xtest_np):
    def _predict_proba(name, mdl, Xs):
        if name == 'mlp':
            return predict_sup_mlp(mdl, Xs)
        else:
            return mdl.predict_proba(Xs)
    lgts = 0
    for name, model in models.items():
        lgts += logits_of(_predict_proba(name, model, Xtest_np))
    return lgts

full_logits_test = predict_full_logits(FULLS, Xtest_np)

# =================================================================
# 7.5) (ADDED) Mean-Teacher FixMatch + Distribution Alignment (SSL)
# =================================================================
ssl = {"enabled": False}
if HAVE_TORCH:
    unlabeled_path = f"{DATA_DIR}/kaggle_full_unlabeled_data.csv"
    if os.path.exists(unlabeled_path):
        unlabeled_df = pd.read_csv(unlabeled_path)
        unl_base = unlabeled_df.drop(columns=[c for c in ['label_source'] if c in unlabeled_df.columns])
        unl_feat = create_features(unl_base).reindex(columns=feature_cols)
        Xu_df = unl_feat.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
        Xu_np = scaler.transform(Xu_df.values).astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d_in, d_out = X_np.shape[1], num_class

        class TabMLP(nn.Module):
            def __init__(self, d_in, d_out):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(),
                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                    nn.Linear(128, d_out)
                )
            def forward(self, x): return self.net(x)

        def ema_update(teacher, student, decay=0.999):
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data.mul_(decay).add_(sp.data * (1.0 - decay))

        def aug_weak(x, noise_std=0.01):
            return x + np.random.normal(0, noise_std, size=x.shape).astype(np.float32)
        def aug_strong(x, drop_prob=0.05, noise_std=0.05):
            mask = (np.random.rand(*x.shape) > drop_prob).astype(np.float32)
            return (x * mask + np.random.normal(0, noise_std, size=x.shape)).astype(np.float32)

        def distribution_align(p, prior_src, prior_tgt, eps=1e-8, strength=1.0):
            adj = p * (prior_tgt[None,:] + eps) / (prior_src[None,:] + eps)
            adj = adj / np.clip(adj.sum(axis=1, keepdims=True), eps, None)
            return strength*adj + (1-strength)*p

        student = TabMLP(d_in, d_out).to(device)
        teacher = TabMLP(d_in, d_out).to(device)
        teacher.load_state_dict(student.state_dict())
        opt = torch.optim.AdamW(student.parameters(), lr=2e-3, weight_decay=1e-5)

        prior_src = np.bincount(y_np, minlength=num_class).astype(np.float64)
        prior_src = prior_src / prior_src.sum()
        target_prior = proba_test_cal.mean(axis=0)

        dl_u = DataLoader(TensorDataset(torch.from_numpy(Xu_np)), batch_size=1024, shuffle=True, drop_last=True)

        X_lab = X_np.astype(np.float32); y_lab = y_np.astype(np.int64)

        THR = 0.90
        UNSUP_W = 3.0
        EPOCHS = 30
        STEPS = 300

        student.train(); teacher.eval()
        for ep in range(EPOCHS):
            it = 0
            for (xb_u,) in dl_u:
                it += 1
                if it > STEPS: break

                idx = np.random.randint(0, len(X_lab), size=xb_u.size(0))
                xb_l = torch.from_numpy(X_lab[idx]).to(device)
                yb_l = torch.from_numpy(y_lab[idx]).to(device)

                xw = torch.from_numpy(aug_weak(xb_u.numpy())).to(device)
                xs = torch.from_numpy(aug_strong(xb_u.numpy())).to(device)

                with torch.no_grad():
                    pw = F.softmax(teacher(xw), dim=1).cpu().numpy()
                pw = distribution_align(pw, prior_src, target_prior, strength=0.7)
                pw = torch.from_numpy(pw).to(device)

                logits_l = student(xb_l)
                logits_s = student(xs)

                sup_loss = F.cross_entropy(logits_l, yb_l)
                conf_w, hard_w = pw.max(dim=1)
                mask = (conf_w >= THR).float()
                unsup_loss = (F.cross_entropy(logits_s, hard_w, reduction='none') * mask).mean()

                loss = sup_loss + UNSUP_W * unsup_loss
                opt.zero_grad(); loss.backward(); opt.step()
                ema_update(teacher, student, decay=0.999)

        teacher.eval()
        with torch.no_grad():
            p_unl = F.softmax(teacher(torch.from_numpy(Xu_np).to(device)), dim=1).cpu().numpy()
            p_tst = F.softmax(teacher(torch.from_numpy(Xtest_np.astype(np.float32)).to(device)), dim=1).cpu().numpy()

        def apply_temp_np(p, T):
            lg = np.log(np.clip(p, 1e-9, 1.0)) / T
            e = np.exp(lg - lg.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        p_unl = apply_temp_np(p_unl, T_global)
        p_tst = apply_temp_np(p_tst, T_global)

        p_unl = distribution_align(p_unl, p_unl.mean(axis=0), target_prior, strength=0.7)
        p_tst = distribution_align(p_tst, p_tst.mean(axis=0), target_prior, strength=0.7)

        ssl = {"enabled": True, "Xu_np": Xu_np, "p_unl": p_unl, "p_tst": p_tst}
    else:
        print("Unlabeled file not found for SSL; skip Mean-Teacher.")
else:
    print("Torch not available; skip Mean-Teacher.")

if ssl.get("enabled", False):
    full_logits_test += logits_of(ssl["p_tst"])

full_proba_test_cal = apply_temp(full_logits_test, T_global)
pred_test_full_ens = full_proba_test_cal.argmax(1) + 1

# ==========================================================
# 8) Pseudo-Labeling — CONSENSUS + PER-CLASS THR + TTA STABILITY
# ==========================================================
USE_PSEUDO   = True
PSEUDO_THRESH = float(os.environ.get("PL_THRESH","0.950"))
PSEUDO_WEIGHT = float(os.environ.get("PL_WEIGHT","0.60"))
PSEUDO_MAX_PER_CLASS = int(os.environ.get("PL_MAX_PER_CLASS","40000"))

TTA_K     = int(os.environ.get("PL_TTA_K", "3"))
TTA_SIGMA = float(os.environ.get("PL_TTA_SIGMA", "0.012"))
STAB_MIN  = float(os.environ.get("PL_STAB_MIN", "0.70"))

ALPHA = float(os.environ.get("PL_ALPHA","2.0"))
BETA  = float(os.environ.get("PL_BETA","1.0"))

def predict_calibrated_proba(models, Xs):
    logits = 0
    out = {}
    for name, mdl in models.items():
        if name == 'mlp':
            p = predict_sup_mlp(mdl, Xs)
        else:
            p = mdl.predict_proba(Xs)
        out[name] = apply_temp(logits_of(p), T_global)
        logits += logits_of(p)
    out['ens'] = apply_temp(logits, T_global)
    return out

class_counts = np.bincount(y_np, minlength=num_class).astype(float)
mean_count = class_counts.mean()
thr_per_class = np.array([PSEUDO_THRESH + (0.01 if class_counts[c] < mean_count else 0.0)
                          for c in range(num_class)], dtype=float)
for c in range(num_class):
    env_key = f"PL_THR_C{c}"
    if env_key in os.environ:
        thr_per_class[c] = float(os.environ[env_key])
print("Per-class PL thresholds:", np.round(thr_per_class,4))

pred_test_full = pred_test_full_ens - 1
proba_test_full = full_proba_test_cal

if USE_PSEUDO:
    unlabeled_path = f"{DATA_DIR}/kaggle_full_unlabeled_data.csv"
    if os.path.exists(unlabeled_path):
        if ssl.get("enabled", False):
            Xu_np = ssl["Xu_np"]
        else:
            unlabeled_df = pd.read_csv(unlabeled_path)
            unl_base = unlabeled_df.drop(columns=[c for c in ['label_source'] if c in unlabeled_df.columns])
            unl_feat = create_features(unl_base)
            Xu_df = unl_feat.reindex(columns=feature_cols)
            Xu_df = Xu_df.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
            Xu_np = scaler.transform(Xu_df.values)

        logits_all = 0
        pred_each_list = []
        for name, mdl in FULLS.items():
            if name == 'mlp':
                p = predict_sup_mlp(mdl, Xu_np)
            else:
                p = mdl.predict_proba(Xu_np)
            logits_all += logits_of(p)
            pred_each_list.append(p.argmax(1))
        if ssl.get("enabled", False):
            p_ssl_u = ssl["p_unl"]
            logits_all += logits_of(p_ssl_u)
            pred_each_list.append(p_ssl_u.argmax(1))

        proba_ens = apply_temp(logits_all, T_global)
        pred_each = np.stack(pred_each_list, axis=0)
        pred_ens = proba_ens.argmax(1)
        conf_ens = proba_ens.max(1)

        agree_counts = (pred_each == pred_ens).sum(axis=0)
        need_votes = 2 if pred_each.shape[0] >= 2 else 1
        consensus_mask = agree_counts >= need_votes

        TTA_votes = np.zeros((Xu_np.shape[0],), dtype=int)
        TTA_conf_sum = np.zeros((Xu_np.shape[0],), dtype=float)
        TTA_margin_sum = np.zeros((Xu_np.shape[0],), dtype=float)
        for k in range(TTA_K):
            noise = np.random.normal(0.0, TTA_SIGMA, size=Xu_np.shape)
            Xu_noisy = Xu_np + noise
            proba_k = predict_calibrated_proba(FULLS, Xu_noisy)['ens']
            pred_k = proba_k.argmax(1)
            conf_k = proba_k.max(1)
            sorted_p = np.sort(proba_k, axis=1)
            margin_k = sorted_p[:, -1] - sorted_p[:, -2]
            TTA_votes += (pred_k == pred_ens).astype(int)
            TTA_conf_sum += conf_k
            TTA_margin_sum += margin_k

        stab_ratio = TTA_votes / float(TTA_K)
        conf_tta = TTA_conf_sum / float(TTA_K)
        margin_tta = TTA_margin_sum / float(TTA_K)

        u_lab = pred_ens
        per_class = {c: 0 for c in range(num_class)}
        selected = []

        idx_order = np.argsort(-(conf_tta))
        for i in idx_order:
            c = int(u_lab[i])
            if not consensus_mask[i]: continue
            if stab_ratio[i] < STAB_MIN: continue
            if conf_tta[i] < thr_per_class[c]: continue
            if per_class[c] >= PSEUDO_MAX_PER_CLASS: continue
            selected.append(i)
            per_class[c] += 1

        selected = np.array(selected, dtype=int)
        print(f"Pseudo selected (consensus+stab): {len(selected)} | per-class:", per_class)

        if len(selected) > 0:
            conf_sel = conf_tta[selected]
            marg_sel = margin_tta[selected]
            w_pl_soft = PSEUDO_WEIGHT * np.power(np.clip(conf_sel, 1e-6, 1.0), ALPHA) * \
                        np.power(np.clip(marg_sel, 1e-6, 1.0), BETA)
            w_pl_soft = np.clip(w_pl_soft, 0.05, 2.0)

            X_aug = np.vstack([X_np, Xu_np[selected]])
            y_aug = np.concatenate([y_np, u_lab[selected]])
            w_aug = np.concatenate([sample_w, w_pl_soft])

            xgb_best_iters = int(np.mean([it for it in BEST_ITERS.get('xgb', []) if it > 0])) if BEST_ITERS.get('xgb') else 2000
            lgb_best_iters = int(np.mean([it for it in BEST_ITERS.get('lgb', []) if it > 0])) if BEST_ITERS.get('lgb') else 2000

            xgb_pl = xgb.XGBClassifier(**{**xgb_params, "n_estimators": max(xgb_best_iters, 50)})
            lgb_pl = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": max(lgb_best_iters, 50)})
            xgb_pl.fit(X_aug, y_aug, sample_weight=w_aug, verbose=False)
            lgb_pl.fit(X_aug, y_aug, sample_weight=w_aug, callbacks=[lgb.log_evaluation(period=0)])

            logits_pl = logits_of(xgb_pl.predict_proba(Xtest_np)) + logits_of(lgb_pl.predict_proba(Xtest_np))
            if HAVE_CAT and BEST_ITERS.get('cat'):
                cat_best_iters = int(np.mean([it for it in BEST_ITERS['cat'] if it > 0])) if len(BEST_ITERS['cat']) else 2000
                cat_pl = CatBoostClassifier(**{**cat_params, "iterations": max(cat_best_iters, 50)})
                cat_pl.fit(X_aug, y_aug, sample_weight=w_aug, verbose=False)
                logits_pl += logits_of(cat_pl.predict_proba(Xtest_np))
            if ssl.get("enabled", False):
                logits_pl += logits_of(ssl["p_tst"])

            proba_pl = apply_temp(logits_pl, T_global)
            pred_test_pl = proba_pl.argmax(1) + 1
        else:
            pred_test_pl = None
    else:
        print("Unlabeled file not found. Skipping Pseudo-Labeling.")
        pred_test_pl = None
else:
    pred_test_pl = None

# ---------------------------
# 9) Save submissions
# ---------------------------
sub_stack = pd.DataFrame({'id': np.arange(len(X_test)), 'risk_level': (test_best.argmax(1)+1).astype(int)})
sub_full  = pd.DataFrame({'id': np.arange(len(X_test)), 'risk_level': pred_test_full_ens.astype(int)})

sub_stack.to_csv('sub_stack.csv', index=False)
sub_full.to_csv('sub_full_ens_cal.csv', index=False)

print("\nsub_stack class dist (%):")
print((sub_stack['risk_level'].value_counts(normalize=True).sort_index()*100).round(2))
print("\nsub_full_ens_cal class dist (%):")
print((sub_full['risk_level'].value_counts(normalize=True).sort_index()*100).round(2))

if pred_test_pl is not None:
    sub_pl = pd.DataFrame({'id': np.arange(len(X_test)), 'risk_level': pred_test_pl.astype(int)})
    sub_pl.to_csv('sub_pseudolabel_consensus_stable.csv', index=False)
    print("\nsub_pseudolabel_consensus_stable class dist (%):")
    print((sub_pl['risk_level'].value_counts(normalize=True).sort_index()*100).round(2))
    print("\nSaved: sub_stack.csv, sub_full_ens_cal.csv, sub_pseudolabel_consensus_stable.csv")
else:
    print("\nSaved: sub_stack.csv, sub_full_ens_cal.csv")

# ---------------------------
# 10) Simple FI từ một model (XGB full) (optional)
# ---------------------------
try:
    if 'xgb' in FULLS:
        importances = FULLS['xgb'].feature_importances_
        fi = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False).head(20)
        print("\nTop-20 XGB features:")
        print(fi)
except Exception as e:
    print("FI error:", e)

gc.collect()


