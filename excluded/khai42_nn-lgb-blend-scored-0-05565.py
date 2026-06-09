import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import cudf


import warnings
warnings.filterwarnings('ignore')
import os, random, gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import lightgbm as lgb


SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

try:
    if tf.config.list_physical_devices("GPU"):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

TARGET, ID_COL = "accident_risk", "id"

if TARGET in test.columns and test[TARGET].nunique(dropna=False) <= 3:
    test = test.drop(columns=[TARGET])


expected_cat = [
    "road_type", "lighting", "weather",
    "road_signs_present", "public_road",
    "time_of_day", "holiday", "school_season"
]
present_cat = [c for c in expected_cat if c in train.columns and c in test.columns]

extra_cat = [c for c in train.columns
             if c not in present_cat and c not in [ID_COL, TARGET]
             and (train[c].dtype == "object" or str(train[c].dtype) == "bool")
             and c in test.columns]

cat_cols = present_cat + extra_cat


from sklearn.preprocessing import LabelEncoder
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    encoders[col] = le

for df_ in (train, test):
    for c in df_.columns:
        if df_[c].dtype == "bool":
            df_[c] = df_[c].astype(int)

def add_freq_encoding(tr, te, cols):
    for c in cols:
        freq = tr[c].value_counts(normalize=True)
        tr[c+"_freq"] = tr[c].map(freq).astype("float32")
        te[c+"_freq"] = te[c].map(freq).fillna(0).astype("float32")

add_freq_encoding(train, test, cat_cols)

def add_interactions(df):
    if {"speed_limit","curvature"}.issubset(df.columns):
        df["speed_x_curve"] = df["speed_limit"].astype("float32") * df["curvature"].astype("float32")
    if {"num_lanes","speed_limit"}.issubset(df.columns):
        df["lanes_x_speed"] = df["num_lanes"].astype("float32") * df["speed_limit"].astype("float32")
    if {"num_lanes","curvature"}.issubset(df.columns):
        df["lanes_x_curve"] = df["num_lanes"].astype("float32") * df["curvature"].astype("float32")
    if {"num_reported_accidents","speed_limit"}.issubset(df.columns):
        df["acc_x_speed"] = df["num_reported_accidents"].astype("float32") * df["speed_limit"].astype("float32")
    if {"num_reported_accidents","curvature"}.issubset(df.columns):
        df["acc_x_curve"] = df["num_reported_accidents"].astype("float32") * df["curvature"].astype("float32")
    if {"lighting","speed_limit"}.issubset(df.columns):
        df["light_x_speed"] = df["lighting"].astype("float32") * df["speed_limit"].astype("float32")
    if {"weather","speed_limit"}.issubset(df.columns):
        df["weather_x_speed"] = df["weather"].astype("float32") * df["speed_limit"].astype("float32")

add_interactions(train)
add_interactions(test)

features = [c for c in train.columns if c not in [ID_COL, TARGET]]
X_full = train[features].astype("float32")
y_full = train[TARGET].astype("float32").values
X_test = test[features].astype("float32")

def make_stratify_bins(y, n_bins=20):
    y_clip = np.clip(y, 0, 1)
    bins = np.floor(y_clip * n_bins).astype(int)
    bins[bins == n_bins] = n_bins - 1
    return bins

bins = make_stratify_bins(y_full, n_bins=20)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


%%time
LGB_AVAILABLE = False
LGB_GPU = False
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    pass

lgb_oof = np.zeros(len(X_full), dtype="float32")
lgb_test = np.zeros(len(X_test), dtype="float32")
lgb_fold_scores = []

if LGB_AVAILABLE:
    lgb_params = dict(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=50,
        reg_alpha=1.0,
        reg_lambda=2.0,
        n_estimators=10000,
        random_state=SEED,
        verbose=-1
    )

    try:
        lgb_params.update({
            "device": "gpu",        
            "gpu_platform_id": 0,
            "gpu_device_id": 0
        })
        tiny_idx = np.random.RandomState(SEED).choice(len(X_full), size=min(1000, len(X_full)), replace=False)
        dtrain_tiny = lgb.Dataset(X_full.iloc[tiny_idx], label=y_full[tiny_idx])
        _ = lgb.train(lgb_params, dtrain_tiny, num_boost_round=1, valid_sets=[dtrain_tiny], verbose_eval=False)
        LGB_GPU = True
        print("[LGB] Using GPU.")
    except Exception as e:
        # Fallback to CPU
        for k in ["device","gpu_platform_id","gpu_device_id"]:
            lgb_params.pop(k, None)
        print(f"[LGB] GPU unavailable, falling back to CPU. Reason: {str(e)[:120]}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, bins), 1):
        X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
        y_tr, y_va = y_full[tr_idx], y_full[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va)

        model_lgb = lgb.train(
            params=lgb_params,
            train_set=dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        va_pred = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
        lgb_oof[va_idx] = va_pred
        fold_rmse = mean_squared_error(y_va, va_pred, squared=False)
        lgb_fold_scores.append(fold_rmse)
        print(f"[LGB] Fold {fold} RMSE: {fold_rmse:.6f}")

        lgb_test += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / skf.n_splits

    lgb_oof_rmse = mean_squared_error(y_full, lgb_oof, squared=False)
    print(f"[LGB] OOF RMSE: {lgb_oof_rmse:.6f}")
else:
    print("[LGB] LightGBM not installed; skipping.")


CUMl_AVAILABLE = False
try:
    import cudf
    import cupy as cp
    from cuml.ensemble import RandomForestRegressor as cuRF
    CUMl_AVAILABLE = True
    print("[cuML] cuML detected. Will train cuML RandomForestRegressor.")
except Exception as e:
    print(f"[cuML] Not available: {str(e)[:120]}")

cuml_oof = np.zeros(len(X_full), dtype="float32")
cuml_test = np.zeros(len(X_test), dtype="float32")
cuml_fold_scores = []

if CUMl_AVAILABLE:
    cuml_params = dict(
        n_estimators=800,
        max_depth=18,
        n_bins=128,           
        max_features="auto",
        min_samples_leaf=2,
        bootstrap=True,
        n_streams=8,
        random_state=SEED
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, bins), 1):
        X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
        y_tr, y_va = y_full[tr_idx], y_full[va_idx]

        gX_tr = cudf.DataFrame.from_pandas(X_tr)
        gX_va = cudf.DataFrame.from_pandas(X_va)
        gX_te = cudf.DataFrame.from_pandas(X_test)
        gy_tr = cudf.Series(y_tr)
        gy_va = cudf.Series(y_va)

        model_rf = cuRF(**cuml_params)
        model_rf.fit(gX_tr, gy_tr)

        va_pred = model_rf.predict(gX_va).to_numpy().astype("float32")
        cuml_oof[va_idx] = va_pred
        fold_rmse = mean_squared_error(y_va, va_pred, squared=False)
        cuml_fold_scores.append(fold_rmse)
        print(f"[cuML] Fold {fold} RMSE: {fold_rmse:.6f}")

        te_pred = model_rf.predict(gX_te).to_numpy().astype("float32")
        cuml_test += te_pred / skf.n_splits

    cuml_oof_rmse = mean_squared_error(y_full, cuml_oof, squared=False)
    print(f"[cuML] OOF RMSE: {cuml_oof_rmse:.6f}")


def build_nn(input_dim: int) -> tf.keras.Model:
    leaky = tf.keras.layers.LeakyReLU(0.1)
    optim = tf.keras.optimizers.Nadam(learning_rate=5e-4, decay=3e-4)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation=leaky, kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(112, activation=leaky, kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation=leaky, kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(80, activation=leaky, kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation=None, dtype="float32"),
    ])
    model.compile(optimizer=optim, loss="mse",
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


%%time
from sklearn.preprocessing import StandardScaler
EPOCHS = 200
BATCH_SIZE = 1024

nn_oof = np.zeros(len(X_full), dtype="float32")
nn_test = np.zeros(len(X_test), dtype="float32")
nn_fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, bins), 1):
    X_tr, X_va = X_full.iloc[tr_idx].copy(), X_full.iloc[va_idx].copy()
    y_tr, y_va = y_full[tr_idx], y_full[va_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_test)

    model_nn = build_nn(X_tr.shape[1])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_rmse", patience=10, mode="min", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_rmse", factor=0.5, patience=5, mode="min", min_lr=1e-6),
    ]
    model_nn.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=0
    )

    va_pred = model_nn.predict(X_va, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    nn_oof[va_idx] = va_pred
    fold_rmse = mean_squared_error(y_va, va_pred, squared=False)
    nn_fold_scores.append(fold_rmse)
    print(f"[NN ] Fold {fold} RMSE: {fold_rmse:.6f}")

    te_pred = model_nn.predict(X_te, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    nn_test += te_pred / skf.n_splits

nn_oof_rmse = mean_squared_error(y_full, nn_oof, squared=False)
print(f"[NN ] OOF RMSE: {nn_oof_rmse:.6f}")


def rmse(a, b): return mean_squared_error(a, b, squared=False)

have_lgb = np.any(lgb_oof != 0) if LGB_AVAILABLE else False
have_cuml = CUMl_AVAILABLE and np.any(cuml_oof != 0)

if have_lgb and have_cuml:
    best_rmse, best_w_lgb, best_w_cuml = 1e9, 0.0, 0.0
    ws = np.linspace(0, 1, 51)
    for w_lgb in ws:
        for w_cuml in ws:
            if w_lgb + w_cuml <= 1.0:
                pred = w_lgb * lgb_oof + w_cuml * cuml_oof + (1 - w_lgb - w_cuml) * nn_oof
                s = rmse(y_full, pred)
                if s < best_rmse:
                    best_rmse, best_w_lgb, best_w_cuml = s, w_lgb, w_cuml
    w_lgb, w_cuml = best_w_lgb, best_w_cuml
    w_nn = 1 - w_lgb - w_cuml
    print(f"[BLEND] OOF RMSE={best_rmse:.6f} | w_lgb={w_lgb:.2f}, w_cuml={w_cuml:.2f}, w_nn={w_nn:.2f}")
    test_pred = w_lgb * lgb_test + w_cuml * cuml_test + w_nn * nn_test

elif have_lgb:
    best_rmse, best_w = 1e9, 0.0
    ws = np.linspace(0, 1, 51)
    for w in ws:
        pred = w * lgb_oof + (1 - w) * nn_oof
        s = rmse(y_full, pred)
        if s < best_rmse:
            best_rmse, best_w = s, w
    print(f"[BLEND] OOF RMSE={best_rmse:.6f} | w_lgb={best_w:.2f}, w_nn={1-best_w:.2f}")
    test_pred = best_w * lgb_test + (1 - best_w) * nn_test

elif have_cuml:
    best_rmse, best_w = 1e9, 0.0
    ws = np.linspace(0, 1, 51)
    for w in ws:
        pred = w * cuml_oof + (1 - w) * nn_oof
        s = rmse(y_full, pred)
        if s < best_rmse:
            best_rmse, best_w = s, w
    print(f"[BLEND] OOF RMSE={best_rmse:.6f} | w_cuml={best_w:.2f}, w_nn={1-best_w:.2f}")
    test_pred = best_w * cuml_test + (1 - best_w) * nn_test

else:
    print("[BLEND] Only NN available; using NN predictions.")
    test_pred = nn_test
test_pred = np.clip(test_pred, 0.0, 1.0)


submission = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_pred})
submission = submission[sample_sub.columns.tolist()]
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(f"Submission saved to: {out_path}")


if LGB_AVAILABLE:
    try:
        print(f"LGB  OOF RMSE: {mean_squared_error(y_full, lgb_oof, squared=False):.6f}")
    except: pass
if CUMl_AVAILABLE:
    try:
        print(f"cuML OOF RMSE: {mean_squared_error(y_full, cuml_oof, squared=False):.6f}")
    except: pass
print(f"NN   OOF RMSE: {nn_oof_rmse:.6f}")




