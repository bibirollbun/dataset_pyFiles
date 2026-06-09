import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

from sklearn.model_selection import TimeSeriesSplit

def custom_score(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        raise ValueError('empty array')

    if (y_true < 0).any():
        raise ValueError('negative y_true')

    if (~ np.isfinite(y_pred)).any():
        raise ValueError('infinite y_pred')

    ape = np.abs((y_true - y_pred) / np.maximum(y_true, eps))

    good_mask = ape <= 1.0
    good_rate = good_mask.mean()
    if good_rate < 0.7:
        return {'score': 0, 'good_rate': good_rate, 'str': f"{Fore.RED}score={0:.3f} {good_rate=:.3f}{Style.RESET_ALL}"}

    good_ape = ape[good_mask]
    mape = np.mean(good_ape)

    scaled_mape = mape / good_rate
    score = 1 - scaled_mape
    # score = max(0.0, score)
    return {'score': score, 'good_rate': good_rate, 'str': f"{score=:.3f} {good_rate=:.3f}"}


ci = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv') # one row per year
csi = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv') # several rows per training month
sp = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv') # at most one row per sector

train_lt = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv')
train_ltns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv')
train_pht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv')
train_phtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv')
train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
train_nhtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv')
test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')

month_codes = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

test_id = test.id.str.split('_', expand=True)
test['month'] = test_id[0]
test['sector'] = test_id[1]
del test_id

for df in [train_lt, train_ltns, train_pht, train_phtns, train_nht, train_nhtns, csi, sp, test]:
    if df is not csi:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
        # print(df.sector_id.min(), df.sector_id.max(), len(np.unique(df.sector_id)), len(df))
    if df is not sp:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1 # min=0, max=66
        print(df['time'].min(), df['time'].max())


amount_new_house_transactions = train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
# Missing values must be filled with zero:
amount_new_house_transactions = amount_new_house_transactions.fillna(0)
# We add sector 95, which has no transactions during the training period:
amount_new_house_transactions[95] = 0
amount_new_house_transactions = amount_new_house_transactions[np.arange(1, 97)]
amount_new_house_transactions.astype(int)


# # =========================
# # GRU (log1p levels) — train excluding zero-sales sectors, predict 6 months
# # Then append months 73..78 as zeros and force excluded sectors to zero at the end
# # =========================

# import os, gc, math, random
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.model_selection import GroupKFold

# # ---- Reproducibility
# SEED = 42
# random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# # =========================
# # Input: amount_new_house_transactions (67 rows x 96 columns)
# # rows: time 0..66, columns: sector_id 1..96 (int)
# # =========================
# df = amount_new_house_transactions[-36:].copy()
# df = df.sort_index()
# df = df[np.arange(1, 97)]

# T = df.shape[0]        # 67
# ALL_SECTORS = np.arange(1, 97)
# print("Data shape (T x N):", df.shape)

# # =========================
# # Excluded sectors (zeroed in final submission, not used for training)
# # =========================
# zero_sales_sector = [12,33,39,41,44,49,52,53,58,72,73,74,75,82,87,89,95,96]
# zero_sales_set = set(zero_sales_sector)
# assert zero_sales_set.issubset(set(ALL_SECTORS)), "zero_sales_sector must be within 1..96"

# train_sectors = [s for s in ALL_SECTORS if s not in zero_sales_set]
# print(f"Training on {len(train_sectors)} sectors; excluding {len(zero_sales_sector)}:", zero_sales_sector)

# df_train = df[train_sectors]   # (67 x N_TRAIN)
# N_SERIES_TRAIN = df_train.shape[1]
# data_levels = df_train.values.T  # (N_TRAIN x 67)

# # =========================
# # Sliding windows
# # =========================
# WIDTH = 12
# H = 6                 # public LB horizon
# pre_len = WIDTH - H   # history length per sample

# assert T >= WIDTH, "Not enough history"
# COPIES = T - WIDTH + 1
# print(f"Window width={WIDTH}, horizon={H}, copies per series={COPIES}")

# x_data = np.zeros((N_SERIES_TRAIN * COPIES, pre_len), dtype=float)
# y_data = np.zeros((N_SERIES_TRAIN * COPIES, H), dtype=float)
# # Group indices for GroupKFold (0..N_SERIES_TRAIN-1)
# groups = np.zeros(N_SERIES_TRAIN * COPIES, dtype=int)

# for s in range(N_SERIES_TRAIN):
#     series = data_levels[s]  # (T,)
#     for k in range(COPIES):
#         i = s * COPIES + k
#         window = series[k:k+WIDTH]
#         x_data[i] = window[:pre_len]
#         y_data[i] = window[pre_len:]
#         groups[i] = s  # group by trained-sector index

# print("x_data shape:", x_data.shape, "| y_data shape:", y_data.shape)

# # =========================
# # Transform to log1p(level) and standardize
# # =========================
# x_log = np.log1p(x_data)
# y_log = np.log1p(y_data)

# mn = x_log.mean()
# sd = x_log.std() + 1e-12
# x_std = (x_log - mn) / sd
# y_std = (y_log - mn) / sd

# # =========================
# # Build training DataFrame (for GroupKFold)
# # =========================
# FEATURES = [f"f{k}" for k in range(pre_len)]
# TARGETS  = [f"y{k}" for k in range(H)]
# train_data = pd.DataFrame(x_std, columns=FEATURES)
# for k in range(H):
#     train_data[TARGETS[k]] = y_std[:, k]
# train_data["group_idx"] = groups  # 0..N_SERIES_TRAIN-1

# print("Train DF shape:", train_data.shape)
# train_data.head()

# # =========================
# # Model
# # =========================
# SEQ_LEN = pre_len

# def build_model():
#     inp = tf.keras.Input(shape=(SEQ_LEN, 1))
#     x = tf.keras.layers.GRU(32, return_sequences=True, dropout=0.1)(inp)
#     x = tf.keras.layers.GRU(32, return_sequences=True, dropout=0.1)(x)
#     x = tf.keras.layers.GRU(32, return_sequences=False, dropout=0.1)(x)
#     out = tf.keras.layers.Dense(H, activation="linear")(x)
#     model = tf.keras.Model(inp, out)
#     model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
#                   loss=tf.keras.losses.Huber(delta=0.5))
#     return model

# model = build_model()
# model.summary()

# # =========================
# # Cross-validation training
# # =========================
# FOLDS = 5
# BATCH_SIZE = 32
# EPOCHS = 8
# VERBOSE = 2

# X_all = train_data[FEATURES].values.astype(np.float32).reshape(-1, SEQ_LEN, 1)
# y_all = train_data[TARGETS].values.astype(np.float32)

# oof_pred_std = np.zeros_like(y_all, dtype=np.float32)

# # Recent windows weighted more
# base_w = np.linspace(1.0, 2.0, COPIES)
# base_w = base_w / base_w.mean()

# gkf = GroupKFold(n_splits=FOLDS)
# for fold, (tr_idx, va_idx) in enumerate(
#     gkf.split(train_data, y=train_data["y0"], groups=train_data["group_idx"])
# ):
#     print("\n" + "#"*28)
#     print(f"### Fold {fold+1}/{FOLDS}")
#     print("#"*28)

#     X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
#     X_va, y_va = X_all[va_idx], y_all[va_idx]

#     w_tr = np.array([base_w[i % COPIES] for i in tr_idx], dtype=np.float32)

#     model = build_model()
#     model.fit(
#         X_tr, y_tr,
#         validation_data=(X_va, y_va),
#         sample_weight=w_tr,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         verbose=VERBOSE,
#         callbacks=[
#             tf.keras.callbacks.ReduceLROnPlateau(
#                 monitor="val_loss", factor=0.5, patience=2, verbose=1
#             ),
#             tf.keras.callbacks.EarlyStopping(
#                 monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
#             )
#         ]
#     )

#     oof_pred_std[va_idx] = model.predict(X_va, verbose=0)

# # =========================
# # OOF inversion and quick checks
# # =========================
# def smape(y_true, y_pred):
#     y_true = np.asarray(y_true).astype(float).ravel()
#     y_pred = np.asarray(y_pred).astype(float).ravel()
#     denom = (np.abs(y_true) + np.abs(y_pred))
#     denom[denom == 0] = 1.0
#     return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

# oof_log = oof_pred_std * sd + mn
# oof_levels = np.expm1(oof_log)
# oof_levels = np.clip(oof_levels, 0, None)

# print("OOF SMAPE (all H):", smape(y_data, oof_levels))
# print("OOF SMAPE (last 2 months):", smape(y_data[:, -2:], oof_levels[:, -2:]))

# m = custom_score(y_data.flatten(), oof_levels.flatten())
# print('custom_score',m['str'])

# # =========================
# # Retrain on ALL training samples
# # =========================
# final_model = build_model()
# w_all = np.array([base_w[i % COPIES] for i in range(len(X_all))], dtype=np.float32)
# final_model.fit(
#     X_all, y_all,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     verbose=VERBOSE,
#     sample_weight=w_all,
#     callbacks=[
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor="loss", factor=0.5, patience=2, verbose=1
#         ),
#         tf.keras.callbacks.EarlyStopping(
#             monitor="loss", patience=3, restore_best_weights=True, verbose=1
#         )
#     ]
# )

# # =========================
# # Inference (months 67..72) for TRAINED sectors only
# # =========================
# X_test_levels = np.zeros((N_SERIES_TRAIN, pre_len), dtype=float)
# for s in range(N_SERIES_TRAIN):
#     series = data_levels[s]  # length T
#     X_test_levels[s] = series[-pre_len:]

# X_test_log = np.log1p(X_test_levels)
# X_test_std = (X_test_log - mn) / sd
# X_test_std = X_test_std.astype(np.float32).reshape(N_SERIES_TRAIN, SEQ_LEN, 1)

# pred_std   = final_model.predict(X_test_std, verbose=0)
# pred_log   = pred_std * sd + mn
# pred_train = np.expm1(pred_log)
# pred_train = np.clip(pred_train, 0, None)   # shape: (N_SERIES_TRAIN, H)

# # =========================
# # Assemble full (96 sectors) predictions for 67..72
# # Put trained sectors' preds, zero for excluded sectors
# # =========================
# pred_full = np.zeros((len(ALL_SECTORS), H), dtype=float)  # (96 x 6)
# # map trained sector id → row index in pred_full
# sector_to_row = {s: i for i, s in enumerate(ALL_SECTORS)}
# # fill trained sectors
# for idx, s in enumerate(train_sectors):
#     pred_full[sector_to_row[s], :] = pred_train[idx]

# # force excluded sectors to zero (already zero, but explicit)
# for s in zero_sales_sector:
#     pred_full[sector_to_row[s], :] = 0.0

# # =========================
# # Build pred_df as time-major (rows=time, cols=sector_id)
# # Months: 67..72 from model, 73..78 zeros
# # =========================
# future_times = np.arange(T, T + H)   # 67..72
# pred_df = (
#     pd.DataFrame(pred_full.T, index=[f"time_{t}" for t in future_times], columns=ALL_SECTORS)
# )

# # (Optional) threshold small values to zero before submission
# # pred_df[pred_df < 10] = 0

# # Add 6 more months 73..78 as zeros
# for t in range(T + H, T + 12):  # 73..78
#     pred_df.loc[f"time_{t}"] = 0

# # Ensure column dtype is int sector ids and row order correct
# pred_df = pred_df.sort_index()  # time_67..time_78
# pred_df = pred_df[ALL_SECTORS]  # columns 1..96

# print("Predictions (time-major) head:")
# display(pred_df.head())

# # =========================
# # Write submission
# # =========================
# test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')

# # IMPORTANT: pred_df is time-major; Kaggle expects stacking by time then sector
# # a_pred.T.unstack().values matches your required order
# test['new_house_transaction_amount'] = pred_df.T.unstack().values

# test[['id', 'new_house_transaction_amount']].to_csv('submission.csv', index=False)

# # quick peek
# !head submission.csv



# =========================
# Preprocess: load & build amount_new_house_transactions + time→month map
# =========================
import numpy as np, pandas as pd

ci = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv')
csi = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv')
sp  = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv')

train_lt   = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv')
train_ltns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv')
train_pht  = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv')
train_phtns= pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv')
train_nht  = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
train_nhtns= pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv')
test       = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')

month_codes = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

test_id = test.id.str.split('_', expand=True)
test['month'] = test_id[0]
test['sector'] = test_id[1]
del test_id

for df_ in [train_lt, train_ltns, train_pht, train_phtns, train_nht, train_nhtns, csi, sp, test]:
    if df_ is not csi:
        df_['sector_id'] = df_.sector.str.slice(7, None).astype(int)
    if df_ is not sp:
        df_['year']  = df_.month.str.slice(0, 4).astype(int)
        df_['month'] = df_.month.str.slice(5, None).map(month_codes)
        df_['time']  = (df_['year'] - 2019) * 12 + df_['month'] - 1  # 0..66

# main matrix: time × sector_id
amount_new_house_transactions = (
    train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
)
amount_new_house_transactions = amount_new_house_transactions.fillna(0)
amount_new_house_transactions[95] = 0  # sector 95 had none
amount_new_house_transactions = amount_new_house_transactions[np.arange(1, 97)]
amount_new_house_transactions = amount_new_house_transactions.astype(float)

# ---- month map: time → month(1..12), from train_nht you already built
time_month_map = (
    train_nht[['time','month']].drop_duplicates().set_index('time')['month'].to_dict()
)

# =========================
# GRU (log1p levels) — exclude zero-sectors, add month sin/cos features, predict 6 months
# =========================
import os, gc, math, random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Use last 36 months (as in your current code)
df = amount_new_house_transactions[-48:].copy()
df = df.sort_index()                   # ensure time ascending
df = df[np.arange(1, 97)]              # 1..96
T  = df.shape[0]                       # 36
ALL_SECTORS = np.arange(1, 97)
print("Data shape (T x N):", df.shape)

# Build aligned month series for these T rows using df.index (which are 'time')
times_in_df  = df.index.values        # actual 'time' ids
months_in_df = np.array([time_month_map[int(t)] for t in times_in_df], dtype=int)  # length T

# Excluded sectors (not used in training)
zero_sales_sector = [12,33,39,41,44,49,52,53,58,72,73,74,75,82,87,89,95,96]
zero_sales_set = set(zero_sales_sector)
assert zero_sales_set.issubset(set(ALL_SECTORS))

train_sectors = [s for s in ALL_SECTORS if s not in zero_sales_set]
print(f"Training on {len(train_sectors)} sectors; excluding {len(zero_sales_sector)}:", zero_sales_sector)

df_train = df[train_sectors]
N_SERIES_TRAIN = df_train.shape[1]
data_levels = df_train.values.T  # (N_TRAIN x T)

# -------------------------
# Sliding windows
# -------------------------
WIDTH = 12
H = 6
pre_len = WIDTH - H
assert T >= WIDTH, "Not enough history"
COPIES = T - WIDTH + 1
print(f"Window width={WIDTH}, horizon={H}, copies per series={COPIES}")

x_data = np.zeros((N_SERIES_TRAIN * COPIES, pre_len), dtype=float)
y_data = np.zeros((N_SERIES_TRAIN * COPIES, H), dtype=float)
groups = np.zeros(N_SERIES_TRAIN * COPIES, dtype=int)

# month features per-sample, per-timestep: sin, cos for the HISTORY part
X_season = np.zeros((N_SERIES_TRAIN * COPIES, pre_len, 2), dtype=np.float32)

for s in range(N_SERIES_TRAIN):
    series = data_levels[s]  # length T
    for k in range(COPIES):
        i = s * COPIES + k
        t0 = k
        window_levels = series[t0:t0+WIDTH]           # len WIDTH
        window_months = months_in_df[t0:t0+WIDTH]     # len WIDTH (aligned)

        x_data[i] = window_levels[:pre_len]
        y_data[i] = window_levels[pre_len:]
        groups[i] = s

        # month cyclical encodings for the history portion (pre_len)
        m_hist = window_months[:pre_len].astype(float)
        X_season[i,:,0] = np.sin(2*np.pi*m_hist/12.0)
        X_season[i,:,1] = np.cos(2*np.pi*m_hist/12.0)

print("x_data shape:", x_data.shape, "| y_data shape:", y_data.shape, "| season:", X_season.shape)

# -------------------------
# Transform to log1p(level) and standardize (levels only; season is already scaled -1..1)
# -------------------------
x_log = np.log1p(x_data)
y_log = np.log1p(y_data)

mn = x_log.mean()
sd = x_log.std() + 1e-12
x_std = (x_log - mn) / sd
y_std = (y_log - mn) / sd

# Build training arrays
FEATURES = [f"f{k}" for k in range(pre_len)]
TARGETS  = [f"y{k}" for k in range(H)]
train_data = pd.DataFrame(x_std, columns=FEATURES)
for k in range(H):
    train_data[TARGETS[k]] = y_std[:, k]
train_data["group_idx"] = groups

SEQ_LEN = pre_len
# stack: [level_chan, sin, cos] → shape (samples, SEQ_LEN, 3)
X_level = x_std.astype(np.float32).reshape(-1, SEQ_LEN, 1)
X_all   = np.concatenate([X_level, X_season], axis=2)
y_all   = y_std.astype(np.float32)

print("X_all shape:", X_all.shape, "y_all shape:", y_all.shape)

# -------------------------
# Model (input channels = 3)
# -------------------------
def build_model():
    inp = tf.keras.Input(shape=(SEQ_LEN, 3))
    x = tf.keras.layers.GRU(32, return_sequences=True, dropout=0.1)(inp)
    x = tf.keras.layers.GRU(32, return_sequences=True, dropout=0.1)(x)
    x = tf.keras.layers.GRU(32, return_sequences=False, dropout=0.1)(x)
    out = tf.keras.layers.Dense(H, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.Huber(delta=0.5))
    return model

model = build_model()
model.summary()

# -------------------------
# Cross-validation training
# -------------------------
from sklearn.model_selection import GroupKFold

FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 20
VERBOSE = 2

oof_pred_std = np.zeros_like(y_all, dtype=np.float32)

base_w = np.linspace(1.0, 2.0, COPIES)
base_w = base_w / base_w.mean()

gkf = GroupKFold(n_splits=FOLDS)
for fold, (tr_idx, va_idx) in enumerate(
    gkf.split(train_data, y=train_data["y0"], groups=train_data["group_idx"])
):
    print("\n" + "#"*28)
    print(f"### Fold {fold+1}/{FOLDS}")
    print("#"*28)

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[va_idx], y_all[va_idx]
    w_tr = np.array([base_w[i % COPIES] for i in tr_idx], dtype=np.float32)

    model = build_model()
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        sample_weight=w_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            )
        ]
    )
    oof_pred_std[va_idx] = model.predict(X_va, verbose=0)

# -------------------------
# OOF inversion & quick checks
# -------------------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true).astype(float).ravel()
    y_pred = np.asarray(y_pred).astype(float).ravel()
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1.0
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

oof_log = oof_pred_std * sd + mn
oof_levels = np.expm1(oof_log)
oof_levels = np.clip(oof_levels, 0, None)

print("OOF SMAPE (all H):", smape(y_data, oof_levels))
print("OOF SMAPE (last 2 months):", smape(y_data[:, -2:], oof_levels[:, -2:]))

# If you defined custom_score earlier:
m = custom_score(y_data.flatten(), oof_levels.flatten())
print('custom_score',m['str'])

# -------------------------
# Retrain on ALL training samples
# -------------------------
final_model = build_model()
w_all = np.array([base_w[i % COPIES] for i in range(len(X_all))], dtype=np.float32)
final_model.fit(
    X_all, y_all,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=VERBOSE,
    sample_weight=w_all,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=2, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=3, restore_best_weights=True, verbose=1
        )
    ]
)

# -------------------------
# Inference (months 67..72) for TRAINED sectors only
# -------------------------
# Build test level history for each trained sector
X_test_levels = np.zeros((N_SERIES_TRAIN, pre_len), dtype=float)
for s in range(N_SERIES_TRAIN):
    series = data_levels[s]         # len T
    X_test_levels[s] = series[-pre_len:]

# Build matching month sin/cos for last pre_len times
m_hist = months_in_df[-pre_len:].astype(float)       # length pre_len
test_season = np.zeros((N_SERIES_TRAIN, pre_len, 2), dtype=np.float32)
test_season[:, :, 0] = np.sin(2*np.pi*m_hist/12.0)[None, :]
test_season[:, :, 1] = np.cos(2*np.pi*m_hist/12.0)[None, :]

X_test_log = np.log1p(X_test_levels)
X_test_std = (X_test_log - mn) / sd
X_test_level = X_test_std.astype(np.float32).reshape(N_SERIES_TRAIN, SEQ_LEN, 1)
X_test_all   = np.concatenate([X_test_level, test_season], axis=2)  # (N_SERIES_TRAIN, SEQ_LEN, 3)

pred_std   = final_model.predict(X_test_all, verbose=0)
pred_log   = pred_std * sd + mn
pred_train = np.expm1(pred_log)
pred_train = np.clip(pred_train, 0, None)   # shape: (N_SERIES_TRAIN, H)

# -------------------------
# Assemble full 96-sector predictions for 67..72
# -------------------------
pred_full = np.zeros((len(ALL_SECTORS), H), dtype=float)  # (96 x 6)
sector_to_row = {s: i for i, s in enumerate(ALL_SECTORS)}
for idx, s in enumerate(train_sectors):
    pred_full[sector_to_row[s], :] = pred_train[idx]
for s in zero_sales_sector:
    pred_full[sector_to_row[s], :] = 0.0

# -------------------------
# Build pred_df (time-major), add months 73..78 as zero rows
# -------------------------
future_times = np.arange(times_in_df[-1] + 1, times_in_df[-1] + 1 + H)   # next 6 time ids (e.g., 67..72 if last was 66)
pred_df = pd.DataFrame(pred_full.T, index=[f"time_{t}" for t in future_times], columns=ALL_SECTORS)

# Optional: threshold small values
# pred_df[pred_df < 10] = 0

# add 6 more months as zeros
for t in range(future_times[-1] + 1, future_times[-1] + 1 + 6):  # 6 months: 73..78 typically
    pred_df.loc[f"time_{t}"] = 0

pred_df = pred_df.sort_index()
pred_df = pred_df[ALL_SECTORS]
print("Predictions (time-major) head:")
display(pred_df.head())

# -------------------------
# Submission
# -------------------------
test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
test['new_house_transaction_amount'] = pred_df.T.unstack().values
test[['id', 'new_house_transaction_amount']].to_csv('submission.csv', index=False)
!head submission.csv


