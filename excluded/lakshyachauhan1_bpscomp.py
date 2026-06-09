# Epic ensemble: Synthetic-aware FE -> XGBoost (GPU) + TensorFlow RNN -> Ridge stack -> refit-on-full
# Paste into a Kaggle GPU notebook cell and run.

import os, gc, time, random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb   # optional, not required
import catboost as cb    # optional, not required

# TensorFlow for RNN model
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses, metrics

# ---------- CONFIG ----------
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

TRAIN_PATH = "/kaggle/input/playground-series-s5e9/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e9/test.csv"

NFOLDS = 5
BAGGING = 3                  # seeds per base model (variance reduction)
EARLY_STOP_BOOST = 200
XGB_ROUNDS = 3000

# RNN training controls (can set MAX_RNN_EPOCHS = 20000)
MAX_RNN_EPOCHS = 5
RNN_BATCH = 2048
RNN_LR = 1e-3
RNN_PATIENCE = 800           # patience for early stopping (large for long training)
EMBED_DIM = 64
RNN_HIDDEN = 256
RNN_LAYERS = 2
KBINS = 64                   # bins per feature to convert to tokens for embedding

MIN_BPM, MAX_BPM = 40, 200

DEVICE = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print("Device:", DEVICE)

# ---------- LOAD ----------
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
id_col = train_df.columns[0]
target_col = train_df.columns[-1]
print("Train", train_df.shape, "Test", test_df.shape)

# ---------- FEATURE ENGINEERING ----------
def float32_bits_features(arr):
    """Given 1D numpy float array (float32), return mantissa/exponent/sign features."""
    a32 = arr.astype(np.float32)
    bits = a32.view(np.uint32)
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & ((1 << 23) - 1)
    return sign.astype(np.int32), exponent.astype(np.int32), mantissa.astype(np.int32)

def synthetic_features(df):
    df = df.copy()
    # numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # fill
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # choose columns excluding id/target
    cont_cols = [c for c in num_cols if c not in [id_col, target_col]]

    # mantissa/exponent digits and digit extraction
    for c in cont_cols:
        arr = df[c].to_numpy(dtype=np.float32)
        sgn, ex, man = float32_bits_features(arr)
        df[f"{c}_sign"] = sgn
        df[f"{c}_exp"] = ex
        df[f"{c}_man_lo3"] = (man % 1000).astype(np.int32)        # low 3 mantissa digits
        df[f"{c}_man_mod97"] = (man % 97).astype(np.int32)

        # decimal digit extraction
        df[f"{c}_d3"] = ((df[c] * 1e3).astype(np.int64) % 1000).astype(np.int32)
        df[f"{c}_d6"] = ((df[c] * 1e6).astype(np.int64) % 1000).astype(np.int32)

        # rounding bins
        df[f"{c}_r2"] = np.round(df[c], 2)
        df[f"{c}_r3"] = np.round(df[c], 3)

    # interaction features (limited)
    for i in range(min(6, len(cont_cols))):
        for j in range(i+1, min(6, len(cont_cols))):
            a = cont_cols[i]; b = cont_cols[j]
            df[f"{a}_plus_{b}"]  = df[a] + df[b]
            df[f"{a}_minus_{b}"] = df[a] - df[b]
            df[f"{a}_mul_{b}"]   = df[a] * df[b]

    # row rank statistics
    df['_row_sum'] = df[cont_cols].sum(axis=1)
    df['_row_mean'] = df[cont_cols].mean(axis=1)
    df['_row_std'] = df[cont_cols].std(axis=1)
    return df

print("Applying synthetic FE...")
train_fe = synthetic_features(train_df)
test_fe  = synthetic_features(test_df)
print("After FE:", train_fe.shape, test_fe.shape)

# ---------- K-Fold target encoding for selected derived categorical-ish features ----------
def kfold_target_encode(train, test, col, target, n_splits=NFOLDS, alpha=20):
    oof = np.zeros(len(train), dtype=np.float32)
    te_test = np.zeros(len(test), dtype=np.float32)
    global_mean = train[target].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in kf.split(train):
        tr = train.iloc[tr_idx]
        val = train.iloc[val_idx]
        stats = tr.groupby(col)[target].agg(['mean','count'])
        smooth = (stats['mean'] * stats['count'] + global_mean * alpha) / (stats['count'] + alpha)
        mapping = smooth.to_dict()
        oof[val_idx] = val[col].map(mapping).fillna(global_mean)
    # full mapping
    stats_full = train.groupby(col)[target].agg(['mean','count'])
    smooth_full = (stats_full['mean'] * stats_full['count'] + global_mean * alpha) / (stats_full['count'] + alpha)
    te_test = test[col].map(smooth_full.to_dict()).fillna(global_mean)
    return oof, te_test

# candidate TE cols: the digit / mantissa fields
cand_te = [c for c in train_fe.columns if c.endswith('_d3') or c.endswith('_d6') or c.endswith('_man_lo3') or c.endswith('_man_mod97') or c.endswith('_exp')]
cand_te = [c for c in cand_te if c in train_fe.columns]
print("TE candidates:", len(cand_te))

# choose up to 12 TE columns
te_cols = cand_te[:12]
print("Applying KFold TE on:", te_cols)
for c in te_cols:
    oof, test_enc = kfold_target_encode(train_fe, test_fe, c, target_col, n_splits=NFOLDS, alpha=20)
    train_fe[f"te_{c}"] = oof
    test_fe[f"te_{c}"] = test_enc

# ---------- select numeric features and prepare tokenization for RNN ----------
features = [c for c in train_fe.columns if c not in [id_col, target_col]]
# drop constants
features = [c for c in features if train_fe[c].nunique()>1]
print("Selected features:", len(features))

# standard scale continuous features for tree and continuous branch of RNN
scaler = StandardScaler()
X = scaler.fit_transform(train_fe[features].values.astype(np.float32))
X_test = scaler.transform(test_fe[features].values.astype(np.float32))
y = train_fe[target_col].values.astype(np.float32)
test_ids = test_fe[id_col].values

# ---------- discretize to produce tokens for sequence input ----------
kb = KBinsDiscretizer(n_bins=KBINS, encode='ordinal', strategy='quantile')
kb.fit(train_fe[features].values)   # fit on train
X_bins = kb.transform(train_fe[features].values).astype(np.int32)   # shape (n, n_features)
X_test_bins = kb.transform(test_fe[features].values).astype(np.int32)

# convert to global token ids: token_id = feature_index * KBINS + bin_index  (reserve 0 for padding if desired)
n_features = X_bins.shape[1]
token_base = np.arange(n_features) * KBINS    # shape (n_features,)
X_tokens = (X_bins + token_base).astype(np.int32)               # tokens per feature
X_test_tokens = (X_test_bins + token_base).astype(np.int32)
VOCAB_SIZE = int(n_features * KBINS) + 2    # +1 safety, +1 padding

print("Token vocab size:", VOCAB_SIZE, "seq length:", n_features)

# ---------- K-Fold CV: train XGBoost + TensorFlow RNN, produce OOF predictions ----------
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

oof_xgb = np.zeros(len(X), dtype=np.float32)
oof_rnn = np.zeros(len(X), dtype=np.float32)
test_xgb_preds = np.zeros((X_test.shape[0], NFOLDS), dtype=np.float32)
test_rnn_preds = np.zeros((X_test.shape[0], NFOLDS), dtype=np.float32)

xgb_histories = []
rnn_histories = []

fold = 0
t0 = time.time()
for tr_idx, val_idx in kf.split(X, y):
    fold += 1
    print(f"\n=== FOLD {fold}/{NFOLDS} ===")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    tokens_tr, tokens_val = X_tokens[tr_idx], X_tokens[val_idx]

    # ----- XGBoost (GPU) -----
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    xgb_params = {
        "objective":"reg:squarederror",
        "eval_metric":"rmse",
        "tree_method":"hist",    # new GPU API
        "device":"cuda",
        "eta":0.01,
        "max_depth":6,
        "subsample":0.8,
        "colsample_bytree":0.8,
        "lambda":3.0,
        "alpha":1.0,
        "seed": SEED + fold
    }
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(xgb_params, dtrain, num_boost_round=XGB_ROUNDS, evals=watchlist, early_stopping_rounds=EARLY_STOP_BOOST, verbose_eval=200)
    best_it = getattr(bst, "best_iteration", None) or XGB_ROUNDS
    print("XGB best_iter:", best_it)
    oof_xgb[val_idx] = bst.predict(dval, iteration_range=(0, best_it))
    test_xgb_preds[:, fold-1] = bst.predict(xgb.DMatrix(X_test), iteration_range=(0, best_it))
    try:
        hist = bst.evals_result()
        if 'val' in hist and 'rmse' in hist['val']:
            xgb_histories.append(np.array(hist['val']['rmse'], dtype=float))
    except Exception:
        pass
    del bst; gc.collect()

    # ----- TensorFlow RNN model -----
    # build sequence + continuous model
    seq_len = n_features
    tf.keras.backend.clear_session()
    # token input (sequence)
    token_input = layers.Input(shape=(seq_len,), dtype="int32", name="tokens")
    emb = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, name="embed")(token_input)
    # RNN stack
    x = emb
    for i in range(RNN_LAYERS):
        return_sequences = (i < RNN_LAYERS - 1)
        x = layers.GRU(RNN_HIDDEN, return_sequences=return_sequences, dropout=0.2, recurrent_dropout=0.0, name=f"gru_{i+1}")(x)
    rnn_out = layers.BatchNormalization(name="bn_rnn")(x)  # shape (batch, hidden)

    # continuous numeric branch
    cont_input = layers.Input(shape=(X_tr.shape[1],), dtype="float32", name="cont")
    c = layers.Dense(512, activation="relu")(cont_input)
    c = layers.BatchNormalization()(c)
    c = layers.Dropout(0.2)(c)
    # combine
    merged = layers.Concatenate()([rnn_out, c])
    h = layers.Dense(512, activation="relu")(merged)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation="relu")(h)
    out = layers.Dense(1, activation="linear")(h)

    model = models.Model(inputs=[token_input, cont_input], outputs=out)
    # use RMSE as loss (user asked), although MSE is standard; RMSE is differentiable via sqrt
    def rmse_tf(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    opt = optimizers.Adam(learning_rate=RNN_LR)
    model.compile(optimizer=opt, loss=rmse_tf, metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

    # callbacks
    ckpt_path = f"/kaggle/working/rnn_fold{fold}.weights.h5"
    cb_list = [
        callbacks.EarlyStopping(monitor='val_rmse', patience=RNN_PATIENCE, mode='min', restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_rmse', save_best_only=True, save_weights_only=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=0.5, patience=200, min_lr=1e-6, verbose=0)
    ]

    # fit
    history = model.fit(
        {"tokens": tokens_tr, "cont": X_tr},
        y_tr,
        validation_data=({"tokens": tokens_val, "cont": X_val}, y_val),
        epochs=MAX_RNN_EPOCHS,
        batch_size=RNN_BATCH,
        callbacks=cb_list,
        verbose=2
    )
    # best val RMSE
    val_rmse_series = history.history.get('val_rmse', [])
    if len(val_rmse_series)>0:
        best_epoch = int(np.argmin(val_rmse_series)) + 1
    else:
        best_epoch = len(val_rmse_series)
    print("RNN best_epoch:", best_epoch, "best_val_rmse:", np.min(val_rmse_series) if len(val_rmse_series)>0 else None)
    rnn_histories.append(np.array(val_rmse_series))

    # OOF and test preds
    val_preds = model.predict({"tokens": tokens_val, "cont": X_val}, batch_size=RNN_BATCH).reshape(-1)
    test_preds = model.predict({"tokens": X_test_tokens, "cont": X_test}, batch_size=RNN_BATCH).reshape(-1)
    oof_rnn[val_idx] = val_preds
    test_rnn_preds[:, fold-1] = test_preds

    # cleanup
    del model, history; gc.collect()
    tf.keras.backend.clear_session()

print("\nCV folds done in {:.2f} min".format((time.time()-t0)/60.0))

# ---------- Aggregate test fold preds (mean across folds) ----------
test_xgb_mean = test_xgb_preds.mean(axis=1)
test_rnn_mean = test_rnn_preds.mean(axis=1)

# ---------- OOF diagnostics ----------
print("OOF RMSE XGB:", mean_squared_error(y, oof_xgb, squared=False))
print("OOF RMSE RNN:", mean_squared_error(y, oof_rnn, squared=False))

# ---------- Meta-learner (Ridge) ----------
stack_oof = np.vstack([oof_xgb, oof_rnn]).T
stack_test = np.vstack([test_xgb_mean, test_rnn_mean]).T

meta = RidgeCV(alphas=[0.01,0.1,1.0,10.0], cv=5, scoring='neg_root_mean_squared_error')
meta.fit(stack_oof, y)
meta_oof = meta.predict(stack_oof)
print("Meta OOF RMSE:", mean_squared_error(y, meta_oof, squared=False), "coefs:", meta.coef_)

# ---------- Refit on full data ----------
print("Refitting XGBoost on full data (using avg best iteration across folds)...")
# compute average best iterations from CV histories (if available) - fallback to XGB_ROUNDS * 0.6
best_its = []
for hist in xgb_histories:
    if len(hist)>0:
        best_its.append(int(np.argmin(hist))+1)
if len(best_its)>0:
    avg_best_it = int(np.mean(best_its))
else:
    avg_best_it = int(XGB_ROUNDS * 0.6)
print("Using xgb num_boost_round:", avg_best_it)

dtrain_full = xgb.DMatrix(X, label=y)
xgb_params_full = dict(xgb_params)
bst_xgb_full = xgb.train(xgb_params_full, dtrain_full, num_boost_round=avg_best_it, verbose_eval=200)
pred_test_xgb_full = bst_xgb_full.predict(xgb.DMatrix(X_test))

# Refit RNN on full data: train for average best epoch or a safe number (use averaged best epoch across folds)
if len(rnn_histories)>0 and any(len(h)>0 for h in rnn_histories):
    avg_rnn_epoch = int(np.mean([np.argmin(h)+1 for h in rnn_histories if len(h)>0]))
    refit_epochs = max(5, int(avg_rnn_epoch * 1.0))  # train that many on full
else:
    refit_epochs = min(50, MAX_RNN_EPOCHS)  # safe fallback
print("Refitting RNN on full data for epochs:", refit_epochs)

# build model again
tf.keras.backend.clear_session()
token_input = layers.Input(shape=(n_features,), dtype="int32", name="tokens")
emb = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, name="embed")(token_input)
x = emb
for i in range(RNN_LAYERS):
    return_sequences = (i < RNN_LAYERS - 1)
    x = layers.GRU(RNN_HIDDEN, return_sequences=return_sequences, dropout=0.2, recurrent_dropout=0.0, name=f"gru_{i+1}")(x)
rnn_out = layers.BatchNormalization()(x)
cont_input = layers.Input(shape=(X.shape[1],), dtype="float32", name="cont")
c = layers.Dense(512, activation="relu")(cont_input)
c = layers.BatchNormalization()(c)
c = layers.Dropout(0.2)(c)
merged = layers.Concatenate()([rnn_out, c])
h = layers.Dense(512, activation="relu")(merged)
h = layers.BatchNormalization()(h)
h = layers.Dropout(0.2)(h)
h = layers.Dense(128, activation="relu")(h)
out = layers.Dense(1, activation="linear")(h)
rnn_full = models.Model(inputs=[token_input, cont_input], outputs=out)
rnn_full.compile(optimizer=optimizers.Adam(learning_rate=RNN_LR), loss=lambda y_true,y_pred: tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred))))
rnn_full.fit({"tokens": X_tokens, "cont": X}, y, epochs=refit_epochs, batch_size=RNN_BATCH, verbose=2)
pred_test_rnn_full = rnn_full.predict({"tokens": X_test_tokens, "cont": X_test}, batch_size=RNN_BATCH).reshape(-1)

# final stack features
stack_final = np.vstack([pred_test_xgb_full, pred_test_rnn_full]).T
final_preds = meta.predict(stack_final)
final_preds = np.clip(final_preds, MIN_BPM, MAX_BPM)

# save submission
out_df = pd.DataFrame({id_col: test_ids, 'BeatsPerMinute': final_preds})
out_df.to_csv("submission_rnn_xgb_stack.csv", index=False)
print("Saved submission_rnn_xgb_stack.csv")

# ---------- Final diagnostics ----------
print("Final meta coefficients:", meta.coef_)
print("CV meta OOF RMSE:", mean_squared_error(y, meta_oof, squared=False))

# plot averaged histories
plt.figure(figsize=(12,6))
if len(xgb_histories)>0:
    minlen = min(len(h) for h in xgb_histories if len(h)>0)
    mean_xgb = np.mean([h[:minlen] for h in xgb_histories if len(h)>=minlen], axis=0)
    plt.plot(np.arange(1,len(mean_xgb)+1), mean_xgb, label='XGB val RMSE')
if len(rnn_histories)>0:
    minlen = min(len(h) for h in rnn_histories if len(h)>0)
    mean_rnn = np.mean([h[:minlen] for h in rnn_histories if len(h)>=minlen], axis=0)
    plt.plot(np.linspace(1, len(mean_rnn), len(mean_rnn)), mean_rnn, label='RNN val RMSE')
plt.xlabel("Iteration / Epoch"); plt.ylabel("Val RMSE"); plt.legend(); plt.grid(True); plt.show()


