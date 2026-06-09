# ======================
# å…±é€šã‚»ãƒƒãƒˆã‚¢ãƒƒãƒ—
# ======================
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

import os, warnings
warnings.filterwarnings("ignore")

print("Input files:", os.listdir("/kaggle/input"))



# ======================
# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
# ======================
train_raw = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test_raw = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

display(train_raw.head())

def add_simple_time_features(df):
    """æ™‚é–“ç‰¹å¾´é‡�"""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    return df

train_simple = add_simple_time_features(train_raw)
test_simple = add_simple_time_features(test_raw)

feature_cols_simple = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "month"
]


# ======================
# ãƒ©ãƒ³ãƒ€ãƒ åˆ†å‰² + RandomForest ãƒ™ãƒ¼ã‚¹ãƒ©ã‚¤ãƒ³
# ======================
X_simple = train_simple[feature_cols_simple]
y_simple = train_simple["count"]

# â€» ã�“ã�®æ™‚ç‚¹ã�§ã�¯æ™‚ç³»åˆ—æ€§ã‚’æ„�è­˜ã�›ã�šã€�ãƒ©ãƒ³ãƒ€ãƒ åˆ†å‰²ã�—ã�¦ã�„ã‚‹
X_tr, X_val, y_tr, y_val = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

rf_baseline = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

rf_baseline.fit(X_tr, y_tr)
pred_val = rf_baseline.predict(X_val)

rmsle_baseline = np.sqrt(mean_squared_log_error(y_val, pred_val))
print(f"[è‡ªåŠ›ãƒ•ã‚§ãƒ¼ã‚º] RandomForest (random split) RMSLE = {rmsle_baseline:.4f}")


# ======================
# 3. ãƒ‡ãƒ¼ã‚¿å†�èª­ã�¿è¾¼ã�¿ ï¼‹ æ™‚é–“ç‰¹å¾´é‡�
# ======================
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

def add_datetime_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday  # 0: Monday, ..., 6: Sunday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    # é€šå‹¤æ™‚é–“å¸¯ï¼ˆæœ�å¤•ï¼‰ãƒ•ãƒ©ã‚°ï¼šä»®èª¬ã�«åŸºã�¥ã��æ‰‹ä½œã‚Šç‰¹å¾´é‡�
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    return df

train = add_datetime_features(train)
test = add_datetime_features(test)

display(train.head())


# ======================
# EDA
# ======================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# æ™‚é–“å¸¯åˆ¥å¹³å�‡éœ€è¦�
sns.lineplot(ax=axes[0, 0], data=train, x="hour", y="count")
axes[0, 0].set_title("Average count by hour")

# å¹³æ—¥ / é��å¹³æ—¥ Ã— æ™‚é–“
sns.lineplot(ax=axes[0, 1], data=train, x="hour", y="count", hue="workingday")
axes[0, 1].set_title("Count by hour & workingday")

# æ°—æ¸© vs éœ€è¦�ï¼ˆã‚µãƒ³ãƒ—ãƒ«æŠ½å‡ºï¼‰
sns.scatterplot(ax=axes[1, 0],
                data=train.sample(5000, random_state=42),
                x="temp", y="count", alpha=0.3)
axes[1, 0].set_title("Temp vs count")

# casual / registered ã�®æ™‚é–“å¸¯åˆ¥
train_melt = train.melt(id_vars=["hour"], value_vars=["casual", "registered"],
                        var_name="user_type", value_name="users")
sns.lineplot(ax=axes[1, 1], data=train_melt, x="hour", y="users", hue="user_type")
axes[1, 1].set_title("Casual vs Registered by hour")

plt.tight_layout()
plt.show()


# ======================
# è©•ä¾¡æŒ‡æ¨™ã�¨CVè¨­å®š
# ======================

def rmsle(y_true, y_pred):
    """RMSLE ã‚’è¨ˆç®—ã�™ã‚‹ãƒ˜ãƒ«ãƒ‘ãƒ¼é–¢æ•°"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# æœ¬ãƒ•ã‚§ãƒ¼ã‚ºã�§ã�¯æ™‚ç³»åˆ—ã‚’æ„�è­˜ã�—ã�Ÿåˆ†å‰²ã‚’æ�¡ç”¨
#tscv = TimeSeriesSplit(n_splits=5) ç‰¹å®šã�®ãƒ†ã‚¹ãƒˆã�§0ã�¨äºˆæ¸¬ã�—å¤§ã��ã��ç²¾åº¦ã�Œæ¸›å°‘ã�—ã�Ÿï¼ˆæ™‚ç³»åˆ—é–¢ä¿‚ã�Œå¼±ã�„ï¼‰

from sklearn.model_selection import KFold
tscv = KFold(n_splits=5, shuffle=True, random_state=42)

# åŸºæœ¬ç‰¹å¾´é‡�ã‚»ãƒƒãƒˆ
base_features = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "year", "month", "day", "hour", "weekday",
    "is_weekend", "is_rush_hour"
]

X_base = train[base_features]
y_count = train["count"].values
y_log = np.log1p(y_count)  # RMSLE ã�¨æ•´å�ˆçš„ã�ª log1p å¤‰æ�›


# ======================
# å®Ÿé¨“Aï¼šLinear Regression
# ======================
oof_lr = np.zeros(len(train))

for fold, (trn_idx, val_idx) in enumerate(tscv.split(X_base, y_log)):
    X_tr, X_val = X_base.iloc[trn_idx], X_base.iloc[val_idx]
    y_tr, y_val = y_log[trn_idx], y_log[val_idx]
    
    model_lr = LinearRegression()
    model_lr.fit(X_tr, y_tr)
    
    pred_val = model_lr.predict(X_val)
    oof_lr[val_idx] = pred_val
    
    rmse_log = np.sqrt(np.mean((pred_val - y_val) ** 2))
    print(f"[Linear] Fold {fold}: RMSE on log1p(count) = {rmse_log:.4f}")

rmsle_lr = np.sqrt(np.mean((oof_lr - y_log) ** 2))
print(f"[Linear] CV RMSLE (approx) = {rmsle_lr:.4f}")



# ======================
# å®Ÿé¨“Bï¼šRandomForestï¼ˆTimeSeriesSplitï¼‰
# ======================
rf_params = dict(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

oof_rf = np.zeros(len(train))

for fold, (trn_idx, val_idx) in enumerate(tscv.split(X_base, y_log)):
    X_tr, X_val = X_base.iloc[trn_idx], X_base.iloc[val_idx]
    y_tr, y_val = y_log[trn_idx], y_log[val_idx]
    
    model_rf = RandomForestRegressor(**rf_params)
    model_rf.fit(X_tr, y_tr)
    
    pred_val = model_rf.predict(X_val)
    oof_rf[val_idx] = pred_val
    
    rmse_log = np.sqrt(np.mean((pred_val - y_val) ** 2))
    print(f"[RF] Fold {fold}: RMSE on log1p(count) = {rmse_log:.4f}")

rmsle_rf = np.sqrt(np.mean((oof_rf - y_log) ** 2))
print(f"[RF] CV RMSLE (approx) = {rmsle_rf:.4f}")



print("train index head:", train.index[:10])
print("X_base index head:", X_base.index[:10])



print("Are indices equal? ->", np.array_equal(train.index, X_base.index))



print("oof_min_max:", oof_rf.min(), oof_rf.max())
print("y_log_min_max:", y_log.min(), y_log.max())



errors = np.abs(oof_rf - y_log)
print("max error:", errors.max())
print("argmax:", errors.argmax())
print("y_log at argmax:", y_log[errors.argmax()])
print("oof at argmax:", oof_rf[errors.argmax()])



# ======================
# LightGBMç”¨ã�®è¿½åŠ ç‰¹å¾´é‡�
# ======================
def add_extra_features(df):
    df = df.copy()
    
    # æ™‚é–“å¸¯ã‚’4åŒºåˆ†ã�«ã�¾ã�¨ã‚�ã‚‹
    def hour_to_period(h):
        if 6 <= h <= 9:
            return 0  # æœ�
        elif 10 <= h <= 16:
            return 1  # æ˜¼
        elif 17 <= h <= 20:
            return 2  # å¤•æ–¹
        else:
            return 3  # å¤œ
    
    df["hour_period"] = df["hour"].apply(hour_to_period)
    df["temp_bin"] = (df["temp"] // 5).astype(int)
    df["humidity_bin"] = (df["humidity"] // 10).astype(int)
    df["year_month"] = df["year"] * 100 + df["month"]
    
    return df

train_lgb = add_extra_features(train)
test_lgb = add_extra_features(test)

lgb_features = base_features + ["hour_period", "temp_bin", "humidity_bin", "year_month"]

X_lgb = train_lgb[lgb_features]
X_test_lgb = test_lgb[lgb_features]



# ======================
# LightGBM CV
# ======================
oof_lgb = np.zeros(len(train))
test_pred_lgb_log = np.zeros(len(test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}

for fold, (trn_idx, val_idx) in enumerate(tscv.split(X_lgb, y_log)):
    X_tr, X_val = X_lgb.iloc[trn_idx], X_lgb.iloc[val_idx]
    y_tr, y_val = y_log[trn_idx], y_log[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        # â˜… ã�“ã�“ã�§ early stopping & ãƒ­ã‚°å‡ºåŠ›ã‚’ callback ã�¨ã�—ã�¦æ¸¡ã�™
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )

    pred_val = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    oof_lgb[val_idx] = pred_val

    test_pred_lgb_log += model_lgb.predict(
        X_test_lgb,
        num_iteration=model_lgb.best_iteration
    ) / tscv.n_splits

    rmse_log = np.sqrt(np.mean((pred_val - y_val) ** 2))
    print(f"[LGB] Fold {fold}: RMSE on log1p(count) = {rmse_log:.4f}")

rmsle_lgb = np.sqrt(np.mean((oof_lgb - y_log) ** 2))
print(f"[LGB] CV RMSLE (approx) = {rmsle_lgb:.4f}")



errors = np.abs(oof_rf - y_log)
print("max error:", errors.max())
print("argmax:", errors.argmax())
print("y_log at argmax:", y_log[errors.argmax()])
print("oof at argmax:", oof_rf[errors.argmax()])



# ======================
# Feature importance ã�®ç¢ºèª�
# ======================
importance = model_lgb.feature_importance(importance_type="gain")
feature_importance = pd.DataFrame({
    "feature": X_lgb.columns,
    "importance": importance
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8, 10))
sns.barplot(data=feature_importance.head(20), x="importance", y="feature")
plt.title("LightGBM Feature Importance (top 20)")
plt.tight_layout()
plt.show()

feature_importance.head(20)



# ============================
# casual & registered å�Œæ™‚CVï¼ˆæ­£ã�—ã�„æ–¹æ³•ï¼‰
# ============================

y_casual_log = np.log1p(train_lgb["casual"].values)
y_reg_log    = np.log1p(train_lgb["registered"].values)

oof_casual_log = np.zeros(len(train))
oof_reg_log    = np.zeros(len(train))

test_pred_casual_log = np.zeros(len(test))
test_pred_reg_log    = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(tscv.split(X_lgb)):

    print(f"\n===== Fold {fold} =====")

    X_tr, X_val = X_lgb.iloc[trn_idx], X_lgb.iloc[val_idx]

    y_tr_c, y_val_c = y_casual_log[trn_idx], y_casual_log[val_idx]
    y_tr_r, y_val_r = y_reg_log[trn_idx],  y_reg_log[val_idx]

    # ---------- casual ----------
    dtrain_c = lgb.Dataset(X_tr, label=y_tr_c)
    dvalid_c = lgb.Dataset(X_val, label=y_val_c, reference=dtrain_c)

    model_c = lgb.train(
        lgb_params,
        dtrain_c,
        valid_sets=[dtrain_c, dvalid_c],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )

    pred_val_c = model_c.predict(X_val, num_iteration=model_c.best_iteration)
    oof_casual_log[val_idx] = pred_val_c

    test_pred_casual_log += model_c.predict(
        X_test_lgb, num_iteration=model_c.best_iteration
    ) / tscv.n_splits

    # ---------- registered ----------
    dtrain_r = lgb.Dataset(X_tr, label=y_tr_r)
    dvalid_r = lgb.Dataset(X_val, label=y_val_r, reference=dtrain_r)

    model_r = lgb.train(
        lgb_params,
        dtrain_r,
        valid_sets=[dtrain_r, dvalid_r],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )

    pred_val_r = model_r.predict(X_val, num_iteration=model_r.best_iteration)
    oof_reg_log[val_idx] = pred_val_r

    test_pred_reg_log += model_r.predict(
        X_test_lgb, num_iteration=model_r.best_iteration
    ) / tscv.n_splits

# ---------- count ã�¸æˆ»ã�—ã�¦ RMSLE è©•ä¾¡ ----------
oof_count = np.expm1(oof_casual_log) + np.expm1(oof_reg_log)
rmsle_count = rmsle(train["count"].values, oof_count)

print("\n===============================")
print(f"[casual+registered] CV RMSLE = {rmsle_count:.4f}")
print("===============================")



# ======================
# Feature importance ã�®ç¢ºèª�
# ======================
importance = model_lgb.feature_importance(importance_type="gain")
feature_importance = pd.DataFrame({
    "feature": X_lgb.columns,
    "importance": importance
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8, 10))
sns.barplot(data=feature_importance.head(20), x="importance", y="feature")
plt.title("LightGBM Feature Importance (top 20)")
plt.tight_layout()
plt.show()

feature_importance.head(20)



from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd


# å¤•æ–¹ã‚¹ãƒ‘ã‚¤ã‚¯ç¢ºèª�ç”¨ã�®åˆ—
train["log_count"] = np.log1p(train["count"])

# TSCV ã�®è¨­å®š
tscv = TimeSeriesSplit(n_splits=5)

# Foldã�”ã�¨ã�«ã€�train/val ã�® count åˆ†å¸ƒã‚’ç¢ºèª�
for fold, (trn_idx, val_idx) in enumerate(tscv.split(train)):

    print(f"\n========================")
    print(f"Fold {fold}")
    print("========================")

    trn = train.iloc[trn_idx]
    val = train.iloc[val_idx]

    # ã�©ã�®æ™‚é–“å¸¯ã�Œå�«ã�¾ã‚Œã�¦ã�„ã‚‹ã�‹
    print("Train hour distribution:")
    print(trn["hour"].value_counts().sort_index())

    print("\nVal hour distribution:")
    print(val["hour"].value_counts().sort_index())

    # æœ€å¤§å€¤ã�®ç¢ºèª�ï¼ˆã‚¹ãƒ‘ã‚¤ã‚¯æ¤œå‡ºï¼‰
    max_trn = trn["log_count"].max()
    max_val = val["log_count"].max()
    idx_trn = trn["log_count"].idxmax()
    idx_val = val["log_count"].idxmax()

    print(f"\nMax log_count in TRAIN: {max_trn:.4f}  (index={idx_trn})")
    print(f"Max log_count in VAL  : {max_val:.4f}  (index={idx_val})")

    # è©²å½“è¡Œã�® hour ã‚‚ç¢ºèª�
    print("TRAIN spike hour:", trn.loc[idx_trn, "hour"])
    print("VAL spike hour  :", val.loc[idx_val, "hour"])

    # å¤•æ–¹ã‚¹ãƒ‘ã‚¤ã‚¯ã�Œ val ã�«ã�—ã�‹ã�ªã�„ã�‹ç¢ºèª�
    print("\nSpike difference (val - train):", max_val - max_trn)




# ======================
# æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ä½œæˆ�
# ======================

# æœ€çµ‚çš„ã�«ã�¯ casual / registered ãƒ¢ãƒ‡ãƒ«ã�®å�ˆç®—ã‚’æ�¡ç”¨
test_pred_count = np.expm1(test_pred_casual_log) + np.expm1(test_pred_reg_log)
test_pred_count = np.clip(test_pred_count, 0, None)

submission = pd.DataFrame({
    "datetime": test_raw["datetime"],
    "count": test_pred_count
})

submission.to_csv("submission.csv", index=False)
submission.head()


