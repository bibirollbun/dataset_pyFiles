import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


ROOT_DIR = "/kaggle/input/jane-street-real-time-market-data-forecasting"

# === ê²½ë¡œ ===
train_path = os.path.join(ROOT_DIR, "train.parquet")

# === ë©”íƒ€ë�°ì�´í„° ===
lags = os.path.join(ROOT_DIR, "lags.parquet")
train_path = os.path.join(ROOT_DIR, "train.parquet")
lags_path  = os.path.join(ROOT_DIR, "lags.parquet")
features   = pl.read_csv(os.path.join(ROOT_DIR, "features.csv"))
responders = pl.read_csv(os.path.join(ROOT_DIR, "responders.csv"))
sample_sub = pl.read_csv(os.path.join(ROOT_DIR, "sample_submission.csv"))


file_path = os.path.join(ROOT_DIR, "train.parquet", "partition_id=0", "part-0.parquet")
data_0 = pl.read_parquet(file_path)

# responder ê³„ì—´ ì»¬ëŸ¼ í™•ì�¸
resp_cols = [c for c in data_0.columns if c.startswith("responder_")]

# responder_6ë§Œ ë‚¨ê¸°ê³  ë‚˜ë¨¸ì§€ drop
resp_drop = [c for c in resp_cols if c != "responder_6"]
data_0 = data_0.drop(resp_drop)


# ì‚­ì œí•  feature ì»¬ëŸ¼ë“¤ : ê²°ì¸¡ì¹˜ ë§�ì�€ ì• ë“¤
drop_features = ["feature_00", "feature_01", "feature_02", "feature_03","feature_04", "feature_26", "feature_27", "feature_31"]

data_0 = data_0.drop(drop_features)

# ë²„ì „1 ë�°ì�´í„°ì…‹ êµ¬ì„± ë¯¸ë¦¬ë³´ê¸° : x ë�°ì�´í„°ì—� featureë§Œ ë‚¨ê¸°ê¸°
data_0_y = data_0["responder_6"]
data_0_X = data_0.drop("responder_6","date_id","time_id","symbol_id", "weight" )


data_0_X


import lightgbm as lgb
import pandas as pd

# Polars â†’ Pandas
X = data_0_X.to_pandas()
y = data_0_y.to_pandas()

# LightGBM ëª¨ë�¸ (íšŒê·€ ì˜ˆì‹œ)
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# í•™ìŠµ (trainë§Œ)
model.fit(X, y)

# í”¼ì²˜ ì¤‘ìš”ë�„ ì¶”ì¶œ
booster = model.booster_
imp_df = pd.DataFrame({
    "feature": model.feature_name_,
    "gain": booster.feature_importance(importance_type="gain"),
    "split": booster.feature_importance(importance_type="split"),
})
imp_df["gain_norm"] = imp_df["gain"] / (imp_df["gain"].sum() + 1e-12)
imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)

print(imp_df.head(30))


# ì²« ë²ˆì§¸ ì½”ë“œì—�ì„œ ì •ì�˜í•œ drop_features ì�¬ì‚¬ìš©
drop_features = ["feature_00", "feature_01", "feature_02", "feature_03",
                 "feature_04", "feature_26", "feature_27", "feature_31"]

# Xì—�ì„œ ì œì™¸í•  ë©”íƒ€ ì»¬ëŸ¼ (ì²« ë²ˆì§¸ ì½”ë“œë°•ìŠ¤ ê¸°ì¤€)
meta_cols = ["date_id", "time_id", "symbol_id", "weight"]

# ê²°ê³¼ ì €ì�¥ dict
high_features_dict = {} 

for pid in range(7):  # partition_id=0~6
    print(f"=== Partition {pid} ===")

    # --------------------
    # 1) ë�°ì�´í„° ë¡œë“œ
    # --------------------
    f = os.path.join(ROOT_DIR, "train.parquet", f"partition_id={pid}", "part-0.parquet")
    df = pl.read_parquet(f)

    # responder ì²˜ë¦¬ (responder_6ë§Œ ë‚¨ê¸°ê¸°)
    df = df.drop([c for c in df.columns if c.startswith("responder_") and c != "responder_6"])

    # ê²°ì¸¡ì¹˜ ë§�ì�€ feature drop
    df = df.drop(drop_features)

    # --------------------
    # 2) X / y ì¤€ë¹„ (ì²« ë²ˆì§¸ ì½”ë“œ ë¡œì§� ì�¬ì‚¬ìš©)
    # --------------------
    df_y = df["responder_6"]
    df_X = df.drop(["responder_6"] + [c for c in meta_cols if c in df.columns])

    X = df_X.to_pandas()
    y = df_y.to_pandas()

    # --------------------
    # 3) LightGBM í•™ìŠµ (partitionë³„)
    # --------------------
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    booster = model.booster_
    imp_df = pd.DataFrame({
        "feature": model.feature_name_,
        "gain": booster.feature_importance(importance_type="gain"),
    })
    imp_df["gain_norm"] = imp_df["gain"] / (imp_df["gain"].sum() + 1e-12)
    imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)

    # --------------------
    # 4) ìƒ�ìœ„ 30ê°œ feature ì¶”ì¶œ (ë¦¬ìŠ¤íŠ¸ë§Œ ì €ì�¥)
    TOP_N = 30
    high_features = imp_df["feature"].iloc[:TOP_N].tolist()

    # íŒŒí‹°ì…˜ë³„ë¡œ ë¦¬ìŠ¤íŠ¸ ì €ì�¥
    high_features_dict[f"high_features_{pid}"] = high_features

    print(f"â†’ high_features_{pid}: {len(high_features)} features")

    # ì�´ì œ high_features_dict["high_features_0"], high_features_dict["high_features_1"], ... ì ‘ê·¼ ê°€ëŠ¥


print(high_features_dict.keys())


# ëª¨ë“  high_features ë¦¬ìŠ¤íŠ¸ì�˜ êµ�ì§‘í•© êµ¬í•˜ê¸°
common_features = set(high_features_dict["high_features_0"])

for pid in range(1, 7):  # 1~6ê¹Œì§€
    common_features &= set(high_features_dict[f"high_features_{pid}"])

common_features = list(common_features)

print(f"ëª¨ë“  íŒŒí‹°ì…˜ì—� ê³µí†µìœ¼ë¡œ ë“±ì�¥í•œ feature ê°œìˆ˜: {len(common_features)}")
print("ê³µí†µ feature ëª©ë¡�:", common_features)

from collections import Counter

# ëª¨ë“  í”¼ì²˜ë¥¼ í•˜ë‚˜ì�˜ ë¦¬ìŠ¤íŠ¸ë¡œ í•©ì¹˜ê¸°
all_features = []
for pid in range(7):
    all_features.extend(high_features_dict[f"high_features_{pid}"])

# ë“±ì�¥ íšŸìˆ˜ ì„¸ê¸°
feature_counts = Counter(all_features)

# 3ê°œ ì�´ìƒ� ë¦¬ìŠ¤íŠ¸ì—�ì„œ ë“±ì�¥í•œ í”¼ì²˜ë§Œ ì¶”ì¶œ
common_3plus_features = [f for f, cnt in feature_counts.items() if cnt >= 3]

print(f"3ê°œ ì�´ìƒ� ë¦¬ìŠ¤íŠ¸ì—� ë‚˜íƒ€ë‚œ feature ê°œìˆ˜: {len(common_3plus_features)}")
print("feature ëª©ë¡�:", common_3plus_features)


# ===============================
# A) ê²°ì¸¡ì¹˜ í™•ì�¸ (NaN Audit)
# - ë‹¨ìœ„: partition_id = 0 ~ 6 (ê°� íŒŒí‹°ì…˜ë³„ë¡œ í™•ì�¸)
# - ëŒ€ìƒ�: FEATURES (í•™ìŠµì—� ì“¸ í”¼ì²˜ ë¦¬ìŠ¤íŠ¸)
# - ì§€í‘œ: í”¼ì²˜ë³„ NaN ë¹„ìœ¨ì�˜ 'í�‰ê· (mean)'ì�„ ì£¼ë¡œ ë³¸ë‹¤
# ===============================

import os
import polars as pl
import pandas as pd

ROOT_DIR = "/kaggle/input/jane-street-real-time-market-data-forecasting"

# âœ… í•™ìŠµì—� ì“¸ í”¼ì²˜ ë¦¬ìŠ¤íŠ¸ë§Œ ë„£ì–´ì£¼ë©´ ë�¨ (Top20 ë˜�ëŠ” êµ�ì§‘í•©20 ì¤‘ íƒ�1)
# ì˜ˆ) FEATURES = top20_features  ë˜�ëŠ”  FEATURES = intersect20_features
FEATURES = [
    'feature_61', 'feature_58', 'feature_60', 'feature_15', 'feature_08', 'feature_24', 'feature_30', 
    'feature_47', 'feature_07', 'feature_29', 'feature_38', 'feature_25', 'feature_62', 
    'feature_06', 'feature_22', 'feature_05', 'feature_23', 'feature_20', 'feature_37', 'feature_28'
]

def na_audit(features, root_dir=ROOT_DIR, n_parts=7):
    """
    ì•„ì£¼ ì‰½ê²Œ ì„¤ëª…:
    - ê°� íŒŒí‹°ì…˜ íŒŒì�¼ì�„ í•˜ë‚˜ì”© ì—°ë‹¤.
    - í•´ë‹¹ íŒŒí‹°ì…˜ì—�ì„œ 'features' ê°� ì»¬ëŸ¼ì�˜ NaN(ë¹ˆì¹¸) ë¹„ìœ¨ì�„ ê³„ì‚°í•œë‹¤.
    - ëª¨ë“  íŒŒí‹°ì…˜ ê²°ê³¼ë¥¼ í•œ í‘œë¡œ ëª¨ì•„ì„œ, í”¼ì²˜ë³„ 'í�‰ê·  NaN ë¹„ìœ¨'ì�„ ì¶”ê°€ë¡œ ê³„ì‚°í•œë‹¤.
    - í�‰ê·  NaN ë¹„ìœ¨ì�´ í�° ìˆœì„œëŒ€ë¡œ ì •ë ¬í•´ì„œ ë�Œë ¤ì¤€ë‹¤.
    """
    part_ratio = {}   # ì˜ˆ: {"partition_0": Series(í”¼ì²˜ë³„ NaNë¹„ìœ¨), ...}
    for pid in range(n_parts):
        f = os.path.join(root_dir, "train.parquet", f"partition_id={pid}", "part-0.parquet")
        # í•„ìš”í•œ ì»¬ëŸ¼ë§Œ ì�½ìœ¼ë©´ ë¹ ë¥´ê³  ë©”ëª¨ë¦¬ ì ˆì•½
        df_pl = pl.read_parquet(f, columns=features)
        df = df_pl.to_pandas()

        # ì»¬ëŸ¼ë³„ NaN ë¹„ìœ¨ ê³„ì‚° (0.0 ~ 1.0)
        part_ratio[f"partition_{pid}"] = df.isna().mean()

        print(f"=== Partition {pid} === ì™„ë£Œ")  # ì§„í–‰ìƒ�í™© ì¶œë ¥

    # íŒŒí‹°ì…˜ë³„ NaN ë¹„ìœ¨ì�„ í•˜ë‚˜ì�˜ DataFrameìœ¼ë¡œ í•©ì¹˜ê¸° (í–‰=í”¼ì²˜, ì—´=partition_x)
    na_df = pd.DataFrame(part_ratio)

    # í”¼ì²˜ë³„ í�‰ê·  NaN ë¹„ìœ¨ ì¶”ê°€(ëª¨ë“  íŒŒí‹°ì…˜ì�„ ë�™ì�¼ ê°€ì¤‘ìœ¼ë¡œ í�‰ê· )
    na_df["mean"] = na_df.mean(axis=1)

    # í�‰ê·  NaN ë¹„ìœ¨ì�´ í�° ìˆœì„œëŒ€ë¡œ ì •ë ¬ (ì–´ë–¤ í”¼ì²˜ê°€ ë�” ë¹„ì–´ì�ˆëŠ”ì§€ í•œëˆˆì—�)
    na_df = na_df.sort_values("mean", ascending=False)

    return na_df

# ===== ì‹¤í–‰ =====
col_name = "feature"  # ì‚¬ë�Œì�´ ì�½ê¸° ì¢‹ê²Œ ì»¬ëŸ¼ëª… í‘œê¸°ìš© ë¬¸ì��ì—´
print(f"{col_name} (ì �ê²€ ëŒ€ìƒ�) ê°œìˆ˜: {len(FEATURES)}")
print(f"{col_name} (ì �ê²€ ëŒ€ìƒ�) ëª©ë¡�: {sorted(FEATURES)}\n")

na_report = na_audit(FEATURES, ROOT_DIR, n_parts=7)

print("\nğŸ”� í”¼ì²˜ë³„ NaN ë¹„ìœ¨ ìš”ì•½ (ì—´=ê°� íŒŒí‹°ì…˜, mean=í�‰ê·  NaN ë¹„ìœ¨)")
print(na_report)

# (ì„ íƒ�) CSVë¡œ ì €ì�¥í•´ì„œ ì—‘ì…€/êµ¬ê¸€ì‹œíŠ¸ë¡œ í™•ì�¸í•˜ê³  ì‹¶ì�„ ë•Œ
# na_report.to_csv("/kaggle/working/na_audit_report.csv")
# print("ì €ì�¥: /kaggle/working/na_audit_report.csv")


# =========================================
# C) LightGBM í•™ìŠµ (ê³ ì • í”Œë¡œìš°)
# - ì�…ë ¥: FEATURES ë¦¬ìŠ¤íŠ¸(Top20 ë˜�ëŠ” êµ�ì§‘í•©20 ì¤‘ íƒ�1)
# - ê²°ì¸¡ì¹˜: ì�…ë ¥ í”¼ì²˜ NaNì�€ "ê·¸ëŒ€ë¡œ" ë‘”ë‹¤ (LightGBMì�´ ì²˜ë¦¬)
# - íƒ€ê¹ƒ ê²°ì¸¡: responder_6 ê²°ì¸¡ í–‰ë§Œ ì œê±°
# - ê²€ì¦�: 5-Fold KFold (shuffle, random_state ê³ ì •)
# - ë³´ê³ : MSE / wMSE ì�˜ í�‰ê·  Â± í‘œì¤€í�¸ì°¨
# =========================================

import os
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 1) ë�°ì�´í„° ë¡œë“œ (í•„ìš” ì»¬ëŸ¼ë§Œ) -------------------------------------------------
cols = FEATURES + ["responder_6", "weight"]  # í•™ìŠµì—� í•„ìš”í•œ ì»¬ëŸ¼ë§Œ ì�½ëŠ”ë‹¤
dfs = []
for pid in range(7):  # partition_id=0~6 ëª¨ë‘� ìˆœíšŒ
    f = os.path.join(ROOT_DIR, "train.parquet", f"partition_id={pid}", "part-0.parquet")
    df_pl = pl.read_parquet(f, columns=cols)
    dfs.append(df_pl)

# ëª¨ë“  íŒŒí‹°ì…˜ì�„ ì„¸ë¡œë¡œ ì�´ì–´ë¶™ì—¬ í•˜ë‚˜ì�˜ í‘œë¡œ ë§Œë“ ë‹¤
df_all = pl.concat(dfs, how="vertical_relaxed").to_pandas()

# 2) íƒ€ê¹ƒ ê²°ì¸¡ ì œê±° -----------------------------------------------------------
# - ì�…ë ¥ í”¼ì²˜ NaNì�€ ë†”ë‘�ê³ , íƒ€ê¹ƒì�´ ë¹„ì–´ì�ˆëŠ” í–‰ë§Œ ì œê±°(í•™ìŠµ ë¶ˆê°€)
df_all = df_all.dropna(subset=["responder_6"])

# 3) í•™ìŠµìš© X, y, w ì¤€ë¹„ ------------------------------------------------------
# - X: ì„ íƒ�í•œ FEATURESë§Œ ì‚¬ìš©
# - y: responder_6
# - w: weight(ìƒ˜í”Œ ê°€ì¤‘ì¹˜)
X = df_all[FEATURES]
y = df_all["responder_6"]
w = df_all["weight"]

# 4) ëª¨ë�¸/ê²€ì¦� ì„¤ì • -----------------------------------------------------------
# - í•˜ì�´í�¼íŒŒë�¼ë¯¸í„°ëŠ” ê³ ì •í•´ ì�¬í˜„ì„± ìœ ì§€
params = dict(
    n_estimators=1000,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5) K-Fold ë£¨í”„ (MSE / wMSE ê³„ì‚°) -------------------------------------------
mses, wmses = [], []

print("ğŸš€ LightGBM 5-Fold í•™ìŠµ ì‹œì�‘")
for fold, (tr, va) in enumerate(kf.split(X), 1):
    # í•™ìŠµ/ê²€ì¦� ë¶„í• 
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]
    w_tr, w_va = w.iloc[tr], w.iloc[va]

    # ëª¨ë�¸ ìƒ�ì„± (LightGBMì�€ ì�…ë ¥ NaNì�„ ì��ë�™ ì²˜ë¦¬)
    model = lgb.LGBMRegressor(**params)

    # í•™ìŠµ (ìƒ˜í”Œ ê°€ì¤‘ì¹˜ ë°˜ì˜�)
    model.fit(X_tr, y_tr, sample_weight=w_tr)

    # ê²€ì¦� ì˜ˆì¸¡
    pred = model.predict(X_va)

    # ì„±ëŠ¥ ê³„ì‚°: MSE(ì�‘ì�„ìˆ˜ë¡� ì¢‹ì�Œ), ê°€ì¤‘ MSE(ì°¸ê³ ìš©)
    mse = mean_squared_error(y_va, pred)
    mse_w = mean_squared_error(y_va, pred, sample_weight=w_va)

    mses.append(mse)
    wmses.append(mse_w)

    print(f"  - Fold {fold}: MSE={mse:.6f} | wMSE={mse_w:.6f}")

# 6) ê²°ê³¼ ìš”ì•½ ì¶œë ¥ -----------------------------------------------------------
mse_mean, mse_std = np.mean(mses), np.std(mses)
wmse_mean, wmse_std = np.mean(wmses), np.std(wmses)

print("\nğŸ“Š CV ìš”ì•½")
print(f"  MSE  : {mse_mean:.6f} Â± {mse_std:.6f}")
print(f"  wMSE : {wmse_mean:.6f} Â± {wmse_std:.6f}")

# (ì„ íƒ�) 7) ì „ì²´ ë�°ì�´í„°ë¡œ ìµœì¢… ëª¨ë�¸ í•œ ë²ˆ ë�” ì �í•© -----------------------------
# - CVê°€ ë��ë‚œ ë’¤, ì‹¤ì‚¬ìš©/ì¶”ë¡  ëŒ€ë¹„ë¡œ ì „ì²´ ë�°ì�´í„°ë¥¼ ì‚¬ìš©í•´ ìµœì¢… ëª¨ë�¸ì�„ ë§Œë“ ë‹¤
final_model = lgb.LGBMRegressor(**params)
final_model.fit(X, y, sample_weight=w)
# â†’ í•„ìš”í•˜ë©´ pickleë¡œ ì €ì�¥ ê°€ëŠ¥:
# import joblib; joblib.dump(final_model, "/kaggle/working/lgbm_final.pkl")
# print("ì €ì�¥: /kaggle/working/lgbm_final.pkl")



# ===============================
# A) ê²°ì¸¡ì¹˜ í™•ì�¸ (ë²„ì „2: êµ�ì§‘í•© 20)
# ===============================
import os, polars as pl, pandas as pd

ROOT_DIR = "/kaggle/input/jane-street-real-time-market-data-forecasting"

# êµ�ì§‘í•© 20ê°œ (ë„¤ê°€ ë½‘ì�€ ë¦¬ìŠ¤íŠ¸ ê·¸ëŒ€ë¡œ ë„£ì–´)
intersect20_features = [
    'feature_06', 'feature_20', 'feature_22', 'feature_07', 'feature_28', 'feature_25', 'feature_30', 'feature_24', 
    'feature_29', 'feature_62', 'feature_61', 'feature_23', 'feature_47', 'feature_38', 'feature_15', 'feature_60', 
    'feature_64', 'feature_58', 'feature_37', 'feature_08', 'feature_59', 'feature_05', 'feature_49', 'feature_69', 'feature_56', 
    'feature_17', 'feature_33', 'feature_74', 'feature_73', 'feature_36', 'feature_21', 'feature_50', 'feature_72'
]

def na_audit(features, root_dir=ROOT_DIR, n_parts=7):
    """
    - ê°� íŒŒí‹°ì…˜ íŒŒì�¼ì�„ ì—°ë‹¤.
    - ì„ íƒ�í•œ featuresì�˜ NaN(ë¹ˆì¹¸) ë¹„ìœ¨ì�„ ê³„ì‚°í•œë‹¤.
    - íŒŒí‹°ì…˜ë³„ ê²°ê³¼ë¥¼ ëª¨ì•„ í”¼ì²˜ë³„ 'í�‰ê·  NaN ë¹„ìœ¨'ì�„ êµ¬í•œë‹¤.
    - í�‰ê·  NaN ë¹„ìœ¨ì�´ í�° ìˆœìœ¼ë¡œ ì •ë ¬í•´ ë°˜í™˜í•œë‹¤.
    """
    part_ratio = {}  # {"partition_0": Series(í”¼ì²˜ë³„ NaNë¹„ìœ¨), ...}
    for pid in range(n_parts):
        f = os.path.join(root_dir, "train.parquet", f"partition_id={pid}", "part-0.parquet")
        df_pl = pl.read_parquet(f, columns=features)  # í•„ìš”í•œ ì»¬ëŸ¼ë§Œ
        df = df_pl.to_pandas()
        part_ratio[f"partition_{pid}"] = df.isna().mean()  # ì»¬ëŸ¼ë³„ NaN ë¹„ìœ¨
        print(f"=== Partition {pid} === ì™„ë£Œ")

    na_df = pd.DataFrame(part_ratio)   # í–‰=í”¼ì²˜, ì—´=partition_x
    na_df["mean"] = na_df.mean(axis=1) # í”¼ì²˜ë³„ í�‰ê·  NaN ë¹„ìœ¨
    na_df = na_df.sort_values("mean", ascending=False)
    return na_df

# ===== ì‹¤í–‰ =====
col_name = "feature"
print(f"{col_name} (ë²„ì „2-êµ�ì§‘í•©20) ê°œìˆ˜: {len(intersect20_features)}")
print(f"{col_name} (ë²„ì „2-êµ�ì§‘í•©20) ëª©ë¡�: {sorted(intersect20_features)}\n")

na_report = na_audit(intersect20_features, ROOT_DIR, n_parts=7)

print("\nğŸ”� í”¼ì²˜ë³„ NaN ë¹„ìœ¨ ìš”ì•½ (ì—´=ê°� íŒŒí‹°ì…˜, mean=í�‰ê·  NaN ë¹„ìœ¨)")
print(na_report)

# (ì„ íƒ�) CSV ì €ì�¥
# na_report.to_csv("/kaggle/working/na_audit_report_v2.csv", index=True)
# print("ì €ì�¥: /kaggle/working/na_audit_report_v2.csv")


# =========================================
# C) LightGBM í•™ìŠµ â€” Version 2 (êµ�ì§‘í•© 20)
# =========================================
import os, polars as pl, pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

ROOT_DIR = "/kaggle/input/jane-street-real-time-market-data-forecasting"

# êµ�ì§‘í•© 20ê°œ
intersect20_features = [
    'feature_61','feature_58','feature_60','feature_15','feature_08','feature_24','feature_30',
    'feature_47','feature_07','feature_29','feature_38','feature_25','feature_62',
    'feature_06','feature_22','feature_05','feature_23','feature_20','feature_37','feature_28'
]
FEATURES = intersect20_features

# ë�°ì�´í„° ë¡œë“œ (í•„ìš” ì»¬ëŸ¼ë§Œ)
need_cols = FEATURES + ["responder_6","weight"]
frames = []
for pid in range(7):
    f = os.path.join(ROOT_DIR,"train.parquet",f"partition_id={pid}","part-0.parquet")
    frames.append(pl.read_parquet(f, columns=need_cols))
df_all = pl.concat(frames, how="vertical_relaxed").to_pandas()

# íƒ€ê¹ƒ ê²°ì¸¡ë§Œ ì œê±° (í”¼ì²˜ NaNì�€ ê·¸ëŒ€ë¡œ)
df_all = df_all.dropna(subset=["responder_6"])
X = df_all[FEATURES]
y = df_all["responder_6"]
w = df_all["weight"]

# ëª¨ë�¸/ê²€ì¦� ì„¤ì •
params = dict(
    n_estimators=1000, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
    force_col_wise=True,  # ë¡œê·¸ê°€ ê¶Œì�¥: ë©€í‹°ìŠ¤ë ˆë“œ ì˜¤ë²„í—¤ë“œ ê°�ì†Œ
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5-Fold CV
mses, wmses = [], []
print("ğŸš€ LightGBM 5-Fold í•™ìŠµ ì‹œì�‘ (Version2: êµ�ì§‘í•©20)")
for fold, (tr, va) in enumerate(kf.split(X), 1):
    model = lgb.LGBMRegressor(**params)
    model.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
    pred = model.predict(X.iloc[va])
    mse = mean_squared_error(y.iloc[va], pred)
    wmse = mean_squared_error(y.iloc[va], pred, sample_weight=w.iloc[va])
    mses.append(mse); wmses.append(wmse)
    print(f"  - Fold {fold}: MSE={mse:.6f} | wMSE={wmse:.6f}")

print("\nğŸ“Š CV ìš”ì•½ (Version2)")
print(f"  MSE  : {np.mean(mses):.6f} Â± {np.std(mses):.6f}")
print(f"  wMSE : {np.mean(wmses):.6f} Â± {np.std(wmses):.6f}")

# (ì„ íƒ�) ì „ì²´ ë�°ì�´í„°ë¡œ ìµœì¢… ì �í•©
final_model_v2 = lgb.LGBMRegressor(**params)
final_model_v2.fit(X, y, sample_weight=w)

