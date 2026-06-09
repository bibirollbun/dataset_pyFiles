!pip install -U scikit-learn imbalanced-learn


!pip install pytorch-tabnet xgboost polars


from pathlib import Path
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import gc
from sklearn.preprocessing import OrdinalEncoder
# Where raw Kaggle CSVs live
DATA_DIR = Path("/kaggle/input/home-credit-default-risk")
# Where we will save the merged parquet files
OUT_DIR  = Path("/kaggle/working/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
drop_cols = []

# Expected filenames
FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",   # optional but recommended
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}

# Leakage policy toggles
# ENABLE_TRAIN_ONLY_CLIPPING = True
# CLIP_COLS = ["APP_CREDIT_INCOME_RATIO","APP_ANNUITY_INCOME_RATIO","APP_CREDIT_GOODS_RATIO"]
# CLIP_QUANTILE = 0.995   # threshold computed on TRAIN ONLY

print("DATA_DIR:", DATA_DIR.as_posix())
print("OUT_DIR :", OUT_DIR.as_posix())



def _read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    print("Loading", name, "from", path)
    # return pd.read_csv(path, low_memory=False)
    chunks = pd.read_csv(path, chunksize=1_000_000, low_memory=False)
    df_list = []
    for chunk in chunks:
        df_list.append(chunk)
    df = pd.concat(df_list, ignore_index=True)
    return df

def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True, chunk_cols: int = 10):
    """
    Memory-efficient one-hot encoder for large DataFrames already in memory.
    Processes categorical columns in chunks to reduce memory pressure.
    Returns transformed df and list of new columns.
    """

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        return df, []

    all_new_cols = []
    n_total = len(cat_cols)
    
    # Process categorical columns in small groups to limit temporary memory use
    for i in range(0, n_total, chunk_cols):
        chunk = cat_cols[i : i + chunk_cols]
        
        # Convert to category dtype if still object (saves memory)
        for c in chunk:
            if df[c].dtype == 'object':
                df[c] = df[c].astype('category')

        # One-hot encode only this chunk
        dummies = pd.get_dummies(
            df[chunk],
            dummy_na=nan_as_category,
            dtype='uint8'
        )

        # Drop original categorical columns for this chunk
        df.drop(columns=chunk, inplace=True)

        # Merge encoded chunk back to main df
        df = pd.concat([df, dummies], axis=1)
        
        all_new_cols.extend(dummies.columns.tolist())

        # Explicitly free temporary memory
        del dummies

    return df, all_new_cols


def agg_numeric(df: pd.DataFrame, group_key: str, prefix: str, stats=None):
    '''Aggregate numeric columns by group_key with given stats (group-local; no global stats).'''
    if stats is None:
        stats = ["mean", "min", "max", "sum", "var"]
    df_num = df.select_dtypes(include=[np.number])
    if group_key not in df_num.columns and group_key in df.columns:
        df_num[group_key] = df[group_key]
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    agg = df_num.groupby(group_key).agg(stats)
    agg.columns = pd.Index([f"{prefix}{c}_{s}".upper() for c, s in agg.columns])
    return agg

def agg_categorical(df: pd.DataFrame, group_key: str, prefix: str):
    '''OHE categoricals, then aggregate mean (proportions) and sum (counts) per group.'''
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ("object", "category")]
    if not cat_cols:
        cnt = df.groupby(group_key).size().rename(f"{prefix}COUNT".upper())
        return cnt.to_frame()
    df_small = df[[group_key] + cat_cols]
    df_ohe, new_cols = one_hot_encoder(df_small)
    agg_dict = {c: ["mean", "sum"] for c in new_cols}  # per-group only
    agg = df_ohe.groupby(group_key).agg(agg_dict)
    agg.columns = pd.Index([f"{prefix}{c}_{s}".upper() for c, s in agg.columns])
    cnt = df.groupby(group_key).size().rename(f"{prefix}COUNT".upper())
    agg = agg.join(cnt)
    return agg

def encode_categorical(df:pd.DataFrame,nan_as_category:bool=True) -> pd.DataFrame:
    binary_cols = [
        "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
        "DAYS_EMPLOYED_ANOM"
    ]

    # 2ï¸�âƒ£ Ordinal columns (monotonic or ordered)
    ordinal_cols = [
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "NAME_EDUCATION_TYPE",       # roughly ordered by level
    ]
    
    low_nominal = [
        "NAME_CONTRACT_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_HOUSING_TYPE",
        "NAME_FAMILY_STATUS",
        "WEEKDAY_APPR_PROCESS_START",
        "NAME_CLIENT_TYPE",
        "NAME_CONTRACT_STATUS",
        "CREDIT_ACTIVE",
        "CREDIT_CURRENCY",
        "CREDIT_TYPE",
    ]
    
    high_nominal = [
        "ORGANIZATION_TYPE",
        "OCCUPATION_TYPE",
        "NAME_GOODS_CATEGORY",
    ]

    def frequency_encode(df,cols):
        for col in cols:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True)
                df[f"{col}_FREQ"] = df[col].map(freq).astype(np.float32)
                drop_cols.append(col)
        return df

    ord_cols_present = [c for c in ordinal_cols if c in df.columns]
    if ord_cols_present:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        df[ord_cols_present] = enc.fit_transform(df[ord_cols_present]).astype(np.float32)

    df = frequency_encode(df,[c for c in high_nominal if c in df.columns])

    low_nominal_present = [c for c in low_nominal if c in df.columns]
    if low_nominal_present:
        print(f"One-hot encoding {len(low_nominal_present)} low-cardinality categorical columns ...")
        df, _ = one_hot_encoder(df, nan_as_category = nan_as_category)

    for c in binary_cols:
        if c in df.columns and df[c].dtype == "object":
            df[c] = df[c].map({"Y":1,"N":0}).astype(np.uint8)

    cat_features = df.select_dtypes(include=["category","object"]).columns
    if len(cat_features):
        df[cat_features] = df[cat_features].apply(lambda x: x.cat.codes if x.dtype.name == "category" else x)

    return df

def safe_ratio(df: pd.DataFrame, num: str, den: str, out_name: str):
    '''Create ratio with safe division (row-wise; no global stats).'''
    if num in df.columns and den in df.columns:
        x = pd.to_numeric(df[num], errors="coerce")
        y = pd.to_numeric(df[den], errors="coerce").replace(0, np.nan)
        df[out_name] = x / y
    return df

def has_cols(df: pd.DataFrame, cols) -> bool:
    return all(c in df.columns for c in cols)




def add_app_level_features(apps: pd.DataFrame) -> pd.DataFrame:
    # Known anomaly
    if "DAYS_EMPLOYED" in apps:
        apps["DAYS_EMPLOYED_ANOM"] = (apps["DAYS_EMPLOYED"] == 365243).astype(np.int8)
        apps["DAYS_EMPLOYED"] = apps["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Age & employment years (DAYS_* are negative)
    if "DAYS_BIRTH" in apps:
        apps["AGE_YEARS"] = -apps["DAYS_BIRTH"] / 365.0
    if "DAYS_EMPLOYED" in apps:
        apps["EMPLOYED_YEARS"] = np.where(
            apps["DAYS_EMPLOYED"] < 0, -apps["DAYS_EMPLOYED"] / 365.0, np.nan
        )

    # Ratios
    ratios = [
        ("AMT_CREDIT", "AMT_INCOME_TOTAL", "APP_CREDIT_INCOME_RATIO"),
        ("AMT_ANNUITY", "AMT_INCOME_TOTAL", "APP_ANNUITY_INCOME_RATIO"),
        ("AMT_CREDIT", "AMT_GOODS_PRICE", "APP_CREDIT_GOODS_RATIO"),
        ("AMT_ANNUITY", "AMT_CREDIT", "APP_CREDIT_TERM"),
        ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "INCOME_PER_PERSON"),
    ]
    for n, d, out in ratios:
        if has_cols(apps, [n, d]):
            apps[out] = np.where(apps[d] == 0, np.nan, apps[n] / apps[d])

    # Family
    if has_cols(apps, ["CNT_FAM_MEMBERS", "CNT_CHILDREN"]):
        apps["ADULTS_COUNT"] = apps["CNT_FAM_MEMBERS"] - apps["CNT_CHILDREN"]

    # EXT_SOURCE interactions
    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in apps.columns]
    if len(ext) >= 2:
        if "EXT_SOURCE_1" in ext and "EXT_SOURCE_2" in ext:
            apps["EXT12_PROD"] = apps["EXT_SOURCE_1"] * apps["EXT_SOURCE_2"]
            apps["EXT12_MIN"]  = apps[["EXT_SOURCE_1","EXT_SOURCE_2"]].min(axis=1)
            apps["EXT12_MAX"]  = apps[["EXT_SOURCE_1","EXT_SOURCE_2"]].max(axis=1)
        if "EXT_SOURCE_1" in ext and "EXT_SOURCE_3" in ext:
            apps["EXT13_PROD"] = apps["EXT_SOURCE_1"] * apps["EXT_SOURCE_3"]
        if "EXT_SOURCE_2" in ext and "EXT_SOURCE_3" in ext:
            apps["EXT23_PROD"] = apps["EXT_SOURCE_2"] * apps["EXT_SOURCE_3"]
        apps["EXT_MEAN"] = apps[ext].mean(axis=1)  # row-wise
        apps["EXT_STD"]  = apps[ext].std(axis=1)   # row-wise

    # Skew reducers
    for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if c in apps.columns:
            apps[f"LOG1P_{c}"] = np.log1p(apps[c].clip(lower=0))

    # Missing flags (keep missingness as signal)
    for col in ["APP_CREDIT_INCOME_RATIO","APP_ANNUITY_INCOME_RATIO","APP_CREDIT_GOODS_RATIO"]:
        if col in apps.columns:
            apps[f"{col}_ISNA"] = apps[col].isna().astype(np.int8)

    drop_cols.append("DAYS_BIRTH")
    drop_cols.append("DAYS_EMPLOYED")

    return apps

def build_applications():
    train = _read_csv("application_train")
    # test  = _read_csv("application_test")

    train_data, test_data = train_test_split(
        train, test_size=0.2, random_state=42, stratify=train["TARGET"]
    )
    train_ids = train_data["SK_ID_CURR"]
    test_ids  = test_data["SK_ID_CURR"]
    y_true = test_data["TARGET"].copy()
    test_data["TARGET"] = np.nan   # align schema
    apps = pd.concat([train_data, test_data], axis=0)
    apps = add_app_level_features(apps)
    return apps,train_ids,test_ids, len(train_data), y_true



def prepare_base_application(train_df,test_df):
    train_df["is_test"] = 0
    test_df["is_test"] = 1
    test_df["TARGET"] = np.nan
    return pd.concat([train_df,test_df],ignore_index=True)



def build_bureau_agg() -> pd.DataFrame:
    bureau = _read_csv("bureau")
    # Merge bureau_balance if available
    try:
        bb = _read_csv("bureau_balance")
        bb_cat = agg_categorical(bb[["SK_ID_BUREAU", "STATUS"]], "SK_ID_BUREAU", "BB_")
        bb_num = bb.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].agg(["min", "max", "size"])
        bb_num.columns = ["BB_MONTHS_MIN", "BB_MONTHS_MAX", "BB_MONTHS_SIZE"]
        bb_agg = bb_cat.join(bb_num, how="left")
        bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    except FileNotFoundError:
        pass

    if has_cols(bureau, ["AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM"]):
        safe_ratio(bureau, "AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM", "BURO_OVERDUE_RATIO")

    if "SK_DPD" in bureau.columns:
        tmp = bureau.groupby("SK_ID_BUREAU")["SK_DPD"].agg(["mean","max"])
        tmp.columns =  ["BURO_DPD_MEAN", "BURO_DPD_MAX"]
        bureau = bureau.merge(tmp, on="SK_ID_BUREAU", how="left")

    buro_num = agg_numeric(bureau.drop(columns=["SK_ID_BUREAU"], errors="ignore"), "SK_ID_CURR", "BURO_")

    keep_cats = [c for c in ["CREDIT_ACTIVE","CREDIT_CURRENCY","CREDIT_TYPE"] if c in bureau.columns]
    buro_cat = agg_categorical(bureau[["SK_ID_CURR"] + keep_cats], "SK_ID_CURR", "BURO_") if keep_cats else None
    buro_agg = buro_num.join(buro_cat, how="left") if buro_cat is not None else buro_num

    if "CREDIT_ACTIVE" in bureau.columns:
        agg_map = {}
        if "AMT_CREDIT_SUM" in bureau.columns:
            agg_map["AMT_CREDIT_SUM"] = ["sum", "mean"]
        if "AMT_CREDIT_SUM_DEBT" in bureau.columns:
            agg_map["AMT_CREDIT_SUM_DEBT"] = ["sum", "mean"]
        if "SK_DPD" in bureau.columns:
            agg_map["SK_DPD"] = ["mean", "max"]

        if agg_map:
            frames = []
            for state, g in bureau.groupby("CREDIT_ACTIVE"):
                g_agg = g.groupby("SK_ID_CURR").agg(agg_map)
                g_agg.columns = [
                    f"BURO_{col}_{stat}_{str(state)}".upper()
                    for col, stat in g_agg.columns
                ]
                frames.append(g_agg)
            if frames:
                wide = pd.concat(frames, axis=1)
                buro_agg = buro_agg.join(wide, how="left")

    drop_cols = ["SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"]
    del bureau, buro_num, buro_cat, bb, bb_agg
    gc.collect()

    return buro_agg




def _aggregate_child_by_prev(df_child: pd.DataFrame, prefix: str) -> pd.DataFrame:
    assert "SK_ID_PREV" in df_child.columns, f"{prefix} missing SK_ID_PREV"
    num_agg = agg_numeric(df_child, "SK_ID_PREV", prefix)
    cat_cols = [c for c in df_child.columns if str(df_child[c].dtype) in ("object", "category")]
    if cat_cols:
        cat_agg = agg_categorical(df_child[["SK_ID_PREV"] + cat_cols], "SK_ID_PREV", prefix)
        return num_agg.join(cat_agg, how="left")
    drop_cols.append("SK_ID_PREV")
    return num_agg

def build_prev_block_agg() -> pd.DataFrame:
    prev = _read_csv("previous_application")

    # POS
    pos = _read_csv("pos_cash")
    pos_agg = _aggregate_child_by_prev(pos, "POS_")

    # Installments
    ins = _read_csv("installments")
    if has_cols(ins, ["AMT_PAYMENT", "AMT_INSTALMENT"]):
        safe_ratio(ins, "AMT_PAYMENT", "AMT_INSTALMENT", "INS_PAYMENT_RATIO")
        ins["INS_PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    if has_cols(ins, ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]):
        ins["INS_DAYS_LATE"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
        ins["INS_PAID_LATE"]  = (ins["INS_DAYS_LATE"] > 0).astype(int)
        ins["INS_PAID_EARLY"] = (ins["INS_DAYS_LATE"] < 0).astype(int)
    late_rate = ins.groupby("SK_ID_PREV")["INS_PAID_LATE"].mean().rename("INS_LATE_RATE") if "INS_PAID_LATE" in ins.columns else None
    pos_late_mean = ins.loc[ins.get("INS_DAYS_LATE", pd.Series(dtype=float)) > 0].groupby("SK_ID_PREV")["INS_DAYS_LATE"].mean().rename("INS_DAYS_LATE_POS_MEAN") if "INS_DAYS_LATE" in ins.columns else None
    ins_agg = _aggregate_child_by_prev(ins, "INS_")
    if late_rate is not None:     ins_agg = ins_agg.join(late_rate, how="left")
    if pos_late_mean is not None: ins_agg = ins_agg.join(pos_late_mean, how="left")

    # Credit Card
    cc = _read_csv("credit_card")
    if has_cols(cc, ["AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]):
        safe_ratio(cc, "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "CC_UTILIZATION")
    if has_cols(cc, ["AMT_DRAWINGS_CURRENT", "AMT_CREDIT_LIMIT_ACTUAL"]):
        safe_ratio(cc, "AMT_DRAWINGS_CURRENT", "AMT_CREDIT_LIMIT_ACTUAL", "CC_DRAW_RATE")
    cc_agg = _aggregate_child_by_prev(cc, "CC_")

    # Join child aggregates back onto previous_application
    prev_enriched = prev.merge(pos_agg, on="SK_ID_PREV", how="left") \
                        .merge(ins_agg, on="SK_ID_PREV", how="left") \
                        .merge(cc_agg, on="SK_ID_PREV", how="left")
    for df_ in [pos, ins, cc]:
        drop_cols.extend([c for c in ["SK_ID_PREV"] if c in df_.columns])
    del pos_agg, ins_agg,cc_agg
    gc.collect()
    

    # Per-previous ratios
    for (n, d, out) in [
        ("AMT_APPLICATION", "AMT_CREDIT", "PREV_APP_CREDIT_RATIO"),
        ("AMT_DOWN_PAYMENT", "AMT_CREDIT", "PREV_DOWNPAY_CREDIT_RATIO"),
        ("AMT_ANNUITY", "AMT_CREDIT", "PREV_ANNUITY_CREDIT_RATIO"),
        ("AMT_CREDIT", "AMT_GOODS_PRICE", "PREV_CREDIT_GOODS_RATIO"),
    ]:
        if has_cols(prev_enriched, [n, d]):
            safe_ratio(prev_enriched, n, d, out)

    # Roll-up to SK_ID_CURR (numeric + selected categoricals)
    prev_num = agg_numeric(prev_enriched.drop(columns=["SK_ID_PREV"], errors="ignore"), "SK_ID_CURR", "PREV_")
    prev_cat_cols = [c for c in ["NAME_CONTRACT_TYPE","WEEKDAY_APPR_PROCESS_START",
                                 "NAME_CONTRACT_STATUS","NAME_CLIENT_TYPE",
                                 "NAME_GOODS_CATEGORY"] if c in prev_enriched.columns]
    prev_cat = agg_categorical(prev_enriched[["SK_ID_CURR"] + prev_cat_cols], "SK_ID_CURR", "PREV_") if prev_cat_cols else None
    prev_agg = prev_num.join(prev_cat, how="left") if prev_cat is not None else prev_num
    drop_cols.append(prev_cat_cols)
    del prev_cat
    gc.collect()
    # Approval/refusal counts & approval rate
    if "NAME_CONTRACT_STATUS" in prev_enriched.columns:
        st = (prev_enriched
              .groupby(["SK_ID_CURR","NAME_CONTRACT_STATUS"])
              .size().unstack(fill_value=0))
        st.columns = [f"PREV_STATUS_{c}".upper() for c in st.columns]
        st["PREV_APPROVAL_RATE"] = st.get("PREV_STATUS_Approved", 0) / st.sum(axis=1).replace(0, np.nan)
        prev_agg = prev_agg.join(st, how="left")

    # Recency of decisions
    if "DAYS_DECISION" in prev_enriched.columns:
        rec = prev_enriched.groupby("SK_ID_CURR")["DAYS_DECISION"].agg(["min","max","mean"])
        rec.columns = ["PREV_DAYS_DEC_MIN","PREV_DAYS_DEC_MAX","PREV_DAYS_DEC_MEAN"]
        rec["PREV_DAYS_DEC_RANGE"] = rec["PREV_DAYS_DEC_MAX"] - rec["PREV_DAYS_DEC_MIN"]
        prev_agg = prev_agg.join(rec, how="left")

    return prev_agg



def train_only_clip(train_df: pd.DataFrame, test_df: pd.DataFrame, cols, q=0.995):
    '''Compute thresholds on TRAIN ONLY and apply to both train and test.'''
    thresholds = {}
    for col in cols:
        if col in train_df.columns:
            thr = np.float32(train_df[col].quantile(q))
            thresholds[col] = thr

            # Clip in place without creating new Series
            train_col = train_df[col].to_numpy(copy=False)
            np.clip(train_col, a_min=None, a_max=thr, out=train_col)
    
            if col in test_df.columns and pd.api.types.is_numeric_dtype(test_df[col]):
                test_col = test_df[col].to_numpy(copy=False)
                np.clip(test_col, a_min=None, a_max=thr, out=test_col)
                
    return train_df, test_df, thresholds

# Orchestrate
train_df = _read_csv("application_train")
test_df  = _read_csv("application_test")
train_ids = train_df["SK_ID_CURR"].unique()
test_ids  = test_df["SK_ID_CURR"].unique()
y_true    = train_df["TARGET"].values
# apps,train_ids,test_ids, n_train, y_true = build_applications()
apps = prepare_base_application(train_df, test_df)
print("Applications (stacked):", apps.shape)

apps = add_app_level_features(apps)
print("Updated Applications (stacked):", apps.shape)

buro_agg = build_bureau_agg()
print("Bureau aggregate:", buro_agg.shape)

prev_agg = build_prev_block_agg()
print("Previous aggregate:", prev_agg.shape)

# Join all blocks
full = apps.merge(buro_agg, on="SK_ID_CURR", how="left") \
           .merge(prev_agg, on="SK_ID_CURR", how="left")
print("Joined shape:", full.shape)

del buro_agg, prev_agg
import gc
gc.collect()
# OHE remaining application-level categoricals on combined frame (allowed)
app_cat_cols = [c for c in full.columns if str(full[c].dtype) in ("object", "category")]
if app_cat_cols:
    print(f"One-hot encoding {len(app_cat_cols)} application-level categorical columns ...")
    full = encode_categorical(full, nan_as_category=True)

print(f"One Hot Encoding Completed....")
# Split back to train/test
# train_merged = full.loc[train_idx]
# test_merged  = full.loc[test_idx]

# train_merged = full[full["SK_ID_CURR"].isin(train_ids)]
# test_merged  = full[full["SK_ID_CURR"].isin(test_ids)]

train_merged = full[full["is_test"] == 0]
test_merged  = full[full["is_test"] == 1]

# TRAIN-ONLY global adjustments (e.g., clipping)
# if ENABLE_TRAIN_ONLY_CLIPPING:
    # print(f"Applying TRAIN-ONLY clipping at q={CLIP_QUANTILE} for: {CLIP_COLS}")
    # train_merged, test_merged, thr = train_only_clip(train_merged, test_merged, CLIP_COLS, q=CLIP_QUANTILE)
    # print("Clip thresholds (train-only):", {k: float(v) if pd.notna(v) else None for k,v in thr.items()})

# Move TARGET to front (if present)
# if "TARGET" in train_merged.columns:
#     cols = ["TARGET"] + [c for c in train_merged.columns if c != "TARGET"]
#     train_merged = train_merged[cols]

# Save
# train_path = OUT_DIR / "train_merged.parquet"
# test_path  = OUT_DIR / "test_merged.parquet"
# print("Saving:", train_path, "and", test_path)
# train_merged.to_parquet(train_path, index=False)
# test_merged.to_parquet(test_path, index=False)

train_merged, test_merged = train_merged.align(test_merged, join='left', axis=1, fill_value=0)

print("\nFinal shapes:")
print("  train:", train_merged.shape, "| TARGET NaNs:", train_merged['TARGET'].isna().sum())
print("  test :", test_merged.shape)


train_merged.drop(columns=["DAYS_BIRTH","DAYS_EMPLOYED"], inplace=True)
test_merged.drop(columns=["DAYS_BIRTH","DAYS_EMPLOYED"], inplace=True)


def handle_missing_values(df,drop_thresh:float=0.90,flag_thresh:float = 0.10,skew_thresh:float=2.0,verbose:bool= True):
    # df = df.copy()
    n = len(df)
    impute_map = {}
    need_drop_cols = []
    flagged_cols = []
    miss = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / n).round(4)
    miss_summary = pd.DataFrame({
        "Missing_Count":miss,
        "MissingPct":miss_pct,
        "Dtype":df.dtypes
    })
    if verbose:
        print(f"Total Columns: {len(df.columns)}")
        print(f"Columns with missing values : {(miss > 0).sum()}")

    
    for col in df.columns:
        if miss_pct[col] > drop_thresh:
            need_drop_cols.append(col)
        elif df[col].nunique(dropna=True) <= 1:
            need_drop_cols.append(col)

    if need_drop_cols:
        if verbose:
            print(f"Dropping {len(need_drop_cols)} columns")
        df.drop(columns=need_drop_cols,inplace=True,errors="ignore")

    flag_df = {}
    for col in df.columns:
        if miss_pct.get(col,0) >= flag_thresh:
            flag_col = f"{col}_ISNA"
            flag_df[flag_col] = df[col].isna().astype(np.int8)
            flagged_cols.append(flag_col)
    if flag_df:
        df = pd.concat([df,pd.DataFrame(flag_df,index=df.index)],axis=1)
        if verbose:
            print(f"Added {len(flagged_cols)} missingness flags.")
        df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            col_skew = abs(df[col].dropna().skew()) if df[col].dropna().size > 0 else 0
            fill_val = df[col].median() if col_skew > skew_thresh else df[col].mean()
            df[col] = df[col].fillna(fill_val)
            # df[col].fillna(fill_val,inplace=True)
            impute_map[col] = fill_val

    special_zero = [c for c in df.columns if "_RATIO" in c or "LOG1P_" in c]
    for col in special_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            # df[col].fillna(0,inplace=True)
            impute_map[col] = 0

    remaining_nans = int(df.isna().sum().sum())
    if verbose:
        print(f"Final Missing values remaining : {remaining_nans}")
        print(f"Final Shape : {df.shape}")

    return df


print("ğŸ§¹ Handling missing values...")
full_cleaned = handle_missing_values(full)
# print(f"âœ… Done: dropped={len(dropped_cols)}, flagged={len(flagged_cols)}, imputed={len(impute_map)} features.")


full.shape


full_cleaned.shape


del apps,full,train_df,test_df,train_ids,test_ids
gc.collect()


train_merged = full_cleaned[full_cleaned["is_test"] == 0]
test_merged  = full_cleaned[full_cleaned["is_test"] == 1]


num_cols = train_merged.select_dtypes(include="number").columns
cols_to_plot = ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AGE_YEARS","APP_CREDIT_INCOME_RATIO","EXT_MEAN","EXT_SOURCE_3"
               ,"BURO_CREDIT_ACTIVE_ACTIVE_SUM"]

for col in cols_to_plot:
    if col in train_merged.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(train_merged[col],bins=60,kde=True)
        plt.title(f"Distribution : {col}")
        plt.tight_layout()
        # plt.savefig(f"{}")
        plt.show()


ratio_cols = [
    "APP_CREDIT_INCOME_RATIO","APP_ANNUITY_INCOME_RATIO","APP_CREDIT_GOODS_RATIO","INCOME_PER_PERSON",
    "EXT_SOURCE_2","EXT_SOURCE_3","APP_CREDIT_TERM", "BURO_AMT_CREDIT_SUM_SUM_ACTIVE","PREV_AMT_ANNUITY_MEAN"
]

for col in ratio_cols:
    if col in train_merged.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=train_merged[col])
        plt.title(f"Boxplot : {col}")
        plt.tight_layout()
        plt.show()


corr = train_merged.corr(numeric_only=True)["TARGET"].sort_values(ascending=False)
corr_top = corr.head(20).dropna()

plt.figure(figsize=(6,8))
sns.barplot(x=corr_top.values,y=corr_top.index,palette="coolwarm")
plt.title("Top Correlation with Target")
plt.tight_layout()
plt.show()


aggs_cols = [c for c in train_merged.columns if c.startswith(("BURO_","PREV_"))]
for c in aggs_cols[:10]:
    plt.figure(figsize=(8,5))
    sns.histplot(train_merged[c].dropna(),bins=60,kde=True)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(5,4))
sns.countplot(x=train_merged["TARGET"])
plt.title("Target Class Distribution")
plt.tight_layout()
plt.show()


variances = train_merged.var().sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.histplot(variances,bins=60,kde=True)
plt.title("Feature Variance Distribution")
plt.tight_layout()
plt.show()


# Create a DataFrame with a single column "column_name"
cols_df = pd.DataFrame(train_merged.columns, columns=['column_name'])

# Save to CSV
cols_df.to_csv('train_merged_columns.csv', index=False)


X_train = train_merged.drop(columns=["TARGET"])
y_train = train_merged["TARGET"]
X_test = test_merged.drop(columns=["TARGET"],errors="ignore")
y_test = test_merged["TARGET"]


X_train.shape,y_train.shape,X_test.shape,y_test.shape


del train_merged,test_merged       # Original full dataset (pre-cleaning)
gc.collect()


# import sys

# # Sort all user variables by size (descending)
# for name, size in sorted(((name, sys.getsizeof(val)) for name, val in globals().items()),
#                          key=lambda x: -x[1])[:20]:
#     print(f"{name:30s}: {size/1024/1024:10.2f} MB")


from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
import polars as pl
def optimize_features(X_train,y_train,X_test,low_var_thresh:float=0.0,corr_thresh:float=0.95,top_k:int=300,
                     save_path:str = "/kaggle/working_feature_selection_artifacts.joblib", verbose:bool = True):
    if verbose:
        print("Dropping low-variance features...")
    selector = VarianceThreshold(threshold=low_var_thresh)
    selector.fit(X_train)
    low_var_kept = X_train.columns[selector.get_support()]
    dropped_low_var = [c for c in X_train.columns if c not in low_var_kept]
    X_train = X_train[low_var_kept]
    if X_test is not None:
        X_test = X_test[low_var_kept]
    if verbose:
        print(f"-> Dropped {len(dropped_low_var)} low variance features")

    del selector, low_var_kept
    gc.collect()

    if verbose:
        print("Removing highly correlated features.")

    cols = list(X_train.columns)
    drop_corr = set()
    batch_size = 200

    for i in range(0,len(cols),batch_size):
        sub_cols = cols[i:i+batch_size]
        corr_sub = X_train[sub_cols].corr().abs()
        upper = corr_sub.where(np.triu(np.ones(corr_sub.shape), k = 1).astype(bool))
        for col in upper.columns:
            if any(upper[col] > corr_thresh):
                drop_corr.add(col)
        
        del corr_sub, upper
        gc.collect()

        if verbose and (i // batch_size + 1) % 2 == 0:
            print(f"Batch {i//batch_size+1}: total {len(drop_corr)} columns marked for drop so far.")

    # corr_matrix = X_train.corr().abs()
    # corr_matrix = pl.from_pandas(X_train).corr().to_pandas().abs()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
    # drop_corr = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    drop_corr = list(drop_corr)
    X_train.drop(columns=drop_corr, inplace=True, errors="ignore")
    if X_test is not None:
        X_test.drop(columns=drop_corr,inplace=True,errors="ignore")
    if verbose:
        print(f"-> Dropped {len(drop_corr)} correlated features")

    # del corr_matrix, upper
    gc.collect()

    if verbose:
        print("Selecting top-K important feature using HGB....")
    # model = HistGradientBoostingClassifier(random_state=42)
    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",   # âœ… strictly CPU
        device="cpu",         # âœ… prevents GPU probing
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train,y_train)
    # importances = get
    importances = pd.Series(model.feature_importances_,index=X_train.columns)
    top_features = importances.nlargest(top_k).index.tolist()

    X_train_opt = X_train[top_features]
    X_test_opt = X_test[top_features]
    if verbose:
        print(f"-> Selected top {len(top_features)} featuress by importance")

    del X_train, X_test, model
    gc.collect()

    meta = {
        "low_var_dropped":dropped_low_var,
        "corr_dropped":drop_corr,
        "top_features":top_features,
        "importance":importances.sort_values(ascending=False).to_dict()
    }
    joblib.dump(meta,save_path)
    if verbose:
        print(f"Feature Selection artifacts saved ->{save_path}")

    return X_train_opt, X_test_opt, meta


print("âš™ï¸� Optimizing features...")
ids = X_test["SK_ID_CURR"].copy()
X_train_opt, X_test_opt, feat_meta = optimize_features(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    low_var_thresh=0.0,
    corr_thresh=0.95,
    top_k=300,
)
print(f"âœ… Final feature count: {X_train_opt.shape[1]}")


# feature_selection_and_modeling_pipeline.py
import os
import warnings
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed, dump, load

OUTDIR = "/kaggle/working"  # or "/mnt/data" locally
os.makedirs(OUTDIR, exist_ok=True)

N_FEATURES = 100       # final number of features to select (adjust)
N_JOBS = -1            # parallel jobs
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5           # StratifiedKFold
VERBOSE = True

# Models to evaluate in CV (you can add more)
def get_basic_models(random_state=RANDOM_STATE, n_jobs=N_JOBS):
    models = {
        "logit": LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=1),  # n_jobs=1 prevents nested threading issues
        "hgb": HistGradientBoostingClassifier(random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=1),
    }
    return models

# Optional: TabNet (try to import)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except Exception:
    TABNET_AVAILABLE = False

# -------------------------
# Utilities
# -------------------------
def printv(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def save_pickle(obj, fname):
    dump(obj, fname)
    printv(f"Saved -> {fname}")

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }
    return metrics

def plot_roc_pr(y_true, y_score, title_prefix, outdir=OUTDIR):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc_score(y_true, y_score):.4f}')
    plt.plot([0,1],[0,1],'--', linewidth=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{title_prefix} ROC")
    plt.legend()
    plt.grid(alpha=0.2)
    fn = os.path.join(outdir, f"{title_prefix}_roc.png")
    plt.savefig(fn); plt.close(); printv("Saved:", fn)

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f'PR AUC={average_precision_score(y_true, y_score):.4f}')
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{title_prefix} Precision-Recall")
    plt.legend()
    plt.grid(alpha=0.2)
    fn = os.path.join(outdir, f"{title_prefix}_pr.png")
    plt.savefig(fn); plt.close(); printv("Saved:", fn)

# -------------------------
# Feature selection methods
# -------------------------
def rfe_select(X, y, n_features=N_FEATURES, random_state=RANDOM_STATE):
    """RFE with LogisticRegression (scaled inside RFE wrapper)."""
    printv("[RFE] Starting...")
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    base = LogisticRegression(max_iter=2000, solver="saga")
    rfe = RFE(estimator=base, n_features_to_select=n_features, step=0.1)
    rfe.fit(Xs, y)
    features = X.columns[rfe.support_].tolist()
    printv(f"[RFE] Selected {len(features)} features.")
    return features

def rf_importance_select(X, y, n_features=N_FEATURES, random_state=RANDOM_STATE):
    printv("[RF] Fitting RandomForest for importances...")
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=1)
    rf.fit(X, y)
    sel = SelectFromModel(rf, prefit=True, max_features=n_features, threshold=-np.inf)
    features = X.columns[sel.get_support()].tolist()
    printv(f"[RF] Selected {len(features)} features.")
    return features, rf

def logistic_coef_select(X, y, n_features=N_FEATURES):
    printv("[LogitCoef] Fitting LogisticRegression for coefficients...")
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=2000, solver="saga")
    lr.fit(Xs, y)
    sel = SelectFromModel(lr, prefit=True, max_features=n_features, threshold=-np.inf)
    features = X.columns[sel.get_support()].tolist()
    printv(f"[LogitCoef] Selected {len(features)} features.")
    return features, lr

def permutation_importance_select(X, y, n_features=N_FEATURES, random_state=RANDOM_STATE):
    # Use a strong model (HGB) for permutation importance
    printv("[PermImp] Fitting HGB for permutation importance...")
    hgb = HistGradientBoostingClassifier(random_state=random_state)
    hgb.fit(X, y)
    printv("[PermImp] Computing permutation importance (this could be slow)...")
    res = permutation_importance(hgb, X, y, n_repeats=10, random_state=random_state, n_jobs=1)
    importances = pd.Series(res.importances_mean, index=X.columns).sort_values(ascending=False)
    features = importances.head(n_features).index.tolist()
    printv(f"[PermImp] Selected {len(features)} features.")
    return features, importances

# -------------------------
# Aggregate & consensus selection
# -------------------------
def consensus_features(selection_lists, n_final=N_FEATURES):
    """Aggregate lists of selected features into a consensus ranking.
       selection_lists: list of lists (features from each method)
    """
    counts = Counter()
    for lst in selection_lists:
        counts.update(lst)
    # score = (count, mean_rank) could be used if you return ranks; but here simple count then fallback to global ranking
    freq_df = pd.DataFrame([(f, counts[f]) for f in counts], columns=["feature","count"])
    freq_df = freq_df.sort_values(by=["count"], ascending=False)
    final = freq_df['feature'].head(n_final).tolist()
    return final, freq_df



def run_feature_selection_and_modeling(
    X, y, X_test=None,
    n_features_final=N_FEATURES, outdir=OUTDIR
):
    start_all = time.time()
    printv("Pipeline start:", time.ctime())

    # 1ï¸�âƒ£ Train/test split
    if X_test is None:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        X_test = X_holdout
    else:
        X_train, y_train = X, y

    # 2ï¸�âƒ£ Feature selection (same as before)
    printv("\nRunning feature selection methods in parallel...")
    results = Parallel(n_jobs=N_JOBS, backend="loky")([
        delayed(rfe_select)(X_train, y_train, n_features=n_features_final),
        delayed(rf_importance_select)(X_train, y_train, n_features=n_features_final),
        delayed(logistic_coef_select)(X_train, y_train, n_features=n_features_final),
        delayed(permutation_importance_select)(X_train, y_train, n_features=n_features_final),
    ])
    lists, stored_models, perm_importances = [], {}, None
    for res in results:
        if isinstance(res, tuple):
            if isinstance(res[1], (RandomForestClassifier, LogisticRegression, HistGradientBoostingClassifier)):
                lists.append(res[0]); stored_models[type(res[1]).__name__] = res[1]
            else:
                lists.append(res[0]); perm_importances = res[1]
        else:
            lists.append(res)

    final_feats, freq_df = consensus_features(lists, n_final=n_features_final)
    dump(final_feats, os.path.join(outdir, "selected_features_consensus.joblib"))
    freq_df.to_csv(os.path.join(outdir, "feature_selection_counts.csv"), index=False)
    printv(f"\nConsensus selected {len(final_feats)} features.")

    # 3ï¸�âƒ£ Modeling loop
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = get_basic_models()
    if TABNET_AVAILABLE:
        models["tabnet"] = "tabnet_placeholder"

    oof_dict = {n: np.zeros(len(X_train)) for n in models}
    test_preds = {n: np.zeros(len(X_test)) for n in models}
    fold_metrics, best_thresholds = defaultdict(list), {}
    models_fitted = defaultdict(list)

    X_sel, X_test_sel = X_train[final_feats], X_test[final_feats]
    scaler = RobustScaler()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_sel, y_train), 1):
        printv(f"\n--- Fold {fold}/{N_SPLITS} ---")
        X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        from imblearn.combine import SMOTETomek
        smote = SMOTETomek(random_state=RANDOM_STATE)
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_tr_s, y_tr_s = smote.fit_resample(X_tr_s, y_tr)
        X_test_s = scaler.transform(X_test_sel)

        for name, model in models.items():
            try:
                if name == "logit":
                    clf = LogisticRegression(max_iter=2000,penalty="l1",solver="saga", class_weight="balanced")
                    clf.fit(X_tr_s, y_tr_s)
                    p_va, p_te = clf.predict_proba(X_va_s)[:,1], clf.predict_proba(X_test_s)[:,1]
                    models_fitted[name].append(clf) 
                elif name == "hgb":
                    clf = HistGradientBoostingClassifier(class_weight='balanced', random_state=RANDOM_STATE)
                    clf.fit(X_tr, y_tr)
                    p_va, p_te = clf.predict_proba(X_va)[:,1], clf.predict_proba(X_test_sel)[:,1]
                    models_fitted[name].append(clf) 
                elif name == "rf":
                    clf = RandomForestClassifier(
                        n_estimators=300, random_state=RANDOM_STATE,
                        n_jobs=1, class_weight="balanced_subsample"
                    )
                    clf.fit(X_tr, y_tr)
                    p_va, p_te = clf.predict_proba(X_va)[:,1], clf.predict_proba(X_test_sel)[:,1]
                    models_fitted[name].append(clf) 
                elif name == "tabnet" and TABNET_AVAILABLE:
                    clf = TabNetClassifier(seed=RANDOM_STATE, verbose=0)
                    clf.fit(X_tr.values, y_tr, eval_set=[(X_va.values, y_va)],
                            max_epochs=100, patience=10,
                            batch_size=1024, virtual_batch_size=64)
                    p_va, p_te = clf.predict_proba(X_va.values)[:,1], clf.predict_proba(X_test_sel.values)[:,1]
                    models_fitted[name].append(clf) 
                else:
                    continue

                oof_dict[name][va_idx] = p_va
                test_preds[name] += p_te / N_SPLITS

                # ğŸ”¹ threshold tuning
                prec, rec, thr = precision_recall_curve(y_va, p_va)
                f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
                best_idx = np.nanargmax(f1)
                best_thr = thr[max(best_idx, 0)] if len(thr) > 0 else 0.5
                best_thresholds.setdefault(name, []).append(best_thr)
                y_pred_opt = (p_va >= best_thr).astype(int)

                m = {
                    "fold": fold,
                    "threshold": float(best_thr),
                    "accuracy": accuracy_score(y_va, y_pred_opt),
                    "precision": precision_score(y_va, y_pred_opt, zero_division=0),
                    "recall": recall_score(y_va, y_pred_opt, zero_division=0),
                    "f1": f1_score(y_va, y_pred_opt, zero_division=0),
                    "roc_auc": roc_auc_score(y_va, p_va),
                    "pr_auc": average_precision_score(y_va, p_va)
                }
                fold_metrics[name].append(m)
                printv(f"{name} fold {fold} | ROC={m['roc_auc']:.3f} thr={m['threshold']:.3f}")

            except Exception as e:
                printv(f"Model {name} failed on fold {fold}: {e}")

    # 4ï¸�âƒ£ Aggregate & threshold summary
    summary_rows, summary = [], {}
    for name, oof in oof_dict.items():
        if np.all(oof == 0): continue
        mean_thr = np.mean(best_thresholds.get(name, [0.5]))
        y_pred_final = (oof >= mean_thr).astype(int)
        metrics_all = compute_metrics(y_train, oof, threshold=mean_thr)
        metrics_all["threshold_mean"] = mean_thr
        summary[name] = {
            "oof_metrics": metrics_all,
            "fold_metrics": fold_metrics[name],
            "test_pred": test_preds[name]
        }
        printv(f"\n{name} | Mean threshold={mean_thr:.3f}")
        printv(f"Accuracy : {metrics_all['accuracy']} | ROC_AUC Score : {metrics_all['roc_auc']}")
        try:
            np.save(os.path.join(outdir, f"oof_preds_{name}.npy"), oof)
            np.save(os.path.join(outdir, f"test_preds_{name}.npy"), test_preds[name])
        except Exception as e:
            printv(f"Saving OOF/Test preds failed for {name}: {e}")

        # --- Plot ROC / PR curves (OOF) using old helper ---
        try:
            plot_roc_pr(y_train, oof, f"{name}_OOF", outdir=outdir)
        except Exception as e:
            printv(f"plot_roc_pr failed for {name}: {e}")

        summary_rows.append({
            "model": name,
            "mean_threshold": round(mean_thr, 3),
            "roc_auc": round(metrics_all["roc_auc"], 4)
        })

    # 5ï¸�âƒ£ Save summary table
    thr_df = pd.DataFrame(summary_rows)
    thr_csv = os.path.join(outdir, "threshold_summary.csv")
    thr_df.to_csv(thr_csv, index=False)
    printv("\n=== Threshold Summary ===")
    print(thr_df)
    printv(f"Saved -> {thr_csv}")

    try:
        if perm_importances is not None:
            perm_importances.loc[final_feats].sort_values(ascending=False).head(50).plot(kind='bar', figsize=(10,4))
            plt.title("Permutation importances (selected features subset)")
            fn = os.path.join(outdir, "perm_importances_selected.png"); plt.savefig(fn); plt.close(); printv("Saved:", fn)
        elif "RandomForestClassifier" in stored_models:
            rf = stored_models["RandomForestClassifier"]
            imp = pd.Series(rf.feature_importances_, index=X.columns).loc[final_feats].sort_values(ascending=False)
            imp.head(50).plot(kind='bar', figsize=(10,4))
            fn = os.path.join(outdir, "rf_importances_selected.png"); plt.savefig(fn); plt.close(); printv("Saved:", fn)
    except Exception as e:
        printv("Feature importance plotting failed:", e)

    dump(summary, os.path.join(outdir, "modeling_summary_with_thresholds.joblib"))
    printv(f"\nPipeline finished in {(time.time()-start_all):.2f}s.")
    return final_feats, summary , models_fitted


import warnings
warnings.filterwarnings('ignore')
final_feats, summary, models_fitted = run_feature_selection_and_modeling(X_train_opt, y_train, X_test=X_test_opt, n_features_final=50)


best_model = max(summary.keys(), key=lambda m: summary[m]["oof_metrics"]["roc_auc"])
print("BEST MODEL =", best_model)



best_test_preds = summary[best_model]["test_pred"]


sub = pd.DataFrame({
    "SK_ID_CURR": ids,
    "TARGET": best_test_preds
})



sub


sub_path = os.path.join(OUTDIR, f"submission_{best_model}.csv")
sub.to_csv(sub_path, index=False)
print("Submission saved:", sub_path)


def simple_average_ensemble(summary, y_train):
    # get model names
    model_names = list(summary.keys())
    
    # load OOF preds
    oof_preds = [np.load(os.path.join(OUTDIR, f"oof_preds_{m}.npy")) 
                 for m in model_names]
    test_preds = [summary[m]["test_pred"] for m in model_names]

    # soft-vote (average)
    oof_ens = np.mean(np.column_stack(oof_preds), axis=1)
    test_ens = np.mean(np.column_stack(test_preds), axis=1)

    # threshold tuning
    prec, rec, thr = precision_recall_curve(y_train, oof_ens)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_thr = thr[np.nanargmax(f1)]

    y_pred = (oof_ens >= best_thr).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_train, oof_ens),
        "threshold": best_thr,
        "oof_pred": oof_ens,
        "test_pred": test_ens,
    }

    return metrics



met = simple_average_ensemble(summary,y_train)


print(met)


import shap

def compute_shap_values(model,X_train,max_samples=5000):
    X_sample = X_train.sample(max_samples,random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values, X_sample


best_model = models_fitted[best_model][0]

explainer, shap_values, X_sample = compute_shap_values(best_model,y_train)

os.makedirs(outdir,exist_ok=True)
plt.figure()
shap.summary_plot(shap_values,X_sample,show=False)
plt.savefig(os.path.join(outdir,"shap_summary.png"),dpi=300,bbox_inches="tight")
plt.close()


plt.figure()
shap.summary_plot(shap_values,X_sample,plot_type="bar",show=False)
plt.savefig(os.path.join(outdir,"shap_bar.png"),dpi=300,bbox_inches="tight")
plt.close()

