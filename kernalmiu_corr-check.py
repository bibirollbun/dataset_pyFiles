
!pip install /kaggle/input/janestreet2025-code/janestreet-0.1-py3-none-any.whl --force-reinstall --no-deps


import polars as pl


base = "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet"

files = [
    f"{base}/partition_id={i}/part-0.parquet"
    for i in range(8)  # partition 0â€“6
]

df = pl.scan_parquet(base)
df = df.filter(
            pl.col("partition_id").is_in([1, 2, 3, 4, 5, 6, 7])
        )
df = df.drop("partition_id")
df = df.fill_null(0.0)


df.limit(5).collect()


time_counts = df.group_by("date_id").agg(pl.col("time_id").n_unique())



time_counts = time_counts.collect().sort('date_id')
time_counts


import matplotlib.pyplot as plt
dates = time_counts["date_id"].to_numpy()
counts = time_counts["time_id"].to_numpy()
plt.figure(figsize=(14, 7))

# ç»˜åˆ¶æ•£ç‚¹å›¾/æŠ˜çº¿å›¾
plt.plot(dates, counts, marker='o', linestyle='-', markersize=4, color='b', label='Timesteps per Day')

# æ ‡è®°å…³é”®çš„ç¨³å®šå€¼ (968)
plt.axhline(y=968, color='r', linestyle='--', label='Stabilization Target (968)') 

# è®¾ç½®æ ‡é¢˜å’Œæ ‡ç­¾
plt.title('Time Steps Stability Over Trading Days (Date ID)', fontsize=16)
plt.xlabel('Date ID', fontsize=14)
plt.ylabel('Number of Time Steps (TimestepsPerDay)', fontsize=14)

# æ˜¾ç¤ºå›¾ä¾‹å’Œç½‘æ ¼
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


stable_dates = time_counts.filter(
    pl.col("time_id") == 968
)
stable_dates[0:5]


df_after677 = df.filter(pl.col("date_id") >= 677)


f61_corr_with_time = df_after677.select(
    corr_with_date = pl.corr("feature_61", "date_id")
).collect()
print(f61_corr_with_time)


target_features = [
    "feature_61", "feature_02", "feature_03", "feature_00", "feature_34", 
    "feature_35", "feature_32", "feature_27", "feature_28", "feature_20", 
    "feature_62"
]

avg_exprs = [
    pl.col(col).mean().alias(f"avg_{col}")
    for col in target_features
]

df_daily_avg = df_after677.group_by("date_id").agg(
    avg_exprs
)
corr_exprs = [
    pl.corr(f"avg_{col}", "date_id").alias(col) # ä½¿ç”¨å�Ÿå§‹ feature name ä½œä¸ºåˆ—å��
    for col in target_features
]
correlation_result = df_daily_avg.select(
    corr_exprs
).collect()

feature_names = correlation_result.columns
result_transposed = correlation_result.transpose(
    column_names=["Correlation Value"], 
    include_header=True, 
    header_name="Feature"
).with_columns(
    pl.Series(feature_names).alias("Feature")
).select(
    "Feature",
    "Correlation Value"
)

result_sorted = result_transposed.with_columns(
    pl.col("Correlation Value").abs().alias("Abs_Corr")
).sort("Abs_Corr", descending=True)


print(result_sorted.drop("Abs_Corr"))


feature_cols = [f"feature_{i:02d}" for i in range(79)]
corr_date_df = df_after677.select(
    [pl.corr(col, "date_id").alias(col) for col in feature_cols]
).collect()

corr_time_df = df_after677.select(
    [pl.corr(col, "time_id").alias(col) for col in feature_cols]
).collect()


corr_data = {
    "Feature": feature_cols,
    "Corr_with_Date": corr_date_df.transpose().to_series().to_list(),
    "Corr_with_Time": corr_time_df.transpose().to_series().to_list()
}

result_df = pl.DataFrame(corr_data)

# 5. æ·»åŠ ç»�å¯¹å€¼åˆ—ä»¥ä¾¿æ�’åº�ï¼ˆæˆ‘ä»¬éœ€è¦�çœ‹ç›¸å…³æ€§å¼ºå¼±ï¼Œä¸�è®ºæ­£è´Ÿï¼‰
result_df = result_df.with_columns(
    pl.col("Corr_with_Date").abs().alias("Abs_Corr_Date"),
    pl.col("Corr_with_Time").abs().alias("Abs_Corr_Time")
)


# --- æ¦œå�• A: ä¸�æ—¥æœŸ(é•¿æœŸè¶‹åŠ¿)ç›¸å…³æ€§æœ€é«˜çš„ç‰¹å¾� ---
print("\nğŸ”¥ Top 10 ä¸� æ—¥æœŸ (Date_ID) ç›¸å…³æ€§æœ€é«˜çš„ç‰¹å¾�ï¼š")
top_date = result_df.sort("Abs_Corr_Date", descending=True).head(10)
print(top_date.select(["Feature", "Corr_with_Date"]))

# --- æ¦œå�• B: ä¸�æ—¥å†…æ—¶é—´(æ—¥å†…æ¨¡å¼�)ç›¸å…³æ€§æœ€é«˜çš„ç‰¹å¾� ---
print("\nâ�° Top 10 ä¸� æ—¥å†…æ—¶é—´ (Time_ID) ç›¸å…³æ€§æœ€é«˜çš„ç‰¹å¾�ï¼š")
top_time = result_df.sort("Abs_Corr_Time", descending=True).head(10)
print(top_time.select(["Feature", "Corr_with_Time"]))


import polars as pl

# 1. å®šä¹‰ç‰¹å¾�åˆ—è¡¨
COLS_FEATURES_CORR = [
    'feature_06', 'feature_04', 'feature_07', 'feature_36',
    'feature_60', 'feature_45', 'feature_56', 'feature_05',
    'feature_51', 'feature_19', 'feature_66', 'feature_59',
    'feature_54', 'feature_70', 'feature_71', 'feature_72',
]

# 2. å®šä¹‰ä½ æƒ³è®¡ç®—çš„ç›®æ ‡åˆ—è¡¨
target_cols = ['responder_6', 'responder_7', 'responder_8']

# 3. å¾ªç�¯è®¡ç®—å¹¶æ‰“å�°ç»“æ�œ
for target in target_cols:
    print(f"\n=== å�„ç‰¹å¾�ä¸� {target} çš„ç›¸å…³æ€§ (Pearson) ===")
    
    try:
        # è®¡ç®—å½“å‰� target ä¸�åˆ—è¡¨ç‰¹å¾�çš„ç›¸å…³æ€§
        # æ³¨æ„�ï¼šå¦‚æ�œ df_after677 å·²ç»�æ˜¯ DataFrame è€Œä¸�æ˜¯ LazyFrameï¼Œè¯·å�»æ�‰ .collect()
        df_corr = df_after677.select(
            [pl.corr(col, target).alias(col) for col in COLS_FEATURES_CORR]
        ).collect()

        # æ��å�–ç»“æ�œå¹¶æ‰“å�°
        results = df_corr.row(0, named=True)
        for feat, corr_val in results.items():
            print(f"{feat:<15}: {corr_val:.6f}")
            
    except Exception as e:
        print(f"è®¡ç®— {target} æ—¶å‡ºé”™: {e}")
        # å�¯èƒ½æ˜¯å› ä¸ºæ•°æ�®ä¸­æ²¡æœ‰ responder_7 æˆ– responder_8




