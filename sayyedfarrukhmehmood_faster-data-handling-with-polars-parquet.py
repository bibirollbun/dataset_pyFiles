import polars as pl

# Load train.csv
csv_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
df = pl.read_csv(csv_path)

# Quick shape and schema
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
df.head(1)


# Output path
output_path = "train_parquet/"

# Save as Parquet, partitioned by "subject" column values
df.write_parquet(
    output_path, 
    partition_by=["subject"]
)



%%time
df_lazy = pl.scan_parquet("train_parquet/")


# Example: load one subject's data
df_subject_0 = pl.read_parquet("train_parquet/subject=SUBJ_000206/*.parquet")
df_subject_0.head()



# Lazy load + compute mean acceleration per sequence
features = (
    df_lazy
    .group_by("sequence_id")
    .agg([
        pl.col("acc_x").mean().alias("acc_x_mean"),
        pl.col("acc_y").mean().alias("acc_y_mean"),
        pl.col("acc_z").mean().alias("acc_z_mean"),
    ])
)

features_df = features.collect()
features_df.head()

