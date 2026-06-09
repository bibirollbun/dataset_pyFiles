import polars as pl
n=6
submission=pl.read_csv("/kaggle/input/user-retention-prediction/submit_sample.csv")
train=pl.read_csv("/kaggle/input/user-retention-prediction/train.csv")
pred=train.with_columns([
    ((1540051200-pl.col("Timestamp"))//(24*60*60)).alias("days")
]).filter(pl.col("days")<n).group_by("ID").agg(
    (pl.col("days").n_unique()/n*7).alias("pred")
)
sub=submission[["ID"]].join(pred,on="ID",how="left").fill_null(0)
sub.write_csv("submission.csv")


sub["pred"].min()




