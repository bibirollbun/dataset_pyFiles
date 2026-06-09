import polars as pl
import numpy as np
train=pl.read_csv("/kaggle/input/user-retention-prediction/train.csv").with_columns([
    ((1540051199-pl.col("Timestamp"))//(24*60*60)).alias("days")
])
sub=pl.read_csv("/kaggle/input/user-retention-prediction/submit_sample.csv")


last_time=train.group_by("ID").agg((1540051199-pl.col("Timestamp").max()).alias("lasttime"))


sub_last_time=sub.join(last_time,on="ID",how="left").fill_null(2*24*3600)
sub_last_time["lasttime"].max()/24/3600


label=train.filter(pl.col("days")==7)[["ID"]].unique().join(
    train.filter(pl.col("days")<7).group_by(["ID"]).agg([
    pl.col("days").n_unique().cast(int).alias("label")
]),on="ID",how="left").fill_null(0)
dist=label.group_by("label").agg((pl.col("ID").count()/len(label)).alias("ratio")).sort("label")
thresholds=np.cumsum(dist["ratio"].to_list())


shift_days=7


features=train.filter(pl.col("days")==shift_days)[["ID"]].unique().join(
    train.filter(pl.col("days").is_in([shift_days+i for i in range(7)])).group_by(["ID","days"]).agg([
    pl.col("ActionType").count().alias("action_count"),
]),on="ID",how="left").fill_null(0)


import numpy as np
score=features.with_columns([
    (1/(pl.col("days")-shift_days+1)).alias("score1"),
    pl.lit(1).alias("score2"),
    (pl.col("action_count").log1p()/(pl.col("days")-shift_days+1)).alias("score3"),
]).group_by(["ID"]).agg([
    pl.col("score1").sum(),
    pl.col("score2").sum(),
    pl.col("score3").sum(),
]).with_columns([
    (pl.col("score1").rank(method="ordinal")/pl.col("ID").count()).alias("pred")
])
score=score.with_columns(
    pl.Series(np.digitize(pl.Series(score["pred"]), thresholds).clip(0,7)).alias("pred")
).sort("ID")


score=score.join(label,on=["ID"],how="left")


score


(200*abs(score["pred"]-score["label"])/(score["pred"]+score["label"])).fill_nan(0).mean()


(200*abs(score["score2"]-score["label"])/(score["score2"]+score["label"])).fill_nan(0).mean()


shift_days=0
features=train.filter(pl.col("days")==shift_days)[["ID"]].unique().join(
    train.filter(pl.col("days").is_in([shift_days+i for i in range(7)])).group_by(["ID","days"]).agg([
    pl.col("ActionType").count().alias("action_count"),
]),on="ID",how="left").fill_null(0)

score=features.with_columns([
    (1/(pl.col("days")-shift_days+1)).alias("score1"),
    pl.lit(1).alias("score2"),
    (pl.col("action_count").log1p()/(pl.col("days")-shift_days+1)).alias("score3"),
]).group_by(["ID"]).agg([
    pl.col("score1").sum(),
    pl.col("score2").sum(),
    pl.col("score3").sum(),
]).with_columns([
    (pl.col("score1").rank(method="ordinal")/pl.col("ID").count()).alias("pred")
])
thresholds=np.cumsum([0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,
       0.07644968, 0.12103906, 0.46647806])
score=score.with_columns(
    pl.Series(np.digitize(pl.Series(score["pred"]), thresholds).clip(0,7)).alias("pred")
).sort("ID")


sub=sub[["ID"]].join(score[["ID","pred"]],on="ID",how="left")
sub.write_csv("submission.csv")




