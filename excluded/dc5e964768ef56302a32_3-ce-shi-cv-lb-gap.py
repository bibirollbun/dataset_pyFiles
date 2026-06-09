import polars as pl
import numpy as np
train=pl.read_csv("/kaggle/input/user-retention-prediction/train.csv").with_columns([
    ((1540051199-pl.col("Timestamp"))//(24*60*60)).alias("days")
])
sub=pl.read_csv("/kaggle/input/user-retention-prediction/submit_sample.csv")


last_time=train.group_by("ID").agg((1540051199-pl.col("Timestamp").max()).alias("lasttime"))


sub_last_time=sub.join(last_time,on="ID",how="left").fill_null(2*24*3600)
sub_last_time["lasttime"].max()/24/3600


def create_sample(shift_days):
    train_id=train.filter(pl.col("days")==shift_days)[["ID"]].unique()
    train_features=train.filter(pl.col("ID").is_in(train_id["ID"].to_list())
                               ).filter(pl.col("days").is_in([shift_days+i for i in range(5)])).with_columns(
        pl.col("days")-shift_days,
        ((1540051199-pl.col("Timestamp"))%(24*60*60)).alias("seconds")
        
    )
    if shift_days>0:
        label=train_id.join(
            train.filter(pl.col("days").is_in([shift_days-i for i in range(1,8)])).group_by(["ID"]).agg([
            pl.col("days").n_unique().cast(int).alias("label")
        ]),on="ID",how="left").fill_null(0)
        dist=label.group_by("label").agg((pl.col("ID").count()/len(label)).alias("ratio")).sort("label")
        thresholds=np.cumsum(dist["ratio"].to_list())
    else:
        label=None
        thresholds=np.cumsum([0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,
                             0.07644968, 0.12103906, 0.46647806])

    return train_features,label,thresholds
    #return sub,label

def create_features(train_features):
    
    pass
    


%%time
train_features,label,thresholds=create_sample(7)


def create_sub(train_features,thresholds):
    sub=train_features.group_by(["ID","days"]).agg([
        pl.col("ActionType").count().alias("action_count"),
        pl.col("ActionType").n_unique().alias("action_nunique"),
    ]).sort(["ID","days"]).with_columns([
        (1/(pl.col("days")+1)).alias("score1"),
        (pl.col("action_nunique")/(pl.col("days")+1)).alias("score2"),
        (pl.col("action_count").log1p()/(pl.col("days")+2)).alias("score3"),
    ]).group_by(["ID"]).agg([
        pl.col("score1").sum(),
        pl.col("score2").sum(),
        pl.col("score3").sum(),
    ]).with_columns([
        ((#(pl.col("score1")+pl.col("score3"))*pl.col("score2")
            # pl.col("score3")
            pl.col("score1")
         ).rank(method="ordinal")/pl.col("ID").count()).alias("pred")
    ])
    sub=sub.with_columns(
        pl.Series(np.digitize(pl.Series(sub["pred"]), thresholds).clip(0,7)).alias("pred")
    ).sort("ID")
    return sub


sub=create_sub(train_features,thresholds)


sub=label.join(sub,on=["ID"],how="left").fill_null(0)
(200*abs(sub["pred"]-sub["label"])/(sub["pred"]+sub["label"])).fill_nan(0).mean()


# cv                        lb
# 46.76443723940843        43.18
# 43.52566836947838
# 43.14134101293236        44.49
# 42.85870686779661
# 42.78157593626163        44.55
# 48.38952297696376        


#线上提交
train_features,label,thresholds=create_sample(0)
sub=create_sub(train_features,thresholds)
sub[["ID","pred"]].write_csv("submission.csv")







