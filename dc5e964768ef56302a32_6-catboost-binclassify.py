import polars as pl
import numpy as np
path="/kaggle/input/user-retention-prediction/"
train=pl.read_csv(f"{path}train.csv").with_columns([
    ((1540051199-pl.col("Timestamp"))//(24*60*60)).alias("days")
])
sub=pl.read_csv(f"{path}submit_sample.csv")


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
            pl.col("days").n_unique().cast(int).alias("label"),
            *[(pl.col("days")==shift_days-i).cast(int).max().alias("label_%s"%i) for i in range(1,8)]
        ]),on="ID",how="left").fill_null(0)
        dist=label.group_by("label").agg((pl.col("ID").count()/len(label)).alias("ratio")).sort("label")
        thresholds=np.cumsum(dist["ratio"].to_list())
    else:
        label=None
        thresholds=np.cumsum([0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,
                             0.07644968, 0.12103906, 0.46647806])

    return train_features,label,thresholds
    #return sub,label


%%time
train_features,train_label,train_thresholds=create_sample(14)
valid_features,valid_label,valid_thresholds=create_sample(7)


def create_features(train_features):
    train_id=train_features[["ID"]].unique()
    for k in range(7):
        train_features_temp=train_features.filter(pl.col("days")==k)
        features=train_features_temp.group_by("ID").agg([
        *[(pl.col("ActionType")==n).sum().alias(f"ActionType_{n}_{k}") for n in range(5)],
        pl.col("seconds").min().alias(f"seconds_min_{k}"),
        pl.col("seconds").max().alias(f"seconds_max_{k}"),
        pl.col("seconds").count().alias(f"seconds_count_{k}"),
    ])
        train_id=train_id.join(features,on="ID",how="left")
    return train_id


train_data=create_features(train_features)
train_data=train_data.join(train_label,on="ID",how="left")

valid_data=create_features(valid_features)
valid_data=valid_data.join(valid_label,on="ID",how="left")


train_data.shape,valid_data.shape


features_name=[i for i in train_data.columns if i not in ["ID","label","label_1","label_2","label_3","label_4","label_5","label_6","label_7"]]


from catboost import CatBoostClassifier,CatBoostRegressor,CatBoostRanker
params={
'iterations':2000,
'loss_function':'Logloss',
'learning_rate':0.05,
'depth':5,
'verbose':False,
'task_type':'GPU',
}

pred=np.zeros(len(valid_data))
for label in ["label_1","label_2","label_3","label_4","label_5","label_6","label_7"]:
    model = CatBoostClassifier(**params)
    model.fit(train_data[features_name].to_numpy(), train_data[label].to_numpy())
    pred+=model.predict_proba(valid_data[features_name].to_numpy())[:,1]


sub=valid_data[["ID","label"]].to_pandas()
sub["pred"]=pred
sub["pred"]=sub["pred"].rank()
sub["pred"]=sub["pred"]/(sub["pred"].max())
sub["pred"]=np.digitize(sub["pred"], valid_thresholds).clip(0,7)


(200*abs(sub["pred"]-sub["label"])/(sub["pred"]+sub["label"])).fillna(0).mean()


test_features,test_label,test_thresholds=create_sample(0)
test_data=create_features(test_features)


pred=np.zeros(len(test_data))
for label in ["label_1","label_2","label_3","label_4","label_5","label_6","label_7"]:
    model = CatBoostClassifier(**params)
    model.fit(valid_data[features_name].to_numpy(), valid_data[label].to_numpy())
    pred+=model.predict_proba(test_data[features_name].to_numpy())[:,1]


sub=test_data[["ID"]].to_pandas()
sub["pred"]=pred
sub["pred"]=sub["pred"].rank()
sub["pred"]=sub["pred"]/(sub["pred"].max())
sub["pred"]=np.digitize(sub["pred"], valid_thresholds).clip(0,7)


sub.to_csv("submission.csv",index=None)




