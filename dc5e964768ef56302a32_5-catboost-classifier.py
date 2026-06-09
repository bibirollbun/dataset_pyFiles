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
            pl.col("days").n_unique().cast(int).alias("label")
        ]),on="ID",how="left").fill_null(0)
        dist=label.group_by("label").agg((pl.col("ID").count()/len(label)).alias("ratio")).sort("label")
#         thresholds=np.cumsum(dist["ratio"].to_list())
        thresholds=dist["ratio"].to_list()
    else:
        label=None
#         thresholds=np.cumsum([0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,
#                              0.07644968, 0.12103906, 0.46647806])
        thresholds=[0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,
                             0.07644968, 0.12103906, 0.46647806]

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


features_name=[i for i in train_data.columns if i not in ["ID","label"]]


from catboost import CatBoostClassifier,CatBoostRegressor,CatBoostRanker
params={
'iterations':2000,
'loss_function':'MultiClass',
'learning_rate':0.05,
'depth':5,
'verbose':100,
# 'eval_metric':"SMAPE",
'task_type':'GPU',
}
model = CatBoostClassifier(**params)
model.fit(train_data[features_name].to_numpy(), train_data["label"].to_numpy(),
#          eval_set=(valid_data[features_name].to_numpy(), valid_data["label"].to_numpy())
         )


pred=model.predict_proba(valid_data[features_name].to_numpy())


import numpy as np
from scipy.optimize import minimize,basinhopping
def post_processing(pred,thresholds):
    n_classes = 8

    def adjust_prob(prob, weights):
        adjusted_prob = prob * weights
        return adjusted_prob / adjusted_prob.sum(axis=1, keepdims=True)

    def objective(weights):
        adjusted_prob = adjust_prob(pred, weights)
        adjusted_distribution = np.bincount(np.argmax(adjusted_prob,axis=1),minlength=n_classes)/len(adjusted_prob)
        return np.sum((adjusted_distribution - np.array(thresholds))**2)

    initial_weights = np.ones(n_classes)
    bounds = [(1e-6, None) for _ in range(n_classes)]

    result = minimize(objective, initial_weights, bounds=bounds,method='Powell')
    optimal_weights = result.x
    adjusted_prob = adjust_prob(pred, optimal_weights)
    return adjusted_prob


adjusted_prob=post_processing(pred,valid_thresholds)


adjusted_distribution = np.bincount(np.argmax(adjusted_prob,axis=1))/len(adjusted_prob)
print("实际分布:", valid_thresholds)
print("调整后分布:", adjusted_distribution)


sub=valid_data[["ID","label"]].to_pandas()
sub["pred"]=np.argmax(adjusted_prob,axis=1)


#转成回归再后处理
sub=valid_data[["ID","label"]].to_pandas()
sub["pred"]=pred@np.array([0,1,2,3,4,5,6,7])
sub["pred"]=sub["pred"].rank()
sub["pred"]=sub["pred"]/(sub["pred"].max())
sub["pred"]=np.digitize(sub["pred"], np.cumsum(valid_thresholds)).clip(0,7)


(200*abs(sub["pred"]-sub["label"])/(sub["pred"]+sub["label"])).fillna(0).mean()


test_features,test_label,test_thresholds=create_sample(0)
test_data=create_features(test_features)
pred=model.predict_proba(test_data[features_name].to_numpy())
adjusted_prob=post_processing(pred,test_thresholds)
sub=test_data[["ID"]].to_pandas()
sub["pred"]=np.argmax(adjusted_prob,axis=1)


adjusted_distribution = np.bincount(np.argmax(adjusted_prob,axis=1))/len(adjusted_prob)
print("实际分布:", test_thresholds)
print("调整后分布:", adjusted_distribution)


#转成回归再后处理
sub=test_data[["ID"]].to_pandas()
sub["pred"]=pred@np.array([0,1,2,3,4,5,6,7])
sub["pred"]=sub["pred"].rank()
sub["pred"]=sub["pred"]/(sub["pred"].max())
sub["pred"]=np.digitize(sub["pred"], np.cumsum(test_thresholds)).clip(0,7)


sub.to_csv("submission.csv",index=None)




