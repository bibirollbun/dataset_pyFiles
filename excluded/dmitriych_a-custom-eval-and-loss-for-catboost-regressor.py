# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import polars as pl
import polars.selectors as cs

from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv

import numpy as np
#import matplotlib.pyplot as plt
#from colorama import Fore, Style

from sklearn.model_selection import TimeSeriesSplit


%%time
pth = "/kaggle/input/china-real-estate-demand-prediction"

ci = pl.read_csv(pth+'/train/city_indexes.csv'
                ).head(6).fill_null(-1).drop("total_fixed_asset_investment_10k"
                                            ).rename(lambda column_name:("" if column_name in ["sector","month"] else  "ci_") + column_name) # one row per year

csi = pl.read_csv(pth+'/train/city_search_index.csv') # several rows per training month

sp = pl.read_csv(pth+'/train/sector_POI.csv'
                ).fill_null(-1).rename(lambda column_name:("" if column_name in ["sector","month"] else  "sp_") + column_name) # at most one row per sector

train_lt = pl.read_csv(pth+'/train/land_transactions.csv', infer_schema_length=10000
                      ).rename(lambda column_name:("" if column_name in ["sector","month"] else  "lt_") + column_name)

train_ltns = pl.read_csv(pth+'/train/land_transactions_nearby_sectors.csv'
                        ).rename(lambda column_name:("" if column_name in ["sector","month"] else  "ltns_") + column_name)

train_pht = pl.read_csv(pth+'/train/pre_owned_house_transactions.csv'
                       ).rename(lambda column_name:("" if column_name in ["sector","month"] else  "pht_") + column_name)

train_phtns = pl.read_csv(pth+'/train/pre_owned_house_transactions_nearby_sectors.csv'
                         ).rename(lambda column_name: ("" if column_name in ["sector","month"] else "phtns_") + column_name)

train_nht = pl.read_csv(pth+'/train/new_house_transactions.csv'
                       ).rename(lambda column_name: ("" if column_name in ["sector","month"] else "nht_") + column_name)

train_nhtns = pl.read_csv(pth+'/train/new_house_transactions_nearby_sectors.csv'
                         ).rename(lambda column_name: ("" if column_name in ["sector","month"] else "nhtns_") + column_name)

test = pl.read_csv(pth+'/test.csv')

test = test.with_columns(id_=pl.col("id").str.split("_")
          ).with_columns(month=pl.col("id_").list.get(0),
                         sector=pl.col("id_").list.get(1)
          ).drop("id_")
test 
train_nhtns
train_nht


month_codes = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}


%%time
data = pl.DataFrame( train_nht["month"].unique() 
            ).join(pl.DataFrame( train_nht["sector"].unique().to_list()+["sector 95"] 
                               ).rename({"column_0":"sector"}), 
                   how="cross" 
            ).with_columns(sector_id=pl.col("sector").str.split(" ").list.get(1).cast(pl.Int8),
                      year=pl.col("month").str.split("-").list.get(0).cast(pl.Int16),
                      month_num=pl.col("month").str.split("-").list.get(1).replace(month_codes).cast(pl.Int8),
            ).with_columns(
                      time=( (pl.col("year") - 2019) * 12 + pl.col("month_num") - 1 ).cast(pl.Int8)
            ).sort("sector_id","time"
                   
            ).join(train_nht, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(0
            ).join(train_nhtns, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(-1
            ).join(train_pht, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(-1
            ).join(train_phtns, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(-1
                       
            ).join(ci.rename({"ci_city_indicator_data_year":"year"}), 
                   on=["year"],
                   how="left"
            ).fill_null(-1
            ).join(sp, 
                   on=["sector"],
                   how="left"
            ).fill_null(-1
                        
            ).join(train_lt, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(-1
            ).join(train_ltns, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(-1
            ).with_columns(cs.float().cast(pl.Float32)
            )
data.max()


#Do you have any ideas how to use csi?
csi


%%time
for col in data.columns:
    if data[col].dtype==pl.Int64:
        c_min = data[col].min()
        c_max = data[col].max()
        if c_min == 0 and c_max == 0:
            data = data.drop(col)
            print(col, "0"*20 )
        elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            data = data.with_columns( pl.col(col).cast(pl.Int8) )
            print(col, data[col].dtype, data[col].min(), data[col].max() )
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            data = data.with_columns( pl.col(col).cast(pl.Int16) )
            print(col, data[col].dtype, data[col].min(), data[col].max() )
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            data = data.with_columns( pl.col(col).cast(pl.Int32) )
            print(col, data[col].dtype, data[col].min(), data[col].max() )
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
            data = data.with_columns( pl.col(col).cast(pl.Int64) )
            print(col, data[col].dtype, data[col].min(), data[col].max() )
            
data = data.drop("month","sector","year")            
data


%%time
data2=data
data2=data2.sort("time","sector_id"
        )

for col in data2.columns[3:]:
    for p in [#3,
              #6,
              #12,
             ]:
            print(col)
            data2=data2.with_columns(
                pl.col(col).rolling_mean(window_size=p).alias(f"{col}_mean{p}"), 
                pl.col(col).rolling_min(window_size=p).alias(f"{col}_min{p}"), 
                pl.col(col).rolling_max(window_size=p).alias(f"{col}_max{p}"), 
            #).with_columns(
            #    ( pl.col(f"{col}_max{p}")-pl.col(f"{col}_min{p}") ).alias(f"{col}_maxmin{p}")
            )

for m in [1,2,12]:
    data2=data2.join(data.drop("month_num").with_columns(pl.col("time")+m),
              on=["sector_id","time"],
              how="left",
              suffix=f"_{m}"
        )
data2=data2.sort("time","sector_id"
        )
data2


lag=-1


%%time
data3 = data2.with_columns(pl.col("nht_amount_new_house_transactions").shift(lag).over("sector_id").alias("label"),
                           #pl.col("amount_new_house_transactions").shift(lag+1).over("sector_id").alias("label1"),
                           #pl.col("amount_new_house_transactions").shift(lag+2).over("sector_id").alias("label2"),
                           #pl.col("amount_new_house_transactions").shift(lag+3).over("sector_id").alias("label3"),
                           #pl.col("amount_new_house_transactions").shift(lag+4).over("sector_id").alias("label4"),
                           #pl.col("amount_new_house_transactions").shift(lag+5).over("sector_id").alias("label5"),
                           
                           cs=((pl.col("month_num")-1)/6*np.pi).cos(),
                           sn=((pl.col("month_num")-1)/6*np.pi).sin(),
                           cs6=((pl.col("month_num")-1)/3*np.pi).cos(),
                           sn6=((pl.col("month_num")-1)/3*np.pi).sin(),
                           cs3=((pl.col("month_num")-1)/1.5*np.pi).cos(),
                           sn3=((pl.col("month_num")-1)/1.5*np.pi).sin(),
#                          ).with_columns( -pl.col("label")+1
                          )#.drop_nulls(subset=["label"])
data3 = data3.drop("sector_id")
data3


%%time
cat_features = [#"sector_id",
                "month_num"
               ]
border=66+lag-12*0-1
#border1=border//2 #-1
border1=border-6*6
border1=6*3
trainPool = Pool(data=data3.filter(pl.col("time")<=border
                                  #).filter(pl.col("label")>10
                                  ).filter(pl.col("time")>border1
                                          ).drop(["label",
                                                  #"label1","label2","label3","label4","label5",
                                                 ]).to_pandas().fillna(-2),
                 label=data3.filter(pl.col("time")<=border
                                  #).filter(pl.col("label")>10
                                   ).filter(pl.col("time")>border1
                                           )["label",
                                             #"label1","label2","label3","label4","label5",
                                            ].to_pandas(),
                 cat_features = cat_features, 
                 #text_features = text_features, 
                )

testPool = Pool(data=data3.filter(pl.col("time")>border
                                  ).filter(pl.col("time")<=66+lag
                                  #).filter(pl.col("label")>10
                                 ).drop(["label",
                                         #"label1","label2","label3","label4","label5",
                                        ]).to_pandas().fillna(-2),
                 label=data3.filter(pl.col("time")>border
                                  ).filter(pl.col("time")<=66+lag
                                  #).filter(pl.col("label")>10
                                   )["label",
                                     #"label1","label2","label3","label4","label5",
                                    ].to_pandas(),
                 cat_features = cat_features, 
                 #text_features = text_features, 
                )
trainPool


#custom eval metric for catboost

# based on https://www.kaggle.com/competitions/china-real-estate-demand-prediction/discussion/598174

def custom_score(y_true, y_pred, eps=1e-12):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        return 0.0

    ape = np.abs((y_true - np.maximum(y_pred,0) ) / np.maximum(y_true, eps))

    bad_rate = np.mean(ape > 1.0)
    if bad_rate > 0.30:
        return 0.0

    mask = ape <= 1.0
    good_ape = ape[mask]

    if good_ape.size == 0:
        return 0.0

    mape = np.mean(good_ape)

    fraction = good_ape.size / y_true.size
    scaled_mape = mape / (fraction + eps)
    score = max(0.0, 1.0 - scaled_mape)
    return score

class CustomMetric:
    def is_max_optimal(self):
        return True # greater is better

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        y_pred = approx
        y_true = target

        output_weight = 1 # weight is not used

        score = custom_score( target, approx )
 
        return score, output_weight

    def get_final_error(self, error, weight):
        return error


#custom loss for catboost
class CustomObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        result = []
        delta = 10**(-6)
        for i in range(len(targets)):
            diff = (targets[i] - approxes[i])#/(targets[i]+delta)
            #der1 = np.sign(diff)
            der1 = np.sign(diff) if (2*targets[i] - approxes[i]) < 0 else np.sign(diff)*5
            der2 = 0
            result.append((der1, der2))
        return result  


testPool2 = Pool(data=data3.filter(pl.col("time")==66
                                 ).drop(["label"]).to_pandas().fillna(-2),
                 cat_features = cat_features, 
                )

cb = CatBoostRegressor(iterations=21000,
                            learning_rate=0.0125,
                            one_hot_max_size=256,
                            custom_metric=[                            
                                           "RMSE",
                                           "MAPE",
                                           "SMAPE",
                                           "MAE",
                            ],
                            loss_function=CustomObjective(),
                            eval_metric=CustomMetric(),
                            l2_leaf_reg=0.3,
                            random_seed=4,
                           )
cb.fit(trainPool,
           eval_set=testPool,
           #early_stopping_rounds=4000,
           #use_best_model=False,
           #plot=True,
           verbose=1000,           
           )
month=np.maximum(cb.predict(testPool2),0)
month


#%%time
importance = cb.get_feature_importance(prettified=True,data=testPool)
importance.to_csv("importance.csv")
importance.head(60)


#importance.tail(60)


#based on https://www.kaggle.com/code/ambrosm/red-explained-baseline/notebook
for i in [11,38,40,43,48,51,52,57,71,72,73,74,81,86,88,94,95]:
    month[i]=0
month


sub = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv")
sub


for m in range(12):
    sub.loc[[i+m*96 for i in range(96)], "new_house_transaction_amount"]=month
sub.to_csv("submission.csv", index=False)
sub

