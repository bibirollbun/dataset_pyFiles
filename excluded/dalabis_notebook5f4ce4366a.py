import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv(
    "/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv",
    dtype={'item_cnt_day': 'Int32'}
)


train.head()


train.dtypes


print(len(train))


agg_features = ["date_block_num", "shop_id", "item_id", "item_cnt_day"]

train_agg = train[agg_features].groupby(["date_block_num", "shop_id", "item_id"]).sum().reset_index()


train_agg.head()


train_agg.describe()


shop_id = 5
item_id = 5037
sample = train_agg[(train_agg["shop_id"] == shop_id) & (train_agg["item_id"] == item_id)]


sample


plt.plot(sample["date_block_num"], sample["item_cnt_day"], 'x')
plt.xlabel("date_block_num")
plt.ylabel("item_cnt_day");


train_agg_total = train_agg.groupby("date_block_num")["item_cnt_day"].sum().reset_index()


train_agg_total


plt.plot(train_agg_total["date_block_num"], train_agg_total["item_cnt_day"], '-x')
plt.xlabel("date_block_num")
plt.ylabel("item_cnt_day");




