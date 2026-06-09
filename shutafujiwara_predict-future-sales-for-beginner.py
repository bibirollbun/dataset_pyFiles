#ライブラリのインポート
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.metrics import mean_squared_error

pd.options.display.float_format = '{:.2f}'.format

print('Setup Complete')


#各データの読み込み
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
sample = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

#外れ値の除去
sales_train = sales_train[sales_train.item_price<100000]
sales_train = sales_train[sales_train.item_cnt_day<1001]

print('load Complete')


#それぞれのデータを見る
#itemsの場合
items


#shopsの場合


#categorisの場合


#sales_trainの場合


#test


#sample


#新たな特徴量を追加する
#日ごとの売上合計列を追加する(商品価格*個数)
sales_train['total'] = sales_train['item_price'] * sales_train['item_cnt_day']

sales_train


#グループ分けをしてみる
#月次(date_block_num)ごとに
sales_train.groupby('date_block_num').mean('total')


#データを図示化してみる
plt.figure(figsize=(10,4))
sns.scatterplot(x=sales_train['date_block_num'],y=sales_train['total'])


#ワークショップ①
#実際にデータ操作してみる


#ワークショップ②
#データの前処理
train_data = sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
train_data.columns = ['item_cnt_month']
train_data.reset_index(inplace=True)

test = test.drop(['ID'],axis=1)
test['date_block_num']=34
test=test.reindex(columns=['date_block_num','shop_id','item_id'])
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

merged= pd.concat([train_data,test],ignore_index=True,sort=False,keys=['date_block_num','shop_id','item_id'])
merged.fillna(0,inplace=True)
print('Pretreatment Completed')


#訓練データ・テストデータ・検証データの作成
X_train = merged[merged.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = merged[merged.date_block_num < 33]['item_cnt_month']
X_valid = merged[merged.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = merged[merged.date_block_num == 33]['item_cnt_month']
X_test = merged[merged.date_block_num ==34].drop(['item_cnt_month'],axis=1)

print('splited_data Complete')



# 学習データの中身を確認
X_train


#modelの作成・実行
model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42
)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    early_stopping_rounds = 10)


prediction = model.predict(X_valid)

#精度の確認
rmse = np.sqrt(mean_squared_error(prediction,Y_valid))
print("RMSE:",str(rmse))


#モデルの精度を上げる


#コンペの提出
Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('xgb_submission.csv', index=False)


