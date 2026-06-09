import pandas as pd
import numpy as np


item_cat = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
sales = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")


item_cat.head()


items.head()


right=pd.merge(item_cat,items , how = 'right', on='item_category_id')
right


left = pd.merge(sales , shops, how='outer', on='shop_id')
left


pd.merge(right,left , how = 'outer' , on ='item_id')


#شرح الدرس 
# https://www.youtube.com/watch?v=zzzkKVRctCo&list=PLuRv1IekA3YVwzaWa2Kp7bgIVcJsJ5XGW&index=6&ab_channel=Ons%D8%A3%D9%86%D8%B3 
a = pd.DataFrame({'cat_name':['Elec','Games','Kitchen'],'cat_id':[0,1,2]})
a



b = pd.DataFrame({'item_name':['MacBook','painting','cup'],'item_id':[0,1,2] , 'cat_id':[0,5,2]})
b


pd.merge(a , b ,how='inner',on = 'cat_id')


pd.merge(a,b , how='outer', on='cat_id')


pd.merge(a,b, how='left', on='cat_id')


pd.merge(a,b,how = 'right', on='cat_id')


lefts = pd.DataFrame({'key':['foo','bar'] , 'val':[1,2]})
rights = pd.DataFrame({'key':['foo','bar'] ,'val' :[4,5]})



lefts.join(rights , how= 'outer', lsuffix='_from_left',rsuffix ='_from_right').drop(columns='key_from_right')

