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


import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def reduce_mem_usage(df, silent=True, allow_categorical=True, float_dtype="float32"):
    """ 
    Iterates through all the columns of a dataframe and downcasts the data type
     to reduce memory usage. Can also factorize categorical columns to integer dtype.
    """
    def _downcast_numeric(series, allow_categorical=allow_categorical):
        """
        Downcast a numeric series into either the smallest possible int dtype or a specified float dtype.
        """
        if pd.api.types.is_sparse(series.dtype) is True:
            return series
        elif pd.api.types.is_numeric_dtype(series.dtype) is False:
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                return series
            else:
                if allow_categorical:
                    return series
                else:
                    codes, uniques = series.factorize()
                    series = pd.Series(data=codes, index=series.index)
                    series = _downcast_numeric(series)
                    return series
        else:
            series = pd.to_numeric(series, downcast="integer")
        if pd.api.types.is_float_dtype(series.dtype):
            series = series.astype(float_dtype)
        return series

    if silent is False:
        start_mem = np.sum(df.memory_usage()) / 1024 ** 2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    if df.ndim == 1:
        df = _downcast_numeric(df)
    else:
        for col in df.columns:
            df.loc[:, col] = _downcast_numeric(df.loc[:,col])
    if silent is False:
        end_mem = np.sum(df.memory_usage()) / 1024 ** 2
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def shrink_mem_new_cols(matrix, oldcols=None, allow_categorical=False):
    # Calls reduce_mem_usage on columns which have not yet been optimized
    if oldcols is not None:
        newcols = matrix.columns.difference(oldcols)
    else:
        newcols = matrix.columns
    matrix.loc[:,newcols] = reduce_mem_usage(matrix.loc[:,newcols], allow_categorical=allow_categorical)
    oldcols = matrix.columns  # This is used to track which columns have already been downcast
    return matrix, oldcols


def list_if_not(s, dtype=str):
    # Puts a variable in a list if it is not already a list
    if type(s) not in (dtype, list):
        raise TypeError
    if (s != "") & (type(s) is not list):
        s = [s]
    return s


path="/kaggle/input/kaggle-predict-future-sales-first-place-solution/abubakar_VVIP/abubakar_VVIP/" # Please adjust this path according to your usage.
items = pd.read_csv(path+"items.csv")
shops = pd.read_csv(path+"shops.csv")
train = pd.read_csv(path+"sales_train.csv")
test = pd.read_csv(path+"test.csv")


params = {
    "num_leaves": 966,
    "cat_smooth": 45.01680827234465,
    "min_child_samples": 27,
    "min_child_weight": 0.021144950289224463,
    "max_bin": 214,
    "learning_rate": 0.01,
    "subsample_for_bin": 300000,
    "min_data_in_bin": 7,
    "colsample_bytree": 0.8,
    "subsample": 0.6,
    "subsample_freq": 5,
    "n_estimators": 750,
}


keep_from_month = 2  # The first couple of months are dropped because of distortions to their features (e.g. wrong item age)
test_month = 33
dropcols = [
    "shop_id",
    "item_id",
    "new_item",
]  # The features are dropped to reduce overfitting



import pandas as pd

df = pd.read_pickle("/kaggle/input/kaggle-predict-future-sales-first-place-solution/abubakar_VVIP/abubakar_VVIP/checkpoint_final_0.84.pkl")  
#pkl is much faster and much lower in memory
df['item_cnt_month'] = df['item_cnt_month'].clip(0,20)
df=df.rename(columns={"item_cnt_month":"item_cnt"})
df=df[df!=np.inf]








import xgboost as xgb
vals_arr_lgb=[]
vals_arr_lgb_84=[]
preds_arr_xgb=[]
vals_arr_xgb=[]
preds_arr_nn=[]
shop_id=[]
item_id=[] 
cat_id=[]
month_arr=[]
actual_y=[]
for i in range(25,35):
    condition = df["date_block_num"]==i
    X_val=df[condition].drop(['item_cnt',"date_block_num"],axis=1)
    y_val =df[condition]["item_cnt"]
    shop_id.append(X_val["shop_id"].values)
    item_id.append(X_val["item_id"].values)
    cat_id.append(X_val["item_category_id"].values)
    month_arr.append(X_val["month"].values)
    actual_y.append(y_val.values)







def post_processing(submission,predict_value):
        path="/kaggle/input/kaggle-predict-future-sales-first-place-solution/abubakar_VVIP/abubakar_VVIP/data/"
        submission['item_cnt_month'] = predict_value.clip(0,20)
        # error_analysis_items_list=error_analysis_file[(error_analysis_file["release_day"].isin([29,30,31]))&\
        #                                               (error_analysis_file["unique_in_33"]==1)]["item_id"].unique()

        #categories in shop 55 
        cat_55_shop=[39,25,31,76]
        submission.loc[((submission["item_category_id"].isin(cat_55_shop))&\
                (submission["shop_id"]!=55)),"item_cnt_month"]=0



        ###############one direction with id 5320 and name ONE DIRECTION Made In The A.M.	 was definitely a hit
        submission.loc[((submission["item_id"]==5320)&(submission["shop_id"]!=55)),"item_cnt_month"]=\
        submission.loc[((submission["item_id"]==5320)&(submission["shop_id"]!=55)),"item_cnt_month"]*3


        submission.loc[((submission["item_id"]==18743)&(submission["shop_id"]!=55)),"item_cnt_month"]=\
        submission.loc[((submission["item_id"]==18743)&(submission["shop_id"]!=55)),"item_cnt_month"]*2




        #category id 34 always remain above 20 so
        submission.loc[((submission["item_category_id"]==34)&\
        (submission["shop_id"]==55)),"item_cnt_month"]=20

        submission.loc[((submission["item_category_id"]==34)&\
        (submission["shop_id"]!=55)),"item_cnt_month"]=0

        submission.loc[(submission["item_id"]==7223),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==7223),"item_cnt_month"]*1.3
        # #

        ##5268 need for speed might be a popular game
        submission.loc[(submission["item_id"]==5268),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==5268),"item_cnt_month"]*3



        #2323 is call of duty black hawks down is very popular and for PS4 it was a super hit as seen from google ratings and wikipedia 
        submission.loc[(submission["item_id"]==2323),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==2323),"item_cnt_month"]*3

        # 2327 s call of duty black hawks down is very popular and for PS4 it was a super hit as seen from google ratings and wikipedia
        submission.loc[(submission["item_id"]==2327),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==2327),"item_cnt_month"]*3


        #3408 Fallout 4 eleased in 2015 is very popular
        submission.loc[(submission["item_id"]==3408),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3408),"item_cnt_month"]*3

        # 3405 Fallout 4 eleased in 2015 is very popular
        submission.loc[(submission["item_id"]==3405),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3405),"item_cnt_month"]*3

        # 3407 Fallout 4 eleased in 2015 is very popular
        submission.loc[(submission["item_id"]==3407),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3407),"item_cnt_month"]*3

        #6729 and 6732,6731  are starwars battlefront variants and they all are extremely popular
        submission.loc[(submission["item_id"]==6729),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==6729),"item_cnt_month"]*1.5

        submission.loc[(submission["item_id"]==6731),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==6731),"item_cnt_month"]*1.5

        submission.loc[(submission["item_id"]==6732),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==6732),"item_cnt_month"]*1.5

        #7782 wasteland is somewhat popular but not so 
        submission.loc[(submission["item_id"]==7782),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==7782),"item_cnt_month"]*1.5
        #PC game football
        submission.loc[(submission["item_id"]==3538),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3538),"item_cnt_month"]*1.5
        #rise of tomb rider xbox1
        submission.loc[(submission["item_id"]==6153),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==6153),"item_cnt_month"]*1.5
        submission.loc[(submission["item_id"]==6152),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==6152),"item_cnt_month"]*1.5

        #rise of tomb rider xbox1
        submission.loc[(submission["item_id"]==2326),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==2326),"item_cnt_month"]*1.5

        #rise of tomb rider xbox1
        submission.loc[(submission["item_id"]==2328),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==2328),"item_cnt_month"]*2
        #rise of tomb rider xbox1
        submission.loc[(submission["item_id"]==1577),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==1577),"item_cnt_month"]*1.5
        # #rise of tomb rider xbox1
        submission.loc[(submission["item_id"]==5269),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==5269),"item_cnt_month"]*1.5
        # #rise of tomb rider xbox1
        submission.loc[(submission["item_id"]==4060),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==4060),"item_cnt_month"]*2








        # .comic high rating
        submission.loc[(submission["item_id"]==13310),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==13310),"item_cnt_month"]*3

        # submission.loc[(submission["item_id"]==13338),"item_cnt_month"]=\
        # submission.loc[(submission["item_id"]==13338),"item_cnt_month"]*1.5

        # submission.loc[(submission["item_id"]==13309),"item_cnt_month"]=\
        # submission.loc[(submission["item_id"]==13309),"item_cnt_month"]*1.5

        #3571  angels and ghost was quite popular
        submission.loc[(submission["item_id"]==3571),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3571),"item_cnt_month"]*2

        #3984 JARRE JEAN MICHEL  Electronika 1  The Time Machine released on october16 15 will also be sold in november
        submission.loc[(submission["item_id"]==3984),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3984),"item_cnt_month"]*2

        #Selena gomez revival
        submission.loc[(submission["item_id"]==3604),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3604),"item_cnt_month"]*1.5

        #botelli 
        submission.loc[(submission["item_id"]==1732),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==1732),"item_cnt_month"]*1.5

        #botelli 
        submission.loc[(submission["item_id"]==1246),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==1246),"item_cnt_month"]*1.5


        # 6335 STEWART ROD  Another Country
        submission.loc[(submission["item_id"]==6335),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==6335),"item_cnt_month"]*1.5

        # 10203 quite popular when starts
        submission.loc[(submission["item_id"]==10203),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==10203),"item_cnt_month"]*1.5

        # 6152 tom raider is a famous game
        submission.loc[(submission["item_id"]==6152),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==6152),"item_cnt_month"]*1.5

        # 6153 tom raider is a famous game
        submission.loc[(submission["item_id"]==6153),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==6153),"item_cnt_month"]*3

        # 6742 statecraft is a famous game
        submission.loc[(submission["item_id"]==6742),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==6742),"item_cnt_month"]*3



        #13745 BLACK WITCHER IS PLAYED AND IS FAMOUS
        submission.loc[(submission["item_id"]==13745),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==13745),"item_cnt_month"]*3
                

        # #5269 \nfs will be a a hot selll
        submission.loc[(submission["item_id"]==5269),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==5269),"item_cnt_month"]*3

        # # #ЛЕВША famous movie
        # submission.loc[(submission["item_id"]==13804),"item_cnt_month"]=\
        # submission.loc[(submission["item_id"]==13804),"item_cnt_month"]*2

        # #mission impossible  famous movie
        submission.loc[(submission["item_id"]==14647),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==14647),"item_cnt_month"]*2






        special_shops=[25,31,42]

        # #mission impossible  famous movie
        submission.loc[(submission["item_id"]==14648),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==14648),"item_cnt_month"]*3
        submission.loc[(submission["item_id"]==14648)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=20

        special_shops=[25,31,42]

        submission.loc[(submission["item_id"]==2427)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=8


        submission.loc[(submission["item_id"]==5268)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==5268)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

        submission.loc[(submission["item_id"]==5269)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==5269)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

        submission.loc[(submission["item_id"]==13745)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==13745)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

        submission.loc[(submission["item_id"]==3408)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3408)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

        submission.loc[(submission["item_id"]==3405)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3405)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

        submission.loc[(submission["item_id"]==3407)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3407)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

        submission.loc[(submission["item_id"]==2327)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==2327)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2

        submission.loc[(submission["item_id"]==6742)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==6742)&(submission["shop_id"].isin(special_shops)),"item_cnt_month"]*2



        special_shops_new=[25,31,42]
        submission.loc[(submission["item_id"]==10447)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==10447)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]*1.5

        submission.loc[(submission["item_id"]==19655)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==19655)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]*1.5

        submission.loc[(submission["item_id"]==19657)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==19657)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]*1.5

        submission.loc[(submission["item_id"]==10449)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==10449)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]*1.5

        submission.loc[(submission["item_id"]==10449)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==10449)&(submission["shop_id"].isin(special_shops_new)),"item_cnt_month"]*1.5
        submission.loc[(submission["item_id"].isin([5268]))&(submission["shop_id"]==39),"item_cnt_month"]=20
        submission.loc[(submission["item_id"].isin([204075]))&(submission["shop_id"]==39),"item_cnt_month"]=20
        submission.loc[(submission["item_id"].isin([204077]))&(submission["shop_id"]==39),"item_cnt_month"]=18
        submission.loc[(submission["item_id"].isin([6742]))&(submission["shop_id"]==39),"item_cnt_month"]=20
        submission.loc[(submission["item_id"].isin([2327]))&(submission["shop_id"]==39),"item_cnt_month"]=20
        submission.loc[(submission["item_id"].isin([2323]))&(submission["shop_id"]==39),"item_cnt_month"]=20

        submission.loc[(submission["item_id"].isin([13247]))&(submission["shop_id"]==39),"item_cnt_month"]=20
        submission.loc[(submission["item_id"].isin([13247]))&(submission["shop_id"]==25),"item_cnt_month"]=20

        submission.loc[(submission["item_id"].isin([14959])),"item_cnt_month"]=\
        submission.loc[(submission["item_id"].isin([14959])),"item_cnt_month"]*2
        submission.loc[(submission["item_id"].isin([13303])),"item_cnt_month"]=\
        submission.loc[(submission["item_id"].isin([13303])),"item_cnt_month"]*2



        submission.loc[(submission["item_id"].isin([21811]))&(submission["shop_id"]==39),"item_cnt_month"]=20
        submission.loc[(submission["item_id"].isin([17270]))&(submission["shop_id"]==39),"item_cnt_month"]=20
        submission.loc[(submission["item_id"].isin([6732]))&(submission["shop_id"]==39),"item_cnt_month"]=20

        submission.loc[(submission["item_id"].isin([2326]))&(submission["shop_id"]==39),"item_cnt_month"]=4
        submission.loc[(submission["item_id"].isin([13246]))&(submission["shop_id"]==39),"item_cnt_month"]=5
        submission.loc[(submission["item_id"].isin([1583]))&(submission["shop_id"]==39),"item_cnt_month"]=5


        #good rating 
        submission.loc[(submission["item_id"].isin([10449]))&(submission["shop_id"]==39),"item_cnt_month"]=16
        submission.loc[(submission["item_id"].isin([16629]))&(submission["shop_id"]==39),"item_cnt_month"]=16
        submission.loc[(submission["item_id"].isin([13805]))&(submission["shop_id"]==39),"item_cnt_month"]=16
        #
        shops=[31,12,42,25]
        submission.loc[(submission["item_id"]==20486),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==20486),"item_cnt_month"]*1.5
        submission.loc[(submission["item_id"]==20486)&(submission["shop_id"]==42),"item_cnt_month"]=10

        submission.loc[(submission["item_id"]==20401),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==20401),"item_cnt_month"]*1.5
        submission.loc[(submission["item_id"]==20401)&(submission["shop_id"]==42),"item_cnt_month"]=10

        submission.loc[(submission["item_id"]==20400),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==20400),"item_cnt_month"]*1.5
        submission.loc[(submission["item_id"]==20400)&(submission["shop_id"]==42),"item_cnt_month"]=10

        submission.loc[(submission["item_id"]==4156)&(submission["shop_id"].isin([22,24,56,57,25,31])),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==4156)&(submission["shop_id"].isin([22,24,56,57,25,31])),"item_cnt_month"]*3

        submission.loc[(submission["item_id"]==20949)&(~submission["shop_id"].isin([55,2])),"item_cnt_month"]=20


        submission.loc[(submission["item_id"]==11373)&(submission["shop_id"]==12),"item_cnt_month"]=20
        submission.loc[(submission["item_id"]==11373)&(submission["shop_id"]!=12),"item_cnt_month"]=0


        submission.loc[(submission["item_id"]==11370)&(submission["shop_id"]==12),"item_cnt_month"]=20
        submission.loc[(submission["item_id"]==11370)&(submission["shop_id"]!=12),"item_cnt_month"]=0

        submission.loc[(submission["item_id"]==11369)&(submission["shop_id"]==12),"item_cnt_month"]=20
        submission.loc[(submission["item_id"]==11369)&(submission["shop_id"]!=12),"item_cnt_month"]=0

        submission.loc[(submission["item_id"]==13342)&(submission["shop_id"]==55),"item_cnt_month"]=20
        submission.loc[(submission["item_id"]==13342)&(submission["shop_id"]!=55),"item_cnt_month"]=0

        submission.loc[(submission["item_id"]==492)&(submission["shop_id"]==55),"item_cnt_month"]=20
        submission.loc[(submission["item_id"]==17717)&(submission["shop_id"]==31),"item_cnt_month"]=20
        submission.loc[(submission["item_id"]==17717)&(submission["shop_id"]==42),"item_cnt_month"]=20



        submission.loc[(submission["item_id"]==7224)&(submission["shop_id"]==31),"item_cnt_month"]=5
        submission.loc[(submission["item_id"]==7224)&(submission["shop_id"]==42),"item_cnt_month"]=8
        submission.loc[(submission["item_id"]==7224)&(submission["shop_id"]==25),"item_cnt_month"]=8


        submission.loc[(submission["item_id"]==4894)&(submission["shop_id"].isin([25,31,42])),"item_cnt_month"]=2
        submission.loc[(submission["item_id"]==19657)&(submission["shop_id"].isin([57])),"item_cnt_month"]=3
        submission.loc[(submission["item_id"]==10449)&(submission["shop_id"].isin([57])),"item_cnt_month"]=3


        submission.loc[(submission["item_id"]==2431)&(submission["shop_id"].isin([26,25,42])),"item_cnt_month"]=12

        shops_new=[42,25,31]
        submission.loc[(submission["item_id"]==3838)&(submission["shop_id"].isin([shops_new])),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3838)&(submission["shop_id"].isin([shops_new])),"item_cnt_month"]*3
        submission.loc[(submission["item_id"]==3838)&(submission["shop_id"].isin([shops_new])),"item_cnt_month"]=\
        submission.loc[(submission["item_id"]==3838)&(submission["shop_id"].isin([shops_new])),"item_cnt_month"].clip(0,8)

        shops_new=[42,25,31,12,28]
        submission.loc[(submission["item_id"]==7728)&(submission["shop_id"].isin([shops_new])),"item_cnt_month"]=8
        important_shop_item_id_pairs=pd.read_csv(path+"important_shop_item_id_files.csv")
        important_shop_item_id_pairs=important_shop_item_id_pairs.rename(columns={"item_cnt":"item_cnt_month_mapped"})
        submission=pd.merge(submission,important_shop_item_id_pairs,on=["shop_id","item_id"],how="left")
        submission.loc[submission["item_cnt_month_mapped"].notnull(),"item_cnt_month"]=\
        submission.loc[submission["item_cnt_month_mapped"].notnull(),"item_cnt_month_mapped"]
        submission['item_cnt_month']=submission['item_cnt_month'].clip(0,20)
        return submission


import numpy as np
import pandas as pd

path2 = "/kaggle/input/kaggle-predict-future-sales-first-place-solution/"

# Tải các mảng NumPy
vals_arr_xgb_np = np.load(path2 + 'vals_arr_xgb_84.npy', allow_pickle=True)

# vals_arr_lgb_np = np.load(path2 + 'vals_arr_lgb_84.npy', allow_pickle=True)

# vals_arr_lgb_np_25_30= np.load('/kaggle/input/vals-arr-lgbm-25-30/vals_arr_lgbm-25-30.pkl', allow_pickle=True)
# vals_arr_lgb_np_30_35 = np.load('/kaggle/input/vals-arr-lgbm-30-35/vals_arr_lgbm-30-35.pkl', allow_pickle=True)
# vals_arr_lgb_np = np.concatenate((vals_arr_lgb_np_25_30,vals_arr_lgb_np_30_35), axis=0)
# vals_arr_lgb_np = [np.array(x[0], dtype=np.float32) for x in vals_arr_lgb_np]


vals_arr_lgb_np = np.load('/kaggle/input/vals-arr-cast/vals_arr_cast.pkl', allow_pickle=True)
# for x in vals_arr_lgb_np:
#     print(x[0])
# vals_arr_lgb_np = [np.array(x[0], dtype=np.float32) for x in vals_arr_lgb_np]

vals_arr_lgbs_np = np.load(path2 + 'vals_arr_lgb_84s.npy', allow_pickle=True) # Sửa tên file nếu cần


column_names = ['item_id', 'shop_id', 'item_category_id', 'month', 'xgb_val', 'lgb_val', 'lgb_new_val']
X_merged_df_list = [
    pd.DataFrame(
        np.column_stack((
            item_id[i],
            shop_id[i],
            cat_id[i],
            month_arr[i],
            vals_arr_xgb_np[i],
            vals_arr_lgb_np[i],
            vals_arr_lgbs_np[i]
            
        )),
        columns=column_names
    )
    for i in range(len(vals_arr_xgb_np))
]

y_merged_df_list = [
    pd.DataFrame(
        np.column_stack((
            actual_y[i],
        )),
        columns=["y_real"]
    )
    for i in range(len(actual_y))
]


def remove_and_process(X_merged):
    tmp  = X_merged
    x1 = post_processing(tmp,tmp["xgb_val"])
    x1['xgb_val']=x1['item_cnt_month']
    tmp = x1.drop(['item_cnt_month_mapped','item_cnt_month'],axis=1)

    x2 = post_processing(tmp,tmp["lgb_val"])
    x2['lgb_val']=x2['item_cnt_month']
    tmp = x2.drop(['item_cnt_month_mapped','item_cnt_month'],axis=1)

    x3 = post_processing(tmp,tmp["lgb_new_val"])
    x3['lgb_new_val']=x3['item_cnt_month']
    tmp = x3.drop(['item_cnt_month_mapped','item_cnt_month'],axis=1)
    return tmp





for i in range (len(X_merged_df_list)):
    X_merged_df_list[i] = remove_and_process(X_merged_df_list[i])





X_train_df = pd.concat(X_merged_df_list[:-1], ignore_index=True)
y_train_df = pd.concat(y_merged_df_list[:-1], ignore_index=True)
X_test_df = pd.concat(X_merged_df_list[-1:], ignore_index=True)


X_train_df = X_train_df.drop(["month"],axis=1)
X_test_df =  X_test_df.drop(["month"],axis=1)


# n_splits=5
# SEED=42
# from sklearn.base import clone
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import StratifiedKFold
# from IPython.display import clear_output
# from scipy.optimize import minimize
# import numpy as np
# from colorama import Fore, Style
# from tqdm import tqdm
# def root_mean_squared_error(y, y_val):
#     mse = mean_squared_error(y, y_val)
#     return np.sqrt(mse)

# def TrainML(model_class,X,y, test_data):
#     SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
#     train_S = []
#     test_S = []
    
#     test_preds = np.zeros((len(test_data), n_splits))
#     oof_non_rounded = np.zeros(len(y), dtype=float) 
#     for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):
#         X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[test_idx] 
#         model = clone(model_class)
#         model.fit(X_train, y_train)
#         y_train_pred = model.predict(X_train)
#         y_val_pred = model.predict(X_val)
#         test_pred = model.predict(test_data)

#         oof_non_rounded[test_idx] = y_val_pred
#         train_rmse = root_mean_squared_error(y_train, y_train_pred)
#         val_rmse = root_mean_squared_error(y_val, y_val_pred)

#         train_S.append(train_rmse)
#         test_S.append(val_rmse)
        
#         test_preds[:, fold] = test_pred
        
#         print(f"Fold {fold+1} - Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")
#         clear_output(wait=True)

#     print(f"Mean Train  --> {np.mean(train_S):.4f}")
#     print(f"Mean Validation RMSE ---> {np.mean(test_S):.4f}")
    
#     tRMSE = root_mean_squared_error(y, oof_non_rounded)

#     print(f"----> || Optimized RMSE SCORE :: {Fore.CYAN}{Style.BRIGHT} {tRMSE:.3f}{Style.RESET_ALL}")
#     tpm = test_preds.mean(axis=1)
#     return tpm


# n_splits = 5
# SEED = 42
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torch.nn.parallel import DataParallel
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from IPython.display import clear_output
# from colorama import Fore, Style
# from tqdm import tqdm

# # Kiểm tra và sử dụng tất cả các GPU có sẵn
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     num_gpus = torch.cuda.device_count()
#     print(f"Đang sử dụng {num_gpus} GPU")
#     for i in range(num_gpus):
#         total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
#         print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Bộ nhớ: {total_memory:.2f} GB")
# else:
#     device = torch.device("cpu")
#     print("Không tìm thấy GPU, đang sử dụng CPU")

# def root_mean_squared_error(y, y_pred):
#     mse = mean_squared_error(y, y_pred)
#     return np.sqrt(mse)

# # Define PyTorch MLP model - Sử dụng mạng lớn hơn để tận dụng bộ nhớ GPU nhiều hơn
# class MLPRegressorTorch(nn.Module):
#     def __init__(self, input_dim, hidden_dims=(512, 256, 128)):
#         super(MLPRegressorTorch, self).__init__()
        
#         layers = []
#         prev_dim = input_dim
        
#         # Create hidden layers
#         for dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.BatchNorm1d(dim))  # Thêm BatchNorm để tăng tốc độ hội tụ và ổn định
#             layers.append(nn.Dropout(0.3))  # Tăng dropout để tránh overfitting
#             prev_dim = dim
        
#         # Output layer
#         layers.append(nn.Linear(prev_dim, 1))
        
#         self.model = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.model(x).squeeze()
# import multiprocessing as mp
# def TrainMLPWithMultiGPU(X, y, test_data, hidden_dims=(100,), batch_size=8192, 
#                          epochs=10, learning_rate=0.001, weight_decay=1e-5):
#     """
#     Train an MLP Regressor với nhiều GPU và k-fold cross-validation
#     Tối ưu hóa cho GPU bộ nhớ cao (15GB)
    
#     Parameters:
#     -----------
#     X : DataFrame hoặc array
#         Features huấn luyện
#     y : Series hoặc array
#         Biến mục tiêu
#     test_data : DataFrame hoặc array
#         Features dữ liệu test
#     hidden_dims : tuple, default=(512, 256, 128)
#         Kiến trúc các hidden layer - đã điều chỉnh lớn hơn để tận dụng bộ nhớ GPU
#     batch_size : int, default=8192
#         Kích thước mini-batch cho huấn luyện - đã tăng lên nhiều để tận dụng GPU
#     epochs : int, default=100
#         Số lượng epochs huấn luyện
#     learning_rate : float, default=0.001
#         Tốc độ học cho Adam optimizer
#     weight_decay : float, default=1e-5
#         Tham số L2 regularization
        
#     Returns:
#     --------
#     numpy.ndarray
#         Dự đoán trung bình cho dữ liệu test
#     """
#     # Chuyển đổi pandas sang numpy nếu cần
#     if hasattr(X, 'values'):
#         X = X.values
#     if hasattr(y, 'values'):
#         y = y.values
#     if hasattr(test_data, 'values'):
#         test_data = test_data.values
    
#     # Đảm bảo y là 1D
#     y = y.ravel() if hasattr(y, 'ravel') else y.flatten()
    
#     # Khởi tạo scaler cho features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     test_data_scaled = scaler.transform(test_data)
    
#     # Chuyển đổi sang PyTorch tensors với precision float16 để tiết kiệm bộ nhớ
#     X_tensor = torch.FloatTensor(X_scaled)
#     y_tensor = torch.FloatTensor(y)
#     test_tensor = torch.FloatTensor(test_data_scaled)
    
#     # KFold cho bài toán regression
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
#     train_scores = []
#     val_scores = []
    
#     test_preds = np.zeros((len(test_data), n_splits))
#     oof_preds = np.zeros(len(y), dtype=float)
    
#     for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), desc="Huấn luyện MLP Folds", total=n_splits)):
#         # Lấy dữ liệu fold
#         X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
        
#         # Tạo DataLoader cho batching hiệu quả với batch_size lớn hơn để tận dụng GPU
#         train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
#                                   pin_memory=True, num_workers=4,multiprocessing_context=mp.get_context("forkserver"))  # Tăng num_workers
        
#         # Tạo model
#         input_dim = X.shape[1]
#         model = MLPRegressorTorch(input_dim, hidden_dims).to(device)
        
#         # Sử dụng DataParallel để tận dụng nhiều GPU
#         if torch.cuda.device_count() > 1:
#             model = DataParallel(model)
#             print(f"Đang sử dụng {torch.cuda.device_count()} GPU cho fold {fold+1}")
        
#         # Sử dụng mixed precision để huấn luyện nhanh hơn
#         scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Loss function và optimizer
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
#         # Training loop
#         best_loss = float('inf')
#         patience_counter = 0
        
#         model.train()
#         for epoch in range(epochs):
#             epoch_loss = 0
            
#             for X_batch, y_batch in train_loader:
#                 # Đưa dữ liệu lên GPU
#                 X_batch = X_batch.to(device, non_blocking=True)
#                 y_batch = y_batch.to(device, non_blocking=True)
                
#                 # Sử dụng mixed precision
#                 with torch.amp.autocast('cuda'):
#                     # Forward pass
#                     optimizer.zero_grad()
#                     y_pred = model(X_batch)
#                     loss = criterion(y_pred, y_batch)
                
#                 # Backward pass với mixed precision
#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()
                
#                 epoch_loss += loss.item() * X_batch.size(0)
            
#             avg_epoch_loss = epoch_loss / len(train_dataset)
#             scheduler.step()
            
#             # Early stopping
#             if avg_epoch_loss < best_loss:
#                 best_loss = avg_epoch_loss
#                 patience_counter = 0
#                 # Lưu model tốt nhất
#                 best_model_state = model.state_dict().copy()
#             else:
#                 patience_counter += 1
#                 if patience_counter >= 15:  # Tăng patience lên để có thể huấn luyện lâu hơn
#                     print(f"Early stopping tại epoch {epoch+1}")
#                     break
            
#             # Hiển thị tiến trình huấn luyện
#             if (epoch + 1) % 10 == 0 or epoch == 0:
#                 print(f"Fold {fold+1}, Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
#                 # clear_output(wait=True)
        
#         # Tải lại model tốt nhất
#         model.load_state_dict(best_model_state)
        
#         # Đánh giá với batch lớn hơn
#         model.eval()
#         with torch.no_grad():
#             # Xử lý dữ liệu validation trên GPU
#             val_batch_size = batch_size * 2  # Tăng batch size khi đánh giá
            
#             # Đánh giá validation data
#             val_dataset = TensorDataset(torch.FloatTensor(X_val))
#             val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, 
#                                    pin_memory=True, num_workers=4,multiprocessing_context=mp.get_context("forkserver"))
            
#             y_val_pred = []
#             for (X_batch,) in val_loader:
#                 X_batch = X_batch.to(device, non_blocking=True)
#                 with torch.amp.autocast('cuda'):
#                     batch_preds = model(X_batch).cpu().numpy()
#                 y_val_pred.extend(batch_preds)
#             y_val_pred = np.array(y_val_pred)
            
#             # Đánh giá training data
#             train_dataset = TensorDataset(torch.FloatTensor(X_train))
#             train_loader = DataLoader(train_dataset, batch_size=val_batch_size, 
#                                      shuffle=False, pin_memory=True, num_workers=4,multiprocessing_context=mp.get_context("forkserver"))
            
#             y_train_pred = []
#             for (X_batch,) in train_loader:
#                 X_batch = X_batch.to(device, non_blocking=True)
#                 with torch.amp.autocast('cuda'):
#                     batch_preds = model(X_batch).cpu().numpy()
#                 y_train_pred.extend(batch_preds)
#             y_train_pred = np.array(y_train_pred)
            
#             # Dự đoán trên dữ liệu test
#             test_dataset = TensorDataset(test_tensor)
#             test_loader = DataLoader(test_dataset, batch_size=val_batch_size, 
#                                     shuffle=False, pin_memory=True, num_workers=4,multiprocessing_context=mp.get_context("forkserver"))
            
#             test_fold_preds = []
#             for (X_batch,) in test_loader:
#                 X_batch = X_batch.to(device, non_blocking=True)
#                 with torch.amp.autocast('cuda'):
#                     batch_preds = model(X_batch).cpu().numpy()
#                 test_fold_preds.extend(batch_preds)
#             test_fold_preds = np.array(test_fold_preds)
        
#         # Tính toán metrics
#         train_rmse = root_mean_squared_error(y_train, y_train_pred)
#         val_rmse = root_mean_squared_error(y_val, y_val_pred)
        
#         # Lưu dự đoán
#         oof_preds[val_idx] = y_val_pred
#         test_preds[:, fold] = test_fold_preds
        
#         train_scores.append(train_rmse)
#         val_scores.append(val_rmse)
        
#         print(f"Fold {fold+1} - Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")
        
#         # Giải phóng bộ nhớ GPU
#         del model
#         torch.cuda.empty_cache()
    
#     # In kết quả cuối cùng
#     print(f"Mean Train RMSE  --> {np.mean(train_scores):.4f}")
#     print(f"Mean Validation RMSE ---> {np.mean(val_scores):.4f}")
    
#     # Tính toán RMSE tổng thể trên out-of-fold predictions
#     total_rmse = root_mean_squared_error(y, oof_preds)
#     print(f"----> || Optimized RMSE SCORE :: {Fore.CYAN}{Style.BRIGHT} {total_rmse:.3f}{Style.RESET_ALL}")
    
#     # Trả về dự đoán trung bình cho dữ liệu test
#     test_preds_mean = test_preds.mean(axis=1)
#     return test_preds_mean




X_train = X_train_df
X_test = X_test_df
y_train = y_train_df


from sklearn.linear_model import LinearRegression, Ridge # Thêm Ridge
# Hoặc import cả 3: from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Tạm thời comment lại hoặc xóa dòng LinearRegression cũ
# reg = LinearRegression().fit(X_train, y_train)
# pred_stack_reg=reg.predict(X_test)





import os
import pandas as pd
from sklearn.linear_model import ElasticNet # Chỉ cần import ElasticNet
alpha_value = 0.05
l1_ratio_value = 0.85

output_dir = "/kaggle/working/" 
os.makedirs(output_dir, exist_ok=True)
path_data_input = "/kaggle/input/kaggle-predict-future-sales-first-place-solution/abubakar_VVIP/abubakar_VVIP/data/" # Đường dẫn file input

try:
    sample_submission_orig = pd.read_csv(path_data_input + 'sample_submission.csv')
except FileNotFoundError:
     print(f"LỖI: Không tìm thấy file sample_submission.csv tại {path_data_input}")
  
     raise

if 'X_train' not in locals() or 'y_train' not in locals() or 'X_test' not in locals():
    print("LỖI: Biến X_train, y_train, hoặc X_test chưa được định nghĩa.")
    print("Vui lòng chạy các cell chuẩn bị dữ liệu theo logic gốc của Notebook 3 trước.")

    raise NameError("X_train, y_train, hoặc X_test không tồn tại.")
elif X_train.shape[1] != 3:
     print(f"CẢNH BÁO: X_train có {X_train.shape[1]} cột, mong đợi 3 cột (dự đoán cơ sở đã xử lý).")
     print("Kết quả có thể không chính xác nếu đầu vào không đúng logic gốc.")

model_name = "ElasticNet"
model_class = ElasticNet


params_key = f"alpha{alpha_value:.3f}_l1r{l1_ratio_value:.2f}"
print(f"  Huấn luyện {model_name} với {params_key}...")
try:

    model = model_class(alpha=alpha_value, l1_ratio=l1_ratio_value,
                        max_iter=2000, tol=1e-4, random_state=42)
    model.fit(X_train, y_train) # Huấn luyện trên X_train đã chuẩn bị
    pred_stack_reg = model.predict(X_test) # Dự đoán trên X_test đã chuẩn bị


    submission_df = sample_submission_orig.copy() # Tạo bản sao từ sample gốc
    submission_df['item_cnt_month'] = pred_stack_reg.clip(0, 20)

    # Lưu file submission
    file_name = f"submission_{model_name}_{params_key}_preprocessed_3feats.csv" # Tên file rõ ràng
    file_path = os.path.join(output_dir, 'submission.csv')
    submission_df[['ID', 'item_cnt_month']].to_csv(file_path, index=False)
    print(f"    Đã lưu: {file_path}")

except Exception as e:
    print(f"    Lỗi khi huấn luyện/lưu {model_name} {params_key}: {e}")
    import traceback
    traceback.print_exc() # In chi tiết lỗi

print(f"--- Hoàn thành tinh chỉnh {model_name} ---")

