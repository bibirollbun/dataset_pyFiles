import numpy as np # linear algebra
import polars as pl # data processing, CSV file I/O 
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
# Input data files are available in the read-only "../input/" directory
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



raw_data = pl.scan_csv ('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv').collect()  
raw_inventory = pl.scan_csv ('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv').collect()  
raw_calendar = pl.scan_csv ('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv').collect()  
weight_test = pl.scan_csv ('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv').collect()  
sales_test = pl.scan_csv ('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv').collect()  

#  source https://kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2


print (raw_data.head())
print (raw_data.shape)
print (raw_data.columns)


raw_data = raw_data.with_columns (pl.col ("date").str.to_date().alias("p-date") )

last_day_train = raw_data.get_column ("p-date").max() 
first_day_train = raw_data.get_column ("p-date").min() 
print (f"{last_day_train = }, {first_day_train = }")
print (f"days in  in training: {(last_day_train - first_day_train) } + 1 ")

sales_test = sales_test.with_columns (pl.col ("date").str.to_date().alias("p-date") )

last_day_submit = sales_test.get_column ("p-date").max() 
first_day_submit =  sales_test.get_column ("p-date").min() 
print (f"{last_day_submit = }, {first_day_submit = }")
print (f"days in  in sales: {(last_day_submit - first_day_submit)} + 1 ")




new_table = raw_data.group_by(["unique_id", "warehouse"]).len()

unique_id_to_warehouse = dict (zip(new_table.get_column("unique_id"), new_table.get_column("warehouse")))
print (  new_table.get_column("unique_id").len())

unique_id_to_warehouse 

raw_data.filter (pl.col("unique_id")==2).sort("date")


train_id = raw_data.get_column("unique_id").unique().to_list()
sales_id = sales_test.get_column("unique_id").unique().to_list()

print (f"unique id in trian = {len(train_id)} unique id in sales = {len(sales_id)}")




# we need a full set for the time series that requires prediction, the others we just keep. 

all_Time_series_train = raw_data.select(pl.col("date")).unique().join(sales_test.select(pl.col("unique_id")).unique(), how = "cross")

all_Time_series_train = all_Time_series_train.join (raw_data, on =["date", "unique_id"], how = "full")
# we need a full set for the time series that requires prediction, the others we just keep. 

all_Time_series_train = all_Time_series_train.with_columns (pl.when (pl.col("date").is_null()).then(
                                                                 pl.col("date_right")).otherwise(
                                                                 pl.col("date")).alias ("date"),
                                                            pl.when (pl.col("unique_id").is_null()).then(
                                                                 pl.col("unique_id_right")).otherwise(
                                                                 pl.col("unique_id")).alias ("unique_id"), 
                                                            pl.when (pl.col("warehouse").is_null()).then (
                                                                 pl.col("unique_id").replace_strict(unique_id_to_warehouse)).otherwise (
                                                                 pl.col("warehouse")).alias("warehouse")  
                                                                 )


all_Time_series_train = all_Time_series_train.drop(["date_right", "unique_id_right"])
display (all_Time_series_train.sort (["unique_id", "date"]))


raw_data.group_by ("unique_id").agg(pl.col("warehouse").unique().len()).sort ("warehouse")



time_series = raw_data.group_by("unique_id").len().sort("len")

with pl.Config (tbl_rows = 2): 
    print (time_series.tail(2))

sns.histplot (time_series.get_column("len").to_numpy() )

most_wanted_list = time_series.tail(2).get_column("unique_id").to_list()

print (most_wanted_list)

# for experimenting : raw_data = raw_data.filter (pl.col("unique_id").is_in (most_wanted_list)) 


display (raw_inventory.head())

raw_inventory.filter (pl.col('unique_id') == 5416)


with pl.Config(tbl_rows = 50) :
    display (raw_inventory.group_by(["product_unique_id","name"]).len().sort("len"))


sales_test.head()


raw_calendar.head()


print (weight_test)
print (weight_test.get_column ("weight").describe())


print (f'training dates from {raw_data.get_column ("date").min()} to {raw_data.get_column ("date").max()}' )
print (f'forcasting dates from {sales_test.get_column ("date").min()} to {sales_test.get_column ("date").max()}' )



import holidays

years_range = [ n for n in range (2020, 2024)]
germany_holidays = holidays.country_holidays("DE", years = years_range)  
hungary_holidays = holidays.country_holidays("HU", years = years_range)  
czech_holidays = holidays.country_holidays("CZ", years = years_range)  

def add_holiday (df : pl.DataFrame) -> pl.DataFrame :
    
    result = df.with_columns(pl.col ("holiday").cast(pl.Boolean))
    result = result.with_columns(pl.when (pl.col("country") == "Germany").then (
                      pl.col('p-date').is_in (germany_holidays.keys()) | pl.col("holiday")).otherwise (
                      pl.col("holiday")).alias ('holiday'))
    result = result.with_columns(pl.when (pl.col("country") == "Hungary").then (
                      pl.col('p-date').is_in (hungary_holidays.keys()) | pl.col("holiday")).otherwise (
                      pl.col("holiday")).alias ('holiday'))
    result = result.with_columns(pl.when (pl.col("country") == "Czech Republic").then (
                      pl.col('p-date').is_in (czech_holidays.keys()) | pl.col("holiday")).otherwise (
                      pl.col("holiday")).alias ('holiday'))
    return result


def merge_tables (raw_data, raw_inventory, raw_calendar : pl.DataFrame ) -> pl.DataFrame :
    
    print (f"data before 1st join = {raw_data.shape}")
    result = raw_data.join(raw_inventory, how = "left", on ="unique_id", suffix = "_inventory")
    print (f"train_clean shape after 1st join = {result.shape}")
    print (f"calendar shape  = {raw_calendar.shape}")
    result = result.join (raw_calendar, how = "left", on = ["date", "warehouse"], suffix = "_date")
    print (f"train_clean shape after 2nd join = {result.shape}")
    return result


from datetime import date
  

num_to_days = {1 : "Mon", 2 : "Tue", 3 : "Wed", 4 : "Thu", 5  :"Fri", 6 : "Sat", 7 : "Sun"}

weekdays_enum = pl.Enum(num_to_days.values())

warehouse_to_country = {     "Budapest_1" : "Hungary" ,
                            "Prague_2"   : "Czech Republic",
                            "Brno_1" : "Czech Republic",
                            "Prague_3" : "Czech Republic", 
                            "Frankfurt_1" : "Germany",
                            "Munich_1"    : "Germany",
                            "Prague_1" : "Czech Republic"
                        
}



def add_features (df : pl.DataFrame) -> pl.DataFrame :
    result = df
    if "unique_id" in df.columns :
        result = result.with_columns (("id_" + pl.col ("unique_id").cast (pl.String)).alias ("unique_id"),
                             ("prod_id_" + pl.col ("product_unique_id").cast (pl.String)).alias ("product_unique_id"),
                              pl.col("name").str.split ("_").list.first().alias ("generic product name"), 
                              pl.max_horizontal ("type_0_discount", "type_1_discount", "type_2_discount",
                                                                   "type_3_discount", "type_4_discount", 
                                                                   "type_5_discount",  "type_6_discount" ).alias ("max_discount"))
        result = result.drop(["type_0_discount","type_1_discount", "type_2_discount",
                              "type_3_discount", "type_4_discount","type_5_discount",  "type_6_discount"])
        
    result = result.with_columns  (pl.col ("shops_closed").cast(pl.Boolean),
                                   pl.col("winter_school_holidays").cast(pl.Boolean),
                                   pl.col("school_holidays").cast(pl.Boolean),
                                   pl.col ("date").str.to_date().alias("p-date"), 
                                      
                                   pl.col("warehouse").replace (warehouse_to_country).alias ("country") )
    
    result = result.with_columns ( (pl.col ('p-date') - first_day_train).dt.total_days().cast(pl.UInt16).alias ("days from start"),
                                   pl.col ('p-date').dt.weekday ().replace_strict (
                                   num_to_days, return_dtype = pl.String).alias ('weekday'), 
                                   pl.col ('p-date').dt.quarter ().cast (pl.UInt8).alias ('quarter'), 
                                   (pl.col ('p-date').dt.year () - 2020).cast (pl.UInt8).alias ('year'), 
                                   pl.col ('p-date').dt.ordinal_day ().cast (pl.UInt16).alias ('day_of_year'), 
                                   pl.col("date").str.tail(5).alias ("month_day"))
    
                              
    result = result.with_columns  (pl.col("weekday").cast(weekdays_enum).alias ('weekday'))
    
    
    result = result.with_columns ((pl.col("day_of_year")/365 *2  * np.pi).sin ().alias ("year_sin"), 
                                  (pl.col("day_of_year")/365 *2 * np.pi).cos ().alias ("year_cos"), 
                                  (pl.col("day_of_year")/365 *4  * np.pi).sin ().alias ("year_sin2x"), 
                                  (pl.col("day_of_year")/365 *4 * np.pi).cos ().alias ("year_cos2x"),
                                 )
    
    result = result.with_columns(pl.col ('weekday').is_in ([  'Sat', 'Sun']).alias ('is_weekend'))
    result = add_holiday (result)     
    
    return result 



train_clean = merge_tables (all_Time_series_train, raw_inventory, raw_calendar)




train_clean = add_features (train_clean)

display (train_clean.select (cs.numeric()).describe())


train_clean.schema


order_sales = train_clean.group_by(["year","quarter", "generic product name"]).agg(pl.col("sales").mean(), 
                                                           pl.col("total_orders").mean())
order_sales = order_sales.with_columns ((pl.col("year").cast(pl.String) + "_" + pl.col("quarter").cast(pl.String)).alias ("date"))


for col in order_sales.get_column("generic product name").unique() : 
    print (f"Display sales for {col =}")
    display_me = order_sales.filter(pl.col("generic product name") == col).sort("date")
    plt.figure(figsize=(22, 8))
    sns.lineplot(data = display_me, x ="date", y = "sales")
    plt.show()             





display (train_clean.schema)


%%time 
!pip install ray==2.10.0
!pip install autogluon.timeseries --no-cache-dir -q
!pip install -U ipywidgets


from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# from autogluon.tabular import TabularPredictor

# from autogluon.common import space

print ('done')


import pandas as pd 

import gc

gc.collect ()

def create_static_features (df : pl.DataFrame) -> pd.DataFrame :
    time_series1 = df.group_by ("unique_id").agg(
                pl.col("L1_category_name_en").max(),
                pl.col("L2_category_name_en").max(),
                pl.col("L3_category_name_en").max(),
                pl.col("L4_category_name_en").max(),
                pl.col("warehouse").max(),
                pl.col("product_unique_id").max(),        
                pl.col("name").max(),
                pl.col("generic product name").max(),
                pl.col("country").max(),
    )
    time_series2 = df.group_by ("unique_id").agg(
                pl.col("L1_category_name_en").min(),
                pl.col("L2_category_name_en").min(),
                pl.col("L3_category_name_en").min(),
                pl.col("L4_category_name_en").min(),
                pl.col("warehouse").min(),
                pl.col("product_unique_id").min(),        
                pl.col("name").min(),
                pl.col("generic product name").min(),
                pl.col("country").min(),
    )
    
    result = time_series1.to_pandas().set_index('unique_id')
    return result 

static_features = create_static_features(train_clean)
train_clean = train_clean.drop(static_features) 

train_data = TimeSeriesDataFrame.from_data_frame(
            train_clean.to_pandas(),
            id_column="unique_id",
            timestamp_column="date")


train_data.static_features = static_features 


known_features_train = train_clean.rename (({"unique_id" : "item_id", 
                                                             "date" : "timestamp"}))

known_features = ['total_orders','sell_price_main', "weekday", "quarter", "year", "day_of_year", "year_sin", "year_cos",  
                  "year_sin2x", "year_cos2x",  'holiday_name',   'holiday', 'shops_closed', 'max_discount', 'days from start',
        'winter_school_holidays', 'school_holidays', "month_day", "is_weekend"]

       


predictor = TimeSeriesPredictor(path = f'/kaggle/working/autogluon_time_series',
                            label='sales', 
                            freq="D",  
                            eval_metric =  'WAPE', 
                            known_covariates_names= known_features,
                            prediction_length = 14)   # matches the time window from the test data 



%%time


predictor.fit(train_data= train_data, 
                    presets= 'high_quality',
                    hyperparameters={
                       "Chronos": [# Zero-shot model WITHOUT covariates
                                #{
                                #"model_path": "bolt_tiny",
                                #"ag_args": {"name_suffix": "ZeroShot"},
                                #},
                                # Chronos-Bolt (Small) combined with CatBoost on covariates
                                {
                "model_path": "bolt_base",
                "covariate_regressor": "CAT",
                "target_scaler": "standard",
                "ag_args": {"name_suffix": "WithRegressor"},
            },
        ],
    },
    enable_ensemble=False,
    time_limit=7200,
             )              


predictor.leaderboard()


forecasting_dates = sales_test.select (pl.col("date")).unique()

all_Time_series_submit = forecasting_dates.join(raw_data.select(pl.col("unique_id")).unique(), how = "cross")

all_Time_series_submit = all_Time_series_submit.join (sales_test, on =["date", "unique_id"], how = "left") 

all_Time_series_submit = all_Time_series_submit.with_columns (
    pl.when (pl.col("warehouse").is_null()).then (
        pl.col("unique_id").replace_strict(unique_id_to_warehouse)).otherwise (
        pl.col("warehouse")).alias("warehouse"))  

display (all_Time_series_submit.filter (pl.col("unique_id") == 1000).head(3))


sales_clean =  merge_tables (all_Time_series_submit, raw_inventory, raw_calendar)

known_features_for_test = add_features (sales_clean)

known_features_for_test = known_features_for_test.fill_null(0)
print (known_features_for_test.select(pl.col("date").unique().sort()))
known_features_for_test_to_pandas = known_features_for_test.to_pandas()

# known_features_for_test_to_pandas.set_index('unique_id', inplace=True)

known_features_for_test_ts = TimeSeriesDataFrame.from_data_frame(
            known_features_for_test_to_pandas,
            id_column="unique_id",
            timestamp_column="date")
print (known_features_for_test_ts.columns)

not_found = [ f for f in known_features if f not in known_features_for_test_ts.columns]
print (f"{not_found = }")
print (known_features_for_test_ts.head(3))

print (known_features_for_test.group_by (by = "unique_id").len().sort("len"))
print ("done")


sales_id_cat = ["id_" + str(id) for id in sales_id]

test_clean = train_clean.filter (pl.col("unique_id").is_in(sales_id_cat)) 

test_data = TimeSeriesDataFrame.from_data_frame(
            test_clean.to_pandas(),
            id_column="unique_id",
            timestamp_column="date")


test_data.static_features = static_features 


%%time


predictions = predictor.predict(test_data, known_covariates = known_features_for_test_ts)


predictor.plot(train_data, predictions, item_ids = ["id_1001","id_5429", "id_4287"])


predictions_clean = pl.DataFrame (predictions.reset_index()) 
predictions_clean =  predictions_clean.with_columns (pl.col("timestamp").cast (pl.String).str.head(10).alias ("date"), 
                                                    pl.col("item_id").str.strip_prefix ("id_").cast (pl.Int64). alias ("unique_id"))

predictions_clean =  predictions_clean.drop (["timestamp",  "0.1","0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", ])



sales_clean_joint = sales_test.join (predictions_clean, on =["date","unique_id"], how = "left")

sales_clean_joint


sales_clean_joint = sales_clean_joint.with_columns((pl.col("unique_id").cast(pl.String) + "_" + pl.col("date")).alias("id"),
                                           pl.max_horizontal (0, pl.col("mean")).alias ("sales_hat"))

submission =  sales_clean_joint.select (["id", "sales_hat"])
submission.write_csv("submission.csv")


submission

