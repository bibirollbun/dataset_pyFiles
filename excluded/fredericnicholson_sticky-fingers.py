import numpy as np # linear algebra
import polars  as pl # data processing, CSV file I/O (e.g. pl.scan_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings

warnings.simplefilter("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



kaggle_path = '/kaggle/input/playground-series-s5e1'

train_df = pl.scan_csv(f'{kaggle_path}/train.csv').collect()
test_df = pl.scan_csv(f'{kaggle_path}/test.csv').collect()
sample_df = pl.scan_csv(f'{kaggle_path}/sample_submission.csv').collect()
gdp_per_capita = pl.scan_csv('/kaggle/input/gdp-per-capita/gdp-per-capita-maddison.csv').collect()

display (train_df.collect_schema())

display (train_df.head(5))

display (gdp_per_capita.head(5))


create_EDA = True


countries = train_df.get_column ('country').unique().to_list()
stores = train_df.get_column ('store').unique().to_list()
products = train_df.get_column ('product').unique().to_list()
print (f" {countries = }")
print (f" {stores = }")
print (f"  {products = }")


for country in countries : 
   print (f"different values for num_sol in {country}")
   with pl.Config(tbl_rows=50):
      display (train_df.filter (pl.col("country") == country).group_by (['product', 'store']).agg(pl.col('num_sold').n_unique()))


dates_train = train_df.with_columns (pl.col("date").str.tail(5).alias ("month_day"))

top_date_group = dates_train.top_k(5000, by = "num_sold").group_by ("month_day").len().sort("len").tail (10)
top_date = top_date_group.get_column ("month_day").to_list()
with pl.Config(tbl_rows=30):
    display (top_date_group)
    
bottom_date_group = dates_train.bottom_k(5000, by = "num_sold").group_by ("month_day").len().sort("len").tail (10)
bottom_date = bottom_date_group.get_column  ("month_day").to_list()
with pl.Config(tbl_rows=30):
    display (bottom_date_group)
    


with pl.Config(tbl_rows=30) :
        print (train_df.group_by (['product', 'country']).agg(pl.col('num_sold').n_unique()))


with pl.Config(tbl_rows=20):
        display (train_df.group_by (['store', 'country']).agg(pl.col('num_sold').n_unique()))


import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from IPython.display import Image


sns.set(rc={"figure.figsize": (20, 10)})


if create_EDA :
    sns.histplot (train_df.filter (pl.col('num_sold').is_not_null()), x= 'num_sold')
    plt.savefig ("/kaggle/working/graph1.jpg")
else :
    display (Image (filename = "/kaggle/working/graph1.jpg"))
    


sales_present = train_df.group_by (by ="date").len().sort("by")
print (sales_present)


cat_features =   ['date', 'num_sold', 'country', 'store',  'product']

for c in cat_features:
    print (f'values for {c}')
    with pl.Config(tbl_rows=50):
       display (train_df.group_by(by = c).agg (pl.col('num_sold').mean(), pl.col('id').len()).sort('num_sold')   )


low_guys = train_df.filter (pl.col('num_sold') < 30)
print (low_guys.group_by (by = 'country').len())


if create_EDA :
    sns.lineplot (data = train_df.filter (pl.col('country') == "Kenya").to_pandas(), x = 'date', y= 'num_sold', hue = "store")
    plt.savefig ("/kaggle/working/graph2.jpg")
else :
    display (Image (filename = "/kaggle/working/graph2.jpg"))
    


if create_EDA :
    sns.lineplot (data = train_df.filter (pl.col('country') == "Kenya").to_pandas(), x = 'date', y= 'num_sold', hue = "product")
    plt.savefig ("/kaggle/working/graph3.jpg")
else :
    display (Image (filename = "/kaggle/working/graph3.jpg"))
    


if create_EDA :
    sns.lineplot (data = train_df.filter (pl.col('country') == "Italy").to_pandas(), x = 'date', y= 'num_sold', hue = "store")
    plt.savefig ("/kaggle/working/graph4.jpg")
else :
    display (Image (filename = "/kaggle/working/graph4.jpg"))
    


print (f"Date start: {train_df.get_column ('date').min()} , Date end: {train_df.get_column ('date').max()}")

print (f"Date start: {test_df.get_column ('date').min()} , Date end: {test_df.get_column ('date').max()}")


if create_EDA :
    sns.lineplot (data = train_df.filter (pl.col('country') == "Italy").to_pandas(), x = 'date', y= 'num_sold', hue = "product")
    plt.savefig ("/kaggle/working/graph5.jpg")
else :
    display (Image (filename = "/kaggle/working/graph5.jpg"))
    


if create_EDA :
    sns.lineplot (data = train_df.filter (pl.col('country') == "Norway").to_pandas(), x = 'date', y= 'num_sold', hue = "store")
    plt.savefig ("/kaggle/working/graph6.jpg")
else :
    display (Image (filename = "/kaggle/working/graph6.jpg"))
    


def add_gdp_per_capita (df : pl.DataFrame, gdp_per_capita : pl.DataFrame ) -> pl.DataFrame :

       result = df.with_columns ((pl.col("country") + "_" + pl.col("date").str.head(4)).alias ("country_year")   )
       lookup = gdp_per_capita.with_columns ((pl.col("Entity") + "_" + pl.col("Year").cast(pl.String)).alias ("country_year")   )
       result = result.join (lookup, how = "left", left_on = "country_year", right_on = "country_year"   )                                      
       return result. drop ([ 'country_year', 'Entity', 'Code', 'Year', '900793-annotations'])

# add_gdp_per_capita (train_df, gdp_per_capita)



import holidays

years_range = [ n for n in range (2010, 2020)]
ken_holidays = holidays.country_holidays("KE", years = years_range)  
usa_holidays = holidays.country_holidays("US", years = years_range)  
fin_holidays = holidays.country_holidays("FI", years = years_range)  
can_holidays = holidays.country_holidays("CA", years = years_range)  
sin_holidays = holidays.country_holidays("SG", years = years_range)  
ita_holidays = holidays.country_holidays("IT", years = years_range)

def add_holiday (df : pl.DataFrame) -> pl.DataFrame :
    result = df
    result = result.with_columns(pl.lit (False).alias ("is_holiday"))
    result = result.with_columns(pl.when (pl.col("country") == "USA").then (
                      pl.col('p-date').is_in (usa_holidays.keys())).otherwise (
                      pl.col("is_holiday")).alias ('is_holiday'))
    result = result.with_columns(pl.when (pl.col("country") == "Finland").then (
                      pl.col('p-date').is_in (fin_holidays.keys())).otherwise (
                      pl.col("is_holiday")).alias ('is_holiday'))
    result = result.with_columns(pl.when (pl.col("country") == "Canada").then (
                      pl.col('p-date').is_in (can_holidays.keys())).otherwise (
                      pl.col("is_holiday")).alias ('is_holiday'))
    result = result.with_columns(pl.when (pl.col("country") == "Italy").then (
                      pl.col('p-date').is_in (ita_holidays.keys())).otherwise (
                      pl.col("is_holiday")).alias ('is_holiday'))
    result = result.with_columns(pl.when (pl.col("country") == "Kenya").then (
                      pl.col('p-date').is_in (ken_holidays.keys())).otherwise (
                      pl.col("is_holiday")).alias ('is_holiday'))
    result = result.with_columns(pl.when (pl.col("country") == "Singapore").then (
                      pl.col('p-date').is_in (sin_holidays.keys())).otherwise (
                      pl.col("is_holiday")).alias ('is_holiday'))
    return result


from datetime import date
  

num_to_days = {1 : "Mon", 2 : "Tue", 3 : "Wed", 4 : "Thu", 5  :"Fri", 6 : "Sat", 7 : "Sun"}

def add_features (raw : pl.DataFrame) -> pl.DataFrame :
    result = raw
    if "num_sold" in result.columns :
        result = result.with_columns (pl.col ("num_sold").log().alias ("num_sold_log"))
        
    result = result.with_columns ((pl.col('country') + '_' + 
                                   pl.col('store') + '_' + 
                                   pl.col ('product')).alias ('countr_store_product'), 
                          pl.col ('date').str.to_date().alias('p-date')) 
                          
    result = result.with_columns(pl.col ('p-date').dt.weekday ().replace_strict (num_to_days, return_dtype = pl.String ).alias ('weekday'),
                                    pl.col ('p-date').dt.quarter ().alias ('quarter'), 
                                    pl.col ('p-date').dt.year ().alias ('year'), 
                                    pl.col ('p-date').dt.ordinal_day ().alias ('day_of_year'), 
                                    pl.col("date").str.tail(5).alias ("month_day"))
    result = result.with_columns ((pl.col("day_of_year")/365 *2  * np.pi).sin ().alias ("year_sin"), 
                                  (pl.col("day_of_year")/365 *2 * np.pi).cos ().alias ("year_cos"), 
                                  (pl.col("day_of_year")/365 *4  * np.pi).sin ().alias ("year_sin2x"), 
                                  (pl.col("day_of_year")/365 *4 * np.pi).cos ().alias ("year_cos2x"),
                                  ((pl.col("day_of_year") + 90) /365 *2  * np.pi).sin ().alias ("year_sinx_plus90"), 
                                  ((pl.col("day_of_year") + 90)/365 *2 * np.pi).cos ().alias ("year_cosx_plus90"))
    result = result.with_columns (pl.col ("month_day").is_in (top_date).alias ("max_date"), 
                                 pl.col ("month_day").is_in (bottom_date).alias ("min_date")) 
    
    # result = result.with_columns((366 - pl.col ('day_of_year')).alias ('remaining_day_of_year'))                                
    # result = result.with_columns( pl.min_horizontal(pl.col ('day_of_year'), pl.col ('remaining_day_of_year')).alias ('close_to_season'))
    result = add_holiday (result)                             
    
    result = result.with_columns(pl.col ('weekday').is_in ([ 'Fri', 'Sat', 'Sun']).alias ('is_weekend'))
    result = add_gdp_per_capita (result, gdp_per_capita)
    if "num_sold" in raw.columns :
        result = result.with_columns(pl.when (pl.col ('num_sold') < 50).then (
            pl.lit(5)).otherwise(pl.lit(1)).alias ('my_weight'))
    
        
    return result


train_clean_df = add_features (train_df.filter (pl.col("num_sold").is_not_null()) )

# train_clean_df = train_clean_df.filter (pl.col ("max_date").not_())

test_clean_df = add_features (test_df)


# train_null_df = train_clean_df.filter(pl.col('num_sold').is_null()).sample (fraction = 1, shuffle = True) 

train_clean_df = train_clean_df.sample (fraction = 1, shuffle = True) 

n = train_clean_df.shape [0]

train_size = int (np.rint (n * 0.99))
validation_size = n - train_size

train_clean_df = train_clean_df.head (train_size)

validation_df = train_clean_df.tail (validation_size)

display (train_clean_df.head())
display (train_clean_df.describe())



!pip install ray==2.10.0


!pip install autogluon.tabular --no-cache-dir -q
!pip install -U ipywidgets


from autogluon.tabular import TabularPredictor

from autogluon.common import space


print ('done')


%%time
 

predictor = TabularPredictor(path = '/kaggle/working/Autogluon3',
                                       label='num_sold', 
                               problem_type = 'regression', 
                               eval_metric =  'mean_absolute_percentage_error',  
                               sample_weight = 'my_weight',
                               learner_kwargs = {'ignored_columns' : [
                                   'id',
                                   'num_sold_log',
#                                  'my_weight'
                                    ]})

print (f'data schema : {train_clean_df.collect_schema}')

predictor.fit(train_data= train_clean_df.to_pandas(), 
                        presets= 'experimental_quality',
# best_quality,  medium_quality, 'experimental_quality',                         
                        time_limit = 30000,
                        num_gpus=1, 
#                        dynamic_stacking=False, num_stack_levels=1
#                        hyperparameters=hyperparameters,
#                        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        )


#in case something went wrong :
# predictor = TabularPredictor.load("/kaggle/working/Autogluon3")


predictor.leaderboard()


 predictor.evaluate (validation_df.to_pandas())


# not 100 % correct if we go after the num_sold with log transformation 

y_pred_log = pl.Series ("predicted_num_sold_log", predictor.predict (validation_df.to_pandas()))
y_pred     = pl.Series ("predicted_num_sold", predictor.predict (validation_df.to_pandas()))
y_true     = validation_df.get_column ('num_sold')
y_true_log = validation_df.get_column ('num_sold_log')
# y_true = y_true_log.exp()
# y_true = y_true_log
# y_pred = y_pred_log.exp()
# y_pred = y_pred_log

compare = pl.DataFrame ([y_pred, y_true])
# compare = compare.rename ({"predicted_num_sold_log" : "predicted_num_sold", "num_sold_log" : "num_sold"})


compare = compare.with_columns (((pl.col ("predicted_num_sold") - pl.col ("num_sold"))/ pl.col ("num_sold")).abs().alias ('relative fault') )

print (compare)



sns.scatterplot (data = compare.to_pandas(), x = "num_sold", y = "predicted_num_sold")
plt.savefig ("/kaggle/working/graph7.jpg")

# display (Image (filename = "/kaggle/working/graph7.jpg"))
    




print (compare)

print (compare.select ("relative fault").describe())

sns.scatterplot (data = compare.to_pandas(),  x = "num_sold", y = "relative fault")
plt.savefig ("/kaggle/working/graph8.jpg")

#    display (Image (filename = "/kaggle/working/graph8.jpg"))    
    


predictions_log = pl.Series (predictor.predict(test_clean_df.to_pandas () ))
#predictions = predictions_log.exp()
predictions = predictions_log
predictions_id  = test_clean_df.with_columns (predictions.alias ("num_sold") )

display (predictions_id)
# predictions_id  = predictions_id.with_columns (pl.when (pl.col("max_date")).then (pl.col ("num_sold") * 1.4).otherwise (pl.col ("num_sold")))

print (predictions_id.describe())

print ('done')



for country in countries :
    print (f"now plotting {country}")
    sns.lineplot (data = predictions_id.filter (pl.col('country') == country).to_pandas(), x = 'date', y= 'num_sold', hue = "product")
    plt.show()
    plt.savefig (f"/kaggle/working/graph9{country}.jpg")
#     display (Image (filename = f"/kaggle/working/graph9{country}.jpg"))    
    



submission = predictions_id.select (['id', 'num_sold'])
submission  = submission.with_columns (pl.max_horizontal (pl.col('num_sold'), 0).round(0)) 

print (submission.describe())
submission.write_csv('submission.csv')
print ('submission finished')


submission.head()


%%time 


predictor.save_space()

import zipfile, os


def zip_files_in_directory(directory, zip_name):
    num_files_deleted = 0 
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory))
                os.remove(file_path)
                num_files_deleted += 1
    return num_files_deleted  
# Example usage
directory = '/kaggle/working/autogluon_time_series'
zip_name = 'Autogluon_Kenya.zip'
n = zip_files_in_directory(directory, zip_name)
print(f"deleted {n } files in {directory}")

directory = '/kaggle/working/Autogluon_bootstrap'
zip_name = 'Autogluon_not_Kenya.zip'
m = zip_files_in_directory(directory, zip_name)

print(f"deleted { m} files in {directory}")

directory = '/kaggle/working/Autogluon3'
zip_name = 'Autogluon3.zip'
m = zip_files_in_directory(directory, zip_name)

print(f"deleted { m} files in {directory}")

