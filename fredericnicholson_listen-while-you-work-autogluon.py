import numpy as np # linear algebra
import polars as pl # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_df = pl.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pl.read_csv ("/kaggle/input/playground-series-s5e4/test.csv")
#original = pl.read_csv('/kaggle/datasets/ysthehurricane/podcast-listening-time-prediction-dataset/podcast_dataset.csv')
#train_df = pl.concat ([train_df, original], how = 'vertical')
sample_submission = pl.read_csv ("/kaggle/input/playground-series-s5e4/sample_submission.csv")
import polars.selectors as cs
train_df.select (cs.numeric()).describe().transpose(include_header=True)


train_df_mark = train_df.with_columns (pl.lit (1).alias ("is_train")).drop(["Listening_Time_minutes"])
test_df_mark = test_df.with_columns (pl.lit (0).alias ("is_train"))                                       
                                       
train_for_episode_length = pl.concat ([train_df_mark, test_df_mark], how = "vertical").filter (
    pl.col("Episode_Length_minutes").is_not_null())
test_for_episode_length = pl.concat ([train_df_mark, test_df_mark], how = "vertical").filter (
    pl.col("Episode_Length_minutes").is_null())



train_for_episode_length = train_for_episode_length.filter (pl.col("Episode_Length_minutes") <= 121)
train_for_episode_length.describe()



!pip install ray==2.10.0
!pip install scikit-learn==1.5.2
!pip install autogluon.tabular --no-cache-dir -q
!pip install -U ipywidgets


CALC_EPISODE = True 

from autogluon.tabular import TabularPredictor

episode_predictor = TabularPredictor(path = '/kaggle/working/Autogluon3',
                                       label='Episode_Length_minutes', 
                               problem_type = 'regression', 
                               eval_metric =  'root_mean_squared_error',  
                               # sample_weight = 'my_weight',
                               verbosity  = 2,
                               learner_kwargs = {'ignored_columns' : [
                                   'id',
                               #    'Price'
                               #   'my_weight'
                                    ]})

if CALC_EPISODE  :
    episode_predictor.fit(train_data= train_for_episode_length.to_pandas(), 
                        presets= 'best_quality',
    # best_quality, high_quality, medium_quality, 'experimental_quality',                         
                        time_limit = 10000,
                        # num_gpus=1,
#                        raise_on_no_models_fitted = True,
#                        dynamic_stacking=False, 
#                        num_stack_levels=1,
                        #hyperparameters=custom_hyperparameters,
#                         hyperparameters = my_search_hyperparameters  ,
#                         hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        )



if CALC_EPISODE  :
    display (episode_predictor.leaderboard())


if CALC_EPISODE  :
    episode_predictions  = episode_predictor.predict( test_for_episode_length.to_pandas()) 
    print (episode_predictions.describe() )
    episode_predictions_length_df = test_for_episode_length.with_columns (pl.Series (episode_predictions).alias ("predicted_episode_length")).select (["id", "predicted_episode_length"])
    print (episode_predictions_length_df)
    episode_predictions_length_df.write_csv("/kaggle/working/estimate_episode_length.csv")
else :
    episode_predictions_length_df = pl.read_csv("/kaggle/working/estimate_episode_length.csv")


episode_predictions_length_df.describe()


cat_features = train_df.select(cs.string()).columns
print (cat_features)


cat_impact = {}
for col in cat_features :
   cat_impact [col] = train_df.group_by(col).agg((pl.col("Listening_Time_minutes") / (pl.col("Episode_Length_minutes") + 0.01) ).mean()) 

print (cat_impact)


train_df.group_by("Genre").agg((pl.col("Listening_Time_minutes") / (1 + pl.col("Episode_Length_minutes"))).mean())


train_df.group_by("Episode_Sentiment").agg((pl.col("Listening_Time_minutes") / (1 + pl.col("Episode_Length_minutes"))).mean())


train_df.group_by("Episode_Title").agg((pl.col("Listening_Time_minutes") / (1 + pl.col("Episode_Length_minutes"))).mean()).sort(
    "Episode_Title")



def insert_estimated_episode_length (df, estimate :pl.DataFrame) -> pl.DataFrame :
    result = df.join (estimate, on = "id", how = "left" )
    result = result.with_columns_seq ((pl.col("Episode_Length_minutes").is_not_null().cast (pl.UInt8) + 1).alias ("is_estimated"),
        pl.when (pl.col("Episode_Length_minutes").is_null()).then (
                pl.col("predicted_episode_length") ).otherwise(
                pl.col("Episode_Length_minutes"). alias ("Episode_Length_minutes")))
    return result.drop("predicted_episode_length")            


def clean_column (df :pl.DataFrame, col : str, lower, upper :float ) -> pl.DataFrame :
    
    result  = df.with_columns (pl.col (col).mean().over (  ["Podcast_Name", 
                                                            "Episode_Sentiment",
                                                            "Episode_Title"]).alias (f"avg_{col}"))
    result = result.with_columns (pl.col(col).fill_null(pl.col(f"avg_{col}")))
    result  = result.with_columns (pl.col(col).clip (lower_bound = lower, upper_bound = upper))
    return result.drop(f"avg_{col}")

def add_feature (df :pl.DataFrame ) -> pl.DataFrame :
    result  = df.with_columns (
        (pl.col("Episode_Sentiment") + "_" + pl.col("Genre")).alias ("Sentiment_Genre"),
        (pl.col("Episode_Sentiment") + "_" + pl.col("Genre") + "_" + pl.col("Episode_Title")).alias ("Sentiment_Genre_title"),
        pl.col("Episode_Title").str.slice (8).cast(pl.UInt8).alias ("episode_num"),
        pl.col("Episode_Length_minutes") / ( 1 + pl.col("Number_of_Ads")).alias ("Length_ads_ratio"),
        pl.col("Host_Popularity_percentage") * pl.col("Guest_Popularity_percentage").alias ("Guest_host_product"),
        pl.col("Episode_Length_minutes").cast(pl.String).str.split (".").list.last().str.len_bytes().alias ("num_digits"))       
    result 
    return result

def add_target_encoding (df :pl.DataFrame, target_distribution : dict ) -> pl.DataFrame :
    result = df 
    for col in target_distribution.keys ():
        target_df = target_distribution [col]
        replace_with_target = dict(zip(target_df.get_column (col).to_list(), 
                                       target_df.get_column ("Listening_Time_minutes").cast(pl.Float32).to_list()))
        result = result.with_columns (pl.col (col).replace_strict(replace_with_target).alias (f"{col}_target_ratio"))
        result = result.with_columns ((pl.col (f"{col}_target_ratio") * pl.col("Episode_Length_minutes")).alias (f"{col}_target"))
        result  = result.drop(col)
    print (result)
    
    return result    

def apply_pipe (df: pl.DataFrame) -> pl.DataFrame :
    return df.pipe (
                insert_estimated_episode_length, episode_predictions_length_df).pipe(
                clean_column, "Episode_Length_minutes", 0, 200).pipe (
                clean_column, "Host_Popularity_percentage", 10, 100).pipe (
                clean_column, "Guest_Popularity_percentage", 10, 100)
    #.pipe(
    #            add_feature)
    #.pipe (
    #                  add_target_encoding, cat_impact  
    #                  )



train_df_clean = apply_pipe (train_df)

train_df_clean = train_df_clean.drop_nulls()
display (train_df_clean.select(cs.numeric()).describe().transpose(include_header=True))

test_df_clean = apply_pipe (test_df)
display (test_df_clean.select(cs.numeric()).describe().transpose(include_header=True))



print (train_df_clean.columns)

train_df_clean.head(5)


train_df_clone = train_df_clean.clone()



train_df = train_df_clone.sample(fraction = 0.95, shuffle = True)

train_df_test = train_df_clone.join (train_df, on = "id", how = 'anti')


print (f"{train_df.shape = }, {train_df_test.shape}")






import sklearn.utils.validation

from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(path = '/kaggle/working/Autogluon2',
                                       label='Listening_Time_minutes', 
                               problem_type = 'regression', 
                               eval_metric =  'root_mean_squared_error',  
                               sample_weight = 'is_estimated',
                               verbosity  = 2,
                               learner_kwargs = {'ignored_columns' : [
                                   'id',
                               #    'Price'
                               #   'my_weight'
                                    ]})

predictor.fit(train_data= train_df.to_pandas(), 
                        presets= 'experimental_quality',
    # best_quality, high_quality, medium_quality, 'experimental_quality',                         
                        time_limit = 24000,
#                        num_gpus=1,
#                        raise_on_no_models_fitted = True,
                        dynamic_stacking=False, 
                        num_stack_levels=1,
                        #hyperparameters=custom_hyperparameters,
#                         hyperparameters = my_search_hyperparameters  ,
#                         hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        )

predictor.leaderboard()


# predictor.feature_importance(train_df_test.to_pandas())


# from sklearn.metrics import mean_squared_error


# predictions_test  = predictor.predict( train_df_test.to_pandas()) 

# train_df_test_compare = train_df_test.get_column ("Listening_Time_minutes").to_numpy()







# from math import sqrt
# print (sqrt (mean_squared_error(predictions_test, train_df_test_compare)))



predictions  = predictor.predict( test_df_clean.to_pandas()) 
                        


def create_submission(df, predictions):
    submission = df.with_columns (pl.Series(predictions).alias ('Listening_Time_minutes'))
    submission = submission.with_columns (pl.col ('Listening_Time_minutes').clip (lower_bound = 0, upper_bound = 120))
    submission = submission.select (["id", "Listening_Time_minutes"])
    print (submission)
    submission.write_csv('submission.csv')
    print("submission.csv created")


create_submission(test_df, predictions)

