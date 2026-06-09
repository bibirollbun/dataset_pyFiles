import numpy as np # linear algebra
import polars as pl # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import os
import shutil  # For handling directory removal (optional)

def delete_files_in_directory(directory_path, remove_empty_directory=False):
    """
    Deletes all files within the specified directory.

    Args:
        directory_path (str): The path to the directory.
        remove_empty_directory (bool, optional):  If True, attempts to remove the directory
                                              after deleting all files.  Defaults to False.
    """
    try:
        # Check if the directory exists
        if not os.path.isdir(directory_path):
            print(f"Error: Directory '{directory_path}' not found.")
            return

        # Iterate through all files and directories in the specified directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Delete the file
                    print(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):  # Handle subdirectories recursively
                    # Option 1: Delete all files in the subdirectory and then remove the subdirectory
                    delete_files_in_directory(file_path, remove_empty_directory=True)  # Recursively delete files
                    if remove_empty_directory:  # Only remove if the flag is set
                        try:
                            os.rmdir(file_path)  # Try to remove empty directory
                            print(f"Removed empty subdirectory: {file_path}")
                        except OSError as e:
                            print(f"Error removing subdirectory {file_path}: {e}")
                                # Option 2:  Completely delete the subdirectory and its contents
                    # shutil.rmtree(file_path) # Use this cautiously!  Deletes the directory and ALL contents.
                    # print(f"Deleted subdirectory (and contents): {file_path}")


            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        # Optionally remove the directory itself if it's now empty
        if remove_empty_directory:
            try:
                os.rmdir(directory_path)
                print(f"Removed empty directory: {directory_path}")
            except OSError as e:
                print(f"Error removing directory {directory_path}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
                

delete_files_in_directory ("/kaggle/working/march_madness/models")
delete_files_in_directory ("/kaggle/working/march_madness/male/models")
delete_files_in_directory ("/kaggle/working/march_madness/female/models")


w_teams = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/WTeams.csv").collect()
m_teams = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/MTeams.csv").collect()
m_compact_results = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv").collect()
w_compact_results = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv").collect()
m_rankings = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv").collect()

w_seeds = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv").collect()
m_seeds = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv").collect()

m_NCAATourneyCompactResults = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv").collect()
w_NCAATourneyCompactResults = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv").collect()
# submission_stage1 = pl.scan_csv ("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv").collect()
submission_stage2 = pl.scan_csv ("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv").collect()

m_truth = pl.scan_csv ("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv").collect()
w_truth = pl.scan_csv ("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv").collect()


m_names = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/MTeams.csv")
w_names = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/WTeams.csv")


m_names.filter (pl.col("TeamName") =="Auburn").collect()



def extract_seeds (seeds : pl.DataFrame) -> pl.DataFrame :
    
    result = seeds.with_columns (pl.col("Seed").str.head(1).alias("Conference"), 
                               pl.col("Seed").str.slice(1,2).cast(pl.UInt8).alias("C_Seed"))
    return result.drop("Seed")

female_seeds_Season = extract_seeds (w_seeds) 
male_seeds_Season = extract_seeds (m_seeds) 


# converting the historic NCAA results into the format used for submission, 

def create_submission_format (compact_NCAA_results : pl.DataFrame) -> pl.DataFrame :
    bare_minimum1 = compact_NCAA_results.select (["Season", "WTeamID", "LTeamID"]).filter (pl.col("WTeamID") < pl.col("LTeamID"))
    submission_format1 =  bare_minimum1.with_columns ((pl.col("Season").cast (pl.String) + "_" + 
                                               pl.col("WTeamID").cast (pl.String) + "_" + pl.col("LTeamID").cast (pl.String)).alias ('ID'), 
                                                     pl.lit(1).alias ('pred'))
    
    bare_minimum2 = compact_NCAA_results.select (["Season", "WTeamID", "LTeamID"]).filter (pl.col("WTeamID") > pl.col("LTeamID"))
    submission_format2 =  bare_minimum2.with_columns ((pl.col("Season").cast (pl.String) + "_" + 
                                               pl.col("LTeamID").cast (pl.String) + "_" + pl.col("WTeamID").cast (pl.String)).alias ('ID'), 
                                                     pl.lit(0).alias ('pred'))
    submission_format = pl.concat ([submission_format1,submission_format2  ]).sample (fraction = 1.0, shuffle = True)
    return submission_format.drop(["Season", "WTeamID", "LTeamID"]) 

historic_NCAA_m = create_submission_format (m_NCAATourneyCompactResults)
historic_NCAA_w = create_submission_format (w_NCAATourneyCompactResults)

historic_NCAA = pl.concat ([historic_NCAA_m, historic_NCAA_w])

display (historic_NCAA)

print  (f"NCAA men start :{historic_NCAA_m.get_column ('ID').min()}, end: {historic_NCAA_m.get_column ('ID').max()}")
print  (f"NCAA women start :{historic_NCAA_w.get_column ('ID').min()}, end: {historic_NCAA_w.get_column ('ID').max()}")


# depreceated, as it does not add value to auto gluon 
def extract_rankings (rankings : pl.DataFrame) -> pl.DataFrame :
    rankings  = rankings.with_columns ((pl.col("Season").cast (pl.String) + "_" + pl.col("TeamID").cast(pl.String)).alias ("Season_Team"))

    last_ranking_day  = rankings.group_by(["Season", "SystemName"]).agg(pl.col("RankingDayNum").max()).sort("Season")

    rankings_last = rankings.join (last_ranking_day, how = "inner", on = ["Season", "SystemName", "RankingDayNum"])


    team_rankings = rankings_last.pivot("SystemName", index="Season_Team", values="OrdinalRank")

    team_rankings = team_rankings.with_columns (pl.col("Season_Team").str.split("_").list.first().cast (pl.Int64).alias ("Season"),
                                               pl.col("Season_Team").str.split("_").list.last().cast (pl.Int64).alias ("TeamID"))

    return team_rankings.drop("Season_Team")

male_team_rankings = extract_rankings (m_rankings)

print (male_team_rankings.head(3))


display (w_compact_results.get_column("Season").unique().to_list())


m_compact_results.filter (pl.col("Season") == 2021).describe()


w_truth


def win_loss_record (teams : pl.DataFrame) -> pl.DataFrame :
    
    result = teams.select ("TeamID")
    base_file =  w_compact_results if (teams.get_column("TeamID").min() > 2000) else m_compact_results
    # season_results = base_file.filter (pl.col("Season") == season)
    # base_file = base_file.with_columns (pl.col("DayNum").max().over("Season").alias ("Season_length"))
    # p-day measures how late in the season a game is played with a min of 0.5 and a max of 1.5.this allows to give games 
    #    later in the season more importance 
    # Wmargin is the difference between win score and loose score 
    base_file = base_file.with_columns ((pl.col("DayNum")/ pl.col("DayNum").max().over("Season") + 0.5).alias ("p_day"), 
                                        (pl.col("WScore") - pl.col ("LScore")).alias ("WMargin"))  
        
    wins = base_file.group_by (["Season","WTeamID"]).len()
    losses = base_file.group_by (["Season","LTeamID"]).len()
    w_wins = base_file.group_by (["Season","WTeamID"]).agg (pl.col("p_day").sum() )
    w_losses = base_file.group_by (["Season","LTeamID"]).agg (pl.col("p_day").sum() )
    wscores =  base_file.group_by (["Season","WTeamID"]).agg (pl.col("WScore").sum())
    lscores = base_file.group_by (["Season","LTeamID"]).agg (pl.col("LScore").sum())
    opp_wscores = base_file.group_by (["Season","WTeamID"]).agg (pl.col("LScore").sum())
    opp_lscores = base_file.group_by (["Season","LTeamID"]).agg (pl.col("WScore").sum())
    wMargin = base_file.group_by (["Season","WTeamID"]).agg (pl.col("WMargin").mean())
    lMargin = base_file.group_by (["Season","LTeamID"]).agg (pl.col("WMargin").mean() * (-1))

    result = result.join (wins, how = "left", left_on = ["TeamID"], right_on = ["WTeamID"])
    result = result.rename ({"len" : "games won"})
    result = result.join (losses, how = "left", left_on = ["TeamID", "Season"], right_on = ["LTeamID", "Season"])
    result = result.rename ({"len" : "games lost"})
    result = result.join (w_wins, how = "left", left_on = ["TeamID", "Season"], right_on = ["WTeamID", "Season"])
    result = result.rename ({"p_day" : "weighted games won"})
    result = result.join (w_losses, how = "left", left_on = ["TeamID", "Season"], right_on = ["LTeamID", "Season"])
    result = result.rename ({"p_day" : "weighted games lost"}) 
    result = result.join (wscores, how = "left", left_on = ["TeamID", "Season"], right_on = ["WTeamID", "Season"])
    result = result.rename ({"WScore" : "points in win"})
    result = result.join (lscores, how = "left", left_on = ["TeamID", "Season"], right_on = ["LTeamID", "Season"])
    result = result.rename ({"LScore" : "points in loss"})
    result = result.join (opp_wscores, how = "left", left_on = ["TeamID", "Season"], right_on = ["WTeamID", "Season"])
    result = result.rename ({"LScore" : "dev points in win"})
    result = result.join (opp_lscores, how = "left", left_on = ["TeamID", "Season"], right_on = ["LTeamID", "Season"])
    result = result.rename ({"WScore" : "dev points in loss"})
    result = result.join (wMargin, how = "left", left_on = ["TeamID", "Season"], right_on = ["WTeamID", "Season"])
    result = result.rename ({"WMargin" : "mean margin win"})
    result = result.join (lMargin, how = "left", left_on = ["TeamID", "Season"], right_on = ["LTeamID", "Season"])
    result = result.rename ({"WMargin" : " mean margin loss"})
    
    result = result.with_columns ((pl.col("games won") + pl.col("games lost")).alias ("total games"), 
                                  (pl.col("points in win") + pl.col("points in loss")).alias ("total points"), 
                                  (pl.col("dev points in win") + pl.col("dev points in loss")).alias ("total dev points"))
    result = result.with_columns ((pl.col("total points") / pl.col("total games")).alias ("points mean"), 
                                  (pl.col("total dev points") / pl.col("total games")).alias ("dev points mean"), 
                                  (pl.col("dev points in win") + pl.col("dev points in loss")).alias ("total dev points"))
    
    just_games_won = result.select(["Season", "TeamID", "games won"]) 
    weight_score_per_game = base_file.join (just_games_won, how = "inner", left_on = ["Season","LTeamID"], right_on = ["Season", "TeamID"])
    weight_score = weight_score_per_game.group_by(["Season", "WTeamID"]).agg(pl.col("games won").sum())
    weight_score = weight_score.rename ({"games won" : "opponent games won"})
    result = result.join (weight_score, how = "left", left_on = ["TeamID", "Season"], right_on = ["WTeamID", "Season"])
        
    return result.fill_null(0)

male_team_results = win_loss_record (m_teams)

display (male_team_results)

female_team_results = win_loss_record (w_teams)

display (female_team_results)


def create_train (submission_format, tournament_results :pl.DataFrame) -> pl.DataFrame :
    
    truth_is_female = (tournament_results.get_column ("WTeamID").min() > 2000)
    
    result = submission_format.with_columns (pl.col("ID").str.split ("_").list.get(0).cast(pl.Int64).alias ("Season"),
                             pl.col("ID").str.split ("_").list.get(1).cast(pl.Int64).alias ("Team1"),
                             pl.col("ID").str.split ("_").list.get(2).cast(pl.Int64).alias ("Team2"))
   
    if truth_is_female :
        result = result.filter (pl.col('Team1') > 2000)
    else :
        result = result.filter (pl.col('Team1') < 2000)
   
    bare_minimum = tournament_results.select (["Season", "WTeamID", "LTeamID", "WScore", "LScore"])
    result = result.join (bare_minimum, how = "left", left_on = ["Season", "Team1", "Team2"], right_on = ["Season", "WTeamID", "LTeamID"])
    result = result.join (bare_minimum, how = "left", left_on = ["Season", "Team1", "Team2"], right_on = ["Season", "LTeamID", "WTeamID"])

    
    result = result.with_columns (pl.when (pl.col("WScore") > 0 ).then (pl.lit(1)).otherwise (
                                  pl.when (pl.col("WScore_right") > 0 ).then (pl.lit(0)).otherwise (
                                           pl.lit(0.5))).alias ("truth"))
    
    for unused in ['Pred', 'DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT', 'DayNum_right', 
                   'WScore_right', 'LScore_right', 'WLoc_right', 'NumOT_right'] :
       if unused in result.columns :
           result = result.drop (unused)
            
    return result
    


male_training = create_train (historic_NCAA_m, m_truth).filter ((pl.col("truth") ==1) | (pl.col("truth") ==0))
female_training = create_train (historic_NCAA_w, w_truth).filter ((pl.col("truth") ==1) | (pl.col("truth") ==0))

male_submission = create_train (submission_stage2, m_truth)
female_submission = create_train (submission_stage2, w_truth)

print (f" total size {male_training.shape  }   ")
print (f" total size {male_submission.shape }   ")

print (male_training.head(3))


def add_info (df, info : pl.DataFrame) -> pl.DataFrame :
  # the information in info for each team in df     
  df = df.join (info, how = "left", left_on = ["Season","Team1"], right_on = ["Season","TeamID"])
  df = df.join (info, how = "left", left_on = ["Season","Team2"], right_on = ["Season","TeamID"])  
  return df 


def build_pipeline (df : pl.DataFrame) -> pl.DataFrame :   
    if  df.get_column ("Team1").min() < 2000 :
        team_results = male_team_results
        seeds_season = male_seeds_Season
    else : 
        team_results = female_team_results
        seeds_season = female_seeds_Season
    return df.pipe (add_info, info = team_results).pipe (add_info, info = seeds_season)
    
    


male_training = build_pipeline (male_training)

try :
    my_weights = pl.scan_csv("/kaggle/working/mncaa_weights.csv").collect()
    male_training = male_training.join (my_weights, how = "right", on ="ID")
    print ("weights added for male NCAA training")
    print (f"final feature list for male NCAA  {male_training.columns = }")
except:
    print ("no weights available for male NCAA")

female_training = build_pipeline (female_training)
try :
    my_weights = pl.scan_csv("/kaggle/working/wncaa_weights.csv").collect()
    female_training = female_training.join (my_weights, how = "right", on ="ID")
    print ("weights added for female NCAA training")
    print (f"final feature list for female NCAA  {female_training.columns = }")
except:
    print ("no weights available for female NCAA")


male_submission = build_pipeline (male_submission)
female_submission = build_pipeline (female_submission)


!pip install ray==2.10.0
!pip install autogluon.tabular --no-cache-dir -q
!pip install -U ipywidgets


print (male_training.shape) 
print (male_submission.shape)
print (male_training.columns) 





from autogluon.tabular import TabularPredictor

if "my_weight" in male_training.columns :
    m_predictor = TabularPredictor(path = '/kaggle/working/march_madness/male',
                                       label='truth', 
                               problem_type = 'binary', 
                               eval_metric =  'accuracy',  
                               sample_weight = 'my_weight',
                               verbosity  = 2,
                               learner_kwargs = {'ignored_columns' : [
                                   'ID', "pred"]})
else :
    m_predictor = TabularPredictor(path = '/kaggle/working/march_madness/male',
                                       label='truth', 
                               problem_type = 'binary', 
                               eval_metric =  'accuracy',  
                               # sample_weight = 'my_weight',
                               verbosity  = 2,
                               learner_kwargs = {'ignored_columns' : [
                                   'ID', "pred"]})
m_predictor.fit(train_data= male_training.to_pandas(), 
                        presets= 'best_quality',
    # best_quality, high_quality, 'medium_quality, 'experimental_quality',                         
                        time_limit = 18000,
                        # num_gpus=0,
                        raise_on_no_models_fitted = True,
#                        dynamic_stacking=False, 
#                        num_bag_folds =2,
#                        num_stack_levels=1,
                        #hyperparameters=hyper_search,
#                         hyperparameters = my_search_hyperparameters  ,
                        #hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        )                              
                                


m_predictor.leaderboard() 


historic_prediction = pl.Series (m_predictor.predict (male_training.to_pandas()))


male_weighted_train = male_training.with_columns (pl.when (pl.col("truth") == historic_prediction).then(
                                                      pl.lit(1)).otherwise (pl.lit(10)).alias ("my_weight")
)

print ("how many are predicted correctly")
print (male_weighted_train.group_by ("my_weight").len())

male_weighted_train = male_weighted_train.select (["ID", "my_weight"])

male_weighted_train.write_csv ("mncaa_weights.csv")



print (female_training.shape) 
print (female_submission.shape)
print (female_training.columns) 




if "my_weight" in female_training.columns :
    w_predictor = TabularPredictor(path = '/kaggle/working/march_madness/female',
                                       label='truth', 
                               problem_type = 'binary', 
                               eval_metric =  'accuracy',  
                               sample_weight = 'my_weight',
                               verbosity  = 2,
                               learner_kwargs = {'ignored_columns' : [
                                   'ID', 'pred']})
else : 
    w_predictor = TabularPredictor(path = '/kaggle/working/march_madness/female',
                                       label='truth', 
                               problem_type = 'binary', 
                               eval_metric =  'accuracy',  
                               # sample_weight = 'my_weight',
                               verbosity  = 2,
                               learner_kwargs = {'ignored_columns' : [
                                   'ID', 'pred']})
    
w_predictor.fit(train_data= female_training.to_pandas(), 
                        presets= 'best_quality',
    # best_quality, high_quality, medium_quality, 'experimental_quality',                         
                        time_limit = 18000,
                        # num_gpus=0,
                        raise_on_no_models_fitted = True,
                        #dynamic_stacking=False, 
                        #num_stack_levels=1,
                        #hyperparameters=hyper_search,
#                         hyperparameters = my_search_hyperparameters  ,
                        #hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        )     


w_predictor.leaderboard()


historic_prediction = pl.Series (w_predictor.predict (female_training.to_pandas()))


female_weighted_train = female_training.with_columns (pl.when (pl.col("truth") == historic_prediction).then(
                                                      pl.lit(1)).otherwise (pl.lit(10)).alias ("my_weight")
)

print ("how many are predicted correctly")
print (female_weighted_train.group_by ("my_weight").len())

female_weighted_train = female_weighted_train.select (["ID", "my_weight"])

female_weighted_train.write_csv ("wncaa_weights.csv")



male_submission = male_submission.fill_null(999.0)
male_submission = male_submission.fill_null("W")

male_march_madness_prediction =  m_predictor.predict_proba(male_submission.to_pandas()) 



female_submission = female_submission.fill_null(999.0)
female_submission = female_submission.fill_null("W")
female_march_madness_prediction =  w_predictor.predict_proba(female_submission.to_pandas())



march_madness_prediction = pl.concat ([pl.DataFrame(male_march_madness_prediction), pl.DataFrame(female_march_madness_prediction)], how = "vertical")

combined_id = pl.concat ([male_submission.get_column("ID"), female_submission.get_column("ID")], how = "vertical")

print (march_madness_prediction.head(3))


probabilty_first_team = march_madness_prediction.get_column ("1")
my_submit = pl.DataFrame ([combined_id, probabilty_first_team]) 
    
my_submit = my_submit.rename ({"1" : "Pred"})

my_submit.filter (pl.col("ID").str.contains ("1120"))




my_submit.write_csv("submission.csv")

