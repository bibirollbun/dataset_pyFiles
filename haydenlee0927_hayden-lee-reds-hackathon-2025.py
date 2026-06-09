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


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn import tree
from sklearn import linear_model
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


savant_data = pd.read_csv("/kaggle/input/reds-hackathon-2025/savant_data_2021_2023.csv")
lahman_data = pd.read_csv("/kaggle/input/reds-hackathon-2025/lahman_people.csv")
sample_submission = pd.read_csv("/kaggle/input/reds-hackathon-2025/sample_submission.csv")


savant_data = savant_data[(savant_data["batter"].isin(sample_submission["PLAYER_ID"])) | (savant_data["pitcher"].isin(sample_submission["PLAYER_ID"].values))]


# List of events that count as a Plate Appearance (PA)
plate_appearance_events = [
    "catcher_interf",
    "double",
    "double_play",
    "field_error",
    "field_out",
    "fielders_choice",
    "fielders_choice_out",
    "force_out",
    "grounded_into_double_play",
    "hit_by_pitch",  
    "home_run",
    "other_out",
    "sac_bunt",  
    "sac_bunt_double_play",
    "sac_fly",
    "sac_fly_double_play",
    "single",
    "strikeout",
    "strikeout_double_play",
    "triple",
    "triple_play",
    "walk" 
]

# List of events that count as an At-Bat (AB)
at_bat_events = [
    "double",
    "double_play",
    "field_error",
    "field_out",
    "fielders_choice",
    "fielders_choice_out",
    "force_out",
    "grounded_into_double_play",
    "home_run",
    "other_out",
    "single",
    "strikeout",
    "strikeout_double_play",
    "triple",
    "triple_play"
]

#Each row representing a season of a player
def create_initial_data(data):
    batter_stats = data.groupby(["batter","game_year"]).agg(
        plate_appearances=("events", lambda x: x.isin(plate_appearance_events).sum()),
        at_bats=("events", lambda x: x.isin(at_bat_events).sum()),
        hits=("events", lambda x: x.isin(["single", "double", "triple", "home_run"]).sum()),
        singles=("events", lambda x: (x == "single").sum()),
        doubles=("events", lambda x: (x == "double").sum()),
        triples=("events", lambda x: (x == "triple").sum()),
        home_runs=("events", lambda x: (x == "home_run").sum()),
        walks=("events", lambda x: (x == "walk").sum()),
        hit_by_pitch=("events", lambda x: (x == "hit_by_pitch").sum()),
        sacrifice_flies=("events", lambda x: (x == "sac_fly").sum()),
        strikeout = ("events", lambda x: (x.isin(["strikeout", "strikeout_double_play"]).sum())),
        rbi_before = ("bat_score", "sum"),
        rbi_after = ("post_bat_score", "sum"),
        run_exp = ("delta_run_exp", "sum"),
        barrels = ("launch_speed_angle", lambda x: (x == 6).sum()),
        good_contact = ("launch_speed_angle", lambda x: (x >= 4).sum()),
        wOBA_num = ("woba_value", "sum"),
        wOBA_denom = ("woba_denom", "sum")
    ).reset_index()
    return batter_stats

#Combining years for all players
def create_aggregate_data(data):
    batter_stats = data.groupby("batter", as_index=False).sum()
    batter_stats = batter_stats.drop("game_year", axis=1)
    return batter_stats

# Calculate Data that can be obtained from aggregate data
def create_ratios(data):
    data["on_base_percentage"] = (data["hits"] + data["walks"] + data["hit_by_pitch"]) / data["plate_appearances"]
    
    data["slugging_percentage"] = (data["singles"] +data["doubles"] * 2 + data["triples"] * 3 +data["home_runs"] * 4) / data["at_bats"]
    # Calculate additional statistics
    data["OPS"] =data["on_base_percentage"] +data["slugging_percentage"]
    
    data["BABIP"] = (data["hits"] -data["home_runs"]) / (data["at_bats"] -data["strikeout"] -data["home_runs"] +data["sacrifice_flies"]).fillna(0)
    data["batting_average"] = data["hits"]/data["at_bats"]
    data["rbi"] = (data["rbi_after"] - data["rbi_before"]) 
    data["barrel_ratio"] = (data["barrels"]/data["at_bats"]).fillna(0)
    data["good_contact_ratio"] = (data["good_contact"]/data["at_bats"]).fillna(0)
    data["wOBA"] = (data["wOBA_num"]/data["wOBA_denom"]).fillna(0)
    data["K%"] = data["strikeout"]/data["plate_appearances"]
    data["BB%"] = data["walks"]/data["plate_appearances"]
    data["BB/K"] = data["walks"]/data["strikeout"]
    data = data.drop(["rbi_before","rbi_after"], axis = 1)
    data = data.drop(["wOBA_num", "wOBA_denom"], axis = 1)


batter_stats = create_initial_data(savant_data).fillna(0)
all_years = [2021, 2022, 2023]
all_batters = batter_stats["batter"].unique()
all_combinations = pd.MultiIndex.from_product([all_batters, all_years], names=["batter", "game_year"])
full_df = pd.DataFrame(index=all_combinations).reset_index()
batter_stats = full_df.merge(batter_stats, on=["batter", "game_year"], how="left").fillna(0)


batter_stats2122 = create_aggregate_data(batter_stats[batter_stats["game_year"] < 2023]) #Training data for model
batter_stats2223 = create_aggregate_data(batter_stats[batter_stats["game_year"] > 2021]) #Input data for prediction
batter_stats21 = create_aggregate_data(batter_stats[batter_stats["game_year"]== 2021]) #For temporary purposes
batter_stats22 = create_aggregate_data(batter_stats[batter_stats["game_year"]== 2022]) #For temporary purposes
batter_stats23 = create_aggregate_data(batter_stats[batter_stats["game_year"]== 2023]) #Testing data for model
create_ratios(batter_stats2122)
create_ratios(batter_stats2223)
create_ratios(batter_stats21)
create_ratios(batter_stats22)
create_ratios(batter_stats23)
batter_stats2122 = batter_stats2122.drop(["wOBA_num","wOBA_denom"],axis=1)
batter_stats2223 = batter_stats2223.drop(["wOBA_num","wOBA_denom"],axis=1)
batter_stats21 = batter_stats21.drop(["wOBA_num","wOBA_denom"],axis=1)
batter_stats22 = batter_stats22.drop(["wOBA_num","wOBA_denom"],axis=1)
batter_stats23 = batter_stats23.drop(["wOBA_num","wOBA_denom"],axis=1)


lahman_data.rename(columns={'player_mlb_id': 'batter'}, inplace=True)
batter_stats23 = batter_stats23.merge(lahman_data, on = "batter", how="inner")
batter_stats23["age"] = 2023 - batter_stats23["birthYear"]
batter_stats22 = batter_stats22.merge(lahman_data, on = "batter", how="inner")
batter_stats22["age"] = 2023 - batter_stats23["birthYear"]
batter_stats2122 = batter_stats2122.merge(lahman_data, on = "batter", how="inner")
batter_stats2122["age"] = 2023 - batter_stats2122["birthYear"]
batter_stats2223 = batter_stats2223.merge(lahman_data, on = "batter", how="inner")
batter_stats2223["age"] = 2024 - batter_stats2223["birthYear"]


np.random.seed(2025)
batter_stats2122 = batter_stats2122.fillna(0) #NA --> 0 for sanity
batter_stats2122.replace([np.inf, -np.inf], 0, inplace=True) #Replacing possible infinities that were uncaught with 0
X = batter_stats2122.drop(["batter","playerID_LAHMAN", "birthYear","birthMonth","birthDay","birthCountry",
                           "bats","throws","debut","birthDate","rbi_before","rbi_after","weight","height"],axis=1) #Dropping unnecessary columns + non-floats
Y = batter_stats23["plate_appearances"]
xtrain,xtest,ytrain,ytest = train_test_split(X, Y, test_size=0.3, shuffle = True)
clf = ElasticNet(max_iter=1000000)
clf1 = clf.fit(xtrain,ytrain)
print("Score: ",clf1.score(xtest,ytest))


batter_stats2122 = batter_stats2122.fillna(0) #NA --> 0 for sanity
batter_stats2122.replace([np.inf, -np.inf], 0, inplace=True) #Replacing possible infinities that were uncaught with 0
X = batter_stats2122.drop(["batter","playerID_LAHMAN", "birthYear","birthMonth","birthDay","birthCountry",
                           "bats","throws","debut","birthDate","rbi_before","rbi_after","weight","height"],axis=1) #Dropping unnecessary columns + non-floats
Y = batter_stats23["plate_appearances"]
model_batter = lgb.LGBMRegressor(n_estimators=80, learning_rate=0.05)
model_batter.fit(xtrain, ytrain)
ypred = model_batter.predict(xtest)
print("RMSE:", np.sqrt(mean_squared_error(ytest, ypred)))


#Each row representing a season of a player
def create_initial_data(data):
    batter_stats = data.groupby(["pitcher","game_year"]).agg(
        plate_appearances=("events", lambda x: x.isin(plate_appearance_events).sum()),
        at_bats=("events", lambda x: x.isin(at_bat_events).sum()),
        hits=("events", lambda x: x.isin(["single", "double", "triple", "home_run"]).sum()),
        singles=("events", lambda x: (x == "single").sum()),
        doubles=("events", lambda x: (x == "double").sum()),
        triples=("events", lambda x: (x == "triple").sum()),
        home_runs=("events", lambda x: (x == "home_run").sum()),
        walks=("events", lambda x: (x == "walk").sum()),
        hit_by_pitch=("events", lambda x: (x == "hit_by_pitch").sum()),
        sacrifice_flies=("events", lambda x: (x == "sac_fly").sum()),
        strikeout = ("events", lambda x: (x.isin(["strikeout", "strikeout_double_play"]).sum())),
        rbi_before = ("bat_score", "sum"),
        rbi_after = ("post_bat_score", "sum"),
        run_exp = ("delta_run_exp", "sum"),
        barrels = ("launch_speed_angle", lambda x: (x == 6).sum()),
        good_contact = ("launch_speed_angle", lambda x: (x >= 4).sum()),
        wOBA_num = ("woba_value", "sum"),
        wOBA_denom = ("woba_denom", "sum"),
        catcher_interf = ("events", lambda x: (x.isin(["catcher_interf"]).sum())),
        innings = ("events", lambda x: ((x.isin(["caught_stealing_2b","caught_stealing_3b","caught_stealing_home",
                                                    "field_out","fielders_choice_out","force_out","other_out",
                                                    "pickoff_caught_stealing_2b","pickoff_caught_stealing_3b",
                                                    "pickoff_caught_stealing_home","strikeout"]).sum())
                                          + (x.isin(["double_play","grounded_into_double_play","strikeout_double_play",
                                                     "sac_bunt_double_play","sac_fly_double_play"]).sum()) * 2
                                          + (x.isin(["catcher_interf"]).sum())*3)/3),
        pitch_speed_num = ("release_speed","sum"),
        pitches = ("pitch_type","count")
    ).reset_index()
    return batter_stats

#Combining years for all players
def create_aggregate_data(data):
    batter_stats = data.groupby("pitcher", as_index=False).sum()
    batter_stats = batter_stats.drop("game_year", axis=1)
    return batter_stats

# Calculate Data that can be obtained from aggregate data
def create_ratios(data):
    data["on_base_percentage"] = (data["hits"] + data["walks"] + data["hit_by_pitch"]) / data["plate_appearances"]
    data["slugging_percentage"] = (data["singles"] +data["doubles"] * 2 + data["triples"] * 3 +data["home_runs"] * 4) / data["at_bats"]
    # Calculate additional statistics
    data["OPS"] =data["on_base_percentage"] +data["slugging_percentage"]
    
    data["BABIP"] = (data["hits"] -data["home_runs"]) / (data["at_bats"] -data["strikeout"] -data["home_runs"] +data["sacrifice_flies"]).fillna(0)
    data["batting_average"] = data["hits"]/data["at_bats"]
    data["rbi"] = (data["rbi_after"] - data["rbi_before"]) 
    data["barrel_ratio"] = (data["barrels"]/data["at_bats"]).fillna(0)
    data["good_contact_ratio"] = (data["good_contact"]/data["at_bats"]).fillna(0)
    data["wOBA"] = (data["wOBA_num"]/data["wOBA_denom"]).fillna(0)
    data["K%"] = data["strikeout"]/data["plate_appearances"]
    data["BB%"] = data["walks"]/data["plate_appearances"]
    data["K/BB"] = data["strikeout"]/data["walks"]
    data["WHIP"] = (data["hits"] + data["walks"])/data["innings"]
    data["release_speed"] = data["pitch_speed_num"]/data["pitches"]
    data = data.drop(["rbi_before","rbi_after"], axis = 1)
    data = data.drop(["wOBA_num", "wOBA_denom","pitch_speed_num"], axis = 1)


pitcher_stats = create_initial_data(savant_data).fillna(0)
all_years = [2021, 2022, 2023]
all_pitchers = pitcher_stats["pitcher"].unique()
all_combinations = pd.MultiIndex.from_product([all_pitchers, all_years], names=["pitcher", "game_year"])
full_df = pd.DataFrame(index=all_combinations).reset_index()
pitcher_stats = full_df.merge(pitcher_stats, on=["pitcher", "game_year"], how="left").fillna(0)
pitcher_stats


pitcher_stats2122 = create_aggregate_data(pitcher_stats[pitcher_stats["game_year"] < 2023]) #Training data for model
pitcher_stats2223 = create_aggregate_data(pitcher_stats[pitcher_stats["game_year"] > 2021]) #Input data for prediction
pitcher_stats21 = create_aggregate_data(pitcher_stats[pitcher_stats["game_year"]== 2021]) #For temporary purposes
pitcher_stats22 = create_aggregate_data(pitcher_stats[pitcher_stats["game_year"]== 2022]) #For temporary purposes
pitcher_stats23 = create_aggregate_data(pitcher_stats[pitcher_stats["game_year"]== 2023]) #Testing data for model
create_ratios(pitcher_stats2122)
create_ratios(pitcher_stats2223)
create_ratios(pitcher_stats21)
create_ratios(pitcher_stats22)
create_ratios(pitcher_stats23)
pitcher_stats2122 = pitcher_stats2122.drop(["wOBA_num","wOBA_denom"],axis=1)
pitcher_stats2223 = pitcher_stats2223.drop(["wOBA_num","wOBA_denom"],axis=1)
pitcher_stats21 = pitcher_stats21.drop(["wOBA_num","wOBA_denom"],axis=1)
pitcher_stats22 = pitcher_stats22.drop(["wOBA_num","wOBA_denom"],axis=1)
pitcher_stats23 = pitcher_stats23.drop(["wOBA_num","wOBA_denom"],axis=1)


#Changed from ID --> Batter. Need to change from Batter -> Pitcher
lahman_data.rename(columns={'batter': 'pitcher'}, inplace=True) 
pitcher_stats23 = pitcher_stats23.merge(lahman_data, on = "pitcher", how="inner")
pitcher_stats23["age"] = 2023 - pitcher_stats23["birthYear"]
pitcher_stats22 = pitcher_stats22.merge(lahman_data, on = "pitcher", how="inner")
pitcher_stats22["age"] = 2023 - pitcher_stats23["birthYear"]
pitcher_stats2122 = pitcher_stats2122.merge(lahman_data, on = "pitcher", how="inner")
pitcher_stats2122["age"] = 2023 - pitcher_stats2122["birthYear"]
pitcher_stats2223 = pitcher_stats2223.merge(lahman_data, on = "pitcher", how="inner")
pitcher_stats2223["age"] = 2024 - pitcher_stats2223["birthYear"]


np.random.seed(2025)
pitcher_stats2122 = pitcher_stats2122.fillna(0) #NA --> 0 for sanity
pitcher_stats2122.replace([np.inf, -np.inf], 0, inplace=True) #Replacing possible infinities that were uncaught with 0
X = pitcher_stats2122.drop(["pitcher","playerID_LAHMAN", "birthYear","birthMonth","birthDay","birthCountry",
                           "bats","throws","debut","birthDate","rbi_before","rbi_after","weight","height"],axis=1) #Dropping unnecessary columns + non-floats
Y = pitcher_stats23["plate_appearances"]
xtrain,xtest,ytrain,ytest = train_test_split(X, Y, test_size=0.3, shuffle = True)
#Appropriate max_iter is around 1000000000, but for sake of running the program, we will decrease this
clf = ElasticNet(max_iter=10000)
clf1 = clf.fit(xtrain,ytrain)
print("Score: ",clf1.score(xtest,ytest))


# Train LightGBM model
pitcher_stats2122 = pitcher_stats2122.fillna(0) #NA --> 0 for sanity
pitcher_stats2122.replace([np.inf, -np.inf], 0, inplace=True) #Replacing possible infinities that were uncaught with 0
X = pitcher_stats2122.drop(["pitcher","playerID_LAHMAN", "birthYear","birthMonth","birthDay","birthCountry",
                           "bats","throws","debut","birthDate","rbi_before","rbi_after","weight","height"],axis=1) #Dropping unnecessary columns + non-floats
Y = pitcher_stats23["plate_appearances"]
xtrain,xtest,ytrain,ytest = train_test_split(X, Y, test_size=0.3, shuffle = True)
model_pitcher = lgb.LGBMRegressor(n_estimators=80, learning_rate=0.05)
model_pitcher.fit(xtrain, ytrain)

# Predict and evaluate
ypred = model_pitcher.predict(xtest)
print("RMSE:", np.sqrt(mean_squared_error(ytest, ypred)))


batter_stats2223 = batter_stats2223.fillna(0) #NA --> 0 for sanity
batter_stats2223.replace([np.inf, -np.inf], 0, inplace=True) #Replacing possible infinities that were uncaught with 0
X = batter_stats2223.drop(["batter","playerID_LAHMAN", "birthYear","birthMonth","birthDay","birthCountry",
                           "bats","throws","debut","birthDate","rbi_before","rbi_after","weight","height"],axis=1) #Dropping unnecessary columns + non-floats
predictions = model_batter.predict(X)
for i in range(len(predictions)):
    if predictions[i] < 0:
        predictions[i] = 0
    predictions[i] = int(predictions[i])
batter_predictions = pd.DataFrame()
batter_predictions["PLAYER_ID"] = batter_stats2223["batter"]
batter_predictions["PLAYING_TIME"] = predictions


pitcher_stats2223 = pitcher_stats2223.fillna(0) #NA --> 0 for sanity
pitcher_stats2223.replace([np.inf, -np.inf], 0, inplace=True) #Replacing possible infinities that were uncaught with 0
X = pitcher_stats2223.drop(["pitcher","playerID_LAHMAN", "birthYear","birthMonth","birthDay","birthCountry",
                           "bats","throws","debut","birthDate","rbi_before","rbi_after","weight","height"],axis=1) #Dropping unnecessary columns + non-floats
predictions = model_pitcher.predict(X)
for i in range(len(predictions)):
    if predictions[i] < 0:
        predictions[i] = 0
    predictions[i] = int(predictions[i])
pitcher_predictions = pd.DataFrame()
pitcher_predictions["PLAYER_ID"] = pitcher_stats2223["pitcher"]
pitcher_predictions["PLAYING_TIME"] = predictions


solution = pd.concat([batter_predictions,pitcher_predictions]).reset_index(drop=True)
solution = solution.groupby("PLAYER_ID", as_index=False)["PLAYING_TIME"].sum().reset_index(drop=True)
for i in range(len(solution)):
    if solution.loc[i]["PLAYER_ID"] not in sample_submission["PLAYER_ID"].values:
        solution = solution.drop(i)
solution = solution.reset_index(drop=True)


solution.to_csv("submission.csv",index=False)

