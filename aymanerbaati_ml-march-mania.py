# Libraries to import 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import seaborn as sns
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
import math 
from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score, StratifiedKFold



section_files = ["Teams.csv", "Seasons.csv", "NCAATourneySeeds.csv",
                 "RegularSeasonCompactResults.csv", "NCAATourneyCompactResults.csv"]
genders = ["M", "W"]
sub_file_name = "SampleSubmissionStage1.csv"
folder_path = "/kaggle/input/march-machine-learning-mania-2025"


def load_first_section(gender):
    dataframes = {}
    for file in section_files:
        filename = gender + file
        name = os.path.splitext(filename)[0] #Gets the name of dataframe 
        dataframes[name] = pd.read_csv(os.path.join(folder_path, filename))
    return dataframes


first_section = load_first_section("M") | load_first_section("W")


first_sec_sub = pd.read_csv(os.path.join(folder_path, sub_file_name))
first_sec_sub.head()


for name in first_section:
    print(name)
    display(first_section[name].head())


SEASON_TO_EXPLORE = 2011 
#Regular season data for 2010 - 2011 season
reg_season = pd.concat([first_section["MRegularSeasonCompactResults"] ,
                         first_section["WRegularSeasonCompactResults"]], axis = 0)
#reg_season = reg_season[reg_season["Season"] == SEASON_TO_EXPLORE]
#NCAA tourney data for 2010 - 2011 season
tourney_res = pd.concat([first_section["MNCAATourneyCompactResults"],
                         first_section["WNCAATourneyCompactResults"]], axis = 0)
#tourney_res = tourney_res[tourney_res["Season"] == SEASON_TO_EXPLORE]
#TEAM SEEDS FOR NCAA TOURNEY
seeds =  pd.concat([first_section["MNCAATourneySeeds"],
                         first_section["WNCAATourneySeeds"]], axis = 0)
#seeds = seeds[seeds["Season"] == SEASON_TO_EXPLORE]


#Some feature engineering to get the seeds of the teams
df = pd.merge(tourney_res, seeds, left_on = ["Season","WTeamID"]
              , right_on = ["Season","TeamID"])
df["WSeed"] = df["Seed"]
df = df.drop(["TeamID", "Seed"], axis = 1)
df = pd.merge(df, seeds, left_on = ["Season","LTeamID"]
              , right_on = ["Season","TeamID"])
df["LSeed"] = df["Seed"]
df = df.drop(["TeamID", "Seed"], axis = 1)
df["LRegion"] = df["LSeed"].str[0]
df["WRegion"] = df["WSeed"].str[0]
df["WSeed"] = df["WSeed"].str[1:3].astype(int)
df["LSeed"] = df["LSeed"].str[1:3].astype(int)
df["SeedDiff"] = df["WSeed"] - df["LSeed"]


df.head()


better_seed_wins = len(df[df["SeedDiff"] < 0])
same_seed = len(df[df["SeedDiff"] == 0])
other_cases = len(df) - (better_seed_wins + same_seed)
labels = ['Winner had smaller Seed', "Same seed", 'Other Cases']
sizes = [better_seed_wins, same_seed, other_cases]
colors = ['blue', 'gray', "red"]

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, 
        startangle=140, wedgeprops={'edgecolor': 'black'})

# Title
plt.title('Proportion of Games Where Winner Had a Better Seed')

# Show plot
plt.show()


seed_diff_ev = df.groupby("Season")["SeedDiff"].mean()
seed_diff_ev.plot(kind='line', marker='o', figsize=(8, 5), color='blue')
plt.xlabel('Tourney Season')
plt.ylabel('Average Seed Difference')
plt.title('Average Seed Difference Over Seasons')
plt.grid(True)
plt.show()


def fixID(df):
    rows = []
    for index, row in df[df["WTeamID"] < df["LTeamID"]].iterrows():
        rows.append({
            "Season" : row["Season"],
            "Team1" : row["WTeamID"],
            "Team2" : row["LTeamID"],
            "Team1Region" : row["WRegion"],
            "Team2Region" : row["LRegion"],
            "Seed1" : row["WSeed"],
            "Seed2" : row["LSeed"], 
            "SeedDiff" : row["SeedDiff"],
            "Result" : 1
        })
    for index, row in df[df["WTeamID"] > df["LTeamID"]].iterrows():
        rows.append({
            "Season" : row["Season"],
            "Team1" : row["LTeamID"],
            "Team2" : row["WTeamID"],
            "Team1Region" : row["LRegion"],
            "Team2Region" : row["WRegion"],
            "Seed1" : row["LSeed"],
            "Seed2" : row["WSeed"], 
            "SeedDiff" : -row["SeedDiff"],
            "Result" : 0
        })

    return pd.DataFrame(rows)


#Let's prepare our data for training the model 
features_to_keep = ["Season", "WTeamID", "LTeamID", "WRegion", "LRegion",
                     "WSeed", "LSeed", "SeedDiff"]
data = df[features_to_keep]
data.head()
data = fixID(data)
data = data.sort_values(by='Season', ascending=True)
#Sanity check 
assert(data[data["Team1"] < data["Team2"]].equals(data))


data.head()


cut_off_date = 2020
train, test = data[data["Season"] < cut_off_date], data[data["Season"] >= cut_off_date]
X_train, y_train = train.drop("Result", axis = 1), train["Result"]
X_test, y_test = test.drop("Result", axis = 1), test["Result"]


#Check for Class imbalances 
print(f"training data classes : {y_train.value_counts()}")
print(f"testing data classes{y_test.value_counts()}")


#Features to drop for logistic regression 
to_drop = ["Season","Team1", "Team2", "Team1Region", "Team2Region"]
X_train, X_test = X_train.drop(to_drop, axis = 1), X_test.drop(to_drop, axis = 1)


#Evaluation benchmark 
random_pred = [0.5] * y_test.shape[0]
print(f"The brier score for the random predictor {brier_score_loss(y_test, random_pred)}")


logr = LogisticRegression(random_state = 0).fit(X_train, y_train)
y_pred = logr.predict(X_train)
proba_pred = logr.predict_proba(X_train)[:,1] #Get the probability for the positive class only
y_val = logr.predict(X_test)
proba_val = logr.predict_proba(X_test)[:,1]


print(f"The accuracy score for the training set {accuracy_score(y_pred, y_train)}")
print(f"The accuracy score for the testing set {accuracy_score(y_val, y_test)}")
print(f"The brier score for the training set {brier_score_loss(y_train, proba_pred)}")
print(f"The brier score for the testing set {brier_score_loss(y_test, proba_val)}")


teams = pd.concat([first_section["MTeams"], first_section["WTeams"]], axis = 0)


teams.head()


detailed_season = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv")
detailed_season.head()


def fixSeasonID(df):
    rows = []
    for index, row in df[df["WTeamID"] < df["LTeamID"]].iterrows():
        rows.append({
            "Season" : row["Season"],
            "DayNum" : row["DayNum"],
            "Team1" : row["WTeamID"],
            "Team2" : row["LTeamID"],
            "Team1Score" : row["WScore"],
            "Team2Score" : row["LScore"],
            "Result" : 1
        })
    for index, row in df[df["WTeamID"] > df["LTeamID"]].iterrows():
        rows.append({
            "Season" : row["Season"],
            "DayNum" : row["DayNum"],
            "Team1" : row["LTeamID"],
            "Team2" : row["WTeamID"],
            "Team1Score" : row["LScore"],
            "Team2Score" : row["WScore"],
            "Result" : 0
        })

    return pd.DataFrame(rows)


reg_season_fixed = fixSeasonID(reg_season)


# Get head to head statistics
h2h_tracker = {}
reg_season_fixed = reg_season_fixed.sort_values(by=['Season', 'DayNum']).reset_index(drop=True)
#innit
reg_season_fixed["H2HWins"] = 0
reg_season_fixed["H2HLosses"] = 0

for index, row in reg_season_fixed.iterrows():
    #get the h2h for this match up 
    team1, team2 = row["Team1"], row["Team2"]
    past_record = h2h_tracker.get((team1, team2), [0, 0])
    #store it in the df
    reg_season_fixed.at[index, 'H2HWins']  = past_record[0]
    reg_season_fixed.at[index, 'H2HLosses']  = past_record[1]
    #Update the past record
    if row["Result"]:
        past_record[0] += 1
    else:
        past_record[1] += 1
    #update tracker 
    h2h_tracker[(team1, team2)] = past_record
reg_season_fixed.head()


reg_season_fixed.tail()


h2h = reg_season_fixed.groupby(["Season", "Team1", "Team2"])[["H2HWins", "H2HLosses"]].last().reset_index()


data = pd.merge(data, h2h, on = ["Season", "Team1", "Team2"], how = "left").fillna(0)


def prepare_model_data(cut_off_date, data):
    train, test = data[data["Season"] < cut_off_date], data[data["Season"] >= cut_off_date]
    X_train, y_train = train.drop("Result", axis = 1), train["Result"]
    X_test, y_test = test.drop("Result", axis = 1), test["Result"]
    to_drop = ["Season","Team1", "Team2", "Team1Region", "Team2Region", "Seed1", "Seed2", "SeedDiff"]
    X_train, X_test = X_train.drop(to_drop, axis = 1), X_test.drop(to_drop, axis = 1)
    return X_train, X_test, y_train, y_test
    
X_train, X_test, y_train, y_test = prepare_model_data(2023, data)


logr = LogisticRegression(random_state = 0).fit(X_train, y_train)
y_pred = logr.predict(X_train)
proba_pred = logr.predict_proba(X_train)[:,1] #Get the probability for the positive class only
y_val = logr.predict(X_test)
proba_val = logr.predict_proba(X_test)[:,1]

print(f"The accuracy score for the training set {accuracy_score(y_pred, y_train)}")
print(f"The accuracy score for the testing set {accuracy_score(y_val, y_test)}")
print(f"The brier score for the training set {brier_score_loss(y_train, proba_pred)}")
print(f"The brier score for the testing set {brier_score_loss(y_test, proba_val)}")


rf = RandomForestClassifier(n_estimators=190).fit(X_train, y_train)
y_pred = rf.predict(X_train)
proba_pred = rf.predict_proba(X_train)[:,1] #Get the probability for the positive class only
y_val = rf.predict(X_test)
proba_val = rf.predict_proba(X_test)[:,1]

print(f"The accuracy score for the training set {accuracy_score(y_pred, y_train)}")
print(f"The accuracy score for the testing set {accuracy_score(y_val, y_test)}")
print(f"The brier score for the training set {brier_score_loss(y_train, proba_pred)}")
print(f"The brier score for the testing set {brier_score_loss(y_test, proba_val)}")


#For each team get won, loss, scored, scored_against

#Get wins
def get_season_wins_team1(df) : 
    return df.groupby(["Season", "Team1"])["Result"].sum().reset_index()
def get_season_wins_team2(df) : 
    return df[df["Result"] == 0].groupby(["Season", "Team2"]).size().reset_index() 
def get_season_wins(df):
    df1 = get_season_wins_team1(df)
    df2 = get_season_wins_team2(df).rename(columns = {0 : "Result2"})
    wins = df1.merge(df2, left_on = ["Season", "Team1"], right_on = ["Season", "Team2"], how = "outer")
    wins[["Result", "Result2"]] = wins[["Result", "Result2"]].fillna(0)
    wins["wins"] = wins["Result"] + wins["Result2"]
    wins["Team1"] = wins["Team1"].fillna(wins["Team2"])
    return wins.drop(["Result", "Result2", "Team2"], axis = 1)

#get loss 
def get_season_loss_team1(df) : 
    return df[df["Result"] == 0].groupby(["Season", "Team1"]).size().reset_index()
    
def get_season_loss_team2(df) : 
    return df.groupby(["Season", "Team2"])["Result"].sum().reset_index() 
def get_season_loss(df):
    df1 = get_season_loss_team1(df).rename(columns = {0 : "Result2"})
    df2 = get_season_loss_team2(df)
    loss = df1.merge(df2, left_on = ["Season", "Team1"], right_on = ["Season", "Team2"], how = "outer")
    loss[["Result", "Result2"]] = loss[["Result", "Result2"]].fillna(0)
    loss["losses"] = loss["Result"] + loss["Result2"]
    loss["Team1"] = loss["Team1"].fillna(loss["Team2"])
    return loss.drop(["Result", "Result2", "Team2"], axis = 1)

#get goals 
def get_team1_goals(df):
    return df.groupby(["Season", "Team1"])["Team1Score"].sum().reset_index()
def get_team2_goals(df):
    return df.groupby(["Season", "Team2"])["Team2Score"].sum().reset_index()
def get_goals(df):
    df1 = get_team1_goals(df)
    df2 = get_team2_goals(df)
    goals = df1.merge(df2, left_on = ["Season", "Team1"], right_on = ["Season", "Team2"], how = "outer")
    goals[["Team1Score", "Team2Score"]] = goals[["Team1Score", "Team2Score"]].fillna(0)
    goals["Goals"] = goals["Team1Score"] + goals["Team2Score"]
    goals["Team1"] = goals["Team1"].fillna(goals["Team2"])
    return goals.drop(["Team2", "Team1Score", "Team2Score"], axis=1)
    
#get goals against
def get_team1_goals_against(df):
    return df.groupby(["Season", "Team1"])["Team2Score"].sum().reset_index()
def get_team2_goals_against(df):
    return df.groupby(["Season", "Team2"])["Team1Score"].sum().reset_index()
def get_goals_against(df):
    df1 = get_team1_goals_against(df)
    df2 = get_team2_goals_against(df)
    goals = df1.merge(df2, left_on = ["Season", "Team1"], right_on = ["Season", "Team2"], how = "outer")
    goals[["Team1Score", "Team2Score"]] = goals[["Team1Score", "Team2Score"]].fillna(0)
    goals["GoalsAgainst"] = goals["Team1Score"] + goals["Team2Score"]
    goals["Team1"] = goals["Team1"].fillna(goals["Team2"])
    return goals.drop(["Team2", "Team1Score", "Team2Score"], axis=1)
#GET ELO

def get_features(df):
    wins = get_season_wins(df)
    losses = wins.merge(get_season_loss(df), on=["Season", "Team1"], how="outer")
    goals = losses.merge(get_goals(reg_season_fixed), on=["Season", "Team1"], how="outer")
    to_return = goals.merge(get_goals_against(reg_season_fixed), on=["Season", "Team1"], how="outer")
    return to_return.fillna(0)


def get_final_df(df, data):
    final_df = get_features(df)
    team1 = final_df.merge(data, on=["Season", "Team1"], how = "right").rename(columns = {"wins" : "wins1", "losses" : "losses1", 
                                                                                     "Goals" : "goals1", "GoalsAgainst" : "GA1"})
    return team1.merge(final_df.rename(columns = {"Team1" : "Team2","wins" : "wins2", "losses" : "losses2", 
                            "Goals" : "goals2", "GoalsAgainst" : "GA2"}), on=["Season", "Team2"])


mydata = get_final_df(reg_season_fixed, data)
mydata.head()


reg_season_fixed.head()


mydata.tail()


X_train, X_test, y_train, y_test = prepare_model_data(2023, mydata)
rf = RandomForestClassifier(n_estimators=1000).fit(X_train, y_train)
y_pred = rf.predict(X_train)
proba_pred = rf.predict_proba(X_train)[:,1] #Get the probability for the positive class only
y_val = rf.predict(X_test)
proba_val = rf.predict_proba(X_test)[:,1]

print(f"The accuracy score for the training set {accuracy_score(y_pred, y_train)}")
print(f"The accuracy score for the testing set {accuracy_score(y_val, y_test)}")
print(f"The brier score for the training set {brier_score_loss(y_train, proba_pred)}")
print(f"The brier score for the testing set {brier_score_loss(y_test, proba_val)}")


logr = LogisticRegression(random_state = 0, max_iter = 1000).fit(X_train, y_train)
y_pred = logr.predict(X_train)
proba_pred = logr.predict_proba(X_train)[:,1] #Get the probability for the positive class only
y_val = logr.predict(X_test)
proba_val = logr.predict_proba(X_test)[:,1]

print(f"The accuracy score for the training set {accuracy_score(y_pred, y_train)}")
print(f"The accuracy score for the testing set {accuracy_score(y_val, y_test)}")
print(f"The brier score for the training set {brier_score_loss(y_train, proba_pred)}")
print(f"The brier score for the testing set {brier_score_loss(y_test, proba_val)}")


def get_elo(reg_season_fixed, k = 40):
    elo_tracker = {}
    reg_season_fixed = reg_season_fixed.sort_values(by=['Season', 'DayNum']).reset_index(drop=True)
    #innit
    reg_season_fixed["Team1Elo"] = 1500
    reg_season_fixed["Team2Elo"] = 1500
    for index, row in reg_season_fixed.iterrows():
        #get the h2h for this match up 
        team1, team2 = row["Team1"], row["Team2"]
        elo1, elo2 = elo_tracker.get(team1, 1500), elo_tracker.get(team2, 1500)
        #store it in the df
        reg_season_fixed.at[index, "Team1Elo"]  = elo1
        reg_season_fixed.at[index, "Team2Elo"]  = elo2
        #elo_diff = max(min(elo2 - elo1, 1000), -1000)
        elo_diff = elo2 - elo1
        expected_win = 1 / (1 + 10 ** (elo_diff / 400))
        expected_loss = 1 - expected_win
        margin = abs(row["Team1Score"] - row["Team2Score"])
        div_fact = abs(elo_diff) if elo_diff != 0 else 1
        mov_factor = math.log(1 + margin) * (2.2 / div_fact)
        adj_k = k * mov_factor
        #Update the past record
        score = row["Result"]
        elo1 += int(adj_k * (score - expected_win))
        elo2 += int(adj_k * (1 - score - expected_loss))
    #update tracker 
        elo_tracker[team1] = elo1
        elo_tracker[team2] = elo2
    return reg_season_fixed


elo_df_40[elo_df_40["Team1Elo"] > elo_df_40["Team2Elo"]]["Result"].value_counts()


elo_df_40[elo_df_40["Team2Elo"] > elo_df_40["Team1Elo"]]["Result"].value_counts()


elo_df1 = elo_df_40.groupby(["Season", "Team1"])["Team1Elo"].last().reset_index()
elo_df2 = elo_df_40.groupby(["Season", "Team2"])["Team2Elo"].last().reset_index().rename(
    columns={"Team2" : "Team1", "Team2Elo" : "Team1Elo"})
elo_df = pd.concat([elo_df1, elo_df2], axis = 0)
elo_df = elo_df.groupby(["Season", "Team1"]).last().reset_index()
elo_df.describe()


df = mydata.merge(elo_df, on=["Season", "Team1"])
mydata = df.merge(elo_df.rename(columns={"Team1":"Team2", "Team1Elo" : "Team2Elo"}), 
                                                    on = ["Season", "Team2"])


X_train, X_test, y_train, y_test = prepare_model_data(2023, mydata)
rf = RandomForestClassifier(n_estimators=5000, max_depth = 5).fit(X_train, y_train)
y_pred = rf.predict(X_train)
proba_pred = rf.predict_proba(X_train)[:,1] #Get the probability for the positive class only
y_val = rf.predict(X_test)
proba_val = rf.predict_proba(X_test)[:,1]

print(f"The accuracy score for the training set {accuracy_score(y_pred, y_train)}")
print(f"The accuracy score for the testing set {accuracy_score(y_val, y_test)}")
print(f"The brier score for the training set {brier_score_loss(y_train, proba_pred)}")
print(f"The brier score for the testing set {brier_score_loss(y_test, proba_val)}")

logr = LogisticRegression(random_state = 0, max_iter = 1000).fit(X_train, y_train)
y_pred = logr.predict(X_train)
proba_pred = logr.predict_proba(X_train)[:,1] #Get the probability for the positive class only
y_val = logr.predict(X_test)
proba_val = logr.predict_proba(X_test)[:,1]

print(f"The accuracy score for the training set {accuracy_score(y_pred, y_train)}")
print(f"The accuracy score for the testing set {accuracy_score(y_val, y_test)}")
print(f"The brier score for the training set {brier_score_loss(y_train, proba_pred)}")
print(f"The brier score for the testing set {brier_score_loss(y_test, proba_val)}")

xgb = XGBClassifier(n_estimators=500, max_depth=2, reg_alpha=1.0,
         reg_lambda=5.0,learning_rate=1, objective='binary:logistic')
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_train)
proba_pred = xgb.predict_proba(X_train)[:,1] #Get the probability for the positive class only
y_val = xgb.predict(X_test)
proba_val = xgb.predict_proba(X_test)[:,1]

print(f"The accuracy score for the training set {accuracy_score(y_pred, y_train)}")
print(f"The accuracy score for the testing set {accuracy_score(y_val, y_test)}")
print(f"The brier score for the training set {brier_score_loss(y_train, proba_pred)}")
print(f"The brier score for the testing set {brier_score_loss(y_test, proba_val)}")


def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    proba_pred = model.predict_proba(X_train)[:,1] #Get the probability for the positive class only
    y_val = model.predict(X_test)
    proba_val = model.predict_proba(X_test)[:,1]
    print(f"\nğŸ“Š Evaluating {model.__class__.__name__}")
    print(f"The accuracy score for the training set {accuracy_score(y_pred, y_train)}")
    print(f"The accuracy score for the testing set {accuracy_score(y_val, y_test)}")
    print(f"The brier score for the training set {brier_score_loss(y_train, proba_pred)}")
    print(f"The brier score for the testing set {brier_score_loss(y_test, proba_val)}")
    return model 



rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42)
logr = LogisticRegression(max_iter=1000, random_state=42)
xgb = XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.1, reg_lambda=5, objective="binary:logistic")

# ğŸ“Œ Voting Classifier (Hard Voting)
voting_clf = VotingClassifier(estimators=[
    ('rf', rf), ('logr', logr), ('xgb', xgb)
], voting='soft')  # 'soft' uses probability averaging

# ğŸ“Œ Stacking Classifier (Using Logistic Regression as Meta-Model)
stacking_clf = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    final_estimator=LogisticRegression(),
    cv=5
)
evaluate_model(rf)
evaluate_model(logr)
evaluate_model(xgb)
evaluate_model(voting_clf)  # Voting Ensemble
evaluate_model(stacking_clf)  # Stacking Ensemble


sub_file = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv")
sub_file.to_csv("random.csv", index = False)


df = sub_file.copy()
df["Season"] = (df["ID"].str[0:4]).astype("int")
df["Team1"] = df["ID"].str[5:9].astype("int")
df["Team2"] = df["ID"].str[10:14].astype("int")
df = df.drop(["ID", "Pred"], axis = 1)
df = df.merge(h2h, on=["Season", "Team1", "Team2"], how="left").fillna(0)
df



features = get_final_df(reg_season_fixed, df)
df = features.merge(elo_df, on=["Season", "Team1"])
final_df = df.merge(elo_df.rename(columns={"Team1":"Team2", "Team1Elo" : "Team2Elo"}), 
                                                    on = ["Season", "Team2"])
final_df


X = final_df.drop(["Season", "Team1", "Team2"], axis = 1)


proba = voting_clf.predict_proba(X)[:, 1]
sub_file["Pred"] = proba
sub_file.to_csv("submission1.csv", index = False)


logr = evaluate_model(logr)


proba = logr.predict_proba(X)[:, 1]
sub_file["Pred"] = proba
sub_file.to_csv("submission2.csv", index = False)


proba = stacking_clf.predict_proba(X)[:, 1]
sub_file["Pred"] = proba
sub_file.to_csv("stacking.csv", index = False)


rf_param_grid = {
    "n_estimators": [500, 1000, 5000],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_grid = HalvingGridSearchCV(RandomForestClassifier(), rf_param_grid,factor=4, cv=2, 
                       scoring="neg_brier_score", n_jobs=-1, verbose=1)
rf_best = rf_grid.fit(X_train, y_train).best_estimator_
evaluate_model(rf_best)


# ğŸ“Œ 2ï¸�âƒ£ Logistic Regression with GridSearchCV
logr_param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "max_iter": [500, 1000, 2000]
}

logr_grid = HalvingGridSearchCV(LogisticRegression(solver="liblinear", random_state=0),
                         logr_param_grid, cv=2, factor=4, scoring="neg_brier_score", n_jobs=-1, verbose = 1)
logr_best = logr_grid.fit(X_train, y_train).best_estimator_
evaluate_model(logr_best)


# ğŸ“Œ 3ï¸�âƒ£ XGBoost with GridSearchCV
xgb_param_grid = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [2, 3, 5],
    "learning_rate": [0.01, 0.1, 0.5, 1],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1, 5, 10]
}

xgb_grid = HalvingGridSearchCV(XGBClassifier(objective="binary:logistic"),
                        xgb_param_grid, cv=2, factor=4, scoring="neg_brier_score", n_jobs=-1, verbose = 1)
xgb_best = xgb_grid.fit(X_train, y_train).best_estimator_
evaluate_model(xgb_best)


pd.read_csv("/kaggle/working/submission1.csv")

