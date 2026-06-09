import numpy as np 
import pandas as pd
import shap
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import make_scorer, brier_score_loss
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit

warnings.filterwarnings("ignore")

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

#Uncomment if you want to see all possible datasets available
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))



DIR_PATH = "/kaggle/input/march-machine-learning-mania-2025/"
ELO_PATH = "/kaggle/input/2025-historical-elo-ratings-ncaa-lennart-haupts/"


df_seeds = pd.concat(
    [
        pd.read_csv(DIR_PATH + "MNCAATourneySeeds.csv").assign(Bracket="M"),
        pd.read_csv(DIR_PATH + "WNCAATourneySeeds.csv").assign(Bracket="W"),
    ],
).reset_index(drop=True)

df_season_compact_results = pd.concat(
    [
        pd.read_csv(DIR_PATH + "MRegularSeasonCompactResults.csv").assign(Bracket="M"),
        pd.read_csv(DIR_PATH + "WRegularSeasonCompactResults.csv").assign(Bracket="W"),
    ]
).reset_index(drop=True)

df_season_detailed_results = pd.concat(
    [
        pd.read_csv(DIR_PATH + "MRegularSeasonDetailedResults.csv").assign(Bracket="M"),
        pd.read_csv(DIR_PATH + "WRegularSeasonDetailedResults.csv").assign(Bracket="W"),
    ]
).reset_index(drop=True)

#Dropping these seasons due to stats only being available since 2003 season (men) or since the 2010 season (women) 
df_season_detailed_results = df_season_detailed_results.query(
    "(Bracket == 'M' and Season >= 2003) or (Bracket == 'W' and Season >= 2010)"
).reset_index(drop=True)

df_tourney_results = pd.concat(
    [
        pd.read_csv(DIR_PATH + "MNCAATourneyCompactResults.csv").assign(Bracket="M"),
        pd.read_csv(DIR_PATH + "WNCAATourneyCompactResults.csv").assign(Bracket="W"),
    ]
).reset_index(drop=True)

#Dropping these seasons due to stats only being available since 2003 season (men) or since the 2010 season (women) 
df_tourney_results = df_tourney_results.query(
    "(Bracket == 'M' and Season >= 2003) or (Bracket == 'W' and Season >= 2010)"
).reset_index(drop=True)

# Elo Ratings from https://www.kaggle.com/code/lennarthaupts/calculate-elo-ratings
df_elo_ratings = pd.concat(
    [
        pd.read_csv(ELO_PATH + "final_mens_elo_rating.csv").assign(Bracket="M"),
        pd.read_csv(ELO_PATH + "final_womens_elo_rating.csv").assign(Bracket="W"),
    ]
).reset_index(drop=True)

df_elo_ratings = df_elo_ratings.drop('Unnamed: 0', axis = 1)
df_sample_submission = pd.read_csv(DIR_PATH + "SampleSubmissionStage2.csv")

df_team_names = pd.read_csv(DIR_PATH + "MTeamSpellings.csv")


df_team_season_detailed_results = pd.concat(
    [
        df_season_detailed_results[["Season", "Bracket", "WTeamID", "LTeamID", "DayNum", "WScore", "LScore", "WBlk", "WOR", "WStl", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WDR", "WAst", "WTO", "WPF", "LBlk", "LOR", "LStl", "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LDR", "LAst", "LTO", "LPF"]]
        .assign(GameResult="W")
        .rename(
            columns={"WTeamID": "TeamID","WScore": "TeamScore", "LScore": "OppScore", "WBlk": "TeamBlk", "WOR": "TeamOR", "WStl": "TeamStl","WFGM": "TeamFGM",
        "WFGA": "TeamFGA",
        "WFGM3": "TeamFGM3",
        "WFGA3": "TeamFGA3",
        "WFTM": "TeamFTM",
        "WFTA": "TeamFTA",
        "WDR": "TeamDR",
        "WAst": "TeamAst",
        "WTO": "TeamTO",
        "WPF": "TeamPF",
        "LBlk": "OBlk", 
        "LOR": "OOR",
        "LStl": "OStl",  
        "LFGM": "OFGM",
        "LFGA": "OFGA",
        "LFGM3": "OFGM3",
        "LFGA3": "OFGA3",
        "LFTM": "OFTM",
        "LFTA": "OFTA",
        "LDR": "ODR",
        "LAst": "OAst",
        "LTO": "OTO",
        "LPF": "OPF"}
        ),
        df_season_detailed_results[["Season", "Bracket", "LTeamID", "WTeamID", "DayNum", "WScore", "LScore", "LBlk", "LOR", "LStl", "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LDR", "LAst", "LTO", "LPF", "WBlk", "WOR", "WStl", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WDR", "WAst", "WTO", "WPF"]]
        .assign(GameResult="L")
        .rename(
            columns={"LTeamID": "TeamID", "LScore": "TeamScore", "WScore": "OppScore", "LBlk": "TeamBlk", "LOR": "TeamOR", "LStl": "TeamStl",  "LFGM": "TeamFGM",
        "LFGA": "TeamFGA",
        "LFGM3": "TeamFGM3",
        "LFGA3": "TeamFGA3",
        "LFTM": "TeamFTM",
        "LFTA": "TeamFTA",
        "LDR": "TeamDR",
        "LAst": "TeamAst",
        "LTO": "TeamTO",
        "LPF": "TeamPF",
        "WBlk": "OBlk", 
        "WOR": "OOR", 
        "WStl": "OStl",
        "WFGM": "OFGM",
        "WFGA": "OFGA",
        "WFGM3": "OFGM3",
        "WFGA3": "OFGA3",
        "WFTM": "OFTM",
        "WFTA": "OFTA",
        "WDR": "ODR",
        "WAst": "OAst",
        "WTO": "OTO",
        "WPF": "OPF"}
        ),
    ]
).reset_index(drop=True)


#Score Differential
df_team_season_detailed_results["ScoreDiff"] = (
    df_team_season_detailed_results["TeamScore"] - df_team_season_detailed_results["OppScore"]
)
df_team_season_detailed_results["Win"] = (df_team_season_detailed_results["GameResult"] == "W").astype(
    "int"
)

#Turnover Differntial
df_team_season_detailed_results["TurnoverDiff"] = df_team_season_detailed_results["TeamTO"] - df_team_season_detailed_results["OTO"]

#FG%
df_team_season_detailed_results["TeamFG_Percent"] = df_team_season_detailed_results["TeamFGM"] / df_team_season_detailed_results["TeamFGA"]
df_team_season_detailed_results["OppFG_Percent"] = df_team_season_detailed_results["OFGM"] / df_team_season_detailed_results["OFGA"]
df_team_season_detailed_results["FG_Percent_Diff"] = df_team_season_detailed_results["TeamFG_Percent"] - df_team_season_detailed_results["OppFG_Percent"]

#FG3%
df_team_season_detailed_results["Team3P_Percent"] = df_team_season_detailed_results["TeamFGM3"] / df_team_season_detailed_results["TeamFGA3"]
df_team_season_detailed_results["Opp3P_Percent"] = df_team_season_detailed_results["OFGM3"] / df_team_season_detailed_results["OFGA3"]
df_team_season_detailed_results["3P_Percent_Diff"] = df_team_season_detailed_results["Team3P_Percent"] - df_team_season_detailed_results["Opp3P_Percent"]

#FT%
df_team_season_detailed_results["TeamFT_Percent"] = df_team_season_detailed_results["TeamFTM"] / df_team_season_detailed_results["TeamFTA"]
df_team_season_detailed_results["OppFT_Percent"] = df_team_season_detailed_results["OFTM"] / df_team_season_detailed_results["OFTA"]
df_team_season_detailed_results["FT_Percent_Diff"] = df_team_season_detailed_results["TeamFT_Percent"] - df_team_season_detailed_results["OppFT_Percent"]


#Possession/Pace Estimation
df_team_season_detailed_results["TeamPossessions"] = df_team_season_detailed_results["TeamFGA"] - df_team_season_detailed_results["TeamOR"] + df_team_season_detailed_results["TeamTO"] + (0.475 * df_team_season_detailed_results["TeamFTA"])
df_team_season_detailed_results["OppPossessions"] = df_team_season_detailed_results["OFGA"] + df_team_season_detailed_results["OTO"] - df_team_season_detailed_results["ODR"] + (0.475 * df_team_season_detailed_results["OFTA"])
df_team_season_detailed_results["GamePace"] = (df_team_season_detailed_results["TeamPossessions"] +df_team_season_detailed_results["OppPossessions"]) / 2


def mean_round(x):
    return round(x.mean(), 2)

team_season_agg = (
    df_team_season_detailed_results.groupby(["Season", "TeamID", "Bracket"])
    .agg(
        AvgScoreDiff=("ScoreDiff", mean_round),
        AvgBlk=("TeamBlk", mean_round),
        AvgOR=("TeamOR", mean_round),
        AvgStl=("TeamStl", mean_round),
        AvgFGM=("TeamFGM", mean_round),
        AvgFGA=("TeamFGA", mean_round),
        AvgFGM3=("TeamFGM3", mean_round),
        AvgFGA3=("TeamFGA3", mean_round),
        AvgFTM=("TeamFTM", mean_round),
        AvgFTA=("TeamFTA", mean_round),
        AvgDR=("TeamDR", mean_round),
        AvgAst=("TeamAst", mean_round),
        AvgTO=("TeamTO", mean_round),
        AvgPF=("TeamPF", mean_round),
        AvgOBlk=("OBlk", mean_round),
        AvgOOR=("OOR", mean_round),
        AvgOStl=("OStl", mean_round),
        AvgOFGM=("OFGM", mean_round),
        AvgOFGA=("OFGA", mean_round),
        AvgOFGM3=("OFGM3", mean_round),
        AvgOFGA3=("OFGA3", mean_round),
        AvgOFTM=("OFTM", mean_round),
        AvgOFTA=("OFTA", mean_round),
        AvgODR=("ODR", mean_round),
        AvgOAst=("OAst", mean_round),
        AvgOTO=("OTO", mean_round),
        AvgOPF=("OPF", mean_round),
        AvgTurnoverDiff=("TurnoverDiff", mean_round),
        AvgTeamFG_Percent=("TeamFG_Percent", mean_round),
        AvgOppFG_Percent=("OppFG_Percent", mean_round),
        AvgFG_Percent_Diff=("FG_Percent_Diff", mean_round),
        AvgTeam3P_Percent=("Team3P_Percent", mean_round),
        AvgOpp3P_Percent=("Opp3P_Percent", mean_round),
        Avg3P_Percent_Diff=("3P_Percent_Diff", mean_round),
        AvgTeamFT_Percent=("TeamFT_Percent", mean_round),
        AvgOppFT_Percent=("OppFT_Percent", mean_round),
        AvgFT_Percent_Diff=("FT_Percent_Diff", mean_round),
        AvgTeamPossesions= ("TeamPossessions", mean_round),
        AvgOppPosessions=("OppPossessions", mean_round),
        AvgGamePace=("GamePace", mean_round),
        Wins=("Win", "sum"),
        Losses=("GameResult", lambda x: (x == "L").sum()),
        WinPercentage=("Win", mean_round),
    )
    .reset_index()
)



df_seeds["ChalkSeed"] = (
    df_seeds["Seed"].str.replace("a", "").str.replace("b", "").str[1:].astype("int")
)

team_season_agg = team_season_agg.merge(
    df_seeds, on=["Season", "TeamID", "Bracket"], how="left"
)
# Handles teams that don't the tournament
team_season_agg["ChalkSeed"] = team_season_agg["ChalkSeed"].fillna(16).astype(int)


team_season_agg = team_season_agg.drop(columns="Seed")
team_season_agg.isna().sum()


team_season_agg.sample(5, random_state=8)


df_team_tourney_results = pd.concat(
    [
        df_tourney_results[
            ["Season", "Bracket", "WTeamID", "LTeamID", "WScore", "LScore"]
        ]
        .assign(GameResult="W")
        .rename(
            columns={
                "WTeamID": "TeamID",
                "LTeamID": "OppTeamID",
                "WScore": "TeamScore",
                "LScore": "OppScore",
            }
        ),
        df_tourney_results[
            ["Season", "Bracket", "LTeamID", "WTeamID", "LScore", "WScore"]
        ]
        .assign(GameResult="L")
        .rename(
            columns={
                "LTeamID": "TeamID",
                "WTeamID": "OppTeamID",
                "LScore": "TeamScore",
                "WScore": "OppScore",
            }
        ),
    ]
).reset_index(drop=True)

df_team_tourney_results["Win"] = (df_team_tourney_results["GameResult"] == "W").astype(
    "int"
)


df_tourney_features = df_team_tourney_results.merge(
    team_season_agg[
        [
            'Season', 'Bracket', 'TeamID', 'AvgScoreDiff', 'AvgBlk', 'AvgOR', 'AvgStl',
            'AvgFGM', 'AvgFGA', 'AvgFGM3', 'AvgFGA3', 'AvgFTM', 'AvgFTA',
            'AvgDR', 'AvgAst', 'AvgTO', 'AvgPF', 'AvgOBlk', 'AvgOOR', 'AvgOStl',
            'AvgOFGM', 'AvgOFGA', 'AvgOFGM3', 'AvgOFGA3', 'AvgOFTM', 'AvgOFTA',
            'AvgODR', 'AvgOAst', 'AvgOTO', 'AvgOPF', 'Wins', 'Losses',
            'WinPercentage', 'ChalkSeed'
        ]
    ],
    on=['Season', 'Bracket', 'TeamID'],
    how='left'
).merge(
    team_season_agg[
        ['Season', 'Bracket', 'TeamID', 'AvgScoreDiff', 'AvgBlk', 'AvgOR', 'AvgStl',
         'AvgFGM', 'AvgFGA', 'AvgFGM3', 'AvgFGA3', 'AvgFTM', 'AvgFTA',
         'AvgDR', 'AvgAst', 'AvgTO', 'AvgPF', 'AvgOBlk', 'AvgOOR', 'AvgOStl',
         'AvgOFGM', 'AvgOFGA', 'AvgOFGM3', 'AvgOFGA3', 'AvgOFTM', 'AvgOFTA',
         'AvgODR', 'AvgOAst', 'AvgOTO', 'AvgOPF', 'Wins', 'Losses','WinPercentage', 'ChalkSeed']
    ].rename(columns={
        'TeamID': 'OppTeamID',
        'AvgScoreDiff': 'OppAvgScoreDiff',
        'AvgBlk': 'OppAvgBlk',
        'AvgOR': 'OppAvgOR',
        'AvgStl': 'OppAvgStl',
        'AvgFGM': 'OppAvgFGM',
        'AvgFGA': 'OppAvgFGA',
        'AvgFGM3': 'OppAvgFGM3',
        'AvgFGA3': 'OppAvgFGA3',
        'AvgFTM': 'OppAvgFTM',
        'AvgFTA': 'OppAvgFTA',
        'AvgDR': 'OppAvgDR',
        'AvgAst': 'OppAvgAst',
        'AvgTO': 'OppAvgTO',
        'AvgPF': 'OppAvgPF',
        'AvgOBlk': 'OppAvgOBlk',
        'AvgOOR': 'OppAvgOOR',
        'AvgOStl': 'OppAvgOStl',
        'AvgOFGM': 'OppAvgOFGM',
        'AvgOFGA': 'OppAvgOFGA',
        'AvgOFGM3': 'OppAvgOFGM3',
        'AvgOFGA3': 'OppAvgOFGA3',
        'AvgOFTM': 'OppAvgOFTM',
        'AvgOFTA': 'OppAvgOFTA',
        'AvgODR': 'OppAvgODR',
        'AvgOAst': 'OppAvgOAst',
        'AvgOTO': 'OppAvgOTO',
        'AvgOPF': 'OppAvgOPF',
        'Wins': 'OppWins',
        'Losses': 'OppLosses',
        'WinPercentage': 'OppWinPercentage',
        'ChalkSeed': 'OppChalkSeed'
    }),
    on=['Season', 'Bracket', 'OppTeamID'],
    how='left'
)


df_tourney_features.sample(5, random_state=8)


df_tourney_features["WinPctDiff"] = (
    df_tourney_features["WinPercentage"]
    - df_tourney_features["OppWinPercentage"]
)

df_tourney_features["ChalkSeedDiff"] = (
    df_tourney_features["ChalkSeed"]
    - df_tourney_features["OppChalkSeed"]
)

df_tourney_features_elo = df_tourney_features.merge(
    df_elo_ratings[
        ['TeamID', 'Season','Rating_Mean', 'Rating_Last'    
        ]
    ],
    on=['Season','TeamID'],
    how='left'
)


# Model Selection & Hyperparameter tuning
pipe = Pipeline([
                 ('selector', VarianceThreshold()),
                 ('clf', XGBClassifier(random_state=0))
                ])

optimal_params = [{
                 'selector__threshold':[0],
                'clf__max_depth': [3],
                'clf__learning_rate': [0.05],
                'clf__n_estimators': [120]
          }]

full_params = [{
                 'selector__threshold':[0, 0.01, 0.05],
                'clf__max_depth': [3, 5, 8],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__n_estimators': [100, 200, 300]
          }]


grid = GridSearchCV(pipe, 
                    optimal_params, 
                    scoring="neg_log_loss", #"neg_brier_score"
                    error_score='raise',
                    verbose=10, 
                    n_jobs=-1,
                    cv=5)


def evaluate_xgb(df_full: pd.DataFrame, year: int):
    df_hist = df_full[df_full['Season'].astype(int) < year]
    df_pred = df_full[df_full['Season'].astype(int) == year] 

    drop_cols = ['Win', 'Bracket', 'Season', 'TeamID', 'OppTeamID', 'GameResult', 'TeamScore', 'OppScore']
    X_train = df_hist.drop(columns=drop_cols)
    print('feature columns after drop')
    print(list(X_train.columns))
    y_train = df_hist[['Win']]
    X_test = df_pred.drop(columns=drop_cols)
    y_test = df_pred[['Win']]
    
    grid.fit(X_train, y_train)

    y_pred_probs = grid.predict_proba(X_test)

    try: 
        brier_score = brier_score_loss(y_test, y_pred_probs[:, 1])
        # logloss = log_loss(result['y'], result['y_pred'])
    except ValueError: 
        print(f"No data for year {year}")
        return np.nan, np.nan, np.nan

    return brier_score, grid


res = pd.DataFrame(columns = ['Bracket','year', 'brier_score'])
m_df_tourney_features_encoded = df_tourney_features_elo[df_tourney_features_elo['Bracket'] == 'M']
w_df_tourney_features_encoded = df_tourney_features_elo[df_tourney_features_elo['Bracket'] == 'W']
for year in range(2004, 2025): 
    if year != 2020:
        if year < 2011:
            m_brier_score, m_grid = evaluate_xgb(m_df_tourney_features_encoded, year)
            print(f"(M) year: {year}, brier score: {m_brier_score}")
            row_m = ['M', year, m_brier_score]
        
            res.loc[len(res)] = row_m

        else:
            m_brier_score, m_grid = evaluate_xgb(m_df_tourney_features_encoded, year)
            print(f"(M) year: {year}, brier score: {m_brier_score}")
            row_m = ['M', year, m_brier_score]       
            res.loc[len(res)] = row_m 
            
            w_brier_score, w_grid = evaluate_xgb(w_df_tourney_features_encoded, year)
            print(f"(W) year: {year}, brier score: {w_brier_score}")
            row_w = ['W', year, w_brier_score]
            res.loc[len(res)] = row_w

res


res['brier_score'].mean()


res.groupby('Bracket').agg(avg_brier_score=("brier_score", 'mean'))


clf = m_grid.best_estimator_.named_steps["clf"]
selected_feature_idx = m_grid.best_estimator_[0].get_support(indices=True)
X_train_validate = m_df_tourney_features_encoded.loc[:, ~m_df_tourney_features_encoded.columns.isin(['Win', 'Bracket','Season', 'TeamID', 'OppTeamID', 'GameResult', 'TeamScore', 'OppScore'])]
selected_columns = X_train_validate.columns[selected_feature_idx].to_list()


feature_importance = pd.DataFrame({
    "feature_names": selected_columns,
    "feature_importance": clf.feature_importances_
}).sort_values(by="feature_importance", ascending=False)
print("Top 10 features:")
print(feature_importance.head(10))


shap_explainer = shap.Explainer(clf, m_df_tourney_features_encoded[selected_columns])

fig = shap.plots.beeswarm(shap_explainer(m_df_tourney_features_encoded[selected_columns]),max_display=20, show=False)

plt.tight_layout()


submission = df_sample_submission.copy()
submission[['Season', 'TeamID', 'OppTeamID']] = submission['ID'].str.split('_', expand=True)
submission['Season'] = submission['Season'].astype(int)
submission['TeamID'] = submission['TeamID'].astype(int)
submission['OppTeamID'] = submission['OppTeamID'].astype(int)


submission.sample(5,random_state=8)


team_season_agg_2025 = team_season_agg.query("Season == 2025")
team_season_agg_2025.sample(5, random_state=8)


df_2025_matchups_with_stats = submission.merge(
    team_season_agg_2025[
        [
            'Bracket', 'TeamID', 'AvgScoreDiff', 'AvgBlk', 'AvgOR', 'AvgStl',
            'AvgFGM', 'AvgFGA', 'AvgFGM3', 'AvgFGA3', 'AvgFTM', 'AvgFTA',
            'AvgDR', 'AvgAst', 'AvgTO', 'AvgPF', 'AvgOBlk', 'AvgOOR', 'AvgOStl',
            'AvgOFGM', 'AvgOFGA', 'AvgOFGM3', 'AvgOFGA3', 'AvgOFTM', 'AvgOFTA',
            'AvgODR', 'AvgOAst', 'AvgOTO', 'AvgOPF', 'Wins', 'Losses',
            'WinPercentage', 'ChalkSeed'
        ]
    ],
    on=['TeamID'],
    how='left'
).merge(
    team_season_agg_2025[
        ['TeamID', 'AvgScoreDiff', 'AvgBlk', 'AvgOR', 'AvgStl',
         'AvgFGM', 'AvgFGA', 'AvgFGM3', 'AvgFGA3', 'AvgFTM', 'AvgFTA',
         'AvgDR', 'AvgAst', 'AvgTO', 'AvgPF', 'AvgOBlk', 'AvgOOR', 'AvgOStl',
         'AvgOFGM', 'AvgOFGA', 'AvgOFGM3', 'AvgOFGA3', 'AvgOFTM', 'AvgOFTA',
         'AvgODR', 'AvgOAst', 'AvgOTO', 'AvgOPF', 'Wins', 'Losses', 'WinPercentage', 'ChalkSeed']
    ].rename(columns={
        'TeamID': 'OppTeamID',
        'AvgScoreDiff': 'OppAvgScoreDiff',
        'AvgBlk': 'OppAvgBlk',
        'AvgOR': 'OppAvgOR',
        'AvgStl': 'OppAvgStl',
        'AvgFGM': 'OppAvgFGM',
        'AvgFGA': 'OppAvgFGA',
        'AvgFGM3': 'OppAvgFGM3',
        'AvgFGA3': 'OppAvgFGA3',
        'AvgFTM': 'OppAvgFTM',
        'AvgFTA': 'OppAvgFTA',
        'AvgDR': 'OppAvgDR',
        'AvgAst': 'OppAvgAst',
        'AvgTO': 'OppAvgTO',
        'AvgPF': 'OppAvgPF',
        'AvgOBlk': 'OppAvgOBlk',
        'AvgOOR': 'OppAvgOOR',
        'AvgOStl': 'OppAvgOStl',
        'AvgOFGM': 'OppAvgOFGM',
        'AvgOFGA': 'OppAvgOFGA',
        'AvgOFGM3': 'OppAvgOFGM3',
        'AvgOFGA3': 'OppAvgOFGA3',
        'AvgOFTM': 'OppAvgOFTM',
        'AvgOFTA': 'OppAvgOFTA',
        'AvgODR': 'OppAvgODR',
        'AvgOAst': 'OppAvgOAst',
        'AvgOTO': 'OppAvgOTO',
        'AvgOPF': 'OppAvgOPF',
        'Wins': 'OppWins',
        'Losses': 'OppLosses',
        'WinPercentage': 'OppWinPercentage',
        'ChalkSeed': 'OppChalkSeed'
    }),
    on=['OppTeamID'],
    how='left'
)


df_2025_matchups_with_stats["WinPctDiff"] = (
    df_2025_matchups_with_stats["WinPercentage"]
    - df_2025_matchups_with_stats["OppWinPercentage"]
)

df_2025_matchups_with_stats["ChalkSeedDiff"] = (
    df_2025_matchups_with_stats["ChalkSeed"]
    - df_2025_matchups_with_stats["OppChalkSeed"]
)


df_2025_matchups_with_stats = df_2025_matchups_with_stats.merge(
    df_elo_ratings[
        ['TeamID', 'Season','Rating_Mean', 'Rating_Last'    
        ]
    ],
    on=['Season','TeamID'],
    how='left'
)


df_2025_matchups_with_stats.columns


drop_cols_pred = ['ID', 'Pred', 'Season', 'Bracket']

X_pred_men = df_2025_matchups_with_stats.query("Bracket == 'M'").drop(columns=drop_cols_pred)
X_pred_women = df_2025_matchups_with_stats.query("Bracket == 'W'").drop(columns=drop_cols_pred)
X_pred_men['Pred'] = m_grid.predict_proba(X_pred_men.set_index(['TeamID', 'OppTeamID']))[:, 1]
X_pred_women['Pred'] = w_grid.predict_proba(X_pred_women.set_index(['TeamID', 'OppTeamID']))[:, 1]
final_submission = pd.concat([X_pred_men, X_pred_women]).reset_index(drop = True)
final_submission["ID"] = "2025_" + final_submission["TeamID"].astype(str) + "_" + final_submission["OppTeamID"].astype(str)

final_submission[['ID', 'Pred']].to_csv('MMM_test_predictions.csv', index=False)





final_submission[['ID','Pred']]




