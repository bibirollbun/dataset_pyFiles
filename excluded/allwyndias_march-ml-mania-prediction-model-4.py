import os
import numpy as np
import pandas as pd

import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, brier_score_loss, mean_squared_error, roc_curve, auc,roc_auc_score
from sklearn.ensemble import RandomForestRegressor #, GradientBoostingRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor

from sklearn.calibration import CalibratedClassifierCV,calibration_curve
from sklearn.isotonic import IsotonicRegression



import warnings
warnings.filterwarnings("ignore")



data_dir = '/kaggle/input/march-machine-learning-mania-2025/**'

data = None
teams = None
seeds = None
games = None
sub = None
gb = None
col = None
model = None # declare model here.
calibration_model = None # declare calibration model here.
iso_reg = None # declare Isotonic Regression here.
platt_scaler = None # declare Logistic Regression here.
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()



def compute_elo(df, K=32, base_rating=1500):
    elo = {}  # Store ratings

    def get_elo(team):
        return elo.get(team, base_rating)

    for index, row in df.iterrows():
        t1, t2 = row['Team1'], row['Team2']
        t1_elo, t2_elo = get_elo(t1), get_elo(t2)

        # Compute win probability
        exp_t1 = 1 / (1 + 10 ** ((t2_elo - t1_elo) / 400))

        # Update Elo
        if row['Pred'] == 1:
            t1_elo += K * (1 - exp_t1)
            t2_elo += K * (0 - (1 - exp_t1))
        else:
            t1_elo += K * (0 - exp_t1)
            t2_elo += K * (1 - (0 - exp_t1))

        # Store updated ratings
        elo[t1], elo[t2] = t1_elo, t2_elo
        df.at[index, 'Team1Elo'] = t1_elo
        df.at[index, 'Team2Elo'] = t2_elo

    return df

def get_latest_elo(team, elo_dict, base_rating=1500):
    return elo_dict.get(team, base_rating)

def apply_elo_to_predictions(df, historical_games):
    # Get the latest Elo ratings from past games
    latest_elos = historical_games[['Team1', 'Team1Elo']].set_index('Team1').to_dict()['Team1Elo']
    latest_elos.update(historical_games[['Team2', 'Team2Elo']].set_index('Team2').to_dict()['Team2Elo'])

    # Compute EloDiff for new matches
    df['Team1Elo'] = df['Team1'].apply(lambda t: get_latest_elo(t, latest_elos))
    df['Team2Elo'] = df['Team2'].apply(lambda t: get_latest_elo(t, latest_elos))
    df['EloDiff'] = df['Team1Elo'] - df['Team2Elo']

    return df

def compute_rcent_win_percentage(games, gameCount):
    max_season = games["Season"].max()
    team_games = games[games['Season']==max_season]
       

    team_games = (
        pd.concat([
            team_games[["DayNum", "WTeamID"]].assign(Win=1).rename(columns={"WTeamID": "TeamID"}),
            team_games[["DayNum", "LTeamID"]].assign(Win=0).rename(columns={"LTeamID": "TeamID"})
        ])
        .sort_values(by=["TeamID", "DayNum"])
    )
    
    team_games["TeamID"] = team_games["TeamID"].astype(int)
    
    def compute_last_N_win_pct(group):
        last_N = group.tail(gameCount)  # Select only the last 10 games
        return last_N["Win"].sum() / len(last_N)  # Compute win percentage
      
    team_games = team_games.groupby("TeamID").apply(compute_last_N_win_pct).reset_index()
    team_games.columns = ["TeamID", "RecentWinPct"]
    
    games = pd.merge(games, team_games[["TeamID", "RecentWinPct"]], left_on=["Team1"],right_on=["TeamID"], how="left")
    games = games.drop(columns=["TeamID"])
    games = games.rename(columns={"RecentWinPct": "Team1RecentWinPct"})
    games["Team1RecentWinPct"] = games["Team1RecentWinPct"].astype("float32")

    
    games = pd.merge(games, team_games[["TeamID", "RecentWinPct"]], left_on=["Team2"],right_on=["TeamID"], how="left")
    games = games.rename(columns={"RecentWinPct": "Team2RecentWinPct"})
    games = games.drop(columns=["TeamID"])
    games["Team2RecentWinPct"] = games["Team2RecentWinPct"].astype("float32")
    
  
    #games["RecentWinPctDiff"] = games["Team1RecentWinPct"] - games["Team2RecentWinPct"]
    
    return games


def load_data():
    global data
    global teams
    global seeds
    global games
    global sub
    global gb
    global col
    files = glob.glob(data_dir)
    data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') for p in files}

    teams = pd.concat([data['MTeams'], data['WTeams']])
    teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']])
    teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
    teams_spelling.columns = ['TeamID', 'TeamNameCount']
    teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])

    season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])
    season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
    tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
    tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])

    season_cresults['ST'] = 'S'
    season_dresults['ST'] = 'S'
    tourney_cresults['ST'] = 'T'
    tourney_dresults['ST'] = 'T'

    seeds_df = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
    seeds = {'_'.join(map(str, [int(k1), k2])): int(v[1:3]) for k1, v, k2 in seeds_df[['Season', 'Seed', 'TeamID']].values}

    games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
    games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})
    games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
    games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
    games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1)
    games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1)
    games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
    games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
    games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
    games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)
    games['ScoreDiff'] = games['WScore'] - games['LScore']
    games['Pred'] = games.apply(lambda r: 1.0 if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0.0, axis=1)
    games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0.0 else r['ScoreDiff'], axis=1)
    games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
    games['AdjSeedDiff'] = np.log1p(np.abs(games['SeedDiff'])) * np.sign(games['SeedDiff'])
    #games['AdjSeedDiff'] = np.log(games['SeedDiff'] + 1) # Adjusted Seed Difference (log transform)
    games = games.fillna(-1)
    
    games = compute_rcent_win_percentage(games,10)
    
  
    games = compute_elo(games)
    games['EloDiff'] = games['Team1Elo'] - games['Team2Elo']
    
    games['GameAge'] = (games['Season'].max() - games['Season']) * 365 + (games['DayNum'].max() - games['DayNum'])
    games['Weight'] = np.exp(-games['GameAge'] / 365.0)  # Adjust denominator for tuning
    
    # Define the stats columns to swap
    stats_columns = ['Score','FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 
                     'Ast', 'TO', 'Stl', 'Blk', 'PF']

    # Create a mask for rows where Team1 is NOT the winner
    mask = games["Team1"] != games["WTeamID"]

    # Swap values for relevant columns
    for col in stats_columns:
        w_col = "W" + col  # e.g., 'WFGM'
        l_col = "L" + col  # e.g., 'LFGM'

        # Swap only for rows where Team1 is not the winning team
        games.loc[mask, [w_col, l_col]] = games.loc[mask, [l_col, w_col]].values
 
    
    c_score_col = ['Team1RecentWinPct','Team2RecentWinPct','NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    #c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
    c_score_agg = ['mean']
    gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
    #gb.columns = [''.join(c) + '_c_score' for c in gb.columns]
    gb.columns = [''.join(c) for c in gb.columns]
    
    # Avoid division by zero (replace 0 with a small value)
    gb['WFGAmean'] = gb['WFGAmean'].replace(0, 1e-6)  
    gb['LDRmean'] = gb['LDRmean'].replace(0, 1e-6)  
    gb['WORmean'] = gb['WORmean'].replace(0, 1e-6) 
    
    # gb["WPossessions"] = gb["WFGAmean"] - gb["WORmean"] + gb["WTOmean"] + (0.45 * gb["WFTAmean"])
    # gb["LPossessions"] = gb["LFGAmean"] - gb["LORmean"] + gb["LTOmean"] + (0.45 * gb["LFTAmean"])
    
    gb['WOffensiveEfficiency'] = gb['WFGMmean'] / gb['WFGAmean'] # Offensive Efficiency (Field Goal Made / Attempted)
    gb['LOffensiveEfficiency'] = gb['LFGMmean'] / gb['LFGAmean'] # Offensive Efficiency (Field Goal Made / Attempted)
    #gb['DefensiveReboundRate'] = gb['LDRmean'] / (gb['LDRmean'] + gb['WORmean']) # Defensive Rebound Rate (Defensive Rebounds / Total Rebounds)
    
    gb["RecentWinPctDiff"] = gb["Team1RecentWinPctmean"] - gb["Team2RecentWinPctmean"]
    
    # Replace infinities and NaNs
    gb = gb.replace([np.inf, -np.inf], np.nan).fillna(0)
    

    #sub = data['SampleSubmissionStage1'].copy()
    sub = data['SampleSubmissionStage2'].copy()

    sub['WLoc'] = 3
    sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0]).astype(int)
    sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
    sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])
    sub['WTeamID'] = None
    sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
    sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
    sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
    sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
    sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)
    sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed']
    sub['AdjSeedDiff'] = np.log1p(np.abs(sub['SeedDiff'])) * np.sign(sub['SeedDiff'])
    #sub['AdjSeedDiff'] = np.log(sub['SeedDiff'] + 1) # Adjusted Seed Difference (log transform)
    sub = sub.fillna(-1)
    
    sub = apply_elo_to_predictions(sub, games)
  
    sub['GameAge'] = sub['Season'].max() - sub['Season']
    sub['Weight'] = np.exp(-sub['GameAge'])  # Exponential decay
      
   
    #games = games[games['ST'] == 'T']
    #games =  games[~games['Season'].isin([2024, 2025])].reset_index(drop=True)

    games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams')
    sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams')
    sub = pd.merge(sub, games[['IDTeams','ScoreDiff']], how='left', left_on='IDTeams', right_on='IDTeams')
    
    #games = games[games['ST'] == 'T']
  
    #exclude_cols = ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc'] + c_score_col
    exclude_cols = ['GameAge','Weight','ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred','ScoreDiff', 'ScoreDiffNorm','WLoc'] + c_score_col
    exclude_Features = ['AdjSeedDiff','NumOTmean']
    #exclude_Features = ['Season','SeedDiff','NumOTmean','WFGMmean','WFGM3mean','WTOmean','WStlmean','WPFmean','LFGAmean','LFGM3mean','LFGA3mean','LTOmean','LStlmean']
    exclude_cols = exclude_cols+exclude_Features
    
    col = [c for c in games.columns if c not in exclude_cols]
    
    
    print('Features : ',col)

    print("Data loading and preprocessing completed.")


def create_models():
    global model
    global calibration_model
    global iso_reg
    
    # Create the models here with the same parameters.    
    model = RandomForestRegressor(
        n_estimators=235,
        random_state=42,
        max_depth=15,
        min_samples_split=2,
        max_features='sqrt',
        n_jobs=-1
    )    
  
    calibration_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)    
    
    iso_reg = IsotonicRegression(out_of_bounds='clip')


def plot_feature_importance(importances, feature_names, top_n=20):
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title('Top {} Feature Importances'.format(top_n))
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def plot_calibration_curve(y_true, y_proba, n_bins=10):
    combined = np.stack([y_proba, y_true], axis=-1)
    combined = combined[np.argsort(combined[:, 0])]
    sorted_probas = combined[:, 0]
    sorted_true = combined[:, 1]

    bins = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = bins[:-1] + (bins[1] - bins[0]) / 2
    bin_assignments = np.digitize(sorted_probas, bins) - 1

    bin_sums = np.bincount(bin_assignments, weights=sorted_probas, minlength=n_bins)
    bin_true = np.bincount(bin_assignments, weights=sorted_true, minlength=n_bins)
    bin_total = np.bincount(bin_assignments, minlength=n_bins)

    fraction_of_positives = bin_true / bin_total
    fraction_of_positives[np.isnan(fraction_of_positives)] = 0

    plt.figure(figsize=(4, 3))
    plt.plot(bin_midpoints, fraction_of_positives, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

    plt.xlabel('Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_calibration_curve_new(y_cal, cal_preds, n_bins=20):
    # Get calibration curve values
    prob_true, prob_pred = calibration_curve(y_cal, cal_preds, n_bins=20, strategy='uniform')

    # Plot the calibration curve
    plt.figure(figsize=(4, 3))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibrated Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration", color='black')

    # Labels and legend
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Win Rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid()
    plt.show()

def plot_prediction_distribution(predictions, title="Distribution of Predictions"):
    """Plots the distribution of model predictions."""
    plt.figure(figsize=(4, 3))
    sns.histplot(predictions, kde=True, color='skyblue')
    plt.title(title)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """Plots the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def train_model():
    # Step 1: Extract weights before scaling
    sample_weights = games['Weight'].values 
    
    # Step 2: Preprocess Data (without Weight)
    X = games[col].fillna(-1)
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    y = games['Pred']
    
    # Step 3: Train/Calibration Split
    #X_train, X_cal, y_train, y_cal = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_cal, y_train, y_cal, w_train, w_cal = train_test_split(X_scaled, y, sample_weights, 
                                                                      test_size=0.2, random_state=42)
    


    # Step 4: Train the Main Model with Weights
    #model.fit(X_train, y_train)
    model.fit(X_train, y_train, sample_weight=w_train)
    #model.fit(X_train, y_train, sample_weight=X.loc[X_train.index, 'Weight'])
    
    # Step 4: Get Predictions
    train_preds = model.predict(X_train).clip(0.001, 0.999) # Avoid extreme probabilities
    cal_preds = model.predict(X_cal).clip(0.001, 0.999)
    
    # Step 5: Train for Calibration 
    #platt_scaler.fit(cal_preds.reshape(-1, 1), y_cal) # Apply Platt Scaling (Logistic Regression)
    iso_reg.fit(cal_preds, y_cal)
    #calibration_model.fit(cal_preds.reshape(-1, 1), y_cal)
   
    # Step 6: Apply Calibration to Predictions
    train_preds_calibrated = iso_reg.predict(train_preds) #iso_reg.transform(train_preds)
    cal_preds_calibrated = iso_reg.predict(cal_preds) #iso_reg.transform(cal_preds)
    
    train_preds_calibrated = np.clip(train_preds_calibrated, 0.001, 0.999)
    cal_preds_calibrated = np.clip(cal_preds_calibrated, 0.001, 0.999)
    
#     train_preds_calibrated = calibration_model.predict(train_preds.reshape(-1, 1)).clip(0.001, 0.999)
#     cal_preds_calibrated = calibration_model.predict(cal_preds.reshape(-1, 1)).clip(0.001, 0.999)

    print(f'Log Loss (Train): {log_loss(y_train, train_preds_calibrated):.4f}')
    print(f'Brier Score (Train): {brier_score_loss(y_train, train_preds_calibrated):.4f}')
    print(f'MSE (Train): {mean_squared_error(y_train, train_preds_calibrated):.4f}')

    # Plot ROC Curve for the calibration set.
    plot_roc_curve(y_cal, cal_preds, "Calibration Set ROC Curve")

    
    feature_importances = model.feature_importances_
    feature_names = col
    plot_feature_importance(feature_importances, feature_names)
    
    # sorted_features = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)

    # for feature, importance in sorted_features[:20]:  # Top 20 features
    #     print(f"{feature}: {importance:.4f}")

    
    print('Calibration Curve : y_cal, cal_preds_calibrated')
    plot_calibration_curve_new(y_cal, cal_preds, n_bins=20)
  
    # Plot the distribution of calibrated predictions.
    plot_prediction_distribution(train_preds_calibrated, "Distribution of Calibrated Training Predictions")


def predict_submission(output_file='submission.csv'):
    # Step 1: Preprocess Submission Data
    sub_X = sub[col].fillna(-1)    
    print(len(sub_X))
    sub_X_imputed = imputer.transform(sub_X)
    sub_X_scaled = scaler.transform(sub_X_imputed)
    
    # Step 2: Make Predictions Using Trained Model
    preds = model.predict(sub_X_scaled).clip(0.001, 0.999)
    print(len(preds))
    
    # Step 3: Apply Calibration    
    preds_calibrated = iso_reg.predict(preds).clip(0.001, 0.999)     # Isotonic Calibration
    #preds_calibrated = calibration_model.predict(preds.reshape(-1, 1)).clip(0.001, 0.999)
    print(len(preds_calibrated))
    
    # Step 4: Save Predictions
    sub['Pred'] = preds_calibrated
    print(len(sub))
    sub[['ID', 'Pred']].drop_duplicates().to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")


def update_datatypes():
    global games
    global sub
    for col in games.select_dtypes(include=["int64"]).columns:
        col_min, col_max = games[col].min(), games[col].max()

        # Convert to smallest possible integer type
        if col_min >= -128 and col_max <= 127:
            games[col] = games[col].astype("int8")  # Fits in int8
        elif col_min >= -32_768 and col_max <= 32_767:
            games[col] = games[col].astype("int16")  # Fits in int16
        else:
            games[col] = games[col].astype("int32")  # Use int32 if needed
    for col in games.select_dtypes(include=["float64"]).columns:         
        games[col] = games[col].astype("float32")
        
    for col in sub.select_dtypes(include=["int64"]).columns:
        col_min, col_max = sub[col].min(), sub[col].max()

        # Convert to smallest possible integer type
        if col_min >= -128 and col_max <= 127:
            sub[col] = sub[col].astype("int8")  # Fits in int8
        elif col_min >= -32_768 and col_max <= 32_767:
            sub[col] = sub[col].astype("int16")  # Fits in int16
        else:
            sub[col] = sub[col].astype("int32")  # Use int32 if needed
    for col in sub.select_dtypes(include=["float64"]).columns:         
        sub[col] = sub[col].astype("float32")


load_data()
#update_datatypes()
create_models()
train_model()
predict_submission()


col = ['IDTeams', 'Team1Seed', 'Team2Seed',  'SeedDiff', 'EloDiff', 
       'WFGMmean', 'WFTMmean', 'WFTAmean', 'WDRmean', 'WAstmean', 
       'LFGMmean', 'LFTMmean', 'LFTAmean', 'LDRmean', 'LAstmean', 
       #'WPossessions', 'LPossessions', 
       'WOffensiveEfficiency', 'LOffensiveEfficiency']

print(col)
create_models()
train_model()
predict_submission()


def evaluate_predictions(df, actual_col, predicted_col):
    actual = df[actual_col]
    predicted_prob = df[predicted_col]
    
    # Compute Metrics
    brier = brier_score_loss(actual, predicted_prob)
    logloss = log_loss(actual, predicted_prob)
    accuracy = accuracy_score(actual, predicted_prob >= 0.5)
    auc = roc_auc_score(actual, predicted_prob)
    
    print(f"Brier Score: {brier:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Calibration Plot
    prob_true, prob_pred = calibration_curve(actual, predicted_prob, n_bins=10)
    plt.figure(figsize=(3,3))
    plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Win Rate')
    plt.title('Calibration Plot')
    plt.legend()
    plt.show()

def compare_yearwise(year, results, prediction):
    results_year = results[results['Season'] == year]
    prediction_year = prediction[prediction['Season'] == year]
    merged_df = pd.merge(prediction_year, results_year, how='inner', on=['ID','Season'])
    #merged_df['Actual'] = merged_df.apply(lambda r: 1 if r['Team1'] == r['WTeamID'] else 0, axis=1).astype(int) 

    merged_df['Actual'] = (merged_df['Team1'] == merged_df['WTeamID']).astype(int)
    evaluate_predictions(merged_df, actual_col='Actual', predicted_col='Pred')


output_dir = '/kaggle/working/'
submissionFile = 'submission.csv'

print(os.path.join(output_dir,submissionFile))

results = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
results['ID'] = results.apply(lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1)

prediction = pd.read_csv(os.path.join(output_dir,submissionFile))
prediction['Season'] = prediction['ID'].map(lambda x: x.split('_')[0]).astype(int)
prediction['Team1'] = prediction['ID'].map(lambda x: x.split('_')[1]).astype('int64')
prediction['Team2'] = prediction['ID'].map(lambda x: x.split('_')[2]).astype('int64')



#compare_yearwise(2024, results, prediction)


# compare_yearwise(2023, results, prediction)
# compare_yearwise(2022, results, prediction)
# compare_yearwise(2021, results, prediction)


# for dirname, _, filenames in os.walk('/kaggle/working/'):    
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#         df = pd.read_csv(os.path.join(dirname, filename))
#         print(df)

