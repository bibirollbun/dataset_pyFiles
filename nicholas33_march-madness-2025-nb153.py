import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import brier_score_loss

def load_data(gender):
    reg_results = pd.read_csv(f"/kaggle/input/march-machine-learning-mania-2025/{gender}RegularSeasonDetailedResults.csv")
    tourney_results = pd.read_csv(f"/kaggle/input/march-machine-learning-mania-2025/{gender}NCAATourneyCompactResults.csv")
    seeds = pd.read_csv(f"/kaggle/input/march-machine-learning-mania-2025/{gender}NCAATourneySeeds.csv")
    assert not reg_results.empty, f"{gender} regular season data failed to load"
    return reg_results, tourney_results, seeds

def compute_team_features(reg_results):
    df_w = reg_results[['Season', 'WTeamID', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'WScore', 'WFTA']].rename(columns={'WTeamID': 'TeamID', 'WFGM': 'FGM', 'WFGA': 'FGA', 'WFGM3': 'FGM3', 'WFGA3': 'FGA3', 'WOR': 'OR', 'WDR': 'DR', 'WAst': 'Ast', 'WTO': 'TO', 'WStl': 'Stl', 'WBlk': 'Blk', 'WPF': 'PF', 'WScore': 'Score', 'WFTA': 'FTA'})
    df_w['OpponentScore'] = reg_results['LScore']
    df_l = reg_results[['Season', 'LTeamID', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'LScore', 'LFTA']].rename(columns={'LTeamID': 'TeamID', 'LFGM': 'FGM', 'LFGA': 'FGA', 'LFGM3': 'FGM3', 'LFGA3': 'FGA3', 'LOR': 'OR', 'LDR': 'DR', 'LAst': 'Ast', 'LTO': 'TO', 'LStl': 'Stl', 'LBlk': 'Blk', 'LPF': 'PF', 'LScore': 'Score', 'LFTA': 'FTA'})
    df_l['OpponentScore'] = reg_results['WScore']
    df_team_games = pd.concat([df_w, df_l])
    mean_stats = df_team_games.groupby(['Season', 'TeamID']).agg({'FGM': 'mean', 'FGA': 'mean', 'FGM3': 'mean', 'FGA3': 'mean', 'OR': 'mean', 'Ast': 'mean', 'TO': 'mean', 'Stl': 'mean', 'PF': 'mean', 'Blk': 'mean', 'Score': 'mean', 'OpponentScore': 'mean', 'FTA': 'mean'}).reset_index()
    mean_stats['PointDiff'] = mean_stats['Score'] - mean_stats['OpponentScore']
    mean_stats['Possessions'] = mean_stats['FGA'] + 0.475 * mean_stats['FTA'] - mean_stats['OR'] + mean_stats['TO']
    
    df_team_games_full = pd.concat([reg_results[['Season', 'DayNum', 'WTeamID', 'WScore']].rename(columns={'WTeamID': 'TeamID', 'WScore': 'Score'}), reg_results[['Season', 'DayNum', 'LTeamID', 'LScore']].rename(columns={'LTeamID': 'TeamID', 'LScore': 'Score'})])
    threshold = 118
    df_last14d = df_team_games_full[df_team_games_full['DayNum'] > threshold]
    win_ratio = df_last14d.groupby(['Season', 'TeamID']).agg({'Score': 'count'}).reset_index().rename(columns={'Score': 'games_14d'})
    wins = reg_results[reg_results['DayNum'] > threshold].groupby(['Season', 'WTeamID']).size().reset_index(name='wins_14d').rename(columns={'WTeamID': 'TeamID'})
    win_ratio = win_ratio.merge(wins, on=['Season', 'TeamID'], how='left').fillna(0)
    win_ratio['win_ratio_14d'] = win_ratio['wins_14d'] / win_ratio['games_14d']
    team_features = mean_stats.merge(win_ratio[['Season', 'TeamID', 'win_ratio_14d']], on=['Season', 'TeamID'], how='left')
    return team_features

def compute_team_quality(reg_results):
    qualities = []
    for season in reg_results['Season'].unique():
        df_season = reg_results[reg_results['Season'] == season]
        df_pairs = pd.DataFrame({
            'T1': np.minimum(df_season['WTeamID'], df_season['LTeamID']),
            'T2': np.maximum(df_season['WTeamID'], df_season['LTeamID']),
            'y': (df_season['WTeamID'] == np.minimum(df_season['WTeamID'], df_season['LTeamID'])).astype(int)
        })
        teams = sorted(pd.unique(df_season[['WTeamID', 'LTeamID']].values.ravel()))
        
        X = pd.DataFrame(0, index=df_pairs.index, columns=[f'team_{team}' for team in teams])
        for team in teams:
            X.loc[df_pairs['T1'] == team, f'team_{team}'] = 1
            X.loc[df_pairs['T2'] == team, f'team_{team}'] = -1
        
        reference_team = max(teams)
        X = X.drop(columns=[f'team_{reference_team}'])
        
        model = LogisticRegression(penalty='l2', C=1.0, fit_intercept=False, solver='lbfgs', max_iter=1000)
        model.fit(X, df_pairs['y'])
        
        qualities_season = pd.DataFrame({
            'TeamID': [int(col.split('_')[1]) for col in X.columns],
            'quality': model.coef_[0],
            'Season': season
        })
        qualities.append(qualities_season)
    return pd.concat(qualities)

lr_features = ['Seed_diff', 'PointDiff', 'T1_quality', 'T2_quality']  # Global definition

def build_and_train(gender, tune=False, validate=True):
    reg_results, tourney_results, seeds = load_data(gender)
    team_features = compute_team_features(reg_results)
    qualities = compute_team_quality(reg_results)
    seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
    tourney_results['T1'] = np.minimum(tourney_results['WTeamID'], tourney_results['LTeamID'])
    tourney_results['T2'] = np.maximum(tourney_results['WTeamID'], tourney_results['LTeamID'])
    tourney_results['y'] = (tourney_results['WTeamID'] == tourney_results['T1']).astype(int)
    
    train_df = tourney_results.merge(team_features.add_prefix('T1_'), left_on=['Season', 'T1'], right_on=['T1_Season', 'T1_TeamID'])
    train_df = train_df.merge(team_features.add_prefix('T2_'), left_on=['Season', 'T2'], right_on=['T2_Season', 'T2_TeamID'])
    train_df = train_df.merge(qualities.add_prefix('T1_'), left_on=['Season', 'T1'], right_on=['T1_Season', 'T1_TeamID'])
    train_df = train_df.merge(qualities.add_prefix('T2_'), left_on=['Season', 'T2'], right_on=['T2_Season', 'T2_TeamID'])
    train_df = train_df.merge(seeds.add_prefix('T1_'), left_on=['Season', 'T1'], right_on=['T1_Season', 'T1_TeamID'])
    train_df = train_df.merge(seeds.add_prefix('T2_'), left_on=['Season', 'T2'], right_on=['T2_Season', 'T2_TeamID'])
    train_df['Seed_diff'] = train_df['T1_SeedNum'] - train_df['T2_SeedNum']
    train_df['PointDiff'] = train_df['T1_PointDiff'] - train_df['T2_PointDiff']
    
    xgb_features = ['T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
                    'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk', 
                    'PointDiff', 'T1_win_ratio_14d', 'T2_win_ratio_14d', 'T1_quality', 'T2_quality', 
                    'T1_SeedNum', 'T2_SeedNum', 'Seed_diff', 'T1_Possessions', 'T2_Possessions']
    
    X_xgb = train_df[xgb_features]
    X_lr = train_df[lr_features]
    y = train_df['y']
    
    if tune:
        param_grid = {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        base_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', 
                                     tree_method='hist', device='cuda', random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, scoring='neg_brier_score', 
                                 cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_xgb, y)
        best_params = grid_search.best_params_
        print(f"{gender} Model - Best Params: {best_params}")
        print(f"{gender} Model - Best CV Brier Score: {-grid_search.best_score_:.4f}")
        xgb_model = xgb.XGBClassifier(**best_params, objective='binary:logistic', 
                                     eval_metric='logloss', tree_method='hist', 
                                     device='cuda', random_state=42)
        xgb_model.fit(X_xgb, y)
        lr_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        lr_model.fit(X_lr, y)
        return xgb_model, lr_model
    elif validate:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        brier_scores = []
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', 
                                    tree_method='hist', device='cuda', random_state=42)
        for train_idx, val_idx in kf.split(X_xgb):
            X_train, X_val = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            xgb_model.fit(X_train, y_train)
            y_pred_val = xgb_model.predict_proba(X_val)[:, 1]
            brier_scores.append(brier_score_loss(y_val, y_pred_val))
        mean_brier = np.mean(brier_scores)
        print(f"{gender} Model - Mean CV Brier Score: {mean_brier:.4f}")
        lr_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        lr_model.fit(X_lr, y)
        return xgb_model, lr_model
    else:
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', 
                                    tree_method='hist', device='cuda', random_state=42)
        xgb_model.fit(X_xgb, y)
        lr_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        lr_model.fit(X_lr, y)
        return xgb_model, lr_model

# Main workflow
print("Checking CUDA availability...")
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

print("\nTuning hyperparameters with cross-validation...")
model_men_tuned_xgb, model_men_tuned_lr = build_and_train('M', tune=True)
print("Men's tuned models completed")
model_women_tuned_xgb, model_women_tuned_lr = build_and_train('W', tune=True)
print("Women's tuned models completed")

print("\nValidating with cross-validation (default params)...")
model_men_cv_xgb, model_men_cv_lr = build_and_train('M', tune=False, validate=True)
print("Men's CV completed")
model_women_cv_xgb, model_women_cv_lr = build_and_train('W', tune=False, validate=True)
print("Women's CV completed")

print("\nRetraining on full data with tuned params for submission...")
model_men_final_xgb, model_men_final_lr = build_and_train('M', tune=False, validate=False)
print("Men's final models trained")
model_women_final_xgb, model_women_final_lr = build_and_train('W', tune=False, validate=False)
print("Women's final models trained")

if model_men_final_xgb is None or model_women_final_xgb is None:
    raise ValueError("Final XGBoost models failed to train")
print("Models verified")

print("Loading submission data...")
df_sub = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')

df_sub['Season'] = df_sub['ID'].apply(lambda x: int(x.split('_')[0]))
df_sub['T1'] = df_sub['ID'].apply(lambda x: int(x.split('_')[1]))
df_sub['T2'] = df_sub['ID'].apply(lambda x: int(x.split('_')[2]))
print("Submission data loaded")

print("Loading additional data for test features...")
reg_results_men = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')
reg_results_women = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv')
seeds_men = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seeds_women = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
print("Additional data loaded")

print("Computing team features...")
team_features_men = compute_team_features(reg_results_men)
team_features_women = compute_team_features(reg_results_women)
qualities_men = compute_team_quality(reg_results_men)
qualities_women = compute_team_quality(reg_results_women)
seeds_men['SeedNum'] = seeds_men['Seed'].apply(lambda x: int(x[1:3]))
seeds_women['SeedNum'] = seeds_women['Seed'].apply(lambda x: int(x[1:3]))
print("Team features computed")

print("Building test DataFrame...")
test_df = df_sub.copy()

test_df = test_df.merge(team_features_men[['Season', 'TeamID', 'FGM', 'FGA', 'FGM3', 'FGA3', 'OR', 'Ast', 'TO', 'Stl', 'PF', 'Blk', 'PointDiff', 'win_ratio_14d', 'Possessions']],
                        left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left', suffixes=('', '_drop')).drop(columns=['TeamID'])
test_df = test_df.rename(columns={col: f'T1_{col}' for col in ['FGM', 'FGA', 'FGM3', 'FGA3', 'OR', 'Ast', 'TO', 'Stl', 'PF', 'Blk', 'PointDiff', 'win_ratio_14d', 'Possessions']})
test_df = test_df.merge(team_features_men[['Season', 'TeamID', 'FGM', 'FGA', 'FGM3', 'FGA3', 'OR', 'Ast', 'TO', 'Stl', 'PF', 'Blk', 'PointDiff', 'win_ratio_14d', 'Possessions']],
                        left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left', suffixes=('', '_drop')).drop(columns=['TeamID'])
test_df = test_df.rename(columns={col: f'T2_{col}' for col in ['FGM', 'FGA', 'FGM3', 'FGA3', 'OR', 'Ast', 'TO', 'Stl', 'PF', 'Blk', 'PointDiff', 'win_ratio_14d', 'Possessions']})

test_df = test_df.merge(team_features_women[['Season', 'TeamID', 'FGM', 'FGA', 'FGM3', 'FGA3', 'OR', 'Ast', 'TO', 'Stl', 'PF', 'Blk', 'PointDiff', 'win_ratio_14d', 'Possessions']],
                        left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left', suffixes=('', '_drop')).drop(columns=['TeamID'])
for col in ['FGM', 'FGA', 'FGM3', 'FGA3', 'OR', 'Ast', 'TO', 'Stl', 'PF', 'Blk', 'PointDiff', 'win_ratio_14d', 'Possessions']:
    test_df[f'T1_{col}'] = test_df[f'T1_{col}'].fillna(test_df[col])
    test_df = test_df.drop(columns=[col])
test_df = test_df.merge(team_features_women[['Season', 'TeamID', 'FGM', 'FGA', 'FGM3', 'FGA3', 'OR', 'Ast', 'TO', 'Stl', 'PF', 'Blk', 'PointDiff', 'win_ratio_14d', 'Possessions']],
                        left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left', suffixes=('', '_drop')).drop(columns=['TeamID'])
for col in ['FGM', 'FGA', 'FGM3', 'FGA3', 'OR', 'Ast', 'TO', 'Stl', 'PF', 'Blk', 'PointDiff', 'win_ratio_14d', 'Possessions']:
    test_df[f'T2_{col}'] = test_df[f'T2_{col}'].fillna(test_df[col])
    test_df = test_df.drop(columns=[col])

test_df = test_df.merge(qualities_men[['Season', 'TeamID', 'quality']], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left').drop(columns=['TeamID']).rename(columns={'quality': 'T1_quality'})
test_df = test_df.merge(qualities_men[['Season', 'TeamID', 'quality']], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left').drop(columns=['TeamID']).rename(columns={'quality': 'T2_quality'})
test_df = test_df.merge(qualities_women[['Season', 'TeamID', 'quality']], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left', suffixes=('', '_drop')).drop(columns=['TeamID'])
test_df['T1_quality'] = test_df['T1_quality'].fillna(test_df['quality'])
test_df = test_df.drop(columns=['quality'])
test_df = test_df.merge(qualities_women[['Season', 'TeamID', 'quality']], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left', suffixes=('', '_drop')).drop(columns=['TeamID'])
test_df['T2_quality'] = test_df['T2_quality'].fillna(test_df['quality'])
test_df = test_df.drop(columns=['quality'])

test_df = test_df.merge(seeds_men[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left').drop(columns=['TeamID']).rename(columns={'SeedNum': 'T1_SeedNum_men'})
test_df = test_df.merge(seeds_men[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left').drop(columns=['TeamID']).rename(columns={'SeedNum': 'T2_SeedNum_men'})
test_df = test_df.merge(seeds_women[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left').drop(columns=['TeamID']).rename(columns={'SeedNum': 'T1_SeedNum_women'})
test_df = test_df.merge(seeds_women[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left').drop(columns=['TeamID']).rename(columns={'SeedNum': 'T2_SeedNum_women'})

test_df['T1_SeedNum'] = np.where(test_df['T1'] < 2000, test_df['T1_SeedNum_men'], test_df['T1_SeedNum_women'])
test_df['T2_SeedNum'] = np.where(test_df['T2'] < 2000, test_df['T2_SeedNum_men'], test_df['T2_SeedNum_women'])

test_df = test_df.drop(columns=['T1_SeedNum_men', 'T2_SeedNum_men', 'T1_SeedNum_women', 'T2_SeedNum_women'])

test_df['Seed_diff'] = test_df['T1_SeedNum'] - test_df['T2_SeedNum']
test_df['PointDiff'] = test_df['T1_PointDiff'] - test_df['T2_PointDiff']
print("Test DataFrame built")

xgb_features = ['T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
                'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk', 
                'PointDiff', 'T1_win_ratio_14d', 'T2_win_ratio_14d', 'T1_quality', 'T2_quality', 
                'T1_SeedNum', 'T2_SeedNum', 'Seed_diff', 'T1_Possessions', 'T2_Possessions']

X_test_xgb = test_df[xgb_features].fillna(0)
X_test_lr = test_df[lr_features].fillna(0)

print("Features prepared")

print("Generating predictions...")
preds_men_xgb = model_men_final_xgb.predict_proba(X_test_xgb)[:, 1]
preds_men_lr = model_men_final_lr.predict_proba(X_test_lr)[:, 1]
print(f"Men's XGBoost preds mean: {preds_men_xgb.mean():.4f}, std: {preds_men_xgb.std():.4f}")
print(f"Men's LR preds mean: {preds_men_lr.mean():.4f}, std: {preds_men_lr.std():.4f}")
preds_men = (preds_men_xgb + preds_men_lr) / 2
print("Men's predictions generated")

preds_women_xgb = model_women_final_xgb.predict_proba(X_test_xgb)[:, 1]
preds_women_lr = model_women_final_lr.predict_proba(X_test_lr)[:, 1]
print(f"Women's XGBoost preds mean: {preds_women_xgb.mean():.4f}, std: {preds_women_xgb.std():.4f}")
print(f"Women's LR preds mean: {preds_women_lr.mean():.4f}, std: {preds_women_lr.std():.4f}")
preds_women = (preds_women_xgb + preds_women_lr) / 2
print("Women's predictions generated")

preds = np.where(df_sub['T1'] < 2000, preds_men, preds_women)
df_sub['Pred'] = np.clip(preds, 0.05, 0.95)
print("Predictions assigned")

df_sub[['ID', 'Pred']].to_csv('submission.csv', index=False)
print("Submission file saved")

print(df_sub.head(10))
print(df_sub['Pred'].describe())

