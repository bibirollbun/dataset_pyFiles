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


from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import brier_score_loss, roc_auc_score, auc,roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


m_seed = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv")
w_seed = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv")
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)

m_season_results = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv")
w_season_results = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv")
season_results = pd.concat([m_season_results, w_season_results], axis=0)

m_tourney_results = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv")
w_tourney_results = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv")
tourney_results = pd.concat([m_tourney_results, w_tourney_results], axis=0)

submission_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv")

def extract_game_info(id_str):
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

def extract_seed_value(seed_str):
    try:
        return int(seed_str[1:])
    except ValueError:
        return 16

seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)

submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(extract_game_info).tolist()

season_results["ScoreDiff"] = season_results["WScore"] - season_results["LScore"]

win_stats = season_results.groupby(["Season", "WTeamID"]).agg(
    Wins=("WTeamID", "count"),
    AvgPointsScored_W=("WScore", "mean"),
    AvgPointsAgainst_W=("LScore", "mean"),
    AvgMargin_W=("ScoreDiff", "mean"),
).reset_index().rename(columns={"WTeamID": "TeamID"})

loss_stats = season_results.groupby(["Season", "LTeamID"]).agg(
    Losses=("LTeamID", "count"),
    AvgPointsScored_L=("LScore", "mean"),
    AvgPointsAgainst_L=("WScore", "mean"),
    AvgMargin_L=("ScoreDiff", "mean"),
).reset_index().rename(columns={"LTeamID": "TeamID"})

team_stats = pd.merge(win_stats, loss_stats, on=["Season", "TeamID"], how="outer").fillna(0)
team_stats["TotalGames"] = team_stats["Wins"] + team_stats["Losses"]
team_stats["WinRate"] = team_stats["Wins"] / team_stats["TotalGames"].replace(0, 1)
team_stats["TotalPointsScored"] = team_stats["AvgPointsScored_W"] * team_stats["Wins"] + team_stats["AvgPointsScored_L"] * team_stats["Losses"]
team_stats["TotalPointsAllowed"] = team_stats["AvgPointsAgainst_W"] * team_stats["Wins"] + team_stats["AvgPointsAgainst_L"] * team_stats["Losses"]
team_stats["PointsDifferential"] = (team_stats["TotalPointsScored"] - team_stats["TotalPointsAllowed"]) / team_stats["TotalGames"].replace(0, 1)

win_tourney_stats = tourney_results.groupby(["Season", "WTeamID"]).agg(
    TourneyWins=("WTeamID", "count"),
).reset_index().rename(columns={"WTeamID": "TeamID"})

loss_tourney_stats = tourney_results.groupby(["Season", "LTeamID"]).agg(
    TourneyLosses=("LTeamID", "count"),
).reset_index().rename(columns={"LTeamID": "TeamID"})

tourney_stats = pd.merge(win_tourney_stats, loss_tourney_stats, on=["Season", "TeamID"], how="outer").fillna(0)
tourney_stats["TotalTourneyGames"] = tourney_stats["TourneyWins"] + tourney_stats["TourneyLosses"]
tourney_stats["TourneyWinRate"] = tourney_stats["TourneyWins"] / tourney_stats["TotalTourneyGames"].replace(0, 1)


results_df = m_tourney_results.copy()
results_df = results_df[['Season', 'WTeamID', 'LTeamID']]

# team 1
train_data1 = results_df.copy()
train_data1['TeamID1'] = train_data1['WTeamID']
train_data1['TeamID2'] = train_data1['LTeamID']
train_data1['Outcome'] = 1
# team 2 
train_data2 = results_df.copy()
train_data2['TeamID1'] = train_data2['LTeamID']
train_data2['TeamID2'] = train_data2['WTeamID']
train_data2['Outcome'] = 0

train_data = pd.concat([train_data1, train_data2], axis=0)
train_data = train_data[['Season', 'TeamID1', 'TeamID2', 'Outcome']]


train_data = train_data.merge(
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

train_data = train_data.merge(
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])


train_data = train_data.merge(
    team_stats[['Season', 'TeamID', 'WinRate', 'PointsDifferential']],
    left_on=['Season', 'TeamID1'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={
    'WinRate': 'WinRate1', 
    'PointsDifferential': 'PointsDifferential1'
}).drop(columns=['TeamID'])

train_data = train_data.merge(
    team_stats[['Season', 'TeamID', 'WinRate', 'PointsDifferential']],
    left_on=['Season', 'TeamID2'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={
    'WinRate': 'WinRate2', 
    'PointsDifferential': 'PointsDifferential2'
}).drop(columns=['TeamID'])


train_data.fillna(0, inplace=True)

train_data['SeedDiff'] = train_data['SeedValue1'] - train_data['SeedValue2']
train_data['WinRateDiff'] = train_data['WinRate1'] - train_data['WinRate2']
train_data['PointsDiffDiff'] = train_data['PointsDifferential1'] - train_data['PointsDifferential2']

recent_seasons = train_data['Season'].unique()[-5:]  
train_data = train_data[train_data['Season'].isin(recent_seasons)]

features = [
    'SeedDiff', 
    'WinRateDiff', 
    'PointsDiffDiff',
]

X = train_data[features]
y = train_data['Outcome']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15, stratify=y
)
models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'model__C': [0.01, 0.1, 1.0, 10.0],
            'model__class_weight': [None, 'balanced']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=1),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_leaf': [3, 5, 7],
            'model__class_weight': [None, 'balanced']
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=5),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [2, 3, 4],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__subsample': [0.8, 0.9, 1.0]
        }
    }
}
results = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model_info in models.items():
    print(f"\nOptimizing {name}...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_info['model'])
    ])
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid=model_info['params'],
        cv=skf,
        scoring='neg_brier_score',  # Brier skoru minimize etmek için
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    
    calibrated_model = CalibratedClassifierCV(best_pipeline, cv=5, method='isotonic')
    calibrated_model.fit(X_train, y_train)
    
    cv_brier_scores = []
    cv_auc_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
        
        calibrated_model.fit(X_cv_train, y_cv_train)
        y_probs = calibrated_model.predict_proba(X_cv_val)[:, 1]
        
        cv_brier_scores.append(brier_score_loss(y_cv_val, y_probs))
        cv_auc_scores.append(roc_auc_score(y_cv_val, y_probs))
    
    y_probs = calibrated_model.predict_proba(X_test)[:, 1]
    test_brier_score = brier_score_loss(y_test, y_probs)
    test_auc_score = roc_auc_score(y_test, y_probs)
    
    results[name] = {
        'CV Brier Score Mean': np.mean(cv_brier_scores),
        'CV Brier Score Std': np.std(cv_brier_scores),
        'CV AUC Mean': np.mean(cv_auc_scores),
        'CV AUC Std': np.std(cv_auc_scores),
        'Test Brier Score': test_brier_score,
        'Test AUC': test_auc_score,
        'Best Parameters': grid_search.best_params_,
        'Best Model': calibrated_model
    }
    results[name]['Test Probs'] = y_probs
    print(f"{name}:")
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  CV Brier Score: {np.mean(cv_brier_scores):.4f} ± {np.std(cv_brier_scores):.4f}")
    print(f"  CV AUC: {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")
    print(f"  Test Brier Score: {test_brier_score:.4f}")
    print(f"  Test AUC: {test_auc_score:.4f}")

best_model_name = min(results, key=lambda x: results[x]['Test Brier Score'])
best_model = results[best_model_name]['Best Model']

print(f"\nBest model: {best_model_name}")
print(f"CV Brier Score: {results[best_model_name]['CV Brier Score Mean']:.4f}")
print(f"CV AUC: {results[best_model_name]['CV AUC Mean']:.4f}")
print(f"Test Brier Score: {results[best_model_name]['Test Brier Score']:.4f}")
print(f"Test AUC: {results[best_model_name]['Test AUC']:.4f}")
print(f"Best parameters: {results[best_model_name]['Best Parameters']}")

best_params = {key.split('__')[1]: value for key, value in results[best_model_name]['Best Parameters'].items()}

best_model = models[best_model_name]['model'].set_params(**best_params)

best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', best_model)
])
best_pipeline.fit(X, y)
submission_df = submission_df.merge(
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

submission_df = submission_df.merge(
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

submission_df = submission_df.merge(
    team_stats[['Season', 'TeamID', 'WinRate', 'PointsDifferential']],
    left_on=['Season', 'TeamID1'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={
    'WinRate': 'WinRate1', 
    'PointsDifferential': 'PointsDifferential1'
}).drop(columns=['TeamID'])

submission_df = submission_df.merge(
    team_stats[['Season', 'TeamID', 'WinRate', 'PointsDifferential']],
    left_on=['Season', 'TeamID2'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={
    'WinRate': 'WinRate2', 
    'PointsDifferential': 'PointsDifferential2'
}).drop(columns=['TeamID'])

submission_df.fillna(0, inplace=True)
submission_df['SeedDiff'] = submission_df['SeedValue1'] - submission_df['SeedValue2']
submission_df['WinRateDiff'] = submission_df['WinRate1'] - submission_df['WinRate2']
submission_df['PointsDiffDiff'] = submission_df['PointsDifferential1'] - submission_df['PointsDifferential2']

X_submission = submission_df[features]

# final model
final_model = CalibratedClassifierCV(best_pipeline, cv=5, method='isotonic')
final_model.fit(X, y)

submission_df['Pred'] = final_model.predict_proba(X_submission)[:, 1]
final_submission = submission_df[['ID', 'Pred']]
final_submission.to_csv('submission.csv', index=False)


best_model_name = min(results, key=lambda x: results[x]['Test Brier Score'])
y_probs_best = results[best_model_name]['Test Probs']  # En iyi modelin test seti tahmin olasılıkları

fpr, tpr, _ = roc_curve(y_test, y_probs_best)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'{best_model_name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Best Model')
plt.legend(loc='lower right')
plt.grid()
plt.show()




