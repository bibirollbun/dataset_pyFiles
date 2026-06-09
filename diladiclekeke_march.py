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


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
import optuna
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')
base_path = "/kaggle/input/march-machine-learning-mania-2025"
m_regular_detailed_file = os.path.join(base_path, "MRegularSeasonDetailedResults.csv")
w_regular_detailed_file = os.path.join(base_path, "WRegularSeasonDetailedResults.csv")
massey_file = os.path.join(base_path, "MMasseyOrdinals.csv")
m_regular_season_file = os.path.join(base_path, "MRegularSeasonCompactResults.csv")
w_regular_season_file = os.path.join(base_path, "WRegularSeasonCompactResults.csv")
m_tourney_file = os.path.join(base_path, "MNCAATourneyCompactResults.csv")
w_tourney_file = os.path.join(base_path, "WNCAATourneyCompactResults.csv")
m_seeds_file = os.path.join(base_path, "MNCAATourneySeeds.csv")
w_seeds_file = os.path.join(base_path, "WNCAATourneySeeds.csv")
submission_file = os.path.join(base_path, "SampleSubmissionStage2.csv")
print("Veri setleri yükleniyor...")
df_m_regular = pd.read_csv(m_regular_season_file)
df_w_regular = pd.read_csv(w_regular_season_file)
df_m_tourney = pd.read_csv(m_tourney_file)
df_w_tourney = pd.read_csv(w_tourney_file)
df_m_seeds = pd.read_csv(m_seeds_file)
df_w_seeds = pd.read_csv(w_seeds_file)
df_submission = pd.read_csv(submission_file)
df_m_regular_detailed = pd.read_csv(m_regular_detailed_file)
df_w_regular_detailed = pd.read_csv(w_regular_detailed_file)
df_massey = pd.read_csv(massey_file)
print(f"df_submission şekli (Stage 1, beklenen 131407): {df_submission.shape}")
if len(df_submission) != 131407:
    raise ValueError(f"df_submission {len(df_submission)} satırlı, Stage 1 için beklenen 131407!")

df_regular = pd.concat([df_m_regular.assign(Gender='M'), df_w_regular.assign(Gender='W')], ignore_index=True)
df_regular_detailed = pd.concat([df_m_regular_detailed.assign(Gender='M'), df_w_regular_detailed.assign(Gender='W')], ignore_index=True)
df_tourney = pd.concat([df_m_tourney.assign(Gender='M'),df_w_tourney.assign(Gender='W')], ignore_index=True)
df_seeds = pd.concat([df_m_seeds.assign(Gender='M'), df_w_seeds.assign(Gender='W')], ignore_index=True)
print(f"Düzenli sezon verisi: {df_regular.shape}")
print(f"Turnuva verisi: {df_tourney.shape}")
print(f"Seed verisi: {df_seeds.shape}")
print(f"Submission verisi: {df_submission.shape}")  
def calculate_elo_ratings(df, initial_elo=1500):
    elo_ratings = {}
    for team in set(df['WTeamID']).union(df['LTeamID']):
        elo_ratings[team] = initial_elo
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Elo hesaplanıyor"):
        winner, loser = row['WTeamID'], row['LTeamID']
        elo_w, elo_l = elo_ratings[winner], elo_ratings[loser]
        expected_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
        k_factor = 20 if row['DayNum'] <= 132 else 30
        elo_ratings[winner] += k_factor * (1- expected_w)
        elo_ratings[loser] += k_factor * (0 - (1- expected_w))
    return elo_ratings
df_seeds['SeedNum'] = df_seeds['Seed'].str.extract('(\d+)').astype(int)
latest_massey = df_massey[df_massey['RankingDayNum'] == 133].groupby('TeamID')['OrdinalRank'].mean()
def team_stats(df_regular):
    team_stats = {}
    stats_columns = ['FGM', 'FGA', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
    for team in set(df_regular['WTeamID']).union(df_regular['LTeamID']):
        wins = df_regular[df_regular['WTeamID'] == team]
        losses = df_regular[df_regular['LTeamID'] == team]
        stats_w = wins[['WFGM', 'WFGA', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].mean()
        stats_l = losses[['LFGM', 'LFGA', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].mean()
        if wins.empty:
            stats_w = pd.Series(0, index=[f'W{stat}' for stat in stats_columns])
        if losses.empty:
            stats_l = pd.Series(0, index=[f'L{stat}' for stat in stats_columns])
        team_stats[team] = pd.concat([stats_w, stats_l])
    return team_stats
def preprocess_data(df, elo_ratings=None, seeds_df=None, regular_stats=None, massey=None, fit=False, is_train=True):
    df_processed = df.copy()

    if is_train:
        df_processed['TeamA'] = df_processed['WTeamID']
        df_processed['TeamB'] = df_processed['LTeamID']
        df_processed['Result'] = 1
        df_reverse = df_processed.copy()
        df_reverse['TeamA'] = df_processed['LTeamID']
        df_reverse['TeamB'] = df_processed['WTeamID']
        df_reverse['Result'] = 0
        df_processed = pd.concat([df_processed, df_reverse], ignore_index=True)
    if fit:
        elo_ratings = calculate_elo_ratings(df)
    df_processed['EloA'] = df_processed['TeamA'].map(elo_ratings)
    df_processed['EloB'] = df_processed['TeamB'].map(elo_ratings)
    df_processed['EloDiff'] = df_processed['EloA'] - df_processed['EloB']

    seeds_df_team_a = seeds_df.rename(columns={'TeamID': 'TeamA', 'SeedNum': 'SeedA'})
    seeds_df_team_b = seeds_df.rename(columns={'TeamID': 'TeamB', 'SeedNum': 'SeedB'})
    df_processed = df_processed.merge(seeds_df_team_a[['Season', 'TeamA', 'SeedA']], on=['Season', 'TeamA'], how='left')
    df_processed = df_processed.merge(seeds_df_team_b[['Season', 'TeamB', 'SeedB']], on=['Season', 'TeamB'], how='left')
    df_processed['SeedDiff'] = df_processed['SeedA'] - df_processed['SeedB']
    df_processed['SeedDiff'] = df_processed['SeedDiff'].fillna(0)
    df_processed['MasseyA'] = df_processed['TeamA'].map(massey)
    df_processed['MasseyB'] = df_processed['TeamB'].map(massey)
    df_processed['MasseyDiff'] = df_processed['MasseyA'] - df_processed['MasseyB']
    df_processed['MasseyDiff'] = df_processed['MasseyDiff'].fillna(0)
    if regular_stats:
        for stat in ['FGM', 'FGA', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
            df_processed[f'{stat}_A'] = df_processed['TeamA'].map(
                lambda x: regular_stats[x][f'W{stat}'] if x in regular_stats and f'W{stat}' in regular_stats[x] else 0
            )
            df_processed[f'{stat}_B'] = df_processed['TeamB'].map(
               lambda x: regular_stats[x][f'W{stat}'] if x in regular_stats and f'W{stat}' in regular_stats[x] else 0
            )
            df_processed[f'{stat}_Diff'] = df_processed[f'{stat}_A'] - df_processed[f'{stat}_B']
                
    features = ['EloDiff', 'SeedDiff', 'MasseyDiff'] + [f'{stat}_Diff' for stat in ['FGM', 'FGA', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']]
    if fit:
        return df_processed[features], df_processed['Result'], elo_ratings
    else:
        return df_processed[features], elo_ratings
df_test = df_submission.copy()
df_test[['Season', 'TeamA', 'TeamB']] = df_test['ID'].str.split('_', expand=True).astype(int)
print("\nDüzenli sezon istatistikleri hesaplanıyor...")
regular_stats = team_stats(df_regular_detailed)
print("\nEğitim verisi ön işleniyor...")
X_train, y_train, elo_ratings = preprocess_data(df_tourney, seeds_df=df_seeds, regular_stats=regular_stats, massey=latest_massey, fit=True, is_train=True)
print("\nTest verisi ön işleniyor...")
X_test, _ = preprocess_data(df_test, elo_ratings, seeds_df=df_seeds, regular_stats=regular_stats, massey=latest_massey, fit=False, is_train=False)

print(f"X_train şekli: {X_train.shape}")
print(f"y_train şekli: {y_train.shape}")
print(f"X_test şekli: {X_test.shape}")
print(f"df_submission şekli: {df_submission.shape}")
if len(X_test) != len(df_submission):
    print(f"UYARI: X_test ({len(X_test)}) ve df_submission ({len(df_submission)} satır sayıları uyuşmuyor!")
if len(df_submission) != 131407:
    raise ValueError(f"df_submission satır sayısı {len(df_submission)}, Stage 1 için")


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
import optuna
import numpy as np
import pandas as pd
import logging

logging.basicConfig(filename='/kaggle/working/optuna_log.txt', level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial, model_name, X, y):
    if model_name == 'xgb':
        params = {
         'n_estimators': trial.suggest_int('n_estimators', 100, 500),
         'max_depth': trial.suggest_int('max_depth', 3, 10),
         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
         'random_state': 42,
         'objective': 'binary:logistic'
        }
        model = XGBClassifier(**params)
    elif model_name == 'lgbm':
        params = {
         'n_estimators': trial.suggest_int('n_estimators', 100, 500),
         'max_depth': trial.suggest_int('max_depth', 3, 10),
         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
         'random_state': 42,
         'objective': 'binary'
        }
        model = LGBMClassifier(**params)
    elif model_name == 'catboost':
        params = {
         'iterations': trial.suggest_int('iterations', 100, 500),
         'depth': trial.suggest_int('depth', 3, 10),
         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
         'random_state': 42,
         'verbose': False
        }
        model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    return scores.mean()
print("\nHiperparametre optimizasyonu başlıyor...")
best_params = {}
for name in ['xgb', 'lgbm', 'catboost']:
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, name, X_train, y_train), n_trials=50)
    print(f"\n{name} için en iyi parametreler: {study.best_params}")
    print(f"\n{name} için en iyi ROC AUC: {study.best_value:.4f}")
    best_params[name] = study.best_params
print("Optimizasyon tamamlandı, stacking başlıyor...")
base_models = [
    ('xgb', XGBClassifier(best_params['xgb'])),
    ('lgbm', LGBMClassifier(best_params['lgbm'])),
    ('catboost', CatBoostClassifier(best_params['catboost'], verbose=False))
]
stacking_model = StackingClassifier(
    estimators= base_models,
    final_estimator=XGBClassifier(random_state=42),
    cv=5
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_predictions = np.zeros(len(df_submission))
train_brier_scores = []
print("\nk-Fold ile eğitim ve tahmin yapılıyor...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    stacking_model.fit(X_fold_train, y_fold_train)
    val_pred = stacking_model.predict_proba(X_fold_val)[:, 1]
    brier_score = brier_score_loss(y_fold_val, val_pred)
    train_brier_scores.append(brier_score)
    print(f"Fold {fold+1} Brier Score: {brier_score:.4f}")

    test_pred = stacking_model.predict_proba(X_test)[:, 1]
    if len(test_pred) != len(df_submission):
        raise ValueError(f"test_pred ({len(test_pred)}) ve df_submission ({len(df_submission)}) boyutları uyuşmuyor!")
    final_predictions += test_pred / 5
print(f"Ortalama Eğitim Brier Score: {np.mean(train_brier_scores):.4f}")
submission = pd.DataFrame({
    'ID': df_submission['ID'],
    'Pred': final_predictions
})
if len(submission) != 131407:
    raise ValueError(f"Submission dosyası {len(submission)} satır içeriyor, Stage 1 için beklenen 131407!")
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f"\nSubmission dosyası oluşturuldu: /kaggle/working/submission.csv, Satır sayısı: {len(submission)}")

cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\nStacking Model ROC AUC: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")    

